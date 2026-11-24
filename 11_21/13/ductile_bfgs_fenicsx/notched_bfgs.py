"""Ductile phase-field fracture of the 2D asymmetrically notched specimen
(Bie et al., Sec. 3.1) solved monolithically with the BFGS algorithm
(Eqs. 39-42): the initial matrix is the uncoupled block-diagonal tangent
diag(K_uu, K_dd), whose inverse is updated by BFGS (Eq. 41).

Units: mm, N, MPa.  Plane strain, thickness 1 mm.
"""
import argparse
import json
import os
import time

import numpy as np
import ufl
import basix.ufl
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

p = argparse.ArgumentParser()
p.add_argument("--du", type=float, default=3.0e-3, help="load increment (mm)")
p.add_argument("--umax", type=float, default=0.30)
p.add_argument("--hf", type=float, default=0.06, help="fine mesh size (mm)")
p.add_argument("--hc", type=float, default=0.6, help="coarse mesh size (mm)")
p.add_argument("--Gc", type=float, default=9.31, help="N/mm")
p.add_argument("--tol", type=float, default=1e-5)
p.add_argument("--maxit", type=int, default=500)
p.add_argument("--refresh", type=int, default=25, help="re-assemble K0 every n iterations")
p.add_argument("--maxcut", type=int, default=4, help="max bisection levels of a failed increment")
p.add_argument("--verbose", action="store_true")
p.add_argument("--rdiv", type=float, default=50.0, help="relative residual treated as divergence")
p.add_argument("--ddmax", type=float, default=0.5, help="max damage change per BFGS iteration")
p.add_argument("--restart", default="", help="checkpoint .npz to restart from")
p.add_argument("--dub", type=float, default=1.2, help="upper bound for d in the projection")
p.add_argument("--out", default="results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
comm = MPI.COMM_WORLD
assert comm.size == 1, "serial driver (numpy BFGS vectors)"

# ---------------------------------------------------------------- material
E, nu = 71.48e3, 0.3
Eh = 0.1 * E                       # hardening modulus E^H (bilinear)
sig_y = 345.0
Gc = args.Gc
lc = 3 * args.hf                   # l = 3*dx = 0.18 mm
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
Hiso = Eh                          # sigma_f = sigma_y + E^H * eps_p
k_res = 1e-6

# ---------------------------------------------------------------- geometry
W, L, R = 18.0, 50.0, 2.5
cL, cR = (0.0, 30.0), (W, 20.0)    # notch centres (20 mm from top / bottom)


def build_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, W, L)
    d1 = occ.addDisk(*cL, 0, R, R)
    d2 = occ.addDisk(*cR, 0, R, R)
    occ.cut([(2, rect)], [(2, d1), (2, d2)])
    occ.synchronize()
    s = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [e[1] for e in s], 1)
    f = gmsh.model.mesh.field
    box = f.add("Box")
    f.setNumber(box, "VIn", args.hf)
    f.setNumber(box, "VOut", args.hc)
    f.setNumber(box, "XMin", 0.0); f.setNumber(box, "XMax", W)
    f.setNumber(box, "YMin", 17.0); f.setNumber(box, "YMax", 33.0)
    f.setNumber(box, "Thickness", 5.0)
    f.setAsBackgroundMesh(box)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    msh, _, _ = io.gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return msh


msh = build_mesh()
ncell = msh.topology.index_map(2).size_local
print(f"cells={ncell}, vertices={msh.geometry.x.shape[0]}", flush=True)

V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
Q = fem.functionspace(msh, ("Lagrange", 1))
qdeg = 1
Qe_s = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(), degree=qdeg)
Qe_v = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(4,), degree=qdeg)
QS = fem.functionspace(msh, Qe_s)
QV = fem.functionspace(msh, Qe_v)
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": qdeg})

u, d = fem.Function(V, name="u"), fem.Function(Q, name="d")
v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
du_t, dd_t = ufl.TrialFunction(V), ufl.TrialFunction(Q)

# state at t_n (quadrature)
epsp_n = fem.Function(QV)     # (xx, yy, zz, xy) plastic strain
p_n = fem.Function(QS)        # equivalent plastic strain
Hpsi_n = fem.Function(QS)     # max_[0,t] psi_e^+
Wp_n = fem.Function(QS)       # accumulated plastic work


def eps3(w):
    e = ufl.sym(ufl.grad(w))
    return ufl.as_tensor([[e[0, 0], e[0, 1], 0], [e[1, 0], e[1, 1], 0], [0, 0, 0]])


I3 = ufl.Identity(3)
ep_n = ufl.as_tensor([[epsp_n[0], epsp_n[3], 0], [epsp_n[3], epsp_n[1], 0], [0, 0, epsp_n[2]]])
pos = lambda x: 0.5 * (x + abs(x))


def constitutive(w):
    """Radial return (linear isotropic hardening, closed form, Eqs. 13-19)."""
    ee_tr = eps3(w) - ep_n
    s_tr = lmbda * ufl.tr(ee_tr) * I3 + 2 * mu * ee_tr
    sdev = ufl.dev(s_tr)
    q_tr = ufl.sqrt(1.5 * ufl.inner(sdev, sdev) + 1e-20)
    f_tr = q_tr - (sig_y + Hiso * p_n)
    dp = pos(f_tr) / (3 * mu + Hiso)
    nflow = 1.5 * sdev / q_tr
    sig = s_tr - 2 * mu * dp * nflow
    ee = ee_tr - dp * nflow
    return sig, ee, dp, nflow


def psi_plus(ee):
    """Eq. (31): spectral split of the elastic strain (plane strain, 3D tensor)."""
    a, b, c = ee[0, 0], ee[1, 1], ee[0, 1]
    m = 0.5 * (a + b)
    r = ufl.sqrt((0.5 * (a - b)) ** 2 + c ** 2 + 1e-30)
    e1, e2, e3 = m + r, m - r, ee[2, 2]
    return 0.5 * lmbda * pos(e1 + e2 + e3) ** 2 + mu * (pos(e1) ** 2 + pos(e2) ** 2 + pos(e3) ** 2)


sig, ee, dp, nflow = constitutive(u)
g = (1 - d) ** 2 + k_res
# Eq. (30): H = max psi_e^+ + int sigma_eq d(eps_p)  (exact for linear hardening)
Wp_inc = (sig_y + Hiso * (p_n + 0.5 * dp)) * dp
Hdrive = ufl.max_value(Hpsi_n, psi_plus(ee)) + Wp_n + Wp_inc

R_u = ufl.inner(g * sig, eps3(v)) * dx
R_d = (-2 * (1 - d) * Hdrive * q + Gc / lc * d * q
       + Gc * lc * ufl.dot(ufl.grad(d), ufl.grad(q))) * dx
F_u, F_d = fem.form(R_u), fem.form(R_d)
# Eq. (42): uncoupled symmetric initial estimate diag(K_uu, K_dd)
J_uu = fem.form(ufl.derivative(R_u, u, du_t))
J_dd = fem.form(ufl.derivative(R_d, d, dd_t))

# ---------------------------------------------------------------- BCs
fdim = 1
bot = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
top = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], L))
dofs_bot = fem.locate_dofs_topological(V, fdim, bot)
dofs_top_x = fem.locate_dofs_topological(V.sub(0), fdim, top)
dofs_top_y = fem.locate_dofs_topological(V.sub(1), fdim, top)
U_top = fem.Constant(msh, PETSc.ScalarType(0.0))
bcs_u = [fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), dofs_bot, V),
         fem.dirichletbc(PETSc.ScalarType(0.0), dofs_top_x, V.sub(0)),
         fem.dirichletbc(U_top, dofs_top_y, V.sub(1))]
# for blocked V, locate_dofs_topological(V, ...) returns block indices
bc_dofs_u = np.unique(np.concatenate([2 * dofs_bot, 2 * dofs_bot + 1, dofs_top_x, dofs_top_y]))
top_y_dofs = dofs_top_y

nu_ = u.x.array.size
nd_ = d.x.array.size

# ---------------------------------------------------------------- linear algebra
A_uu = fem.petsc.create_matrix(J_uu)
A_dd = fem.petsc.create_matrix(J_dd)
b_u = fem.petsc.create_vector(F_u)
b_d = fem.petsc.create_vector(F_d)


def make_ksp(A):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    return ksp


ksp_u, ksp_d = make_ksp(A_uu), make_ksp(A_dd)
xu, xd = A_uu.createVecRight(), A_dd.createVecRight()
ru, rd = A_uu.createVecRight(), A_dd.createVecRight()


def assemble_K0():
    A_uu.zeroEntries()
    assemble_matrix(A_uu, J_uu, bcs=bcs_u)
    A_uu.assemble()
    A_dd.zeroEntries()
    assemble_matrix(A_dd, J_dd)
    A_dd.assemble()
    ksp_u.setOperators(A_uu)
    ksp_d.setOperators(A_dd)
    ksp_u.setUp()
    ksp_d.setUp()


def residual(z):
    """g(z) with Dirichlet rows zeroed; also returns the top reaction."""
    u.x.array[:] = z[:nu_]
    d.x.array[:] = z[nu_:]
    u.x.scatter_forward(); d.x.scatter_forward()
    with b_u.localForm() as loc:
        loc.set(0.0)
    assemble_vector(b_u, F_u)
    b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    gu = b_u.array.copy()
    react = gu[top_y_dofs].sum()
    gu[bc_dofs_u] = 0.0
    with b_d.localForm() as loc:
        loc.set(0.0)
    assemble_vector(b_d, F_d)
    b_d.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    gd = b_d.array.copy()
    return np.concatenate([gu, gd]), react


def apply_H0(gvec):
    ru.array[:] = gvec[:nu_]
    rd.array[:] = gvec[nu_:]
    ksp_u.solve(ru, xu)
    ksp_d.solve(rd, xd)
    return np.concatenate([xu.array, xd.array])


def apply_H(gvec, S, Y):
    """Two-loop recursion of the BFGS inverse update, Eq. (41)."""
    qv = gvec.copy()
    alphas = []
    for s, y in reversed(list(zip(S, Y))):
        rho = 1.0 / y.dot(s)
        a = rho * s.dot(qv)
        alphas.append(a)
        qv -= a * y
    r = apply_H0(qv)
    for (s, y), a in zip(zip(S, Y), reversed(alphas)):
        rho = 1.0 / y.dot(s)
        b = rho * y.dot(r)
        r += (a - b) * s
    return r


def project(zz, lb):
    """Admissible damage: d_n <= d <= 1 (irreversibility and upper bound)."""
    zz[nu_:] = np.clip(zz[nu_:], lb, args.dub)
    return zz


def reduced(gv, zz, lb):
    """Zero the damage residual on active bounds (projected/active-set gradient)."""
    gr = gv.copy()
    dz_ = zz[nu_:]
    gd = gr[nu_:]
    act = ((dz_ <= lb + 1e-12) & (gd > 0)) | ((dz_ >= args.dub - 1e-12) & (gd < 0))
    gd[act] = 0.0
    return gr


def ok_res(gv, n0):
    return np.all(np.isfinite(gv)) and np.linalg.norm(gv) < 1e3 * n0


def line_search(z, s, g0, lb):
    """Secant line search on phi(a) = s.g(P(z + a s)) (Matthies & Strang)."""
    phi0 = s.dot(g0)
    n0 = np.linalg.norm(g0)
    a = 1.0
    for _ in range(20):   # backtrack on non-finite / exploding residuals
        zt = project(z + a * s, lb)
        gnew, react = residual(zt)
        gnew = reduced(gnew, zt, lb)
        if ok_res(gnew, n0):
            break
        a *= 0.5
    if a < 1.0 or phi0 >= 0:
        return zt, gnew, react
    phi1 = s.dot(gnew)
    if abs(phi1) <= 0.8 * abs(phi0):
        return zt, gnew, react
    a0, f0, a1, f1 = 0.0, phi0, 1.0, phi1
    for _ in range(6):
        if abs(f1 - f0) < 1e-30:
            break
        a2 = float(np.clip(a1 - f1 * (a1 - a0) / (f1 - f0), 0.1, 2.0))
        z2 = project(z + a2 * s, lb)
        g2, r2 = residual(z2)
        g2 = reduced(g2, z2, lb)
        if not ok_res(g2, n0):
            break
        f2 = s.dot(g2)
        a0, f0, a1, f1 = a1, f1, a2, f2
        zt, gnew, react = z2, g2, r2
        if abs(f2) <= 0.8 * abs(phi0):
            break
    return zt, gnew, react


def bfgs_step():
    """Solve g(z)=0 for the current load level with monolithic BFGS."""
    z = np.concatenate([u.x.array, d.x.array])
    lb = np.zeros_like(d.x.array)   # irreversibility is carried by the history H (Eq. 30)
    for b in bcs_u:   # impose the new boundary displacement
        b.set(z[:nu_])
    assemble_K0()
    gk, react = residual(z)
    gk = reduced(gk, z, lb)
    S, Y = [], []
    n0u = max(np.linalg.norm(gk[:nu_]), 1e-8)
    n0d = max(np.linalg.norm(gk[nu_:]), 1e-8)
    nrefresh = 0
    z_prev = z.copy()
    last_refresh = 0
    for it in range(1, args.maxit + 1):
        s = -apply_H(gk, S, Y)
        smax = np.abs(s[nu_:]).max()
        if smax > args.ddmax:          # step-length control on the damage update
            s *= args.ddmax / smax
        znew, gnew, react = line_search(z, s, gk, lb)
        dz = znew - z
        dg = gnew - gk
        z, gk = znew, gnew
        if dz.dot(dg) > 1e-10 * np.linalg.norm(dz) * np.linalg.norm(dg):
            S.append(dz); Y.append(dg)
        eu = np.linalg.norm(gk[:nu_])
        ed = np.linalg.norm(gk[nu_:])
        if args.verbose:
            print(f"    it {it:3d} |g_u|={eu:.3e}/{n0u:.3e} |g_d|={ed:.3e}/{n0d:.3e} "
                  f"dmax={z[nu_:].max():.4f} pairs={len(S)}", flush=True)
        if (eu <= args.tol * n0u or eu < 1e-6) and (ed <= args.tol * n0d or ed < 1e-8):
            residual(z)
            return it, nrefresh, react, True
        # blow-up test, with floors so that an already-equilibrated field is not flagged
        diverging = not np.isfinite(eu + ed) or eu > args.rdiv * max(n0u, 1.0) \
            or ed > args.rdiv * max(n0d, 0.1)
        if diverging or it - last_refresh >= args.refresh:
            # refresh the initial estimate K0 (Eq. 42) and restart the BFGS updates,
            # from the previous iterate if the residual is blowing up
            if diverging:
                z = z_prev.copy()
            gk, react = residual(z)
            gk = reduced(gk, z, lb)
            assemble_K0()
            S, Y = [], []
            nrefresh += 1
            last_refresh = it
        z_prev = z.copy()
    residual(z)
    return it, nrefresh, react, False


# ---------------------------------------------------------------- state update
expr_epsp = fem.Expression(ufl.as_vector([ep_n[0, 0] + dp * nflow[0, 0], ep_n[1, 1] + dp * nflow[1, 1],
                                          ep_n[2, 2] + dp * nflow[2, 2], ep_n[0, 1] + dp * nflow[0, 1]]),
                           QV.element.interpolation_points())
expr_p = fem.Expression(p_n + dp, QS.element.interpolation_points())
expr_H = fem.Expression(ufl.max_value(Hpsi_n, psi_plus(ee)), QS.element.interpolation_points())
expr_W = fem.Expression(Wp_n + Wp_inc, QS.element.interpolation_points())
tmp_v, tmp_p, tmp_H, tmp_W = fem.Function(QV), fem.Function(QS), fem.Function(QS), fem.Function(QS)


def update_state():
    tmp_v.interpolate(expr_epsp); tmp_p.interpolate(expr_p)
    tmp_H.interpolate(expr_H); tmp_W.interpolate(expr_W)
    epsp_n.x.array[:] = tmp_v.x.array; p_n.x.array[:] = tmp_p.x.array
    Hpsi_n.x.array[:] = tmp_H.x.array; Wp_n.x.array[:] = tmp_W.x.array


def solve_increment(Ua, Ub, level):
    """BFGS solve of one load increment; if it fails, bisect the increment."""
    z_u, z_d = u.x.array.copy(), d.x.array.copy()
    U_top.value = Ub
    its, nref, react, ok = bfgs_step()
    if ok or level >= args.maxcut:
        update_state()
        return its, nref, react, ok, 1
    u.x.array[:] = z_u; d.x.array[:] = z_d
    Um = 0.5 * (Ua + Ub)
    i1, r1, _, ok1, n1 = solve_increment(Ua, Um, level + 1)
    i2, r2, react, ok2, n2 = solve_increment(Um, Ub, level + 1)
    return its + i1 + i2, nref + r1 + r2, react, ok1 and ok2, n1 + n2


# ---------------------------------------------------------------- output
xyz = msh.geometry.x[:, :2].copy()
cells = msh.geometry.dofmap.copy()
np.savez(os.path.join(args.out, "mesh.npz"), x=xyz, cells=cells)
# map P1 dof -> geometry node (identical for P1 on a P1 mesh, but be safe)
Qdofmap = Q.dofmap.list
d_geom_order = np.empty(xyz.shape[0], dtype=np.int64)
d_geom_order[cells.reshape(-1)] = Qdofmap.reshape(-1)

snap_targets = [0.150, 0.180, 0.186, 0.189, 0.192, 0.195, 0.21, 0.30]
saved = set()
history = [(0.0, 0.0, 0, 0, 0.0, 0)]
nsteps = int(round(args.umax / args.du))
t0 = time.time()
n_start = 1
state_funcs = dict(u=u, d=d, epsp=epsp_n, p=p_n, Hpsi=Hpsi_n, Wp=Wp_n)
if args.restart:
    ck = np.load(args.restart)
    for k, f_ in state_funcs.items():
        f_.x.array[:] = ck[k]
    history = [tuple(r) for r in ck["history"]]
    n_start = int(ck["n"]) + 1
    saved = {t for t in snap_targets if t <= history[-1][0] + 1e-9}
log = open(os.path.join(args.out, "log.txt"), "a" if args.restart else "w")
for n in range(n_start, nsteps + 1):
    Ut = n * args.du
    ts = time.time()
    its, nref, react, ok, nsub = solve_increment(Ut - args.du, Ut, 0)
    dmax = d.x.array.max()
    history.append((Ut, react, its, nref, dmax, nsub))
    msg = (f"step {n:4d} u={Ut:.4f} F={react:9.2f} N it={its:3d} K0-refresh={nref} "
           f"dmax={dmax:.3f} conv={ok} substeps={nsub} ({time.time() - ts:.1f}s)")
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    np.savez(os.path.join(args.out, f"checkpoint_{n % 2}.npz"), n=n, history=np.array(history),
             **{k: f_.x.array for k, f_ in state_funcs.items()})
    np.savetxt(os.path.join(args.out, "force_displacement.csv"), np.array(history),
               delimiter=",", header="u_mm,F_N,iterations,K0_refreshes,dmax,substeps", comments="")
    for tgt in snap_targets:
        if tgt not in saved and Ut >= tgt - 1e-9:
            saved.add(tgt)
            np.savez(os.path.join(args.out, f"damage_u{tgt:.3f}.npz"),
                     d=d.x.array[d_geom_order], u=Ut,
                     disp=u.x.array.reshape(-1, 2)[d_geom_order])
log.close()
with open(os.path.join(args.out, "run_info.json"), "w") as fh:
    json.dump(dict(vars(args), cells=int(ncell), wall_time_s=time.time() - t0,
                   E=E, nu=nu, Eh=Eh, sig_y=sig_y, lc=lc), fh, indent=2)
print("done", time.time() - t0, flush=True)
