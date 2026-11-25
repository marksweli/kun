"""CMAS-assisted fatigue crack growth in a porous APS top coat (Sec. 4.4 of 11.md).

Coupled diffusion - Walker viscoplasticity - AT2 phase field, staggered scheme,
FEniCSx/dolfinx 0.9.  Units: length um, stress MPa, time s, energy density MJ/m^3,
fracture energy MPa*um = J/m^2, diffusivity um^2/s.

Steps
  1. random porous 100x100 um top coat (gmsh, h = 1 um)
  2. CMAS penetration: dC/dt = D lap C in the solid, C = 4 mol% on the top edge,
     zero flux elsewhere (incl. pore surfaces), t = 10 min ("Pre-CMAS")
  3. 10 strain-controlled cycles on the right edge (range 3 %, R = 0.1, 0.5 %/s)
     with g_f = (1 - chi*theta(C)) g_f0, theta = C/(C + K_L)
"""
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import gmsh
import ufl
from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, apply_lifting,
                               set_bc, LinearProblem)

comm = MPI.COMM_WORLD

# ----------------------------------------------------------------- parameters
P = dict(
    Lx=100.0, Ly=100.0, h=1.0,            # model 0.1 x 0.1 mm, mesh 1 um
    porosity=0.09, dmean=6.0, dstd=2.5,   # TP-CLD chord (diameter) ~ N(6, 2.5) um
    dmin=1.5, seed=1,
    E=11.5e3, nu=0.12,                    # Table 1 (MPa)
    K1=59.3, n=1.23,                      # Walker drag stress / exponent (Table 1)
    edot0=1.0,                            # reference rate of Eq. (10) (1/s)
    Gc0=28.8, lc=2.0,                     # J/m^2, um (calibrated, see README)
    chi=0.66, KL=1.78,                    # Eq. (9); K_L = exp(-dg_b/RT) in mol%
    D=5.0, C0=4.0, t_cmas=600.0, dt_c=2.0,  # 5e-12 m^2/s = 5 um^2/s, 10 min
    erange=0.03, R=0.1, rate=0.005, ncyc=10, de=2.5e-4,
    kappa=1e-6, dfail=0.95, out="results",
)
if len(sys.argv) > 1:
    P.update(json.loads(sys.argv[1]))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), P["out"])
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------ geometry / mesh
def make_pores(p):
    rng = np.random.default_rng(p["seed"])
    target = p["porosity"] * p["Lx"] * p["Ly"]
    pores, area, tries = [], 0.0, 0
    while area < target and tries < 200000:
        tries += 1
        dia = rng.normal(p["dmean"], p["dstd"])
        if dia < p["dmin"]:
            continue
        r = 0.5 * dia
        gap = 1.0
        x = rng.uniform(r + gap, p["Lx"] - r - gap)
        y = rng.uniform(r + gap, p["Ly"] - r - gap)
        if any(np.hypot(x - a, y - b) < r + c + gap for a, b, c in pores):
            continue
        pores.append((x, y, r))
        area += np.pi * r * r
    return pores


def build_mesh(p):
    pores = make_pores(p)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("topcoat")
    occ = gmsh.model.occ
    box = occ.addRectangle(0, 0, 0, p["Lx"], p["Ly"])
    disks = [(2, occ.addDisk(x, y, 0, r, r)) for x, y, r in pores]
    occ.cut([(2, box)], disks)
    occ.synchronize()
    surf = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surf], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", p["h"])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", p["h"])
    gmsh.model.mesh.generate(2)
    from dolfinx.io import gmshio
    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return msh, pores


t0 = time.time()
msh, pores = build_mesh(P)
tdim = msh.topology.dim
ncell = msh.topology.index_map(tdim).size_local
area_solid = comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx)))
por = 1 - area_solid / (P["Lx"] * P["Ly"])
print(f"pores={len(pores)} porosity={por:.4f} cells={ncell}", flush=True)
np.savetxt(os.path.join(OUT, "pores.csv"), np.array(pores), delimiter=",",
           header="x_um,y_um,r_um", comments="")

x = ufl.SpatialCoordinate(msh)
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 1})


def locate(f):
    return dmesh.locate_entities_boundary(msh, 0, f)


# ------------------------------------------------------------ CMAS diffusion
Vc = fem.functionspace(msh, ("Lagrange", 1))
C, Cn = fem.Function(Vc, name="C"), fem.Function(Vc)
c_, w = ufl.TrialFunction(Vc), ufl.TestFunction(Vc)
top_dofs = fem.locate_dofs_topological(
    Vc, 0, locate(lambda X: np.isclose(X[1], P["Ly"])))
bc_c = fem.dirichletbc(PETSc.ScalarType(P["C0"]), top_dofs, Vc)
dtc = P["dt_c"]
ac = (c_ * w + dtc * P["D"] * ufl.dot(ufl.grad(c_), ufl.grad(w))) * ufl.dx
Lc = Cn * w * ufl.dx
diff = LinearProblem(ac, Lc, bcs=[bc_c], u=C,
                     petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
xc = Vc.tabulate_dof_coordinates()


def depth_profile(f, nb=34):
    yb = np.linspace(0, P["Ly"], nb + 1)
    depth = P["Ly"] - xc[:, 1]
    idx = np.clip(np.digitize(depth, yb) - 1, 0, nb - 1)
    prof = np.array([f.x.array[idx == i].mean() for i in range(nb)])
    return 0.5 * (yb[1:] + yb[:-1]), prof


profiles = {}
nsteps_c = int(round(P["t_cmas"] / dtc))
for k in range(1, nsteps_c + 1):
    diff.solve()
    Cn.x.array[:] = C.x.array
    tmin = k * dtc / 60
    if np.isclose(tmin, [1, 5, 10]).any():
        profiles[int(round(tmin))] = depth_profile(C)
bot = C.x.array[np.isclose(xc[:, 1], 0.0)]
print(f"CMAS 10 min: bottom C min/mean = {bot.min():.3f}/{bot.mean():.3f} mol%",
      flush=True)
with io.XDMFFile(comm, os.path.join(OUT, "cmas_10min.xdmf"), "w") as f:
    f.write_mesh(msh)
    f.write_function(C)
np.savetxt(os.path.join(OUT, "cmas_profile.csv"),
           np.column_stack([profiles[1][0]] + [profiles[m][1] for m in (1, 5, 10)]),
           delimiter=",", header="depth_um,C_1min,C_5min,C_10min", comments="")

# ----------------------------------------------- mechanics + phase-field set-up
E, nu = P["E"], P["nu"]
G = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
Kb = lam + 2 * G / 3
s2 = np.sqrt(2.0)
mvec = np.array([1.0, 1.0, 1.0, 0.0])
Idev = np.eye(4) - np.outer(mvec, mvec) / 3
Cel = 2 * G * np.eye(4) + lam * np.outer(mvec, mvec)

V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
Vd = fem.functionspace(msh, ("Lagrange", 1))
u, du, v = fem.Function(V, name="u"), ufl.TrialFunction(V), ufl.TestFunction(V)
d, dd, q = fem.Function(Vd, name="d"), ufl.TrialFunction(Vd), ufl.TestFunction(Vd)

qe4 = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(4,), degree=1)
qe44 = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(4, 4), degree=1)
qe1 = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(), degree=1)
sig_q = fem.Function(fem.functionspace(msh, qe4))
Ct_q = fem.Function(fem.functionspace(msh, qe44))
H_q = fem.Function(fem.functionspace(msh, qe1))


def eps_m(w_):
    e = ufl.sym(ufl.grad(w_))
    return ufl.as_vector([e[0, 0], e[1, 1], 0.0, s2 * e[0, 1]])


pts, _ = basix.make_quadrature(basix.CellType.triangle, 1)
cells = np.arange(ncell, dtype=np.int32)
eps_expr = fem.Expression(eps_m(u), pts)
theta = C / (C + P["KL"])
gf = P["Gc0"] * (1 - P["chi"] * theta)
gf_expr = fem.Expression(gf, pts)
gf_cell = gf_expr.eval(msh, cells).ravel()
deg = (1 - d) ** 2 + P["kappa"]

Fu = ufl.inner(deg * sig_q, eps_m(v)) * dx
Ju = ufl.inner(deg * ufl.dot(Ct_q, eps_m(du)), eps_m(v)) * dx
Fu_f, Ju_f = fem.form(Fu), fem.form(Ju)
lc = P["lc"]
ad = ((gf / lc + 2 * H_q) * dd * q + gf * lc * ufl.dot(ufl.grad(dd), ufl.grad(q))) * dx
Ld = 2 * H_q * q * dx
dprob = LinearProblem(ad, Ld, u=d, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

left = locate(lambda X: np.isclose(X[0], 0.0))
right = locate(lambda X: np.isclose(X[0], P["Lx"]))
corners = locate(lambda X: np.isclose(X[1], 0.0) & (np.isclose(X[0], 0.0) | np.isclose(X[0], P["Lx"])))
ux_right = fem.Constant(msh, PETSc.ScalarType(0.0))
bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(V.sub(0), 0, left), V.sub(0)),
       fem.dirichletbc(ux_right, fem.locate_dofs_topological(V.sub(0), 0, right), V.sub(0)),
       fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(V.sub(1), 0, corners), V.sub(1))]

# internal variables at the (single) quadrature point of each cell
ep = np.zeros((ncell, 4))     # viscoplastic strain (Mandel, incl. zz)
Gin = np.zeros(ncell)         # accumulated inelastic energy, Eq. (5)
H = np.zeros(ncell)           # history field, Eq. (6)/(16a)


def walker(eps, ep_old, dt):
    """Backward-Euler radial return for Eq. (10) (n1 = K2 = 0, back stress neglected)."""
    str_tr = (eps - ep_old) @ Cel.T
    s = str_tr @ Idev.T
    ns = np.linalg.norm(s, axis=1)
    seq = np.sqrt(1.5) * ns
    a = 3 * G * P["edot0"] * dt
    K, n = P["K1"], P["n"]
    # f(y) = seq - a y^n - K y = 0, y = (dp/(edot0 dt))^(1/n); concave -> Newton from y=0
    y = seq / K
    for _ in range(60):
        f = seq - a * y ** n - K * y
        y_new = y - f / (-a * n * y ** (n - 1) - K)
        y_new = np.maximum(y_new, 0.0)
        if np.max(np.abs(y_new - y) / (y + 1e-30)) < 1e-12:
            y = y_new
            break
        y = y_new
    dp = P["edot0"] * dt * y ** n
    act = (seq > 1e-12) & (dp > 0)
    N = np.zeros_like(s)
    N[act] = s[act] / ns[act, None]
    dep = np.sqrt(1.5) * dp[:, None] * N
    sig = str_tr - 2 * G * dep
    Ct = np.broadcast_to(Cel, (len(eps), 4, 4)).copy()
    if act.any():
        ya = y[act]
        h = K / (P["edot0"] * dt * n * ya ** (n - 1))
        Na = N[act]
        NN = Na[:, :, None] * Na[:, None, :]
        c1 = 6 * G ** 2 * dp[act] / seq[act]
        c2 = 6 * G ** 2 / (3 * G + h)
        Ct[act] -= c1[:, None, None] * (Idev[None] - NN) + c2[:, None, None] * NN
    return sig, Ct, dep


A = assemble_matrix(Ju_f, bcs=bcs)
A.assemble()
b = A.createVecRight()
ksp = PETSc.KSP().create(comm)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("petsc")
ddu = fem.Function(V)
bc_dofs = np.concatenate([bc.dof_indices()[0] for bc in bcs])


def solve_u(dt, tol=1e-8, maxit=25):
    """Newton for the displacement at fixed d; returns committed constitutive state."""
    for bc in bcs:
        bc.set(u.x.array)
    for it in range(maxit):
        eps = eps_expr.eval(msh, cells).reshape(ncell, 4)
        sig, Ct, dep = walker(eps, ep, dt)
        sig_q.x.array[:] = sig.ravel()
        Ct_q.x.array[:] = Ct.ravel()
        A.zeroEntries()
        assemble_matrix(A, Ju_f, bcs=bcs)
        A.assemble()
        with b.localForm() as bl:
            bl.set(0.0)
        assemble_vector(b, Fu_f)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b.scale(-1.0)
        b.array[bc_dofs[bc_dofs < b.local_size]] = 0.0
        rn = b.norm()
        if it == 0:
            r0 = max(rn, 1e-12)
        if rn < tol * max(r0, 1.0) or rn < 1e-10:
            return sig, dep, it
        ksp.setOperators(A)
        ksp.solve(b, ddu.x.petsc_vec)
        ddu.x.scatter_forward()
        u.x.array[:] += ddu.x.array
    print(f"  warning: Newton not converged, |r|={rn:.3e}", flush=True)
    return sig, dep, maxit


# loading path: 0 -> emax, then (emin <-> emax) x ncyc, cycle k ends at emin
emax = P["erange"] / (1 - P["R"])
emin = P["R"] * emax
path = [0.0, emax]
for k in range(P["ncyc"]):
    path += [emin, emax] if k < P["ncyc"] - 1 else [emin]
yc = msh.geometry.x[msh.geometry.dofmap][:, :, 1].mean(axis=1)
top = yc > P["Ly"] / 2
cell_area_f = fem.Function(fem.functionspace(msh, ("DG", 0)))
cell_area_f.x.array[:] = fem.assemble_vector(fem.form(
    ufl.TestFunction(cell_area_f.function_space) * ufl.dx)).array
cell_area = cell_area_f.x.array[:ncell]

d_dg = fem.functionspace(msh, ("DG", 0))
d_cell_f = fem.Function(d_dg, name="d_cell")
d_cell_expr = fem.Expression(d, d_dg.element.interpolation_points())

xdmf = io.XDMFFile(comm, os.path.join(OUT, "fatigue_10min.xdmf"), "w")
xdmf.write_mesh(msh)
xdmf.write_function(C, 0.0)

rec, cyc_rec = [], []
snap = {}
eps_app, tcur, step = 0.0, 0.0, 0
half = 0
for seg in range(1, len(path)):
    e0, e1 = path[seg - 1], path[seg]
    nst = max(1, int(round(abs(e1 - e0) / P["de"])))
    dt = abs(e1 - e0) / P["rate"] / nst
    for j in range(1, nst + 1):
        eps_app = e0 + (e1 - e0) * j / nst
        tcur += dt
        step += 1
        ux_right.value = eps_app * P["Lx"]
        sig, dep, nit = solve_u(dt)
        ep += dep
        Gin += np.einsum("ij,ij->i", dep, sig)
        epsv = eps_expr.eval(msh, cells).reshape(ncell, 4)
        epse = epsv - ep
        psi = 0.5 * np.einsum("ij,ij->i", epse, sig) + Gin
        H = np.maximum(H, psi)
        H_q.x.array[:] = H
        dprob.solve()
        d.x.array[:] = np.clip(d.x.array, 0.0, 1.0)
        F = fem.assemble_scalar(fem.form(deg * sig_q[0] * dx))
        sxx = F / (P["Lx"] * P["Ly"])
        rec.append((step, tcur, eps_app, sxx, d.x.array.max(), nit))
    # end of a segment that finishes at emin -> end of a cycle
    if seg >= 2 and np.isclose(e1, emin):
        ncy = seg // 2
        d_cell_f.interpolate(d_cell_expr)
        dc = d_cell_f.x.array[:ncell]
        fail = dc >= P["dfail"]
        nt, nb = int((fail & top).sum()), int((fail & ~top).sum())
        at, ab = cell_area[fail & top].sum(), cell_area[fail & ~top].sum()
        cyc_rec.append((ncy, nb, nt, ab, at))
        snap[ncy] = d.x.array.copy()
        xdmf.write_function(d, float(ncy))
        print(f"cycle {ncy:2d}: t={tcur:6.1f}s  dmax={d.x.array.max():.3f} "
              f"failed cells bottom/top = {nb}/{nt}, failed area = {ab:.1f}/{at:.1f} um^2 "
              f"({time.time() - t0:.0f}s)", flush=True)
xdmf.close()

np.savetxt(os.path.join(OUT, "stress_strain.csv"), np.array(rec), delimiter=",",
           header="step,t_s,strain,stress_MPa,dmax,newton_its", comments="")
np.savetxt(os.path.join(OUT, "failed_elements.csv"), np.array(cyc_rec),
           delimiter=",", fmt=["%d", "%d", "%d", "%.2f", "%.2f"],
           header="cycle,cells_bottom,cells_top,area_bottom_um2,area_top_um2", comments="")
np.savez(os.path.join(OUT, "fields.npz"), xy=msh.geometry.x[:, :2],
         tri=msh.geometry.dofmap, C=C.x.array, xd=Vd.tabulate_dof_coordinates()[:, :2],
         **{f"d{k}": v_ for k, v_ in snap.items()})
with open(os.path.join(OUT, "params.json"), "w") as f:
    json.dump(dict(P, n_pores=len(pores), porosity_mesh=por, n_cells=int(ncell),
                   emax=emax, emin=emin, runtime_s=time.time() - t0), f, indent=2)
print(f"done in {time.time() - t0:.0f}s", flush=True)
