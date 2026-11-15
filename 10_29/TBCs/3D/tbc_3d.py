"""
3D thermal shock (quenching) of TBCs (Sec. 3.4.4 of 10_29/8.md) reproduced with FEniCSx (dolfinx 0.9).

Model (Fig. 19, 3D): x = length (L = 2500 um), y = thickness (BC 0..100 um, TC 100..100+h_TC um),
z = width w (50 or 100 um). By symmetry only the half 0 <= x <= L/2 is computed (plane x = L/2:
u_x = 0, adiabatic). Initial 1000 C, quench to 10 C on the thermal shock surfaces:
  case 1: upper surface (y = H)
  case 2: upper + left/right ends (x = 0, L)
  case 3: upper + left/right + front/back (z = 0, w)
Other surfaces adiabatic. Mechanics: u_x = 0 on x = L/2; bottom --bottom clamped (default, u = 0)
or roller (u_y = 0, u_z = 0 at one bottom node).
TC/BC interface: band of TC cells (thickness --band, default ell) with G_c^{TC/BC}.

Numerics as in ../2D/tbc_2d.py (AT1 phase field, Amor split, staggered scheme), with a
CG + GAMG solver with rigid-body near-nullspace (or MUMPS, --solver lu) for the 3D elasticity problem.
"""
import argparse
import json
import os
import time

import numpy as np
import basix.ufl
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from mpi4py import MPI
from petsc4py import PETSc

p = argparse.ArgumentParser()
p.add_argument("--case", type=int, choices=[1, 2, 3], default=1)
p.add_argument("--w", type=float, default=100.0, help="width [um]")
p.add_argument("--Gi", type=float, default=0.001, help="TC/BC interface fracture toughness [N/mm]")
p.add_argument("--hTC", type=float, default=100.0, help="TC thickness [um]")
p.add_argument("--h", type=float, default=10.0, help="element size [um]")
p.add_argument("--ell", type=float, default=None, help="phase-field length [um] (default 2h)")
p.add_argument("--band", type=float, default=None, help="interface band thickness [um] (default ell)")
p.add_argument("--bottom", choices=["clamped", "roller"], default="clamped",
               help="bottom of BC: clamped (u = 0, rigid substrate) or roller (u_y = 0 + pin)")
p.add_argument("--T", type=float, default=0.03, help="total time [s]")
p.add_argument("--noise", type=float, default=0.02)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--solver", choices=["lu", "gamg"], default="gamg", help="elasticity solver")
p.add_argument("--maxit", type=int, default=30)
p.add_argument("--tol", type=float, default=1e-3)
p.add_argument("--outdir", type=str, required=True)
a = p.parse_args()

comm = MPI.COMM_WORLD
assert comm.size == 1, "run in serial"
os.makedirs(a.outdir, exist_ok=True)
UM = 1e-6

MAT = {  # E, alpha, rho, kappa, cv, Gc  (Table 6, SI)
    "TC": (11.7e9, 10.04e-6, 5650.0, 1.05, 450.0, 0.008e3),
    "BC": (71.9e9, 17.65e-6, 7320.0, 4.3, 780.0, 0.3e3),
}
nu = 0.25  # 3D bond-based PD
Gi = a.Gi * 1e3
theta0, theta_s = 1000.0, 10.0
L, hBC, hTC, W = 2500 * UM, 100 * UM, a.hTC * UM, a.w * UM
Lh = L / 2
Htot = hBC + hTC
h = a.h * UM
ell = (a.ell if a.ell is not None else 2 * a.h) * UM
band = (a.band if a.band is not None else ell / UM) * UM
kres = 1e-6

nx, ny, nz = int(round(Lh / h)), int(round(Htot / h)), int(round(W / h))
msh = mesh.create_box(comm, [[0, 0, 0], [Lh, Htot, W]], [nx, ny, nz], mesh.CellType.hexahedron)
gdim, fdim = 3, 2

V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
S = fem.functionspace(msh, ("Lagrange", 1))
DG0 = fem.functionspace(msh, ("DG", 0))
qdeg = 2
Q = fem.functionspace(msh, basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(), degree=qdeg))
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": qdeg})

yc = DG0.tabulate_dof_coordinates()[:, 1]
is_tc = yc > hBC
is_band = is_tc & (yc < hBC + band)
rng = np.random.default_rng(a.seed)
fields = {}
for k, name in enumerate(["E", "alpha", "rho", "kappa", "cv", "Gc"]):
    f = fem.Function(DG0, name=name)
    f.x.array[:] = np.where(is_tc, MAT["TC"][k], MAT["BC"][k])
    fields[name] = f
fields["Gc"].x.array[is_band] = Gi
fields["Gc"].x.array[:] *= 1.0 + a.noise * (2 * rng.random(yc.size) - 1)
Ef, alf, Gc = fields["E"], fields["alpha"], fields["Gc"]
mu = Ef / (2 * (1 + nu))
lmbda = Ef * nu / ((1 + nu) * (1 - 2 * nu))
Kb = lmbda + 2 * mu / 3

u = fem.Function(V, name="u")
theta = fem.Function(S, name="theta")
theta_n = fem.Function(S)
d = fem.Function(S, name="damage")
d_n = fem.Function(S)
d_it = fem.Function(S)
H_n = fem.Function(Q)
H = fem.Function(Q)
Hpsi = fem.Function(Q)
theta.x.array[:] = theta0
theta_n.x.array[:] = theta0
H_n.interpolate(fem.Expression(3 * Gc / (16 * ell), Q.element.interpolation_points()))
H.x.array[:] = H_n.x.array


# ---------------- boundary conditions ----------------
def shock_marker(x):
    m = np.isclose(x[1], Htot)
    if a.case >= 2:
        m |= np.isclose(x[0], 0.0)
    if a.case >= 3:
        m |= np.isclose(x[2], 0.0) | np.isclose(x[2], W)
    return m


shock_dofs = fem.locate_dofs_geometrical(S, shock_marker)
bc_T = fem.dirichletbc(PETSc.ScalarType(theta_s), shock_dofs, S)
f_bot = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
f_sym = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], Lh))
v_pin = mesh.locate_entities_boundary(
    msh, 0, lambda x: np.isclose(x[1], 0.0) & np.isclose(x[0], Lh) & np.isclose(x[2], 0.0))
bcs_roller = [fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(V.sub(1), fdim, f_bot), V.sub(1)),
         fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(V.sub(0), fdim, f_sym), V.sub(0)),
         fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(V.sub(2), 0, v_pin), V.sub(2))]
bcs_u = bcs_roller if a.bottom == "roller" else [
    fem.dirichletbc(np.zeros(gdim), fem.locate_dofs_topological(V, fdim, f_bot), V),
    bcs_roller[1]]


def eps(w):
    return ufl.sym(ufl.grad(w))


def sig_lin(e):
    return lmbda * ufl.tr(e) * ufl.Identity(gdim) + 2 * mu * e


def psi_pos(e):
    tr = ufl.tr(e)
    dev = e - tr / 3 * ufl.Identity(gdim)
    return 0.5 * Kb * (0.5 * (tr + abs(tr))) ** 2 + mu * ufl.inner(dev, dev)


g = (1 - d) ** 2 + kres
eps_th = alf * (theta - theta0) * ufl.Identity(gdim)
eps_e = eps(u) - eps_th

# ---------------- heat conduction ----------------
Tt, Tw = ufl.TrialFunction(S), ufl.TestFunction(S)
a_K = fem.form(fields["kappa"] * ufl.inner(ufl.grad(Tt), ufl.grad(Tw)) * dx)
mL = assemble_vector(fem.form(fields["rho"] * fields["cv"] * Tw * dx)).array.copy()
is_bcT = np.zeros(mL.size, bool)
is_bcT[shock_dofs] = True
A_T = assemble_matrix(a_K, bcs=[bc_T], diagonal=0.0)
A_T.assemble()
K_T = A_T.copy()
b_T = PETSc.Vec().createSeq(mL.size)
dg = PETSc.Vec().createSeq(mL.size)
ksp_T = PETSc.KSP().create(comm)
ksp_T.setType("cg")
ksp_T.getPC().setType("hypre")
ksp_T.setTolerances(rtol=1e-10)
cur_dt = [None]


def solve_heat(dt):
    if cur_dt[0] != dt:
        K_T.copy(A_T)
        dg.array[:] = np.where(is_bcT, 1.0, mL / dt)
        A_T.setDiagonal(dg, addv=PETSc.InsertMode.ADD_VALUES)
        ksp_T.setOperators(A_T)
        cur_dt[0] = dt
    b_T.array[:] = mL / dt * theta_n.x.array
    apply_lifting(b_T, [a_K], bcs=[[bc_T]])
    set_bc(b_T, [bc_T])
    ksp_T.solve(b_T, theta.x.petsc_vec)
    theta.x.scatter_forward()


# ---------------- mechanics: CG + GAMG (default) or MUMPS with rigid body modes ----------------
def rigid_body_modes(Vs):
    xv = Vs.tabulate_dof_coordinates()
    nn = xv.shape[0]
    bs = Vs.dofmap.index_map_bs
    basis = [np.zeros(nn * bs) for _ in range(6)]
    for i in range(3):
        basis[i][i::3] = 1.0
    x0, x1, x2 = xv[:, 0], xv[:, 1], xv[:, 2]
    basis[3][0::3], basis[3][1::3] = -x1, x0
    basis[4][0::3], basis[4][2::3] = x2, -x0
    basis[5][2::3], basis[5][1::3] = x1, -x2
    vecs = []
    for b in basis:
        v = PETSc.Vec().createWithArray(b.copy())
        vecs.append(v)
    ns = PETSc.NullSpace().create(vectors=vecs)
    return ns, vecs


ut, uw = ufl.TrialFunction(V), ufl.TestFunction(V)
a_u = fem.form(g * ufl.inner(sig_lin(eps(ut)), eps(uw)) * dx)
L_u = fem.form(g * ufl.inner(sig_lin(eps_th), eps(uw)) * dx)
A_u = assemble_matrix(a_u, bcs=bcs_u)
A_u.assemble()
nsp, _nsvecs = rigid_body_modes(V)
A_u.setNearNullSpace(nsp)
A_u.setBlockSize(3)
b_u = fem.petsc.create_vector(L_u)
ksp_u = PETSc.KSP().create(comm)
if a.solver == "lu":
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("lu")
    ksp_u.getPC().setFactorSolverType("mumps")
else:
    ksp_u.setType("cg")
    ksp_u.setTolerances(rtol=1e-8, max_it=2000)
    ksp_u.setInitialGuessNonzero(True)
    ksp_u.getPC().setType("gamg")
    opts = PETSc.Options()
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"
    opts["pc_gamg_threshold"] = 0.01
    ksp_u.setFromOptions()
n_lin = [0]


def solve_u():
    A_u.zeroEntries()
    assemble_matrix(A_u, a_u, bcs=bcs_u)
    A_u.assemble()
    with b_u.localForm() as loc:
        loc.set(0)
    assemble_vector(b_u, L_u)
    apply_lifting(b_u, [a_u], bcs=[bcs_u])
    set_bc(b_u, bcs_u)
    ksp_u.setOperators(A_u)
    ksp_u.solve(b_u, u.x.petsc_vec)
    n_lin[0] = max(n_lin[0], ksp_u.getIterationNumber())
    u.x.scatter_forward()


# ---------------- damage ----------------
dt_, dw = ufl.TrialFunction(S), ufl.TestFunction(S)
a_d = fem.form(2 * H * dt_ * dw * dx + 0.75 * Gc * ell * ufl.inner(ufl.grad(dt_), ufl.grad(dw)) * dx)
L_d = fem.form((2 * H - 3 * Gc / (8 * ell)) * dw * dx)
A_d = assemble_matrix(a_d)
A_d.assemble()
b_d = fem.petsc.create_vector(L_d)
ksp_d = PETSc.KSP().create(comm)
ksp_d.setType("cg")
ksp_d.getPC().setType("hypre")
ksp_d.setTolerances(rtol=1e-9)
dsol = fem.Function(S)
psi_expr = fem.Expression(psi_pos(eps_e), Q.element.interpolation_points())


def solve_d():
    Hpsi.interpolate(psi_expr)
    np.maximum(H_n.x.array, Hpsi.x.array, out=H.x.array)
    A_d.zeroEntries()
    assemble_matrix(A_d, a_d)
    A_d.assemble()
    with b_d.localForm() as loc:
        loc.set(0)
    assemble_vector(b_d, L_d)
    ksp_d.setOperators(A_d)
    ksp_d.solve(b_d, dsol.x.petsc_vec)
    d.x.array[:] = np.clip(np.maximum(d_n.x.array, dsol.x.array), 0.0, 1.0)


E_el_form = fem.form(g * 0.5 * ufl.inner(sig_lin(eps_e), eps_e) * dx)
E_f_form = fem.form(3 * Gc / 8 * (d / ell + ell * ufl.inner(ufl.grad(d), ufl.grad(d))) * dx)

# ---------------- time schedule & output ----------------
sched = [(2e-3, 1e-5), (1e-2, 5e-5), (5e-2, 2.5e-4), (np.inf, 2.5e-3)]
times = [0.0]
while times[-1] < a.T - 1e-12:
    step = next(s for tl, s in sched if times[-1] < tl - 1e-12)
    times.append(min(times[-1] + step, a.T))
times = np.array(times)
snap_targets = [s for s in (0.25e-3, 0.4875e-3, 1e-3, 2e-3, 3e-3, 5e-3, 10e-3, 20e-3, 30e-3, 0.1, 0.3)
                if s <= a.T + 1e-12]

xs = S.tabulate_dof_coordinates()
ix = np.rint(xs[:, 0] / h).astype(int)
iy = np.rint(xs[:, 1] / h).astype(int)
iz = np.rint(xs[:, 2] / h).astype(int)
j_int = int(round((hBC + 0.5 * band) / h))
j_bc = int(round(hBC / h))


def grid(arr):
    out = np.zeros((nz + 1, ny + 1, nx + 1), np.float32)
    out[iz, iy, ix] = arr
    return out


def metrics(dg_):
    plane = dg_[:, j_int, :]
    tc = dg_[:, j_bc + 1:, :]
    top_half = tc[:, tc.shape[1] // 2:, :]
    col = (top_half > 0.9).mean(axis=1).mean(axis=0) > 0.5  # transverse cracks through the width
    n_trans = int(np.sum(np.diff(np.r_[0, col.astype(int)]) == 1))
    return float((plane > 0.9).mean()), n_trans, float(dg_[:, : max(1, int(round((hBC - 2 * ell) / h))), :].max())


print(f"3D case {a.case}: w={a.w} um, Gi={Gi} N/m, hTC={a.hTC} um, half model {nx}x{ny}x{nz} "
      f"({nx*ny*nz} hex), h={h/UM:g} um, ell={ell/UM:g} um, steps={len(times)-1}", flush=True)
hist = {k: [0.0] for k in ("t", "E_el", "E_frac", "interface_damaged_fraction", "n_transverse", "max_d_BC")}
hist["n_iter"] = [0]
snaps = {"t": [], "theta": [], "d": []}
t0 = time.time()
k_snap = 0
for n in range(1, len(times)):
    t, dt = times[n], times[n] - times[n - 1]
    solve_heat(dt)
    n_lin[0] = 0
    for it in range(1, a.maxit + 1):
        d_it.x.array[:] = d.x.array
        solve_u()
        solve_d()
        err = np.abs(d.x.array - d_it.x.array).max()
        if err < a.tol:
            break
    d_n.x.array[:] = d.x.array
    H_n.x.array[:] = H.x.array
    theta_n.x.array[:] = theta.x.array
    dgrid = grid(d.x.array)
    fi, ntr, dbc = metrics(dgrid)
    for k, v in (("t", t), ("E_el", fem.assemble_scalar(E_el_form)), ("E_frac", fem.assemble_scalar(E_f_form)),
                 ("interface_damaged_fraction", fi), ("n_transverse", ntr), ("max_d_BC", dbc), ("n_iter", it)):
        hist[k].append(v)
    if k_snap < len(snap_targets) and t >= snap_targets[k_snap] - 1e-12:
        snaps["t"].append(t)
        snaps["theta"].append(grid(theta.x.array))
        snaps["d"].append(dgrid)
        k_snap += 1
    if n % 20 == 0 or n == len(times) - 1:
        print(f"step {n}/{len(times)-1} t={t*1e3:.3f} ms it={it} cg={n_lin[0]} err={err:.1e} iface={fi:.2f} "
              f"ntrans={ntr} maxdBC={dbc:.2f} Eel={hist['E_el'][-1]:.3e} wall={time.time()-t0:.0f}s", flush=True)

wall = time.time() - t0
with io.XDMFFile(comm, os.path.join(a.outdir, "final_fields.xdmf"), "w") as xf:
    xf.write_mesh(msh)
    xf.write_function(d, times[-1])
    xf.write_function(theta, times[-1])
np.savez_compressed(os.path.join(a.outdir, "snapshots.npz"), t=np.array(snaps["t"]), theta=np.array(snaps["theta"]),
                    d=np.array(snaps["d"]), L=L, W=W, hBC=hBC, hTC=hTC, h=h)
np.savez_compressed(os.path.join(a.outdir, "history.npz"), **{k: np.array(v) for k, v in hist.items()})
with open(os.path.join(a.outdir, "run_info.json"), "w") as f:
    json.dump({"bottom": a.bottom, "case": a.case, "w_um": a.w, "theta0_C": theta0, "theta_shock_C": theta_s, "Gi_N_per_mm": a.Gi,
               "hTC_um": a.hTC, "hBC_um": 100, "L_um": 2500, "computed_half_length_um": 1250,
               "h_um": h / UM, "ell_um": ell / UM, "band_um": band / UM, "nu": nu, "nx": nx, "ny": ny, "nz": nz,
               "n_cells_half": nx * ny * nz, "steps": len(times) - 1, "T_s": a.T, "Gc_noise": a.noise,
               "seed": a.seed, "wallclock_s": wall, "wallclock_per_step_s": wall / (len(times) - 1)}, f, indent=2)
print(f"done in {wall:.0f} s")
