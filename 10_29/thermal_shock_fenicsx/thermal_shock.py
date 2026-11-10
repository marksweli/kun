"""
Thermal shock (quenching) of a ceramic plate (Sec. 3.3 of 10_29/8.md) reproduced with FEniCSx (dolfinx 0.9).

Quarter model 25 mm x 5 mm (lower-left part of the 50 x 10 mm plate, Fig. 14):
  - left (x=0) and lower (y=0) edges: thermal shock, theta = theta* = 20 C
  - right (x=25 mm) edge: symmetry u_x = 0, adiabatic
  - top (y=5 mm) edge:   symmetry u_y = 0, adiabatic
Initial temperature theta0 (300/400/500/600 C). Plane stress. Material: Table 5.
Cracks do not affect heat transfer (one-way coupling, as in the paper).

Model: transient heat conduction (backward Euler, lumped capacity) + quasi-static
phase-field fracture (AT1 with elastic threshold, hybrid formulation with Amor split
driving force), staggered alternate minimisation in every time step.
"""
import argparse
import json
import os
import time

import numpy as np
import basix.ufl
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from mpi4py import MPI
from petsc4py import PETSc

p = argparse.ArgumentParser()
p.add_argument("--theta0", type=float, default=600.0, help="initial temperature [C]")
p.add_argument("--h", type=float, default=0.0625, help="element size [mm]")
p.add_argument("--ell", type=float, default=None, help="phase-field length [mm] (default 2h)")
p.add_argument("--T", type=float, default=0.3, help="total time [s]")
p.add_argument("--noise", type=float, default=0.02, help="relative random perturbation of Gc per cell")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--maxit", type=int, default=50, help="max staggered iterations per step")
p.add_argument("--tol", type=float, default=1e-3, help="staggered tolerance on max|d_k - d_k-1|")
p.add_argument("--outdir", type=str, required=True)
a = p.parse_args()

comm = MPI.COMM_WORLD
assert comm.size == 1, "run in serial"
os.makedirs(a.outdir, exist_ok=True)

# ---------------- Table 5 (SI) ----------------
E, kappa, Gc0, cv, rho, alpha = 370e9, 31.0, 42.47, 880.0, 3980.0, 7.5e-6
nu = 1.0 / 3.0  # bond-based PD, plane stress
theta_s = 20.0
theta0 = a.theta0
Lx, Ly = 25e-3, 5e-3
h = a.h * 1e-3
ell = (a.ell if a.ell is not None else 2 * a.h) * 1e-3
kres = 1e-7

mu = E / (2 * (1 + nu))
lmbda = E * nu / (1 - nu**2)  # plane stress
K2 = lmbda + mu
sig_c = np.sqrt(3 * Gc0 * E / (8 * ell))

nx, ny = int(round(Lx / h)), int(round(Ly / h))
msh = mesh.create_rectangle(comm, [[0, 0], [Lx, Ly]], [nx, ny], mesh.CellType.quadrilateral)
gdim = 2
fdim = 1

V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
S = fem.functionspace(msh, ("Lagrange", 1))
DG0 = fem.functionspace(msh, ("DG", 0))
qdeg = 2
Q = fem.functionspace(msh, basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(), degree=qdeg))
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": qdeg})

# random fracture toughness field (triggers symmetry breaking of the periodic crack pattern)
rng = np.random.default_rng(a.seed)
Gc = fem.Function(DG0)
Gc.x.array[:] = Gc0 * (1.0 + a.noise * (2 * rng.random(Gc.x.array.size) - 1))

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
edges = {
    "shock": lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0),
    "right": lambda x: np.isclose(x[0], Lx),
    "top": lambda x: np.isclose(x[1], Ly),
}
shock_dofs = fem.locate_dofs_geometrical(S, edges["shock"])
bc_T = fem.dirichletbc(PETSc.ScalarType(theta_s), shock_dofs, S)
f_r = mesh.locate_entities_boundary(msh, fdim, edges["right"])
f_t = mesh.locate_entities_boundary(msh, fdim, edges["top"])
bc_ux = fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(V.sub(0), fdim, f_r), V.sub(0))
bc_uy = fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(V.sub(1), fdim, f_t), V.sub(1))
bcs_u = [bc_ux, bc_uy]


def eps(w):
    return ufl.sym(ufl.grad(w))


def sig_lin(e):
    return lmbda * ufl.tr(e) * ufl.Identity(gdim) + 2 * mu * e


def psi_pos(e):
    tr = ufl.tr(e)
    dev = e - 0.5 * tr * ufl.Identity(gdim)
    return 0.5 * K2 * (0.5 * (tr + abs(tr))) ** 2 + mu * ufl.inner(dev, dev)


g = (1 - d) ** 2 + kres
eps_th = alpha * (theta - theta0) * ufl.Identity(gdim)
eps_e = eps(u) - eps_th

# ---------------- heat conduction: (M_L/dt + K) theta = M_L/dt theta_n ----------------
Tt, Tw = ufl.TrialFunction(S), ufl.TestFunction(S)
a_K = fem.form(kappa * ufl.inner(ufl.grad(Tt), ufl.grad(Tw)) * dx)
mL = assemble_vector(fem.form(rho * cv * Tw * dx)).array.copy()  # row-sum lumped capacity
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


# ---------------- mechanics (hybrid: fully degraded linear stress) ----------------
ut, uw = ufl.TrialFunction(V), ufl.TestFunction(V)
a_u = fem.form(g * ufl.inner(sig_lin(eps(ut)), eps(uw)) * dx)
L_u = fem.form(g * ufl.inner(sig_lin(eps_th), eps(uw)) * dx)
A_u = assemble_matrix(a_u, bcs=bcs_u)
A_u.assemble()
b_u = fem.petsc.create_vector(L_u)
ksp_u = PETSc.KSP().create(comm)
ksp_u.setType("preonly")
ksp_u.getPC().setType("lu")
ksp_u.getPC().setFactorSolverType("mumps")


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
    u.x.scatter_forward()


# ---------------- damage: 2H d - 3Gc ell/4 lap d = 2H - 3Gc/(8 ell), H >= psi_c ----------------
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

# ---------------- time schedule ----------------
sched = [(2e-3, 2e-5), (2e-2, 1e-4), (1e-1, 5e-4), (np.inf, 2e-3)]
times = [0.0]
while times[-1] < a.T - 1e-12:
    tc = times[-1]
    step = next(s for tl, s in sched if tc < tl - 1e-12)
    times.append(min(tc + step, a.T))
times = np.array(times)
snap_targets = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]
snap_targets = [s for s in snap_targets if s <= a.T + 1e-12]

xs = S.tabulate_dof_coordinates()[:, :2]
ix = np.rint(xs[:, 0] / h).astype(int)
iy = np.rint(xs[:, 1] / h).astype(int)


def grid(arr):
    out = np.zeros((ny + 1, nx + 1), np.float32)
    out[iy, ix] = arr
    return out


print(f"theta0={theta0} C, mesh {nx}x{ny} quads ({nx*ny} cells), h={h*1e3:.4g} mm, ell={ell*1e3:.4g} mm, "
      f"sigma_c(AT1)={sig_c/1e6:.1f} MPa, thermal stress E*alpha*dT/(1-nu)="
      f"{E*alpha*(theta0-theta_s)/(1-nu)/1e6:.0f} MPa, steps={len(times)-1}", flush=True)

hist = {"t": [0.0], "E_el": [0.0], "E_frac": [0.0], "E_kin": [0.0], "n_iter": [0]}
snaps = {"t": [], "theta": [], "d": []}
t0 = time.time()
k_snap = 0
for n in range(1, len(times)):
    t, dt = times[n], times[n] - times[n - 1]
    solve_heat(dt)
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
    hist["t"].append(t)
    hist["E_el"].append(fem.assemble_scalar(E_el_form))
    hist["E_frac"].append(fem.assemble_scalar(E_f_form))
    hist["E_kin"].append(0.0)  # quasi-static mechanics
    hist["n_iter"].append(it)
    if k_snap < len(snap_targets) and t >= snap_targets[k_snap] - 1e-12:
        snaps["t"].append(t)
        snaps["theta"].append(grid(theta.x.array))
        snaps["d"].append(grid(d.x.array))
        k_snap += 1
    if n % 20 == 0 or n == len(times) - 1:
        print(f"step {n}/{len(times)-1} t={t*1e3:.3f} ms it={it} err={err:.1e} "
              f"maxd={d.x.array.max():.3f} Eel={hist['E_el'][-1]:.3e} Ef={hist['E_frac'][-1]:.3e} "
              f"wall={time.time()-t0:.0f}s", flush=True)

wall = time.time() - t0
np.savez_compressed(os.path.join(a.outdir, "snapshots.npz"), t=np.array(snaps["t"]),
                    theta=np.array(snaps["theta"]), d=np.array(snaps["d"]), Lx=Lx, Ly=Ly, h=h)
np.savez_compressed(os.path.join(a.outdir, "history.npz"), **{k: np.array(v) for k, v in hist.items()})
with open(os.path.join(a.outdir, "run_info.json"), "w") as f:
    json.dump({"theta0_C": theta0, "theta_shock_C": theta_s, "h_mm": h * 1e3, "ell_mm": ell * 1e3,
               "nx": nx, "ny": ny, "n_cells": nx * ny, "steps": len(times) - 1, "T_s": a.T,
               "E": E, "nu": nu, "kappa": kappa, "Gc": Gc0, "cv": cv, "rho": rho, "alpha": alpha,
               "Gc_noise": a.noise, "seed": a.seed, "sigma_c_AT1_MPa": sig_c / 1e6,
               "wallclock_s": wall}, f, indent=2)
print(f"done in {wall:.0f} s")
