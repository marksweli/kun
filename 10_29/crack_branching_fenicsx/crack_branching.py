"""
Dynamic crack branching (Sec. 3.2 of 10_29/8.md) reproduced with FEniCSx (dolfinx 0.9).

Pre-cracked brittle plate 100 mm x 40 mm, pre-crack from (0, 20 mm) to (50 mm, 20 mm),
tensile traction sigma0 suddenly applied on top and bottom edges. Plane stress.
E = 72 GPa, rho = 2235 kg/m^3, Gc = 3.8 N/m, nu = 1/3 (bond-based PD, plane stress).

Model: phase-field fracture (AT1 with elastic threshold, Amor vol/dev split),
explicit central-difference time integration with lumped mass, staggered damage update.
"""
import argparse
import json
import os
import time

import numpy as np
import basix
import basix.ufl
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from mpi4py import MPI
from petsc4py import PETSc

p = argparse.ArgumentParser()
p.add_argument("--sigma", type=float, default=2.0, help="applied stress [MPa]")
p.add_argument("--h", type=float, default=0.25, help="element size [mm]")
p.add_argument("--ell", type=float, default=None, help="phase-field length [mm] (default 2h)")
p.add_argument("--dt", type=float, default=None, help="time step [ns] (default 80*h[mm])")
p.add_argument("--T", type=float, default=50.0, help="total time [us]")
p.add_argument("--nsnap", type=int, default=10, help="number of field snapshots")
p.add_argument("--outdir", type=str, required=True)
a = p.parse_args()

comm = MPI.COMM_WORLD
assert comm.size == 1, "run in serial"
os.makedirs(a.outdir, exist_ok=True)

# ---------------- parameters (SI) ----------------
E, rho, Gc, nu = 72e9, 2235.0, 3.8, 1.0 / 3.0
Lx, Ly, a0 = 0.1, 0.04, 0.05
h = a.h * 1e-3
ell = (a.ell if a.ell is not None else 2 * a.h) * 1e-3
dt = (a.dt if a.dt is not None else 80.0 * a.h) * 1e-9
T_end = a.T * 1e-6
sigma0 = a.sigma * 1e6
kres = 1e-7

mu = E / (2 * (1 + nu))
lmbda = E * nu / (1 - nu**2)  # plane stress
K2 = lmbda + mu  # 2D bulk modulus
cd = np.sqrt(E / (rho * (1 - nu**2)))
psi_c = 3 * Gc / (16 * ell)  # AT1 elastic threshold
sig_c = np.sqrt(3 * Gc * E / (8 * ell))

nx, ny = int(round(Lx / h)), int(round(Ly / h))
msh = mesh.create_rectangle(comm, [[0, 0], [Lx, Ly]], [nx, ny], mesh.CellType.quadrilateral)
gdim = 2

V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
D = fem.functionspace(msh, ("Lagrange", 1))
qdeg = 2
Qe = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(), degree=qdeg)
Q = fem.functionspace(msh, Qe)
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": qdeg})

u = fem.Function(V, name="u")
vel = fem.Function(V)
d = fem.Function(D, name="damage")
H = fem.Function(Q)
Hnew = fem.Function(Q)
H.x.array[:] = psi_c

# boundaries
fdim = 1
top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], Ly))
bot = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
facets = np.hstack([top, bot]).astype(np.int32)
tags = np.hstack([np.full_like(top, 1), np.full_like(bot, 2)]).astype(np.int32)
order = np.argsort(facets)
ft = mesh.meshtags(msh, fdim, facets[order], tags[order])
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

# pre-crack: d = 1 on nodes of the line y = Ly/2, x <= a0
crack_dofs = fem.locate_dofs_geometrical(
    D, lambda x: np.isclose(x[1], Ly / 2) & (x[0] <= a0 + 1e-9))
bc_d = fem.dirichletbc(PETSc.ScalarType(1.0), crack_dofs, D)


def eps(w):
    return ufl.sym(ufl.grad(w))


def mp(x):
    return 0.5 * (x + abs(x))


def mn(x):
    return 0.5 * (x - abs(x))


def psi_pos(w):
    e = eps(w)
    tr = ufl.tr(e)
    dev = e - 0.5 * tr * ufl.Identity(gdim)
    return 0.5 * K2 * mp(tr) ** 2 + mu * ufl.inner(dev, dev)


def sigma(w, dd):
    e = eps(w)
    tr = ufl.tr(e)
    dev = e - 0.5 * tr * ufl.Identity(gdim)
    g = (1 - dd) ** 2 + kres
    return g * (K2 * mp(tr) * ufl.Identity(gdim) + 2 * mu * dev) + K2 * mn(tr) * ufl.Identity(gdim)


v_ = ufl.TestFunction(V)
# lumped (row-sum) mass
mvec = assemble_vector(fem.form(rho * ufl.inner(ufl.as_vector((1.0, 1.0)), v_) * dx))
minv = 1.0 / mvec.array.copy()
fext = assemble_vector(fem.form(sigma0 * v_[1] * ds(1) - sigma0 * v_[1] * ds(2)))
fext_arr = fext.array.copy()
fint_form = fem.form(ufl.inner(sigma(u, d), eps(v_)) * dx)
fint = fem.petsc.create_vector(fint_form)

# damage problem (AT1 with history H >= psi_c): 2H d - 3Gc*ell/4 lap d = 2H - 3Gc/(8 ell)
dt_, w_ = ufl.TrialFunction(D), ufl.TestFunction(D)
a_d = fem.form(2 * H * dt_ * w_ * dx + 0.75 * Gc * ell * ufl.inner(ufl.grad(dt_), ufl.grad(w_)) * dx)
L_d = fem.form((2 * H - 3 * Gc / (8 * ell)) * w_ * dx)
A_d = assemble_matrix(a_d, bcs=[bc_d])
A_d.assemble()
b_d = fem.petsc.create_vector(L_d)
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A_d)
ksp.setType("cg")
ksp.getPC().setType("hypre")
ksp.setTolerances(rtol=1e-8)
dsol = fem.Function(D)

psi_expr = fem.Expression(psi_pos(u), Q.element.interpolation_points())


def solve_damage():
    Hnew.interpolate(psi_expr)
    np.maximum(H.x.array, Hnew.x.array, out=H.x.array)
    A_d.zeroEntries()
    assemble_matrix(A_d, a_d, bcs=[bc_d])
    A_d.assemble()
    with b_d.localForm() as loc:
        loc.set(0)
    assemble_vector(b_d, L_d)
    apply_lifting(b_d, [a_d], bcs=[[bc_d]])
    b_d.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_d, [bc_d])
    ksp.setOperators(A_d)
    ksp.solve(b_d, dsol.x.petsc_vec)
    dsol.x.scatter_forward()
    # irreversibility + bounds
    d.x.array[:] = np.clip(np.maximum(d.x.array, dsol.x.array), 0.0, 1.0)


# structured-grid indexing of vertex dofs (for snapshots / tip tracking)
xd = D.tabulate_dof_coordinates()[:, :2]
ix = np.rint(xd[:, 0] / h).astype(int)
iy = np.rint(xd[:, 1] / h).astype(int)
xv = V.tabulate_dof_coordinates()[:, :2]
jx = np.rint(xv[:, 0] / h).astype(int)
jy = np.rint(xv[:, 1] / h).astype(int)


def grid_scalar(arr):
    g = np.zeros((ny + 1, nx + 1))
    g[iy, ix] = arr
    return g


def grid_umag():
    ua = u.x.array.reshape(-1, 2)
    g = np.zeros((ny + 1, nx + 1))
    g[jy, jx] = np.linalg.norm(ua, axis=1)
    return g


def crack_tip(thr=0.9):
    """Rightmost damaged point in the upper half (follows the upper branch)."""
    m = (d.x.array > thr) & (xd[:, 1] >= Ly / 2 - 0.5 * h) & (xd[:, 0] > a0 + 0.5 * h)
    if not np.any(m):
        return a0, Ly / 2
    idx = np.where(m)[0]
    k = idx[np.argmax(xd[idx, 0])]
    return xd[k, 0], xd[k, 1]


ekin_form = fem.form(0.5 * rho * ufl.inner(vel, vel) * dx)
epot_form = fem.form(0.5 * ufl.inner(sigma(u, d), eps(u)) * dx)

nsteps = int(round(T_end / dt))
snap_every = max(1, nsteps // a.nsnap)
hist_every = max(1, int(round(0.1e-6 / dt)))
print(f"mesh {nx}x{ny} quads, h={h*1e3:.4g} mm, ell={ell*1e3:.4g} mm, dt={dt*1e9:.3g} ns "
      f"(CFL {h/cd*1e9:.3g} ns), steps={nsteps}, sigma_c(AT1)={sig_c/1e6:.3g} MPa", flush=True)

snaps_t, snaps_d, snaps_u = [], [], []
hist = {"t": [], "tip_x": [], "tip_y": [], "Ekin": [], "Eel": []}
t = 0.0
t0 = time.time()
ua = u.x.array
va = vel.x.array
for n in range(1, nsteps + 1):
    # central difference: v_{n+1/2} = v_{n-1/2} + dt*a_n ; u_{n+1} = u_n + dt*v_{n+1/2}
    with fint.localForm() as loc:
        loc.set(0)
    assemble_vector(fint, fint_form)
    acc = (fext_arr - fint.array) * minv
    va += (0.5 * dt if n == 1 else dt) * acc
    ua += dt * va
    t += dt
    solve_damage()
    if n % hist_every == 0 or n == nsteps:
        tx, ty = crack_tip()
        hist["t"].append(t)
        hist["tip_x"].append(tx)
        hist["tip_y"].append(ty)
        hist["Ekin"].append(fem.assemble_scalar(ekin_form))
        hist["Eel"].append(fem.assemble_scalar(epot_form))
    if n % snap_every == 0 or n == nsteps:
        snaps_t.append(t)
        snaps_d.append(grid_scalar(d.x.array).astype(np.float32))
        snaps_u.append(grid_umag().astype(np.float32))
        print(f"step {n}/{nsteps}  t={t*1e6:.2f} us  max d={d.x.array.max():.3f}  "
              f"tip=({hist['tip_x'][-1]*1e3:.2f},{hist['tip_y'][-1]*1e3:.2f}) mm  "
              f"wall={time.time()-t0:.0f}s", flush=True)

wall = time.time() - t0
np.savez_compressed(os.path.join(a.outdir, "snapshots.npz"), t=np.array(snaps_t),
                    d=np.array(snaps_d), umag=np.array(snaps_u), Lx=Lx, Ly=Ly, h=h)
np.savez_compressed(os.path.join(a.outdir, "history.npz"), **{k: np.array(v) for k, v in hist.items()})
with open(os.path.join(a.outdir, "run_info.json"), "w") as f:
    json.dump({"sigma0_MPa": a.sigma, "h_mm": h * 1e3, "ell_mm": ell * 1e3, "dt_ns": dt * 1e9,
               "T_us": T_end * 1e6, "nx": nx, "ny": ny, "n_cells": nx * ny, "steps": nsteps,
               "E": E, "rho": rho, "Gc": Gc, "nu": nu, "sigma_c_AT1_MPa": sig_c / 1e6,
               "wallclock_s": wall}, f, indent=2)
print(f"done in {wall:.0f} s")
