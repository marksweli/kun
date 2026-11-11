"""Case 3 -- Quenching of a ceramic plate (paper Sec. 5.2, Fig. 8/9, Table 5).

Units: mm, N, MPa, s, K.  Plane stress (thin plate).
Table 5 (from the PDF): rho = 2.45e-6 kg/mm^3, E = 340 GPa, nu = 0.22,
kappa = 0.3 W/(mm K), Gc = 42.47e-6 J/mm^2 (= 0.04247 N/mm), C = 775 J/(kg K),
alpha = 8e-6 1/K.  l = 0.08 mm, eta = 1e-8 kN s/mm^2 (= 1e-5 N s/mm^2).
Plate 50 x 9.8 mm, T0 = 680 K, all outer edges held at 300 K, t_end = 0.01 s.

Assumptions (not given in the paper):
  * ft = 260 MPa (AT1-equivalent strength sqrt(3 E Gc / 8 l) for l = 0.08 mm);
  * double symmetry -> only a quarter (25 x 4.9 mm) is computed and mirrored for
    the figures (cooled edges: x = 0 and y = 0);
  * refined size h = 0.05 mm (paper: 0.02 mm) inside a 2.4 mm band along the
    cooled edges and 100 implicit time steps (dt = 1e-4 s) to keep the run
    tractable on 2 CPU cores; pass --h 0.02 --n 500 to use finer settings;
  * heat capacity lumped (vertex quadrature) to avoid temperature overshoot.
"""
import sys, os, time, json
import numpy as np
import gmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.io import gmshio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfczm as pf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
comm = MPI.COMM_WORLD

Lx, Ly = 25.0, 4.9                     # quarter of 50 x 9.8
rho, E, nu, kappa = 2.45e-6, 340e3, 0.22, 0.3
Gc, Cp, alph = 42.47e-3, 775.0, 8e-6
ft = 260.0
l, eta = 0.08, 1e-5
T0, Tw = 680.0, 300.0
t_end, n_steps = 0.01, 100
h_fine, band = 0.05, 2.4
if "--h" in sys.argv:
    h_fine = float(sys.argv[sys.argv.index("--h") + 1])
if "--quick" in sys.argv:
    h_fine, n_steps = 0.1, 40
if "--tend" in sys.argv:
    t_end = float(sys.argv[sys.argv.index("--tend") + 1])
if "--n" in sys.argv:
    n_steps = int(sys.argv[sys.argv.index("--n") + 1])
dt_val = t_end / n_steps

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("quench")
occ = gmsh.model.occ
s = occ.addRectangle(0, 0, 0, Lx, Ly)
occ.synchronize()
gmsh.model.addPhysicalGroup(2, [s], 1)
cooled = []
for dim, tag in gmsh.model.getBoundary([(2, s)], oriented=False):
    xc = occ.getCenterOfMass(1, tag)
    if np.isclose(xc[0], 0) or np.isclose(xc[1], 0):
        cooled.append(tag)
f = gmsh.model.mesh.field
dist = f.add("Distance")
f.setNumbers(dist, "CurvesList", cooled)
f.setNumber(dist, "Sampling", 500)
thr = f.add("Threshold")
f.setNumber(thr, "InField", dist)
f.setNumber(thr, "SizeMin", h_fine)
f.setNumber(thr, "SizeMax", 0.4)
f.setNumber(thr, "DistMin", band)
f.setNumber(thr, "DistMax", band + 1.5)
f.setAsBackgroundMesh(thr)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
gmsh.finalize()
ncell = msh.topology.index_map(2).size_global
print(f"cells: {ncell}, h={h_fine}", flush=True)

V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
W = fem.functionspace(msh, ("Lagrange", 1))
W0 = fem.functionspace(msh, ("DG", 0))
a1 = pf.a1_coef(E, Gc, ft, l)
print(f"a1 = {a1:.3f}, l_ch = {E*Gc/ft**2:.3f} mm")

u = fem.Function(V, name="u")
T = fem.Function(W, name="T")
T_old = fem.Function(W)
T.x.array[:] = T0
T_old.x.array[:] = T0
d = fem.Function(W, name="d")
d_old, d_it = fem.Function(W), fem.Function(W)
dt = fem.Constant(msh, dt_val)
lm, mu = pf.lame(E, nu, plane_stress=True)
w_d = pf.omega(d, a1)

# heat conduction, Eq. (99), backward Euler
Tt, s_ = ufl.TrialFunction(W), ufl.TestFunction(W)
# vertex quadrature for the whole form = lumped heat capacity (no over/undershoot
# at the sudden boundary cooling); one rule per form, see dolfinx quadrature pitfall
DXv = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
a_T = (rho * Cp * Tt * s_ / dt + w_d * kappa * ufl.dot(ufl.grad(Tt), ufl.grad(s_))) * DXv
L_T = rho * Cp * T_old * s_ / dt * DXv
cool_dofs = fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0) | np.isclose(x[1], 0))
heat = pf.LinearSolver(a_T, L_T, T, [fem.dirichletbc(PETSc.ScalarType(Tw), cool_dofs, W)])

# mechanics (plane stress), thermal eigenstrain
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eth = alph * (T - T0) * ufl.Identity(2)
a_u = w_d * ufl.inner(pf.sigma0(pf.eps(du), lm, mu), pf.eps(v)) * pf.DX
L_u = w_d * ufl.inner(pf.sigma0(eth, lm, mu), pf.eps(v)) * pf.DX
Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
sx = fem.locate_dofs_geometrical((V.sub(0), Vx), lambda x: np.isclose(x[0], Lx))
sy = fem.locate_dofs_geometrical((V.sub(1), Vy), lambda x: np.isclose(x[1], Ly))
bcs = [fem.dirichletbc(PETSc.ScalarType(0), sx[0], V.sub(0)),
       fem.dirichletbc(PETSc.ScalarType(0), sy[0], V.sub(1))]
mech = pf.LinearSolver(a_u, L_u, u, bcs)

ee = pf.eps(u) - eth
H = pf.History(W0, pf.psi_plus(ee, lm, mu), fem.Constant(msh, ft**2 / (2 * E)))
pfs = pf.PhaseFieldSolver(pf.phase_field_energy(d, d_old, d_it, H.H, Gc, a1, l, eta=eta, dt=dt), d, d_old)
E_frac = fem.form(Gc * pf.gamma_density(d, l) * pf.DX)


def solve_TU():
    heat.solve()
    mech.solve()


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
tri = pf.p1_triangulation(W)


def full_plot(ax, vals, cmap, vmin, vmax):
    """mirror the quarter onto the whole 50 x 9.8 plate"""
    import matplotlib.tri as mtri
    for sxm in (1, -1):
        for sym in (1, -1):
            xx = Lx + sxm * (tri.x - Lx)
            yy = Ly + sym * (tri.y - Ly)
            c = ax.tripcolor(mtri.Triangulation(xx, yy, tri.triangles), vals, shading="gouraud",
                             cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xlim(0, 2 * Lx); ax.set_ylim(0, 2 * Ly)
    ax.set_xticks([]); ax.set_yticks([])
    return c


xdmf = io.XDMFFile(comm, os.path.join(OUT, "quench.xdmf"), "w")
xdmf.write_mesh(msh)
snaps = {int(round(n_steps * s)): s * t_end for s in (0.25, 0.5, 0.75, 1.0)}
fig, axes = plt.subplots(4, 2, figsize=(13, 8.5))
rows, t0, row = [], time.time(), 0
for k in range(1, n_steps + 1):
    t = k * dt_val
    it, err = pf.staggered(solve_TU, H, pfs, d, d_it, tol=1e-4, max_it=100)
    H.commit()
    d_old.x.array[:] = d.x.array
    T_old.x.array[:] = T.x.array
    Ed = pf.assemble_scalar(E_frac) * 4          # whole plate, N mm per mm thickness
    rows.append((t, Ed, d.x.array.max(), T.x.array.min(), T.x.array.max(), it))
    print(f"step {k:4d} t={t:.5f} Ed={Ed:.4e} dmax={d.x.array.max():.3f} Tmax={T.x.array.max():.1f} "
          f"it={it} err={err:.1e} wall={time.time()-t0:.0f}s", flush=True)
    if k in snaps:
        xdmf.write_function(d, t); xdmf.write_function(T, t)
        c1 = full_plot(axes[row, 0], d.x.array, "jet", 0, 1)
        c2 = full_plot(axes[row, 1], T.x.array, "jet", 300, 680)
        axes[row, 0].set_title(f"phase field, t = {t:.4f} s")
        axes[row, 1].set_title(f"temperature (K), t = {t:.4f} s")
        row += 1
xdmf.close()
fig.colorbar(c1, ax=axes[:, 0], shrink=0.6); fig.colorbar(c2, ax=axes[:, 1], shrink=0.6)
fig.savefig(os.path.join(OUT, "fig8_damage_temperature.png"), dpi=150)
fig, ax = plt.subplots(figsize=(13, 3))
full_plot(ax, d.x.array, "jet", 0, 1); ax.set_title(f"fracture pattern at t = {t_end:g} s (cf. Fig. 9)")
fig.savefig(os.path.join(OUT, "fig9_final_pattern.png"), dpi=200, bbox_inches="tight")
np.savetxt(os.path.join(OUT, "history.csv"), np.array(rows), delimiter=",",
           header="t_s,fracture_energy_Nmm_per_mm,dmax,Tmin,Tmax,stagger_iters", comments="")
json.dump(dict(cells=int(ncell), h=h_fine, l=l, ft_assumed=ft, dt=dt_val, wall_s=time.time() - t0),
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("done")
