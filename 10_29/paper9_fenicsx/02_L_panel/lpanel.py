"""Case 2 -- L-shaped panel (Winkler test), paper Sec. 5.1, Fig. 6/7, Table 4.

Units: mm, N, MPa.  Plane stress, thickness B = 100 mm.
E = 25.28 GPa, nu = 0.18, Gc = 0.09 N/mm, ft = 2.7 MPa, l = 2.5 mm.
Bottom edge clamped, upward displacement at (470, 250) (30 mm from the right
edge on the lower face of the right arm), up to 0.8 mm.
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

E, nu, Gc, ft, l, B = 25280.0, 0.18, 0.09, 2.7, 2.5, 100.0
h_fine, h_coarse = 1.0, 10.0       # h = l/2.5 in the critical zone
u_max, n_steps = 0.8, 160
load_x = 470.0
if "--quick" in sys.argv:
    h_fine, n_steps = 2.5, 80

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("L")
occ = gmsh.model.occ
pts = [(0, 0), (250, 0), (250, 250), (load_x, 250), (500, 250), (500, 500), (0, 500)]
p = [occ.addPoint(x, y, 0) for x, y in pts]
ln = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
s = occ.addPlaneSurface([occ.addCurveLoop(ln)])
occ.synchronize()
gmsh.model.addPhysicalGroup(2, [s], 1)
f = gmsh.model.mesh.field
box = f.add("Box")
f.setNumber(box, "VIn", h_fine)
f.setNumber(box, "VOut", h_coarse)
f.setNumber(box, "XMin", 0); f.setNumber(box, "XMax", 262)
f.setNumber(box, "YMin", 238); f.setNumber(box, "YMax", 330)
f.setNumber(box, "Thickness", 40)
f.setAsBackgroundMesh(box)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
gmsh.finalize()
ncell = msh.topology.index_map(2).size_global
print(f"cells: {ncell}", flush=True)

V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
W = fem.functionspace(msh, ("Lagrange", 1))
W0 = fem.functionspace(msh, ("DG", 0))
a1 = pf.a1_coef(E, Gc, ft, l)

u = fem.Function(V, name="u")
d = fem.Function(W, name="d")
d_old, d_it = fem.Function(W), fem.Function(W)
lm, mu = pf.lame(E, nu, plane_stress=True)
w_d = pf.omega(d, a1)
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a_u = w_d * ufl.inner(pf.sigma0(pf.eps(du), lm, mu), pf.eps(v)) * pf.DX
L_u = ufl.inner(fem.Constant(msh, PETSc.ScalarType((0.0, 0.0))), v) * pf.DX

bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))
Vy, _ = V.sub(1).collapse()
load_y = fem.locate_dofs_geometrical((V.sub(1), Vy), lambda x: np.isclose(x[0], load_x) & np.isclose(x[1], 250.0))
uload = fem.Function(Vy)
bcs = [fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), bottom, V),
       fem.dirichletbc(uload, load_y, V.sub(1))]
mech = pf.LinearSolver(a_u, L_u, u, bcs)
load_nodes = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], load_x) & np.isclose(x[1], 250.0))

H = pf.History(W0, pf.psi_plus(pf.eps(u), lm, mu), fem.Constant(msh, ft**2 / (2 * E)))
pfs = pf.PhaseFieldSolver(pf.phase_field_energy(d, d_old, d_it, H.H, Gc, a1, l), d, d_old)
f_int = fem.form(w_d * ufl.inner(pf.sigma0(pf.eps(u), lm, mu), pf.eps(v)) * pf.DX)
E_frac = fem.form(Gc * pf.gamma_density(d, l) * pf.DX)

xdmf = io.XDMFFile(comm, os.path.join(OUT, "lpanel.xdmf"), "w")
xdmf.write_mesh(msh)
rows, t0 = [], time.time()
snap = {int(n_steps * s) for s in (0.25, 0.5, 1.0)}
for k in range(1, n_steps + 1):
    disp = u_max * k / n_steps
    uload.x.array[:] = disp
    it, err = pf.staggered(mech.solve, H, pfs, d, d_it, tol=1e-4, max_it=200)
    H.commit()
    d_old.x.array[:] = d.x.array
    vec = fem.petsc.assemble_vector(f_int)
    Fy = np.sum(vec.array[load_nodes * 2 + 1]) * B / 1000.0      # kN
    Ed = pf.assemble_scalar(E_frac) * B / 1000.0                 # J
    rows.append((disp, Fy, Ed, d.x.array.max(), it))
    print(f"step {k:4d} u={disp:.4f} F={Fy:.4f} kN Ed={Ed:.4f} J dmax={d.x.array.max():.3f} "
          f"it={it} err={err:.1e} t={time.time()-t0:.0f}s", flush=True)
    if k in snap:
        xdmf.write_function(d, disp)
xdmf.close()
np.savetxt(os.path.join(OUT, "load_displacement.csv"), np.array(rows), delimiter=",",
           header="u_mm,F_kN,fracture_energy_J,dmax,stagger_iters", comments="")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
tri = pf.p1_triangulation(W)
c = ax[0].tripcolor(tri, d.x.array, shading="gouraud", cmap="jet", vmin=0, vmax=1)
ax[0].set_aspect("equal"); ax[0].set_title("phase field d (u = %.2f mm)" % u_max); fig.colorbar(c, ax=ax[0])
ax[1].plot(R[:, 0], R[:, 1], "r-", label="FEniCSx (this repro)")
ax[1].set_xlabel("Displacement (mm)"); ax[1].set_ylabel("Reaction force (kN)"); ax[1].grid(alpha=.3)
ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_crack_and_load.png"), dpi=150)
json.dump(dict(cells=int(ncell), h=h_fine, l=l, peak_kN=float(R[:, 1].max()),
               u_at_peak=float(R[np.argmax(R[:, 1]), 0]), wall_s=time.time() - t0),
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("done")
