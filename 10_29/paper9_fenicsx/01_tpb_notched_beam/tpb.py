"""Case 1 -- Three-point bending of a notched concrete beam with an interface
above the notch (paper Sec. 4, Fig. 4/5). Only h = 0.2 mm, r = l/l_phi = 0.25.

Units: mm, N, MPa.  Plane stress, thickness B = 100 mm.
Assumed material data (not listed in the paper, standard notched concrete beam
of Wu 2017): E = 20 GPa, nu = 0.2, ft = 2.4 MPa, Gc_interface = 0.113 N/mm.
Paper: Gc_bulk = 10 Gc_interface, l = 2.5 mm.  Interface strength taken = ft.
Interface: vertical line above the notch tip, length 30 mm (measured from Fig. 4).
"""
import sys, os, time, json
import numpy as np
import gmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh as dmesh
from dolfinx.io import gmshio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfczm as pf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
comm = MPI.COMM_WORLD

# ----------------------------------------------------------- parameters
Lb, Hb, B = 450.0, 100.0, 100.0
notch_w, notch_h = 5.0, 50.0
E, nu, ft = 20000.0, 0.2, 2.4
Gc_i = 0.113
Gc_b = 10.0 * Gc_i
L_int = 30.0           # interface length above the notch tip (Fig. 4)
ft_i = ft
l = 2.5
r = 0.25
lphi = l / r
h_fine, h_coarse = 0.2, 5.0
u_max = 1.0
# load steps: du = 0.002 mm up to 0.2 mm (pre-peak and peak), then du = 0.01 mm
u_steps = np.r_[np.arange(0.002, 0.2 + 1e-9, 0.002), np.arange(0.21, u_max + 1e-9, 0.01)]
load_halfwidth = 2.0
if "--quick" in sys.argv:
    h_fine = 1.0
    u_steps = np.r_[np.arange(0.005, 0.2 + 1e-9, 0.005), np.arange(0.22, u_max + 1e-9, 0.02)]
n_steps = len(u_steps)

# ----------------------------------------------------------- mesh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("tpb")
occ = gmsh.model.occ
beam = occ.addRectangle(0, 0, 0, Lb, Hb)
notch = occ.addRectangle(Lb / 2 - notch_w / 2, -1, 0, notch_w, notch_h + 1)
cut, _ = occ.cut([(2, beam)], [(2, notch)])
line = occ.addLine(occ.addPoint(Lb / 2, notch_h, 0), occ.addPoint(Lb / 2, notch_h + L_int, 0))
occ.fragment(cut, [(1, line)])
occ.synchronize()
gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], 1)
f = gmsh.model.mesh.field
box = f.add("Box")
f.setNumber(box, "VIn", h_fine)
f.setNumber(box, "VOut", h_coarse)
f.setNumber(box, "XMin", Lb / 2 - 10)
f.setNumber(box, "XMax", Lb / 2 + 10)
f.setNumber(box, "YMin", 40)
f.setNumber(box, "YMax", Hb)
f.setNumber(box, "Thickness", 30)
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

# ----------------------------------------------------------- interface PF
tol = 1e-6
idofs = fem.locate_dofs_geometrical(
    W, lambda x: np.isclose(x[0], Lb / 2, atol=tol) & (x[1] >= notch_h - tol) & (x[1] <= notch_h + L_int + tol))
phi = pf.solve_interface_phase_field(W, idofs, lphi)
phi.name = "phi"
Gbar = pf.gc_bar(Gc_i, Gc_b, Gc_b, l, lphi)
print(f"R(r)={pf.R_of_r(r):.4f} < pi Gi/(2GcA)={np.pi*Gc_i/(2*Gc_b):.4f};  Gc_bar={Gbar:.5f}")
g = 1 - (1 - phi) ** 2
Gc = g * Gbar + (1 - g) * Gc_b                                   # Eq. (10)
ftx = g * ft_i + (1 - g) * ft                                    # Eq. (4)
a1 = pf.a1_coef(E, Gc, ftx, l)

# ----------------------------------------------------------- fields
u = fem.Function(V, name="u")
d = fem.Function(W, name="d")
d_old, d_it = fem.Function(W), fem.Function(W)
lm, mu = pf.lame(E, nu, plane_stress=True)
w_d = pf.omega(d, a1)
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a_u = w_d * ufl.inner(pf.sigma0(pf.eps(du), lm, mu), pf.eps(v)) * pf.DX
L_u = ufl.inner(fem.Constant(msh, PETSc.ScalarType((0.0, 0.0))), v) * pf.DX

# BCs
left = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
Vy, _ = V.sub(1).collapse()
right_y = fem.locate_dofs_geometrical((V.sub(1), Vy), lambda x: np.isclose(x[0], Lb) & np.isclose(x[1], 0))
top_y = fem.locate_dofs_geometrical((V.sub(1), Vy), lambda x: np.isclose(x[1], Hb) & (np.abs(x[0] - Lb / 2) <= load_halfwidth + tol))
uload = fem.Function(Vy)
bcs = [fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), left, V),
       fem.dirichletbc(PETSc.ScalarType(0.0), right_y[0], V.sub(1)),
       fem.dirichletbc(uload, top_y, V.sub(1))]
mech = pf.LinearSolver(a_u, L_u, u, bcs)

H = pf.History(W0, pf.psi_plus(pf.eps(u), lm, mu), ftx**2 / (2 * E))
F_d = pf.phase_field_energy(d, d_old, d_it, H.H, Gc, a1, l)
pfs = pf.PhaseFieldSolver(F_d, d, d_old)

# reaction force: internal force vector, summed over loaded y dofs
f_int = fem.form(w_d * ufl.inner(pf.sigma0(pf.eps(u), lm, mu), pf.eps(v)) * pf.DX)
E_frac = fem.form(Gc * pf.gamma_density(d, l) * pf.DX)


def reaction():
    vec = fem.petsc.assemble_vector(f_int)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vec.array


# y-dofs of V for the loaded nodes (blocked index * 2 + 1)
top_nodes = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], Hb) & (np.abs(x[0] - Lb / 2) <= load_halfwidth + tol))

xdmf = io.XDMFFile(comm, os.path.join(OUT, "tpb.xdmf"), "w")
xdmf.write_mesh(msh)
xdmf.write_function(phi, 0.0)

rows = []
t0 = time.time()
snap = {int(n_steps * s) for s in (0.05, 0.25, 0.5, 1.0)}
for k in range(1, n_steps + 1):
    disp = float(u_steps[k - 1])
    uload.x.array[:] = -disp
    it, err = pf.staggered(mech.solve, H, pfs, d, d_it, tol=1e-4, max_it=200)
    H.commit()
    d_old.x.array[:] = d.x.array
    arr = reaction()
    Fy = -np.sum(arr[top_nodes * 2 + 1]) * B / 1000.0         # kN
    Ed = pf.assemble_scalar(E_frac) * B / 1000.0                # J (N mm -> J)
    rows.append((disp, Fy, Ed, d.x.array.max(), it))
    print(f"step {k:4d} u={disp:.4f} F={Fy:.4f} kN  Ed={Ed:.5f} J  dmax={d.x.array.max():.3f} it={it} "
          f"err={err:.1e}  t={time.time()-t0:.0f}s", flush=True)
    if k in snap:
        xdmf.write_function(d, disp)
        xdmf.write_function(u, disp)
xdmf.close()
np.savetxt(os.path.join(OUT, "load_displacement.csv"), np.array(rows),
           delimiter=",", header="u_mm,F_kN,fracture_energy_J,dmax,stagger_iters", comments="")

# ----------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(R[:, 0], R[:, 1], "r-", label=f"h={h_fine} mm, r={r}")
ax[0].set_xlabel("Displacement (mm)"); ax[0].set_ylabel("Reaction force (kN)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(R[:, 0], R[:, 2], "r-")
ax[1].set_xlabel("Displacement (mm)"); ax[1].set_ylabel("Fracture energy (J)"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_load_energy.png"), dpi=150)

tri = pf.p1_triangulation(W)
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
for a, fld, ttl in ((ax[0], d, "phase field d"), (ax[1], phi, "interface phase field phi")):
    c = a.tripcolor(tri, fld.x.array, shading="gouraud", cmap="jet", vmin=0, vmax=1)
    a.set_aspect("equal"); a.set_title(ttl); fig.colorbar(c, ax=a)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "damage_full.png"), dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(8, 5))
for a, fld, ttl in ((ax[0], d, "d (zoom)"), (ax[1], phi, "phi (zoom)")):
    c = a.tripcolor(tri, fld.x.array, shading="gouraud", cmap="jet", vmin=0, vmax=1)
    a.set_xlim(Lb / 2 - 25, Lb / 2 + 25); a.set_ylim(0, Hb); a.set_aspect("equal"); a.set_title(ttl)
    fig.colorbar(c, ax=a)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "damage_zoom.png"), dpi=150)
json.dump(dict(cells=int(ncell), h=h_fine, l=l, r=r, lphi=lphi, Gc_bar=float(Gbar),
               peak_kN=float(R[:, 1].max()), wall_s=time.time() - t0),
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("done")
