"""Common geometry, material data and solvers for reproducing the TBC
cooling-induced cracking cases of
  H. Wang et al., "Peridynamic-based investigation of the cracking behavior of
  multilayer thermal barrier coatings", Ceram. Int. 48 (2022) 23543-23553
with FEniCSx (dolfinx 0.9). The bond-based PD fracture model of the paper is
replaced by a phase-field cohesive zone model (Wu's PF-CZM) solved by FEM.

Units: length um, stress MPa, energy release rate MPa*um (= J/m^2), T in K.
"""
import json
import time
import numpy as np
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem import petsc as fpetsc

BC_TAG, TGO_TAG, TC_TAG = 1, 2, 3

# Table 1 of the paper (E in MPa, alpha in 1/K, k in W/(m K))
MAT = {
    TC_TAG: dict(E=40.0e3, nu=0.25, alpha=1.07e-5, k=1.05),
    TGO_TAG: dict(E=360.0e3, nu=0.25, alpha=8.0e-6, k=25.2),
    BC_TAG: dict(E=86.0e3, nu=0.25, alpha=1.75e-5, k=4.3),
}
GC_TC = 20.0  # J/m^2 = MPa*um (also used for the TC/TGO interface, as in the paper)

H_TC, H_BC, H_TGO = 280.0, 200.0, 6.0
PERIOD = 80.0
H_TOT = H_TC + H_BC  # 480 um


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def tgo_curves(W, A, n_per_um=1.0):
    """TC/TGO (top) and TGO/BC (bottom, 6 um normal offset) interface points.

    A is the interface amplitude taken as peak-to-valley height (Fig. 6/9 of the
    paper: TGO top runs from ~193 to ~213 um for A = 20 um).
    Valleys at x = 0, 80, ...; peaks at x = 40, 120, ...
    """
    x = np.linspace(0.0, W, int(W * n_per_um) + 1)
    k = 2 * np.pi / PERIOD
    yt = H_BC + H_TGO / 2 - 0.5 * A * np.cos(k * x)
    dy = 0.5 * A * k * np.sin(k * x)
    nrm = np.sqrt(1 + dy**2)
    xb = x + H_TGO * dy / nrm
    yb = yt - H_TGO / nrm
    xb[0], xb[-1] = 0.0, W
    return np.c_[x, yt], np.c_[xb, yb]


def build_mesh(W, A, h_fine, fine_box, cracks=(), h_coarse=10.0, comm=MPI.COMM_WORLD):
    """gmsh mesh of the 2D TBC cell (Fig. 3). cracks: ((x0,y0),(x1,y1)) segments
    embedded in the TC so that a line of nodes lies on each pre-crack."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tbc")
    g = gmsh.model.geo
    top, bot = tgo_curves(W, A)
    lc = h_coarse

    def pts(arr):
        return [g.addPoint(px, py, 0, lc) for px, py in arr]

    pt_top, pt_bot = pts(top), pts(bot)
    p00, pW0 = g.addPoint(0, 0, 0, lc), g.addPoint(W, 0, 0, lc)
    p0H, pWH = g.addPoint(0, H_TOT, 0, lc), g.addPoint(W, H_TOT, 0, lc)
    s_top, s_bot = g.addSpline(pt_top), g.addSpline(pt_bot)
    l_bottom = g.addLine(p00, pW0)
    r1 = g.addLine(pW0, pt_bot[-1])
    r2 = g.addLine(pt_bot[-1], pt_top[-1])
    r3 = g.addLine(pt_top[-1], pWH)
    l_top = g.addLine(pWH, p0H)
    q3 = g.addLine(p0H, pt_top[0])
    q2 = g.addLine(pt_top[0], pt_bot[0])
    q1 = g.addLine(pt_bot[0], p00)
    s_bc = g.addPlaneSurface([g.addCurveLoop([l_bottom, r1, -s_bot, q1])])
    s_tgo = g.addPlaneSurface([g.addCurveLoop([s_bot, r2, -s_top, q2])])
    s_tc = g.addPlaneSurface([g.addCurveLoop([s_top, r3, l_top, q3])])
    crack_lines = []
    for (a, b) in cracks:
        pa, pb = g.addPoint(*a, 0, h_fine), g.addPoint(*b, 0, h_fine)
        crack_lines.append(g.addLine(pa, pb))
    g.synchronize()
    if crack_lines:
        gmsh.model.mesh.embed(1, crack_lines, 2, s_tc)
    for tag, s in ((BC_TAG, s_bc), (TGO_TAG, s_tgo), (TC_TAG, s_tc)):
        gmsh.model.addPhysicalGroup(2, [s], tag)
    f = gmsh.model.mesh.field
    boxes = []
    for (y0, y1, hb) in ((fine_box[0], fine_box[1], h_fine),
                         (H_BC - A / 2 - 12, H_BC + A / 2 + 12, max(h_fine, 1.5))):
        box = f.add("Box")
        f.setNumber(box, "VIn", hb)
        f.setNumber(box, "VOut", h_coarse)
        f.setNumber(box, "XMin", -1)
        f.setNumber(box, "XMax", W + 1)
        f.setNumber(box, "YMin", y0)
        f.setNumber(box, "YMax", y1)
        f.setNumber(box, "Thickness", 40)
        boxes.append(box)
    fmin = f.add("Min")
    f.setNumbers(fmin, "FieldsList", boxes)
    f.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    msh, ct, _ = io.gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return msh, ct


def cell_property(msh, ct, key):
    V0 = fem.functionspace(msh, ("DG", 0))
    f = fem.Function(V0)
    for tag, m in MAT.items():
        f.x.array[ct.indices[ct.values == tag]] = m[key]
    f.x.scatter_forward()
    return f


def tc_indicator(msh, ct):
    V0 = fem.functionspace(msh, ("DG", 0))
    f = fem.Function(V0)
    f.x.array[ct.indices[ct.values == TC_TAG]] = 1.0
    f.x.scatter_forward()
    return f


# --------------------------------------------------------------------------
# thermal problem (steady conduction, Fig. 11a)
# --------------------------------------------------------------------------
def solve_temperature(msh, ct, T_top, T_bot):
    V = fem.functionspace(msh, ("Lagrange", 1))
    k = cell_property(msh, ct, "k")
    T, q = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = k * ufl.inner(ufl.grad(T), ufl.grad(q)) * ufl.dx
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * q * ufl.dx
    top = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], H_TOT))
    bot = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
    bcs = [fem.dirichletbc(PETSc.ScalarType(T_top), fem.locate_dofs_topological(V, 1, top), V),
           fem.dirichletbc(PETSc.ScalarType(T_bot), fem.locate_dofs_topological(V, 1, bot), V)]
    prob = fpetsc.LinearProblem(a, L, bcs=bcs, petsc_options={
        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "petsc"})
    Th = prob.solve()
    Th.name = "T"
    return Th


# --------------------------------------------------------------------------
# PF-CZM thermo-elastic fracture solver
# --------------------------------------------------------------------------
class PFCZM:
    """Staggered PF-CZM (Wu 2017, linear softening) for the TC layer.

    * TGO and BC are unbreakable (paper: s_c set very large); the TC and the
      TC/TGO interface share G_c = 20 J/m^2 (paper Sec. 2.2.2).
    * 2D elasticity with plane-strain constants (nu = 1/4 as in 2D BB-PD) and
      thermal load dT(x) = lam * dT_ref(x). kinematics="pd2d" uses only the
      in-plane thermal strain alpha*dT*I_2 (as the 2D BB-PD bond force
      c*(s - alpha*T) does); "plane_strain" is the classical plane strain.
    * Crack driving force: max in-plane principal elastic strain
      Y = E <eps_1>^2 / 2, i.e. a stretch-type criterion analogous to the
      critical bond stretch of BB-PD ("strain") or Rankine <sigma_1>^2/(2E).
    * Mechanical BCs (Fig. 3): u_y = 0 at the bottom, MPC on both sides:
      u_x = 0 on the left, u_x = U (single unknown, zero resultant force) on
      the right; U is obtained by superposition of two linear solves.
    """

    def __init__(self, msh, ct, W, dT_ref, l0, ft, cracks=(), criterion="strain",
                 kinematics="pd2d", a2=-0.5, crack_tol=1e-6, Gc=GC_TC):
        self.msh, self.ct, self.W = msh, ct, W
        self.l0, self.ft, self.Gc, self.crit = l0, ft, Gc, criterion
        E_tc = MAT[TC_TAG]["E"]
        self.a1 = 4 * E_tc * self.Gc / (np.pi * l0 * ft**2)
        self.a2 = a2
        c0 = np.pi
        dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
        self.dx = dx

        self.V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
        self.Vd = fem.functionspace(msh, ("Lagrange", 1))
        self.V0 = fem.functionspace(msh, ("DG", 0))
        self.u, self.u0, self.u1 = (fem.Function(self.V, name=n) for n in ("u", "u0", "u1"))
        self.d = fem.Function(self.Vd, name="d")
        self.dk = fem.Function(self.Vd)
        self.dn = fem.Function(self.Vd)
        self.H = fem.Function(self.V0, name="H")
        self.Hn = fem.Function(self.V0)
        self.dT = fem.Function(self.Vd, name="dT")
        self.dT_ref = dT_ref

        E = cell_property(msh, ct, "E")
        nu = cell_property(msh, ct, "nu")
        al = cell_property(msh, ct, "alpha")
        chi = tc_indicator(msh, ct)
        lmb = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        if kinematics == "pd2d":
            # 2D BB-PD analogue: in-plane isotropic thermal strain alpha*dT*I_2 only
            beta = 2 * (lmb + mu) * al
        else:
            # classical plane strain (alpha*dT also acts out of plane)
            beta = (3 * lmb + 2 * mu) * al
        I = ufl.Identity(2)
        eps = lambda w: ufl.sym(ufl.grad(w))

        d = self.d
        omega = lambda s: (1 - s) ** 2 / ((1 - s) ** 2 + self.a1 * s * (1 + self.a2 * s))
        w = chi * omega(d) + (1 - chi)
        self.omega = omega
        sig0 = lambda w_: lmb * ufl.tr(eps(w_)) * I + 2 * mu * eps(w_)
        sig_th = beta * self.dT * I
        uT, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.a_u = fem.form(w * ufl.inner(sig0(uT), eps(v)) * dx)
        self.L_u = fem.form(w * ufl.inner(sig_th, eps(v)) * dx)
        self.r0 = fem.form(w * ufl.inner(sig0(self.u0) - sig_th, eps(v)) * dx)
        self.r1 = fem.form(w * ufl.inner(sig0(self.u1), eps(v)) * dx)
        self.sig = lambda: w * (sig0(self.u) - sig_th)

        # mechanical BCs
        fdim = 1
        bottom = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
        left = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
        right = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], W))
        Vx, Vy = self.V.sub(0), self.V.sub(1)
        self.Ubar = fem.Constant(msh, PETSc.ScalarType(0.0))
        self.right_dofs = fem.locate_dofs_topological(Vx, fdim, right)
        self.bcs_u = [
            fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(Vy, fdim, bottom), Vy),
            fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_topological(Vx, fdim, left), Vx),
            fem.dirichletbc(self.Ubar, self.right_dofs, Vx),
        ]
        self.A = fpetsc.create_matrix(self.a_u)
        self.b = fpetsc.create_vector(self.L_u)
        self.R = fpetsc.create_vector(self.r0)
        self.ksp = PETSc.KSP().create(msh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("petsc")
        bs = self.V.dofmap.index_map_bs
        nloc = self.V.dofmap.index_map.size_local
        self.right_owned = self.right_dofs[self.right_dofs < nloc * bs]

        # crack driving force (DG0 is exact for P1 strains)
        ee = eps(self.u) - al * self.dT * I
        if criterion == "strain":
            e1 = 0.5 * ufl.tr(ee) + ufl.sqrt(((ee[0, 0] - ee[1, 1]) / 2) ** 2 + ee[0, 1] ** 2 + 1e-30)
            Y = 0.5 * E * ufl.max_value(e1, 0.0) ** 2
        else:
            s0 = sig0(self.u) - sig_th
            s1 = 0.5 * ufl.tr(s0) + ufl.sqrt(((s0[0, 0] - s0[1, 1]) / 2) ** 2 + s0[0, 1] ** 2 + 1e-30)
            Y = ufl.max_value(s1, 0.0) ** 2 / (2 * E)
        self.Y_expr = fem.Expression(chi * Y, self.V0.element.interpolation_points())
        self.Yh = fem.Function(self.V0)
        self.H_floor = ft**2 / (2 * E_tc)

        # phase-field problem (alpha = 2d - d^2, -d^2 linearised at stagger iterate)
        q = ufl.TestFunction(self.Vd)
        dv = ufl.variable(d)
        domega = ufl.diff(omega(dv), dv)
        Gc = self.Gc
        F = (domega * self.H * q + Gc / (c0 * l0) * (2 - 2 * self.dk) * q
             + 2 * Gc * l0 / c0 * ufl.inner(ufl.grad(d), ufl.grad(q))) * dx(TC_TAG)
        # zero-valued term only to give the non-TC rows a diagonal sparsity entry
        F = F + fem.Constant(msh, PETSc.ScalarType(0.0)) * d * q * (dx(BC_TAG) + dx(TGO_TAG))
        self.F_d = fem.form(F)
        self.J_d = fem.form(ufl.derivative(F, d, ufl.TrialFunction(self.Vd)))
        # dofs not touching TC cells are fixed to 0
        tc_cells = ct.indices[ct.values == TC_TAG]
        tc_dofs = np.unique(self.Vd.dofmap.list[tc_cells].ravel())
        all_dofs = np.arange(self.Vd.dofmap.index_map.size_local + self.Vd.dofmap.index_map.num_ghosts)
        non_tc = np.setdiff1d(all_dofs, tc_dofs).astype(np.int32)
        self.bcs_d = [fem.dirichletbc(PETSc.ScalarType(0.0), non_tc, self.Vd)]

        # pre-cracks: d = 1 on nodes lying on the segments (lower bound)
        xd = self.Vd.tabulate_dof_coordinates()[:, :2]
        for (a, b) in cracks:
            a, b = np.array(a), np.array(b)
            t = np.clip(((xd - a) @ (b - a)) / np.dot(b - a, b - a), 0, 1)
            dist = np.linalg.norm(xd - (a + t[:, None] * (b - a)), axis=1)
            self.dn.x.array[dist < crack_tol + 1e-3] = 1.0
        self.d.x.array[:] = self.dn.x.array
        self.dk.x.array[:] = self.dn.x.array
        self.H.x.array[:] = self.H_floor
        self.Hn.x.array[:] = self.H_floor

        self.Jmat = fpetsc.create_matrix(self.J_d)
        self.Fvec = fpetsc.create_vector(self.F_d)
        self.snes = PETSc.SNES().create(msh.comm)
        self.snes.setType("vinewtonrsls")
        self.snes.setFunction(self._F, self.Fvec)
        self.snes.setJacobian(self._J, self.Jmat)
        self.snes.setTolerances(rtol=1e-9, atol=1e-9, max_it=50)
        kd = self.snes.getKSP()
        kd.setType("preonly")
        kd.getPC().setType("lu")
        kd.getPC().setFactorSolverType("petsc")
        self.lb = self.dn.x.petsc_vec.copy()
        self.ub = self.dn.x.petsc_vec.copy()
        self.ub.set(1.0)
        self.xd = self.d.x.petsc_vec.copy()

    # ---------------- phase-field SNES callbacks
    def _F(self, snes, x, b):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.d.x.petsc_vec)
        self.d.x.scatter_forward()
        with b.localForm() as bl:
            bl.set(0.0)
        fpetsc.assemble_vector(b, self.F_d)
        fpetsc.apply_lifting(b, [self.J_d], [self.bcs_d], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, self.bcs_d, x, -1.0)

    def _J(self, snes, x, A, P):
        A.zeroEntries()
        fpetsc.assemble_matrix(A, self.J_d, bcs=self.bcs_d)
        A.assemble()

    # ---------------- elastic solve with MPC by superposition
    def _solve_linear(self, ubar, thermal, out):
        self.Ubar.value = ubar
        with self.b.localForm() as bl:
            bl.set(0.0)
        if thermal:
            fpetsc.assemble_vector(self.b, self.L_u)
        fpetsc.apply_lifting(self.b, [self.a_u], [self.bcs_u])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(self.b, self.bcs_u)
        self.ksp.solve(self.b, out.x.petsc_vec)
        out.x.scatter_forward()

    def _reaction(self, form):
        with self.R.localForm() as rl:
            rl.set(0.0)
        fpetsc.assemble_vector(self.R, form)
        self.R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        loc = self.R.array_r[self.right_owned].sum()
        return self.msh.comm.allreduce(loc, op=MPI.SUM)

    def solve_u(self):
        self.A.zeroEntries()
        fpetsc.assemble_matrix(self.A, self.a_u, bcs=self.bcs_u)
        self.A.assemble()
        self.ksp.setOperators(self.A)
        self._solve_linear(0.0, True, self.u0)
        self._solve_linear(1.0, False, self.u1)
        R0, R1 = self._reaction(self.r0), self._reaction(self.r1)
        U = -R0 / R1
        self.u.x.array[:] = self.u0.x.array + U * self.u1.x.array
        self.u.x.scatter_forward()
        self.Ubar.value = U
        return U

    def update_history(self):
        self.Yh.interpolate(self.Y_expr)
        self.H.x.array[:] = np.maximum(self.Hn.x.array, np.maximum(self.Yh.x.array, self.H_floor))
        self.H.x.scatter_forward()

    def solve_d(self):
        self.dk.x.array[:] = self.d.x.array
        self.d.x.petsc_vec.copy(self.xd)
        self.dn.x.petsc_vec.copy(self.lb)
        self.snes.setVariableBounds(self.lb, self.ub)
        self.snes.solve(None, self.xd)
        self.xd.copy(self.d.x.petsc_vec)
        self.d.x.scatter_forward()
        return self.snes.getConvergedReason()

    def step(self, lam, tol=2e-3, maxit=300, minit=2):
        self.dT.x.array[:] = lam * self.dT_ref.x.array
        self.dT.x.scatter_forward()
        it, err = 0, 1.0
        for it in range(1, maxit + 1):
            d_old = self.d.x.array.copy()
            self.solve_u()
            self.update_history()
            reason = self.solve_d()
            err = self.msh.comm.allreduce(np.abs(self.d.x.array - d_old).max(), op=MPI.MAX)
            if err < tol and it >= minit:
                break
        self.Hn.x.array[:] = self.H.x.array
        self.dn.x.array[:] = self.d.x.array
        return it, err, reason

    def peak_Y_ratio(self):
        """max sqrt(Y/Y_floor) in TC (= equivalent stress / f_t)."""
        self.Yh.interpolate(self.Y_expr)
        return np.sqrt(self.Yh.x.array.max() / self.H_floor)


# --------------------------------------------------------------------------
# post-processing helpers
# --------------------------------------------------------------------------
def p1_triangulation(V):
    xy = V.tabulate_dof_coordinates()[:, :2]
    cells = V.dofmap.list
    return xy, cells


def plot_field(ax, xy, cells, vals, W, A, title="", vmin=0, vmax=1, cmap="jet", xlim=None, ylim=None,
               label="Damage d"):
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], cells)
    pc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    top, bot = tgo_curves(W, A)
    ax.plot(top[:, 0], top[:, 1], "w-", lw=0.6)
    ax.plot(bot[:, 0], bot[:, 1], "w-", lw=0.6)
    ax.set_aspect("equal")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$x$ / $\mu$m")
    ax.set_ylabel(r"$y$ / $\mu$m")
    ax.set_title(title, fontsize=9)
    return pc


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# generic driver
# --------------------------------------------------------------------------
def run_simulation(name, outdir, W, A, cracks, thermal, l0, ft, h_fine, fine_box,
                   lam_steps, snap_lams, view, criterion="strain", kinematics="pd2d",
                   tol=2e-3, maxit=300, Gc=GC_TC):
    """thermal: ("iso", dT) or ("gradient", T_top, T_bot, ref), ref in {"room", "top"}."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    comm = MPI.COMM_WORLD
    log = open(os.path.join(outdir, "log.txt"), "w")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    t0 = time.time()
    msh, ct = build_mesh(W, A, h_fine, fine_box, cracks)
    Vd = fem.functionspace(msh, ("Lagrange", 1))
    dT_ref = fem.Function(Vd, name="dT_ref")
    if thermal[0] == "iso":
        dT_ref.x.array[:] = thermal[1]
        T = None
    else:
        T = solve_temperature(msh, ct, thermal[1], thermal[2])
        ref = thermal[3] if len(thermal) > 3 else "room"
        if ref == "room":
            # temperatures relative to room T: cooling from the steady gradient to room T
            dT_ref.x.array[:] = -T.x.array
        else:
            # stress-free at uniform T_top, cooled to the steady gradient field
            dT_ref.x.array[:] = T.x.array - thermal[1]
    s = PFCZM(msh, ct, W, dT_ref, l0, ft, cracks=cracks, criterion=criterion, kinematics=kinematics, Gc=Gc)
    say(f"[{name}] cells={msh.topology.index_map(2).size_local} u-dofs={2 * s.V.dofmap.index_map.size_local}"
        f" l0={l0} ft={ft} Gc={Gc} a1={s.a1:.1f} h={h_fine} cracks={list(cracks)} thermal={thermal}"
        f" criterion={criterion} kinematics={kinematics} mesh {time.time() - t0:.1f}s")
    params = dict(name=name, W=W, A=A, cracks=[list(map(list, c)) for c in cracks], thermal=list(thermal),
                  l0=l0, ft=ft, Gc=Gc, l_ch=MAT[TC_TAG]["E"] * Gc / ft**2, a1=s.a1, h_fine=h_fine, criterion=criterion,
                  kinematics=kinematics, materials={str(k): v for k, v in MAT.items()})
    save_json(os.path.join(outdir, "params.json"), params)

    xy, cells = p1_triangulation(s.Vd)
    dT_max = float(np.abs(dT_ref.x.array).max())
    hist, snaps = [], {}
    gamma_form = fem.form((1.0 / np.pi) * ((2 * s.d - s.d**2) / l0
                                           + l0 * ufl.inner(ufl.grad(s.d), ufl.grad(s.d))) * s.dx(TC_TAG))
    d_store = []
    for k, lam in enumerate(lam_steps):
        tt = time.time()
        it, err, reason = s.step(lam, tol=tol, maxit=maxit)
        gam = comm.allreduce(fem.assemble_scalar(gamma_form), op=MPI.SUM)
        dmax = s.d.x.array.max()
        rec = dict(step=k + 1, lam=float(lam), dT_max=float(lam * dT_max), iters=it, err=float(err),
                   U=float(s.Ubar.value), crack_length_um=float(gam), d_max=float(dmax),
                   eq_ratio=float(s.peak_Y_ratio()), time=time.time() - tt)
        hist.append(rec)
        say(f"step {k + 1:3d} lam={lam:.4f} |dT|max={lam * dT_max:6.1f}K it={it:3d} err={err:.1e} "
            f"Gamma={gam:7.2f}um dmax={dmax:.3f} U={s.Ubar.value:.4f} ({rec['time']:.1f}s)")
        d_store.append(s.d.x.array.astype(np.float32).copy())
        for sl in snap_lams:
            if abs(lam - sl) < 1e-9:
                snaps[sl] = s.d.x.array.copy()
    save_json(os.path.join(outdir, "history.json"), hist)
    np.savez_compressed(os.path.join(outdir, "fields.npz"), xy=xy, cells=cells, lam=np.array(lam_steps),
                        d=np.array(d_store), dT_ref=dT_ref.x.array,
                        u=s.u.x.array.reshape(-1, 2), T=(T.x.array if T is not None else np.zeros(0)))
    with io.XDMFFile(comm, os.path.join(outdir, "final.xdmf"), "w") as xf:
        xf.write_mesh(msh)
        xf.write_function(s.d)
        Vout = fem.functionspace(msh, ("Lagrange", 1, (2,)))
        uo = fem.Function(Vout, name="u")
        uo.interpolate(s.u)
        xf.write_function(uo)
        if T is not None:
            xf.write_function(T)

    # ---- figures: damage evolution (like Figs. 6/9/11 of the paper)
    keys = sorted(snaps)
    n = len(keys)
    ncol = 2 if n == 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow * view["aspect"]), squeeze=False)
    labels = "abcdefgh"
    for i, kk in enumerate(keys):
        ax = axs.flat[i]
        pc = plot_field(ax, xy, cells, snaps[kk], W, A, xlim=view["xlim"], ylim=view["ylim"],
                        title=f"({labels[i]}) $|\\Delta T|_{{max}}$ = {kk * dT_max:.0f} K")
        fig.colorbar(pc, ax=ax, label="Damage $d$", shrink=0.85)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "damage_evolution.png"), dpi=200)
    plt.close(fig)

    # ---- final displacement contours (like Fig. 10g-h)
    u = s.u.x.array.reshape(-1, 2)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.6 * view["aspect"]))
    for j, comp in enumerate("xy"):
        pc = plot_field(axs[j], xy, cells, u[:, j], W, A, vmin=None, vmax=None, xlim=view["xlim"],
                        ylim=view["ylim"], title=f"$u_{comp}$ / $\\mu$m (final)")
        fig.colorbar(pc, ax=axs[j], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "displacement_final.png"), dpi=200)
    plt.close(fig)

    if T is not None:
        fig, ax = plt.subplots(figsize=(4.5, 7))
        pc = plot_field(ax, xy, cells, T.x.array, W, A, vmin=200, vmax=400, cmap="jet",
                        title="Steady temperature (relative) / K")
        fig.colorbar(pc, ax=ax, label="$T$ / K")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "temperature.png"), dpi=200)
        plt.close(fig)

    lam_a = np.array([h["dT_max"] for h in hist])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lam_a, [h["crack_length_um"] for h in hist], "o-", ms=3)
    ax.set_xlabel(r"$|\Delta T|_{max}$ / K")
    ax.set_ylabel(r"crack length $\Gamma_l$ / $\mu$m")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crack_length.png"), dpi=200)
    plt.close(fig)
    say(f"[{name}] done in {time.time() - t0:.1f}s")
    log.close()
    return s
