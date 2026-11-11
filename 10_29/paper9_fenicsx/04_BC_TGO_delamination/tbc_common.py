"""Shared TBC model for paper Sec. 5.3-5.5 (chemo-thermo-mechanical PF-CZM).

Units: mm, N, MPa, s, degC; oxygen concentration c in mol/m^3.

Geometry (Fig. 10a): substrate 3 mm, BC 100 um, TC 500 um, sinusoidal TC/BC
interface y = A cos(2 pi x / L) (peak at x = 0 so that both vertical edges are
symmetry lines).  BCs: u_x = 0 on the left edge, u_y = 0 on the bottom, right
edge kept straight (u_x = const, unknown).  The constraint is imposed exactly
by superposition u = u0 + a u1 with a chosen such that the resultant x-force on
the right edge vanishes.  Generalised plane strain with eps_zz = substrate thermal
strain (thick substrate controls the out-of-plane strain), uniform temperature.

Oxidation (Sec. 2.5.2, Eqs. 45, 47, 83):
    dc/dt = D Lap c - M dn/dt,    dn/dt = omega(d) G (1-n) c   (T > 850 C)
BC material properties are mixed with the TGO ones via n (Eq. 83) and the
growth strain alpha_n n I is added (Eq. 14).
"""
import os, sys, time, json
import numpy as np
import gmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.io import gmshio
import pfczm as pf

comm = MPI.COMM_WORLD

# ------------------------------------------------ Table 1 (T-dependent)
TAB_T = np.array([20, 220, 420, 620, 820, 1020.0])
TAB = {
    "TC": dict(alpha=[9.68, 9.67, 9.71, 9.81, 10.02, 10.04], E=[17.5, 16.3, 15.2, 14.0, 12.9, 11.7],
               nu=[0.2] * 6),
    "BC": dict(alpha=[12.36, 13.04, 13.91, 14.97, 16.22, 17.65], E=[152, 151, 145, 132, 109, 71.9],
               nu=[0.32, 0.33, 0.33, 0.34, 0.35, 0.35]),
    "TGO": dict(alpha=[5.08, 5.90, 6.73, 7.55, 8.38, 9.20], E=[380, 369, 361, 352, 336, 309],
                nu=[0.27] * 6),
}
SUB_T = np.array([20, 700, 1000.0])
SUB = dict(alpha=[14, 15, 16.6], E=[200, 150, 150], nu=[0.3, 0.3, 0.3])
# ------------------------------------------------ Table 2 (mm, N, MPa)
GC = dict(TC=8e-3, BC=0.3, TGO=0.06, SUB=1e3, I_TC_TGO=0.01, I_TGO_BC=0.02)
FT = dict(TC=200.0, BC=500.0, TGO=380.0, SUB=1e5, I_TC_TGO=50.0, I_TGO_BC=200.0)
# substrate: not part of Table 2 -> elastic (huge ft); Gc chosen so that a1 > 3 (convex split)
# ------------------------------------------------ Table 3
D_OX = 2.0e-13 * 1e6         # m^2/s -> mm^2/s
G_OX = 1.0e-4                # m^3 mol^-1 s^-1  (c in mol/m^3 -> 1/s)
M_OX = 1.11e5                # mol/m^3
ALPHA_N = 8e-3
C_BAR = 1.55                 # mol/m^3 on TC/TGO interface
T_OX = 850.0                 # oxidation only above this temperature
T_REF = 1150.0               # stress-free temperature


def interp_ext(T, xs, ys):
    """linear interpolation with linear extrapolation (needed up to 1150 C)"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if T > xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (T - xs[-1])
    return float(np.interp(T, xs, ys))


def props(mat, T):
    """E (MPa), nu, thermal strain relative to T_REF (secant CTE from 20 C)."""
    xs, tb = (SUB_T, SUB) if mat == "SUB" else (TAB_T, TAB[mat])
    E = interp_ext(T, xs, tb["E"]) * 1e3
    nu = interp_ext(T, xs, tb["nu"])
    a = lambda TT: interp_ext(TT, xs, tb["alpha"]) * 1e-6
    eth = a(T) * (T - 20.0) - a(T_REF) * (T_REF - 20.0)
    return E, nu, eth


class TBCModel:
    def __init__(self, Lw, A, W, t_tgo, h_fine, lphi, l, out, refine=(-0.02, 0.02),
                 damage=True, oxidation=False, h_coarse=0.05):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.Lw, self.A, self.Wd_width, self.t_tgo = Lw, A, W, t_tgo
        self.l, self.lphi, self.damage, self.oxidation = l, lphi, damage, oxidation
        self._mesh(h_fine, h_coarse, refine)
        self._setup()

    # -------------------------------------------------------------- mesh
    def _mesh(self, h_fine, h_coarse, refine):
        Lw, A, W, tt = self.Lw, self.A, self.Wd_width, self.t_tgo
        y_top, y_bc, y_sub = 0.5, -0.1, -3.1
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("tbc")
        g = gmsh.model.geo
        npts = 121
        xs = np.linspace(0, W, npts)

        def curve(shift):
            pts = [g.addPoint(x, A * np.cos(2 * np.pi * x / Lw) + shift, 0) for x in xs]
            return pts, g.addSpline(pts)

        pI, cI = curve(0.0)                      # TC/TGO (or TC/BC) interface
        P = lambda x, y: g.addPoint(x, y, 0)
        pTL, pTR = P(0, y_top), P(W, y_top)
        pBL, pBR = P(0, y_bc), P(W, y_bc)
        pSL, pSR = P(0, y_sub), P(W, y_sub)
        lT = g.addLine(pTL, pTR)
        lTr = g.addLine(pI[-1], pTR)
        lTl = g.addLine(pI[0], pTL)
        sTC = g.addPlaneSurface([g.addCurveLoop([cI, lTr, -lT, -lTl])])
        surf = {"TC": [sTC]}
        if tt > 0:
            pJ, cJ = curve(-tt)                  # TGO/BC interface
            l1, l2 = g.addLine(pJ[0], pI[0]), g.addLine(pJ[-1], pI[-1])
            surf["TGO"] = [g.addPlaneSurface([g.addCurveLoop([cJ, l2, -cI, -l1])])]
            low, lowc = pJ, cJ
        else:
            low, lowc = pI, cI
        lB = g.addLine(pBL, pBR)
        lBr, lBl = g.addLine(pBR, low[-1]), g.addLine(pBL, low[0])
        surf["BC"] = [g.addPlaneSurface([g.addCurveLoop([lB, lBr, -lowc, -lBl])])]
        lS = g.addLine(pSL, pSR)
        lSr, lSl = g.addLine(pSR, pBR), g.addLine(pSL, pBL)
        surf["SUB"] = [g.addPlaneSurface([g.addCurveLoop([lS, lSr, -lB, -lSl])])]
        g.synchronize()
        self.tags = {"TC": 1, "TGO": 2, "BC": 3, "SUB": 4}
        for k, v in surf.items():
            gmsh.model.addPhysicalGroup(2, v, self.tags[k])
        gmsh.model.addPhysicalGroup(1, [cI], 11)
        if tt > 0:
            gmsh.model.addPhysicalGroup(1, [cJ], 12)
        f = gmsh.model.mesh.field
        box = f.add("Box")
        f.setNumber(box, "VIn", h_fine)
        f.setNumber(box, "VOut", h_coarse)
        f.setNumber(box, "XMin", -1); f.setNumber(box, "XMax", W + 1)
        f.setNumber(box, "YMin", refine[0]); f.setNumber(box, "YMax", refine[1])
        f.setNumber(box, "Thickness", 0.03)
        f.setAsBackgroundMesh(box)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        self.msh, self.ct, self.ft = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
        gmsh.finalize()
        self.ncell = self.msh.topology.index_map(2).size_global
        print(f"cells: {self.ncell}", flush=True)

    # -------------------------------------------------------------- fields
    def _setup(self):
        msh, l = self.msh, self.l
        self.V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
        self.Ws = fem.functionspace(msh, ("Lagrange", 1))
        self.W0 = fem.functionspace(msh, ("DG", 0))
        W, W0 = self.Ws, self.W0
        self.chi = {}
        for k, tg in self.tags.items():
            c = fem.Function(W0)
            c.x.array[self.ct.find(tg)] = 1.0
            self.chi[k] = c
        # interface phase fields
        msh.topology.create_connectivity(1, 2)
        self.phi = []
        for tag in (11, 12):
            facets = self.ft.find(tag)
            if facets.size == 0:
                continue
            dofs = fem.locate_dofs_topological(W, 1, facets)
            ph = pf.solve_interface_phase_field(W, dofs, self.lphi)
            ph.name = f"phi{tag}"
            self.phi.append(ph)
        # oxidation fields
        self.n = fem.Function(W, name="n")
        self.c = fem.Function(W, name="c")
        # temperature-dependent constants
        self.T = 1150.0
        self.K = {}
        for m in ("TC", "BC", "TGO", "SUB"):
            self.K[m] = [fem.Constant(msh, 0.0) for _ in range(3)]   # E, nu, eth
        self.update_T(T_REF)
        ch, n = self.chi, self.n

        def mix(i):
            K = self.K
            bc = (1 - n) * K["BC"][i] + n * K["TGO"][i]
            return ch["TC"] * K["TC"][i] + ch["TGO"] * K["TGO"][i] + ch["BC"] * bc + ch["SUB"] * K["SUB"][i]

        E, nu, eth = mix(0), mix(1), mix(2)
        eig = eth + ch["BC"] * ALPHA_N * n                      # Eqs. (13), (14)
        self.E_x = E
        lm = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        # bulk Gc / ft
        bulk = lambda P: (ch["TC"] * P["TC"] + ch["TGO"] * P["TGO"]
                          + ch["BC"] * ((1 - n) * P["BC"] + n * P["TGO"]) + ch["SUB"] * P["SUB"])
        Gc, ftx = bulk(GC), bulk(FT)
        # smeared interfaces, Eq. (10) & (4)
        self.Gbar = []
        spec = [("I_TC_TGO", "TC", "TGO"), ("I_TGO_BC", "TGO", "BC")]
        for ph, (key, mA, mB) in reversed(list(zip(self.phi, spec))):
            Gb = pf.gc_bar(GC[key], GC[mA], GC[mB], l, self.lphi)
            self.Gbar.append((key, Gb))
            print(f"{key}: Gi={GC[key]}, Gc_bar={Gb:.4e}")
            gph = 1 - (1 - ph) ** 2
            Gc = gph * Gb + (1 - gph) * Gc
            ftx = gph * FT[key] + (1 - gph) * ftx
        self.Gc, self.ftx = Gc, ftx
        a1 = pf.a1_coef(E, Gc, ftx, l)
        V = self.V
        self.u = fem.Function(V, name="u")
        self.u1 = fem.Function(V)
        self.d = fem.Function(W, name="d")
        self.d_old, self.d_it = fem.Function(W), fem.Function(W)
        d = self.d
        w_d = pf.omega(d, a1) if self.damage else 1.0
        self.w_d = w_d
        du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        I2 = ufl.Identity(2)
        a_u = w_d * ufl.inner(pf.sigma0(pf.eps(du), lm, mu), pf.eps(v)) * pf.DX
        # generalised plane strain: the out-of-plane total strain follows the (3 mm thick)
        # substrate, eps_zz = eth_SUB(T); elastic out-of-plane strain ezz_e = eps_zz - eig.
        # sigma = lm (tr eps + eps_zz - 3 eig) I + 2 mu (eps - eig I)
        ezz0 = self.K["SUB"][2]
        ezz_e = ezz0 - eig
        L_u = w_d * ufl.inner(((3 * lm + 2 * mu) * eig - lm * ezz0) * I2, pf.eps(v)) * pf.DX
        L_0 = ufl.inner(fem.Constant(msh, PETSc.ScalarType((0.0, 0.0))), v) * pf.DX
        Vx, _ = V.sub(0).collapse()
        Vy, _ = V.sub(1).collapse()
        Wd = self.Wd_width
        left = fem.locate_dofs_geometrical((V.sub(0), Vx), lambda x: np.isclose(x[0], 0.0))
        right = fem.locate_dofs_geometrical((V.sub(0), Vx), lambda x: np.isclose(x[0], Wd))
        bot = fem.locate_dofs_geometrical((V.sub(1), Vy), lambda x: np.isclose(x[1], -3.1))
        z = PETSc.ScalarType(0.0)
        bc_l = fem.dirichletbc(z, left[0], V.sub(0))
        bc_b = fem.dirichletbc(z, bot[0], V.sub(1))
        self.bcs0 = [bc_l, bc_b, fem.dirichletbc(z, right[0], V.sub(0))]
        self.bcs1 = [bc_l, bc_b, fem.dirichletbc(PETSc.ScalarType(1.0), right[0], V.sub(0))]
        self.mech0 = pf.LinearSolver(a_u, L_u, self.u, self.bcs0)
        self.mech1 = pf.LinearSolver(a_u, L_0, self.u1, self.bcs1)
        self.mech1.ksp = self.mech0.ksp        # same matrix -> reuse factorisation
        self.mech1.A = self.mech0.A
        x = ufl.SpatialCoordinate(msh)
        psi = ufl.as_vector((x[0] / Wd, 0.0))
        e_el = lambda uu: pf.eps(uu) - eig * I2
        self.R0 = fem.form(w_d * ufl.inner(pf.sigma0(e_el(self.u), lm, mu, ezz_e), pf.eps(psi)) * pf.DX)
        self.R1 = fem.form(w_d * ufl.inner(pf.sigma0(pf.eps(self.u1), lm, mu), pf.eps(psi)) * pf.DX)
        ee = e_el(self.u)
        self.sig = w_d * pf.sigma0(ee, lm, mu, ezz_e)
        self.H = pf.History(W0, pf.psi_plus(ee, lm, mu, ezz_e), ftx**2 / (2 * E))
        if self.damage:
            self.pfs = pf.PhaseFieldSolver(
                pf.phase_field_energy(d, self.d_old, self.d_it, self.H.H, Gc, a1, l), d, self.d_old)
        self.E_frac = fem.form(Gc * pf.gamma_density(d, l) * pf.DX)
        # max principal (in-plane) stress, DG0
        s = self.sig
        s1 = (s[0, 0] + s[1, 1]) / 2 + ufl.sqrt(((s[0, 0] - s[1, 1]) / 2) ** 2 + s[0, 1] ** 2 + 1e-30)
        self.s1 = fem.Function(W0, name="sigma_max_principal")
        self.s1_expr = fem.Expression(s1, W0.element.interpolation_points())
        self.syy = fem.Function(W0, name="sigma_yy")
        self.syy_expr = fem.Expression(s[1, 1], W0.element.interpolation_points())
        if self.oxidation:
            self._setup_oxidation()

    def update_T(self, T):
        self.T = T
        for m in ("TC", "BC", "TGO", "SUB"):
            E, nu, eth = props(m, T)
            self.K[m][0].value, self.K[m][1].value, self.K[m][2].value = E, nu, eth

    # -------------------------------------------------------------- oxidation
    def _setup_oxidation(self):
        W, msh = self.Ws, self.msh
        self.c_old = fem.Function(W)
        self.dt_c = fem.Constant(msh, 1.0)
        self.Gon = fem.Constant(msh, G_OX)
        cc, s = ufl.TrialFunction(W), ufl.TestFunction(W)
        chiBC = self.chi["BC"]
        w_d = self.w_d
        # vertex quadrature (lumped reaction/capacity, one rule for the whole form):
        # M-matrix -> c >= 0 and a nodal oxygen balance that is consistent with the
        # nodal update of n, even though the reaction length sqrt(D/(M G)) ~ 0.13 um < h
        DXv = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
        a = (cc * s / self.dt_c + D_OX * ufl.dot(ufl.grad(cc), ufl.grad(s))
             + M_OX * w_d * self.Gon * chiBC * (1 - self.n) * cc * s) * DXv
        L = self.c_old * s / self.dt_c * DXv
        tc_cells = self.ct.find(self.tags["TC"])
        tc_dofs = np.unique(W.dofmap.list[tc_cells].ravel())
        bc_cells = self.ct.find(self.tags["BC"])
        self.bc_dofs = np.unique(W.dofmap.list[bc_cells].ravel())
        self.c.x.array[tc_dofs] = C_BAR
        self.csolver = pf.LinearSolver(a, L, self.c, [fem.dirichletbc(PETSc.ScalarType(C_BAR), tc_dofs, W)])
        # omega(d) at nodes for the n update
        self.wd_nodes = fem.Function(W)
        self.wd_nodes.x.array[:] = 1.0
        self.wd_expr = (fem.Expression(self.w_d, W.element.interpolation_points())
                        if self.damage else None)

    def oxidise(self, dt):
        """one backward-Euler step of the reaction-diffusion Eq. (45) followed by
        the exact integration of Eq. (47) with frozen c."""
        if self.T <= T_OX:
            return
        self.dt_c.value = dt
        self.c_old.x.array[:] = self.c.x.array
        self.csolver.solve()
        self.c.x.array[:] = np.maximum(self.c.x.array, 0.0)
        if self.wd_expr is not None:
            self.wd_nodes.interpolate(self.wd_expr)
        idx = self.bc_dofs
        n = self.n.x.array
        n[idx] = 1 - (1 - n[idx]) * np.exp(-self.wd_nodes.x.array[idx] * G_OX * self.c.x.array[idx] * dt)

    # -------------------------------------------------------------- mechanics
    def solve_u(self):
        self.mech0.solve()
        self.mech1.solve(assemble=False)
        R0 = pf.assemble_scalar(self.R0)
        R1 = pf.assemble_scalar(self.R1)
        a = -R0 / R1
        self.u.x.array[:] += a * self.u1.x.array
        self.u.x.scatter_forward()

    def step(self, tol=1e-4, max_it=100):
        if not self.damage:
            self.solve_u()
            return 1, 0.0
        it, err = pf.staggered(self.solve_u, self.H, self.pfs, self.d, self.d_it, tol=tol, max_it=max_it)
        self.H.commit()
        self.d_old.x.array[:] = self.d.x.array
        return it, err

    def sample(self, f, pts):
        """evaluate a scalar field at points (N x 2), serial"""
        from dolfinx import geometry
        if not hasattr(self, "_bb"):
            self._bb = geometry.bb_tree(self.msh, 2)
        p3 = np.c_[pts, np.zeros(len(pts))]
        cand = geometry.compute_collisions_points(self._bb, p3)
        coll = geometry.compute_colliding_cells(self.msh, cand, p3)
        cells = np.array([coll.links(i)[0] for i in range(len(p3))], dtype=np.int32)
        return f.eval(p3, cells).ravel()

    def tgo_thickness(self, x0, y_top, depth=0.02, npts=2001, thr=0.8):
        """TGO thickness (n > thr) below y_top along the vertical line x = x0"""
        ys = np.linspace(y_top - 1e-7, y_top - depth, npts)
        nv = self.sample(self.n, np.c_[np.full(npts, x0), ys])
        inside = np.where(nv > thr)[0]
        if inside.size == 0:
            return 0.0
        # contiguous from the interface downward
        k = 0
        while k < len(inside) and inside[k] == k:
            k += 1
        return float(y_top - ys[k - 1]) if k > 0 else 0.0

    def dmax_layers(self):
        out = []
        for k in ("TC", "TGO", "BC", "SUB"):
            cells = self.ct.find(self.tags[k])
            dofs = np.unique(self.Ws.dofmap.list[cells].ravel()) if cells.size else []
            out.append(round(float(self.d.x.array[dofs].max()), 3) if len(dofs) else None)
        return out

    def fracture_energy(self):
        return pf.assemble_scalar(self.E_frac)

    def update_stress(self):
        self.s1.interpolate(self.s1_expr)
        self.syy.interpolate(self.syy_expr)

    # -------------------------------------------------------------- plotting
    def tri(self):
        if not hasattr(self, "_tri"):
            self._tri = pf.p1_triangulation(self.Ws)
        return self._tri

    def plot_nodal(self, ax, f, title, ylim=(-0.04, 0.04), cmap="jet", vmin=None, vmax=None):
        c = ax.tripcolor(self.tri(), f.x.array, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        self._decor(ax, title, ylim)
        return c

    def plot_cell(self, ax, f, title, ylim=(-0.04, 0.04), cmap="jet", vmin=None, vmax=None,
                  layers=None):
        """DG0 field; `layers` (e.g. ["TC"]) restricts the plot to those layers."""
        import matplotlib.tri as mtri
        t0 = self.tri()
        vals = f.x.array.copy()
        tri = t0
        if layers is not None:
            keep = sum(self.chi[k].x.array for k in layers) > 0.5
            tri = mtri.Triangulation(t0.x, t0.y, t0.triangles[keep])
            vals = vals[keep]
            if vmin is None:
                vmin, vmax = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
        c = ax.tripcolor(tri, facecolors=vals, cmap=cmap, vmin=vmin, vmax=vmax)
        self._decor(ax, title, ylim)
        return c

    def _decor(self, ax, title, ylim):
        xs = np.linspace(0, self.Wd_width, 200)
        y = self.A * np.cos(2 * np.pi * xs / self.Lw)
        ax.plot(xs, y, "w-", lw=0.6)
        if self.t_tgo > 0:
            ax.plot(xs, y - self.t_tgo, "w-", lw=0.6)
        ax.set_xlim(0, self.Wd_width); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
