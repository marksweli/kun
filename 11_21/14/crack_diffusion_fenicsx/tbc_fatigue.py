"""Thermo-chemo-mechanical fatigue phase-field model of a TBC unit cell under
cyclic gas exposure -- FEniCSx reproduction of Sec. 4.1.2 ("Cracks affecting
diffusion process") of

  W. Wang, B. Li, B. Yang, Y. Li, Int. J. Fatigue 201 (2025) 109134.

Fields / equations (per time step, staggered):
  1. heat conduction            rho cp dT/dt = div(kappa grad T)          (Eq. 36)
  2. O2 diffusion + oxidation   dc/dt = div(D grad c) - R dxi/dt          (Eq. 47)
                                dxi/dt = (1-xi) M(T) c                    (Eq. 38)
                                D = D0 exp(-Qd/RT) (1 + k_d <d-d_th>)     (Eq. 45, 56)
  3. mechanics (gen. plane strain, sigma = omega(d) C:(eps-eps_th-eps_g))  (Eq. 35)
  4. fatigue PF-CZM phase field (Eq. 27-34, 52-54), staggered with 3.
"""
import argparse
import json
import os
import time

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, io
from dolfinx.fem import petsc as fpetsc

COMM = MPI.COMM_WORLD
RGAS = 8.314
UM = 1e-6

# ------------------------------------------------------------ parameters
P = argparse.ArgumentParser()
P.add_argument("--cycles", type=int, default=800)
P.add_argument("--h", type=float, default=0.6, help="fine mesh size [um]")
P.add_argument("--amp", type=float, default=15.0, help="interface amplitude A [um]")
P.add_argument("--k", type=float, default=1.0, help="fatigue accumulation parameter k")
P.add_argument("--kd", type=float, default=4.737e8, help="crack diffusion amplification (0: ideal crack)")
P.add_argument("--dth", type=float, default=0.8)
P.add_argument("--Th", type=float, default=0.5 * (1573.0 + 1000.0), help="hold temperature [K]")
P.add_argument("--cb", type=float, default=2.8, help="interface O2 concentration at Th [mol/m3]")
P.add_argument("--Dair", type=float, default=2.0e-5, help="O2 diffusivity in air [m2/s] (calibrates Qd)")
P.add_argument("--Rcons", type=float, default=1.5 * 3950 / 0.10196, help="O2 consumed per unit TGO volume [mol/m3]")
P.add_argument("--front", type=float, default=1.0, help="reaction front width sqrt(D/(R M)) at Th [um]")
P.add_argument("--n_heat", type=int, default=2)
P.add_argument("--n_hold", type=int, default=3)
P.add_argument("--n_cool", type=int, default=2)
P.add_argument("--stag_tol", type=float, default=1e-3)
P.add_argument("--stag_max", type=int, default=60)
P.add_argument("--snap", type=str, default="132,229,300,324,459,511,744")
P.add_argument("--diag", action="store_true")
P.add_argument("--pu", type=int, default=2, help="displacement degree")
P.add_argument("--band_up", type=float, default=12.0, help="fine band above interface [um]")
P.add_argument("--grade", type=float, default=0.2, help="mesh size growth per um outside band")
P.add_argument("--outdir", type=str, default="results")
args = P.parse_args()
os.makedirs(args.outdir, exist_ok=True)

W, H_SUB, H_BC, H_TC = 100.0, 3000.0, 100.0, 200.0   # um
Y_PEAK = H_SUB + H_BC                                  # TC/TGO peak level
A = args.amp
T0 = 293.15                                            # stress-free / ambient
Th = args.Th
B0 = 1.2e-6                                            # phase-field length b0
H0_TGO = 1.0                                           # initial TGO [um]
EPS_G = 0.003                                          # TGO growth strain

# Table 1-2 (+ substrate made unbreakable)
MAT = {  # E, nu, ft, rho, kappa, cp, alpha
    "TC":  dict(E=30e9, nu=0.12, ft=380e6, rho=5650, ka=1.16, cp=450, al=9.8e-6),
    "TGO": dict(E=320e9, nu=0.25, ft=900e6, rho=3978, ka=4.4, cp=1000, al=9.2e-6),
    "BC":  dict(E=110e9, nu=0.33, ft=600e6, rho=7320, ka=14.5, cp=780, al=14e-6),
    "SUB": dict(E=160e9, nu=0.33, ft=1e10, rho=8110, ka=26.0, cp=680, al=15e-6),
}
for k_, m in MAT.items():
    lm = m["E"] * m["nu"] / ((1 + m["nu"]) * (1 - 2 * m["nu"]))
    mu = m["E"] / (2 * (1 + m["nu"]))
    # G_c = 256 f_t^2 b0 / (27 Ebar), Ebar = lambda + 2 mu (Sec. 2.3)
    m["Gc"] = 256 * m["ft"] ** 2 * B0 / (27 * (lm + 2 * mu)) if k_ != "SUB" else 1e4
    m["a1"] = 4 * m["E"] * m["Gc"] / (np.pi * B0 * m["ft"] ** 2)

# diffusion / oxidation (Table 3 + calibration, see README)
D0, CMAX = 5.1e-9, 1.11e5
D_TH = args.Dair / (1 + args.kd * (1 - args.dth)) if args.kd > 0 else 2.0e-5 / (1 + 4.737e8 * 0.2)
QD = RGAS * Th * np.log(D0 / D_TH)                     # D(Th) = D_TH
P_O2 = 15 * 1.01325 * 0.21                             # bar
MU0 = -RGAS * Th * np.log(args.cb / (CMAX * np.sqrt(P_O2)))   # Eq. 44 -> c(Th) = cb
RC = args.Rcons
M_TH = D_TH / (RC * (args.front * UM) ** 2)

t_heat, t_hold, t_cool = 900.0, 5400.0, 900.0
T_CYC = t_heat + t_hold + t_cool


def T_bc(t):
    s = t % T_CYC
    if s < t_heat:
        return T0 + (Th - T0) * s / t_heat
    if s < t_heat + t_hold:
        return Th
    return Th - (Th - T0) * (s - t_heat - t_hold) / t_cool


def schedule(ncyc):
    out = []
    for n in range(ncyc):
        off = 0.0
        for dt, m in ((t_heat, args.n_heat), (t_hold, args.n_hold), (t_cool, args.n_cool)):
            for j in range(m):
                out.append((n + 1, n * T_CYC + off + dt * (j + 1) / m, dt / m))
            off += dt
    return out


# ------------------------------------------------------------ mesh (gmsh)
def y_int(x):
    return Y_PEAK - A + A * np.cos(2 * np.pi * x / W)


def build_mesh():
    import gmsh
    from dolfinx.io import gmshio
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if COMM.rank == 0:
        g = gmsh.model.geo
        p = [g.addPoint(0, 0, 0), g.addPoint(W, 0, 0), g.addPoint(W, H_SUB, 0), g.addPoint(0, H_SUB, 0)]
        xs = np.linspace(0, W, 201)
        ip = [g.addPoint(x, y_int(x), 0) for x in xs]
        pt = [g.addPoint(W, Y_PEAK + H_TC, 0), g.addPoint(0, Y_PEAK + H_TC, 0)]
        l_bot = g.addLine(p[0], p[1]); l_sr = g.addLine(p[1], p[2]); l_sbc = g.addLine(p[2], p[3])
        l_sl = g.addLine(p[3], p[0])
        l_br = g.addLine(p[2], ip[-1]); l_int = g.addSpline(ip); l_bl = g.addLine(ip[0], p[3])
        l_tr = g.addLine(ip[-1], pt[0]); l_top = g.addLine(pt[0], pt[1]); l_tl = g.addLine(pt[1], ip[0])
        s_sub = g.addPlaneSurface([g.addCurveLoop([l_bot, l_sr, l_sbc, l_sl])])
        s_bc = g.addPlaneSurface([g.addCurveLoop([-l_sbc, l_br, -l_int, l_bl])])
        s_tc = g.addPlaneSurface([g.addCurveLoop([l_int, l_tr, l_top, l_tl])])
        g.synchronize()
        m = gmsh.model
        m.addPhysicalGroup(2, [s_tc], 1); m.addPhysicalGroup(2, [s_bc], 2); m.addPhysicalGroup(2, [s_sub], 3)
        m.addPhysicalGroup(1, [l_top], 11); m.addPhysicalGroup(1, [l_bot], 12)
        m.addPhysicalGroup(1, [l_sl, l_bl, l_tl], 13); m.addPhysicalGroup(1, [l_sr, l_br, l_tr], 14)
        m.addPhysicalGroup(1, [l_int], 15)
        h = args.h
        s = f"(y-({Y_PEAK - A}+{A}*Cos(2*Pi*x/{W})))"
        below = f"(Fabs({s}+26)-({s}+26))/2"     # distance below the fine band
        above = f"(Fabs({s}-{args.band_up})+({s}-{args.band_up}))/2"       # distance above the fine band
        f = m.mesh.field.add("MathEval")
        m.mesh.field.setString(f, "F", f"{h}+{args.grade}*{below}+{args.grade}*{above}")
        f2 = m.mesh.field.add("MathEval")
        m.mesh.field.setString(f2, "F", "60")
        fm = m.mesh.field.add("Min")
        m.mesh.field.setNumbers(fm, "FieldsList", [f, f2])
        m.mesh.field.setAsBackgroundMesh(fm)
        for o in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
            gmsh.option.setNumber(o, 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        m.mesh.generate(2)
    msh, ct, ft = gmshio.model_to_mesh(gmsh.model, COMM, 0, gdim=2)
    gmsh.finalize()
    msh.geometry.x[:, :2] *= UM
    return msh, ct, ft


t_start = time.time()
msh, ctags, ftags = build_mesh()
ncell = msh.topology.index_map(2).size_global
if COMM.rank == 0:
    print(f"mesh: {ncell} triangles, {msh.topology.index_map(0).size_global} vertices", flush=True)
tdim = 2
msh.topology.create_connectivity(1, 2)

# ------------------------------------------------------------ spaces & material indicators
V1 = fem.functionspace(msh, ("Lagrange", 1))
Vu = fem.functionspace(msh, ("Lagrange", args.pu, (2,)))
W0 = fem.functionspace(msh, ("DG", 0))
x = ufl.SpatialCoordinate(msh)

chi = {}
for name, tag in (("TC", 1), ("BC", 2), ("SUB", 3)):
    chi[name] = fem.Function(W0)
    chi[name].x.array[ctags.find(tag)] = 1.0

xi = fem.Function(V1, name="xi")          # oxidation extent (TGO fraction)
xi_n = fem.Function(V1)
xyz = V1.tabulate_dof_coordinates()
yint_dofs = y_int(xyz[:, 0] / UM) * UM
xi.x.array[:] = np.where((xyz[:, 1] <= yint_dofs + 1e-9) & (xyz[:, 1] >= yint_dofs - H0_TGO * UM), 1.0, 0.0)
xi0 = fem.Function(V1)
xi0.x.array[:] = xi.x.array
xi_n.x.array[:] = xi.x.array


def prop(key):
    """Eq. (40): BC/TGO mixture by xi in the BC domain."""
    return (chi["TC"] * MAT["TC"][key] + chi["SUB"] * MAT["SUB"][key]
            + chi["BC"] * ((1 - xi) * MAT["BC"][key] + xi * MAT["TGO"][key]))


E_, nu_ = prop("E"), prop("nu")
lam_ = E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_))
mu_ = E_ / (2 * (1 + nu_))
ft_, Gc_ = prop("ft"), prop("Gc")
a1_ = 4 * E_ * Gc_ / (np.pi * B0 * ft_ ** 2)
Y0_ = ft_ ** 2 / (2 * E_)

# ------------------------------------------------------------ 1. heat
T = fem.Function(V1, name="T")
T.x.array[:] = T0
T_n = fem.Function(V1)
T_n.x.array[:] = T0
dt_c = fem.Constant(msh, 1.0)
Tt, wT = ufl.TrialFunction(V1), ufl.TestFunction(V1)
dxv = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
rcp = prop("rho") * prop("cp")
aT = (rcp * Tt * wT / dt_c) * dxv + prop("ka") * ufl.dot(ufl.grad(Tt), ufl.grad(wT)) * dxv
LT = rcp * T_n * wT / dt_c * dxv
top_dofs = fem.locate_dofs_topological(V1, 1, ftags.find(11))
bot_dofs = fem.locate_dofs_topological(V1, 1, ftags.find(12))
T_top, T_bot = fem.Constant(msh, T0), fem.Constant(msh, T0)
bcT = [fem.dirichletbc(T_top, top_dofs, V1), fem.dirichletbc(T_bot, bot_dofs, V1)]
heat = fpetsc.LinearProblem(aT, LT, bcs=bcT, u=T, petsc_options={
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

# ------------------------------------------------------------ 2. diffusion / oxidation
d = fem.Function(V1, name="d")
c = fem.Function(V1, name="c")
c_n = fem.Function(V1)
wc = ufl.TestFunction(V1)
D_T = D0 * ufl.exp(-QD / (RGAS * T))
M_T = M_TH * ufl.exp(-QD / RGAS * (1 / T - 1 / Th))
D_eff = D_T * (1 + args.kd * ufl.max_value(d - args.dth, 0.0))
ct = ufl.TrialFunction(V1)
# linear-implicit consumption R (1-xi_n) M c^{n+1}; then exact local update of xi (M c dt << 1)
ac = (ct / dt_c + RC * (1 - xi_n) * M_T * ct) * wc * dxv + D_eff * ufl.dot(ufl.grad(ct), ufl.grad(wc)) * dxv
Lc = c_n / dt_c * wc * dxv
dxi = (1 - xi_n) * (1 - ufl.exp(-M_T * ufl.max_value(c, 0.0) * dt_c))
# c = cb(T) on TC dofs (incl. the TC/TGO interface), c = 0 in the substrate
tc_cells, sub_cells = ctags.find(1), ctags.find(3)
dofs_tc = np.unique(V1.dofmap.list[tc_cells])
dofs_sub_all = np.unique(V1.dofmap.list[sub_cells])
dofs_bc_all = np.unique(V1.dofmap.list[ctags.find(2)])
dofs_sub = np.setdiff1d(dofs_sub_all, dofs_bc_all)
c_b = fem.Constant(msh, 0.0)
bcc = [fem.dirichletbc(c_b, dofs_tc, V1), fem.dirichletbc(PETSc.ScalarType(0.0), dofs_sub, V1)]
prob_c = fpetsc.LinearProblem(ac, Lc, bcs=bcc, u=c, petsc_options={
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
xi_expr = fem.Expression(xi_n + dxi, V1.element.interpolation_points())


def c_boundary(Tv):
    return CMAX * np.exp(-MU0 / (RGAS * Tv)) * np.sqrt(P_O2)


# ------------------------------------------------------------ 3. mechanics
u = fem.Function(Vu, name="u")
u0, u1 = fem.Function(Vu), fem.Function(Vu)
du, v = ufl.TrialFunction(Vu), ufl.TestFunction(Vu)
DXq = ufl.dx(metadata={"quadrature_degree": 2})
K_RES = 1e-8


def omega(dd):
    Q = a1_ * dd * (1 - 0.5 * dd)                     # p=2, a2=-0.5, a3=0 (linear softening)
    return (1 - dd) ** 2 / ((1 - dd) ** 2 + Q) * (1 - K_RES) + K_RES


eps_star = (chi["TC"] * MAT["TC"]["al"] * (T - T0) + chi["SUB"] * MAT["SUB"]["al"] * (T - T0)
            + chi["BC"] * (((1 - xi) * MAT["BC"]["al"] + xi * MAT["TGO"]["al"]) * (T - T0)
                           # oxide is formed stress-free at Th (Sec. 'Assumptions' in README)
                           + xi * (MAT["BC"]["al"] - MAT["TGO"]["al"]) * (Th - T0)
                           + EPS_G * (xi - xi0)))
ezz = MAT["SUB"]["al"] * (T - T0)                    # generalized plane strain


def sig_eff(uu, with_eig=True):
    e = ufl.sym(ufl.grad(uu))
    if with_eig:
        return lam_ * (ufl.tr(e) + ezz - 3 * eps_star) * ufl.Identity(2) + 2 * mu_ * (e - eps_star * ufl.Identity(2))
    return lam_ * ufl.tr(e) * ufl.Identity(2) + 2 * mu_ * e


a_u = fem.form(omega(d) * ufl.inner(sig_eff(du, False), ufl.sym(ufl.grad(v))) * DXq)
e_v = ufl.sym(ufl.grad(v))
L_u = fem.form(omega(d) * (lam_ * (3 * eps_star - ezz) * ufl.tr(e_v) + 2 * mu_ * eps_star * ufl.tr(e_v)) * DXq)
L_zero = fem.form(ufl.inner(fem.Constant(msh, (0.0, 0.0)), v) * DXq)
R0_form = fem.form(omega(d) * ufl.inner(sig_eff(u0, True), e_v) * DXq)
R1_form = fem.form(omega(d) * ufl.inner(sig_eff(u1, False), e_v) * DXq)

Vx, _ = Vu.sub(0).collapse()
left = np.concatenate([ftags.find(13)])
right = ftags.find(14)
dl = fem.locate_dofs_topological((Vu.sub(0), Vx), 1, left)
dr = fem.locate_dofs_topological((Vu.sub(0), Vx), 1, right)
Vy, _ = Vu.sub(1).collapse()
db = fem.locate_dofs_topological((Vu.sub(1), Vy), 1, ftags.find(12))
zx, zy = fem.Function(Vx), fem.Function(Vy)
onex = fem.Function(Vx)
onex.x.array[:] = 1e-6
bc_l, bc_b = fem.dirichletbc(zx, dl, Vu.sub(0)), fem.dirichletbc(zy, db, Vu.sub(1))
bcs0 = [bc_l, bc_b, fem.dirichletbc(zx, dr, Vu.sub(0))]
bcs1 = [bc_l, bc_b, fem.dirichletbc(onex, dr, Vu.sub(0))]
right_x_dofs = dr[0]
bs = Vu.dofmap.index_map_bs

Au = fpetsc.create_matrix(a_u)
bu = fpetsc.create_vector(L_u)
rvec = fpetsc.create_vector(R0_form)
ksp_u = PETSc.KSP().create(COMM)
ksp_u.setType("preonly"); ksp_u.getPC().setType("lu"); ksp_u.getPC().setFactorSolverType("mumps")


def _solve_lin(L, bcs, out):
    with bu.localForm() as bl:
        bl.set(0)
    fpetsc.assemble_vector(bu, L)
    fpetsc.apply_lifting(bu, [a_u], bcs=[bcs])
    bu.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fpetsc.set_bc(bu, bcs)
    ksp_u.solve(bu, out.x.petsc_vec)
    out.x.scatter_forward()


def _reaction(form):
    with rvec.localForm() as rl:
        rl.set(0)
    fpetsc.assemble_vector(rvec, form)
    rvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    arr = rvec.array
    nloc = Vu.dofmap.index_map.size_local * bs
    loc = right_x_dofs[right_x_dofs < nloc]
    return COMM.allreduce(float(np.sum(arr[loc])), op=MPI.SUM)


def solve_mech():
    """Right edge kept straight & traction-free on average: u = u0 + a*u1."""
    Au.zeroEntries()
    fpetsc.assemble_matrix(Au, a_u, bcs=bcs0)    # same constrained dofs for both
    Au.assemble()
    ksp_u.setOperators(Au)
    _solve_lin(L_u, bcs0, u0)
    _solve_lin(L_zero, bcs1, u1)
    a = -_reaction(R0_form) / _reaction(R1_form)
    u.x.array[:] = u0.x.array + a * u1.x.array


# effective (undamaged) stress -> crack driving force Y_n = <sigma1>^2 / 2E  (Eq. 54)
s_eff = sig_eff(u, True)
s1 = 0.5 * (s_eff[0, 0] + s_eff[1, 1]) + ufl.sqrt(0.25 * (s_eff[0, 0] - s_eff[1, 1]) ** 2 + s_eff[0, 1] ** 2 + 1e-12)
Yn = ufl.max_value(s1, 0.0) ** 2 / (2 * E_)

# ------------------------------------------------------------ 4. fatigue phase field
d_old, d_k = fem.Function(V1), fem.Function(V1)
Hn = fem.Function(W0)          # max_t Y_n
Hf = fem.Function(W0, name="H")
abar = fem.Function(W0, name="abar")
alpha_s_old = fem.Function(W0)
fdeg = fem.Function(W0, name="f")
fdeg.x.array[:] = 1.0
pts0 = W0.element.interpolation_points()
# floor f*Y0: damage starts when sigma1 > sqrt(f) f_t, i.e. the degraded cohesive strength of Eq. (57)
H_expr = fem.Expression(ufl.max_value(ufl.max_value(Hn, Yn), (1 - 1e-3) * fdeg * Y0_), pts0)
Hn_expr = fem.Expression(ufl.max_value(Hn, Yn), pts0)
alpha_s = omega(d) * Yn                                  # Eq. (34)
alpha_s_expr = fem.Expression(alpha_s, pts0)
alphaT_expr = fem.Expression(Gc_ / (args.k * B0), pts0)  # Eq. (32)
alphaT = fem.Function(W0)

C0 = np.pi
wd = ufl.TestFunction(V1)
gd = ufl.grad(d)
Pi = (omega(d) * Hf + fdeg * Gc_ / (C0 * B0) * ((2 - 2 * d_k) * d + B0 ** 2 * ufl.dot(gd, gd))) * DXq
Fd = ufl.derivative(Pi, d, wd)
Fd_form, Jd_form = fem.form(Fd), fem.form(ufl.derivative(Fd, d))
Ad = fpetsc.create_matrix(Jd_form)
bd = fpetsc.create_vector(Fd_form)
lb, ub = fem.Function(V1), fem.Function(V1)
ub.x.array[:] = 1.0


def _Fd(snes, xx, b):
    xx.copy(d.x.petsc_vec); d.x.scatter_forward()
    with b.localForm() as bl:
        bl.set(0.0)
    fpetsc.assemble_vector(b, Fd_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


def _Jd(snes, xx, Am, Pm):
    Am.zeroEntries(); fpetsc.assemble_matrix(Am, Jd_form); Am.assemble()


snes = PETSc.SNES().create(COMM)
snes.setFunction(_Fd, bd); snes.setJacobian(_Jd, Ad)
snes.setType("vinewtonrsls")
snes.setTolerances(rtol=1e-9, atol=1e-9, max_it=60)
snes.getKSP().setType("preonly"); snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("petsc"); snes.getKSP().getPC().setFactorOrdering("nd")
xd = d.x.petsc_vec.copy()


def solve_pf():
    lb.x.array[:] = d_old.x.array
    snes.setVariableBounds(lb.x.petsc_vec, ub.x.petsc_vec)
    d.x.petsc_vec.copy(xd)
    snes.solve(None, xd)
    xd.copy(d.x.petsc_vec); d.x.scatter_forward()
    return snes.getConvergedReason()


def staggered():
    err, it = 1.0, 0
    while it < args.stag_max:
        it += 1
        d_k.x.array[:] = d.x.array
        solve_mech()
        Hf.interpolate(H_expr)
        r = solve_pf()
        err = COMM.allreduce(np.max(np.abs(d.x.array - d_k.x.array)), op=MPI.MAX)
        if err < args.stag_tol:
            break
    return it, err, r


# ------------------------------------------------------------ post-processing
crack_len_form = fem.form((2 * d - d ** 2 + B0 ** 2 * ufl.dot(gd, gd)) / (C0 * B0) * (chi["TC"] + chi["BC"]) * DXq)
frac_en_form = fem.form(fdeg * Gc_ * (2 * d - d ** 2 + B0 ** 2 * ufl.dot(gd, gd)) / (C0 * B0) * DXq)
diff_en_form = fem.form(RGAS * T * c * chi["BC"] * dxv)
oxy_form = fem.form(c * chi["BC"] * dxv)


def scalar(f):
    return COMM.allreduce(fem.assemble_scalar(f), op=MPI.SUM)


bbt = geometry.bb_tree(msh, tdim)


def probe_setup(x0):
    ys = np.arange(y_int(x0) - 45.0, y_int(x0) + 1e-6, 0.05)
    pts = np.zeros((ys.size, 3)); pts[:, 0] = x0 * UM; pts[:, 1] = ys * UM
    cand = geometry.compute_collisions_points(bbt, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    keep, cells = [], []
    for i in range(pts.shape[0]):
        lk = coll.links(i)
        if len(lk):
            keep.append(i); cells.append(lk[0])
    return pts[keep], np.array(cells, dtype=np.int32)


probes = {"peak": probe_setup(0.25), "peak_r": probe_setup(W - 0.25), "valley": probe_setup(W / 2)}


def tgo_thickness(name):
    pts, cells = probes[name]
    val = xi.eval(pts, cells).ravel()
    return float(np.sum(val > 0.5) * 0.05)


# DG0 cell-wise fields for plotting snapshots
d_fun_expr = fem.Expression(omega(d), pts0)
s1_dam_expr = fem.Expression(omega(d) * s1, pts0)
s1_fun = fem.Function(W0, name="sigma1")

snap_cycles = set(int(s) for s in args.snap.split(",") if s)
snapshots = {}
xdmf = io.XDMFFile(COMM, os.path.join(args.outdir, "fields.xdmf"), "w")
xdmf.write_mesh(msh)

hist_keys = ["step", "cycle", "t_h", "T", "stag_it", "stag_err", "crack_len_um", "frac_energy_Jpm",
             "diff_energy_Jpm", "oxygen_molpm", "tgo_peak_um", "tgo_valley_um", "d_max", "f_min", "phase"]
hist = []
info = dict(vars(args), ncell=int(ncell), QD=QD, D_Th=D_TH, MU0=MU0, M_Th=M_TH, P_O2_bar=P_O2,
            materials={k_: {kk: float(vv) for kk, vv in m.items()} for k_, m in MAT.items()})
if COMM.rank == 0:
    print(json.dumps({k_: v_ for k_, v_ in info.items() if k_ != "materials"}, indent=1), flush=True)
    print("Gc [J/m2]:", {k_: round(m["Gc"], 2) for k_, m in MAT.items()},
          " a1:", {k_: round(m["a1"], 2) for k_, m in MAT.items()}, flush=True)
    with open(os.path.join(args.outdir, "params.json"), "w") as fo:
        json.dump(info, fo, indent=1)

# ------------------------------------------------------------ time loop
sched = schedule(args.cycles)
alphaT.interpolate(alphaT_expr)
t_prev = 0.0
for istep, (cyc, t, dt) in enumerate(sched, start=1):
    dt_c.value = dt
    Tb = T_bc(t)
    T_top.value = Tb; T_bot.value = Tb       # TC surface and substrate bottom (uniform, see README)
    heat.solve(); T_n.x.array[:] = T.x.array
    Tm = float(Tb)
    phase = "heat" if (t - (cyc - 1) * T_CYC) <= t_heat + 1e-6 else (
        "hold" if (t - (cyc - 1) * T_CYC) <= t_heat + t_hold + 1e-6 else "cool")

    # diffusion / oxidation (d from previous step)
    c_b.value = c_boundary(Tm)
    c_n.x.array[:] = c.x.array
    prob_c.solve()
    xi.interpolate(xi_expr)
    xi.x.array[:] = np.clip(xi.x.array, 0.0, 1.0)
    xi_n.x.array[:] = xi.x.array

    # mechanics + fatigue phase field
    d_old.x.array[:] = d.x.array
    it, err, r = staggered()

    # history and fatigue variables (Eq. 31-34)
    Hn.interpolate(Hn_expr)
    tmp = fem.Function(W0); tmp.interpolate(alpha_s_expr)
    abar.x.array[:] += np.maximum(tmp.x.array - alpha_s_old.x.array, 0.0)
    alpha_s_old.x.array[:] = tmp.x.array
    alphaT.interpolate(alphaT_expr)
    aT_ = alphaT.x.array
    fdeg.x.array[:] = np.where(abar.x.array <= aT_, 1.0, (2 * aT_ / (abar.x.array + aT_)) ** 2)

    row = dict(step=istep, cycle=cyc, t_h=t / 3600, T=Tm, stag_it=it, stag_err=err,
               crack_len_um=scalar(crack_len_form) / UM, frac_energy_Jpm=scalar(frac_en_form),
               diff_energy_Jpm=scalar(diff_en_form), oxygen_molpm=scalar(oxy_form),
               tgo_peak_um=0.5 * (tgo_thickness("peak") + tgo_thickness("peak_r")),
               tgo_valley_um=tgo_thickness("valley"),
               d_max=COMM.allreduce(float(d.x.array.max()), op=MPI.MAX),
               f_min=COMM.allreduce(float(fdeg.x.array.min()), op=MPI.MIN), phase=phase)
    hist.append(row)
    end_hold = phase == "hold" and abs((t - (cyc - 1) * T_CYC) - (t_heat + t_hold)) < 1e-6
    end_cycle = abs(t - cyc * T_CYC) < 1e-6
    if args.diag:
        s1_fun.interpolate(fem.Expression(s1, pts0))
        sv = s1_fun.x.array
        tgo_c = np.unique(np.where(xi.x.array[V1.dofmap.list].mean(axis=1) > 0.5, np.arange(ncell), -1))[1:]
        print(f"   diag {phase} T={Tm:.0f}: max s1_eff TC={sv[tc_cells].max()/1e6:.0f} "
              f"BC={sv[ctags.find(2)].max()/1e6:.0f} TGO={sv[tgo_c].max()/1e6 if tgo_c.size else 0:.0f} MPa, "
              f"Yn/Y0 max={np.max(Hf.x.array):.3e}", flush=True)
    if COMM.rank == 0 and (end_cycle or it >= 5):
        print(f"[{time.time() - t_start:7.0f}s] cyc {cyc:4d} {phase:4s} T={Tm:7.1f} it={it:2d} err={err:.1e} "
              f"L={row['crack_len_um']:7.2f}um dmax={row['d_max']:.3f} fmin={row['f_min']:.3f} "
              f"TGO pk/vl={row['tgo_peak_um']:.2f}/{row['tgo_valley_um']:.2f} Ediff={row['diff_energy_Jpm']:.3e}",
              flush=True)
    if end_hold and (cyc in snap_cycles or cyc == args.cycles):
        s1_fun.interpolate(s1_dam_expr)
        snapshots[cyc] = dict(c=c.x.array.copy(), xi=xi.x.array.copy(), d=d.x.array.copy(),
                              s1=s1_fun.x.array.copy(), f=fdeg.x.array.copy())
        for fn in (c, xi, d, s1_fun, fdeg, T):
            xdmf.write_function(fn, t)
    if end_cycle and cyc in snapshots:
        s1_fun.interpolate(s1_dam_expr)
        snapshots[cyc]["s1cold"] = s1_fun.x.array.copy()
    if end_cycle and (cyc % 10 == 0 or cyc == args.cycles):
        if COMM.rank == 0:
            with open(os.path.join(args.outdir, "history.csv"), "w") as fo:
                fo.write(",".join(hist_keys) + "\n")
                for rr in hist:
                    fo.write(",".join(str(rr[k_]) for k_ in hist_keys) + "\n")
            xyz1 = V1.tabulate_dof_coordinates()
            cells1 = V1.dofmap.list
            np.savez_compressed(os.path.join(args.outdir, "snapshots.npz"), xy=xyz1[:, :2], tri=cells1,
                                cycles=np.array(sorted(snapshots)),
                                **{f"{kk}_{cc}": vv for cc, sd in snapshots.items() for kk, vv in sd.items()},
                                xi_last=xi.x.array, d_last=d.x.array, c_last=c.x.array)

xdmf.close()
if COMM.rank == 0:
    print(f"done in {time.time() - t_start:.0f} s", flush=True)
