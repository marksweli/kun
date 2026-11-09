"""
二维等温氧化 化学-力学-损伤耦合模型 (FEniCSx / dolfinx 0.9)
复现: Sun et al., "Synergistic effects of porosity and interface roughness on
oxidation failure in thermal barrier coating system" (10_29/7.md)

模块 (交错求解, 见论文 Fig. 4):
  1) 氧扩散-氧化反应 (c, n)       Eq.(1)-(11)
  2) 固体力学 (u), 生长应变 + 混合律 Eq.(12)-(17), (40)-(41)
  3) 历史变量 H = max psi_e^+        Eq.(33), (48)
  4) 相场 (phi, 仅 TC 层)            Eq.(49)

单位: SI (m, s, Pa, mol/m^3); 网格在 gmsh 中以 um 建立后缩放到 m.
"""
import argparse
import json
import os
import time

import numpy as np
import gmsh
import matplotlib.tri as mtri
import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

TC, BC, SUB = 1, 2, 3
UM = 1.0e-6
HOUR = 3600.0


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--porosity", type=float, default=0.07, help="TC 孔隙率 phi")
    p.add_argument("--amp", type=float, default=10.0, help="界面振幅 A [um]")
    p.add_argument("--hours", type=float, default=2000.0, help="氧化时间 [h]")
    p.add_argument("--gamma", type=float, default=0.0143, help="TGO 生长应变 gamma (论文未给; 标定使 7%%,A=10um 时 phi=0.4 约在 520 h)")
    p.add_argument("--omega_rt", type=float, default=5e-23, help="耦合系数 Omega/RT [1/Pa] (论文值 5e-23)")
    p.add_argument("--alpha", type=float, default=1.0, help="TC 吸附项系数 alpha [1/s] (Langmuir 高压极限)")
    p.add_argument("--hmin", type=float, default=0.5, help="界面附近网格尺寸 [um]")
    p.add_argument("--dt_max", type=float, default=5.0, help="最大时间步 [h]")
    p.add_argument("--dphi_max", type=float, default=0.05, help="单步允许的最大相场增量")
    p.add_argument("--snap_every", type=float, default=50.0, help="快照间隔 [h]")
    p.add_argument("--split", choices=["spectral", "amor"], default="spectral",
                   help="拉/压能量分解: spectral(Miehe, 仅拉伸主应变驱动) 或 amor(Eq.33 体积-偏量)")
    p.add_argument("--outdir", type=str, default="results")
    return p.parse_args()


def material_params(a):
    P = {}
    # --- 氧扩散 ---
    Dm, DK = 1.8e-4, 3.18e-3                         # Eq.(2),(3) 论文给出的数值 [m^2/s]
    P["D_TC"] = a.porosity**2 / (1.0 / Dm + 1.0 / DK)  # Eq.(1)
    P["D_BC"] = 2.82e-12                             # 1373.15 K
    P["D_TGO"] = 2.63e-13
    P["alpha"] = a.alpha                             # Eq.(5)
    P["c0"] = 2.5                                    # TC 顶面氧浓度 [mol/m^3]
    P["xi"] = 1e-4                                   # Eq.(10) [m^3/(mol s)]
    rho, M, eta = 3950.0, 0.10196, 1.5               # Al2O3 密度, 摩尔质量, O2 化学计量比
    P["kappa"] = rho / M * eta                       # Eq.(11) [mol/m^3]
    P["omega_rt"] = a.omega_rt                       # Eq.(7)
    # --- 力学 (Table 2, 1373.15 K) ---
    P["E"] = {"TC": 40e9, "TGO": 320e9, "BC": 120e9, "SUB": 160e9}
    P["nu"] = {"TC": 0.12, "TGO": 0.25, "BC": 0.32, "SUB": 0.33}
    P["gamma"] = a.gamma
    # --- 相场 ---
    P["Gc"], P["l0"], P["k_res"] = 30.0, 2e-6, 1e-6
    return P


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


# ----------------------------------------------------------------------------
# 几何与网格 (Fig. 5)
# ----------------------------------------------------------------------------
W_UM, H_SUB, H_BC, H_TC, LAMBDA = 60.0, 3000.0, 200.0, 300.0, 60.0
Y_IF = H_SUB + H_BC          # TC/BC 平均界面高度 [um]
Y_TOP = Y_IF + H_TC


def y_interface_um(x_um, A):
    # 峰 (BC 凸起) 在 x=0, 60 um; 谷 在 x=30 um
    return Y_IF + A * np.cos(2 * np.pi * x_um / LAMBDA)


def build_mesh(A, hmin, comm):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tbc")
    g = gmsh.model.geo
    hmax = 60.0
    p_b0, p_b1 = g.addPoint(0, 0, 0), g.addPoint(W_UM, 0, 0)
    p_s1, p_s0 = g.addPoint(W_UM, H_SUB, 0), g.addPoint(0, H_SUB, 0)
    xs = np.linspace(0, W_UM, 121)
    ip = [g.addPoint(x, y_interface_um(x, A), 0) for x in xs]
    p_t0, p_t1 = g.addPoint(0, Y_TOP, 0), g.addPoint(W_UM, Y_TOP, 0)

    l_bot = g.addLine(p_b0, p_b1)
    l_sr = g.addLine(p_b1, p_s1)
    l_sb = g.addLine(p_s1, p_s0)       # SUB/BC 界面 (从右到左)
    l_sl = g.addLine(p_s0, p_b0)
    l_br = g.addLine(p_s1, ip[-1])
    l_if = g.addSpline(ip[::-1])        # 界面 (从右到左)
    l_bl = g.addLine(ip[0], p_s0)
    l_tr = g.addLine(ip[-1], p_t1)
    l_top = g.addLine(p_t1, p_t0)
    l_tl = g.addLine(p_t0, ip[0])

    s_sub = g.addPlaneSurface([g.addCurveLoop([l_bot, l_sr, l_sb, l_sl])])
    s_bc = g.addPlaneSurface([g.addCurveLoop([-l_sb, l_br, l_if, l_bl])])
    s_tc = g.addPlaneSurface([g.addCurveLoop([-l_if, l_tr, l_top, l_tl])])
    g.synchronize()
    gmsh.model.addPhysicalGroup(2, [s_tc], TC)
    gmsh.model.addPhysicalGroup(2, [s_bc], BC)
    gmsh.model.addPhysicalGroup(2, [s_sub], SUB)

    f = gmsh.model.mesh.field
    fd = f.add("Distance")
    f.setNumbers(fd, "CurvesList", [l_if])
    f.setNumber(fd, "Sampling", 600)
    ft = f.add("Threshold")
    f.setNumber(ft, "InField", fd)
    f.setNumber(ft, "SizeMin", hmin)
    f.setNumber(ft, "SizeMax", hmax)
    f.setNumber(ft, "DistMin", 18.0)     # TGO 生长区与 TC 断裂区
    f.setNumber(ft, "DistMax", 400.0)
    ft2 = f.add("Threshold")             # 过渡区: 避免网格尺寸突变
    f.setNumber(ft2, "InField", fd)
    f.setNumber(ft2, "SizeMin", 2.0)
    f.setNumber(ft2, "SizeMax", hmax)
    f.setNumber(ft2, "DistMin", 30.0)
    f.setNumber(ft2, "DistMax", 600.0)
    fm = f.add("Min")
    f.setNumbers(fm, "FieldsList", [ft, ft2])
    f.setAsBackgroundMesh(fm)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    msh, ct, _ = io.gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    msh.geometry.x[:, :2] *= UM
    return msh, ct


# ----------------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------------
def main():
    a = get_args()
    P = material_params(a)
    comm = MPI.COMM_WORLD
    os.makedirs(a.outdir, exist_ok=True)
    A = a.amp
    msh, ct = build_mesh(A, a.hmin, comm)
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    print(f"[mesh] cells={ncells}, D_TC={P['D_TC']:.3e} m^2/s, kappa={P['kappa']:.4e} mol/m^3")

    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    dxv = ufl.Measure("dx", domain=msh, subdomain_data=ct,
                      metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})  # 集中质量
    # 注意: dolfinx 0.9 中同一子域内不同积分规则的核会相互覆盖, 因此扩散方程全部使用 vertex 规则
    # (P1 下扩散项被积函数为线性, vertex 规则精确; 质量/反应项即为集中质量格式)
    dxq = ufl.Measure("dx", domain=msh, subdomain_data=ct, metadata={"quadrature_degree": 2})

    # ---------------- 函数空间 ----------------
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    Wsp = fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))   # (c, n)
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    Vu = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    V0 = fem.functionspace(msh, ("DG", 0))
    VD1 = fem.functionspace(msh, ("DG", 1))
    Qe = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(), degree=2)
    Q = fem.functionspace(msh, Qe)

    def yif(x):  # 界面高度 [m], x 为 [m]
        return y_interface_um(x / UM, A) * UM

    # ---------------- 初始 TGO (法向厚度 1 um) ----------------
    xs_fine = np.linspace(0, W_UM, 6001) * UM
    curve = np.c_[xs_fine, yif(xs_fine)]
    from scipy.spatial import cKDTree
    tree = cKDTree(curve)

    def n_init(x):
        d, _ = tree.query(np.c_[x[0], x[1]])
        below = x[1] <= yif(x[0]) + 1e-12
        return np.where(below & (d <= 1.0 * UM + 1e-12), 1.0, 0.0)

    # ---------------- 扩散-氧化 ----------------
    w, w_old = fem.Function(Wsp), fem.Function(Wsp)
    c, n = ufl.split(w)
    c_old, n_old = ufl.split(w_old)
    q, pn = ufl.TestFunctions(Wsp)
    w.sub(1).interpolate(n_init)
    w.x.scatter_forward()
    w_old.x.array[:] = w.x.array
    n0 = fem.Function(V1)
    n0.interpolate(n_init)

    sig_m = fem.Function(VD1)        # 静水应力 (上一步力学结果)
    dt_c = fem.Constant(msh, PETSc.ScalarType(60.0))
    D_b = (1 - n) * P["D_BC"] + n * P["D_TGO"]
    rate = P["xi"] * (1 - n) * c                                    # Eq.(10)
    F_d = ((c - c_old) / dt_c * q * dxv
           + P["D_TC"] * ufl.inner(ufl.grad(c), ufl.grad(q)) * dxv(TC)
           + P["alpha"] * c * q * dxv(TC)                            # Eq.(5)
           + D_b * ufl.inner(ufl.grad(c) - P["omega_rt"] * c * ufl.grad(sig_m), ufl.grad(q)) * (dxv(BC) + dxv(SUB))  # Eq.(6)-(9)
           + P["kappa"] * rate * q * dxv(BC)                         # Eq.(11)
           + (n - n_old) / dt_c * pn * dxv
           - rate * pn * dxv(BC))
    Wc, _ = Wsp.sub(0).collapse()
    top = fem.locate_dofs_geometrical((Wsp.sub(0), Wc), lambda x: np.isclose(x[1], Y_TOP * UM))
    sub = fem.locate_dofs_geometrical((Wsp.sub(0), Wc), lambda x: x[1] <= H_SUB * UM * (1 + 1e-12))
    c_top, c_zero = fem.Function(Wc), fem.Function(Wc)
    c_top.x.array[:] = P["c0"]
    bcs_d = [fem.dirichletbc(c_top, top, Wsp.sub(0)), fem.dirichletbc(c_zero, sub, Wsp.sub(0))]
    prob_d = NonlinearProblem(F_d, w, bcs=bcs_d)
    newton = NewtonSolver(comm, prob_d)
    newton.convergence_criterion = "incremental"
    newton.rtol, newton.atol, newton.max_it = 1e-9, 1e-12, 30
    ksp = newton.krylov_solver
    ksp.setType("preonly"); ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")

    # ---------------- 力学 ----------------
    u = fem.Function(Vu, name="u")
    phi = fem.Function(V1, name="phase")
    n_f = fem.Function(V1, name="TGO_n")
    v, du = ufl.TestFunction(Vu), ufl.TrialFunction(Vu)
    I2 = ufl.Identity(2)
    eps = lambda z: ufl.sym(ufl.grad(z))
    lam_tc, mu_tc = lame(P["E"]["TC"], P["nu"]["TC"])
    lam_bc, mu_bc = lame(P["E"]["BC"], P["nu"]["BC"])
    lam_tg, mu_tg = lame(P["E"]["TGO"], P["nu"]["TGO"])
    lam_sb, mu_sb = lame(P["E"]["SUB"], P["nu"]["SUB"])
    gam = P["gamma"]
    gdeg = (1 - phi) ** 2 + P["k_res"]                          # Eq.(36), n=2
    lam_m = (1 - n_f) * lam_bc + n_f * lam_tg                   # Eq.(16)
    mu_m = (1 - n_f) * mu_bc + n_f * mu_tg
    eg = ufl.max_value(n_f - n0, 0.0) * gam                     # Eq.(14) 以初始 TGO 为无应力参考

    def sig_lin(e, lam, mu):  # 平面应变
        return lam * ufl.tr(e) * I2 + 2 * mu * e

    sig_bc_u = sig_lin(eps(du), lam_m, mu_m)
    sig_bc_g = (3 * lam_m + 2 * mu_m) * eg * I2                 # 生长应变产生的本征应力
    a_u = (gdeg * ufl.inner(sig_lin(eps(du), lam_tc, mu_tc), eps(v)) * dx(TC)  # Eq.(40)-(41)
           + ufl.inner(sig_bc_u, eps(v)) * dx(BC)
           + ufl.inner(sig_lin(eps(du), lam_sb, mu_sb), eps(v)) * dx(SUB))
    L_u = ufl.inner(sig_bc_g, eps(v)) * dx(BC)
    fdim = tdim - 1
    Vx, _ = Vu.sub(0).collapse()
    Vy, _ = Vu.sub(1).collapse()
    sides = lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], W_UM * UM)
    bottom = lambda x: np.isclose(x[1], 0.0)
    bcs_u = [fem.dirichletbc(fem.Function(Vx), fem.locate_dofs_geometrical((Vu.sub(0), Vx), sides), Vu.sub(0)),
             fem.dirichletbc(fem.Function(Vy), fem.locate_dofs_geometrical((Vu.sub(1), Vy), bottom), Vu.sub(1))]
    lu_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    prob_u = LinearProblem(a_u, L_u, bcs=bcs_u, u=u, petsc_options=lu_opts)

    # 应力 (未退化本构 * g 只在 TC)
    eu = eps(u)
    sig_tc = gdeg * sig_lin(eu, lam_tc, mu_tc)
    sig_bc = sig_lin(eu, lam_m, mu_m) - sig_bc_g
    sig_sb = sig_lin(eu, lam_sb, mu_sb)
    s33_bc = lam_m * ufl.tr(eu) - (3 * lam_m + 2 * mu_m) * eg
    # 静水应力 (逐单元按材料选择), 用指示函数实现
    ind = {k: fem.Function(V0) for k in (TC, BC, SUB)}
    for k, f_ in ind.items():
        f_.x.array[ct.find(k)] = 1.0
    sm_expr = (ind[TC] * (ufl.tr(sig_tc) + gdeg * lam_tc * ufl.tr(eu))
               + ind[BC] * (ufl.tr(sig_bc) + s33_bc)
               + ind[SUB] * (ufl.tr(sig_sb) + lam_sb * ufl.tr(eu))) / 3.0
    sm_e = fem.Expression(sm_expr, VD1.element.interpolation_points())
    s22_expr = ind[TC] * sig_tc[1, 1] + ind[BC] * sig_bc[1, 1] + ind[SUB] * sig_sb[1, 1]
    S22 = fem.Function(V0, name="S22")
    s22_e = fem.Expression(s22_expr, V0.element.interpolation_points())

    # ---------------- 历史变量 & 相场 ----------------
    K_tc = lam_tc + 2 * mu_tc / 3
    tr_e = ufl.tr(eu)
    if a.split == "amor":
        psi_p = 0.5 * K_tc * ufl.max_value(tr_e, 0) ** 2 + mu_tc * (ufl.inner(eu, eu) - tr_e**2 / 3)  # Eq.(33)
    else:
        rad = ufl.sqrt(((eu[0, 0] - eu[1, 1]) / 2) ** 2 + eu[0, 1] ** 2 + 1e-30)
        e1, e2 = tr_e / 2 + rad, tr_e / 2 - rad          # 平面应变主应变 (e3=0)
        psi_p = 0.5 * lam_tc * ufl.max_value(tr_e, 0) ** 2 + mu_tc * (ufl.max_value(e1, 0) ** 2 + ufl.max_value(e2, 0) ** 2)
    ipts = Q.element.interpolation_points
    ipts = ipts() if callable(ipts) else ipts
    psi_e = fem.Expression(psi_p, ipts)
    H, H_old, psi_q = fem.Function(Q), fem.Function(Q), fem.Function(Q)

    dphi, wphi = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    Gc, l0 = P["Gc"], P["l0"]
    a_p = ((2 * l0 * H / Gc + 1) * dphi * wphi + l0**2 * ufl.inner(ufl.grad(dphi), ufl.grad(wphi))) * dxq(TC)  # Eq.(49)
    a_p += fem.Constant(msh, PETSc.ScalarType(0.0)) * dphi * wphi * (dx(BC) + dx(SUB))  # 仅为补全稀疏结构
    L_p = 2 * l0 * H / Gc * wphi * dxq(TC)
    tc_cells = ct.find(TC)
    tc_dofs = np.unique(V1.dofmap.list[tc_cells].ravel())
    non_tc = np.setdiff1d(np.arange(V1.dofmap.index_map.size_local + V1.dofmap.index_map.num_ghosts), tc_dofs)
    bcs_p = [fem.dirichletbc(PETSc.ScalarType(0.0), non_tc.astype(np.int32), V1)]
    prob_p = LinearProblem(a_p, L_p, bcs=bcs_p, u=phi, petsc_options=lu_opts)

    # ---------------- 后处理工具 ----------------
    cO = fem.Function(V1, name="oxygen")

    def sync_cn():
        n_f.interpolate(w.sub(1))
        cO.interpolate(w.sub(0))

    xd1 = V1.tabulate_dof_coordinates()[:, :2] / UM
    yint_d = y_interface_um(xd1[:, 0], A)
    near_if_tc = np.where(np.isin(np.arange(len(xd1)), tc_dofs) & (xd1[:, 1] - yint_d < 3.0))[0]
    valley_dof = near_if_tc[np.argmin(np.hypot(xd1[near_if_tc, 0] - 30.0, xd1[near_if_tc, 1] - (Y_IF - A)))]
    tc_mask = np.isin(np.arange(len(xd1)), tc_dofs)
    triang_all = mtri.Triangulation(xd1[:, 0], xd1[:, 1], V1.dofmap.list)

    def tgo_thickness(xum):  # 沿竖直线 n>=0.5 的长度 [um] (峰/谷处界面法向即竖直方向)
        y0 = y_interface_um(xum, A)
        ys = np.linspace(y0 - 40.0, y0 - 1e-3, 4001)
        vals = mtri.LinearTriInterpolator(triang_all, n_f.x.array)(np.full_like(ys, xum), ys)
        return float(np.sum(np.ma.filled(vals, 0.0) >= 0.5) * (ys[1] - ys[0]))

    # 输出窗口 (界面附近) 用于绘图
    win_lo, win_hi = Y_IF - A - 30.0, Y_IF + A + 40.0
    cells_all = V1.dofmap.list
    cmid = xd1[cells_all].mean(axis=1)
    win_cells = np.where((cmid[:, 1] > win_lo) & (cmid[:, 1] < win_hi))[0]
    win_dofs, tri = np.unique(cells_all[win_cells], return_inverse=True)
    tri = tri.reshape(-1, 3)
    snaps = {"xy": xd1[win_dofs], "tri": tri, "cell_tag": ct.values[np.argsort(ct.indices)][win_cells],
             "t": [], "label": [], "n": [], "c": [], "phi": [], "S22": [], "sigm": []}
    smP1 = fem.Function(V1)

    def snapshot(t_h, label="periodic"):
        S22.interpolate(s22_e)
        smP1.interpolate(sig_m)
        snaps["t"].append(t_h)
        snaps["label"].append(label)
        snaps["n"].append(n_f.x.array[win_dofs].copy())
        snaps["c"].append(cO.x.array[win_dofs].copy())
        snaps["phi"].append(phi.x.array[win_dofs].copy())
        snaps["S22"].append(S22.x.array[win_cells].copy())
        snaps["sigm"].append(smP1.x.array[win_dofs].copy())

    xdmf = io.XDMFFile(comm, os.path.join(a.outdir, "fields.xdmf"), "w")
    xdmf.write_mesh(msh)
    u1 = fem.Function(fem.functionspace(msh, ("Lagrange", 1, (2,))), name="u")

    def write_xdmf(t_h):
        S22.interpolate(s22_e)
        u1.interpolate(u)
        for f_ in (cO, n_f, phi, S22, u1):
            xdmf.write_function(f_, t_h)

    # ---------------- 时间推进 ----------------
    def solve_mech_phase(max_it=50, tol=1e-4):
        for it in range(max_it):
            prob_u.solve()
            psi_q.interpolate(psi_e)
            H.x.array[:] = np.maximum(H_old.x.array, psi_q.x.array)     # Eq.(48)
            phi_prev = phi.x.array.copy()
            prob_p.solve()
            phi.x.array[:] = np.clip(phi.x.array, 0.0, 1.0)
            err = np.max(np.abs(phi.x.array - phi_prev))
            if err < tol:
                break
        sig_m.interpolate(sm_e)
        return it + 1

    T_end = a.hours * HOUR
    t, dt = 0.0, 60.0
    dt_max = a.dt_max * HOUR
    hist = {k: [] for k in ("t_h", "phi_valley", "phi_max", "x_phimax", "y_phimax",
                            "h_peak", "h_valley", "h_mean", "c_if_valley", "crack_len")}
    sync_cn()
    solve_mech_phase()
    next_snap = 0.0
    arc = np.sum(np.hypot(np.diff(xs_fine), np.diff(curve[:, 1]))) / UM
    # 界面上方 3 um 内 phi>=0.95 的 TC 节点在 x 方向的投影覆盖长度 (裂纹长度近似)
    xb = np.linspace(0, W_UM, 61)   # 1 um 分箱

    def record(t_h):
        hist["t_h"].append(t_h)
        hist["phi_valley"].append(float(phi.x.array[valley_dof]))
        ph = np.where(tc_mask, phi.x.array, -1)
        im = int(np.argmax(ph))
        hist["phi_max"].append(float(ph[im]))
        hist["x_phimax"].append(float(xd1[im, 0]))
        hist["y_phimax"].append(float(xd1[im, 1] - y_interface_um(xd1[im, 0], A)))
        hist["h_peak"].append(tgo_thickness(0.0))
        hist["h_valley"].append(tgo_thickness(30.0))
        nbc = fem.assemble_scalar(fem.form(n_f * dx(BC))) / UM**2
        hist["h_mean"].append(float(nbc / arc))
        hist["c_if_valley"].append(float(cO.x.array[valley_dof]))
        cr = near_if_tc[phi.x.array[near_if_tc] >= 0.95]
        covered = np.zeros(len(xb) - 1, bool)
        if len(cr):
            idx = np.clip(np.searchsorted(xb, xd1[cr, 0]) - 1, 0, len(xb) - 2)
            covered[idx] = True
        hist["crack_len"].append(float(covered.sum() * (xb[1] - xb[0])))

    events = [dict(key="phi_max", val=0.4, label="phi0.4"),
              dict(key="phi_max", val=0.95, label="initiation"),
              dict(key="crack_len", val=15.0, label="propagation"),
              dict(key="crack_len", val=40.0, label="failure")]
    for ev in events:
        ev["done"] = False
    record(0.0)
    snapshot(0.0); write_xdmf(0.0)
    next_snap = a.snap_every
    t0 = time.time()
    step = 0
    while t < T_end - 1e-6:
        dt = min(dt, T_end - t, next_snap * HOUR - t if next_snap * HOUR > t + 1 else dt)
        saved = (w.x.array.copy(), phi.x.array.copy(), H.x.array.copy(), u.x.array.copy(), sig_m.x.array.copy())
        dt_c.value = dt
        try:
            nit, conv = newton.solve(w)
        except RuntimeError:
            conv = False
        if not conv:
            w.x.array[:], phi.x.array[:], H.x.array[:], u.x.array[:], sig_m.x.array[:] = saved
            dt *= 0.5
            continue
        w.x.scatter_forward()
        sync_cn()
        nst = solve_mech_phase()
        dphi = np.max(np.abs(phi.x.array - saved[1]))
        if dphi > a.dphi_max and dt > 30.0:
            w.x.array[:], phi.x.array[:], H.x.array[:], u.x.array[:], sig_m.x.array[:] = saved
            sync_cn()
            dt *= 0.5
            continue
        # 接受
        t += dt
        step += 1
        w_old.x.array[:] = w.x.array
        H_old.x.array[:] = H.x.array
        t_h = t / HOUR
        record(t_h)
        if t_h >= next_snap - 1e-6:
            snapshot(t_h); write_xdmf(t_h)
            next_snap += a.snap_every
        for ev in events:   # 事件快照: 相场阈值 / 裂纹长度
            if not ev["done"] and hist[ev["key"]][-1] >= ev["val"]:
                ev["done"] = True
                snapshot(t_h, ev["label"])
                print(f"  -> event {ev['label']} at t={t_h:.2f} h", flush=True)
        if step % 10 == 0 or t >= T_end - 1e-6:
            print(f"[{step:5d}] t={t_h:8.2f} h dt={dt/HOUR:6.3f} h newton={nit} stag={nst} "
                  f"phi_v={hist['phi_valley'][-1]:.3f} phi_max={hist['phi_max'][-1]:.3f} "
                  f"h_pk={hist['h_peak'][-1]:.2f} h_vl={hist['h_valley'][-1]:.2f} um "
                  f"crack={hist['crack_len'][-1]:.1f} um  ({time.time()-t0:.0f}s)", flush=True)
        if dphi < 0.4 * a.dphi_max:
            dt = min(dt * 1.5, dt_max)
    xdmf.close()

    np.savez_compressed(os.path.join(a.outdir, "snapshots.npz"),
                        xy=snaps["xy"], tri=snaps["tri"], cell_tag=snaps["cell_tag"],
                        t=np.array(snaps["t"]), label=np.array(snaps["label"]), n=np.array(snaps["n"]), c=np.array(snaps["c"]),
                        phi=np.array(snaps["phi"]), S22=np.array(snaps["S22"]), sigm=np.array(snaps["sigm"]))
    np.savez(os.path.join(a.outdir, "history.npz"), **{k: np.array(v) for k, v in hist.items()})
    meta = dict(vars(a)); meta.update({"D_TC": P["D_TC"], "kappa": P["kappa"], "ncells": int(ncells),
                                       "steps": step, "wall_s": time.time() - t0})
    with open(os.path.join(a.outdir, "run_info.json"), "w") as fo:
        json.dump(meta, fo, indent=2)
    print("done", meta)


if __name__ == "__main__":
    main()
