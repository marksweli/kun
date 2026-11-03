# -*- coding: utf-8 -*-
"""
FEniCSx (dolfinx 0.11) 复现：
  Song et al., "Numerical investigation on the cracking behaviors of thermal barrier
  coating system under different thermal cycle loading waveforms"  (10_29/5.md)

两个算例：
  tri  : 三角波热循环（保温 1 s），不激活 TGO 生长，不激活 TC/TGO 界面 cohesive 损伤
  trap : 梯形波热循环（保温 2400 s），激活 TGO 生长 + TC/TGO 界面 cohesive 损伤

模型要点（单位制：μm, MPa, s, °C）
  * 2D 平面应变周期单胞，正弦界面：y = -A cos(2πx/L)，L = 21.43 μm, A = 6.88 μm
    x=0, L 处为 valley，x=L/2 为 peak（BC 凸起伸入 TC）
  * 层：基体 / BC / TGO(初始 0.7 μm，沿法向偏置) / cohesive 薄层(0.2 μm) / TC
  * 边界：左边 u_x=0（固定/对称）；底边 u_y=0；右边 u_x 全部相等（equation constraint，
    用 dolfinx_mpc 多点约束实现），右边合力自然为零
  * 本构：热弹性 + Norton 蠕变 ε̇cr = B σ^n（J2 型多轴推广，向后欧拉径向回映 + 一致切线）
  * TGO 生长：Eq.(2)/(3) 的 swelling 形式（r22=3, r11=r33=0）。默认 --growth-dir y：
    与 ABAQUS 未定义局部方向时一致，"2" 方向即全局 y 方向：Δε_yy = Δh/h；
    --growth-dir normal：沿局部界面法向 Δε_g = (Δh/h) n⊗n（对比变体）
    厚度动力学：Eq.(2) ḣ = 0.194(0.582t)^(-2/3) 积分 → h = (0.582 t)^{1/3} (μm, t 单位 h)；
    可选 --growth-law sec32 使用 3.2 节标定的 h = 0.48 t^{1/3}
  * TC/TGO 界面：cohesive 零膜刚度薄层（模拟 COH2D4），初始判据为 Eq.(4)
        D = (<σn>/σn0)^2 + (σs/σs0)^2 = 1，σn0 = σs0 = 80 MPa（35~125 的均值）
    之后线性软化，萌生后再张开 0.6 μm（文中 critical opening displacement）完全失效
"""
import argparse
import json
import os
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl
import dolfinx
import dolfinx.fem.petsc  # noqa: F401
import dolfinx_mpc
import gmsh
import ufl
from dolfinx import fem, io
from scipy.spatial import cKDTree

comm = MPI.COMM_WORLD
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 几何参数 (μm)
# ---------------------------------------------------------------------------
L_WAVE = 21.43
AMP = 6.88
H0_TGO = 0.7
T_COH = 0.2
H_TC = 300.0       # TC 顶面 y
H_BC = 100.0       # BC/基体界面在 y = -H_BC
H_SUB = 500.0      # 基体厚度
KW = 2.0 * np.pi / L_WAVE

TAG_SUB, TAG_BC, TAG_TGO, TAG_COH, TAG_TC = 1, 2, 3, 4, 5


def f_if(x):
    return -AMP * np.cos(KW * x)


def df_if(x):
    return AMP * KW * np.sin(KW * x)


def offset_curve(x, h):
    fp = df_if(x)
    nrm = np.sqrt(1.0 + fp ** 2)
    return x - h * fp / nrm, f_if(x) + h / nrm


# ---------------------------------------------------------------------------
# 材料参数（Table 1 数值在 md 中缺失，采用文中引用文献 [26] Aktaa 2005 /
# Ranjbar-Far 2010 中常用的 APS-TBC 温度相关数据）
# ---------------------------------------------------------------------------
T_TAB = np.array([20., 200., 400., 600., 800., 1000., 1100.])
MAT = {
    TAG_TC: dict(E=[48e3, 47e3, 44e3, 40e3, 34e3, 26e3, 22e3],
                 nu=[0.10, 0.10, 0.10, 0.11, 0.11, 0.12, 0.12],
                 alpha=[9.68, 9.61, 9.72, 9.82, 10.0, 10.2, 10.34]),
    TAG_TGO: dict(E=[400e3, 390e3, 380e3, 370e3, 355e3, 325e3, 320e3],
                  nu=[0.23, 0.23, 0.24, 0.24, 0.25, 0.25, 0.25],
                  alpha=[8.0, 8.2, 8.4, 8.7, 9.0, 9.3, 9.6]),
    TAG_BC: dict(E=[200e3, 190e3, 175e3, 160e3, 145e3, 120e3, 110e3],
                 nu=[0.30, 0.30, 0.31, 0.31, 0.32, 0.33, 0.33],
                 alpha=[13.6, 14.2, 14.6, 15.2, 16.1, 17.2, 17.6]),
    TAG_SUB: dict(E=[220e3, 210e3, 200e3, 190e3, 170e3, 155e3, 120e3],
                  nu=[0.31, 0.32, 0.33, 0.33, 0.34, 0.35, 0.35],
                  alpha=[12.5, 13.0, 13.6, 14.1, 14.8, 15.8, 16.4]),
}
T_ALPHA_REF = 20.0     # 割线热膨胀系数参考温度

# Norton 蠕变 (Table 2 + 正文 median 值)
B_TC_1000, B_TGO_1000 = 1.8e-9, 7.3e-9   # s^-1 MPa^-1, n = 1
Q_CERAMIC = 400e3                        # J/mol，陶瓷层蠕变 Arrhenius 温度依赖（假设）
R_GAS = 8.314
BC_CREEP_T = np.array([600., 700., 800., 850.])
BC_CREEP_LOGB = np.log10([6.54e-19, 2.20e-21, 1.84e-7, 2.15e-8])
BC_CREEP_N = np.array([4.57, 2.99, 1.55, 2.45])

# cohesive 界面
SIG0_N = SIG0_S = 80.0          # MPa
# 文中 "critical opening displacement δn0 = δs0 = 0.6 μm" 按 ABAQUS displacement 型损伤演化理解：
# 从损伤萌生到完全失效的有效位移增量 δf - δ0 = 0.6 μm（线性软化）
DELTA_FAIL = 0.6
E_PEN = 1.0e5                   # cohesive 法向罚刚度 × 层厚 (MPa)，K = E_PEN/T_COH
G_PEN = 0.4e5
E_MEMB = 1.0e2                  # 零膜刚度的数值稳定小量
D_MAX = 0.999

# TGO 生长动力学 h = (k_p t)^(1/3)，t 单位 h, h 单位 μm
#  "eq2"  : Eq.(2) 原式 ḣ = 0.194(0.582t)^(-2/3)  积分得  h = (0.582 t)^(1/3)   （默认）
#  "sec32": 由 3.2 节 9/20/36 h → 1.0/1.3/1.6 μm 标定的 h = 0.48 t^(1/3)
GROWTH_KP = {"eq2": 0.582, "sec32": 0.48 ** 3}

T_HIGH, T_LOW = 1000.0, 25.0


def interp_prop(tag, key, T):
    return np.interp(T, T_TAB, MAT[tag][key])


def eps_thermal(tag, T, T_sf):
    a = lambda TT: interp_prop(tag, "alpha", TT) * 1e-6
    return a(T) * (T - T_ALPHA_REF) - a(T_sf) * (T_sf - T_ALPHA_REF)


def creep_params(tag, T):
    """返回 (B, n)；B=0 表示无蠕变"""
    TK, TrefK = T + 273.15, 1273.15
    if tag == TAG_TC:
        return B_TC_1000 * np.exp(-Q_CERAMIC / R_GAS * (1.0 / TK - 1.0 / TrefK)), 1.0
    if tag == TAG_TGO:
        return B_TGO_1000 * np.exp(-Q_CERAMIC / R_GAS * (1.0 / TK - 1.0 / TrefK)), 1.0
    if tag == TAG_BC:
        return 10.0 ** np.interp(T, BC_CREEP_T, BC_CREEP_LOGB), float(np.interp(T, BC_CREEP_T, BC_CREEP_N))
    return 0.0, 1.0


# ---------------------------------------------------------------------------
# 网格
# ---------------------------------------------------------------------------
def build_mesh(h_min=0.1, h_near=0.3, h_max=8.0, npts=161):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tbc")
    geo = gmsh.model.geo
    xs = np.linspace(0.0, L_WAVE, npts)

    def spline(h):
        if h == 0.0:
            px, py = xs, f_if(xs)
        else:
            px, py = offset_curve(xs, h)
        px[0], px[-1] = 0.0, L_WAVE
        tags = [geo.addPoint(a, b, 0.0) for a, b in zip(px, py)]
        return geo.addSpline(tags), tags[0], tags[-1]

    c1, p1l, p1r = spline(0.0)
    c2, p2l, p2r = spline(H0_TGO)
    c3, p3l, p3r = spline(H0_TGO + T_COH)
    yb, ys = -H_BC, -H_BC - H_SUB
    q_sl, q_sr = geo.addPoint(0, ys, 0), geo.addPoint(L_WAVE, ys, 0)
    q_bl, q_br = geo.addPoint(0, yb, 0), geo.addPoint(L_WAVE, yb, 0)
    q_tl, q_tr = geo.addPoint(0, H_TC, 0), geo.addPoint(L_WAVE, H_TC, 0)
    l = geo.addLine
    bot, sr, mid, sl = l(q_sl, q_sr), l(q_sr, q_br), l(q_br, q_bl), l(q_bl, q_sl)
    br, bl = l(q_br, p1r), l(p1l, q_bl)
    tr, tl = l(p1r, p2r), l(p2l, p1l)
    cr, cl = l(p2r, p3r), l(p3l, p2l)
    tcr, top, tcl = l(p3r, q_tr), l(q_tr, q_tl), l(q_tl, p3l)
    surf = {}
    surf[TAG_SUB] = geo.addPlaneSurface([geo.addCurveLoop([bot, sr, mid, sl])])
    surf[TAG_BC] = geo.addPlaneSurface([geo.addCurveLoop([-mid, br, -c1, bl])])
    surf[TAG_TGO] = geo.addPlaneSurface([geo.addCurveLoop([c1, tr, -c2, tl])])
    surf[TAG_COH] = geo.addPlaneSurface([geo.addCurveLoop([c2, cr, -c3, cl])])
    surf[TAG_TC] = geo.addPlaneSurface([geo.addCurveLoop([c3, tcr, top, tcl])])
    geo.synchronize()
    for tag, s in surf.items():
        gmsh.model.addPhysicalGroup(2, [s], tag)
    fld = gmsh.model.mesh.field
    fd = fld.add("Distance")
    fld.setNumbers(fd, "CurvesList", [c1, c2, c3])
    fld.setNumber(fd, "Sampling", 800)
    f1 = fld.add("Threshold")
    fld.setNumber(f1, "InField", fd)
    fld.setNumber(f1, "SizeMin", h_min)
    fld.setNumber(f1, "SizeMax", h_near)
    fld.setNumber(f1, "DistMin", 0.25)
    fld.setNumber(f1, "DistMax", 2.5)
    fld.setNumber(f1, "StopAtDistMax", 1)
    f2 = fld.add("Threshold")
    fld.setNumber(f2, "InField", fd)
    fld.setNumber(f2, "SizeMin", h_near)
    fld.setNumber(f2, "SizeMax", h_max)
    fld.setNumber(f2, "DistMin", 2.5)
    fld.setNumber(f2, "DistMax", 80.0)
    fm = fld.add("Min")
    fld.setNumbers(fm, "FieldsList", [f1, f2])
    fld.setAsBackgroundMesh(fm)
    for k in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber("Mesh." + k, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    md = io.gmsh.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return md.mesh, md.cell_tags


# ---------------------------------------------------------------------------
# 本构：向量化的应力更新（逐积分点）
# 应变/应力分量顺序 [xx, yy, zz, xy]（张量分量）
# ---------------------------------------------------------------------------
I4 = np.array([1.0, 1.0, 1.0, 0.0])


def norm4(s):
    return np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + 2.0 * s[:, 3] ** 2)


def creep_update(eps_e_tr, E, nu, B, n, dt):
    """Norton 蠕变的向后欧拉径向回映。返回 σ(4), Δεcr(4), Ct(3x3 Voigt, 工程剪应变)"""
    npt = eps_e_tr.shape[0]
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    tr = eps_e_tr[:, :3].sum(axis=1)
    p = K * tr
    s_tr = 2.0 * G[:, None] * (eps_e_tr - (tr / 3.0)[:, None] * I4)
    snorm = norm4(s_tr)
    q_tr = np.sqrt(1.5) * snorm
    c = 3.0 * G * dt * B
    q = q_tr.copy()
    if np.any(c > 0):
        m = (c > 0) & (q_tr > 1e-12)
        qt, cc, nn = q_tr[m], c[m], n[m] if np.ndim(n) else np.full(m.sum(), n)
        lin = np.isclose(nn, 1.0)
        qq = np.empty_like(qt)
        qq[lin] = qt[lin] / (1.0 + cc[lin])
        nl = ~lin
        if np.any(nl):
            a, cN, nN = qt[nl], cc[nl], nn[nl]
            lo, hi = np.zeros_like(a), a.copy()
            x = np.minimum(a, (a / cN) ** (1.0 / nN))
            for _ in range(60):
                g = x + cN * x ** nN - a
                lo = np.where(g < 0, x, lo)
                hi = np.where(g > 0, x, hi)
                dg = 1.0 + cN * nN * np.maximum(x, 1e-30) ** (nN - 1.0)
                xn = x - g / dg
                bad = (xn <= lo) | (xn >= hi)
                xn = np.where(bad, 0.5 * (lo + hi), xn)
                if np.max(np.abs(xn - x) / np.maximum(a, 1e-12)) < 1e-12:
                    x = xn
                    break
                x = xn
            qq[nl] = x
        q[m] = qq
    theta = np.where(q_tr > 1e-12, q / np.maximum(q_tr, 1e-30), 1.0)
    nn_all = n if np.ndim(n) else np.full(npt, n)
    hfac = np.where(c > 0, 1.0 / (1.0 + c * nn_all * np.maximum(q, 1e-30) ** (nn_all - 1.0)), 1.0)
    hfac = np.where(q_tr > 1e-12, hfac, 1.0)
    sig = p[:, None] * I4 + theta[:, None] * s_tr
    deps_cr = ((q_tr - q) / np.maximum(3.0 * G, 1e-30))[:, None] * 1.5 * s_tr / np.maximum(q_tr, 1e-30)[:, None]
    # 一致切线
    N = s_tr / np.maximum(snorm, 1e-30)[:, None]
    Nv = np.stack([N[:, 0], N[:, 1], N[:, 3]], axis=1)
    Ct = np.zeros((npt, 3, 3))
    Ct[:, :2, :2] += K[:, None, None]
    gt = 2.0 * G * theta
    Ct[:, 0, 0] += gt * 2.0 / 3.0
    Ct[:, 1, 1] += gt * 2.0 / 3.0
    Ct[:, 0, 1] -= gt / 3.0
    Ct[:, 1, 0] -= gt / 3.0
    Ct[:, 2, 2] += G * theta
    Ct += (2.0 * G * (hfac - theta))[:, None, None] * Nv[:, :, None] * Nv[:, None, :]
    return sig, deps_cr, Ct


def cohesive_update(eps, nvec, d):
    """零膜刚度 cohesive 薄层（局部 n-s 坐标）。eps: [xx,yy,xy_tensor]"""
    nx, ny = nvec[:, 0], nvec[:, 1]
    sx, sy = ny, -nx
    npt = eps.shape[0]
    Tm = np.zeros((npt, 3, 3))  # [ε_ss, ε_nn, γ_sn] = T [ε_xx, ε_yy, γ_xy]
    Tm[:, 0] = np.stack([sx * sx, sy * sy, sx * sy], 1)
    Tm[:, 1] = np.stack([nx * nx, ny * ny, nx * ny], 1)
    Tm[:, 2] = np.stack([2 * sx * nx, 2 * sy * ny, sx * ny + sy * nx], 1)
    ev = np.stack([eps[:, 0], eps[:, 1], 2.0 * eps[:, 2]], 1)
    el = np.einsum("pij,pj->pi", Tm, ev)
    open_ = el[:, 1] > 0
    kn = E_PEN * np.where(open_, 1.0 - d, 1.0)
    ks = G_PEN * (1.0 - d)
    Cl = np.zeros((npt, 3, 3))
    Cl[:, 0, 0] = E_MEMB
    Cl[:, 1, 1] = kn
    Cl[:, 2, 2] = ks
    Ct = np.einsum("pki,pkl,plj->pij", Tm, Cl, Tm)
    sv = np.einsum("pij,pj->pi", Ct, ev)
    sig = np.stack([sv[:, 0], sv[:, 1], np.zeros(npt), sv[:, 2]], 1)
    # 未损伤的界面牵引力（用于 Eq.(4)）与分离位移
    tn_und = E_PEN * el[:, 1]
    ts_und = G_PEN * el[:, 2]
    tn = kn * el[:, 1]
    ts = ks * el[:, 2]
    dn = T_COH * el[:, 1]
    ds = T_COH * el[:, 2]
    return sig, Ct, dict(tn=tn, ts=ts, tn_und=tn_und, ts_und=ts_und, dn=dn, ds=ds)


# ---------------------------------------------------------------------------
# 载荷谱
# ---------------------------------------------------------------------------
def load_schedule(case, ncycles, hold, t_ramp=300.0, n_ramp=20, n_hold=16):
    """返回步列表: (Δt, T_end, is_hold, cycle, phase)"""
    steps = []
    for c in range(1, ncycles + 1):
        if case == "trap":
            # 保温段对数分布时间步（生长与蠕变初期速率高）
            r = np.geomspace(5.0, hold, n_hold)
            r = np.concatenate([[0.0], r])
            r[-1] = hold
            for dt in np.diff(r):
                steps.append((dt, T_HIGH, True, c, "hold"))
        else:
            steps.append((hold, T_HIGH, True, c, "hold"))
        for k in range(1, n_ramp + 1):
            steps.append((t_ramp / n_ramp, T_HIGH + (T_LOW - T_HIGH) * k / n_ramp, False, c, "cool"))
        if c < ncycles:
            for k in range(1, n_ramp + 1):
                steps.append((t_ramp / n_ramp, T_LOW + (T_HIGH - T_LOW) * k / n_ramp, False, c, "heat"))
    return steps


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def run(case, ncycles, hold, outdir, h_min, growth_law="eq2", damage=None, growth_dir="y"):
    os.makedirs(outdir, exist_ok=True)
    growth_on = case == "trap"
    damage_on = (case == "trap") if damage is None else damage
    kp = GROWTH_KP[growth_law]
    t0_wall = time.time()

    mesh, ctags = build_mesh(h_min=h_min)
    tdim = mesh.topology.dim
    ncell = mesh.topology.index_map(tdim).size_local
    cell_mat = np.zeros(ncell, dtype=np.int32)
    cell_mat[ctags.indices] = ctags.values
    print(f"[{case}] cells = {ncell}", flush=True)

    V = fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    qdeg = 2
    Qe3 = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(3,), degree=qdeg)
    Qe9 = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(9,), degree=qdeg)
    Q3, Q9 = fem.functionspace(mesh, Qe3), fem.functionspace(mesh, Qe9)
    Xq, _ = basix.make_quadrature(basix.CellType.triangle, qdeg)
    nq = Xq.shape[0]
    cells = np.arange(ncell, dtype=np.int32)
    qdofs = Q3.dofmap.list[:ncell].reshape(-1)          # (ncell*nq) -> 块自由度
    npq = ncell * nq
    mat_q = np.repeat(cell_mat, nq)

    # 积分点坐标
    xq_expr = fem.Expression(ufl.SpatialCoordinate(mesh), Xq)
    xq = xq_expr.eval(mesh, cells).reshape(npq, 2)

    # 界面法向（最近点投影到 BC/TGO 正弦曲线）
    xs = np.linspace(-0.2 * L_WAVE, 1.2 * L_WAVE, 20001)
    tree = cKDTree(np.stack([xs, f_if(xs)], 1))
    _, idx = tree.query(xq)
    xp = xs[idx]
    for _ in range(20):
        g = (xp - xq[:, 0]) + (f_if(xp) - xq[:, 1]) * df_if(xp)
        dg = 1.0 + df_if(xp) ** 2 + (f_if(xp) - xq[:, 1]) * AMP * KW ** 2 * np.cos(KW * xp)
        xp = xp - g / dg
    fp = df_if(xp)
    nrm = np.sqrt(1 + fp ** 2)
    nvec = np.stack([-fp / nrm, 1.0 / nrm], 1)
    dist_if = np.sum((xq - np.stack([xp, f_if(xp)], 1)) * nvec, axis=1)

    # 状态量
    eps_cr = np.zeros((npq, 4))
    eps_g = np.zeros((npq, 4))
    sig_n = np.zeros((npq, 4))
    coh = mat_q == TAG_COH
    d_dam = np.zeros(npq)
    D_idx = np.zeros(npq)          # Eq.(4) 历史最大值
    dm_max = np.zeros(npq)
    r_fail = np.zeros(npq)         # 牵引力丧失比例 (δ-δ0)/(δf-δ0)，=1 表示完全开裂
    delta0 = np.full(npq, np.nan)
    tgo = mat_q == TAG_TGO
    # swelling 方向：r22=3, r11=r33=0。ABAQUS 中未定义局部材料方向时 "2" 即全局 y 方向（默认 "y"）；
    # "normal" 表示沿局部界面法向 n⊗n
    if growth_dir == "normal":
        nn_g = np.stack([nvec[:, 0] ** 2, nvec[:, 1] ** 2, np.zeros(npq), nvec[:, 0] * nvec[:, 1]], 1)
    else:
        nn_g = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (npq, 1))

    # 有限元形式
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    sig_q = fem.Function(Q3)
    Ct_q = fem.Function(Q9)

    def epsv(z):
        return ufl.as_vector([z[0].dx(0), z[1].dx(1), z[0].dx(1) + z[1].dx(0)])

    Cm = ufl.as_matrix([[Ct_q[3 * i + j] for j in range(3)] for i in range(3)])
    dxq = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": qdeg, "quadrature_scheme": "default"})
    a_form = fem.form(ufl.inner(ufl.dot(Cm, epsv(w)), epsv(v)) * dxq)

    # 边界条件
    fdim = tdim - 1
    ymin = -H_BC - H_SUB
    left = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], ymin))
    bcs = [fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(0), fdim, left), V.sub(0)),
           fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(1), fdim, bottom), V.sub(1))]

    # 右边 equation constraint: u_x(右边所有节点) = u_x(主节点)
    V0, _ = V.sub(0).collapse()
    right_dofs, right0 = fem.locate_dofs_geometrical((V.sub(0), V0), lambda x: np.isclose(x[0], L_WAVE))
    bs = V.dofmap.index_map_bs
    xr = V0.tabulate_dof_coordinates()[right0]
    order = np.argsort(xr[:, 1])
    right_dofs = right_dofs[order]
    master = right_dofs[len(right_dofs) // 2]
    slaves = np.array([d for d in right_dofs if d != master], dtype=np.int32)
    imap = V.dofmap.index_map
    gmaster = imap.local_to_global(np.array([master // bs], dtype=np.int32))[0] * bs + master % bs
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_constraint(V, slaves, np.full(len(slaves), gmaster, dtype=np.int64),
                       np.ones(len(slaves), dtype=PETSc.ScalarType),
                       np.zeros(len(slaves), dtype=np.int32),
                       np.arange(len(slaves) + 1, dtype=np.int32))
    mpc.finalize()
    u = fem.Function(mpc.function_space, name="u")
    strain_expr = fem.Expression(ufl.as_vector([u[0].dx(0), u[1].dx(1), 0.5 * (u[0].dx(1) + u[1].dx(0))]), Xq)
    L_form = fem.form(ufl.inner(sig_q, epsv(v)) * dxq)

    A = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs)
    ksp = PETSc.KSP().create(comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    dvec = dolfinx.la.petsc.create_vector([(mpc.function_space.dofmap.index_map, bs)])

    print(f"[{case}] dofs = {V.dofmap.index_map.size_global * bs}, qp = {npq}", flush=True)

    # 监测点：TC 侧，界面法向偏置 h0 + t_coh + 0.3 μm
    def tc_point(x0, off=H0_TGO + T_COH + 0.3):
        px, py = offset_curve(np.array([x0]), off)
        return np.array([px[0], py[0]])
    mon_x = {"peak": 0.5 * L_WAVE, "off-peak": 0.375 * L_WAVE, "mid-flank": 0.25 * L_WAVE,
             "off-valley": 0.125 * L_WAVE, "valley": 0.0}
    tc_idx = np.where(mat_q == TAG_TC)[0]
    tc_tree = cKDTree(xq[tc_idx])
    mon_q = {k: tc_idx[tc_tree.query(tc_point(x))[1]] for k, x in mon_x.items()}
    coh_idx = np.where(coh)[0]
    coh_tree = cKDTree(xq[coh_idx])
    mon_c = {k: coh_idx[coh_tree.query(tc_point(x, H0_TGO + 0.5 * T_COH))[1]] for k, x in mon_x.items()}

    # TGO 厚度
    t_ox = H0_TGO ** 3 / kp * 3600.0   # 与初始厚度对应的等效氧化时间 (s)
    h_tgo = H0_TGO
    T_sf = T_HIGH

    hist = {k: [] for k in ["t", "T", "cycle", "h_tgo", "phase"]}
    for k in mon_q:
        for comp in ("s11", "s22", "s12"):
            hist[f"{k}:{comp}"] = []
        hist[f"{k}:D"] = []
        hist[f"{k}:d"] = []
        hist[f"{k}:r"] = []
    snapshots = {}

    def constitutive(eps3, T, dt, commit=False):
        nonlocal eps_cr, sig_n
        sig = np.zeros((npq, 4))
        Ct = np.zeros((npq, 3, 3))
        eps4 = np.stack([eps3[:, 0], eps3[:, 1], np.zeros(npq), eps3[:, 2]], 1)
        new_cr = eps_cr.copy()
        cinfo = None
        for tag in (TAG_SUB, TAG_BC, TAG_TGO, TAG_TC):
            m = mat_q == tag
            E = np.full(m.sum(), interp_prop(tag, "E", T))
            nu = np.full(m.sum(), interp_prop(tag, "nu", T))
            eth = eps_thermal(tag, T, T_sf)
            ee = eps4[m] - eth * I4 - eps_cr[m] - eps_g[m]
            B, n = creep_params(tag, T)
            s, dcr, C = creep_update(ee, E, nu, np.full(m.sum(), B), np.full(m.sum(), n), dt)
            sig[m], Ct[m] = s, C
            new_cr[m] = eps_cr[m] + dcr
        s, C, cinfo = cohesive_update(eps3[coh], nvec[coh], d_dam[coh])
        sig[coh], Ct[coh] = s, C
        if commit:
            eps_cr = new_cr
            sig_n = sig
        return sig, Ct, cinfo

    def set_q(sig, Ct):
        sa = sig_q.x.array.reshape(-1, 3)
        sa[qdofs] = sig[:, [0, 1, 3]]
        ca = Ct_q.x.array.reshape(-1, 9)
        ca[qdofs] = Ct.reshape(npq, 9)
        sig_q.x.scatter_forward()
        Ct_q.x.scatter_forward()

    def get_strain():
        return strain_expr.eval(mesh, cells).reshape(npq, 3)

    def newton(T, dt, tol=1e-7, maxit=25):
        r0 = None
        for it in range(maxit):
            sig, Ct, _ = constitutive(get_strain(), T, dt)
            set_q(sig, Ct)
            b = dolfinx_mpc.assemble_vector(L_form, mpc)
            dolfinx.fem.petsc.set_bc(b, bcs)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            rn = b.norm()
            if r0 is None:
                r0 = max(rn, 1e-12)
            if rn < tol * r0 or rn < 1e-9:
                b.destroy()
                return it
            A.zeroEntries()
            dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs, A=A)
            A.assemble()
            ksp.setOperators(A)
            b.scale(-1.0)
            ksp.solve(b, dvec)
            mpc.backsubstitution(dvec)
            u.x.array[:] += dvec.array_r[: u.x.array.size]
            u.x.scatter_forward()
            b.destroy()
        print(f"   !! Newton not converged (r/r0 = {rn / r0:.2e})", flush=True)
        return maxit

    steps = load_schedule(case, ncycles, hold)
    t = 0.0
    T = T_HIGH
    log = open(os.path.join(outdir, "log.txt"), "w")
    for istep, (dt, T_new, is_hold, cyc, phase) in enumerate(steps):
        # TGO 生长（仅高温保温段，Eq.(2)(3)）
        if growth_on and is_hold:
            t_ox_new = t_ox + dt
            h_new = (kp * t_ox_new / 3600.0) ** (1.0 / 3.0)
            deps = (h_new - h_tgo) / h_tgo
            eps_g[tgo] += deps * nn_g[tgo]
            t_ox, h_tgo = t_ox_new, h_new
        T = T_new
        t += dt
        its = newton(T, dt)
        sig, Ct, cinfo = constitutive(get_strain(), T, dt, commit=True)
        # 界面损伤（显式更新）
        # Eq.(4)：以实际界面牵引力计算，D>=1 即判定损伤萌生（之后保持 D=1）
        Dnow = (np.maximum(cinfo["tn"], 0.0) / SIG0_N) ** 2 + (cinfo["ts"] / SIG0_S) ** 2
        Dc = np.maximum(D_idx[coh], np.minimum(Dnow, 1.0))
        dm = np.sqrt(np.maximum(cinfo["dn"], 0.0) ** 2 + cinfo["ds"] ** 2)
        dmax = np.maximum(dm_max[coh], dm)
        d0 = delta0[coh]
        if damage_on:
            init = np.isnan(d0) & (Dnow >= 1.0)
            d0[init] = dm[init] / np.sqrt(np.maximum(Dnow[init], 1.0))
            ok = ~np.isnan(d0)
            dd = np.zeros_like(dmax)
            df_ = d0[ok] + DELTA_FAIL
            dd[ok] = df_ * (dmax[ok] - d0[ok]) / (np.maximum(dmax[ok], 1e-30) * DELTA_FAIL)
            rr = np.zeros_like(dmax)
            rr[ok] = np.clip((dmax[ok] - d0[ok]) / DELTA_FAIL, 0.0, 1.0)
            r_fail[coh] = np.maximum(r_fail[coh], rr)
            dd = np.clip(dd, 0.0, D_MAX)
            d_dam[coh] = np.maximum(d_dam[coh], dd)
            delta0[coh] = d0
            Dc[ok] = 1.0
        D_idx[coh] = Dc
        dm_max[coh] = dmax

        hist["t"].append(t); hist["T"].append(T); hist["cycle"].append(cyc)
        hist["h_tgo"].append(h_tgo); hist["phase"].append(phase)
        for k, iq in mon_q.items():
            hist[f"{k}:s11"].append(sig[iq, 0])
            hist[f"{k}:s22"].append(sig[iq, 1])
            hist[f"{k}:s12"].append(sig[iq, 3])
            ic = mon_c[k]
            hist[f"{k}:D"].append(D_idx[ic])
            hist[f"{k}:d"].append(d_dam[ic])
            hist[f"{k}:r"].append(r_fail[ic])
        msg = (f"[{case}] step {istep + 1}/{len(steps)} cyc {cyc} {phase:4s} t={t:9.1f}s T={T:7.1f}C "
               f"h_TGO={h_tgo:.3f}um newton={its} s22_TC[max]={sig[mat_q == TAG_TC, 1].max():8.1f} "
               f"D=1 frac={np.mean(D_idx[coh] >= 0.999):.3f} cracked(r=1) frac={np.mean(r_fail[coh] >= 0.999):.3f}")
        print(msg, flush=True)
        log.write(msg + "\n")
        last_of_cool = (phase == "cool") and (istep + 1 == len(steps) or steps[istep + 1][4] != "cool")
        if last_of_cool:
            snapshots[cyc] = dict(sig=sig.copy(), D=D_idx.copy(), d=d_dam.copy(), h_tgo=h_tgo)
            np.savez_compressed(os.path.join(outdir, f"snap_cycle{cyc:02d}.npz"), xq=xq, mat=mat_q,
                                sig=sig, D=D_idx, d=d_dam, r=r_fail, eps_cr=eps_cr, eps_g=eps_g, nvec=nvec,
                                dist_if=dist_if, xproj=xp, h_tgo=h_tgo, T=T, t=t)
    log.close()

    # 输出 ParaView 文件（最终时刻）：位移 P1 + 应力 DG1 L2 投影
    W1 = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    u1 = fem.Function(W1, name="u")
    u1.interpolate(u)
    S = fem.functionspace(mesh, ("DG", 1, (4,)))
    ps, qs = ufl.TrialFunction(S), ufl.TestFunction(S)
    Q4 = fem.functionspace(mesh, basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(4,), degree=qdeg))
    fq = fem.Function(Q4)
    arr = fq.x.array.reshape(-1, 4)
    arr[qdofs] = np.stack([sig_n[:, 0], sig_n[:, 1], sig_n[:, 3], np.zeros(npq)], 1)
    arr[qdofs[coh], 3] = D_idx[coh]
    proj = dolfinx.fem.petsc.LinearProblem(ufl.inner(ps, qs) * dxq, ufl.inner(fq, qs) * dxq,
                                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                           petsc_options_prefix="proj_")
    sh = proj.solve()
    if isinstance(sh, tuple):
        sh = sh[0]
    sh.name = "s11_s22_s12_D"
    mt = fem.functionspace(mesh, ("DG", 0))
    mf = fem.Function(mt, name="material")
    mf.x.array[:ncell] = cell_mat
    with io.XDMFFile(comm, os.path.join(outdir, "final_u_material.xdmf"), "w") as xf:
        xf.write_mesh(mesh)
        xf.write_function(u1)
        xf.write_function(mf)
    with io.VTXWriter(comm, os.path.join(outdir, "final_stress.bp"), [sh], engine="BP4") as vtx:
        vtx.write(0.0)

    np.savez_compressed(os.path.join(outdir, "history.npz"), **{k: np.array(v) for k, v in hist.items()})
    meta = dict(case=case, ncycles=ncycles, hold=hold, h_tgo_final=h_tgo, ncell=int(ncell),
                wall_time_s=time.time() - t0_wall, monitor_x={k: float(v) for k, v in mon_x.items()},
                L=L_WAVE, A=AMP, h0=H0_TGO, t_coh=T_COH, growth=growth_on, damage=damage_on,
                growth_law=growth_law, kp_um3_per_h=kp, growth_dir=growth_dir)
    with open(os.path.join(outdir, "meta.json"), "w") as fo:
        json.dump(meta, fo, indent=2, ensure_ascii=False)
    print(f"[{case}] done in {meta['wall_time_s']:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["tri", "trap"], required=True)
    ap.add_argument("--ncycles", type=int, default=10)
    ap.add_argument("--hold", type=float, default=None, help="高温保温时间 s（tri 默认 1，trap 默认 2400）")
    ap.add_argument("--hmin", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--amp", type=float, default=None, help="正弦界面振幅 μm（默认 AMP）")
    ap.add_argument("--growth-law", choices=["eq2", "sec32"], default="eq2")
    ap.add_argument("--no-damage", action="store_true", help="关闭界面 cohesive 损伤（对比用）")
    ap.add_argument("--growth-dir", choices=["y", "normal"], default="y",
                    help="TGO swelling 方向：y=全局 2 方向（ABAQUS 默认），normal=局部界面法向")
    a = ap.parse_args()
    if a.amp is not None:
        AMP = a.amp
    hold = a.hold if a.hold is not None else (1.0 if a.case == "tri" else 2400.0)
    out = a.out or os.path.join(HERE, "results", a.case)
    run(a.case, a.ncycles, hold, out, a.hmin, a.growth_law, False if a.no_damage else None, a.growth_dir)
