"""
FEniCSx 复现 (Case A, AR = 10/24, off-peak 水平预制裂纹):
  热梯度 (瞬态导热 + 对流换热, heat transfer input)  +  TGO 连续生长 (BC -> TGO 材料切换)
  + TC 烧结 (模量随时间-温度增大 + 烧结收缩)  +  Norton 蠕变松弛 (TC/TGO/BC)
  + Miehe 应力型相场损伤 (sigma_c, zeta, l_int, 粘性正则 tau)
用于 TBC 界面 off-peak 处裂纹扩展演化模拟 (参考论文 10_29/6.md, Sect. 3-4)

离散: 位移 P2 (二次三角形单元, 避免体积锁定/应力震荡), 温度 P1, 相场 P1,
      蠕变/烧结/历史场等内变量存储在 2 阶积分点 (每单元 3 点) 上.
单位: 长度 µm, 应力 MPa, 时间 s, 温度 K
运行: python tbc_phasefield_caseA.py [n_cycles]
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import sys
import json
import time

import numpy as np
from mpi4py import MPI
import ufl
import basix
import basix.ufl
from dolfinx import fem, io, geometry
from dolfinx.mesh import compute_midpoints
from dolfinx.fem.petsc import LinearProblem
import gmsh

import mesh_tbc as MT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("TBC_OUT", os.path.join(HERE, "results"))
os.makedirs(os.path.join(OUT, "snapshots"), exist_ok=True)

N_CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
comm = MPI.COMM_WORLD

# =============================================================================
# 1. 参数
# =============================================================================
T_REF = 293.15            # 无应力参考温度 / 室温
T_ROOM = 293.15
T_GAS = 1323.0            # 外部燃气温度
T_COOL = 923.0            # 冷却空气温度 (Table 5)
H_COOL = 420.0            # W/m^2K (Wang et al. 1996)
P_GAS = 3.5               # MPa (35 bar)

# 热循环 (Fig. 3, heat transfer input): t=0~100 s 室温, 之后每循环 3600 s = 3100 s 加热 + 500 s 冷却
# (第 n 循环结束于 3600 n + 100 s, 与论文 "cycle 1 at 3700 s, cycle 2 at 7300 s" 一致).
# 加热/冷却由燃气 (及冷却空气) 温度驱动的对流换热实现; 燃气温度按 Fig. 3 红线的指数形状变化:
#   加热 f = 1 - exp(-tau/TAU_UP),  冷却 f = exp(-(tau-3100)/TAU_DOWN)
T_PRE, T_HEAT, T_CYCLE = 100.0, 3100.0, 3600.0
TAU_UP, TAU_DOWN = 8.0, 40.0
HEAT_DT = [2, 3, 5, 10, 20, 40, 70, 150, 300, 500, 1000, 1000]
COOL_DT = [5, 5, 10, 10, 10, 10, 15, 15, 20, 20, 30, 50, 100, 200]
CYCLE_STEPS = [("heat", float(v)) for v in HEAT_DT] + [("cool", float(v)) for v in COOL_DT]
assert abs(sum(HEAT_DT) - T_HEAT) < 1e-9 and abs(sum(COOL_DT) - (T_CYCLE - T_HEAT)) < 1e-9

# 热物性 (Table 4, 按原表列: Substrate, BC, TGO, TC)
K_TH = {4: 0.9, 3: 15.5, 2: 15.1, 1: 80.0}            # W/mK
CP = {4: 650.0, 3: 880.0, 2: 610.0, 1: 460.0}         # J/kgK
RHO = {4: 6000.0, 3: 3970.0, 2: 7500.0, 1: 8700.0}    # kg/m^3

# 力学性能 (Table 1), E[GPa], nu, alpha[1e-6/K]
TAB = {
    4: dict(TE=[293, 493, 693, 893, 1093, 1293], E=[123, 116, 108, 100, 92, 85],
            Tn=[293, 493, 693, 893, 1093, 1293], nu=[.359, .365, .370, .374, .378, .380],
            Ta=[293, 493, 693, 893, 1093, 1293], a=[12.0, 12.1, 12.5, 13.4, 14.7, 15.8]),
    3: dict(TE=[293, 493, 693, 893, 1093, 1293], E=[152, 151, 145, 132, 109, 72],
            Tn=[293, 493, 693, 893, 1093, 1293], nu=[.319, .327, .334, .341, .346, .351],
            Ta=[293, 493, 693, 893, 1093, 1293], a=[12.3, 13.0, 13.9, 14.9, 16.2, 17.6]),
    2: dict(TE=[301, 673, 923, 1123, 1253, 1373], E=[400, 390, 380, 355, 325, 320],
            Tn=[301, 673, 923, 1123, 1253, 1373], nu=[.23, .23, .24, .25, .25, .25],
            Ta=[301, 673, 923, 1123, 1253, 1373], a=[8.2, 8.4, 8.7, 9.0, 9.3, 9.6]),
    1: dict(TE=[293, 373, 473, 573, 673, 873, 1073, 1173, 1323, 1423],
            E=[62.5, 59.6, 55.7, 50.3, 44.9, 29.8, 22.2, 21.5, 19.8, 18.6],
            Tn=[301, 473, 573, 673, 873, 923, 1073, 1253, 1323, 1423],
            nu=[.10, .10, .10, .10, .10, .11, .11, .12, .12, .12],
            Ta=[293, 493, 693, 893, 1093, 1293], a=[9.68, 9.67, 9.71, 9.81, 10.02, 10.04]),
}

# 蠕变 (Table 3): eps_dot = A sigma^n, 1123 K 以上激活
CREEP_T_ON, CREEP_T_W = 1123.0, 25.0
CREEP = {1: (9.0e-9, 1.0), 2: (7.3e-7, 1.0), 3: (1.39e-7, 3.0), 4: (0.0, 1.0)}
BC_Q = np.log(2.35e-6 / 1.39e-7) / (1 / 1273.0 - 1 / 1423.0)  # BC 两点 Arrhenius 拟合 [K]

# TGO 生长 (Eq. 4, Table 2): h = K A0 t^m exp(-Q0/R (1/T - 1/Tref)),
# 论文: "the growth rate is accelerated to 6 µm in the first 30 h at 1343 K" -> 以 1343 K 为基准的等效时间
#   dt_eq = dt exp(-Q0/(R m) (1/T - 1/T_ACC)),  h = max(1, H_TGO_END (t_eq / 30 h)^m)
A0, M_TGO, Q0, R_GAS, T_REF_TGO = 1.0, 0.332, 76.7e3, 8.314, 2424.0
H_TGO0, H_TGO_END, T_ACC, T_ACC_TIME = 1.0, 6.0, 1343.0, 30 * 3100.0

# TC 烧结 (时间-温度等效, 论文未给出具体参数, 此处为假设值):
#   E_TC = E_dep(T) [1 + (F_MAX-1) S],  eps_sint = -EPS_S_MAX S,  S = 1 - exp(-(t_eq / T_C_SINT)^N_AV)
#   (Avrami 型致密化动力学; F_MAX, T_C_SINT, N_AV 按论文 Case A 的裂纹扩展历程标定)
#   dt_eq = dt exp(-Q_S/R (1/T - 1/T_S_REF))
F_MAX = float(os.environ.get("TBC_F_MAX", 1.5))
N_AV = float(os.environ.get("TBC_N_AV", 4.0))
EPS_S_MAX = float(os.environ.get("TBC_EPS_S", 1.0e-3))
T_C_SINT = float(os.environ.get("TBC_TC_SINT_H", 1.0)) * 3600.0
Q_S, T_S_REF, T_SINT_ON = 200e3, 1373.0, 1123.0

# 相场损伤 (Table 6)
SIG_C, ZETA, L_INT, TAU_V, K_RES = 300.0, 0.2, 0.06, 1.0, 1e-6
# 裂纹只在 TC 体内扩展 (论文: 不穿透 TGO/BC, 不沿界面剥离):
# 紧贴 TC/TGO 界面厚 D_EXCL 的 TC 薄层不施加损伤驱动力
D_EXCL = float(os.environ.get("TBC_D_EXCL", 0.15))
DD_MAX, DT_MIN = 0.2, 0.25    # 单步最大相场增量 / 最小时间步 (自适应)
MAX_IT = int(os.environ.get("TBC_MAX_IT", 300))   # 交错迭代上限 (裂纹失稳扩展时需迭代至收敛)
QDEG = 2                      # 积分点阶次 (P2 位移 -> 线性应变, 3 点)

W, H_TOP = MT.W, MT.H_TOP


def h_in_forced_convection(T_air=T_GAS, p=35e5, U=488.0, L=0.2):
    """Eq. (10): 外掠平板强制对流 (COMSOL 公式), 空气物性用 Sutherland 公式"""
    mu = 1.716e-5 * (T_air / 273.15) ** 1.5 * (273.15 + 110.4) / (T_air + 110.4)
    k = 0.0241 * (T_air / 273.15) ** 1.5 * (273.15 + 194.0) / (T_air + 194.0)
    cp = 1180.0
    rho = p / (287.05 * T_air)
    Pr, Re = mu * cp / k, rho * U * L / mu
    if Re <= 5e5:
        h = 2 * k / L * 0.3387 * Pr ** (1 / 3) * Re ** 0.5 / (1 + (0.0468 / Pr) ** (2 / 3)) ** 0.25
    else:
        h = 2 * k / L * Pr ** (1 / 3) * (0.037 * Re ** 0.8 - 871.0)
    return h, Re, Pr


H_IN, RE_IN, PR_IN = h_in_forced_convection()


def gas_factor(phase, tau):
    if phase == "heat":
        return 1.0 - np.exp(-tau / TAU_UP)
    return np.exp(-(tau - T_HEAT) / TAU_DOWN)


# =============================================================================
# 2. 网格
# =============================================================================
msh_file = os.path.join(HERE, "tbc_caseA.msh")
if comm.rank == 0:
    MT.build(msh_file)
    gmsh.finalize()
comm.barrier()
gmsh.initialize()
gmsh.open(msh_file)
md = io.gmsh.model_to_mesh(gmsh.model, comm, 0, gdim=2)
gmsh.finalize()
domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim - 1, tdim)
n_cells = domain.topology.index_map(tdim).size_local + domain.topology.index_map(tdim).num_ghosts
cells_all = np.arange(n_cells, dtype=np.int32)

tag0 = np.zeros(n_cells, dtype=np.int32)
tag0[cell_tags.indices] = cell_tags.values
mid = compute_midpoints(domain, tdim, cells_all)
xc, yc = mid[:, 0], mid[:, 1]
s_below = MT.y_int(xc) - yc          # 单元形心到 TC/TGO 界面的竖直距离 (向下为正)

# DG0 (单元常数) 与积分点空间
V0 = fem.functionspace(domain, ("DG", 0))
c2d = V0.dofmap.list[:, 0]
Xq, _ = basix.make_quadrature(basix.CellType.triangle, QDEG)
NQ = Xq.shape[0]
NQP = n_cells * NQ
VQ = fem.functionspace(domain, basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(), degree=QDEG))
VQ4 = fem.functionspace(domain, basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(4,), degree=QDEG))
q2d = VQ.dofmap.list.reshape(-1)
q2d4 = VQ4.dofmap.list.reshape(-1)


def dg0(name):
    return fem.Function(V0, name=name)


def set_dg0(f, vals):
    f.x.array[c2d] = vals
    f.x.scatter_forward()


def set_q(f, vals):
    f.x.array[q2d] = vals
    f.x.scatter_forward()


def set_q4(f, vals):
    f.x.array.reshape(-1, 4)[q2d4] = vals
    f.x.scatter_forward()


def per_qp(cell_vals):
    return np.repeat(cell_vals, NQ, axis=0)


def cell_mean(q_vals):
    return q_vals.reshape(n_cells, NQ, *q_vals.shape[1:]).mean(axis=1)


dx = ufl.Measure("dx", domain=domain)
dxq = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": QDEG})
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
centroid = np.array([[1.0 / 3.0, 1.0 / 3.0]])

# =============================================================================
# 3. 热传导 (µm 单位换算: k*1, rho*cp*1e-12, h*1e-6)
# =============================================================================
VT = fem.functionspace(domain, ("Lagrange", 1))
T = fem.Function(VT, name="T")
T_n = fem.Function(VT)
T.x.array[:] = T_ROOM
T_n.x.array[:] = T_ROOM
k_c, rc_c = dg0("k"), dg0("rhocp")
dtT = fem.Constant(domain, 1.0)
Tg_c = fem.Constant(domain, T_ROOM)
Tc_c = fem.Constant(domain, T_ROOM)
hin_c = fem.Constant(domain, H_IN * 1e-6)
hco_c = fem.Constant(domain, H_COOL * 1e-6)
Tt, wt = ufl.TrialFunction(VT), ufl.TestFunction(VT)
aT = (rc_c * 1e-12 / dtT * Tt * wt * dx + k_c * ufl.dot(ufl.grad(Tt), ufl.grad(wt)) * dx
      + hin_c * Tt * wt * ds(14) + hco_c * Tt * wt * ds(13))
LT = (rc_c * 1e-12 / dtT * T_n * wt * dx + hin_c * Tg_c * wt * ds(14) + hco_c * Tc_c * wt * ds(13))
LU_OPTS = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
heat = LinearProblem(aT, LT, u=T, bcs=[], petsc_options_prefix="heat_", petsc_options=LU_OPTS)

# =============================================================================
# 4. 力学: P2 位移 (广义平面应变: eps_zz = delta/W)
# =============================================================================
VU = fem.functionspace(domain, ("Lagrange", 2, (2,)))
u = fem.Function(VU, name="u")
Vd = fem.functionspace(domain, ("Lagrange", 1))
d = fem.Function(Vd, name="phase_field")
d_n = fem.Function(Vd)
K_q, G_q, r_q, g_q = (fem.Function(VQ) for _ in range(4))
eig = fem.Function(VQ4)
ezz = fem.Constant(domain, 0.0)
delta = fem.Constant(domain, 0.0)
pres = fem.Constant(domain, 0.0)
r_q.x.array[:] = 1.0
g_q.x.array[:] = 1.0
I3 = ufl.Identity(3)


def eps3(w):
    e = ufl.sym(ufl.grad(w))
    return ufl.as_matrix([[e[0, 0], e[0, 1], 0], [e[1, 0], e[1, 1], 0], [0, 0, 0]])


def sig_eff(Et):
    tr = ufl.tr(Et)
    return K_q * tr * I3 + 2 * G_q * r_q * (Et - tr / 3 * I3)


E0 = ufl.as_matrix([[-eig[0], -eig[3], 0], [-eig[3], -eig[1], 0], [0, 0, ezz - eig[2]]])
uu, vv = ufl.TrialFunction(VU), ufl.TestFunction(VU)
aU = g_q * ufl.inner(sig_eff(eps3(uu)), eps3(vv)) * dxq
LU = -g_q * ufl.inner(sig_eff(E0), eps3(vv)) * dxq - pres * vv[1] * ds(14)

fdim = tdim - 1
dofs_l = fem.locate_dofs_topological(VU.sub(0), fdim, facet_tags.find(11))
dofs_r = fem.locate_dofs_topological(VU.sub(0), fdim, facet_tags.find(12))
dofs_b = fem.locate_dofs_topological(VU.sub(1), fdim, facet_tags.find(13))
bcs_u = [fem.dirichletbc(fem.Constant(domain, 0.0), dofs_l, VU.sub(0)),
         fem.dirichletbc(delta, dofs_r, VU.sub(0)),
         fem.dirichletbc(fem.Constant(domain, 0.0), dofs_b, VU.sub(1))]
mech = LinearProblem(aU, LU, u=u, bcs=bcs_u, petsc_options_prefix="mech_", petsc_options=LU_OPTS)

# 相场 (Miehe et al. 2015): tau d_dot = (1-d) H - (d - l^2 lap d), H 在积分点上
Hq = fem.Function(VQ)
dtd = fem.Constant(domain, 1.0)
dd, wd = ufl.TrialFunction(Vd), ufl.TestFunction(Vd)
ad = ((TAU_V / dtd + 1.0 + Hq) * dd * wd + L_INT ** 2 * ufl.dot(ufl.grad(dd), ufl.grad(wd))) * dxq
Ld = (TAU_V / dtd * d_n + Hq) * wd * dxq
pf = LinearProblem(ad, Ld, u=d, bcs=[], petsc_options_prefix="pf_", petsc_options=LU_OPTS)

# 积分点 / 形心值表达式
eu = ufl.sym(ufl.grad(u))
ex_strain = fem.Expression(ufl.as_vector([eu[0, 0], eu[1, 1], eu[0, 1]]), Xq)
ex_Tq = fem.Expression(T, Xq)
ex_dq = fem.Expression(d, Xq)
ex_Tc = fem.Expression(T, centroid)
ex_xq = fem.Expression(ufl.SpatialCoordinate(domain), Xq)
alpha_c = dg0("alpha")
# 右边界加权热膨胀 -> 规定位移 delta (Eq. 7 的连续形式)
form_delta = fem.form(alpha_c * (T - T_REF) * ds(12))


def q_eval(expr):
    v = expr.eval(domain, cells_all)
    return v.reshape(NQP, -1)


xq = q_eval(ex_xq)
cell_of_q = np.repeat(cells_all, NQ)
dist_int_q = xq[:, 1] - MT.y_int(xq[:, 0])         # 积分点到 TC/TGO 界面的竖直距离 (向上为正)

# =============================================================================
# 5. 状态变量 (积分点)
# =============================================================================
st = dict(
    eps_in=np.zeros((NQP, 4)),     # 非弹性 (蠕变 + 材料切换) 应变 xx, yy, zz, xy
    H=np.zeros(NQP),
    t_sint=np.zeros(NQP),
    t_ox=0.0,
    h_tgo=H_TGO0,
    mat=tag0.copy(),               # 材料按单元
    strain=np.zeros((NQP, 3)),
    ezz=0.0,
)
grow_candidate = (tag0 == 3) & (s_below > 0)
drive_ok_q = per_qp(tag0 == 1) & (dist_int_q >= D_EXCL)    # 允许产生损伤驱动力的积分点


def mat_props(Tv, mat, t_sint):
    N = Tv.shape[0]
    E = np.zeros(N); nu = np.zeros(N); al = np.zeros(N); A = np.zeros(N); n = np.ones(N)
    act = np.clip((Tv - CREEP_T_ON) / CREEP_T_W, 0.0, 1.0)
    for m, tb in TAB.items():
        sel = mat == m
        if not np.any(sel):
            continue
        Tm = Tv[sel]
        E[sel] = np.interp(Tm, tb["TE"], tb["E"]) * 1e3
        nu[sel] = np.interp(Tm, tb["Tn"], tb["nu"])
        al[sel] = np.interp(Tm, tb["Ta"], tb["a"]) * 1e-6
        A0m, nm = CREEP[m]
        Am = A0m * act[sel]
        if m == 3:
            Am = Am * np.exp(-BC_Q * (1.0 / np.maximum(Tm, 300.0) - 1.0 / 1273.0))
        A[sel] = Am
        n[sel] = nm
    S = 1.0 - np.exp(-(t_sint / T_C_SINT) ** N_AV)
    E = np.where(mat == 1, E * (1.0 + (F_MAX - 1.0) * S), E)
    eps_s = np.where(mat == 1, -EPS_S_MAX * S, 0.0)
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    return K, G, al, A, n, eps_s, E


def full_strain(strain, ez):
    return np.column_stack([strain[:, 0], strain[:, 1], np.full(len(strain), ez), strain[:, 2]])


def stress_from(estar, K, G, r):
    """estar: (N,4) xx,yy,zz,xy 弹性试应变 -> 有效应力 (N,4)"""
    tr = estar[:, 0] + estar[:, 1] + estar[:, 2]
    dev = estar.copy()
    dev[:, :3] -= (tr / 3.0)[:, None]
    sig = 2 * (G * r)[:, None] * dev
    sig[:, :3] += (K * tr)[:, None]
    return sig, dev


def vm_of_dev(dev, G):
    s = 2 * G[:, None] * dev
    return np.sqrt(1.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + 2 * s[:, 3] ** 2))


def creep_ratio(A, n, G, dt, svm_tr):
    """隐式 Norton: r + 3 G dt A (r svm_tr)^(n-1) r = 1"""
    r = np.ones(len(A))
    lin = (n == 1.0) & (A > 0)
    r[lin] = 1.0 / (1.0 + 3 * G[lin] * dt * A[lin])
    nl = (n > 1.0) & (A > 0)
    if np.any(nl):
        nn = n[nl]
        c = 3 * G[nl] * dt * A[nl] * svm_tr[nl] ** (nn - 1)
        x = np.ones(nl.sum())
        for _ in range(60):
            f = x + c * x ** nn - 1.0
            fp = 1.0 + c * nn * x ** (nn - 1)
            x = np.clip(x - f / fp, 1e-12, 1.0)
            if np.max(np.abs(f)) < 1e-12:
                break
        r[nl] = x
    return r


def principal2d(sig):
    a = 0.5 * (sig[:, 0] + sig[:, 1])
    b = np.sqrt((0.5 * (sig[:, 0] - sig[:, 1])) ** 2 + sig[:, 3] ** 2)
    return a + b, a - b


# 采样点 (界面上方 0.1 µm, Fig. 6)
SAMPLE_X = {"peak": 0.05, "off_peak": MT.X_OFF, "mid": 6.0, "off_valley": 10.0, "valley": 11.95}
bb = geometry.bb_tree(domain, tdim)
sample_cells = {}
for k_, x_ in SAMPLE_X.items():
    p = np.array([[x_, float(MT.y_int(x_)) + 0.1, 0.0]])
    coll = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(bb, p), p)
    sample_cells[k_] = int(coll.links(0)[0])

dcoords = Vd.tabulate_dof_coordinates()[:, :2]
tip = np.array(MT.CRACK_TIP)
x0b, x1b, y0b, y1b = MT.FINE_BOX
in_box = ((dcoords[:, 0] > x0b) & (dcoords[:, 0] < x1b) & (dcoords[:, 1] > y0b) & (dcoords[:, 1] < y1b))
not_notch_left = dcoords[:, 0] >= MT.CRACK_TIP[0] - 0.05     # 只统计裂尖前方的新扩展


def crack_extension(dv, thr):
    """裂尖 (初始方形裂尖中心) 到 d>=thr 区域最远点的距离 = 裂纹扩展长度"""
    sel = in_box & not_notch_left & (dv >= thr)
    if not np.any(sel):
        return 0.0, np.nan
    dist = np.linalg.norm(dcoords[sel] - tip, axis=1)
    i = np.argmax(dist)
    vec = dcoords[sel][i] - tip
    return float(dist[i]), float(np.degrees(np.arctan2(vec[1], vec[0])))


def branch_damage(dv):
    """裂纹口左侧 (朝波峰方向) 的最大相场值, 用于检查界面分支"""
    sel = in_box & (dcoords[:, 0] < MT.X_OFF - 0.35)
    return float(dv[sel].max()) if np.any(sel) else 0.0


# =============================================================================
# 6. 单步求解 (交错: 热 -> 生长/烧结 -> [力学 <-> 蠕变 <-> 相场])
# =============================================================================
def thermal_step(phase, tau, dt):
    f = gas_factor(phase, tau)
    Tg_c.value = T_ROOM + (T_GAS - T_ROOM) * f
    Tc_c.value = T_ROOM + (T_COOL - T_ROOM) * f
    dtT.value = dt
    T_n.x.array[:] = T.x.array
    heat.solve()
    pres.value = P_GAS * f
    return f


def update_thermal_props(mat):
    set_dg0(k_c, np.vectorize(K_TH.get)(mat).astype(float))
    set_dg0(rc_c, np.vectorize(lambda m: RHO[m] * CP[m])(mat).astype(float))


def T_interface_offpeak():
    p = np.array([[MT.X_OFF, MT.Y_C + 0.1, 0.0]])
    return float(T.eval(p, np.array([sample_cells["off_peak"]], dtype=np.int32))[0])


def solve_step(state, phase, tau, dt):
    """从 state 出发推进 dt, 返回新 state 与诊断信息 (不修改输入 state)"""
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    Tq = q_eval(ex_Tq)[:, 0]
    T_cell = ex_Tc.eval(domain, cells_all).reshape(-1)
    T_int = T_interface_offpeak()

    # --- TGO 生长 (Arrhenius 等效时间, 自然只在高温下有效) ---
    new_tgo = np.zeros(n_cells, dtype=bool)
    if phase == "heat":     # 只在加热段生长, 冷却段暂停
        s["t_ox"] += dt * np.exp(-Q0 / (R_GAS * M_TGO) * (1.0 / T_int - 1.0 / T_ACC))
    s["h_tgo"] = max(H_TGO0, H_TGO_END * (s["t_ox"] / T_ACC_TIME) ** M_TGO)
    new_tgo = grow_candidate & (s["mat"] == 3) & (s_below <= s["h_tgo"])
    if np.any(new_tgo):
        s["mat"][new_tgo] = 2
        # 新生成 TGO 在当前温度下无应力 (蠕变消除切换应力)
        nq_new = per_qp(new_tgo)
        _, _, al_new, *_ = mat_props(Tq[nq_new], np.full(nq_new.sum(), 2), s["t_sint"][nq_new])
        etot = full_strain(s["strain"][nq_new], s["ezz"])
        etot[:, :3] -= (al_new * (Tq[nq_new] - T_REF))[:, None]
        s["eps_in"][nq_new] = etot
        update_thermal_props(s["mat"])
    matq = per_qp(s["mat"])

    # --- TC 烧结 ---
    act = (Tq > T_SINT_ON) & (matq == 1)
    s["t_sint"][act] += dt * np.exp(-Q_S / R_GAS * (1.0 / Tq[act] - 1.0 / T_S_REF))

    K, G, al, A, n, eps_s, E = mat_props(Tq, matq, s["t_sint"])
    _, _, al_c, *_ = mat_props(T_cell, s["mat"], np.zeros(n_cells))
    set_q(K_q, K); set_q(G_q, G); set_dg0(alpha_c, al_c)
    th = al * (Tq - T_REF) + eps_s
    eig_v = s["eps_in"].copy()
    eig_v[:, :3] += th[:, None]
    set_q4(eig, eig_v)
    dlt = W / H_TOP * fem.assemble_scalar(form_delta)
    delta.value = dlt
    ezz.value = dlt / W
    s["ezz"] = dlt / W
    dtd.value = dt

    r = np.ones(NQP)
    d_prev = d_n.x.array.copy()
    u_prev = u.x.array.copy()
    H_old = state["H"]
    it_conv = False
    for it in range(MAX_IT):
        set_q(r_q, r)
        set_q(g_q, (1 - q_eval(ex_dq)[:, 0]) ** 2 + K_RES)
        mech.solve()
        strain = q_eval(ex_strain)
        estar = full_strain(strain, s["ezz"]) - eig_v
        _, dev = stress_from(estar, K, G, np.ones(NQP))
        svm_tr = vm_of_dev(dev, G)
        r_new = creep_ratio(A, n, G, dt, svm_tr)
        sig_eff_q, _ = stress_from(estar, K, G, r_new)
        s1, s2 = principal2d(sig_eff_q)
        Dd = ZETA * np.maximum((np.maximum(s1, 0) / SIG_C) ** 2 + (np.maximum(s2, 0) / SIG_C) ** 2 - 1.0, 0.0)
        H_new = np.where(drive_ok_q & (matq == 1), np.maximum(H_old, Dd), H_old * (matq == 1))
        set_q(Hq, H_new)
        pf.solve()
        d.x.array[:] = np.clip(d.x.array, d_n.x.array, 1.0)   # 不可逆
        if np.max(d.x.array - d_n.x.array) > DD_MAX and dt > DT_MIN:
            return None, dict(dmax_inc=float(np.max(d.x.array - d_n.x.array)))   # 步长过大, 提前中止
        err_d = np.max(np.abs(d.x.array - d_prev))
        err_u = np.linalg.norm(u.x.array - u_prev) / max(np.linalg.norm(u.x.array), 1e-12)
        err_r = np.max(np.abs(r_new - r))
        d_prev = d.x.array.copy(); u_prev = u.x.array.copy()
        r = r_new
        if it > 0 and err_d < 1e-3 and err_u < 1e-5 and err_r < 1e-3:
            it_conv = True
            break
    # 提交蠕变应变
    s["eps_in"] = s["eps_in"] + (1.0 - r)[:, None] * dev
    s["H"] = H_new
    s["strain"] = strain
    dq = q_eval(ex_dq)[:, 0]
    sig_q = sig_eff_q * ((1 - dq) ** 2 + K_RES)[:, None]
    info = dict(iters=it + 1, conv=it_conv, T_int=T_int, delta=dlt,
                sig=cell_mean(sig_q), sig_eff=cell_mean(sig_eff_q),
                s1q_dam=principal2d(sig_q)[0], matq=matq,
                d_cell=cell_mean(dq), E=cell_mean(E), n_new_tgo=int(new_tgo.sum()),
                dmax_inc=float(np.max(d.x.array - d_n.x.array)))
    return s, info


# =============================================================================
# 7. 输出
# =============================================================================
xdmf = io.XDMFFile(comm, os.path.join(OUT, "fields_caseA.xdmf"), "w")
xdmf.write_mesh(domain)
VU1 = fem.functionspace(domain, ("Lagrange", 1, (2,)))
u_out = fem.Function(VU1, name="u")
sig1_f, sig22_f, mat_f, sig1d_f = dg0("sigma1_damaged"), dg0("sigma22"), dg0("material"), dg0("sigma1_x_d")


def write_fields(tt, info, state):
    s1, _ = principal2d(info["sig"])
    set_dg0(sig1_f, s1)
    set_dg0(sig22_f, info["sig"][:, 1])
    set_dg0(sig1d_f, principal2d(info["sig_eff"])[0] * info["d_cell"])
    set_dg0(mat_f, state["mat"].astype(float))
    u_out.interpolate(u)
    for f in (T, u_out, d, sig1_f, sig22_f, sig1d_f, mat_f):
        xdmf.write_function(f, tt)


def snapshot(tag_, tt, info, state):
    s1, _ = principal2d(info["sig"])
    np.savez_compressed(os.path.join(OUT, "snapshots", f"{tag_}.npz"), t=tt,
                        d=d.x.array.astype(np.float32), T=T.x.array.astype(np.float32),
                        sig1=s1.astype(np.float32), sig22=info["sig"][:, 1].astype(np.float32),
                        sig1_eff=principal2d(info["sig_eff"])[0].astype(np.float32),
                        mat=state["mat"].astype(np.int8), h_tgo=state["h_tgo"])


np.savez_compressed(os.path.join(OUT, "mesh_data.npz"),
                    x=dcoords, cells=Vd.dofmap.list, tag0=tag0, xc=xc, yc=yc)
assert np.allclose(VT.tabulate_dof_coordinates()[:, :2], dcoords)

# =============================================================================
# 8. 稳态热分析 -> TGO 生长温度标定
# =============================================================================
update_thermal_props(st["mat"])
Tg_c.value, Tc_c.value, dtT.value = T_GAS, T_COOL, 1e8
heat.solve()
T0_cal = T_interface_offpeak()
yy = np.linspace(0, H_TOP, 4000)
line = np.column_stack([np.full_like(yy, 11.9), yy, np.zeros_like(yy)])
coll = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(bb, line), line)
cl = np.array([coll.links(i)[0] for i in range(len(yy))], dtype=np.int32)
np.savetxt(os.path.join(OUT, "steady_temperature_profile.csv"),
           np.column_stack([yy, T.eval(line, cl)[:, 0]]), delimiter=",", header="y_um,T_K", comments="")
ptop = np.array([[6.0, H_TOP - 1e-6, 0]])
T_top = float(T.eval(ptop, np.array([geometry.compute_colliding_cells(
    domain, geometry.compute_collisions_points(bb, ptop), ptop).links(0)[0]], dtype=np.int32))[0])
busso_raw = A0 * T_ACC_TIME ** M_TGO * np.exp(-Q0 / R_GAS * (1 / T_ACC - 1 / T_REF_TGO))
h30_expected = H_TGO_END * np.exp(-Q0 / R_GAS * (1 / T0_cal - 1 / T_ACC))
T.x.array[:] = T_ROOM
T_n.x.array[:] = T_ROOM

meta = dict(h_in=H_IN, Re=RE_IN, Pr=PR_IN, T_surface_steady=T_top, T_interface_steady=T0_cal,
            busso_h_30h_1343K_unaccelerated_um=float(busso_raw), TGO_acceleration_factor=float(H_TGO_END / busso_raw),
            h_tgo_30cycles_expected_um=float(h30_expected), F_MAX=F_MAX, N_AV=N_AV, EPS_S_MAX=EPS_S_MAX, T_C_SINT_h=T_C_SINT / 3600,
            n_cells=int(n_cells), n_qp=int(NQP), n_dofs_u_P2=int(VU.dofmap.index_map.size_global * 2),
            n_dofs_d=int(Vd.dofmap.index_map.size_global), crack_tip=list(MT.CRACK_TIP),
            x_offpeak=MT.X_OFF, D_EXCL=D_EXCL, sample_cells=sample_cells)
if comm.rank == 0:
    print(json.dumps(meta, indent=1), flush=True)
    json.dump(meta, open(os.path.join(OUT, "run_info.json"), "w"), indent=1)

# =============================================================================
# 9. 时间推进
# =============================================================================
hist_f = open(os.path.join(OUT, "history_caseA.csv"), "w")
cols = ["t", "cycle", "phase", "T_gas", "T_offpeak_int", "delta_um", "h_tgo_um", "n_tgo_cells", "d_max",
        "crack_ext_d09_um", "crack_ext_d05_um", "crack_angle_deg", "branch_d_left", "E_TC_int_GPa"]
for k_ in SAMPLE_X:
    cols += [f"sig1_{k_}", f"sig22_{k_}"]
cols += ["sig1_max_TC", "iters"]
hist_f.write(",".join(cols) + "\n")

t = T_PRE
t0_wall = time.time()
state = st
for cyc in range(1, N_CYCLES + 1):
    tau = 0.0
    for phase, dt_nom in CYCLE_STEPS:
        tau_end = tau + dt_nom
        dt_next = np.inf
        while tau < tau_end - 1e-9:
            dt = min(tau_end - tau, dt_next)
            if tau_end - tau - dt < 0.25 * dt:
                dt = tau_end - tau
            while True:
                T_backup = (T.x.array.copy(), T_n.x.array.copy())
                u_backup = u.x.array.copy()
                fgas = thermal_step(phase, tau + dt, dt)
                new_state, info = solve_step(state, phase, tau + dt, dt)
                if new_state is None or (info["dmax_inc"] > DD_MAX and dt > DT_MIN):
                    T.x.array[:], T_n.x.array[:] = T_backup
                    u.x.array[:] = u_backup
                    d.x.array[:] = d_n.x.array
                    update_thermal_props(state["mat"])
                    dt = max(dt / 2, DT_MIN)
                    dt_next = dt
                    continue
                break
            if np.isfinite(dt_next):
                dt_next = dt * (1.5 if info["dmax_inc"] < 0.5 * DD_MAX else 1.0)
            state = new_state
            d_n.x.array[:] = d.x.array
            tau += dt
            t += dt
            ext9, ang = crack_extension(d.x.array, 0.9)
            ext5, _ = crack_extension(d.x.array, 0.5)
            s1, _ = principal2d(info["sig"])
            sc_off = sample_cells["off_peak"]
            s1_tc = info["s1q_dam"][info["matq"] == 1]
            row = [t, cyc, phase, float(Tg_c.value), info["T_int"], info["delta"], state["h_tgo"],
                   int((state["mat"] == 2).sum()), float(d.x.array.max()), ext9, ext5, ang,
                   branch_damage(d.x.array), info["E"][sc_off] / 1e3]
            for k_, c_ in sample_cells.items():
                row += [s1[c_], info["sig"][c_, 1]]
            row += [float(s1_tc.max()), info["iters"]]
            hist_f.write(",".join(str(v) if isinstance(v, str) else f"{v:.6g}" for v in row) + "\n")
            hist_f.flush()
            if comm.rank == 0:
                print(f"cyc {cyc:2d} t={t:8.1f} {phase:4s} dt={dt:6.1f} Tgas={float(Tg_c.value):6.0f} "
                      f"Tint={info['T_int']:7.1f} h_tgo={state['h_tgo']:.3f} dmax={d.x.array.max():.3f} "
                      f"ext(d>.9)={ext9:.3f} ang={ang:5.1f} br={branch_damage(d.x.array):.2f} "
                      f"s1_off={s1[sc_off]:7.1f} it={info['iters']:2d}{'' if info['conv'] else '*'} "
                      f"wall={time.time() - t0_wall:7.0f}s", flush=True)
        if phase == "heat" and abs(tau - T_HEAT) < 1e-9:
            write_fields(t, info, state)
            snapshot(f"c{cyc:02d}_end_heating", t, info, state)
    write_fields(t, info, state)
    snapshot(f"c{cyc:02d}_end", t, info, state)

xdmf.close()
hist_f.close()
if comm.rank == 0:
    print(f"done, wall time {time.time() - t0_wall:.0f} s", flush=True)
