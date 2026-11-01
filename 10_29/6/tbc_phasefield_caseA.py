"""
FEniCSx 复现 (Case A, AR = 10/24, off-peak 水平预制裂纹):
  热梯度 (瞬态导热 + 对流换热)  +  TGO 连续生长 (BC -> TGO 材料切换)
  + TC 烧结 (模量随时间-温度增大 + 烧结收缩)  +  Norton 蠕变松弛 (TC/TGO/BC)
  + Miehe 应力型相场损伤 (sigma_c, zeta, l_int, 粘性正则 tau)
用于 TBC 界面 off-peak 处裂纹扩展演化模拟 (参考论文 10_29/6.md, Sect. 3-4)

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
from dolfinx import fem, io, geometry
from dolfinx.mesh import compute_midpoints, locate_entities_boundary
from dolfinx.fem.petsc import LinearProblem
import gmsh

import mesh_tbc as MT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "snapshots"), exist_ok=True)

N_CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
comm = MPI.COMM_WORLD

# =============================================================================
# 1. 参数
# =============================================================================
T_REF = 293.15            # 无应力参考温度 / 室温
T_ROOM = 293.15
T_GAS = 1323.0            # 外部燃气温度 (参考工况)
T_COOL = 923.0            # 冷却空气温度 (Table 5)
H_COOL = 420.0            # W/m^2K (Wang et al. 1996)
P_GAS = 3.5               # MPa (35 bar)

# 热循环 (每循环 3600 s: 3100 s 加热 [含 300 s 升温] + 500 s 冷却 [300 s 降温 + 200 s 室温])
T_RAMP, T_HEAT, T_COOLDOWN, T_CYCLE = 300.0, 3100.0, 3400.0, 3600.0
CYCLE_STEPS = ([("ramp", 50.0)] * 6 +
               [("hold", dt) for dt in (50, 100, 150, 250, 450, 800, 1000)] +
               [("cool", 20.0)] * 15 +
               [("room", 100.0)] * 2)
assert abs(sum(s[1] for s in CYCLE_STEPS) - T_CYCLE) < 1e-9

# 热物性 (Table 4; 原表列顺序与密度/导热系数明显颠倒, 此处按 TC, TGO, BC, SUB 对应)
K_TH = {1: 0.9, 2: 15.5, 3: 15.1, 4: 80.0}            # W/mK
CP = {1: 650.0, 2: 880.0, 3: 610.0, 4: 460.0}         # J/kgK
RHO = {1: 6000.0, 2: 3970.0, 3: 7500.0, 4: 8700.0}    # kg/m^3

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

# TGO 生长 (Eq. 4, Table 2): h = K A0 t^m exp(-Q0/R (1/T - 1/Tref)), 按论文加速到 30 h 内 6 µm
A0, M_TGO, Q0, R_GAS, T_REF_TGO = 1.0, 0.332, 76.7e3, 8.314, 2424.0
H_TGO0, H_TGO_END = 1.0, 6.0

# TC 烧结 (时间-温度等效): E_TC = E_dep(T) * [1 + (F_MAX-1) S],  eps_sint = -EPS_S_MAX * S
# S = 1 - exp(-sqrt(t_eq / T_C_SINT)), dt_eq = dt exp(-Q_S/R (1/T - 1/T_S_REF))
F_MAX, EPS_S_MAX, Q_S, T_S_REF, T_C_SINT, T_SINT_ON = 2.0, 2.0e-3, 200e3, 1373.0, 20 * 3600.0, 1123.0

# 相场损伤 (Table 6)
SIG_C, ZETA, L_INT, TAU_V, K_RES = 300.0, 0.2, 0.06, 1.0, 1e-6
DD_MAX, DT_MIN = 0.2, 0.5     # 单步最大相场增量 / 最小时间步 (自适应)

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

V0 = fem.functionspace(domain, ("DG", 0))
V04 = fem.functionspace(domain, ("DG", 0, (4,)))
c2d = V0.dofmap.list[:, 0]
c2d4 = V04.dofmap.list[:, 0]


def dg0(name):
    f = fem.Function(V0, name=name)
    return f


def set_dg0(f, vals):
    f.x.array[c2d] = vals
    f.x.scatter_forward()


def set_dg04(f, vals):
    f.x.array.reshape(-1, 4)[c2d4] = vals
    f.x.scatter_forward()


dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
centroid = np.array([[1.0 / 3.0, 1.0 / 3.0]])

# =============================================================================
# 3. 热传导 (µm 单位换算: k*1, rho*cp*1e-12, h*1e-6)
# =============================================================================
VT = fem.functionspace(domain, ("Lagrange", 1))
T = fem.Function(VT, name="T")
T_n = fem.Function(VT)
T_hold_end = fem.Function(VT)
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
heat = LinearProblem(aT, LT, u=T, bcs=[], petsc_options_prefix="heat_",
                     petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                    "pc_factor_mat_solver_type": "mumps"})

# =============================================================================
# 4. 力学 (广义平面应变: eps_zz = delta/W, 与周期单元面内膨胀一致)
# =============================================================================
VU = fem.functionspace(domain, ("Lagrange", 1, (2,)))
u = fem.Function(VU, name="u")
Vd = fem.functionspace(domain, ("Lagrange", 1))
d = fem.Function(Vd, name="phase_field")
d_n = fem.Function(Vd)
K_c, G_c, r_c = dg0("K"), dg0("G"), dg0("r_creep")
eig = fem.Function(V04, name="eigenstrain")
ezz = fem.Constant(domain, 0.0)
delta = fem.Constant(domain, 0.0)
pres = fem.Constant(domain, 0.0)
r_c.x.array[:] = 1.0
I3 = ufl.Identity(3)


def eps3(w):
    e = ufl.sym(ufl.grad(w))
    return ufl.as_matrix([[e[0, 0], e[0, 1], 0], [e[1, 0], e[1, 1], 0], [0, 0, 0]])


def sig_eff(Et):
    tr = ufl.tr(Et)
    return K_c * tr * I3 + 2 * G_c * r_c * (Et - tr / 3 * I3)


E0 = ufl.as_matrix([[-eig[0], -eig[3], 0], [-eig[3], -eig[1], 0], [0, 0, ezz - eig[2]]])
gdeg = (1 - d) ** 2 + K_RES
uu, vv = ufl.TrialFunction(VU), ufl.TestFunction(VU)
aU = gdeg * ufl.inner(sig_eff(eps3(uu)), eps3(vv)) * dx
LU = -gdeg * ufl.inner(sig_eff(E0), eps3(vv)) * dx - pres * vv[1] * ds(14)

fdim = tdim - 1
dofs_l = fem.locate_dofs_topological(VU.sub(0), fdim, facet_tags.find(11))
dofs_r = fem.locate_dofs_topological(VU.sub(0), fdim, facet_tags.find(12))
dofs_b = fem.locate_dofs_topological(VU.sub(1), fdim, facet_tags.find(13))
bcs_u = [fem.dirichletbc(fem.Constant(domain, 0.0), dofs_l, VU.sub(0)),
         fem.dirichletbc(delta, dofs_r, VU.sub(0)),
         fem.dirichletbc(fem.Constant(domain, 0.0), dofs_b, VU.sub(1))]
mech = LinearProblem(aU, LU, u=u, bcs=bcs_u, petsc_options_prefix="mech_",
                     petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                    "pc_factor_mat_solver_type": "mumps"})

# 相场 (Miehe et al. 2015): tau d_dot = (1-d) H - (d - l^2 lap d)
Hh = dg0("H")
dtd = fem.Constant(domain, 1.0)
dd, wd = ufl.TrialFunction(Vd), ufl.TestFunction(Vd)
ad = ((TAU_V / dtd + 1.0 + Hh) * dd * wd + L_INT ** 2 * ufl.dot(ufl.grad(dd), ufl.grad(wd))) * dx
Ld = (TAU_V / dtd * d_n + Hh) * wd * dx
pf = LinearProblem(ad, Ld, u=d, bcs=[], petsc_options_prefix="pf_",
                   petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                  "pc_factor_mat_solver_type": "mumps"})

# 单元形心值表达式
ex_strain = fem.Expression(ufl.as_vector([ufl.sym(ufl.grad(u))[0, 0], ufl.sym(ufl.grad(u))[1, 1],
                                          ufl.sym(ufl.grad(u))[0, 1]]), centroid)
ex_T = fem.Expression(T, centroid)
ex_d = fem.Expression(d, centroid)
alpha_c = dg0("alpha")
# 右边界加权热膨胀 -> 规定位移 delta (Eq. 7 的连续形式)
form_delta = fem.form(alpha_c * (T - T_REF) * ds(12))


def cell_eval(expr):
    return expr.eval(domain, cells_all).reshape(n_cells, -1)


# =============================================================================
# 5. 状态变量
# =============================================================================
st = dict(
    eps_in=np.zeros((n_cells, 4)),     # 非弹性(蠕变+切换)应变 xx,yy,zz,xy
    H=np.zeros(n_cells),
    t_sint=np.zeros(n_cells),
    t_ox=0.0,
    h_tgo=H_TGO0,
    mat=tag0.copy(),
    strain=np.zeros((n_cells, 3)),
    ezz=0.0,
)
is_tc = tag0 == 1
grow_candidate = (tag0 == 3) & (s_below > 0)


def mat_props(T_cell, mat, t_sint):
    E = np.zeros(n_cells); nu = np.zeros(n_cells); al = np.zeros(n_cells)
    A = np.zeros(n_cells); n = np.ones(n_cells)
    act = np.clip((T_cell - CREEP_T_ON) / CREEP_T_W, 0.0, 1.0)
    for m, tb in TAB.items():
        sel = mat == m
        if not np.any(sel):
            continue
        Tm = T_cell[sel]
        E[sel] = np.interp(Tm, tb["TE"], tb["E"]) * 1e3
        nu[sel] = np.interp(Tm, tb["Tn"], tb["nu"])
        al[sel] = np.interp(Tm, tb["Ta"], tb["a"]) * 1e-6
        A0m, nm = CREEP[m]
        Am = A0m * act[sel]
        if m == 3:
            Am = Am * np.exp(-BC_Q * (1.0 / np.maximum(Tm, 300.0) - 1.0 / 1273.0))
        A[sel] = Am
        n[sel] = nm
    S = 1.0 - np.exp(-np.sqrt(t_sint / T_C_SINT))
    E = np.where(mat == 1, E * (1.0 + (F_MAX - 1.0) * S), E)
    eps_s = np.where(mat == 1, -EPS_S_MAX * S, 0.0)
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    return K, G, al, A, n, eps_s, E


def full_strain(strain, ez):
    return np.column_stack([strain[:, 0], strain[:, 1], np.full(n_cells, ez), strain[:, 2]])


def stress_from(estar, K, G, r):
    """estar: (n,4) xx,yy,zz,xy 弹性试应变 -> 有效应力 (n,4)"""
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
    r = np.ones(n_cells)
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


def crack_extension(dv, thr):
    sel = in_box & (dv >= thr)
    if not np.any(sel):
        return 0.0, np.nan
    dist = np.linalg.norm(dcoords[sel] - tip, axis=1)
    i = np.argmax(dist)
    vec = dcoords[sel][i] - tip
    return float(dist[i]), float(np.degrees(np.arctan2(vec[1], vec[0])))


# =============================================================================
# 6. 单步求解 (交错: 热 -> 生长/烧结 -> [力学 <-> 蠕变 <-> 相场])
# =============================================================================
def thermal_step(phase, tau, dt):
    if phase in ("ramp", "hold"):
        f = min(tau / T_RAMP, 1.0) if phase == "ramp" else 1.0
        Tg_c.value = T_ROOM + (T_GAS - T_ROOM) * f
        Tc_c.value = T_ROOM + (T_COOL - T_ROOM) * f
        dtT.value = dt
        T_n.x.array[:] = T.x.array
        heat.solve()
        pres.value = P_GAS * f
    elif phase == "cool":
        s = min((tau - T_HEAT) / (T_COOLDOWN - T_HEAT), 1.0)
        T.x.array[:] = (1 - s) * T_hold_end.x.array + s * T_ROOM       # 等温输入降温
        pres.value = P_GAS * (1 - s)
    else:
        T.x.array[:] = T_ROOM
        pres.value = 0.0
    T.x.scatter_forward()


def update_thermal_props(mat):
    set_dg0(k_c, np.vectorize(K_TH.get)(mat).astype(float))
    set_dg0(rc_c, np.vectorize(lambda m: RHO[m] * CP[m])(mat).astype(float))


def T_interface_offpeak():
    p = np.array([[MT.X_OFF, MT.Y_C + 0.1, 0.0]])
    return float(T.eval(p, np.array([sample_cells["off_peak"]], dtype=np.int32))[0])


def solve_step(state, phase, tau, dt, T0_cal, KTGO):
    """从 state 出发推进 dt, 返回新 state 与诊断信息 (不修改输入 state)"""
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    T_cell = cell_eval(ex_T)[:, 0]
    T_int = T_interface_offpeak()

    # --- TGO 生长 (加热保温阶段), Arrhenius 等效时间 ---
    new_tgo = np.zeros(n_cells, dtype=bool)
    if phase == "hold":
        fac = np.exp(-Q0 / (R_GAS * M_TGO) * (1.0 / T_int - 1.0 / T0_cal))
        s["t_ox"] += dt * fac
        s["h_tgo"] = max(H_TGO0, KTGO * s["t_ox"] ** M_TGO)
        new_tgo = grow_candidate & (s["mat"] == 3) & (s_below <= s["h_tgo"])
        if np.any(new_tgo):
            s["mat"][new_tgo] = 2
            # 新生成 TGO 在当前温度下无应力 (蠕变消除切换应力)
            _, _, al_new, *_ = mat_props(T_cell, s["mat"], s["t_sint"])
            etot = full_strain(s["strain"], s["ezz"])
            th = (al_new * (T_cell - T_REF))[new_tgo]
            s["eps_in"][new_tgo] = etot[new_tgo]
            s["eps_in"][new_tgo, :3] -= th[:, None]
            update_thermal_props(s["mat"])

    # --- TC 烧结 ---
    act = (T_cell > T_SINT_ON) & (s["mat"] == 1)
    s["t_sint"][act] += dt * np.exp(-Q_S / R_GAS * (1.0 / T_cell[act] - 1.0 / T_S_REF))

    K, G, al, A, n, eps_s, E = mat_props(T_cell, s["mat"], s["t_sint"])
    set_dg0(K_c, K); set_dg0(G_c, G); set_dg0(alpha_c, al)
    th = al * (T_cell - T_REF) + eps_s
    eig_v = s["eps_in"].copy()
    eig_v[:, :3] += th[:, None]
    set_dg04(eig, eig_v)
    dlt = W / H_TOP * fem.assemble_scalar(form_delta)
    delta.value = dlt
    ezz.value = dlt / W
    s["ezz"] = dlt / W
    dtd.value = dt

    r = np.ones(n_cells)
    d_prev = d_n.x.array.copy()
    u_prev = u.x.array.copy()
    H_old = state["H"]
    it_conv = False
    for it in range(40):
        set_dg0(r_c, r)
        mech.solve()
        strain = cell_eval(ex_strain)
        estar = full_strain(strain, s["ezz"]) - eig_v
        _, dev = stress_from(estar, K, G, np.ones(n_cells))
        svm_tr = vm_of_dev(dev, G)
        r_new = creep_ratio(A, n, G, dt, svm_tr)
        sig_eff_c, _ = stress_from(estar, K, G, r_new)
        s1, s2 = principal2d(sig_eff_c)
        Dd = ZETA * np.maximum((np.maximum(s1, 0) / SIG_C) ** 2 + (np.maximum(s2, 0) / SIG_C) ** 2 - 1.0, 0.0)
        H_new = np.where(s["mat"] == 1, np.maximum(H_old, Dd), 0.0)
        set_dg0(Hh, H_new)
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
    d_cell = cell_eval(ex_d)[:, 0]
    g_cell = (1 - d_cell) ** 2 + K_RES
    sig = sig_eff_c * g_cell[:, None]
    info = dict(iters=it + 1, conv=it_conv, T_int=T_int, delta=dlt, sig=sig, sig_eff=sig_eff_c,
                d_cell=d_cell, E=E, T_cell=T_cell, n_new_tgo=int(new_tgo.sum()),
                dmax_inc=float(np.max(d.x.array - d_n.x.array)))
    return s, info


# =============================================================================
# 7. 输出
# =============================================================================
xdmf = io.XDMFFile(comm, os.path.join(OUT, "fields_caseA.xdmf"), "w")
xdmf.write_mesh(domain)
sig1_f, sig22_f, mat_f, sig1d_f = dg0("sigma1_damaged"), dg0("sigma22"), dg0("material"), dg0("sigma1_x_d")


def write_fields(tt, info, state):
    s1, _ = principal2d(info["sig"])
    set_dg0(sig1_f, s1)
    set_dg0(sig22_f, info["sig"][:, 1])
    set_dg0(sig1d_f, principal2d(info["sig_eff"])[0] * info["d_cell"])
    set_dg0(mat_f, state["mat"].astype(float))
    for f in (T, u, d, sig1_f, sig22_f, sig1d_f, mat_f):
        xdmf.write_function(f, tt)


def snapshot(tag_, tt, info, state):
    s1, _ = principal2d(info["sig"])
    np.savez_compressed(os.path.join(OUT, "snapshots", f"{tag_}.npz"), t=tt,
                        d=d.x.array.astype(np.float32), T=T.x.array.astype(np.float32),
                        sig1=s1.astype(np.float32), sig22=info["sig"][:, 1].astype(np.float32),
                        sig1_eff=principal2d(info["sig_eff"])[0].astype(np.float32),
                        mat=state["mat"].astype(np.int8), h_tgo=state["h_tgo"])


# 网格信息 (供后处理绘图)
np.savez_compressed(os.path.join(OUT, "mesh_data.npz"),
                    x=dcoords, cells=Vd.dofmap.list, tag0=tag0, xc=xc, yc=yc)
# 确保 P1 dof 顺序与几何节点一致
assert np.allclose(VT.tabulate_dof_coordinates()[:, :2], dcoords)

# =============================================================================
# 8. 稳态热分析 -> TGO 生长温度标定
# =============================================================================
update_thermal_props(st["mat"])
Tg_c.value, Tc_c.value, dtT.value = T_GAS, T_COOL, 1e8
heat.solve()
T0_cal = T_interface_offpeak()
yy = np.linspace(0, H_TOP, 2000)
line = np.column_stack([np.full_like(yy, 11.9), yy, np.zeros_like(yy)])
coll = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(bb, line), line)
cl = np.array([coll.links(i)[0] for i in range(len(yy))], dtype=np.int32)
np.savetxt(os.path.join(OUT, "steady_temperature_profile.csv"),
           np.column_stack([yy, T.eval(line, cl)[:, 0]]), delimiter=",", header="y_um,T_K", comments="")
T_top = float(T.eval(np.array([[6.0, H_TOP, 0]]), np.array(
    [geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(bb, np.array([[6.0, H_TOP - 1e-6, 0]])),
                                      np.array([[6.0, H_TOP - 1e-6, 0]])).links(0)[0]], dtype=np.int32))[0])
t_ox_total = 30 * (T_HEAT - T_RAMP)
KTGO = H_TGO_END / t_ox_total ** M_TGO
busso_raw = A0 * t_ox_total ** M_TGO * np.exp(-Q0 / R_GAS * (1 / T0_cal - 1 / T_REF_TGO))
T.x.array[:] = T_ROOM
T_n.x.array[:] = T_ROOM

meta = dict(h_in=H_IN, Re=RE_IN, Pr=PR_IN, T_surface_steady=T_top, T_interface_steady=T0_cal,
            busso_h_30h_unaccelerated_um=float(busso_raw), TGO_acceleration_factor=float(H_TGO_END / busso_raw),
            n_cells=int(n_cells), n_dofs_u=int(VU.dofmap.index_map.size_global * 2),
            crack_tip=list(MT.CRACK_TIP), x_offpeak=MT.X_OFF, sample_cells=sample_cells)
if comm.rank == 0:
    print(json.dumps(meta, indent=1), flush=True)
    json.dump(meta, open(os.path.join(OUT, "run_info.json"), "w"), indent=1)

# =============================================================================
# 9. 时间推进
# =============================================================================
hist_f = open(os.path.join(OUT, "history_caseA.csv"), "w")
cols = ["t", "cycle", "phase", "T_offpeak_int", "delta_um", "h_tgo_um", "n_tgo_cells", "d_max",
        "crack_ext_d09_um", "crack_ext_d05_um", "crack_len_d09_um", "crack_angle_deg", "E_TC_int_GPa"]
for k_ in SAMPLE_X:
    cols += [f"sig1_{k_}", f"sig22_{k_}"]
cols += ["sig1_max_TC", "iters"]
hist_f.write(",".join(cols) + "\n")

t = 0.0
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
                thermal_step(phase, tau + dt, dt)
                new_state, info = solve_step(state, phase, tau + dt, dt, T0_cal, KTGO)
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
            if phase == "hold" and abs(tau - T_HEAT) < 1e-9:
                T_hold_end.x.array[:] = T.x.array
            ext9, ang = crack_extension(d.x.array, 0.9)
            ext5, _ = crack_extension(d.x.array, 0.5)
            s1, _ = principal2d(info["sig"])
            sc_off = sample_cells["off_peak"]
            row = [t, cyc, phase, info["T_int"], info["delta"], state["h_tgo"], int((state["mat"] == 2).sum()),
                   float(d.x.array.max()), ext9, ext5, MT.CRACK_L + ext9, ang, info["E"][sc_off] / 1e3]
            for k_, c_ in sample_cells.items():
                row += [s1[c_], info["sig"][c_, 1]]
            row += [float(s1[is_tc].max()), info["iters"]]
            hist_f.write(",".join(str(v) if isinstance(v, str) else f"{v:.6g}" for v in row) + "\n")
            hist_f.flush()
            if comm.rank == 0:
                print(f"cyc {cyc:2d} t={t:8.1f} {phase:5s} dt={dt:6.1f} Tint={info['T_int']:7.1f} "
                      f"h_tgo={state['h_tgo']:.3f} dmax={d.x.array.max():.3f} L(d>.9)={MT.CRACK_L + ext9:.3f} "
                      f"s1_off={s1[sc_off]:7.1f} it={info['iters']:2d}{'' if info['conv'] else '*'} "
                      f"wall={time.time() - t0_wall:7.0f}s", flush=True)
        if phase == "hold" and abs(tau - T_HEAT) < 1e-9:
            write_fields(t, info, state)
            snapshot(f"c{cyc:02d}_end_heating", t, info, state)
        if phase == "cool" and abs(tau - T_COOLDOWN) < 1e-9:
            snapshot(f"c{cyc:02d}_cooled", t, info, state)
    write_fields(t, info, state)
    snapshot(f"c{cyc:02d}_end", t, info, state)

xdmf.close()
hist_f.close()
if comm.rank == 0:
    print(f"done, wall time {time.time() - t0_wall:.0f} s", flush=True)
