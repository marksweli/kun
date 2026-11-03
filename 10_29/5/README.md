# FEniCSx 复现：三角波 / 梯形波热循环下 TBC 的开裂行为（10_29/5.md）

## 1. 文件结构

```
5/
├── tbc_fenicsx.py        # 求解器（dolfinx 0.11 + dolfinx_mpc + gmsh）
├── postprocess.py        # 后处理：作图 + figures/summary.json
├── results/
│   ├── tri/                  # 三角波（保温 1 s，无 TGO 生长，无界面损伤）10 周次
│   ├── trap/                 # 梯形波（保温 2400 s，TGO 生长 + 界面损伤）10 周次  ← 最终结果
│   ├── trap_normalgrowth/    # 对比：TGO 沿局部界面法向生长
│   └── trap_sec32growth/     # 对比：法向生长 + 3.2 节标定的较慢生长律
│      每个目录：snap_cycleXX.npz（每次冷却结束时的积分点场）、history.npz（监测点时程）、
│      meta.json、log.txt、final_u_material.xdmf/h5（位移+材料）、final_stress.bp（ParaView 可读）
└── figures/
    ├── fig_stress_fields_and_crack.png  # s22 / s12 云图 + 最大主应力裂纹路径 + 界面裂纹（对应论文 Fig.5/6/10/17）
    ├── fig_interface_profiles.png       # TC 侧沿界面 s22 / s12 分布（第 1/5/10 周次）
    ├── fig_variants_profiles.png        # 各变体第 10 周次 s22 / s12 沿界面分布对比
    ├── fig_history_tri.png / fig_history_trap.png   # 监测点应力时程（论文 Fig.7/14/16）
    ├── fig_cycle_evolution.png          # 每周次冷却结束时监测点应力
    ├── fig_interface_damage_trap.png    # 界面 D(Eq.4) 与真实开裂 r 沿界面分布
    ├── fig_damage_history_trap.png      # 监测点界面损伤时程
    └── summary.json                     # 关键数值
```

## 2. 运行

```bash
# 环境（一次）
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba
/tmp/bin/micromamba create -y -p /opt/fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.11 dolfinx_mpc gmsh python-gmsh matplotlib scipy mumps

cd 10_29/5
export OMP_NUM_THREADS=1
/opt/fenicsx/bin/python tbc_fenicsx.py --case tri  --ncycles 10 --out results/tri      # ~10 min
/opt/fenicsx/bin/python tbc_fenicsx.py --case trap --ncycles 10 --out results/trap     # ~40 min
# 对比变体
/opt/fenicsx/bin/python tbc_fenicsx.py --case trap --growth-dir normal --out results/trap_normalgrowth
/opt/fenicsx/bin/python tbc_fenicsx.py --case trap --growth-dir normal --growth-law sec32 --out results/trap_sec32growth
/opt/fenicsx/bin/python postprocess.py
```

其他参数：`--hold`（保温时间，trap 默认 2400 s）、`--hmin`（界面最小网格尺寸）、`--amp`（界面振幅）、`--no-damage`。

## 3. 模型

- **几何**：平面应变周期单胞，$x\in[0,L]$，$L=21.43\,\mu$m，界面 $y=-A\cos(2\pi x/L)$，$A=6.88\,\mu$m；
  $x=0,L$ 为 valley，$x=L/2$ 为 peak。层次：基体（500 μm）/ BC（100 μm）/ TGO（初始 $h_0=0.7\,\mu$m，沿法向偏置）/
  cohesive 薄层（0.2 μm）/ TC（约 300 μm）。gmsh 网格约 3.0 万个三角形，P2 单元，约 12.2 万自由度，界面处 $h_{\min}=0.1\,\mu$m。
- **边界条件**：左边 $u_x=0$；底边 $u_y=0$；右边 $u_x$ 相等（dolfinx_mpc 多点约束，与 ABAQUS equation 等价）。
- **载荷**：无应力温度 1000 °C。每周次：保温（三角波 1 s / 梯形波 2400 s）→ 300 s 冷却到 25 °C → 300 s 升温，共 10 周次。
- **本构**：热弹性 + Norton 蠕变 $\dot\varepsilon^{cr}=B\,\sigma^n$（$J_2$ 推广，向后欧拉径向回映 + 一致切线）。
- **TGO 生长**（只在梯形波保温段）：Eq.(2) $\dot h = 0.194\,(0.582t)^{-2/3}$ 积分得 $h=(0.582\,t)^{1/3}$（μm，t 单位 h），
  10 周次后 $h:0.7\to1.62\,\mu$m。按 Eq.(3) swelling 形式施加，$r_{11}=r_{33}=0,\ r_{22}=3$：
  $$\Delta\varepsilon^{g}_{yy}=\frac{\Delta h}{h}.$$
  ABAQUS 中未定义局部材料方向时，"2" 方向即全局 $y$ 方向（默认 `--growth-dir y`）。
- **TC/TGO 界面**：cohesive 薄层（零膜刚度），初始判据 Eq.(4)
  $$D=\left(\frac{\langle\sigma_n\rangle}{\sigma_n^0}\right)^2+\left(\frac{\sigma_s}{\sigma_s^0}\right)^2=1,\qquad \sigma_n^0=\sigma_s^0=80\ \text{MPa},$$
  之后线性软化，萌生后再张开 $\delta_f-\delta_0=0.6\,\mu$m 即完全失效。输出牵引力丧失比 $r$（$r=1$ 表示真实界面裂纹）。三角波关闭损伤，D 仅作诊断量。
- **TC 内裂纹**：按最大主应力准则，从近界面 $\sigma_1$ 最大点沿垂直于 $\sigma_1$ 的方向追踪路径（近似论文 XFEM 的萌生/扩展方向）。

### 假设（md 中缺失或不明确）
1. Table 1 的数值缺失，弹性模量、泊松比、热膨胀系数使用 Aktaa (2005) / Ranjbar-Far (2010) 常用的 APS-TBC 温度相关数据。
2. TC/TGO 蠕变在 1000 °C 时取 $B=1.8\times10^{-9}$ / $7.3\times10^{-9}$（n=1），用 Arrhenius 外推到其他温度，$Q=400$ kJ/mol；BC 按 Table 2 插值；基体为弹性。
3. "critical opening displacement 0.6 μm" 解释为线性软化的失效位移增量。
4. 梯形波的升温/降温时间与三角波相同（各 300 s），两者只差保温时间。

## 4. 结果（第 10 周次冷却结束，summary.json）

| 指标 | 三角波 tri | 梯形波 trap |
|---|---|---|
| TGO 厚度 | 0.70 μm（不生长） | 1.62 μm |
| TC 近界面 s22 最大值 | 181 MPa | 758 MPa |
| s22 沿界面分布最大值位置 | $x/L=0.425$ → **off-peak** | $x/L\approx0$ → **valley / off-valley 侧** |
| 裂纹萌生 | off-peak / peak 附近（$x/L=0.395$，$\sigma_1=184$ MPa） | 谷部界面 |
| 扩展方式 | 在 TC 内沿 x 方向（与 x 轴约 10°） | **沿 TC/TGO 界面**：$x/L\in[0,0.26]$ 完全开裂（$r=1$，占界面 48%），其余部分 $r=0.5\sim1$ 软化 |

对比变体（第 10 周次，s22 沿界面最大值）：

| 变体 | 生长方向 | $h_{TGO}$ | s22 最大位置 |
|---|---|---|---|
| trap（最终） | 全局 y | 1.62 μm | valley 侧（713 MPa） |
| trap_normalgrowth | 局部法向 | 1.62 μm | peak（82 MPa），界面无真实开裂 |
| trap_sec32growth | 局部法向（较慢生长律） | 1.03 μm | peak（79 MPa），界面无真实开裂 |

### 结论
- **三角波**：复现论文结论。最大法向拉应力 s22 位于 TC/TGO 界面 off-peak 附近；微裂纹在 peak/off-peak 附近萌生，在 TC 内基本沿 x 方向扩展，不沿界面扩展。
- **梯形波**：TGO 生长 + 界面损伤使 s22 最大值从 off-peak 移到谷侧（off-valley/valley），界面从谷部到侧壁中部完全开裂，即裂纹沿界面曲线扩展，与论文趋势一致。监测点时程见 `fig_history_trap.png`。

### 与论文的差异
- 梯形波 s22 数值偏大（约 700 MPa），极大值更靠近 valley 而不是严格的 off-valley。可能原因：Table 1 材料数据和 TC/TGO 高温蠕变参数是假设值，TGO 应力松弛不足；y 方向 swelling 在 valley 处等同于法向增厚，引起的应力过大。
- 若 TGO 沿局部法向生长（变体 trap_normalgrowth），s22 最大值仍在 peak/off-peak，不出现向谷部转移的现象。这说明论文中的转移主要来自 swelling 的全局 y 方向分量，即侧壁上的横向生长。
- 本实现中，梯形波大部分界面在第一次冷却时 D 已达到 1；真实开裂由 $r$ 判断。
- 梯形波少数时间步 Newton 未达到 1e-7 容差，但最终残差 $r/r_0\le3\times10^{-4}$，已接受，详见 log.txt。
