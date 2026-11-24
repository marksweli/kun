# 论文 9（Min et al., CMAME 411 (2023) 116044）6 个算例的 FEniCSx 复现

论文：*A chemo-thermo-mechanical coupled phase-field cohesive zone model for thermal barrier coatings*（见 `../9.pdf` / `../9.md`）。

| 文件夹 | 算例 | 主脚本 |
|---|---|---|
| `01_tpb_notched_beam` | 带缺口混凝土梁三点弯曲（仅 $h=0.2$ mm, $r=l/l_\phi=0.25$） | `tpb.py` |
| `02_L_panel` | L 型板断裂 | `lpanel.py` |
| `03_quenching` | 陶瓷淬火试验 | `quench.py` |
| `04_BC_TGO_delamination` | 5.3 节 BC/TGO 界面分层（`--damage 1/0` 损伤开关） | `bc_tgo.py` |
| `05_TC_crack` | 5.4 节 TC 裂纹（一周期、初始 TGO≈0、冷却 30 min、氧扩散/氧化） | `tc_crack.py` |
| `06_TC_TGO_delamination` | 5.5 节 TC/TGO 界面分层（800 h 等温氧化 + 30 min 冷却） | `tc_tgo.py` |

公共模块：`pfczm.py`（PF-CZM 求解器）和 `tbc_common.py`（TBC 多层模型，供算例 4–6 使用）。每个文件夹中都放有一份拷贝，因此可以在各文件夹内独立运行。每个算例的结果（图片、`history.csv`、`summary.json`、`run.log`、XDMF/HDF5 场文件）都保存在该文件夹的 `results/` 下。

## 环境与运行

```bash
# 安装 micromamba（从 github.com/mamba-org/micromamba-releases 下载）后：
micromamba create -p /opt/fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.9 mpich matplotlib scipy
export PATH=/opt/fenicsx/bin:$PATH HDF5_USE_FILE_LOCKING=FALSE
cd 01_tpb_notched_beam && python -u tpb.py > results/run.log      # 各算例同理
python bc_tgo.py --damage 0        # 算例 4：关闭损伤（纯弹性）
python bc_tgo.py --damage 1        # 算例 4：考虑损伤（默认）
# 所有脚本都支持 --quick（粗网格、少步数的快速测试）
```

## 模型与数值实现

**相场内聚区模型（PF-CZM，Wu 型）**

- 几何函数 $\alpha(d)=2d-d^2$，$c_0=\pi$。
- 能量退化函数：

$$
\omega(d)=\frac{(1-d)^p}{(1-d)^p+a_1 d\,(1+a_2 d+a_2a_3 d^2)},\qquad a_1=\frac{4EG_c}{\pi l f_t^2}
$$

- 软化律：Cornelissen 指数软化（$p=2$, $a_2=1.3868$, $a_3=0.6567$）。
- 相场方程：

$$
\omega'(d)\,\mathcal H+\frac{G_c}{c_0 l}\Big(\alpha'(d)-2l^2\Delta d\Big)=0,\qquad
\mathcal H=\max_{\tau\le t}\Big\{\max\big(\psi_0^+,\ \tfrac{f_t^2}{2E}\big)\Big\}
$$

- 拉压分解采用 hybrid 格式。
- 历史变量为 DG0 场。对于温度相关的 $E(T)$，阈值 $\tfrac{f_t^2}{2E(T)}$ 按当前温度更新。
- 界面（TC/TGO、TGO/BC、混凝土梁的界面）采用论文中的光滑化处理：用宽度 $l_\phi$ 的界面相场 $\phi$ 插值，

$$
\bar G_c = G_c^{i}\,\frac{l}{l_\phi}\ \ (\text{界面附近})
$$

  界面参数取自论文 Table 2。

**热–化–力耦合（算例 4–6）**

- 热应变以 1150 °C 为无应力参考温度。
- 材料参数按 Table 1 线性插值，超出表格范围时线性外推。
- 平面问题采用**广义平面应变**：$\varepsilon_{zz}=\varepsilon_{th}^{SUB}(T)$，即厚度方向随基体自由收缩。
- 右边界施加"保持直线"约束，用叠加法实现：

$$
u=u_0+a\,u_1,\qquad a=-R_0/R_1
$$

- 氧扩散–反应（Table 3 参数）：

$$
\dot c=\nabla\!\cdot(D\nabla c)-\frac{\partial n}{\partial t}M,\qquad
\dot n=g\,c\,(1-n)
$$

  - 仅在 $T>850$ °C 时进行。
  - TC 中取 $c=\bar c=1.55$。
  - TGO 体积膨胀应变为 $\varepsilon_g=\alpha_n\,n\,\mathbf I$（$\alpha_n=8\times10^{-3}$）。
  - 材料性质按氧化程度 $n$ 在 BC 与 TGO 之间混合。

**与论文的主要差异（为在 2 核 / 7 GB 机器上完成计算）**

1. **求解方式**：论文用显式动力学，这里改为隐式准静态交错求解（staggered）。算例 5 例外，已按论文的显式算法实现，见该算例说明。
   - 相场子问题按 convex-split 线性化，用 PETSc SNES-VI（`vinewtonrsls`）保证 $d_{n}\le d\le1$。
   - 每步最多交错迭代 200 次（三点弯、L 型板）或 100 次（淬火、TBC 算例），容差 $10^{-4}$；未收敛的步继续推进，并在 `run.log` 中记录。
   - 淬火算例的热方程用集中质量（顶点积分），以避免温度过冲。
2. **TBC 界面形状**：取 $y=A\cos(2\pi x/L)$（论文为 sin，二者只相差相位）。这样左右边界恰为对称线/周期线，左边界施加 $u_x=0$。
3. **基体（SUB）**：不参与断裂，取很大的 $f_t$，同时取 $G_c=10^3$，使 $a_1\ge3$，保证 convex-split 适用。
4. **网格**：最小网格尺寸见各算例。淬火算例为 $h=0.05$ mm（论文 0.02 mm）。
5. **材料数据**：论文未给出完整数据时，采用文献常用值，详见各算例说明。

## 各算例说明与结果

（见下节；更详细的数值输出见各文件夹 `results/run.log` 和 `summary.json`。）

### 1. 带缺口混凝土梁三点弯曲（`01_tpb_notched_beam`）

**设置**

- 几何按 Fig. 4：梁 $450\times100$ mm，缺口 $5\times50$ mm，跨中加载，平面应力。
- 网格：加密区 $h=0.2$ mm；$l=2.5$ mm，$r=l/l_\phi=0.25$（$l_\phi=10$ mm）。
- 界面：缺口尖端正上方的竖直线段。长度在 Fig. 4 上按像素量得约 **30 mm**（$y=50\to80$ mm）。
- $G_c^b=10G_c^i$。界面断裂能按 Eq. (10) 做耗散等效补偿：$\bar G_c=0.0960$ N/mm。
- 加载：位移控制至 1 mm，峰前 $\Delta u=0.002$ mm，峰后 $\Delta u=0.01$ mm。

**假设**：论文没有给出材料参数和厚度，取 Wu 带缺口混凝土梁的常用数据：

$$E=20\ \text{GPa},\quad \nu=0.2,\quad f_t=2.4\ \text{MPa},\quad G_c^i=0.113\ \text{N/mm},\quad B=100\ \text{mm}$$

**结果**（`results/`）

- 本计算（$B=100$ mm）：峰值荷载 1.47 kN（$u=0.11$ mm）；$u=1$ mm 时断裂能 0.488 J。
- 论文 Fig. 5c/d：峰值约 0.2 kN；$u=1$ mm 时断裂能约 0.07 J。
- 平面应力下荷载和能量都与 $B$ 成正比。用 $B_{eq}=13.6$ mm 使峰值等于 0.2 kN 后：
  - $u=1$ mm 时断裂能为 0.066 J，与论文的 0.07 J 吻合，这是独立校核。
  - 软化段与论文基本重合（见 `fig5_compare_scaled.png`；论文数据点为人工读图的近似值）。
- 主要差别：本计算峰值位移约 0.11 mm，论文约 0.04 mm。原因应是论文实际使用的 $E$、$f_t$ 与这里的假设不同。
- 裂纹从缺口尖端沿界面扩展到 $y\approx80$ mm，之后进入断裂能高 10 倍的基体，扩展明显减慢（`damage_zoom.png`）。
- 运行时间约 64 min。
- 其他输出：`fig5_load_energy.png`（荷载–位移和断裂能–位移）、`load_displacement.csv`。

### 2. L 型板（`02_L_panel`）

**设置**

- 几何按 Fig. 6；材料按 Table 4：$E=25.28$ GPa，$\nu=0.18$，$G_c=0.09$ N/mm，$f_t=2.7$ MPa，$l=2.5$ mm。
- 平面应力，$B=100$ mm。临界区 $h=1$ mm。
- 底边固定；在距右端 30 mm 的下表面点施加向上位移，至 0.8 mm（160 步）。

**结果**

- 峰值荷载 **7.42 kN**（$u=0.17$ mm）。论文本方法约 6.8 kN，实验约 7.6 kN，吻合良好。
- 裂纹从内角出发，向左上方弯曲后水平扩展，与 Fig. 7a 的实验裂纹区一致。
- 输出：`results/fig7_crack_and_load.png`、`load_displacement.csv`；运行时间约 67 min。

### 3. 陶瓷淬火（`03_quenching`）

**设置**

- 材料按 Table 5；$l=0.08$ mm，$\eta=10^{-8}$ kN·s/mm²。
- 板 $50\times9.8$ mm，$T_0=680$ K，四周边界保持 300 K，计算至 $t=0.01$ s。
- 热–力耦合：

$$\rho C\dot T=\kappa\Delta T,\qquad \boldsymbol\varepsilon_{th}=\alpha_\theta(T-T_0)\mathbf I$$

  相场方程中加入粘性项 $\eta\dot d$。

**假设**

- $f_t=260$ MPa。论文未给出，按 $\sqrt{3EG_c/(8l)}$ 估计。
- 利用双对称性只算 1/4 板，作图时镜像。
- 平面应力。
- $h=0.05$ mm（论文 0.02 mm），100 个时间步。可用 `--h 0.02 --n 500` 加密。
- 比热项采用集中质量。

**结果**

- 四周边界上形成大量平行、近似等间距的表面裂纹，且长短裂纹交替出现，与论文 Fig. 8/9 及实验定性一致（`fig9_final_pattern.png`、`fig8_damage_temperature.png`）。
- 运行时间约 1.9 h。

### 4. 5.3 节 BC/TGO 界面分层（`04_BC_TGO_delamination`）

**设置**

- 半周期模型：$L=60$ µm，起伏幅值 $A=20$ µm，TGO 厚 5 µm（$n=1$），不考虑氧化。
- 冷却：1150 °C 线性降至 20 °C，用时 0.3 h，每步 10 °C。
- 界面参数按 Table 2；$l=0.5$ µm，$l_\phi=2$ µm；最小网格 0.2 µm（108 588 个单元）。
- 用 `--damage 0/1` 开关控制是否考虑损伤，结果分别保存在 `results/no_damage/` 和 `results/damage/`。

**结果**

- 无损伤（Fig. 11）：TGO 内为数 GPa 的压应力，BC 在波峰处受拉，TC 中最大主应力约 115 MPa（20 °C）。
- 有损伤（Fig. 12）：损伤在冷却中逐渐增长，$d_{max}$ 随温度变化如下：

| 温度 | $d_{max}$ |
|---|---|
| 900 °C | 0.017 |
| 700 °C | 0.23 |
| 400 °C | 0.54 |
| 20 °C | 0.69 |

  - 损伤在**波峰处的 TC/TGO 界面**萌生，并沿界面向侧翼扩展。BC/TGO 界面上只有很小的损伤（$dpprox0.035$）。
  - **这与论文的 BC/TGO 界面分层不同。** 可能原因：TC/TGO 界面强度最低（$f_t=50$ MPa）、广义平面应变假设、cos 形界面，以及隐式准静态求解。
  - 冷却后期部分交错迭代达到 100 次上限（残差约 $10^{-3}$）。
  - 运行时间：有损伤约 1.5 h，无损伤约 3 min。

### 5. 5.4 节 TC 裂纹（`05_TC_crack`）

本算例按论文 3.1 节的**显式算法**重写（`tc_crack.py` + `explicit_pfczm.py`）。原隐式准静态版本保留为 `tc_crack_implicit.py`，结果在 `results_implicit/`。

**设置**

- 一周期模型：$L=30$ µm，$A=5$ µm，TC 500 µm / BC 100 µm / 基体 3 mm，初始 TGO 厚度为 0（只有 TC/BC 界面，取 TC/TGO 界面参数）。
- 边界条件同 Fig. 10a：左边 $u_x=0$，底边 $u_y=0$，右边保持直线。
- 冷却：1150 °C 降至 20 °C，用时 30 min；$T>850$ °C 时计算氧扩散和氧化。
- 网格：界面附近 $h=0.2$ µm（33 688 个单元）；$l=0.5$ µm，$l_\phi=2$ µm。

**算法（逐式对应论文）**

- 位移：中心差分 + 集中质量，式 (56)–(59)、(71)、(75)$_1$：

$$\{\mathbf u\}_{n+1}=2\{\mathbf u\}_n-\{\mathbf u\}_{n-1}-\Delta t^2[M]^{-1}\{F_u^{\rm int}\}_n,\qquad
\{F_u^{\rm int}\}=\int\omega(d)[B_u]^T[D](\boldsymbol\epsilon-\boldsymbol\epsilon_\theta-\boldsymbol\epsilon_n)\,\mathrm dV$$

- 相场：前向 Euler + 集中粘性矩阵，式 (64)–(67)、(75)$_3$：

$$\{d\}_{n+1}=\{d\}_n+\Delta t\,[H]^{-1}\left(\{Y\}_n-\{F_d\}_n\right)$$

- 氧浓度 $c$ 和体积分数 $n$：前向 Euler，式 (45)、(47)（含 $\omega(d)$）、(68)–(70)。
- 历史变量按式 (52) 字面实现：$\mathcal H=\max_\tau\{\psi_0^+,\ f_t^2/2E\}$，$\psi_0^+$ 按式 (38) 取二维谱分解。
- 应力用 hybrid 格式；界面参数按式 (4)、(10)；BC 性质按式 (83) 随 $n$ 混合。
- 时间步按式 (77)–(78)，$\beta=0.5$：$\Delta t=6.0\times10^{-12}$ s。
- 单元循环是 cffi 编译的 C 内核；`verify_kernel.py` 验证它的内力向量和集中质量与 dolfinx 组装结果一致（相对误差 $<10^{-12}$）。

**论文没有给出、只能自行选取的参数**

- 模拟时长 $t_{sim}$：$\Delta t\sim10^{-11}$ s，显式格式无法按物理时间算完 30 min，必须把冷却压缩到很短的模拟时长（各过程速率乘以 $S=1800\ \mathrm s/t_{sim}$）。论文没有说明它的压缩方式或步数。
- 相场粘性 $\eta$：论文只在淬火算例给了 $10^{-8}$ kN·s/mm²（$=10^{-5}$ N·s/mm²），这里沿用。
- 式 (54) 的 $a_1$ 在论文中印作 $4EG_c/(\pi f_t^2)$，量纲不对，按 Wu 的原式取 $a_1=4EG_c/(\pi l f_t^2)$。

**结果**

| 运行 | $t_{sim}$ (s) | 步数 | 最终 $d_{max}$ | 裂纹是否贯穿整个周期 |
|---|---|---|---|---|
| `results/`（默认） | $1.2\times10^{-5}$ | $1.99\times10^{6}$ | 1.00 | 是，$T\approx438$ °C 时贯通 |
| `results_quasistatic/` | $2.4\times10^{-5}$ | $3.98\times10^{6}$ | 0.26 | 否 |
| `results_implicit/`（隐式） | — | — | 0.28 | 否 |

- 默认运行（`fig13_tc_crack.png`）：
  - 损伤约在 777 °C 于界面**波峰**处萌生，与论文一致。
  - 随后沿整条 TC/BC 界面的 TC 一侧发展，在 $T\approx438$ °C 时 $d\to1$ 的带贯通整个周期，TC 与 BC 完全分离。
- **与论文 Fig. 13 的差别：裂纹路径不同。** 这里的贯穿裂纹沿界面走；论文的裂纹离开界面后水平穿过 TC。
- **贯穿只在惯性效应显著时出现。** $t_{sim}=1.2\times10^{-5}$ s 时，冷却收缩在涂层中激起幅值约 ±20 MPa 的竖向应力波（`history.png` 中的 $\sigma_{yy}$ 振荡），反复加载使界面损伤逐步累积直至分离。$t_{sim}$ 加倍后惯性应力减半，显式解回到与隐式准静态解相同的结果（$d_{max}\approx0.26$–$0.28$，只在波峰处有损伤）。
- 粗网格（$h=0.4$ µm）参数扫描见 `results_sweep_coarse/sweep.csv`：

| $t_{sim}$ (s) | $\eta$ (N·s/mm²) | 最终 $d_{max}$ | 贯穿 |
|---|---|---|---|
| $6\times10^{-6}$ | $10^{-4}$ | 0.99 | 是 |
| $6\times10^{-6}$ | $10^{-3}$ | 0.12 | 否 |
| $1.2\times10^{-5}$ | $10^{-5}$ | 0.99 | 是 |
| $1.2\times10^{-5}$ | $10^{-4}$ | 0.28 | 否 |
| $1.6\times10^{-5}$ | $10^{-5}$ | 0.30 | 否 |
| $2.0\times10^{-5}$ | $10^{-5}$ | 0.27 | 否 |
| $2.4\times10^{-5}$ | $10^{-5}$ | 0.17 | 否 |
| $4.8\times10^{-5}$ | $10^{-5}$ | 0.11 | 否 |

  所有出现贯穿的组合，裂纹都沿界面走，没有一组得到水平穿过 TC 的路径。
- 结论：按论文写出的算法和 Table 1–3 的参数，准静态下驱动力不足以形成贯穿裂纹（波峰处最大主应力约 50–60 MPa，与界面强度 50 MPa 相当，远低于 TC 的 200 MPa）。论文的结果应依赖于未报告的数值设置（时间压缩/质量缩放、$\eta$ 等），仅凭论文无法唯一确定。
- 氧化：$n_{max}=0.057$，未形成 $n>0.8$ 的 TGO，与"TGO 厚度约为 0"一致。
- 式 (52) 的字面实现在 $E$ 随温度升高时会使 TC 内出现约 0.03–0.04 的均匀背景损伤，断裂能数值主要由它贡献。
- 运行时间：默认运行约 41 min，`results_quasistatic` 约 83 min（2 核各跑一个）。

### 6. 5.5 节 TC/TGO 界面分层（`06_TC_TGO_delamination`）

**设置**

- 几何同 5.4 节。
- 先在 1150 °C 下等温氧化 800 h：400 个对数间隔的氧化步；力学/相场方程在 40 个时刻求解（20 h 前对数间隔，之后均匀间隔）。
- 再在 30 min 内冷却至 20 °C。

**结果**

- 氧扩散和氧化（Fig. 14）：TGO（$n>0.8$）沿界面均匀生长。
- TGO 厚度（Fig. 15，谷底处）：800 h 时为 **3.37 µm**。一维抛物线估计 $\sqrt{2D\bar c t/M}=4.0$ µm，曲线形状同样为抛物线型。约 55 h 处的跳变是 $n>0.8$ 判据遇到初期较平缓的氧化前沿造成的测量伪影。
- 损伤（Fig. 16）：
  - 等温阶段，氧化膨胀导致 TC/TGO 界面附近出现弱损伤。
  - 冷却阶段，损伤在 TC/TGO 界面靠近波峰的侧翼增长，位置与论文 A–D 点定性一致。
  - 但幅值较小：$d_{max}=0.077$，未形成论文中的界面贯通裂纹。
- 最大主应力和断裂能（Fig. 17）：等温阶段应力随氧化增长，断裂能保持很低；冷却阶段断裂能持续增加，趋势与论文一致。
- 运行时间约 10 min。

## 总结

| 算例 | 与论文的一致性 |
|---|---|
| L 型板 | 峰值和裂纹路径定量吻合 |
| 三点弯 | 厚度等效缩放后，软化段和断裂能吻合；峰值位移偏大 |
| 淬火 | 裂纹形貌定性一致 |
| TBC 算例 4、6 | 应力、氧化/TGO 生长和损伤萌生位置定性一致；损伤程度低于论文结果，详见上文 |
| TBC 算例 5（显式） | 萌生位置一致；贯穿裂纹只在惯性显著时出现，且沿界面而非水平穿过 TC |
