# Case A 复现（FEniCSx 0.11）：TBC 界面 off-peak 预制裂纹的相场扩展模拟

参考论文：`10_29/6.md`（APS-TBC 多物理场分层失效，Sect. 3–4，Table 1–7，Fig. 3）。
耦合：**off-peak 水平预制裂纹 + 相场损伤 + TGO 连续生长（BC→TGO 材料切换）+ 厚度方向传热（heat transfer input）+ TC 烧结/蠕变松弛**。

## 文件

| 文件 | 说明 |
|---|---|
| `mesh_tbc.py` | Gmsh 几何/网格（半周期单元、正弦界面、预制裂纹、加密区） |
| `tbc_phasefield_caseA.py` | 主求解器（传热 → TGO 生长/烧结 → 力学-蠕变-相场交错迭代） |
| `postprocess.py` | 后处理与绘图 |
| `tbc_caseA.msh` | 网格（30 449 个三角形；位移 P2 共 125 730 个自由度） |
| `run_caseA.log` | 30 个循环的完整运行日志 |
| `results/history_caseA.csv` | 时间历程：燃气/界面温度、δ、TGO 厚度、裂纹扩展长度/角度、界面分支检测量、采样点 σ1/σ22 |
| `results/crack_length_per_cycle.csv`, `summary.json`, `run_info.json` | 每个循环的裂纹扩展长度、标定参数 |
| `results/fields_caseA.xdmf/.h5` | 每个循环加热结束/冷却结束时的场（T, u, d, σ1, σ22, 材料），可用 ParaView 打开 |
| `results/snapshots/*.npz` | 快照（绘图用） |
| `results/figures/*.png` | 网格、热循环（Fig. 3）、热梯度、历程、相场（Fig. 10）、损伤主应力（Fig. 11）、σ22 |

运行（单核约 1.3 h）：
```bash
micromamba create -n fenicsx -c conda-forge python=3.12 fenics-dolfinx mpich petsc4py python-gmsh matplotlib scipy
python tbc_phasefield_caseA.py 30
python postprocess.py
```

## 模型

**几何/边界（Fig. 2、Fig. 9）**：宽 $W=12\,\mu m$（半波长，$\lambda=24\,\mu m$），界面 $y_I(x)=5130+5\cos(2\pi x/\lambda)$（AR=10/24）；
TC 250 µm / BC 130 µm（含初始 TGO 1 µm）/ 基体 5 mm。左侧（波峰）对称 $u_x=0$；底部 $u_y=0$；顶面气压 3.5 MPa（随燃气温度同步加载）；右侧规定位移

$$\delta=\frac{W}{H}\int_0^{H}\alpha(T)\,(T-T_{ref})\,dy\qquad(\text{Eq. 7 的连续形式})$$

面外应变取 $\varepsilon_{zz}=\delta/W$（广义平面应变）。
**预制裂纹**：off-peak，距波峰 $W/6=2\,\mu m$；水平、长 0.5 µm、宽 0.06 µm、方形裂尖，左端与 TC/TGO 界面相交，只从 TC 中切除。裂纹区网格 $h=0.03\,\mu m=l_{int}/2$。

**离散**：位移采用 **P2 二次三角形单元**（论文为 8 节点二次四边形），温度和相场采用 P1；蠕变应变、烧结时间、历史场 $\mathcal H$ 等内变量存于每个单元的 3 个 2 阶积分点上。

**传热（Table 4 按原表顺序：基体 / BC / TGO / TC）**

| | 基体 | BC | TGO | TC |
|---|---|---|---|---|
| $k$ [W/mK] | 0.9 | 15.5 | 15.1 | 80 |
| $C_p$ [J/kgK] | 650 | 880 | 610 | 460 |
| $\rho$ [kg/m³] | 6000 | 3970 | 7500 | 8700 |

控制方程为 $\rho c_p\dot T=\nabla\cdot(k\nabla T)$。顶面边界 $q=h_{in}(T_{gas}-T)$，其中 $h_{in}=16\,453$ W/m²K（Eq. 10，$Re=1.84\times10^7$）；底面边界 $q=h_{cool}(T_{cool}-T)$，$h_{cool}=420$。
按此表计算，稳态时 TC 表面为 1320 K，TC/TGO 界面为 1319.8 K。由于 TC 导热系数为 80，温降主要集中在 5 mm 的基体中。

**热循环（Fig. 3，heat transfer input）**
- 0–100 s 为室温。之后每个循环 3600 s，其中加热 3100 s、冷却 500 s；第 $n$ 个循环结束于 $3600n+100$ s，例如第 1 圈结束于 3700 s，第 2 圈结束于 7300 s。
- 加热与冷却都通过对流换热实现。燃气温度与冷却空气温度按 Fig. 3 红线的形状变化：

$$T_{gas}=293+(1323-293)f,\quad f=\begin{cases}1-e^{-\tau/8\,s}&\text{加热}\\ e^{-(\tau-3100)/40\,s}&\text{冷却}\end{cases}$$

  得到的界面温度曲线见 `fig_thermal_cycle.png`：约 30 s 内升到保温温度，冷却约 100 s 后降到 430 K，约 300 s 后回到室温。

**TGO 生长（Eq. 4）**
- 按论文 “accelerated to 6 µm in the first 30 h at 1343 K”，采用以 1343 K 为基准的 Arrhenius 等效时间：$dt_{eq}=dt\,e^{-\frac{Q_0}{Rm}(\frac1T-\frac1{1343})}$，$h=\max\!\big(1,\,6\,(t_{eq}/30\,\text{h})^{0.332}\big)$。
- 只在加热段生长。
- BC 单元形心落入 $h$ 内时切换为 TGO，并在切换时刻令其无应力（论文：“creep model eliminates the stress induced by the material switch”）。

**蠕变（Table 3）**：采用 J2-Norton 定律，1123 K 以上激活，隐式径向返回。

**TC 烧结**（论文没有给出参数，本文按 Case A 的裂纹历程标定）：

$$E_{TC}=E_{dep}(T)\,[1+(F-1)S],\quad \varepsilon^{sint}=-\varepsilon_s S,\quad S=1-\exp[-(t_{eq}/t_c)^{N}]$$

取 $F=1.5$，$N=4$（Avrami 型），$t_c=1$ h（以 1373 K 为基准，$Q_s=200$ kJ/mol），$\varepsilon_s=10^{-3}$。
在约 1320 K 下，$S$ 在第 1/2/3/5 圈约为 0.03 / 0.41 / 0.93 / 1.0，即模量主要在第 2–3 圈增大，随后饱和。

**相场**（Miehe 2015 应力型，Table 6）：

$$\tau\dot d=(1-d)\mathcal H-(d-l^2\Delta d),\qquad \mathcal H=\max_t\ \zeta\Big\langle\sum_{i=1}^{2}\Big(\frac{\langle\tilde\sigma_i\rangle}{\sigma_c}\Big)^2-1\Big\rangle$$

- 参数：$\sigma_c=300$ MPa，$\zeta=0.2$，$l=0.06\,\mu m$，$\tau=1$ s，$\boldsymbol\sigma=[(1-d)^2+10^{-6}]\tilde{\boldsymbol\sigma}$。
- **裂纹只在 TC 中扩展**：TGO/BC 不施加驱动力；紧贴 TC/TGO 界面、厚 0.15 µm 的一层 TC 同样不施加驱动力。这样裂纹不会穿透 TGO，也不会在裂纹口沿界面向波峰方向产生分支。
- 力学、蠕变和相场在每个时间步内交错迭代至收敛（上限 300 次）。单步 $\Delta d>0.2$ 时，时间步自动减半。

## 结果（30 个循环）

见下表与 `results/figures/fig_crack_length_per_cycle.png`。

| 循环 | TGO 厚度 [µm] | 本模型扩展量 [µm] | 论文 [µm] |
|---|---|---|---|
| 1 | 1.715 | 0.645 | 0.58 |
| 2 | 2.158 | 1.470 | 1.48 |
| 3 | 2.469 | 2.056 | 1.97 |
| 4 | 2.717 | 2.086 | – |
| 7 | 3.271 | 2.086 | 1.97（止裂） |
| 10 | 3.683 | 2.086 | – |
| 20 | 4.635 | 2.086 | – |
| 30 | 5.303 | 2.086 | 1.97 |

计算条件：2 核 CPU，30 圈墙钟时间约 4775 s。完整日志见 `run_caseA.log`。

- 裂纹从方形裂尖向**右上方**（约 9–11°）扩展，**只位于 TC 内**。裂纹口左侧没有界面分支：历程文件中 `branch_d_left` 列始终约为 0。
- 扩展只发生在冷却段。各圈扩展量的趋势与论文一致：第 1 圈约 0.6 µm，第 2–3 圈扩展最快；之后由于 TGO 增厚，最大第一主应力从波峰迁移到 mid-level（`fig11_damaged_sigma1.png`），裂尖应力下降，裂纹停止扩展。
- 与论文的差异：本模型在第 4 圈后即完全止裂（论文为第 7 圈），最终扩展长度约 2.09 µm（论文约 1.97 µm）。

## 说明与假设

1. Table 4 按原表列顺序使用。由此得到的 TC 内温降很小（界面约 1320 K），与论文正文所述的 1270 K 界面温度不同。
2. 热循环采用 heat transfer input。燃气温度的指数变化形状是根据 Fig. 3 红线拟合的。
3. 烧结参数 $F, N, t_c, \varepsilon_s$ 论文未给出，本文按 Case A 的裂纹历程（0.58 / 1.48 / 1.97 µm）标定。如果不考虑烧结，第 1 圈扩展 0.615 µm，但第 2 圈只到约 0.83 µm，第 3 圈不再扩展。
4. 相场驱动力只使用面内主应力；"damaged σ₁" 输出为 $g(d)\sigma_1$，XDMF 中另存了 $\tilde\sigma_1 d$（`sigma1_x_d`）。
5. 裂纹扩展长度定义为初始裂尖中心到 $d\ge0.9$ 区域最远点的距离（不含初始 0.5 µm 裂纹）。
