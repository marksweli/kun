# 11.md §4.4 复现：顶层涂层 CMAS 辅助疲劳裂纹扩展（渗透 10 min）

使用 FEniCSx (dolfinx 0.9) 实现论文中扩散、Walker 粘塑性与 AT2 相场的弱耦合（交错）模型，只计算 **CMAS 渗透 10 min** 这一工况。

## 运行

```bash
# 环境（见仓库其它 *_fenicsx 目录）：micromamba create -p /opt/fenicsx -c conda-forge \
#   python=3.12 fenics-dolfinx=0.9 mpich matplotlib scipy gmsh python-gmsh
cd 11_21/11
OMP_NUM_THREADS=1 /opt/fenicsx/bin/python cmas_fatigue_fenicsx/cmas_fatigue.py '{"edot0":0.04}'  # ~26 min, 2 核
/opt/fenicsx/bin/python cmas_fatigue_fenicsx/plot_results.py
```
可以通过 JSON 参数覆盖 `P` 字典中的任意参数，例如 `'{"ncyc":4,"de":0.001}'`。

## 模型

| 项目 | 设置 |
|---|---|
| 几何 | $0.1\times0.1$ mm 多孔顶层涂层（论文 model 1 的统计等效模型），共 34 个随机圆孔，孔隙率 8.9%（目标 9%）。孔径取弦长 $r_1$（Eq. 28，即直径），服从 $N(6,2.5)$ μm |
| 网格 | gmsh 线性三角形单元，$h=1$ μm，共 22992 个单元（论文 model 1 为 18984 个四边形单元） |
| 扩散 | $\dot C = D\nabla^2 C$，$D=5\times10^{-12}\,\mathrm{m^2/s}$；顶面 $C=4$ mol%，其余边界及孔壁零通量；时间 600 s，向后 Euler，$\Delta t=2$ s。之后浓度场冻结（图17中的 "Pre-CMAS"） |
| 强度退化 | $g_f=(1-\chi\theta)g_{f0}$，$\chi=0.66$，$\theta=C/(C+K_L)$，$K_L=e^{-\Delta g_b/RT}$ |
| 粘塑性 | Eq.(10) Walker 模型：$\dot p=\dot\varepsilon_0\langle\sigma_{eq}/K\rangle^n$，$K=K_1=59.3$ MPa，$n=1.23$（Table 1）；$K_2=n_4=n_5=n_7=n_1=0$ 时无硬化。背应力的饱和值仅为 $\Omega'+n_2/n_3\approx1$ MPa，远小于流动应力，故忽略。积分点上采用向后 Euler 径向返回映射和一致切线 |
| 相场 | AT2：$(g_f/l_c+2\mathcal H)d-g_f l_c\nabla^2 d=2\mathcal H$；$\mathcal H=\max(\tfrac12\boldsymbol\varepsilon_e:\boldsymbol\sigma'+G_{in})$，$\mathrm dG_{in}=\mathrm d\boldsymbol\varepsilon_{in}:\boldsymbol\sigma'$（Eq. 5/6），按论文不做拉压分解；退化函数 $(1-d)^2+\kappa$ |
| 边界/载荷 | 左边 $u_x=0$，底部两角点 $u_y=0$；右边施加位移控制循环应变，应变范围 $\Delta\varepsilon=3\%$，$R=0.1$（$\varepsilon_{max}=3.33\%$，$\varepsilon_{min}=0.333\%$），加载速率 0.5%/s，共 10 周（每周 12 s）；载荷沿水平方向，垂直于 CMAS 渗透方向 |
| 求解 | 每个增量（$\Delta\varepsilon=0.025\%$，$\Delta t=0.05$ s）先用 Newton 法求解 $u$，再更新历史场并线性求解 $d$（交错格式、固定增量，与论文 §4.1.1 一致）。$\Delta\varepsilon=0.1\%$ 的结果与此差别 <3%，说明增量已收敛 |
| 失效单元 | $d\ge0.95$ 的单元。由于本文网格单元面积平均约 0.39 μm²，为与论文 1 μm 网格比较，统计时用**失效面积 (μm²) 作为等效 1 μm² 单元数**；按单元形心 $y>50$ μm 划分上半部分 |

### 论文未给出、需要标定的参数（重要）

| 参数 | 取值 | 依据 |
|---|---|---|
| $\dot\varepsilon_0$（Eq. 10 未写参考速率） | 0.04 s⁻¹ | 若按字面取 1 s⁻¹，0.5%/s 下的流动应力只有约 0.88 MPa（已测试），损伤几乎为零，也与论文图13同一模型约 12 MPa 的拉伸峰值矛盾。取 0.04 s⁻¹ 时，固体在 0.5%/s 下的流动应力约 10.9 MPa，与图13的量级一致。另外测试了 0.03 和 0.06：前者失效发展偏快，后者在前期偏慢 |
| $g_{f0}$, $l_c$ | 28.8 J/m², 2 μm | 由 AT2 一维峰值应力 $\sigma_c=\tfrac{9}{16}\sqrt{Eg_f/(3l_c)}=132$ MPa（图5，0 mol%）得 $g_f/l_c=14.4$ MPa；取 $l_c=2h$ |
| $K_L$ | 1.78 mol%（以摩尔分数计时对应 $\Delta g_b\approx54$ kJ/mol，$T=1350$ °C） | 由图5各浓度下峰值应力比 $\sigma(C)/\sigma(0)=1-\chi\theta(C)$ 拟合：2/4/6/8 mol% 分别给出 1.77/1.80/1.79/1.76，结果自洽 |

此外，论文中的参数之间存在矛盾，例如 Table 1 的 $E=11.5$ GPa 与图5的初始斜率约 90 GPa 不符。因此本复现只追求定性一致和量级一致。

## 结果（`results/`）

| 文件 | 内容 |
|---|---|
| `fig11_cmas_profile.png`, `cmas_profile.csv` | 沿深度的平均 CMAS 浓度：10 min 曲线与论文图11几乎重合（底部 1.38 mol%，论文约 1.4） |
| `fig17_fatigue_10min.png` | 对应图17(c)：10 min CMAS 场以及第 3/5/7/9 周的相场 |
| `fig18_19_failed_elements.png`, `failed_elements.csv` | 对应图18/19：上下两半的失效单元数与比例，并与论文 10 min 数据（图中读取）对比 |
| `stress_strain.png`, `stress_strain.csv` | 名义应力–应变/时间曲线 |
| `cmas_10min.xdmf`, `fatigue_10min.xdmf` | ParaView 场文件（C，以及每周结束时的 d） |
| `fields.npz`, `pores.csv`, `params.json` | 网格、场数据、孔隙几何和全部参数 |

失效单元数（等效 1 μm²）：

| 周次 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 本文 下半/上半 | 0/0 | 2/2 | 6/11 | 20/35 | 48/57 | 92/60 | 109/61 | 111/62 | 113/63 | 113/63 |
| 论文 下半/上半 | 2/0 | 29/18 | 32/50 | 35/76 | 35/92 | 35/100 | 35/108 | 35/117 | 35/121 | 37/125 |

### 讨论

- **与论文一致的方面**：裂纹从孔边萌生，沿相邻孔之间的韧带垂直于加载方向扩展；背景中可以看到弥散的低相场带，与图17相似。CMAS 使上半部分的 $g_f$ 降低约 30%，因此第 2–4 周的失效以上半部分为主（第 3–4 周上半占 64%，论文为 61%–68%），即表面裂纹先萌生。失效单元总数先快速增长，随后趋于饱和（第10周 176，论文 162）。
- **差异**：本文随机孔隙模型与论文 model 1 的几何并不相同，主裂纹在第 5–6 周贯穿整个厚度，之后试样卸载、失效数饱和，下半部分的最终占比也因此偏高（约 64%，论文约 23%）。在论文中，底部裂纹萌生于第 1–2 周；而本模型第 1 周没有单元达到 $d\ge0.95$。这些差异主要来自孔隙几何以及上述标定参数。
