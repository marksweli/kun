# TBCs 热冲击开裂（FEniCSx 复现 `10_29/8.md` 第 3.4 节）

论文用 TM-DH-BPD 近场动力学（Abaqus）计算热障涂层（TC/BC 双层）的热冲击开裂。这里用 FEniCSx（dolfinx 0.9）实现，断裂模型换成**热-力耦合 AT1 相场模型**。几何、材料（Table 6）、热边界、网格尺寸（2D 中 $\Delta x=5$ μm）和总时长 $T=0.3$ s 都和论文保持一致。

| 目录 | 内容 |
|---|---|
| [`2D/`](2D/README.md) | 3.4.1 淬火/加热、3.4.2 界面韧性 $G_c^{TC/BC}$、3.4.3 TC 厚度 |
| [`3D/`](3D/README.md) | 3.4.4 三维模型，3 种热冲击面 × 宽度 $w=50/100$ μm |

环境：`micromamba create -p /opt/fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.9 mpich matplotlib scipy`，串行运行。

## 共同模型

- **材料（Table 6）**：

  | 层 | $E$ | $\alpha$ | $\rho$ | $\kappa$ | $c_v$ | $G_c$ |
  |---|---|---|---|---|---|---|
  | TC | 11.7 GPa | $10.04\times10^{-6}$ | 5650 | 1.05 | 450 | 0.008 N/mm |
  | BC | 71.9 GPa | $17.65\times10^{-6}$ | 7320 | 4.3 | 780 | 0.3 N/mm |

  界面韧性 $G_c^{TC/BC}$ 赋给紧贴界面、厚度为 $\ell$ 的一层 TC 单元。
- **传热**：$\rho c_v\dot\theta=\kappa\Delta\theta$，上表面（3D 还包括 case 2/3 的侧面）突然变为冲击温度，其余面绝热。向后 Euler 格式，热容矩阵集中，裂纹不影响传热。
- **力学**：准静态，$\boldsymbol\varepsilon_e=\boldsymbol\varepsilon(\mathbf u)-\alpha(\theta-\theta_0)\mathbf I$，应力为 $\boldsymbol\sigma=g(d)\,\mathbb C:\boldsymbol\varepsilon_e$，其中 $g=(1-d)^2+k$。
- **相场（AT1，Amor 分解）**：

$$2\mathcal H d-\tfrac34G_c\ell\Delta d=2\mathcal H-\frac{3G_c}{8\ell},\qquad \mathcal H=\max\Big(\tfrac{3G_c}{16\ell},\max_{\tau\le t}\psi^+\Big),\qquad \ell=2\Delta x$$

  每个时间步内做交错迭代，收敛条件为 $\max|\Delta d|<10^{-3}$。$G_c$ 加 ±2% 的随机扰动。
- **底部力学边界**：默认 `--bottom clamped`，取 $\mathbf u=0$，模拟刚性基底。论文的 Fig. 19 图片不可得，边界条件是根据文字推断的。另有 `--bottom roller` 选项（$u_y=0$ 加一个固定点），对比结果见 `2D/results/roller/`。

## 与论文的主要差异

1. 用 PD 换成相场：相场裂纹有扩散宽度 $\ell$，界面用厚度为 $\ell$ 的带来表示。
2. 时间尺度：传热时间 $h_{TC}^2/a\approx24$ ms（$a=\kappa/\rho c_v\approx4.1\times10^{-7}\,\mathrm{m^2/s}$）。所以这里界面剥离明显晚于论文给出的 28 ms 完全剥落。
3. 固支底部会使 BC 底层在长时间冷却后受到约束拉应力，出现一条贴底的损伤带。这是边界条件造成的人工效应，界面附近的 BC 没有损伤。
4. 3D 模型取半长（利用对称），$\Delta x=10$ μm（论文为 5 μm），以控制计算量。
