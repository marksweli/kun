# TBC 二维等温氧化 化学-力学-损伤耦合模型（FEniCSx 复现）

复现 `10_29/7.md`（Sun et al., *Synergistic effects of porosity and interface roughness on oxidation failure in thermal barrier coating system*）中的二维等温氧化模型。

**本次工况**：TC 孔隙率 7%，TC/TGO 界面振幅 $A=10\,\mu\mathrm{m}$，$T=1373.15\,\mathrm{K}$，氧化 2000 h。

## 文件

| 文件 | 说明 |
|---|---|
| `tbc_oxidation.py` | 主程序：gmsh 建模 + dolfinx 交错求解（扩散-氧化 → 力学 → 历史变量 → 相场） |
| `plot_results.py` | 后处理：生成相场-时间曲线、TGO 厚度、裂纹/S22 云图、场分布图和 `summary.json` |
| `results_P7_A10/` | 本工况结果：`history.npz`（时间历程）、`snapshots.npz`（界面窗口快照）、`fields.xdmf/.h5`（全场，ParaView 可直接打开）、`run_info.json`、`summary.json`、`fig_*.png` |
| `results_run.log` | 本工况的运行日志 |

## 运行

```bash
# 环境 (conda-forge)
micromamba create -p /opt/fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.9 mpich python-gmsh matplotlib scipy
# 计算 (串行, 2 核机器上约 1 h)
/opt/fenicsx/bin/python tbc_oxidation.py --porosity 0.07 --amp 10 --hours 2000 --outdir results_P7_A10
/opt/fenicsx/bin/python plot_results.py --outdir results_P7_A10
```

其他工况直接修改 `--porosity`（如 0.02/0.04/0.10/0.15）和 `--amp`（5/10/15）即可。

## 模型

### 几何与网格（Fig. 5）
- 宽 $60\,\mu\mathrm{m}$（一个波长），SUB 3 mm、BC 200 μm、TC 300 μm。
- 界面 $y=y_0+A\cos(2\pi x/60)$：$x=0,60$ 处为峰（BC 凸起），$x=30$ 处为谷。
- 初始 TGO：界面下方法向距离 1 μm 内取 $n=1$。
- 界面两侧 18 μm 内网格尺寸 0.5 μm（满足 $h\le l_0/4$），远场逐渐粗化，约 3.0 万个三角形单元。

### 控制方程
1. **TC 中的氧扩散**（Eq.1–5）：

$$D^{\mathrm{eff}}=\phi^2\left(\frac{1}{D_m}+\frac{1}{D_K}\right)^{-1},\qquad \frac{\partial c}{\partial t}-\nabla\cdot(D^{\mathrm{eff}}\nabla c)=-\alpha c$$

   其中 $D_m=1.8\times10^{-4}$、$D_K=3.18\times10^{-3}\,\mathrm{m^2/s}$，当 $\phi=7\%$ 时 $D^{\mathrm{eff}}=8.35\times10^{-7}\,\mathrm{m^2/s}$。
2. **BC/TGO 中的应力辅助扩散**（Eq.6–9, 11）：

$$\frac{\partial c}{\partial t}-\nabla\cdot\left[D\left(\nabla c-\frac{\Omega}{RT}c\nabla\sigma_m\right)\right]=-\kappa\frac{\partial n}{\partial t},\qquad D=(1-n)D_{BC}+nD_{TGO}$$

3. **TGO 生长**（Eq.10）：$\dfrac{\partial n}{\partial t}=\xi(1-n)c$，$\xi=10^{-4}$，$\kappa=\dfrac{\rho_{\mathrm{Al_2O_3}}}{M_{\mathrm{Al_2O_3}}}\eta_{\mathrm{O_2}}=\dfrac{3950}{0.10196}\times1.5=5.81\times10^4\,\mathrm{mol/m^3}$。
4. **力学**（Eq.12–17, 40–41）：平面应变、小变形；BC/TGO 混合区刚度用混合律 $\mathbf K=(1-n)\mathbf K_{BC}+n\mathbf K_{TGO}$；生长应变 $\boldsymbol\epsilon_g=(n-n_0)\gamma\mathbf I$（3D 各向同性）；TC 刚度退化 $\mathbb C_c=g(c)\mathbb C$，其中 $g=(1-c)^2+k$。
5. **相场**（Eq.48–49，只在 TC 中求解，界面上为自然边界）：

$$\left[\frac{2l_0H}{G_c}+1\right]c-l_0^2\Delta c=\frac{2l_0H}{G_c},\qquad H=\max_{\tau\le t}\psi_e^+$$

### 边界条件
- 氧浓度：TC 顶面 $c=2.5\,\mathrm{mol/m^3}$，BC 底面（及 SUB）$c=0$。
- 力学：底面 $u_y=0$，两侧 $u_x=0$。
- 相场：$\nabla c\cdot\mathbf n=0$，初值为 0。

### 数值方法
- 扩散-反应 $(c,n)$ 用 P1×P1 混合元整体 Newton 求解，时间离散为向后 Euler；采用顶点积分（集中质量），保证反应前沿处单调。
- 位移用 P2；$H$ 存在 2 阶积分点上；相场用 P1。
- 每个时间步内交错迭代"力学 ↔ 相场"，直到 $\|\Delta c\|_\infty<10^{-4}$（对应 Fig. 4 的流程）。
- 自适应步长：最大 5 h；当单步相场增量超过 0.05 时步长减半（最小 30 s）。

## 论文未给出或含糊、需要自行假设的参数（重要）

| 项 | 本文取值 | 说明 |
|---|---|---|
| 生长应变 $\gamma$ | **0.0143** | 论文未给出。标定目标为 7%、A=10 μm 时 $\max c=0.4$ 出现在约 520 h（论文 Fig.15a）。γ=0.012/0.014/0.015/0.018 对应的 0.4 到达时间分别为 >700/550/455/275 h，结果对 γ 很敏感 |
| 吸附系数 $\alpha$ | 1 s⁻¹ | Langmuir 高压极限（论文："increases asymptotically to 1"） |
| $\eta_{O_2}$、$\rho$、$M$ | 1.5、3950 kg/m³、0.10196 kg/mol | Al₂O₃ 标准值 |
| $\Omega/RT$ | $5\times10^{-23}$ Pa⁻¹（论文原值） | 按 SI 单位理解时，这一耦合项实际上可以忽略。若想考察应力辅助扩散，可用 `--omega_rt 1e-9` 之类的物理量级值 |
| 拉/压分解 | **谱分解（Miehe）**，`--split spectral` | 论文 Eq.33 写的是 Amor 体积-偏量分解，但正文要求"只由拉伸部分驱动，压应力不贡献"。实测 Amor 分解（`--split amor`）下损伤最大值出现在**峰**附近受压区，和论文的谷处萌生不符；谱分解得到谷处萌生，与 Fig.15b 一致 |
| 初始 TGO 应力 | 以初始 TGO 为无应力参考，即 $\epsilon_g=(n-n_0)\gamma$ | 保证 $t=0$ 时相场为 0（Fig.15a 曲线从 0 开始） |
| 侧边周期约束 | 两侧 $u_x=0$ | 论文要求侧边各点 $u_x$ 相同。由于 3 mm 基体主导，侧边的均匀位移量可以忽略 |
| 残余刚度 $k$ | $10^{-6}$ | — |

## 结果（7%，A=10 μm，1373.15 K，2000 h）

网格约 3.0 万单元，共 446 个时间步，串行耗时约 41 min。

| 指标 | 本计算 | 论文 |
|---|---|---|
| 200 h TGO 厚度（峰/谷/平均） | 5.19 / 4.21 / 4.69 μm | 平均约 5 μm（实验，Fig.9/11） |
| 1000 h TGO 厚度（峰/谷） | 12.1 / 8.4 μm | 约 11 / 约 7 μm（Fig.13a） |
| 2000 h TGO 厚度（峰/谷/平均） | 17.7 / 11.5 / 12.9 μm | — |
| 相场 $c$ 达到 0.4 的时间 | 515 h（γ 按此标定） | 约 520 h（Fig.15a） |
| 失稳开裂（$c\to1$）的时间 | 534 h | 约 545 h |
| $c=0.4$ 时谷处 TGO 厚度 | 6.3 μm | 5 μm |
| 萌生位置 | TC/TGO 界面谷附近（x≈34 μm，左右对称） | 谷处 |
| TC 中 S22 | 谷两侧受拉、峰处受压，量级为 −400 ~ +300 MPa | −400 ~ 300 MPa（Fig.15b） |
| 最终状态 | 裂纹沿界面扩展：534 h 时约 36 μm，1175 h 时达到 40 μm，此后基本稳定 | 沿界面横向扩展 |

图件：
- `fig_phase_vs_time.png`：对应 Fig.15a
- `fig_crack_S22.png`：对应 Fig.15b，三个面板依次为 $c=0.4$、开裂、失效时的相场和 TC 中的 S22
- `fig_tgo_thickness.png`：对应 Fig.11/12
- `fig_fields.png`：200/1000/2000 h 时的 TGO 分布、氧浓度、S22 和相场，对应 Fig.13

说明：
1. 由于 $\Omega/RT$ 采用论文原值后耦合项可忽略，本计算中"峰厚谷薄"完全来自几何效应：峰处 BC 凸起，接收氧通量的面积更大。论文认为应力也加剧了这种不均匀，所以本计算谷处厚度偏大约 1 μm。
2. 开裂是失稳过程：$c>0.4$ 之后约 20 h 内相场升到 1，裂纹在一个步长内扩展到约 36 μm。因此"萌生"和"扩展"出现在同一时刻，这一点与 Fig.15a 中曲线近乎竖直的形态一致。
