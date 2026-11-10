# 陶瓷板淬火热冲击（FEniCSx 复现 `10_29/8.md` 第 3.3 节）

本目录用 FEniCSx（dolfinx 0.9）复现论文中氧化铝陶瓷板的淬火开裂算例（Fig. 14–17，实验见 Jiang et al. 2012 [76]）。论文的 TM-DH-BPD 近场动力学模型在这里换成**热-力耦合相场断裂模型**，与论文参考的相场工作 [77] 思路一致。几何、边界条件、材料（Table 5）、网格密度和总时长都和论文保持一致。

## 文件

| 文件 | 说明 |
|---|---|
| `thermal_shock.py` | 主程序：瞬态热传导 → 准静态力学 ↔ 相场交错迭代 |
| `run_all.sh` | 4 个初始温度的运行脚本（2 核并行） |
| `plot_results.py` | 后处理：生成 `results/fig*.png` 和 `results/summary.json` |
| `results/theta0_{300,400,500,600}C/` | `snapshots.npz`（$t=1,3,10,30,100,300$ ms 时的温度场与损伤场）、`history.npz`（能量历程）、`run_info.json` |
| `results/log_*.txt` | 运行日志 |

## 运行

```bash
micromamba create -p /opt/fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.9 mpich matplotlib scipy
./run_all.sh            # 4 个温度；单个工况单核约 40 min（两个工况同时跑时约 1.7 h）
/opt/fenicsx/bin/python thermal_shock.py --theta0 600 --outdir results/theta0_600C   # 单个工况
/opt/fenicsx/bin/python plot_results.py
```

## 模型

- **几何（Fig. 14）**：取 $50\times10$ mm 板的左下 1/4，$25\times5$ mm。平面应力。
- **边界条件**：
  - 左边 $x=0$、下边 $y=0$ 为热冲击面，$\theta=\theta^*=20\,^\circ$C，力学上自由。
  - 右边 $x=25$ mm 为对称面，$u_x=0$，绝热。
  - 上边 $y=5$ mm 为对称面，$u_y=0$，绝热。
- **初始温度**：$\theta_0=300/400/500/600\,^\circ$C。
- **材料（Table 5）**：$E=370$ GPa，$\kappa=31$ W/(m·°C)，$G_c=42.47$ N/m，$C_v=880$ J/(kg·°C)，$\rho=3980\ \mathrm{kg/m^3}$，$\alpha=7.5\times10^{-6}$ /°C；$\nu=1/3$（二维键型 PD）。
- **热传导**：$\rho C_v\dot\theta=\kappa\Delta\theta$，向后 Euler，热容矩阵用行和集中（避免早期小 $\Delta t$ 下温度出现非物理振荡）。和论文一样，裂纹不影响传热（单向耦合）。
- **力学**：准静态，$\boldsymbol\varepsilon_e=\boldsymbol\varepsilon(\mathbf u)-\alpha(\theta-\theta_0)\mathbf I$。采用 hybrid 格式：应力为 $g(d)\,\mathbb C:\boldsymbol\varepsilon_e$，对 $\mathbf u$ 是线性的，用 MUMPS 求解。
- **相场（AT1，带弹性阈值）**：

$$2\mathcal H\,d-\tfrac34G_c\ell\,\Delta d=2\mathcal H-\frac{3G_c}{8\ell},\qquad \mathcal H=\max\Big(\frac{3G_c}{16\ell},\ \max_{\tau\le t}\psi^+(\boldsymbol\varepsilon_e)\Big)$$

  其中 $\psi^+$ 采用 Amor 分解，$\ell=2\Delta x=0.125$ mm，对应 $\sigma_c=\sqrt{3G_cE/(8\ell)}=217$ MPa。$G_c$ 按单元加 ±2% 的均匀随机扰动（`--noise`，固定种子），用来打破完全规则网格上的对称性，类似于实际材料的非均匀性。
- **网格与时间**：$\Delta x=0.0625$ mm，$400\times80=32{,}000$ 个四边形单元，和论文的 32,000 个物质点一致。总时长 0.3 s。时间步随时间增大：0–2 ms 为 20 μs，2–20 ms 为 0.1 ms，20–100 ms 为 0.5 ms，100–300 ms 为 2 ms，共 540 步。每步交错迭代，直到 $\max|d_k-d_{k-1}|<10^{-3}$（上限 50 次）。

## 结果

| Fig. | 本复现 |
|---|---|
| Fig. 15 能量演化 | `results/fig15_energy.png`（应变能、断裂耗散能） |
| Fig. 16 600 °C 温度/损伤演化 | `results/fig16_evolution_600C.png` |
| Fig. 17 最终裂纹形态 | `results/fig17_crack_patterns.png`（由 1/4 模型镜像得到整板） |

从下冷却面起算、深度超过不同值的裂纹条数（统计范围为 1/4 模型下边 $x>1.5$ mm 的部分）：

| $\theta_0$ (°C) | >0.5 mm | >1 mm | >2 mm | >3 mm | 应变能峰值时刻（ms） | 最终断裂能（J/m） |
|---|---|---|---|---|---|---|
| 300 | 14 | 9 | 6 | 3 | 258 | 2.74 |
| 400 | 18 | 10 | 6 | 5 | 124 | 3.31 |
| 500 | 18 | 12 | 8 | 6 | 140 | 3.68 |
| 600 | 22 | 14 | 7 | 6 | 164 | 4.02 |

- **演化过程（Fig. 16）**：冷却面先出现损伤（1 ms），随后形成大致等间距的短裂纹（3–10 ms）。冷却前沿向内推进时，部分裂纹继续扩展成长裂纹，相邻裂纹的扩展受到抑制，呈现论文所述的**周期性和分级性**（长短裂纹交替出现）。
- **温度的影响（Fig. 17）**：初始温度越高，裂纹越密，耗散能越大，和实验 [76] 及论文的趋势一致。
- **能量（Fig. 15）**：应变能先上升到峰值，然后随裂纹扩展而下降，温度越高峰值越大，与论文描述一致。这里力学按准静态求解（不需要论文中的质量缩放），所以动能恒为 0。
- **与论文的差别**：冷却面上有一层宽约 $\ell$ 的损伤带。原因是边界节点的温度在 $t=0^+$ 突降到 20 °C，冷却层厚度小于 $\ell$，相场无法在这么薄的层内局部化。论文 Fig. 16(a) 也在冷却面上观察到损伤。少数时间步的交错迭代达到 50 次上限，残差约 $2.5\times10^{-3}$，出现在裂纹失稳扩展的时刻，对最终形态影响很小。
