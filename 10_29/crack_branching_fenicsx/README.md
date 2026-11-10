# 动态裂纹分叉（FEniCSx 复现 `10_29/8.md` 第 3.2 节）

本目录用 FEniCSx（dolfinx 0.9）复现论文的动态裂纹分叉算例（Fig. 9–13）。论文用的是 Abaqus/VUEL 中的 DH-BPD 近场动力学；FEniCSx 是有限元框架，所以这里改用**相场断裂模型**（论文第 3.2 节也把相场模型列为本算例的标准对照方法 [73–75]），材料参数、几何、载荷、网格和时间步都和论文保持一致。

## 文件

| 文件 | 说明 |
|---|---|
| `crack_branching.py` | 主程序：显式中心差分动力学 + 相场（每个时间步交错更新） |
| `run_all.sh` | 本目录所有工况的运行脚本（2 核并行） |
| `plot_results.py` | 后处理：生成 `results/fig*.png` 和 `results/summary.json` |
| `results/sigma{2,4,6,12}MPa/` | 不同外加应力，$\Delta x=0.25$ mm，$\Delta t=20$ ns |
| `results/mesh_h{0.5,0.25,0.125}/` | 网格收敛性，$\sigma_0=2$ MPa，$\Delta t=10$ ns（同论文 Table 4） |
| `results/log_*.txt` | 运行日志 |

每个工况目录包含：`snapshots.npz`（每 5 μs 的损伤场 `d` 和位移幅值 `umag`，规则网格节点值）、`history.npz`（每 0.1 μs 的裂尖坐标和能量）、`run_info.json`。

## 运行

```bash
micromamba create -p /opt/fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.9 mpich matplotlib scipy
./run_all.sh                                  # 全部工况（2 核机器约 2 h，耗时主要来自 0.125 mm 网格）
/opt/fenicsx/bin/python crack_branching.py --sigma 2 --h 0.25 --dt 20 --outdir results/sigma2MPa   # 单个工况
/opt/fenicsx/bin/python plot_results.py
```

## 模型

- **几何与载荷（Fig. 9）**：平板 $100\times40$ mm，预制裂纹为 $y=20$ mm、$0\le x\le50$ mm；$t=0$ 时在上、下边突加拉伸面力 $\sigma_0$。平面应力。
- **材料**：$E=72$ GPa，$\rho=2235\ \mathrm{kg/m^3}$，$G_c=3.8$ N/m；$\nu=1/3$（二维键型 PD 隐含的泊松比）。
- **相场（AT1，带弹性阈值）**：

$$\psi=g(d)\,\psi^+(\boldsymbol\varepsilon)+\psi^-(\boldsymbol\varepsilon),\qquad g(d)=(1-d)^2+k$$

$$2\mathcal H\,d-\tfrac34G_c\ell\,\Delta d=2\mathcal H-\frac{3G_c}{8\ell},\qquad \mathcal H=\max\Big(\frac{3G_c}{16\ell},\ \max_{\tau\le t}\psi^+\Big)$$

  这里 $\psi^\pm$ 采用 Amor 体积/偏量分解（平面应力下的二维体积模量 $K=\lambda+\mu$）。取 $\ell=2\Delta x$。AT1 的强度为 $\sigma_c=\sqrt{3G_cE/(8\ell)}$，当 $\Delta x=0.25$ mm 时 $\sigma_c=14.3$ MPa。
- **预制裂纹**：裂纹线上的节点施加 Dirichlet 条件 $d=1$。
- **时间积分**：中心差分，采用行和集中质量；每个时间步先显式更新位移，再更新历史场、求解相场（CG+BoomerAMG），并施加不可逆约束 $d\ge d_{n}$。
- **网格**：$\Delta x=0.25$ mm，共 $400\times160=64{,}000$ 个四边形单元，和论文的 64,000 个物质点一致。计算到 $t=50$ μs。

## 结果

| Fig. | 本复现 |
|---|---|
| Fig. 10 位移/损伤演化 | `results/fig10_evolution_sigma2MPa.png` |
| Fig. 11 不同外加应力 | `results/fig11_damage_vs_stress.png`（2/4/6 MPa） |
| Fig. 12 网格收敛（损伤） | `results/fig12_mesh_damage.png` |
| Fig. 13 裂尖速度 | `results/fig13_tip_speed.png` |

`summary.json` 中的主要数据（$\sigma_0=2$ MPa）：

| $\Delta x$ (mm) | 单元数 | 起裂（μs） | 分叉时刻（μs） | 分叉位置 $x$（mm） | 分叉前平均速度（m/s） | 墙钟时间/步（s，单核） |
|---|---|---|---|---|---|---|
| 0.5 | 16,000 | 8.2 | 21.6 | 68.5 | 1460 | 0.10 |
| 0.25 | 64,000 | 8.1 | 20.4 | 66.3 | 1338 | 0.37 |
| 0.125 | 256,000 | 8.0 | 20.7 | 66.6 | 1292 | 1.28 |

- **分叉**：$\sigma_0=2$ MPa 时，裂纹约在 $t\approx20.5$ μs、$x\approx66$ mm 处分成对称的两支，然后延伸到右边界，和论文的分叉时刻 $t=20$ μs 以及 Fig. 10/11 的形态一致。三套网格得到的分叉时刻和位置几乎相同，裂纹带宽度随 $\Delta x$（即 $\ell$）减小而变窄，与 Fig. 12 一致。
- **起裂时刻**：本复现约为 8 μs（判据为裂尖越过预制裂纹尖端 0.5 mm）；Fig. 13 的速度曲线从约 7 μs 开始增大，论文为 5 μs。差别来自相场的弥散裂尖：需要的驱动能量比 PD 的键断裂判据更高。
- **裂尖速度**：分叉前约 1300 m/s，分叉后约 1600–2200 m/s，都低于 Rayleigh 波速 $c_R\approx3200$ m/s。裂尖位置在网格上是离散跳动的，所以速度曲线有锯齿；细网格的曲线更平滑。
- **应力更高时**：4 MPa 下分叉提前到约 10.5 μs，并发生二次分叉，与文献中高载荷下多次分叉的趋势一致。6 MPa 下除了分叉，加载边附近也出现损伤带；12 MPa 下中面和加载边出现层裂（上、下两个应力波在中面叠加后超过 AT1 强度）。这种现象来自相场模型的有限强度，和 PD 的结果不同，因此 Fig. 11 只画 2/4/6 MPa，12 MPa 的数据保留在 `results/sigma12MPa/` 供参考。
- 论文 Fig. 11 的三个应力水平在原文图中给出，正文未列出数值；这里选用 2/4/6 MPa，如需其他值可直接改 `--sigma`。
