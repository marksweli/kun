# Case A 复现（FEniCSx 0.11）：TBC 界面 off-peak 预制裂纹的相场扩展模拟

参考论文：`10_29/6.md`（APS-TBC 多物理场分层失效，Sect. 3–4，Table 1–7）。
耦合：**off-peak 水平预制裂纹 + 相场损伤 + TGO 连续生长（BC→TGO 材料切换）+ 厚度方向热梯度 + TC 烧结/蠕变松弛**。

## 文件

| 文件 | 说明 |
|---|---|
| `mesh_tbc.py` | Gmsh 几何/网格（半周期单元、正弦界面、预制裂纹、加密区） |
| `tbc_phasefield_caseA.py` | 主求解器（热传导 → TGO 生长/烧结 → 力学-蠕变-相场交错迭代） |
| `postprocess.py` | 后处理与绘图 |
| `tbc_caseA.msh` | 生成的网格（30 449 个 P1 三角形） |
| `run_caseA.log` | 30 个循环的完整运行日志 |
| `results/history_caseA.csv` | 时间历程：界面温度、δ、TGO 厚度、裂纹长度/角度、采样点 σ1/σ22 |
| `results/crack_length_per_cycle.csv`, `summary.json`, `run_info.json` | 每个循环的裂纹长度、热分析标定量 |
| `results/fields_caseA.xdmf/.h5` | 每个循环加热结束/循环结束时的场（T, u, d, σ1, σ22, 材料），可用 ParaView 打开 |
| `results/snapshots/*.npz` | 各循环快照（绘图用） |
| `results/figures/*.png` | 网格、热梯度、历程、相场（对应 Fig. 10）、损伤主应力（对应 Fig. 11）、σ22 |

运行：
```bash
micromamba create -n fenicsx -c conda-forge python=3.12 fenics-dolfinx mpich petsc4py python-gmsh matplotlib scipy
python tbc_phasefield_caseA.py 30     # 2 核 CPU 约 15 分钟
python postprocess.py
```

## 模型

**几何/边界（Fig. 2、Fig. 9）**：宽 $W=12\,\mu m$（半波长，$\lambda=24\,\mu m$），界面 $y_I(x)=5130+5\cos(2\pi x/\lambda)$（峰谷差 $10\,\mu m$，AR=10/24）；
TC 250 µm / BC 130 µm（含初始 TGO 1 µm）/ 基体 5 mm。左侧 $x=0$（波峰）对称 $u_x=0$；底部滚动支座 $u_y=0$；顶面气压 3.5 MPa；右侧规定位移

$$\delta=\frac{W}{H}\int_0^{H}\alpha(T)\,(T-T_{ref})\,dy\quad(\text{Eq. 7 的连续形式，随热梯度变化})$$

面外应变取 $\varepsilon_{zz}=\delta/W$（广义平面应变近似）。
**预制裂纹**：off-peak 位于距波峰 $W/6=2\,\mu m$ 处；水平、长 0.5 µm、宽 0.06 µm、方形裂尖，左端与 TC/TGO 界面相交，仅从 TC 中切除。裂纹扩展区网格 $h=0.03\,\mu m=l_{int}/2$，TGO 生长带网格 0.2 µm。

**热传导**：$\rho c_p\dot T=\nabla\cdot(k\nabla T)$，顶面 $q=h_{in}(T_{gas}-T)$，底面 $q=h_{cool}(T_{cool}-T)$，侧面绝热。$h_{in}$ 由 Eq. 10（湍流，$Re=1.84\times10^7$）算得 $16\,453\ W/m^2K$，$h_{cool}=420$，$T_{gas}=1323\,K$，$T_{cool}=923\,K$。
得到 TC 表面 **1314 K**、TC/TGO 界面 **1275 K**（论文为 1313 K / 约 1271 K）。
每个循环 3600 s：升温 300 s → 保温至 3100 s（对流换热）→ 300 s 等温线性降温至 293 K → 室温保持 200 s。

**TGO 生长**（Eq. 4）：$h=K A_0 t_{eq}^{m}$，$m=0.332$，等效氧化时间按 Arrhenius（$Q_0=76.7$ kJ/mol）随界面温度累积；按论文的加速设定取 $K$，使 30 个循环后 $h=6\,\mu m$（$K\approx4.3$）。BC 单元形心到界面的竖直距离 $\le h$ 时切换为 TGO（导热和力学性能同时切换），并在切换时刻令该单元无应力（即蠕变消除切换应力）。

**蠕变**（Table 3）：J2-Norton $\dot{\boldsymbol\varepsilon}^{cr}=\tfrac32A\sigma_{vm}^{n-1}\mathbf s$，在 1123 K 以上激活；BC 的 $A(T)$ 由 1273/1423 K 两点按 Arrhenius 插值；采用隐式（向后 Euler）径向返回 $r+3G\Delta tA(r\sigma^{tr}_{vm})^{n-1}r=1$。

**TC 烧结**（时间-温度等效）：$E_{TC}=E_{dep}(T)\,[1+(F-1)S]$，$\varepsilon^{sint}=-\varepsilon_s S$，$S=1-\exp(-\sqrt{t_{eq}/t_c})$。

**相场**（Miehe 2015 应力型，Table 6）：

$$\tau\dot d=(1-d)\mathcal H-(d-l^2\Delta d),\qquad \mathcal H=\max_t\ \zeta\Big\langle\sum_{i=1}^{2}\Big(\frac{\langle\tilde\sigma_i\rangle}{\sigma_c}\Big)^2-1\Big\rangle$$

$\sigma_c=300$ MPa，$\zeta=0.2$，$l=0.06\,\mu m$，$\tau=1$ s，$\boldsymbol\sigma=[(1-d)^2+k]\tilde{\boldsymbol\sigma}$。损伤驱动力只施加在 TC 中。
力学/蠕变/相场在每个时间步内交错迭代至收敛，并在单步 $\Delta d>0.2$ 时自动减小步长。

## 结果（30 个循环）

| 循环 | 1 | 2 | 3 | 7 | 30 |
|---|---|---|---|---|---|
| TGO 厚度 [µm] | 1.94 | 2.44 | 2.79 | 3.70 | 6.00 |
| 裂纹长度（d≥0.9）[µm] | 2.35 | 2.65 | 2.65 | 2.65 | 2.65 |

- 裂纹从方形裂尖向**右上方**（约 27°）扩展，扩展只发生在**冷却阶段**；之后在第 2 个循环继续扩展，从第 3 个循环起止裂。止裂原因是 TGO 增厚后，最大第一主应力沿界面从波峰向 mid-level 迁移（见 `fig11_damaged_sigma1.png`、`fig_stress_history_sampling_points.png`）。这与论文 Case A 的结论一致（论文：0.58→1.48→1.97 µm，第 7 个循环止裂）。
- 后期 mid-level 处的 σ1 随循环增加（达到约 180 MPa），反映了 TC 烧结硬化的影响，这也与论文结论一致。
- 与论文的差异：本模型第 1 个循环的扩展量偏大，最终长度约 2.65 µm（论文约 2 µm）；另外在裂纹口处，沿 TC/TGO 界面朝波峰方向还出现了一条短的界面分支（界面分层）。

## 假设（论文未给出或无法直接对应的部分）

1. Table 4 的列顺序与物理量明显颠倒（例如密度 6000 kg/m³ 应属于 YSZ），因此按 TC/TGO/BC/基体的顺序反向使用。
2. 热循环的波形（Fig. 3 未给出）：升温 300 s，降温 300 s，然后室温保持 200 s；气压随加热过程施加，冷却时卸载。
3. 广义平面应变近似为 $\varepsilon_{zz}=\delta/W$；采用 P1 位移单元（论文为二次四边形单元）。
4. 烧结参数（论文未给出数值）：$F=2$，$\varepsilon_s=0.2\%$，$Q_s=200$ kJ/mol，$t_c=20$ h（1373 K）。TGO/TC 的蠕变系数在 1123 K 以上取常数。
5. 相场驱动力只取面内主应力，且只在 TC 中施加；损伤对整个应力张量进行退化。
6. 文中所称“损伤第一主应力”在本代码中定义为 $g(d)\,\sigma_1$；同时也输出了 $\tilde\sigma_1\cdot d$（XDMF 中的 `sigma1_x_d`）。
