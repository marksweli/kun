# 14.md 4.1.2 节“裂纹影响扩散过程”FEniCSx 复现（代码，进行中）

热-化-力耦合疲劳相场（PF-CZM）模型：氧扩散 + TGO 生长 + 疲劳断裂，裂纹处扩散系数放大
$D = D_0 e^{-Q_d/RT}\,(1 + k_d\langle d-d_{th}\rangle)$。

- `tbc_fatigue.py`：求解器（gmsh 网格、热传导、O₂ 扩散/氧化、广义平面应变力学（P2 位移）、Wu PF-CZM 疲劳相场交错求解）。
- `plot_results.py`：生成 Fig. 12–15 对应图（TGO 厚度、c/ξ/d 云图、裂纹长度与断裂能、扩散能）。

## 运行

```bash
export HDF5_USE_FILE_LOCKING=FALSE
python tbc_fatigue.py --k 2 --cycles 800 --outdir results          # 裂纹影响扩散
python tbc_fatigue.py --k 2 --kd 0 --cycles 800 --outdir results_ideal  # 理想裂纹对照
python plot_results.py results results_ideal
```

环境：dolfinx 0.9 + python-gmsh（conda-forge，需同时指定 `fenics-dolfinx=0.9`，否则 dolfinx 会被升级）。

## 主要假设

- 边界温度取热流体与冷流体入口平均值 $T_h=(1573+1000)/2=1286.5$ K，施加于 TC 顶面和基体底面；循环为 900 s 升温、5400 s 保温、900 s 降温。
- 界面为余弦形，振幅 $A=15\,\mu$m，宽 100 µm；初始 TGO 厚 1 µm。
- $Q_d$ 的标定条件：$D(T_h)(1+0.2k_d)=D_{air}=2\times10^{-5}$ m²/s。$\mu^0$ 的标定条件：$c_b(T_h)=2.8$ mol/m³。
- 氧耗 $\mathcal R=1.5\cdot3950/0.10196$ mol/m³；反应速率 $M=D/(\mathcal R\ell^2)$，其中 $\ell=1\,\mu$m。
- 假设 TGO 在 $T_h$ 时无应力；Eq. 38 中的应力项未计入。
- 疲劳历史量下限取 $f\bar Y_0$；取 $k=2$。

## 状态

- 试算（k=2）中，TGO 增长约 0.025 µm/周，论文约 0.03 µm/周。
- 约第 110 周沿界面侧翼起裂分层，与论文一致。
- 800 周的完整计算尚未完成，因此结果未提交。
