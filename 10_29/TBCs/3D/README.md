# TBCs 三维算例（论文 3.4.4 节）

## 文件

| 文件 | 说明 |
|---|---|
| `tbc_3d.py` | 主程序：半长模型，以 $x=L/2$ 为对称面，$\nu=0.25$，CG + GAMG（带刚体近零空间） |
| `run_all.sh` | case 1/2/3 × $w=50/100$ μm，$G_i=0.001$ N/mm，$h_{TC}=100$ μm，$T=10$ ms |
| `plot_results.py` | 后处理：`fig23_cases_*`（正面、界面平面、三维裂纹点云，半模型镜像为全长）、`fig24_evolution_*`、`history.png`、`summary.json` |

热冲击面：

- case 1：上表面
- case 2：上表面 + 左右端面
- case 3：再加上前后面

初温 1000 °C，冲击温度 10 °C。$\Delta x=10$ μm，$\ell=20$ μm，单元数为 $125\times20\times10=25$k（$w=100$）。

```bash
./run_all.sh            # 单个工况约 5–18 s/步，耗时较长
/opt/fenicsx/bin/python tbc_3d.py --case 3 --w 50 --T 0.01 --outdir results/case3_w50_Gi0.001_hTC100
/opt/fenicsx/bin/python plot_results.py
```

## 结果

正式计算尚未完成，本目录目前只有代码。

冒烟测试（滚支，case 3，$w=50$）中，界面在 0.2 ms 内完全损伤，与论文 case 3 在 0.4875 ms 时界面即被破坏的描述一致。但其中部分原因是粗网格和较大的 $\ell$。
