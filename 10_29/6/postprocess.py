"""
后处理: 读取 results/ 中的历程与快照, 生成论文风格图 (Fig. 4, 5/9, 7, 10, 11 对应图)
运行: python postprocess.py
"""
import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import mesh_tbc as MT

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.sans-serif": ["DejaVu Sans"]})

md = np.load(os.path.join(RES, "mesh_data.npz"))
X, CELLS, TAG0 = md["x"], md["cells"], md["tag0"]
tri = mtri.Triangulation(X[:, 0], X[:, 1], CELLS)
YC = MT.Y_C
xs = np.linspace(0, MT.W, 400)


def interfaces(ax, h_tgo, **kw):
    ax.plot(xs, MT.y_int(xs), "k-", lw=0.6, **kw)
    ax.plot(xs, MT.y_int(xs) - h_tgo, "k--", lw=0.6, **kw)


# ---------------- 1. 网格 / 几何 (Fig. 5, Fig. 9) ----------------
fig, axs = plt.subplots(1, 3, figsize=(13, 5.5), gridspec_kw=dict(width_ratios=[0.6, 1.2, 1.4]))
colors = {1: "#9ecae1", 2: "#fc9272", 3: "#a1d99b", 4: "#fdd0a2"}
for ax, lim in zip(axs, [(0, 12, 5080, 5200), (0, 12, 5118, 5142), (1.6, 3.6, 5133.7, 5135.4)]):
    for m, c in colors.items():
        sel = TAG0 == m
        ax.tripcolor(mtri.Triangulation(X[:, 0], X[:, 1], CELLS[sel]), np.ones(sel.sum()),
                     cmap=matplotlib.colors.ListedColormap([c]), shading="flat")
    ax.triplot(tri, lw=0.15, color="k")
    ax.set_xlim(lim[:2]); ax.set_ylim(lim[2:]); ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
axs[0].set_ylabel("y [µm]")
axs[0].set_title("TC / TGO / BC")
axs[1].set_title("TGO growth band (0.2 µm) + crack zone")
axs[2].set_title(f"off-peak pre-crack 0.5×0.06 µm (x={MT.X_OFF:.1f} µm)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_mesh_geometry.png"), dpi=200); plt.close(fig)

# ---------------- 2. 稳态热梯度 (Fig. 4) ----------------
tp = np.loadtxt(os.path.join(RES, "steady_temperature_profile.csv"), delimiter=",", skiprows=1)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(tp[:, 0], tp[:, 1]); axs[0].set_xlabel("y [µm]"); axs[0].set_ylabel("T [K]")
axs[0].set_title("Steady thermal gradient (x = 11.9 µm)")
sel = tp[:, 0] > 4900
axs[1].plot(tp[sel, 0], tp[sel, 1])
for yv, lb in [(5000, "SUB/BC"), (5125, "BC/TGO"), (5380, "TC top")]:
    axs[1].axvline(yv, color="gray", ls=":"); axs[1].text(yv, tp[sel, 1].min(), lb, rotation=90, va="bottom")
axs[1].set_xlabel("y [µm]"); axs[1].set_title("zoom: coating")
for a in axs:
    a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_temperature_gradient.png"), dpi=200); plt.close(fig)

# ---------------- 3. 历程 ----------------
import csv
rows = list(csv.DictReader(open(os.path.join(RES, "history_caseA.csv"))))
H = {k: np.array([float(r[k]) if k != "phase" else 0 for r in rows]) for k in rows[0] if k != "phase"}
th = H["t"] / 3600.0
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axs[0].plot(th, H["T_offpeak_int"], lw=0.8); axs[0].set_ylabel("T at off-peak interface [K]")
ax2 = axs[0].twinx(); ax2.plot(th, H["h_tgo_um"], "r", lw=1.2); ax2.set_ylabel("TGO thickness [µm]", color="r")
axs[1].plot(th, H["crack_len_d09_um"], lw=1.2, label="crack length (d ≥ 0.9)")
axs[1].plot(th, MT.CRACK_L + H["crack_ext_d05_um"], lw=1.0, ls="--", label="crack length (d ≥ 0.5)")
axs[1].set_ylabel("crack length [µm]"); axs[1].legend()
axs[2].plot(th, H["E_TC_int_GPa"], lw=0.8); axs[2].set_ylabel("E of TC near off-peak [GPa]\n(T- and sintering-dependent)")
axs[2].set_xlabel("time [h] (= cycles)")
for a in axs:
    a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_history_T_TGO_crack.png"), dpi=200); plt.close(fig)

fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for k, c in [("peak", "C0"), ("off_peak", "C3"), ("mid", "C2"), ("off_valley", "C1")]:
    axs[0].plot(th, H[f"sig1_{k}"], c, lw=0.8, label=k)
    axs[1].plot(th, H[f"sig22_{k}"], c, lw=0.8, label=k)
axs[0].set_ylabel(r"$\sigma_1$ [MPa] (0.1 µm above interface)")
axs[1].set_ylabel(r"$\sigma_{22}$ [MPa]"); axs[1].set_xlabel("time [h]")
for a in axs:
    a.grid(alpha=0.3); a.legend(ncol=4)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_stress_history_sampling_points.png"), dpi=200); plt.close(fig)

# 每个循环结束 (室温) 的裂纹长度表
cyc_end = []
for c in range(1, int(H["cycle"].max()) + 1):
    i = np.where(H["cycle"] == c)[0][-1]
    cyc_end.append((c, H["t"][i], H["h_tgo_um"][i], H["crack_len_d09_um"][i], H["crack_angle_deg"][i],
                    H["sig1_max_TC"][i]))
np.savetxt(os.path.join(RES, "crack_length_per_cycle.csv"), np.array(cyc_end), delimiter=",",
           header="cycle,t_s,h_tgo_um,crack_length_d09_um,crack_angle_deg,sig1_max_TC_MPa", comments="", fmt="%.5g")
fig, ax = plt.subplots(figsize=(6, 4))
ce = np.array(cyc_end)
ax.plot(ce[:, 0], ce[:, 3], "o-"); ax.set_xlabel("cycle"); ax.set_ylabel("crack length at end of cycle [µm]")
ax2 = ax.twinx(); ax2.plot(ce[:, 0], ce[:, 2], "s--r", ms=3); ax2.set_ylabel("TGO thickness [µm]", color="r")
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_crack_length_per_cycle.png"), dpi=200)
plt.close(fig)

# ---------------- 4. 相场 / 损伤第一主应力云图 (Fig. 10, 11) ----------------
snaps = sorted(glob.glob(os.path.join(RES, "snapshots", "*.npz")))


def pick(names):
    out = []
    for n in names:
        f = os.path.join(RES, "snapshots", n + ".npz")
        if os.path.exists(f):
            out.append((n, np.load(f)))
    return out


ncyc = int(H["cycle"].max())
wanted = ["c01_end_heating", "c01_cooled", "c01_end", "c02_end", "c03_end", "c05_end", "c07_end",
          "c10_end", "c15_end", "c20_end", "c25_end", "c30_end"]
sel_snaps = pick([w for w in wanted if int(w[1:3]) <= ncyc])


def label(n, s):
    return f"{n.replace('_', ' ')}\nt = {float(s['t']):.0f} s, h_TGO = {float(s['h_tgo']):.2f} µm"


def panel_plot(field, fname, cmap, vmin, vmax, lim, nodal, title):
    n = len(sel_snaps)
    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.3 * nr), squeeze=False)
    for ax, (nm, s) in zip(axs.flat, sel_snaps):
        v = s[field]
        if nodal:
            im = ax.tripcolor(tri, v, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
        else:
            im = ax.tripcolor(tri, facecolors=v, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        interfaces(ax, float(s["h_tgo"]))
        ax.set_xlim(lim[:2]); ax.set_ylim(lim[2:]); ax.set_aspect("equal")
        ax.set_title(label(nm, s), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(im, ax=axs, shrink=0.6)
    fig.savefig(os.path.join(FIG, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


crack_win = (1.5, 5.0, YC - 1.2, YC + 1.8)
panel_plot("d", "fig10_phase_field_crack_zoom.png", "jet", 0, 1, crack_win, True,
           "Case A: crack phase field d (off-peak, AR 10/24)")
panel_plot("d", "fig10b_phase_field_interface.png", "jet", 0, 1, (0, 12, YC - 14, YC + 6), True,
           "Case A: crack phase field d (half period)")
panel_plot("sig1", "fig11_damaged_sigma1.png", "jet", -100, 300, (0, 12, YC - 14, YC + 6), False,
           r"Case A: damaged first principal stress $g(d)\,\sigma_1$ [MPa]")
panel_plot("sig1", "fig11b_damaged_sigma1_zoom.png", "jet", -100, 400, crack_win, False,
           r"Case A: damaged first principal stress $g(d)\,\sigma_1$ [MPa] (crack zone)")
panel_plot("sig22", "fig_sigma22.png", "coolwarm", -300, 300, (0, 12, YC - 14, YC + 6), False,
           r"Case A: radial stress $\sigma_{22}$ [MPa]")
panel_plot("T", "fig_temperature_snapshots.png", "inferno", 290, 1330, (0, 12, YC - 14, YC + 6), True,
           "Temperature [K]")

info = json.load(open(os.path.join(RES, "run_info.json")))
summary = dict(run_info=info, crack_length_per_cycle=[dict(cycle=int(c[0]), h_tgo_um=round(c[2], 3),
                                                          crack_length_um=round(c[3], 3),
                                                          angle_deg=None if np.isnan(c[4]) else round(c[4], 1))
                                                     for c in cyc_end])
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps(summary["crack_length_per_cycle"], indent=0))
