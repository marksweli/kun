"""绘制 tbc_oxidation.py 的结果 (对应论文 Fig.11-13, Fig.15)."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})

p = argparse.ArgumentParser()
p.add_argument("--outdir", default="results")
a = p.parse_args()
d = a.outdir
info = json.load(open(os.path.join(d, "run_info.json")))
h = np.load(os.path.join(d, "history.npz"))
s = np.load(os.path.join(d, "snapshots.npz"))
A = info["amp"]
Y_IF = 3200.0
tag = s["cell_tag"]
xy = s["xy"]
T = mtri.Triangulation(xy[:, 0], xy[:, 1] - Y_IF, s["tri"])   # y 以平均界面为 0
tc_tri = tag == 1
title = f"TC porosity {info['porosity']*100:.0f}%, A = {A:g} um, 1373.15 K"


def first_cross(key, val):
    i = np.where(h[key] >= val)[0]
    return float(h["t_h"][i[0]]) if len(i) else None


# ---------- 1) 相场-时间曲线 (Fig.15a) ----------
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(h["t_h"], h["phi_max"], "b-", label="max phase field at TC/TGO interface")
ax.plot(h["t_h"], h["phi_valley"], "r--", label="phase field at valley (x = 30 um)")
ax.axhline(0.4, color="k", ls="--", lw=0.8)
ax.set_xlim(0, info["hours"]); ax.set_ylim(0, 1.0)
ax.set_xlabel("Time / h"); ax.set_ylabel("Phase field (c)")
ax.set_title(title, fontsize=9); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(d, "fig_phase_vs_time.png"), dpi=200); plt.close(fig)

# ---------- 2) TGO 厚度 (Fig.11/12) ----------
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(h["t_h"], h["h_peak"], "r-", label="peak (x = 0)")
ax.plot(h["t_h"], h["h_valley"], "b-", label="valley (x = 30 um)")
ax.plot(h["t_h"], h["h_mean"], "k--", label="mean (TGO area / interface length)")
ax.plot([200], [5.0], "ks", mfc="none", label="exp. mean, 7%, 200 h (paper)")
ax.set_xlabel("Time / h"); ax.set_ylabel("TGO thickness / um")
ax.set_xlim(0, info["hours"]); ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(d, "fig_tgo_thickness.png"), dpi=200); plt.close(fig)

# ---------- 3) 裂纹萌生/扩展/失效: 相场 + S22 (Fig.15b) ----------
labels = list(s["label"])
pick = []
names = {"phi0.4": "damage c=0.4", "initiation": "initiation / propagation (c=1)", "failure": "failure (crack 40 um)"}
for lb in ("phi0.4", "initiation", "failure"):
    if lb in labels:
        pick.append((names[lb], labels.index(lb)))
if len(pick) < 3:  # 若未达到事件, 用末态补齐
    pick.append(("final", len(labels) - 1))
pick = pick[:3]
ylim = (-A - 15, A + 30)
fig, axs = plt.subplots(2, len(pick), figsize=(3.4 * len(pick), 6.2), squeeze=False)
for j, (lb, k) in enumerate(pick):
    ph = np.where(tc_tri, s["phi"][k][s["tri"]].mean(axis=1), np.nan)
    m0 = axs[0, j].tripcolor(T, np.ma.masked_invalid(ph), cmap="jet", vmin=0, vmax=1)
    axs[0, j].set_title(f"{lb}\nt = {s['t'][k]:.0f} h", fontsize=9)
    S = np.where(tc_tri, s["S22"][k] / 1e6, np.nan)
    m1 = axs[1, j].tripcolor(T, np.ma.masked_invalid(S), cmap="jet", vmin=-400, vmax=300)
    for ax in axs[:, j]:
        ax.set_aspect("equal"); ax.set_ylim(*ylim); ax.set_xlim(0, 60)
        ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(m0, ax=axs[0, :].tolist(), label="Phase field", shrink=0.8)
fig.colorbar(m1, ax=axs[1, :].tolist(), label="S22 in TC / MPa", shrink=0.8)
fig.suptitle(title, fontsize=9)
fig.savefig(os.path.join(d, "fig_crack_S22.png"), dpi=200); plt.close(fig)

# ---------- 4) TGO 形貌/氧浓度/S22 (Fig.13) ----------
times = [t for t in (200.0, 1000.0, 2000.0) if np.any(np.isclose(s["t"], t))]
fig, axs = plt.subplots(len(times), 4, figsize=(14, 3.0 * len(times)), squeeze=False)
for i, t in enumerate(times):
    k = int(np.where(np.isclose(s["t"], t) & (s["label"] == "periodic"))[0][0])
    m = axs[i, 0].tripcolor(T, s["n"][k], shading="gouraud", cmap="jet", vmin=0, vmax=1)
    fig.colorbar(m, ax=axs[i, 0]); axs[i, 0].set_title(f"TGO fraction n, {t:.0f} h", fontsize=9)
    m = axs[i, 1].tripcolor(T, s["c"][k], shading="gouraud", cmap="jet")
    fig.colorbar(m, ax=axs[i, 1]); axs[i, 1].set_title("O2 concentration (mol/m3)", fontsize=9)
    S = s["S22"][k] / 1e6
    m = axs[i, 2].tripcolor(T, S, cmap="jet", vmin=np.percentile(S, 1), vmax=np.percentile(S, 99))
    fig.colorbar(m, ax=axs[i, 2]); axs[i, 2].set_title("S22 all layers (MPa)", fontsize=9)
    ph = np.where(tc_tri, s["phi"][k][s["tri"]].mean(axis=1), np.nan)
    m = axs[i, 3].tripcolor(T, np.ma.masked_invalid(ph), cmap="jet", vmin=0, vmax=1)
    fig.colorbar(m, ax=axs[i, 3]); axs[i, 3].set_title("Phase field in TC", fontsize=9)
    for ax in axs[i]:
        ax.set_aspect("equal"); ax.set_ylim(-A - 25, A + 25); ax.set_xlim(0, 60)
        ax.set_xlabel("x / um"); ax.set_ylabel("y - y_if / um")
fig.tight_layout(); fig.savefig(os.path.join(d, "fig_fields.png"), dpi=160); plt.close(fig)

# ---------- 汇总 ----------
summary = {
    "t_phi_max_0.4_h": first_cross("phi_max", 0.4),
    "t_phi_max_0.95_h(initiation)": first_cross("phi_max", 0.95),
    "t_phi_valley_0.4_h": first_cross("phi_valley", 0.4),
    "t_crack_15um_h": first_cross("crack_len", 15.0),
    "t_crack_40um_h": first_cross("crack_len", 40.0),
}
for t in (200.0, 1000.0, 2000.0):
    i = int(np.argmin(np.abs(h["t_h"] - t)))
    summary[f"TGO_at_{t:.0f}h_um(peak/valley/mean)"] = [round(float(h[k][i]), 2) for k in ("h_peak", "h_valley", "h_mean")]
i = None if summary["t_phi_max_0.4_h"] is None else int(np.where(h["phi_max"] >= 0.4)[0][0])
if i is not None:
    summary["TGO_valley_at_phi0.4_um"] = round(float(h["h_valley"][i]), 2)
    summary["phi_max_location_at_0.4(x_um, y_above_if_um)"] = [round(float(h["x_phimax"][i]), 2), round(float(h["y_phimax"][i]), 2)]
summary["final_crack_len_um"] = float(h["crack_len"][-1])
json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
