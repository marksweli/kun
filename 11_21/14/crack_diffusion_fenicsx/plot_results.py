"""Plot Figs. 12-15 analogues from tbc_fatigue.py output.

usage: python plot_results.py results [results_ideal]
"""
import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def load_hist(d):
    with open(os.path.join(d, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    out = {k: np.array([float(r[k]) if k != "phase" else 0 for r in rows]) for k in rows[0] if k != "phase"}
    out["phase"] = np.array([r["phase"] for r in rows])
    return out


def per_cycle(h, key, phase):
    m = h["phase"] == phase
    return h["cycle"][m], h[key][m]


res = sys.argv[1] if len(sys.argv) > 1 else "results"
ideal = sys.argv[2] if len(sys.argv) > 2 else None
fig_dir = os.path.join(res, "figs")
os.makedirs(fig_dir, exist_ok=True)
H = load_hist(res)
HI = load_hist(ideal) if ideal and os.path.exists(os.path.join(ideal, "history.csv")) else None

# Fig. 12: TGO thickness at peak / valley
fig, ax = plt.subplots(figsize=(6, 4.2))
for hh, ls, lab in ((H, "-", "crack-affected diffusion"), (HI, "--", "ideal crack ($k_d=0$)")):
    if hh is None:
        continue
    n, tp = per_cycle(hh, "tgo_peak_um", "cool")
    _, tv = per_cycle(hh, "tgo_valley_um", "cool")
    ax.plot(n, tp, "r" + ls, label=f"peak, {lab}")
    ax.plot(n, tv, "b" + ls, label=f"valley, {lab}")
ax.set_xlabel("Cycles"); ax.set_ylabel(r"TGO thickness ($\mu$m)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig12_tgo_thickness.png"), dpi=150)

# Fig. 14: crack length and fracture energy
fig, ax = plt.subplots(figsize=(6, 4.2))
n, L = per_cycle(H, "crack_len_um", "cool")
_, E = per_cycle(H, "frac_energy_Jpm", "cool")
ax.plot(n, L, "k-", label="crack length")
ax.set_xlabel("Cycles"); ax.set_ylabel(r"Crack length ($\mu$m)")
ax2 = ax.twinx(); ax2.plot(n, E, "r-", label="fracture energy"); ax2.set_ylabel("Fracture energy (J/m)", color="r")
if HI is not None:
    ni, Li = per_cycle(HI, "crack_len_um", "cool")
    ax.plot(ni, Li, "k--", label="crack length, ideal crack")
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig14_crack_energy.png"), dpi=150)

# Fig. 15: diffusion energy (all steps -> cyclic)
fig, ax = plt.subplots(figsize=(6, 4.2))
tcyc = H["t_h"] / 2.0          # one cycle = 7200 s = 2 h
ax.plot(tcyc, H["diff_energy_Jpm"], "b-", lw=0.6)
ax.set_xlabel("Cycles"); ax.set_ylabel("Diffusion energy in BC (J/m)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig15_diffusion_energy.png"), dpi=150)

# Fig. 13: c, xi, d snapshots
z = np.load(os.path.join(res, "snapshots.npz"))
xy = z["xy"] * 1e6
tri = mtri.Triangulation(xy[:, 0], xy[:, 1], z["tri"])
cyc = [int(c) for c in z["cycles"]]
A = float(os.environ.get("AMP", 15.0))
y_int = 3100.0 - A + A * np.cos(2 * np.pi * xy[:, 0] / 100.0)
fig, axs = plt.subplots(3, len(cyc), figsize=(2.3 * len(cyc), 8.5), squeeze=False)
for j, c in enumerate(cyc):
    for i, (key, lab, cm) in enumerate((("c", r"$c$ (mol/m$^3$)", "jet"), ("xi", r"$\xi$", "jet"), ("d", r"$d$", "jet"))):
        a = axs[i, j]
        v = z[f"{key}_{c}"].copy()
        if key == "xi":   # show oxide only (TC carries xi=1 by construction)
            v[xy[:, 1] > y_int + 1e-3] = 0.0
        vmax = 2.8 if key == "c" else 1.0
        im = a.tripcolor(tri, v, cmap=cm, shading="gouraud", vmin=0, vmax=vmax)
        a.set_xlim(0, 100); a.set_ylim(3040, 3140); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(f"N = {c}")
        if j == 0:
            a.set_ylabel(lab)
    for i in range(3):
        fig.colorbar(axs[i, -1].collections[0], ax=axs[i, :].tolist(), shrink=0.8, pad=0.01)
fig.savefig(os.path.join(fig_dir, "fig13_snapshots.png"), dpi=150, bbox_inches="tight")
print("figures written to", fig_dir)
