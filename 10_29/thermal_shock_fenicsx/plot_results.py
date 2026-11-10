"""Post-processing for the ceramic quenching runs (Figs. 15-17 of 10_29/8.md)."""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
TEMPS = (300, 400, 500, 600)


def load(T0):
    p = os.path.join(R, f"theta0_{T0}C")
    if not os.path.exists(os.path.join(p, "history.npz")):
        return None
    return (np.load(os.path.join(p, "snapshots.npz")), np.load(os.path.join(p, "history.npz")),
            json.load(open(os.path.join(p, "run_info.json"))))


def full_plate(q):
    """Mirror the lower-left quarter (25 x 5 mm) to the full 50 x 10 mm plate."""
    lower = np.hstack([q, q[:, -2::-1]])
    return np.vstack([lower, lower[-2::-1, :]])


def count_cracks(line, thr=0.9):
    """Number of separate d > thr segments along a line."""
    m = (line > thr).astype(int)
    return int(np.sum(np.diff(np.r_[0, m]) == 1))


def crack_stats(dq, h):
    """Count cracks from the bottom shock surface at several depths (hierarchy)."""
    out = {}
    for depth_mm in (0.5, 1.0, 2.0, 3.0, 4.0):
        j = int(round(depth_mm * 1e-3 / h))
        out[f"bottom_edge_cracks_deeper_than_{depth_mm}mm"] = count_cracks(dq[j, int(round(1.5e-3 / h)):])
    return out


summary = {}
runs = {T0: load(T0) for T0 in TEMPS}
runs = {k: v for k, v in runs.items() if v}

# ---- Fig. 15 analogue: energies ----
if runs:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for T0, (sn, hs, info) in runs.items():
        axs[0].plot(hs["t"] * 1e3, hs["E_el"], label=f"θ0 = {T0} °C")
        axs[1].plot(hs["t"] * 1e3, hs["E_frac"], label=f"θ0 = {T0} °C")
    axs[0].plot([], [], "k:", label="kinetic energy ≡ 0 (quasi-static)")
    for ax, tl in zip(axs, ("strain energy (J/m)", "fracture (dissipated) energy (J/m)")):
        ax.set_xscale("log")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(tl)
        ax.legend(fontsize=8)
    fig.suptitle("energy evolution of the quarter model, per unit thickness (cf. Fig. 15)")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig15_energy.png"), dpi=130)
    plt.close(fig)

# ---- Fig. 16 analogue: temperature and damage evolution, 600 C ----
if 600 in runs:
    sn, hs, info = runs[600]
    n = len(sn["t"])
    fig, axs = plt.subplots(n, 2, figsize=(12, 1.55 * n + 0.5))
    for i in range(n):
        im = axs[i, 0].imshow(sn["theta"][i], origin="lower", extent=[0, 25, 0, 5], cmap="jet",
                              vmin=20, vmax=600)
        axs[i, 0].set_title(f"temperature (°C), t = {sn['t'][i]*1e3:g} ms", fontsize=9)
        fig.colorbar(im, ax=axs[i, 0], fraction=0.015)
        im = axs[i, 1].imshow(sn["d"][i], origin="lower", extent=[0, 25, 0, 5], cmap="jet", vmin=0, vmax=1)
        axs[i, 1].set_title(f"damage d, t = {sn['t'][i]*1e3:g} ms", fontsize=9)
        fig.colorbar(im, ax=axs[i, 1], fraction=0.015)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig16_evolution_600C.png"), dpi=130)
    plt.close(fig)

# ---- Fig. 17 analogue: final crack patterns (full plate by symmetry) ----
if runs:
    fig, axs = plt.subplots(len(runs), 1, figsize=(10, 2.3 * len(runs)))
    axs = np.atleast_1d(axs)
    for ax, (T0, (sn, hs, info)) in zip(axs, runs.items()):
        ax.imshow(full_plate(sn["d"][-1]), origin="lower", extent=[0, 50, 0, 10], cmap="gray_r",
                  vmin=0, vmax=1)
        ax.set_title(f"θ0 = {T0} °C → 20 °C, t = {sn['t'][-1]:g} s", fontsize=10)
        st = crack_stats(sn["d"][-1], info["h_mm"] * 1e-3)
        summary[f"theta0_{T0}C"] = {**st,
                                    "max_strain_energy_J_per_m": float(hs["E_el"].max()),
                                    "t_at_max_strain_energy_ms": float(hs["t"][hs["E_el"].argmax()] * 1e3),
                                    "final_fracture_energy_J_per_m": float(hs["E_frac"][-1]),
                                    "steps": info["steps"], "max_staggered_iter": int(hs["n_iter"].max()),
                                    "wallclock_s": round(info["wallclock_s"])}
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig17_crack_patterns.png"), dpi=130)
    plt.close(fig)

json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
