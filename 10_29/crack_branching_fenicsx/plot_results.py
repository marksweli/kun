"""Post-processing for the dynamic crack branching runs (Figs. 10-13 of 10_29/8.md)."""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
EXT = [0, 100, 0, 40]


def load(name):
    p = os.path.join(R, name)
    if not os.path.exists(os.path.join(p, "history.npz")):
        return None
    return (np.load(os.path.join(p, "snapshots.npz")), np.load(os.path.join(p, "history.npz")),
            json.load(open(os.path.join(p, "run_info.json"))))


def tip_metrics(hst, info):
    t, x, y = hst["t"], hst["tip_x"], hst["tip_y"]
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    # speed: central difference of the crack path length over a 2 us window
    w = max(1, int(round(1.0e-6 / (t[1] - t[0]))))
    v = np.full_like(t, np.nan)
    v[w:-w] = (s[2 * w:] - s[:-2 * w]) / (t[2 * w:] - t[:-2 * w])
    a0, yc = 0.05, 0.02
    init = t[np.argmax(x > a0 + 0.5e-3)] if np.any(x > a0 + 0.5e-3) else None
    br = t[np.argmax(y > yc + 1e-3)] if np.any(y > yc + 1e-3) else None
    xb = x[np.argmax(y > yc + 1e-3)] if br is not None else None
    stats = {}
    if init is not None and br is not None:
        m1 = (t >= init) & (t <= br)
        m2 = (t > br) & (x < 0.099)
        stats["mean_speed_before_branching_m_s"] = round(float(np.nanmean(v[m1])), 1)
        stats["mean_speed_after_branching_m_s"] = round(float(np.nanmean(v[m2])), 1)
        stats["max_speed_2us_window_m_s"] = round(float(np.nanmax(v)), 1)
    return t, v, s, init, br, xb, stats


def field_plot(ax, f, cmap, title, vmin=None, vmax=None):
    im = ax.imshow(f, origin="lower", extent=EXT, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 20, 40])
    return im


summary = {}

# ---- Fig. 10 analogue: displacement and damage evolution, sigma0 = 2 MPa ----
r = load("sigma2MPa")
if r:
    sn, hst, info = r
    idx = [k for k in (1, 3, 5, 7, 9) if k < len(sn["t"])]
    fig, axs = plt.subplots(len(idx), 2, figsize=(10, 2.3 * len(idx)))
    for i, k in enumerate(idx):
        im = field_plot(axs[i, 0], sn["umag"][k] * 1e6, "jet", f"|u| (μm), t = {sn['t'][k]*1e6:.0f} μs")
        fig.colorbar(im, ax=axs[i, 0], fraction=0.02)
        im = field_plot(axs[i, 1], sn["d"][k], "jet", f"damage d, t = {sn['t'][k]*1e6:.0f} μs", 0, 1)
        fig.colorbar(im, ax=axs[i, 1], fraction=0.02)
    fig.suptitle("σ0 = 2 MPa, Δx = 0.25 mm (cf. Fig. 10)")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig10_evolution_sigma2MPa.png"), dpi=130)
    plt.close(fig)

# ---- Fig. 11 analogue: final damage for different applied stresses ----
cases = [(s, load(f"sigma{s}MPa")) for s in (2, 4, 6)]
cases = [(s, c) for s, c in cases if c]
if cases:
    fig, axs = plt.subplots(len(cases), 1, figsize=(6, 2.6 * len(cases)))
    axs = np.atleast_1d(axs)
    for ax, (s, (sn, hst, info)) in zip(axs, cases):
        field_plot(ax, sn["d"][-1], "jet", f"σ0 = {s} MPa, t = {sn['t'][-1]*1e6:.0f} μs", 0, 1)
        t, v, sp, init, br, xb, st = tip_metrics(hst, info)
        summary[f"sigma{s}MPa"] = {"initiation_us": None if init is None else round(init * 1e6, 2),
                                   "branching_us": None if br is None else round(br * 1e6, 2),
                                   "branching_x_mm": None if xb is None else round(xb * 1e3, 2),
                                   **st, "wallclock_s": round(info["wallclock_s"])}
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig11_damage_vs_stress.png"), dpi=130)
    plt.close(fig)

# ---- Fig. 12/13 analogue: mesh convergence ----
mesh = [(hh, load(f"mesh_h{hh}")) for hh in ("0.5", "0.25", "0.125")]
mesh = [(hh, c) for hh, c in mesh if c]
if mesh:
    fig, axs = plt.subplots(len(mesh), 2, figsize=(10, 2.3 * len(mesh)))
    axs = np.atleast_2d(axs)
    for i, (hh, (sn, hst, info)) in enumerate(mesh):
        field_plot(axs[i, 0], sn["d"][0], "jet", f"Δx = {hh} mm, t = {sn['t'][0]*1e6:.0f} μs", 0, 1)
        field_plot(axs[i, 1], sn["d"][-1], "jet", f"Δx = {hh} mm, t = {sn['t'][-1]*1e6:.0f} μs", 0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig12_mesh_damage.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for hh, (sn, hst, info) in mesh:
        t, v, sp, init, br, xb, st = tip_metrics(hst, info)
        ax.plot(t * 1e6, v, label=f"Δx = {hh} mm")
        summary[f"mesh_h{hh}"] = {"initiation_us": None if init is None else round(init * 1e6, 2),
                                  "branching_us": None if br is None else round(br * 1e6, 2),
                                  "branching_x_mm": None if xb is None else round(xb * 1e3, 2),
                                  **st,
                                  "n_cells": info["n_cells"], "dt_ns": info["dt_ns"],
                                  "steps": info["steps"], "wallclock_s": round(info["wallclock_s"]),
                                  "wallclock_per_step_s": round(info["wallclock_s"] / info["steps"], 4)}
    E, rho, nu = 72e9, 2235.0, 1.0 / 3.0
    cR = 0.9194 * np.sqrt(E / (2 * (1 + nu)) / rho)  # Rayleigh speed (plane stress, nu=1/3)
    ax.axhline(cR, ls="--", c="gray", label=f"Rayleigh speed {cR:.0f} m/s")
    ax.set_xlabel("t (μs)")
    ax.set_ylabel("crack tip speed (m/s)")
    ax.set_title("upper branch tip speed (cf. Fig. 13)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig13_tip_speed.png"), dpi=130)
    plt.close(fig)

json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
