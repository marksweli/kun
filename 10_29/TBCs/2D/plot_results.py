"""Post-processing of the 2D TBC runs (Figs. 20-22 analogues of 10_29/8.md) -> results/*.png, summary.json."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
STRETCH = 3.0  # vertical exaggeration of the thin coating in the field plots


def load(name):
    d = os.path.join(RES, name)
    if not os.path.isfile(os.path.join(d, "run_info.json")):
        return None
    return dict(info=json.load(open(os.path.join(d, "run_info.json"))),
                snap=np.load(os.path.join(d, "snapshots.npz")), hist=np.load(os.path.join(d, "history.npz")))


def field_ax(ax, run, key, t, title=None, cmap="jet", vmin=0, vmax=1):
    s = run["snap"]
    i = int(np.argmin(abs(s["t"] - t)))
    F = s[key][i]
    L, H = float(s["L"]) * 1e6, float(s["hBC"] + s["hTC"]) * 1e6
    x, y = np.linspace(0, L, F.shape[1]), np.linspace(0, H, F.shape[0])
    m = ax.pcolormesh(x, y, F, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)
    ax.axhline(float(s["hBC"]) * 1e6, color="w", lw=0.5, ls="--")
    ax.set_aspect(STRETCH)
    ax.set_xticks([]); ax.set_yticks([0, H])
    ax.set_title(title or f"t = {s['t'][i]*1e3:g} ms", fontsize=9)
    return m


def finish(fig, m, label, fn):
    fig.colorbar(m, ax=fig.axes, shrink=0.6, label=label)
    fig.savefig(os.path.join(RES, fn), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fn)


def first_time(t, y, thr):
    k = np.nonzero(y >= thr)[0]
    return float(t[k[0]] * 1e3) if len(k) else None


def summarize(run):
    i, h, s = run["info"], run["hist"], run["snap"]
    ell = i["ell_um"]
    d = s["d"][-1]
    ny = d.shape[0] - 1
    yy = np.linspace(0, i["hBC_um"] + i["hTC_um"], ny + 1)
    bc_rows = yy < i["hBC_um"] - 2 * ell
    t, fi = h["t"], h["interface_damaged_fraction"]
    return dict(mode=i["mode"], Gi=i["Gi_N_per_mm"], hTC=i["hTC_um"], bottom=i.get("bottom", "roller"),
                t_first_transverse_ms=first_time(t, h["n_transverse"], 1),
                n_transverse_final=int(h["n_transverse"][-1]),
                t_interface_10pct_ms=first_time(t, fi, 0.1), t_interface_50pct_ms=first_time(t, fi, 0.5),
                t_interface_90pct_ms=first_time(t, fi, 0.9), interface_fraction_final=float(fi[-1]),
                max_d_BC_beyond_2ell=float(d[bc_rows].max()), wallclock_s=i["wallclock_s"])


def main():
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RES, "*_Gi*_hTC*")))
    runs = {n: load(n) for n in names}
    runs = {n: r for n, r in runs.items() if r is not None}
    roller = {"roller/" + os.path.basename(p): load("roller/" + os.path.basename(p))
              for p in glob.glob(os.path.join(RES, "roller", "*_Gi*"))}
    roller = {n: r for n, r in roller.items() if r is not None}

    q, ht = runs.get("quench_Gi0.001_hTC100"), runs.get("heat_Gi0.001_hTC100")
    # Fig. 20 analogue: damage evolution, quench (1, 2, 3, 28 ms) vs heating (8, 10, 12, 20 ms)
    if q and ht:
        fig, axs = plt.subplots(4, 2, figsize=(14, 6))
        for k, (tq, th) in enumerate(zip([1e-3, 2e-3, 3e-3, 28e-3], [8e-3, 10e-3, 12e-3, 20e-3])):
            m = field_ax(axs[k, 0], q, "d", tq, f"quench t = {tq*1e3:g} ms")
            field_ax(axs[k, 1], ht, "d", th, f"heating t = {th*1e3:g} ms")
        finish(fig, m, "damage d", "fig20_quench_vs_heat_damage.png")
        fig, axs = plt.subplots(4, 2, figsize=(14, 6))
        for k, (tq, th) in enumerate(zip([1e-3, 2e-3, 3e-3, 28e-3], [8e-3, 10e-3, 12e-3, 20e-3])):
            m = field_ax(axs[k, 0], q, "theta", tq, f"quench t = {tq*1e3:g} ms", cmap="coolwarm", vmin=10, vmax=1000)
            field_ax(axs[k, 1], ht, "theta", th, f"heating t = {th*1e3:g} ms", cmap="coolwarm", vmin=10, vmax=1000)
        finish(fig, m, "temperature [C]", "fig20_quench_vs_heat_temperature.png")
    for name, r in [("quench", q), ("heat", ht)]:
        if r:
            ts = r["snap"]["t"]
            fig, axs = plt.subplots(len(ts), 1, figsize=(10, 1.1 * len(ts)))
            for ax, t in zip(axs, ts):
                m = field_ax(ax, r, "d", t)
            finish(fig, m, "damage d", f"evolution_{name}_Gi0.001_hTC100.png")

    # Fig. 21 analogue: final damage vs interface toughness
    gis = [g for g in ["0.001", "0.002", "0.004", "0.008"] if f"quench_Gi{g}_hTC100" in runs]
    if gis:
        for tt, tag in [(28e-3, "28ms"), (0.3, "final")]:
            fig, axs = plt.subplots(len(gis), 1, figsize=(10, 1.4 * len(gis)), squeeze=False)
            for ax, g in zip(axs[:, 0], gis):
                m = field_ax(ax, runs[f"quench_Gi{g}_hTC100"], "d", tt, f"$G_c^{{TC/BC}}$ = {g} N/mm, t = {tt*1e3:g} ms")
            finish(fig, m, "damage d", f"fig21_damage_vs_Gi_{tag}.png")

    # Fig. 22 analogue: final damage vs TC thickness
    hs = [hh for hh in ["100", "300", "500"] if f"quench_Gi0.001_hTC{hh}" in runs]
    if hs:
        for tt, tag in [(28e-3, "28ms"), (0.3, "final")]:
            fig, axs = plt.subplots(len(hs), 1, figsize=(10, 2.5 * len(hs)), squeeze=False)
            for ax, hh in zip(axs[:, 0], hs):
                m = field_ax(ax, runs[f"quench_Gi0.001_hTC{hh}"], "d", tt, f"$h_{{TC}}$ = {hh} um, t = {tt*1e3:g} ms")
            finish(fig, m, "damage d", f"fig22_damage_vs_hTC_{tag}.png")

    # histories
    allr = {**runs, **roller}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for n, r in allr.items():
        h = r["hist"]
        ls = "--" if n.startswith("roller") else "-"
        axs[0].semilogx(h["t"][1:] * 1e3, h["interface_damaged_fraction"][1:], ls, label=n)
        axs[1].semilogx(h["t"][1:] * 1e3, h["n_transverse"][1:], ls, label=n)
    axs[0].set_xlabel("t [ms]"); axs[0].set_ylabel("damaged interface fraction (d > 0.9)")
    axs[1].set_xlabel("t [ms]"); axs[1].set_ylabel("number of transverse cracks")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "history.png"), dpi=150); plt.close(fig)
    print("wrote history.png")

    summary = {n: summarize(r) for n, r in allr.items()}
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    for n, s in summary.items():
        print(n, s)


if __name__ == "__main__":
    main()
