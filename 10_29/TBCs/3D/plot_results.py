"""Post-processing of the 3D TBC runs (Figs. 23-24 analogues of 10_29/8.md) -> results/*.png, summary.json.
The half model (0 <= x <= L/2) is mirrored to the full length L for plotting."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CASES = [1, 2, 3]
WIDTHS = ["50", "100"]


def load(name):
    d = os.path.join(RES, name)
    if not os.path.isfile(os.path.join(d, "run_info.json")):
        return None
    return dict(info=json.load(open(os.path.join(d, "run_info.json"))),
                snap=np.load(os.path.join(d, "snapshots.npz")), hist=np.load(os.path.join(d, "history.npz")))


def mirror(F):
    """(nz+1, ny+1, nx+1) half-model grid -> full length (last axis)."""
    return np.concatenate([F, F[..., -2::-1]], axis=-1)


def snap(run, key, t):
    s = run["snap"]
    i = int(np.argmin(abs(s["t"] - t)))
    return mirror(s[key][i]), float(s["t"][i])


def axes_xy(run, F):
    i = run["info"]
    x = np.linspace(0, i["L_um"], F.shape[-1])
    y = np.linspace(0, i["hBC_um"] + i["hTC_um"], F.shape[1])
    z = np.linspace(0, i["w_um"], F.shape[0])
    return x, y, z


def j_int(run):
    i = run["info"]
    return int(round((i["hBC_um"] + 0.5 * i["band_um"]) / i["h_um"]))


def name(case, w):
    return f"case{case}_w{w}_Gi0.001_hTC100"


def fig23(runs, t, tag):
    keys = [(c, w) for w in WIDTHS for c in CASES if name(c, w) in runs]
    if not keys:
        return
    fig, axs = plt.subplots(len(keys), 2, figsize=(15, 1.5 * len(keys)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1, 1]))
    for k, (c, w) in enumerate(keys):
        r = runs[name(c, w)]
        D, tt = snap(r, "d", t)
        x, y, z = axes_xy(r, D)
        m = axs[k, 0].pcolormesh(x, y, D[0], cmap="jet", vmin=0, vmax=1, shading="gouraud", rasterized=True)
        axs[k, 0].axhline(r["info"]["hBC_um"], color="w", lw=0.5, ls="--")
        axs[k, 0].set_aspect(3); axs[k, 0].set_xticks([])
        axs[k, 0].set_title(f"case {c}, w = {w} um: front face z = 0 (t = {tt*1e3:g} ms)", fontsize=8)
        axs[k, 1].pcolormesh(x, z, D[:, j_int(r), :], cmap="jet", vmin=0, vmax=1, shading="gouraud", rasterized=True)
        axs[k, 1].set_aspect(3); axs[k, 1].set_xticks([])
        axs[k, 1].set_title(f"case {c}, w = {w} um: interface plane (x-z)", fontsize=8)
    fig.colorbar(m, ax=axs, shrink=0.5, label="damage d")
    fig.savefig(os.path.join(RES, f"fig23_cases_damage_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"fig23_cases_damage_{tag}.png")


def fig23_3d(runs, t, tag):
    keys = [(c, w) for w in WIDTHS for c in CASES if name(c, w) in runs]
    if not keys:
        return
    fig = plt.figure(figsize=(15, 3.2 * len(WIDTHS)))
    for k, (c, w) in enumerate(keys):
        r = runs[name(c, w)]
        D, tt = snap(r, "d", t)
        x, y, z = axes_xy(r, D)
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        sel = D > 0.9
        ax = fig.add_subplot(len(WIDTHS), 3, WIDTHS.index(w) * 3 + c, projection="3d")
        ax.scatter(X[sel], Z[sel], Y[sel], c=np.where(Y[sel] < r["info"]["hBC_um"] + r["info"]["band_um"], "r", "b"),
                   s=0.3, alpha=0.5, rasterized=True)
        ax.set_xlim(0, x[-1]); ax.set_ylim(0, z[-1]); ax.set_zlim(0, y[-1])
        ax.set_box_aspect((6, 1.2, 1.5))
        ax.set_xlabel("x [um]", fontsize=7); ax.set_ylabel("z", fontsize=7); ax.set_zlabel("y", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_title(f"case {c}, w = {w} um, t = {tt*1e3:g} ms\n(d > 0.9; red: interface, blue: TC cracks)", fontsize=8)
    fig.savefig(os.path.join(RES, f"fig23_cases_3d_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"fig23_cases_3d_{tag}.png")


def fig24(run, tag):
    ts = run["snap"]["t"]
    fig, axs = plt.subplots(len(ts), 2, figsize=(15, 1.1 * len(ts)), squeeze=False)
    for k, t in enumerate(ts):
        Th, tt = snap(run, "theta", t)
        D, _ = snap(run, "d", t)
        x, y, z = axes_xy(run, D)
        mt = axs[k, 0].pcolormesh(x, y, Th[0], cmap="coolwarm", vmin=10, vmax=1000, shading="gouraud", rasterized=True)
        md = axs[k, 1].pcolormesh(x, y, D[0], cmap="jet", vmin=0, vmax=1, shading="gouraud", rasterized=True)
        for ax in axs[k]:
            ax.set_aspect(3); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t = {tt*1e3:g} ms", fontsize=8)
    fig.colorbar(mt, ax=axs[:, 0], shrink=0.4, label="temperature [C]")
    fig.colorbar(md, ax=axs[:, 1], shrink=0.4, label="damage d")
    fig.savefig(os.path.join(RES, f"fig24_evolution_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"fig24_evolution_{tag}.png")


def first_time(t, y, thr):
    k = np.nonzero(y >= thr)[0]
    return float(t[k[0]] * 1e3) if len(k) else None


def main():
    runs = {os.path.basename(p): load(os.path.basename(p)) for p in glob.glob(os.path.join(RES, "case*"))}
    runs = {n: r for n, r in runs.items() if r is not None and len(r["snap"]["t"])}
    T = max(r["info"]["T_s"] for r in runs.values())
    for t, tag in [(0.4875e-3, "0.4875ms"), (2e-3, "2ms"), (T, "final")]:
        fig23(runs, t, tag)
        fig23_3d(runs, t, tag)
    for c in CASES:
        if name(c, "100") in runs:
            fig24(runs[name(c, "100")], f"case{c}_w100")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for n, r in sorted(runs.items()):
        h = r["hist"]
        axs[0].semilogx(h["t"][1:] * 1e3, h["interface_damaged_fraction"][1:], label=n)
        axs[1].semilogx(h["t"][1:] * 1e3, h["n_transverse"][1:], label=n)
    axs[0].set_xlabel("t [ms]"); axs[0].set_ylabel("damaged interface fraction (d > 0.9)")
    axs[1].set_xlabel("t [ms]"); axs[1].set_ylabel("transverse cracks (half model)")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "history.png"), dpi=150); plt.close(fig)
    print("wrote history.png")

    summary = {}
    for n, r in sorted(runs.items()):
        i, h = r["info"], r["hist"]
        t, fi = h["t"], h["interface_damaged_fraction"]
        summary[n] = dict(case=i["case"], w=i["w_um"], bottom=i["bottom"], h_um=i["h_um"], ell_um=i["ell_um"],
                          t_first_transverse_ms=first_time(t, h["n_transverse"], 1),
                          n_transverse_half_final=int(h["n_transverse"][-1]),
                          t_interface_10pct_ms=first_time(t, fi, 0.1), t_interface_50pct_ms=first_time(t, fi, 0.5),
                          t_interface_90pct_ms=first_time(t, fi, 0.9), interface_fraction_final=float(fi[-1]),
                          max_d_BC_beyond_2ell=float(h["max_d_BC"][-1]), wallclock_s=i["wallclock_s"])
        print(n, summary[n])
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
