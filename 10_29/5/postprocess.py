# -*- coding: utf-8 -*-
"""
后处理：读取 results/tri 与 results/trap 的结果，生成与论文 Fig.5-10、14-17 对应的图与 summary.json
用法：python postprocess.py [--res results] [--cycle 10]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import LinearNDInterpolator

L = 21.43
A = 6.88
H0 = 0.7
TC_ = 0.2
KW = 2 * np.pi / L
TAG_TC, TAG_COH, TAG_TGO = 5, 4, 3
CASE_NAME = {"tri": "Triangular (hold 1 s)", "trap": "Trapezoidal (hold 2400 s)"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def f_if(x):
    return -A * np.cos(KW * x)


def offset_curve(x, h):
    fp = A * KW * np.sin(KW * x)
    n = np.sqrt(1 + fp ** 2)
    return x - h * fp / n, f_if(x) + h / n


def fold(x):
    """利用对称性把 x 折叠到 [0, L/2]（0=valley, L/2=peak）"""
    x = np.mod(x, L)
    return np.minimum(x, L - x)


def classify(xf):
    r = xf / L
    if r < 0.06:
        return "valley"
    if r < 0.19:
        return "off-valley"
    if r < 0.31:
        return "mid-flank"
    if r < 0.44:
        return "off-peak"
    return "peak"


def load(res, case, cyc):
    return dict(np.load(os.path.join(res, case, f"snap_cycle{cyc:02d}.npz")))


def tc_triangulation(s, ywin=(-A - 2.5, A + 7.0)):
    m = (s["mat"] == TAG_TC) & (s["xq"][:, 1] > ywin[0]) & (s["xq"][:, 1] < ywin[1])
    x, y = s["xq"][m, 0], s["xq"][m, 1]
    tri = mtri.Triangulation(x, y)
    cx, cy = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
    xs = np.linspace(0, L, 2001)
    ox, oy = offset_curve(xs, H0 + TC_)
    below = cy < np.interp(cx, ox, oy) + 0.02
    ex = x[tri.triangles]
    ey = y[tri.triangles]
    edge = np.max(np.hypot(ex - np.roll(ex, 1, 1), ey - np.roll(ey, 1, 1)), 1)
    tri.set_mask(below | (edge > 1.2))
    return tri, m


def draw_layers(ax):
    xs = np.linspace(0, L, 400)
    ax.fill_between(xs, -A - 3, f_if(xs), color="0.85", zorder=0)
    ox, oy = offset_curve(xs, H0)
    ax.fill(np.r_[xs, ox[::-1]], np.r_[f_if(xs), oy[::-1]], color="0.55", zorder=0)
    for h, c in ((0, "k"), (H0, "k"), (H0 + TC_, "k")):
        px, py = offset_curve(xs, h)
        ax.plot(px, py, c, lw=0.5)
    ax.text(0.5, -A - 2.0, "BC", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(0, L)
    ax.set_ylim(-A - 2.5, A + 7.0)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")


def principal(sxx, syy, sxy):
    c = 0.5 * (sxx + syy)
    r = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    s1 = c + r
    th = 0.5 * np.arctan2(2 * sxy, sxx - syy)  # σ1 方向
    return s1, th


def trace_crack(s, ds=0.05, frac=0.35, smax_len=15.0):
    """最大主应力准则裂纹路径：在 TC 近界面区 σ1 最大处萌生，沿垂直于 σ1 的方向双向扩展"""
    m = (s["mat"] == TAG_TC) & (s["dist_if"] < H0 + TC_ + 12.0)
    xy = s["xq"][m]
    sig = s["sig"][m]
    itp = LinearNDInterpolator(xy, np.stack([sig[:, 0], sig[:, 1], sig[:, 3]], 1))
    s1, _ = principal(sig[:, 0], sig[:, 1], sig[:, 3])
    near = s["dist_if"][m] < H0 + TC_ + 1.5
    i0 = np.argmax(np.where(near, s1, -np.inf))
    p0 = xy[i0].copy()
    s1_0 = s1[i0]
    xs = np.linspace(0, L, 2001)
    ox, oy = offset_curve(xs, H0 + TC_)
    paths = []
    for sgn in (1.0, -1.0):
        p = p0.copy()
        prev = None
        path = [p.copy()]
        for _ in range(int(smax_len / ds)):
            v = itp(p[0], p[1])
            if np.any(np.isnan(v)):
                break
            v = np.ravel(v)
            sv1, th = principal(v[0], v[1], v[2])
            if sv1 < frac * s1_0:
                break
            t = np.array([-np.sin(th), np.cos(th)])  # 裂纹面方向 ⟂ σ1
            if prev is None:
                t = t if t[0] * sgn >= 0 else -t
            elif np.dot(t, prev) < 0:
                t = -t
            q = p + ds * t
            if q[0] < 0 or q[0] > L or q[1] < np.interp(q[0], ox, oy):
                break
            p, prev = q, t
            path.append(p.copy())
        paths.append(np.array(path))
    full = np.vstack([paths[1][::-1], paths[0][1:]])
    return p0, s1_0, full


def interface_profile(s, comp, off=0.3, n=301):
    xs = np.linspace(0, L / 2, n)
    px, py = offset_curve(xs, H0 + TC_ + off)
    m = (s["mat"] == TAG_TC) & (s["dist_if"] < H0 + TC_ + 3.0)
    itp = LinearNDInterpolator(s["xq"][m], s["sig"][m, comp])
    v = itp(px, py)
    # 对称：叠加右半支取平均
    v2 = LinearNDInterpolator(s["xq"][m], s["sig"][m, comp])(L - px, py)
    sgn = -1.0 if comp == 3 else 1.0   # s12 关于 peak 反对称
    return xs, 0.5 * (v + sgn * v2)


def coh_profile(s, key):
    m = s["mat"] == TAG_COH
    xf = fold(s["xproj"][m])
    order = np.argsort(xf)
    xb = np.linspace(0, L / 2, 120)
    idx = np.clip(np.digitize(xf[order], xb) - 1, 0, len(xb) - 2)
    vals = s[key][m][order]
    out = np.array([vals[idx == k].max() if np.any(idx == k) else np.nan for k in range(len(xb) - 1)])
    return 0.5 * (xb[1:] + xb[:-1]), out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--res", default=os.path.join(here, "results"))
    ap.add_argument("--cycle", type=int, default=10)
    ap.add_argument("--fig", default=os.path.join(here, "figures"))
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    cases = [c for c in ("tri", "trap") if os.path.isdir(os.path.join(a.res, c))]
    summary = {}

    # ---------------- Fig.5/6 应力云图 (s22, s12) + Fig.10 裂纹路径 ----------------
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 4.6 * len(cases)), squeeze=False)
    for i, case in enumerate(cases):
        ncyc = max(int(p[-6:-4]) for p in glob.glob(os.path.join(a.res, case, "snap_cycle*.npz")))
        cyc = min(a.cycle, ncyc)
        s = load(a.res, case, cyc)
        tri, m = tc_triangulation(s)
        sig = s["sig"][m]
        for j, (comp, name) in enumerate(((1, "s22"), (3, "s12"))):
            ax = axes[i, j]
            v = sig[:, comp]
            lim = np.percentile(np.abs(v), 99.8)
            cs = ax.tricontourf(tri, v, levels=np.linspace(-lim, lim, 41), cmap="jet", extend="both")
            plt.colorbar(cs, ax=ax, shrink=0.8, label=f"{name} (MPa)")
            draw_layers(ax)
            # 最大值位置
            mm = (s["mat"] == TAG_TC) & (s["dist_if"] < H0 + TC_ + 2.0)
            k = np.argmax(s["sig"][mm, comp] if comp == 1 else np.abs(s["sig"][mm, comp]))
            pk = s["xq"][mm][k]
            ax.plot(*pk, "w*", ms=12, mec="k")
            ax.set_title(f"{CASE_NAME[case]}: {name}, end of cooling, cycle {cyc}", fontsize=8)
        # 裂纹路径
        ax = axes[i, 2]
        s1, _ = principal(sig[:, 0], sig[:, 1], sig[:, 3])
        lim = np.percentile(np.abs(s1), 99.8)
        cs = ax.tricontourf(tri, s1, levels=np.linspace(-lim, lim, 41), cmap="jet", extend="both")
        plt.colorbar(cs, ax=ax, shrink=0.8, label="σ1 (MPa)")
        draw_layers(ax)
        p0, s10, path = trace_crack(s)
        ax.plot(path[:, 0], path[:, 1], "m-", lw=2.5, label="predicted TC crack (max σ1 criterion)")
        ax.plot(*p0, "w*", ms=12, mec="k", label=f"initiation, σ1={s10:.0f} MPa")
        mc = s["mat"] == TAG_COH
        ini = mc & (s["D"] >= 0.999) & (s["r"] < 0.999) & (case != "tri")
        dmg = mc & (s["r"] >= 0.999)
        if ini.any():
            ax.plot(s["xq"][ini, 0], s["xq"][ini, 1], ".", color="orange", ms=2, label="interface damaged (D=1, softening)")
        if dmg.any():
            ax.plot(s["xq"][dmg, 0], s["xq"][dmg, 1], ".", color="w", ms=2.5, label="interface crack (fully failed)")
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
        ax.set_title(f"{CASE_NAME[case]}: crack prediction, cycle {cyc}", fontsize=8)

        # 统计
        mm = (s["mat"] == TAG_TC) & (s["dist_if"] < H0 + TC_ + 2.0)
        k = np.argmax(s["sig"][mm, 1])
        pk = s["xq"][mm][k]
        xpk = fold(s["xproj"][mm][k])
        xs, prof = interface_profile(s, 1)
        kk = np.nanargmax(prof)
        dirv = path[-1] - path[0]
        ang = np.degrees(np.arctan2(abs(dirv[1]), abs(dirv[0])))
        mc = s["mat"] == TAG_COH
        summary[case] = dict(
            cycle=cyc,
            h_tgo_um=float(s["h_tgo"]),
            s22_max_TC_near_interface_MPa=float(s["sig"][mm, 1].max()),
            s22_max_point_xy_um=[float(pk[0]), float(pk[1])],
            s22_max_x_over_L_folded=float(xpk / L),
            s22_max_location=classify(xpk),
            s22_profile_max_x_over_L=float(xs[kk] / L),
            s22_profile_max_location=classify(xs[kk]),
            s22_profile_max_MPa=float(prof[kk]),
            crack_initiation_xy_um=[float(p0[0]), float(p0[1])],
            crack_initiation_x_over_L_folded=float(fold(p0[0]) / L),
            crack_initiation_location=classify(fold(p0[0])),
            crack_path_length_um=float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))),
            crack_path_mean_angle_to_x_deg=float(ang),
            interface_D_eq4_max=float(s["D"][mc].max()),
            interface_cracked_fraction_r_eq_1=float(np.mean(s["r"][mc] >= 0.999)),
            interface_initiated_fraction_D_eq_1=float(np.mean(s["D"][mc] >= 0.999)),
        )
    fig.tight_layout()
    fig.savefig(os.path.join(a.fig, "fig_stress_fields_and_crack.png"))
    plt.close(fig)

    # ---------------- s22 / s12 沿界面分布（TC 侧，距 cohesive 层 0.3 μm） ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for case, ls in zip(cases, ("-", "--")):
        ncyc = summary[case]["cycle"]
        for cyc, col in zip(sorted({1, max(1, ncyc // 2), ncyc}), ("C0", "C1", "C3")):
            s = load(a.res, case, cyc)
            for ax, comp, name in ((axes[0], 1, "s22"), (axes[1], 3, "s12")):
                xs, prof = interface_profile(s, comp)
                ax.plot(xs / L, prof, ls, color=col, label=f"{case}, cycle {cyc}")
    for ax, name in zip(axes, ("s22", "s12")):
        for xv, lab in ((0, "valley"), (0.125, "off-valley"), (0.25, "mid"), (0.375, "off-peak"), (0.5, "peak")):
            ax.axvline(xv, color="0.7", lw=0.6)
            ax.text(xv, ax.get_ylim()[1] if False else 0, lab, rotation=90, fontsize=7, va="bottom", ha="right")
        ax.set_xlabel("x / L   (0 = valley, 0.5 = peak)")
        ax.set_ylabel(f"{name} in TC near TC/TGO interface (MPa)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Stress along the TC/TGO interface (TC side), end of cooling")
    fig.tight_layout()
    fig.savefig(os.path.join(a.fig, "fig_interface_profiles.png"))
    plt.close(fig)

    # ---------------- Fig.7 / 14 / 16：监测点应力演化 ----------------
    for case in cases:
        h = dict(np.load(os.path.join(a.res, case, "history.npz")))
        t = h["t"] / 3600.0
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for key, col in zip(("peak", "off-peak", "mid-flank", "off-valley"), ("C3", "C1", "C2", "C0")):
            axes[0].plot(t, h[f"{key}:s22"], color=col, lw=0.9, label=key)
            axes[1].plot(t, h[f"{key}:s12"], color=col, lw=0.9, label=key)
        axes[2].plot(t, h["T"], "k", lw=0.8)
        ax2 = axes[2].twinx()
        ax2.plot(t, h["h_tgo"], "r", lw=1.2)
        ax2.set_ylabel("TGO thickness (μm)", color="r")
        axes[0].set_ylabel("s22 (MPa)")
        axes[1].set_ylabel("s12 (MPa)")
        axes[2].set_ylabel("T (°C)")
        axes[2].set_xlabel("time (h)")
        for ax in axes[:2]:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=4)
        axes[0].set_title(f"{CASE_NAME[case]}: stress evolution in TC at peak / off-peak / off-valley")
        fig.tight_layout()
        fig.savefig(os.path.join(a.fig, f"fig_history_{case}.png"))
        plt.close(fig)

    # 每循环冷却结束时的监测点应力
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for case, ls in zip(cases, ("--", "-")):
        h = dict(np.load(os.path.join(a.res, case, "history.npz")))
        ph = h["phase"]
        end = [i for i in range(len(ph)) if ph[i] == "cool" and (i + 1 == len(ph) or ph[i + 1] != "cool")]
        cyc = h["cycle"][end]
        for key, col in zip(("peak", "off-peak", "off-valley"), ("C3", "C1", "C0")):
            axes[0].plot(cyc, h[f"{key}:s22"][end], ls, marker="o", ms=3, color=col, label=f"{case}: {key}")
            axes[1].plot(cyc, h[f"{key}:s12"][end], ls, marker="o", ms=3, color=col, label=f"{case}: {key}")
    for ax, n in zip(axes, ("s22", "s12")):
        ax.set_xlabel("cycle")
        ax.set_ylabel(f"{n} at end of cooling (MPa)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.fig, "fig_cycle_evolution.png"))
    plt.close(fig)

    # ---------------- Fig.8/9/17：界面损伤 ----------------
    if "trap" in cases:
        ncyc = summary["trap"]["cycle"]
        cyc_list = sorted({1, min(4, ncyc), ncyc})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for cyc, col in zip(cyc_list, ("C0", "C1", "C3")):
            s = load(a.res, "trap", cyc)
            xb, Dv = coh_profile(s, "D")
            _, dv = coh_profile(s, "r")
            axes[0].plot(xb / L, Dv, color=col, label=f"cycle {cyc}")
            axes[1].plot(xb / L, dv, color=col, label=f"cycle {cyc}")
        for ax, n in zip(axes[:2], ("initiation index D, Eq.(4)", "traction-loss ratio r (1 = crack)")):
            ax.set_xlabel("x / L   (0 = valley, 0.5 = peak)")
            ax.set_ylabel(n)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            ax.set_ylim(-0.02, 1.05)
        s = load(a.res, "trap", ncyc)
        mc = s["mat"] == TAG_COH
        ax = axes[2]
        draw_layers(ax)
        sc = ax.scatter(s["xq"][mc, 0], s["xq"][mc, 1], c=s["r"][mc], s=3, cmap="jet", vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="r (1 = crack)")
        ax.set_title(f"TC/TGO interface damage, cycle {ncyc}")
        fig.tight_layout()
        fig.savefig(os.path.join(a.fig, "fig_interface_damage_trap.png"))
        plt.close(fig)

        h = dict(np.load(os.path.join(a.res, "trap", "history.npz")))
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for key, col in zip(("peak", "off-peak", "mid-flank", "off-valley", "valley"), ("C3", "C1", "C2", "C0", "C4")):
            ax.plot(h["t"] / 3600, h[f"{key}:r"], color=col, label=f"r @ {key}")
            ax.plot(h["t"] / 3600, h[f"{key}:D"], ":", color=col)
        ax.set_xlabel("time (h)")
        ax.set_ylabel("D (dotted, Eq.4) / r (solid)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
        ax.set_title("Trapezoidal: interface damage evolution at monitoring points")
        fig.tight_layout()
        fig.savefig(os.path.join(a.fig, "fig_damage_history_trap.png"))
        plt.close(fig)

    # ---------------- 变体对比（results 下所有子目录） ----------------
    variants = sorted(d for d in os.listdir(a.res) if glob.glob(os.path.join(a.res, d, "snap_cycle*.npz")))
    if len(variants) > 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for vname in variants:
            ncyc = max(int(p[-6:-4]) for p in glob.glob(os.path.join(a.res, vname, "snap_cycle*.npz")))
            sv = load(a.res, vname, ncyc)
            for ax, comp in zip(axes, (1, 3)):
                xs, prof = interface_profile(sv, comp)
                ax.plot(xs / L, prof, label=f"{vname} (cycle {ncyc}, h_TGO={float(sv['h_tgo']):.2f} μm)")
            xs, prof = interface_profile(sv, 1)
            kk = np.nanargmax(prof)
            summary.setdefault("variants", {})[vname] = dict(
                cycle=ncyc, h_tgo_um=float(sv["h_tgo"]), s22_profile_max_x_over_L=float(xs[kk] / L),
                s22_profile_max_location=classify(xs[kk]), s22_profile_max_MPa=float(prof[kk]))
        for ax, n in zip(axes, ("s22", "s12")):
            for xv in (0, 0.125, 0.25, 0.375, 0.5):
                ax.axvline(xv, color="0.7", lw=0.6)
            ax.set_xlabel("x / L   (0 = valley, 0.125 = off-valley, 0.375 = off-peak, 0.5 = peak)")
            ax.set_ylabel(f"{n} in TC near interface (MPa)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(os.path.join(a.fig, "fig_variants_profiles.png"))
        plt.close(fig)

    with open(os.path.join(a.fig, "summary.json"), "w") as fo:
        json.dump(summary, fo, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
