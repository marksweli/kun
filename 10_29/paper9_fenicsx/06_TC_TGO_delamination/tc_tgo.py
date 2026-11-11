"""Paper Sec. 5.5: TC/TGO interface delamination.

Same one-period model as Sec. 5.4 (L = 30 um, A = 5 um, no initial TGO).
Stage 1: 800 h isothermal oxidation at 1150 C (oxygen diffusion + BC -> TGO
conversion + swelling strain alpha_n n), mechanics/phase field solved at
log-spaced instants.  Stage 2: cooling 1150 C -> 20 C in 30 min.
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dolfinx import io
import tbc_common as tbc

ap = argparse.ArgumentParser()
ap.add_argument("--damage", type=int, default=1)
ap.add_argument("--h", type=float, default=0.0002)
ap.add_argument("--dT", type=float, default=10.0)
ap.add_argument("--n_ox", type=int, default=400, help="oxidation (c, n) steps in 800 h")
ap.add_argument("--n_mech", type=int, default=40, help="mechanics solves during the isothermal stage")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.h, args.dT, args.n_ox, args.n_mech = 0.0008, 50.0, 120, 10

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results" + ("_quick" if args.quick else ""))
L, A = 0.030, 0.005
m = tbc.TBCModel(Lw=L, A=A, W=L, t_tgo=0.0, h_fine=args.h, lphi=0.002, l=0.0005, out=out,
                 refine=(-A - 0.010, A + 0.010), damage=bool(args.damage), oxidation=True,
                 h_coarse=0.02)
HOUR = 3600.0
t_iso = 800 * HOUR
# log-spaced oxidation instants (fine at the start where the TGO grows fastest)
t_ox = np.r_[0.0, np.geomspace(10.0, t_iso, args.n_ox)]
# mechanics instants: log-spaced up to 20 h, then uniform in time up to 800 h
t_mech = np.r_[np.geomspace(10.0, 20 * HOUR, max(args.n_mech // 4, 2)),
               np.linspace(20 * HOUR, t_iso, args.n_mech - max(args.n_mech // 4, 2) + 1)[1:]]
mech_idx = set(int(np.argmin(abs(t_ox - tm))) for tm in t_mech)
snap_h = [0, 267, 533, 800]
snap_idx = {int(np.argmin(abs(t_ox - s * HOUR))): s for s in snap_h}
ylim = (-0.012, 0.012)
xdmf = io.XDMFFile(m.msh.comm, os.path.join(out, "tc_tgo.xdmf"), "w")
xdmf.write_mesh(m.msh)
rows, ox_snaps, d_snaps, t0 = [], {}, {}, time.time()
tgo_rows = []   # TGO thickness at the valley after every oxidation step
peak_cells = None


def s1_peak():
    """max principal stress near the roughness peaks (x = 0 and x = L, y = A)"""
    global peak_cells
    if peak_cells is None:
        from dolfinx import mesh as dmesh
        mid = dmesh.compute_midpoints(m.msh, 2, np.arange(m.ncell, dtype=np.int32))
        r = np.minimum(np.hypot(mid[:, 0], mid[:, 1] - A), np.hypot(mid[:, 0] - L, mid[:, 1] - A))
        peak_cells = np.where(r < 0.003)[0]
    return float(m.s1.x.array[peak_cells].max())


def record(t, stage, it=0, err=0.0):
    m.update_stress()
    Ef = m.fracture_energy()
    rows.append([t / HOUR, m.T, Ef, float(m.d.x.array.max()), s1_peak(), float(m.n.x.array.max()),
                 m.tgo_thickness(L / 2, -A), stage])
    print(f"[{'iso' if stage == 0 else 'cool'}] t={t/HOUR:9.4f} h T={m.T:7.1f} dmax={rows[-1][3]:.3f} "
          f"Efrac={Ef:.3e} s1_peak={rows[-1][4]:.1f} TGO={rows[-1][6]*1e3:.3f} um it={it} err={err:.1e} "
          f"wall={time.time()-t0:.0f}s dmax[TC,TGO,BC,SUB]={m.dmax_layers()}", flush=True)


# ------------------------------------------------ stage 1: isothermal oxidation
m.update_T(1150.0)
for k in range(len(t_ox)):
    if k > 0:
        m.oxidise(t_ox[k] - t_ox[k - 1])
    tgo_rows.append([t_ox[k] / HOUR, m.tgo_thickness(L / 2, -A)])
    if k in snap_idx:
        ox_snaps[snap_idx[k]] = dict(c=m.c.x.array.copy(), n=m.n.x.array.copy())
    if k in mech_idx or k == len(t_ox) - 1:
        it, err = m.step()
        record(t_ox[k], 0, it, err)
        if k in snap_idx or k == len(t_ox) - 1:
            for f in (m.d, m.s1, m.n, m.c):
                xdmf.write_function(f, t_ox[k])
d_snaps["end of isothermal (800 h)"] = m.d.x.array.copy()

# ------------------------------------------------ stage 2: cooling in 30 min
t_cool, T0, T1 = 1800.0, 1150.0, 20.0
dt = args.dT / (T0 - T1) * t_cool
tc = np.r_[np.arange(dt, t_cool, dt), t_cool]
snap_T = {700.0: "cooling, T = 700 C", 400.0: "cooling, T = 400 C", 20.0: "end of cooling, T = 20 C"}
tp = 0.0
for t in tc:
    for j in range(4):
        m.update_T(T0 - (T0 - T1) * (tp + (j + 0.5) * (t - tp) / 4) / t_cool)
        m.oxidise((t - tp) / 4)
    m.update_T(T0 - (T0 - T1) * t / t_cool)
    it, err = m.step()
    record(t_iso + t, 1, it, err)
    for Ts, lab in snap_T.items():
        if abs(m.T - Ts) < 1e-6:
            d_snaps[lab] = m.d.x.array.copy()
            for f in (m.d, m.s1, m.n, m.c):
                xdmf.write_function(f, t_iso + t)
    tp = t
xdmf.close()
np.savetxt(os.path.join(out, "history.csv"), np.array(rows), delimiter=",",
           header="t_h,T_C,E_frac_Nmm_per_mm,d_max,s1max_peak_MPa,n_max,TGO_thickness_valley_mm,stage(0=iso,1=cool)",
           comments="")
np.savetxt(os.path.join(out, "tgo_thickness.csv"), np.array(tgo_rows), delimiter=",",
           header="t_h,TGO_thickness_valley_mm", comments="")

# ------------------------------------------------ figures
# Fig. 14: oxygen and TGO (n > 0.8)
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for j, s in enumerate(snap_h):
    m.c.x.array[:] = ox_snaps[s]["c"]; m.n.x.array[:] = ox_snaps[s]["n"]
    c1 = m.plot_nodal(ax[0, j], m.c, f"oxygen c (mol/m$^3$), {s} h", ylim, vmin=0, vmax=tbc.C_BAR)
    c2 = m.plot_nodal(ax[1, j], m.n, f"oxide fraction n, {s} h (contour n=0.8)", ylim, vmin=0, vmax=1)
    ax[1, j].tricontour(m.tri(), m.n.x.array, levels=[0.8], colors="w", linewidths=0.8)
fig.colorbar(c1, ax=ax[0, :], shrink=0.7); fig.colorbar(c2, ax=ax[1, :], shrink=0.7)
fig.suptitle("Fig. 14 reproduction: oxygen distribution and TGO growth")
fig.savefig(os.path.join(out, "fig14_oxidation.png"), dpi=160)

# Fig. 15: TGO thickness at the valley vs time + parabolic estimate
r = np.array(rows)
iso = r[:, 7] == 0
fig, ax = plt.subplots(figsize=(6, 4))
tg = np.array(tgo_rows)
ax.plot(tg[:, 0], tg[:, 1] * 1e3, "r-", label="this work (FEniCSx), n > 0.8 at valley")
tt = np.linspace(0, 800, 200)
ax.plot(tt, np.sqrt(2 * tbc.D_OX * tbc.C_BAR * tt * HOUR / tbc.M_OX) * 1e3, "k--",
        label=r"parabolic estimate $\sqrt{2D\bar c t/M}$")
ax.set_xlabel("time (h)"); ax.set_ylabel(r"TGO thickness ($\mu$m)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig15_tgo_thickness.png"), dpi=160)

# Fig. 16: damage (top: fixed scale 0..1, bottom: scaled to the max damage)
fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
dtop = max(float(dv.max()) for dv in d_snaps.values())
cb = [None, None]
for j, (lab, dv) in enumerate(d_snaps.items()):
    m.d.x.array[:] = dv
    for i, vmax in enumerate((1.0, dtop)):
        c = m.plot_nodal(ax[i, j], m.d, f"({'abcd'[j]}) d, {lab}" + ("" if i == 0 else f"\n(colour scale 0-{dtop:.3f})"),
                         ylim, vmin=0, vmax=vmax)
        cb[i] = c
        ax[i, j].tricontour(m.tri(), m.n.x.array, levels=[0.8], colors="w", linewidths=0.6, linestyles="--")
fig.colorbar(cb[0], ax=ax[0, :], shrink=0.8)
fig.colorbar(cb[1], ax=ax[1, :], shrink=0.8)
fig.suptitle("Fig. 16 reproduction: damage after oxidation and during cooling (dashed: n = 0.8)")
fig.savefig(os.path.join(out, "fig16_damage.png"), dpi=160)

# Fig. 17: max principal stress at the peak and fracture energy vs time
x = np.where(iso, r[:, 0], 800 + (r[:, 0] - 800) * 3600 / 60)   # cooling stage in minutes, shifted
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, k, lab in ((ax[0], 4, "max principal stress at peak (MPa)"), (ax[1], 2, "fracture energy (N mm/mm)")):
    a.plot(r[iso, 0], r[iso, k], "b.-", label="isothermal")
    a.set_xlabel("isothermal time (h)"); a.set_ylabel(lab)
    a2 = a.twiny(); a2.plot((r[~iso, 0] - 800) * 60, r[~iso, k], "r.-", label="cooling")
    a2.set_xlabel("cooling time (min)", color="r")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig17_stress_energy.png"), dpi=160)
json.dump(dict(case="5.5 TC/TGO delamination", damage=bool(args.damage), h=args.h, cells=m.ncell,
               n_ox=args.n_ox, n_mech=args.n_mech, dT=args.dT,
               tgo_thickness_800h_um=float(tg[-1, 1] * 1e3), final_dmax=rows[-1][3],
               final_Efrac=rows[-1][2], wall_s=time.time() - t0),
          open(os.path.join(out, "summary.json"), "w"), indent=2)
print("done", flush=True)
