"""Paper Sec. 5.4: top-coat cracking during cooling (one-period model).

L = 30 um, A = 5 um, initial TGO thickness 0 (no TGO subdomain), uniform cooling
1150 C -> 20 C in 30 min, oxygen diffusion / oxidation of the BC (Eqs. 45, 47)
active while T > 850 C, prescribed c = 1.55 mol/m^3 in the TC (TC/BC interface).
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
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.h, args.dT = 0.0008, 50.0

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results" + ("_quick" if args.quick else ""))
L, A = 0.030, 0.005
m = tbc.TBCModel(Lw=L, A=A, W=L, t_tgo=0.0, h_fine=args.h, lphi=0.002, l=0.0005, out=out,
                 refine=(-A - 0.010, A + 0.012), damage=bool(args.damage), oxidation=True,
                 h_coarse=0.02)

t_end, T0, T1 = 1800.0, 1150.0, 20.0
T_of_t = lambda t: T0 - (T0 - T1) * t / t_end
dt = args.dT / (T0 - T1) * t_end
times = np.r_[np.arange(dt, t_end, dt), t_end]
snap_T = [700.0, 400.0, 20.0]
ylim = (-0.012, 0.016)
xdmf = io.XDMFFile(m.msh.comm, os.path.join(out, "tc_crack.xdmf"), "w")
xdmf.write_mesh(m.msh)
rows, snaps, t0, tp = [], {}, time.time(), 0.0
for k, t in enumerate(times):
    # oxidation over [tp, t] at the temperature of the interval start, then mechanics at T(t)
    nsub = 4
    for j in range(nsub):
        m.update_T(T_of_t(tp + (j + 0.5) * (t - tp) / nsub))
        m.oxidise((t - tp) / nsub)
    T = T_of_t(t)
    m.update_T(T)
    it, err = m.step()
    m.update_stress()
    Ef = m.fracture_energy()
    chiTC = m.chi["TC"].x.array > 0.5
    rows.append([t, T, Ef, float(m.d.x.array.max()), float(m.s1.x.array[chiTC].max()),
                 float(m.n.x.array.max()), m.tgo_thickness(L / 2, -A), it])
    print(f"step {k+1:4d}/{len(times)} t={t:7.1f} s T={T:7.1f} dmax={rows[-1][3]:.3f} Efrac={Ef:.3e} "
          f"s1max_TC={rows[-1][4]:.1f} nmax={rows[-1][5]:.3f} it={it} err={err:.1e} "
          f"wall={time.time()-t0:.0f}s dmax[TC,TGO,BC,SUB]={m.dmax_layers()}", flush=True)
    for Ts in snap_T:
        if abs(T - Ts) < 1e-6:
            snaps[Ts] = dict(d=m.d.x.array.copy(), s1=m.s1.x.array.copy())
            for f in (m.d, m.s1, m.n, m.c):
                xdmf.write_function(f, t)
    tp = t
xdmf.close()
np.savetxt(os.path.join(out, "history.csv"), np.array(rows), delimiter=",",
           header="t_s,T_C,E_frac_Nmm_per_mm,d_max,s1max_TC_MPa,n_max,TGO_thickness_valley_mm,stagger_it",
           comments="")

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for j, Ts in enumerate(snap_T):
    m.d.x.array[:] = snaps[Ts]["d"]
    c = m.plot_nodal(ax[0, j], m.d, f"({'abc'[j]}) d, T = {Ts:.0f} C", ylim, vmin=0, vmax=1)
    m.s1.x.array[:] = snaps[Ts]["s1"]
    c2 = m.plot_cell(ax[1, j], m.s1, f"max principal stress (MPa), T = {Ts:.0f} C", ylim,
                     vmin=-300, vmax=300)
fig.colorbar(c, ax=ax[0, :], shrink=0.7); fig.colorbar(c2, ax=ax[1, :], shrink=0.7)
fig.suptitle("Fig. 13 reproduction: TC crack initiation / propagation during cooling")
fig.savefig(os.path.join(out, "fig13_tc_crack.png"), dpi=170)

r = np.array(rows)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(r[:, 0] / 60, r[:, 4], "r-"); ax.set_xlabel("t (min)"); ax.set_ylabel("max principal stress in TC (MPa)", color="r")
ax2 = ax.twinx(); ax2.plot(r[:, 0] / 60, r[:, 2], "b-"); ax2.set_ylabel("fracture energy (N mm / mm)", color="b")
fig.tight_layout(); fig.savefig(os.path.join(out, "history.png"), dpi=150)
json.dump(dict(case="5.4 TC crack", damage=bool(args.damage), h=args.h, cells=m.ncell, dT=args.dT,
               final_dmax=rows[-1][3], final_Efrac=rows[-1][2], n_max=rows[-1][5],
               tgo_thickness_valley_mm=rows[-1][6], wall_s=time.time() - t0),
          open(os.path.join(out, "summary.json"), "w"), indent=2)
print("done", flush=True)
