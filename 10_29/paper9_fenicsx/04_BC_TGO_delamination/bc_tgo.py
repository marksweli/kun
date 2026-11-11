"""Paper Sec. 5.3: BC/TGO interface delamination during cooling (half-period model).

Fixed TGO (n = 1 inside TGO, 5 um), no growth/swelling, uniform cooling
1150 C -> 20 C in 0.3 h.  Switch damage on/off with --damage 1/0
(0 gives the residual stress state of Fig. 11, 1 the fracture process of Fig. 12).
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dolfinx import io
import tbc_common as tbc

ap = argparse.ArgumentParser()
ap.add_argument("--damage", type=int, default=1, help="1: PF-CZM damage, 0: elastic only")
ap.add_argument("--h", type=float, default=0.0002)
ap.add_argument("--dT", type=float, default=10.0, help="temperature step (C)")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.h, args.dT = 0.0008, 50.0

tag = "damage" if args.damage else "no_damage"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results" + ("_quick" if args.quick else ""), tag)
L, A, TGO = 0.060, 0.020, 0.005
m = tbc.TBCModel(Lw=L, A=A, W=L / 2, t_tgo=TGO, h_fine=args.h, lphi=0.002, l=0.0005, out=out,
                 refine=(-A - TGO - 0.008, A + 0.008), damage=bool(args.damage), oxidation=False,
                 h_coarse=0.02)
m.n.x.array[:] = 0.0          # BC stays BC, TGO material assigned via its own subdomain (n = 1)

t_end, T0, T1 = 0.3 * 3600.0, 1150.0, 20.0
T_of_t = lambda t: T0 - (T0 - T1) * t / t_end
snap_h = [0.003, 0.150, 0.210, 0.300]
times = sorted(set(np.round(np.r_[np.arange(0, t_end, args.dT / (T0 - T1) * t_end)[1:], t_end,
                                    np.array(snap_h) * 3600.0], 6)))
ylim = (-0.045, 0.035)
xdmf = io.XDMFFile(m.msh.comm, os.path.join(out, "bc_tgo.xdmf"), "w")
xdmf.write_mesh(m.msh)
rows, snaps, t0 = [], {}, time.time()
for k, t in enumerate(times):
    T = T_of_t(t)
    m.update_T(T)
    it, err = m.step()
    m.update_stress()
    Ef = m.fracture_energy()
    chiTC = m.chi["TC"].x.array > 0.5
    rows.append([t / 3600, T, Ef, float(m.d.x.array.max()), float(m.s1.x.array[chiTC].max()),
                 float(m.s1.x.array.max()), it])
    print(f"step {k+1:4d}/{len(times)} t={t/3600:.4f} h T={T:7.1f} dmax={rows[-1][3]:.3f} "
          f"Efrac={Ef:.3e} s1max_TC={rows[-1][4]:.1f} it={it} err={err:.1e} wall={time.time()-t0:.0f}s "
          f"dmax[TC,TGO,BC,SUB]={m.dmax_layers()}", flush=True)
    for th in snap_h:
        if abs(t - th * 3600) < 1e-6:
            snaps[th] = dict(d=m.d.x.array.copy(), s1=m.s1.x.array.copy(), syy=m.syy.x.array.copy())
            xdmf.write_function(m.d, t); xdmf.write_function(m.s1, t); xdmf.write_function(m.syy, t)
xdmf.close()
np.savetxt(os.path.join(out, "history.csv"), np.array(rows), delimiter=",",
           header="t_h,T_C,E_frac_Nmm_per_mm,d_max,s1max_TC_MPa,s1max_all_MPa,stagger_it", comments="")

# ---------------------------------------------------------------- figures
def setf(f, a):
    f.x.array[:] = a
    return f

if args.damage:
    fig, ax = plt.subplots(2, 4, figsize=(13, 7.5))
    for j, th in enumerate(snap_h):
        c = m.plot_nodal(ax[0, j], setf(m.d, snaps[th]["d"]), f"d, t={th:.3f} h", ylim, vmin=0, vmax=1)
        c2 = m.plot_cell(ax[1, j], setf(m.s1, snaps[th]["s1"]), f"TC max principal (MPa), t={th:.3f} h",
                         ylim, layers=["TC"])
        fig.colorbar(c2, ax=ax[1, j], fraction=0.046)
    fig.colorbar(c, ax=ax[0, -1], fraction=0.046)
    fig.suptitle("Fig. 12 reproduction: BC/TGO delamination during cooling (with damage)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig12_damage_stress.png"), dpi=170)
else:
    th = 0.3
    fig, ax = plt.subplots(2, 4, figsize=(13, 7.5))
    setf(m.syy, snaps[th]["syy"])
    for j, (lay, nm) in enumerate([(None, "(a) TBC"), (["TC"], "(b) TC"), (["TGO"], "(c) TGO"), (["BC"], "(d) BC")]):
        c = m.plot_cell(ax[0, j], m.syy, f"{nm} sigma_yy (MPa), t=0.3 h", ylim, layers=lay,
                        **({} if lay else dict(vmin=-300, vmax=300)))
        fig.colorbar(c, ax=ax[0, j], fraction=0.046)
    for j, th in enumerate(snap_h):
        c = m.plot_cell(ax[1, j], setf(m.s1, snaps[th]["s1"]), f"TC max principal (MPa), t={th:.3f} h",
                        ylim, layers=["TC"])
        fig.colorbar(c, ax=ax[1, j], fraction=0.046)
    fig.suptitle("Fig. 11 reproduction: residual stress after cooling (no damage)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig11_stress.png"), dpi=170)

r = np.array(rows)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(r[:, 0], r[:, 4], "r-", label="max principal stress in TC")
ax.set_xlabel("t (h)"); ax.set_ylabel("MPa")
ax2 = ax.twinx(); ax2.plot(r[:, 0], r[:, 2] * 1e3, "b-", label="fracture energy")
ax2.set_ylabel("fracture energy (mJ/m)  [= N mm/mm x 1e3]")
fig.legend(loc="upper left"); fig.tight_layout(); fig.savefig(os.path.join(out, "history.png"), dpi=150)
json.dump(dict(case="5.3 BC/TGO delamination", damage=bool(args.damage), h=args.h, cells=m.ncell,
               dT=args.dT, final_dmax=rows[-1][3], final_Efrac=rows[-1][2], wall_s=time.time() - t0),
          open(os.path.join(out, "summary.json"), "w"), indent=2)
print("done", flush=True)
