"""Paper Sec. 5.4: top-coat (TC) crack during cooling -- explicit solver of the paper.

One-period model (L = 30 um, A = 5 um, TC 500 um / BC 100 um / substrate 3 mm,
Fig. 10a boundary conditions: u_x = 0 on the left edge, u_y = 0 on the bottom,
right edge kept straight), initial TGO thickness 0, uniform cooling
1150 C -> 20 C in 30 min, oxygen diffusion / oxidation of the BC while T > 850 C
(c = 1.55 mol/m^3 prescribed in the TC, i.e. on the TC/TGO interface).

The coupled problem is integrated with the explicit scheme of Sec. 3.1.2 of the
paper (central difference + lumped mass for u, forward Euler + lumped matrices
for d, c and n; see explicit_pfczm.py).  The stable time step (Eqs. 77-78,
beta = 0.5) is ~6e-12 s, so the 30 min of cooling cannot be resolved in physical
time: as in any explicit analysis of a slow process the loading is applied over
a shorter simulated duration t_sim (--tsim).  The paper does not report the
duration / number of steps it used; the rates of the time-dependent processes
(cooling, diffusion, oxidation) are scaled by S = 1800 s / t_sim so that they
follow the physical 30 min history.

The previous implicit quasi-static version of this case is kept as
tc_crack_implicit.py (results in results_implicit/).
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dolfinx import fem, io
import tbc_common as tbc
import explicit_pfczm as ex

ap = argparse.ArgumentParser()
ap.add_argument("--tsim", type=float, default=1.2e-5, help="simulated duration of the 30 min cooling (s)")
ap.add_argument("--eta", type=float, default=1.0e-5, help="phase-field viscosity (N s/mm^2); paper Sec. 5.2: 1e-8 kN s/mm^2")
ap.add_argument("--damage", type=int, default=1)
ap.add_argument("--h", type=float, default=0.0002)
ap.add_argument("--tag", default="")
ap.add_argument("--nout", type=int, default=100, help="number of monitoring / snapshot instants")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.h, args.nout = 0.0006, 20

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "results" + ("_quick" if args.quick else "") + args.tag)
L, A, l, lphi = 0.030, 0.005, 0.0005, 0.002
# dolfinx: mesh (refined critical zone around the interface), spaces, interface phase field
m = tbc.TBCModel(Lw=L, A=A, W=L, t_tgo=0.0, h_fine=args.h, lphi=lphi, l=l, out=out,
                 refine=(-A - 0.003, A + 0.005), damage=True, oxidation=False, h_coarse=0.02)
GC = dict(TC=tbc.GC["TC"], BC=tbc.GC["BC"], TGO=tbc.GC["TGO"])
FT = dict(TC=tbc.FT["TC"], BC=tbc.FT["BC"], TGO=tbc.FT["TGO"], I=tbc.FT["I_TC_TGO"])
ox = dict(alpha_n=tbc.ALPHA_N, D=tbc.D_OX, G=tbc.G_OX, M=tbc.M_OX, cbar=tbc.C_BAR)
s = ex.ExplicitTBC(m.Ws, m.ct, m.tags, m.phi[0], tbc.props, GC, FT, dict(m.Gbar)["I_TC_TGO"], l, args.eta, L,
                   ox=ox, damage=bool(args.damage), workdir=os.path.join(here, "_kernel"))

t_phys, T0, T1 = 1800.0, 1150.0, 20.0
S = t_phys / args.tsim
nsteps = int(np.ceil(args.tsim / s.dt))
dt_c = s.check_oxidation_step(S)
print(f"nodes {s.nn}, cells {s.nc}, h_min {s.hmin*1e3:.3f} um")
print(f"dt_u = {s.dt_u:.3e} s, dt_d = {s.dt_d:.3e} s, dt_c = {dt_c:.3e} s -> dt = {s.dt:.3e} s (beta = {s.beta})")
print(f"t_sim = {args.tsim:.3e} s, {nsteps} steps, time scale S = {S:.3e}", flush=True)
assert s.dt <= s.beta * dt_c, "concentration update unstable for this time scale"

T_of = lambda k: T0 - (T0 - T1) * k / nsteps
edges = np.unique(np.round(np.linspace(0, nsteps, args.nout + 1)).astype(int))
tc_cells, far = s.mat == 0, (s.mat == 0) & (s.X[s.conn, 1].mean(axis=1) > 0.05)
xb = np.linspace(0, L, 61)


def crack_extent(thr=0.9):
    """fraction of the period covered by d > thr (projection on x)"""
    xs = s.X[s.d > thr, 0]
    return np.unique(np.clip(np.digitize(xs, xb), 1, xb.size - 1)).size / (xb.size - 1) if xs.size else 0.0


rows, snaps, t0 = [], [], time.time()
for k0, k1 in zip(edges[:-1], edges[1:]):
    # split a block at T = 850 C so that the oxidation switch is exact
    ks = [k0, k1]
    kx = int(np.ceil((T0 - tbc.T_OX) / (T0 - T1) * nsteps))
    if k0 < kx < k1:
        ks = [k0, kx, k1]
    for a, b in zip(ks[:-1], ks[1:]):
        s.advance(b - a, T_of(a), T_of(b), S=S, oxid=T_of(a) > tbc.T_OX)
    T = T_of(k1)
    cf = s.cell_fields(T)
    row = [k1 * s.dt * S, T, s.fracture_energy(T), float(s.d.max()), float(s.d[(s.flag & 8) > 0].max()),
           crack_extent(), float(cf["s1"][tc_cells].max()), float(np.average(cf["syy"][far], weights=s.area[far])),
           s.kinetic_energy(), float(s.n.max())]
    rows.append(row)
    snaps.append(dict(T=T, d=s.d.copy(), s1=cf["s1"].copy(), n=s.n.copy(), c=s.c.copy(),
                      ux=s.ux.copy(), uy=s.uy.copy()))
    print(f"step {k1:9d}/{nsteps} t={row[0]:7.1f} s T={T:7.1f} dmax={row[3]:.3f} dmaxTC={row[4]:.3f} "
          f"extent={row[5]:.2f} Efrac={row[2]:.3e} s1max_TC={row[6]:.1f} syy_TC_far={row[7]:.1f} "
          f"Ekin={row[8]:.2e} nmax={row[9]:.3f} wall={time.time()-t0:.0f}s", flush=True)
    if not np.isfinite(s.ux).all():
        raise RuntimeError("explicit solution diverged")
r = np.array(rows)
np.savetxt(os.path.join(out, "history.csv"), r, delimiter=",", comments="",
           header="t_s,T_C,E_frac_Nmm_per_mm,d_max,d_max_TC,crack_extent,s1max_TC_MPa,syy_TC_far_MPa,E_kin,n_max")

np.savez_compressed(os.path.join(out, "snapshots.npz"), t=r[:, 0], T=r[:, 1],
                    d=np.array([sn["d"] for sn in snaps], dtype=np.float32), x=s.X.astype(np.float32))

# ------------------------------------------------ pick the Fig. 13 instants
ext = r[:, 5]
first = lambda mask: int(np.argmax(mask)) if mask.any() else len(r) - 1
ia = first(r[:, 4] >= 0.25)                    # damage initiation
ic = first(ext >= 0.999)                       # crack spans the whole period
ib = min(first(r[:, 4] >= 0.6), ic)            # propagation
pick = [ia, ib, ic]

# ------------------------------------------------ dolfinx output (XDMF)
u, dfun, s1f = fem.Function(m.V, name="u"), fem.Function(m.Ws, name="d"), fem.Function(m.W0, name="s1")
nf, cfun = fem.Function(m.Ws, name="n"), fem.Function(m.Ws, name="c")
with io.XDMFFile(m.msh.comm, os.path.join(out, "tc_crack.xdmf"), "w") as xdmf:
    xdmf.write_mesh(m.msh)
    for i in sorted(set(pick + [len(r) - 1])):
        sn = snaps[i]
        u.x.array[0::2], u.x.array[1::2] = sn["ux"], sn["uy"]
        dfun.x.array[:], s1f.x.array[:], nf.x.array[:], cfun.x.array[:] = sn["d"], sn["s1"], sn["n"], sn["c"]
        for f in (u, dfun, s1f, nf, cfun):
            xdmf.write_function(f, r[i, 0])

# ------------------------------------------------ figures
ylim = (-0.012, 0.016)
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for j, i in enumerate(pick):
    m.d.x.array[:] = snaps[i]["d"]
    c1 = m.plot_nodal(ax[0, j], m.d, f"({'abc'[j]}) d,  t = {r[i,0]/60:.1f} min, T = {r[i,1]:.0f} C", ylim, vmin=0, vmax=1)
    s1f.x.array[:] = snaps[i]["s1"]
    c2 = m.plot_cell(ax[1, j], s1f, f"max principal stress (MPa), T = {r[i,1]:.0f} C", ylim, vmin=-300, vmax=300)
fig.colorbar(c1, ax=ax[0, :], shrink=0.7); fig.colorbar(c2, ax=ax[1, :], shrink=0.7)
fig.suptitle("Fig. 13 reproduction (explicit scheme of the paper): TC crack initiation, propagation, coalescence")
fig.savefig(os.path.join(out, "fig13_tc_crack.png"), dpi=170)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(r[:, 0] / 60, r[:, 3], "r-", label="d max"); ax[0].plot(r[:, 0] / 60, r[:, 5], "k--", label="crack extent (d > 0.9) / L")
ax[0].set_xlabel("t (min)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(r[:, 0] / 60, r[:, 6], "r-", label="max principal stress in TC")
ax[1].plot(r[:, 0] / 60, r[:, 7], "b-", label=r"mean $\sigma_{yy}$ in TC, y > 50 um")
ax[1].set_xlabel("t (min)"); ax[1].set_ylabel("MPa"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].plot(r[:, 0] / 60, r[:, 2], "b-"); ax[2].set_xlabel("t (min)"); ax[2].set_ylabel("fracture energy (N mm / mm)"); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(out, "history.png"), dpi=150)

json.dump(dict(case="5.4 TC crack", scheme="explicit (paper Sec. 3.1.2)", damage=bool(args.damage), h=args.h,
               nodes=int(s.nn), cells=int(s.nc), dt=s.dt, dt_u=s.dt_u, dt_d=s.dt_d, beta=s.beta, eta=args.eta,
               t_sim=args.tsim, steps=nsteps, time_scale=S, final_dmax=float(r[-1, 3]),
               final_crack_extent=float(r[-1, 5]), final_Efrac=float(r[-1, 2]), n_max=float(r[-1, 9]),
               T_initiation_C=float(r[ia, 1]) if r[ia, 4] >= 0.25 else None,
               T_coalescence_C=float(r[ic, 1]) if ext[ic] >= 0.999 else None,
               wall_s=time.time() - t0),
          open(os.path.join(out, "summary.json"), "w"), indent=2)
print("done", flush=True)
