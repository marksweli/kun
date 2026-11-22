"""Case 1 (paper Sec. 3.1): TC interfacial cracking of an isothermally cooled
TBC (dT = -285 K), interface amplitude A = 20 um, optional 6-um horizontal
pre-existing micro-crack in the TC near the interface (Fig. 9/10).

    python run_case1_isothermal.py --precrack 1   # with micro-crack (Figs. 9-10)
    python run_case1_isothermal.py --precrack 0   # without (Figs. 6-7)
"""
import argparse
import numpy as np
import tbc_common as tc

p = argparse.ArgumentParser()
p.add_argument("--precrack", type=int, default=1, help="1: pre-existing micro-crack on, 0: off")
p.add_argument("--A", type=float, default=20.0, help="interface amplitude (peak-to-valley) / um")
p.add_argument("--ft", type=float, default=20.0, help="TC tensile strength / MPa (calibrated)")
p.add_argument("--Gc", type=float, default=0.3, help="effective TC/interface G_c / (J/m^2)")
p.add_argument("--l0", type=float, default=2.0, help="phase-field length scale / um")
p.add_argument("--h", type=float, default=0.8, help="mesh size in the fine zone / um")
p.add_argument("--nsteps", type=int, default=60)
p.add_argument("--out", type=str, default=None)
a = p.parse_args()

W = 2 * tc.PERIOD  # two interface cycles, 160 um
# 6-um horizontal micro-crack above the valley/peak flank, cf. Fig. 9(d)
cracks = [((89.0, 236.0), (95.0, 236.0))] if a.precrack else []
lam = np.linspace(0, 1, a.nsteps + 1)[1:]
snaps = [lam[int(round(f * a.nsteps)) - 1] for f in (0.80, 0.88, 0.94, 1.0)]
out = a.out or f"results/case1_isothermal_precrack{a.precrack}"
tc.run_simulation("case1", out, W, a.A, cracks, ("iso", -285.0), a.l0, a.ft, a.h, (183.0, 300.0),
                  lam, snaps, Gc=a.Gc, view=dict(xlim=(0, W), ylim=(150, 310), aspect=1.0))
