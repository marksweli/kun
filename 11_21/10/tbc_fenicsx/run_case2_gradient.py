"""Case 2 (paper Sec. 3.2): TC internal cracking of a TBC cooled from a steady
thermal gradient (T_top = 400 K, T_bottom = 200 K, relative to room
temperature), optional randomly placed pre-existing micro-defects (Fig. 11).

    python run_case2_gradient.py --defects 1   # with micro-defects (Fig. 11)
    python run_case2_gradient.py --defects 0   # without
"""
import argparse
import numpy as np
import tbc_common as tc

p = argparse.ArgumentParser()
p.add_argument("--defects", type=int, default=1, help="1: pre-existing micro-defects on, 0: off")
p.add_argument("--ref", type=str, default="room", choices=["room", "top"],
               help="room: dT = -T (cooled from the gradient state to room T); "
                    "top: dT = T - T_top (stress free at uniform T_top, cooled to the gradient)")
p.add_argument("--A", type=float, default=20.0)
p.add_argument("--ft", type=float, default=20.0, help="TC tensile strength / MPa (calibrated)")
p.add_argument("--Gc", type=float, default=0.3, help="effective TC/interface G_c / (J/m^2)")
p.add_argument("--l0", type=float, default=2.0)
p.add_argument("--h", type=float, default=1.0)
p.add_argument("--nsteps", type=int, default=50)
p.add_argument("--out", type=str, default=None)
a = p.parse_args()

W = 3 * tc.PERIOD  # 240 um wide, as in Fig. 11
# seven 6-um horizontal defects at the positions read from Fig. 11(b)
centres = [(25, 449), (35, 399), (128, 421), (183, 427), (143, 349), (127, 319), (212, 291)]
cracks = [((x - 3.0, y), (x + 3.0, y)) for x, y in centres] if a.defects else []
lam = np.linspace(0, 1, a.nsteps + 1)[1:]
snaps = [lam[int(round(f * a.nsteps)) - 1] for f in (0.70, 0.85, 0.95, 1.0)]
out = a.out or f"results/case2_gradient_defects{a.defects}"
tc.run_simulation("case2", out, W, a.A, cracks, ("gradient", 400.0, 200.0, a.ref), a.l0, a.ft, a.h,
                  (265.0, 480.0), lam, snaps, Gc=a.Gc, view=dict(xlim=(0, W), ylim=(260, 480), aspect=0.75))
