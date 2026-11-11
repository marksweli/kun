"""Post-processing: compare with paper Fig. 5c/d (r = 0.25) after thickness scaling.

The paper does not give the out-of-plane thickness B nor the concrete data.
For plane stress, F and the fracture energy are proportional to B, so the
curves computed with B = 100 mm are rescaled with B_eq chosen such that the
peak load equals the paper's 0.2 kN.  The energy curve then serves as an
independent check.  Paper values are rough manual readings of Fig. 5c/d.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
r = np.loadtxt(os.path.join(OUT, "load_displacement.csv"), delimiter=",", skiprows=1)
u, F, Ed = r[:, 0], r[:, 1], r[:, 2]
B = 100.0
paper_u = np.array([0.0, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
paper_F = np.array([0.0, 0.17, 0.198, 0.18, 0.145, 0.09, 0.06, 0.045, 0.032])
paper_E = np.array([0.0, 0.0, 0.001, 0.009, 0.02, 0.037, 0.05, 0.061, 0.071])
B_eq = B * 0.2 / F.max()
s = B_eq / B

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(u, F * s, "r-", label=f"FEniCSx, scaled to B = {B_eq:.1f} mm")
ax[0].plot(paper_u, paper_F, "ko", ms=4, label="paper Fig. 5c, r = 0.25 (approx.)")
ax[0].set_xlabel("displacement (mm)"); ax[0].set_ylabel("reaction force (kN)")
ax[1].plot(u, Ed * s, "r-", label=f"FEniCSx, scaled to B = {B_eq:.1f} mm")
ax[1].plot(paper_u, paper_E, "ko", ms=4, label="paper Fig. 5d, r = 0.25 (approx.)")
ax[1].set_xlabel("displacement (mm)"); ax[1].set_ylabel("fracture energy (J)")
for a in ax:
    a.grid(alpha=.3); a.legend()
fig.suptitle("TPB, h = 0.2 mm, r = 0.25: comparison after thickness scaling")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_compare_scaled.png"), dpi=160)
info = dict(B_computed_mm=B, peak_F_kN_B100=float(F.max()), u_peak_mm=float(u[F.argmax()]),
            B_eq_mm=float(B_eq), E_frac_end_J_B100=float(Ed[-1]), E_frac_end_J_Beq=float(Ed[-1] * s))
json.dump(info, open(os.path.join(OUT, "compare_scaled.json"), "w"), indent=2)
print(info)
