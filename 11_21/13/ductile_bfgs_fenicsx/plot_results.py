"""Plot force-displacement curve (vs. Fig. 4) and damage contours (vs. Fig. 3c)."""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

runs = sys.argv[1:] or sorted(glob.glob("results_*"))

# Fig. 4, "residual controlled staggered algorithm" [82] (digitised from the paper figure)
ref = np.array([[0.0, 0.0], [0.031, 830], [0.102, 2680], [0.146, 3740], [0.167, 4210],
                [0.187, 4600], [0.189, 4350], [0.189, 3850], [0.189, 1810], [0.189, 0.0],
                [0.227, 0.0], [0.260, 0.0], [0.300, 0.0]])

fig, ax = plt.subplots(figsize=(7, 4))
for r in runs:
    fd = np.loadtxt(os.path.join(r, "force_displacement.csv"), delimiter=",", skiprows=1)
    du = r.rstrip("/").split("results_du")[-1].replace("e-", "E-")
    lab = f"FEniCSx BFGS monolithic ($\\Delta u$={du} mm)"
    ax.plot(fd[:, 0], fd[:, 1], "--", lw=2, label=lab)
ax.plot(ref[:, 0], ref[:, 1], "ko", mfc="none", ms=9, mew=1.8,
        label="Residual controlled staggered [82] (digitised Fig. 4)")
ax.set_xlabel("Displacement u (mm)"); ax.set_ylabel("Force (N)")
ax.set_xlim(0, 0.3); ax.set_ylim(0, 6000); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("force_displacement.png", dpi=200)

for r in runs:
    m = np.load(os.path.join(r, "mesh.npz"))
    tri = mtri.Triangulation(m["x"][:, 0], m["x"][:, 1], m["cells"])
    snaps = sorted(glob.glob(os.path.join(r, "damage_u*.npz")))
    pick = [s for s in snaps if any(k in s for k in ("0.180", "0.189", "0.192", "0.195"))] or snaps
    fig, axs = plt.subplots(1, len(pick), figsize=(2.6 * len(pick) + 1, 6.2))
    axs = np.atleast_1d(axs)
    for a, s in zip(axs, pick):
        dd = np.load(s)
        tc = a.tripcolor(tri, np.clip(dd["d"], 0, 1), shading="gouraud", cmap="jet", vmin=0, vmax=1)
        a.set_aspect("equal"); a.axis("off")
        a.set_title(f"u = {float(dd['u']):.3f} mm")
    fig.colorbar(tc, ax=axs.tolist(), label="Dmg", shrink=0.8)
    fig.savefig(os.path.join(r, "damage_evolution.png"), dpi=200, bbox_inches="tight")
print("ok")
