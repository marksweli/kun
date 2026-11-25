"""Post-processing: Fig. 11 / 17 / 18 / 19 style plots for the 10 min CMAS case."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
F = np.load(os.path.join(OUT, "fields.npz"))
tri = mtri.Triangulation(F["xy"][:, 0], F["xy"][:, 1], F["tri"])


def nodal(vals, xd):
    # dof coordinates -> geometry node ordering
    from scipy.spatial import cKDTree
    idx = cKDTree(xd).query(F["xy"])[1]
    return vals[idx]


xd = F["xd"]
C = nodal(F["C"], xd)

# Fig. 11-type depth profile (paper, 10 min curve digitised from Fig. 11)
prof = np.loadtxt(os.path.join(OUT, "cmas_profile.csv"), delimiter=",", skiprows=1)
paper10 = np.array([[0, 4.0], [10, 3.6], [20, 3.2], [30, 2.85], [40, 2.5], [50, 2.2],
                    [60, 1.95], [70, 1.72], [80, 1.55], [90, 1.42], [100, 1.37]])
fig, ax = plt.subplots(figsize=(5, 4))
for k, m in enumerate((1, 5, 10)):
    ax.plot(prof[:, 0] / 1000, prof[:, k + 1], "-o", ms=3, label=f"FEniCSx {m} min")
ax.plot(paper10[:, 0] / 1000, paper10[:, 1], "k--", label="paper Fig. 11, 10 min")
ax.set(xlabel="Depth from surface / mm", ylabel="CMAS concentration / mol%", xlim=(0, 0.1), ylim=(0, 4.5))
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig11_cmas_profile.png"), dpi=150)

# Fig. 17-type: CMAS field + phase field at cycles 3,5,7,9
cyc = [k for k in (3, 5, 7, 9) if f"d{k}" in F]
fig, axs = plt.subplots(1, 1 + len(cyc), figsize=(3.2 * (1 + len(cyc)), 3.4))
im = axs[0].tripcolor(tri, C, shading="gouraud", cmap="jet", vmin=0, vmax=4)
axs[0].set_title("CMAS, 10 min")
fig.colorbar(im, ax=axs[0], fraction=0.046, label="mol%")
for a, k in zip(axs[1:], cyc):
    im = a.tripcolor(tri, nodal(F[f"d{k}"], xd), shading="gouraud", cmap="jet", vmin=0, vmax=1)
    a.set_title(f"d, {k} cycle")
fig.colorbar(im, ax=axs[-1], fraction=0.046, label="phase field")
for a in axs:
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig17_fatigue_10min.png"), dpi=150)

# Fig. 18 / 19-type bars, with the paper's 10 min values (digitised)
fe = np.loadtxt(os.path.join(OUT, "failed_elements.csv"), delimiter=",", skiprows=1, ndmin=2)
paper_tot = np.array([2, 47, 82, 111, 127, 135, 143, 152, 156, 162])
paper_bot = np.array([2, 29, 32, 35, 35, 35, 35, 35, 35, 37])
n = fe[:, 0].astype(int)
# failed area in um^2 == number of equivalent 1 um x 1 um elements (paper mesh)
fb, ft = fe[:, 3], fe[:, 4]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
w = 0.38
axs[0].bar(n - w / 2, fb, w, color="#1f4e9c", label="FEniCSx bottom half")
axs[0].bar(n - w / 2, ft, w, bottom=fb, color="#c0392b", label="FEniCSx top half")
axs[0].bar(n + w / 2, paper_bot[:len(n)], w, color="#8ec3ff", label="paper bottom half")
axs[0].bar(n + w / 2, (paper_tot - paper_bot)[:len(n)], w, bottom=paper_bot[:len(n)],
           color="#ffcccc", label="paper top half")
axs[0].set(xlabel="Number of cycles", ylabel="Failure elements (equiv. 1 $\\mu$m$^2$)", xticks=n)
axs[0].legend()
tot = np.maximum(fb + ft, 1e-12)
pb = paper_bot / paper_tot
axs[1].bar(n - w / 2, np.where(fb + ft > 0, fb / tot, np.nan), w, color="#1f4e9c", label="FEniCSx bottom")
axs[1].bar(n - w / 2, ft / tot, w, bottom=fb / tot, color="#c0392b", label="FEniCSx top")
axs[1].bar(n + w / 2, pb[:len(n)], w, color="#8ec3ff", label="paper bottom")
axs[1].bar(n + w / 2, 1 - pb[:len(n)], w, bottom=pb[:len(n)], color="#ffcccc", label="paper top")
axs[1].set(xlabel="Number of cycles", ylabel="Proportion", xticks=n, ylim=(0, 1))
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig18_19_failed_elements.png"), dpi=150)

ss = np.loadtxt(os.path.join(OUT, "stress_strain.csv"), delimiter=",", skiprows=1)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].plot(ss[:, 2] * 100, ss[:, 3], lw=0.8)
axs[0].set(xlabel="Applied strain / %", ylabel="Nominal stress / MPa")
axs[1].plot(ss[:, 1], ss[:, 3], lw=0.8)
axs[1].set(xlabel="Time / s", ylabel="Nominal stress / MPa")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stress_strain.png"), dpi=150)
print("plots written to", OUT)
