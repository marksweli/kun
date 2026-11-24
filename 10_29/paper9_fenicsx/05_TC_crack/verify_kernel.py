"""Consistency check of the explicit C kernel (explicit_pfczm.py) against dolfinx:
internal force vector, Eq. (58), and lumped mass, Eq. (57), for a random state.

    python verify_kernel.py
"""
import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc
import tbc_common as tbc
import explicit_pfczm as ex

L, A, l = 0.030, 0.005, 0.0005
m = tbc.TBCModel(Lw=L, A=A, W=L, t_tgo=0.0, h_fine=0.0006, lphi=0.002, l=l, out="/tmp/verify_kernel",
                 refine=(-A - 0.003, A + 0.005), damage=True, oxidation=False, h_coarse=0.02)
GC = dict(TC=tbc.GC["TC"], BC=tbc.GC["BC"], TGO=tbc.GC["TGO"])
FT = dict(TC=tbc.FT["TC"], BC=tbc.FT["BC"], TGO=tbc.FT["TGO"], I=tbc.FT["I_TC_TGO"])
s = ex.ExplicitTBC(m.Ws, m.ct, m.tags, m.phi[0], tbc.props, GC, FT, dict(m.Gbar)["I_TC_TGO"], l, 1e-5, L,
                   ox=None, damage=False, workdir="/tmp/verify_kernel/_kernel")
assert np.allclose(m.V.tabulate_dof_coordinates()[:, :2], s.X)
rng = np.random.default_rng(0)
T = 600.0
s.ux[:] = 1e-5 * rng.standard_normal(s.nn)
s.uy[:] = 1e-5 * rng.standard_normal(s.nn)
s.uxo[:], s.uyo[:] = s.ux, s.uy
u = fem.Function(m.V)
u.x.array[0::2], u.x.array[1::2] = s.ux, s.uy
cf = s.cell_fields(T)


def dg(a):
    f = fem.Function(m.W0)
    f.x.array[:] = a
    return f


E, nu, eth = dg(cf["E"]), dg(cf["nu"]), dg(cf["eth"])
lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
v = ufl.TestFunction(m.V)
ee = ufl.sym(ufl.grad(u)) - eth * ufl.Identity(2)
sig = lam * ufl.tr(ee) * ufl.Identity(2) + 2 * mu * ee
F = fem.petsc.assemble_vector(fem.form(ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx)).array
u0x, u0y = s.ux.copy(), s.uy.copy()
s.advance(1, T, T)                       # from rest: u_new = u - dt^2 M^-1 F_int
free = (s.flag & 7) == 0
kfx = -(s.ux - u0x) * s.mass / s.dt ** 2
kfy = -(s.uy - u0y) * s.mass / s.dt ** 2
e1 = max(np.abs(kfx - F[0::2])[free].max(), np.abs(kfy - F[1::2])[free].max()) / np.abs(F).max()
rho = dg(np.array([s.RHO["TC"], s.RHO["BC"], s.RHO["SUB"]])[s.mat])
ML = fem.petsc.assemble_vector(fem.form(rho * ufl.TestFunction(m.Ws) * ufl.dx)).array
e2 = np.abs(ML - s.mass).max() / s.mass.max()
print(f"internal force, C kernel vs dolfinx: rel. diff {e1:.2e}")
print(f"lumped mass,    C kernel vs dolfinx: rel. diff {e2:.2e}")
assert e1 < 1e-8 and e2 < 1e-10
print("OK")
