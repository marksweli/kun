"""Shared helpers for the phase-field cohesive zone model (PF-CZM) used in
Min et al., CMAME 411 (2023) 116044 -- "A chemo-thermo-mechanical coupled
phase field framework for failure in thermal barrier coatings".

Model ingredients (paper Sec. 2):
  * crack geometric function  alpha(d) = 2d - d^2,  c0 = pi           (Eq. 2-3)
  * Wu's rational degradation  omega(d) = (1-d)^p / ((1-d)^p + Q(d))   (Eq. 53)
    with Cornelissen softening  p=2, a1=4 E Gc/(pi l ft^2), a2=1.3868, a3=0.6567
  * hybrid formulation (Ambati): sigma = omega(d) C:eps_e, driving force
    from spectral tensile energy psi0+                                (Eq. 37-39)
  * history  H = max(psi0+, ft^2/(2E))                                 (Eq. 52)
  * smeared interface: Gc(x) = [1-(1-phi)^2] Gc_bar + (1-phi)^2 Gc_bulk (Eq. 10)
    with Gc_bar from the dissipation equivalence (Eq. 6)

The paper integrates all fields with an explicit scheme. Here every problem is
solved implicitly / quasi-statically with an alternate-minimisation (staggered)
scheme, and the irreversibility d_{n+1} >= d_n is enforced with PETSc SNES VI.

This file is copied verbatim into every case folder so that each case is
self-contained.
"""
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc as fpetsc

C0 = np.pi
A2, A3 = 1.3868, 0.6567
K_RES = 1.0e-8  # residual stiffness
# fixed quadrature (UFL would otherwise estimate a huge degree for the rational omega)
DX = ufl.dx(metadata={"quadrature_degree": 2})


# ----------------------------------------------------------------- PF-CZM
def alpha(d):
    return 2 * d - d**2


def omega(d, a1, p=2):
    Q = a1 * d * (1 + A2 * d * (1 + A3 * d))
    return (1 - d) ** p / ((1 - d) ** p + Q) * (1 - K_RES) + K_RES


def a1_coef(E, Gc, ft, l):
    return 4.0 * E * Gc / (np.pi * l * ft**2)


def gamma_density(d, l):
    return (alpha(d) + l**2 * ufl.dot(ufl.grad(d), ufl.grad(d))) / (C0 * l)


def phase_field_energy(d, d_old, d_k, H, Gc, a1, l, eta=0.0, dt=1.0):
    """Convexified phase-field energy (H fixed).  alpha(d) = 2d - d^2 is
    split into its convex part 2d and concave part -d^2; the latter is
    linearised at the previous staggered iterate d_k (convex-concave
    procedure).  Since omega(d) is convex for a1 >~ 3, every sub-problem is
    convex and Newton-VI is robust; at convergence (d = d_k) the stationarity
    condition is exactly Eq. (100)."""
    Pi = (omega(d, a1) * H
          + Gc / (C0 * l) * ((2 - 2 * d_k) * d + l**2 * ufl.dot(ufl.grad(d), ufl.grad(d)))) * DX
    if eta > 0:
        Pi += 0.5 * eta / dt * (d - d_old) ** 2 * DX
    return Pi


def eps(u):
    return ufl.sym(ufl.grad(u))


def principal_2d(e):
    tr = e[0, 0] + e[1, 1]
    rad = ufl.sqrt(((e[0, 0] - e[1, 1]) / 2) ** 2 + e[0, 1] ** 2 + 1e-30)
    return tr / 2 + rad, tr / 2 - rad


def mp(x):
    return (x + abs(x)) / 2


def psi_plus(e2, lmbda, mu, ezz=0.0):
    """Tensile part of the elastic energy, Eq. (38). e2: in-plane elastic
    strain, ezz: out-of-plane elastic strain (plane strain + eigenstrain)."""
    e1, e2_ = principal_2d(e2)
    tr = e1 + e2_ + ezz
    return 0.5 * lmbda * mp(tr) ** 2 + mu * (mp(e1) ** 2 + mp(e2_) ** 2 + mp(ezz) ** 2)


def sigma0(e2, lmbda, mu, ezz=0.0):
    """Undamaged in-plane stress (plane strain if ezz given; for plane stress
    pass the plane-stress lambda and ezz=0)."""
    return lmbda * (ufl.tr(e2) + ezz) * ufl.Identity(2) + 2 * mu * e2


def lame(E, nu, plane_stress=False):
    mu = E / (2 * (1 + nu))
    lm = E * nu / ((1 + nu) * (1 - 2 * nu))
    if plane_stress:
        lm = 2 * lm * mu / (lm + 2 * mu)
    return lm, mu


# --------------------------------------------------- interface (Sec. 2.2)
def gc_bar(Gi, GcA, GcB, l, lphi, n=20001):
    """Modified interface toughness, Eq. (6), generalised to an interface
    between two different bulk materials A (x<0) and B (x>0)."""
    b = np.pi * l / 2
    x = np.linspace(-b, b, n)
    gam = 2 * np.cos(x / l) ** 2 / (np.pi * l)          # gamma of profile Eq. (3)
    phi = np.exp(-np.abs(x) / lphi)                      # Eq. (7)
    g = 1 - (1 - phi) ** 2
    Gb = np.where(x < 0, GcA, GcB)
    num = Gi - np.trapezoid((1 - g) * Gb * gam, x)
    den = np.trapezoid(g * gam, x)
    return num / den


def R_of_r(r):
    """Left-hand side of the constraint Eq. (9)."""
    return (np.pi / 2 - (8 - (4 * r * r + 8) * np.exp(-np.pi / 2 * r)) / (r**3 + 4 * r)
            + (1 - (1 + 2 * r * r) * np.exp(-np.pi * r)) / (r**3 + 2 * r))


def solve_interface_phase_field(V, interface_dofs, lphi):
    """Steady interface phase field (Sec. 3.2): phi - lphi^2 Lap(phi) = 0,
    phi = 1 on the interface. Its 1D solution is exp(-|x|/lphi), Eq. (7)."""
    phi, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (phi * w + lphi**2 * ufl.dot(ufl.grad(phi), ufl.grad(w))) * ufl.dx
    L = fem.Constant(V.mesh, 0.0) * w * ufl.dx
    bc = fem.dirichletbc(PETSc.ScalarType(1.0), interface_dofs, V)
    pr = fpetsc.LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    out = pr.solve()
    out.x.array[:] = np.clip(out.x.array, 0.0, 1.0)
    return out


# ------------------------------------------------------------ solvers
class LinearSolver:
    """Re-assembled linear solve with a direct (MUMPS) factorisation."""

    def __init__(self, a, L, u, bcs):
        self.a, self.L = fem.form(a), fem.form(L)
        self.u, self.bcs = u, bcs
        self.A = fpetsc.create_matrix(self.a)
        self.b = fpetsc.create_vector(self.L)
        self.ksp = PETSc.KSP().create(u.function_space.mesh.comm)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self.ksp.setOperators(self.A)

    def assemble_matrix(self, bcs=None):
        bcs = self.bcs if bcs is None else bcs
        self.A.zeroEntries()
        fpetsc.assemble_matrix(self.A, self.a, bcs=bcs)
        self.A.assemble()
        self.ksp.setOperators(self.A)

    def solve(self, bcs=None, assemble=True, out=None):
        bcs = self.bcs if bcs is None else bcs
        out = self.u if out is None else out
        if assemble:
            self.assemble_matrix(bcs)
        with self.b.localForm() as bl:
            bl.set(0)
        fpetsc.assemble_vector(self.b, self.L)
        fpetsc.apply_lifting(self.b, [self.a], bcs=[bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(self.b, bcs)
        self.ksp.solve(self.b, out.x.petsc_vec)
        out.x.scatter_forward()
        return out


class PhaseFieldSolver:
    """Bound-constrained Newton (SNES vinewtonrsls) on the convexified
    phase-field energy; d_old <= d <= 1 enforces irreversibility."""

    def __init__(self, Pi, d, d_old, prefix="pf_"):
        self.d, self.d_old = d, d_old
        F = ufl.derivative(Pi, d, ufl.TestFunction(d.function_space))
        self.L, self.a = fem.form(F), fem.form(ufl.derivative(F, d))
        self.A = fpetsc.create_matrix(self.a)
        self.b = fpetsc.create_vector(self.L)
        V = d.function_space
        self.lb, self.ub = fem.Function(V), fem.Function(V)
        self.ub.x.array[:] = 1.0
        snes = PETSc.SNES().create(V.mesh.comm)
        snes.setFunction(self._F, self.b)
        snes.setJacobian(self._J, self.A)
        snes.setType("vinewtonrsls")
        snes.setTolerances(rtol=1e-9, atol=1e-10, max_it=100)
        ksp = snes.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        # PETSc native LU: MUMPS crashes when the VI reduced (inactive) set is empty
        ksp.getPC().setFactorSolverType("petsc")
        ksp.getPC().setFactorOrdering("nd")
        snes.setOptionsPrefix(prefix)
        snes.setFromOptions()
        self.snes = snes
        self.x = d.x.petsc_vec.copy()

    def _F(self, snes, x, b):
        x.copy(self.d.x.petsc_vec)
        self.d.x.scatter_forward()
        with b.localForm() as bl:
            bl.set(0.0)
        fpetsc.assemble_vector(b, self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def _J(self, snes, x, A, P):
        A.zeroEntries()
        fpetsc.assemble_matrix(A, self.a)
        A.assemble()

    def solve(self):
        self.lb.x.array[:] = self.d_old.x.array
        self.snes.setVariableBounds(self.lb.x.petsc_vec, self.ub.x.petsc_vec)
        self.d.x.petsc_vec.copy(self.x)
        self.snes.solve(None, self.x)
        self.x.copy(self.d.x.petsc_vec)
        self.d.x.scatter_forward()
        r = self.snes.getConvergedReason()
        if r < 0:
            print(f"  [warning] phase-field SNES not converged (reason {r})", flush=True)
        return self.snes.getIterationNumber(), r


class History:
    """Cell-wise (DG0) history variable H = max(max_t psi0+, ft^2/2E).

    Hn stores only the historical maximum of psi0+; the floor ft^2/2E is
    evaluated with the *current* (possibly temperature-dependent) E, so that
    a1(T) * H0(T) always equals the damage threshold."""

    def __init__(self, W0, psi_expr, H0_expr, eps0=1e-3):
        # H0 slightly below ft^2/2E: at exactly H0 the PF-CZM energy has a
        # saddle at d=0 (f'(0)=0, f''(0)<0) that a minimiser would escape.
        H0_expr = (1 - eps0) * H0_expr
        self.H = fem.Function(W0, name="H")
        self.Hn = fem.Function(W0)
        pts = W0.element.interpolation_points()
        self.expr = fem.Expression(ufl.max_value(ufl.max_value(self.Hn, psi_expr), H0_expr), pts)
        self.expr_psi = fem.Expression(ufl.max_value(self.Hn, psi_expr), pts)
        self.expr0 = fem.Expression(H0_expr, pts)
        self.H.interpolate(self.expr0)

    def update(self):
        self.H.interpolate(self.expr)

    def commit(self):
        self.Hn.interpolate(self.expr_psi)


def assemble_scalar(form):
    comm = form.mesh.comm if hasattr(form, "mesh") else MPI.COMM_WORLD
    return comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)


def staggered(mech_solve, H, pf, d, d_iter, tol=1e-4, max_it=100):
    """Alternate minimisation: u -> H -> d until max|d_k - d_{k-1}| < tol."""
    it, err = 0, 1.0
    while it < max_it:
        it += 1
        d_iter.x.array[:] = d.x.array
        mech_solve()
        H.update()
        pf.solve()
        err = np.max(np.abs(d.x.array - d_iter.x.array)) if d.x.array.size else 0.0
        err = d.function_space.mesh.comm.allreduce(err, op=MPI.MAX)
        if err < tol:
            break
    return it, err


# ------------------------------------------------------------ plotting
def p1_triangulation(V):
    import matplotlib.tri as mtri
    x = V.tabulate_dof_coordinates()
    cells = V.dofmap.list
    return mtri.Triangulation(x[:, 0], x[:, 1], cells)
