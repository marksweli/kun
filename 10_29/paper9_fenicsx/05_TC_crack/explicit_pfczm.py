"""Explicit chemo-thermo-mechanical PF-CZM solver, written equation by equation
after Sec. 3.1 of Min et al., CMAME 411 (2023) 116044.

  displacement  : central difference, lumped mass                (Eqs. 56-59, 71, 75_1)
  phase field   : forward Euler, lumped viscosity matrix [H]     (Eqs. 64-67, 72_2, 75_3)
  oxygen c, n   : forward Euler, lumped                          (Eqs. 45, 47, 68-70, 75_4)
  stress        : hybrid, sigma = omega(d) D (eps - eps_th - eps_n)   (Eq. 58)
  history       : H = max_tau { psi0+, ft^2 / 2E }               (Eq. 52), psi0+ from Eq. (38), dim = 2
  degradation   : Eqs. (53)-(54) (Cornelissen), a1 = 4 E Gc / (pi l ft^2)
  interface     : Eqs. (4), (10) with the interface phase field phi
  time step     : dt = beta min(dt_u, dt_d, dt_c), beta = 0.5     (Eqs. 77-78)

dolfinx provides the mesh, the P1 spaces, the interface phase field and the output;
the element loop of the explicit update (linear triangles, one-point quadrature,
row-sum lumping) is a small C kernel compiled with cffi (the tool chain FFCx uses).
Units: mm, N, MPa, s, tonne.
"""
import os
import numpy as np
import cffi

A2, A3 = 1.3868, 0.6567

C_SRC = r"""
#include <math.h>
#define A2 1.3868
#define A3 0.6567
#define PI 3.14159265358979323846

static inline double omega(double d, double a1) {
    double N = (1 - d) * (1 - d);
    double Q = a1 * d * (1 + A2 * d * (1 + A3 * d));
    return N / (N + Q);
}
static inline double domega(double d, double a1) {
    double N = (1 - d) * (1 - d), dN = -2 * (1 - d);
    double Q = a1 * d * (1 + A2 * d * (1 + A3 * d));
    double dQ = a1 * (1 + 2 * A2 * d + 3 * A2 * A3 * d * d);
    double s = N + Q;
    return (dN * Q - N * dQ) / (s * s);
}

/* flags: 1 ux = 0, 2 uy = 0, 4 right edge (common ux), 8 node of TC (c prescribed),
          16 node of BC (n, c evolve) */
void pfczm_advance(int nsteps, int nn, int nc,
             const int *conn, const double *dNx, const double *dNy, const double *area,
             const int *mat, const double *g,
             const double *minv, const double *ms, const double *msbc, const int *flag,
             double *ux, double *uy, double *uxo, double *uyo,
             double *d, double *H, double *c, double *n,
             double *fx, double *fy, double *rd, double *rc, double *wn,
             const double *P0, const double *P1, const double *par)
{
    const double dt = par[0], l = par[1], eta = par[2], Gbar = par[3], fti = par[4];
    const double GcM[3] = {par[5], par[6], par[7]}, ftM[3] = {par[8], par[9], par[10]};
    const double alpha_n = par[11], Dox = par[12], Gox = par[13], Mox = par[14];
    const double S = par[15];
    const int oxid = par[16] > 0.5, damage = par[17] > 0.5;
    for (int s = 0; s < nsteps; s++) {
        double w1 = (double)s / nsteps, w0 = 1 - w1, P[12];
        for (int k = 0; k < 12; k++) P[k] = w0 * P0[k] + w1 * P1[k];
        for (int i = 0; i < nn; i++) { fx[i] = fy[i] = rd[i] = rc[i] = wn[i] = 0; }
        for (int e = 0; e < nc; e++) {
            const int *nd = conn + 3 * e;
            const double *bx = dNx + 3 * e, *by = dNy + 3 * e;
            const double A = area[e];
            int m = mat[e];
            double E, nu, eth, Gc = 0, ft = 1;
            if (m == 0) { E = P[0]; nu = P[1]; eth = P[2]; Gc = GcM[0]; ft = ftM[0]; }
            else if (m == 1) {                                   /* Eq. (83) */
                double ne = (n[nd[0]] + n[nd[1]] + n[nd[2]]) / 3;
                E = (1 - ne) * P[3] + ne * P[6];
                nu = (1 - ne) * P[4] + ne * P[7];
                eth = (1 - ne) * P[5] + ne * P[8] + alpha_n * ne;   /* Eqs. (13), (14) */
                Gc = (1 - ne) * GcM[1] + ne * GcM[2];
                ft = (1 - ne) * ftM[1] + ne * ftM[2];
            } else { E = P[9]; nu = P[10]; eth = P[11]; }
            int pfon = damage && m != 2;
            double a1 = 0, w = 1;
            if (m != 2) {                                        /* Eqs. (4), (10) */
                Gc = g[e] * Gbar + (1 - g[e]) * Gc;
                ft = g[e] * fti + (1 - g[e]) * ft;
                a1 = 4 * E * Gc / (PI * l * ft * ft);            /* Eq. (54) */
            }
            if (pfon) w = omega((d[nd[0]] + d[nd[1]] + d[nd[2]]) / 3, a1);
            double lam = E * nu / ((1 + nu) * (1 - 2 * nu)), mu = E / (2 * (1 + nu));
            double exx = 0, eyy = 0, gxy = 0;
            for (int a = 0; a < 3; a++) {
                exx += bx[a] * ux[nd[a]]; eyy += by[a] * uy[nd[a]];
                gxy += by[a] * ux[nd[a]] + bx[a] * uy[nd[a]];
            }
            double ex = exx - eth, ey = eyy - eth, tr = ex + ey;
            double sxx = w * (lam * tr + 2 * mu * ex), syy = w * (lam * tr + 2 * mu * ey);
            double sxy = w * mu * gxy;
            for (int a = 0; a < 3; a++) {                        /* Eq. (58) */
                fx[nd[a]] += A * (bx[a] * sxx + by[a] * sxy);
                fy[nd[a]] += A * (by[a] * syy + bx[a] * sxy);
            }
            if (m != 2) {                                        /* Eqs. (38), (52) */
                double mm = 0.5 * tr, r = sqrt(0.25 * (ex - ey) * (ex - ey) + 0.25 * gxy * gxy);
                double e1 = mm + r, e2 = mm - r;
                double trp = tr > 0 ? tr : 0, e1p = e1 > 0 ? e1 : 0, e2p = e2 > 0 ? e2 : 0;
                double psi = 0.5 * lam * trp * trp + mu * (e1p * e1p + e2p * e2p);
                double fl = ft * ft / (2 * E);
                if (psi < fl) psi = fl;
                if (H[e] < psi) H[e] = psi;
            }
            if (pfon) {                                          /* Eqs. (66), (67) */
                double gx = 0, gy = 0;
                for (int a = 0; a < 3; a++) { gx += bx[a] * d[nd[a]]; gy += by[a] * d[nd[a]]; }
                for (int a = 0; a < 3; a++) {
                    double da = d[nd[a]];
                    rd[nd[a]] += A / 3 * (-domega(da, a1) * H[e] - Gc * (2 - 2 * da) / (PI * l))
                                 - 2 * Gc * l / PI * A * (bx[a] * gx + by[a] * gy);
                }
            }
            if (oxid && m == 1) {                                /* Eqs. (69), (70) */
                double gx = 0, gy = 0;
                for (int a = 0; a < 3; a++) { gx += bx[a] * c[nd[a]]; gy += by[a] * c[nd[a]]; }
                for (int a = 0; a < 3; a++) {
                    rc[nd[a]] -= Dox * A * (bx[a] * gx + by[a] * gy);
                    wn[nd[a]] += A / 3 * (damage ? omega(d[nd[a]], a1) : 1.0);
                }
            }
        }
        double Fr = 0, Mr = 0;
        for (int i = 0; i < nn; i++) if (flag[i] & 4) { Fr += fx[i]; Mr += 1 / minv[i]; }
        for (int i = 0; i < nn; i++) {
            double ax = -fx[i] * minv[i], ay = -fy[i] * minv[i];     /* Eq. (56) */
            if (flag[i] & 4) ax = -Fr / Mr;
            if (flag[i] & 1) ax = 0;
            if (flag[i] & 2) ay = 0;
            double un = 2 * ux[i] - uxo[i] + dt * dt * ax;          /* Eq. (75)_1 */
            uxo[i] = ux[i]; ux[i] = un;
            un = 2 * uy[i] - uyo[i] + dt * dt * ay;
            uyo[i] = uy[i]; uy[i] = un;
            if (damage) {                                           /* Eq. (75)_3 */
                double dn = d[i] + dt * rd[i] / (eta * ms[i]);
                d[i] = dn < 0 ? 0 : (dn > 1 ? 1 : dn);
            }
            if (oxid && (flag[i] & 16)) {                           /* Eqs. (47), (75)_4 */
                double dtp = S * dt;
                double ndot = wn[i] / msbc[i] * Gox * (1 - n[i]) * c[i];
                if (!(flag[i] & 8)) {
                    double cn = c[i] + dtp * (rc[i] / msbc[i] - Mox * ndot);
                    c[i] = cn < 0 ? 0 : cn;
                }
                double nn_ = n[i] + dtp * ndot;
                n[i] = nn_ > 1 ? 1 : nn_;
            }
        }
    }
}
"""


def build_kernel(workdir):
    """compile (once) and import the C kernel"""
    import importlib, sys
    os.makedirs(workdir, exist_ok=True)
    ffi = cffi.FFI()
    ffi.cdef("""void pfczm_advance(int, int, int, const int*, const double*, const double*, const double*,
             const int*, const double*, const double*, const double*, const double*, const int*,
             double*, double*, double*, double*, double*, double*, double*, double*,
             double*, double*, double*, double*, double*,
             const double*, const double*, const double*);""")
    ffi.set_source("_pfczm_kernel", C_SRC, extra_compile_args=["-O3", "-march=native"], libraries=["m"])
    ffi.compile(tmpdir=workdir, verbose=False)
    sys.path.insert(0, workdir)
    mod = importlib.import_module("_pfczm_kernel")
    return mod.ffi, mod.lib


def omega(d, a1):
    N = (1 - d) ** 2
    return N / (N + a1 * d * (1 + A2 * d * (1 + A3 * d)))


class ExplicitTBC:
    """explicit solver on a dolfinx P1 triangle mesh with TC (tag 1) / BC (3) / SUB (4) cells"""

    RHO = dict(TC=5.65e-9, BC=7.32e-9, TGO=3.978e-9, SUB=8.22e-9)       # Table 2, tonne/mm^3

    def __init__(self, W, cell_tags, tags, phi, props, GC, FT, Gbar, l, eta, width,
                 ox=None, damage=True, beta=0.5, workdir="/tmp/pfczm_kernel"):
        self.ffi, self.lib = build_kernel(workdir)
        msh = W.mesh
        X = W.tabulate_dof_coordinates()[:, :2]
        nc = msh.topology.index_map(2).size_local
        conn = np.ascontiguousarray(W.dofmap.list[:nc], dtype=np.int32)
        self.X, self.conn, self.nn, self.nc = X, conn, X.shape[0], nc
        x, y = X[conn, 0], X[conn, 1]
        det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        self.area = np.abs(det) / 2
        self.dNx = np.ascontiguousarray(np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / det[:, None])
        self.dNy = np.ascontiguousarray(np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / det[:, None])
        mat = np.full(nc, -1, dtype=np.int32)
        for k, v in (("TC", 0), ("BC", 1), ("SUB", 2)):
            mat[cell_tags.find(tags[k])] = v
        assert mat.min() >= 0
        self.mat = mat
        ph = phi.x.array[conn].mean(axis=1)
        self.g = 1 - (1 - ph) ** 2
        self.props, self.GC, self.FT, self.Gbar, self.l, self.eta = props, GC, FT, Gbar, l, eta
        self.ox, self.damage = ox, damage
        # lumped (row-sum) matrices
        rho = np.array([self.RHO["TC"], self.RHO["BC"], self.RHO["SUB"]])[mat]
        lump = lambda w: np.bincount(conn.ravel(), np.repeat(w / 3, 3), minlength=self.nn)
        self.mass = lump(rho * self.area)                       # Eq. (57)
        self.ms = lump(self.area)                               # Eqs. (65), (66)
        msbc = lump(self.area * (mat == 1))
        tol = 1e-9
        flag = np.zeros(self.nn, dtype=np.int32)
        flag[np.abs(X[:, 0]) < tol] |= 1
        flag[np.abs(X[:, 1] - X[:, 1].min()) < tol] |= 2
        flag[np.abs(X[:, 0] - width) < tol] |= 4
        flag[np.unique(conn[mat == 0])] |= 8
        flag[msbc > 0] |= 16
        self.flag = flag
        self.msbc = np.where(msbc > 0, msbc, 1.0)
        z = lambda k: np.zeros(k)
        self.ux, self.uy, self.uxo, self.uyo = z(self.nn), z(self.nn), z(self.nn), z(self.nn)
        self.d, self.c, self.n, self.H = z(self.nn), z(self.nn), z(self.nn), z(nc)
        if ox is not None:
            self.c[(flag & 8) > 0] = ox["cbar"]
        self._work = [z(self.nn) for _ in range(5)]
        self.minv = 1.0 / self.mass
        self.t = 0.0
        self._time_step(beta)

    # ---------------------------------------------------------------- Eqs. (77), (78)
    def _time_step(self, beta):
        x, y = self.X[self.conn, 0], self.X[self.conn, 1]
        hmin = np.min([np.hypot(x[:, i] - x[:, j], y[:, i] - y[:, j]) for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
        cd = np.zeros(self.nc)
        for T in (20.0, 1150.0):
            for k, v in (("TC", 0), ("BC", 1), ("SUB", 2)):
                E, nu, _ = self.props(k, T)
                if k == "BC":        # partially oxidised BC is stiffer and lighter
                    Eo, nuo, _ = self.props("TGO", T)
                    E, nu = max(E, Eo), min(nu, nuo)
                rho = min(self.RHO[k], self.RHO["TGO"]) if k == "BC" else self.RHO[k]
                c = np.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) / rho)
                cd[self.mat == v] = np.maximum(cd[self.mat == v], c)
        self.dt_u = float(np.min(hmin / cd))
        pfc = self.mat != 2
        Gc_max = max(self.GC["TC"], self.GC["BC"], self.GC["TGO"], self.Gbar)
        self.dt_d = float(self.eta * hmin[pfc].min() ** 2 / (2 * self.l * Gc_max)) if self.damage else np.inf
        self.hmin = float(hmin.min())
        self.dt = beta * min(self.dt_u, self.dt_d)
        self.beta = beta

    def check_oxidation_step(self, S):
        """stability of the (time-scaled) concentration update, Eq. (77)_4"""
        if self.ox is None:
            return np.inf
        hb = np.sqrt(self.area[self.mat == 1].min() * 4 / np.sqrt(3))
        dt_c = hb ** 2 / (2 * self.ox["D"]) / S
        dt_r = 1.0 / (self.ox["M"] * self.ox["G"]) / S
        return min(dt_c, dt_r)

    def _pvec(self, T):
        out = []
        for k in ("TC", "BC", "TGO", "SUB"):
            out += list(self.props(k, T))
        return np.array(out)

    def advance(self, nsteps, T0, T1, S=1.0, oxid=False):
        """nsteps explicit steps while the uniform temperature goes linearly from T0 to T1"""
        ox = self.ox or dict(alpha_n=0.0, D=0.0, G=0.0, M=0.0)
        par = np.array([self.dt, self.l, self.eta, self.Gbar, self.FT["I"],
                        self.GC["TC"], self.GC["BC"], self.GC["TGO"],
                        self.FT["TC"], self.FT["BC"], self.FT["TGO"],
                        ox["alpha_n"], ox["D"], ox["G"], ox["M"], S,
                        1.0 if (oxid and self.ox is not None) else 0.0, 1.0 if self.damage else 0.0])
        P0, P1 = self._pvec(T0), self._pvec(T1)
        f = self.ffi
        pd = lambda a: f.cast("double *", f.from_buffer(a))
        pi = lambda a: f.cast("int *", f.from_buffer(a))
        self.lib.pfczm_advance(nsteps, self.nn, self.nc, pi(self.conn), pd(self.dNx), pd(self.dNy), pd(self.area),
                         pi(self.mat), pd(self.g), pd(self.minv), pd(self.ms), pd(self.msbc), pi(self.flag),
                         pd(self.ux), pd(self.uy), pd(self.uxo), pd(self.uyo),
                         pd(self.d), pd(self.H), pd(self.c), pd(self.n),
                         *[pd(w) for w in self._work], pd(P0), pd(P1), pd(par))
        self.t += nsteps * self.dt
        self.T = T1

    # ---------------------------------------------------------------- post-processing
    def cell_fields(self, T):
        """material arrays, strains, hybrid stress (Eq. 58), max principal stress, psi0+"""
        P = {k: self.props(k, T) for k in ("TC", "BC", "TGO", "SUB")}
        ne = self.n[self.conn].mean(axis=1) * (self.mat == 1)
        pick = lambda i: np.choose(self.mat, [P["TC"][i], (1 - ne) * P["BC"][i] + ne * P["TGO"][i], P["SUB"][i]])
        E, nu, eth = pick(0), pick(1), pick(2)
        if self.ox is not None:
            eth = eth + self.ox["alpha_n"] * ne
        Gc = np.choose(self.mat, [self.GC["TC"], (1 - ne) * self.GC["BC"] + ne * self.GC["TGO"], 1.0])
        ft = np.choose(self.mat, [self.FT["TC"], (1 - ne) * self.FT["BC"] + ne * self.FT["TGO"], 1.0])
        Gc = self.g * self.Gbar + (1 - self.g) * Gc
        ft = self.g * self.FT["I"] + (1 - self.g) * ft
        a1 = 4 * E * Gc / (np.pi * self.l * ft ** 2)
        w = np.where((self.mat != 2) & self.damage, omega(self.d[self.conn].mean(axis=1), a1), 1.0)
        lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        ux, uy = self.ux[self.conn], self.uy[self.conn]
        ex = (self.dNx * ux).sum(1) - eth
        ey = (self.dNy * uy).sum(1) - eth
        gxy = (self.dNy * ux).sum(1) + (self.dNx * uy).sum(1)
        tr = ex + ey
        sxx, syy, sxy = w * (lam * tr + 2 * mu * ex), w * (lam * tr + 2 * mu * ey), w * mu * gxy
        s1 = 0.5 * (sxx + syy) + np.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2)
        return dict(E=E, nu=nu, eth=eth, Gc=Gc, ft=ft, a1=a1, w=w, sxx=sxx, syy=syy, sxy=sxy, s1=s1)

    def internal_force(self, T):
        cf = self.cell_fields(T)
        A = self.area[:, None]
        fx = np.bincount(self.conn.ravel(), (A * (self.dNx * cf["sxx"][:, None] + self.dNy * cf["sxy"][:, None])).ravel(), minlength=self.nn)
        fy = np.bincount(self.conn.ravel(), (A * (self.dNy * cf["syy"][:, None] + self.dNx * cf["sxy"][:, None])).ravel(), minlength=self.nn)
        return fx, fy

    def fracture_energy(self, T):
        """int Gc gamma dV, Eqs. (1), (48)"""
        cf = self.cell_fields(T)
        dl = self.d[self.conn]
        gx, gy = (self.dNx * dl).sum(1), (self.dNy * dl).sum(1)
        al = (2 * dl - dl ** 2).mean(axis=1)
        dens = cf["Gc"] / (np.pi * self.l) * (al + self.l ** 2 * (gx ** 2 + gy ** 2))
        return float((dens * self.area)[self.mat != 2].sum())

    def kinetic_energy(self):
        vx, vy = (self.ux - self.uxo) / self.dt, (self.uy - self.uyo) / self.dt
        return float(0.5 * (self.mass * (vx ** 2 + vy ** 2)).sum())
