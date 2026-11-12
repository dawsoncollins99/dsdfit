"""Steady Fokker-Planck solver (internal).

Most users never call this module directly: ``model_fractions``, ``simulate``,
``model_evolution`` and ``fit`` use it for you.

The model, in y = (r + ell)^2 (um^2), where ell is the kinetic length from
``Conditions`` (ell = 0 by default, so y = r^2, radius squared):

    d/dy [ v N - D_x dN/dy ] + k r^2 N = 0,      v = a (s_bar - A_k / r + B_k / r^3),   r = sqrt(y) - ell

with droplets injected at the smallest size (flux F_in = 1) and no flux at the
largest size. Working in y keeps the equation in the same form with the kinetic
correction: a droplet grows as dy/dt = a (s - s_eq(r)), so supersaturation
fluctuations still spread y with a constant D_x = a^2 sigma_s^2 tau.

It is solved with finite volumes on a uniform grid in y, using Scharfetter-Gummel
fluxes (as in Steve Krueger's MATLAB solver); the result is a tridiagonal linear
system.

``Model.fractions_and_sensitivities`` also returns how each bin's fraction changes
with ln(D_x) and s_bar (tangent-linear method: the same tridiagonal matrix with a
new right-hand side). The fitting code needs these.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import solve_banded

CELL_WIDTH = 0.007     # target width of one grid cell in y = (r + ell)^2 [um^2]
R_MAX_MIN = 30.0       # the grid always extends at least to this radius [um]
MAX_CELLS = 2_000_000


# ---------------------------------------------------------------- Bernoulli-function helpers
def _bern(z):
    """B(z) = z / (e^z - 1), accurate for small z."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-3
    zs = z[small]
    out[small] = 1 - zs / 2 + zs**2 / 12
    zl = z[~small]
    with np.errstate(over="ignore"):
        out[~small] = zl / np.expm1(zl)
    return out


def _bern_prime(z):
    """B'(z)."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-3
    zs = z[small]
    out[small] = -0.5 + zs / 6 - zs**3 / 180
    zl = z[~small]
    b = _bern(zl)
    out[~small] = b * (1 - zl - b) / zl
    return out


def _diff_weight(z):
    """G(z) = B(z) - z B'(z) = z^2 e^z / (e^z - 1)^2."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-3
    zs = z[small]
    out[small] = 1 - zs**2 / 12 + zs**4 / 240
    za = np.abs(z[~small])
    with np.errstate(over="ignore", under="ignore"):
        out[~small] = za**2 * np.exp(-za) / np.expm1(-za) ** 2
    return out


# ---------------------------------------------------------------- the model
class Model:
    """The discretized model for one set of conditions and one set of bins."""

    def __init__(self, cond, edges_diameter):
        self.cond = cond
        r_edges = 0.5 * np.asarray(edges_diameter, float)          # bin edges as radii [um]
        r_min = 0.5 * (0.5 * cond.d_dry)                           # half the dry radius
        if r_edges[0] <= r_min:
            raise ValueError("The smallest bin edge must be larger than half the dry particle diameter.")
        r_max = max(R_MAX_MIN, 1.5 * r_edges[-1])
        ell = cond.ell
        self.ell = ell
        n_cells = int(np.ceil(((r_max + ell)**2 - (r_min + ell)**2) / CELL_WIDTH))
        if n_cells > MAX_CELLS:
            raise ValueError(f"The largest bin edge ({2 * r_edges[-1]:g} um) needs too fine a grid. "
                             "Drop bins well beyond the largest droplets.")
        self.n_cells = n_cells
        self.yf = np.linspace((r_min + ell)**2, (r_max + ell)**2, n_cells + 1)   # cell faces [um^2]
        self.dy = self.yf[1] - self.yf[0]
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])                            # cell centers [um^2]
        self.rc = np.sqrt(self.yc) - ell                                          # cell-center radii [um]

        # which cells lie in which bin (cells cut by a bin edge are split by length)
        ye = (r_edges + ell)**2
        j = np.clip(np.floor((ye - self.yf[0]) / self.dy).astype(int), 0, n_cells - 1)
        self._j = j
        self._f = np.clip((ye - self.yf[j]) / self.dy, 0.0, 1.0)
        self.n_bins = len(r_edges) - 1

    # ---- the tridiagonal system
    def _assemble(self, D_x, s_bar):
        c = self.cond
        v_center = c.a * (s_bar - c.A_k / self.rc + c.B_k / self.rc**3)   # drift [um^2/s]
        v_face = 0.5 * (v_center[:-1] + v_center[1:])
        Pe = v_face * self.dy / D_x                                         # cell Peclet number
        alpha = (D_x / self.dy) * _bern(-Pe)
        beta = -(D_x / self.dy) * _bern(Pe)
        diag = c.k * self.rc**2 * self.dy                                   # fallout
        diag[:-1] += alpha
        diag[1:] -= beta
        ab = np.zeros((3, self.n_cells))
        ab[0, 1:] = beta
        ab[1] = diag
        ab[2, :-1] = -alpha
        return ab, Pe

    def _bin(self, X):
        """Integrate cell values over each bin (X has one row per cell, any number of columns)."""
        X = X[:, None] if X.ndim == 1 else X
        out = np.empty((self.n_bins, X.shape[1]))
        for b in range(self.n_bins):
            ja, jb, fa, fb = self._j[b], self._j[b + 1], self._f[b], self._f[b + 1]
            if ja == jb:
                out[b] = (fb - fa) * X[ja]
            else:
                out[b] = (1 - fa) * X[ja] + X[ja + 1:jb].sum(axis=0) + fb * X[jb]
        return out * self.dy

    def _check(self, D_x, s_bar):
        if not (np.isfinite(D_x) and D_x > 0):
            raise ValueError(f"D_x must be a positive number (got {D_x!r}).")
        if not np.isfinite(s_bar):
            raise ValueError(f"s_bar must be a finite number (got {s_bar!r}).")

    def density(self, D_x, s_bar):
        """Droplet density on the grid (unit injection).

        Returns (r, N): cell-center radii [um] and the number of droplets per unit
        y = (r + ell)^2 in each cell (per unit r^2 when ell = 0).
        """
        self._check(D_x, s_bar)
        ab, _ = self._assemble(D_x, s_bar)
        rhs = np.zeros(self.n_cells)
        rhs[0] = 1.0
        return self.rc, solve_banded((1, 1), ab, rhs)

    def evolve(self, D_x, s_bar, times, N0=None, injection=0.0, dt=0.5, return_density=False):
        """Time-dependent solution (Steve Krueger's backward-Euler solver).

        The steady system is A N = b. In time, each cell obeys dy dN/dt = b - A N, and
        backward Euler gives (dy/dt + A) N_new = (dy/dt) N_old + b.

        Parameters
        ----------
        times : increasing times [s] at which to report, all >= 0.
        N0 : density on the grid at t = 0 (default: empty).
        injection : particles per second entering at the smallest size.
        dt : largest time step [s]; steps are shortened to land exactly on each time.
        return_density : also return the density on the grid at each time.

        Returns
        -------
        (numbers, total): numbers[i, k] is the number of droplets in bin k at times[i];
        total[i] is the number of particles of any size in the chamber. With
        return_density, a third array holds the density on the grid at each time.
        """
        self._check(D_x, s_bar)
        times = np.asarray(times, float)
        if np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("times must be increasing and not negative.")
        if not dt > 0:
            raise ValueError("dt must be positive.")
        ab, _ = self._assemble(D_x, s_bar)
        N = np.zeros(self.n_cells) if N0 is None else np.array(N0, float)
        b = np.zeros(self.n_cells)
        b[0] = injection
        numbers, total = np.empty((len(times), self.n_bins)), np.empty(len(times))
        density = np.empty((len(times), self.n_cells)) if return_density else None
        t = 0.0
        for i, t_next in enumerate(times):
            gap = t_next - t
            if gap > 0:
                n_steps = int(np.ceil(gap / dt - 1e-9))
                h = gap / n_steps
                M = ab.copy()
                M[1] += self.dy / h
                for _ in range(n_steps):
                    N = solve_banded((1, 1), M, (self.dy / h) * N + b, check_finite=False)
                t = t_next
            numbers[i] = self._bin(N)[:, 0]
            total[i] = N.sum() * self.dy
            if return_density:
                density[i] = N
        return (numbers, total, density) if return_density else (numbers, total)

    def fractions(self, D_x, s_bar):
        """Fraction of detected droplets expected in each bin."""
        _, N = self.density(D_x, s_bar)
        m = self._bin(N)[:, 0]
        total = m.sum()
        if not total > 0:
            raise ValueError("The model puts (almost) no droplets in the bins for these parameters.")
        return m / total

    def fractions_and_sensitivities(self, D_x, s_bar):
        """Fractions p_k, and dp_k/d(ln D_x), dp_k/d(s_bar) as a (K, 2) array."""
        self._check(D_x, s_bar)
        ab, Pe = self._assemble(D_x, s_bar)
        rhs = np.zeros(self.n_cells)
        rhs[0] = 1.0
        N = solve_banded((1, 1), ab, rhs)

        # derivatives of each face flux, holding N fixed
        w = 1.0 + _bern_prime(Pe)
        dF_dlnD = (D_x / self.dy) * _diff_weight(Pe) * (N[:-1] - N[1:])
        dF_ds = self.cond.a * (w * N[:-1] + (1 - w) * N[1:])

        def minus_divergence(q):
            d = np.zeros(self.n_cells)
            d[:-1] -= q
            d[1:] += q
            return d

        S = solve_banded((1, 1), ab, np.column_stack([minus_divergence(dF_dlnD), minus_divergence(dF_ds)]))
        binned = self._bin(np.column_stack([N, S]))
        m, dm = binned[:, 0], binned[:, 1:]
        total = m.sum()
        if not total > 0:
            raise ValueError("The model puts (almost) no droplets in the bins for these parameters.")
        p = m / total
        dp = (dm - np.outer(p, dm.sum(axis=0))) / total
        return p, dp


@lru_cache(maxsize=16)
def _cached_model(cond, edges_key):
    return Model(cond, np.array(edges_key))


def get_model(cond, edges):
    """Model for these conditions and bin edges (reused if asked for again)."""
    return _cached_model(cond, tuple(np.round(np.asarray(edges, float), 12)))
