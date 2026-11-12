"""Estimate D_x and s_bar from a histogram.

What ``fit`` does, in words: it searches for the D_x and s_bar whose predicted
size distribution best matches the histogram, and reports how uncertain those
values are. "Best match" means the largest likelihood: the parameters under
which the observed counts are most probable. The details (the loss, the
Gauss-Newton search and the error bars) are in the appendix of the guide.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.stats import poisson
from .solver import get_model

# starting grid: the fit begins from whichever of these shapes best matches the data
_START_D = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
_START_S = (-0.0005, 0.0, 0.0003, 0.0006, 0.0009, 0.0012, 0.002, 0.003, 0.0045, 0.007, 0.01, 0.015)
# allowed range, and the largest step allowed per iteration, for (ln D_x, s_bar)
_LOW = np.array([math.log(1e-4), -0.005])
_HIGH = np.array([math.log(1e4), 0.02])
_MAX_STEP = np.array([1.0, 5e-4])


@dataclass
class FitResult:
    """Result of ``fit``.

    Attributes
    ----------
    D_x, s_bar          estimates (D_x in um^4/s, s_bar as a fraction)
    D_x_err, s_bar_err  error bars: one standard deviation (about 68% confidence)
    correlation         correlation between the D_x and s_bar estimates (-1 to 1)
    J_min               the loss at the best fit (the fit check, see below)
    J_expected          the value J_min should have, roughly, if the model fits
    J_spread            how much J_min typically varies around J_expected
    fit_ok              True if J_min is within 3 spreads of J_expected
    n                   number of droplets in the histogram
    edges, fractions    bins and the fitted model's fraction in each bin
    iterations, converged
    """
    D_x: float
    D_x_err: float
    s_bar: float
    s_bar_err: float
    correlation: float
    J_min: float
    J_expected: float
    J_spread: float
    n: int
    iterations: int
    converged: bool
    edges: np.ndarray = field(repr=False)
    fractions: np.ndarray = field(repr=False)

    @property
    def fit_ok(self):
        return self.J_min <= self.J_expected + 3.0 * self.J_spread

    def summary(self):
        check = ("consistent with the model" if self.fit_ok else
                 "MUCH LARGER than expected: the model does not describe these data well")
        lines = [
            f"D_x   = {self.D_x:.4g} +/- {self.D_x_err:.2g} um^4/s   ({100 * self.D_x_err / self.D_x:.1f}%)",
            f"s_bar = {self.s_bar:.5g} +/- {self.s_bar_err:.2g}      ({100 * self.s_bar:.4g}% +/- "
            f"{100 * self.s_bar_err:.2g} percentage points)",
            f"correlation between the two estimates: {self.correlation:+.2f}",
            f"fit check: J_min = {self.J_min:.1f}, expected about {self.J_expected:.0f} +/- "
            f"{self.J_spread:.0f} -> {check}",
            f"({self.n} droplets, {self.iterations} iterations"
            + ("" if self.converged else ", DID NOT CONVERGE") + ")",
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def _deviance(counts, n, p):
    """The loss J = 2 sum_k c_k ln(c_k / (n p_k))."""
    have = counts > 0
    return 2.0 * float(np.sum(counts[have] * np.log(counts[have] / (n * np.maximum(p[have], 1e-300)))))


def _expected_unit_deviance(mu):
    """Average contribution of a bin with expected count mu to J (about 1 for large mu)."""
    out = np.empty(len(mu))
    for i, m in enumerate(mu):
        if m < 1e-8:                     # a bin that essentially never sees a droplet
            out[i] = 0.0
            continue
        if m > 200:
            out[i] = 1.0 + 1.0 / (6.0 * m)
            continue
        hi = int(m + 12 * math.sqrt(m) + 20)
        c = np.arange(hi + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(c > 0, c * np.log(c / m), 0.0) - c + m
        out[i] = 2.0 * np.sum(poisson.pmf(c, m) * d)
    return out


def _start(model, counts, n):
    """Best starting point on a coarse grid of (D_x, s_bar)."""
    if not hasattr(model, "_start_shapes"):
        shapes = []
        for D in _START_D:
            for s in _START_S:
                try:
                    shapes.append((D, s, model.fractions(D, s)))
                except ValueError:
                    pass
        model._start_shapes = shapes
    D, s, _ = min(model._start_shapes, key=lambda item: _deviance(counts, n, item[2]))
    return np.array([math.log(D), s])


def fit(hist, cond, start=None, tol=1e-8, max_iter=100):
    """Estimate D_x and s_bar, with error bars, from a histogram.

    Parameters
    ----------
    hist : Histogram
        The observed (or simulated) droplet counts.
    cond : Conditions
        The physical conditions the fit should assume.
    start : (D_x, s_bar), optional
        Starting guess. By default the fit picks one automatically.

    Returns
    -------
    FitResult. ``print(result)`` gives a short summary.
    """
    model = get_model(cond, hist.edges)
    counts = hist.counts.astype(float)
    n = float(counts.sum())

    theta = _start(model, counts, n) if start is None else np.array([math.log(start[0]), start[1]])
    theta = np.clip(theta, _LOW, _HIGH)

    converged = False
    for iteration in range(1, max_iter + 1):
        p, dp = model.fractions_and_sensitivities(math.exp(theta[0]), theta[1])
        use = p > 1e-14 * p.max()
        A = n * (dp[use].T / p[use]) @ dp[use]                      # n x Fisher information
        b = dp[use].T @ ((counts[use] - n * p[use]) / p[use])       # (minus half the gradient of J)
        try:
            step = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise RuntimeError("fit: D_x and s_bar cannot be told apart for these data "
                               "(the information matrix is singular).") from None
        # a parameter sitting at its limit and being pushed past it stays put; solve for the other one
        stuck = ((theta <= _LOW) & (step < 0)) | ((theta >= _HIGH) & (step > 0))
        if stuck.any():
            free = ~stuck
            step = np.zeros(2)
            if free.any():
                step[free] = np.linalg.solve(A[np.ix_(free, free)], b[free])
        predicted = float(b @ step)                                 # predicted decrease of J
        if predicted < tol:
            converged = True
            break
        # shorten a step that is too long, keeping its direction (so it still goes downhill)
        step = step * min(1.0, float(np.min(_MAX_STEP / np.maximum(np.abs(step), 1e-300))))
        J_now = _deviance(counts, n, p)
        t = 1.0
        while True:                                                 # halve the step until J goes down
            trial = np.clip(theta + t * step, _LOW, _HIGH)
            J_trial = _deviance(counts, n, model.fractions(math.exp(trial[0]), trial[1]))
            if J_trial < J_now or t < 1e-6:
                break
            t /= 2
        if t < 1e-6:                    # no downhill step found: converged only if already at the bottom
            converged = predicted < 1e-3
            break
        theta = trial

    p, dp = model.fractions_and_sensitivities(math.exp(theta[0]), theta[1])
    use = p > 1e-14 * p.max()
    Sigma = np.linalg.inv(n * (dp[use].T / p[use]) @ dp[use])
    sd = np.sqrt(np.diag(Sigma))
    D_x = math.exp(theta[0])
    J_exp = float(np.sum(_expected_unit_deviance(n * p)) - 3.0)
    return FitResult(
        D_x=D_x, D_x_err=D_x * sd[0], s_bar=float(theta[1]), s_bar_err=float(sd[1]),
        correlation=float(Sigma[0, 1] / (sd[0] * sd[1])),
        J_min=_deviance(counts, n, p), J_expected=J_exp, J_spread=math.sqrt(2.0 * max(J_exp, 1.0)),
        n=int(n), iterations=iteration, converged=converged,
        edges=np.array(hist.edges), fractions=p,
    )
