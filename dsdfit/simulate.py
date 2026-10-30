"""Model predictions and synthetic histograms."""
import numpy as np
from .solver import get_model
from .histogram import Histogram


def model_fractions(D_x, s_bar, cond, edges):
    """Fraction of detected droplets the model expects in each bin.

    Parameters
    ----------
    D_x : float
        Broadening by supersaturation fluctuations [um^4/s].
    s_bar : float
        Mean supersaturation, as a fraction (0.0009 means 0.09%).
    cond : Conditions
        Physical conditions.
    edges : array
        Bin edges as droplet diameters [um] (for example ``welas_bins()``).

    Returns
    -------
    array with one fraction per bin; the fractions sum to 1.
    """
    return get_model(cond, edges).fractions(D_x, s_bar)


def simulate(D_x, s_bar, cond, edges, n, seed=None):
    """Make a synthetic histogram: solve the model, then sample n droplets from it.

    Each of the n droplets lands in bin k with the probability the model predicts,
    independently of the others.

    Parameters
    ----------
    D_x : float
        True broadening by supersaturation fluctuations [um^4/s].
    s_bar : float
        True mean supersaturation, as a fraction (0.0009 means 0.09%).
    cond : Conditions
        True physical conditions.
    edges : array
        Bin edges as droplet diameters [um] (for example ``welas_bins()``).
    n : int
        Number of droplets to count.
    seed : int, optional
        Random seed, for repeatable results.

    Returns
    -------
    Histogram, which also remembers the true D_x and s_bar (``hist.truth``) and
    the exact model fractions (``hist.true_fractions``).
    """
    n = int(n)
    if n <= 0:
        raise ValueError("simulate: n must be a positive number of droplets.")
    p = model_fractions(D_x, s_bar, cond, edges)
    counts = np.random.default_rng(seed).multinomial(n, p / p.sum())
    return Histogram(edges, counts, truth={"D_x": D_x, "s_bar": s_bar}, true_fractions=p)
