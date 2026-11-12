"""Time-dependent model: how droplet numbers change over time.

The steady model assumes the chamber has settled. ``model_evolution`` follows the
same Fokker-Planck model in time instead (backward Euler, as in Steve Krueger's
time-dependent solver), for two kinds of experiment:

    start="pulse"   one particle is injected at t = 0 and nothing after,
                    for example a short burst of aerosol into the chamber;
    start="empty"   the chamber starts with no droplets and particles are
                    injected steadily, one per second, from t = 0 on.

The model is linear in the number of particles, so multiply the result by the
size of the pulse (or the injection rate) to get actual numbers.
"""
import numpy as np
from .solver import get_model


def model_evolution(D_x, s_bar, cond, edges, times, start="pulse", dt=0.5):
    """Expected number of droplets in each bin at each time.

    Parameters
    ----------
    D_x : float
        Broadening by supersaturation fluctuations [um^4/s].
    s_bar : float
        Mean supersaturation, as a fraction (0.0009 means 0.09%).
    cond : Conditions
        Physical conditions.
    edges : array
        Bin edges as droplet diameters [um].
    times : array
        Times [s] at which to report, increasing, all >= 0.
    start : "pulse" or "empty"
        "pulse": one particle injected at t = 0 (at the smallest size), none after.
        "empty": empty chamber at t = 0, one particle injected per second from then on.
    dt : float
        Largest time step [s]. Smaller is more accurate; 0.5 s is plenty for most uses.

    Returns
    -------
    (numbers, total): numbers has one row per time and one column per bin, the
    expected number of droplets in that bin; total is the expected number of
    particles of any size (including ones too small for the bins) still in the chamber.
    """
    model = get_model(cond, edges)
    if start == "pulse":
        N0 = np.zeros(model.n_cells)
        N0[0] = 1.0 / model.dy             # one particle in the smallest cell
        return model.evolve(D_x, s_bar, times, N0=N0, injection=0.0, dt=dt)
    if start == "empty":
        return model.evolve(D_x, s_bar, times, N0=None, injection=1.0, dt=dt)
    raise ValueError('start must be "pulse" or "empty".')
