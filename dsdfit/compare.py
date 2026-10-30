"""Compare a fitted D_x with what the supersaturation statistics predict."""


def predicted_D_x(sigma_s, tau, cond):
    """D_x predicted from supersaturation fluctuations: a^2 * sigma_s^2 * tau.

    Parameters
    ----------
    sigma_s : float
        Standard deviation of the supersaturation experienced by droplets, as a
        fraction (0.002 means 0.2%).
    tau : float
        Integral time scale of those fluctuations [s]: how long a fluctuation
        lasts (the area under the autocorrelation function).
    cond : Conditions
        Physical conditions (supplies the growth constant a).

    Returns
    -------
    D_x in um^4/s.

    This relation holds when the fluctuations are short-lived compared with the
    time a droplet takes to grow. See the guide for how to compute sigma_s and tau.
    """
    return cond.a**2 * sigma_s**2 * tau
