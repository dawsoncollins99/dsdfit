"""Best case: the same model makes and fits the data. The truth must be recovered."""
import numpy as np
from dsdfit import Conditions, welas_bins, simulate, fit


def test_recovery_best_case():
    cond, edges = Conditions(), welas_bins()
    zD, zS = [], []
    for seed in range(10):
        h = simulate(0.3, 0.0009, cond, edges, n=10**5, seed=seed)
        r = fit(h, cond)
        assert r.converged
        zD.append((r.D_x - 0.3) / r.D_x_err)
        zS.append((r.s_bar - 0.0009) / r.s_bar_err)
    zD, zS = np.array(zD), np.array(zS)
    assert np.all(np.abs(zD) < 4) and np.all(np.abs(zS) < 4)       # every fit within 4 error bars
    assert abs(zD.mean()) < 1.5 and abs(zS.mean()) < 1.5            # no systematic offset


def test_fit_check_flags_wrong_physics():
    """Fitting with a badly wrong chamber height should fail the fit check at large n."""
    edges = welas_bins()
    h = simulate(0.3, 0.0009, Conditions(H=1.0), edges, n=10**6, seed=0)
    assert fit(h, Conditions(H=1.0)).fit_ok
    assert not fit(h, Conditions(H=2.0)).fit_ok
