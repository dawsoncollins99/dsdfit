"""Best case: the same model makes and fits the data. The truth must be recovered."""
import numpy as np
import pytest
from dsdfit import Conditions, welas_bins, simulate, fit


@pytest.mark.parametrize("alpha_c", [None, 0.04])
def test_recovery_best_case(alpha_c):
    cond, edges = Conditions(alpha_c=alpha_c), welas_bins()
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


def test_recovery_with_size_dependent_fallout():
    """Data made with faster fallout for big droplets, fitted with the same fallout: the truth is recovered.
    Fitted with plain k r^2 fallout instead, s_bar comes out too low."""
    cond = Conditions(alpha_c=0.04, fallout_factor=lambda r: 1 + 0.005 * r**2)
    edges = welas_bins(2.0)
    zD, zS, s_plain = [], [], []
    for seed in range(5):
        h = simulate(0.3, 0.0009, cond, edges, n=10**5, seed=seed)
        r = fit(h, cond)
        assert r.converged
        zD.append((r.D_x - 0.3) / r.D_x_err)
        zS.append((r.s_bar - 0.0009) / r.s_bar_err)
        s_plain.append(fit(h, Conditions(alpha_c=0.04)).s_bar)
    assert np.all(np.abs(zD) < 4) and np.all(np.abs(zS) < 4)
    assert np.all(np.array(s_plain) < 0.0009)


def test_fit_check_flags_wrong_physics():
    """Fitting with a badly wrong chamber height should fail the fit check at large n."""
    edges = welas_bins()
    h = simulate(0.3, 0.0009, Conditions(H=1.0), edges, n=10**6, seed=0)
    assert fit(h, Conditions(H=1.0)).fit_ok
    assert not fit(h, Conditions(H=2.0)).fit_ok


def test_fit_reaches_the_minimum_when_the_model_is_wrong():
    """Data made with the kinetic correction, fitted without it: the fit must still find the lowest J."""
    import math
    from scipy.optimize import minimize
    from dsdfit.solver import get_model
    from dsdfit.fit import _deviance
    edges = welas_bins()
    h = simulate(0.3, 0.003, Conditions(alpha_c=0.04), edges, n=10**5, seed=0)
    r = fit(h, Conditions())
    assert r.converged
    model, counts = get_model(Conditions(), edges), h.counts.astype(float)
    J = lambda th: _deviance(counts, counts.sum(), model.fractions(math.exp(th[0]), th[1]))
    best = minimize(J, [math.log(r.D_x), r.s_bar], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "initial_simplex": [
                        [math.log(r.D_x) - 0.3, r.s_bar], [math.log(r.D_x) + 0.3, r.s_bar + 2e-4],
                        [math.log(r.D_x), r.s_bar - 2e-4]]})
    assert r.J_min <= best.fun + 1e-3
