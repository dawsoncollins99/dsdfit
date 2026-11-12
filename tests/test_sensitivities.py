import math
import numpy as np
import pytest
from dsdfit import Conditions, welas_bins
from dsdfit.solver import get_model


@pytest.mark.parametrize("alpha_c", [None, 0.04])
@pytest.mark.parametrize("D_x,s_bar", [(0.3, 0.0009), (1.0, 0.0001), (1.0, 0.003)])
def test_tangent_matches_finite_differences(D_x, s_bar, alpha_c):
    model = get_model(Conditions(alpha_c=alpha_c), welas_bins())
    p, dp = model.fractions_and_sensitivities(D_x, s_bar)
    h = 1e-4
    fd_lnD = (model.fractions(D_x * math.exp(h), s_bar) - model.fractions(D_x * math.exp(-h), s_bar)) / (2 * h)
    hs = 1e-7
    fd_s = (model.fractions(D_x, s_bar + hs) - model.fractions(D_x, s_bar - hs)) / (2 * hs)
    big = p > 1e-4 * p.max()
    for tangent, fd in [(dp[:, 0], fd_lnD), (dp[:, 1], fd_s)]:
        scale = np.max(np.abs(tangent[big]))
        assert np.max(np.abs(tangent[big] - fd[big])) < 1e-4 * scale


@pytest.mark.parametrize("alpha_c", [None, 0.04])
def test_fractions_sum_to_one_and_sensitivities_to_zero(alpha_c):
    model = get_model(Conditions(alpha_c=alpha_c), welas_bins())
    p, dp = model.fractions_and_sensitivities(0.3, 0.0009)
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(dp.sum(axis=0), 0.0, atol=1e-10 * np.abs(dp).max())
