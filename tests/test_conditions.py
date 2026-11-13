import numpy as np
import pytest
from dsdfit import Conditions, predicted_D_x, model_fractions, welas_bins


def test_defaults_match_matlab_constants():
    c = Conditions()
    assert c.a == pytest.approx(205.8, rel=1e-3)
    assert c.A_k == pytest.approx(1.128e-3, rel=1e-3)
    assert c.B_k == pytest.approx(3.650e-4, rel=1e-3)
    assert c.k == pytest.approx(1.2097e-4, rel=1e-3)
    assert c.critical_supersaturation == pytest.approx(7.63e-4, rel=1e-2)


def test_fallout_scales_with_height():
    assert Conditions(H=2.0).k == pytest.approx(Conditions(H=1.0).k / 2)


@pytest.mark.parametrize("bad", [dict(T=-1.0), dict(p=0.0), dict(d_dry=-0.1), dict(H=float("nan")), dict(T=400.0),
                                 dict(alpha_c=0.0), dict(alpha_c=1.5), dict(alpha_c=0.04, alpha_T=0.0),
                                 dict(fallout_factor=0.0), dict(fallout_factor=-1.0),
                                 dict(fallout_factor=float("nan")), dict(fallout_factor="fast")])
def test_bad_inputs_raise(bad):
    with pytest.raises(ValueError):
        Conditions(**bad)


def test_default_fallout_rate_is_k_r_squared():
    c = Conditions()
    r = np.array([1.0, 5.0, 12.0])
    assert np.allclose(c.fallout_rate(r), c.k * r**2)


def test_constant_fallout_factor_is_like_a_shorter_chamber():
    """Doubling the fallout rate at every size is the same as halving the chamber height."""
    edges = welas_bins()
    p_factor = model_fractions(0.3, 0.0009, Conditions(fallout_factor=2.0), edges)
    p_short = model_fractions(0.3, 0.0009, Conditions(H=0.5), edges)
    assert np.allclose(p_factor, p_short, rtol=1e-10, atol=0)


def test_fallout_factor_number_and_function_agree():
    edges = welas_bins()
    p_number = model_fractions(0.3, 0.0009, Conditions(fallout_factor=1.3), edges)
    p_function = model_fractions(0.3, 0.0009, Conditions(fallout_factor=lambda r: 1.3), edges)
    assert np.array_equal(p_number, p_function)


def test_size_dependent_fallout_removes_big_droplets():
    """Faster fallout for big droplets leaves relatively fewer of them."""
    edges = welas_bins()
    p_plain = model_fractions(0.3, 0.0009, Conditions(), edges)
    p_fast = model_fractions(0.3, 0.0009, Conditions(fallout_factor=lambda r: 1 + 0.01 * r**2), edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    assert np.sum(mid * p_fast) < np.sum(mid * p_plain)


def test_fallout_factor_must_be_positive_at_every_size():
    with pytest.raises(ValueError):
        model_fractions(0.3, 0.0009, Conditions(fallout_factor=lambda r: 1 - 0.01 * r**2), welas_bins())


def test_predicted_D_x():
    c = Conditions()
    assert predicted_D_x(0.002, 1.5, c) == pytest.approx(c.a**2 * 0.002**2 * 1.5)


def test_no_kinetic_correction_by_default():
    assert Conditions().ell == 0.0


def test_kinetic_length_matches_codt():
    """C-ODT (alpha_c = 0.04, alpha_T = 1) at 288 K: about 1.5 um, as measured from its trajectories."""
    c = Conditions(alpha_c=0.04, alpha_T=1.0)
    assert c.ell == pytest.approx(1.53, abs=0.03)
    assert c.a == Conditions().a                    # the growth constant itself does not change


def test_kinetic_length_shrinks_as_more_molecules_stick():
    ells = [Conditions(alpha_c=x).ell for x in (0.01, 0.04, 0.1, 0.3, 1.0)]
    assert all(e1 > e2 for e1, e2 in zip(ells, ells[1:]))
    assert Conditions(alpha_c=1.0).ell == pytest.approx(0.2, abs=0.02)
