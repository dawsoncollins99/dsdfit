import pytest
from dsdfit import Conditions, predicted_D_x


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
                                 dict(alpha_c=0.0), dict(alpha_c=1.5), dict(alpha_c=0.04, alpha_T=0.0)])
def test_bad_inputs_raise(bad):
    with pytest.raises(ValueError):
        Conditions(**bad)


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
