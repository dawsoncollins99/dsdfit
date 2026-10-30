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


@pytest.mark.parametrize("bad", [dict(T=-1.0), dict(p=0.0), dict(d_dry=-0.1), dict(H=float("nan")), dict(T=400.0)])
def test_bad_inputs_raise(bad):
    with pytest.raises(ValueError):
        Conditions(**bad)


def test_predicted_D_x():
    c = Conditions()
    assert predicted_D_x(0.002, 1.5, c) == pytest.approx(c.a**2 * 0.002**2 * 1.5)
