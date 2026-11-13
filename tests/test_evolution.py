"""The time-dependent model: exact identities with the steady model, and Steve Krueger's reference numbers."""
import numpy as np
import pytest
from dsdfit import Conditions, welas_bins, model_evolution
import dsdfit.solver as solver


@pytest.mark.parametrize("alpha_c,fallout_factor", [(None, 1.0), (0.04, 1.0), (0.04, 1.3)])
def test_pulse_summed_over_time_equals_steady(alpha_c, fallout_factor):
    """A pulse of one particle, summed over all time, is the steady solution for one particle per second.
    With backward Euler this holds exactly, for any time step."""
    cond, edges = Conditions(alpha_c=alpha_c, fallout_factor=fallout_factor), welas_bins()
    dt = 10.0
    times = np.arange(dt, 4000.0 + dt / 2, dt)
    numbers, total = model_evolution(0.3, 0.003, cond, edges, times, start="pulse", dt=dt)
    assert total[-1] < 1e-12                                  # everything has fallen out
    model = solver.get_model(cond, edges)
    steady = model._bin(model.density(0.3, 0.003)[1])[:, 0]
    assert np.allclose(numbers.sum(axis=0) * dt, steady, rtol=1e-8, atol=1e-12 * steady.max())


def test_empty_chamber_settles_to_steady():
    cond, edges = Conditions(), welas_bins()
    numbers, _ = model_evolution(0.3, 0.003, cond, edges, [4000.0], start="empty", dt=10.0)
    model = solver.get_model(cond, edges)
    steady = model._bin(model.density(0.3, 0.003)[1])[:, 0]
    assert np.allclose(numbers[0], steady, rtol=1e-8, atol=1e-12 * steady.max())


def test_matches_steve_krueger_reference(monkeypatch):
    """README_FP_time_v2: 8000 cells up to r = 30 um, empty start, dt = 1 s, s = 0.09%, D_x = 1e-6.
    Reference: steady mean radius 4.206 um; relative L1 distance to steady 6.11e-2 at 500 s, 7.87e-5 at 1000 s."""
    x_min = 0.0325**2
    monkeypatch.setattr(solver, "CELL_WIDTH", (900.0 - x_min) / 8000 * (1 + 1e-12))
    model = solver.Model(Conditions(T=288.0, p=1e5, d_dry=0.13, H=1.0), np.array([0.1, 40.0]))
    assert model.n_cells == 8000
    _, N_ss = model.density(1e-6, 0.0009)
    assert np.sum(model.rc * N_ss) / np.sum(N_ss) == pytest.approx(4.206, abs=5e-4)
    _, _, N = model.evolve(1e-6, 0.0009, [500.0, 1000.0], injection=1.0, dt=1.0, return_density=True)
    l1 = np.abs(N - N_ss).sum(axis=1) / N_ss.sum()
    assert l1[0] == pytest.approx(6.11e-2, rel=1e-2)
    assert l1[1] == pytest.approx(7.87e-5, rel=1e-2)


def test_bad_start_raises():
    with pytest.raises(ValueError):
        model_evolution(0.3, 0.003, Conditions(), welas_bins(), [10.0], start="burst")
