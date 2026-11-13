"""The time-dependent model: identities with the steady model, accuracy of the time stepping,
and Steve Krueger's reference numbers."""
import numpy as np
import pytest
from dsdfit import Conditions, welas_bins, model_evolution
import dsdfit.solver as solver


def _pulse(model):
    N0 = np.zeros(model.n_cells)
    N0[0] = 1.0 / model.dy
    return N0


@pytest.mark.parametrize("alpha_c,fallout_factor", [(None, 1.0), (0.04, 1.0), (0.04, 1.3)])
def test_pulse_summed_over_time_equals_steady_backward_euler(alpha_c, fallout_factor):
    """A pulse of one particle, summed over all time, is the steady solution for one particle per second.
    With backward Euler this holds exactly, for any time step."""
    cond, edges = Conditions(alpha_c=alpha_c, fallout_factor=fallout_factor), welas_bins()
    model = solver.get_model(cond, edges)
    dt = 10.0
    times = np.arange(dt, 4000.0 + dt / 2, dt)
    numbers, total = model.evolve(0.3, 0.003, times, N0=_pulse(model), dt=dt, scheme="backward_euler")
    assert total[-1] < 1e-12                                  # everything has fallen out
    steady = model._bin(model.density(0.3, 0.003)[1])[:, 0]
    assert np.allclose(numbers.sum(axis=0) * dt, steady, rtol=1e-8, atol=1e-12 * steady.max())


def test_pulse_summed_over_time_approaches_steady_trbdf2(monkeypatch):
    """With the default scheme the same identity holds up to a time-step error that shrinks with the step."""
    monkeypatch.setattr(solver, "CELL_WIDTH", 0.05)            # coarse grid, to keep the test fast
    model = solver.Model(Conditions(alpha_c=0.04), welas_bins())
    steady = model._bin(model.density(0.3, 0.003)[1])[:, 0]
    errors = []
    for dt in (8.0, 4.0, 2.0):
        times = np.arange(dt, 4000.0 + dt / 2, dt)
        numbers, total = model.evolve(0.3, 0.003, times, N0=_pulse(model), dt=dt)
        assert abs(total[-1]) < 1e-12
        errors.append(np.max(np.abs(numbers.sum(axis=0) * dt - steady)) / steady.max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 5e-3


def test_default_scheme_adds_no_artificial_spreading(monkeypatch):
    """Spread of y = (r + ell)^2 for a pulse after 60 s, at C-ODT-like conditions. The default scheme with
    1 s steps agrees with 0.25 s steps; backward Euler with 0.5 s steps spreads the pulse too much
    (artificial spreading of about v^2 dt / 2)."""
    monkeypatch.setattr(solver, "CELL_WIDTH", 0.02)
    model = solver.Model(Conditions(alpha_c=0.04), np.array([2.0, 40.0]))
    big = model.rc >= 1.0

    def spread(dt, scheme):
        _, _, dens = model.evolve(0.57, 0.0031, [60.0], N0=_pulse(model), dt=dt, return_density=True, scheme=scheme)
        w, y = dens[0][big], model.yc[big]
        mean = (w * y).sum() / w.sum()
        return (w * (y - mean) ** 2).sum() / w.sum()

    reference = spread(0.25, "trbdf2")
    assert spread(1.0, "trbdf2") == pytest.approx(reference, rel=2e-3)
    assert spread(0.5, "backward_euler") > 1.05 * reference


def test_empty_chamber_settles_to_steady():
    cond, edges = Conditions(), welas_bins()
    numbers, _ = model_evolution(0.3, 0.003, cond, edges, [4000.0], start="empty", dt=10.0)
    model = solver.get_model(cond, edges)
    steady = model._bin(model.density(0.3, 0.003)[1])[:, 0]
    assert np.allclose(numbers[0], steady, rtol=1e-8, atol=1e-12 * steady.max())


def test_matches_steve_krueger_reference(monkeypatch):
    """README_FP_time_v2: 8000 cells up to r = 30 um, empty start, dt = 1 s (backward Euler), s = 0.09%,
    D_x = 1e-6. Reference: steady mean radius 4.206 um; relative L1 distance to steady 6.11e-2 at 500 s,
    7.87e-5 at 1000 s."""
    x_min = 0.0325**2
    monkeypatch.setattr(solver, "CELL_WIDTH", (900.0 - x_min) / 8000 * (1 + 1e-12))
    model = solver.Model(Conditions(T=288.0, p=1e5, d_dry=0.13, H=1.0), np.array([0.1, 40.0]))
    assert model.n_cells == 8000
    _, N_ss = model.density(1e-6, 0.0009)
    assert np.sum(model.rc * N_ss) / np.sum(N_ss) == pytest.approx(4.206, abs=5e-4)
    _, _, N = model.evolve(1e-6, 0.0009, [500.0, 1000.0], injection=1.0, dt=1.0, return_density=True,
                           scheme="backward_euler")
    l1 = np.abs(N - N_ss).sum(axis=1) / N_ss.sum()
    assert l1[0] == pytest.approx(6.11e-2, rel=1e-2)
    assert l1[1] == pytest.approx(7.87e-5, rel=1e-2)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        model_evolution(0.3, 0.003, Conditions(), welas_bins(), [10.0], start="burst")
    model = solver.get_model(Conditions(), welas_bins())
    with pytest.raises(ValueError):
        model.evolve(0.3, 0.003, [10.0], scheme="rk4")
