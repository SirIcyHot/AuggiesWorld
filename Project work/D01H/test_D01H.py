"""Tests for D01H, the H dissolver on the F01H cake.

Each fixture solves the R01H -> F01H -> D01H chain once.

Units: mass flow [kg/s], concentration [mol/L], time [s].
"""

import warnings

import numpy as np
import pytest

# D01H puts F01H, R01H and CustomUnitOps on the path, so it is imported first.
import D01H
import F01H


def solve(h_flow=D01H.H_FLOW, k_h=D01H.ANTISOLVENT_K_H):
    originals = D01H.H_FLOW, D01H.ANTISOLVENT_K_H
    D01H.H_FLOW, D01H.ANTISOLVENT_K_H = h_flow, k_h
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cake, cake_flow, tank = D01H.run_flowsheet()
            return (cake, cake_flow, tank, D01H.history(tank, cake_flow),
                    D01H.final_balance(tank))
    finally:
        D01H.H_FLOW, D01H.ANTISOLVENT_K_H = originals


@pytest.fixture(scope="module")
def default_run():
    return solve()


@pytest.fixture(scope="module")
def no_antisolvent_effect_run():
    """K_H = 0: H only dilutes, so more F dissolves."""
    return solve(k_h=0.0)


def test_regrid_keeps_the_cake(default_run):
    _, cake_flow, *_ = default_run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        *_, filt = F01H.run_flowsheet()

    assert cake_flow.solid_mass_flow == pytest.approx(
        filt.cake_flow.solid_mass_flow, rel=1e-12)
    np.testing.assert_array_equal(cake_flow.liquid_mass_j,
                                  filt.cake_flow.liquid_mass_j)
    np.testing.assert_array_equal(cake_flow.x_grid, D01H.CRYSTAL_SIZE_GRID)


def test_residence_time_is_charge_over_feed(default_run):
    cake, cake_flow, _, profiles, _ = default_run
    flow = (cake_flow.solid_mass_flow + cake_flow.liquid_mass_flow
            + D01H.H_FLOW)

    assert D01H.residence_time(cake) == pytest.approx(
        D01H.INITIAL_MASS / flow, rel=1e-9)
    assert profiles.time[-1] == pytest.approx(
        D01H.NUM_RESIDENCE_TIMES * D01H.residence_time(cake), rel=1e-6)


def test_tank_receives_the_cake_crystals(default_run):
    _, cake_flow, _, _, balance = default_run
    assert balance.crystal["inlet"] == pytest.approx(
        cake_flow.solid_mass_flow, rel=1e-3)


def test_reaches_steady_state(default_run):
    *_, balance = default_run
    assert D01H.steady_state_gap(balance) < D01H.STEADY_STATE_TOL


def test_steady_mass_balance(default_run):
    *_, balance = default_run
    *_, (_, cake, h, _, out, _) = D01H.balance_rows(balance)

    assert out == pytest.approx(cake + h, rel=1e-4)


def test_f_dissolves_marginally(default_run):
    *_, balance = default_run
    assert 0.005 < D01H.fraction_dissolved(balance) < 0.10


def test_liquid_ends_just_below_saturation(default_run):
    *_, profiles, _ = default_run
    sigma = profiles.conc[-1] / profiles.solubility[-1] - 1
    assert -1e-2 < sigma < 0


def test_crystals_lose_what_the_liquid_gains(default_run):
    *_, balance = default_run
    names = D01H.species_names()
    gained = balance.to_liquid[names.index(D01H.CRYSTAL_SPECIES)]

    assert balance.crystal["dissolved"] == pytest.approx(gained, rel=0.01)


def test_only_f_changes_phase(default_run):
    *_, balance = default_run
    for j, name in enumerate(D01H.species_names()):
        if name != D01H.CRYSTAL_SPECIES:
            assert balance.to_liquid[j] == pytest.approx(0.0, abs=1e-15), name


def test_antisolvent_holds_back_dissolution(default_run,
                                            no_antisolvent_effect_run):
    *_, with_k = default_run
    *_, without_k = no_antisolvent_effect_run

    assert (D01H.fraction_dissolved(without_k)
            > 2 * D01H.fraction_dissolved(with_k))


def test_figures_are_written(default_run, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    cake, cake_flow, tank, _, balance = default_run
    D01H.save_figures(cake, cake_flow, tank, out_dir=tmp_path,
                      balance=balance)

    for name in ("D01H_profiles", "D01H_csd", "D01H_settings",
                 "D01H_mass_balance"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
