"""Tests for D01H, the dynamic dissolver on the F01H cake.

Each fixture solves the whole R01 -> F01 -> D01 chain once (about 40 s).

Units: mass flow [kg/s], concentration [mol/L], temperature [K], time [s].
"""

import warnings

import numpy as np
import pytest

# D01H puts CustomUnitOps on the path, so it is imported first.
import D01H

from continuous_dissolver import ContinuousDissolver  # noqa: E402
from continuous_filter import flow_from_vessel  # noqa: E402


def solve(dissolution=D01H.DISSOLUTION):
    original = D01H.DISSOLUTION
    D01H.DISSOLUTION = dissolution
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filt, tank = D01H.run_flowsheet()
    finally:
        D01H.DISSOLUTION = original
    return filt, tank, D01H.history(filt, tank)


@pytest.fixture(scope="module")
def default_run():
    return solve()


@pytest.fixture(scope="module")
def fast_run():
    return solve((200.0, 0.0, 1.0))


def test_residence_time_is_charge_over_feed(default_run):
    filt, _, profiles = default_run
    feed = (filt.cake_flow.liquid_mass_flow + filt.cake_flow.solid_mass_flow
            + D01H.SOLVENT_FLOW)

    assert D01H.residence_time(filt) == pytest.approx(D01H.INITIAL_MASS / feed)
    assert profiles["time"][-1] == pytest.approx(
        D01H.NUM_RESIDENCE_TIMES * D01H.residence_time(filt), rel=1e-6)


def test_tank_starts_as_pure_solvent(default_run):
    _, tank, profiles = default_run
    mass_j = np.asarray(tank.result.mass_j_liquid0)[0]

    assert mass_j.sum() == pytest.approx(D01H.INITIAL_MASS)
    assert profiles["crystals_held"][0] == pytest.approx(0.0, abs=1e-12)


def test_cake_crystals_enter_the_tank(default_run):
    """Guards the PharmaPy crystal-feed fix: without it this stays zero."""
    _, _, profiles = default_run
    assert profiles["crystals_held"][-1] > 1e-3  # kg


def test_reaches_steady_state(default_run):
    filt, _, profiles = default_run
    assert D01H.steady_state_gap(filt, profiles) < D01H.STEADY_STATE_TOL


def test_steady_mass_balance(default_run):
    filt, tank, _ = default_run
    *_, (_, cake, solvent, out) = [r for r in D01H.balance_rows(filt, tank)]

    assert out == pytest.approx(cake + solvent, rel=1e-4)


def test_only_f_changes_phase(default_run):
    filt, tank, _ = default_run
    out = flow_from_vessel(tank)
    names = list(out.name_species)
    cake = filt.cake_flow

    for name in ("A", "C"):
        j = names.index(name)
        assert out.liquid_mass_j[j] == pytest.approx(cake.liquid_mass_j[j],
                                                     rel=1e-3)


def test_dissolving_tank_stays_undersaturated(default_run):
    _, _, profiles = default_run
    assert np.all(profiles["conc_f"] <= profiles["solubility"] * (1 + 1e-9))


def test_fast_kinetics_reach_the_equilibrium_dissolver(fast_run):
    """Two independent models: with fast dissolution the dynamic tank's
    steady state must match the steady-state equilibrium dissolver."""
    filt, _, profiles = fast_run
    names = tuple(filt.feed.name_species)

    eq = ContinuousDissolver(
        solubility=D01H.CR01H.make_solubility_function(names),
        temp=D01H.DISSOLVER_TEMPERATURE)
    eq.Inlets = [filt.cake_flow, D01H.solvent_stream(names)]
    eq.solve()

    assert profiles["fraction_dissolved"][-1] == pytest.approx(
        eq.summary()["fraction_dissolved"], abs=0.005)
    assert profiles["conc_f"][-1] / profiles["solubility"][-1] > 0.99


def test_slower_kinetics_dissolve_less(default_run, fast_run):
    assert (default_run[2]["fraction_dissolved"][-1]
            < fast_run[2]["fraction_dissolved"][-1])


def test_figures_are_written(default_run, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    filt, tank, _ = default_run
    D01H.save_figures(filt, tank, out_dir=tmp_path)

    for name in ("D01_profiles", "D01_csd", "D01_settings",
                 "D01_mass_balance"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
