"""Tests for D01R, the antisolvent crystallizer on the R02R outlet.

Each fixture solves the R02R -> D01R chain once (about a minute).

Units: mass flow [kg/s], concentration [mol/L], time [s].
"""

import warnings

import numpy as np
import pytest

# D01R puts R02R on the path, so it is imported first.
import D01R


def solve(antisolvent_flow=D01R.ANTISOLVENT_FLOW):
    original = D01R.ANTISOLVENT_FLOW
    D01R.ANTISOLVENT_FLOW = antisolvent_flow
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feed, tank = D01R.run_flowsheet()
            return (feed, tank, D01R.history(tank),
                    D01R.final_balance(tank))
    finally:
        D01R.ANTISOLVENT_FLOW = original


@pytest.fixture(scope="module")
def default_run():
    return solve()


@pytest.fixture(scope="module")
def no_antisolvent_run():
    return solve(antisolvent_flow=1.0e-9)


def test_residence_time_is_charge_over_feed(default_run):
    feed, _, profiles, _ = default_run
    flow = (sum(float(np.sum(s.mass_flow)) for s in feed.Streams)
            + D01R.ANTISOLVENT_FLOW)

    assert D01R.residence_time(feed) == pytest.approx(D01R.INITIAL_MASS / flow)
    assert profiles.time[-1] == pytest.approx(
        D01R.NUM_RESIDENCE_TIMES * D01R.residence_time(feed), rel=1e-6)


def test_tank_starts_as_pure_solvent_without_crystals(default_run):
    _, tank, profiles, _ = default_run
    mass_j = np.asarray(tank.result.mass_j_liquid0)[0]

    assert mass_j.sum() == pytest.approx(D01R.INITIAL_MASS)
    for species in D01R.CRYSTAL_SPECIES:
        assert profiles.crystal_mass[species][0] == pytest.approx(0.0,
                                                                  abs=1e-12)


def test_reaches_steady_state(default_run):
    *_, balance = default_run
    assert D01R.steady_state_gap(balance) < D01R.STEADY_STATE_TOL


def test_steady_mass_balance(default_run):
    *_, balance = default_run
    *_, (_, r02r, h, _, out, _) = D01R.balance_rows(balance)

    assert out == pytest.approx(r02r + h, rel=1e-3)


def test_d_e_j_crystallize_mostly(default_run):
    """K_B = 0.2 cuts the steady solubility to ~2 % of S0."""
    *_, balance = default_run
    names = D01R.species_names()

    for species in D01R.CRYSTAL_SPECIES:
        j = names.index(species)
        fraction = balance.to_solid[j] / balance.feeds[:, j].sum()
        assert 0.90 < fraction < 0.99, species


def test_crystals_stay_on_the_size_grid(default_run):
    """Under 0.1 % of each species' crystal mass sits in the top tenth of
    the grid, so next to nothing grows past its end."""
    _, tank, *_ = default_run
    grid = np.asarray(D01R.CRYSTAL_SIZE_GRID)
    top = grid > grid[0] + 0.9 * (grid[-1] - grid[0])

    for k, species in enumerate(D01R.CRYSTAL_SPECIES):
        distrib = np.asarray(getattr(tank.result, f"distrib_solid{k}"))[-1]
        mass = distrib * D01R.crystal_weight(tank, k)
        assert mass[top].sum() < 1e-3 * mass.sum(), species


def test_only_d_e_j_change_phase(default_run):
    *_, balance = default_run
    names = D01R.species_names()

    for j, name in enumerate(names):
        if name not in D01R.CRYSTAL_SPECIES:
            assert balance.to_solid[j] == pytest.approx(0.0, abs=1e-12), name


def test_crystals_gain_what_the_liquid_loses(default_run):
    *_, balance = default_run
    names = D01R.species_names()

    for species, terms in balance.crystal.items():
        lost = balance.to_solid[names.index(species)]
        assert terms["growth"] == pytest.approx(lost, rel=0.01), species


def test_no_antisolvent_no_crystals_at_steady_state(no_antisolvent_run):
    """R02R's outlet alone is below every S0, so whatever crystallizes
    against the B-rich charge at start-up must dissolve again."""
    _, _, profiles, _ = no_antisolvent_run

    for species in D01R.CRYSTAL_SPECIES:
        assert profiles.conc[species][-1] < profiles.solubility[species][-1]
        assert profiles.crystal_mass[species][-1] < 1e-9, species


def test_figures_are_written(default_run, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    feed, tank, _, balance = default_run
    D01R.save_figures(feed, tank, out_dir=tmp_path, balance=balance)

    for name in ("D01R_profiles", "D01R_csd", "D01R_settings",
                 "D01R_mass_balance"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
