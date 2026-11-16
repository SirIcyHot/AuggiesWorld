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
    """0.02 kg/s of B crystallizes ~99 % or more of each species."""
    *_, balance = default_run
    names = D01R.species_names()

    for species in D01R.CRYSTAL_SPECIES:
        j = names.index(species)
        fraction = balance.to_solid[j] / balance.feeds[:, j].sum()
        assert 0.98 < fraction < 1.0, species


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


def feed_conc(feed, species):
    """R02R outlet concentration of ``species`` [mol/L]."""
    names = list(D01R.species_names())
    conc = np.asarray(feed.Streams[0].mole_conc, dtype=float).ravel()
    return conc[names.index(species)]


def test_without_antisolvent_only_supersaturated_species_crystallize(
        no_antisolvent_run, default_run):
    """Without B, a species R02R delivers below S0 (J) must redissolve
    whatever crystallized against the B-rich charge at start-up, while one
    R02R delivers above S0 (D, E: assumption A4, R02R forms no solids)
    crystallizes down to S0 on its own, and less than with B."""
    feed, _, profiles, _ = no_antisolvent_run
    *_, with_b = default_run
    names = D01R.species_names()

    for species in D01R.CRYSTAL_SPECIES:
        conc = profiles.conc[species][-1]
        solubility = profiles.solubility[species][-1]
        if feed_conc(feed, species) < D01R.SOLUBILITY_S0[species]:
            assert conc < solubility, species
            assert profiles.crystal_mass[species][-1] < 1e-9, species
        else:
            assert conc == pytest.approx(solubility, rel=0.02), species
            j = names.index(species)
            assert (profiles.fraction_crystallized[species][-1]
                    < with_b.to_solid[j] / with_b.feeds[:, j].sum()), species


def test_figures_are_written(default_run, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    feed, tank, _, balance = default_run
    D01R.save_figures(feed, tank, out_dir=tmp_path, balance=balance)

    for name in ("D01R_profiles", "D01R_csd", "D01R_settings",
                 "D01R_mass_balance"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
