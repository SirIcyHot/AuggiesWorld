"""Tests for D02R, the L dissolver on the F01R cake.

Each fixture solves the R02R -> F01R -> D02R chain once.

Units: mass flow [kg/s], concentration [mol/L], time [s].
"""

import json
import warnings

import numpy as np
import pytest

# D02R puts F01R, R02R and CustomUnitOps on the path, so it is imported first.
import D02R


def solve(**settings):
    originals = {name: getattr(D02R, name) for name in settings}
    for name, value in settings.items():
        setattr(D02R, name, value)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cake, cake_flow, tank = D02R.run_flowsheet()
            return (cake, cake_flow, tank, D02R.history(tank, cake_flow),
                    D02R.final_balance(tank))
    finally:
        for name, value in originals.items():
            setattr(D02R, name, value)


@pytest.fixture(scope="module")
def default_run():
    return solve()


@pytest.fixture(scope="module")
def no_solvent_effect_run():
    """K_L = 0: L only dilutes, so far less dissolves."""
    return solve(SOLVENT_K_L=0.0)


def test_l_is_in_the_database():
    with open(D02R.DB_PATH) as handle:
        entry = json.load(handle)["L"]
    for key in ("mw", "cp_liq", "rho_liq", "rho_solid", "visc_liq",
                "surf_tension", "p_vap"):
        assert key in entry
    assert "L" in D02R.species_names()


def test_split_cake_keeps_each_species_crystals(default_run):
    _, cake_flow, *_ = default_run
    densities = D02R.F01R.solid_densities(D02R.DB_PATH)
    numbers = D02R.split_cake(cake_flow)
    fed = D02R.crystals_fed(cake_flow)

    np.testing.assert_allclose(sum(numbers.values()), cake_flow.crystal_flow,
                               rtol=1e-12)
    for name in D02R.CRYSTAL_SPECIES:
        assert D02R.crystal_mass_flow(numbers[name], densities[name]) == \
            pytest.approx(fed[name], rel=1e-9)


def test_cake_stream_carries_the_cake(default_run):
    cake, cake_flow, *_ = default_run
    total = sum(float(np.sum(s.mass_flow)) for s in cake.Streams)

    assert [s.phase_family for s in cake.Streams] == \
        ["liquid", "solid", "solid", "solid"]
    assert total == pytest.approx(
        cake_flow.liquid_mass_flow + cake_flow.solid_mass_flow, rel=1e-9)


def test_tank_receives_the_cake_crystals(default_run):
    """Weighted by each solid's own slurry volume, PharmaPy's crystal feed
    term is the cake crystals."""
    _, cake_flow, _, _, balance = default_run
    fed = D02R.crystals_fed(cake_flow)
    for name, terms in balance.crystal.items():
        assert terms["inlet"] == pytest.approx(fed[name], rel=1e-3), name


def test_crystals_lose_what_the_liquid_gains(default_run):
    *_, balance = default_run
    names = D02R.species_names()
    for name, terms in balance.crystal.items():
        assert terms["dissolved"] == pytest.approx(
            balance.to_liquid[names.index(name)], rel=1e-3), name


def test_residence_time_is_charge_over_feed(default_run):
    cake, cake_flow, _, profiles, _ = default_run
    flow = (cake_flow.solid_mass_flow + cake_flow.liquid_mass_flow
            + D02R.L_FLOW)

    assert D02R.residence_time(cake) == pytest.approx(
        D02R.INITIAL_MASS / flow, rel=1e-9)
    assert profiles.time[-1] == pytest.approx(
        D02R.NUM_RESIDENCE_TIMES * D02R.residence_time(cake), rel=1e-6)


def test_reaches_steady_state(default_run):
    *_, balance = default_run
    assert D02R.steady_state_gap(balance) < D02R.STEADY_STATE_TOL


def test_steady_mass_balance_against_the_real_feed(default_run):
    """Liquid plus crystals out equals cake plus L in."""
    _, cake_flow, _, profiles, balance = default_run
    fed = cake_flow.liquid_mass_flow + cake_flow.solid_mass_flow + D02R.L_FLOW
    out = balance.outlet.sum() + sum(profiles.crystal_out[s][-1]
                                     for s in D02R.CRYSTAL_SPECIES)

    assert out == pytest.approx(fed, rel=1e-3)


def test_d_e_j_dissolve_almost_completely(default_run):
    _, cake_flow, _, _, balance = default_run
    for name, fraction in D02R.fraction_dissolved(balance,
                                                  cake_flow).items():
        assert 0.95 < fraction < 1.0, name


def test_only_d_e_j_change_phase(default_run):
    *_, balance = default_run
    for j, name in enumerate(D02R.species_names()):
        if name not in D02R.CRYSTAL_SPECIES:
            assert balance.to_liquid[j] == pytest.approx(0.0, abs=1e-15), name


def test_solvent_effect_drives_the_dissolution(default_run,
                                               no_solvent_effect_run):
    _, cake_flow, _, _, with_k = default_run
    *_, without_k = no_solvent_effect_run
    with_l = D02R.fraction_dissolved(with_k, cake_flow)
    without_l = D02R.fraction_dissolved(without_k, cake_flow)

    for name in D02R.CRYSTAL_SPECIES:
        assert without_l[name] < 0.5 * with_l[name], name


def test_figures_are_written(default_run, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    cake, cake_flow, tank, _, balance = default_run
    D02R.save_figures(cake, cake_flow, tank, out_dir=tmp_path,
                      balance=balance)

    for name in ("D02R_profiles", "D02R_csd", "D02R_settings",
                 "D02R_mass_balance"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
