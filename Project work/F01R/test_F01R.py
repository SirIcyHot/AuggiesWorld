"""Tests for F01R, the filter on the D01R slurry.

Units: mass flow [kg/s], concentration [kmol/m3 = mol/L], temperature [K],
density [kg/m3].
"""

import warnings

import numpy as np
import pytest

import F01R  # puts D01R, R02R and CustomUnitOps on the path
import D01R


@pytest.fixture(scope="module")
def flowsheet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return F01R.run_flowsheet()


def test_feed_is_the_whole_d01r_discharge(flowsheet):
    """The merged outlet carries D01R's own liquid and crystal outflow."""
    tank, outlet, _ = flowsheet
    balance = D01R.final_balance(tank)

    assert outlet.liquid_mass_j == pytest.approx(balance.outlet, rel=1e-6,
                                                 abs=1e-12)
    crystals = F01R.crystal_mass_by_species(outlet)
    for name in F01R.CRYSTAL_SPECIES:
        assert crystals[name] == pytest.approx(
            balance.crystal[name]["outlet"], rel=1e-6, abs=1e-12)


def test_crystals_come_from_d01r_not_a_placeholder(flowsheet):
    """Every species is partly dissolved, partly crystalline."""
    _, outlet, _ = flowsheet

    for name, value in F01R.crystalline_fraction(outlet).items():
        assert 0.0 < value < 1.0, name


def test_mixed_density_is_mass_over_volume(flowsheet):
    _, outlet, _ = flowsheet
    crystals = F01R.crystal_mass_by_species(outlet)
    densities = F01R.solid_densities(F01R.DEFAULT_DATABASE_PATH)

    volume = sum(crystals[n] / densities[n] for n in F01R.CRYSTAL_SPECIES)
    assert outlet.solid_density == pytest.approx(
        sum(crystals.values()) / volume, rel=1e-12)


def test_grid_is_d01rs(flowsheet):
    _, outlet, _ = flowsheet
    np.testing.assert_array_equal(outlet.x_grid, D01R.CRYSTAL_SIZE_GRID)


def test_filter_balance_and_settings(flowsheet):
    *_, filt = flowsheet
    summary = filt.summary()

    assert abs(summary["mass_balance_error_kg_s"]) < 1e-15
    assert summary["solids_recovery"] == pytest.approx(1 - F01R.SOLIDS_LOSS)
    assert summary["cake_lod"] == pytest.approx(F01R.CAKE_MOISTURE)


def test_species_table_closes(flowsheet):
    *_, filt = flowsheet
    _, feed, cake, filtrate = F01R.species_table(filt)[-1]

    assert cake + filtrate == pytest.approx(feed, rel=1e-12)
