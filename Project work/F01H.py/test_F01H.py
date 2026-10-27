"""Tests for F01H, the filter on the R01H product stream.

Units: mass flow [kg/s], concentration [kmol/m3 = mol/L], temperature [K].
"""

import warnings

import numpy as np
import pytest

import F01H


@pytest.fixture(scope="module")
def flowsheet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return F01H.run_flowsheet()


def test_feed_carries_the_r01h_outlet_flow(flowsheet):
    """Crystallizing F moves mass between phases; it creates none."""
    _, outlet, feed, _ = flowsheet

    assert feed.liquid_mass_flow + feed.solid_mass_flow == pytest.approx(
        outlet.liquid_mass_flow, rel=1e-12)
    assert feed.temp == outlet.temp


def test_only_f_crystallizes(flowsheet):
    _, outlet, feed, _ = flowsheet
    j_f = list(feed.name_species).index("F")
    others = np.arange(len(feed.name_species)) != j_f

    np.testing.assert_array_equal(feed.liquid_mass_j[others],
                                  outlet.liquid_mass_j[others])
    assert feed.solid_mass_flow == pytest.approx(
        outlet.liquid_mass_j[j_f] - feed.liquid_mass_j[j_f], rel=1e-12)
    assert feed.solid_mass_flow > 0


def test_equilibrium_leaves_the_liquid_saturated(flowsheet):
    """Dissolved F sits exactly at CR01H's solubility for the left liquid."""
    _, _, feed, _ = flowsheet
    liquid = F01H._liquid_stream(feed, feed.liquid_mass_j, feed.temp)
    conc = np.asarray(liquid.mole_conc, dtype=float)
    solubility = F01H.CR01H.make_solubility_function(
        tuple(feed.name_species))

    j_f = list(feed.name_species).index("F")
    assert conc[j_f] == pytest.approx(float(solubility(feed.temp, conc)),
                                      rel=1e-9)


def test_fixed_solid_fraction_overrides_equilibrium(flowsheet):
    _, outlet, _, _ = flowsheet
    feed = F01H.crystallize_f(outlet, solid_fraction=0.5)

    assert F01H.crystalline_fraction(feed) == pytest.approx(0.5, rel=1e-9)


def test_filter_balance_and_settings(flowsheet):
    *_, filt = flowsheet
    summary = filt.summary()

    assert abs(summary["mass_balance_error_kg_s"]) < 1e-15
    assert summary["solids_recovery"] == pytest.approx(1 - F01H.SOLIDS_LOSS)
    assert summary["cake_lod"] == pytest.approx(F01H.CAKE_MOISTURE)


def test_species_table_closes(flowsheet):
    _, outlet, _, filt = flowsheet
    total = F01H.species_table(outlet, filt)[-1]

    _, r01_out, feed, cake, filtrate = total
    assert feed == pytest.approx(r01_out, rel=1e-12)
    assert cake + filtrate == pytest.approx(feed, rel=1e-12)
