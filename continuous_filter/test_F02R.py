"""Tests for F02R, the filter on the R02R product stream.

Units: mass flow [kg/s], concentration [kmol/m3 = mol/L], temperature [K],
density [kg/m3].
"""

import warnings

import numpy as np
import pytest

import F02R


@pytest.fixture(scope="module")
def r02r():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return F02R.r02r_outlet()


@pytest.fixture(scope="module")
def flowsheet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return F02R.run_flowsheet()


def test_feed_carries_the_r02r_outlet_flow(flowsheet):
    """Crystallizing moves mass between phases; it creates none."""
    _, outlet, feed, _ = flowsheet

    assert feed.liquid_mass_flow + feed.solid_mass_flow == pytest.approx(
        outlet.liquid_mass_flow, rel=1e-12)
    assert feed.temp == outlet.temp


def test_only_d_e_and_j_crystallize(flowsheet):
    _, outlet, feed, _ = flowsheet
    names = list(feed.name_species)
    others = [j for j, n in enumerate(names) if n not in F02R.CRYSTAL_SPECIES]

    np.testing.assert_array_equal(feed.liquid_mass_j[others],
                                  outlet.liquid_mass_j[others])

    crystals = F02R.crystal_mass_by_species(feed)
    for name in F02R.CRYSTAL_SPECIES:
        j = names.index(name)
        assert crystals[name] == pytest.approx(
            outlet.liquid_mass_j[j] - feed.liquid_mass_j[j], rel=1e-9)


def test_fixed_fractions_are_applied(r02r):
    _, outlet = r02r
    settings = {"D": 0.2, "E": 0.5, "J": 0.9}
    feed = F02R.crystallize(outlet, settings=settings)

    for name, value in F02R.crystalline_fraction(feed).items():
        assert value == pytest.approx(settings[name], rel=1e-9)


def test_mixed_density_is_mass_over_volume(r02r):
    _, outlet = r02r
    feed = F02R.crystallize(outlet)
    crystals = F02R.crystal_mass_by_species(feed)
    densities = F02R.solid_densities(F02R.DEFAULT_DATABASE_PATH)

    volume = sum(crystals[n] / densities[n] for n in F02R.CRYSTAL_SPECIES)
    assert feed.solid_density == pytest.approx(
        sum(crystals.values()) / volume, rel=1e-12)


def test_equilibrium_with_coupled_solubilities(r02r):
    """Two solubility functions at once: both species end up saturated.

    The functions are illustrative, not data. E's solubility depends on [D],
    and crystallizing either species changes the liquid volume, so the two
    equations are coupled and need the sweep to converge.
    """
    _, outlet = r02r
    names = list(outlet.name_species)
    j_d, j_e = names.index("D"), names.index("E")

    def solub_d(temp, conc):
        return 1e-3  # mol/L, about a tenth of R02R's outlet [D]

    def solub_e(temp, conc):
        return 1e-5 * (temp - 273.15) + 0.1 * conc[j_d]

    settings = {"D": solub_d, "E": solub_e, "J": 1.0}
    feed = F02R.crystallize(outlet, settings=settings)

    liquid = F02R._liquid_stream(feed, feed.liquid_mass_j, feed.temp)
    conc = np.asarray(liquid.mole_conc, dtype=float)

    assert conc[j_d] == pytest.approx(solub_d(feed.temp, conc), rel=1e-8)
    assert conc[j_e] == pytest.approx(solub_e(feed.temp, conc), rel=1e-8)
    assert F02R.crystalline_fraction(feed)["J"] == pytest.approx(1.0)


def test_rejects_unknown_species(r02r):
    _, outlet = r02r

    with pytest.raises(KeyError):
        F02R.crystallize(outlet, settings={"A": 0.5})


def test_filter_balance_and_settings(flowsheet):
    *_, filt = flowsheet
    summary = filt.summary()

    assert abs(summary["mass_balance_error_kg_s"]) < 1e-15
    assert summary["solids_recovery"] == pytest.approx(1 - F02R.SOLIDS_LOSS)
    assert summary["cake_lod"] == pytest.approx(F02R.CAKE_MOISTURE)


def test_species_table_closes(flowsheet):
    _, outlet, _, filt = flowsheet
    _, r02r_out, feed, cake, filtrate = F02R.species_table(outlet, filt)[-1]

    assert feed == pytest.approx(r02r_out, rel=1e-12)
    assert cake + filtrate == pytest.approx(feed, rel=1e-12)
