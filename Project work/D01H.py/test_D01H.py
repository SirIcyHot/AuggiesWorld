"""Tests for D01H, the dissolver on the F01H cake.

Units: mass flow [kg/s], concentration [kmol/m3 = mol/L], temperature [K].
"""

import warnings

import numpy as np
import pytest

from PharmaPy.Streams_Refactored import LiquidStream

import D01H


@pytest.fixture(scope="module")
def flowsheet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return D01H.run_flowsheet()


def test_feeds_are_the_cake_and_the_solvent(flowsheet):
    filt, solvent, diss = flowsheet

    cake_total = filt.cake_flow.liquid_mass_flow + filt.cake_flow.solid_mass_flow
    feed_total = diss.feed.liquid_mass_flow + diss.feed.solid_mass_flow

    assert feed_total == pytest.approx(cake_total + D01H.SOLVENT_FLOW,
                                       rel=1e-12)
    assert diss.feed.solid_mass_flow == pytest.approx(
        filt.cake_flow.solid_mass_flow, rel=1e-12)


def test_only_f_dissolves(flowsheet):
    filt, solvent, diss = flowsheet
    names = list(diss.feed.name_species)
    j_f = names.index("F")
    others = np.arange(len(names)) != j_f

    np.testing.assert_allclose(diss.outlet_flow.liquid_mass_j[others],
                               diss.feed.liquid_mass_j[others], rtol=1e-12)
    assert diss.outlet_flow.liquid_mass_j[j_f] == pytest.approx(
        diss.feed.liquid_mass_j[j_f] + diss.dissolved, rel=1e-12)


def test_outlet_liquid_is_saturated_or_crystals_are_gone(flowsheet):
    _, _, diss = flowsheet
    out = diss.outlet_flow
    names = list(out.name_species)

    liquid = LiquidStream(str(D01H.DB_PATH), mass_flow=out.liquid_mass_flow,
                          mass_frac=out.liquid_mass_j / out.liquid_mass_flow,
                          temp=out.temp, name_solv="B", verbose=False)
    conc = np.asarray(liquid.mole_conc, dtype=float)
    solubility = float(D01H.CR01H.make_solubility_function(tuple(names))(
        out.temp, conc))

    if out.solid_mass_flow > 0:
        assert conc[names.index("F")] == pytest.approx(solubility, rel=1e-8)
    else:
        assert conc[names.index("F")] <= solubility


def test_mass_balance_closes(flowsheet):
    _, _, diss = flowsheet
    assert abs(diss.summary()["mass_balance_error_kg_s"]) < 1e-15


def test_species_table_closes(flowsheet):
    total = D01H.species_table(*flowsheet)[-1]
    _, cake, solvent, out = total

    assert out == pytest.approx(cake + solvent, rel=1e-12)
