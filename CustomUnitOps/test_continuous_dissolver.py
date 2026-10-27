"""Tests for the standalone continuous dissolver.

Units: mass flow [kg/s], size [um], concentration [kmol/m3 = mol/L],
temperature [K].
"""

import os
import warnings

import numpy as np
import pytest

from PharmaPy.Streams_Refactored import LiquidStream

from continuous_dissolver import ContinuousDissolver, shrink, shrink_for_mass
from continuous_filter import SlurryFlow, flow_from_stream

DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "tests",
                         "Flowsheet", "data", "compound_database.json")

SPECIES = ["A", "B", "C", "D", "solvent"]
J_C = SPECIES.index("C")
LIQUID_FRAC = np.array([0.0, 0.0, 0.02, 0.0, 0.98])
SOLID_FRAC = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
X_GRID = np.arange(1.0, 201.0)                      # [um]
DENSITY = 1200.0                                    # [kg/m3]
TEMP = 300.0                                        # [K]

LIQUID_FLOW = 0.004                                 # [kg/s]
SOLID_FLOW = 0.002                                  # [kg/s]


def cake(liquid_flow=LIQUID_FLOW, solid_flow=SOLID_FLOW,
         solid_frac=SOLID_FRAC):
    shape = np.exp(-0.5 * ((X_GRID - 80.0) / 20.0) ** 2)
    scale = solid_flow / (DENSITY * np.sum(shape * X_GRID**3) * 1e-18)

    return SlurryFlow(
        path_thermo=DATA_PATH, name_species=SPECIES, temp=TEMP,
        liquid_mass_j=liquid_flow * LIQUID_FRAC, name_solv="solvent",
        x_grid=X_GRID, crystal_flow=scale * shape,
        solid_mass_frac=np.asarray(solid_frac, dtype=float),
        solid_density=DENSITY, target_components=["C"],
        solvent_name="solvent")


def solvent(flow=0.01, temp=TEMP):
    return LiquidStream(DATA_PATH, mass_flow=flow, mass_frac=[0, 0, 0, 0, 1],
                        temp=temp, name_solv="solvent", verbose=False)


def constant_solubility(value):
    return lambda temp, conc: value


def liquid_conc(dissolver):
    out = dissolver.outlet_flow
    total = out.liquid_mass_flow
    liquid = LiquidStream(DATA_PATH, mass_flow=total,
                          mass_frac=out.liquid_mass_j / total, temp=out.temp,
                          name_solv="solvent", verbose=False)
    return np.asarray(liquid.mole_conc, dtype=float)


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------


def test_shrink_keeps_number_and_mass_on_the_grid():
    """Mass matches the exact shrunken sizes. Number matches too, except for
    the first bin, whose crystals drop below the first node and keep only
    their mass there."""
    feed = cake()
    delta = 0.37                                     # [um]

    out = shrink(feed, delta)
    counts_in = feed.crystal_flow * feed.dx
    counts_out = out * feed.dx

    below = (X_GRID[0] - delta) / X_GRID[0]
    expected_number = counts_in[1:].sum() + counts_in[0] * below**3

    assert counts_out.sum() == pytest.approx(expected_number, rel=1e-12)
    assert np.sum(counts_out * X_GRID**3) == pytest.approx(
        np.sum(counts_in * (X_GRID - delta) ** 3), rel=1e-12)


def test_shrink_for_mass_hits_the_target():
    feed = cake()
    target = 0.4 * SOLID_FLOW

    delta = shrink_for_mass(feed, target)
    remaining = DENSITY * 1e-18 * np.sum(shrink(feed, delta) * feed.dx
                                         * X_GRID**3)

    assert SOLID_FLOW - remaining == pytest.approx(target, rel=1e-9)


def test_shrink_moves_the_distribution_down():
    feed = cake()
    out = shrink(feed, 10.0)

    mean_in = np.sum(feed.crystal_flow * X_GRID) / feed.crystal_flow.sum()
    mean_out = np.sum(out * X_GRID) / out.sum()

    assert mean_out == pytest.approx(mean_in - 10.0, abs=0.1)


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------


def test_equilibrium_leaves_the_liquid_saturated():
    diss = ContinuousDissolver(solubility=constant_solubility(0.2))
    diss.Inlets = [cake(), solvent()]
    diss.solve()

    assert liquid_conc(diss)[J_C] == pytest.approx(0.2, rel=1e-9)
    assert 0 < diss.summary()["fraction_dissolved"] < 1


def test_temperature_dependent_solubility():
    def solubility(temp, conc):
        return 0.01 * (temp - 273.15)               # mol/L

    diss = ContinuousDissolver(solubility=solubility, temp=310.0)
    diss.Inlets = [cake(), solvent()]
    diss.solve()

    assert liquid_conc(diss)[J_C] == pytest.approx(0.01 * (310.0 - 273.15),
                                                   rel=1e-9)


def test_enough_solvent_dissolves_everything():
    diss = ContinuousDissolver(solubility=constant_solubility(5.0))
    diss.Inlets = [cake(), solvent(flow=0.1)]
    outlet = diss.solve()

    assert diss.summary()["fraction_dissolved"] == pytest.approx(1.0)
    assert [p.phase_family for p in outlet.Phases] == ["liquid"]


def test_saturated_feed_dissolves_nothing():
    diss = ContinuousDissolver(solubility=constant_solubility(1e-6))
    diss.Inlets = [cake()]

    with pytest.warns(UserWarning, match="saturation"):
        diss.solve()

    assert diss.summary()["dissolved_kg_s"] == 0


def test_approach_scales_the_equilibrium_dissolution():
    full = ContinuousDissolver(solubility=constant_solubility(0.2))
    full.Inlets = [cake(), solvent()]
    full.solve()

    half = ContinuousDissolver(solubility=constant_solubility(0.2),
                               approach=0.5)
    half.Inlets = [cake(), solvent()]
    half.solve()

    assert half.dissolved == pytest.approx(0.5 * full.dissolved, rel=1e-9)


def test_more_solvent_dissolves_more():
    amounts = []
    for flow in (0.005, 0.01, 0.02):
        diss = ContinuousDissolver(solubility=constant_solubility(0.2))
        diss.Inlets = [cake(), solvent(flow=flow)]
        diss.solve()
        amounts.append(diss.dissolved)

    assert np.all(np.diff(amounts) > 0)


# ---------------------------------------------------------------------------
# Fixed fraction, balances, outlet
# ---------------------------------------------------------------------------


def test_fixed_fraction():
    diss = ContinuousDissolver(fraction_dissolved=0.3)
    diss.Inlets = [cake(), solvent()]
    diss.solve()

    assert diss.summary()["fraction_dissolved"] == pytest.approx(0.3,
                                                                 rel=1e-9)


def test_mass_balance_closes_and_dissolved_mass_joins_the_liquid():
    diss = ContinuousDissolver(solubility=constant_solubility(0.2))
    diss.Inlets = [cake(), solvent()]
    diss.solve()
    s = diss.summary()

    assert abs(s["mass_balance_error_kg_s"]) < 1e-15
    assert diss.outlet_flow.liquid_mass_j[J_C] == pytest.approx(
        LIQUID_FLOW * LIQUID_FRAC[J_C] + s["dissolved_kg_s"], rel=1e-12)


def test_outlet_stream_carries_the_remaining_crystals():
    diss = ContinuousDissolver(fraction_dissolved=0.4)
    diss.Inlets = [cake(), solvent()]
    outlet = diss.solve()

    back = flow_from_stream(outlet)
    np.testing.assert_allclose(back.crystal_flow,
                               diss.outlet_flow.crystal_flow, rtol=1e-12)


def test_mixed_crystals_need_a_fixed_fraction():
    mixed = cake(solid_frac=[0, 0, 0.5, 0.5, 0])
    diss = ContinuousDissolver(solubility=constant_solubility(0.2))
    diss.Inlets = [mixed, solvent()]

    with pytest.raises(ValueError, match="one species"):
        diss.solve()

    diss = ContinuousDissolver(fraction_dissolved=0.5)
    diss.Inlets = [mixed, solvent()]
    diss.solve()
    assert abs(diss.summary()["mass_balance_error_kg_s"]) < 1e-15


@pytest.mark.parametrize("kwargs", [
    {},
    {"solubility": constant_solubility(1.0), "fraction_dissolved": 0.5},
    {"fraction_dissolved": 1.5},
    {"solubility": constant_solubility(1.0), "approach": -0.1},
])
def test_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        ContinuousDissolver(**kwargs)


def test_needs_crystals():
    diss = ContinuousDissolver(fraction_dissolved=0.5)
    diss.Inlets = [solvent()]

    with warnings.catch_warnings(), pytest.raises(ValueError):
        warnings.simplefilter("ignore")
        diss.solve()
