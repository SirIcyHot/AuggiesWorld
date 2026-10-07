"""Fidelity of the refactored solid-liquid separation units.

`Filter` and `DeliquoringStep` were moved onto `MultiPhaseVessel`, and the
standard they are held to is the pre-refactor unit in `SolidLiquidSep`: the
physics is meant to be unchanged, so the numbers must be reproduced before
anything about them is revisited. These tests drive the legacy unit and the
refactored one on the same charge and compare.

Two differences are expected and are asserted as such rather than tuned away:

* **Quadrature.** The refactored stack integrates a distribution with the
  rectangle rule, because the FVM stores cell averages, where the legacy
  `getMoments` uses the trapezoid. Anything that reaches the crystal size
  distribution therefore differs by O(dx), and the test asserts that the gap
  *shrinks with the grid* - which an error would not.
* **Step cap.** `AssimuloBackend` defaults to `options={'maxh': 1}`, a one
  second ceiling. Deliquoring runs over 1e4 to 1e6 s of physical time, so that
  default forces 1e4 to 1e6 steps and the solve never finishes. These units are
  given `options={}` deliberately; it is a property of the default, not of the
  models.
"""

import os

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import AssimuloBackend, ScipyBackend
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.MixedPhases import Cake, Slurry
from PharmaPy.Phases import LiquidPhase as LegacyLiquid
from PharmaPy.Phases import SolidPhase as LegacySolid
from PharmaPy.Phases_Refactored import LiquidPhase as Liquid
from PharmaPy.Phases_Refactored import SolidPhase as Solid
from PharmaPy.SolidLiquidSep import DeliquoringStep as LegacyDeliquoring
from PharmaPy.SolidLiquidSep import Filter as LegacyFilter
from PharmaPy.SolidLiquidSep_Refactored import DeliquoringStep, Filter

pytestmark = pytest.mark.unit

pytest.importorskip("assimulo", reason="the reference unit needs assimulo")

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

MASSFRAC_SOLID = [0, 0, 1, 0, 0]
MASSFRAC_LIQUID = [0, 0, 0, 0, 1]

FILT_DIAM = 0.3       # [m]
ALPHA = 1e10          # [m/kg]
RESIST_MEDIUM = 1e9   # [1/m]
DELTA_P = 5e4         # [Pa]

VOL_LIQUID = 2750e-6  # [m**3]
MASS_SOLID = VOL_LIQUID * 2.4e-2 * 1e3  # [kg]


def size_grid(num_nodes=500):
    return np.linspace(1, 500, num_nodes)


def legacy_phases(x_grid, mass=MASS_SOLID, vol_liquid=VOL_LIQUID):
    """An old solid and liquid. `distrib` is read as a volume-fraction shape."""

    liquid = LegacyLiquid(path_thermo=DATA_PATH, vol=vol_liquid,
                          mass_frac=MASSFRAC_LIQUID)
    solid = LegacySolid(DATA_PATH, mass=mass, x_distrib=x_grid,
                        distrib=np.ones_like(x_grid),
                        mass_frac=MASSFRAC_SOLID)

    return liquid, solid


def refactored_phases(x_grid, vol_slurry, density, mass=MASS_SOLID,
                      vol_liquid=VOL_LIQUID):
    """The same physical phases, with the distribution stated through the flag.

    Handing the refactored solid the same raw array would NOT give the same
    distribution: the legacy phase reads it as a volume-fraction shape, this
    stack reads it as an intensive number density. `distrib_basis` is what
    makes the two comparable.
    """

    liquid = Liquid(DATA_PATH, vol=vol_liquid, mass_frac=MASSFRAC_LIQUID,
                    name_solv="solvent")
    solid = Solid(DATA_PATH, mass=mass, mass_frac=MASSFRAC_SOLID)

    solid.mechanisms = OneDFVMMechanism(
        solid, target_components="C", solvent_name="solvent", x_grid=x_grid,
        distrib_init=np.ones_like(x_grid), distrib_basis="vol_perc",
        basis_mass=mass, vol_slurry=vol_slurry, density=density,
    )

    return liquid, solid


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def filtration_times(num_nodes, backend=AssimuloBackend):
    """Legacy and refactored filtration time on one charge."""

    x_grid = size_grid(num_nodes)

    liquid, solid = legacy_phases(x_grid)
    slurry = Slurry()
    slurry.Phases = (solid, liquid)

    reference = LegacyFilter(FILT_DIAM, ALPHA, RESIST_MEDIUM)
    reference.Phases = slurry
    reference.solve_unit(deltaP=DELTA_P, verbose=False)

    vol_slurry = VOL_LIQUID + solid.vol

    new_liquid, new_solid = refactored_phases(x_grid, vol_slurry,
                                             solid.getDensity())

    candidate = Filter(FILT_DIAM, deltaP=DELTA_P, alpha=ALPHA,
                       resist_medium=RESIST_MEDIUM, integrator=backend())
    candidate.Phases = [new_liquid, new_solid]

    time, _ = candidate.solve_unit(runtime=1e4, verbose=False)

    return float(reference.timeProf[-1]), float(np.asarray(time)[-1])


def test_filter_converges_to_the_legacy_filtration_time():
    """The gap is the distribution quadrature, so it must shrink with the grid."""

    gaps = []

    for num_nodes in (500, 2000, 8000):
        reference, candidate = filtration_times(num_nodes)
        gaps.append(abs(candidate - reference) / reference)

    gaps = np.array(gaps)

    assert np.all(np.diff(gaps) < 0), gaps
    assert gaps[0] < 5e-3, gaps
    assert gaps[-1] < 5e-4, gaps


def test_filter_agrees_across_backends():
    """The controller formulation keeps it an ODE, so scipy must handle it."""

    _, assimulo = filtration_times(500, AssimuloBackend)
    _, scipy = filtration_times(500, ScipyBackend)

    assert abs(scipy - assimulo) / assimulo < 1e-3


def test_filter_offers_both_outlets():
    """The original dropped the filtrate; the cake stays the default output."""

    x_grid = size_grid(200)
    liquid, solid = legacy_phases(x_grid)
    vol_slurry = VOL_LIQUID + solid.vol

    new_liquid, new_solid = refactored_phases(x_grid, vol_slurry,
                                             solid.getDensity())

    unit = Filter(FILT_DIAM, deltaP=DELTA_P, alpha=ALPHA,
                  resist_medium=RESIST_MEDIUM, integrator=AssimuloBackend())
    unit.Phases = [new_liquid, new_solid]

    outlet = unit.Outlet

    assert set(outlet) == {"cake", "filtrate"}
    assert unit.default_output == "cake"
    assert outlet["filtrate"] is not None


# ---------------------------------------------------------------------------
# Deliquoring
# ---------------------------------------------------------------------------


NUM_NODES_CAKE = 20


def deliquoring_pair(num_nodes=NUM_NODES_CAKE):
    """A legacy deliquoring unit and its refactored twin on one cake."""

    x_grid = size_grid(500)

    liquid, solid = legacy_phases(x_grid, vol_liquid=1e-4)

    cake = Cake(num_discr=num_nodes)
    cake.Phases = (solid, liquid)

    reference = LegacyDeliquoring(num_nodes=num_nodes, diam_unit=0.01,
                                  resist_medium=RESIST_MEDIUM)
    reference.Phases = cake

    return reference, cake, solid, x_grid


def refactored_deliquoring(cake, solid, x_grid, backend=AssimuloBackend):

    new_liquid, new_solid = refactored_phases(
        x_grid, cake.cake_vol, solid.getDensity(), vol_liquid=1e-4)

    # options={} on purpose - see the module docstring on the maxh default.
    unit = DeliquoringStep(
        num_nodes=NUM_NODES_CAKE, cake_vol=cake.cake_vol, alpha=cake.alpha,
        deltaP=DELTA_P, diam_unit=0.01, resist_medium=RESIST_MEDIUM,
        integrator=backend(options={}),
    )
    unit.Phases = [new_liquid, new_solid]

    return unit


def saturation_of(unit, states):
    collection = unit.solver_state_collection
    key = next(k for k in collection.states if k.name == "sat_red")

    return np.asarray(states)[-1][collection.slices[key]]


def test_deliquoring_constants_match_the_original():
    """Everything solve_unit computed before integrating."""

    reference, cake, solid, x_grid = deliquoring_pair()
    reference.solve_unit(deltaP=DELTA_P, runtime=1.0, verbose=False)

    unit = refactored_deliquoring(cake, solid, x_grid)
    unit.compile_structure()

    # Both sides reach the size distribution, so both carry the same O(dx)
    # quadrature difference and nothing larger.
    assert abs(unit.sat_inf - reference.sat_inf) / reference.sat_inf < 1e-8
    assert abs(unit.p_thresh - reference.p_thresh) / reference.p_thresh < 1e-4
    assert (abs(unit.theta_conv - reference.theta_conv)
            / reference.theta_conv) < 1e-4


@pytest.mark.parametrize("theta", [1e-4, 1e-3, 1e-2])
def test_deliquoring_profile_matches_the_original(theta):
    """The saturation profile, after a run that actually moves it."""

    reference, cake, solid, x_grid = deliquoring_pair()
    reference.solve_unit(deltaP=DELTA_P, runtime=1.0, verbose=False)

    runtime = theta / reference.theta_conv

    fresh, cake, solid, x_grid = deliquoring_pair()
    _, states = fresh.solve_unit(deltaP=DELTA_P, runtime=runtime,
                                 verbose=False)
    expected = np.asarray(states)[-1].reshape(NUM_NODES_CAKE, -1)[:, 0]

    # The profile has to have gone somewhere, or this proves nothing.
    assert expected.max() - expected.min() > 1e-3
    assert expected.max() < 0.99

    unit = refactored_deliquoring(cake, solid, x_grid)
    _, new_states = unit.solve_unit(runtime=runtime, verbose=False)

    np.testing.assert_allclose(saturation_of(unit, new_states), expected,
                               rtol=1e-4, atol=1e-8)


def test_deliquoring_runs_on_the_scipy_backend():
    reference, cake, solid, x_grid = deliquoring_pair()
    reference.solve_unit(deltaP=DELTA_P, runtime=1.0, verbose=False)

    runtime = 1e-3 / reference.theta_conv

    fresh, cake, solid, x_grid = deliquoring_pair()
    _, states = fresh.solve_unit(deltaP=DELTA_P, runtime=runtime,
                                 verbose=False)
    expected = np.asarray(states)[-1].reshape(NUM_NODES_CAKE, -1)[:, 0]

    unit = refactored_deliquoring(cake, solid, x_grid, ScipyBackend)
    _, new_states = unit.solve_unit(runtime=runtime, verbose=False)

    np.testing.assert_allclose(saturation_of(unit, new_states), expected,
                               rtol=1e-3, atol=1e-8)


def test_deliquoring_declares_only_the_cake_profile():
    """No phase inventory, matching the original, and no output that needs one."""

    reference, cake, solid, x_grid = deliquoring_pair()
    unit = refactored_deliquoring(cake, solid, x_grid)
    unit.compile_structure()

    names = {key.name for key in unit.solver_state_collection.keys}
    assert names == {"sat_red", "conc_star"}

    # mole_conc is computed from mass_j, which is deliberately not a state.
    output_names = {key.name for key in unit.output_state_collection.keys}
    assert "mole_conc" not in output_names
