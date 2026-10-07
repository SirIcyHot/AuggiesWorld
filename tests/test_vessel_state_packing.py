"""Packing contract between the vessel balances and the solver vector.

MultiPhaseVessel evaluates the material and energy balances separately and
then packs both into one flat derivative vector for the integrator. Solver
states are split by whether they belong to a phase: phase-owned differential
states are packed from a flat material array, while states with no owning
phase, currently only the vessel temperature, come from the global balances.
A state that is visited by neither keeps the zero its buffer was filled with,
which is silent and looks like a physically stationary quantity rather than a
missing one.

The fixture is a closed jacketed vessel whose contents sit about 25 K above
the coolant, so the correct temperature rate is unambiguously negative and
cannot be confused with the zero a dropped state would produce.
"""

import os

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import BatchReactor
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

CHARGE_MASS = 1.0  # [kg]
CHARGE_MASS_FRAC = [0.4, 0.6, 0.0, 0.0, 0.0]  # [-]
CHARGE_TEMP = 298.15  # [K], the phase default

UTILITY_MASS_FLOW = 100.0  # [kg/s]
UTILITY_TEMP_IN = 273.55  # [K], below the charge so the jacket removes heat
HEAT_TRANSFER_COEFF = 1e4  # [W/m**2/K]
VESSEL_DIAMETER = 0.01  # [m]

# The packed value is the same float the energy balance produced, so any
# difference would be a transcription error rather than accumulated roundoff.
PACKING_RTOL = 1e-12  # [-]


def _build_vessel(isothermal=False):
    """Build a closed jacketed batch vessel with a cold utility.

    Parameters
    ----------
    isothermal : bool, optional
        When True the vessel carries no temperature state, which removes the
        energy balance from the solver vector entirely.

    Returns
    -------
    BatchReactor
        Vessel with its array layout compiled, ready for ``unit_model``.
    """
    vessel = BatchReactor(
        integrator=AssimuloBackend(),
        h_conv=HEAT_TRANSFER_COEFF,
        diam=VESSEL_DIAMETER,
        isothermal=isothermal,
    )
    vessel.Phases = LiquidPhase(
        DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC
    )
    vessel.Utility = CoolingWater(
        mass_flow=UTILITY_MASS_FLOW, temp_in=UTILITY_TEMP_IN
    )
    # compile_structure builds the array layout the balances index into. It
    # runs inside solve_unit, but these tests call unit_model directly.
    vessel.compile_structure()
    return vessel


def _temperature_slice(vessel):
    """Return the slice of the solver vector holding the vessel temperature.

    Parameters
    ----------
    vessel : MultiPhaseVessel
        Vessel whose structure has been compiled.

    Returns
    -------
    slice
        Position of ``global_temp`` within the packed solver vector.
    """
    collection = vessel.solver_state_collection
    key = next(k for k in collection.states if k.name == "global_temp")
    return collection.slices[key]


def test_energy_rate_reaches_the_packed_state_vector():
    """The computed temperature rate survives packing into the solver vector.

    The energy balance is evaluated on its own and then written into the flat
    derivative vector. This asserts the value the integrator receives is the
    value the balance produced, because a packing loop that skips states
    without an owning phase returns a well-formed vector carrying a zero in
    that slot instead.
    """
    vessel = _build_vessel()
    states = vessel.create_solver_init_states()

    packed = np.asarray(vessel.unit_model(0.0, states))
    temperature_rate = packed[_temperature_slice(vessel)][0]  # [K/s]

    # A jacket below the contents must cool them, so a zero here is not a
    # plausible answer and the assertion cannot pass vacuously.
    assert temperature_rate < 0

    # Recompute the balance directly and require the packed value to match.
    unpacked = vessel.solver_state_collection.unpack(states)
    completed = vessel.complete_state(unpacked, 0.0)
    vessel.update_phases_from_state(completed)
    _, buffer = vessel.material_balances(0.0, completed, limiter_dt=1.0)
    energy_rates = vessel.energy_balances(0.0, completed, buffer)

    expected = float(np.asarray(list(energy_rates.values())[0]))  # [K/s]
    np.testing.assert_allclose(temperature_rate, expected, rtol=PACKING_RTOL)


def test_material_only_request_leaves_the_temperature_untouched():
    """A material-only evaluation reports no temperature rate.

    unit_model(mat_bce=True) is a diagnostic path that skips the energy
    balance, so it supplies no global rates at all. Packing must leave that
    slot at zero rather than failing on the missing input.
    """
    vessel = _build_vessel()
    states = vessel.create_solver_init_states()

    packed = np.asarray(vessel.unit_model(0.0, states, mat_bce=True))

    assert len(packed) == vessel.solver_state_collection.dim
    assert packed[_temperature_slice(vessel)][0] == 0.0


def test_energy_only_request_packs_the_temperature():
    """An energy-only evaluation reports the temperature and no material rates.

    This path supplies global rates without material rates, the mirror image
    of the case above, and must not fail on the absent material input.
    """
    vessel = _build_vessel()
    states = vessel.create_solver_init_states()

    packed = np.asarray(vessel.unit_model(0.0, states, enrgy_bce=True))

    assert packed[_temperature_slice(vessel)][0] < 0

    material_slices = vessel.solver_state_collection.material_slices
    solver_slices = vessel.solver_state_collection.slices
    for key in material_slices:
        np.testing.assert_allclose(packed[solver_slices[key]], 0.0)


def test_isothermal_vessel_carries_no_temperature_state():
    """An isothermal vessel has no temperature state to pack.

    The temperature state is registered only when an energy balance exists, so
    packing must not demand a global rate for a state that was never created.
    """
    vessel = _build_vessel(isothermal=True)
    states = vessel.create_solver_init_states()

    names = [key.name for key in vessel.solver_state_collection.states]
    assert "global_temp" not in names

    packed = np.asarray(vessel.unit_model(0.0, states))
    assert len(packed) == vessel.solver_state_collection.dim
