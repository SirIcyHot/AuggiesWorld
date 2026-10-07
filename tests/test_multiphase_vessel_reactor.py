"""Balance regressions for the MultiPhaseVessel reactor path.

The refactored vessel is otherwise exercised only by the manual scripts under
``PharmaPy/``, which hard-code paths, plot to screen, and are read by eye.
These tests cover the reactor path as an automated check, driving
``unit_model`` directly so they need no integrator and run without Assimulo.

Fixtures use the five-species ``tests/Flowsheet/data`` database, whose molar
masses make both reactions exactly mass conserving (A 100 + B 50 = C 150;
C 150 + A 100 = D 250). The charge is deliberately richer in B than in A
(0.4 kg A, 0.6 kg B) so that A is the limiting reactant and an A/B mix-up is
visible in the stoichiometric ratio.

Two kinetic regimes are exercised. ``RATE_CONSTANT_KINETIC`` is slow enough
that the extent follows the Arrhenius expression, while
``RATE_CONSTANT_CLAMPED`` is fast enough that ``RxnKinetics.get_rxn_rates``
rescales the extent to stop a concentration going negative. Both regimes must
satisfy the same stoichiometric invariants.
"""

import os

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import (BatchReactor, ContinuousReactor,
                                        SemiBatchReactor)
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

SPECIES = ("A", "B", "C", "D", "solvent")
MOLAR_MASS = {"A": 100.0, "B": 50.0, "C": 150.0, "D": 250.0}  # [g/mol]

CHARGE_MASS = 1.0  # [kg]
CHARGE_MASS_FRAC = [0.4, 0.6, 0.0, 0.0, 0.0]  # [-] ordered as SPECIES
INLET_MASS_FLOW = 0.1  # [kg/s]

REACTIONS = ["A + B --> C", "C + A --> D"]
# Low enough that the Arrhenius temperature factor stays near unity, so the
# regime is set by the pre-exponential factor alone.
ACTIVATION_ENERGY = 1e2  # [J/mol]
RATE_CONSTANT_KINETIC = 1e-2  # [L/mol/s], slow enough to avoid the extent clamp
RATE_CONSTANT_CLAMPED = 1e4  # [L/mol/s], fast enough to hit the extent clamp

UTILITY_MASS_FLOW = 100.0  # [kg/s]
UTILITY_TEMP_IN = 273.55  # [K], below the 298.15 K charge so the jacket removes heat
HEAT_TRANSFER_COEFF = 1e4  # [W/m**2/K]
VESSEL_DIAMETER = 0.01  # [m]

# Species mass rates that must cancel are compared against an absolute floor
# rather than a relative one, because the exact answer is zero. The budget sits
# a few orders above double-precision roundoff on rates of order 1 kg/s.
MASS_CLOSURE_ATOL = 1e-12  # [kg/s]
# Ratios fixed by stoichiometry and molar mass are exact up to accumulated
# roundoff in the rate evaluation.
RATIO_RTOL = 1e-10  # [-]


def _build_reactor(reactor_cls, rate_constant=None, with_inlet=False):
    """Construct a vessel charged with the standard A/B mixture.

    Parameters
    ----------
    reactor_cls : type
        ``BatchReactor`` or ``ContinuousReactor``.
    rate_constant : float, optional
        Pre-exponential factor applied to both reactions [L/mol/s]. When
        None, no kinetics are attached and the vessel has no reaction
        source term.
    with_inlet : bool, optional
        When True, attach a liquid feed at ``INLET_MASS_FLOW`` whose
        composition equals the initial charge composition.

    Returns
    -------
    MultiPhaseVessel
        Vessel ready for a ``unit_model`` evaluation.
    """
    vessel = reactor_cls(
        integrator=AssimuloBackend(),
        h_conv=HEAT_TRANSFER_COEFF,
        diam=VESSEL_DIAMETER,
    )
    vessel.Phases = LiquidPhase(
        DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC
    )
    vessel.Utility = CoolingWater(
        mass_flow=UTILITY_MASS_FLOW, temp_in=UTILITY_TEMP_IN
    )

    if rate_constant is not None:
        vessel.RxnKinetics = RxnKinetics(
            path=DATA_PATH,
            rxn_list=REACTIONS,
            k_params=np.array([rate_constant, rate_constant]),
            ea_params=np.array([ACTIVATION_ENERGY, ACTIVATION_ENERGY]),
        )

    if with_inlet:
        vessel.Inlet = LiquidStream(
            DATA_PATH,
            mass_flow=INLET_MASS_FLOW,
            mass_frac=CHARGE_MASS_FRAC,
        )

    # compile_structure builds the array layout the balances index into. It
    # runs inside solve_unit, but these tests call unit_model directly, so
    # they have to compile it themselves.
    vessel.compile_structure()

    return vessel


def _species_rates(vessel):
    """Evaluate the material balance once and return species mass rates.

    Parameters
    ----------
    vessel : MultiPhaseVessel
        Configured vessel.

    Returns
    -------
    numpy.ndarray
        Rate of change of species mass holdup, ordered as ``SPECIES``
        [kg/s].
    """
    states = vessel.create_solver_init_states()
    packed = np.asarray(vessel.unit_model(0.0, states, mat_bce=True))

    # unit_model returns the whole solver vector, so take the species block
    # by its own slice rather than assuming it spans the whole array.
    collection = vessel.solver_state_collection
    key = next(k for k in collection.states if k.name == "mass_j")
    return packed[collection.slices[key]]


def test_solver_states_are_species_masses_then_temperature():
    """The solver vector is the per-species holdup followed by temperature.

    Every assertion in this module indexes the state vector positionally, so
    the layout and its units are pinned here rather than assumed.
    """
    vessel = _build_reactor(BatchReactor)
    collection = vessel.solver_state_collection

    layout = [
        (key.name, state.dim, state.units)
        for key, state in collection.states.items()
    ]
    assert layout == [("mass_j", len(SPECIES), "kg"), ("global_temp", 1, "K")]

    states = vessel.create_solver_init_states()
    expected_masses = CHARGE_MASS * np.array(CHARGE_MASS_FRAC)  # [kg]
    np.testing.assert_allclose(states[:len(SPECIES)], expected_masses)

    # unit_model unpacks and pack_state_rates repacks on every evaluation, so
    # the round trip must be lossless and order preserving.
    round_tripped = collection.pack(collection.unpack(states))
    np.testing.assert_allclose(round_tripped, states)


def test_semibatch_feed_accumulates_at_the_inlet_composition():
    """A fed vessel with no outlet grows at the inlet species mass flows.

    SemiBatchReactor declares no outlet connection, which isolates the inlet
    term from the outlet term and the volume controller.
    """
    vessel = _build_reactor(SemiBatchReactor, with_inlet=True)
    assert len(vessel.outlet_connections) == 0

    rates = _species_rates(vessel)

    # Feed only: 0.1 kg/s split 40/60 between A and B.
    expected = INLET_MASS_FLOW * np.array(CHARGE_MASS_FRAC)  # [kg/s]
    np.testing.assert_allclose(rates, expected)


def test_reaction_conserves_total_mass():
    """Reaction source terms cancel across species.

    The database molar masses satisfy A + B = C exactly, so a batch vessel
    with no feed must show zero net mass rate whatever the extent.
    """
    vessel = _build_reactor(BatchReactor, rate_constant=RATE_CONSTANT_KINETIC)

    rates = _species_rates(vessel)

    assert abs(rates.sum()) < MASS_CLOSURE_ATOL
    # Guard against a vacuous pass in which nothing reacted at all.
    assert rates[SPECIES.index("A")] < -MASS_CLOSURE_ATOL


def test_reaction_consumes_reactants_in_stoichiometric_proportion():
    """A and B are consumed in their molar-mass ratio and reappear as C.

    The first reaction is one to one in moles, so the consumed mass ratio is
    fixed at mw_A / mw_B = 2. The second reaction cannot proceed at t = 0
    because C is absent, so D must stay at zero and every gram of consumed
    reactant must appear as C.
    """
    vessel = _build_reactor(BatchReactor, rate_constant=RATE_CONSTANT_KINETIC)

    rates = _species_rates(vessel)
    rate_a, rate_b, rate_c, rate_d, rate_solvent = rates

    assert rate_a < 0 and rate_b < 0 and rate_c > 0

    expected_ratio = MOLAR_MASS["A"] / MOLAR_MASS["B"]  # [-], equals 2.0
    np.testing.assert_allclose(rate_a / rate_b, expected_ratio, rtol=RATIO_RTOL)
    np.testing.assert_allclose(rate_c, -(rate_a + rate_b), rtol=RATIO_RTOL)

    # C is absent at t = 0, so the C + A reaction is inert, and the solvent
    # participates in neither reaction.
    assert abs(rate_d) < MASS_CLOSURE_ATOL
    assert abs(rate_solvent) < MASS_CLOSURE_ATOL


def test_reaction_rate_is_first_order_in_the_preexponential_factor():
    """Doubling the pre-exponential factor doubles the unclamped rate.

    The Arrhenius form is linear in that factor, so this holds without
    reimplementing the rate expression. It also demonstrates that the slow
    fixture is limited by the kinetics rather than by the extent clamp, which
    would otherwise pin both cases to the same value.
    """
    base = _species_rates(
        _build_reactor(BatchReactor, rate_constant=RATE_CONSTANT_KINETIC)
    )
    doubled = _species_rates(
        _build_reactor(BatchReactor, rate_constant=2 * RATE_CONSTANT_KINETIC)
    )

    np.testing.assert_allclose(doubled, 2 * base, rtol=RATIO_RTOL)


def test_fast_kinetics_are_clamped_to_the_available_inventory():
    """Pseudo-instantaneous kinetics cannot drive a species below zero.

    ``RxnKinetics.get_rxn_rates`` rescales the extent when a species would go
    negative, comparing a rate against a concentration and so implicitly over
    a one second step. In this regime the limiting
    reactant A is consumed at exactly its own inventory per second.
    """
    vessel = _build_reactor(BatchReactor, rate_constant=RATE_CONSTANT_CLAMPED)

    rates = _species_rates(vessel)

    initial_masses = CHARGE_MASS * np.array(CHARGE_MASS_FRAC)  # [kg]
    limiting = SPECIES.index("A")

    np.testing.assert_allclose(
        rates[limiting], -initial_masses[limiting], rtol=RATIO_RTOL
    )
    # One second at this rate must leave every species non-negative.
    assert np.all(initial_masses + rates >= -MASS_CLOSURE_ATOL)
    # Mass closure must survive the clamp.
    assert abs(rates.sum()) < MASS_CLOSURE_ATOL


def test_continuous_vessel_holds_volume_by_matching_outlet_to_inlet():
    """The default volume controller withdraws exactly the fed volume.

    DefaultContinuousVesselVolume adopts the initial vessel volume as its
    target, so at t = 0 the volume error is zero and the outlet reduces to the
    inlet volumetric flow. The feed composition equals the charge composition
    here, so every species holdup is then stationary.
    """
    vessel = _build_reactor(ContinuousReactor, with_inlet=True)

    rates = _species_rates(vessel)

    np.testing.assert_allclose(
        rates, np.zeros(len(SPECIES)), atol=MASS_CLOSURE_ATOL
    )

    # Guard against a vacuous pass in which nothing flows at all: the
    # controller must have posted a positive outlet volumetric flow. The feed
    # and the charge share a composition, hence a density, so the fed volume
    # is the charge volume scaled by the fed mass fraction of the charge.
    assert len(vessel.inlet_connections) == 1
    expected_vol_flow = vessel.Phases.vol * (
        INLET_MASS_FLOW / CHARGE_MASS
    )  # [m**3/s]

    posted = [
        value
        for key, value in vessel.controller.operating_conditions.items()
        if key.name == "vol_flow" and key.port == "outlet"
    ]
    assert len(posted) == 1
    assert posted[0] > 0
    np.testing.assert_allclose(posted[0], expected_vol_flow, rtol=RATIO_RTOL)


def test_reaction_runs_while_the_continuous_vessel_holds_volume():
    """Feed, outlet and reaction superpose in one balance.

    With inlet and outlet cancelling, the remaining species rates are the
    reaction terms alone, so the vessel still converts reactants while its
    total mass stays stationary.

    The kinetic rate constant is required, not the clamped one. The outlet is
    no longer hard-clamped to the phase volume (that compared m**3/s against
    m**3); draining a phase faster than it can supply is now throttled by the
    positivity limiter instead. Under the clamped rate constant this fixture
    sits exactly on that boundary, so the limiter scales the outlet down by
    2.5% and the flow terms stop cancelling, which is the limiter working
    rather than a balance defect. The slow rate constant keeps the vessel off
    the boundary, where the premise of this test actually holds.
    """
    vessel = _build_reactor(
        ContinuousReactor,
        rate_constant=RATE_CONSTANT_KINETIC,
        with_inlet=True,
    )

    rates = _species_rates(vessel)

    # Flow terms cancel and reactions conserve mass, so the total is zero.
    assert abs(rates.sum()) < MASS_CLOSURE_ATOL
    # The chemistry is nonetheless active.
    assert rates[SPECIES.index("C")] > 0
    assert rates[SPECIES.index("A")] < 0


def test_cold_jacket_removes_heat_from_the_charge():
    """A jacket below the charge temperature gives a negative dT/dt.

    Only the sign is asserted. The magnitude is currently correct just for a
    1 kg charge, because ``MultiPhaseVessel.energy_balances`` divides the heat
    rate by the specific heat without multiplying by the total mass. The
    mass-scaling contract is asserted separately once that is fixed.
    """
    vessel = _build_reactor(BatchReactor)

    states = vessel.create_solver_init_states()
    rates = np.asarray(vessel.unit_model(0.0, states))

    temperature_rate = rates[-1]  # [K/s]
    assert np.isfinite(temperature_rate)
    assert temperature_rate < 0
