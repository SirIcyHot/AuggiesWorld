"""Withdrawal and flow contracts for the continuous crystallizer.

Tests evaluate ``unit_model`` at t = 0, so they need no integrator.

Units: charge mass [kg], temp [K], x_grid [um], distrib [#/(um m**3)],
feed [kg/s], vol_flow [m**3/s], residence time [s], d(mass_j)/dt [kg/s],
d(distrib)/dt [#/(um m**3 s)].

Every test runs undersaturated, where nucleation and growth are inactive, so
the only mechanism acting on the crystal population is withdrawal through the
product stream. Formation is covered in the batch module.
"""

import os

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Streams_Refactored import LiquidStream

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

SPECIES = ("A", "B", "C", "D", "solvent")  # solvent LAST
TARGET = "C"
TARGET_INDEX = SPECIES.index(TARGET)

NUM_GRID = 199  # [-] size classes
GRID_START = 1.0  # [um]
GRID_STOP = 199.0  # [um]

CHARGE_MASS = 1.0  # [kg]
CHARGE_MASS_FRAC = [0.0, 0.0, 0.2, 0.0, 0.8]  # [-]
FEED_MASS_FLOW = 0.01  # [kg/s], feed composition matches the charge

# Undersaturated, so the kinetics select the unparameterised dissolution
# branch and only withdrawal acts on the population.
WARM_TEMP = 313.0  # [K]

APELBLAT_COEFFS = np.array([-28.13909202, 0.001, 5.900800253])  # [-, K, -]
NUCL_PRIM = (6.26855218e18, 0.0, 8.30671806)
NUCL_SEC = (0.0, 0.0, 0.0, 0.0)
GROWTH = (1.45782420e6, 0.0, 4.52241037, 1.0, 3.93676056)

VESSEL_DIAMETER = 0.01  # [m]

SEED_PEAK = 1.0e10  # [#/(um m**3)]
SEED_CENTRE = 50.0  # [um]
SEED_WIDTH = 15.0  # [um]

# Recorded from this fixture. The liquid-only figure is listed because it is
# what a residence time built on the wrong volume would produce.
REFERENCE_RESIDENCE_TIME = 106.34780878  # [s], on the slurry volume
LIQUID_ONLY_RESIDENCE_TIME = 100.0  # [s], the wrong answer

# Closed-form comparisons (a division, a trapezoid) sit a few orders above
# double-precision roundoff.
ALGEBRAIC_RTOL = 1e-12  # [-]
REFERENCE_RTOL = 1e-6  # [-]
# An idle vessel resolves an outlet of order 1e-20 m**3/s, so its rates are
# roundoff rather than exact zeros.
IDLE_RATE_ATOL = 1e-15  # [kg/s] and [#/(um m**3 s)]


def _size_grid():
    """Return the internal crystal size coordinate.

    Returns
    -------
    numpy.ndarray
        Uniform grid of ``NUM_GRID`` sizes [um], spacing 1 um.
    """
    return np.arange(GRID_START, GRID_STOP + 1.0)


def _seed_distribution(scale=1.0):
    """Return a Gaussian crystal population.

    Parameters
    ----------
    scale : float, optional
        Multiplier on the peak number density [-].

    Returns
    -------
    numpy.ndarray
        Crystal number density [#/(um m**3)].
    """
    grid = _size_grid()
    return scale * SEED_PEAK * np.exp(
        -0.5 * ((grid - SEED_CENTRE) / SEED_WIDTH) ** 2
    )


def _build_crystallizer(distrib=None, with_feed=True, temp=WARM_TEMP):
    """Build a continuous crystallizer ready for a solver-free evaluation.

    Parameters
    ----------
    distrib : numpy.ndarray, optional
        Initial crystal number density [#/(um m**3)]. Defaults to unseeded.
    with_feed : bool, optional
        When True, attach a liquid feed whose composition matches the charge.
    temp : float, optional
        Vessel temperature [K].

    Returns
    -------
    ContinuousCrystallizer
        Vessel with its array layout compiled.

    Notes
    -----
    The mechanism must be attached to the solid before the vessel sees the
    phases, and the kinetics must be set through the vessel property, which
    is what wires the phase connections.
    """
    vessel = ContinuousCrystallizer(
        integrator=None, h_conv=0, diam=VESSEL_DIAMETER, isothermal=True
    )

    liquid = LiquidPhase(
        DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC, temp=temp
    )
    solid = SolidPhase(
        DATA_PATH, mass=0, mass_frac=[0.0, 0.0, 1.0, 0.0, 0.0], temp=temp
    )

    mechanism = OneDFVMMechanism(
        solid,
        target_components=TARGET,
        solvent_name="solvent",
        x_grid=_size_grid(),
        distrib_init=np.zeros(NUM_GRID) if distrib is None else distrib,
    )
    solid.mechanisms = mechanism

    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = CrystKinetics(
        APELBLAT_COEFFS,
        nucl_prim=NUCL_PRIM,
        nucl_sec=NUCL_SEC,
        growth=GROWTH,
        solubility_type="apelblat",
    )

    if with_feed:
        vessel.Inlet = LiquidStream(
            DATA_PATH,
            mass_flow=FEED_MASS_FLOW,
            mass_frac=CHARGE_MASS_FRAC,
            temp=temp,
        )

    # compile_structure builds the array layout the balances index into. It
    # runs inside solve_unit, but these tests call unit_model directly.
    vessel.compile_structure()
    return vessel


def _mechanism(vessel):
    """Return the population balance attached to the solid phase.

    Parameters
    ----------
    vessel : ContinuousCrystallizer
        Compiled vessel.

    Returns
    -------
    OneDFVMMechanism
        The crystallization mechanism.
    """
    from PharmaPy.DataClasses import PhaseRef

    solid = vessel.Phases.get_phase_from_ref(PhaseRef("solid", 0))
    return solid.get_mechanism(OneDFVMMechanism)


def _liquid(vessel):
    """Return the vessel's liquid phase.

    Parameters
    ----------
    vessel : ContinuousCrystallizer
        Compiled vessel.

    Returns
    -------
    LiquidPhase
        The continuous phase.
    """
    from PharmaPy.DataClasses import PhaseRef

    return vessel.Phases.get_phase_from_ref(PhaseRef("liquid", 0))


def _derivatives(vessel):
    """Evaluate the material balance once and split it by state.

    Parameters
    ----------
    vessel : ContinuousCrystallizer
        Compiled vessel.

    Returns
    -------
    tuple of numpy.ndarray
        Liquid species mass rates [kg/s] and crystal number density rates
        [#/(um m**3 s)].

    Notes
    -----
    ``unit_model`` returns the vessel's reusable buffer, so the slices are
    copied before they are returned.
    """
    states = vessel.create_solver_init_states()
    packed = np.asarray(vessel.unit_model(0.0, states, mat_bce=True)).copy()

    collection = vessel.solver_state_collection
    species_key = next(k for k in collection.states if k.name == "mass_j")
    distrib_key = next(k for k in collection.states if k.name == "distrib")
    return (
        packed[collection.slices[species_key]],
        packed[collection.slices[distrib_key]],
    )


def _outlet_vol_flow(vessel):
    """Return the volumetric discharge the controller posted.

    Parameters
    ----------
    vessel : ContinuousCrystallizer
        Compiled vessel, already evaluated at least once.

    Returns
    -------
    float
        Outlet volumetric flow [m**3/s].
    """
    posted = [
        value
        for key, value in vessel.controller.operating_conditions.items()
        if key.name == "vol_flow" and key.port == "outlet"
    ]
    assert len(posted) == 1
    return float(posted[0])


def test_suspended_crystals_wash_out_at_the_slurry_residence_time():
    """Withdrawal removes the population at minus one over the residence time.

    With the vessel undersaturated nothing nucleates or grows, so the only
    term acting on the distribution is discharge through the product. The
    residence time is built on the slurry volume, not the liquid volume; the
    two differ here by more than six percent, so this pins the basis.
    """
    seed = _seed_distribution()
    vessel = _build_crystallizer(distrib=seed)

    species_rates, distribution_rates = _derivatives(vessel)
    mechanism = _mechanism(vessel)

    outlet_flow = _outlet_vol_flow(vessel)  # [m**3/s]
    slurry_volume = mechanism._slurry_volume()  # [m**3]
    residence_time = slurry_volume / outlet_flow  # [s]

    np.testing.assert_allclose(
        distribution_rates, -seed / residence_time, rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        residence_time, REFERENCE_RESIDENCE_TIME, rtol=REFERENCE_RTOL
    )

    # Guard against a vacuous pass: crystals are present and really leaving.
    assert np.max(seed) > 0
    assert np.min(distribution_rates) < 0

    # A residence time built on the liquid volume alone would be the round
    # 100 s, which the assertion above would not tolerate.
    liquid_only = _liquid(vessel).vol / outlet_flow  # [s]
    np.testing.assert_allclose(
        liquid_only, LIQUID_ONLY_RESIDENCE_TIME, rtol=REFERENCE_RTOL
    )
    assert residence_time > liquid_only


def test_residence_time_responds_to_the_solid_holdup():
    """More suspended solid lengthens the residence time.

    Because the slurry volume carries the crystals, doubling the population
    does not double the washout rate: the vessel holds more, so each crystal
    stays longer. The withdrawal law still holds at both loadings with its
    own residence time.
    """
    light = _build_crystallizer(distrib=_seed_distribution())
    heavy = _build_crystallizer(distrib=_seed_distribution(scale=2.0))

    _, light_rates = _derivatives(light)
    _, heavy_rates = _derivatives(heavy)

    light_tau = _mechanism(light)._slurry_volume() / _outlet_vol_flow(light)
    heavy_tau = _mechanism(heavy)._slurry_volume() / _outlet_vol_flow(heavy)

    np.testing.assert_allclose(
        light_rates, -_seed_distribution() / light_tau, rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        heavy_rates,
        -_seed_distribution(scale=2.0) / heavy_tau,
        rtol=ALGEBRAIC_RTOL,
    )

    # The extra solid really did lengthen the residence time, so the washout
    # rate grows by less than the factor of two applied to the population.
    assert heavy_tau > light_tau
    peak_ratio = np.min(heavy_rates) / np.min(light_rates)  # [-]
    assert 1.0 < peak_ratio < 2.0
    np.testing.assert_allclose(
        peak_ratio, 2.0 * light_tau / heavy_tau, rtol=ALGEBRAIC_RTOL
    )


def test_matched_feed_leaves_the_liquid_stationary():
    """A crystal-free vessel fed its own composition holds every species.

    The controller withdraws the fed volume, and with no solid present the
    discharge is pure liquid, so inflow and outflow cancel exactly.
    """
    vessel = _build_crystallizer()

    species_rates, distribution_rates = _derivatives(vessel)

    assert np.all(species_rates == 0.0)
    assert np.all(distribution_rates == 0.0)
    # Guard against a vacuous pass in which nothing flows at all.
    assert _outlet_vol_flow(vessel) > 0


def test_volume_controller_withdraws_the_fed_volumetric_flow():
    """At its target volume the controller discharges exactly what is fed."""
    vessel = _build_crystallizer()
    _derivatives(vessel)

    feed = LiquidStream(
        DATA_PATH,
        mass_flow=FEED_MASS_FLOW,
        mass_frac=CHARGE_MASS_FRAC,
        temp=WARM_TEMP,
    )

    np.testing.assert_allclose(
        _outlet_vol_flow(vessel), feed.vol_flow, rtol=ALGEBRAIC_RTOL
    )


def test_slurry_discharge_leaves_the_liquid_accumulating():
    """Crystals occupy part of the discharge, so liquid builds up.

    The controller withdraws a fixed volume of slurry. When a fraction of
    that volume is solid, less liquid leaves than enters, and the liquid
    holdup grows at the volumetric flow times the solid volume fraction times
    the liquid density. The accumulation carries the feed composition.
    """
    vessel = _build_crystallizer(distrib=_seed_distribution())

    species_rates, _ = _derivatives(vessel)
    mechanism = _mechanism(vessel)
    liquid = _liquid(vessel)

    outlet_flow = _outlet_vol_flow(vessel)  # [m**3/s]
    slurry_volume = mechanism._slurry_volume()  # [m**3]
    solid_fraction = (slurry_volume - liquid.vol) / slurry_volume  # [-]

    expected_gain = (
        outlet_flow * solid_fraction * liquid.getDensity()
    )  # [kg/s]
    np.testing.assert_allclose(
        species_rates.sum(), expected_gain, rtol=ALGEBRAIC_RTOL
    )

    # Guard against a vacuous pass: solids really occupy part of the volume.
    assert solid_fraction > 0
    assert species_rates.sum() > 0

    # What accumulates is feed, so it arrives in the feed composition.
    np.testing.assert_allclose(
        species_rates / species_rates.sum(),
        CHARGE_MASS_FRAC,
        atol=ALGEBRAIC_RTOL,
    )


def test_idle_undersaturated_vessel_does_nothing():
    """An unfed, crystal-free, undersaturated vessel is inert.

    A seeded vessel would not qualify: the controller still resolves a
    residual outlet of order 1e-20 m**3/s, and against a population of 1e10
    that still withdraws crystals at a measurable rate. With no crystals
    there is nothing to withdraw and nothing to form, so what remains is
    roundoff, compared against an absolute floor for that reason.
    """
    vessel = _build_crystallizer(with_feed=False)

    species_rates, distribution_rates = _derivatives(vessel)

    np.testing.assert_allclose(
        species_rates, np.zeros(len(SPECIES)), atol=IDLE_RATE_ATOL
    )
    np.testing.assert_allclose(
        distribution_rates, np.zeros(NUM_GRID), atol=IDLE_RATE_ATOL
    )

    # Guard against a vacuous pass: the same vessel with a feed is active.
    fed_species, fed_distribution = _derivatives(
        _build_crystallizer(distrib=_seed_distribution())
    )
    assert np.min(fed_distribution) < -IDLE_RATE_ATOL


def test_product_stream_carries_both_phases():
    """The discharge is a slurry, not a clarified liquid.

    A continuous crystallizer exists to remove crystals, so its outlet must
    map the solid phase as well as the liquid one.
    """
    vessel = _build_crystallizer(distrib=_seed_distribution())

    assert len(vessel.outlet_connections) == 1
    mapped = {
        mapping.sink_phaseref.phase_type
        for mapping in vessel.outlet_connections[0].phase_mappings
    }
    assert mapped == {"liquid", "solid"}

    families = [phase.phase_family for phase in vessel.Outlet.Phases]
    assert "liquid" in families
    assert "solid" in families
