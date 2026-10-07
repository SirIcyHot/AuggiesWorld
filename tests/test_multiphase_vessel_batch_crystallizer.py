"""Driving-force and population-balance contracts for the batch crystallizer.

Tests evaluate ``unit_model`` at t = 0, so they need no integrator.

Units: charge mass [kg], temp [K], x_grid [um], distrib [#/(um m**3)],
conc and solubility [kg/m**3 of PURE SOLVENT], supersat [-] relative,
d(mass_j)/dt [kg/s], d(distrib)/dt [#/(um m**3 s)].

The solvent must be the LAST species: ``compute_supersaturation`` divides by
``mass_j[-1]`` rather than by the solvent index it resolves by name. WARM_TEMP
sits below saturation, COLD_TEMP and DEEP_COLD_TEMP above it.
"""

import os

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.DataClasses import PhaseRef
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

SPECIES = ("A", "B", "C", "D", "solvent")  # solvent LAST, see module docstring
TARGET = "C"
TARGET_INDEX = SPECIES.index(TARGET)
SOLVENT_INDEX = SPECIES.index("solvent")
SPECTATOR_INDICES = [0, 1, 3, 4]  # every species but the target

NUM_GRID = 199  # [-] size classes
GRID_START = 1.0  # [um], also the nucleation size
GRID_STOP = 199.0  # [um]

CHARGE_MASS = 1.0  # [kg]
CHARGE_MASS_FRAC = [0.0, 0.0, 0.2, 0.0, 0.8]  # [-]
RHO_PURE_SOLVENT = 887.6  # [kg/m**3], rho_liq of "solvent" in the database

WARM_TEMP = 313.0  # [K], below saturation
COLD_TEMP = 283.0  # [K], mildly supersaturated
DEEP_COLD_TEMP = 273.15  # [K], strongly supersaturated

# Apelblat solubility coefficients: c_sat = exp(a1 + a2/T + a3*ln T) [kg/m**3].
APELBLAT_COEFFS = np.array([-28.13909202, 0.001, 5.900800253])  # [-, K, -]
NUCL_PRIM = (6.26855218e18, 0.0, 8.30671806)  # k_b [#/m**3/s], E_b [J/mol], b [-]
NUCL_SEC = (0.0, 0.0, 0.0, 0.0)  # disabled
# k_g [um/s], E_g [J/mol], g [-], alpha [-], beta [1/um]. The five-element form
# selects the size-dependent growth branch of the population balance.
GROWTH = (1.45782420e6, 0.0, 4.52241037, 1.0, 3.93676056)

HEAT_TRANSFER_COEFF = 1e4  # [W/m**2/K], unused: the controller owns temperature
VESSEL_DIAMETER = 0.01  # [m]

LIMITER_HORIZON = 1.0  # [s], the step the vessel inventory limiter assumes

# Quantities related to their expected value by closed-form algebra (a
# division, an exponential, a trapezoid) are compared a few orders above
# double-precision roundoff, so a change of evaluation order is not a failure.
ALGEBRAIC_RTOL = 1e-12  # [-]
# Literals transcribed from a run of this fixture are quoted to eight digits.
REFERENCE_RTOL = 1e-6  # [-]
# Anti-vacuity floors. Any real crystallization exceeds these by decades.
ACTIVE_MASS_RATE_FLOOR = 1e-6  # [kg/s]
ACTIVE_NUCLEATION_FLOOR = 1.0  # [#/(um m**3 s)]

# Values recorded from this fixture; see the module docstring for the basis.
REFERENCE_CONC_TARGET = 221.9  # [kg/m**3 of pure solvent]
REFERENCE_SOLUBILITY = {
    WARM_TEMP: 319.93198151,  # [kg/m**3]
    COLD_TEMP: 176.54326883,
    DEEP_COLD_TEMP: 143.24057176,
}
REFERENCE_SUPERSAT = {
    WARM_TEMP: -0.30641507,  # [-]
    COLD_TEMP: 0.25691566,
    DEEP_COLD_TEMP: 0.54914210,
}
# Re-recorded after the population balance moved to an intensive basis. The
# previous value, -0.16583064, was exactly three times this one: the factor of
# three in d(mu_3)/dt = 3*G*mu_2 + B*rad**3 was distributing over the
# nucleation term as well, and with an empty seed the transfer is nucleation
# alone, so the old literal recorded the defect rather than the physics.
REFERENCE_TARGET_RATE = -5.52768793e-02  # [kg/s] at DEEP_COLD_TEMP
# Re-recorded for the same reason. The distribution is now a number density
# per m**3 of slurry rather than an absolute count, so this is larger than the
# previous 4.1349151e11 by roughly the reciprocal of the slurry volume.
REFERENCE_NUCLEATION_RATE = 3.87173920e14  # [#/(um m**3 s)] at COLD_TEMP


def _build_crystallizer(temp, distrib=None, charge_mass=CHARGE_MASS,
                        nucl_pre=NUCL_PRIM[0], growth_pre=GROWTH[0],
                        sup_sat_type="relative", target_components=TARGET):
    """Build a batch crystallizer ready for a solver-free evaluation.

    Parameters
    ----------
    temp : float
        Vessel temperature [K], imposed by a constant-value controller so that
        the energy balance is absent and temperature is not a solver state.
    distrib : numpy.ndarray, optional
        Initial crystal number density on ``x_grid`` [#/(um m**3)]. Defaults to
        an unseeded grid.
    charge_mass : float, optional
        Liquid holdup [kg].
    nucl_pre : float, optional
        Primary nucleation pre-exponential [#/m**3/s].
    growth_pre : float, optional
        Growth pre-exponential [um/s].
    sup_sat_type : str, optional
        Driving-force form: ``'absolute'``, ``'relative'`` or ``'ratio'``.
    target_components : str or list of str, optional
        Crystallizing species.

    Returns
    -------
    BatchCrystallizer
        Vessel with its array layout compiled.

    Notes
    -----
    The mechanism must be attached to the solid before the vessel sees the phases, and the kinetics must be set
    through the vessel property rather than the mechanism constructor, which
    would bypass the setter that builds the size-dependent growth factor.
    """
    vessel = BatchCrystallizer(
        integrator=None,
        h_conv=HEAT_TRANSFER_COEFF,
        diam=VESSEL_DIAMETER,
        controller=SimpleTemperatureController(temp_func=lambda time: temp),
    )

    liquid = LiquidPhase(
        DATA_PATH, mass=charge_mass, mass_frac=CHARGE_MASS_FRAC, temp=temp
    )
    solid = SolidPhase(
        DATA_PATH, mass=0, mass_frac=[0.0, 0.0, 1.0, 0.0, 0.0], temp=temp
    )

    grid = _size_grid()
    mechanism = OneDFVMMechanism(
        solid,
        target_components=target_components,
        solvent_name="solvent",
        x_grid=grid,
        distrib_init=np.zeros(NUM_GRID) if distrib is None else distrib,
    )
    solid.mechanisms = mechanism

    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = CrystKinetics(
        APELBLAT_COEFFS,
        nucl_prim=(nucl_pre,) + NUCL_PRIM[1:],
        nucl_sec=NUCL_SEC,
        growth=(growth_pre,) + GROWTH[1:],
        solubility_type="apelblat",
        sup_sat_type=sup_sat_type,
        # Stated rather than assumed. The mechanism used to hard-code this
        # basis; it is now declared on the kinetics and defaults to a
        # solution basis, so the fixture has to name the one these Apelblat
        # coefficients and the reference values below were recorded in.
        solubility_basis="mass_per_volume_solvent",
    )

    # compile_structure builds the array layout the balances index into. It
    # runs inside solve_unit, but these tests call unit_model directly.
    vessel.compile_structure()
    return vessel


def _size_grid():
    """Return the internal crystal size coordinate.

    Returns
    -------
    numpy.ndarray
        Uniform grid of ``NUM_GRID`` sizes [um], spacing 1 um.
    """
    return np.arange(GRID_START, GRID_STOP + 1.0)


def _mechanism(vessel):
    """Return the population balance attached to the vessel's solid phase.

    Parameters
    ----------
    vessel : BatchCrystallizer
        Compiled vessel.

    Returns
    -------
    OneDFVMMechanism
        The crystallization mechanism.
    """
    solid = vessel.Phases.get_phase_from_ref(PhaseRef("solid", 0))
    return solid.get_mechanism(OneDFVMMechanism)


def _liquid(vessel):
    """Return the vessel's liquid phase.

    Parameters
    ----------
    vessel : BatchCrystallizer
        Compiled vessel.

    Returns
    -------
    LiquidPhase
        The continuous phase.
    """
    return vessel.Phases.get_phase_from_ref(PhaseRef("liquid", 0))


def _derivatives(vessel):
    """Evaluate the material balance once and split it by state.

    Parameters
    ----------
    vessel : BatchCrystallizer
        Compiled vessel.

    Returns
    -------
    tuple of numpy.ndarray
        Liquid species mass rates [kg/s] and crystal number density rates
        [#/(um m**3 s)].

    Notes
    -----
    ``unit_model`` returns the vessel's reusable rate buffer, so the slices are
    copied before they are handed back. Two successive calls otherwise alias.
    """
    states = vessel.create_solver_init_states()
    packed = np.asarray(vessel.unit_model(0.0, states, mat_bce=True)).copy()

    collection = vessel.solver_state_collection
    species_key = next(k for k in collection.states if k.name == "mass_j")
    distrib_key = next(k for k in collection.states if k.name == "distrib")
    return packed[collection.slices[species_key]], packed[collection.slices[distrib_key]]


def _seed_distribution():
    """Return a Gaussian seed carrying substantial crystal surface.

    Returns
    -------
    numpy.ndarray
        Crystal number density [#/(um m**3)] centred at 50 um.
    """
    peak = 1.0e10  # [#/(um m**3)]
    centre = 50.0  # [um]
    width = 15.0  # [um]
    return peak * np.exp(-0.5 * ((_size_grid() - centre) / width) ** 2)


def _apelblat_solubility(temp):
    """Evaluate the declared Apelblat solubility correlation.

    Parameters
    ----------
    temp : float
        Temperature [K].

    Returns
    -------
    float
        Saturation concentration [kg/m**3 of pure solvent].

    Notes
    -----
    Re-implemented from the documented form rather than called from the
    kinetics object, so the assertion is independent of the production path.
    """
    first, second, third = APELBLAT_COEFFS
    return float(np.exp(first + second / temp + third * np.log(temp)))


def test_solver_states_are_liquid_species_then_crystal_distribution():
    """The solver vector is the liquid species holdup then the crystal density.

    Every other test indexes these blocks, so the layout, its units and the
    limiter flags are pinned here rather than assumed. The distribution is a
    number density, not an inventory, so the vessel's negative-inventory
    limiter must not rescale it.
    """
    vessel = _build_crystallizer(WARM_TEMP)
    collection = vessel.solver_state_collection

    layout = [
        (key.name, key.phaseref, state.dim, state.units)
        for key, state in collection.states.items()
    ]
    assert layout == [
        ("mass_j", PhaseRef("liquid", 0), len(SPECIES), "kg"),
        ("distrib", PhaseRef("solid", 0), NUM_GRID, "#/(micron m3 slurry)"),
    ]
    assert collection.dim == len(SPECIES) + NUM_GRID

    states = list(collection.states.values())
    assert states[0].limit_negative_inventory is True
    assert states[1].limit_negative_inventory is False

    initial = vessel.create_solver_init_states()
    expected_masses = CHARGE_MASS * np.array(CHARGE_MASS_FRAC)  # [kg]
    np.testing.assert_allclose(initial[:len(SPECIES)], expected_masses)
    np.testing.assert_array_equal(initial[len(SPECIES):], np.zeros(NUM_GRID))

    round_tripped = collection.pack(collection.unpack(initial))
    np.testing.assert_allclose(round_tripped, initial)


def test_undersaturated_slurry_with_seed_crystals_is_completely_quiescent():
    """Below saturation nothing nucleates, grows or dissolves.

    Asserted as an exact zero rather than against a tolerance. Every path
    multiplies by an identically zero rate, so a nonzero result is a spurious
    source term, which is precisely what this test exists to catch.
    """
    seed = _seed_distribution()
    vessel = _build_crystallizer(WARM_TEMP, distrib=seed)
    mechanism = _mechanism(vessel)

    _, supersat, _ = mechanism.compute_supersaturation(_liquid(vessel))
    assert float(np.atleast_1d(supersat)[0]) < 0

    # The seed carries real crystal surface, so a leaking growth branch would
    # produce an enormous rate rather than a subtle one.
    surface_moment = mechanism.compute_moments(seed, _size_grid())[2]
    assert surface_moment > 0

    species_rates, distribution_rates = _derivatives(vessel)
    assert np.all(species_rates == 0.0)
    assert np.all(distribution_rates == 0.0)

    # Guard against a vacuous pass in which the kinetics are simply dead: the
    # same fixture below saturation must be active.
    active_species, _ = _derivatives(_build_crystallizer(DEEP_COLD_TEMP))
    assert active_species[TARGET_INDEX] < -ACTIVE_MASS_RATE_FLOOR


def test_supersaturation_uses_a_pure_solvent_concentration_basis():
    """Concentration is solute mass per cubic metre of pure solvent.

    Per cubic metre of solution the target would read about 187 kg/m**3 rather
    than 221.9, so this pins the basis and not merely the arithmetic. Cooling
    must lower the solubility and raise the driving force.
    """
    measured = {}
    for temp in (WARM_TEMP, COLD_TEMP, DEEP_COLD_TEMP):
        vessel = _build_crystallizer(temp)
        liquid = _liquid(vessel)
        conc, supersat, solubility = _mechanism(vessel).compute_supersaturation(liquid)

        expected_conc = (
            liquid.mass_j / liquid.mass_j[-1] * RHO_PURE_SOLVENT
        )  # [kg/m**3 of pure solvent]
        np.testing.assert_allclose(conc, expected_conc, rtol=ALGEBRAIC_RTOL)
        np.testing.assert_allclose(
            conc[TARGET_INDEX], REFERENCE_CONC_TARGET, rtol=REFERENCE_RTOL
        )
        # The solvent's own concentration on this basis is its pure density.
        assert conc[SOLVENT_INDEX] == RHO_PURE_SOLVENT

        solubility_value = float(np.atleast_1d(solubility)[0])
        np.testing.assert_allclose(
            solubility_value, _apelblat_solubility(temp), rtol=ALGEBRAIC_RTOL
        )
        np.testing.assert_allclose(
            solubility_value, REFERENCE_SOLUBILITY[temp], rtol=REFERENCE_RTOL
        )
        assert solubility_value > 0

        supersat_value = float(np.atleast_1d(supersat)[0])
        expected_supersat = (
            conc[TARGET_INDEX] - solubility_value
        ) / solubility_value  # [-]
        np.testing.assert_allclose(
            supersat_value, expected_supersat, rtol=ALGEBRAIC_RTOL
        )
        np.testing.assert_allclose(
            supersat_value, REFERENCE_SUPERSAT[temp], rtol=REFERENCE_RTOL
        )

        measured[temp] = (solubility_value, supersat_value)

    # Cooling lowers solubility and raises the driving force.
    assert (
        measured[WARM_TEMP][0] > measured[COLD_TEMP][0] > measured[DEEP_COLD_TEMP][0]
    )
    assert (
        measured[WARM_TEMP][1] < measured[COLD_TEMP][1] < measured[DEEP_COLD_TEMP][1]
    )


def test_supersaturation_type_selects_the_declared_driving_force():
    """The three driving-force conventions are distinct and self-consistent.

    Only the driving force is evaluated, so the sign of the supersaturation is
    irrelevant here and the undersaturated fixture is used.
    """
    values = {}
    for sup_sat_type in ("absolute", "relative", "ratio"):
        vessel = _build_crystallizer(WARM_TEMP, sup_sat_type=sup_sat_type)
        conc, supersat, solubility = _mechanism(vessel).compute_supersaturation(
            _liquid(vessel)
        )
        values[sup_sat_type] = (
            float(np.atleast_1d(supersat)[0]),
            float(conc[TARGET_INDEX]),
            float(np.atleast_1d(solubility)[0]),
        )

    absolute, conc_target, saturation = values["absolute"]
    relative = values["relative"][0]
    ratio = values["ratio"][0]

    np.testing.assert_allclose(
        absolute, conc_target - saturation, rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        relative, (conc_target - saturation) / saturation, rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(ratio, relative + 1.0, rtol=ALGEBRAIC_RTOL)
    # "ratio" means c / c_sat; the code reaches it by adding one to the
    # relative form, so the two expressions are only algebraically equal.
    np.testing.assert_allclose(
        ratio, conc_target / saturation, rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        absolute, relative * saturation, rtol=ALGEBRAIC_RTOL
    )

    # A silently ignored sup_sat_type would return the same value three times.
    assert len({absolute, relative, ratio}) == 3


def test_supersaturation_is_intensive_in_the_charge_size():
    """The driving force depends on composition, not on how much is charged."""
    charges = (0.5, 1.0, 2.0, 137.0)  # [kg], spanning more than two decades

    reference = None
    volumes = []
    for charge_mass in charges:
        vessel = _build_crystallizer(WARM_TEMP, charge_mass=charge_mass)
        liquid = _liquid(vessel)
        conc, supersat, _ = _mechanism(vessel).compute_supersaturation(liquid)
        volumes.append(liquid.vol)

        if reference is None:
            reference = (conc, float(np.atleast_1d(supersat)[0]))
            continue

        np.testing.assert_allclose(conc, reference[0], rtol=ALGEBRAIC_RTOL)
        np.testing.assert_allclose(
            float(np.atleast_1d(supersat)[0]), reference[1], rtol=ALGEBRAIC_RTOL
        )

    # Guard against a vacuous pass: the vessels really do differ in size, and
    # the driving force is not trivially zero.
    assert len(set(volumes)) == len(charges)
    assert reference[1] != 0.0


def test_moment_operators_agree_and_are_linear():
    """The fast moment operators match the generic one and the map is linear.

    The mass-transfer term uses the second and third moments through dedicated
    routines; they must integrate the same distribution as ``compute_moments``.
    """
    mechanism = _mechanism(_build_crystallizer(WARM_TEMP))
    grid = _size_grid()

    generator = np.random.default_rng(0)
    first = generator.random(NUM_GRID) * 1.0e6  # [#/(um m**3)]
    second = generator.random(NUM_GRID) * 1.0e6

    # compute_moments reports mu_n in m**n; compute_second_moment and
    # compute_third_moment stay micron-based, because the mass transfer wants
    # them that way. The conversion is one factor of 1e-6 per moment order, so
    # the two operators agree only once it is applied.
    moments = mechanism.compute_moments(first, grid)
    np.testing.assert_allclose(
        moments[2], mechanism.compute_second_moment(first) * 1e-12,
        rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        moments[3], mechanism.compute_third_moment(first) * 1e-18,
        rtol=ALGEBRAIC_RTOL
    )
    assert moments[3] > 0

    combined = mechanism.compute_third_moment(2.0 * first + 3.0 * second)
    expected = 2.0 * mechanism.compute_third_moment(
        first
    ) + 3.0 * mechanism.compute_third_moment(second)
    np.testing.assert_allclose(combined, expected, rtol=ALGEBRAIC_RTOL)

    # A constant distribution gives closed-form zeroth and first moments, so
    # the quadrature rule can be pinned exactly. The second and third are
    # deliberately not checked against analytic integrals, because quadrature
    # error there is expected rather than a defect.
    #
    # OPEN QUESTION for review. These expectations were the continuum
    # integrals, height*(stop - start) and height*(stop**2 - start**2)/2,
    # which is what the trapezoid rule gives. The scheme now sums over cells
    # instead, on the argument that dcsd_dt = -diff(flux)/dx advances cell
    # averages and the trapezoid halves the first cell, which is the one cell
    # nucleation injects into. That is a real argument, but it also means the
    # moment operator is no longer the integral of the distribution: on this
    # grid mu_0 is 597 rather than 594. The expectations below follow the
    # scheme, written in closed form from this fixture's own grid rather than
    # from the mechanism's internals, so they still pin the rule. Confirm the
    # rectangle rule is intended before treating these literals as settled.
    height = 3.0  # [#/(um m**3)]
    step = grid[1] - grid[0]  # [um], the uniform cell width of this fixture
    flat = np.full(NUM_GRID, height)
    flat_moments = mechanism.compute_moments(flat, grid)
    np.testing.assert_allclose(
        flat_moments[0], height * NUM_GRID * step, rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        flat_moments[1],
        height * step * NUM_GRID * (GRID_START + GRID_STOP) / 2.0 * 1e-6,
        rtol=ALGEBRAIC_RTOL,
    )


def test_crystallization_draws_liquid_mass_only_from_the_target_species():
    """Crystal formation removes liquid mass through the target species alone.

    The solvent and the spectator species are untouched, and the charge is not
    over-drawn within the one-second horizon the inventory limiter assumes.
    """
    vessel = _build_crystallizer(DEEP_COLD_TEMP)
    species_rates, _ = _derivatives(vessel)

    assert np.all(species_rates[SPECTATOR_INDICES] == 0.0)
    # Guard against a vacuous pass: something must actually be crystallizing.
    assert species_rates[TARGET_INDEX] < -ACTIVE_MASS_RATE_FLOOR

    np.testing.assert_allclose(
        species_rates[TARGET_INDEX], species_rates.sum(), rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        species_rates[TARGET_INDEX], REFERENCE_TARGET_RATE, rtol=REFERENCE_RTOL
    )

    remaining = (
        CHARGE_MASS * np.array(CHARGE_MASS_FRAC)
        + species_rates * LIMITER_HORIZON
    )  # [kg]
    assert np.all(remaining >= 0)

    # With a single target the split above is the identity, so the equal-split
    # contract is asserted at mechanism level on a two-target mechanism. It is
    # deliberately not driven through unit_model, which would ask half the
    # crystal mass from a species the charge does not contain.
    two_target = _build_crystallizer(WARM_TEMP, target_components=["C", "D"])
    split = _mechanism(two_target).compute_species_transfer(1.0, _liquid(two_target))
    np.testing.assert_allclose(
        split, [0.0, 0.0, -0.5, -0.5, 0.0], rtol=ALGEBRAIC_RTOL
    )


def test_nucleation_enters_the_grid_only_at_the_smallest_size_class():
    """On an empty grid the only crystal source is the nucleation boundary.

    Its magnitude is set by the nucleation rate alone: the boundary condition
    is ``f(L_0) = B / G`` and the flux through that face is ``G * s(L_0) * f``,
    so the growth rate cancels and only the size factor survives.

    COLD_TEMP is required. At DEEP_COLD_TEMP a doubled nucleation rate asks for
    more target species than the charge holds within the limiter horizon.
    """
    baseline_species, baseline_distrib = _derivatives(
        _build_crystallizer(COLD_TEMP)
    )
    fast_nucleation, fast_distrib = _derivatives(
        _build_crystallizer(COLD_TEMP, nucl_pre=2.0 * NUCL_PRIM[0])
    )
    fast_growth, growth_distrib = _derivatives(
        _build_crystallizer(COLD_TEMP, growth_pre=2.0 * GROWTH[0])
    )

    # Nuclei appear in the smallest size class and nowhere else.
    assert np.all(baseline_distrib[1:] == 0.0)
    assert baseline_distrib[0] > ACTIVE_NUCLEATION_FLOOR
    np.testing.assert_allclose(
        baseline_distrib[0], REFERENCE_NUCLEATION_RATE, rtol=REFERENCE_RTOL
    )

    np.testing.assert_allclose(
        fast_distrib[0], 2.0 * baseline_distrib[0], rtol=ALGEBRAIC_RTOL
    )
    np.testing.assert_allclose(
        growth_distrib[0], baseline_distrib[0], rtol=ALGEBRAIC_RTOL
    )

    # The same holds for the mass drawn from the liquid: with no crystals
    # suspended there is no surface to grow on, so only nucleation moves mass.
    np.testing.assert_allclose(
        fast_nucleation[TARGET_INDEX],
        2.0 * baseline_species[TARGET_INDEX],
        rtol=ALGEBRAIC_RTOL,
    )
    np.testing.assert_allclose(
        fast_growth[TARGET_INDEX],
        baseline_species[TARGET_INDEX],
        rtol=ALGEBRAIC_RTOL,
    )
