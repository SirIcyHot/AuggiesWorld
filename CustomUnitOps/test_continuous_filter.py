"""Tests for the standalone continuous filter.

The strongest check is the handover: the cake is fed to a
`ContinuousCrystallizer`, the next unit in the intended line, and the crystal
flow that vessel reads from it must equal the crystal flow the filter put in
it. That is the quantity the per-m3-of-slurry storage makes easy to lose.

Units: mass flow [kg/s], size [um], number density [#/(um m3)], number flow
[#/(um s)].
"""

import os
import warnings

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
from PharmaPy.DataClasses import PhaseRef
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Streams_Refactored import LiquidStream

from continuous_filter import (ContinuousFilter, SlurryFlow, flow_from_vessel,
                               flow_from_stream, mix, stream_from_flow)

DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'tests',
                         'Flowsheet', 'data', 'compound_database.json')

SPECIES = ['A', 'B', 'C', 'D', 'solvent']
LIQUID_FRAC = np.array([0.0, 0.0, 0.05, 0.0, 0.95])
SOLID_FRAC = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
X_GRID = np.arange(1.0, 200.0)                       # [um]
DENSITY = 1200.0                                     # [kg/m3]
TEMP = 300.0                                         # [K]

LIQUID_FLOW = 0.010                                  # [kg/s]
SOLID_FLOW = 0.002                                   # [kg/s]


def crystal_shape():
    return np.exp(-0.5 * ((X_GRID - 80.0) / 20.0) ** 2)


def slurry_feed(liquid_flow=LIQUID_FLOW, solid_flow=SOLID_FLOW, temp=TEMP):
    """A SlurryFlow with a known crystal mass flow."""
    shape = crystal_shape()
    mu_3 = np.sum(shape * X_GRID**3 * 1.0)           # dx = 1 um
    scale = solid_flow / (DENSITY * mu_3 * 1e-18)

    return SlurryFlow(
        path_thermo=DATA_PATH, name_species=SPECIES, temp=temp,
        liquid_mass_j=liquid_flow * LIQUID_FRAC, name_solv='solvent',
        x_grid=X_GRID, crystal_flow=scale * shape,
        solid_mass_frac=SOLID_FRAC, solid_density=DENSITY,
        target_components=['C'], solvent_name='solvent')


def receiving_crystallizer(stream, seed=0.0):
    """A ContinuousCrystallizer fed `stream`, evaluated once at t = 0.

    Run warm with zero nucleation and growth, so the only terms in its
    population balance are the feed and the product withdrawal.

    Parameters
    ----------
    seed : float, optional
        Peak of the crystal population already in the vessel [#/(um m3)].
    """
    vessel = ContinuousCrystallizer(integrator=None, h_conv=0, diam=0.01,
                                    isothermal=True)

    liquid = LiquidPhase(DATA_PATH, mass=1.0, mass_frac=[0, 0, 0, 0, 1],
                         temp=320.0, name_solv='solvent')
    solid = SolidPhase(DATA_PATH, mass=0, mass_frac=SOLID_FRAC, temp=320.0)
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components='C', solvent_name='solvent',
        x_grid=X_GRID, distrib_init=seed * crystal_shape(), density=DENSITY)

    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = CrystKinetics(
        np.array([-28.13909202, 0.001, 5.900800253]),
        nucl_prim=(0.0, 0.0, 1.0), nucl_sec=(0.0, 0.0, 0.0, 0.0),
        growth=(0.0, 0.0, 1.0), solubility_type='apelblat')
    vessel.Inlet = stream
    vessel.compile_structure()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        states = vessel.create_solver_init_states()
        vessel.unit_model(0.0, states, mat_bce=True)

    return vessel


def resolved_crystal_feed(vessel):
    """Crystal flow [#/(um s)] the vessel resolved from its feed.

    This is the vessel's own reading, before any mechanism uses it: the
    stream's crystal density times the slurry flow summed over the feed's
    phases, which is what add_inlet_terms hands the population balance.
    """
    buffer = vessel._material_contributions
    transfers = buffer.aux[buffer.INLET]

    slurry_vol_flow = sum(t.vol_flow or 0.0 for t in transfers)
    solid = next(t for t in transfers
                 if t.mapping.source_phaseref.phase_type == 'solid')
    density = solid.stream_phase.get_mechanism(OneDFVMMechanism).distrib

    return slurry_vol_flow * np.asarray(density)


def crystal_feed_in_balance(vessel):
    """Crystal flow [#/(um s)] implied by the population balance's feed term.

    The feed term is dn/dt = (Q / V_slurry) * n_in, so the crystal flow it
    represents is dn/dt * V_slurry.
    """
    buffer = vessel._material_contributions
    collection = vessel.solver_state_collection
    key = next(k for k in collection.states if k.name == 'distrib')
    dndt = np.asarray(buffer.contributions[buffer.INLET]
                      [collection.material_slices[key]])

    mechanism = vessel.Phases.get_phase_from_ref(
        PhaseRef('solid', 0)).get_mechanism(OneDFVMMechanism)

    return dndt * mechanism._slurry_volume(mechanism.distrib)


# ---------------------------------------------------------------------------
# Mass balance and settings
# ---------------------------------------------------------------------------


def test_mass_balance_closes():
    filt = ContinuousFilter(solids_loss=0.02, cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    filt.solve()
    s = filt.summary()

    assert abs(s['mass_balance_error_kg_s']) < 1e-15
    assert s['feed_solid_kg_s'] == pytest.approx(SOLID_FLOW, rel=1e-12)


def test_solids_loss_sets_recovery():
    filt = ContinuousFilter(solids_loss=0.05, cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    filt.solve()
    s = filt.summary()

    assert s['solids_recovery'] == pytest.approx(0.95, rel=1e-12)
    assert s['filtrate_solid_kg_s'] == pytest.approx(0.05 * SOLID_FLOW,
                                                     rel=1e-12)


def test_size_dependent_loss_removes_fines_only():
    loss = np.where(X_GRID < 50.0, 1.0, 0.0)          # everything under 50 um
    filt = ContinuousFilter(solids_loss=loss, cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    filt.solve()

    assert np.all(filt.cake_flow.crystal_flow[X_GRID < 50.0] == 0)
    assert np.all(filt.filtrate_flow.crystal_flow[X_GRID >= 50.0] == 0)


@pytest.mark.parametrize('basis, moisture, expected_ratio', [
    ('wet', 0.3, 0.3 / 0.7),
    ('dry', 0.5, 0.5),
])
def test_cake_moisture_sets_retained_liquid(basis, moisture, expected_ratio):
    filt = ContinuousFilter(solids_loss=0.0, cake_moisture=moisture,
                            moisture_basis=basis)
    filt.Inlets = slurry_feed()
    filt.solve()
    s = filt.summary()

    assert s['cake_liquid_kg_s'] / s['cake_solid_kg_s'] == pytest.approx(
        expected_ratio, rel=1e-12)

    if basis == 'wet':
        assert s['cake_lod'] == pytest.approx(moisture, rel=1e-12)


def test_retained_liquid_is_mother_liquor():
    filt = ContinuousFilter(cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    cake, filtrate = filt.solve()

    np.testing.assert_allclose(cake.Phases[0].mass_frac, LIQUID_FRAC)
    np.testing.assert_allclose(filtrate.Phases[0].mass_frac, LIQUID_FRAC)


def test_too_little_liquid_warns_and_keeps_it_all():
    filt = ContinuousFilter(cake_moisture=0.9)       # wants 9 kg/kg solid
    filt.Inlets = slurry_feed()

    with pytest.warns(UserWarning, match='drier'):
        filt.solve()

    s = filt.summary()
    assert s['filtrate_liquid_kg_s'] == 0
    assert s['cake_liquid_kg_s'] == pytest.approx(LIQUID_FLOW)


def test_clean_filtrate_has_no_solid_phase():
    """A vessel with no solid phase rejects a stream that carries one."""
    filt = ContinuousFilter(solids_loss=0.0)
    filt.Inlets = slurry_feed()
    _, filtrate = filt.solve()

    assert [p.phase_family for p in filtrate.Phases] == ['liquid']


@pytest.mark.parametrize('value, basis', [(1.0, 'wet'), (-0.1, 'dry')])
def test_rejects_impossible_moisture(value, basis):
    filt = ContinuousFilter(cake_moisture=value, moisture_basis=basis)
    filt.Inlets = slurry_feed()

    with pytest.raises(ValueError):
        filt.solve()


# ---------------------------------------------------------------------------
# Several feeds
# ---------------------------------------------------------------------------


def test_feeds_add():
    a = slurry_feed(liquid_flow=0.01, solid_flow=0.002)
    b = slurry_feed(liquid_flow=0.004, solid_flow=0.001)
    wash = LiquidStream(DATA_PATH, mass_flow=0.003, mass_frac=[0, 0, 0, 0, 1],
                        temp=TEMP, name_solv='solvent', verbose=False)

    filt = ContinuousFilter(cake_moisture=0.3)
    filt.Inlets = [a, b, wash]
    filt.solve()
    s = filt.summary()

    assert s['feed_solid_kg_s'] == pytest.approx(0.003, rel=1e-12)
    assert s['feed_liquid_kg_s'] == pytest.approx(0.017, rel=1e-12)


def test_mixed_temperature_is_mass_weighted():
    a = slurry_feed(liquid_flow=0.01, solid_flow=0.0, temp=300.0)
    a.crystal_flow[:] = 0
    b = slurry_feed(liquid_flow=0.01, solid_flow=0.0, temp=320.0)
    b.crystal_flow[:] = 0

    assert mix([a, b]).temp == pytest.approx(310.0)


def test_rejects_mismatched_grids():
    a = slurry_feed()
    b = slurry_feed()
    b.x_grid = b.x_grid * 2

    with pytest.raises(ValueError, match='size grid'):
        mix([a, b])


# ---------------------------------------------------------------------------
# Stream round trip and the handover to a crystallizer
# ---------------------------------------------------------------------------


def test_stream_round_trip_preserves_flows():
    feed = slurry_feed()
    back = flow_from_stream(stream_from_flow(feed))

    np.testing.assert_allclose(back.crystal_flow, feed.crystal_flow,
                               rtol=1e-12)
    np.testing.assert_allclose(back.liquid_mass_j, feed.liquid_mass_j,
                               rtol=1e-12)


def test_solid_stream_reports_the_crystal_mass_flow():
    filt = ContinuousFilter(cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    cake, _ = filt.solve()

    assert cake.Phases[1].mass_flow == pytest.approx(
        filt.cake_flow.solid_mass_flow, rel=1e-12)


def test_crystallizer_resolves_the_cake_crystals_the_filter_sent():
    filt = ContinuousFilter(solids_loss=0.02, cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    cake, _ = filt.solve()

    read = resolved_crystal_feed(receiving_crystallizer(cake))

    np.testing.assert_allclose(read, filt.cake_flow.crystal_flow, rtol=1e-9)


def test_crystallizer_resolves_the_cake_liquid_the_filter_sent():
    filt = ContinuousFilter(solids_loss=0.02, cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    cake, _ = filt.solve()

    vessel = receiving_crystallizer(cake)
    buffer = vessel._material_contributions
    liquid = next(t for t in buffer.aux[buffer.INLET]
                  if t.mapping.source_phaseref.phase_type == 'liquid')

    np.testing.assert_allclose(liquid.species_flow,
                               filt.cake_flow.liquid_mass_j, rtol=1e-9)


@pytest.mark.parametrize('seed', [0.0, 1e9])
def test_crystallizer_balance_receives_the_cake_crystals(seed):
    filt = ContinuousFilter(solids_loss=0.02, cake_moisture=0.3)
    filt.Inlets = slurry_feed()
    cake, _ = filt.solve()

    received = crystal_feed_in_balance(receiving_crystallizer(cake, seed))

    np.testing.assert_allclose(received, filt.cake_flow.crystal_flow,
                               rtol=1e-6)


def test_reads_a_solved_vessel_product_stream():
    """flow_from_vessel splits the published slurry flow between the phases.

    A stand-in carries a real vessel's phases and the trajectory format
    MultiPhaseVessel.outputs publishes, so no integration is needed.
    """
    vessel = receiving_crystallizer(stream_from_flow(slurry_feed()), seed=1e9)
    liquid = vessel.Phases.get_phase_from_ref(PhaseRef('liquid', 0))
    mechanism = vessel.Phases.get_phase_from_ref(
        PhaseRef('solid', 0)).get_mechanism(OneDFVMMechanism)

    vol_flow = 1e-5                                   # [m3/s]

    class SolvedVessel:
        Phases = vessel.Phases
        outputs = {
            'vol_flow': np.array([2e-5, vol_flow]),
            'temp': np.array([310.0, 320.0]),
            'mole_conc': np.vstack([liquid.mole_conc, liquid.mole_conc]),
            'distrib': np.vstack([0 * mechanism.distrib, mechanism.distrib]),
        }

    flow = flow_from_vessel(SolvedVessel())           # last time point

    phi = mechanism.solid_volume_fraction(mechanism.distrib)
    expected_liquid = (liquid.getDensity() * (1 - phi) * vol_flow
                       * np.asarray(liquid.mass_frac))

    assert flow.temp == 320.0
    np.testing.assert_allclose(flow.crystal_flow, mechanism.distrib * vol_flow)
    np.testing.assert_allclose(flow.liquid_mass_j, expected_liquid,
                               rtol=1e-9, atol=1e-15)
