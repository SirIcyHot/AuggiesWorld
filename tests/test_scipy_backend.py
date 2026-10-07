"""Regressions for the scipy.integrate.solve_ivp integrator backend.

ScipyBackend exists so PharmaPy can be run in an environment that has nothing
but scipy, which is already a hard dependency. Most of what is worth pinning
here is therefore behaviour that needs no Assimulo at all: option translation,
the refusal of a differential-algebraic unit, and the event handling. Only
``test_matches_assimulo_on_a_shared_grid`` compares against CVode, and it skips
when Assimulo is absent.

The fixtures use the five-species ``tests/Flowsheet/data`` database, whose
molar masses make both reactions exactly mass conserving (A 100 + B 50 = C 150;
C 150 + A 100 = D 250), matching ``test_multiphase_vessel_reactor.py``.

The event tests attach synthetic StateEvents through the ``state_events``
constructor argument rather than building a crystallizer. A synthetic event is
a sharper instrument here: because it does not influence the right-hand side,
segmenting on it must leave the trajectory unchanged, which is the property
being asserted.
"""

import os

import numpy as np
import pytest

from PharmaPy.DataClasses import StateEvent
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import BatchReactor
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

CHARGE_MASS = 1.0  # [kg]
CHARGE_MASS_FRAC = [0.4, 0.6, 0.0, 0.0, 0.0]  # [-] A, B, C, D, solvent

REACTIONS = ["A + B --> C", "C + A --> D"]
RATE_CONSTANT = 1e-2  # [L/mol/s], slow enough to avoid the extent clamp
ACTIVATION_ENERGY = 1e2  # [J/mol]

UTILITY_MASS_FLOW = 100.0  # [kg/s]
UTILITY_TEMP_IN = 273.55  # [K]
HEAT_TRANSFER_COEFF = 1e4  # [W/m**2/K]
VESSEL_DIAMETER = 0.01  # [m]

RUNTIME = 10.0  # [s]
GRID = np.linspace(0.0, RUNTIME, 21)


def build_reactor(integrator, state_events=()):
    """A batch reactor with no outlet, so it declares no events of its own."""

    vessel = BatchReactor(
        integrator=integrator,
        h_conv=HEAT_TRANSFER_COEFF,
        diam=VESSEL_DIAMETER,
        state_events=list(state_events),
    )

    vessel.Phases = LiquidPhase(
        DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC
    )
    vessel.Utility = CoolingWater(
        mass_flow=UTILITY_MASS_FLOW, temp_in=UTILITY_TEMP_IN
    )
    vessel.RxnKinetics = RxnKinetics(
        path=DATA_PATH,
        rxn_list=REACTIONS,
        k_params=np.array([RATE_CONSTANT, RATE_CONSTANT]),
        ea_params=np.array([ACTIVATION_ENERGY, ACTIVATION_ENERGY]),
    )

    return vessel


def state_event(name, function, direction=0, terminal=False):
    return StateEvent(
        name=name,
        function=function,
        direction=direction,
        terminal=terminal,
        source=None,
    )


# ---------------------------------------------------------------------------
# Fidelity against the Assimulo backend
# ---------------------------------------------------------------------------


def test_matches_assimulo_on_a_shared_grid():
    """Same model, same grid, same answer under CVode and scipy BDF."""

    pytest.importorskip("assimulo", reason="AssimuloBackend needs assimulo")

    from PharmaPy.IntegratorBackends import AssimuloBackend

    # Tight tolerances on both. At the 1e-6 default the two disagree by about
    # 2e-4 on the smallest species, which is ordinary truncation difference
    # between two BDF implementations rather than a model discrepancy - at
    # 1e-10 they converge to 1e-9 on every state, which is what makes this a
    # test of the model and the packing rather than of the step controllers.
    tolerances = {"atol": 1e-10, "rtol": 1e-10}

    # Separate instances: solving mutates the unit's phases.
    reference = build_reactor(
        AssimuloBackend(options=dict(maxh=1, **tolerances))
    )
    candidate = build_reactor(ScipyBackend(options=dict(tolerances)))

    time_ref, states_ref = reference.solve_unit(time_grid=GRID)
    time_new, states_new = candidate.solve_unit(time_grid=GRID)

    # Both must report exactly the requested grid. This is the assertion that
    # catches an off-by-one in the per-segment grid slicing, which a tolerance
    # on the states would hide.
    assert np.array_equal(np.asarray(time_ref), GRID)
    assert np.array_equal(np.asarray(time_new), GRID)

    np.testing.assert_allclose(
        np.asarray(states_new), np.asarray(states_ref), rtol=1e-6, atol=1e-9
    )


# ---------------------------------------------------------------------------
# What the backend refuses
# ---------------------------------------------------------------------------


def test_algebraic_states_are_refused_without_touching_the_unit():
    """solve_ivp has no DAE mode, so a residual row must be refused."""

    vessel = build_reactor(ScipyBackend())
    vessel.compile_structure()

    collection = vessel.solver_state_collection
    key = next(k for k in collection.states if k.name == "global_temp")
    collection.states[key].update_variable("state_type", "alg")
    collection.compile()

    assert vessel.has_algebraic_balance

    backend = ScipyBackend()

    with pytest.raises(NotImplementedError, match="solves ODEs only"):
        backend.compile_integrator(vessel)

    # The refusal happens before unit.reset(), so the caller gets their unit
    # back exactly as it was.
    assert not backend._compiled


def test_sensitivities_are_refused():
    vessel = build_reactor(ScipyBackend())
    vessel.compile_structure()

    with pytest.raises(NotImplementedError, match="sensitivities"):
        ScipyBackend().compile_integrator(vessel, eval_sens=True)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="Unknown method"):
        ScipyBackend(method="CVode")


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_option_translation():
    backend = ScipyBackend(
        options={
            "maxh": 0.5,
            "inith": 1e-3,
            "atol": 1e-8,
            "verbosity": 50,
            "time_limit": 30,
            "linear_solver": "krylov",
        }
    )

    assert backend.translate_options() == {
        "max_step": 0.5,
        "first_step": 1e-3,
        "atol": 1e-8,
        "rtol": 1e-6,
    }

    # The hint is received and recorded even though scipy cannot act on it.
    assert backend.linear_solver == "krylov"
    assert backend.statistics == {}


def test_reserved_options_are_refused():
    backend = ScipyBackend(options={"t_eval": [0.0, 1.0]})

    with pytest.raises(ValueError, match="set by ScipyBackend itself"):
        backend.translate_options()


def test_default_options_are_not_shared_between_instances():
    """AssimuloBackend's options={'maxh': 1} default is one shared dict."""

    first = ScipyBackend()
    first.options["maxh"] = 1

    assert ScipyBackend().options == {}


def test_krylov_request_does_not_raise():
    """Crystallizers_Refactored asks for it on every solve."""

    backend = ScipyBackend()
    backend.set_linear_solver("krylov")

    assert backend.linear_solver == "krylov"


# ---------------------------------------------------------------------------
# Output shape and the right-hand side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_grid", [True, False])
def test_output_shape_and_initial_point(use_grid):
    vessel = build_reactor(ScipyBackend())

    kwargs = {"time_grid": GRID} if use_grid else {"runtime": RUNTIME}
    time, states = vessel.solve_unit(**kwargs)

    time = np.asarray(time)
    states = np.asarray(states)

    assert states.ndim == 2
    assert states.shape[0] == time.size
    assert states.shape[1] == vessel.solver_state_collection.dim

    assert time[0] == pytest.approx(0.0)
    assert time[-1] == pytest.approx(RUNTIME)
    assert np.all(np.diff(time) > 0)

    # retrieve_results consumed the same trajectory.
    assert vessel.result.time.size == time.size

    if use_grid:
        assert np.array_equal(time, GRID)


def test_rhs_result_is_copied_out_of_the_unit_buffer():
    """unit_model refills and returns one buffer; scipy would keep a handle."""

    vessel = build_reactor(ScipyBackend())
    vessel.compile_structure()

    backend = ScipyBackend()
    rhs = backend.make_rhs(vessel)

    states = vessel.create_solver_init_states()

    first = rhs(0.0, states)
    second = rhs(0.0, states * 0.5)

    assert first is not second
    assert first is not vessel._solver_rate_buffer
    assert second is not vessel._solver_rate_buffer
    assert not np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_identically_zero_event_does_not_spin():
    """A surface that never leaves zero must cost nothing.

    A level controller holding its target makes ``unit.Phases.vol - target``
    identically zero over an interval. scipy counts ``g <= 0 and g_new >= 0``
    as a crossing, so such a surface is active on every single step; without
    the disarm band this run would exhaust max_restarts instead of finishing.
    """

    backend = ScipyBackend(max_restarts=5)
    vessel = build_reactor(
        backend,
        [state_event("always_zero", lambda time, state, unit: 0.0)],
    )

    time, _ = vessel.solve_unit(time_grid=GRID)

    assert np.array_equal(np.asarray(time), GRID)
    assert backend.statistics["nrestarts"] == 0
    assert backend.statistics["nsegments"] == 1


def test_state_event_restarts_the_integration_at_each_root():
    backend = ScipyBackend()
    vessel = build_reactor(
        backend,
        [state_event("sine", lambda time, state, unit: np.sin(time))],
    )

    time, _ = vessel.solve_unit(time_grid=GRID)
    time = np.asarray(time)

    assert backend.statistics["nsegments"] > 1
    assert np.all(np.diff(time) > 0)

    for root in (np.pi, 2 * np.pi, 3 * np.pi):
        assert np.min(np.abs(time - root)) < 1e-6

    # Every requested grid point still comes back.
    assert np.all(np.isin(GRID, time))


def test_segmenting_does_not_change_the_trajectory():
    """The synthetic event does not enter the model, so it must not matter."""

    event = [state_event("sine", lambda time, state, unit: np.sin(time))]

    segmented = build_reactor(ScipyBackend(), event)
    straight = build_reactor(ScipyBackend(segment_events=False), event)

    _, states_segmented = segmented.solve_unit(time_grid=GRID)
    _, states_straight = straight.solve_unit(time_grid=GRID)

    # Both report the grid plus, for the segmented run, the event roots.
    time_segmented = np.asarray(segmented.result.time)
    on_grid = np.isin(time_segmented, GRID)

    np.testing.assert_allclose(
        np.asarray(states_segmented)[on_grid],
        np.asarray(states_straight),
        rtol=1e-4,
        atol=1e-8,
    )


def test_segment_events_false_runs_in_one_segment():
    backend = ScipyBackend(segment_events=False)
    vessel = build_reactor(
        backend,
        [state_event("sine", lambda time, state, unit: np.sin(time))],
    )

    vessel.solve_unit(time_grid=GRID)

    assert backend.statistics["nsegments"] == 1
    assert backend.statistics["nrestarts"] == 0


def test_terminal_event_stops_the_run():
    backend = ScipyBackend()
    vessel = build_reactor(
        backend,
        [
            state_event(
                "stop_at_four",
                lambda time, state, unit: time - 4.0,
                terminal=True,
            )
        ],
    )

    time, states = vessel.solve_unit(time_grid=GRID)
    time = np.asarray(time)

    assert time[-1] == pytest.approx(4.0)
    assert np.asarray(states).shape[0] == time.size
    assert vessel.elapsed_time == pytest.approx(4.0)


def test_statistics_report_only_counters_scipy_tracks():
    backend = ScipyBackend()
    vessel = build_reactor(backend)

    vessel.solve_unit(time_grid=GRID)

    statistics = backend.statistics

    assert statistics["nfcns"] > 0
    assert statistics["nsegments"] == 1
    assert statistics["nrestarts"] == 0

    # Counters CVode prints and scipy does not track are absent, not zero.
    for absent in ("nsteps", "nerrfails", "nniters", "nnfails"):
        assert absent not in statistics
