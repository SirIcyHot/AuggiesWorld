"""Plant-identification data for CR02R (used by ``CR02R_control_theory.ipynb``).

The plant for the temperature loop is

    u = jacket (coolant) inlet temperature  -->  CR02R  -->  y = vessel T

with the level held by PharmaPy's PI level loop throughout (it is part of
the plant as the temperature loop sees it). This script produces the two
heavy pieces the notebook needs and caches them in ``results/control/``:

1. ``plant_step_test.npz`` -- an open-loop step test on the full nonlinear
   model. The jacket inlet sits at ``U_BASE`` until ``T_STEP1`` (the plant
   settles), steps by ``+DU`` (identification data), then at ``T_STEP2`` to
   ``-DU`` (a validation step twice the size, opposite sign).
2. ``plant_linearization.npz`` -- the Jacobian linearization of the full
   model at the pre-step operating point: ``A = df/dx``, ``B = df/du`` by
   finite differences of PharmaPy's own right-hand side
   (``MultiPhaseVessel.unit_model``), and ``C`` picking out the
   temperature. This is the exact LTI state-space model the step test is
   approximated by, with every population-balance bin as a state.

Run it with::

    python CR02R_plant_id.py            # step test + linearization (~35 min)
    python CR02R_plant_id.py --linearize-only   # reuse the step test
    python CR02R_plant_id.py --closed-loop      # closed-loop validation run

``--closed-loop`` runs ``CR02R_control``'s ``jacket`` mode (PharmaPy's PI
jacket loop, default tuning) with the set point stepping -1 K at
``T_STEP1`` and saves ``closed_loop_jacket.npz``, so the notebook can check
its predicted closed-loop response against the nonlinear model.
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "R02R"))

import CR02R  # noqa: E402
import R02R_control as control  # noqa: E402
from PharmaPy.DataClasses import StateKey  # noqa: E402
from PharmaPy.ProcessControl_Refactored import (  # noqa: E402
    JacketPIDContinuousVesselController,
    PIDContinuousVesselVolume,
)

OUT_DIR = HERE / "results" / "control"
STEP_FILE = OUT_DIR / "plant_step_test.npz"
LIN_FILE = OUT_DIR / "plant_linearization.npz"
CLOSED_FILE = OUT_DIR / "closed_loop_jacket.npz"

U_BASE = 60 + 273.15      # K, CR02R's usual coolant inlet temperature
DU = 2.0                  # K, identification step
T_STEP1 = 6 * 3600.0      # s, plant settled by now (~5 residence times)
T_STEP2 = 9 * 3600.0      # s
RUNTIME = 12 * 3600.0     # s
DT_SAVE = 30.0            # s, well under the ~3.6 min thermal time constant

SETPOINT_BASE = 60 + 273.15
SETPOINT_STEP = -1.0      # K, closed-loop validation step at T_STEP1
CLOSED_RUNTIME = 8 * 3600.0


def jacket_schedule(t):
    """Open-loop input u(t) [K]."""
    if t < T_STEP1:
        return U_BASE
    if t < T_STEP2:
        return U_BASE + DU
    return U_BASE - DU


class JacketInputController(PIDContinuousVesselVolume):
    """PI level loop (PharmaPy's) plus an open-loop jacket inlet
    temperature ``u(t) + du``. ``du`` is the input perturbation used for
    the B matrix."""

    def __init__(self, u_func, **level):
        super().__init__(**level)
        self.u_func = u_func
        self.du = 0.0

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):
        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)
        if resolved_inlets is None:
            self.operating_conditions[unit.UTILITY_TEMPERATURE] = (
                self.u_func(time) + self.du)


def build_open_loop(u_func=jacket_schedule):
    vessel = control.install_controller(
        CR02R.build_reactor(), JacketInputController(u_func))
    return control.wrap("CR02R", vessel, None)


def state_at(vessel, index):
    """The packed solver-state vector at saved point ``index``."""
    collection = vessel.solver_state_collection
    result = vessel.result
    state = {key: np.asarray(getattr(result, collection.format_key(key)))[index]
             for key in collection.states}
    return np.asarray(collection.pack(state), dtype=float)


def state_labels(vessel):
    labels = []
    for key in vessel.solver_state_collection.states:
        dim = int(np.prod(np.shape(np.asarray(
            getattr(vessel.result,
                    vessel.solver_state_collection.format_key(key)))[0])))
        phase = "" if key.phaseref is None else \
            f"{key.phaseref.phase_type}{key.phaseref.index}."
        labels += [f"{phase}{key.name}[{i}]" for i in range(dim)] \
            if dim > 1 else [f"{phase}{key.name}"]
    return labels


def plant_outputs(vessel):
    """time, u, T, fraction crystallized (D, E, J), volume, outlet flow."""
    hist = CR02R.history(vessel)
    t = hist.time
    return dict(
        time=t,
        u=np.array([vessel.controller.u_func(ti) for ti in t]),
        temp=hist.temp,
        frac_D=hist.fraction["D"], frac_E=hist.fraction["E"],
        frac_J=hist.fraction["J"],
        volume=np.ravel(vessel.result.vessel_vol),
        outlet_flow=np.ravel(vessel.result.outlet_vol_flow),
    )


def run_step_test():
    cv = build_open_loop()
    points = int(RUNTIME / DT_SAVE) + 1
    wall = control.solve(cv, RUNTIME, points)
    vessel = cv["CR02R"]
    data = plant_outputs(vessel)
    i0 = int(np.searchsorted(data["time"], T_STEP1)) - 1
    np.savez_compressed(
        STEP_FILE, **data, x0=state_at(vessel, i0), t0=data["time"][i0],
        u0=U_BASE, du=DU, t_step1=T_STEP1, t_step2=T_STEP2,
        labels=np.array(state_labels(vessel)), wall=wall)
    print(f"step test: {wall / 60:.1f} min, saved {STEP_FILE}")
    return cv


def rhs(vessel, t, x):
    # unit_model returns a buffer it reuses on the next call: copy it.
    return np.array(vessel.unit_model(t, x), dtype=float, copy=True).ravel()


def linearize(x0, t0):
    """A, B, C, D of the full model at (x0, U_BASE), by central differences
    of PharmaPy's right-hand side. The input enters as a constant jacket
    temperature, so the operating point is time-invariant."""
    cv = build_open_loop(lambda t: U_BASE)
    vessel = cv["CR02R"]
    vessel.compile_structure()
    n = x0.size
    f0 = rhs(vessel, t0, x0)

    start = clock.time()
    A = np.empty((n, n))
    scale = np.maximum(np.abs(x0), 1e-6 * max(np.abs(x0).max(), 1.0))
    for j in range(n):
        h = 1e-6 * scale[j]
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        A[:, j] = (rhs(vessel, t0, xp) - rhs(vessel, t0, xm)) / (2 * h)
        if j == 9:
            per = (clock.time() - start) / 10
            print(f"  {per * 1e3:.0f} ms per column, ~{per * n / 60:.1f} min "
                  f"for {n} states")

    controller = vessel.controller
    h = 1e-3
    controller.du = +h
    fp = rhs(vessel, t0, x0)
    controller.du = -h
    fm = rhs(vessel, t0, x0)
    controller.du = 0.0
    B = ((fp - fm) / (2 * h))[:, None]

    labels = state_labels_from_collection(vessel)
    i_temp = labels.index("global_temp")
    C = np.zeros((1, n))
    C[0, i_temp] = 1.0
    D = np.zeros((1, 1))

    np.savez_compressed(LIN_FILE, A=A, B=B, C=C, D=D, x0=x0, t0=t0, u0=U_BASE,
                        f0=f0, labels=np.array(labels), i_temp=i_temp,
                        wall=clock.time() - start)
    print(f"linearization: {n} states, {(clock.time() - start) / 60:.1f} min, "
          f"|f(x0)|/|x0| = {np.linalg.norm(f0) / np.linalg.norm(x0):.2e}, "
          f"saved {LIN_FILE}")


def state_labels_from_collection(vessel):
    """State names without needing a result (dims from the collection)."""
    labels = []
    collection = vessel.solver_state_collection
    for key in collection.states:
        dim = int(collection.states[key].dim)
        phase = "" if key.phaseref is None else \
            f"{key.phaseref.phase_type}{key.phaseref.index}."
        labels += ([f"{phase}{key.name}[{i}]" for i in range(dim)]
                   if dim > 1 else [f"{phase}{key.name}"])
    return labels


def run_closed_loop():
    """PharmaPy's jacket PI loop (default tuning) on a -1 K set-point step."""
    def setpoint(t):
        return SETPOINT_BASE + (SETPOINT_STEP if t >= T_STEP1 else 0.0)

    ctrl = JacketPIDContinuousVesselController(
        setpoint, temperature=dict(jacket_limits=control.JACKET_LIMITS))
    vessel = control.install_controller(CR02R.build_reactor(), ctrl)
    cv = control.wrap("CR02R", vessel, None)
    wall = control.solve(cv, CLOSED_RUNTIME, int(CLOSED_RUNTIME / DT_SAVE) + 1)
    control.STEP_TIME = T_STEP1
    h = control.controller_history(vessel)
    gain, ti, _, _ = ctrl.temperature_tuning()
    np.savez_compressed(
        CLOSED_FILE, time=h.time, temp=h.temp,
        setpoint=np.array([setpoint(t) for t in h.time]), jacket=h.jacket,
        kc=gain, ti=ti, tau_ht=ctrl.heat_transfer_time, t_step=T_STEP1,
        wall=wall)
    print(f"closed loop: {wall / 60:.1f} min, Kc = {gain}, Ti = {ti:.1f} s, "
          f"saved {CLOSED_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--linearize-only", action="store_true")
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="short run + linearization, to check timing")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.closed_loop:
        run_closed_loop()
    elif args.probe:
        cv = build_open_loop(lambda t: U_BASE)
        control.solve(cv, 600.0, 21)
        v = cv["CR02R"]
        print("states:", state_at(v, -1).size)
        LIN_FILE = OUT_DIR / "probe_linearization.npz"
        linearize(state_at(v, -1), 600.0)
    else:
        if not args.linearize_only:
            run_step_test()
        step = np.load(STEP_FILE)
        linearize(step["x0"], float(step["t0"]))
