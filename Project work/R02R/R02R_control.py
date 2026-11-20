"""R02R as a one-member ConcurrentVessel, for control studies.

The vessel is ``R02R.build_reactor()`` unchanged (same feed, kinetics,
jacket); this file only chooses its controller, wraps it in a
``PharmaPy.ConcurrentVessel.ConcurrentVessel`` and integrates it with CVode.
A ConcurrentVessel with one member solves exactly what the bare vessel does;
the point of the wrapper is that more members (CR02R, D01R, ...) can be added
later, and a controller on one member can then read another member's state.

Where control lives in PharmaPy
-------------------------------
Every MultiPhaseVessel owns exactly one ``controller``
(``PharmaPy.ProcessControl_Refactored``). On every right-hand-side
evaluation the vessel calls it twice:

1. ``observe`` + ``compute_operating_conditions`` with the vessel state only
   (sets anything that does not depend on the feed: jacket temperature,
   inlet overrides);
2. the same again with the resolved inlets (sets the outlet flow, which is
   trimmed around the feed).

The controller returns *operating conditions*, ``{OperatingKey: value}``:

- ``OperatingKey("vol_flow", connection=0, port="outlet")``  outlet flow
- ``MultiPhaseVessel.UTILITY_TEMPERATURE``  (``temp_in``, port "utility")
  jacket inlet temperature, overriding ``Utility.temp_in``
- ``OperatingKey(..., phaseref=..., port="inlet")``  inlet overrides

Controller memory (a PI integral, a filtered measurement for D action) is
declared as solver states, so CVode integrates it with the vessel and the
replay reproduces it exactly.

The modes compared here (``--mode``, or ``--compare`` for all four)
------------------------------------------------------------------
``open``     ``DefaultContinuousVesselVolume`` -- what a ContinuousReactor
             gets when no controller is given. P-only level loop on the
             outlet, ``F_out = F_in + K (V - V_set)``, K = 1/tau with tau a
             hundredth of a residence time. Temperature is open loop: the
             jacket stays at ``Utility.temp_in``.
``pi``       ``PIDContinuousVesselVolume``: the same level loop with integral
             action, so no level offset. Temperature still open loop.
``jacket``   ``JacketPIDContinuousVesselController``: PI level loop plus a
             PI temperature loop that moves the jacket inlet temperature to
             track ``setpoint(t)``, clipped to ``JACKET_LIMITS``, with
             back-calculation anti-windup. The energy balance is kept.
``imposed``  ``PIDContinuousVesselController``: PI level loop, and the vessel
             temperature is *set* to ``setpoint(t)`` (the energy balance is
             removed). The "perfect temperature control" bound.

The set point is ``SETPOINT_START`` until ``STEP_TIME``, then
``SETPOINT_END``: a step test of the temperature loop.

One PharmaPy gotcha this file works around: the controller's solver states
are registered when the vessel's phases are set, so assigning
``vessel.controller = ...`` afterwards silently drops the integral (the PI
loop runs as P-only). ``install_controller`` re-lays the solver states.

Run it with::

    python R02R_control.py --compare --no-show
    python R02R_control.py --mode jacket --runtime 7200

Figures go to ``results/control/``.
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import R02R  # noqa: E402
from PharmaPy.ConcurrentVessel import ConcurrentVessel  # noqa: E402
from PharmaPy.DataClasses import StateCollection, StateKey  # noqa: E402
from PharmaPy.IntegratorBackends import AssimuloBackend  # noqa: E402
from PharmaPy.ProcessControl_Refactored import (  # noqa: E402
    DefaultContinuousVesselVolume,
    JacketPIDContinuousVesselController,
    PIDContinuousVesselController,
    PIDContinuousVesselVolume,
)

RESULTS_DIR = HERE / "results" / "control"

# ------------------------------------------------------------ settings
RUNTIME = 10 * 3600.0          # s
POINTS = 601
SETPOINT_START = 60 + 273.15   # K, R02R's initial and coolant temperature
SETPOINT_END = 55 + 273.15     # K
STEP_TIME = 5 * 3600.0         # s; --step-time, default half the runtime
JACKET_LIMITS = (5 + 273.15, 95 + 273.15)  # K, what the utility can deliver

# None = PharmaPy's defaults (see the module docstring of
# ProcessControl_Refactored): level Kc = 1/(0.01 tau_res), Ti = 4/Kc;
# temperature Kc = 4, Ti = m cp / (U A).
LEVEL_TUNING = dict(tau=None, integral_time=None)
TEMPERATURE_TUNING = dict(gain=None, integral_time=None)

SOLVER_OPTIONS = dict(rtol=1e-6, atol=1e-8)

MODES = ("open", "pi", "jacket", "imposed")
MODE_LABELS = {
    "open": "P level, open-loop T (PharmaPy default)",
    "pi": "PI level, open-loop T",
    "jacket": "PI level, PI jacket T",
    "imposed": "PI level, imposed T",
}
MODE_COLORS = {"open": "#5f6b7a", "pi": "#2a78d6", "jacket": "#eb6834",
               "imposed": "#1baf7a"}
# Line weights/dashes so modes that coincide stay visible on top of each other.
MODE_STYLES = {"open": dict(lw=4.0, alpha=0.45), "pi": dict(lw=2.0),
               "jacket": dict(lw=2.0), "imposed": dict(lw=2.0, ls=(0, (4, 2)))}
TEXT_PRIMARY = R02R.TEXT_PRIMARY
TEXT_MUTED = R02R.TEXT_MUTED
GRID_COLOR = "#e5e4e0"


def setpoint(t):
    """Temperature set point [K]: a step at STEP_TIME."""
    return np.where(np.asarray(t) < STEP_TIME, SETPOINT_START, SETPOINT_END) \
        if np.ndim(t) else (SETPOINT_START if t < STEP_TIME else SETPOINT_END)


# ------------------------------------------------------------ controllers
def make_controller(mode: str):
    """A fresh controller for one of MODES."""
    if mode == "open":
        return DefaultContinuousVesselVolume(tau=LEVEL_TUNING["tau"])
    if mode == "pi":
        return PIDContinuousVesselVolume(**LEVEL_TUNING)
    if mode == "jacket":
        return JacketPIDContinuousVesselController(
            setpoint, **LEVEL_TUNING,
            temperature=dict(TEMPERATURE_TUNING, jacket_limits=JACKET_LIMITS))
    if mode == "imposed":
        return PIDContinuousVesselController(setpoint, **LEVEL_TUNING)
    raise ValueError(f"mode must be one of {MODES}")


def install_controller(vessel, controller):
    """Give an already-built vessel a new controller, and re-lay its solver
    states so the controller's own states (integrals) are included and, for
    an imposed temperature, the energy balance is dropped."""
    vessel.controller = controller
    vessel.solver_state_collection = StateCollection()
    vessel.output_state_collection = StateCollection()
    vessel.nomenclature()
    return vessel


def wrap(name: str, vessel, solver_options=SOLVER_OPTIONS) -> ConcurrentVessel:
    """One-member ConcurrentVessel on CVode (Assimulo's defaults if
    ``solver_options`` is None)."""
    integrator = (AssimuloBackend() if solver_options is None else
                  AssimuloBackend(options=dict(solver_options)))
    return ConcurrentVessel({name: vessel}, integrator=integrator)


def build(mode: str) -> ConcurrentVessel:
    vessel = install_controller(R02R.build_reactor(), make_controller(mode))
    return wrap("R02R", vessel)


def solve(cv: ConcurrentVessel, runtime=RUNTIME, points=POINTS):
    start = clock.time()
    cv.solve_unit(time_grid=np.linspace(0.0, runtime, points), verbose=False)
    return clock.time() - start


# ------------------------------------------------------------ results
def controller_history(vessel) -> SimpleNamespace:
    """What the loops saw and did at every saved point.

    ``temp`` [K], ``setpoint`` [K] (None if open loop), ``jacket`` [K]
    (applied jacket inlet temperature; None when the temperature is imposed),
    ``volume`` and ``target`` [m3], ``inlet_flow`` and ``outlet_flow``
    [m3/s], and ``integrals`` {state name: array}.
    """
    result = vessel.result
    control = vessel.controller
    t = np.asarray(result.time)

    integrals = {}
    for state in control.solver_states:
        values = getattr(result, state.name, None)
        if values is not None:
            integrals[state.name] = np.asarray(values, dtype=float).ravel()

    has_setpoint = getattr(control, "temp_func", None) is not None
    temp = (np.asarray(result.global_temp, dtype=float) if
            hasattr(result, "global_temp") else setpoint(t))

    if hasattr(control, "jacket_temperature"):
        key = control.temperature_loop.integral_key
        jacket = np.array([
            float(control.jacket_temperature(
                ti, {StateKey("global_temp"): Ti,
                     key: integrals[key.name][i]}, vessel)[1])
            for i, (ti, Ti) in enumerate(zip(t, temp))])
    elif has_setpoint:
        jacket = None  # imposed: no energy balance, so no jacket
    else:
        jacket = np.full_like(t, vessel.Utility.temp_in)

    return SimpleNamespace(
        time=t, temp=temp,
        setpoint=setpoint(t) if has_setpoint else None,
        jacket=jacket,
        volume=np.asarray(result.vessel_vol, dtype=float).ravel(),
        target=control.target_volume,
        inlet_flow=float(DefaultContinuousVesselVolume.total_inlet_flow(vessel)),
        outlet_flow=np.asarray(result.outlet_vol_flow, dtype=float).ravel(),
        integrals=integrals,
    )


def step_metrics(h: SimpleNamespace) -> dict:
    """Classical step-response numbers for the temperature step."""
    if h.setpoint is None:
        return {}
    after = h.time >= STEP_TIME
    if after.sum() < 2:
        return {}
    t, T = h.time[after] - STEP_TIME, h.temp[after]
    step = SETPOINT_END - SETPOINT_START
    final = T[-1]
    band = 0.02 * abs(step)
    outside = np.nonzero(np.abs(T - final) > band)[0]
    settle = t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(t) \
        else (0.0 if not len(outside) else np.nan)
    overshoot = max(0.0, (final - T.min()) / abs(step)) if step < 0 else \
        max(0.0, (T.max() - final) / abs(step))
    return dict(offset=final - SETPOINT_END, overshoot=overshoot,
                settling_time=settle,
                iae=float(np.trapezoid(np.abs(T - SETPOINT_END), t)))


def report(name, mode, h, wall):
    m = step_metrics(h)
    line = (f"  {mode:<8} {MODE_LABELS[mode]:<40} wall {wall:6.1f} s | "
            f"T end {h.temp[-1] - 273.15:6.2f} C | "
            f"V offset {(h.volume[-1] - h.target) * 1e6:+9.2f} mL")
    if m:
        line += (f" | T offset {m['offset']:+.3f} K, overshoot "
                 f"{m['overshoot']:.1%}, 2% settling "
                 f"{m['settling_time'] / 60:.1f} min")
    print(line)


# ------------------------------------------------------------ plotting
def _style(axis):
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(TEXT_MUTED)
    axis.tick_params(colors=TEXT_MUTED, labelsize=9)


def plot_control(histories: dict, unit_name: str):
    """Four panels: T vs set point, jacket, volume error, outlet flow."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    (ax_t, ax_j), (ax_v, ax_f) = axes
    for mode, h in histories.items():
        hours = h.time / 3600
        style = dict(color=MODE_COLORS[mode], **MODE_STYLES[mode])
        ax_t.plot(hours, h.temp - 273.15, label=MODE_LABELS[mode], **style)
        if h.jacket is not None:
            ax_j.plot(hours, h.jacket - 273.15, **style)
        ax_v.plot(hours, (h.volume - h.target) * 1e6, **style)
        ax_f.plot(hours, h.outlet_flow * 1e6 * 60, **style)

    any_h = next(iter(histories.values()))
    hours = any_h.time / 3600
    ax_t.plot(hours, setpoint(any_h.time) - 273.15, color=TEXT_PRIMARY,
              lw=1, ls="--", label="set point")
    ax_v.axhline(0.0, color=TEXT_PRIMARY, lw=1, ls="--")
    ax_f.axhline(any_h.inlet_flow * 1e6 * 60, color=TEXT_PRIMARY, lw=1,
                 ls="--", label="inlet flow")

    ax_t.set_ylabel("Vessel temperature [C]")
    ax_j.set_ylabel("Jacket inlet temperature [C]")
    ax_v.set_ylabel("V - V_target [mL]")
    ax_f.set_ylabel("Outlet flow [mL/min]")
    for ax in (ax_v, ax_f):
        ax.set_xlabel("Time [h]")
    for ax in axes.flat:
        _style(ax)
    ax_f.legend(frameon=False, fontsize=8, loc="best")
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{unit_name} in a ConcurrentVessel: native controllers, "
                 f"set point step at {STEP_TIME / 3600:.0f} h",
                 color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_integrals(histories: dict, unit_name: str):
    """The controllers' own solver states (their memory)."""
    import matplotlib.pyplot as plt

    names = sorted({n for h in histories.values() for n in h.integrals})
    if not names:
        return None
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 3.8),
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        for mode, h in histories.items():
            if name in h.integrals:
                ax.plot(h.time / 3600, h.integrals[name],
                        color=MODE_COLORS[mode], label=MODE_LABELS[mode],
                        **MODE_STYLES[mode])
        ax.set_title(name, color=TEXT_PRIMARY, fontsize=10)
        ax.set_xlabel("Time [h]")
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{unit_name}: controller integral states", color=TEXT_PRIMARY)
    fig.tight_layout()
    return fig


def save_figures(histories, unit_name, out_dir=RESULTS_DIR, tag=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix, fig in (("control", plot_control(histories, unit_name)),
                        ("integrals", plot_integrals(histories, unit_name))):
        if fig is not None:
            path = out_dir / f"{unit_name}_{suffix}{tag}.png"
            fig.savefig(path, dpi=150)
            saved.append(path)
    return saved


def parse_args(doc, default_mode="jacket"):
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default=default_mode)
    parser.add_argument("--compare", action="store_true",
                        help="run every mode and overlay them")
    parser.add_argument("--runtime", type=float, default=RUNTIME)
    parser.add_argument("--points", type=int, default=POINTS)
    parser.add_argument("--step-time", type=float, default=None,
                        help="set point step time [s]; default runtime / 2")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    return args


def main(unit_name, build_fn, doc, default_mode="jacket",
         out_dir=RESULTS_DIR):
    args = parse_args(doc, default_mode)
    global STEP_TIME
    STEP_TIME = args.runtime / 2 if args.step_time is None else args.step_time
    modes = MODES if args.compare else (args.mode,)
    histories = {}
    print(f"{unit_name} as a one-member ConcurrentVessel, "
          f"{args.runtime / 3600:.1f} h:")
    for mode in modes:
        cv = build_fn(mode)
        wall = solve(cv, args.runtime, args.points)
        vessel = cv[unit_name]
        histories[mode] = controller_history(vessel)
        report(unit_name, mode, histories[mode], wall)
    tag = "" if args.compare else f"_{args.mode}"
    for path in save_figures(histories, unit_name, out_dir, tag):
        print(f"  saved {path}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return histories


if __name__ == "__main__":
    main("R02R", build, __doc__, default_mode="jacket")
