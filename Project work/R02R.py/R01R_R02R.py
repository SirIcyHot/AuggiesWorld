"""Two jacketed CSTRs in series, each held at its own temperature by
feedback control, on PharmaPy's refactored ContinuousReactor.

The flowsheet is::

    FEED_A --+
             +--> R01R (30 C) --> R02R (60 C)
    FEED_B --+

``R01R`` is the upstream reactor. It takes R02R's two feeds (``FEED_A`` and
``FEED_B`` from ``R02R.py``, unchanged) and no others. ``R02R`` gets no fresh
feed at all: its only inlet is R01R's outlet, handed over by
``PharmaPy.SimExec.SimulationExec`` as R01R's solved outlet trajectory. Both
reactors are the R02R vessel (same charge, jacket, coolant flow and three
parallel reactions); only the temperature they are held at differs.

Feedback control
----------------
Each reactor carries two loops, both in one controller object
(``JacketTemperatureController``):

* Level: PharmaPy's ``DefaultContinuousVesselVolume`` trims the outlet flow
  around the inlet flow to hold the vessel volume. Because R02R's inlet is
  R01R's outlet, R01R's level loop sets the flow R02R receives; R02R's level
  loop then follows whatever arrives.
* Temperature: a proportional controller measures the reactor temperature
  and sets the jacket coolant inlet temperature::

      T_jacket_in = T_sp + K_c (T_sp - T),   clipped to [T_min, T_max]

  The vessel's energy balance stays a solved state, so the reaction heat,
  the feed enthalpy and the jacket duty all still show up as real terms.
  Being proportional-only, the loop settles with a small offset,
  ``Q_load / (U A (1 + K_c))``; a larger ``CONTROLLER_GAIN`` shrinks it.

Run it with::

    python R01R_R02R.py               # solve, save figures, show them
    python R01R_R02R.py --no-show     # solve and save figures only

Figures are written to ``results/`` next to this file:

    R01R_R02R_profiles.png      per reactor: w_i vs time, T and jacket T vs
                                time, heat sources vs time
    R01R_R02R_mass_balance.png  per reactor: in / generated / out /
                                accumulated at the final time
    R01R_R02R_settings.png      settings and final values for the run

Settings shared with R02R (kinetics, feeds, vessel, coolant flow) are read
from ``R02R.py`` so the two scripts cannot drift apart.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ProcessControl_Refactored import DefaultContinuousVesselVolume
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater

import R02R as base


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FLOWSHEET = "R01R --> R02R"

# ---------------------------------------------------------------- settings
# Temperature each reactor starts at and is held at.
SETPOINTS = {
    "R01R": 303.15,  # K, 30 C
    "R02R": 333.15,  # K, 60 C
}
CONTROLLER_GAIN = 100.0  # K of jacket inlet per K of reactor error
# The coolant is liquid water, so its inlet temperature is kept between
# 5 C and 95 C whatever the controller asks for.
JACKET_TEMP_MIN = 278.15  # K
JACKET_TEMP_MAX = 368.15  # K

# Two residence times in series take longer to settle than one; this is
# ~10.8 residence times of each vessel.
RUNTIME = 5400.0  # s


class JacketTemperatureController(DefaultContinuousVesselVolume):
    """Level control on the outlet plus P control of the jacket inlet.

    The level loop is PharmaPy's own. The temperature loop writes the
    coolant inlet temperature onto ``unit.Utility`` on every evaluation,
    which the vessel reads when it adds the jacket duty to its energy
    balance. The law has no internal state, so it gives the same answer on
    a rejected step, a retried step and the balance replay.
    """

    def __init__(self, setpoint, gain=CONTROLLER_GAIN,
                 temp_min=JACKET_TEMP_MIN, temp_max=JACKET_TEMP_MAX):
        super().__init__()
        self.setpoint = setpoint
        self.gain = gain
        self.temp_min = temp_min
        self.temp_max = temp_max

    def jacket_inlet_temperature(self, temp):
        """Coolant inlet temperature [K] asked for at reactor temperature
        ``temp`` [K]; works on scalars and arrays."""
        demand = self.setpoint + self.gain * (self.setpoint - temp)
        return np.clip(demand, self.temp_min, self.temp_max)

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):
        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)
        temp = float(np.ravel(unit.get_heat_transfer_temperature())[0])
        unit.Utility.temp_in = float(self.jacket_inlet_temperature(temp))


# ---------------------------------------------------------------- builders
def build_reactor(name: str, database_path: Path,
                  inlet=None) -> base.BalanceRecordingReactor:
    """One R02R-type vessel held at ``SETPOINTS[name]``.

    ``inlet`` is left unset for a unit fed by the flowsheet.
    """
    species_names = base.database_species(database_path)
    solvent_name = base.database_solvent(species_names)
    setpoint = SETPOINTS[name]

    reactor = base.BalanceRecordingReactor(
        integrator=ScipyBackend(),
        h_conv=base.REACTOR_H_CONV,
        diam=base.REACTOR_DIAMETER,
        controller=JacketTemperatureController(setpoint),
    )
    reactor.Phases = LiquidPhase(
        str(database_path),
        temp=setpoint,
        mass_frac=base.named_mass_fraction(base.INITIAL_REACTOR,
                                           species_names, solvent_name),
        mass=base.REACTOR_MASS,
        name_solv=solvent_name,
    )
    reactor.RxnKinetics = RxnKinetics(
        path=str(database_path),
        rxn_list=base.REACTIONS,
        k_params=base.RATE_CONSTANTS,
        ea_params=base.ACTIVATION_ENERGIES,
        params_f=base.reaction_order_matrix(species_names),
        temp_ref=base.KINETICS_TEMP_REF,
        delta_hrxn=base.HEAT_OF_REACTION,
    )
    reactor.Utility = CoolingWater(
        mass_flow=base.COOLANT_MASS_FLOW,
        temp_in=setpoint,
        h_conv=base.COOLANT_H_CONV,
    )
    if inlet is not None:
        reactor.Inlet = inlet
    return reactor


def build_flowsheet(
    database_path: Path | str = base.DEFAULT_DATABASE_PATH,
) -> SimulationExec:
    """R01R takes R02R's two feeds; R02R takes only R01R's outlet."""
    database_path = Path(database_path)
    species_names = base.database_species(database_path)
    solvent_name = base.database_solvent(species_names)

    flowsheet = SimulationExec(str(database_path), flowsheet=FLOWSHEET)
    flowsheet.R01R = build_reactor("R01R", database_path, inlet=[
        base.make_feed(database_path, base.FEED_A, base.FLOW_A,
                       species_names, solvent_name),
        base.make_feed(database_path, base.FEED_B, base.FLOW_B,
                       species_names, solvent_name),
    ])
    flowsheet.R02R = build_reactor("R02R", database_path)
    return flowsheet


def run_flowsheet(
    database_path: Path | str = base.DEFAULT_DATABASE_PATH,
    runtime: float = RUNTIME,
) -> SimulationExec:
    """Solve R01R then R02R and return the flowsheet."""
    flowsheet = build_flowsheet(database_path)
    with warnings.catch_warnings():
        # SimExec's object check does not recognise a refactored vessel's
        # Inlet (a list here, and on R02R supplied by the flowsheet itself),
        # so it warns that Inlet is missing on both units. It is not.
        warnings.filterwarnings("ignore", message=".*not detected.*")
        flowsheet.SolveFlowsheet(
            kwargs_run={name: {"runtime": runtime, "verbose": False}
                        for name in SETPOINTS},
            verbose=False,
        )
    return flowsheet


def reactors(flowsheet: SimulationExec) -> dict:
    return {name: getattr(flowsheet, name) for name in SETPOINTS}


def temperatures(reactor) -> SimpleNamespace:
    """Reactor and jacket-inlet temperature histories [K]."""
    temp = np.ravel(reactor.result.global_temp)
    return SimpleNamespace(
        reactor=temp,
        jacket_in=reactor.controller.jacket_inlet_temperature(temp),
        setpoint=reactor.controller.setpoint,
    )


# ---------------------------------------------------------------- plotting
def plot_profiles(units: dict, balances: dict):
    """One row per reactor: w_i vs time, T vs time, heat terms vs time."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(units), 3, figsize=(15, 4.2 * len(units)),
                             squeeze=False)
    for row, (name, reactor) in zip(axes, units.items()):
        ax_w, ax_t, ax_q = row
        time_min = np.asarray(reactor.result.time) / 60.0
        fractions = base.mass_fractions(reactor)
        index = {s: j for j, s in enumerate(reactor.name_species)}
        species = base.reacting_species(reactor)

        for color, s in zip(base.SERIES_COLORS * len(species), species):
            ax_w.plot(time_min, fractions[:, index[s]], color=color, lw=2,
                      label=s)
        ax_w.set_ylabel("$w_i$ [kg/kg]")
        ax_w.set_title(f"{name}: mass fraction of each species")
        ax_w.legend(title="i", frameon=False, ncol=2)

        temps = temperatures(reactor)
        ax_t.plot(time_min, temps.reactor - 273.15, color=base.TEXT_PRIMARY,
                  lw=2, label="Reactor")
        ax_t.plot(time_min, temps.jacket_in - 273.15,
                  color=base.SERIES_COLORS[0], lw=1.5,
                  label="Jacket inlet (manipulated)")
        ax_t.axhline(temps.setpoint - 273.15, color=base.TEXT_MUTED, lw=1,
                     ls="--", label="Setpoint")
        ax_t.set_ylabel("$T$ [°C]")
        ax_t.set_title(f"{name}: temperature control "
                       f"(final {temps.reactor[-1] - 273.15:.2f} °C)")
        ax_t.legend(frameon=False)

        heat = balances[name].heat
        active = [k for k in base.HEAT_TERM_LABELS
                  if np.any(np.abs(heat.get(k, 0.0)) > 1e-9)]
        bal_min = balances[name].time / 60.0
        ax_q.axhline(0.0, color=base.TEXT_MUTED, lw=0.8)
        for color, k in zip(base.SERIES_COLORS, active):
            ax_q.plot(bal_min, heat[k], color=color, lw=2,
                      label=base.HEAT_TERM_LABELS[k])
        ax_q.plot(bal_min, sum(heat.values()), color=base.TEXT_PRIMARY,
                  lw=1.5, ls="--", label="Net (sum)")
        ax_q.set_ylabel("Heat rate into vessel [W]")
        ax_q.set_title(f"{name}: heat sources")
        ax_q.legend(frameon=False)

        for axis in row:
            axis.set_xlabel("Time [min]")
            axis.grid(True, color="#e5e4e0", lw=0.8)
            axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{FLOWSHEET}: feedback-controlled CSTRs in series")
    fig.tight_layout()
    return fig


INLET_LABELS = {"R01R": ["Feed A", "Feed B"], "R02R": ["From R01R"]}


def plot_mass_balance(units: dict, balances: dict):
    """Final-time species balance of each reactor, one table per reactor."""
    import matplotlib.pyplot as plt

    to_g = 1.0e3  # kg/s -> g/s

    def cell(value):
        return "0" if abs(value) < 5e-7 else f"{value:.4f}"

    n_species = len(next(iter(units.values())).name_species)
    fig, axes = plt.subplots(len(units), 1,
                             figsize=(10, len(units) * (0.3 * n_species + 1.6)))
    for ax, (name, reactor) in zip(np.atleast_1d(axes), units.items()):
        bal = balances[name]
        feeds = bal.feeds[-1] * to_g
        generation = bal.generation[-1] * to_g
        outlet = bal.outlet[-1] * to_g
        accumulation = bal.accumulation[-1] * to_g

        columns = (["Species"]
                   + [f"{label} [g/s]" for label in INLET_LABELS[name]]
                   + ["Generated [g/s]", "Out [g/s]", "Accumulated [g/s]"])
        rows = [[s] + [cell(f[j]) for f in feeds]
                + [cell(generation[j]), cell(outlet[j]),
                   cell(accumulation[j])]
                for j, s in enumerate(reactor.name_species)]
        rows.append(["Total"] + [cell(f.sum()) for f in feeds]
                    + [cell(generation.sum()), cell(outlet.sum()),
                       cell(accumulation.sum())])

        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=columns, cellLoc="right",
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        for (r, c), cell_obj in table.get_celld().items():
            cell_obj.set_edgecolor("#e5e4e0")
            if c == 0:
                cell_obj.set_text_props(ha="left")
            if r == 0 or r == len(rows):
                cell_obj.set_text_props(weight="bold")
            if r == 0:
                cell_obj.set_facecolor("#f0efec")

        gap = base.steady_state_gap(bal)
        status = ("steady state" if gap < base.STEADY_STATE_TOL
                  else "NOT yet at steady state")
        ax.set_title(f"{name} at t = {bal.time[-1]:g} s: in "
                     f"{feeds.sum():.4f} g/s, out {outlet.sum():.4f} g/s; "
                     f"max accumulation {gap * 100:.2g}% of inlet ({status})",
                     fontsize=10)
    fig.tight_layout()
    return fig


def settings_rows(units: dict) -> list[tuple[str, str, str]]:
    def composition(named):
        return ", ".join(f"{k} {v:g}" for k, v in named.items()) \
            + " (mass frac.)"

    tau = base.REACTOR_MASS / (base.FLOW_A + base.FLOW_B)
    rows = [
        ("Flowsheet", "Topology", FLOWSHEET),
        ("Feeds (R01R only)", "Feed A",
         f"{base.FLOW_A:g} kg/s, {composition(base.FEED_A)}"),
        ("Feeds (R01R only)", "Feed B",
         f"{base.FLOW_B:g} kg/s, {composition(base.FEED_B)}"),
        ("Feeds (R01R only)", "Feed temperature",
         f"{base.FEED_TEMPERATURE:g} K"),
        ("Vessels (each)", "Initial charge",
         f"{base.REACTOR_MASS:g} kg solvent, at its setpoint"),
        ("Vessels (each)", "Residence time", f"{tau:g} s"),
        ("Control", "Level", "DefaultContinuousVesselVolume (outlet flow)"),
        ("Control", "Temperature",
         f"P on jacket inlet T, Kc {CONTROLLER_GAIN:g} K/K, limits "
         f"{JACKET_TEMP_MIN:g}-{JACKET_TEMP_MAX:g} K"),
        ("Control", "Coolant flow", f"{base.COOLANT_MASS_FLOW:g} kg/s"),
        ("Run", "Runtime", f"{RUNTIME:g} s"),
    ]
    for name, reactor in units.items():
        temps = temperatures(reactor)
        solvent = base.database_solvent(tuple(reactor.name_species))
        final = {s: w for s, w in zip(reactor.name_species,
                                      base.mass_fractions(reactor)[-1])
                 if s != solvent and w > 1e-9}
        rows += [
            (name, "Setpoint", f"{temps.setpoint:g} K "
                               f"({temps.setpoint - 273.15:g} °C)"),
            (name, "Final T / jacket inlet T",
             f"{temps.reactor[-1]:.3f} K / {temps.jacket_in[-1]:.3f} K"),
            (name, "Final [mass frac.]",
             ", ".join(f"{k} {v:.5f}" for k, v in final.items())),
        ]
    return rows


def plot_settings(units: dict):
    import matplotlib.pyplot as plt

    rows = settings_rows(units)
    fig, ax = plt.subplots(figsize=(13, 0.3 * len(rows) + 0.6))
    ax.axis("off")
    cell_text, previous = [], None
    for group, setting, value in rows:
        cell_text.append([group if group != previous else "", setting, value])
        previous = group
    table = ax.table(cellText=cell_text, colLabels=["Group", "Setting",
                                                    "Value"],
                     colWidths=[0.15, 0.2, 0.65], cellLoc="left",
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e4e0")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0efec")
    ax.set_title(f"{FLOWSHEET} settings", pad=12)
    fig.tight_layout()
    return fig


def save_figures(units: dict, balances: dict, out_dir: Path = RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "R01R_R02R_profiles": plot_profiles(units, balances),
        "R01R_R02R_mass_balance": plot_mass_balance(units, balances),
        "R01R_R02R_settings": plot_settings(units),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    database = Path(os.environ.get("PHARMAPY_DATABASE",
                                   base.DEFAULT_DATABASE_PATH))
    units = reactors(run_flowsheet(database))
    balances = {name: base.replay_balances(r) for name, r in units.items()}

    for name, reactor in units.items():
        temps = temperatures(reactor)
        bal = balances[name]
        print(f"\n{name}  (setpoint {temps.setpoint:g} K)")
        print(f"  Final T {temps.reactor[-1]:.3f} K, jacket inlet "
              f"{temps.jacket_in[-1]:.3f} K, offset "
              f"{temps.reactor[-1] - temps.setpoint:+.3f} K")
        print(f"  {'species':<8}{'w_out':>10}{'in g/s':>10}{'gen g/s':>10}"
              f"{'out g/s':>10}")
        for j, s in enumerate(reactor.name_species):
            print(f"  {s:<8}{base.mass_fractions(reactor)[-1][j]:>10.5f}"
                  f"{bal.inlet[-1][j] * 1e3:>10.4f}"
                  f"{bal.generation[-1][j] * 1e3:>10.4f}"
                  f"{bal.outlet[-1][j] * 1e3:>10.4f}")
        gap = base.steady_state_gap(bal)
        if gap >= base.STEADY_STATE_TOL:
            print(f"  WARNING: accumulation is {gap:.2%} of the inlet flow; "
                  "increase RUNTIME to reach steady state.")
        print("  Heat terms at the final time (W): " + ", ".join(
            f"{base.HEAT_TERM_LABELS.get(k, k)} {v[-1]:.1f}"
            for k, v in bal.heat.items() if abs(v[-1]) > 1e-9))

    # The hand-over: what R02R takes in is what R01R put out.
    handover = np.abs(balances["R02R"].inlet[-1]
                      - balances["R01R"].outlet[-1]).max()
    print(f"\nR01R outlet vs R02R inlet at the final time: max species "
          f"difference {handover * 1e3:.2e} g/s")

    save_figures(units, balances)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
