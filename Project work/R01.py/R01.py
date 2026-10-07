"""Two-feed jacketed CSTR on PharmaPy's refactored ContinuousReactor.

The flowsheet is::

    FEED_A --+
             +--> R01
    FEED_B --+

``R01`` is ``PharmaPy.Reactors_Refactored.ContinuousReactor``. The refactored
vessel takes a list of inlet streams directly, so the separate mixer unit the
legacy ``Reactors.CSTR`` needed is gone. Feed compositions are supplied by
compound name; the database controls the order PharmaPy uses internally.

The reactor is jacketed rather than isothermal so that temperature is a solved
state: the heat of reaction raises it, the jacket and the feeds pull it back,
and the Arrhenius term feeds it back into the rate. That is what makes the
Ca-vs-T phase portrait meaningful.

Run it with::

    python R01.py                 # solve, save figures, show them
    python R01.py --no-show       # solve and save figures only

Figures are written to ``results/`` next to this file:

    R01_profiles.png   Ca vs time, T vs time, Ca vs T
    R01_native.png     the same run through PharmaPy.Plotting.plot_function
    R01_settings.png   the settings and initial conditions used for the run

The compound database defaults to ``../compound_database.json``; set
``PHARMAPY_DATABASE`` to use another one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.ThermoModule import ParseDatabase
from PharmaPy.Utilities import CoolingWater


HERE = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = HERE.parent / "compound_database.json"
RESULTS_DIR = HERE / "results"

# ---------------------------------------------------------------- settings
REACTIONS = ["A + B --> C"]
RATE_CONSTANTS = np.array([1.0e-2])  # L/mol/s, at KINETICS_TEMP_REF
ACTIVATION_ENERGIES = np.array([4.0e4])  # J/mol
KINETICS_TEMP_REF = 313.15  # K, rate constants above are quoted here
HEAT_OF_REACTION = -5.0e4  # J/mol, negative = exothermic

FLOW_A = 1.0e-5  # m**3/s
FLOW_B = 1.0e-5  # m**3/s
FEED_TEMPERATURE = 313.15  # K
FEED_A = {"A": 0.33}  # mol/L
FEED_B = {"B": 0.33}  # mol/L

REACTOR_VOLUME = 0.01  # m**3
REACTOR_DIAMETER = 0.2  # m, jacket area = 4 V / D
REACTOR_H_CONV = 1000.0  # W/m**2/K, vessel side
INITIAL_TEMPERATURE = 313.15  # K
INITIAL_REACTOR = {"A": 0.0, "B": 0.0, "C": 0.0}  # mol/L, rest is solvent

COOLANT_MASS_FLOW = 0.1  # kg/s
COOLANT_TEMPERATURE = 313.15  # K
COOLANT_H_CONV = 1000.0  # W/m**2/K, jacket side

RUNTIME = 1800.0  # s

# Validated categorical slots 1-3 (blue, orange, aqua).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
TEXT_MUTED = "#52514e"


# ---------------------------------------------------------------- builders
def database_species(database_path: Path) -> tuple[str, ...]:
    """Return the database species names in PharmaPy's declared order."""
    parsed_database = ParseDatabase(str(database_path), to_arrays=False)
    return tuple(parsed_database["name_species"])


def named_mole_concentration(
    composition: Mapping[str, float],
    species_names: tuple[str, ...],
) -> np.ndarray:
    """Convert a named composition to PharmaPy's ordered concentration array."""
    unknown_names = set(composition) - set(species_names)
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise KeyError(f"Unknown compounds in composition: {names}")
    if any(value < 0.0 for value in composition.values()):
        raise ValueError("Mole concentrations cannot be negative.")

    return np.array(
        [composition.get(name, 0.0) for name in species_names],
        dtype=float,
    )


def make_feed(
    database_path: Path,
    composition: Mapping[str, float],
    flow_rate: float,
    species_names: tuple[str, ...],
) -> LiquidStream:
    """Create one named-composition liquid feed."""
    return LiquidStream(
        str(database_path),
        temp=FEED_TEMPERATURE,
        mole_conc=named_mole_concentration(composition, species_names),
        vol_flow=flow_rate,
        name_solv="solvent",
    )


def build_reactor(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
) -> ContinuousReactor:
    """Build R01 with both feeds attached as separate inlets."""
    database_path = Path(database_path)
    species_names = database_species(database_path)

    reactor = ContinuousReactor(
        integrator=ScipyBackend(),
        h_conv=REACTOR_H_CONV,
        diam=REACTOR_DIAMETER,
    )
    reactor.Phases = LiquidPhase(
        str(database_path),
        temp=INITIAL_TEMPERATURE,
        mole_conc=named_mole_concentration(INITIAL_REACTOR, species_names),
        vol=REACTOR_VOLUME,
        name_solv="solvent",
    )
    reactor.RxnKinetics = RxnKinetics(
        path=str(database_path),
        rxn_list=REACTIONS,
        k_params=RATE_CONSTANTS,
        ea_params=ACTIVATION_ENERGIES,
        temp_ref=KINETICS_TEMP_REF,
        delta_hrxn=HEAT_OF_REACTION,
    )
    reactor.Utility = CoolingWater(
        mass_flow=COOLANT_MASS_FLOW,
        temp_in=COOLANT_TEMPERATURE,
        h_conv=COOLANT_H_CONV,
    )
    reactor.Inlet = [
        make_feed(database_path, FEED_A, FLOW_A, species_names),
        make_feed(database_path, FEED_B, FLOW_B, species_names),
    ]
    return reactor


def run_reactor(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    runtime: float = RUNTIME,
) -> ContinuousReactor:
    """Solve R01 and return it with ``result`` populated."""
    reactor = build_reactor(database_path)
    reactor.solve_unit(runtime=runtime, verbose=False)
    return reactor


# ---------------------------------------------------------------- plotting
def species_conc(reactor: ContinuousReactor, name: str) -> np.ndarray:
    """Liquid-phase concentration history of one species [mol/L]."""
    index = list(reactor.name_species).index(name)
    return np.asarray(reactor.result.mole_conc_liquid0)[:, index]


def plot_profiles(reactor: ContinuousReactor):
    """Ca vs time, T vs time and the Ca-T phase portrait, side by side."""
    import matplotlib.pyplot as plt

    result = reactor.result
    time_min = np.asarray(result.time) / 60.0
    conc_a = species_conc(reactor, "A")
    temp = np.ravel(result.global_temp)

    fig, (ax_ca, ax_t, ax_phase) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax_ca.plot(time_min, conc_a, color=SERIES_COLORS[0], lw=2)
    ax_ca.set_xlabel("Time [min]")
    ax_ca.set_ylabel("$C_A$ [mol/L]")
    ax_ca.set_title("Concentration of A")

    ax_t.plot(time_min, temp, color=SERIES_COLORS[1], lw=2)
    ax_t.axhline(COOLANT_TEMPERATURE, color=TEXT_MUTED, lw=1, ls="--")
    ax_t.annotate("coolant inlet", (time_min[-1], COOLANT_TEMPERATURE),
                  xytext=(0, 4), textcoords="offset points", ha="right",
                  va="bottom", color=TEXT_MUTED, fontsize=9)
    ax_t.set_xlabel("Time [min]")
    ax_t.set_ylabel("$T$ [K]")
    ax_t.set_title("Reactor temperature")

    ax_phase.plot(conc_a, temp, color=SERIES_COLORS[2], lw=2)
    ax_phase.plot(conc_a[0], temp[0], "o", ms=8, color=SERIES_COLORS[2],
                  mec="white", mew=2)
    ax_phase.plot(conc_a[-1], temp[-1], "s", ms=8, color=SERIES_COLORS[2],
                  mec="white", mew=2)
    ax_phase.annotate("start", (conc_a[0], temp[0]), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=9)
    ax_phase.annotate(f"t = {time_min[-1]:.0f} min", (conc_a[-1], temp[-1]),
                      xytext=(-8, 0), textcoords="offset points",
                      ha="right", va="center", fontsize=9)
    ax_phase.set_xlabel("$C_A$ [mol/L]")
    ax_phase.set_ylabel("$T$ [K]")
    ax_phase.set_title("$C_A$ vs $T$")

    for axis in (ax_ca, ax_t, ax_phase):
        axis.grid(True, color="#e5e4e0", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("R01 - refactored ContinuousReactor")
    fig.tight_layout()
    return fig


def plot_native(reactor: ContinuousReactor):
    """The same run through PharmaPy's own ``Plotting.plot_function``.

    ``plot_function`` reads ``uo.states_di`` and ``uo.fstates_di`` alongside
    ``uo.result``. The refactored vessel only exposes the former, but its
    result carries both descriptions as ``di_states``/``di_fstates``, so a thin
    view built from the result is all the plotter needs.
    """
    from PharmaPy.Plotting import plot_function

    result = reactor.result
    view = SimpleNamespace(
        result=result,
        states_di=result.di_states,
        fstates_di=result.di_fstates,
    )
    solutes = [name for name in reactor.name_species if name != "solvent"]
    fig, axes = plot_function(
        view,
        (["mole_conc_liquid0", solutes], "global_temp", "q_ht"),
        fig_map=(0, 1, 2),
        ncols=3,
        ylabels=("C_j", "T", "Q_ht"),
        figsize=(15, 4.5),
    )
    for axis in axes:
        axis.set_xlabel("time (s)")
    fig.suptitle("R01 - PharmaPy.Plotting.plot_function")
    fig.tight_layout()
    return fig


def settings_rows(reactor: ContinuousReactor) -> list[tuple[str, str, str]]:
    """(group, setting, value) rows describing this run."""
    def composition(named):
        solutes = {k: v for k, v in named.items() if v}
        if not solutes:
            return "solvent only"
        return ", ".join(f"{k} {v:g}" for k, v in solutes.items())

    tau = REACTOR_VOLUME / (FLOW_A + FLOW_B)
    area = 4 * REACTOR_VOLUME / REACTOR_DIAMETER
    u_ht = 1.0 / (1.0 / REACTOR_H_CONV + 1.0 / COOLANT_H_CONV)
    final = reactor.result
    final_conc = np.asarray(final.mole_conc_liquid0)[-1]
    # Clip solver roundoff (e.g. -1e-25) so it prints as 0, not -0.0000.
    final_solutes = {name: max(c, 0.0)
                     for name, c in zip(reactor.name_species, final_conc)
                     if name != "solvent"}

    return [
        ("Initial conditions", "Temperature", f"{INITIAL_TEMPERATURE:g} K"),
        ("Initial conditions", "Volume", f"{REACTOR_VOLUME:g} m³"),
        ("Initial conditions", "Charge [mol/L]",
         composition(INITIAL_REACTOR)),
        ("Feeds", "Feed A", f"{FLOW_A:g} m³/s, {composition(FEED_A)} mol/L"),
        ("Feeds", "Feed B", f"{FLOW_B:g} m³/s, {composition(FEED_B)} mol/L"),
        ("Feeds", "Feed temperature", f"{FEED_TEMPERATURE:g} K"),
        ("Feeds", "Residence time", f"{tau:g} s ({tau / 60:.1f} min)"),
        ("Kinetics", "Reactions", "; ".join(REACTIONS)),
        ("Kinetics", "k at T_ref",
         ", ".join(f"{k:g}" for k in RATE_CONSTANTS) + " L/mol/s"),
        ("Kinetics", "Ea",
         ", ".join(f"{e:g}" for e in ACTIVATION_ENERGIES) + " J/mol"),
        ("Kinetics", "T_ref", f"{KINETICS_TEMP_REF:g} K"),
        ("Kinetics", "ΔH_rxn", f"{HEAT_OF_REACTION:g} J/mol"),
        ("Jacket", "Coolant", f"{COOLANT_MASS_FLOW:g} kg/s at "
                              f"{COOLANT_TEMPERATURE:g} K"),
        ("Jacket", "U · A", f"{u_ht:g} W/m²/K × {area:g} m² "
                            f"= {u_ht * area:g} W/K"),
        ("Run", "Runtime", f"{RUNTIME:g} s"),
        ("Run", "Integrator", type(reactor.integrator).__name__),
        ("Run", "Final T", f"{np.ravel(final.global_temp)[-1]:.3f} K"),
        ("Run", "Final [mol/L]",
         ", ".join(f"{k} {v:.4f}" for k, v in final_solutes.items())),
    ]


def plot_settings(reactor: ContinuousReactor):
    """Render the run's settings and initial conditions as a table figure."""
    import matplotlib.pyplot as plt

    rows = settings_rows(reactor)
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(rows) + 0.6))
    ax.axis("off")

    cell_text = []
    previous_group = None
    for group, setting, value in rows:
        cell_text.append([group if group != previous_group else "",
                          setting, value])
        previous_group = group

    table = ax.table(
        cellText=cell_text,
        colLabels=["Group", "Setting", "Value"],
        colWidths=[0.22, 0.22, 0.56],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e4e0")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0efec")

    ax.set_title("R01 settings and initial conditions", pad=12)
    fig.tight_layout()
    return fig


def save_figures(reactor: ContinuousReactor, out_dir: Path = RESULTS_DIR):
    """Draw all three figures, save them as PNGs and return them by name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "R01_profiles": plot_profiles(reactor),
        "R01_native": plot_native(reactor),
        "R01_settings": plot_settings(reactor),
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

    database = Path(os.environ.get("PHARMAPY_DATABASE", DEFAULT_DATABASE_PATH))
    r01 = run_reactor(database)

    print("Final CSTR concentrations (mol/L):")
    for name, concentration in zip(r01.name_species,
                                   r01.result.mole_conc_liquid0[-1]):
        print(f"  {name}: {concentration:.6f}")
    print(f"Final temperature: {np.ravel(r01.result.global_temp)[-1]:.3f} K")

    save_figures(r01)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
