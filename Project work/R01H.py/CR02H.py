"""Reactive continuous crystallizer driven to zero F solubility with extra A.

The flowsheet is::

    FEED_A ---+
    FEED_C ---+--> CR02
    A_PUSH ---+

``CR02`` is ``CR01`` with two changes: the vessel (initial charge and jacket
coolant) runs at ``CR02_TEMPERATURE``, and a third feed of A (``A_PUSH``)
raises the liquid [A] until the F solubility correlation reaches zero. The
feeds, the reaction ``2 A + C -> F``, the vessel and the crystallization
kinetics (nucleation, growth, size grid and solubility correlation) are all
taken from ``CR01H``, which in turn takes the feeds and reaction from
``R01H``.

The CR01 solubility correlation::

    c_sat = (m1 [A] + b1) T_C + m2 [A] + b2

falls with [A] and crosses zero at ``zero_solubility_a_conc(T)``. Past that
point it goes negative, which is not physical, so CR02 clamps it at zero.
``A_PUSH`` is sized so the steady-state [A] sits ``PUSH_MARGIN`` above the
zero point: the clamp then holds the solubility at exactly zero, and the
steady state stays clear of the clamp's corner. With no margin the solver
crawls along that corner and the run stalls. ``python CR02H.py --size-feed``
re-sizes ``A_PUSH`` after ``CR02_TEMPERATURE`` or the feeds change.

Supersaturation is absolute (``c - c_sat``) rather than CR01's relative
``(c - c_sat) / c_sat``. The relative form divides by the solubility, so it
has no value at zero solubility; the absolute form keeps the same rate
constants and orders and stays finite there.

Run it with::

    python CR02H.py               # solve, save figures, show them
    python CR02H.py --no-show     # solve and save figures only
    python CR02H.py --size-feed   # find the A_PUSH flow, then solve

Figures are written to ``results/`` next to this file:

    CR02_profiles.png      w_i vs time, T vs time, w_i vs T (reacting species)
    CR02_solubility.png    liquid F, F solubility and [A] vs its zero point
    CR02_native.png        the same run through PharmaPy.Plotting.plot_function
    CR02_settings.png      the settings and initial conditions used for the run
    CR02_mass_balance.png  inlet vs outlet species balance at the final time
    CR02_heat_sources.png  energy-balance terms (heat sources) vs time

The compound database defaults to ``compound_databaseActual.json``; set
``PHARMAPY_DATABASE`` to use another one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef
from PharmaPy.Mechanisms import OneDFVMMechanism, ReactionMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
from PharmaPy.Utilities import CoolingWater
from CR01H import (
    ACTIVATION_ENERGIES,
    COOLANT_H_CONV,
    COOLANT_MASS_FLOW,
    CRYSTAL_KV,
    CRYSTAL_SIZE_GRID,
    DEFAULT_DATABASE_PATH,
    FEED_A,
    FEED_B,
    FEED_TEMPERATURE,
    FLOW_A,
    FLOW_B,
    GROWTH,
    HEAT_OF_REACTION,
    HEAT_TERM_LABELS,
    INITIAL_REACTOR,
    KINETICS_TEMP_REF,
    NUCL_PRIM,
    NUCL_SEC,
    NUM_RESIDENCE_TIMES,
    RATE_CONSTANTS,
    REACTOR_DIAMETER,
    REACTOR_H_CONV,
    REACTOR_VOLUME,
    REACTIONS,
    RESULTS_DIR,
    SERIES_COLORS,
    SOLUBILITY_B1,
    SOLUBILITY_B2,
    SOLUBILITY_M1,
    SOLUBILITY_M2,
    STEADY_STATE_TOL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    build_kinetics,
    database_solvent,
    database_species,
    make_feed,
    make_solubility_function,
    mass_fractions,
    named_mass_fraction,
    plot_heat_sources,
    plot_native,
    reacting_species,
    replay_balances,
    steady_state_gap,
)


# CR02-only settings. Everything else comes from CR01H.
CR02_TEMPERATURE = 5 + 273.15  # K, initial charge and jacket coolant inlet
FEED_A_PUSH = {"A": 1.0}  # mass fraction, pure A
# kg/s. Gives [A] ~1.92 mol/L at steady state against a zero point of
# ~1.74 mol/L at the ~282.9 K the vessel settles at; see --size-feed.
FLOW_A_PUSH = 1.0e-4
PUSH_MARGIN = 0.10  # steady-state [A] / zero-solubility [A] - 1
SUPERSATURATION_TYPE = "absolute"  # relative is undefined at c_sat = 0


def zero_solubility_a_conc(temp) -> np.ndarray:
    """Liquid [A] [mol/L] at which the CR01 F solubility reaches zero."""
    temp_c = np.asarray(temp) - 273.15
    return -((SOLUBILITY_B1 * temp_c + SOLUBILITY_B2)
             / (SOLUBILITY_M1 * temp_c + SOLUBILITY_M2))


def make_clamped_solubility_function(species_names: tuple[str, ...]):
    """CR01's F solubility correlation, held at zero once it goes negative."""
    correlation = make_solubility_function(species_names)

    def solubility(temp, conc):
        return np.maximum(correlation(temp, conc), 0.0)

    return solubility


def total_feed_flow(flow_a_push: float = FLOW_A_PUSH) -> float:
    return FLOW_A + FLOW_B + flow_a_push


def reactor_residence_time(reactor: ContinuousCrystallizer,
                           flow_a_push: float = FLOW_A_PUSH) -> float:
    """Initial residence time; call before solving, while the phases still
    hold the initial charge."""
    initial_mass = float(reactor.Phases.Liquids[0].mass)
    return initial_mass / total_feed_flow(flow_a_push)


def build_reactor(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    flow_a_push: float = FLOW_A_PUSH,
) -> ContinuousCrystallizer:
    """Build CR02: CR01's feeds and kinetics plus the A push feed."""
    database_path = Path(database_path)
    species_names = database_species(database_path)
    solvent_name = database_solvent(species_names)

    reactor = ContinuousCrystallizer(
        integrator=ScipyBackend(),
        h_conv=REACTOR_H_CONV,
        diam=REACTOR_DIAMETER,
    )
    liquid = LiquidPhase(
        str(database_path),
        temp=CR02_TEMPERATURE,
        mass_frac=named_mass_fraction(INITIAL_REACTOR, species_names,
                                      solvent_name),
        vol=REACTOR_VOLUME,
        name_solv=solvent_name,
    )
    solid_mass_frac = np.zeros(len(species_names))
    solid_mass_frac[species_names.index("F")] = 1.0
    solid = SolidPhase(
        str(database_path),
        temp=CR02_TEMPERATURE,
        mass=0.0,
        mass_frac=solid_mass_frac,
        name_solv=solvent_name,
    )
    solid.mechanisms = OneDFVMMechanism(
        solid,
        target_components="F",
        solvent_name=solvent_name,
        x_grid=CRYSTAL_SIZE_GRID,
        distrib_init=np.zeros(len(CRYSTAL_SIZE_GRID)),
        kv=CRYSTAL_KV,
    )
    reactor.Phases = [liquid, solid]
    reactor.CrystKinetics = CrystKinetics(
        solub_fn=make_clamped_solubility_function(species_names),
        nucl_prim=NUCL_PRIM,
        nucl_sec=NUCL_SEC,
        growth=GROWTH,
        sup_sat_type=SUPERSATURATION_TYPE,
        solubility_basis="mole_per_volume_solution",
    )
    reactor.intraphase_processes = [
        IntraPhaseProcess(
            phaseref=PhaseRef("liquid", 0),
            mechanism=ReactionMechanism(build_kinetics(database_path), liquid),
        )
    ]
    reactor.nomenclature(overwrite=True)
    reactor.Utility = CoolingWater(
        mass_flow=COOLANT_MASS_FLOW,
        temp_in=CR02_TEMPERATURE,
        h_conv=COOLANT_H_CONV,
    )
    feeds = [
        make_feed(database_path, FEED_A, FLOW_A, species_names, solvent_name),
        make_feed(database_path, FEED_B, FLOW_B, species_names, solvent_name),
    ]
    if flow_a_push > 0.0:
        feeds.append(make_feed(database_path, FEED_A_PUSH, flow_a_push,
                               species_names, solvent_name))
    reactor.Inlet = feeds
    return reactor


def run_reactor(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    runtime: float | None = None,
    flow_a_push: float = FLOW_A_PUSH,
) -> ContinuousCrystallizer:
    """Solve CR02 and return it with ``result`` populated."""
    reactor = build_reactor(database_path, flow_a_push)
    if runtime is None:
        runtime = (NUM_RESIDENCE_TIMES
                   * reactor_residence_time(reactor, flow_a_push))
    reactor.solve_unit(runtime=runtime, verbose=False)
    return reactor


def a_conc_history(reactor: ContinuousCrystallizer) -> np.ndarray:
    """Liquid [A] [mol/L] at each saved time."""
    index_a = list(reactor.name_species).index("A")
    return np.asarray(reactor.result.mole_conc_liquid0, dtype=float)[:, index_a]


def push_excess(reactor: ContinuousCrystallizer) -> float:
    """Final [A] relative to the zero-solubility [A] at the final T, minus 1.

    Zero or above means the solubility has been pushed to zero.
    """
    final_temp = np.ravel(reactor.result.global_temp)[-1]
    return float(a_conc_history(reactor)[-1]
                 / zero_solubility_a_conc(final_temp) - 1.0)


def size_push_feed(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    margin: float = PUSH_MARGIN,
    flow_guess: float = FLOW_A_PUSH,
    tol: float = 0.01,
    max_iter: int = 8,
) -> float:
    """Find the A push flow that leaves [A] ``margin`` above its zero point.

    Secant iteration on ``push_excess(flow) - margin``, one full solve per
    step. ``tol`` is on the excess, so 0.01 lands within one percentage point
    of ``margin``.
    """
    def miss(flow):
        reactor = run_reactor(database_path, flow_a_push=flow)
        excess = push_excess(reactor)
        print(f"  A push {flow * 1e3:.4f} g/s -> [A] {excess:+.1%} "
              "vs zero point")
        return excess - margin

    flows = [flow_guess]
    misses = [miss(flow_guess)]
    for _ in range(max_iter):
        if abs(misses[-1]) < tol:
            return flows[-1]
        if len(flows) == 1:
            next_flow = 1.2 * flow_guess
        else:
            slope = (misses[-1] - misses[-2]) / (flows[-1] - flows[-2])
            next_flow = max(flows[-1] - misses[-1] / slope, 0.0)
        flows.append(next_flow)
        misses.append(miss(next_flow))
    raise RuntimeError(
        f"A push flow did not converge in {max_iter} solves; last "
        f"{flows[-1]:g} kg/s missed the margin by {misses[-1]:+.3f}.")


# ---------------------------------------------------------------- plotting
def retitle(fig, title: str, suptitle: bool = True):
    """Replace the CR01 title on a figure drawn by a CR01H plot function."""
    if suptitle:
        fig.suptitle(title)
    else:
        fig.axes[0].set_title(title)
    return fig


def plot_profiles(reactor: ContinuousCrystallizer):
    """w_i vs time, T vs time and the w_i-T phase portrait, side by side.

    i runs over the species in REACTIONS (the solvent left out).
    """
    import matplotlib.pyplot as plt

    result = reactor.result
    time_min = np.asarray(result.time) / 60.0
    fractions = mass_fractions(reactor)
    index = {name: j for j, name in enumerate(reactor.name_species)}
    species = reacting_species(reactor)
    colors = dict(zip(species, SERIES_COLORS * len(species)))
    temp = np.ravel(result.global_temp)

    fig, (ax_ca, ax_t, ax_phase) = plt.subplots(1, 3, figsize=(15, 4.5))

    for name in species:
        ax_ca.plot(time_min, fractions[:, index[name]], color=colors[name],
                   lw=2, label=name)
    ax_ca.set_xlabel("Time [min]")
    ax_ca.set_ylabel("$w_i$ [kg/kg]")
    ax_ca.set_title("Mass fraction of each species")
    ax_ca.legend(title="i", frameon=False)

    ax_t.plot(time_min, temp, color=TEXT_PRIMARY, lw=2)
    ax_t.axhline(CR02_TEMPERATURE, color=TEXT_MUTED, lw=1, ls="--")
    ax_t.annotate("coolant inlet", (time_min[-1], CR02_TEMPERATURE),
                  xytext=(0, 4), textcoords="offset points", ha="right",
                  va="bottom", color=TEXT_MUTED, fontsize=9)
    ax_t.set_xlabel("Time [min]")
    ax_t.set_ylabel("$T$ [K]")
    ax_t.set_title("Crystallizer temperature")

    for name in species:
        frac = fractions[:, index[name]]
        ax_phase.plot(frac, temp, color=colors[name], lw=2, label=name)
        ax_phase.plot(frac[0], temp[0], "o", ms=7, color=colors[name],
                      mec="white", mew=1.5)
        ax_phase.plot(frac[-1], temp[-1], "s", ms=7, color=colors[name],
                      mec="white", mew=1.5)
    ax_phase.set_xlabel("$w_i$ [kg/kg]")
    ax_phase.set_ylabel("$T$ [K]")
    ax_phase.set_title(f"$w_i$ vs $T$  (● t = 0, ■ t = "
                       f"{time_min[-1]:.0f} min)")
    ax_phase.legend(title="i", frameon=False)

    for axis in (ax_ca, ax_t, ax_phase):
        axis.grid(True, color="#e5e4e0", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CR02 - reactive crystallizer with A push")
    fig.tight_layout()
    return fig


def plot_solubility(reactor: ContinuousCrystallizer):
    """Liquid F against its solubility, and [A] against its zero point."""
    import matplotlib.pyplot as plt

    result = reactor.result
    time_min = np.asarray(result.time) / 60.0
    species_index = {name: j for j, name in enumerate(reactor.name_species)}
    liquid_conc = np.asarray(result.mole_conc_liquid0, dtype=float)
    f_conc = liquid_conc[:, species_index["F"]]
    a_conc = liquid_conc[:, species_index["A"]]
    temperature = np.ravel(result.global_temp)
    solubility = np.asarray(result.solubility, dtype=float).reshape(
        len(time_min), -1
    )[:, 0]

    fig, (ax_f, ax_a) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_f.plot(time_min, f_conc, color=SERIES_COLORS[0], lw=2,
              label="[F] liquid")
    ax_f.plot(time_min, solubility, color=SERIES_COLORS[1], lw=2,
              label="F solubility")
    ax_f.set_xlabel("Time [min]")
    ax_f.set_ylabel("Concentration [mol/L]")
    ax_f.set_title("F concentration and solubility")
    ax_f.legend(frameon=False)

    ax_a.plot(time_min, a_conc, color=SERIES_COLORS[2], lw=2, label="[A]")
    ax_a.plot(time_min, zero_solubility_a_conc(temperature),
              color=TEXT_MUTED, lw=1.5, ls="--",
              label="[A] for zero F solubility")
    ax_a.set_xlabel("Time [min]")
    ax_a.set_ylabel("[A] [mol/L]")
    ax_a.set_title("A pushed past the zero-solubility point")
    ax_a.legend(frameon=False)

    for axis in (ax_f, ax_a):
        axis.grid(True, color="#e5e4e0", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CR02 - F solubility pushed to zero by feeding A")
    fig.tight_layout()
    return fig


def settings_rows(reactor: ContinuousCrystallizer) -> list[tuple[str, str, str]]:
    """(group, setting, value) rows describing this run."""
    def composition(named):
        solutes = {k: v for k, v in named.items() if v}
        if not solutes:
            return "solvent only"
        return (", ".join(f"{k} {v:g}" for k, v in solutes.items())
                + " (mass frac.)")

    push_flow = sum(float(np.sum(conn.stream.Phases[0].mass_flow))
                    for conn in reactor.inlet_connections[2:])
    # After the solve the phases hold the final state, so read the initial
    # charge off the result instead.
    initial_vol = float(np.ravel(reactor.result.vessel_vol)[0])
    initial_mass = float(np.sum(reactor.result.mass_j_liquid0[0]))
    tau = initial_mass / total_feed_flow(push_flow)
    area = 4 * initial_vol / REACTOR_DIAMETER
    u_ht = 1.0 / (1.0 / REACTOR_H_CONV + 1.0 / COOLANT_H_CONV)
    final = reactor.result
    final_temp = np.ravel(final.global_temp)[-1]
    final_frac = mass_fractions(reactor)[-1]
    final_solub = np.ravel(final.solubility)[-1]
    solvent_name = database_solvent(tuple(reactor.name_species))
    # Clip solver roundoff (e.g. -1e-25) so it prints as 0, not -0.0000.
    # Species that never appear are left out so the row fits the table.
    final_solutes = {name: max(w, 0.0)
                     for name, w in zip(reactor.name_species, final_frac)
                     if name != solvent_name and w > 1e-9}

    return [
        ("Initial conditions", "Temperature", f"{CR02_TEMPERATURE:g} K"),
        ("Initial conditions", "Volume",
         f"{initial_vol:.4g} m³ ({initial_mass:g} kg derived)"),
        ("Initial conditions", "Charge", composition(INITIAL_REACTOR)),
        ("Feeds", "Feed A", f"{FLOW_A:g} kg/s, {composition(FEED_A)}"),
        ("Feeds", "Feed B", f"{FLOW_B:g} kg/s, {composition(FEED_B)}"),
        ("Feeds", "A push", f"{push_flow:g} kg/s, {composition(FEED_A_PUSH)}"),
        ("Feeds", "Feed temperature", f"{FEED_TEMPERATURE:g} K"),
        ("Feeds", "Residence time", f"{tau:g} s ({tau / 60:.1f} min)"),
        ("Kinetics", "Reactions", "; ".join(REACTIONS)),
        ("Kinetics", "k at T_ref",
         ", ".join(f"{k:g}" for k in RATE_CONSTANTS) + " L/mol/s"),
        ("Kinetics", "Ea",
         ", ".join(f"{e:g}" for e in ACTIVATION_ENERGIES) + " J/mol"),
        ("Kinetics", "T_ref", f"{KINETICS_TEMP_REF:g} K"),
        ("Kinetics", "ΔH_rxn", f"{HEAT_OF_REACTION:g} J/mol"),
        ("Crystallization", "Nucleation (k, Ea, b)",
         ", ".join(f"{p:g}" for p in NUCL_PRIM)),
        ("Crystallization", "Growth (k, Ea, g)",
         ", ".join(f"{p:g}" for p in GROWTH)),
        ("Crystallization", "Supersaturation", SUPERSATURATION_TYPE),
        ("Jacket", "Coolant", f"{COOLANT_MASS_FLOW:g} kg/s at "
                              f"{CR02_TEMPERATURE:g} K"),
        ("Jacket", "U · A", f"{u_ht:g} W/m²/K × {area:g} m² "
                            f"= {u_ht * area:g} W/K"),
        ("Run", "Runtime", f"{np.ravel(final.time)[-1]:g} s"),
        ("Run", "Integrator", type(reactor.integrator).__name__),
        ("Run", "Final T", f"{final_temp:.3f} K"),
        ("Run", "Final [A]",
         f"{a_conc_history(reactor)[-1]:.4f} mol/L (zero point "
         f"{zero_solubility_a_conc(final_temp):.4f} mol/L)"),
        ("Run", "Final F solubility", f"{final_solub:.4g} mol/L"),
        ("Run", "Final [mass frac.]",
         ", ".join(f"{k} {v:.5f}" for k, v in final_solutes.items())),
    ]


def plot_settings(reactor: ContinuousCrystallizer):
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

    ax.set_title("CR02 settings and initial conditions", pad=12)
    fig.tight_layout()
    return fig


def plot_mass_balance(reactor: ContinuousCrystallizer,
                      balances: SimpleNamespace):
    """Inlet vs outlet species balance at the final time, as a table."""
    import matplotlib.pyplot as plt

    to_g = 1.0e3  # kg/s -> g/s
    feeds = balances.feeds[-1] * to_g
    generation = balances.generation[-1] * to_g
    outlet = balances.outlet[-1] * to_g
    accumulation = balances.accumulation[-1] * to_g
    feed_labels = ["Feed A", "Feed C", "A push"] + [
        f"Feed {i + 1}" for i in range(3, len(feeds))]

    def cell(value):
        return "0" if abs(value) < 5e-7 else f"{value:.4f}"

    columns = (["Species"]
               + [f"{name} in [g/s]" for name in feed_labels[:len(feeds)]]
               + ["Generated [g/s]", "Out [g/s]", "Accumulated [g/s]"])
    rows = []
    for j, name in enumerate(reactor.name_species):
        rows.append([name] + [cell(f[j]) for f in feeds]
                    + [cell(generation[j]), cell(outlet[j]),
                       cell(accumulation[j])])
    rows.append(["Total"] + [cell(f.sum()) for f in feeds]
                + [cell(generation.sum()), cell(outlet.sum()),
                   cell(accumulation.sum())])

    total_in = feeds.sum()
    gap = steady_state_gap(balances)
    status = ("steady state" if gap < STEADY_STATE_TOL
              else "NOT yet at steady state")

    fig, ax = plt.subplots(figsize=(11, 0.3 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="right",
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor("#e5e4e0")
        if col == 0:
            cell_obj.set_text_props(ha="left")
        if row == 0 or row == len(rows):
            cell_obj.set_text_props(weight="bold")
        if row == 0:
            cell_obj.set_facecolor("#f0efec")

    ax.set_title(
        f"CR02 mass balance at t = {balances.time[-1]:g} s  [g/s]\n"
        f"in {total_in:.4f} g/s, out {outlet.sum():.4f} g/s; "
        f"max accumulation {gap * 100:.2g}% of inlet flow ({status})",
        pad=12)
    fig.text(0.5, 0.02,
             "Terms read from PharmaPy's material contribution buffer "
             "(inlet, intraphase, outlet) and its net species rates.",
             ha="center", color=TEXT_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def save_figures(reactor: ContinuousCrystallizer,
                 out_dir: Path = RESULTS_DIR,
                 balances: SimpleNamespace | None = None):
    """Draw all figures, save them as PNGs and return them by name."""
    if balances is None:
        balances = replay_balances(reactor)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "CR02_profiles": plot_profiles(reactor),
        "CR02_solubility": plot_solubility(reactor),
        "CR02_native": retitle(plot_native(reactor),
                               "CR02 - PharmaPy.Plotting.plot_function"),
        "CR02_settings": plot_settings(reactor),
        "CR02_mass_balance": plot_mass_balance(reactor, balances),
        "CR02_heat_sources": retitle(
            plot_heat_sources(balances),
            "CR02 heat sources from PharmaPy's energy balance",
            suptitle=False),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    parser.add_argument("--size-feed", action="store_true",
                        help="find the A push flow for PUSH_MARGIN first")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    database = Path(os.environ.get("PHARMAPY_DATABASE", DEFAULT_DATABASE_PATH))

    flow_a_push = FLOW_A_PUSH
    if args.size_feed:
        print(f"Sizing the A push for [A] {PUSH_MARGIN:.0%} above the "
              "zero-solubility point:")
        flow_a_push = size_push_feed(database)
        print(f"Set FLOW_A_PUSH = {flow_a_push:.4g}  # kg/s\n")

    cr02 = run_reactor(database, flow_a_push=flow_a_push)

    print("Final crystallizer liquid mass fractions:")
    for name, fraction in zip(cr02.name_species, mass_fractions(cr02)[-1]):
        print(f"  {name}: {fraction:.6f}")
    final_temp = np.ravel(cr02.result.global_temp)[-1]
    print(f"Final temperature: {final_temp:.3f} K")
    print(f"Final [A]: {a_conc_history(cr02)[-1]:.4f} mol/L "
          f"(zero-solubility point {zero_solubility_a_conc(final_temp):.4f})")
    print(f"Final F solubility: {np.ravel(cr02.result.solubility)[-1]:.4g} "
          "mol/L")
    if push_excess(cr02) < 0.0:
        print("  WARNING: [A] is below the zero-solubility point; raise "
              "FLOW_A_PUSH or run with --size-feed.")

    balances = replay_balances(cr02)
    print(f"\nMass balance at t = {balances.time[-1]:g} s (g/s):")
    print(f"  {'species':<8}{'in':>10}{'generated':>11}{'out':>10}"
          f"{'accum.':>10}")
    for j, name in enumerate(cr02.name_species):
        print(f"  {name:<8}{balances.inlet[-1][j] * 1e3:>10.4f}"
              f"{balances.generation[-1][j] * 1e3:>11.4f}"
              f"{balances.outlet[-1][j] * 1e3:>10.4f}"
              f"{balances.accumulation[-1][j] * 1e3:>10.4f}")
    gap = steady_state_gap(balances)
    if gap >= STEADY_STATE_TOL:
        print(f"  WARNING: accumulation is {gap:.2%} of the inlet flow; "
              "increase NUM_RESIDENCE_TIMES to reach steady state.")

    print("\nHeat terms at the final time (W):")
    for name, values in balances.heat.items():
        print(f"  {HEAT_TERM_LABELS.get(name, name):<20}{values[-1]:>10.3f}")

    save_figures(cr02, balances=balances)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
