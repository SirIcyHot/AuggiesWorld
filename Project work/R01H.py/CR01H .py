"""Reactive continuous crystallizer on PharmaPy's refactored vessel API.

The flowsheet is::

    FEED_A --+
             +--> CR01
    FEED_C --+

``CR01`` is a ``PharmaPy.Crystallizers_Refactored.ContinuousCrystallizer``
with a liquid reaction region and an F-only 1D-FVM solid population balance.
Feed compositions are supplied by compound name; the database controls the
order PharmaPy uses internally.

Everything is specified and reported on a mass basis: feeds as mass flows
[kg/s] with solute mass fractions, the initial charge as a mass [kg] with
solute mass fractions, and compositions plotted as mass fractions. The
solvent makes up whatever mass fraction the solutes leave. The kinetics are
the exception: PharmaPy evaluates the rate law on molar concentrations, so
rate constants stay in L/mol/s.

The crystallizer is jacketed rather than isothermal. The liquid reaction is
``2 A + C -> F`` and the custom rate law is
``k_rate * exp(-[A]) * [C]``. F crystallizes with a temperature- and
A-dependent solubility correlation.

Run it with::

    python R01H.py                # solve, save figures, show them
    python R01H.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    CR01_profiles.png   w_i vs time, T vs time, w_i vs T (reacting species)
    CR01_solubility.png  liquid F and F solubility vs time
    CR01_native.png     the same run through PharmaPy.Plotting.plot_function
    CR01_settings.png   the settings and initial conditions used for the run
    CR01_mass_balance.png  inlet vs outlet species balance at the final time
    CR01_heat_sources.png  energy-balance terms (heat sources) vs time

Both balance figures read PharmaPy's own balance terms rather than
re-deriving them: the solved trajectory is replayed through the vessel's
``material_balances`` and ``energy_balances`` (the same calls the integrator
makes), and each term is recorded as PharmaPy computed it.

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
from R01H import (
    ACTIVATION_ENERGIES,
    COOLANT_H_CONV,
    COOLANT_MASS_FLOW,
    COOLANT_TEMPERATURE,
    DEFAULT_DATABASE_PATH,
    FEED_A,
    FEED_B,
    FEED_TEMPERATURE,
    FLOW_A,
    FLOW_B,
    HEAT_OF_REACTION,
    INITIAL_REACTOR,
    INITIAL_TEMPERATURE,
    KINETICS_TEMP_REF,
    PARAMS_F_DUMMY,
    RATE_CONSTANTS,
    REACTOR_DIAMETER,
    REACTOR_H_CONV,
    REACTOR_VOLUME,
    REACTIONS,
    RESULTS_DIR,
    SERIES_COLORS,
    NUM_RESIDENCE_TIMES,
    STEADY_STATE_TOL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    build_kinetics,
    database_solvent,
    database_species,
    make_feed,
    named_mass_fraction,
)


# Crystallization-only settings.
KB_PRIM, B_ORDER = 1.0e3, 3.0
KG_GROWTH, G_ORDER = 1.0e5, 2.0
NUCL_PRIM = (KB_PRIM, 0.0, B_ORDER)
GROWTH = (KG_GROWTH, 0.0, G_ORDER)
NUCL_SEC = (0.0, 0.0, 0.0, 0.0)
CRYSTAL_KV = 1.0
CRYSTAL_SIZE_GRID = np.arange(1.0, 201.0)

SOLUBILITY_M1 = -0.00461
SOLUBILITY_B1 = 0.014636164
SOLUBILITY_M2 = -0.23273
SOLUBILITY_B2 = 0.341571

def make_solubility_function(species_names: tuple[str, ...]):
    """Create F solubility correlation using liquid [A] and temperature."""
    index_a = species_names.index("A")

    def solubility(temp, conc):
        conc = np.asarray(conc)
        a_conc = conc[..., index_a]
        temp_c = np.asarray(temp) - 273.15
        return ((SOLUBILITY_M1 * a_conc + SOLUBILITY_B1) * temp_c
                + SOLUBILITY_M2 * a_conc + SOLUBILITY_B2)

    return solubility


def reactor_residence_time(reactor: ContinuousCrystallizer) -> float:
    """Return the initial residence time implied by the configured mass."""
    initial_mass = float(reactor.Phases.Liquids[0].mass)
    return initial_mass / (FLOW_A + FLOW_B)


def build_reactor(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
) -> ContinuousCrystallizer:
    """Build R01 with both feeds attached as separate inlets."""
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
        temp=INITIAL_TEMPERATURE,
        mass_frac=named_mass_fraction(INITIAL_REACTOR, species_names,
                                      solvent_name),
        vol=REACTOR_VOLUME,
        name_solv=solvent_name,
    )
    solid_mass_frac = np.zeros(len(species_names))
    solid_mass_frac[species_names.index("F")] = 1.0
    solid = SolidPhase(
        str(database_path),
        temp=INITIAL_TEMPERATURE,
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
        solub_fn=make_solubility_function(species_names),
        nucl_prim=NUCL_PRIM,
        nucl_sec=NUCL_SEC,
        growth=GROWTH,
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
        temp_in=COOLANT_TEMPERATURE,
        h_conv=COOLANT_H_CONV,
    )
    reactor.Inlet = [
        make_feed(database_path, FEED_A, FLOW_A, species_names, solvent_name),
        make_feed(database_path, FEED_B, FLOW_B, species_names, solvent_name),
    ]
    return reactor


def run_reactor(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    runtime: float | None = None,
) -> ContinuousCrystallizer:
    """Solve R01 and return it with ``result`` populated."""
    reactor = build_reactor(database_path)
    if runtime is None:
        runtime = NUM_RESIDENCE_TIMES * reactor_residence_time(reactor)
    reactor.solve_unit(runtime=runtime, verbose=False)
    return reactor


# ---------------------------------------------------------------- balances
def replay_balances(reactor: ContinuousCrystallizer) -> SimpleNamespace:
    """Re-evaluate PharmaPy's balances at every saved time point.

    Mirrors ``MultiPhaseVessel.find_output_states_from_replay``: an
    independent copy of the vessel is set to each accepted solver state and
    asked for its balances through the same methods the integrator calls,
    including the positivity limiter. Nothing here re-derives a balance; the
    terms are read out of PharmaPy's contribution buffer.

    Returns
    -------
    SimpleNamespace
        ``time`` [s]; ``feeds`` (time, feed, species) inlet mass flows [kg/s];
        ``inlet``, ``generation``, ``outlet``, ``accumulation`` (time,
        species) [kg/s], outlet positive; ``heat`` {term: array} [W].
    """
    pseudo = reactor.create_pseudo()
    pseudo.phase_states.compile(pseudo.Phases)
    collection = reactor.solver_state_collection
    result = reactor.result
    history = {
        key: np.asarray(getattr(result, collection.format_key(key)))
        for key in collection.states
    }

    feeds, inlet, generation, outlet, accumulation, heat = ([] for _ in
                                                            range(6))
    for i, time in enumerate(np.asarray(result.time)):
        state = {key: values[i] for key, values in history.items()}
        completed = pseudo.complete_state(state, time)
        pseudo.update_phases_from_state(completed)

        rates, buffer = pseudo.material_balances(
            time, completed, limiter_dt=pseudo.positivity_horizon)

        contributions = {
            "inlet": 0.0,
            "intraphase": 0.0,
            "crossphase": 0.0,
            "outlet": 0.0,
            "utility": 0.0,
            "mixing": 0.0,
        }
        pseudo.add_inlet_energy_terms(
            contributions, buffer.aux[buffer.INLET], time, completed
        )
        pseudo.add_intraphase_energy_terms(
            contributions, buffer.aux[buffer.INTRAPHASE], time, completed
        )
        pseudo.add_crossphase_energy_terms(
            contributions, buffer.aux[buffer.CROSSPHASE], time, completed
        )
        pseudo.add_outlet_energy_terms(
            contributions, buffer.aux[buffer.OUTLET], time, completed
        )
        pseudo.add_utility_energy_terms(contributions, time, completed)
        pseudo.add_mixing_energy_terms(contributions, time, completed)

        feeds.append([np.array(transfer.species_flow, dtype=float)
                      for transfer in buffer.aux[buffer.INLET]])
        inlet.append(np.asarray(buffer.contributions[buffer.INLET], dtype=float)
                     .copy())
        generation.append(np.asarray(buffer.contributions[buffer.INTRAPHASE],
                                    dtype=float).copy())
        outlet.append(-np.asarray(buffer.contributions[buffer.OUTLET],
                                 dtype=float))
        accumulation.append(np.array(rates, dtype=float))
        heat.append({name: float(value) for name, value in contributions.items()})

    return SimpleNamespace(
        time=np.asarray(result.time),
        feeds=np.array(feeds),
        inlet=np.array(inlet),
        generation=np.array(generation),
        outlet=np.array(outlet),
        accumulation=np.array(accumulation),
        heat={name: np.array([h[name] for h in heat]) for name in heat[0]},
    )


def steady_state_gap(balances: SimpleNamespace) -> float:
    """Largest species accumulation at the final time, relative to the
    total inlet mass flow."""
    return float(np.abs(balances.accumulation[-1]).max()
                 / balances.inlet[-1].sum())


# ---------------------------------------------------------------- plotting
def mass_fractions(reactor: ContinuousCrystallizer) -> np.ndarray:
    """Liquid-phase mass-fraction history, (time, species)."""
    mass_j = np.asarray(reactor.result.mass_j_liquid0, dtype=float)
    return mass_j / mass_j.sum(axis=1, keepdims=True)


def reacting_species(reactor: ContinuousCrystallizer) -> list[str]:
    """Species named in REACTIONS, in order of appearance, minus the solvent."""
    solvent_name = database_solvent(tuple(reactor.name_species))
    names = []
    for reaction in REACTIONS:
        for side in reaction.split("-->"):
            for term in side.split("+"):
                name = term.strip().split()[-1]
                if name not in names and name != solvent_name:
                    names.append(name)
    return names


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
    ax_t.axhline(COOLANT_TEMPERATURE, color=TEXT_MUTED, lw=1, ls="--")
    ax_t.annotate("coolant inlet", (time_min[-1], COOLANT_TEMPERATURE),
                  xytext=(0, 4), textcoords="offset points", ha="right",
                  va="bottom", color=TEXT_MUTED, fontsize=9)
    ax_t.set_xlabel("Time [min]")
    ax_t.set_ylabel("$T$ [K]")
    ax_t.set_title("Reactor temperature")

    for name in species:
        frac = fractions[:, index[name]]
        ax_phase.plot(frac, temp, color=colors[name], lw=2, label=name)
        ax_phase.plot(frac[0], temp[0], "o", ms=7, color=colors[name],
                      mec="white", mew=1.5)
        ax_phase.plot(frac[-1], temp[-1], "s", ms=7, color=colors[name],
                      mec="white", mew=1.5)
    ax_phase.set_xlabel("$w_i$ [kg/kg]")
    ax_phase.set_ylabel("$T$ [K]")
    ax_phase.set_title(f"$w_i$ vs $T$  (\u25cf t = 0, \u25a0 t = "
                       f"{time_min[-1]:.0f} min)")
    ax_phase.legend(title="i", frameon=False)

    for axis in (ax_ca, ax_t, ax_phase):
        axis.grid(True, color="#e5e4e0", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CR01 - reactive continuous crystallizer")
    fig.tight_layout()
    return fig


def plot_solubility(reactor: ContinuousCrystallizer):
    """Plot liquid F concentration and its temperature/A-dependent solubility."""
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

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time_min, f_conc, color=SERIES_COLORS[0], lw=2,
            label="[F] liquid")
    ax.plot(time_min, solubility, color=SERIES_COLORS[1], lw=2,
            label="F solubility")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Concentration [mol/L]")
    ax.set_title("F concentration and temperature/A-dependent solubility")
    ax.legend(frameon=False)
    ax.grid(True, color="#e5e4e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(time_min, a_conc, color=SERIES_COLORS[2], ls="--", lw=1.5,
             label="[A]")
    ax2.set_ylabel("[A] [mol/L]")
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_native(reactor: ContinuousCrystallizer):
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
    solvent_name = database_solvent(tuple(reactor.name_species))
    solutes = [name for name in reactor.name_species if name != solvent_name]
    fig, axes = plot_function(
        view,
        (["mass_j_liquid0", solutes], "global_temp", "q_ht"),
        fig_map=(0, 1, 2),
        ncols=3,
        ylabels=("m_j", "T", "Q_ht"),
        figsize=(15, 4.5),
    )
    for axis in axes:
        axis.set_xlabel("time (s)")
    fig.suptitle("CR01 - PharmaPy.Plotting.plot_function")
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

    initial_vol = float(np.ravel(reactor.result.vessel_vol)[0])
    initial_mass = float(reactor.Phases.Liquids[0].mass)
    tau = reactor_residence_time(reactor)
    area = 4 * initial_vol / REACTOR_DIAMETER
    u_ht = 1.0 / (1.0 / REACTOR_H_CONV + 1.0 / COOLANT_H_CONV)
    final = reactor.result
    final_frac = mass_fractions(reactor)[-1]
    solvent_name = database_solvent(tuple(reactor.name_species))
    # Clip solver roundoff (e.g. -1e-25) so it prints as 0, not -0.0000.
    # Species that never appear are left out so the row fits the table.
    final_solutes = {name: max(w, 0.0)
                     for name, w in zip(reactor.name_species, final_frac)
                     if name != solvent_name and w > 1e-9}

    return [
        ("Initial conditions", "Temperature", f"{INITIAL_TEMPERATURE:g} K"),
        ("Initial conditions", "Volume",
         f"{initial_vol:.4g} m³ ({initial_mass:g} kg derived)"),
        ("Initial conditions", "Charge", composition(INITIAL_REACTOR)),
        ("Feeds", "Feed A", f"{FLOW_A:g} kg/s, {composition(FEED_A)}"),
        ("Feeds", "Feed B", f"{FLOW_B:g} kg/s, {composition(FEED_B)}"),
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
        ("Run", "Runtime",
         f"{NUM_RESIDENCE_TIMES * tau:g} s"),
        ("Run", "Integrator", type(reactor.integrator).__name__),
        ("Run", "Final T", f"{np.ravel(final.global_temp)[-1]:.3f} K"),
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

    ax.set_title("CR01 settings and initial conditions", pad=12)
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
    feed_labels = ["Feed A", "Feed C"] + [
        f"Feed {i + 1}" for i in range(2, len(feeds))]

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

    fig, ax = plt.subplots(figsize=(10, 0.3 * len(rows) + 1.2))
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
        f"CR01 mass balance at t = {balances.time[-1]:g} s  [g/s]\n"
        f"in {total_in:.4f} g/s, out {outlet.sum():.4f} g/s; "
        f"max accumulation {gap * 100:.2g}% of inlet flow ({status})",
        pad=12)
    fig.text(0.5, 0.02,
             "Terms read from PharmaPy's material contribution buffer "
             "(inlet, intraphase, outlet) and its net species rates.",
             ha="center", color=TEXT_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


HEAT_TERM_LABELS = {
    "intraphase": "Reaction (ΔH_rxn)",
    "utility": "Jacket",
    "inlet": "Feeds (sensible)",
    "outlet": "Outlet",
    "crossphase": "Phase transfer",
    "mixing": "Mixing",
}


def plot_heat_sources(balances: SimpleNamespace):
    """Each energy-balance term vs time; positive adds heat to the vessel."""
    import matplotlib.pyplot as plt

    time_min = balances.time / 60.0
    active = [name for name in HEAT_TERM_LABELS
              if np.any(np.abs(balances.heat.get(name, 0.0)) > 1e-9)]
    inactive = [HEAT_TERM_LABELS[name] for name in HEAT_TERM_LABELS
                if name in balances.heat and name not in active]
    net = sum(balances.heat[name] for name in balances.heat)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color=TEXT_MUTED, lw=0.8)
    for color, name in zip(SERIES_COLORS, active):
        ax.plot(time_min, balances.heat[name], color=color, lw=2,
                label=HEAT_TERM_LABELS[name])
    ax.plot(time_min, net, color=TEXT_PRIMARY, lw=1.5, ls="--",
            label="Net (sum)")

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Heat rate into vessel [W]")
    ax.set_title("CR01 heat sources from PharmaPy's energy balance")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False)
    ax.grid(True, color="#e5e4e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if inactive:
        fig.text(0.5, 0.01, "Zero throughout: " + ", ".join(inactive),
                 ha="center", color=TEXT_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def save_figures(reactor: ContinuousCrystallizer,
                 out_dir: Path = RESULTS_DIR,
                 balances: SimpleNamespace | None = None):
    """Draw all figures, save them as PNGs and return them by name."""
    if balances is None:
        balances = replay_balances(reactor)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "CR01_profiles": plot_profiles(reactor),
        "CR01_solubility": plot_solubility(reactor),
        "CR01_native": plot_native(reactor),
        "CR01_settings": plot_settings(reactor),
        "CR01_mass_balance": plot_mass_balance(reactor, balances),
        "CR01_heat_sources": plot_heat_sources(balances),
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

    print("Final crystallizer liquid mass fractions:")
    for name, fraction in zip(r01.name_species, mass_fractions(r01)[-1]):
        print(f"  {name}: {fraction:.6f}")
    print(f"Final temperature: {np.ravel(r01.result.global_temp)[-1]:.3f} K")

    balances = replay_balances(r01)
    print(f"\nMass balance at t = {balances.time[-1]:g} s (g/s):")
    print(f"  {'species':<8}{'in':>10}{'generated':>11}{'out':>10}"
          f"{'accum.':>10}")
    for j, name in enumerate(r01.name_species):
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

    save_figures(r01, balances=balances)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
