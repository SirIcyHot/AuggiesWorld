"""Two-feed jacketed CSTR with three parallel reactions, on PharmaPy's
refactored ContinuousReactor.

The flowsheet is::

    FEED_A --+
             +--> R02R
    FEED_B --+

``R02R`` is ``PharmaPy.Reactors_Refactored.ContinuousReactor``. The refactored
vessel takes a list of inlet streams directly, so the separate mixer unit the
legacy ``Reactors.CSTR`` needed is gone. Feed compositions are supplied by
compound name; the database controls the order PharmaPy uses internally.

Three parallel reactions consume the same reactants::

    A + C + G + I --> D
    A + C + G + I --> E
    A + C + G + I --> J

each with its own power-law orders in A, C and G. I is consumed by every
reaction but carries no order, so its concentration does not enter any rate.

Everything is specified and reported on a mass basis: feeds as mass flows
[kg/s] with solute mass fractions, the initial charge as a mass [kg] with
solute mass fractions, and compositions plotted as mass fractions. The
solvent makes up whatever mass fraction the solutes leave. The kinetics are
the exception: PharmaPy evaluates the rate law on molar concentrations, so
rate constants stay in L/mol/s.

The reactor is jacketed rather than isothermal so that temperature is a solved
state: the heat of reaction raises it, the jacket and the feeds pull it back,
and the Arrhenius term feeds it back into the rate. That is what makes the
w_i-vs-T phase portrait meaningful.

Run it with::

    python R02R.py                # solve, save figures, show them
    python R02R.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    R02R_profiles.png   w_i vs time, T vs time, w_i vs T (reacting species)
    R02R_native.png     the same run through PharmaPy.Plotting.plot_function
    R02R_settings.png   the settings and initial conditions used for the run
    R02R_mass_balance.png  inlet vs outlet species balance at the final time
    R02R_heat_sources.png  energy-balance terms (heat sources) vs time

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
DEFAULT_DATABASE_PATH = HERE / "compound_databaseActual.json"
RESULTS_DIR = HERE / "results"

# ---------------------------------------------------------------- settings
REACTIONS = [
    "A + C + G + I --> D",
    "A + C + G + I --> E",
    "A + C + G + I --> J",
]
# Orders for each reaction, keyed by reactant. I is deliberately not given
# an order: PharmaPy needs one per reactant, so it gets 0 (C_I**0 = 1).
REACTION_ORDERS = [
    {"A": 0.29, "C": 1.46, "G": 0.29},  # --> D
    {"A": 0.21, "C": 1.60, "G": 0.21},  # --> E
    {"A": 0.50, "C": 0.97, "G": 0.50},  # --> J
]
RATE_CONSTANTS = np.array([1.1667e-3, 6.3333e-4, 3.333e-3])  # L/mol/s, at KINETICS_TEMP_REF
ACTIVATION_ENERGIES = np.array([74863.4, -54320.2, -104523.8])  # J/mol
KINETICS_TEMP_REF = 313.15  # K, rate constants above are quoted here
HEAT_OF_REACTION = np.array([-6.22e5, -6.22e5, -6.22e5])  # J/mol, negative = exothermic

# Compositions are solute mass fractions; the solvent (B) is the remainder,
# so it never needs listing.
FLOW_A = 1.0e-2  # kg/s
FLOW_B = 1.0e-2  # kg/s
FEED_TEMPERATURE = 313.15  # K
FEED_A = {"A": 0.0207, "G": 0.0337}  # mass fraction, ~0.33 mol/L each
FEED_B = {"C": 0.0459, "I": 0.0264}  # mass fraction, ~0.33 mol/L each

REACTOR_MASS = 10.0  # kg, initial charge (~0.01 m**3 of water)
REACTOR_DIAMETER = 0.2  # m, jacket area = 4 V / D
REACTOR_H_CONV = 1000.0  # W/m**2/K, vessel side
INITIAL_TEMPERATURE = 313.15  # K
INITIAL_REACTOR = {}  # mass fraction, pure solvent

COOLANT_MASS_FLOW = 0.1  # kg/s
COOLANT_TEMPERATURE = 313.15  # K
COOLANT_H_CONV = 1000.0  # W/m**2/K, jacket side

# 7.2 residence times: long enough that the end of the run is a steady
# state (accumulation below STEADY_STATE_TOL of the throughput).
RUNTIME = 3600.0  # s
STEADY_STATE_TOL = 1.0e-2  # accumulation / inlet mass flow, per species

# Categorical slots 1-7 (blue, orange, aqua, yellow, violet, pink, slate),
# one per reacting species.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1",
                 "#d6457a", "#5f6b7a")
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"


class BalanceRecordingReactor(ContinuousReactor):
    """ContinuousReactor that keeps the energy-balance terms it computes.

    ``MultiPhaseVessel.energy_balances`` collects every heat term into a
    local ``contributions`` dict, sums it and returns only dT/dt. The
    shaft-work hook is the last term it adds before summing, so copying the
    dict there captures every term exactly as PharmaPy computed it. Nothing
    about the balance itself changes.
    """

    def add_shaftwork_energy_terms(self, contributions, time,
                                   completed_state):
        super().add_shaftwork_energy_terms(contributions, time,
                                           completed_state)
        self.energy_contributions = {
            name: float(value) for name, value in contributions.items()
        }


# ---------------------------------------------------------------- builders
def database_species(database_path: Path) -> tuple[str, ...]:
    """Return the database species names in PharmaPy's declared order."""
    parsed_database = ParseDatabase(str(database_path), to_arrays=False)
    return tuple(parsed_database["name_species"])


def database_solvent(species_names: tuple[str, ...]) -> str:
    """Select the configured solvent without relying on species positions."""
    for candidate in ("solvent", "Water", "B"):
        if candidate in species_names:
            return candidate
    raise KeyError(
        "The database must contain one of the named solvent entries: "
        "solvent, Water, or B."
    )


def named_mass_fraction(
    composition: Mapping[str, float],
    species_names: tuple[str, ...],
    solvent_name: str,
) -> np.ndarray:
    """Convert named solute mass fractions to PharmaPy's ordered array.

    The solvent takes the remainder. Listing the solvent explicitly is
    allowed, but then the fractions must already sum to 1.
    """
    unknown_names = set(composition) - set(species_names)
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise KeyError(f"Unknown compounds in composition: {names}")
    if any(value < 0.0 for value in composition.values()):
        raise ValueError("Mass fractions cannot be negative.")

    fractions = np.array(
        [composition.get(name, 0.0) for name in species_names],
        dtype=float,
    )
    total = fractions.sum()

    if solvent_name in composition:
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Mass fractions including the solvent sum to {total:g}, "
                "not 1.")
        return fractions

    if total > 1.0:
        raise ValueError(f"Solute mass fractions sum to {total:g} > 1.")
    fractions[species_names.index(solvent_name)] = 1.0 - total
    return fractions


def reaction_order_matrix(species_names: tuple[str, ...]) -> list[list[float]]:
    """Expand REACTION_ORDERS to PharmaPy's ``params_f`` layout.

    PharmaPy expects one row per reaction holding an order for every
    reactant of that reaction, in database order. Reactants without an
    entry in REACTION_ORDERS (here I) get 0, i.e. no concentration term.
    """
    matrix = []
    for reaction, orders in zip(REACTIONS, REACTION_ORDERS):
        reactants = [term.strip().split()[-1]
                     for term in reaction.split("-->")[0].split("+")]
        unknown = set(orders) - set(reactants)
        if unknown:
            raise KeyError(f"Orders given for non-reactants of '{reaction}': "
                           + ", ".join(sorted(unknown)))
        ordered = sorted(reactants, key=species_names.index)
        matrix.append([float(orders.get(name, 0.0)) for name in ordered])
    return matrix


def make_feed(
    database_path: Path,
    composition: Mapping[str, float],
    flow_rate: float,
    species_names: tuple[str, ...],
    solvent_name: str,
) -> LiquidStream:
    """Create one named-composition liquid feed; flow_rate in kg/s."""
    return LiquidStream(
        str(database_path),
        temp=FEED_TEMPERATURE,
        mass_frac=named_mass_fraction(composition, species_names,
                                      solvent_name),
        mass_flow=flow_rate,
        name_solv=solvent_name,
    )

def build_reactor(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
) -> BalanceRecordingReactor:
    """Build R02R with both feeds attached as separate inlets."""
    database_path = Path(database_path)
    species_names = database_species(database_path)
    solvent_name = database_solvent(species_names)

    reactor = BalanceRecordingReactor(
        integrator=ScipyBackend(),
        h_conv=REACTOR_H_CONV,
        diam=REACTOR_DIAMETER,
    )
    reactor.Phases = LiquidPhase(
        str(database_path),
        temp=INITIAL_TEMPERATURE,
        mass_frac=named_mass_fraction(INITIAL_REACTOR, species_names,
                                      solvent_name),
        mass=REACTOR_MASS,
        name_solv=solvent_name,
    )
    reactor.RxnKinetics = RxnKinetics(
        path=str(database_path),
        rxn_list=REACTIONS,
        k_params=RATE_CONSTANTS,
        ea_params=ACTIVATION_ENERGIES,
        params_f=reaction_order_matrix(species_names),
        temp_ref=KINETICS_TEMP_REF,
        delta_hrxn=HEAT_OF_REACTION,
    )
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
    runtime: float = RUNTIME,
) -> ContinuousReactor:
    """Solve R02R and return it with ``result`` populated."""
    reactor = build_reactor(database_path)
    reactor.solve_unit(runtime=runtime, verbose=False)
    return reactor


# ---------------------------------------------------------------- balances
def replay_balances(reactor: BalanceRecordingReactor) -> SimpleNamespace:
    """Re-evaluate PharmaPy's balances at every saved time point.

    Mirrors ``MultiPhaseVessel.find_output_states_from_replay``: an
    independent copy of the vessel is set to each accepted solver state and
    asked for its balances through the same methods the integrator calls,
    including the positivity limiter. Nothing here re-derives a balance; the
    terms are read out of PharmaPy's contribution buffer and energy dict.

    Returns
    -------
    SimpleNamespace
        ``time`` [s]; ``feeds`` (time, feed, species) inlet mass flows [kg/s];
        ``inlet``, ``generation``, ``outlet``, ``accumulation`` (time,
        species) [kg/s], outlet positive; ``heat`` {term: array} [W].
    """
    pseudo = reactor.create_pseudo()
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
        pseudo.energy_balances(time, completed, buffer)

        terms = buffer.contributions
        feeds.append([np.array(transfer.species_flow, dtype=float)
                      for transfer in buffer.aux[buffer.INLET]])
        inlet.append(terms[buffer.INLET].copy())
        generation.append(terms[buffer.INTRAPHASE].copy())
        outlet.append(-terms[buffer.OUTLET])
        accumulation.append(np.array(rates, dtype=float))
        heat.append(pseudo.energy_contributions)

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
def mass_fractions(reactor: ContinuousReactor) -> np.ndarray:
    """Liquid-phase mass-fraction history, (time, species)."""
    mass_j = np.asarray(reactor.result.mass_j_liquid0, dtype=float)
    return mass_j / mass_j.sum(axis=1, keepdims=True)


def reacting_species(reactor: ContinuousReactor) -> list[str]:
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


def plot_profiles(reactor: ContinuousReactor):
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

    fig.suptitle("R02R - refactored ContinuousReactor")
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
    fig.suptitle("R02R - PharmaPy.Plotting.plot_function")
    fig.tight_layout()
    return fig


def settings_rows(reactor: ContinuousReactor) -> list[tuple[str, str, str]]:
    """(group, setting, value) rows describing this run."""
    def composition(named):
        solutes = {k: v for k, v in named.items() if v}
        if not solutes:
            return "solvent only"
        return (", ".join(f"{k} {v:g}" for k, v in solutes.items())
                + " (mass frac.)")

    initial_vol = float(np.ravel(reactor.result.vessel_vol)[0])
    tau = REACTOR_MASS / (FLOW_A + FLOW_B)
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
        ("Initial conditions", "Mass",
         f"{REACTOR_MASS:g} kg ({initial_vol:.4g} m³)"),
        ("Initial conditions", "Charge", composition(INITIAL_REACTOR)),
        ("Feeds", "Feed A", f"{FLOW_A:g} kg/s, {composition(FEED_A)}"),
        ("Feeds", "Feed B", f"{FLOW_B:g} kg/s, {composition(FEED_B)}"),
        ("Feeds", "Feed temperature", f"{FEED_TEMPERATURE:g} K"),
        ("Feeds", "Residence time", f"{tau:g} s ({tau / 60:.1f} min)"),
        *[("Kinetics", f"Rxn {i + 1}: {rxn}",
           "orders " + ", ".join(f"{k} {v:g}" for k, v in orders.items())
           + f"; k {k_i:g} L/mol/s; Ea {ea:.1f} J/mol; ΔH {dh:g} J/mol")
          for i, (rxn, orders, k_i, ea, dh) in enumerate(zip(
              REACTIONS, REACTION_ORDERS, RATE_CONSTANTS,
              ACTIVATION_ENERGIES, HEAT_OF_REACTION))],
        ("Kinetics", "T_ref", f"{KINETICS_TEMP_REF:g} K"),
        ("Jacket", "Coolant", f"{COOLANT_MASS_FLOW:g} kg/s at "
                              f"{COOLANT_TEMPERATURE:g} K"),
        ("Jacket", "U · A", f"{u_ht:g} W/m²/K × {area:g} m² "
                            f"= {u_ht * area:g} W/K"),
        ("Run", "Runtime", f"{RUNTIME:g} s"),
        ("Run", "Integrator", type(reactor.integrator).__name__),
        ("Run", "Final T", f"{np.ravel(final.global_temp)[-1]:.3f} K"),
        ("Run", "Final [mass frac.]",
         ", ".join(f"{k} {v:.5f}" for k, v in final_solutes.items())),
    ]


def plot_settings(reactor: ContinuousReactor):
    """Render the run's settings and initial conditions as a table figure."""
    import matplotlib.pyplot as plt

    rows = settings_rows(reactor)
    fig, ax = plt.subplots(figsize=(13, 0.3 * len(rows) + 0.6))
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
        colWidths=[0.13, 0.22, 0.65],
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

    ax.set_title("R02R settings and initial conditions", pad=12)
    fig.tight_layout()
    return fig


def plot_mass_balance(reactor: ContinuousReactor,
                      balances: SimpleNamespace):
    """Inlet vs outlet species balance at the final time, as a table."""
    import matplotlib.pyplot as plt

    to_g = 1.0e3  # kg/s -> g/s
    feeds = balances.feeds[-1] * to_g
    generation = balances.generation[-1] * to_g
    outlet = balances.outlet[-1] * to_g
    accumulation = balances.accumulation[-1] * to_g
    feed_labels = ["Feed A", "Feed B"] + [
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
        f"R02R mass balance at t = {balances.time[-1]:g} s  [g/s]\n"
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
    "shaftwork": "Shaft work",
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
    ax.set_title("R02R heat sources from PharmaPy's energy balance")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False)
    ax.grid(True, color="#e5e4e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if inactive:
        fig.text(0.5, 0.01, "Zero throughout: " + ", ".join(inactive),
                 ha="center", color=TEXT_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def save_figures(reactor: BalanceRecordingReactor,
                 out_dir: Path = RESULTS_DIR,
                 balances: SimpleNamespace | None = None):
    """Draw all figures, save them as PNGs and return them by name."""
    if balances is None:
        balances = replay_balances(reactor)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "R02R_profiles": plot_profiles(reactor),
        "R02R_native": plot_native(reactor),
        "R02R_settings": plot_settings(reactor),
        "R02R_mass_balance": plot_mass_balance(reactor, balances),
        "R02R_heat_sources": plot_heat_sources(balances),
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
    r02r = run_reactor(database)

    print("Final CSTR mass fractions:")
    for name, fraction in zip(r02r.name_species, mass_fractions(r02r)[-1]):
        print(f"  {name}: {fraction:.6f}")
    print(f"Final temperature: {np.ravel(r02r.result.global_temp)[-1]:.3f} K")

    balances = replay_balances(r02r)
    print(f"\nMass balance at t = {balances.time[-1]:g} s (g/s):")
    print(f"  {'species':<8}{'in':>10}{'generated':>11}{'out':>10}"
          f"{'accum.':>10}")
    for j, name in enumerate(r02r.name_species):
        print(f"  {name:<8}{balances.inlet[-1][j] * 1e3:>10.4f}"
              f"{balances.generation[-1][j] * 1e3:>11.4f}"
              f"{balances.outlet[-1][j] * 1e3:>10.4f}"
              f"{balances.accumulation[-1][j] * 1e3:>10.4f}")
    gap = steady_state_gap(balances)
    if gap >= STEADY_STATE_TOL:
        print(f"  WARNING: accumulation is {gap:.2%} of the inlet flow; "
              "increase RUNTIME to reach steady state.")

    print("\nHeat terms at the final time (W):")
    for name, values in balances.heat.items():
        print(f"  {HEAT_TERM_LABELS.get(name, name):<20}{values[-1]:>10.3f}")

    save_figures(r02r, balances=balances)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
