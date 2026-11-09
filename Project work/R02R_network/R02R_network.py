"""Isothermal CSTR with the Z1/F -> INT1 -> ... -> D/E reaction network, on
PharmaPy's refactored ContinuousReactor.

This is ``R02R_onemethod/R02R.py`` with three changes:

1. **Kinetics.** The three parallel power-law reactions are replaced by the
   22-step network in ``STEPS`` (reported activation energies, kJ/mol). Every
   step is taken as written in the table, i.e. first order in the species it
   consumes: ``X --> Y`` with ``r = k * C_X``.
2. **Isothermal.** ``isothermal=True``, so there is no energy balance, no
   jacket and no heat-of-reaction data needed. The heat-source figure is gone.
3. **Database / feed.** Uses ``compound_database_expanded.json`` (which holds
   Z1, INT1, ... INT2ONO2). The feed is R02R's, with the substrate C swapped
   for Z1 at the same 0.509 mol/L. A, G and I are now inert carriers.

Rate constants
--------------
Only activation energies were reported, so the rate constant at the
operating temperature is estimated with transition-state (Eyring) theory,
treating each reported barrier as a free energy of activation::

    k_i(T) = (k_B T / h) * exp(-Ea_i / (R T))          [1/s]

That is a pre-exponential of k_B T / h ~ 7e12 1/s at 60 C. This is the
standard assumption when only computed barriers are available, but it is a
starting point to calibrate, not data (a 6 kJ/mol error in a barrier is a
factor of ~10 in k).

Two consequences you need to know about:

* **Fast steps are capped** (``K_MAX``). Eyring gives k up to 7e12 1/s
  (the barrierless INT1 trapping steps), which is far too stiff to integrate.
  More importantly, PharmaPy's positivity limiter scales *every* reaction
  extent by one common factor when any species would empty within
  ``positivity_horizon`` (1 s). A first-order step with k > 1/s trips it and
  slows the whole network. So each species' total first-order loss is
  capped at ``K_MAX`` = 0.5 1/s, scaling all of that species' branches by
  the same factor so the branching ratios Eyring predicts are kept. 0.5 1/s
  is still ~2000x faster than 1/tau, so capped steps still act instantly on
  the residence-time scale and the steady state does not depend on K_MAX.
* **High barriers do nothing at 60 C.** With tau ~ 67 min, a step needs
  roughly Ea < 105 kJ/mol to make progress at 60 C. Everything past Z2/Z5
  (118.9 to 248.9 kJ/mol) is effectively off, so D and E barely form. That
  follows directly from the barriers. To get D/E you need a hotter run, a
  longer residence time, or barriers fitted to measured yields.

Because k is evaluated at ``OPERATING_TEMPERATURE`` and that is also passed
as ``temp_ref``, the Arrhenius correction inside PharmaPy is exactly 1 here.
Ea still goes into RxnKinetics, so the object is valid if you move the
temperature, but for a temperature sweep rebuild the reactor with
``rate_constants(T)`` at each T so the cap is re-applied.

Mass basis caveat
-----------------
The steps are written without co-reactants (e.g. Z1 203.2 g/mol -> INT1
186.2 g/mol), so they do not conserve mass. PharmaPy converts molar rates to
mass with each species' MW, so the "Generated" column of the mass balance
does not sum to zero. The molar species balances are exact. Add the missing
co-reactants/co-products to ``STEPS`` (e.g. ``Z1 --> INT1 + B``) when the
full stoichiometry is known.

Run it with::

    python R02R_network.py              # solve, save figures, show them
    python R02R_network.py --no-show    # solve and save figures only

Figures go to ``results/``:

    R02R_network_profiles.png      w_i vs time (linear and log)
    R02R_network_native.png        PharmaPy.Plotting.plot_function view
    R02R_network_settings.png      settings, feed and outlet
    R02R_network_kinetics.png      Ea, Eyring k, capped k used, k * tau
    R02R_network_mass_balance.png  inlet / generated / outlet at final time
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


HERE = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = (HERE.parents[1] / "compounddatabase"
                         / "compound_database_expanded.json")
RESULTS_DIR = HERE / "results"

# ---------------------------------------------------------------- kinetics
# (consumes, produces, reported Ea [kJ/mol]). The table's "IN1AcO" is read
# as INT1AcO, the species every later row and the database use.
STEPS = [
    ("Z1", "INT1", 48.8),
    ("F", "INT1", 48.8),
    ("INT1", "INT1ONO2", 0.0),
    ("INT1", "INT1OH", 0.0),
    ("INT1", "INT1AcO", 0.0),
    ("INT1ONO2", "Z2", 147.2),
    ("INT1OH", "Z2", 65.2),
    ("INT1AcO", "Z2", 43.5),
    ("INT1AcO", "INT3", 88.7),
    ("INT3", "INT4", 12.1),
    ("INT4", "Z3", 33.1),
    ("Z2", "INT5", 118.9),
    ("INT5", "Z4", 248.9),
    ("INT5", "D", 123.6),
    ("Z2", "Z5", 73.3),
    ("Z5", "Z6", 166.0),
    ("Z5", "INT2OH", 206.9),
    ("Z5", "E", 140.9),
    ("INT2OH", "Z7", 116.81),
    ("INT2OH", "INT2ONO2", 147.2),
    ("INT2OH", "E", 129.0),
    ("INT2ONO2", "E", 183.1),
]
REACTIONS = [f"{reactant} --> {product}" for reactant, product, _ in STEPS]
ACTIVATION_ENERGIES = np.array([ea for *_, ea in STEPS]) * 1e3  # J/mol

OPERATING_TEMPERATURE = 60 + 273.15  # K, reactor held here
KINETICS_TEMP_REF = OPERATING_TEMPERATURE  # k is evaluated here (see doc)
TRANSMISSION_COEFF = 1.0  # Eyring kappa
K_MAX = 0.5  # 1/s, cap on a species' total first-order loss (see doc)
HEAT_OF_REACTION = np.zeros(len(STEPS))  # unused: isothermal

BOLTZMANN = 1.380649e-23  # J/K
PLANCK = 6.62607015e-34  # J s
GAS_CONSTANT = 8.314462618  # J/mol/K


def eyring_rate_constants(temp: float) -> np.ndarray:
    """Uncapped Eyring k for every step at ``temp`` [K], in 1/s."""
    prefactor = TRANSMISSION_COEFF * BOLTZMANN * temp / PLANCK
    return prefactor * np.exp(-ACTIVATION_ENERGIES / (GAS_CONSTANT * temp))


# What-if factors on individual Eyring k values, keyed by reaction string
# ("Z5 --> E": 1e6) or "all". A factor f is the same as lowering that barrier
# by R*T*ln(f), i.e. 6.4 kJ/mol per factor of 10 at 60 C. Applied before
# the K_MAX cap, so branching stays consistent. Empty = reported barriers.
RATE_MULTIPLIERS: dict[str, float] = {}


def rate_constants(temp: float = KINETICS_TEMP_REF,
                   k_max: float = K_MAX,
                   multipliers: Mapping[str, float] | None = None
                   ) -> np.ndarray:
    """Eyring k at ``temp`` with each species' total loss capped at k_max.

    ``multipliers`` (default RATE_MULTIPLIERS) scales chosen steps first.
    All steps leaving one species are then scaled by the same factor, so the
    branching between them is the (multiplied) Eyring value.
    """
    if multipliers is None:
        multipliers = RATE_MULTIPLIERS
    unknown = set(multipliers) - set(REACTIONS) - {"all"}
    if unknown:
        raise KeyError("RATE_MULTIPLIERS keys must be reaction strings or "
                       "'all': " + ", ".join(sorted(unknown)))
    k = eyring_rate_constants(temp) * np.array([
        multipliers.get(rxn, multipliers.get("all", 1.0))
        for rxn in REACTIONS])
    reactants = [reactant for reactant, _, _ in STEPS]
    total_out = {name: 0.0 for name in reactants}
    for name, k_i in zip(reactants, k):
        total_out[name] += k_i
    scale = np.array([min(1.0, k_max / total_out[name])
                      for name in reactants])
    return k * scale


RATE_CONSTANTS = rate_constants()  # 1/s at KINETICS_TEMP_REF

# ---------------------------------------------------------------- feed etc.
# R02R's feed with C replaced by Z1. mol/L, full composition incl. solvent H.
FEEDS = [
    ("Feed", 2.0e-2, "mol/L",
     {"A": 3.841, "Z1": 0.509, "G": 5.276, "H": 5.938, "I": 1.110}),
]
FEED_TEMPERATURE = 303.15  # K (stream property; the vessel is isothermal)

REACTOR_MASS = 80.0  # kg, initial charge
REACTOR_DIAMETER = 0.2  # m
INITIAL_REACTOR = {}  # mass fraction, pure solvent

RUNTIME = 10 * 3600.0  # s, ~9 residence times
STEADY_STATE_TOL = 1.0e-2  # accumulation / inlet mass flow, per species
PLOT_FLOOR = 1e-6  # mass fractions below this at all times are not plotted

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1",
                 "#d6457a", "#5f6b7a", "#0f8a8a", "#b5651d", "#4b4fb8")
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"


# ---------------------------------------------------------------- builders
def database_species(database_path: Path) -> tuple[str, ...]:
    parsed_database = ParseDatabase(str(database_path), to_arrays=False)
    return tuple(parsed_database["name_species"])


def database_solvent(species_names: tuple[str, ...]) -> str:
    for candidate in ("solvent", "Water", "H"):
        if candidate in species_names:
            return candidate
    raise KeyError("The database must contain a solvent entry: "
                   "solvent, Water, or H.")


def named_mass_fraction(composition: Mapping[str, float],
                        species_names: tuple[str, ...],
                        solvent_name: str) -> np.ndarray:
    """Named solute mass fractions -> ordered array; solvent is the rest."""
    unknown_names = set(composition) - set(species_names)
    if unknown_names:
        raise KeyError("Unknown compounds in composition: "
                       + ", ".join(sorted(unknown_names)))
    if any(value < 0.0 for value in composition.values()):
        raise ValueError("Mass fractions cannot be negative.")
    fractions = np.array([composition.get(name, 0.0)
                          for name in species_names], dtype=float)
    total = fractions.sum()
    if solvent_name in composition:
        if not np.isclose(total, 1.0):
            raise ValueError(f"Mass fractions including the solvent sum to "
                             f"{total:g}, not 1.")
        return fractions
    if total > 1.0:
        raise ValueError(f"Solute mass fractions sum to {total:g} > 1.")
    fractions[species_names.index(solvent_name)] = 1.0 - total
    return fractions


def reaction_order_matrix(species_names: tuple[str, ...]) -> list[list[float]]:
    """``params_f``: one row per reaction, order 1 in its single reactant."""
    matrix = []
    for reaction in REACTIONS:
        reactants = [term.strip().split()[-1]
                     for term in reaction.split("-->")[0].split("+")]
        ordered = sorted(reactants, key=species_names.index)
        matrix.append([1.0 for _ in ordered])
    return matrix


def molar_to_mass_fraction(conc: Mapping[str, float],
                           database_path: Path) -> dict[str, float]:
    parsed_database = ParseDatabase(str(database_path), to_arrays=False)
    mw = dict(zip(parsed_database["name_species"], parsed_database["mw"]))
    mass_conc = {name: value * mw[name] for name, value in conc.items()}
    total = sum(mass_conc.values())
    return {name: value / total for name, value in mass_conc.items()}


def make_feeds(database_path: Path) -> list[LiquidStream]:
    species_names = database_species(database_path)
    solvent_name = database_solvent(species_names)
    feeds = []
    for _, flow_rate, basis, composition in FEEDS:
        if basis == "mol/L":
            composition = molar_to_mass_fraction(composition, database_path)
        elif basis != "mass_frac":
            raise ValueError(f"Unknown composition basis '{basis}'.")
        feeds.append(LiquidStream(
            str(database_path),
            temp=FEED_TEMPERATURE,
            mass_frac=named_mass_fraction(composition, species_names,
                                          solvent_name),
            mass_flow=flow_rate,
            name_solv=solvent_name,
        ))
    return feeds


def total_feed_flow() -> float:
    return sum(flow_rate for _, flow_rate, _, _ in FEEDS)


def build_reactor(database_path: Path | str = DEFAULT_DATABASE_PATH,
                  temp: float = OPERATING_TEMPERATURE) -> ContinuousReactor:
    """Isothermal CSTR at ``temp`` [K] with k re-evaluated at that T."""
    database_path = Path(database_path)
    species_names = database_species(database_path)
    solvent_name = database_solvent(species_names)

    reactor = ContinuousReactor(
        integrator=ScipyBackend(),
        isothermal=True,
        diam=REACTOR_DIAMETER,
    )
    reactor.Phases = LiquidPhase(
        str(database_path),
        temp=temp,
        mass_frac=named_mass_fraction(INITIAL_REACTOR, species_names,
                                      solvent_name),
        mass=REACTOR_MASS,
        name_solv=solvent_name,
    )
    reactor.RxnKinetics = RxnKinetics(
        path=str(database_path),
        rxn_list=REACTIONS,
        k_params=rate_constants(temp),
        ea_params=ACTIVATION_ENERGIES,
        params_f=reaction_order_matrix(species_names),
        temp_ref=temp,
        delta_hrxn=HEAT_OF_REACTION,
    )
    reactor.Inlet = make_feeds(database_path)
    return reactor


def run_reactor(database_path: Path | str = DEFAULT_DATABASE_PATH,
                runtime: float = RUNTIME,
                temp: float = OPERATING_TEMPERATURE) -> ContinuousReactor:
    reactor = build_reactor(database_path, temp)
    reactor.solve_unit(runtime=runtime, verbose=False)
    return reactor


# ---------------------------------------------------------------- balances
def replay_balances(reactor: ContinuousReactor) -> SimpleNamespace:
    """Re-evaluate PharmaPy's material balance at every saved time point.

    Same replay as R02R_onemethod, minus the energy balance (isothermal).
    """
    pseudo = reactor.create_pseudo()
    collection = reactor.solver_state_collection
    result = reactor.result
    history = {
        key: np.asarray(getattr(result, collection.format_key(key)))
        for key in collection.states
    }

    feeds, inlet, generation, outlet, accumulation = ([] for _ in range(5))
    for i, time in enumerate(np.asarray(result.time)):
        state = {key: values[i] for key, values in history.items()}
        completed = pseudo.complete_state(state, time)
        pseudo.update_phases_from_state(completed)
        rates, buffer = pseudo.material_balances(
            time, completed, limiter_dt=pseudo.positivity_horizon)
        terms = buffer.contributions
        feeds.append([np.array(transfer.species_flow, dtype=float)
                      for transfer in buffer.aux[buffer.INLET]])
        inlet.append(terms[buffer.INLET].copy())
        generation.append(terms[buffer.INTRAPHASE].copy())
        outlet.append(-terms[buffer.OUTLET])
        accumulation.append(np.array(rates, dtype=float))

    return SimpleNamespace(
        time=np.asarray(result.time),
        feeds=np.array(feeds),
        inlet=np.array(inlet),
        generation=np.array(generation),
        outlet=np.array(outlet),
        accumulation=np.array(accumulation),
    )


def steady_state_gap(balances: SimpleNamespace) -> float:
    return float(np.abs(balances.accumulation[-1]).max()
                 / balances.inlet[-1].sum())


# ---------------------------------------------------------------- plotting
def mass_fractions(reactor: ContinuousReactor) -> np.ndarray:
    mass_j = np.asarray(reactor.result.mass_j_liquid0, dtype=float)
    return mass_j / mass_j.sum(axis=1, keepdims=True)


def network_species() -> list[str]:
    names = []
    for reactant, product, _ in STEPS:
        for name in (reactant, product):
            if name not in names:
                names.append(name)
    return names


def visible_species(reactor: ContinuousReactor) -> list[str]:
    """Network species that ever exceed PLOT_FLOOR."""
    fractions = mass_fractions(reactor)
    index = {name: j for j, name in enumerate(reactor.name_species)}
    return [name for name in network_species()
            if fractions[:, index[name]].max() > PLOT_FLOOR]


def plot_profiles(reactor: ContinuousReactor):
    """w_i vs time, linear and log, for network species that appear."""
    import matplotlib.pyplot as plt

    time_min = np.asarray(reactor.result.time) / 60.0
    fractions = mass_fractions(reactor)
    index = {name: j for j, name in enumerate(reactor.name_species)}
    species = visible_species(reactor)
    colors = dict(zip(species, SERIES_COLORS * (len(species) // 10 + 1)))
    styles = dict(zip(species, (["-"] * 10 + ["--"] * 10 + [":"] * 10)))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 4.8))
    for name in species:
        w = fractions[:, index[name]]
        for axis in (ax_lin, ax_log):
            axis.plot(time_min, np.maximum(w, 0.0), color=colors[name],
                      ls=styles[name], lw=2, label=name)
    ax_log.set_yscale("log")
    ax_log.set_ylim(PLOT_FLOOR, None)
    ax_lin.set_title("Mass fraction of each network species")
    ax_log.set_title("Same, log scale")
    for axis in (ax_lin, ax_log):
        axis.set_xlabel("Time [min]")
        axis.set_ylabel("$w_i$ [kg/kg]")
        axis.grid(True, color="#e5e4e0", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    ax_log.legend(title="i", frameon=False, bbox_to_anchor=(1.02, 1),
                  loc="upper left")
    fig.suptitle(f"R02R network - isothermal CSTR at "
                 f"{OPERATING_TEMPERATURE - 273.15:g} °C")
    fig.tight_layout()
    return fig


def plot_native(reactor: ContinuousReactor):
    from PharmaPy.Plotting import plot_function

    result = reactor.result
    view = SimpleNamespace(result=result, states_di=result.di_states,
                           fstates_di=result.di_fstates)
    fig, axes = plot_function(
        view,
        (["mass_j_liquid0", visible_species(reactor)], "vessel_vol"),
        fig_map=(0, 1),
        ncols=2,
        ylabels=("m_j", "V"),
        figsize=(12, 4.5),
    )
    for axis in axes:
        axis.set_xlabel("time (s)")
    fig.suptitle("R02R network - PharmaPy.Plotting.plot_function")
    fig.tight_layout()
    return fig


def _table_figure(rows, columns, title, col_widths=None, footnote=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 0.3 * len(rows) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, colWidths=col_widths,
                     cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e4e0")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0efec")
    ax.set_title(title, pad=12)
    if footnote:
        fig.text(0.5, 0.01, footnote, ha="center", color=TEXT_MUTED,
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.03 if footnote else 0, 1, 1))
    return fig


def plot_kinetics():
    """Ea, raw Eyring k, k actually used and k*tau for every step."""
    tau = REACTOR_MASS / total_feed_flow()
    k_raw = eyring_rate_constants(KINETICS_TEMP_REF)
    rows = []
    for i, (rxn, ea, kr, ku) in enumerate(zip(
            REACTIONS, ACTIVATION_ENERGIES, k_raw, RATE_CONSTANTS)):
        note = ("capped" if ku < kr * (1 - 1e-12)
                else "inactive" if ku * tau < 1e-2 else "")
        rows.append([str(i + 1), rxn, f"{ea / 1e3:g}", f"{kr:.3e}",
                     f"{ku:.3e}", f"{ku * tau:.3g}", note])
    return _table_figure(
        rows,
        ["#", "Step", "Ea [kJ/mol]", "Eyring k [1/s]", "k used [1/s]",
         "k·τ", "Note"],
        f"R02R network kinetics at {KINETICS_TEMP_REF:g} K "
        f"(τ = {tau:g} s, K_MAX = {K_MAX:g} 1/s)",
        col_widths=[0.04, 0.26, 0.11, 0.15, 0.15, 0.12, 0.10],
        footnote="k = (k_B T/h) exp(-Ea/RT); capped: species' total loss "
                 "scaled to K_MAX, branching kept; inactive: k·τ < 0.01.")


def plot_settings(reactor: ContinuousReactor):
    def composition(named, basis="mass_frac"):
        solutes = {k: v for k, v in named.items() if v}
        if not solutes:
            return "solvent only"
        unit = "mol/L" if basis == "mol/L" else "mass frac."
        return ", ".join(f"{k} {v:g}" for k, v in solutes.items()) + \
            f" ({unit})"

    initial_vol = float(np.ravel(reactor.result.vessel_vol)[0])
    tau = REACTOR_MASS / total_feed_flow()
    final_frac = mass_fractions(reactor)[-1]
    index = {name: j for j, name in enumerate(reactor.name_species)}
    final = {name: max(final_frac[index[name]], 0.0)
             for name in network_species()
             if final_frac[index[name]] > PLOT_FLOOR}
    rows = [
        ("Reactor", "Mode", "Isothermal ContinuousReactor"),
        ("Reactor", "Temperature", f"{OPERATING_TEMPERATURE:g} K"),
        ("Reactor", "Initial mass",
         f"{REACTOR_MASS:g} kg ({initial_vol:.4g} m³)"),
        ("Reactor", "Initial charge", composition(INITIAL_REACTOR)),
        *[("Feeds", label, f"{flow:g} kg/s, {composition(named, basis)}")
          for label, flow, basis, named in FEEDS],
        ("Feeds", "Residence time", f"{tau:g} s ({tau / 60:.1f} min)"),
        ("Kinetics", "Steps", f"{len(STEPS)} first-order steps"),
        ("Kinetics", "k estimate",
         f"Eyring at {KINETICS_TEMP_REF:g} K, κ = {TRANSMISSION_COEFF:g}, "
         f"species loss capped at {K_MAX:g} 1/s"),
        ("Run", "Runtime", f"{RUNTIME:g} s"),
        ("Run", "Integrator", type(reactor.integrator).__name__),
        ("Run", "Final network w_i",
         ", ".join(f"{k} {v:.4g}" for k, v in final.items())),
    ]
    cells, previous = [], None
    for group, setting, value in rows:
        cells.append([group if group != previous else "", setting, value])
        previous = group
    return _table_figure(cells, ["Group", "Setting", "Value"],
                         "R02R network settings and initial conditions",
                         col_widths=[0.12, 0.18, 0.70])


def plot_mass_balance(reactor: ContinuousReactor,
                      balances: SimpleNamespace):
    to_g = 1.0e3
    feeds = balances.feeds[-1] * to_g
    generation = balances.generation[-1] * to_g
    outlet = balances.outlet[-1] * to_g
    accumulation = balances.accumulation[-1] * to_g
    labels = [label for label, *_ in FEEDS]

    def cell(value):
        return "0" if abs(value) < 5e-7 else f"{value:.4f}"

    active = [j for j, _ in enumerate(reactor.name_species)
              if max(abs(feeds[:, j]).max(), abs(generation[j]),
                     abs(outlet[j])) >= 5e-7]
    rows = [[reactor.name_species[j]] + [cell(f[j]) for f in feeds]
            + [cell(generation[j]), cell(outlet[j]), cell(accumulation[j])]
            for j in active]
    rows.append(["Total"] + [cell(f.sum()) for f in feeds]
                + [cell(generation.sum()), cell(outlet.sum()),
                   cell(accumulation.sum())])
    gap = steady_state_gap(balances)
    status = ("steady state" if gap < STEADY_STATE_TOL
              else "NOT yet at steady state")
    return _table_figure(
        rows,
        ["Species"] + [f"{n} in [g/s]" for n in labels]
        + ["Generated [g/s]", "Out [g/s]", "Accumulated [g/s]"],
        f"R02R network mass balance at t = {balances.time[-1]:g} s\n"
        f"max accumulation {gap * 100:.2g}% of inlet flow ({status})",
        footnote="Generated total is non-zero because the steps are written "
                 "without co-reactants (MW not conserved); molar balances "
                 "are exact.")


def save_figures(reactor, out_dir: Path = RESULTS_DIR, balances=None):
    if balances is None:
        balances = replay_balances(reactor)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "R02R_network_profiles": plot_profiles(reactor),
        "R02R_network_native": plot_native(reactor),
        "R02R_network_settings": plot_settings(reactor),
        "R02R_network_kinetics": plot_kinetics(),
        "R02R_network_mass_balance": plot_mass_balance(reactor, balances),
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
                                   DEFAULT_DATABASE_PATH))

    tau = REACTOR_MASS / total_feed_flow()
    print(f"Rate constants at {KINETICS_TEMP_REF:g} K "
          f"(Eyring, capped at {K_MAX:g} 1/s per species):")
    for rxn, ea, k in zip(REACTIONS, ACTIVATION_ENERGIES, RATE_CONSTANTS):
        print(f"  {rxn:<22} Ea {ea / 1e3:7.2f} kJ/mol  k {k:.3e} 1/s"
              f"  k*tau {k * tau:.3g}")

    r02r = run_reactor(database)

    fractions = mass_fractions(r02r)[-1]
    print("\nFinal CSTR mass fractions (network species):")
    for name in network_species():
        j = r02r.name_species.index(name)
        print(f"  {name:<9} {max(fractions[j], 0.0):.6g}")

    balances = replay_balances(r02r)
    gap = steady_state_gap(balances)
    if gap >= STEADY_STATE_TOL:
        print(f"  WARNING: accumulation is {gap:.2%} of the inlet flow; "
              "increase RUNTIME to reach steady state.")

    save_figures(r02r, balances=balances)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
