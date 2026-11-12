"""D02H: H wash of the F01H filter cake (the dissolver half of block W02H).

The flowsheet is::

                                         +----------- W02H -----------+
    CR01H --> CR02H --> F01H --cake----> | D02H --> F02H -------cake--+--> D01H
                          |              |  ^         |               |
                          |              +--|---------|---------------+
                          +--> filtrate    FEED_H     +--> wash filtrate

W02H is a reslurry wash: D02H re-suspends F01H's cake in fresh H and F02H
filters it again, so the cake reaching D01H carries H instead of most of the
B-based mother liquor. The two are separate unit ops, shown as one block in
the flowsheet.

D02H is identical to D01H: the same ``ContinuousCrystallizer`` charged with H,
the same F solubility, kinetics, size grid and tank settings, all read from
``D01H`` so the two cannot drift apart (``build_dissolver`` is D01H's own).
Standalone it is fed F01H's cake (``f01h_cake`` solves R01H -> F01H, as D01H
used to before the wash went in) and pure H, and so dissolves the same few
per mille of the crystals D01H does.

Solubility of F (mol/L), with H as antisolvent (see ``D01H``)::

    c_sat = S0 * exp(-K_H * [H])

S0 is the cake liquid's F concentration, F01H's mother liquor being
saturated.

The residence time comes from the solvent charge::

    tau = INITIAL_MASS / (cake mass flow + H_FLOW)

and the run lasts ``NUM_RESIDENCE_TIMES`` of them, enough to reach steady
state.

Run it with::

    python D02H.py                # solve, save figures, show them
    python D02H.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    D02H_profiles.png      crystals held, saturation, crystal flow in vs out
    D02H_csd.png           F crystal size distribution, fed vs leaving
    D02H_settings.png      the settings and initial conditions used
    D02H_mass_balance.png  inlet vs outlet species balance at the final time
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
from PharmaPy.DataClasses import PhaseRef
from PharmaPy.Mechanisms import PopulationBalanceMechanism
from PharmaPy.MixedPhases_Refactored import MixedStream
from PharmaPy.Streams_Refactored import LiquidStream

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/D02H
F01H_DIR = HERE.parent / "F01H"
D01H_DIR = HERE.parent / "D01H"
RESULTS_DIR = HERE / "results"

sys.path[:0] = [str(F01H_DIR), str(D01H_DIR)]

import F01H  # noqa: E402  (puts R01H and CustomUnitOps on the path)
import R01H  # noqa: E402
import D01H  # noqa: E402
from continuous_filter import SlurryFlow, stream_from_flow  # noqa: E402

DB_PATH = F01H.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
# Identical to D01H: every setting is D01H's, so change them there.
CRYSTAL_SPECIES = D01H.CRYSTAL_SPECIES
SOLVENT_NAME = D01H.SOLVENT_NAME
ANTISOLVENT_NAME = D01H.ANTISOLVENT_NAME

H_FLOW = D01H.H_FLOW  # kg/s
H_TEMPERATURE = D01H.H_TEMPERATURE  # K

INITIAL_MASS = D01H.INITIAL_MASS  # kg
INITIAL_COMPOSITION = D01H.INITIAL_COMPOSITION
DISSOLVER_TEMPERATURE = D01H.DISSOLVER_TEMPERATURE  # K, None: cake T
DISSOLVER_DIAMETER = D01H.DISSOLVER_DIAMETER  # m

ANTISOLVENT_K_H = D01H.ANTISOLVENT_K_H  # L/mol

NUCL_PRIM = D01H.NUCL_PRIM
NUCL_SEC = D01H.NUCL_SEC
GROWTH = D01H.GROWTH
DISSOLUTION = D01H.DISSOLUTION

CRYSTAL_SIZE_GRID = D01H.CRYSTAL_SIZE_GRID  # um
CRYSTAL_KV = D01H.CRYSTAL_KV

NUM_RESIDENCE_TIMES = D01H.NUM_RESIDENCE_TIMES
STEADY_STATE_TOL = D01H.STEADY_STATE_TOL

# Categorical slots 1-2 (blue, orange): fed vs leaving / liquid vs c_sat.
SERIES_COLORS = ("#2a78d6", "#eb6834")
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID_COLOR = "#e5e4e0"


# ---------------------------------------------------------------- builders
def species_names() -> tuple[str, ...]:
    return R01H.database_species(DB_PATH)


def regrid(flow: SlurryFlow, grid=CRYSTAL_SIZE_GRID) -> SlurryFlow:
    """``flow`` with its crystals re-binned onto ``grid``.

    The mass density (number * x**3) is interpolated linearly, since the
    number density of a mass-shaped distribution goes as 1/x**3 and would
    invent mass between coarse nodes. It is then scaled so the crystal mass
    flow is unchanged. The liquid is untouched.
    """
    grid = np.asarray(grid, dtype=float)
    mass_shape = np.interp(grid, flow.x_grid,
                           flow.crystal_flow * flow.x_grid**3, right=0.0)
    number = mass_shape / grid**3
    new = dataclasses.replace(flow, x_grid=grid, crystal_flow=number,
                              dx=None)
    new.crystal_flow = number * flow.solid_mass_flow / new.solid_mass_flow
    return new


def f01h_cake():
    """Solve R01H -> F01H and re-bin the cake onto CRYSTAL_SIZE_GRID.

    Returns (cake MixedStream, cake SlurryFlow), both on D02H's grid.
    """
    *_, filt = F01H.run_flowsheet(DB_PATH)
    cake_flow = regrid(filt.cake_flow)
    return stream_from_flow(cake_flow, include_solid=True), cake_flow


def h_stream(names) -> LiquidStream:
    """Fresh H feed."""
    return LiquidStream(
        str(DB_PATH), mass_flow=H_FLOW, temp=H_TEMPERATURE,
        mass_frac=R01H.named_mass_fraction({}, names, ANTISOLVENT_NAME),
        name_solv=ANTISOLVENT_NAME, verbose=False)


def solubility_s0(cake: MixedStream) -> float:
    """F solubility in H-free liquid [mol/L]: the cake liquid's [F]."""
    liquid = cake.Phases[0]
    return float(np.asarray(liquid.mole_conc)[
        list(liquid.name_species).index(CRYSTAL_SPECIES)])


def dissolver_temperature(cake: MixedStream) -> float:
    if DISSOLVER_TEMPERATURE is None:
        return float(cake.Phases[0].temp)
    return DISSOLVER_TEMPERATURE


def feed_mass_flow(cake: MixedStream) -> float:
    """Total mass flow into the tank: cake (liquid + crystals) plus H."""
    return sum(float(np.sum(s.mass_flow)) for s in cake.Streams) + H_FLOW


def residence_time(cake: MixedStream) -> float:
    """Nominal residence time [s] from the solvent charge."""
    return INITIAL_MASS / feed_mass_flow(cake)


def build_dissolver(cake: MixedStream, s0: float | None = None
                    ) -> ContinuousCrystallizer:
    """D02H: D01H's own vessel (``D01H.build_dissolver``), fed ``cake``."""
    return D01H.build_dissolver(cake, s0)


def run_flowsheet(runtime: float | None = None):
    """R01H -> F01H -> D02H. Returns (cake, cake flow, dissolver)."""
    cake, cake_flow = f01h_cake()
    tank = build_dissolver(cake)

    if runtime is None:
        runtime = NUM_RESIDENCE_TIMES * residence_time(cake)

    tank.solve_unit(runtime=runtime, verbose=False)
    return cake, cake_flow, tank


# ---------------------------------------------------------------- results
def population_balance(tank):
    return tank.Phases.get_phase_from_ref(PhaseRef("solid", 0)) \
        .get_mechanism(PopulationBalanceMechanism)


def crystal_weight(tank) -> np.ndarray:
    """Number density [#/m**3/um] -> crystal mass per m**3 slurry, per cell."""
    pbm = population_balance(tank)
    return (pbm.density * pbm.kv * np.asarray(pbm.x_grid)**3 * pbm.dx
            * 1e-18)


def history(tank, cake_flow) -> SimpleNamespace:
    """Time profiles read from the solved tank.

    ``time`` [s], ``crystal_mass`` held [kg], ``crystal_out`` and
    ``crystal_in`` [kg/s], liquid ``conc`` of F and its ``solubility``
    [mol/L], and ``fraction_dissolved`` [-], 1 - crystals out / crystals
    fed (meaningful at steady state; during start-up it includes the
    crystals the tank is still accumulating).
    """
    res = tank.result
    names = species_names()
    j = names.index(CRYSTAL_SPECIES)
    time = np.asarray(res.time)
    vessel_vol = np.ravel(res.vessel_vol)
    vol_flow = np.ravel(tank.outputs["vol_flow"])
    conc = np.asarray(res.mole_conc_liquid0, dtype=float)
    distrib = np.asarray(res.distrib_solid0, dtype=float)
    crystal_kg_m3 = distrib @ crystal_weight(tank)
    crystal_in = cake_flow.solid_mass_flow
    crystal_out = crystal_kg_m3 * vol_flow
    return SimpleNamespace(
        time=time,
        crystal_mass=crystal_kg_m3 * vessel_vol,
        crystal_out=crystal_out,
        crystal_in=np.full_like(time, crystal_in),
        conc=conc[:, j],
        solubility=tank.solubility(float(tank.Phases.Liquids[0].temp), conc),
        fraction_dissolved=1.0 - crystal_out / crystal_in,
    )


def final_balance(tank) -> SimpleNamespace:
    """PharmaPy's own balance terms at the final time [kg/s].

    The final solver state is set on an independent copy of the vessel and
    its ``material_balances`` called, as the integrator does; each term is
    read from the contribution buffer rather than re-derived.

    ``feeds`` (feed phase, species) inlet flows, in the order cake liquid,
    cake crystals, H; liquid ``inlet``, ``to_liquid`` (dissolved),
    ``outlet``, ``accumulation`` (species); and ``crystal`` {inlet,
    dissolved, outlet, accumulation}. Crystal ``inlet`` is PharmaPy's feed
    term on the population balance, which includes its dilution term.
    """
    pseudo = tank.create_pseudo()
    pseudo.phase_states.compile(pseudo.Phases)
    collection = tank.solver_state_collection
    res = tank.result
    time = float(np.asarray(res.time)[-1])
    state = {key: np.asarray(getattr(res, collection.format_key(key)))[-1]
             for key in collection.states}
    completed = pseudo.complete_state(state, time)
    pseudo.update_phases_from_state(completed)
    rates, buffer = pseudo.material_balances(
        time, completed, limiter_dt=pseudo.positivity_horizon)

    slices = collection.material_slices
    liquid = next(s for key, s in slices.items() if key.name != "distrib")
    cells = next(s for key, s in slices.items() if key.name == "distrib")
    rows = {name: np.asarray(buffer.contributions[getattr(buffer, name)],
                             dtype=float)
            for name in ("INLET", "CROSSPHASE", "OUTLET")}
    rates = np.asarray(rates, dtype=float)
    weight = crystal_weight(tank) * float(np.ravel(res.vessel_vol)[-1])

    def mass_rate(row):
        return float(weight @ row[cells])

    return SimpleNamespace(
        time=time,
        feeds=np.array([np.array(t.species_flow, dtype=float)
                        for t in buffer.aux[buffer.INLET]]),
        feed_phases=[t.mapping.source_phaseref.phase_type
                     for t in buffer.aux[buffer.INLET]],
        inlet=rows["INLET"][liquid],
        to_liquid=rows["CROSSPHASE"][liquid],
        outlet=-rows["OUTLET"][liquid],
        accumulation=rates[liquid],
        crystal={
            "inlet": mass_rate(rows["INLET"]),
            "dissolved": -mass_rate(rows["CROSSPHASE"]),
            "outlet": -mass_rate(rows["OUTLET"]),
            "accumulation": mass_rate(rates),
        },
    )


def steady_state_gap(balance: SimpleNamespace) -> float:
    """Largest relative accumulation: liquid species against the liquid
    feed, crystals against the crystal feed."""
    liquid = np.abs(balance.accumulation).max() / balance.inlet.sum()
    crystal = (abs(balance.crystal["accumulation"])
               / max(abs(balance.crystal["inlet"]), 1e-30))
    return float(max(liquid, crystal))


def fraction_dissolved(balance: SimpleNamespace) -> float:
    """Share of the fed crystal mass that dissolves, at the final time."""
    return balance.crystal["dissolved"] / balance.crystal["inlet"]


def balance_rows(balance: SimpleNamespace) -> list[list]:
    """[component, cake in, H in, dissolved, out, accumulated] [kg/s]."""
    names = species_names()
    phases = balance.feed_phases
    cake_liquid = balance.feeds[phases.index("liquid")]
    h = balance.feeds[len(phases) - 1 - phases[::-1].index("liquid")]
    j_f = names.index(CRYSTAL_SPECIES)
    rows = []
    for j, name in enumerate(names):
        values = [cake_liquid[j], h[j], balance.to_liquid[j],
                  balance.outlet[j], balance.accumulation[j]]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)
    c = balance.crystal
    rows.append([f"{names[j_f]} (crystals)", c["inlet"], 0.0,
                 -c["dissolved"], c["outlet"], c["accumulation"]])
    rows.append(["Total", cake_liquid.sum() + c["inlet"], h.sum(), 0.0,
                 balance.outlet.sum() + c["outlet"],
                 balance.accumulation.sum() + c["accumulation"]])
    return rows


# ---------------------------------------------------------------- plotting
def _style(axis):
    axis.grid(True, color=GRID_COLOR, lw=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_profiles(cake, profiles):
    """Crystals held, F in the liquid vs solubility, crystal flows."""
    import matplotlib.pyplot as plt

    t = profiles.time / 60.0
    tau = residence_time(cake) / 60.0
    fed, out = SERIES_COLORS
    fig, (ax_m, ax_c, ax_f) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax_m.plot(t, profiles.crystal_mass * 1e3, color=fed, lw=2)
    ax_m.set_ylabel("F crystals in tank [g]")
    ax_m.set_title("Crystals held in D02H")

    ax_c.plot(t, profiles.conc, color=fed, lw=2, label="$c_F$ liquid")
    ax_c.plot(t, profiles.solubility, color=out, lw=2, ls="--",
              label="$c_{sat}$")
    ax_c.set_ylabel("F [mol/L]")
    ax_c.set_title("Dissolved F vs solubility")
    ax_c.legend(frameon=False)

    ax_f.plot(t, profiles.crystal_in * 1e3, color=fed, lw=2,
              label="fed (cake)")
    ax_f.plot(t, profiles.crystal_out * 1e3, color=out, lw=2,
              label="leaving")
    ax_f.set_ylabel("F crystals [g/s]")
    ax_f.set_title(f"Crystal flow (final: "
                   f"{profiles.fraction_dissolved[-1]:.1%} dissolved)",
                   fontsize=11)
    ax_f.legend(frameon=False)

    for axis in (ax_m, ax_c, ax_f):
        axis.set_xlabel("Time [min]")
        _style(axis)

    fig.suptitle(f"D02H - H wash of the F01H cake (W02H)  (residence time "
                 f"{tau:.1f} min, run {t[-1] / tau:.0f} residence times)")
    fig.tight_layout()
    return fig


def plot_csd(tank, cake_flow):
    """Mass-weighted CSD of the cake crystals and of those leaving."""
    import matplotlib.pyplot as plt

    pbm = population_balance(tank)
    x, dx = np.asarray(pbm.x_grid), pbm.dx
    distrib = np.asarray(tank.result.distrib_solid0)[-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    for number, label, color, style in (
            (cake_flow.crystal_flow, "fed (F01H cake)", SERIES_COLORS[0],
             "-"),
            (distrib, "leaving D02H", SERIES_COLORS[1], "--")):
        mass = np.asarray(number) * x**3 * dx
        w = mass / mass.sum() / dx
        mean = np.sum(w * x * dx)
        ax.plot(x, w * 100, color=color, lw=2, ls=style,
                label=f"{label}  (mean {mean:.0f} um)")
    fed_mass = np.asarray(cake_flow.crystal_flow) * x**3
    occupied = x[fed_mass > 1e-4 * fed_mass.max()]
    ax.set_xlim(0, min(x[-1], 1.5 * occupied.max()))
    ax.set_xlabel("Crystal size [um]")
    ax.set_ylabel("Mass fraction [% per um]")
    ax.set_title("F crystal size distribution by mass, end of run")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def settings_rows(cake, cake_flow, profiles, balance):
    names = species_names()
    liquid = cake.Phases[0]
    tau = residence_time(cake)
    gap = steady_state_gap(balance)
    solutes = ", ".join(
        f"{n} {w:.3g}" for n, w in zip(names, liquid.mass_frac) if w > 1e-9)

    def kinetics(params):
        return ", ".join(f"{p:g}" for p in params)

    return [
        ("Tank", "Charge", f"{INITIAL_MASS:g} kg of {SOLVENT_NAME}"),
        ("Tank", "Temperature",
         f"{dissolver_temperature(cake):.2f} K (isothermal, cake T)"),
        ("Tank", "Residence time",
         f"{tau:.1f} s ({tau / 60:.2f} min) = charge / feed"),
        ("Feeds", "F01H cake",
         f"{cake_flow.solid_mass_flow * 1e3:.4g} g/s crystals + "
         f"{cake_flow.liquid_mass_flow * 1e3:.4g} g/s liquid at "
         f"{cake_flow.temp:.4g} K"),
        ("Feeds", "Cake liquid (w)", solutes),
        ("Feeds", f"Fresh {ANTISOLVENT_NAME}",
         f"{H_FLOW * 1e3:g} g/s at {H_TEMPERATURE:g} K"),
        ("Solubility", "Model",
         f"S0 exp(-{ANTISOLVENT_K_H:g} [{ANTISOLVENT_NAME}]) mol/L"),
        ("Solubility", "S0", f"{solubility_s0(cake):.4g} mol/L "
                             "(cake liquid [F], saturated)"),
        ("Kinetics F", "Nucl. / growth / dissol.",
         f"({kinetics(NUCL_PRIM)}) / ({kinetics(GROWTH)}) / "
         f"({kinetics(DISSOLUTION)})"),
        ("Run", "Runtime", f"{profiles.time[-1]:.0f} s "
                           f"({NUM_RESIDENCE_TIMES:g} residence times)"),
        ("Run", "Steady state",
         f"accumulation {gap:.2%} of feed "
         f"({'reached' if gap < STEADY_STATE_TOL else 'NOT reached'})"),
        ("Run", "Crystals dissolved",
         f"{fraction_dissolved(balance):.2%} of the cake crystals"),
    ]


def _table_figure(columns, rows, title, col_widths=None, align="left",
                  width=9):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, 0.32 * len(rows) + 0.9))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, colWidths=col_widths,
                     cellLoc=align, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if col == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0efec")
    ax.set_title(title, pad=12)
    fig.tight_layout()
    return fig


def plot_settings(cake, cake_flow, profiles, balance):
    rows, previous = [], None
    for group, setting, value in settings_rows(cake, cake_flow, profiles,
                                               balance):
        rows.append([group if group != previous else "", setting, value])
        previous = group
    return _table_figure(["Group", "Setting", "Value"], rows,
                         "D02H settings and initial conditions",
                         col_widths=[0.16, 0.24, 0.60], width=10)


def plot_mass_balance(balance):
    def cell(value):
        return "0" if abs(value) < 5e-10 else f"{value * 1e3:.5f}"

    rows = [[name] + [cell(v) for v in values]
            for name, *values in balance_rows(balance)]
    return _table_figure(
        ["Component", "Cake in [g/s]", f"{ANTISOLVENT_NAME} in [g/s]",
         "Dissolved [g/s]", "Out [g/s]", "Accumulated [g/s]"], rows,
        f"D02H mass balance at t = {balance.time:.0f} s\n'Dissolved' leaves "
        "the crystal row and enters the dissolved F row; crystal 'Cake in' "
        "is PharmaPy's feed term",
        align="right", width=11)


def save_figures(cake, cake_flow, tank, out_dir: Path = RESULTS_DIR,
                 balance: SimpleNamespace | None = None):
    profiles = history(tank, cake_flow)
    if balance is None:
        balance = final_balance(tank)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "D02H_profiles": plot_profiles(cake, profiles),
        "D02H_csd": plot_csd(tank, cake_flow),
        "D02H_settings": plot_settings(cake, cake_flow, profiles, balance),
        "D02H_mass_balance": plot_mass_balance(balance),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


# ---------------------------------------------------------------- report
def print_report(cake, cake_flow, tank, balance: SimpleNamespace):
    profiles = history(tank, cake_flow)
    tau = residence_time(cake)
    print(f"D02H: {INITIAL_MASS:g} kg {SOLVENT_NAME} charge, "
          f"tau = {tau:.1f} s, run {profiles.time[-1]:.0f} s\n")
    print(f"  F liquid {profiles.conc[-1]:.5f} mol/L, c_sat "
          f"{profiles.solubility[-1]:.5f} mol/L, crystals held "
          f"{profiles.crystal_mass[-1] * 1e3:.3f} g")
    print(f"  Crystals fed {cake_flow.solid_mass_flow * 1e3:.5f} g/s, "
          f"dissolved {balance.crystal['dissolved'] * 1e3:.5f} g/s "
          f"({fraction_dissolved(balance):.2%})")

    print(f"\n  {'component':<16}{'cake in':>11}{'H in':>11}"
          f"{'dissolved':>11}{'out':>11}{'accum.':>11}   [g/s]")
    for name, *values in balance_rows(balance):
        print(f"  {name:<16}" + "".join(f"{v * 1e3:>11.5f}" for v in values))

    gap = steady_state_gap(balance)
    if gap >= STEADY_STATE_TOL:
        print(f"\n  WARNING: accumulation is {gap:.2%} of the feed; "
              "increase NUM_RESIDENCE_TIMES.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    d02h_cake, d02h_cake_flow, d02h = run_flowsheet()
    d02h_balance = final_balance(d02h)
    print_report(d02h_cake, d02h_cake_flow, d02h, d02h_balance)
    save_figures(d02h_cake, d02h_cake_flow, d02h, balance=d02h_balance)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
