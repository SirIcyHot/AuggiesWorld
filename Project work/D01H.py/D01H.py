"""D01H: dynamic continuous dissolver on the F01H filter cake.

The flowsheet is::

    FEED_A --+
             +--> R01 --> F01 --(cake)--+
    FEED_C --+                          +--> D01 --> slurry
                         SOLVENT -------+

``D01`` is a ``PharmaPy.Crystallizers_Refactored.ContinuousCrystallizer``
run as a dissolver: a stirred tank charged with fresh solvent, fed the
F01H cake (F crystals plus the mother liquor held in the cake) and more
solvent, and solved in time. The crystals dissolve at the rate set by the
dissolution kinetics below, until the liquid is saturated (CR01H's F
solubility) or the crystals are gone, while the product is withdrawn
continuously. Nucleation and growth are off by default, so the tank only
dissolves.

The residence time comes from the solvent charge::

    tau = INITIAL_MASS / (cake mass flow + solvent mass flow)

and the run lasts ``NUM_RESIDENCE_TIMES`` of them, enough to reach steady
state.

Dissolution rate (PharmaPy's CrystKinetics), in um/s, negative = shrinking::

    D = k_D * exp(-E_a / (R T)) * sigma * |sigma|**(order - 1)
    sigma = (c_F - S) / S        relative supersaturation, < 0 here

k_D is the pre-exponential factor: with E_a = 0 it is the rate at sigma = -1
(pure solvent), whatever the temperature.

Needs the PharmaPy fix for crystal feeds into a crystallizer
(``PopulationBalanceMechanism.get_inlet_contributions``); without it the
cake's crystals never enter the tank.

Run it with::

    python D01H.py                # solve, save figures, show them
    python D01H.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    D01_profiles.png      crystals held, [F] vs solubility, fraction dissolved
    D01_csd.png           crystal size distribution: cake vs tank at the end
    D01_settings.png      the settings and initial conditions used
    D01_mass_balance.png  inlet vs outlet species balance at the final time
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Streams_Refactored import LiquidStream

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/D01H.py
ROOT = HERE.parents[1]                          # AuggiesWorld

F01H_DIR = ROOT / "Project work" / "F01H.py"
UNIT_OPS_DIR = ROOT / "CustomUnitOps"
RESULTS_DIR = HERE / "results"

sys.path[:0] = [str(UNIT_OPS_DIR), str(F01H_DIR)]

from continuous_filter import flow_from_vessel  # noqa: E402
import F01H  # noqa: E402

CR01H = F01H.CR01H
DB_PATH = F01H.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = F01H.CRYSTAL_SPECIES  # F
SOLVENT_NAME = "B"

# Fresh solvent feed.
SOLVENT_FLOW = 1.0e-3  # kg/s
SOLVENT_TEMPERATURE = 20 + 273.15  # K

# Tank charge: pure solvent. Its mass sets the residence time.
INITIAL_MASS = 0.2  # kg
INITIAL_COMPOSITION = {}  # solute mass fractions; solvent is the remainder
DISSOLVER_TEMPERATURE = 20 + 273.15  # K, held constant (isothermal tank)
DISSOLVER_DIAMETER = 0.2  # m

# Dissolution kinetics: (k_D [um/s], E_a [J/mol], order [-]).
DISSOLUTION = (5.0, 0.0, 1.0)

# Off: the tank only dissolves. Use CR01H.NUCL_PRIM / CR01H.GROWTH to let
# crystals also nucleate and grow if the liquid becomes supersaturated.
NUCL_PRIM = (0.0, 0.0, 1.0)
NUCL_SEC = (0.0, 0.0, 0.0, 0.0)
GROWTH = (0.0, 0.0, 1.0)

CRYSTAL_SIZE_GRID = F01H.CRYSTAL_SIZE_GRID  # um, must match the cake's
CRYSTAL_KV = F01H.CRYSTAL_KV

NUM_RESIDENCE_TIMES = 8.0
STEADY_STATE_TOL = 1.0e-2  # |d(mass in tank)/dt| / feed mass flow

# Categorical slots 1-4 (blue, orange, aqua, yellow), as in R01H and CR01H.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID_COLOR = "#e5e4e0"


# ---------------------------------------------------------------- builders
def solvent_stream(names) -> LiquidStream:
    """Fresh solvent feed."""
    return LiquidStream(
        str(DB_PATH), mass_flow=SOLVENT_FLOW, temp=SOLVENT_TEMPERATURE,
        mass_frac=CR01H.named_mass_fraction({}, names, SOLVENT_NAME),
        name_solv=SOLVENT_NAME, verbose=False)


def feed_mass_flow(filt) -> float:
    """Total mass flow into the tank: cake plus solvent [kg/s]."""
    cake = filt.cake_flow
    return cake.liquid_mass_flow + cake.solid_mass_flow + SOLVENT_FLOW


def residence_time(filt) -> float:
    """Nominal residence time [s] from the solvent charge."""
    return INITIAL_MASS / feed_mass_flow(filt)


def build_dissolver(filt) -> ContinuousCrystallizer:
    """D01: a ContinuousCrystallizer charged with solvent and fed the cake."""
    names = tuple(filt.feed.name_species)

    tank = ContinuousCrystallizer(integrator=ScipyBackend(), h_conv=0,
                                  diam=DISSOLVER_DIAMETER, isothermal=True)

    liquid = LiquidPhase(
        str(DB_PATH), temp=DISSOLVER_TEMPERATURE, mass=INITIAL_MASS,
        mass_frac=CR01H.named_mass_fraction(INITIAL_COMPOSITION, names,
                                            SOLVENT_NAME),
        name_solv=SOLVENT_NAME)

    solid_mass_frac = np.zeros(len(names))
    solid_mass_frac[names.index(CRYSTAL_SPECIES)] = 1.0
    solid = SolidPhase(str(DB_PATH), temp=DISSOLVER_TEMPERATURE, mass=0.0,
                       mass_frac=solid_mass_frac, name_solv=SOLVENT_NAME)
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components=CRYSTAL_SPECIES, solvent_name=SOLVENT_NAME,
        x_grid=CRYSTAL_SIZE_GRID, distrib_init=np.zeros(len(CRYSTAL_SIZE_GRID)),
        kv=CRYSTAL_KV)

    tank.Phases = [liquid, solid]
    tank.CrystKinetics = CrystKinetics(
        solub_fn=CR01H.make_solubility_function(names),
        nucl_prim=NUCL_PRIM, nucl_sec=NUCL_SEC, growth=GROWTH,
        dissolution=DISSOLUTION,
        solubility_basis="mole_per_volume_solution")
    tank.Inlet = [filt.Outlet["cake"], solvent_stream(names)]
    return tank


def run_flowsheet(runtime: float | None = None):
    """R01 -> F01 -> D01. Returns (filter, dissolver)."""
    _, _, _, filt = F01H.run_flowsheet(DB_PATH)
    tank = build_dissolver(filt)

    if runtime is None:
        runtime = NUM_RESIDENCE_TIMES * residence_time(filt)

    tank.solve_unit(runtime=runtime, verbose=False)
    return filt, tank


# ---------------------------------------------------------------- results
def history(filt, tank):
    """Time profiles read from the solved tank, as a dict of arrays.

    time [s]; crystals_held [kg]; conc_f and solubility [mol/L];
    crystals_out [kg/s]; fraction_dissolved [-] of the cake's crystal feed.
    """
    res = tank.result
    time = np.asarray(res.time)
    liquid_mass = np.asarray(res.mass_j_liquid0).sum(axis=1)
    total = np.ravel(res.Total_m_in_vessel)
    names = list(filt.feed.name_species)

    outlets = [flow_from_vessel(tank, k) for k in range(len(time))]
    crystals_out = np.array([o.solid_mass_flow for o in outlets])
    crystals_in = filt.cake_flow.solid_mass_flow

    return {
        "time": time,
        "crystals_held": total - liquid_mass,
        "conc_f": np.asarray(res.mole_conc_liquid0)[:, names.index(
            CRYSTAL_SPECIES)],
        "solubility": np.ravel(np.asarray(res.solubility))[:len(time)],
        "crystals_out": crystals_out,
        "fraction_dissolved": 1.0 - crystals_out / crystals_in,
        "total_mass": total,
    }


def steady_state_gap(filt, profiles) -> float:
    """|d(mass in tank)/dt| at the end, relative to the feed mass flow."""
    t, m = profiles["time"], profiles["total_mass"]
    rate = (m[-1] - m[-2]) / (t[-1] - t[-2])
    return abs(rate) / feed_mass_flow(filt)


def balance_rows(filt, tank) -> list[list]:
    """[component, cake, solvent, outlet] rows at the final time, kg/s."""
    cake = filt.cake_flow
    out = flow_from_vessel(tank)
    names = list(cake.name_species)
    solvent = SOLVENT_FLOW * CR01H.named_mass_fraction({}, tuple(names),
                                                       SOLVENT_NAME)
    rows = []
    for j, name in enumerate(names):
        values = [cake.liquid_mass_j[j], solvent[j], out.liquid_mass_j[j]]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)
    rows.append([f"{CRYSTAL_SPECIES} (crystals)", cake.solid_mass_flow, 0.0,
                 out.solid_mass_flow])
    rows.append(["Total", cake.liquid_mass_flow + cake.solid_mass_flow,
                 SOLVENT_FLOW, out.liquid_mass_flow + out.solid_mass_flow])
    return rows


# ---------------------------------------------------------------- plotting
def _style(axis):
    axis.grid(True, color=GRID_COLOR, lw=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_profiles(filt, profiles):
    """Crystals held, [F] vs solubility, and fraction dissolved vs time."""
    import matplotlib.pyplot as plt

    t = profiles["time"] / 60.0
    tau = residence_time(filt) / 60.0
    fig, (ax_m, ax_c, ax_f) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax_m.plot(t, profiles["crystals_held"] * 1e3, color=SERIES_COLORS[0],
              lw=2)
    ax_m.set_ylabel("Crystals in tank [g]")
    ax_m.set_title("F crystals held in D01")

    ax_c.plot(t, profiles["conc_f"], color=SERIES_COLORS[0], lw=2)
    ax_c.plot(t, profiles["solubility"], color=SERIES_COLORS[1], lw=2,
              ls="--")
    ax_c.annotate("[F] liquid", (t[-1], profiles["conc_f"][-1]),
                  xytext=(-4, -12), textcoords="offset points", ha="right",
                  color=TEXT_PRIMARY, fontsize=9)
    ax_c.annotate("F solubility", (t[-1], profiles["solubility"][-1]),
                  xytext=(-4, 6), textcoords="offset points", ha="right",
                  color=TEXT_PRIMARY, fontsize=9)
    ax_c.legend(["[F] liquid", "F solubility"], frameon=False,
                loc="lower right", bbox_to_anchor=(1.0, 0.12))
    ax_c.set_ylabel("Concentration [mol/L]")
    ax_c.set_title("Approach to saturation")

    ax_f.plot(t, profiles["fraction_dissolved"] * 100, color=SERIES_COLORS[0],
              lw=2)
    ax_f.set_ylim(0, 105)
    ax_f.set_ylabel("Feed crystals dissolved [%]")
    ax_f.set_title(f"Fraction dissolved (final "
                   f"{profiles['fraction_dissolved'][-1]:.1%})")

    for axis in (ax_m, ax_c, ax_f):
        axis.set_xlabel("Time [min]")
        _style(axis)

    fig.suptitle(f"D01 - dynamic dissolver  (residence time {tau * 60:.0f} s, "
                 f"run {t[-1] / tau:.0f} residence times)")
    fig.tight_layout()
    return fig


def mass_distribution(number_density, x_grid, dx):
    """Mass fraction per um of size, from a number density."""
    mass = number_density * x_grid**3 * dx
    total = mass.sum()
    return mass / total / dx if total > 0 else mass


def plot_csd(filt, tank):
    """Mass-weighted CSD of the cake fed in and of the tank at the end."""
    import matplotlib.pyplot as plt

    cake = filt.cake_flow
    x, dx = cake.x_grid, cake.dx
    final = np.asarray(tank.result.distrib_solid0)[-1]

    w_cake = mass_distribution(cake.crystal_flow, x, dx)
    w_tank = mass_distribution(final, x, dx)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, w_cake * 100, color=SERIES_COLORS[0], lw=2,
            label="Cake fed to D01")
    ax.plot(x, w_tank * 100, color=SERIES_COLORS[1], lw=2,
            label="Tank / outlet at the end")
    # The tank's mean sits below the cake's, so its label goes on the left
    # of its line and the cake's on the right, where they cannot overlap.
    top = ax.get_ylim()[1]
    for w, color, side in ((w_cake, SERIES_COLORS[0], "left"),
                           (w_tank, SERIES_COLORS[1], "right")):
        mean = np.sum(w * x * dx)
        ax.axvline(mean, color=color, lw=1, ls=":")
        ax.annotate(f"mean {mean:.0f} um", (mean, top * 0.97),
                    xytext=(5 if side == "left" else -5, 0),
                    textcoords="offset points", ha=side, va="top",
                    color=TEXT_MUTED, fontsize=9)
    ax.set_xlabel("Crystal size [um]")
    ax.set_ylabel("Mass fraction [% per um]")
    ax.set_title("Crystal size distribution by mass")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def settings_rows(filt, tank, profiles) -> list[tuple[str, str, str]]:
    tau = residence_time(filt)
    k_d, e_a, order = DISSOLUTION
    gap = steady_state_gap(filt, profiles)
    return [
        ("Tank", "Charge", f"{INITIAL_MASS:g} kg of {SOLVENT_NAME}"),
        ("Tank", "Temperature", f"{DISSOLVER_TEMPERATURE:g} K (isothermal)"),
        ("Tank", "Residence time",
         f"{tau:.1f} s ({tau / 60:.2f} min) = charge / feed"),
        ("Feeds", "Cake (from F01H)",
         f"{filt.cake_flow.solid_mass_flow * 1e3:.4f} g/s crystals + "
         f"{filt.cake_flow.liquid_mass_flow * 1e3:.4f} g/s liquid"),
        ("Feeds", "Solvent", f"{SOLVENT_FLOW * 1e3:g} g/s of {SOLVENT_NAME} at "
                             f"{SOLVENT_TEMPERATURE:g} K"),
        ("Kinetics", "Dissolution",
         f"k_D {k_d:g} um/s, E_a {e_a:g} J/mol, order {order:g}"),
        ("Kinetics", "Nucleation / growth",
         "off" if NUCL_PRIM[0] == 0 and GROWTH[0] == 0 else
         f"nucl {NUCL_PRIM}, growth {GROWTH}"),
        ("Kinetics", "Solubility", "CR01H (temperature and [A])"),
        ("Run", "Runtime", f"{profiles['time'][-1]:.0f} s "
                           f"({NUM_RESIDENCE_TIMES:g} residence times)"),
        ("Run", "Steady state",
         f"d(mass)/dt = {gap:.2%} of feed "
         f"({'reached' if gap < STEADY_STATE_TOL else 'NOT reached'})"),
        ("Run", "Final fraction dissolved",
         f"{profiles['fraction_dissolved'][-1]:.1%}"),
    ]


def _table_figure(columns, rows, title, col_widths=None, align="left",
                  figsize=(9, None)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(figsize[0], 0.32 * len(rows) + 0.9))
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


def plot_settings(filt, tank, profiles):
    rows, previous = [], None
    for group, setting, value in settings_rows(filt, tank, profiles):
        rows.append([group if group != previous else "", setting, value])
        previous = group
    return _table_figure(["Group", "Setting", "Value"], rows,
                         "D01 settings and initial conditions",
                         col_widths=[0.16, 0.26, 0.58])


def plot_mass_balance(filt, tank):
    rows = [[name] + [f"{v * 1e3:.5f}" for v in values]
            for name, *values in balance_rows(filt, tank)]
    fig = _table_figure(["Component", "Cake [g/s]", "Solvent [g/s]",
                         "D01 out [g/s]"], rows,
                        "D01 mass balance at the final time", align="right")
    return fig


def save_figures(filt, tank, out_dir: Path = RESULTS_DIR):
    profiles = history(filt, tank)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "D01_profiles": plot_profiles(filt, profiles),
        "D01_csd": plot_csd(filt, tank),
        "D01_settings": plot_settings(filt, tank, profiles),
        "D01_mass_balance": plot_mass_balance(filt, tank),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


# ---------------------------------------------------------------- report
def print_report(filt, tank):
    profiles = history(filt, tank)
    tau = residence_time(filt)
    print(f"D01: {INITIAL_MASS:g} kg {SOLVENT_NAME} charge, tau = {tau:.1f} s, "
          f"run {profiles['time'][-1]:.0f} s")
    print(f"Final: {profiles['fraction_dissolved'][-1]:.1%} of the cake "
          f"crystals dissolved, {profiles['crystals_held'][-1] * 1e3:.3f} g "
          f"held, [F] {profiles['conc_f'][-1]:.4f} vs solubility "
          f"{profiles['solubility'][-1]:.4f} mol/L\n")

    print(f"  {'component':<16}{'cake':>12}{'solvent':>12}{'D01 out':>12}"
          "   [g/s]")
    for name, *values in balance_rows(filt, tank):
        print(f"  {name:<16}" + "".join(f"{v * 1e3:>12.5f}" for v in values))

    gap = steady_state_gap(filt, profiles)
    if gap >= STEADY_STATE_TOL:
        print(f"\n  WARNING: mass in the tank is still changing at {gap:.2%} "
              "of the feed; increase NUM_RESIDENCE_TIMES.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    d01_filter, d01 = run_flowsheet()
    print_report(d01_filter, d01)
    save_figures(d01_filter, d01)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
