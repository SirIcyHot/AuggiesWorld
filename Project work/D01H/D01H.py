"""D01H: continuous dissolver on the washed (F02H) filter cake.

The flowsheet is::

                         +----------- W02H -----------+
    R01H --> F01H -cake->| D02H --> F02H -------cake--+--+
              |          |  ^         |               |  |
              |          +--|---------|---------------+  +--> D01H --> slurry
              |            FEED_H     +--> wash filtrate |
              +--> filtrate                   FEED_H ----+

``D01H`` is a ``PharmaPy.Crystallizers_Refactored.ContinuousCrystallizer``
charged with H and fed two streams: F02H's cake (``f02h_cake`` solves
R01H -> F01H -> D02H -> F02H and takes the cake MixedStream, F crystals plus
the mostly-H liquor the washed cake holds) and pure H. D02H, the wash ahead
of it, is the same vessel (it calls this module's ``build_dissolver``). The
F crystals live in one solid
phase with a population balance on F01H's size grid, with nucleation,
growth and dissolution kinetics. The R01H reactions are not carried into
the tank: D01H only dissolves (or crystallizes) F.

The cake is fed through ``continuous_filter.stream_from_flow``, the form
whose crystal density a vessel reads back as exactly the crystal flow the
filter sent (see ``CustomUnitOps/test_continuous_filter.py``). Its crystals
are first re-binned from F01H's 20 um grid onto D01H's 1 um grid, keeping
the crystal mass flow exactly. Dissolving a few percent only shrinks each
crystal by about 1 um. On the 20 um grid the crystals lost undercount the F
the liquid gains by ~10 %; on the 1 um grid the two agree to ~0.2 %.

Solubility of F (mol/L), with H as antisolvent::

    c_sat = S0 * exp(-K_H * [H])

S0 is the F solubility in H-free liquid, taken as F01H's cake liquid's own F
concentration: F01H leaves its B-based mother liquor saturated. The washed
F02H cake holds mostly H, so its [F] is not S0; ``run_flowsheet`` passes
D02H's S0 (F01H's cake) instead, and ``build_dissolver`` reads the cake it
is given only when no S0 is passed. Adding H dilutes
the cake liquid, so some F dissolves to re-saturate it, but H also lowers
the solubility. K_H is set so the two nearly cancel at D01H's steady [H]
(~14 mol/L): only a few percent of the fed crystals dissolve. Lower K_H or
raise H_FLOW to dissolve more.

The pure-H charge holds no F, so the first crystals to arrive dissolve
completely until the liquid saturates.

Crystallization rates (PharmaPy's CrystKinetics), relative supersaturation
sigma = (c - c_sat) / c_sat::

    nucleation   B = k_b * exp(-E_b / (R T)) * sigma**b      [#/(s m**3)]
    growth       G = k_g * exp(-E_g / (R T)) * sigma**g      [um/s]
    dissolution  D = k_d * exp(-E_d / (R T)) * sigma*|sigma|**(d-1) [um/s]

Growth and nucleation act when sigma > 0, dissolution when sigma < 0.

The residence time comes from the solvent charge::

    tau = INITIAL_MASS / (cake mass flow + H_FLOW)

and the run lasts ``NUM_RESIDENCE_TIMES`` of them, enough to reach steady
state.

Run it with::

    python D01H.py                # solve, save figures, show them
    python D01H.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    D01H_profiles.png      crystals held, saturation, crystal flow in vs out
    D01H_csd.png           F crystal size distribution, fed vs leaving
    D01H_settings.png      the settings and initial conditions used
    D01H_mass_balance.png  inlet vs outlet species balance at the final time
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
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import (
    OneDFVMMechanism,
    PopulationBalanceMechanism,
)
from PharmaPy.MixedPhases_Refactored import MixedStream
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Streams_Refactored import LiquidStream

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/D01H
F01H_DIR = HERE.parent / "F01H"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(F01H_DIR))

import F01H  # noqa: E402  (puts R01H and CustomUnitOps on the path)
import R01H  # noqa: E402
from continuous_filter import SlurryFlow, stream_from_flow  # noqa: E402

DB_PATH = F01H.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = "F"
SOLVENT_NAME = "H"  # the tank charge (the cake liquid's solvent is B)
ANTISOLVENT_NAME = "H"

# Fresh H feed, pure H.
H_FLOW = 0.5e-3  # kg/s
H_TEMPERATURE = 25 + 273.15  # K

# Tank charge: pure H. Its mass sets the residence time.
INITIAL_MASS = 1.0  # kg
INITIAL_COMPOSITION = {}  # solute mass fractions; solvent is the remainder
# None: held at the cake temperature (isothermal tank).
DISSOLVER_TEMPERATURE = None  # K
DISSOLVER_DIAMETER = 0.1  # m

# Solubility c_sat = S0 exp(-K_H [H]) [mol/L], S0 = the cake liquid's F
# concentration (saturated by F01H). L/mol: each 1 mol/L of H multiplies
# the solubility by exp(-K_H) = 0.932; at the ~14 mol/L of H the tank
# settles at, exp(-K_H * 14) = 0.37.
ANTISOLVENT_K_H = 0.07

# Kinetics (k, E_a [J/mol], order). Nucleation and growth are CR01H's.
# Units: nucleation #/(s m**3 slurry), growth and dissolution um/s.
NUCL_PRIM = F01H.CR01H.NUCL_PRIM
NUCL_SEC = (0.0, 0.0, 0.0, 0.0)
GROWTH = F01H.CR01H.GROWTH
# k_d is the shrink rate [um/s] at sigma = -1 (crystals in F-free liquid).
# 1 um/s keeps the steady liquid within 0.1 % of saturation. D01R's 5e5
# pins sigma near 1e-9, where the solver chatters across saturation and
# the balance replay can no longer resolve the dissolution rate.
DISSOLUTION = (1.0, 0.0, 1.0)

# um. The cake is re-binned onto this grid (see the module docstring). The
# cake crystals (80 +/- 20 um) end well below 401 um, and nothing grows.
CRYSTAL_SIZE_GRID = np.arange(1.0, 402.0, 1.0)
CRYSTAL_KV = F01H.CRYSTAL_KV

NUM_RESIDENCE_TIMES = 10.0
# Accumulation / feed mass flow per liquid species, and accumulation /
# crystal feed for the crystals.
STEADY_STATE_TOL = 1.0e-2

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

    Returns (cake MixedStream, cake SlurryFlow), both on D01H's grid. D02H,
    the wash, is fed this cake.
    """
    *_, filt = F01H.run_flowsheet(DB_PATH)
    cake_flow = regrid(filt.cake_flow)
    return stream_from_flow(cake_flow, include_solid=True), cake_flow


def f02h_cake():
    """Solve R01H -> F01H -> D02H -> F02H: the washed cake D01H is fed.

    Returns (cake MixedStream, cake SlurryFlow, S0), the cake on D01H's grid
    and S0 the F solubility in H-free liquid that D02H used (F01H's cake
    liquid [F]); the washed cake's own [F] is not S0.
    """
    sys.path.insert(0, str(HERE.parent / "F02H"))
    import F02H  # here, not at the top: F02H imports D02H, which imports D01H

    d02h_cake, _, _, filt = F02H.run_flowsheet()
    cake_flow = regrid(filt.cake_flow)
    return (stream_from_flow(cake_flow, include_solid=True), cake_flow,
            solubility_s0(d02h_cake))


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


def make_solubility_function(names, s0: float):
    """c_sat of F [mol/L] from the liquid's H content."""
    index_h = names.index(ANTISOLVENT_NAME)

    def solubility(temp, conc):
        conc = np.asarray(conc)
        return s0 * np.exp(-ANTISOLVENT_K_H * conc[..., index_h])

    return solubility


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
    """D01H: a ContinuousCrystallizer charged with H, one F solid.

    The liquid <-> solid connections come from ContinuousCrystallizer's
    ``CrystKinetics`` setter: a forward (nucleation and growth) and a
    reverse (dissolution) transfer, both driven by the F population balance.
    ``s0`` [mol/L] is the F solubility in H-free liquid; None reads it from
    ``cake`` (``solubility_s0``), right only for an unwashed cake.
    """
    if s0 is None:
        s0 = solubility_s0(cake)
    names = species_names()
    temp = dissolver_temperature(cake)

    tank = ContinuousCrystallizer(integrator=ScipyBackend(), h_conv=0,
                                  diam=DISSOLVER_DIAMETER, isothermal=True)

    liquid = LiquidPhase(
        str(DB_PATH), temp=temp, mass=INITIAL_MASS,
        mass_frac=R01H.named_mass_fraction(INITIAL_COMPOSITION, names,
                                           SOLVENT_NAME),
        name_solv=SOLVENT_NAME)

    mass_frac = np.zeros(len(names))
    mass_frac[names.index(CRYSTAL_SPECIES)] = 1.0
    solid = SolidPhase(str(DB_PATH), temp=temp, mass=0.0,
                       mass_frac=mass_frac, name_solv=SOLVENT_NAME)
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components=CRYSTAL_SPECIES, solvent_name=SOLVENT_NAME,
        x_grid=CRYSTAL_SIZE_GRID,
        distrib_init=np.zeros(len(CRYSTAL_SIZE_GRID)), kv=CRYSTAL_KV)
    tank.Phases = [liquid, solid]

    # Kept on the tank so the results can evaluate c_sat after the solve.
    tank.solubility = make_solubility_function(names, s0)
    tank.solubility_s0 = s0
    kinetics = CrystKinetics(
        solub_fn=tank.solubility,
        nucl_prim=NUCL_PRIM, nucl_sec=NUCL_SEC, growth=GROWTH,
        dissolution=DISSOLUTION,
        solubility_basis="mole_per_volume_solution")
    tank.CrystKinetics = kinetics
    tank.Inlet = [cake, h_stream(names)]
    return tank


def run_flowsheet(runtime: float | None = None):
    """R01H -> F01H -> W02H -> D01H. Returns (cake, cake flow, dissolver)."""
    cake, cake_flow, s0 = f02h_cake()
    tank = build_dissolver(cake, s0)

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
        s0=float(tank.solubility_s0),
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
    ax_m.set_title("Crystals held in D01H")

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

    fig.suptitle(f"D01H - H dissolver on the washed F02H cake  (residence time "
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
            (cake_flow.crystal_flow, "fed (F02H cake)", SERIES_COLORS[0],
             "-"),
            (distrib, "leaving D01H", SERIES_COLORS[1], "--")):
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
        ("Feeds", "F02H cake",
         f"{cake_flow.solid_mass_flow * 1e3:.4g} g/s crystals + "
         f"{cake_flow.liquid_mass_flow * 1e3:.4g} g/s liquid at "
         f"{cake_flow.temp:.4g} K"),
        ("Feeds", "Cake liquid (w)", solutes),
        ("Feeds", f"Fresh {ANTISOLVENT_NAME}",
         f"{H_FLOW * 1e3:g} g/s at {H_TEMPERATURE:g} K"),
        ("Solubility", "Model",
         f"S0 exp(-{ANTISOLVENT_K_H:g} [{ANTISOLVENT_NAME}]) mol/L"),
        ("Solubility", "S0", f"{profiles.s0:.4g} mol/L "
                             "(F01H cake liquid [F], saturated, as D02H)"),
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
                         "D01H settings and initial conditions",
                         col_widths=[0.16, 0.24, 0.60], width=10)


def plot_mass_balance(balance):
    def cell(value):
        return "0" if abs(value) < 5e-10 else f"{value * 1e3:.5f}"

    rows = [[name] + [cell(v) for v in values]
            for name, *values in balance_rows(balance)]
    return _table_figure(
        ["Component", "Cake in [g/s]", f"{ANTISOLVENT_NAME} in [g/s]",
         "Dissolved [g/s]", "Out [g/s]", "Accumulated [g/s]"], rows,
        f"D01H mass balance at t = {balance.time:.0f} s\n'Dissolved' leaves "
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
        "D01H_profiles": plot_profiles(cake, profiles),
        "D01H_csd": plot_csd(tank, cake_flow),
        "D01H_settings": plot_settings(cake, cake_flow, profiles, balance),
        "D01H_mass_balance": plot_mass_balance(balance),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


# ---------------------------------------------------------------- report
def print_report(cake, cake_flow, tank, balance: SimpleNamespace):
    profiles = history(tank, cake_flow)
    tau = residence_time(cake)
    print(f"D01H: {INITIAL_MASS:g} kg {SOLVENT_NAME} charge, "
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

    d01h_cake, d01h_cake_flow, d01h = run_flowsheet()
    d01h_balance = final_balance(d01h)
    print_report(d01h_cake, d01h_cake_flow, d01h, d01h_balance)
    save_figures(d01h_cake, d01h_cake_flow, d01h, balance=d01h_balance)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
