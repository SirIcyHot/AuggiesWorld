"""D02R: continuous dissolver on the F01R filter cake, with solvent L.

The flowsheet is::

    R02R --> F01R --+-- filtrate
                    |
                    +-- cake --+
                               +--> D02R --> solution (+ a few crystals)
                     FEED_L ---+

``D02R`` is a ``PharmaPy.Crystallizers_Refactored.ContinuousCrystallizer``
charged with L and fed two streams: F01R's cake (``f01r_cake`` solves
R02R -> F01R) and pure L. L is a new compound with DUMMY properties in
``compound_databaseActual.json``; replace them when real data exist.

F01R carries D, E and J crystals as one mixed population. Here each
species gets its own solid phase, population balance and kinetics, as in
D01R, so each dissolves against its own solubility. ``split_cake`` splits
the mixed population by crystal volume: every species keeps the cake's
size distribution shape, and its own crystal mass flow exactly. The cake
enters as one stream (liquid, then the D, E and J solids, in that order);
PharmaPy maps a feed's k-th solid onto the vessel's k-th solid.

Solubility of each species i (mol/L), with L as the dissolving solvent::

    c_sat,i = S0_i * exp(K_L * [L])

S0_i is the solubility in L-free liquid (R02R's H-based liquid); for now
D, E and J share one value, D01R's S0 for J. K_L > 0, so L raises the
solubility. It was set, for an earlier F01R cake, so that at
D02R's steady [L] (~11 mol/L) the solubility sits just below what full
dissolution of the cake crystals would give: D, E and J dissolve almost
completely. Raise K_L or L_FLOW to dissolve the rest.

Crystallization rates (PharmaPy's CrystKinetics), relative supersaturation
sigma_i = (c_i - c_sat,i) / c_sat,i::

    nucleation   B = k_b * exp(-E_b / (R T)) * sigma**b      [#/(s m**3)]
    growth       G = k_g * exp(-E_g / (R T)) * sigma**g      [um/s]
    dissolution  D = k_d * exp(-E_d / (R T)) * sigma*|sigma|**(d-1) [um/s]

Growth and nucleation act when sigma > 0, dissolution when sigma < 0.

The residence time comes from the solvent charge::

    tau = INITIAL_MASS / (cake mass flow + L_FLOW)

and the run lasts ``NUM_RESIDENCE_TIMES`` of them, enough to reach steady
state.

Run it with::

    python D02R.py                # solve, save figures, show them
    python D02R.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    D02R_profiles.png      crystals held, saturation, fraction dissolved
    D02R_csd.png           D, E, J size distributions, fed vs leaving
    D02R_settings.png      the settings and initial conditions used
    D02R_mass_balance.png  inlet vs outlet species balance at the final time
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
from PharmaPy.DataClasses import PhaseConnection, PhaseRef
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import (
    OneDFVMMechanism,
    PopulationBalanceMechanism,
    ReversibleTransferMechanism,
)
from PharmaPy.MixedPhases_Refactored import MixedStream
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Streams_Refactored import LiquidStream

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/D02R.py
F01R_DIR = HERE.parent / "F01R"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(F01R_DIR))

import F01R  # noqa: E402  (puts R02R and CustomUnitOps on the path)
import R02R  # noqa: E402
from continuous_filter import SlurryFlow  # noqa: E402

DB_PATH = F01R.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = F01R.CRYSTAL_SPECIES  # ("D", "E", "J")
SOLVENT_NAME = "L"  # the tank charge (the cake liquid's solvent is H)

# Fresh solvent feed, pure L.
L_FLOW = 2.0e-3  # kg/s
L_TEMPERATURE = 25 + 273.15  # K

# Tank charge: pure L. Its mass sets the residence time.
INITIAL_MASS = 2.5  # kg
INITIAL_COMPOSITION = {}  # solute mass fractions; solvent is the remainder
# None: held at the cake temperature (isothermal tank).
DISSOLVER_TEMPERATURE = None  # K
DISSOLVER_DIAMETER = 0.15  # m

# Solubility c_sat = S0 exp(K_L [L]) [mol/L]. For now D, E and J share one
# S0, D01R's J value (R02R's liquid, no L); D01R's own per-species values
# were D 0.0284, E 0.01505, J 0.0886. L/mol: each 1 mol/L of L multiplies
# every solubility by exp(K_L) = 1.18. Both were tuned for an earlier F01R
# cake (~98.6 % dissolved); retune them for the current one.
SHARED_SOLUBILITY_S0 = 0.0886  # mol/L
SOLUBILITY_S0 = {name: SHARED_SOLUBILITY_S0 for name in CRYSTAL_SPECIES}
SOLVENT_K_L = 0.166

# Kinetics per species, (k, E_a [J/mol], order), D01R's nucleation and
# growth. Units: nucleation #/(s m**3 slurry), growth and dissolution um/s.
NUCL_PRIM = {name: (1.0e10, 0.0, 2.0) for name in CRYSTAL_SPECIES}
NUCL_SEC = {name: (0.0, 0.0, 0.0, 0.0) for name in CRYSTAL_SPECIES}
GROWTH = {name: (10.0, 0.0, 1.0) for name in CRYSTAL_SPECIES}
# k_d is the shrink rate [um/s] at sigma = -1 (crystals in solute-free
# liquid).
DISSOLUTION = {name: (1.0e3, 0.0, 1.0) for name in CRYSTAL_SPECIES}

# The cake's size grid (F01R's, 1-200 um); the tank's solids must match it.
CRYSTAL_SIZE_GRID = F01R.CRYSTAL_SIZE_GRID
CRYSTAL_KV = F01R.CRYSTAL_KV

NUM_RESIDENCE_TIMES = 10.0
# Accumulation / feed mass flow per liquid species, and accumulation /
# crystal feed per crystal species.
STEADY_STATE_TOL = 1.0e-2

# Categorical slots 1-3 (blue, orange, aqua), one per species.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID_COLOR = "#e5e4e0"


# ---------------------------------------------------------------- builders
def species_names() -> tuple[str, ...]:
    return R02R.database_species(DB_PATH)


def split_cake(cake_flow: SlurryFlow) -> dict[str, np.ndarray]:
    """Crystal number flow [#/(um s)] of each species in the mixed cake.

    Each species takes its share of the crystal volume, w_i rho_mix /
    rho_i, at every size: the shares sum to one, so the crystal count is
    kept, and species i carries w_i of the crystal mass.
    """
    names = list(cake_flow.name_species)
    densities = F01R.solid_densities(DB_PATH)
    return {name: (np.asarray(cake_flow.crystal_flow)
                   * cake_flow.solid_mass_frac[names.index(name)]
                   * cake_flow.solid_density / densities[name])
            for name in CRYSTAL_SPECIES}


def crystal_mass_flow(number_flow, density: float) -> float:
    """Crystal mass flow [kg/s] of a number flow [#/(um s)]."""
    grid = np.asarray(CRYSTAL_SIZE_GRID, dtype=float)
    dx = OneDFVMMechanism._cell_widths(grid)
    return float(density * CRYSTAL_KV * 1e-18
                 * np.sum(np.asarray(number_flow) * grid**3 * dx))


def cake_stream(cake_flow: SlurryFlow) -> MixedStream:
    """The cake as one stream: its liquid, then a D, an E and a J solid.

    Follows the convention ``continuous_filter.stream_from_flow`` uses for
    one solid: each solid's crystal density is per m**3 of the whole
    stream, so the vessel, which multiplies it by the stream's total
    volumetric flow, reads back exactly that species' crystal flow.
    """
    names = list(cake_flow.name_species)
    densities = F01R.solid_densities(DB_PATH)
    numbers = split_cake(cake_flow)

    liquid = LiquidStream(
        str(DB_PATH), mass_flow=cake_flow.liquid_mass_flow,
        mass_frac=cake_flow.liquid_mass_j / cake_flow.liquid_mass_flow,
        temp=cake_flow.temp, name_solv=cake_flow.name_solv, verbose=False)
    slurry_vol_flow = float(liquid.vol_flow) + sum(
        crystal_mass_flow(numbers[n], densities[n]) / densities[n]
        for n in CRYSTAL_SPECIES)

    solids = []
    for name in CRYSTAL_SPECIES:
        mass_frac = np.zeros(len(names))
        mass_frac[names.index(name)] = 1.0
        solid = SolidPhase(str(DB_PATH), mass=0, mass_frac=mass_frac,
                           temp=cake_flow.temp, verbose=False)
        mechanism = OneDFVMMechanism(
            solid, target_components=name, solvent_name=cake_flow.name_solv,
            x_grid=CRYSTAL_SIZE_GRID,
            distrib_init=numbers[name] / slurry_vol_flow,
            density=densities[name], kv=CRYSTAL_KV)
        # A stream's "volume" is its slurry flow, so the solid's mass reads
        # as its crystal mass flow.
        mechanism.slurry_volume = slurry_vol_flow
        solid.mechanisms = mechanism
        solids.append(solid.to_stream())

    return MixedStream([liquid] + solids)


def f01r_cake():
    """Solve R02R -> F01R. Returns (cake MixedStream, cake SlurryFlow)."""
    *_, filt = F01R.run_flowsheet(DB_PATH)
    return cake_stream(filt.cake_flow), filt.cake_flow


def l_stream(names) -> LiquidStream:
    """Fresh L feed."""
    return LiquidStream(
        str(DB_PATH), mass_flow=L_FLOW, temp=L_TEMPERATURE,
        mass_frac=R02R.named_mass_fraction({}, names, SOLVENT_NAME),
        name_solv=SOLVENT_NAME, verbose=False)


def make_solubility_function(names, species: str):
    """c_sat of ``species`` [mol/L] from the liquid's L content."""
    index_l = names.index(SOLVENT_NAME)
    s0 = SOLUBILITY_S0[species]

    def solubility(temp, conc):
        conc = np.asarray(conc)
        return s0 * np.exp(SOLVENT_K_L * conc[..., index_l])

    return solubility


def dissolver_temperature(cake: MixedStream) -> float:
    if DISSOLVER_TEMPERATURE is None:
        return float(cake.Phases[0].temp)
    return DISSOLVER_TEMPERATURE


def feed_mass_flow(cake: MixedStream) -> float:
    """Total mass flow into the tank: cake (liquid + crystals) plus L."""
    return sum(float(np.sum(s.mass_flow)) for s in cake.Streams) + L_FLOW


def residence_time(cake: MixedStream) -> float:
    """Nominal residence time [s] from the solvent charge."""
    return INITIAL_MASS / feed_mass_flow(cake)


def build_dissolver(cake: MixedStream) -> ContinuousCrystallizer:
    """D02R: a ContinuousCrystallizer charged with L, one solid per species.

    The CrystKinetics convenience setter only links liquid 0 to solid 0, so
    each species' liquid <-> solid connection is built by hand, a forward
    (nucleation and growth) and a reverse (dissolution) transfer, both
    driven by that species' population balance, as in D01R.
    """
    names = species_names()
    temp = dissolver_temperature(cake)
    liquid_ref = PhaseRef("liquid", 0)

    tank = ContinuousCrystallizer(integrator=ScipyBackend(), h_conv=0,
                                  diam=DISSOLVER_DIAMETER, isothermal=True)

    liquid = LiquidPhase(
        str(DB_PATH), temp=temp, mass=INITIAL_MASS,
        mass_frac=R02R.named_mass_fraction(INITIAL_COMPOSITION, names,
                                           SOLVENT_NAME),
        name_solv=SOLVENT_NAME)

    solids = []
    for species in CRYSTAL_SPECIES:
        mass_frac = np.zeros(len(names))
        mass_frac[names.index(species)] = 1.0
        solid = SolidPhase(str(DB_PATH), temp=temp, mass=0.0,
                           mass_frac=mass_frac, name_solv=SOLVENT_NAME)
        solid.mechanisms = OneDFVMMechanism(
            solid, target_components=species, solvent_name=SOLVENT_NAME,
            x_grid=CRYSTAL_SIZE_GRID,
            distrib_init=np.zeros(len(CRYSTAL_SIZE_GRID)), kv=CRYSTAL_KV)
        solids.append(solid)
    tank.Phases = [liquid] + solids

    connections = []
    for k, species in enumerate(CRYSTAL_SPECIES):
        solid_ref = PhaseRef("solid", k)
        kinetics = CrystKinetics(
            solub_fn=make_solubility_function(names, species),
            nucl_prim=NUCL_PRIM[species], nucl_sec=NUCL_SEC[species],
            growth=GROWTH[species], dissolution=DISSOLUTION[species],
            solubility_basis="mole_per_volume_solution")
        pbm = tank.Phases.get_phase_from_ref(solid_ref) \
            .get_mechanism(PopulationBalanceMechanism)
        pbm.liquid_phase_ref = liquid_ref
        pbm.owning_phase_ref = solid_ref
        pbm.liquid_phase = tank.Phases.get_phase_from_ref(liquid_ref)
        kinetics.target_idx = pbm.target_ind[0]
        pbm.mechanism_kinetics = kinetics
        transfer = ReversibleTransferMechanism(source_mechanism=pbm)
        connections += [
            PhaseConnection(liquid_ref, solid_ref,
                            species_weights=pbm.fraction,
                            mechanism=transfer.forward),
            PhaseConnection(solid_ref, liquid_ref,
                            species_weights=pbm.fraction,
                            mechanism=transfer.reverse),
        ]
    tank.phase_connections = connections
    tank.nomenclature(overwrite=True)
    tank.Inlet = [cake, l_stream(names)]
    return tank


def run_flowsheet(runtime: float | None = None):
    """R02R -> F01R -> D02R. Returns (cake, cake flow, dissolver)."""
    cake, cake_flow = f01r_cake()
    tank = build_dissolver(cake)

    if runtime is None:
        runtime = NUM_RESIDENCE_TIMES * residence_time(cake)

    tank.solve_unit(runtime=runtime, verbose=False)
    return cake, cake_flow, tank


# ---------------------------------------------------------------- results
def population_balance(tank, k: int):
    return tank.Phases.get_phase_from_ref(PhaseRef("solid", k)) \
        .get_mechanism(PopulationBalanceMechanism)


def crystal_weight(tank, k: int) -> np.ndarray:
    """Number density [#/m**3/um] -> crystal mass per m**3 slurry, per cell."""
    pbm = population_balance(tank, k)
    return (pbm.density * pbm.kv * np.asarray(pbm.x_grid)**3 * pbm.dx
            * 1e-18)


def slurry_volumes(tank, distribs, vessel_vol) -> list[np.ndarray]:
    """The slurry volume [m**3] each population balance's density is per.

    PharmaPy takes it per solid as V_liquid / (1 - phi_k), counting only
    that solid's own crystal volume fraction phi_k. With three solids that
    is not the vessel volume, which holds the liquid and every crystal::

        vessel_vol = V_liquid * (1 + sum_k phi_k / (1 - phi_k))

    ``distribs`` holds one number density (or a time series of them) per
    solid, in solid order; ``vessel_vol`` matches their shape.
    """
    phis = [np.asarray(d) @ crystal_weight(tank, k)
            / population_balance(tank, k).density
            for k, d in enumerate(distribs)]
    v_liquid = np.asarray(vessel_vol) / (1.0 + sum(p / (1.0 - p)
                                                   for p in phis))
    return [v_liquid / (1.0 - p) for p in phis]


def crystals_fed(cake_flow: SlurryFlow) -> dict[str, float]:
    """Crystal mass flow of each species in the cake [kg/s]."""
    names = list(cake_flow.name_species)
    return {name: cake_flow.solid_mass_flow
            * cake_flow.solid_mass_frac[names.index(name)]
            for name in CRYSTAL_SPECIES}


def history(tank, cake_flow) -> SimpleNamespace:
    """Time profiles read from the solved tank.

    ``time`` [s]; per species (dicts keyed by name): ``crystal_mass`` held
    [kg], ``crystal_out`` [kg/s], liquid ``conc`` and ``solubility``
    [mol/L], and ``fraction_dissolved`` [-], 1 - crystals out / crystals
    fed (meaningful at steady state; during start-up it includes the
    crystals the tank is still accumulating).
    """
    res = tank.result
    names = species_names()
    time = np.asarray(res.time)
    vessel_vol = np.ravel(res.vessel_vol)
    vol_flow = np.ravel(tank.outputs["vol_flow"])
    conc = np.asarray(res.mole_conc_liquid0, dtype=float)
    temp = float(tank.Phases.Liquids[0].temp)
    fed = crystals_fed(cake_flow)

    distribs = [np.asarray(getattr(res, f"distrib_solid{k}"), dtype=float)
                for k in range(len(CRYSTAL_SPECIES))]
    volumes = slurry_volumes(tank, distribs, vessel_vol)

    out = SimpleNamespace(time=time, crystal_mass={}, crystal_out={},
                          conc={}, solubility={}, fraction_dissolved={})
    for k, species in enumerate(CRYSTAL_SPECIES):
        crystal_kg_m3 = distribs[k] @ crystal_weight(tank, k)
        out.crystal_mass[species] = crystal_kg_m3 * volumes[k]
        out.crystal_out[species] = crystal_kg_m3 * vol_flow
        out.conc[species] = conc[:, names.index(species)]
        out.solubility[species] = make_solubility_function(names, species)(
            temp, conc)
        out.fraction_dissolved[species] = (1.0 - out.crystal_out[species]
                                           / fed[species])
    return out


def final_balance(tank) -> SimpleNamespace:
    """PharmaPy's own balance terms at the final time [kg/s].

    The final solver state is set on an independent copy of the vessel and
    its ``material_balances`` called, as the integrator does; each term is
    read from the contribution buffer rather than re-derived.

    ``feeds`` (feed phase, species) inlet flows and ``feed_phases`` their
    phase types, in the order cake liquid, cake D, E, J solids, L; liquid
    ``inlet``, ``to_liquid`` (dissolved), ``outlet``, ``accumulation``
    (species); and per species ``crystal`` {name: {inlet, dissolved,
    outlet, accumulation}}, each weighted by that solid's own slurry
    volume (``slurry_volumes``). Crystal ``inlet`` is PharmaPy's feed term
    on the population balance, which includes its dilution term.
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
    rows = {name: np.asarray(buffer.contributions[getattr(buffer, name)],
                             dtype=float)
            for name in ("INLET", "CROSSPHASE", "OUTLET")}
    rates = np.asarray(rates, dtype=float)
    volumes = slurry_volumes(
        tank, [np.asarray(getattr(res, f"distrib_solid{k}"))[-1]
               for k in range(len(CRYSTAL_SPECIES))],
        float(np.ravel(res.vessel_vol)[-1]))

    crystal = {}
    for k, species in enumerate(CRYSTAL_SPECIES):
        cells = slices[next(key for key in slices
                            if key.name == "distrib"
                            and key.phaseref == PhaseRef("solid", k))]
        weight = crystal_weight(tank, k) * float(volumes[k])

        def mass_rate(row):
            return float(weight @ row[cells])

        crystal[species] = {
            "inlet": mass_rate(rows["INLET"]),
            "dissolved": -mass_rate(rows["CROSSPHASE"]),
            "outlet": -mass_rate(rows["OUTLET"]),
            "accumulation": mass_rate(rates),
        }

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
        crystal=crystal,
    )


def steady_state_gap(balance: SimpleNamespace) -> float:
    """Largest relative accumulation: liquid species against the liquid
    feed, each crystal species against its crystal feed."""
    liquid = np.abs(balance.accumulation).max() / balance.inlet.sum()
    crystals = [abs(t["accumulation"]) / max(abs(t["inlet"]), 1e-30)
                for t in balance.crystal.values()]
    return float(max([liquid] + crystals))


def fraction_dissolved(balance: SimpleNamespace,
                       cake_flow: SlurryFlow) -> dict[str, float]:
    """Share of each species' fed crystal mass that dissolves, final time:
    what the liquid gains over the crystals the cake carries."""
    names = species_names()
    fed = crystals_fed(cake_flow)
    return {name: balance.to_liquid[names.index(name)] / fed[name]
            for name in CRYSTAL_SPECIES}


def liquid_feeds(balance: SimpleNamespace):
    """(cake liquid, L) species flows [kg/s] from the inlet transfers."""
    liquids = [f for f, phase in zip(balance.feeds, balance.feed_phases)
               if phase == "liquid"]
    return liquids[0], liquids[-1]


def balance_rows(balance: SimpleNamespace) -> list[list]:
    """[component, cake in, L in, dissolved, out, accumulated] [kg/s]."""
    names = species_names()
    cake, fresh = liquid_feeds(balance)
    rows = []
    for j, name in enumerate(names):
        values = [cake[j], fresh[j], balance.to_liquid[j],
                  balance.outlet[j], balance.accumulation[j]]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)
    for species, t in balance.crystal.items():
        rows.append([f"{species} (crystals)", t["inlet"], 0.0,
                     -t["dissolved"], t["outlet"], t["accumulation"]])
    crystals = balance.crystal.values()
    rows.append(["Total", cake.sum() + sum(t["inlet"] for t in crystals),
                 fresh.sum(),
                 balance.to_liquid.sum() - sum(t["dissolved"]
                                               for t in crystals),
                 balance.outlet.sum() + sum(t["outlet"] for t in crystals),
                 balance.accumulation.sum()
                 + sum(t["accumulation"] for t in crystals)])
    return rows


# ---------------------------------------------------------------- plotting
def _style(axis):
    axis.grid(True, color=GRID_COLOR, lw=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_profiles(cake, profiles):
    """Crystals held, saturation ratio and fraction dissolved vs time."""
    import matplotlib.pyplot as plt

    t = profiles.time / 60.0
    tau = residence_time(cake) / 60.0
    colors = dict(zip(CRYSTAL_SPECIES, SERIES_COLORS))
    fig, (ax_m, ax_s, ax_f) = plt.subplots(1, 3, figsize=(15, 4.5))

    for species in CRYSTAL_SPECIES:
        color = colors[species]
        ax_m.plot(t, profiles.crystal_mass[species] * 1e3, color=color, lw=2,
                  label=species)
        ax_s.plot(t, profiles.conc[species] / profiles.solubility[species],
                  color=color, lw=2, label=species)
        ax_f.plot(t, profiles.fraction_dissolved[species] * 100,
                  color=color, lw=2, label=species)

    ax_m.set_ylabel("Crystals in tank [g]")
    ax_m.set_title("Crystals held in D02R")
    ax_s.axhline(1.0, color=TEXT_MUTED, lw=1, ls="--")
    ax_s.annotate("saturated", (t[-1], 1.0), xytext=(0, -4),
                  textcoords="offset points", ha="right", va="top",
                  color=TEXT_MUTED, fontsize=9)
    ax_s.set_ylabel("$c_i / c_{sat,i}$ [-]")
    ax_s.set_title("Saturation ratio of the liquid")
    final = ", ".join(f"{s} {profiles.fraction_dissolved[s][-1]:.1%}"
                      for s in CRYSTAL_SPECIES)
    ax_f.set_ylim(0, 105)
    ax_f.set_ylabel("Fed crystals dissolved [%]")
    ax_f.set_title(f"Fraction dissolved (final {final})", fontsize=11)

    for axis in (ax_m, ax_s, ax_f):
        axis.set_xlabel("Time [min]")
        axis.legend(title="i", frameon=False)
        _style(axis)

    fig.suptitle(f"D02R - L dissolver on the F01R cake  (residence time "
                 f"{tau:.1f} min, run {t[-1] / tau:.0f} residence times)")
    fig.tight_layout()
    return fig


def plot_csd(tank, cake_flow):
    """Each species' crystal mass flow by size, fed vs leaving.

    Absolute flows on a log axis, so the few crystals left are visible
    next to the cake they came from.
    """
    import matplotlib.pyplot as plt

    numbers = split_cake(cake_flow)
    vol_flow = float(np.ravel(tank.outputs["vol_flow"])[-1])
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (species, color) in enumerate(zip(CRYSTAL_SPECIES,
                                             SERIES_COLORS)):
        pbm = population_balance(tank, k)
        x = np.asarray(pbm.x_grid)
        per_um = pbm.density * pbm.kv * x**3 * 1e-18 * 1e3  # g/(s um)
        leaving = np.asarray(getattr(tank.result,
                                     f"distrib_solid{k}"))[-1] * vol_flow
        ax.plot(x, numbers[species] * per_um, color=color, lw=2,
                label=f"{species} fed")
        ax.plot(x, leaving * per_um, color=color, lw=2, ls="--",
                label=f"{species} leaving")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-9)
    ax.set_xlabel("Crystal size [um]")
    ax.set_ylabel("Crystal mass flow [g/(s um)]")
    ax.set_title("Crystal size distribution by mass flow, end of run")
    ax.legend(frameon=False, ncol=3)
    _style(ax)
    fig.tight_layout()
    return fig


def settings_rows(cake, cake_flow, profiles, balance):
    names = species_names()
    tau = residence_time(cake)
    gap = steady_state_gap(balance)
    fed = crystals_fed(cake_flow)
    liquid = ", ".join(
        f"{n} {w:.3g}" for n, w in zip(names, cake.Phases[0].mass_frac)
        if w > 1e-9)

    def kinetics(params):
        return ", ".join(f"{p:g}" for p in params)

    rows = [
        ("Tank", "Charge", f"{INITIAL_MASS:g} kg of {SOLVENT_NAME}"),
        ("Tank", "Temperature",
         f"{dissolver_temperature(cake):.2f} K (isothermal, cake T)"),
        ("Tank", "Residence time",
         f"{tau:.1f} s ({tau / 60:.2f} min) = charge / feed"),
        ("Feeds", "F01R cake crystals",
         ", ".join(f"{n} {fed[n] * 1e3:.4g}" for n in CRYSTAL_SPECIES)
         + " g/s"),
        ("Feeds", "F01R cake liquid",
         f"{cake_flow.liquid_mass_flow * 1e3:.4g} g/s at "
         f"{cake_flow.temp:.4g} K (w: {liquid})"),
        ("Feeds", f"Fresh {SOLVENT_NAME}",
         f"{L_FLOW * 1e3:g} g/s at {L_TEMPERATURE:g} K"),
        ("Solubility", "Model",
         f"S0 exp({SOLVENT_K_L:g} [{SOLVENT_NAME}]) mol/L"),
        ("Solubility", "S0",
         ", ".join(f"{s} {SOLUBILITY_S0[s]:g}" for s in CRYSTAL_SPECIES)
         + " mol/L"),
    ]
    for species in CRYSTAL_SPECIES:
        rows.append((f"Kinetics {species}", "Nucl. / growth / dissol.",
                     f"({kinetics(NUCL_PRIM[species])}) / "
                     f"({kinetics(GROWTH[species])}) / "
                     f"({kinetics(DISSOLUTION[species])})"))
    rows += [
        ("Run", "Runtime", f"{profiles.time[-1]:.0f} s "
                           f"({NUM_RESIDENCE_TIMES:g} residence times)"),
        ("Run", "Steady state",
         f"accumulation {gap:.2%} of feed "
         f"({'reached' if gap < STEADY_STATE_TOL else 'NOT reached'})"),
        ("Run", "Crystals dissolved",
         ", ".join(f"{s} {v:.2%}"
                   for s, v in fraction_dissolved(balance,
                                                  cake_flow).items())),
    ]
    return rows


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
                         "D02R settings and initial conditions",
                         col_widths=[0.16, 0.22, 0.62], width=11)


def plot_mass_balance(balance):
    def cell(value):
        return "0" if abs(value) < 5e-10 else f"{value * 1e3:.5f}"

    rows = [[name] + [cell(v) for v in values]
            for name, *values in balance_rows(balance)]
    return _table_figure(
        ["Component", "Cake in [g/s]", f"{SOLVENT_NAME} in [g/s]",
         "Dissolved [g/s]", "Out [g/s]", "Accumulated [g/s]"], rows,
        f"D02R mass balance at t = {balance.time:.0f} s\n'Dissolved' leaves "
        "a crystal row and enters its dissolved row; crystal 'Cake in' is "
        "PharmaPy's feed term",
        align="right", width=11)


def save_figures(cake, cake_flow, tank, out_dir: Path = RESULTS_DIR,
                 balance: SimpleNamespace | None = None):
    profiles = history(tank, cake_flow)
    if balance is None:
        balance = final_balance(tank)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "D02R_profiles": plot_profiles(cake, profiles),
        "D02R_csd": plot_csd(tank, cake_flow),
        "D02R_settings": plot_settings(cake, cake_flow, profiles, balance),
        "D02R_mass_balance": plot_mass_balance(balance),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


# ---------------------------------------------------------------- report
def print_report(cake, cake_flow, tank, balance: SimpleNamespace):
    profiles = history(tank, cake_flow)
    tau = residence_time(cake)
    fed = crystals_fed(cake_flow)
    dissolved = fraction_dissolved(balance, cake_flow)
    print(f"D02R: {INITIAL_MASS:g} kg {SOLVENT_NAME} charge, "
          f"tau = {tau:.1f} s, run {profiles.time[-1]:.0f} s\n")

    print(f"  {'species':<8}{'liquid':>10}{'c_sat':>10}{'fed':>10}"
          f"{'held':>10}{'dissolved':>11}")
    print(f"  {'':<8}{'[mol/L]':>10}{'[mol/L]':>10}{'[g/s]':>10}"
          f"{'[g]':>10}{'[% fed]':>11}")
    for species in CRYSTAL_SPECIES:
        print(f"  {species:<8}{profiles.conc[species][-1]:>10.5f}"
              f"{profiles.solubility[species][-1]:>10.5f}"
              f"{fed[species] * 1e3:>10.5f}"
              f"{profiles.crystal_mass[species][-1] * 1e3:>10.3f}"
              f"{dissolved[species]:>11.2%}")

    print(f"\n  {'component':<16}{'cake in':>11}{'L in':>11}"
          f"{'dissolved':>11}{'out':>11}{'accum.':>11}   [g/s]")
    for name, *values in balance_rows(balance):
        print(f"  {name:<16}" + "".join(f"{v * 1e3:>11.5f}" for v in values))

    gap = steady_state_gap(balance)
    if gap >= STEADY_STATE_TOL:
        print(f"\n  WARNING: accumulation is {gap:.2%} of the feed (or of "
              "a crystal feed); increase NUM_RESIDENCE_TIMES.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    d02r_cake, d02r_cake_flow, d02r = run_flowsheet()
    d02r_balance = final_balance(d02r)
    print_report(d02r_cake, d02r_cake_flow, d02r, d02r_balance)
    save_figures(d02r_cake, d02r_cake_flow, d02r, balance=d02r_balance)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
