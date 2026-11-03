"""D01R: continuous antisolvent crystallizer on the R02R outlet.

The flowsheet is::

    FEED_A --+                R02R outlet --+
             +--> R02R  ...                 +--> D01R --> slurry
    FEED_B --+                ANTISOLVENT --+

``D01R`` is a ``PharmaPy.Crystallizers_Refactored.ContinuousCrystallizer``
charged with B and fed two streams: R02R's outlet (``r02r_feed`` solves
``R02R`` and takes its outlet at the final, steady state) and pure B as
antisolvent. R02R runs in H, so its outlet holds no B. B lowers the
solubility of R02R's three products D, E and J faster than it dilutes them,
and each crystallizes as its own solid phase with its own population
balance, nucleation, growth and dissolution kinetics. The R02R reactions
are not carried into the tank: D01R only crystallizes.

Solubility of each crystallizing species i (mol/L), with B as antisolvent::

    c_sat,i = S0_i * exp(-K_B * [B])

S0_i is the solubility in B-free liquid, i.e. in R02R's outlet. It sits
25 % above R02R's outlet concentration, so R02R's liquid alone stays
undersaturated. With K_B = 0.2 L/mol, D01R's steady [B] (~19.4 mol/L)
cuts every solubility to ~2 % of S0, so ~96 % of the D, E and J fed
crystallizes. K_B = 0.0364 leaves them only ~5 % supersaturated, and a few
percent crystallizes. Raise K_B or ANTISOLVENT_FLOW to crystallize more.

The pure-B charge is far richer in B than the steady state, so the first
solute to arrive meets a low solubility and crystallizes; those crystals
redissolve as R02R's liquid displaces the charge. That start-up is what the
dissolution kinetics act on.

Crystallization rates (PharmaPy's CrystKinetics), relative supersaturation
sigma_i = (c_i - c_sat,i) / c_sat,i::

    nucleation   B = k_b * exp(-E_b / (R T)) * sigma**b      [#/(s m**3)]
    growth       G = k_g * exp(-E_g / (R T)) * sigma**g      [um/s]
    dissolution  D = k_d * exp(-E_d / (R T)) * sigma*|sigma|**(d-1) [um/s]

Growth and nucleation act when sigma > 0, dissolution when sigma < 0. At
steady state the liquid stays supersaturated, so dissolution only acts on
crystals that meet undersaturated liquid (start-up, or a change in the
feeds).

The residence time comes from the solvent charge::

    tau = INITIAL_MASS / (R02R outlet mass flow + ANTISOLVENT_FLOW)

and the run lasts ``NUM_RESIDENCE_TIMES`` of them, enough to reach steady
state.

Run it with::

    python D01R.py                # solve, save figures, show them
    python D01R.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    D01R_profiles.png      crystals held, saturation, fraction crystallized
    D01R_csd.png           crystal size distribution of D, E and J at the end
    D01R_settings.png      the settings and initial conditions used
    D01R_mass_balance.png  inlet vs outlet species balance at the final time
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
HERE = Path(__file__).resolve().parent          # Project work/D01R.py
R02R_DIR = HERE.parent / "R02R"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(R02R_DIR))

import R02R  # noqa: E402

DB_PATH = R02R.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = ("D", "E", "J")
SOLVENT_NAME = "B"  # the tank charge (R02R's solvent is H)
ANTISOLVENT_NAME = "B"

# Fresh antisolvent feed, pure ANTISOLVENT_NAME.
ANTISOLVENT_FLOW = 1.0e-2  # kg/s
ANTISOLVENT_TEMPERATURE = 25 + 273.15  # K

# Tank charge: pure B. Its mass sets the residence time.
INITIAL_MASS = 80.0  # kg
INITIAL_COMPOSITION = {}  # solute mass fractions; solvent is the remainder
CRYSTALLIZER_TEMPERATURE = 20 + 273.15  # K, held constant (isothermal tank)
CRYSTALLIZER_DIAMETER = 0.2  # m

# Solubility c_sat = S0 * exp(-K_B [B]) [mol/L]. S0 is 1.25x R02R's outlet
# concentration.
SOLUBILITY_S0 = {"D": 0.0284, "E": 0.01505, "J": 0.0886}  # mol/L
# L/mol: each 1 mol/L of B in the liquid multiplies every solubility by
# exp(-K_B) = 0.82; at the ~19.4 mol/L of B the tank settles at
# (0.01 kg/s B), exp(-K_B * 19.4) = 0.021, so ~96 % of D, E and J
# crystallizes. 0.0364 (0.49 at steady state) crystallizes ~4 %.
ANTISOLVENT_K_B = 0.2

# Kinetics per species, (k, E_a [J/mol], order). Units: nucleation
# #/(s m**3 slurry), growth and dissolution um/s.
NUCL_PRIM = {name: (1.0e10, 0.0, 2.0) for name in CRYSTAL_SPECIES}
NUCL_SEC = {name: (0.0, 0.0, 0.0, 0.0) for name in CRYSTAL_SPECIES}
GROWTH = {name: (10.0, 0.0, 1.0) for name in CRYSTAL_SPECIES}
# Dissolution kinetics for each species: k_d is the shrink rate [um/s] at
# sigma = -1 (crystals in solute-free liquid).
DISSOLUTION = {
    "D": (5e5, 0.0, 1.0),
    "E": (5e5, 0.0, 1.0),
    "J": (5e5, 0.0, 1.0),
}

# um. Crystals that grow past the last cell leave the population balance
# untracked; the report warns if that exceeds 1 % of the crystal mass. At
# these settings 99.99 % of the crystal mass stays below ~1.4 mm (J, the
# largest); the old 1-601 um grid lost up to 28 % of it.
CRYSTAL_SIZE_GRID = np.arange(1.0, 2002.0, 10.0)
CRYSTAL_KV = 1.0

# The liquid settles in ~5 residence times; the crystal populations, which
# follow the liquid's [B], take more than twice that.
NUM_RESIDENCE_TIMES = 12.0
# Accumulation / feed mass flow per liquid species, and accumulation / rate
# of formation per crystal species.
STEADY_STATE_TOL = 1.0e-2

# Categorical slots 1-3 (blue, orange, aqua), one per crystallizing species.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID_COLOR = "#e5e4e0"


# ---------------------------------------------------------------- builders
def species_names() -> tuple[str, ...]:
    return R02R.database_species(DB_PATH)


def r02r_feed() -> MixedStream:
    """Solve R02R and return its outlet at the final time.

    The outlet holds constant flows, temperature and composition, so it
    feeds D01R as a steady stream rather than R02R's start-up transient.
    """
    return R02R.run_reactor(DB_PATH).Outlet


def antisolvent_stream(names) -> LiquidStream:
    """Fresh antisolvent feed."""
    return LiquidStream(
        str(DB_PATH), mass_flow=ANTISOLVENT_FLOW,
        temp=ANTISOLVENT_TEMPERATURE,
        mass_frac=R02R.named_mass_fraction({}, names, ANTISOLVENT_NAME),
        name_solv=ANTISOLVENT_NAME, verbose=False)


def make_solubility_function(names, species: str):
    """c_sat of ``species`` [mol/L] from the liquid's antisolvent content."""
    index_b = names.index(ANTISOLVENT_NAME)
    s0 = SOLUBILITY_S0[species]

    def solubility(temp, conc):
        conc = np.asarray(conc)
        return s0 * np.exp(-ANTISOLVENT_K_B * conc[..., index_b])

    return solubility


def feed_mass_flow(feed: MixedStream) -> float:
    """Total mass flow into the tank: R02R outlet plus antisolvent [kg/s]."""
    r02r = sum(float(np.sum(s.mass_flow)) for s in feed.Streams)
    return r02r + ANTISOLVENT_FLOW


def residence_time(feed: MixedStream) -> float:
    """Nominal residence time [s] from the solvent charge."""
    return INITIAL_MASS / feed_mass_flow(feed)


def build_crystallizer(feed: MixedStream) -> ContinuousCrystallizer:
    """D01R: a ContinuousCrystallizer charged with B, one solid per species.

    The CrystKinetics convenience setter only links liquid 0 to solid 0, so
    each species' liquid <-> solid connection is built here the same way:
    a forward (nucleation and growth) and a reverse (dissolution) transfer,
    both driven by that species' population balance.
    """
    names = species_names()
    liquid_ref = PhaseRef("liquid", 0)

    tank = ContinuousCrystallizer(integrator=ScipyBackend(), h_conv=0,
                                  diam=CRYSTALLIZER_DIAMETER, isothermal=True)

    liquid = LiquidPhase(
        str(DB_PATH), temp=CRYSTALLIZER_TEMPERATURE, mass=INITIAL_MASS,
        mass_frac=R02R.named_mass_fraction(INITIAL_COMPOSITION, names,
                                           SOLVENT_NAME),
        name_solv=SOLVENT_NAME)

    solids = []
    for species in CRYSTAL_SPECIES:
        mass_frac = np.zeros(len(names))
        mass_frac[names.index(species)] = 1.0
        solid = SolidPhase(str(DB_PATH), temp=CRYSTALLIZER_TEMPERATURE,
                           mass=0.0, mass_frac=mass_frac,
                           name_solv=SOLVENT_NAME)
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
    tank.Inlet = [feed, antisolvent_stream(names)]
    return tank


def run_flowsheet(runtime: float | None = None):
    """R02R -> D01R. Returns (R02R outlet feed, crystallizer)."""
    feed = r02r_feed()
    tank = build_crystallizer(feed)

    if runtime is None:
        runtime = NUM_RESIDENCE_TIMES * residence_time(feed)

    tank.solve_unit(runtime=runtime, verbose=False)
    return feed, tank


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


def history(tank) -> SimpleNamespace:
    """Time profiles read from the solved tank.

    ``time`` [s]; per species (dicts keyed by name): ``crystal_mass`` held
    [kg], ``crystal_conc`` [mol/L slurry], ``conc`` liquid [mol/L],
    ``solubility`` [mol/L] and ``fraction_crystallized`` [-], the share of
    the species in the tank that is solid.
    """
    res = tank.result
    names = species_names()
    time = np.asarray(res.time)
    vessel_vol = np.ravel(res.vessel_vol)
    conc = np.asarray(res.mole_conc_liquid0, dtype=float)
    mw = tank.Phases.Liquids[0].mw

    distribs = [np.asarray(getattr(res, f"distrib_solid{k}"), dtype=float)
                for k in range(len(CRYSTAL_SPECIES))]
    volumes = slurry_volumes(tank, distribs, vessel_vol)

    out = SimpleNamespace(time=time, crystal_mass={}, crystal_conc={},
                          conc={}, solubility={}, fraction_crystallized={})
    for k, species in enumerate(CRYSTAL_SPECIES):
        j = names.index(species)
        crystal_kg_m3 = distribs[k] @ crystal_weight(tank, k)
        out.crystal_mass[species] = crystal_kg_m3 * volumes[k]
        out.crystal_conc[species] = crystal_kg_m3 / mw[j]
        out.conc[species] = conc[:, j]
        out.solubility[species] = make_solubility_function(names, species)(
            CRYSTALLIZER_TEMPERATURE, conc)
        total = out.crystal_conc[species] + conc[:, j]
        out.fraction_crystallized[species] = np.divide(
            out.crystal_conc[species], total, out=np.zeros_like(total),
            where=total > 0)
    return out


def final_balance(tank) -> SimpleNamespace:
    """PharmaPy's own balance terms at the final time [kg/s].

    The final solver state is set on an independent copy of the vessel and
    its ``material_balances`` called, as the integrator does; each term is
    read from the contribution buffer rather than re-derived.

    ``feeds`` (feed, species) inlet flows; liquid ``inlet``, ``to_solid``,
    ``outlet``, ``accumulation`` (species); and per crystallizing species
    ``crystal`` {name: {growth, inlet, outlet, accumulation}}, each weighted
    by that solid's own slurry volume (``slurry_volumes``). ``inlet`` on a
    crystal is PharmaPy's dilution term (no crystals are fed).
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
            "growth": mass_rate(rows["CROSSPHASE"]),
            "inlet": mass_rate(rows["INLET"]),
            "outlet": -mass_rate(rows["OUTLET"]),
            "accumulation": mass_rate(rates),
        }

    return SimpleNamespace(
        time=time,
        feeds=np.array([np.array(t.species_flow, dtype=float)
                        for t in buffer.aux[buffer.INLET]]),
        inlet=rows["INLET"][liquid],
        to_solid=-rows["CROSSPHASE"][liquid],
        outlet=-rows["OUTLET"][liquid],
        accumulation=rates[liquid],
        crystal=crystal,
    )


def steady_state_gap(balance: SimpleNamespace) -> float:
    """Largest relative accumulation: liquid species against the feed flow,
    crystal species against their rate of formation."""
    liquid = np.abs(balance.accumulation).max() / balance.inlet.sum()
    crystals = [abs(t["accumulation"]) / max(abs(t["growth"]), 1e-30)
                for t in balance.crystal.values()]
    return float(max([liquid] + crystals))


def balance_rows(balance: SimpleNamespace) -> list[list]:
    """[component, R02R in, antisolvent in, to crystals, out, accumulated]
    [kg/s]."""
    names = species_names()
    r02r, h = balance.feeds[0], balance.feeds[1]
    rows = []
    for j, name in enumerate(names):
        values = [r02r[j], h[j], balance.to_solid[j], balance.outlet[j],
                  balance.accumulation[j]]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)
    for species, terms in balance.crystal.items():
        rows.append([f"{species} (crystals)", terms["inlet"], 0.0,
                     terms["growth"], terms["outlet"],
                     terms["accumulation"]])
    liquid_out = balance.outlet.sum()
    crystals_out = sum(t["outlet"] for t in balance.crystal.values())
    rows.append(["Total", r02r.sum(), h.sum(), 0.0,
                 liquid_out + crystals_out,
                 balance.accumulation.sum()
                 + sum(t["accumulation"] for t in balance.crystal.values())])
    return rows


# ---------------------------------------------------------------- plotting
def _style(axis):
    axis.grid(True, color=GRID_COLOR, lw=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_profiles(feed, profiles):
    """Crystals held, saturation ratio and fraction crystallized vs time."""
    import matplotlib.pyplot as plt

    t = profiles.time / 60.0
    tau = residence_time(feed) / 60.0
    colors = dict(zip(CRYSTAL_SPECIES, SERIES_COLORS))
    fig, (ax_m, ax_s, ax_f) = plt.subplots(1, 3, figsize=(15, 4.5))

    for species in CRYSTAL_SPECIES:
        color = colors[species]
        ax_m.plot(t, profiles.crystal_mass[species] * 1e3, color=color, lw=2,
                  label=species)
        ax_s.plot(t, profiles.conc[species] / profiles.solubility[species],
                  color=color, lw=2, label=species)
        ax_f.plot(t, profiles.fraction_crystallized[species] * 100,
                  color=color, lw=2, label=species)

    ax_m.set_ylabel("Crystals in tank [g]")
    ax_m.set_title("Crystals held in D01R")
    ax_s.axhline(1.0, color=TEXT_MUTED, lw=1, ls="--")
    ax_s.annotate("saturated", (t[-1], 1.0), xytext=(0, 4),
                  textcoords="offset points", ha="right", va="bottom",
                  color=TEXT_MUTED, fontsize=9)
    ax_s.set_ylabel("$c_i / c_{sat,i}$ [-]")
    ax_s.set_title("Saturation ratio of the liquid")
    final = ", ".join(f"{s} {profiles.fraction_crystallized[s][-1]:.1%}"
                      for s in CRYSTAL_SPECIES)
    ax_f.set_ylabel("Solid share of species in tank [%]")
    ax_f.set_title(f"Fraction crystallized (final {final})", fontsize=11)

    for axis in (ax_m, ax_s, ax_f):
        axis.set_xlabel("Time [min]")
        axis.legend(title="i", frameon=False)
        _style(axis)

    fig.suptitle(f"D01R - antisolvent crystallizer on the R02R outlet  "
                 f"(residence time {tau:.1f} min, run "
                 f"{t[-1] / tau:.0f} residence times)")
    fig.tight_layout()
    return fig


def plot_csd(tank):
    """Mass-weighted CSD of each crystallizing species at the end."""
    import matplotlib.pyplot as plt

    res = tank.result
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (species, color) in enumerate(zip(CRYSTAL_SPECIES,
                                             SERIES_COLORS)):
        pbm = population_balance(tank, k)
        x, dx = np.asarray(pbm.x_grid), pbm.dx
        mass = np.asarray(getattr(res, f"distrib_solid{k}"))[-1] * x**3 * dx
        total = mass.sum()
        w = mass / total / dx if total > 0 else mass
        mean = np.sum(w * x * dx)
        ax.plot(x, w * 100, color=color, lw=2,
                label=f"{species}  (mean {mean:.0f} um)")
    ax.set_xlabel("Crystal size [um]")
    ax.set_ylabel("Mass fraction [% per um]")
    ax.set_title("Crystal size distribution by mass, end of run")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def settings_rows(feed, tank, profiles, balance):
    names = species_names()
    liquid = feed.Streams[0]
    tau = residence_time(feed)
    gap = steady_state_gap(balance)
    def solutes(crystallizing):
        return ", ".join(
            f"{n} {w:.3g}" for n, w in zip(names, liquid.mass_frac)
            if n != SOLVENT_NAME and w > 1e-9
            and (n in CRYSTAL_SPECIES) == crystallizing)

    def kinetics(params):
        return ", ".join(f"{p:g}" for p in params)

    rows = [
        ("Tank", "Charge", f"{INITIAL_MASS:g} kg of {SOLVENT_NAME}"),
        ("Tank", "Temperature",
         f"{CRYSTALLIZER_TEMPERATURE:g} K (isothermal)"),
        ("Tank", "Residence time",
         f"{tau:.1f} s ({tau / 60:.2f} min) = charge / feed"),
        ("Feeds", "R02R outlet",
         f"{float(np.sum(liquid.mass_flow)) * 1e3:.4g} g/s at "
         f"{float(liquid.temp):.4g} K"),
        ("Feeds", "R02R products (w)", solutes(True)),
        ("Feeds", "R02R others (w)", solutes(False)),
        ("Feeds", f"Antisolvent {ANTISOLVENT_NAME}",
         f"{ANTISOLVENT_FLOW * 1e3:g} g/s at {ANTISOLVENT_TEMPERATURE:g} K"),
        ("Solubility", "Model",
         f"S0 exp(-{ANTISOLVENT_K_B:g} [{ANTISOLVENT_NAME}]) mol/L"),
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
        ("Run", "Fraction crystallized",
         ", ".join(f"{s} {profiles.fraction_crystallized[s][-1]:.1%}"
                   for s in CRYSTAL_SPECIES)),
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


def plot_settings(feed, tank, profiles, balance):
    rows, previous = [], None
    for group, setting, value in settings_rows(feed, tank, profiles,
                                               balance):
        rows.append([group if group != previous else "", setting, value])
        previous = group
    return _table_figure(["Group", "Setting", "Value"], rows,
                         "D01R settings and initial conditions",
                         col_widths=[0.16, 0.24, 0.60], width=10)


def plot_mass_balance(balance):
    def cell(value):
        return "0" if abs(value) < 5e-10 else f"{value * 1e3:.5f}"

    rows = [[name] + [cell(v) for v in values]
            for name, *values in balance_rows(balance)]
    return _table_figure(
        ["Component", "R02R in [g/s]", f"{ANTISOLVENT_NAME} in [g/s]",
         "To crystals [g/s]", "Out [g/s]", "Accumulated [g/s]"], rows,
        f"D01R mass balance at t = {balance.time:.0f} s\n'To crystals' leaves "
        "a dissolved row and enters a crystal row; crystal 'R02R in' is "
        "PharmaPy's dilution term (no crystals are fed)",
        align="right", width=11)


def save_figures(feed, tank, out_dir: Path = RESULTS_DIR,
                 balance: SimpleNamespace | None = None):
    profiles = history(tank)
    if balance is None:
        balance = final_balance(tank)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "D01R_profiles": plot_profiles(feed, profiles),
        "D01R_csd": plot_csd(tank),
        "D01R_settings": plot_settings(feed, tank, profiles, balance),
        "D01R_mass_balance": plot_mass_balance(balance),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


# ---------------------------------------------------------------- report
def print_report(feed, tank, balance: SimpleNamespace):
    profiles = history(tank)
    tau = residence_time(feed)
    print(f"D01R: {INITIAL_MASS:g} kg {SOLVENT_NAME} charge, "
          f"tau = {tau:.1f} s, run {profiles.time[-1]:.0f} s\n")

    print(f"  {'species':<8}{'liquid':>10}{'c_sat':>10}{'crystals':>11}"
          f"{'crystallized':>14}")
    print(f"  {'':<8}{'[mol/L]':>10}{'[mol/L]':>10}{'held [g]':>11}"
          f"{'[% of fed]':>14}")
    names = species_names()
    for species in CRYSTAL_SPECIES:
        j = names.index(species)
        fed = balance.feeds[:, j].sum()
        print(f"  {species:<8}{profiles.conc[species][-1]:>10.5f}"
              f"{profiles.solubility[species][-1]:>10.5f}"
              f"{profiles.crystal_mass[species][-1] * 1e3:>11.3f}"
              f"{balance.to_solid[j] / fed:>14.2%}")

    print(f"\n  {'component':<16}{'R02R in':>11}{'anti. in':>11}"
          f"{'to cryst.':>11}{'out':>11}{'accum.':>11}   [g/s]")
    for name, *values in balance_rows(balance):
        print(f"  {name:<16}" + "".join(f"{v * 1e3:>11.5f}" for v in values))

    for species in CRYSTAL_SPECIES:
        j = names.index(species)
        past_grid = balance.to_solid[j] - balance.crystal[species]["growth"]
        if past_grid > 0.01 * max(balance.to_solid[j], 1e-30):
            print(f"\n  WARNING: {past_grid / balance.to_solid[j]:.1%} of the "
                  f"{species} crystal mass grows past the last size cell; "
                  "extend CRYSTAL_SIZE_GRID.")
    gap = steady_state_gap(balance)
    if gap >= STEADY_STATE_TOL:
        print(f"\n  WARNING: accumulation is {gap:.2%} of the feed (or of "
              "a crystal's formation rate); increase NUM_RESIDENCE_TIMES.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    d01r_feed, d01r = run_flowsheet()
    d01r_balance = final_balance(d01r)
    print_report(d01r_feed, d01r, d01r_balance)
    save_figures(d01r_feed, d01r, balance=d01r_balance)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
