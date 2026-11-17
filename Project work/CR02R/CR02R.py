"""CR02R: R02R as a continuous reactive crystallizer.

The flowsheet is::

    D01H slurry (F crystals) --+
                               +--> CR02R --> slurry (D, E, J crystals)
    MAKEUP (A, G, I, H)  ------+

``CR02R`` is ``PharmaPy.Reactive_Crystallizers.ContinuousReactiveCrystallizer``:
R02R's vessel, jacket and three parallel reactions (taken from ``R02R.py``
unchanged), plus four solid phases, each with its own population balance:

- F, fed as crystals in D01H's slurry. F's solubility is far above anything
  the liquid reaches, so the crystals dissolve, and the dissolved F reacts.
  Before CR02R, R02R had no solid phase and the flowsheet turned D01H's F
  crystals into dissolved F inside the link.
- D, E and J, the products. Each crystallizes once the liquid passes its
  solubility. The solubilities are made up, set so that about 20 % of each
  product crystallizes here at the flowsheet's conditions; D01R
  crystallizes the rest of D and E, and J dissolves again in D01R's B.

Solubility (PharmaPy's Apelblat form, kg per kg of H, the solvent)::

    ln c_sat,i = a_i + b_i / T + c_i ln T

b_i and c_i are invented, giving solubilities that rise ~3 %/K at the
~333 K CR02R settles at. a_i was then set so that c_sat,i at 333 K sits
where ~20 % of species i crystallizes (see ``SOLUBILITY_APELBLAT``).

Crystallization kinetics (nucleation, growth, dissolution) are D01R's, on
the relative supersaturation (c - c_sat) / c_sat. The products sit on
D01R's 20 um size grid so their crystals pass into D01R's solids directly;
in the flowsheet F sits on the 100 um H-side grid, the grid D01H's
crystals arrive on; standalone it sits on a 20 um grid (``F_SIZE_GRID``).
The reactions are not affected by the crystals: their rates
depend on A, F and G only.

Standalone, CR02R is fed ``DESIGN_FEED``: D01H's discharge at the end of a
flowsheet run (its liquid, and its F crystals given a made-up 1 mm Gaussian
size distribution), plus R02R's feed with the F taken out (``makeup_stream``).
That is what CR02R sees in the flowsheet, so the 20 % carries over (the
full flowsheet and this file agree to ~0.3 % in T and F converted).

Run it with::

    python CR02R.py                # solve, save figures, show them
    python CR02R.py --no-show      # solve and save figures only

Figures are written to ``results/`` next to this file:

    CR02R_profiles.png      fraction crystallized, F crystals, temperature
    CR02R_csd.png           crystal size distribution of each solid at the end
    CR02R_settings.png      the settings used for the run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PharmaPy.DataClasses import PhaseConnection, PhaseRef
from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
from PharmaPy.Mechanisms import (
    OneDFVMMechanism,
    PopulationBalanceMechanism,
    ReversibleTransferMechanism,
)
from PharmaPy.MixedPhases_Refactored import MixedStream
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Reactive_Crystallizers import ContinuousReactiveCrystallizer
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Utilities import CoolingWater

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent          # Project work/CR02R
PROJECT = HERE.parent
RESULTS_DIR = HERE / "results"

for folder in ("R02R", "D01R", "F01R"):
    sys.path.insert(0, str(PROJECT / folder))

import D01R  # noqa: E402
import F01R  # noqa: E402
import R02R  # noqa: E402

DB_PATH = R02R.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = ("D", "E", "J")  # the products, which crystallize here
# F comes first: D01H's slurry carries one solid, which PharmaPy maps onto
# the vessel's solid 0.
SOLID_SPECIES = ("F",) + CRYSTAL_SPECIES
SOLVENT_NAME = "H"

# ln c_sat = a + b / T + c ln T [kg/kg H]. b and c are invented; a puts
# c_sat at 333 K where ~20 % of each product crystallizes in the flowsheet
# (D 0.0201, E 0.0375, J 0.00454 kg/kg H made in total, from the full
# flowsheet run; the liquid ends only ~0.2 % supersaturated, so c_sat ~ 0.8x
# those). F's sits at ~0.15 kg/kg H at 333 K, ~2.5x the F fed per kg of H,
# so F crystals always meet undersaturated liquid and dissolve.
SOLUBILITY_APELBLAT = {
    "F": (-1.6980, -2000.0, 1.0),
    "D": (-0.9292, -3000.0, 1.0),
    "E": (-7.6155, -2500.0, 2.0),
    "J": (1.9880, -3500.0, 0.5),
}
SOLUBILITY_BASIS = "mass_per_mass_solvent"

# Kinetics (k, E_a [J/mol], order): D01R's for the products. F only
# dissolves, at D01R's rate; its nucleation and growth are off.
NUCL_PRIM = {"F": (0.0, 0.0, 2.0), **{s: D01R.NUCL_PRIM[s] for s in
                                      ("D", "E")}, "J": (1.0e10, 0.0, 2.0)}
NUCL_SEC = {name: (0.0, 0.0, 0.0, 0.0) for name in SOLID_SPECIES}
GROWTH = {"F": (0.0, 0.0, 1.0), **{s: D01R.GROWTH[s] for s in ("D", "E")},
          "J": (10.0, 0.0, 1.0)}
DISSOLUTION = {name: (50.0, 0.0, 1.0) for name in SOLID_SPECIES}

# um. The products use D01R's grid, so their crystals feed D01R's solids
# as they are. Flowsheet.py sets F's to its 100 um H-side grid. Standalone
# F uses 20 um: on 100 um cells the made-up 1 mm Gaussian of DESIGN_FEED
# dissolves into ~16 % more dissolved F than its crystal mass (0.405 vs
# 0.349 g/s converted), which inflates every product. At 20 um it converts
# 0.3505 g/s, as the flowsheet's own CSD does on 100 um.
PRODUCT_SIZE_GRID = D01R.CRYSTAL_SIZE_GRID
F_SIZE_GRID = np.arange(1.0, 10002.0, 20.0)
CRYSTAL_KV = 1.0

# D01H's discharge at the end of the flowsheet run [kg/s], CR02R's
# standalone slurry feed. Its F crystals get a made-up size distribution:
# the flowsheet's are mm sized.
DESIGN_FEED = {
    "temp": 281.01,  # K
    "liquid": {"A": 1.02e-6, "B": 5.80e-6, "F": 4.9e-7, "H": 5.5483e-4},
    "f_crystals": 3.4926e-4,
    "csd_mean": 1000.0,  # um, mass-weighted Gaussian
    "csd_std": 300.0,  # um
}

RUNTIME = 10 * 3600.0  # s, ~8.5 residence times
STEADY_STATE_TOL = 1.0e-2

# Categorical slots, the same per species as in the flowsheet figures.
SERIES_COLORS = {"F": "#2a78d6", "D": "#eb6834", "E": "#1baf7a",
                 "J": "#8a5cd1"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID_COLOR = "#e5e4e0"


# ---------------------------------------------------------------- streams
def species_names() -> tuple[str, ...]:
    return R02R.database_species(DB_PATH)


def solubility(species: str, temp):
    """Apelblat solubility of ``species`` [kg/kg H] at ``temp`` [K]."""
    a, b, c = SOLUBILITY_APELBLAT[species]
    temp = np.asarray(temp, dtype=float)
    return np.exp(a + b / temp + c * np.log(temp))


def makeup_stream(names=None) -> LiquidStream:
    """R02R's feed with the F taken out, every other mass flow unchanged.

    In the flowsheet all of CR02R's F comes from D01H.
    """
    names = names or species_names()
    _, flow, basis, composition = R02R.FEEDS[0]
    fractions = R02R.molar_to_mass_fraction(composition, DB_PATH)
    without_f = {k: v for k, v in composition.items() if k != "F"}
    return R02R.make_feed(DB_PATH, without_f, flow * (1.0 - fractions["F"]),
                          names, R02R.database_solvent(names), basis)


def gaussian_number_flow(grid, solid_mass_flow, density, mean, std):
    """Crystal number flow [#/(um s)] with a Gaussian mass distribution."""
    grid = np.asarray(grid, dtype=float)
    dx = OneDFVMMechanism._cell_widths(grid)
    number_shape = np.exp(-0.5 * ((grid - mean) / std) ** 2) / grid**3
    mass_per_shape = (density * CRYSTAL_KV * 1e-18
                      * np.sum(number_shape * grid**3 * dx))
    return number_shape * solid_mass_flow / mass_per_shape


def slurry_stream(names, temp, liquid_mass_j, solids, solvent=SOLVENT_NAME):
    """A liquid and one solid per entry of ``solids``, as one stream.

    ``solids`` is a list of (species, grid, number flow [#/(um s)]). As in
    ``continuous_filter.stream_from_flow``, each crystal density is per m**3
    of the whole stream, so a vessel multiplying it by the stream's flow
    reads back that solid's number flow.
    """
    names = list(names)
    liquid_mass_j = np.asarray(liquid_mass_j, dtype=float)
    liquid = LiquidStream(
        str(DB_PATH), mass_flow=float(liquid_mass_j.sum()),
        mass_frac=liquid_mass_j / liquid_mass_j.sum(), temp=temp,
        name_solv=solvent, verbose=False)
    densities = F01R.solid_densities(DB_PATH, [s for s, *_ in solids])

    def crystal_volume(grid, numbers):
        dx = OneDFVMMechanism._cell_widths(np.asarray(grid, dtype=float))
        return CRYSTAL_KV * 1e-18 * float(np.sum(numbers * grid**3 * dx))

    slurry_vol_flow = float(liquid.vol_flow) + sum(
        crystal_volume(grid, numbers) for _, grid, numbers in solids)

    streams = [liquid]
    for species, grid, numbers in solids:
        mass_frac = np.zeros(len(names))
        mass_frac[names.index(species)] = 1.0
        solid = SolidPhase(str(DB_PATH), mass=0, mass_frac=mass_frac,
                           temp=temp, verbose=False)
        mechanism = OneDFVMMechanism(
            solid, target_components=species, solvent_name=solvent,
            x_grid=np.asarray(grid, dtype=float),
            distrib_init=np.asarray(numbers, dtype=float) / slurry_vol_flow,
            density=densities[species], kv=CRYSTAL_KV)
        mechanism.slurry_volume = slurry_vol_flow
        solid.mechanisms = mechanism
        streams.append(solid.to_stream())
    return MixedStream(streams)


def design_feed(names=None) -> MixedStream:
    """D01H's discharge (``DESIGN_FEED``): its liquid, then F crystals."""
    names = names or species_names()
    liquid = np.array([DESIGN_FEED["liquid"].get(n, 0.0) for n in names])
    density = F01R.solid_densities(DB_PATH, ["F"])["F"]
    numbers = gaussian_number_flow(F_SIZE_GRID, DESIGN_FEED["f_crystals"],
                                   density, DESIGN_FEED["csd_mean"],
                                   DESIGN_FEED["csd_std"])
    return slurry_stream(names, DESIGN_FEED["temp"], liquid,
                         [("F", F_SIZE_GRID, numbers)])


# ---------------------------------------------------------------- builder
def reaction_kinetics() -> RxnKinetics:
    """R02R's three reactions, as R02R builds them."""
    names = species_names()
    return RxnKinetics(
        path=str(DB_PATH), rxn_list=R02R.REACTIONS,
        k_params=R02R.RATE_CONSTANTS, ea_params=R02R.ACTIVATION_ENERGIES,
        params_f=R02R.reaction_order_matrix(names),
        temp_ref=R02R.KINETICS_TEMP_REF, delta_hrxn=R02R.HEAT_OF_REACTION)


def size_grid(species: str) -> np.ndarray:
    return np.asarray(F_SIZE_GRID if species == "F" else PRODUCT_SIZE_GRID,
                      dtype=float)


def build_reactor(feed: MixedStream | None = None,
                  makeup: LiquidStream | None = None):
    """CR02R: R02R's vessel and reactions, plus F, D, E and J solids.

    The CrystKinetics convenience setter only links liquid 0 to solid 0, so
    each solid's liquid <-> solid connection is built here as in D01R: a
    forward (nucleation and growth) and a reverse (dissolution) transfer.
    """
    names = species_names()
    liquid_ref = PhaseRef("liquid", 0)
    feed = design_feed(names) if feed is None else feed
    makeup = makeup_stream(names) if makeup is None else makeup

    reactor = ContinuousReactiveCrystallizer(
        integrator=AssimuloBackend(), h_conv=R02R.REACTOR_H_CONV,
        diam=R02R.REACTOR_DIAMETER)

    liquid = LiquidPhase(
        str(DB_PATH), temp=R02R.INITIAL_TEMPERATURE, mass=R02R.REACTOR_MASS,
        mass_frac=R02R.named_mass_fraction(R02R.INITIAL_REACTOR, names,
                                           SOLVENT_NAME),
        name_solv=SOLVENT_NAME)
    solids = []
    for species in SOLID_SPECIES:
        mass_frac = np.zeros(len(names))
        mass_frac[names.index(species)] = 1.0
        solid = SolidPhase(str(DB_PATH), temp=R02R.INITIAL_TEMPERATURE,
                           mass=0.0, mass_frac=mass_frac,
                           name_solv=SOLVENT_NAME)
        grid = size_grid(species)
        solid.mechanisms = OneDFVMMechanism(
            solid, target_components=species, solvent_name=SOLVENT_NAME,
            x_grid=grid, distrib_init=np.zeros(len(grid)), kv=CRYSTAL_KV)
        solids.append(solid)
    reactor.Phases = [liquid] + solids

    connections = []
    for k, species in enumerate(SOLID_SPECIES):
        solid_ref = PhaseRef("solid", k)
        kinetics = CrystKinetics(
            coeff_solub=SOLUBILITY_APELBLAT[species],
            solubility_type="apelblat",
            nucl_prim=NUCL_PRIM[species], nucl_sec=NUCL_SEC[species],
            growth=GROWTH[species], dissolution=DISSOLUTION[species],
            solubility_basis=SOLUBILITY_BASIS)
        pbm = reactor.Phases.get_phase_from_ref(solid_ref) \
            .get_mechanism(PopulationBalanceMechanism)
        pbm.liquid_phase_ref = liquid_ref
        pbm.owning_phase_ref = solid_ref
        pbm.liquid_phase = reactor.Phases.get_phase_from_ref(liquid_ref)
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
    reactor.phase_connections = connections
    reactor.RxnKinetics = reaction_kinetics()
    reactor.Utility = CoolingWater(
        mass_flow=R02R.COOLANT_MASS_FLOW, temp_in=R02R.COOLANT_TEMPERATURE,
        h_conv=R02R.COOLANT_H_CONV)
    reactor.Inlet = [feed, makeup]
    return reactor


def run_reactor(runtime: float = RUNTIME, points: int = 201):
    """Solve CR02R standalone on DESIGN_FEED; returns the reactor."""
    reactor = build_reactor()
    reactor.solve_unit(time_grid=np.linspace(0.0, runtime, points),
                       verbose=False)
    return reactor


# ---------------------------------------------------------------- results
def population_balance(reactor, k: int):
    return reactor.Phases.get_phase_from_ref(PhaseRef("solid", k)) \
        .get_mechanism(PopulationBalanceMechanism)


def outlet(reactor, index=-1) -> SimpleNamespace:
    """The discharge at one saved point [kg/s]: ``dissolved`` (all species)
    and ``crystals`` (per solid), with the temperature [K].

    Each solid's density is per V_liquid / (1 - phi_k) (see
    ``D01R.slurry_volumes``), so with several solids the liquid flow is
    Q / (1 + sum_k phi_k / (1 - phi_k)).
    """
    res = reactor.result
    names = species_names()
    out = reactor.outputs
    vol_flow = float(np.ravel(out["vol_flow"])[index])
    conc = np.maximum(np.atleast_2d(res.mole_conc_liquid0)[index], 0.0)
    mw = np.asarray(reactor.Phases.Liquids[0].mw, dtype=float)
    phis, masses = [], []
    for k in range(len(SOLID_SPECIES)):
        pbm = population_balance(reactor, k)
        n = np.maximum(np.atleast_2d(getattr(res, f"distrib_solid{k}"))[index],
                       0.0)
        vol = pbm.kv * np.asarray(pbm.x_grid)**3 * pbm.dx * 1e-18
        phis.append(float(n @ vol))
        masses.append(float(n @ vol) * pbm.density)
    total = 1.0 + sum(p / (1.0 - p) for p in phis)
    liquid_flow = vol_flow / total
    crystals = {s: m * vol_flow / ((1.0 - p) * total)
                for s, m, p in zip(SOLID_SPECIES, masses, phis)}
    dissolved = dict(zip(names, conc * mw * liquid_flow))
    return SimpleNamespace(temp=float(np.ravel(res.global_temp)[index]),
                           dissolved=dissolved, crystals=crystals)


def fraction_crystallized(out: SimpleNamespace) -> dict[str, float]:
    """Crystal share of each product in the discharge."""
    shares = {}
    for s in CRYSTAL_SPECIES:
        total = out.crystals[s] + out.dissolved[s]
        shares[s] = out.crystals[s] / total if total > 0 else 0.0
    return shares


def history(reactor) -> SimpleNamespace:
    """Discharge profiles: ``time`` [s], ``temp`` [K], ``fraction``
    crystallized per product, ``f_crystals`` leaving [kg/s], and the liquid
    ``saturation`` c / c_sat per solid."""
    time = np.asarray(reactor.result.time, dtype=float)
    names = species_names()
    j_solv = names.index(SOLVENT_NAME)
    outs = [outlet(reactor, i) for i in range(time.size)]
    saturation = {}
    for s in SOLID_SPECIES:
        conc = np.array([o.dissolved[s] / max(o.dissolved[names[j_solv]],
                                              1e-30) for o in outs])
        saturation[s] = conc / solubility(s, [o.temp for o in outs])
    return SimpleNamespace(
        time=time, temp=np.array([o.temp for o in outs]),
        fraction={s: np.array([fraction_crystallized(o)[s] for o in outs])
                  for s in CRYSTAL_SPECIES},
        f_crystals=np.array([o.crystals["F"] for o in outs]),
        saturation=saturation)


def mass_closure(reactor) -> tuple[float, float]:
    """(total fed, total discharged) at the final time [kg/s]."""
    names = species_names()
    fed = sum(float(np.sum(s.mass_flow)) for s in design_feed(names).Streams
              if s.phase_family == "liquid") + DESIGN_FEED["f_crystals"] \
        + float(makeup_stream(names).mass_flow)
    out = outlet(reactor)
    return fed, sum(out.dissolved.values()) + sum(out.crystals.values())


# ---------------------------------------------------------------- figures
def _style(axis):
    axis.grid(True, color=GRID_COLOR, lw=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_profiles(profiles):
    import matplotlib.pyplot as plt

    t = profiles.time / 3600.0
    fig, (ax_f, ax_s, ax_t) = plt.subplots(1, 3, figsize=(15, 4.5))
    for s in CRYSTAL_SPECIES:
        ax_f.plot(t, profiles.fraction[s] * 100, color=SERIES_COLORS[s],
                  lw=2, label=s)
    ax_f.set_ylabel("Crystal share of discharge [%]")
    final = ", ".join(f"{s} {profiles.fraction[s][-1]:.1%}"
                      for s in CRYSTAL_SPECIES)
    ax_f.set_title(f"Fraction crystallized (final {final})", fontsize=11)
    ax_f.legend(title="i", frameon=False)

    for s in SOLID_SPECIES:
        ax_s.plot(t, profiles.saturation[s], color=SERIES_COLORS[s], lw=2,
                  label=s)
    ax_s.axhline(1.0, color=TEXT_MUTED, lw=1, ls="--")
    ax_s.set_ylabel("$c_i / c_{sat,i}$ [-]")
    fed = DESIGN_FEED["f_crystals"]
    ax_s.set_title(f"Saturation ratio (F crystals leaving "
                   f"{profiles.f_crystals[-1] / fed:.2%} of fed)",
                   fontsize=11)
    ax_s.legend(title="i", frameon=False)

    ax_t.plot(t, profiles.temp, color=TEXT_PRIMARY, lw=2)
    ax_t.axhline(R02R.COOLANT_TEMPERATURE, color=TEXT_MUTED, lw=1, ls="--")
    ax_t.set_ylabel("$T$ [K]")
    ax_t.set_title("Reactor temperature")
    for axis in (ax_f, ax_s, ax_t):
        axis.set_xlabel("Time [h]")
        _style(axis)
    fig.suptitle("CR02R - R02R's reactions with F dissolving and D, E, J "
                 "crystallizing")
    fig.tight_layout()
    return fig


def plot_csd(reactor):
    import matplotlib.pyplot as plt

    res = reactor.result
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, s in enumerate(SOLID_SPECIES):
        pbm = population_balance(reactor, k)
        x, dx = np.asarray(pbm.x_grid), pbm.dx
        mass = np.asarray(getattr(res, f"distrib_solid{k}"))[-1] * x**3 * dx
        total = mass.sum()
        if total <= 0:
            continue
        w = mass / total / dx
        ax.plot(x, w * 100, color=SERIES_COLORS[s], lw=2,
                label=f"{s}  (mean {np.sum(w * x * dx):.0f} um)")
    ax.set_xlim(0, PRODUCT_SIZE_GRID[-1])
    ax.set_xlabel("Crystal size [um]")
    ax.set_ylabel("Mass fraction [% per um]")
    ax.set_title("Crystal size distribution by mass, end of run")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def settings_rows(reactor, profiles):
    fed, out = mass_closure(reactor)
    rows = [
        ("Vessel", "Reactions, jacket, charge", "R02R's (see R02R.py)"),
        ("Feeds", "D01H slurry",
         f"{sum(DESIGN_FEED['liquid'].values()) * 1e3:.4g} g/s liquid + "
         f"{DESIGN_FEED['f_crystals'] * 1e3:.4g} g/s F crystals "
         f"(~{DESIGN_FEED['csd_mean']:.0f} um) at {DESIGN_FEED['temp']} K"),
        ("Feeds", "Makeup",
         f"R02R's feed without F, {float(makeup_stream().mass_flow) * 1e3:.4g}"
         " g/s"),
        ("Solubility", "Model", "ln c_sat = a + b/T + c ln T [kg/kg H]"),
    ]
    temp = profiles.temp[-1]
    for s in SOLID_SPECIES:
        a, b, c = SOLUBILITY_APELBLAT[s]
        rows.append(("Solubility", s, f"a {a:g}, b {b:g}, c {c:g}; "
                     f"{float(solubility(s, temp)):.4g} kg/kg H at "
                     f"{temp:.1f} K"))
    for s in SOLID_SPECIES:
        rows.append((f"Kinetics {s}", "Nucl. / growth / dissol.",
                     f"{NUCL_PRIM[s]} / {GROWTH[s]} / {DISSOLUTION[s]}"))
    rows += [
        ("Run", "Runtime", f"{profiles.time[-1] / 3600:.1f} h"),
        ("Run", "Fraction crystallized",
         ", ".join(f"{s} {profiles.fraction[s][-1]:.1%}"
                   for s in CRYSTAL_SPECIES)),
        ("Run", "Mass in / out", f"{fed * 1e3:.4f} / {out * 1e3:.4f} g/s "
                                 f"({(fed - out) / fed:.2%} gap)"),
    ]
    return rows


def plot_settings(reactor, profiles):
    rows, previous = [], None
    for group, setting, value in settings_rows(reactor, profiles):
        rows.append([group if group != previous else "", setting, value])
        previous = group
    return D01R._table_figure(["Group", "Setting", "Value"], rows,
                              "CR02R settings", col_widths=[0.14, 0.22, 0.64],
                              width=11)


def save_figures(reactor, out_dir: Path = RESULTS_DIR, profiles=None):
    profiles = history(reactor) if profiles is None else profiles
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {"CR02R_profiles": plot_profiles(profiles),
               "CR02R_csd": plot_csd(reactor),
               "CR02R_settings": plot_settings(reactor, profiles)}
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


def print_report(reactor):
    out = outlet(reactor)
    solvent = out.dissolved[SOLVENT_NAME]
    print(f"CR02R at t = {reactor.result.time[-1] / 3600:.1f} h, "
          f"T = {out.temp:.2f} K\n")
    print(f"  {'solid':<6}{'liquid':>10}{'c_sat':>10}{'dissolved':>11}"
          f"{'crystals':>10}{'crystallized':>14}")
    print(f"  {'':<6}{'[kg/kg H]':>10}{'[kg/kg H]':>10}{'[g/s]':>11}"
          f"{'[g/s]':>10}")
    shares = fraction_crystallized(out)
    for s in SOLID_SPECIES:
        share = (f"{shares[s]:.1%}" if s in shares else
                 f"({out.crystals[s] / DESIGN_FEED['f_crystals']:.2%} of fed)")
        print(f"  {s:<6}{out.dissolved[s] / solvent:>10.5f}"
              f"{float(solubility(s, out.temp)):>10.5f}"
              f"{out.dissolved[s] * 1e3:>11.5f}{out.crystals[s] * 1e3:>10.5f}"
              f"  {share:>12}")
    fed, discharged = mass_closure(reactor)
    print(f"\n  mass in {fed * 1e3:.4f} g/s, out {discharged * 1e3:.4f} g/s "
          f"({(fed - discharged) / fed:.2%} gap)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    parser.add_argument("--runtime", type=float, default=RUNTIME)
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    cr02r = run_reactor(args.runtime)
    print_report(cr02r)
    save_figures(cr02r)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
