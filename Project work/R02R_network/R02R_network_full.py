"""Isothermal R02R network CSTR with the full DFT stoichiometry.

``R02R_network.py`` treats every step as ``X --> Y``, first order in X. This
version writes each step with the co-reactants, co-products and catalysts
from the DFT mechanism, so

* rates depend on the bulk species (acetyl nitrate, HNO3, NH4NO3, AcOH ...)
  as well as the network species, and
* the mass balance closes (apart from the database MW issues listed below).

Everything else - feed, vessel, residence time, isothermal operation - is
``R02R_network.py`` unchanged.

Species names
-------------
The database keeps its letter names for the bulk species:

    A = HNO3   B = H2O   G = Ac2O   H = AcOH (solvent)   I = NH4NO3
    K = AcONO2 (acetyl nitrate)

and ``compound_database_network.json`` adds Hplus (H+), NO3minus (NO3-),
CH2O and NH2CH2COOCH3 with DUMMY properties (MW is real). PharmaPy does not
track charge, so the ions are ordinary species.

How the DFT table was turned into reactions
-------------------------------------------
* A species listed on both sides is a **catalyst**: it is in the rate law
  but not consumed (NH4NO3, AcOH, AcONO2 in several steps).
* A species listed only as consumed, where the mass balance says it is not
  consumed, is also treated as a catalyst (HNO3 in step 15, NH4NO3 in steps
  18 and 21, NO2 in step 20).
* "NO2 + OH" (steps 12-14) is read as HNO3, and the lone NO2 in step 20 as
  HNO3 too: neither NO2 nor OH is made anywhere else in the network.
* Steps 1-5 get the ion the mass balance is missing: NO3- out of steps 1-2,
  H+ out of steps 3-5.
* HNO3 + Ac2O --> AcONO2 + AcOH (acetyl nitrate formation) has no reported
  barrier; it is treated as fast. The two AcONO2 --> NO2+ + AcO- rows are
  left out: nothing in the network consumes NO2+ or AcO-, so they would
  only drain acetyl nitrate.
* Database MWs that do not balance (left as they are, see ``check_mass``):
  INT3 is 2.016 below steps 9/10, INT5 is 14.03 below steps 12/14, and Z4
  is ~3 below step 13. All three carry negligible flux at 60 C.

Rate law and rate constants
---------------------------
Each step is elementary: first order in every species in its transition
state (consumed species and catalysts), with an Eyring rate constant at
1 mol/L standard state::

    k = (k_B T / h) * exp(-Ea / RT)        [(L/mol)^(n-1) / s]

where n is the molecularity. The reported barriers are SMD electronic
energies with ZPE (E + ZPE), not free energies, so the activation entropy
is missing. For a step with several molecules in the transition state that
entropy loss would *raise* the free-energy barrier, so these k are upper
bounds. ``TS_ENTROPY_PENALTY`` adds a fixed kJ/mol per extra molecule if
you want to test that.

Fast steps are capped as in ``R02R_network.py``: for each step's primary
(network) species, the pseudo-first-order loss at the bulk concentrations
is capped at ``K_MAX``, scaling that species' branches together so the
branching is kept. The bulk concentrations used are the solved steady-state
ones (``run_reactor`` iterates; see there), so the cap never throttles a
step below what its Eyring k gives at the conditions actually reached.

Acetyl nitrate formation has no barrier, so it runs at the cap. That sets
how little free HNO3 is left, and HNO3 catalyses Z2 --> Z5 and is consumed
by the D route (steps 12-14). Results that depend on HNO3 therefore depend
on this unreported step; a measured rate or equilibrium constant for
HNO3 + Ac2O <=> AcONO2 + AcOH would pin it down.

Run it with::

    python R02R_network_full.py            # solve, save figures, show them
    python R02R_network_full.py --no-show  # solve and save figures only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import R02R_network as net
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.ThermoModule import ParseDatabase

HERE = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = (HERE.parents[1] / "compounddatabase"
                         / "compound_database_network.json")
RESULTS_DIR = net.RESULTS_DIR

DISPLAY = {"A": "HNO3", "B": "H2O", "G": "Ac2O", "H": "AcOH",
           "I": "NH4NO3", "K": "AcONO2", "Hplus": "H+", "NO3minus": "NO3-"}

# ---------------------------------------------------------------- reactions
# Each step: consumed {species: coeff}, produced {species: coeff},
# catalysts (in the rate law, not consumed), Ea [kJ/mol] (None = not
# reported, treated as fast), and a note on any change from the DFT table.
STEPS = [
    dict(n=1, consumes={"Z1": 1, "K": 1}, produces={"INT1": 1, "H": 1,
         "NO3minus": 1}, catalysts=("I",), ea=48.8,
         note="NO3- added"),
    dict(n=2, consumes={"F": 1}, produces={"INT1": 1, "NO3minus": 1},
         catalysts=("H", "I"), ea=48.8, note="NO3- added"),
    dict(n=3, consumes={"INT1": 1, "A": 1},
         produces={"INT1ONO2": 1, "Hplus": 1}, catalysts=(), ea=0.0,
         note="H+ added"),
    dict(n=4, consumes={"INT1": 1, "B": 1},
         produces={"INT1OH": 1, "Hplus": 1}, catalysts=(), ea=0.0,
         note="H+ added"),
    dict(n=5, consumes={"INT1": 1, "H": 1},
         produces={"INT1AcO": 1, "Hplus": 1}, catalysts=(), ea=0.0,
         note="H+ added"),
    dict(n=6, consumes={"INT1ONO2": 1}, produces={"Z2": 1, "CH2O": 1},
         catalysts=("K",), ea=147.2, note=""),
    dict(n=7, consumes={"INT1OH": 1, "K": 1},
         produces={"Z2": 1, "CH2O": 1, "H": 1}, catalysts=("I",),
         ea=65.2, note=""),
    dict(n=8, consumes={"INT1AcO": 1, "K": 1, "I": 1},
         produces={"Z2": 1, "H": 1, "A": 1, "NH2CH2COOCH3": 1},
         catalysts=(), ea=43.5, note=""),
    dict(n=9, consumes={"INT1AcO": 1, "K": 1}, produces={"INT3": 1},
         catalysts=(), ea=88.7, note="INT3 MW low"),
    dict(n=10, consumes={"INT3": 1, "K": 1}, produces={"INT4": 1},
         catalysts=(), ea=12.1, note="INT3 MW low"),
    dict(n=11, consumes={"INT4": 1, "K": 1}, produces={"Z3": 1, "BP1": 1},
         catalysts=(), ea=33.1, note=""),
    dict(n=12, consumes={"Z2": 1, "A": 1}, produces={"INT5": 1},
         catalysts=("H",), ea=118.9,
         note="NO2+OH read as HNO3"),
    dict(n=13, consumes={"INT5": 1, "A": 1}, produces={"Z4": 1, "BP2": 1},
         catalysts=("H",), ea=248.9,
         note="NO2+OH read as HNO3"),
    dict(n=14, consumes={"INT5": 1, "A": 1}, produces={"D": 1, "BP2": 1},
         catalysts=("H",), ea=123.6,
         note="NO2+OH read as HNO3"),
    dict(n=15, consumes={"Z2": 1, "K": 1}, produces={"Z5": 1},
         catalysts=("I", "A"), ea=73.3, note="HNO3 catalyst (mass)"),
    dict(n=16, consumes={"Z5": 1, "K": 1}, produces={"Z6": 1},
         catalysts=(), ea=166.0, note=""),
    dict(n=17, consumes={"Z5": 1, "H": 1}, produces={"INT2OH": 1, "G": 1},
         catalysts=(), ea=206.9, note=""),
    dict(n=18, consumes={"Z5": 1, "K": 1},
         produces={"E": 1, "G": 1, "CH2O": 1}, catalysts=("I",),
         ea=140.9, note="NH4NO3 catalyst (mass)"),
    dict(n=19, consumes={"INT2OH": 1, "G": 1},
         produces={"Z7": 1, "H": 1, "CH2O": 1}, catalysts=(),
         ea=116.81, note=""),
    dict(n=20, consumes={"INT2OH": 1, "K": 1},
         produces={"INT2ONO2": 1, "H": 1}, catalysts=("A",), ea=147.2,
         note="NO2 → HNO3 catalyst (mass)"),
    dict(n=21, consumes={"INT2OH": 1, "K": 1},
         produces={"E": 1, "H": 1, "CH2O": 1}, catalysts=("I",), ea=129.0,
         note="NH4NO3 catalyst (mass)"),
    dict(n=22, consumes={"INT2ONO2": 1}, produces={"E": 1, "CH2O": 1},
         catalysts=("K",), ea=183.1, note=""),
    dict(n=23, consumes={"A": 1, "G": 1}, produces={"K": 1, "H": 1},
         catalysts=(), ea=None,
         note="no Ea reported: fast"),
]

OPERATING_TEMPERATURE = net.OPERATING_TEMPERATURE
K_MAX = net.K_MAX  # 1/s, cap on a species' pseudo-first-order loss
TS_ENTROPY_PENALTY = 0.0  # kJ/mol per transition-state molecule beyond 1
# Upper-bound bulk concentrations [mol/L] used only to size the cap: the
# quoted feed values, with AcOH allowed to grow by the HNO3 it can make.
C_REF = {"A": 3.841, "G": 5.276, "H": 5.938 + 3.841, "I": 1.110,
         "K": 3.841}
C_REF_DEFAULT = 1.0  # mol/L, any other co-reactant (e.g. water)
BULK_SPECIES = ("A", "B", "G", "H", "I", "K")  # co-reactants/catalysts
C_REF_FLOOR = 1e-3  # mol/L, so an absent species (water) still sizes a cap
CAP_ITERATIONS = 6
RUNTIME = net.RUNTIME


def step_label(step) -> str:
    def side(terms):
        return " + ".join((f"{c:g} " if c != 1 else "") + DISPLAY.get(s, s)
                          for s, c in terms.items())
    cats = (" [cat: " + ", ".join(DISPLAY.get(c, c) for c in step["catalysts"])
            + "]") if step["catalysts"] else ""
    return f"{side(step['consumes'])} --> {side(step['produces'])}{cats}"


def molecularity(step) -> int:
    return int(sum(step["consumes"].values())) + len(step["catalysts"])


def participating_species() -> list[str]:
    names = []
    for step in STEPS:
        for name in [*step["consumes"], *step["produces"], *step["catalysts"]]:
            if name not in names:
                names.append(name)
    return names


def eyring_rate_constants(temp: float) -> np.ndarray:
    """Uncapped Eyring k [(L/mol)^(n-1)/s]; unreported Ea taken as 0."""
    prefactor = net.BOLTZMANN * temp / net.PLANCK
    ea = np.array([(s["ea"] or 0.0)
                   + TS_ENTROPY_PENALTY * (molecularity(s) - 1)
                   for s in STEPS]) * 1e3
    return prefactor * np.exp(-ea / (net.GAS_CONSTANT * temp))


def pseudo_first_order(k: np.ndarray,
                       c_ref: dict[str, float] | None = None) -> np.ndarray:
    """k times the reference concentration of every TS partner of the
    step's primary (first-listed) species, in 1/s."""
    c_ref = C_REF if c_ref is None else c_ref
    out = np.empty_like(k)
    for i, (step, k_i) in enumerate(zip(STEPS, k)):
        primary = next(iter(step["consumes"]))
        partners = [s for s in step["consumes"] if s != primary]
        partners += list(step["catalysts"])
        out[i] = k_i * np.prod([c_ref.get(s, C_REF_DEFAULT)
                                for s in partners])
    return out


def rate_constants(temp: float = OPERATING_TEMPERATURE,
                   k_max: float = K_MAX,
                   c_ref: dict[str, float] | None = None) -> np.ndarray:
    """Eyring k with each primary species' total loss capped at k_max,
    sized at the bulk concentrations ``c_ref`` (default C_REF)."""
    k = eyring_rate_constants(temp)
    k_eff = pseudo_first_order(k, c_ref)
    primaries = [next(iter(s["consumes"])) for s in STEPS]
    total = {p: 0.0 for p in primaries}
    for p, ke in zip(primaries, k_eff):
        total[p] += ke
    return k * np.array([min(1.0, k_max / total[p]) if total[p] > 0 else 1.0
                         for p in primaries])


def check_mass(database_path: Path = DEFAULT_DATABASE_PATH) -> list[float]:
    """Product minus reactant mass [g/mol] for each step (0 = balanced)."""
    database = ParseDatabase(str(database_path), to_arrays=False)
    mw = dict(zip(database["name_species"], database["mw"]))
    return [sum(c * mw[s] for s, c in st["produces"].items())
            - sum(c * mw[s] for s, c in st["consumes"].items())
            for st in STEPS]


# ---------------------------------------------------------------- reactor
def build_kinetics(database_path: Path, temp: float,
                   c_ref: dict[str, float] | None = None) -> RxnKinetics:
    """RxnKinetics from an explicit stoichiometric matrix.

    PharmaPy's reaction-string parser cannot express a catalyst (a species
    on both sides ends up with +1), and its elementary rate law only gives
    orders to net reactants. So the matrix is built here and the order
    matrix is set afterwards, with order 1 for every consumed species and
    every catalyst.
    """
    partic = participating_species()
    stoich = np.zeros((len(STEPS), len(partic)))
    for i, step in enumerate(STEPS):
        for s, c in step["consumes"].items():
            stoich[i, partic.index(s)] -= c
        for s, c in step["produces"].items():
            stoich[i, partic.index(s)] += c

    kinetics = RxnKinetics(
        path=str(database_path),
        stoich_matrix=stoich,
        partic_species=partic,
        k_params=rate_constants(temp, c_ref=c_ref),
        ea_params=np.array([(s["ea"] or 0.0) for s in STEPS]) * 1e3,
        temp_ref=temp,
        delta_hrxn=np.zeros(len(STEPS)),
    )
    orders = np.zeros((len(STEPS), len(kinetics.partic_species)))
    for i, step in enumerate(STEPS):
        for s, c in step["consumes"].items():
            orders[i, kinetics.partic_species.index(s)] = c
        for s in step["catalysts"]:
            orders[i, kinetics.partic_species.index(s)] = 1.0
    kinetics.params_f = orders
    return kinetics


def build_reactor(database_path: Path | str = DEFAULT_DATABASE_PATH,
                  temp: float = OPERATING_TEMPERATURE,
                  c_ref: dict[str, float] | None = None) -> ContinuousReactor:
    database_path = Path(database_path)
    species = net.database_species(database_path)
    solvent = net.database_solvent(species)
    reactor = ContinuousReactor(integrator=ScipyBackend(), isothermal=True,
                                diam=net.REACTOR_DIAMETER)
    reactor.Phases = LiquidPhase(
        str(database_path), temp=temp,
        mass_frac=net.named_mass_fraction(net.INITIAL_REACTOR, species,
                                          solvent),
        mass=net.REACTOR_MASS, name_solv=solvent)
    reactor.RxnKinetics = build_kinetics(database_path, temp, c_ref)
    reactor.Inlet = net.make_feeds(database_path)
    return reactor


def run_reactor(database_path: Path | str = DEFAULT_DATABASE_PATH,
                runtime: float = RUNTIME,
                temp: float = OPERATING_TEMPERATURE,
                max_iter: int = CAP_ITERATIONS,
                verbose: bool = False) -> ContinuousReactor:
    """Solve, re-sizing the cap at the steady-state bulk concentrations.

    The cap is first sized at C_REF (feed values). Bulk concentrations at
    steady state can be far from those (HNO3 is almost all converted to
    acetyl nitrate), and a cap sized for 3.8 mol/L HNO3 throttles an
    HNO3-catalysed step ~350x too hard. So the run is repeated with C_REF
    replaced by the solved concentrations (floored at C_REF_FLOOR) until the
    rate constants stop changing by more than 10%.
    """
    c_ref = dict(C_REF)
    k_prev = rate_constants(temp, c_ref=c_ref)
    for iteration in range(1, max_iter + 1):
        reactor = build_reactor(database_path, temp, c_ref)
        reactor.solve_unit(runtime=runtime, verbose=False)
        conc = final_concentrations(reactor, database_path)
        c_ref = {s: max(conc[s], C_REF_FLOOR) for s in BULK_SPECIES}
        k_new = rate_constants(temp, c_ref=c_ref)
        change = float(np.max(np.abs(np.log(k_new / k_prev))))
        if verbose:
            print(f"  cap iteration {iteration}: max k change "
                  f"{np.exp(change):.3g}x")
        reactor.cap_reference = c_ref
        if change < np.log(1.1):
            break
        k_prev = k_new
    return reactor


def final_concentrations(reactor, database_path=DEFAULT_DATABASE_PATH):
    """Final molar concentration of every species [mol/L]."""
    database = ParseDatabase(str(database_path), to_arrays=False)
    mw = dict(zip(database["name_species"], database["mw"]))
    mass = np.asarray(reactor.result.mass_j_liquid0, dtype=float)[-1]
    vol = float(np.ravel(reactor.result.vessel_vol)[-1])
    return {n: max(m, 0.0) / mw[n] / vol
            for n, m in zip(reactor.name_species, mass)}


def molar_yields(reactor, balances, products=("D", "E", "Z5", "Z3",
                                               "INT1ONO2"),
                 database_path=DEFAULT_DATABASE_PATH) -> dict[str, float]:
    """Outlet molar flow of each product per mol Z1 fed."""
    database = ParseDatabase(str(database_path), to_arrays=False)
    mw = dict(zip(database["name_species"], database["mw"]))
    idx = {n: j for j, n in enumerate(reactor.name_species)}
    z1_in = balances.inlet[-1][idx["Z1"]] / mw["Z1"]
    return {p: max(balances.outlet[-1][idx[p]], 0.0) / mw[p] / z1_in
            for p in products}


# ---------------------------------------------------------------- figures
def plot_profiles(reactor, database_path=DEFAULT_DATABASE_PATH):
    """Concentration vs time on a log scale: network and bulk species."""
    import matplotlib.pyplot as plt

    database = ParseDatabase(str(database_path), to_arrays=False)
    mw = dict(zip(database["name_species"], database["mw"]))
    time_min = np.asarray(reactor.result.time) / 60.0
    mass = np.asarray(reactor.result.mass_j_liquid0, dtype=float)
    vol = np.ravel(reactor.result.vessel_vol)
    conc = {n: np.maximum(mass[:, j], 0) / mw[n] / vol
            for j, n in enumerate(reactor.name_species)}
    network = [n for n in net.network_species() if conc[n].max() > 1e-8]
    bulk = [n for n in ("A", "G", "K", "I", "Hplus", "NO3minus", "CH2O",
                        "NH2CH2COOCH3") if conc[n].max() > 1e-8]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, names, title in ((axes[0], network, "Network species"),
                             (axes[1], bulk, "Bulk reagents and by-products")):
        for i, n in enumerate(names):
            ax.plot(time_min, np.maximum(conc[n], 1e-12),
                    color=net.SERIES_COLORS[i % 10],
                    ls="-" if i < 10 else "--", lw=2,
                    label=DISPLAY.get(n, n))
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 20)
        ax.set_xlabel("Time [min]")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, color="#e5e4e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 1),
                  loc="upper left")
    axes[0].set_ylabel("Concentration [mol/L]")
    fig.suptitle("R02R network, full DFT stoichiometry - isothermal CSTR at "
                 f"{OPERATING_TEMPERATURE - 273.15:g} °C", x=0.05, ha="left")
    fig.tight_layout()
    return fig


def plot_kinetics(database_path=DEFAULT_DATABASE_PATH, c_ref=None):
    """Kinetics table; k_eff uses ``c_ref`` (the run's final cap basis)."""
    tau = net.REACTOR_MASS / net.total_feed_flow()
    k_raw = eyring_rate_constants(OPERATING_TEMPERATURE)
    k_used = rate_constants(c_ref=c_ref)
    k_eff = pseudo_first_order(k_used, c_ref)
    dmass = check_mass(database_path)
    units = {1: "1/s", 2: "L/mol/s", 3: "L²/mol²/s", 4: "L³/mol³/s"}
    rows = []
    for st, kr, ku, ke, dm in zip(STEPS, k_raw, k_used, k_eff, dmass):
        flag = ("capped" if ku < kr * (1 - 1e-12)
                else "inactive" if ke * tau < 1e-2 else "")
        rows.append([str(st["n"]), step_label(st),
                     "fast" if st["ea"] is None else f"{st['ea']:g}",
                     f"{ku:.2e} {units[molecularity(st)]}",
                     f"{ke * tau:.2g}", flag,
                     "" if abs(dm) < 0.01 else f"{dm:+.2f}", st["note"]])
    return net._table_figure(
        rows, ["#", "Step", "Ea", "k used", "k_eff·τ", "", "Δm", "Note"],
        f"Full-stoichiometry kinetics at {OPERATING_TEMPERATURE:g} K "
        f"(τ = {tau:g} s; k_eff at steady-state bulk concentrations)",
        col_widths=[0.03, 0.42, 0.05, 0.13, 0.07, 0.06, 0.05, 0.19],
        footnote="Eyring with E+ZPE barriers (entropy omitted, so k is an "
                 "upper bound for multi-molecule steps); Δm = product - "
                 "reactant mass [g/mol] where the database MWs do not balance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    database = Path(os.environ.get("PHARMAPY_DATABASE",
                                   DEFAULT_DATABASE_PATH))
    reactor = run_reactor(database, verbose=True)
    balances = net.replay_balances(reactor)
    conc = final_concentrations(reactor, database)

    print("Steady-state concentrations [mol/L] (> 1e-12):")
    for name, c in sorted(conc.items(), key=lambda kv: -kv[1]):
        if c > 1e-12 and name != "H":
            print(f"  {DISPLAY.get(name, name):<14}{c:.4g}")
    print("\nMolar yield per mol Z1 fed:")
    for name, y in molar_yields(reactor, balances,
                                database_path=database).items():
        print(f"  {name:<10}{y:.4g}")
    gen = balances.generation[-1].sum() * 1e3
    print(f"\nNet mass generated: {gen:+.2e} g/s "
          f"(feed {balances.inlet[-1].sum() * 1e3:.2f} g/s)")
    gap = net.steady_state_gap(balances)
    print(f"Max accumulation: {gap:.2%} of inlet flow")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {"R02R_network_full_profiles": plot_profiles(reactor, database),
               "R02R_network_full_kinetics": plot_kinetics(
                   database, reactor.cap_reference)}
    for name, fig in figures.items():
        fig.savefig(RESULTS_DIR / f"{name}.png", dpi=150,
                    bbox_inches="tight")
    print(f"Figures saved to {RESULTS_DIR}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
