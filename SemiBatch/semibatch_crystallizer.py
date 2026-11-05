"""Reactive semibatch crystallizer paired with ``semibatch_reactor.py``.

The setup is::

    FEED_A ------+
    FEED_G ------+--> SBC (4 g I made up to 17 mL with H) --> slurry
    FEED_C_in_H -+

``SBC`` is ``PharmaPy.Crystallizers_Refactored.SemiBatchCrystallizer``. It
takes every input from ``semibatch_reactor``: the same charge (I in H), the
same three linear feeds over 37 min, the same hold, vessel and thermal mode
(70 C isothermal by default), and the reactor's selected rate law (``sbr.KINETICS_MODEL``, R02R
by default; ``--kinetics`` here too) as a liquid reaction region. The
only addition is that the products D, E and J can crystallize.

Each of D, E and J has its own solid phase and 1D-FVM population balance,
with the nucleation, growth and dissolution kinetics of
``Project work/D01R/D01R.py``. D01R's solubility is::

    c_sat,i = S0_i * exp(-K_B * [B])

There is no B here, so each solubility is the B-free S0_i: D 0.0284,
E 0.01505, J 0.0886 mol/L, independent of temperature. The products form
at far higher concentrations, so most of what forms crystallizes.

Supersaturation is relative, ``sigma_i = (c_i - c_sat,i) / c_sat,i``, as in
D01R.

Known issue: a few percent of the product (about 4 % with R02R kinetics,
7 % with second order, at the default settings) does not close the mole
balance. PharmaPy dilutes the crystal number density by the feed's volume
flow, which overstates the vessel's growth here (the reactions destroy
mass and liquid becomes denser crystal). See ``unaccounted_product``.

Run it with::

    python semibatch_crystallizer.py              # solve, save figures, show
    python semibatch_crystallizer.py --no-show    # solve and save figures only

Figures are written to ``results/`` next to this file (with a
``_first_order`` or ``_second_order`` suffix for those models):

    SBC_profiles.png      species moles (liquid and solid), T, vessel contents
    SBC_normalized.png    the same totals over their peaks and over C fed
    SBC_crystals.png      D, E, J dissolved vs crystallized, and final CSDs
    SBC_settings.png      the settings and initial conditions used
    SBC_mass_balance.png  charged, dosed, reacted and final moles per species

PharmaPy needs the conda environment ``PharmaPy`` (activate it first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PharmaPy.Crystallizers_Refactored import SemiBatchCrystallizer
from PharmaPy.DataClasses import PhaseConnection, PhaseRef
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import (
    OneDFVMMechanism,
    PopulationBalanceMechanism,
    ReversibleTransferMechanism,
)
from PharmaPy.Phases_Refactored import SolidPhase

import semibatch_reactor as sbr

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
D01R_DIR = HERE.parent / "Project work" / "D01R"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(D01R_DIR))

import D01R  # noqa: E402

DB_PATH = sbr.DB_PATH

# ---------------------------------------------------------------- settings
# Charge, dose, runtime and vessel all come from semibatch_reactor; the
# crystallization settings come from D01R.
CRYSTAL_SPECIES = D01R.CRYSTAL_SPECIES  # ("D", "E", "J")
NUCL_PRIM = D01R.NUCL_PRIM
NUCL_SEC = D01R.NUCL_SEC
GROWTH = D01R.GROWTH
DISSOLUTION = D01R.DISSOLUTION
SOLUBILITY_S0 = D01R.SOLUBILITY_S0  # mol/L; no B, so c_sat = S0
CRYSTAL_SIZE_GRID = D01R.CRYSTAL_SIZE_GRID  # um
CRYSTAL_KV = D01R.CRYSTAL_KV
# Warn when more than this share of the product made is neither dissolved
# nor crystal (see unaccounted_product). Crystals stay below ~250 um here,
# well inside the grid.
LOSS_TOL = 1.0e-2

TEXT_MUTED = sbr.TEXT_MUTED


# ---------------------------------------------------------------- builders
def build_crystallizer(integrator=None) -> SemiBatchCrystallizer:
    """The semibatch reactor's vessel with D, E and J solid phases added.

    As in D01R, each species' liquid <-> solid connection is built by hand
    (the CrystKinetics setter only links liquid 0 to solid 0): a forward
    (nucleation and growth) and a reverse (dissolution) transfer, both
    driven by that species' population balance.
    """
    names = sbr.species_names()
    solvent = sbr.solvent_name()
    liquid_ref = PhaseRef("liquid", 0)

    tank = SemiBatchCrystallizer(integrator=integrator or ScipyBackend(),
                                 **sbr.vessel_kwargs())

    liquid = sbr.make_charge()
    solids = []
    for species in CRYSTAL_SPECIES:
        mass_frac = np.zeros(len(names))
        mass_frac[names.index(species)] = 1.0
        solid = SolidPhase(str(DB_PATH), temp=sbr.INITIAL_TEMPERATURE,
                           mass=0.0, mass_frac=mass_frac, name_solv=solvent)
        solid.mechanisms = OneDFVMMechanism(
            solid, target_components=species, solvent_name=solvent,
            x_grid=CRYSTAL_SIZE_GRID,
            distrib_init=np.zeros(len(CRYSTAL_SIZE_GRID)), kv=CRYSTAL_KV)
        solids.append(solid)
    tank.Phases = [liquid] + solids

    connections = []
    for k, species in enumerate(CRYSTAL_SPECIES):
        solid_ref = PhaseRef("solid", k)
        kinetics = CrystKinetics(
            solub_fn=D01R.make_solubility_function(names, species),
            nucl_prim=NUCL_PRIM[species], nucl_sec=NUCL_SEC[species],
            growth=GROWTH[species], dissolution=DISSOLUTION[species],
            solubility_basis="mole_per_volume_solution")
        pbm = population_balance(tank, k)
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
    tank.intraphase_processes = sbr.reaction_processes(
        tank.Phases.get_phase_from_ref(liquid_ref))
    tank.nomenclature(overwrite=True)

    utility = sbr.make_utility()
    if utility is not None:
        tank.Utility = utility
    tank.Inlet = sbr.make_dose()
    return tank


def run_crystallizer(runtime: float = sbr.RUNTIME) -> SemiBatchCrystallizer:
    """Solve the semibatch crystallizer and return it with ``result``."""
    tank = build_crystallizer()
    tank.solve_unit(runtime=runtime, verbose=False)
    return tank


# ---------------------------------------------------------------- results
def population_balance(tank, k: int):
    return tank.Phases.get_phase_from_ref(PhaseRef("solid", k)) \
        .get_mechanism(PopulationBalanceMechanism)


def crystal_weight(tank, k: int) -> np.ndarray:
    """Number density [#/m**3/um] -> crystal mass per m**3 slurry, per cell."""
    pbm = population_balance(tank, k)
    return (pbm.density * pbm.kv * np.asarray(pbm.x_grid)**3 * pbm.dx
            * 1e-18)


def crystal_moles(tank) -> dict[str, np.ndarray]:
    """Moles of each crystallizing species held as crystals, over time.

    Each population balance is a number density per m**3 of its own slurry
    volume, which D01R.slurry_volumes recovers from the vessel volume.
    """
    res = tank.result
    mw = sbr.molecular_weights()
    distribs = [np.asarray(getattr(res, f"distrib_solid{k}"), dtype=float)
                for k in range(len(CRYSTAL_SPECIES))]
    volumes = D01R.slurry_volumes(tank, distribs, np.ravel(res.vessel_vol))
    return {species: distribs[k] @ crystal_weight(tank, k) * volumes[k]
            / (mw[species] * 1e-3)
            for k, species in enumerate(CRYSTAL_SPECIES)}


def total_moles(tank) -> dict[str, np.ndarray]:
    """Liquid plus crystal moles of every species, over time."""
    totals = dict(sbr.mole_history(tank))
    for species, n in crystal_moles(tank).items():
        totals[species] = totals[species] + n
    return totals


def mole_balance(tank) -> SimpleNamespace:
    """sbr.mole_balance with crystals counted in ``final``, plus the split.

    ``liquid`` and ``solid`` split ``final`` for the crystallizing species.
    """
    time = float(np.asarray(tank.result.time)[-1])
    liquid = {name: float(n[-1]) for name, n in
              sbr.mole_history(tank).items()}
    solid = {name: float(n[-1]) for name, n in crystal_moles(tank).items()}
    final = {name: liquid[name] + solid.get(name, 0.0) for name in liquid}
    charge = sbr.charge_moles()
    charged = {name: charge.get(name, 0.0) for name in final}
    dosed_now = sbr.dosed_moles(time)
    dosed = {name: float(dosed_now.get(name, 0.0)) for name in final}
    reacted = {name: final[name] - charged[name] - dosed[name]
               for name in final}
    # F, the second-order intermediate, has taken its C (0 otherwise).
    products = sum(reacted[name] for name in CRYSTAL_SPECIES) + reacted["F"]
    return SimpleNamespace(time=time, charged=charged, dosed=dosed,
                           reacted=reacted, final=final, liquid=liquid,
                           solid=solid, closure=-reacted["C"] - products)


def unaccounted_product(tank) -> dict[str, float]:
    """Product the reactions made that is neither dissolved nor crystal.

    Every reaction path consumes one C and makes one D, E or J (via the
    intermediate F in the second-order model), so C consumed minus
    D + E + J + F held should be ~0. It is not, for two reasons:

    * crystals that grow past the last size cell leave the population
      balance (the CSD plot shows whether any reach it);
    * PharmaPy's ``add_dilution_terms`` dilutes the number density by the
      feed's volumetric inflow, but here the vessel grows by less than
      that: the reactions destroy mass and the liquid becomes denser
      crystals. The population is over-diluted and crystals are lost.

    Refining the size grid does not change this gap, so at these settings
    it is the dilution, not the grid.
    """
    balance = mole_balance(tank)
    made = -balance.reacted["C"]
    held = (sum(balance.final[name] for name in CRYSTAL_SPECIES)
            + balance.final["F"])
    return {"lost_mol": made - held,
            "share": (made - held) / made if made > 0 else 0.0}


# ---------------------------------------------------------------- plotting
def plot_profiles(tank):
    """The reactor's profile figure, with species totals (liquid + solid)."""
    fig = sbr.plot_profiles(tank, title=(
        f"Semibatch crystallizer ({sbr.KINETICS_MODEL} kinetics, "
        "D01R crystallization)"))
    ax_n = fig.axes[0]
    ax_n.clear()
    time_min = np.asarray(tank.result.time) / 60.0
    totals = total_moles(tank)
    for name in sbr.report_species():
        ax_n.plot(time_min, totals[name], color=sbr.SPECIES_COLORS[name],
                  lw=2, label=name)
    ax_n.set_xlabel("Time [min]")
    ax_n.set_ylabel("Amount [mol]")
    ax_n.set_title("Species in the vessel (liquid + crystals)")
    ax_n.legend(frameon=False, ncol=2)
    sbr.style(ax_n)
    sbr.mark_dose_end(ax_n)
    return fig


def plot_crystals(tank):
    """Dissolved vs crystallized D, E, J over time, and the final CSDs."""
    import matplotlib.pyplot as plt

    res = tank.result
    time_min = np.asarray(res.time) / 60.0
    liquid = sbr.mole_history(tank)
    solid = crystal_moles(tank)
    names = sbr.species_names()
    conc = np.asarray(res.mole_conc_liquid0, dtype=float)

    fig, (ax_n, ax_s, ax_csd) = plt.subplots(1, 3, figsize=(15, 4.5))
    for species in CRYSTAL_SPECIES:
        color = sbr.SPECIES_COLORS[species]  # as in the other figures
        ax_n.plot(time_min, solid[species], color=color, lw=2,
                  label=f"{species} crystals")
        ax_n.plot(time_min, liquid[species], color=color, lw=1.5,
                  ls="--", label=f"{species} dissolved")
        ax_s.plot(time_min, conc[:, names.index(species)]
                  / SOLUBILITY_S0[species], color=color, lw=2, label=species)
        distrib = np.asarray(getattr(res, f"distrib_solid"
                                          f"{CRYSTAL_SPECIES.index(species)}"))
        ax_csd.plot(CRYSTAL_SIZE_GRID, distrib[-1], color=color, lw=2,
                    label=species)

    ax_n.set_xlabel("Time [min]")
    ax_n.set_ylabel("Amount [mol]")
    ax_n.set_title("Crystallized vs dissolved")
    ax_n.legend(frameon=False, fontsize=8, ncol=2)

    ax_s.axhline(1.0, color=TEXT_MUTED, lw=1, ls=":")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("Time [min]")
    ax_s.set_ylabel("$c_i / c_{sat,i}$")
    ax_s.set_title("Saturation ratio")
    ax_s.legend(frameon=False)

    ax_csd.set_xlabel("Size [µm]")
    ax_csd.set_ylabel("Number density [#/m³/µm]")
    ax_csd.set_title(f"Size distribution at t = {time_min[-1]:.0f} min")
    ax_csd.legend(frameon=False)

    for axis in (ax_n, ax_s):
        sbr.style(axis)
        sbr.mark_dose_end(axis)
    sbr.style(ax_csd)
    fig.suptitle("Semibatch crystallizer: D, E, J crystals")
    fig.tight_layout()
    return fig


def plot_settings(tank):
    fig = sbr.plot_settings(tank, title="Semibatch crystallizer settings")
    return fig


def crystal_settings_rows() -> list[list[str]]:
    rows = []
    for species in CRYSTAL_SPECIES:
        rows.append([species, f"{SOLUBILITY_S0[species]:g}",
                     "k {:g}, Ea {:g}, b {:g}".format(*NUCL_PRIM[species]),
                     "k {:g}, Ea {:g}, g {:g}".format(*GROWTH[species]),
                     "k {:g}, Ea {:g}, d {:g}".format(*DISSOLUTION[species])])
    return rows


def plot_crystal_settings():
    return sbr.table_figure(
        crystal_settings_rows(),
        ["Species", "c_sat [mol/L]", "Nucleation [#/s/m³]", "Growth [µm/s]",
         "Dissolution [µm/s]"],
        "Semibatch crystallizer: crystallization kinetics (from D01R)",
        note=(f"Size grid {CRYSTAL_SIZE_GRID[0]:g}-{CRYSTAL_SIZE_GRID[-1]:g} "
              f"µm, {len(CRYSTAL_SIZE_GRID)} cells; kv {CRYSTAL_KV:g}."))


def plot_mass_balance(tank):
    balance = mole_balance(tank)

    def cell(value):
        return "0" if abs(value) < 5e-10 else f"{value:.6f}"

    rows = [[name, cell(balance.charged[name]), cell(balance.dosed[name]),
             cell(balance.reacted[name]), cell(balance.liquid[name]),
             cell(balance.solid.get(name, 0.0)), cell(balance.final[name])]
            for name in tank.name_species
            if any(abs(getattr(balance, term)[name]) > 5e-10
                   for term in ("charged", "dosed", "final"))]
    return sbr.table_figure(
        rows, ["Species", "Charged", "Dosed", "Reacted", "Dissolved",
               "Crystals", "Final"],
        f"Semibatch crystallizer mole balance at t = {balance.time:g} s "
        "[mol]",
        note=(f"Reacted = final - charged - dosed. C consumed minus "
              f"D + E + J (+ F) held: {balance.closure:.2g} mol "
              "(PharmaPy's crystal over-dilution shows up here)."))


def save_figures(tank, out_dir: Path = RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "SBC_profiles": plot_profiles(tank),
        "SBC_normalized": sbr.plot_normalized(
            tank, total_moles(tank),
            title=f"Semibatch crystallizer ({sbr.KINETICS_MODEL} kinetics), "
                  "normalized (liquid + crystals)"),
        "SBC_crystals": plot_crystals(tank),
        "SBC_settings": plot_settings(tank),
        "SBC_crystal_settings": plot_crystal_settings(),
        "SBC_mass_balance": plot_mass_balance(tank),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}{sbr.figure_suffix()}.png", dpi=150,
                    bbox_inches="tight")
    return figures


def print_summary(tank):
    balance = mole_balance(tank)
    temp = np.ravel(tank.result.global_temp)
    print(f"Moles at t = {balance.time:g} s (mol):")
    print(f"  {'species':<8}{'charged':>10}{'dosed':>10}{'reacted':>10}"
          f"{'dissolved':>11}{'crystals':>10}")
    for name in sbr.report_species() + (sbr.solvent_name(),):
        print(f"  {name:<8}{balance.charged[name] :>10.6f}"
              f"{balance.dosed[name] :>10.6f}"
              f"{balance.reacted[name] :>10.6f}"
              f"{balance.liquid[name] :>11.6f}"
              f"{balance.solid.get(name, 0.0) :>10.6f}")
    print(f"  C consumed - (D + E + J + F) held: {balance.closure:.2g} mol")
    loss = unaccounted_product(tank)
    if loss["share"] > LOSS_TOL:
        print(f"  WARNING: {loss['share']:.1%} of the product is unaccounted "
              "for (PharmaPy over-dilutes the crystals; see "
              "unaccounted_product).")
    print(f"Temperature: peak {temp.max():.2f} K, final {temp[-1]:.2f} K")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    sbr.add_kinetics_argument(parser)
    args = parser.parse_args()
    sbr.KINETICS_MODEL = args.kinetics

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    sbc = run_crystallizer()
    print_summary(sbc)
    save_figures(sbc)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
