"""Isothermal temperature sweep of R02R: the F-starvation case study.

Each point solves R02R with the vessel held isothermal at the sweep
temperature (``isothermal=True``, so there is no energy balance and no
jacket). Feeds, kinetics and charge are taken unchanged from ``R02R.py``.

The case study: the three reactions all consume F, but J has the highest
order in it (0.5, against 0.21 for E and 0.29 for D). F is fed at about an
eighth of A and a tenth of G and is used up almost completely, so J is
starved of it and E takes most of the product in the mid range, although
the rate constants alone favour J. J only leads below about 30 C, and D, the
one reaction with a positive activation energy, wins when hot.

Selectivity is molar, from the outlet at the end of the run::

    S_i = n_i / (n_D + n_E + n_J),   i in {D, E, J}

F conversion is ``1 - w_F,out / w_F,in`` on the combined feed, which at
steady state is the fraction of the fed F that reacted.

Run it with::

    python R02R_temp_sweep.py             # solve, save figure, show it
    python R02R_temp_sweep.py --no-show   # solve and save figure only

The figure is written to ``results/R02R_temp_sweep_selectivity.png``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ThermoModule import ParseDatabase

import R02R


SWEEP_TEMPS_C = np.arange(40.0, 80.0 + 1e-9, 5.0)  # degC
PRODUCTS = ("D", "E", "J")
LIMITING = "F"


def build_isothermal_reactor(temp: float, database_path: Path):
    """R02R as in R02R.build_reactor, but held isothermal at ``temp`` [K]."""
    species_names = R02R.database_species(database_path)
    solvent_name = R02R.database_solvent(species_names)

    reactor = R02R.ContinuousReactor(
        integrator=ScipyBackend(),
        isothermal=True,
        diam=R02R.REACTOR_DIAMETER,
    )
    reactor.Phases = LiquidPhase(
        str(database_path),
        temp=temp,
        mass_frac=R02R.named_mass_fraction(R02R.INITIAL_REACTOR,
                                           species_names, solvent_name),
        mass=R02R.REACTOR_MASS,
        name_solv=solvent_name,
    )
    reactor.RxnKinetics = RxnKinetics(
        path=str(database_path),
        rxn_list=R02R.REACTIONS,
        k_params=R02R.RATE_CONSTANTS,
        ea_params=R02R.ACTIVATION_ENERGIES,
        params_f=R02R.reaction_order_matrix(species_names),
        temp_ref=R02R.KINETICS_TEMP_REF,
        delta_hrxn=R02R.HEAT_OF_REACTION,
    )
    reactor.Inlet = R02R.make_feeds(database_path)
    return reactor


def inlet_mass_fraction(database_path: Path) -> np.ndarray:
    """Mass fractions of ``R02R.FEEDS`` once combined."""
    feeds = R02R.make_feeds(database_path)
    flows = np.array([flow_rate for _, flow_rate, _, _ in R02R.FEEDS])
    fracs = np.array([feed.mass_frac for feed in feeds])
    return flows @ fracs / flows.sum()


def selectivity(reactor, mw: dict[str, float]) -> np.ndarray:
    """Molar selectivity to each of PRODUCTS at the final time."""
    index = {name: j for j, name in enumerate(reactor.name_species)}
    frac = R02R.mass_fractions(reactor)[-1]
    moles = np.array([max(frac[index[p]], 0.0) / mw[p] for p in PRODUCTS])
    return moles / moles.sum()


def run_sweep(database_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Selectivity (temperature, product) and F conversion (temperature)."""
    database = ParseDatabase(str(database_path), to_arrays=False)
    mw = dict(zip(database["name_species"], database["mw"]))
    c_index = list(database["name_species"]).index(LIMITING)
    w_c_in = inlet_mass_fraction(database_path)[c_index]

    rows, conversion = [], []
    for temp_c in SWEEP_TEMPS_C:
        reactor = build_isothermal_reactor(temp_c + 273.15, database_path)
        reactor.solve_unit(runtime=R02R.RUNTIME, verbose=False)
        rows.append(selectivity(reactor, mw))
        w_c_out = max(R02R.mass_fractions(reactor)[-1][c_index], 0.0)
        conversion.append(1.0 - w_c_out / w_c_in)
        print(f"  {temp_c:5.1f} C  " + "  ".join(
            f"S_{p} {s:.4f}" for p, s in zip(PRODUCTS, rows[-1]))
            + f"   X_{LIMITING} {conversion[-1]:.3f}")
    return np.array(rows), np.array(conversion)


def plot_selectivity(sel: np.ndarray, conversion: np.ndarray):
    """Selectivities vs temperature, with F conversion dashed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, (name, color) in enumerate(zip(PRODUCTS, R02R.SERIES_COLORS)):
        ax.plot(SWEEP_TEMPS_C, sel[:, j], "o-", color=color, lw=2, ms=5,
                label=f"$S_{name}$")
    ax.plot(SWEEP_TEMPS_C, conversion, "--", color=R02R.TEXT_MUTED, lw=1.5,
            label=f"$X_{LIMITING}$ (conversion)")
    ax.set_xlabel("Reactor temperature [°C]")
    ax.set_ylabel("Selectivity [mol/mol products], conversion [-]")
    ax.set_ylim(0, 1.05)
    ax.set_title("R02R isothermal sweep: steady-state selectivity, "
                 "F-limited feed")
    ax.legend(frameon=False, loc="center right")
    ax.grid(True, color="#e5e4e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figure without opening a window")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    database = Path(os.environ.get("PHARMAPY_DATABASE",
                                   R02R.DEFAULT_DATABASE_PATH))
    print("Isothermal sweep, steady-state molar selectivity:")
    sel, conversion = run_sweep(database)

    R02R.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = R02R.RESULTS_DIR / "R02R_temp_sweep_selectivity.png"
    plot_selectivity(sel, conversion).savefig(out, dpi=150,
                                              bbox_inches="tight")
    print(f"Figure saved to {out}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
