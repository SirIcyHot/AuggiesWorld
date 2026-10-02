"""Two-feed PharmaPy CSTR flowsheet.

The flowsheet is::

    MIX01 --> R01

``MIX01`` combines two continuous liquid feeds and ``R01`` is PharmaPy's
continuous stirred-tank reactor. Feed compositions are supplied by compound
name; the database controls the order required by PharmaPy internally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace

import numpy as np


from PharmaPy.Containers import Mixer as PharmaPyMixer
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import CSTR
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.ThermoModule import ParseDatabase


DEFAULT_DATABASE_PATH = Path("Z:/AuggiesWorld/Project work/compound_database.json")

REACTIONS = ["A + B --> C"]
RATE_CONSTANTS = np.array([1.0e-2])
ACTIVATION_ENERGIES = np.array([0.0])

FLOW_A = 1.0e-5  # m**3/s
FLOW_B = 1.0e-5  # m**3/s
REACTOR_VOLUME = 0.01  # m**3
TEMPERATURE = 313.15  # K
RUNTIME = 1800.0  # s

FEED_A = {"A": 0.33}
FEED_B = {"B": 0.33}
INITIAL_REACTOR = {"A": 0.0, "B": 0.0, "C": 0.0}


class Mixer(PharmaPyMixer):
    """Steady two-feed mixer with vector-shaped connection outputs."""

    def solve_unit(self) -> None:
        total_flow = sum(feed.mass_flow for feed in self.Inlets)
        mass_frac = sum(
            feed.mass_flow * feed.mass_frac for feed in self.Inlets
        ) / total_flow
        temperature = sum(
            feed.mass_flow * feed.temp for feed in self.Inlets
        ) / total_flow
        time = np.array([0.0, 1.0])

        self.names_states_out = ["temp", "mass_frac", "mass_flow"]
        self.Outlet = LiquidStream(
            path_thermo=self.Inlets[0].path_data,
            temp=temperature,
            mass_frac=mass_frac,
            mass_flow=total_flow,
            verbose=False,
        )
        self.outputs = {
            "temp": np.full(time.shape, temperature),
            "mass_frac": np.vstack((mass_frac, mass_frac)),
            "mass_flow": np.full(time.shape, total_flow),
        }
        self.result = SimpleNamespace(time=time)
        self.timeProf = time

    def flatten_states(self) -> None:
        return None


def database_species(database_path: Path) -> tuple[str, ...]:
    """Return the database species names in PharmaPy's declared order."""
    parsed_database = ParseDatabase(str(database_path), to_arrays=False)
    return tuple(parsed_database["name_species"])


def named_mole_concentration(
    composition: Mapping[str, float],
    species_names: tuple[str, ...],
) -> np.ndarray:
    """Convert a named composition to PharmaPy's ordered concentration array."""
    unknown_names = set(composition) - set(species_names)
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise KeyError(f"Unknown compounds in composition: {names}")
    if any(value < 0.0 for value in composition.values()):
        raise ValueError("Mole concentrations cannot be negative.")

    return np.array(
        [composition.get(name, 0.0) for name in species_names],
        dtype=float,
    )


def make_feed(
    database_path: Path,
    composition: Mapping[str, float],
    flow_rate: float,
    species_names: tuple[str, ...],
) -> LiquidStream:
    """Create one named-composition liquid feed."""
    feed = LiquidStream(
        path_thermo=str(database_path),
        temp=TEMPERATURE,
        mole_conc=named_mole_concentration(composition, species_names),
        vol_flow=flow_rate,
        name_solv="solvent",
        verbose=False,
    )
    # Mixer.dynamic_balances expects continuous inlet fields to have a time
    # axis. Two identical points represent a steady feed without scalar
    # fields reaching the vectorized balance code.
    steady_time = np.array([0.0, 1.0])
    steady_profile = {
        "mass_flow": np.full(steady_time.shape, feed.mass_flow),
        "mass_frac": np.vstack((feed.mass_frac, feed.mass_frac)),
        "temp": np.full(steady_time.shape, feed.temp),
    }
    feed.time_upstream = steady_time
    feed.y_upstream = steady_profile
    feed.y_inlet = steady_profile
    return feed


def build_flowsheet(database_path: Path | str = DEFAULT_DATABASE_PATH) -> SimulationExec:
    """Build the two-feed mixer-to-CSTR PharmaPy flowsheet."""
    database_path = Path(database_path)
    species_names = database_species(database_path)

    flowsheet = SimulationExec(
        str(database_path),
        flowsheet="MIX01 --> R01",
    )

    flowsheet.MIX01 = Mixer()
    flowsheet.MIX01.Inlets = [
        make_feed(database_path, FEED_A, FLOW_A, species_names),
        make_feed(database_path, FEED_B, FLOW_B, species_names),
    ]

    flowsheet.R01 = CSTR(isothermal=True)
    flowsheet.R01.Phases = LiquidPhase(
        str(database_path),
        temp=TEMPERATURE,
        mole_conc=named_mole_concentration(INITIAL_REACTOR, species_names),
        vol=REACTOR_VOLUME,
        name_solv="solvent",
    )
    flowsheet.R01.Kinetics = RxnKinetics(
        path=str(database_path),
        rxn_list=REACTIONS,
        k_params=RATE_CONSTANTS,
        ea_params=ACTIVATION_ENERGIES,
    )

    return flowsheet


def run_flowsheet(
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    runtime: float = RUNTIME,
) -> SimulationExec:
    """Run the mixer-to-CSTR flowsheet and return its populated executor."""
    flowsheet = build_flowsheet(database_path)
    flowsheet.SolveFlowsheet(
        kwargs_run={"MIX01": {}, "R01": {"runtime": runtime}},
        verbose=False,
    )
    return flowsheet


if __name__ == "__main__":
    database = Path(os.environ.get("PHARMAPY_DATABASE", DEFAULT_DATABASE_PATH))
    simulation = run_flowsheet(database)
    result = simulation.R01.result

    print("Final CSTR concentrations (mol/L):")
    for name, concentration in zip(simulation.R01.name_species, result.mole_conc[-1]):
        print(f"  {name}: {concentration:.6f}")
