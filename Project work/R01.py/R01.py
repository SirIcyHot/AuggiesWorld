"""Two-feed CSTR using PharmaPy's refactored unit operations."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

PHARMAPY_SOURCE_PATH = (
    Path(Path(__file__).anchor)
    / "PharmaPy-dev-upd2"
    / "PharmaPy-dev-upd2"
)
if PHARMAPY_SOURCE_PATH.is_dir():
    sys.path.insert(0, str(PHARMAPY_SOURCE_PATH))

from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.MixedPhases_Refactored import MixedStream
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.ThermoModule import ParseDatabase
from PharmaPy.DataClasses import PhaseMapping, PhaseRef, StreamConnection


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


class DirectInletContinuousReactor(ContinuousReactor):
    """Refactored reactor with local outlet and inlet compatibility fixes."""

    def configure_default_connections(self):
        """Create the outlet without using the broken phase copy path."""
        if len(self.outlet_connections) > 0:
            return

        phase = self.Phases[0]
        outlet_phase = LiquidStream(
            path_thermo=phase.path_data,
            temp=phase.temp,
            pres=phase.pres,
            mole_conc=phase.mole_conc,
            vol_flow=0.0,
            name_solv=phase.name_solv,
            verbose=False,
        )
        outlet_stream = MixedStream([outlet_phase])
        phase_ref = PhaseRef("liquid", 0)
        self.outlet_connections = [
            StreamConnection(
                stream=outlet_stream,
                phase_mappings=[
                    PhaseMapping(
                        source_phaseref=phase_ref,
                        sink_phaseref=phase_ref,
                    )
                ],
            )
        ]

    @property
    def Inlet(self):
        return self.inlet_connections


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
    """Create one refactored PharmaPy liquid feed."""
    return LiquidStream(
        path_thermo=str(database_path),
        temp=TEMPERATURE,
        mole_conc=named_mole_concentration(composition, species_names),
        vol_flow=flow_rate,
        name_solv="solvent",
        verbose=False,
    )


def build_flowsheet(database_path: Path | str = DEFAULT_DATABASE_PATH) -> SimulationExec:
    """Build a one-unit flowsheet with two direct reactor inlets."""
    database_path = Path(database_path)
    species_names = database_species(database_path)

    flowsheet = SimulationExec(str(database_path), flowsheet="R01")
    flowsheet.R01 = DirectInletContinuousReactor(
        integrator=ScipyBackend(),
        isothermal=True,
    )
    flowsheet.R01.Phases = LiquidPhase(
        str(database_path),
        temp=TEMPERATURE,
        mole_conc=named_mole_concentration(INITIAL_REACTOR, species_names),
        vol=REACTOR_VOLUME,
        name_solv="solvent",
    )
    MultiPhaseVessel.Inlet.fset(flowsheet.R01, [
        make_feed(database_path, FEED_A, FLOW_A, species_names),
        make_feed(database_path, FEED_B, FLOW_B, species_names),
    ])
    flowsheet.R01.RxnKinetics = RxnKinetics(
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
    """Run the direct two-inlet refactored CSTR flowsheet."""
    flowsheet = build_flowsheet(database_path)
    flowsheet.SolveFlowsheet(
        kwargs_run={"R01": {"runtime": runtime}},
        verbose=False,
    )
    return flowsheet


if __name__ == "__main__":
    database = Path(os.environ.get("PHARMAPY_DATABASE", DEFAULT_DATABASE_PATH))
    simulation = run_flowsheet(database)
    result = simulation.R01.result

    print("Final CSTR concentrations (mol/L):")
    for name, concentration in zip(
        simulation.R01.Phases.name_species,
        result.mole_conc_liquid0[-1],
    ):
        print(f"  {name}: {concentration:.6f}")
