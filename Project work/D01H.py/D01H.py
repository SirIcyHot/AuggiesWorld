"""D01H: continuous dissolver on the F01H filter cake.

The flowsheet is::

    FEED_A --+
             +--> R01 --> F01 --(cake)--+
    FEED_C --+                          +--> D01 --> slurry
                         SOLVENT -------+

``R01`` and ``F01`` are run as-is from ``Project work/F01H.py``, so D01 is
fed exactly the cake F01H produces: F crystals plus the mother liquor held
in the cake (A, B, C and dissolved F).

Fresh solvent B is added and part of the F crystals redissolve. How much is
set by CR01H's F solubility (temperature and [A] dependent): crystals
dissolve until the liquid is saturated or none are left. ``APPROACH``
below 1 stops short of equilibrium, as a stand-in for a residence time too
short to get there.

Dissolution shrinks every crystal by the same length (see
``CustomUnitOps/continuous_dissolver.py`` for the equations), so the
outlet's size distribution is the cake's, moved down.

Run it with::

    python D01H.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from PharmaPy.Streams_Refactored import LiquidStream

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/D01H.py
ROOT = HERE.parents[1]                          # AuggiesWorld

F01H_DIR = ROOT / "Project work" / "F01H.py"
UNIT_OPS_DIR = ROOT / "CustomUnitOps"

sys.path[:0] = [str(UNIT_OPS_DIR), str(F01H_DIR)]

from continuous_dissolver import ContinuousDissolver  # noqa: E402
import F01H  # noqa: E402

CR01H = F01H.CR01H
DB_PATH = F01H.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
SOLVENT_NAME = "B"
SOLVENT_FLOW = 1.0e-3  # kg/s of fresh solvent
SOLVENT_TEMPERATURE = 20 + 273.15  # K

# None: the dissolver runs at the mixed temperature of cake and solvent.
DISSOLVER_TEMPERATURE = None  # K

# Fraction of the way to equilibrium reached [-]; 1 is equilibrium.
APPROACH = 1.0


# ---------------------------------------------------------------- flowsheet
def solvent_stream(species_names) -> LiquidStream:
    """Fresh solvent feed."""
    mass_frac = np.zeros(len(species_names))
    mass_frac[list(species_names).index(SOLVENT_NAME)] = 1.0
    return LiquidStream(str(DB_PATH), mass_flow=SOLVENT_FLOW,
                        mass_frac=mass_frac, temp=SOLVENT_TEMPERATURE,
                        name_solv=SOLVENT_NAME, verbose=False)


def build_dissolver(species_names) -> ContinuousDissolver:
    return ContinuousDissolver(
        solubility=CR01H.make_solubility_function(tuple(species_names)),
        approach=APPROACH,
        temp=DISSOLVER_TEMPERATURE,
    )


def run_flowsheet():
    """R01 -> F01 -> D01. Returns (filter, solvent, dissolver)."""
    _, _, _, filt = F01H.run_flowsheet(DB_PATH)
    names = list(filt.feed.name_species)

    solvent = solvent_stream(names)
    diss = build_dissolver(names)
    diss.Inlets = [filt.cake_flow, solvent]
    diss.solve()

    return filt, solvent, diss


# ---------------------------------------------------------------- reporting
def species_table(filt, solvent, diss) -> list[list]:
    """Rows of [component, cake, solvent, D01 out] in kg/s."""
    cake, out = filt.cake_flow, diss.outlet_flow
    solvent_j = float(solvent.mass_flow) * np.asarray(solvent.mass_frac)
    crystal = F01H.CRYSTAL_SPECIES

    rows = []
    for j, name in enumerate(cake.name_species):
        values = [cake.liquid_mass_j[j], solvent_j[j], out.liquid_mass_j[j]]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)

    rows.append([f"{crystal} (crystals)", cake.solid_mass_flow, 0.0,
                 out.solid_mass_flow])
    rows.append(["Total",
                 cake.liquid_mass_flow + cake.solid_mass_flow,
                 float(solvent.mass_flow),
                 out.liquid_mass_flow + out.solid_mass_flow])
    return rows


def print_report(filt, solvent, diss):
    s = diss.summary()

    print(f"D01 at {s['temp_K']:.2f} K, solvent {SOLVENT_FLOW * 1e3:g} g/s "
          f"of {SOLVENT_NAME} at {SOLVENT_TEMPERATURE:.2f} K, "
          f"approach {APPROACH:g}")
    print(f"Dissolved {s['dissolved_kg_s'] * 1e3:.5f} g/s "
          f"({s['fraction_dissolved']:.1%} of the crystals); every crystal "
          f"shrank {s['shrinkage_um']:.2f} um\n")

    header = ["component", "cake", "solvent", "D01 out"]
    print(f"  {header[0]:<16}" + "".join(f"{h:>12}" for h in header[1:])
          + "   [g/s]")
    for row in species_table(filt, solvent, diss):
        print(f"  {row[0]:<16}" + "".join(f"{v * 1e3:>12.5f}"
                                          for v in row[1:]))

    print(f"\nMass balance error {s['mass_balance_error_kg_s']:.2e} kg/s")


if __name__ == "__main__":
    print_report(*run_flowsheet())
