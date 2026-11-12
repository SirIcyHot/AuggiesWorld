"""F02R: continuous filter on the D02R wash slurry (the filter half of W02R).

The flowsheet is::

                                     +------------ W02R ------------+
    R02R --> D01R --> F01R --cake--> | D02R --> F02R ---------cake--+--> D03R
                                     |  ^         |                 |
                                     +--|---------|-----------------+
                                       FEED_B     +--> wash filtrate

``D02R`` is the B wash dissolver, run as-is from ``Project work/D02R/D02R.py``
(which itself solves R02R -> D01R -> F01R for its feed). Its outlet is taken
at the final, steady time, so the filter sees exactly D02R's discharge:
liquid, temperature and the real D, E and J crystals, merged into one mixed
population by ``F01R.outlet_flow`` (D02R holds one solid phase per species,
as D01R does).

``F02`` is ``ContinuousFilter`` from ``continuous_filter.py``, with F01R's
settings: ``SOLIDS_LOSS``, ``CAKE_MOISTURE`` and ``MOISTURE_BASIS`` are read
from ``F01R``, so the two filters stay the same.

Run it with::

    python F02R.py

The compound database defaults to ``compounddatabase/compound_databaseActual.json``;
set ``PHARMAPY_DATABASE`` to use another one.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/F02R
ROOT = HERE.parents[1]                          # AuggiesWorld

D02R_DIR = ROOT / "Project work" / "D02R"
F01R_DIR = ROOT / "Project work" / "F01R"

sys.path[:0] = [str(D02R_DIR), str(F01R_DIR)]

import F01R  # noqa: E402  (puts D01R, R02R and CustomUnitOps on the path)
import D02R  # noqa: E402
from continuous_filter import ContinuousFilter, SlurryFlow  # noqa: E402

DEFAULT_DATABASE_PATH = F01R.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = F01R.CRYSTAL_SPECIES  # ("D", "E", "J")
CRYSTAL_SIZE_GRID = D02R.CRYSTAL_SIZE_GRID  # um, D02R's (= F01R's = D01R's)
CRYSTAL_KV = D02R.CRYSTAL_KV

# Same settings as the F01R filter before it.
SOLIDS_LOSS = F01R.SOLIDS_LOSS  # fraction of crystals passing into the filtrate
CAKE_MOISTURE = F01R.CAKE_MOISTURE  # kg liquid / kg wet cake (LOD)
MOISTURE_BASIS = F01R.MOISTURE_BASIS


# ---------------------------------------------------------------- builders
def d02r_outlet():
    """Run R02R -> ... -> D02R. Returns (D02R cake MixedStream, its
    SlurryFlow, D02R, D02R's discharge at the final time)."""
    cake, cake_flow, tank = D02R.run_flowsheet()
    return cake, cake_flow, tank, F01R.outlet_flow(tank)


def build_filter() -> ContinuousFilter:
    return ContinuousFilter(solids_loss=SOLIDS_LOSS,
                            cake_moisture=CAKE_MOISTURE,
                            moisture_basis=MOISTURE_BASIS)


def run_flowsheet(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """R02R -> D01R -> F01R -> D02R -> F02. Returns (D02R, D02R outlet,
    filter). ``database_path`` is accepted for symmetry with the other
    scripts; D02R reads its own."""
    _, _, tank, outlet = d02r_outlet()

    filt = build_filter()
    filt.Inlets = [outlet]
    filt.solve()

    return tank, outlet, filt


# ---------------------------------------------------------------- reporting
def print_report(filt: ContinuousFilter, f01r_cake: SlurryFlow | None = None):
    """The wash filter's split; with ``f01r_cake``, also what the wash did
    to the cake liquid (mother liquor in vs out of W02R)."""
    feed = filt.feed
    summary = filt.summary()
    feed_total = feed.liquid_mass_flow + feed.solid_mass_flow

    print(f"D02R outlet: {feed_total:.6g} kg/s at {feed.temp:.2f} K")
    print(f"Settings:    solids loss {SOLIDS_LOSS:g}, cake moisture "
          f"{CAKE_MOISTURE:g} ({MOISTURE_BASIS} basis), as F01R\n")

    header = ["component", "D02R out", "cake", "filtrate"]
    print(f"  {header[0]:<16}" + "".join(f"{h:>12}" for h in header[1:])
          + "   [g/s]")
    for row in F01R.species_table(filt):
        print(f"  {row[0]:<16}" + "".join(f"{v * 1e3:>12.5f}"
                                          for v in row[1:]))

    print(f"\nCake LOD {summary['cake_lod']:.3f}, solids recovery "
          f"{summary['solids_recovery']:.2%}, mass balance error "
          f"{summary['mass_balance_error_kg_s']:.2e} kg/s")

    if f01r_cake is not None:
        names = list(feed.name_species)
        washed = filt.cake_flow
        print("\nW02R wash, cake liquid in (F01R cake) vs out (F02R cake) [g/s]:")
        for j, name in enumerate(names):
            before, after = f01r_cake.liquid_mass_j[j], washed.liquid_mass_j[j]
            if max(before, after) > 1e-9:
                print(f"  {name:<4}{before * 1e3:>12.5f}{after * 1e3:>12.5f}")


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    d02r_cake, d02r_cake_flow, d02r, outlet = d02r_outlet()
    f02 = build_filter()
    f02.Inlets = [outlet]
    f02.solve()
    print_report(f02, d02r_cake_flow)
