"""F02H: continuous filter on the D02H wash slurry (the filter half of W02H).

The flowsheet is::

                         +----------- W02H -----------+
    R01H --> F01H -cake->| D02H --> F02H -------cake--+--> D01H
                         |  ^         |               |
                         +--|---------|---------------+
                           FEED_H     +--> wash filtrate

``D02H`` is the H wash dissolver, run as-is from ``Project work/D02H/D02H.py``
(which itself solves R01H -> F01H for its feed). Its outlet is taken at the
final, steady time with ``continuous_filter.flow_from_vessel``, so the filter
sees exactly D02H's discharge: liquid, temperature and the real F crystals.

``F02`` is ``ContinuousFilter`` from ``continuous_filter.py``, with F01H's
settings: ``SOLIDS_LOSS``, ``CAKE_MOISTURE`` and ``MOISTURE_BASIS`` are read
from ``F01H``, so the two filters stay the same.

Run it with::

    python F02H.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/F02H
ROOT = HERE.parents[1]                          # AuggiesWorld

D02H_DIR = ROOT / "Project work" / "D02H"
F01H_DIR = ROOT / "Project work" / "F01H"

sys.path[:0] = [str(D02H_DIR), str(F01H_DIR)]

import F01H  # noqa: E402  (puts R01H and CustomUnitOps on the path)
import D02H  # noqa: E402
from continuous_filter import (  # noqa: E402
    ContinuousFilter, SlurryFlow, flow_from_vessel)

DEFAULT_DATABASE_PATH = F01H.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = F01H.CRYSTAL_SPECIES  # "F"

# Same settings as the F01H filter before it.
SOLIDS_LOSS = F01H.SOLIDS_LOSS  # fraction of crystals passing into the filtrate
CAKE_MOISTURE = F01H.CAKE_MOISTURE  # kg liquid / kg wet cake (LOD)
MOISTURE_BASIS = F01H.MOISTURE_BASIS


# ---------------------------------------------------------------- builders
def build_filter() -> ContinuousFilter:
    return ContinuousFilter(solids_loss=SOLIDS_LOSS,
                            cake_moisture=CAKE_MOISTURE,
                            moisture_basis=MOISTURE_BASIS)


def run_flowsheet():
    """R01H -> F01H -> D02H -> F02. Returns (D02H cake MixedStream, D02H
    cake SlurryFlow, D02H, filter)."""
    cake, cake_flow, tank = D02H.run_flowsheet()

    filt = build_filter()
    filt.Inlets = [flow_from_vessel(tank)]
    filt.solve()

    return cake, cake_flow, tank, filt


# ---------------------------------------------------------------- reporting
def species_table(filt: ContinuousFilter) -> list[list]:
    """Rows of [component, D02H out, cake, filtrate] in kg/s."""
    streams = [filt.feed, filt.cake_flow, filt.filtrate_flow]
    rows = []
    for j, name in enumerate(filt.feed.name_species):
        values = [s.liquid_mass_j[j] for s in streams]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)
    rows.append([f"{CRYSTAL_SPECIES} (crystals)"]
                + [s.solid_mass_flow for s in streams])
    rows.append(["Total"]
                + [s.liquid_mass_flow + s.solid_mass_flow for s in streams])
    return rows


def print_report(filt: ContinuousFilter, f01h_cake: SlurryFlow | None = None):
    """The wash filter's split; with ``f01h_cake``, also what the wash did
    to the cake liquid (mother liquor in vs out of W02H)."""
    feed = filt.feed
    summary = filt.summary()
    feed_total = feed.liquid_mass_flow + feed.solid_mass_flow

    print(f"D02H outlet: {feed_total:.6g} kg/s at {feed.temp:.2f} K")
    print(f"Settings:    solids loss {SOLIDS_LOSS:g}, cake moisture "
          f"{CAKE_MOISTURE:g} ({MOISTURE_BASIS} basis), as F01H\n")

    header = ["component", "D02H out", "cake", "filtrate"]
    print(f"  {header[0]:<16}" + "".join(f"{h:>12}" for h in header[1:])
          + "   [g/s]")
    for row in species_table(filt):
        print(f"  {row[0]:<16}" + "".join(f"{v * 1e3:>12.5f}"
                                          for v in row[1:]))

    print(f"\nCake LOD {summary['cake_lod']:.3f}, solids recovery "
          f"{summary['solids_recovery']:.2%}, mass balance error "
          f"{summary['mass_balance_error_kg_s']:.2e} kg/s")

    if f01h_cake is not None:
        washed = filt.cake_flow
        print("\nW02H wash, cake liquid in (F01H cake) vs out (F02H cake) [g/s]:")
        for j, name in enumerate(feed.name_species):
            before, after = f01h_cake.liquid_mass_j[j], washed.liquid_mass_j[j]
            if max(before, after) > 1e-9:
                print(f"  {name:<4}{before * 1e3:>12.5f}{after * 1e3:>12.5f}")


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    _, d02h_cake_flow, _, f02 = run_flowsheet()
    print_report(f02, d02h_cake_flow)
