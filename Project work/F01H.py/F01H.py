"""F01H: continuous filter on the R01H product stream.

The flowsheet is::

    FEED_A --+
             +--> R01 --(A, C, F in B)--> F01 --> cake
    FEED_C --+                               \\--> filtrate

``R01`` is the R01H jacketed CSTR, run as-is from ``Project work/R01H.py``.
Its outlet is taken at the final time, so the filter sees exactly R01H's
outlet mass flow, composition and temperature.

That outlet is all liquid, and a filter needs crystals. F is the species
that crystallizes (CR01H), so the dissolved F is split here into liquid and
solid using CR01H's own F solubility correlation, which depends on
temperature and [A]:

- ``F_SOLID_FRACTION = None``: equilibrium. Just enough F stays dissolved to
  saturate the remaining liquid at the filter temperature.
- ``F_SOLID_FRACTION = x``: that fraction of the F is crystalline, whatever
  the solubility says.

The crystals get a Gaussian size distribution on CR01H's size grid. The
split conserves mass, so the slurry entering F01 carries R01H's outlet mass
flow exactly.

``F01`` is ``ContinuousFilter`` from ``CustomUnitOps/continuous_filter.py``.
Two settings govern it: ``SOLIDS_LOSS`` (fraction of crystals passing into
the filtrate) and ``CAKE_MOISTURE`` (liquid held in the cake, as LOD by
default).

Run it with::

    python F01H.py

The compound database defaults to R01H's ``compound_databaseActual.json``;
set ``PHARMAPY_DATABASE`` to use another one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Streams_Refactored import LiquidStream

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/F01H.py
ROOT = HERE.parents[1]                          # AuggiesWorld

R01H_DIR = ROOT / "Project work" / "R01H.py"
DB_PATH = R01H_DIR / "compound_databaseActual.json"
FILTER_DIR = ROOT / "CustomUnitOps"

sys.path[:0] = [str(FILTER_DIR), str(R01H_DIR)]

from continuous_filter import (  # noqa: E402
    ContinuousFilter, SlurryFlow, flow_from_vessel)
import R01H  # noqa: E402
import CR01H  # noqa: E402

DEFAULT_DATABASE_PATH = DB_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = "F"

# None: the filter runs at R01H's outlet temperature.
FILTER_TEMPERATURE = None  # K

# None: equilibrium split from CR01H's solubility. A number in [0, 1]: that
# fraction of the F is crystalline.
F_SOLID_FRACTION = None

# Crystal size distribution, by mass, on CR01H's size grid.
CRYSTAL_SIZE_GRID = CR01H.CRYSTAL_SIZE_GRID  # um
CRYSTAL_KV = CR01H.CRYSTAL_KV
CSD_MEAN = 80.0  # um
CSD_STD = 20.0  # um

SOLIDS_LOSS = 0.01  # fraction of crystals passing into the filtrate
CAKE_MOISTURE = 0.30  # kg liquid / kg wet cake (LOD)
MOISTURE_BASIS = "wet"


# ---------------------------------------------------------------- builders
def solid_density(database_path: Path, name: str = CRYSTAL_SPECIES) -> float:
    """Crystal density [kg/m3] from the compound database."""
    with open(database_path) as handle:
        return float(json.load(handle)[name]["rho_solid"])


def r01h_outlet(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """Run R01H and return its outlet at the final time.

    Returns
    -------
    reactor : ContinuousReactor
        The solved R01.
    outlet : SlurryFlow
        Its product stream, liquid only.
    """
    reactor = R01H.run_reactor(database_path)
    return reactor, flow_from_vessel(reactor)


def _liquid_stream(flow: SlurryFlow, liquid_mass_j: np.ndarray, temp: float):
    total = float(np.sum(liquid_mass_j))
    return LiquidStream(flow.path_thermo, mass_flow=total,
                        mass_frac=liquid_mass_j / total, temp=temp,
                        name_solv=flow.name_solv, verbose=False)


def equilibrium_dissolved_f(flow: SlurryFlow, temp: float) -> float:
    """Dissolved F mass flow [kg/s] that saturates the liquid left behind.

    Solves conc_F(m) = solubility(T, [A](m)), where m is the F left in
    solution. Both sides move with m: crystallizing F shrinks the liquid
    volume and so concentrates everything else, A included. The solubility
    is CR01H's correlation, in mol/L of solution (= kmol/m3 of liquid).
    """
    names = list(flow.name_species)
    j_f = names.index(CRYSTAL_SPECIES)
    total_f = flow.liquid_mass_j[j_f]
    solubility = CR01H.make_solubility_function(tuple(names))

    def excess(dissolved):
        mass_j = flow.liquid_mass_j.copy()
        mass_j[j_f] = dissolved
        liquid = _liquid_stream(flow, mass_j, temp)
        conc = np.asarray(liquid.mole_conc, dtype=float)
        return conc[j_f] - float(solubility(temp, conc))

    if excess(total_f) <= 0:
        return total_f  # undersaturated: nothing crystallizes

    if excess(0.0) >= 0:
        return 0.0  # solubility is zero or negative here: all F crystallizes

    return brentq(excess, 0.0, total_f, xtol=1e-16, rtol=1e-12)


def crystal_number_flow(solid_mass_flow: float, density: float) -> np.ndarray:
    """Gaussian crystal number flow [#/(um s)] carrying solid_mass_flow."""
    grid = np.asarray(CRYSTAL_SIZE_GRID, dtype=float)
    # The FVM cell widths, so this mass matches the one the filter and any
    # downstream vessel compute from the same distribution.
    dx = OneDFVMMechanism._cell_widths(grid)
    mass_shape = np.exp(-0.5 * ((grid - CSD_MEAN) / CSD_STD) ** 2)

    # A mass-shaped distribution becomes a number distribution by dividing
    # by the crystal mass at each size.
    number_shape = mass_shape / grid**3
    mass_per_shape = (density * CRYSTAL_KV * 1e-18
                      * np.sum(number_shape * grid**3 * dx))

    return number_shape * solid_mass_flow / mass_per_shape


def crystallize_f(flow: SlurryFlow, temp: float | None = None,
                  solid_fraction: float | None = F_SOLID_FRACTION
                  ) -> SlurryFlow:
    """Split the F in a liquid flow into dissolved and crystalline F."""
    if temp is None:
        temp = flow.temp

    names = list(flow.name_species)
    j_f = names.index(CRYSTAL_SPECIES)
    total_f = float(flow.liquid_mass_j[j_f])

    if solid_fraction is None:
        dissolved = equilibrium_dissolved_f(flow, temp)
    else:
        if not 0.0 <= solid_fraction <= 1.0:
            raise ValueError("F_SOLID_FRACTION must lie in [0, 1].")
        dissolved = total_f * (1.0 - solid_fraction)

    solid_f = total_f - dissolved

    liquid_mass_j = flow.liquid_mass_j.copy()
    liquid_mass_j[j_f] = dissolved

    solid_mass_frac = np.zeros(len(names))
    solid_mass_frac[j_f] = 1.0

    density = solid_density(Path(flow.path_thermo))

    return SlurryFlow(
        path_thermo=flow.path_thermo,
        name_species=names,
        temp=temp,
        liquid_mass_j=liquid_mass_j,
        name_solv=flow.name_solv,
        x_grid=np.asarray(CRYSTAL_SIZE_GRID, dtype=float),
        crystal_flow=crystal_number_flow(solid_f, density),
        solid_mass_frac=solid_mass_frac,
        solid_density=density,
        kv=CRYSTAL_KV,
        target_components=[CRYSTAL_SPECIES],
        solvent_name=flow.name_solv,
    )


def build_filter() -> ContinuousFilter:
    return ContinuousFilter(solids_loss=SOLIDS_LOSS,
                            cake_moisture=CAKE_MOISTURE,
                            moisture_basis=MOISTURE_BASIS)


def run_flowsheet(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """R01 -> F01. Returns (reactor, r01h outlet, filter feed, filter)."""
    reactor, outlet = r01h_outlet(database_path)
    feed = crystallize_f(outlet, temp=FILTER_TEMPERATURE)

    filt = build_filter()
    filt.Inlets = [feed]
    filt.solve()

    return reactor, outlet, feed, filt


# ---------------------------------------------------------------- reporting
def species_table(outlet: SlurryFlow, filt: ContinuousFilter) -> list[list]:
    """Rows of [component, R01 out, F01 feed, cake, filtrate] in kg/s."""
    streams = [filt.feed, filt.cake_flow, filt.filtrate_flow]
    rows = []

    for j, name in enumerate(outlet.name_species):
        values = [outlet.liquid_mass_j[j]] + [s.liquid_mass_j[j]
                                              for s in streams]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)

    rows.append([f"{CRYSTAL_SPECIES} (crystals)", 0.0]
                + [s.solid_mass_flow for s in streams])
    rows.append(["Total", outlet.liquid_mass_flow]
                + [s.liquid_mass_flow + s.solid_mass_flow for s in streams])
    return rows


def crystalline_fraction(feed: SlurryFlow) -> float:
    """Fraction of the F in a flow that is crystalline [-]."""
    dissolved = feed.liquid_mass_j[list(feed.name_species).index(
        CRYSTAL_SPECIES)]
    total = feed.solid_mass_flow + dissolved
    return feed.solid_mass_flow / total if total > 0 else 0.0


def print_report(outlet: SlurryFlow, filt: ContinuousFilter):
    feed = filt.feed
    summary = filt.summary()
    feed_total = feed.liquid_mass_flow + feed.solid_mass_flow

    print(f"R01H outlet: {outlet.liquid_mass_flow:.6g} kg/s at "
          f"{outlet.temp:.2f} K")
    print(f"F01 feed:    {feed_total:.6g} kg/s at {feed.temp:.2f} K, "
          f"{crystalline_fraction(feed):.1%} of F crystalline")
    print(f"Settings:    solids loss {SOLIDS_LOSS:g}, cake moisture "
          f"{CAKE_MOISTURE:g} ({MOISTURE_BASIS} basis)\n")

    header = ["component", "R01 out", "F01 feed", "cake", "filtrate"]
    print(f"  {header[0]:<16}" + "".join(f"{h:>12}" for h in header[1:])
          + "   [g/s]")
    for row in species_table(outlet, filt):
        print(f"  {row[0]:<16}" + "".join(f"{v * 1e3:>12.5f}"
                                          for v in row[1:]))

    print(f"\nCake LOD {summary['cake_lod']:.3f}, solids recovery "
          f"{summary['solids_recovery']:.2%}, mass balance error "
          f"{summary['mass_balance_error_kg_s']:.2e} kg/s")


if __name__ == "__main__":
    database = Path(os.environ.get("PHARMAPY_DATABASE",
                                   DEFAULT_DATABASE_PATH))
    _, r01_outlet, _, f01 = run_flowsheet(database)
    print_report(r01_outlet, f01)
