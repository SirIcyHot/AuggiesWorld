"""F01R: continuous filter on the D01R slurry.

The flowsheet is::

                       FEED_B --+
                                |
    R02R --(D, E, J in H)--> D01R --(D, E, J crystals)--> F01 --> cake
                                                             \\--> filtrate

``D01R`` is the antisolvent crystallizer, run as-is from
``Project work/D01R/D01R.py`` (which itself solves R02R for its feed). Its
outlet is taken at the final, steady time, so the filter sees exactly
D01R's discharge: liquid, temperature and the real D, E and J crystals.

D01R holds one solid phase per species. The filter carries one solid, so
``d01r_outlet`` merges the three into one mixed population on D01R's size
grid: number flows add cell by cell, the composition is each species'
share of the crystal mass, and the density is total mass / total crystal
volume. Mass by species is exact, because the filter treats every crystal
alike; D02R splits the population back into D, E and J by volume.

D01R's three slurry volumes are not its vessel volume (see
``D01R.slurry_volumes``), so the outlet is split into liquid and crystal
flows the same way: Q_L = Q / (1 + sum_k phi_k / (1 - phi_k)) and
N_k = n_k Q_L / (1 - phi_k).

``F01`` is ``ContinuousFilter`` from ``continuous_filter.py``. Two settings
govern it: ``SOLIDS_LOSS`` (fraction of crystals passing into the filtrate)
and ``CAKE_MOISTURE`` (liquid held in the cake, as LOD by default).

Run it with::

    python F01R.py

The compound database defaults to ``compounddatabase/compound_databaseActual.json``;
set ``PHARMAPY_DATABASE`` to use another one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from PharmaPy.DataClasses import PhaseRef
from PharmaPy.Mechanisms import OneDFVMMechanism

# ---------------------------------------------------------------- paths
# Built from this file's location, so they work whichever folder Python is
# started from. Update them here if any of these files move.
HERE = Path(__file__).resolve().parent          # Project work/F01R
ROOT = HERE.parents[1]                          # AuggiesWorld

D01R_DIR = ROOT / "Project work" / "D01R"
R02R_DIR = ROOT / "Project work" / "R02R"
DB_PATH = ROOT / "compounddatabase" / "compound_databaseActual.json"
FILTER_DIR = ROOT / "CustomUnitOps"

sys.path[:0] = [str(FILTER_DIR), str(D01R_DIR), str(R02R_DIR)]

from continuous_filter import ContinuousFilter, SlurryFlow  # noqa: E402
import D01R  # noqa: E402

DEFAULT_DATABASE_PATH = DB_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = D01R.CRYSTAL_SPECIES  # ("D", "E", "J")

# The crystals arrive from D01R, so they keep D01R's size grid.
CRYSTAL_SIZE_GRID = D01R.CRYSTAL_SIZE_GRID  # um
CRYSTAL_KV = D01R.CRYSTAL_KV

# Only for ``crystal_number_flow``: a Gaussian stand-in used to build stream
# templates before any crystals exist (e.g. by the coupled flowsheet).
CSD_MEAN = 80.0  # um
CSD_STD = 20.0  # um

SOLIDS_LOSS = 0.01  # fraction of crystals passing into the filtrate
CAKE_MOISTURE = 0.30  # kg liquid / kg wet cake (LOD)
MOISTURE_BASIS = "wet"


# ---------------------------------------------------------------- builders
def solid_densities(database_path: Path,
                    names=CRYSTAL_SPECIES) -> dict[str, float]:
    """Crystal density of each species [kg/m3], from the database."""
    with open(database_path) as handle:
        database = json.load(handle)
    return {name: float(database[name]["rho_solid"]) for name in names}


def mixed_density(solid_mass: dict[str, float],
                  densities: dict[str, float]) -> float:
    """Density of a crystal mix [kg/m3]: total mass / total volume."""
    total = sum(solid_mass.values())
    if total <= 0:
        # No crystals: any density will do; take the mass-free average.
        return float(np.mean(list(densities.values())))
    volume = sum(mass / densities[name] for name, mass in solid_mass.items())
    return total / volume


def crystal_number_flow(solid_mass_flow: float, density: float) -> np.ndarray:
    """Gaussian crystal number flow [#/(um s)] carrying solid_mass_flow.

    A template only; the filter itself gets D01R's real crystals.
    """
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


def outlet_flow(tank, time_index: int = -1) -> SlurryFlow:
    """D01R's discharge at one saved time, with D, E, J merged into one solid.

    Returns a SlurryFlow on D01R's grid: liquid mass flows [kg/s] and a
    crystal number flow [#/(um s)] whose composition and mixed density
    carry each species' crystal mass exactly.
    """
    res = tank.result
    outputs = tank.outputs
    liquid = tank.Phases.get_phase_from_ref(PhaseRef("liquid", 0))
    names = list(liquid.name_species)

    vol_flow = max(float(np.ravel(outputs["vol_flow"])[time_index]), 0.0)
    temp = (float(np.ravel(outputs["temp"])[time_index]) if "temp" in outputs
            else float(liquid.temp))
    conc = np.maximum(np.atleast_2d(outputs["mole_conc"])[time_index], 0.0)

    distribs, phis, masses, densities = [], [], {}, {}
    for k, species in enumerate(CRYSTAL_SPECIES):
        pbm = D01R.population_balance(tank, k)
        n = np.maximum(np.atleast_2d(getattr(res, f"distrib_solid{k}"))
                       [time_index], 0.0)
        distribs.append(n)
        phis.append(float(n @ D01R.crystal_weight(tank, k)) / pbm.density)
        densities[species] = float(pbm.density)

    liquid_flow = vol_flow / (1.0 + sum(p / (1.0 - p) for p in phis))
    crystal_flow = np.zeros(len(CRYSTAL_SIZE_GRID))
    for k, species in enumerate(CRYSTAL_SPECIES):
        number = distribs[k] * liquid_flow / (1.0 - phis[k])
        crystal_flow += number
        masses[species] = float(number @ D01R.crystal_weight(tank, k))

    total_solid = sum(masses.values())
    solid_mass_frac = np.zeros(len(names))
    for species, mass in masses.items():
        solid_mass_frac[names.index(species)] = (
            mass / total_solid if total_solid > 0
            else 1.0 / len(CRYSTAL_SPECIES))

    return SlurryFlow(
        path_thermo=liquid.path_thermo,
        name_species=names,
        temp=temp,
        liquid_mass_j=conc * np.asarray(liquid.mw) * liquid_flow,
        name_solv=liquid.name_solv,
        x_grid=np.asarray(CRYSTAL_SIZE_GRID, dtype=float),
        crystal_flow=crystal_flow,
        solid_mass_frac=solid_mass_frac,
        solid_density=mixed_density(masses, densities),
        kv=CRYSTAL_KV,
        target_components=list(CRYSTAL_SPECIES),
        solvent_name=liquid.name_solv,
    )


def d01r_outlet(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """Run R02R -> D01R and return D01R and its discharge at the final time.

    ``database_path`` is accepted for symmetry with the other scripts;
    D01R reads its own (``D01R.DB_PATH``).
    """
    _, tank = D01R.run_flowsheet()
    return tank, outlet_flow(tank)


def build_filter() -> ContinuousFilter:
    return ContinuousFilter(solids_loss=SOLIDS_LOSS,
                            cake_moisture=CAKE_MOISTURE,
                            moisture_basis=MOISTURE_BASIS)


def run_flowsheet(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """R02R -> D01R -> F01. Returns (D01R, D01R outlet, filter)."""
    tank, outlet = d01r_outlet(database_path)

    filt = build_filter()
    filt.Inlets = [outlet]
    filt.solve()

    return tank, outlet, filt


# ---------------------------------------------------------------- reporting
def crystal_mass_by_species(flow: SlurryFlow) -> dict[str, float]:
    """Crystal mass flow of each crystallizing species [kg/s]."""
    names = list(flow.name_species)
    return {name: flow.solid_mass_flow * flow.solid_mass_frac[names.index(name)]
            for name in CRYSTAL_SPECIES}


def species_table(filt: ContinuousFilter) -> list[list]:
    """Rows of [component, F01 feed (D01R out), cake, filtrate] in kg/s."""
    streams = [filt.feed, filt.cake_flow, filt.filtrate_flow]
    rows = []

    for j, name in enumerate(filt.feed.name_species):
        values = [s.liquid_mass_j[j] for s in streams]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)

    crystals = [crystal_mass_by_species(s) for s in streams]
    for name in CRYSTAL_SPECIES:
        rows.append([f"{name} (crystals)"] + [c[name] for c in crystals])

    rows.append(["Total"]
                + [s.liquid_mass_flow + s.solid_mass_flow for s in streams])
    return rows


def crystalline_fraction(feed: SlurryFlow) -> dict[str, float]:
    """Fraction of each crystallizing species that is crystalline [-]."""
    names = list(feed.name_species)
    crystals = crystal_mass_by_species(feed)
    out = {}
    for name in CRYSTAL_SPECIES:
        total = crystals[name] + feed.liquid_mass_j[names.index(name)]
        out[name] = crystals[name] / total if total > 0 else 0.0
    return out


def print_report(filt: ContinuousFilter):
    feed = filt.feed
    summary = filt.summary()
    feed_total = feed.liquid_mass_flow + feed.solid_mass_flow
    fractions = ", ".join(f"{name} {value:.1%}"
                          for name, value in crystalline_fraction(feed).items())

    print(f"D01R outlet: {feed_total:.6g} kg/s at {feed.temp:.2f} K; "
          f"crystalline: {fractions}")
    print(f"Crystals:    {feed.solid_density:.1f} kg/m3 mixed density")
    print(f"Settings:    solids loss {SOLIDS_LOSS:g}, cake moisture "
          f"{CAKE_MOISTURE:g} ({MOISTURE_BASIS} basis)\n")

    header = ["component", "D01R out", "cake", "filtrate"]
    print(f"  {header[0]:<16}" + "".join(f"{h:>12}" for h in header[1:])
          + "   [g/s]")
    for row in species_table(filt):
        print(f"  {row[0]:<16}" + "".join(f"{v * 1e3:>12.5f}"
                                          for v in row[1:]))

    print(f"\nCake LOD {summary['cake_lod']:.3f}, solids recovery "
          f"{summary['solids_recovery']:.2%}, mass balance error "
          f"{summary['mass_balance_error_kg_s']:.2e} kg/s")


if __name__ == "__main__":
    _, _, f01 = run_flowsheet()
    print_report(f01)
