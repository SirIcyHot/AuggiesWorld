"""F02R: continuous filter on the R02R product stream.

The flowsheet is::

    FEED_A --+
             +--> R02R --(A, C, D, E, G, I, J in B)--> F02 --> cake
    FEED_B --+                                            \\--> filtrate

``R02R`` is the three-reaction jacketed CSTR, run as-is from
``Project work/R02R.py``. Its outlet is taken at the final time, so the
filter sees exactly R02R's outlet mass flow, composition and temperature.

That outlet is all liquid, and a filter needs crystals. D, E and J are the
products that crystallize, so each is split here into dissolved and
crystalline. ``SOLID_FRACTION`` sets the split per species, as either:

- a number in [0, 1]: that fraction of the species is crystalline, or
- a solubility function ``solubility(temp, conc) -> mol/L of solution``,
  with ``conc`` the liquid's molar concentrations in database order: just
  enough stays dissolved to saturate the liquid left behind (equilibrium).

There is no solubility data for D, E or J yet, so the defaults are numbers.

The filter carries one solid phase, so D, E and J crystals travel as one
mixed population: a single size distribution, with the composition of the
mix and its density (total mass / total crystal volume). Mass by species is
exact regardless, because the filter treats every crystal alike.

The split conserves mass, so the slurry entering F02 carries R02R's outlet
mass flow exactly.

``F02`` is ``ContinuousFilter`` from ``continuous_filter.py``. Two settings
govern it: ``SOLIDS_LOSS`` (fraction of crystals passing into the filtrate)
and ``CAKE_MOISTURE`` (liquid held in the cake, as LOD by default).

Run it with::

    python F02R.py

The compound database defaults to R02R's ``compound_databaseActual.json``;
set ``PHARMAPY_DATABASE`` to use another one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Streams_Refactored import LiquidStream

from continuous_filter import ContinuousFilter, SlurryFlow, flow_from_vessel

HERE = Path(__file__).resolve().parent
R02R_DIR = HERE.parent / "Project work" / "R02R.py"

# R02R lives in a folder that is not a package, so it is loaded by path.
sys.path.insert(0, str(R02R_DIR))
import R02R  # noqa: E402

DEFAULT_DATABASE_PATH = R02R.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
CRYSTAL_SPECIES = ("D", "E", "J")

# None: the filter runs at R02R's outlet temperature.
FILTER_TEMPERATURE = None  # K

# Per crystallizing species: a fraction in [0, 1] that is crystalline, or a
# solubility function (temp, conc) -> mol/L for an equilibrium split.
# Placeholders: complete crystallization until solubility data exist.
SOLID_FRACTION = {
    "D": 1.0,
    "E": 1.0,
    "J": 1.0,
}

# Crystal size distribution, by mass, for the mixed D/E/J population. The
# grid matches CR01H's, so both filters hand on crystals on the same grid.
CRYSTAL_SIZE_GRID = np.arange(1.0, 201.0)  # um
CRYSTAL_KV = 1.0
CSD_MEAN = 80.0  # um
CSD_STD = 20.0  # um

SOLIDS_LOSS = 0.01  # fraction of crystals passing into the filtrate
CAKE_MOISTURE = 0.30  # kg liquid / kg wet cake (LOD)
MOISTURE_BASIS = "wet"

# Equilibrium with more than one solubility function is solved one species
# at a time, repeated until no dissolved flow moves by more than this.
EQUILIBRIUM_TOL = 1e-12  # relative
EQUILIBRIUM_MAX_SWEEPS = 100


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


def r02r_outlet(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """Run R02R and return its outlet at the final time.

    Returns
    -------
    reactor : ContinuousReactor
        The solved R02R.
    outlet : SlurryFlow
        Its product stream, liquid only.
    """
    reactor = R02R.run_reactor(database_path)
    return reactor, flow_from_vessel(reactor)


def _liquid_stream(flow: SlurryFlow, liquid_mass_j: np.ndarray, temp: float):
    total = float(np.sum(liquid_mass_j))
    return LiquidStream(flow.path_thermo, mass_flow=total,
                        mass_frac=liquid_mass_j / total, temp=temp,
                        name_solv=flow.name_solv, verbose=False)


def dissolved_flows(flow: SlurryFlow, temp: float,
                    settings: dict = SOLID_FRACTION) -> np.ndarray:
    """Liquid mass flows [kg/s] once D, E and J have crystallized.

    Species given a fraction are set directly. Species given a solubility
    function are solved for: conc_i(m_i) = solubility_i(T, conc), where m_i
    is what stays dissolved. Crystallizing any species shrinks the liquid
    and so concentrates all the others, which couples the equations; they
    are solved one species at a time, sweeping until nothing moves.
    """
    names = list(flow.name_species)
    mass_j = flow.liquid_mass_j.copy()

    unknown = set(settings) - set(CRYSTAL_SPECIES)
    if unknown:
        raise KeyError(f"SOLID_FRACTION names non-crystallizing species: "
                       f"{', '.join(sorted(unknown))}")

    equilibrium = []
    for name in CRYSTAL_SPECIES:
        setting = settings.get(name, 0.0)
        j = names.index(name)
        if callable(setting):
            equilibrium.append((j, setting))
        else:
            if not 0.0 <= setting <= 1.0:
                raise ValueError(
                    f"SOLID_FRACTION[{name!r}] must lie in [0, 1].")
            mass_j[j] = flow.liquid_mass_j[j] * (1.0 - setting)

    def excess(j, solubility, dissolved):
        trial = mass_j.copy()
        trial[j] = dissolved
        conc = np.asarray(_liquid_stream(flow, trial, temp).mole_conc,
                          dtype=float)
        return conc[j] - float(solubility(temp, conc))

    for _ in range(EQUILIBRIUM_MAX_SWEEPS):
        moved = 0.0
        for j, solubility in equilibrium:
            total = flow.liquid_mass_j[j]
            if excess(j, solubility, total) <= 0:
                new = total  # undersaturated: none of it crystallizes
            elif excess(j, solubility, 0.0) >= 0:
                new = 0.0  # zero solubility here: all of it crystallizes
            else:
                new = brentq(lambda m: excess(j, solubility, m), 0.0, total,
                             xtol=1e-16, rtol=1e-12)
            moved = max(moved, abs(new - mass_j[j]) / max(total, 1e-300))
            mass_j[j] = new
        if moved <= EQUILIBRIUM_TOL:
            return mass_j

    raise RuntimeError("The equilibrium split did not converge.")


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


def crystallize(flow: SlurryFlow, temp: float | None = None,
                settings: dict = SOLID_FRACTION) -> SlurryFlow:
    """Split D, E and J in a liquid flow into dissolved and crystalline."""
    if temp is None:
        temp = flow.temp

    names = list(flow.name_species)
    liquid_mass_j = dissolved_flows(flow, temp, settings)
    solid_mass_j = flow.liquid_mass_j - liquid_mass_j

    solid_mass = {name: float(solid_mass_j[names.index(name)])
                  for name in CRYSTAL_SPECIES}
    total_solid = sum(solid_mass.values())

    solid_mass_frac = np.zeros(len(names))
    for name, mass in solid_mass.items():
        solid_mass_frac[names.index(name)] = (
            mass / total_solid if total_solid > 0
            else 1.0 / len(CRYSTAL_SPECIES))

    density = mixed_density(solid_mass,
                            solid_densities(Path(flow.path_thermo)))

    return SlurryFlow(
        path_thermo=flow.path_thermo,
        name_species=names,
        temp=temp,
        liquid_mass_j=liquid_mass_j,
        name_solv=flow.name_solv,
        x_grid=np.asarray(CRYSTAL_SIZE_GRID, dtype=float),
        crystal_flow=crystal_number_flow(total_solid, density),
        solid_mass_frac=solid_mass_frac,
        solid_density=density,
        kv=CRYSTAL_KV,
        target_components=list(CRYSTAL_SPECIES),
        solvent_name=flow.name_solv,
    )


def build_filter() -> ContinuousFilter:
    return ContinuousFilter(solids_loss=SOLIDS_LOSS,
                            cake_moisture=CAKE_MOISTURE,
                            moisture_basis=MOISTURE_BASIS)


def run_flowsheet(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """R02R -> F02. Returns (reactor, R02R outlet, filter feed, filter)."""
    reactor, outlet = r02r_outlet(database_path)
    feed = crystallize(outlet, temp=FILTER_TEMPERATURE)

    filt = build_filter()
    filt.Inlets = [feed]
    filt.solve()

    return reactor, outlet, feed, filt


# ---------------------------------------------------------------- reporting
def crystal_mass_by_species(flow: SlurryFlow) -> dict[str, float]:
    """Crystal mass flow of each crystallizing species [kg/s]."""
    names = list(flow.name_species)
    return {name: flow.solid_mass_flow * flow.solid_mass_frac[names.index(name)]
            for name in CRYSTAL_SPECIES}


def species_table(outlet: SlurryFlow, filt: ContinuousFilter) -> list[list]:
    """Rows of [component, R02R out, F02 feed, cake, filtrate] in kg/s."""
    streams = [filt.feed, filt.cake_flow, filt.filtrate_flow]
    rows = []

    for j, name in enumerate(outlet.name_species):
        values = [outlet.liquid_mass_j[j]] + [s.liquid_mass_j[j]
                                              for s in streams]
        if max(abs(v) for v in values) > 1e-12:
            rows.append([f"{name} (dissolved)"] + values)

    crystals = [crystal_mass_by_species(s) for s in streams]
    for name in CRYSTAL_SPECIES:
        rows.append([f"{name} (crystals)", 0.0] + [c[name] for c in crystals])

    rows.append(["Total", outlet.liquid_mass_flow]
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


def print_report(outlet: SlurryFlow, filt: ContinuousFilter):
    feed = filt.feed
    summary = filt.summary()
    feed_total = feed.liquid_mass_flow + feed.solid_mass_flow
    fractions = ", ".join(f"{name} {value:.1%}"
                          for name, value in crystalline_fraction(feed).items())

    print(f"R02R outlet: {outlet.liquid_mass_flow:.6g} kg/s at "
          f"{outlet.temp:.2f} K")
    print(f"F02 feed:    {feed_total:.6g} kg/s at {feed.temp:.2f} K; "
          f"crystalline: {fractions}")
    print(f"Crystals:    {feed.solid_density:.1f} kg/m3 mixed density")
    print(f"Settings:    solids loss {SOLIDS_LOSS:g}, cake moisture "
          f"{CAKE_MOISTURE:g} ({MOISTURE_BASIS} basis)\n")

    header = ["component", "R02R out", "F02 feed", "cake", "filtrate"]
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
    _, r02r_out, _, f02 = run_flowsheet(database)
    print_report(r02r_out, f02)
