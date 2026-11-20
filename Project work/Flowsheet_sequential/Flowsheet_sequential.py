"""The whole process with every unit operation its own ConcurrentVessel,
solved one after another (sequential modular).

``Flowsheet.py`` puts all eight vessels in ONE ConcurrentVessel on one
integrator, so every unit reads the live outlet of the unit feeding it. Here
each vessel is wrapped in a ConcurrentVessel of its own, with its own
integrator, and the units are solved in process order, each over the whole
run::

    CR01H -> CR02H -> [F01H] -> D02H -> [F02H] -> D01H -> CR02R -> D01R
          -> [F01R] -> D02R -> [F02R] -> D03R

A unit's feed from upstream is the upstream unit's recorded outlet
trajectory, interpolated linearly in time between the saved points
(``RecordedSlurryInlet``). On the way it goes through the same algebraic step
as ``Flowsheet.SlurryInlet``: the F01H / F02H / F01R / F02R filter split,
the top size cell fold, and dissolving any solid the sink has no phase for.

Why the filters are not ConcurrentVessels
-----------------------------------------
A ConcurrentVessel only accepts MultiPhaseVessel members, and a ConcurrentVessel
is not one itself, so neither a filter (a ContinuousFilter) nor another
ConcurrentVessel can be a member. The filters stay algebraic steps in the
feed of the unit after them, exactly as in ``Flowsheet.py``.

What is shared with Flowsheet.py
--------------------------------
Everything except the coupling: each vessel's builder and settings, the H and
R size grids, the inlet templates, the solver options and the whole report
and figure code are imported from ``../Flowsheet/Flowsheet.py``. A change
there (a grid, a design value, a builder) carries over to this version.

Differences from the concurrent version
---------------------------------------
- There is no recycle in the plant, so the sequential solve is the same
  model; the results should agree up to the interpolation of the recorded
  feeds (linear between the saved points, ``REPORT_POINTS``) and solver
  tolerance. The report states where they differ.
- Each unit has its own CVode integrator, its own step sizes and its own
  Jacobian, which is far smaller (one vessel, ~100-400 states instead of
  ~1000).

Run from an activated environment, or LAPACK fails to load::

    conda activate PharmaPy
    python Flowsheet_sequential.py                 # solve, save, show figures
    python Flowsheet_sequential.py --no-show       # solve and save figures only
    python Flowsheet_sequential.py --runtime 600   # a short run, to check wiring

Figures are written to ``results/`` next to this file, with the same names
as Flowsheet.py's (Flowsheet_profiles.png, Flowsheet_streams.png).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import sys
import time as clock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent          # Project work/Flowsheet_sequential
PROJECT = HERE.parent                           # Project work
RESULTS_DIR = HERE / "results"
sys.path.insert(0, str(PROJECT / "Flowsheet"))

# Importing Flowsheet sets the shared H and R size grids on every unit module
# and puts the unit folders on sys.path.
import Flowsheet as base  # noqa: E402
from Flowsheet import (  # noqa: E402
    CR01H, CR02H, CR02R, D01H, D01R, D02H, D02R, D03R, F01H, F01R, F02H,
    F02R, DB_PATH, REPORT_POINTS, RUNTIME, SOLVER_OPTIONS,
    SlurryInlet, SolidCakeInlet, cake_template_f01h, cake_template_r,
    fold_top_cell, liquid_template, outlet_at, product_template,
    species_names, split_cake, with_true_slurry_feed)
from PharmaPy.ConcurrentVessel import ConcurrentVessel  # noqa: E402
from PharmaPy.IntegratorBackends import AssimuloBackend  # noqa: E402

# Process order. Each entry: (unit, upstream unit, filter module or None).
SEQUENCE = (
    ("CR01H", None, None),
    ("CR02H", "CR01H", None),
    ("D02H", "CR02H", F01H),
    ("D01H", "D02H", F02H),
    ("CR02R", "D01H", None),
    ("D01R", "CR02R", None),
    ("D02R", "D01R", F01R),
    ("D03R", "D02R", F02R),
)
FILTER_NAMES = {F01H: "F01H", F02H: "F02H", F01R: "F01R", F02R: "F02R"}


# ------------------------------------------------- recorded feed
class RecordedSlurryInlet(SlurryInlet):
    """A unit's feed: the upstream unit's solved outlet, read back in time.

    ``Flowsheet.SlurryInlet`` reads the source's live outlet; this reads the
    source's saved outlet (absolute liquid and crystal flows at every saved
    point, from ``Flowsheet.outlet_at``) and interpolates it linearly at the
    time asked. The filter split, top cell fold and dissolving of unmatched
    solids are the same. It depends on time alone (``follows_state`` False),
    so the sink caches it per time like any recorded upstream trajectory, and
    it holds only arrays, so the results replay can copy it freely.
    """

    follows_state = False

    def __init__(self, source, sink_species, separator=None):
        super().__init__(source, sink_species, separator)
        points = [outlet_at(source, i)
                  for i in range(len(np.ravel(source.result.time)))]
        self.time = np.asarray(source.result.time, dtype=float).ravel()
        self.temp = np.array([p.temp for p in points])
        self.conc = np.array([p.conc for p in points])
        self.liquid_flow = np.array([p.liquid_flow for p in points])
        self.crystal_flows = [np.array([p.crystals[k] for p in points])
                              for k in range(len(self.shape.mass_weights))]
        self.source = None  # the history replaces the vessel
        self._now = 0.0

    def evaluate_inputs(self, time):
        self._now = float(time)
        return super().evaluate_inputs(time)

    def _recorded(self, time):
        t = self.time
        time = min(max(time, t[0]), t[-1])
        i = min(int(np.searchsorted(t, time, side="right")) - 1, t.size - 2)
        w = (time - t[i]) / (t[i + 1] - t[i])

        def lerp(a):
            return (1.0 - w) * a[i] + w * a[i + 1]

        return (float(lerp(self.temp)), np.maximum(lerp(self.conc), 0.0),
                max(float(lerp(self.liquid_flow)), 0.0),
                [np.maximum(lerp(n), 0.0) for n in self.crystal_flows])

    def live_values(self):
        temp, conc, liquid_flow, crystals = self._recorded(self._now)
        if self.separator is not None:
            liquid_flow, crystals = split_cake(
                self.separator, self.shape, conc, liquid_flow, crystals)
        crystals = [fold_top_cell(n, w) for n, w in
                    zip(crystals, self.shape.vol_weights)]

        self.dissolved = 0.0
        if self.dissolve and liquid_flow > 0:
            conc = conc.copy()
            for k, j in self.dissolve:
                mass = float(crystals[k] @ self.shape.mass_weights[k])
                conc[j] += mass / (self.shape.mw[j] * liquid_flow)
                self.dissolved += mass

        kept = [crystals[k] for k in self.picks]
        vol_weights = [self.shape.vol_weights[k] for k in self.picks]
        cake_flow = liquid_flow + sum(
            float(n @ w) for n, w in zip(kept, vol_weights))

        if cake_flow > 0:
            self.solids = [n / cake_flow for n in kept]
        else:
            self.solids = [np.zeros_like(n) for n in kept]

        return {
            "vol_flow": cake_flow,
            "mole_conc": conc,
            "temp": temp,
            "distrib": np.mean(self.solids, axis=0),
        }


def plug(sink, inlet, template, index=0):
    """Make ``inlet`` the sink's inlet ``index``, on a copy of ``template``.

    The same wiring as ``Flowsheet.attach``, without a link: each solid phase
    of the stream gets its own share of the inlet (``SolidCakeInlet``).
    """
    connection = sink.make_stream_connection(copy.deepcopy(template))
    solid_index = 0

    for phase in connection.stream:
        if phase.phase_family == "solid":
            phase.DynamicInlet = SolidCakeInlet(inlet, solid_index)
            solid_index += 1
        else:
            phase.DynamicInlet = inlet

    connections = list(sink.inlet_connections)
    connections[index] = connection
    sink.inlet_connections = connections


# ---------------------------------------------------------------- builders
def build_vessels() -> dict:
    """Every vessel from its own script, as in Flowsheet.build_flowsheet."""
    names = species_names()
    return {
        "CR01H": CR01H.build_reactor(DB_PATH),
        "CR02H": with_true_slurry_feed(CR02H.build_reactor(
            DB_PATH, feed=liquid_template(names, "H"))),
        "D02H": with_true_slurry_feed(
            D02H.build_dissolver(cake_template_f01h(names))),
        "D01H": with_true_slurry_feed(
            D01H.build_dissolver(cake_template_f01h(names))),
        "CR02R": with_true_slurry_feed(CR02R.build_reactor(
            feed=cake_template_f01h(names),
            makeup=CR02R.makeup_stream(names))),
        "D01R": with_true_slurry_feed(
            D01R.build_crystallizer(product_template(names))),
        "D02R": with_true_slurry_feed(
            D02R.build_dissolver(cake_template_r(names, D02R))),
        "D03R": with_true_slurry_feed(
            D03R.build_dissolver(cake_template_r(names, D03R))),
    }


def feed_template(unit, upstream, vessels, names):
    """The inlet stream shape each unit is fed through (as Flowsheet.py)."""
    if unit == "CR02H":  # a plain link there: a copy of CR01H's outlet
        return vessels[upstream].outlet_connections[0].stream
    if unit in ("D02H", "D01H", "CR02R"):
        return cake_template_f01h(names)
    if unit == "D01R":
        return product_template(names)
    return cake_template_r(names, D02R if unit == "D02R" else D03R)


def sink_species(unit):
    return {"CR02H": ("F",), "D02H": ("F",), "D01H": ("F",),
            "CR02R": ("F",), "D01R": D01R.SOLID_SPECIES,
            "D02R": D02R.CRYSTAL_SPECIES,
            "D03R": D03R.CRYSTAL_SPECIES}[unit]


def run_flowsheet(runtime: float = RUNTIME, points: int = REPORT_POINTS,
                  verbose: bool = True) -> SimpleNamespace:
    """Solve every unit's ConcurrentVessel in process order."""
    names = species_names()
    vessels = build_vessels()
    grid = np.linspace(0.0, runtime, points)
    units, walls, inlets = {}, {}, {}

    for unit, upstream, filter_module in SEQUENCE:
        vessel = vessels[unit]
        if upstream is not None:
            separator = (filter_module.build_filter()
                         if filter_module is not None else None)
            inlet = RecordedSlurryInlet(vessels[upstream], sink_species(unit),
                                        separator)
            plug(vessel, inlet, feed_template(unit, upstream, vessels, names))
            inlets[unit] = inlet

        cv = ConcurrentVessel(
            {unit: vessel},
            integrator=AssimuloBackend(options=dict(SOLVER_OPTIONS)))
        start = clock.time()
        cv.solve_unit(time_grid=grid, verbose=False)
        walls[unit] = clock.time() - start
        units[unit] = cv
        if verbose:
            via = (f" via {FILTER_NAMES[filter_module]}"
                   if filter_module is not None else "")
            fed = f"fed by {upstream}{via}" if upstream else "fresh feeds only"
            print(f"  {unit:<6} {cv.solver_state_collection.dim:>5} states  "
                  f"{walls[unit] / 60:>6.1f} min  ({fed})", flush=True)

    return SimpleNamespace(units=units, vessels=vessels, walls=walls,
                           inlets=inlets)


# ---------------------------------------------------------------- results
def as_plant(run) -> SimpleNamespace:
    """The solved units in the shape Flowsheet.py's report code reads."""
    last = run.units["D03R"]
    return SimpleNamespace(
        members=run.vessels,
        result=SimpleNamespace(time=last.result.time),
        tear_statistics={"evaluations": 0}, tears=(),
        solver_state_collection=SimpleNamespace(
            dim=sum(cv.solver_state_collection.dim
                    for cv in run.units.values())),
        integrator=last.integrator)


def print_report(run, wall: float):
    """Flowsheet.py's report, with a header for the sequential solve."""
    plant = as_plant(run)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        base.print_report(plant, wall)
    lines = buffer.getvalue().splitlines()
    header = (f"Sequential solve: {len(run.units)} ConcurrentVessels, one per "
              f"unit, {plant.solver_state_collection.dim} states in all, "
              f"{len(plant.result.time)} saved points, t = "
              f"{plant.result.time[-1] / 3600:.1f} h, {wall / 60:.1f} min "
              f"wall, linear solver {plant.integrator.linear_solver}")
    for i, line in enumerate(lines):
        if line.startswith("Concurrent solve:"):
            lines[i] = header
    print("\n".join(lines))

    print("\nDissolved inside the feeds [g/s] (unmatched solids):")
    for unit, inlet in run.inlets.items():
        if inlet.dissolve:
            print(f"  -> {unit:<6} {inlet.dissolved * 1e3:.3e}")


def save_figures(run, out_dir: Path = RESULTS_DIR):
    return base.save_figures(as_plant(run), out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    parser.add_argument("--runtime", type=float, default=RUNTIME,
                        help=f"simulated time [s], default {RUNTIME:g}")
    parser.add_argument("--points", type=int, default=REPORT_POINTS,
                        help="saved time points")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    print("Solving each unit's ConcurrentVessel in process order:")
    start = clock.time()
    solved = run_flowsheet(args.runtime, args.points)
    print_report(solved, clock.time() - start)
    save_figures(solved)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
