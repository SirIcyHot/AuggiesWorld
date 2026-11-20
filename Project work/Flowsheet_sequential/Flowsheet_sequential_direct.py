"""The simplified (DirectTransfer) process with every unit operation its own
ConcurrentVessel, solved one after another (sequential modular).

The model is ``../Flowsheet/Flowsheet_simplified.py``'s: every solid fed to
CR02H, D02H, D01H, CR02R, D01R and D03R goes onto an extra "feed solid"
phase that PharmaPy's native ``DirectTransfer.first_order`` mechanism turns
into liquid, all dissolution rates are zero, and D02R (the W02R wash) takes
the F01R cake's crystals into its own population balances. The coupling is
``Flowsheet_sequential.py``'s: each vessel is wrapped in a ConcurrentVessel
of its own, with its own integrator, and the units are solved in process
order, each over the whole run::

    CR01H -> CR02H -> [F01H] -> D02H -> [F02H] -> D01H -> CR02R -> D01R
          -> [F01R] -> D02R -> [F02R] -> D03R

A unit's feed from upstream is the upstream unit's recorded outlet,
interpolated linearly in time between the saved points. It goes through the
very same link step as in Flowsheet_simplified (``DirectFeedInlet``, or
``SlurryInlet`` into D02R, filters included): only the source changes, from
the upstream vessel's live outlet to ``RecordedOutlet``, a stand-in that
answers the same questions from the saved trajectory.

The filters stay algebraic steps in the feed of the unit after them: a
ConcurrentVessel takes only MultiPhaseVessel members, and neither a
ContinuousFilter nor another ConcurrentVessel is one.

Everything else (the DirectTransfer setup, builders, grids, solver options,
report and figures) is imported from Flowsheet_simplified, so a change there
carries over here. Do not import Flowsheet.py into the same process: both
modules set settings on the shared unit modules at import.

Run from an activated environment, or LAPACK fails to load::

    conda activate PharmaPy
    python Flowsheet_sequential_direct.py                 # solve, save, show
    python Flowsheet_sequential_direct.py --no-show       # solve and save only
    python Flowsheet_sequential_direct.py --runtime 600   # a short wiring check

Figures are written to ``results_direct/`` next to this file, with
Flowsheet_simplified's names (Flowsheet_profiles.png, Flowsheet_streams.png).
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
RESULTS_DIR = HERE / "results_direct"
sys.path.insert(0, str(PROJECT / "Flowsheet"))

# Importing it applies the simplification (zero dissolution, dropped solids,
# D02R's kinetics) and the shared size grids to the unit modules.
import Flowsheet_simplified as base  # noqa: E402
from Flowsheet_simplified import (  # noqa: E402
    CR01H, CR02H, CR02R, D01H, D01R, D02H, D02R, D03R, F01H, F01R, F02H,
    F02R, CR_SOLVENT, DB_PATH, DIRECT_FED, REPORT_POINTS, RUNTIME,
    SOLVER_OPTIONS, DirectFeedInlet, FeedSolidInlet, SlurryInlet,
    SolidCakeInlet, add_feed_solid, cake_template_f01h, cake_template_r,
    direct_template, liquid_template, pass_through, product_template,
    species_names)
from PharmaPy.ConcurrentVessel import ConcurrentVessel  # noqa: E402
from PharmaPy.DataClasses import (  # noqa: E402
    PhaseMapping, PhaseRef, StreamConnection)
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


# ------------------------------------------------- recorded feeds
class RecordedOutlet:
    """A solved vessel's outlet, read back in time like a live vessel.

    Answers what ``Flowsheet_simplified.live_outlet`` asks of a vessel
    (``outlet_values``, ``_last_evaluation``, ``_state_value_at`` for each
    crystal solid's distribution), from the saved trajectory, interpolated
    linearly at ``_last_evaluation``. Holds only arrays.
    """

    def __init__(self, vessel, n_solids):
        out = vessel.outputs
        liquid = vessel.Phases.get_phase_from_ref(PhaseRef("liquid", 0))
        self.time = np.asarray(vessel.result.time, dtype=float).ravel()
        self.vol_flow = np.ravel(out["vol_flow"]).astype(float)
        self.mole_conc = np.atleast_2d(out["mole_conc"]).astype(float)
        self.temp = (np.ravel(out["temp"]).astype(float) if "temp" in out
                     else np.full(self.time.size, float(liquid.temp)))
        self.distribs = [np.atleast_2d(getattr(vessel.result,
                                               f"distrib_solid{k}"))
                         .astype(float) for k in range(n_solids)]
        self._last_evaluation = 0.0

    def _lerp(self, values, time):
        t = self.time
        time = min(max(float(time), t[0]), t[-1])
        i = min(int(np.searchsorted(t, time, side="right")) - 1, t.size - 2)
        w = (time - t[i]) / (t[i + 1] - t[i])
        return (1.0 - w) * values[i] + w * values[i + 1]

    def outlet_values(self):
        now = self._last_evaluation
        return {"vol_flow": float(self._lerp(self.vol_flow, now)),
                "mole_conc": self._lerp(self.mole_conc, now),
                "temp": float(self._lerp(self.temp, now))}

    def _state_value_at(self, key, point):
        return self._lerp(self.distribs[key.phaseref.index], point)


class Recorded:
    """Mixin: a Flowsheet_simplified link inlet fed by a RecordedOutlet.

    The link's own ``live_values`` runs unchanged; it just reads the stand-in.
    It depends on time alone, so the sink caches it per time like any
    recorded upstream trajectory, and copies freely in the results replay.
    """

    follows_state = False

    def __init__(self, source, *args, **kwargs):
        super().__init__(source, *args, **kwargs)  # shape from the real vessel
        self.source = RecordedOutlet(source, len(self.shape.mass_weights))

    def evaluate_inputs(self, time):
        self.source._last_evaluation = float(time)
        return super().evaluate_inputs(time)

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        for name, value in self.__dict__.items():
            copied.__dict__[name] = copy.deepcopy(value, memo)
        return copied


class RecordedDirectFeedInlet(Recorded, DirectFeedInlet):
    """DirectFeedInlet (liquid + crystal mass onto the feed solid), recorded."""


class RecordedSlurryInlet(Recorded, SlurryInlet):
    """SlurryInlet (crystals into D02R's population balances), recorded."""


def plug_direct(sink, upstream, template, separator=None, index=0):
    """Flowsheet_simplified.attach without a link: the stream's liquid onto
    the sink's liquid and its one solid onto the sink's feed solid."""
    stream = copy.deepcopy(template)
    liquid, solid = stream.Phases
    inlet = RecordedDirectFeedInlet(upstream, liquid.mass_frac, separator)
    liquid.DynamicInlet = inlet
    solid.DynamicInlet = FeedSolidInlet(inlet, solid.mass_frac)

    connection = StreamConnection(
        stream=stream,
        phase_mappings=[PhaseMapping(PhaseRef("liquid", 0),
                                     PhaseRef("liquid", 0)),
                        PhaseMapping(PhaseRef("solid", 0),
                                     sink.feed_solid_ref)])
    connections = list(sink.inlet_connections)
    connections[index] = connection
    sink.inlet_connections = connections
    return inlet


def plug_slurry(sink, inlet, template, index=0):
    """Flowsheet_simplified.attach_slurry without a link."""
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
    return inlet


# ---------------------------------------------------------------- builders
def build_vessels() -> dict:
    """Every vessel as in Flowsheet_simplified.build_flowsheet."""
    names = species_names()
    vessels = {
        "CR01H": CR01H.build_reactor(DB_PATH),
        "CR02H": CR02H.build_reactor(DB_PATH,
                                     feed=liquid_template(names, "H")),
        "D02H": D02H.build_dissolver(cake_template_f01h(names)),
        "D01H": D01H.build_dissolver(cake_template_f01h(names)),
        "CR02R": CR02R.build_reactor(feed=cake_template_f01h(names),
                                     makeup=CR02R.makeup_stream(names)),
        "D01R": D01R.build_crystallizer(product_template(names)),
        "D02R": pass_through(D02R.build_dissolver(
            cake_template_r(names, D02R))),
        "D03R": D03R.build_dissolver(cake_template_r(names, D03R)),
    }
    for name in DIRECT_FED:
        add_feed_solid(vessels[name])
    return vessels


def wire(unit, upstream, filter_module, vessels, names):
    """Plug ``unit``'s feed from ``upstream``'s recorded outlet, with the
    template Flowsheet_simplified uses for that link."""
    sink, source = vessels[unit], vessels[upstream]
    separator = (filter_module.build_filter() if filter_module is not None
                 else None)
    h_cake = direct_template(cake_template_f01h(names), ("F",))

    if unit == "D02R":
        return plug_slurry(sink, RecordedSlurryInlet(
            source, D02R.CRYSTAL_SPECIES, separator),
            cake_template_r(names, D02R))
    if unit == "CR02H":
        template = direct_template(liquid_template(names, CR_SOLVENT), ("F",))
    elif unit == "D01R":
        template = direct_template(product_template(names),
                                   CR02R.CRYSTAL_SPECIES)
    elif unit == "D03R":
        template = direct_template(cake_template_r(names, D03R),
                                   D03R.CRYSTAL_SPECIES)
    else:  # D02H, D01H, CR02R
        template = h_cake
    return plug_direct(sink, source, template, separator)


def run_flowsheet(runtime: float = RUNTIME, points: int = REPORT_POINTS,
                  verbose: bool = True) -> SimpleNamespace:
    """Solve every unit's ConcurrentVessel in process order."""
    names = species_names()
    vessels = build_vessels()
    grid = np.linspace(0.0, runtime, points)
    units, walls = {}, {}

    for unit, upstream, filter_module in SEQUENCE:
        if upstream is not None:
            wire(unit, upstream, filter_module, vessels, names)

        cv = ConcurrentVessel(
            {unit: vessels[unit]},
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

    return SimpleNamespace(units=units, vessels=vessels, walls=walls)


# ---------------------------------------------------------------- results
def as_plant(run) -> SimpleNamespace:
    """The solved units in the shape Flowsheet_simplified's report reads."""
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
    """Flowsheet_simplified's report, with a header for the sequential solve."""
    plant = as_plant(run)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        base.print_report(plant, wall)
    header = (f"Sequential solve (DirectTransfer): {len(run.units)} "
              f"ConcurrentVessels, one per unit, "
              f"{plant.solver_state_collection.dim} states in all, "
              f"{len(plant.result.time)} saved points, t = "
              f"{plant.result.time[-1] / 3600:.1f} h, {wall / 60:.1f} min "
              f"wall, linear solver {plant.integrator.linear_solver}")
    print("\n".join(header if line.startswith("Concurrent solve:") else line
                    for line in buffer.getvalue().splitlines()))


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
