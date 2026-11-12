"""The whole process as one ConcurrentVessel, on one integrator.

The flowsheet is::

    FEED_A --+              A_PUSH --+           WASH_H --+           FEED_H --+
             +--> CR01H --> CR02H ---+--> [F01H] ---> W02H(D02H -> [F02H]) ---+--> D01H --+
    FEED_C --+                              |                  |                          |
                                            +--> filtrate 1    +--> wash filtrate H       |
                                                                                          |
       +----------------------------------------------------------------------------------+
       |
       +--> R02R --> D01R --> [F01R] ---> W02R(D02R -> [F02R]) ---> D03R --> product
             ^        ^         |           ^         |             ^
    R02R makeup   FEED_B        |        WASH_B       |           FEED_L
                                +--> filtrate 2       +--> wash filtrate B

W02H and W02R are reslurry washes, each simulated as two unit ops (a wash
dissolver and a filter) but reported as one block. D02H is identical to D01H
(H, which dissolves very little F); D02R uses B, D01R's antisolvent, with
D01R's solubility. F02H and F02R have the settings of F01H and F01R.

``PharmaPy.ConcurrentVessel`` (pharmapy-org >= 0.3.3) hands every member to one
integrator as one state vector, so all units advance through each time step
together and each reads the live outlet of the unit feeding it. Every vessel
comes from its own script's builder, with that script's settings: this file
only connects them.

Five connections are not plain links, because ConcurrentVessel members must
be MultiPhaseVessels and a link assumes the sink looks like the source. Each
one is a small algebraic step inside the link, re-evaluated on every
right-hand side (see the ``CoupledInlet`` subclasses below):

- ``F01H``, ``F02H``, ``F01R`` and ``F02R`` are the steady-state filters, with
  their settings taken from each script's ``build_filter``. The standalone
  F01H and F01R scripts invent crystals from a Gaussian because their upstream
  reactor is all liquid; here the filters get the real crystals from CR02H,
  D02H, D01R and D02R instead.
- ``D01H -> R02R``: D01H discharges a slurry but R02R has no solid phase. The
  crystals are assumed to dissolve on entering R02R, conserving mass. See
  ``SLURRY_TO_LIQUID`` for why this is defensible and what it hides.

Where this differs from running the scripts one at a time
---------------------------------------------------------
- CR02H receives CR01H's crystals (``CR02H.FEED_SOLIDS`` is False standalone).
- D02H and D01H run on CR01H's size grid, D02R and D03R on D01R's. The real
  crystals are mm sized, which the 1-401 um grid of D01H cannot hold.
- R02R's F comes only from D01H. ``R02R_MAKEUP`` is R02R's own feed with the F
  taken out, at the same mass flows of everything else.
- D02H's and D01H's S0 (the same, the vessels being identical) is fixed at the
  CR02H mother liquor given by ``CAKE_LIQUID_DESIGN``, since a kinetics object
  cannot read a stream that starts empty. The report warns if the run ends up
  elsewhere.

Solver
------
CVode (Assimulo) with GMRES, as the crystallizers ask for, plus a preconditioner
of one Jacobian block per vessel (``BlockPreconditionedCVode``). ~2800 states
rule out a dense Jacobian, and GMRES alone stalls on the dissolution kinetics.
Run from an activated environment, or LAPACK fails to load::

    conda activate PharmaPy
    python Flowsheet.py                 # solve, save figures, show them
    python Flowsheet.py --no-show       # solve and save figures only
    python Flowsheet.py --runtime 600   # a short run, to check the wiring

Figures are written to ``results/`` next to this file:

    Flowsheet_profiles.png  what each vessel discharges vs time
    Flowsheet_streams.png   every stream at the final time
"""

from __future__ import annotations

import argparse
import copy
import sys
import time as clock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from PharmaPy.ConcurrentVessel import ConcurrentVessel, CoupledInlet, StreamLink
from PharmaPy.DataClasses import PhaseRef, StateKey
from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Mechanisms import PopulationBalanceMechanism
from PharmaPy.MixedPhases_Refactored import MixedStream

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent          # Project work/Flowsheet
PROJECT = HERE.parent                           # Project work
RESULTS_DIR = HERE / "results"

for folder in ("D03R", "F02R", "D02R", "F01R", "D01R", "R02R", "F02H", "D02H",
               "D01H", "F01H", "R01H"):
    sys.path.insert(0, str(PROJECT / folder))

import CR01H  # noqa: E402
import CR02H  # noqa: E402
import D01H  # noqa: E402
import D01R  # noqa: E402
import D02H  # noqa: E402
import D02R  # noqa: E402
import D03R  # noqa: E402
import F01H  # noqa: E402
import F01R  # noqa: E402
import F02H  # noqa: E402
import F02R  # noqa: E402
import R02R  # noqa: E402
from continuous_filter import (  # noqa: E402
    SlurryFlow, flow_from_vessel, stream_from_flow)

DB_PATH = F01H.DB_PATH

# ---------------------------------------------------------------- settings
# 16 h: R02R and D01R are the slow units (tau ~1.1 h and ~0.7 h, D01R needing
# 12 of them), and D01R only starts settling once R02R has.
RUNTIME = 16 * 3600.0  # s
REPORT_POINTS = 481  # saved time points; the results replay visits each one
SOLVER_OPTIONS = {"maxh": 120.0, "rtol": 1e-6, "atol": 1e-8,
                  "precond": "PREC_RIGHT"}

# The crystal populations travel between these vessels unchanged, so each pair
# shares one grid; see the module docstring.
D01H.CRYSTAL_SIZE_GRID = CR01H.CRYSTAL_SIZE_GRID  # D02H builds through D01H
D02H.CRYSTAL_SIZE_GRID = CR01H.CRYSTAL_SIZE_GRID
F01R.CRYSTAL_SIZE_GRID = D01R.CRYSTAL_SIZE_GRID
D02R.CRYSTAL_SIZE_GRID = D01R.CRYSTAL_SIZE_GRID
D03R.CRYSTAL_SIZE_GRID = D01R.CRYSTAL_SIZE_GRID

# CR02H's mother liquor at steady state [kg/s liquid], which fixes the S0 of
# D02H and D01H (D01H.solubility_s0 reads the cake liquid's [F]) and their
# temperature.
CAKE_LIQUID_DESIGN = {"A": 1.4445e-4, "B": 8.2248e-4, "C": 8.76e-8,
                      "F": 2.846e-6}
CAKE_TEMPERATURE_DESIGN = 281.01  # K
CAKE_DESIGN_TOL = 0.05  # relative; the report warns beyond this

# R02R has no solid phase, so D01H's crystals join its liquid as dissolved F.
# R02R has no F solubility model to check this against. The standalone feed
# carries 0.509 mol/L of dissolved F, well above what reaches R02R here, which
# is the basis for assuming it all dissolves.
SLURRY_TO_LIQUID = {"species": "F"}

# Categorical slots, fixed per species across every panel.
SPECIES_COLORS = {"F": "#2a78d6", "D": "#eb6834", "E": "#1baf7a", "J": "#8a5cd1"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID_COLOR = "#e5e4e0"

CR_SOLVENT = "B"  # the CR01H and CR02H feeds are B-based

UNIT_CRYSTALS = {"CR01H": ("F",), "CR02H": ("F",), "D02H": ("F",),
                 "D01H": ("F",), "R02R": (), "D01R": D01R.CRYSTAL_SPECIES,
                 "D02R": D02R.CRYSTAL_SPECIES, "D03R": D03R.CRYSTAL_SPECIES}

# Wash blocks: each is two unit ops, reported as one block.
WASH_BLOCKS = {"W02H": ("D02H", "F02H"), "W02R": ("D02R", "F02R")}
PANEL_TITLES = {"D02H": "W02H (D02H wash)", "D02R": "W02R (D02R wash)"}


# ------------------------------------------------- slurry bookkeeping
class SlurryShape:
    """What is needed to turn a vessel's outlet into absolute flows."""

    def __init__(self, vessel):
        liquid = vessel.Phases.get_phase_from_ref(PhaseRef("liquid", 0))
        self.mw = np.asarray(liquid.mw, dtype=float)
        self.names = list(liquid.name_species)
        self.mass_weights = []  # [kg/#] x dx per cell, one array per solid
        self.vol_weights = []  # [m3/#] x dx per cell, one array per solid

        for phase in vessel.Phases:
            if phase.phase_family != "solid":
                continue
            pbm = phase.get_mechanism(PopulationBalanceMechanism)
            vol = pbm.kv * np.asarray(pbm.x_grid)**3 * pbm.dx * 1e-18
            self.vol_weights.append(vol)
            self.mass_weights.append(pbm.density * vol)

    def absolute(self, vol_flow, mole_conc, distribs):
        """Liquid volume flow [m3/s] and crystal number flows [#/(um s)].

        PharmaPy keeps each solid's density per V_liquid / (1 - phi_k), which
        with several solids is not the vessel volume (see D01R.slurry_volumes).
        With one solid this reduces to Q_L = (1 - phi) Q and N = n Q, the same
        as ``continuous_filter.flow_from_vessel``.
        """
        phis = [float(d @ w) for d, w in zip(distribs, self.vol_weights)]
        total = 1.0 + sum(phi / (1.0 - phi) for phi in phis)
        liquid_flow = vol_flow / total
        crystal_flows = [d * vol_flow / ((1.0 - phi) * total)
                         for d, phi in zip(distribs, phis)]
        return liquid_flow, crystal_flows

    def liquid_mass_j(self, liquid_flow, mole_conc):
        return np.asarray(mole_conc, dtype=float) * self.mw * liquid_flow

    def crystal_mass(self, crystal_flows):
        return [float(n @ w) for n, w in zip(crystal_flows, self.mass_weights)]


def live_outlet(vessel, shape):
    """The vessel's outlet at the point last evaluated, as absolute flows."""
    values = vessel.outlet_values()
    point = vessel._last_evaluation
    # outlet_values only publishes solid 0, and D01R has three solids.
    distribs = [
        np.maximum(np.asarray(vessel._state_value_at(
            StateKey("distrib", PhaseRef("solid", k)), point), dtype=float),
            0.0)
        for k in range(len(shape.mass_weights))
    ]
    conc = np.maximum(np.asarray(values["mole_conc"], dtype=float), 0.0)
    vol_flow = max(float(values["vol_flow"]), 0.0)
    liquid_flow, crystal_flows = shape.absolute(vol_flow, conc, distribs)
    return float(values["temp"]), conc, liquid_flow, crystal_flows


def split_cake(separator, shape, conc, liquid_flow, crystal_flows):
    """ContinuousFilter's split, without building any streams.

    Returns (cake liquid flow [m3/s], cake crystal flows). The filter itself
    costs too much to call on every right-hand side, since it constructs
    PharmaPy phases; ``check_filter`` compares the two at the final time.
    """
    keep = 1.0 - float(separator.solids_loss)
    cake_crystals = [n * keep for n in crystal_flows]
    retained = (sum(shape.crystal_mass(cake_crystals))
                * separator._liquid_per_solid())
    available = float(shape.liquid_mass_j(liquid_flow, conc).sum())
    held = min(retained / available, 1.0) if available > 0 else 0.0
    return held * liquid_flow, cake_crystals


def fold_top_cell(distrib, vol_weights):
    """Move the top size cell's crystals into the cell below, keeping volume.

    PharmaPy's finite-volume scheme pads the grid with a copy of its last
    cell (Mechanisms.py, ``f_aug = [..., csd, csd[-1]]``). For growth the top
    face is an outflow and that is harmless; for dissolution it is the inflow,
    so a populated top cell draws in crystals from beyond the grid at its own
    density, never drains, and fills every cell below. Those crystals then
    dissolve into solute nobody fed: once D01R's E crystals reached the top
    cell (~14 h), the L dissolver (now D03R) made ~0.03 g/s of E from nothing.
    A dissolver fed an empty top cell keeps it empty, so the cake links feed
    one.
    """
    n = np.array(distrib, dtype=float)
    n[-2] += n[-1] * vol_weights[-1] / vol_weights[-2]
    n[-1] = 0.0
    return n


# ------------------------------------------------- algebraic link steps
class AlgebraicInlet(CoupledInlet):
    """A coupled inlet that keeps its class when copied.

    CoupledInlet.__deepcopy__ builds a plain CoupledInlet, which would silently
    turn a filter back into a straight pipe in the results replay.
    """

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied

        for name, value in self.__dict__.items():
            if name == "source":
                copied.source = memo.get(id(value))
            else:
                copied.__dict__[name] = copy.deepcopy(value, memo)

        return copied


class FilterCakeInlet(AlgebraicInlet):
    """The cake of a filter on the source's outlet: F01H or F01R.

    Serves the sink's liquid phase. The liquid carries the mean of the solid
    densities under ``distrib`` so that MultiPhaseVessel._inlet_phase_shares,
    which reads one trajectory for all of a connection's solids, adds up the
    correct total solid volume. ``SolidCakeInlet`` serves each solid its own.
    """

    def __init__(self, source, separator):
        super().__init__(source)
        self.separator = separator
        self.shape = SlurryShape(source)
        self.solids = None

    def live_values(self):
        if self.source is None:
            raise RuntimeError("This filter inlet was copied without its "
                               "source; solve through the ConcurrentVessel.")

        temp, conc, liquid_flow, crystals = live_outlet(self.source, self.shape)
        cake_liquid, cake_crystals = split_cake(
            self.separator, self.shape, conc, liquid_flow, crystals)
        cake_crystals = [fold_top_cell(n, w) for n, w in
                         zip(cake_crystals, self.shape.vol_weights)]

        cake_flow = cake_liquid + sum(
            float(n @ w) for n, w in zip(cake_crystals, self.shape.vol_weights))

        if cake_flow > 0:
            self.solids = [n / cake_flow for n in cake_crystals]
        else:
            self.solids = [np.zeros_like(n) for n in cake_crystals]

        return {
            "vol_flow": cake_flow,
            "mole_conc": conc,
            "temp": temp,
            "distrib": np.mean(self.solids, axis=0),
        }


class SolidCakeInlet:
    """One solid's share of a FilterCakeInlet."""

    follows_state = True

    def __init__(self, cake, index):
        self.cake = cake
        self.index = index
        self.controllable = ()
        self.parent_instance = None

    def evaluate_inputs(self, time):
        # The cake evaluates first, or is frozen at what it last served, and
        # either way self.cake.solids is the population that goes with it.
        values = self.cake.evaluate_inputs(time)
        return {"temp": values["temp"],
                "distrib": self.cake.solids[self.index].copy()}


class DissolvedSlurryInlet(AlgebraicInlet):
    """A slurry delivered to a liquid-only vessel; see SLURRY_TO_LIQUID."""

    def __init__(self, source, species):
        super().__init__(source)
        self.shape = SlurryShape(source)
        self.target = self.shape.names.index(species)

    def live_values(self):
        if self.source is None:
            raise RuntimeError("This inlet was copied without its source; "
                               "solve through the ConcurrentVessel.")

        temp, conc, liquid_flow, crystals = live_outlet(self.source, self.shape)
        mass_j = self.shape.liquid_mass_j(liquid_flow, conc)
        mass_j[self.target] += sum(self.shape.crystal_mass(crystals))
        total = float(mass_j.sum())

        if total <= 0.0:  # a solver iterate can throttle the outlet to nothing
            return {"mass_flow": 0.0, "temp": temp}

        return {"mass_flow": total, "mass_frac": mass_j / total, "temp": temp}


class SlurryFeedMassFlow:
    """Mixin: report a live solid feed's true mass flow.

    On a live link the solid stream's mass is its density times a slurry volume
    still holding the source vessel's volume (m3) rather than a flow (m3/s), so
    it reads ~1000x too high. The material balance never uses it, since a solid
    has no material slice, but the inlet enthalpy term does: it added ~60 W to
    CR02H's 1 L and held it 5.6 K above its true temperature. Here the flow is
    recomputed from the distribution and the stream's slurry flow.
    """

    def resolve_inlets(self, completed_state, operating_conditions,
                       inlet_inputs=None):
        resolved = super().resolve_inlets(
            completed_state, operating_conditions, inlet_inputs)

        for connection in resolved:
            slurry_flow = sum(float(t.vol_flow or 0.0) for t in connection)

            for transfer in connection:
                phase = transfer.stream_phase

                if phase.phase_family != "solid":
                    continue

                pbm = transfer.vessel_phase.get_mechanism(
                    PopulationBalanceMechanism)
                fed = phase.get_mechanism(PopulationBalanceMechanism)
                distrib = np.asarray(getattr(fed, fed.flow_state_name),
                                     dtype=float)
                mass = float(pbm.density * pbm.kv * 1e-18 * slurry_flow
                             * pbm.compute_third_moment(distrib))
                transfer.species_flow = (np.asarray(phase.mass_frac,
                                                    dtype=float) * mass)

        return resolved


def with_true_slurry_feed(vessel):
    cls = type(vessel)
    vessel.__class__ = type(cls.__name__, (SlurryFeedMassFlow, cls), {})
    return vessel


class BlockPreconditionedCVode(AssimuloBackend):
    """CVode with GMRES, preconditioned by one Jacobian block per vessel.

    Unpreconditioned GMRES stalls at ~1150 s, when crystals reach D01R and
    the L dissolver (now D03R): dissolution at k_d = 5e5 um/s on a 10 um grid has eigenvalues near
    5e4 /s. A dense Jacobian of the whole plant would cost ~2800 evaluations of
    every vessel each time. But this plant has no recycle, so its Jacobian
    is block lower triangular: each vessel's own block carries the stiffness,
    and the links only couple it to the vessel upstream. Each block is found by
    finite differences on that vessel alone with the couplings frozen, and GMRES
    still multiplies by the true Jacobian, so the links are not approximated in
    the answer, only in the preconditioner.
    """

    PERTURBATION = np.sqrt(np.finfo(float).eps)

    def set_ode_problem(self, unit, states_init):
        super().set_ode_problem(unit, states_init)
        self._unit = unit
        self._blocks = None
        self._factors = None
        self.refreshes = 0
        self._started = clock.time()
        self._problem.prec_setup = self._prec_setup
        self._problem.prec_solve = self._prec_solve

    def _member_block(self, member, time, states):
        """d(rates)/d(states) of one vessel, its inlets held fixed."""
        states = np.array(states, dtype=float)
        collection = member.solver_state_collection

        # A state at zero is perturbed on the scale of its own group: a crystal
        # size cell by the distribution's largest, a species by the total mass.
        floor = np.ones(states.size)
        for block in collection.slices.values():
            floor[block] = max(1.0e-3 * float(np.abs(states[block]).max()),
                               1.0e-12)

        member.invalidate_inlet_cache()
        base = np.array(member.unit_model(time=time, states=states),
                        dtype=float)
        block = np.zeros((states.size, states.size))

        for j in range(states.size):
            step = self.PERTURBATION * max(abs(states[j]), floor[j])
            trial = states.copy()
            trial[j] += step
            member.invalidate_inlet_cache()
            block[:, j] = (np.asarray(member.unit_model(time=time,
                                                        states=trial))
                           - base) / step

        # One bad column must not poison the factorization: it only makes the
        # preconditioner weaker, and GMRES still solves the true system.
        return np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

    def _prec_setup(self, t, y, fy, jok, gamma, data):
        unit = self._unit
        y = np.asarray(y, dtype=float)
        refreshed = False

        if not jok or self._blocks is None:
            unit._eval_key = None
            unit._evaluate(t, y)
            with unit._frozen_couplings():
                self._blocks = {
                    name: self._member_block(
                        unit.members[name], t, y[unit._member_slices[name]])
                    for name in unit.order
                }
            # The vessels were left at the last perturbed state.
            unit._eval_key = None
            refreshed = True
            self.refreshes += 1
            if self.refreshes % 25 == 0:
                print(f"  t = {float(t) / 3600:.2f} h, {self.refreshes} "
                      f"preconditioner refreshes, "
                      f"{(clock.time() - self._started) / 60:.1f} min wall",
                      flush=True)

        self._factors = {
            name: lu_factor(np.eye(block.shape[0]) - gamma * block,
                            check_finite=False)
            for name, block in self._blocks.items()
        }
        return [refreshed, data]

    def _prec_solve(self, t, y, fy, r, gamma, delta, data):
        unit = self._unit
        solved = np.array(r, dtype=float)
        for name, factors in self._factors.items():
            piece = unit._member_slices[name]
            solved[piece] = lu_solve(factors, solved[piece],
                                     check_finite=False)
        return solved


def attach(cv, link, coupling, stream):
    """Add a link, then give it ``coupling`` and the sink's own inlet stream.

    add_link hands the sink a copy of the source's outlet stream and a plain
    coupling; neither fits a link with a filter in it.
    """
    cv.add_link(link)
    sink = cv.members[link.sink]
    connection = sink.make_stream_connection(copy.deepcopy(stream))
    solid_index = 0

    for phase in connection.stream:
        if phase.phase_family == "solid":
            phase.DynamicInlet = SolidCakeInlet(coupling, solid_index)
            solid_index += 1
        else:
            phase.DynamicInlet = coupling

    connections = list(sink.inlet_connections)
    connections[link.inlet_index] = connection
    sink.inlet_connections = connections
    link.coupling = coupling


# ---------------------------------------------------------------- builders
def species_names() -> tuple[str, ...]:
    return R02R.database_species(DB_PATH)


def cake_template_f01h(names) -> MixedStream:
    """Inlet template for D01H. Its liquid sets S0 and the tank temperature."""
    liquid = np.array([CAKE_LIQUID_DESIGN.get(n, 0.0) for n in names])
    solid_frac = np.zeros(len(names))
    solid_frac[names.index("F")] = 1.0
    density = F01H.solid_density(DB_PATH)
    flow = SlurryFlow(
        path_thermo=str(DB_PATH), name_species=list(names),
        temp=CAKE_TEMPERATURE_DESIGN, liquid_mass_j=liquid, name_solv="B",
        x_grid=np.asarray(F01H.CRYSTAL_SIZE_GRID, dtype=float),
        crystal_flow=F01H.crystal_number_flow(3.5e-4, density),
        solid_mass_frac=solid_frac, solid_density=density,
        kv=F01H.CRYSTAL_KV, target_components=["F"], solvent_name="B")
    return stream_from_flow(flow, include_solid=True)


def cake_template_r(names, dissolver=D02R) -> MixedStream:
    """Inlet template for D02R or D03R (``dissolver``, the module): a liquid,
    then a D, an E and a J solid, built by that dissolver's ``cake_stream``."""
    liquid = np.zeros(len(names))
    liquid[names.index("H")] = 1.0e-4
    species = dissolver.CRYSTAL_SPECIES
    solid_frac = np.zeros(len(names))
    solid_frac[[names.index(s) for s in species]] = 1.0 / len(species)
    density = F01R.mixed_density({s: 1.0 for s in species},
                                 F01R.solid_densities(DB_PATH))
    flow = SlurryFlow(
        path_thermo=str(DB_PATH), name_species=list(names),
        temp=D01R.CRYSTALLIZER_TEMPERATURE, liquid_mass_j=liquid,
        name_solv="H",
        x_grid=np.asarray(dissolver.CRYSTAL_SIZE_GRID, dtype=float),
        crystal_flow=F01R.crystal_number_flow(1.0e-4, density),
        solid_mass_frac=solid_frac, solid_density=density,
        kv=dissolver.CRYSTAL_KV, target_components=list(species),
        solvent_name="H")
    return dissolver.cake_stream(flow)


def r02r_makeup(names):
    """R02R's own feed with the F taken out, every other mass flow unchanged."""
    _, flow, basis, composition = R02R.FEEDS[0]
    fractions = R02R.molar_to_mass_fraction(composition, DB_PATH)
    without_f = {k: v for k, v in composition.items() if k != "F"}
    return R02R.make_feed(DB_PATH, without_f, flow * (1.0 - fractions["F"]),
                          names, R02R.database_solvent(names), basis)


def liquid_template(names, solvent):
    return MixedStream([R02R.make_feed(DB_PATH, {}, 1.0e-3, names, solvent)])


def build_flowsheet() -> ConcurrentVessel:
    """Every vessel from its own script, connected as one system."""
    names = species_names()

    cr01 = CR01H.build_reactor(DB_PATH)
    cr02 = with_true_slurry_feed(CR02H.build_reactor(
        DB_PATH, feed=liquid_template(names, "H")))
    # D02H and D01H: the same vessel, the same design template (S0, T).
    d02h = with_true_slurry_feed(D02H.build_dissolver(cake_template_f01h(names)))
    d01h = with_true_slurry_feed(D01H.build_dissolver(cake_template_f01h(names)))
    r02r = R02R.build_reactor(DB_PATH)
    r02r.Inlet = [r02r_makeup(names)]
    d01r = D01R.build_crystallizer(liquid_template(names, "H"))
    d02r = with_true_slurry_feed(D02R.build_dissolver(cake_template_r(names, D02R)))
    d03r = with_true_slurry_feed(D03R.build_dissolver(cake_template_r(names, D03R)))

    cv = ConcurrentVessel(
        {"CR01H": cr01, "CR02H": cr02, "D02H": d02h, "D01H": d01h,
         "R02R": r02r, "D01R": d01r, "D02R": d02r, "D03R": d03r},
        integrator=BlockPreconditionedCVode(options=dict(SOLVER_OPTIONS)),
    )

    cv.add_link(StreamLink("CR01H", "CR02H", inlet=0))
    attach(cv, StreamLink("CR02H", "D02H", inlet=0, name="F01H"),
           FilterCakeInlet(cr02, F01H.build_filter()),
           cake_template_f01h(names))
    attach(cv, StreamLink("D02H", "D01H", inlet=0, name="F02H"),
           FilterCakeInlet(d02h, F02H.build_filter()),
           cake_template_f01h(names))
    attach(cv, StreamLink("D01H", "R02R", name="D01H->R02R"),
           DissolvedSlurryInlet(d01h, SLURRY_TO_LIQUID["species"]),
           liquid_template(names, "H"))
    cv.add_link(StreamLink("R02R", "D01R", inlet=0))
    attach(cv, StreamLink("D01R", "D02R", inlet=0, name="F01R"),
           FilterCakeInlet(d01r, F01R.build_filter()),
           cake_template_r(names, D02R))
    attach(cv, StreamLink("D02R", "D03R", inlet=0, name="F02R"),
           FilterCakeInlet(d02r, F02R.build_filter()),
           cake_template_r(names, D03R))
    return cv


def run_flowsheet(runtime: float = RUNTIME,
                  points: int = REPORT_POINTS) -> ConcurrentVessel:
    cv = build_flowsheet()
    cv.solve_unit(time_grid=np.linspace(0.0, runtime, points), verbose=False)
    return cv


# ---------------------------------------------------------------- results
def outlet_at(vessel, index=-1) -> SimpleNamespace:
    """A solved vessel's outlet at one saved point, as absolute flows."""
    shape = SlurryShape(vessel)
    out = vessel.outputs
    liquid = vessel.Phases.get_phase_from_ref(PhaseRef("liquid", 0))
    vol_flow = float(np.ravel(out["vol_flow"])[index])
    conc = np.maximum(np.atleast_2d(out["mole_conc"])[index], 0.0)
    temp = (float(np.ravel(out["temp"])[index]) if "temp" in out
            else float(liquid.temp))
    distribs = [np.maximum(np.atleast_2d(getattr(vessel.result,
                                                 f"distrib_solid{k}"))[index],
                           0.0)
                for k in range(len(shape.mass_weights))]
    liquid_flow, crystals = shape.absolute(vol_flow, conc, distribs)
    return SimpleNamespace(
        temp=temp, conc=conc, shape=shape, liquid_flow=liquid_flow,
        crystals=crystals,
        liquid_mass_j=shape.liquid_mass_j(liquid_flow, conc),
        solid_mass=shape.crystal_mass(crystals))


def stream(temp, liquid_mass_j, solids=None) -> SimpleNamespace:
    """A stream table entry: dissolved and crystal mass flows [kg/s]."""
    return SimpleNamespace(temp=temp, liquid=np.asarray(liquid_mass_j),
                           solids=dict(solids or {}))


def total(entry) -> float:
    return float(entry.liquid.sum()) + sum(entry.solids.values())


def from_outlet(out, species) -> SimpleNamespace:
    return stream(out.temp, out.liquid_mass_j, zip(species, out.solid_mass))


def filter_streams(out, separator, species):
    """(cake, filtrate) of a filter fed by outlet ``out``."""
    held_flow, cake_crystals = split_cake(separator, out.shape, out.conc,
                                          out.liquid_flow, out.crystals)
    cake_liquid = out.shape.liquid_mass_j(held_flow, out.conc)
    cake_mass = out.shape.crystal_mass(cake_crystals)
    cake = stream(out.temp, cake_liquid, zip(species, cake_mass))
    filtrate = stream(
        out.temp, out.liquid_mass_j - cake_liquid,
        ((s, m - c) for s, m, c in zip(species, out.solid_mass, cake_mass)))
    return cake, filtrate


def fresh_feed(feed) -> SimpleNamespace:
    """A stream table entry for one of the fresh liquid feeds."""
    return stream(float(feed.temp),
                  float(feed.mass_flow) * np.asarray(feed.mass_frac))


def stream_table(cv) -> dict[str, SimpleNamespace]:
    """Every stream at the final time, in process order."""
    names = species_names()
    outlets = {name: outlet_at(unit) for name, unit in cv.members.items()}
    entries = {
        name: from_outlet(outlets[name], UNIT_CRYSTALS[name])
        for name in cv.members
    }
    cake_1, filtrate_1 = filter_streams(outlets["CR02H"], F01H.build_filter(),
                                        UNIT_CRYSTALS["CR02H"])
    wash_h_cake, wash_h_filtrate = filter_streams(
        outlets["D02H"], F02H.build_filter(), UNIT_CRYSTALS["D02H"])
    cake_2, filtrate_2 = filter_streams(outlets["D01R"], F01R.build_filter(),
                                        UNIT_CRYSTALS["D01R"])
    wash_r_cake, wash_r_filtrate = filter_streams(
        outlets["D02R"], F02R.build_filter(), UNIT_CRYSTALS["D02R"])

    return {
        "Feed A": fresh_feed(CR01H.make_feed(
            DB_PATH, CR01H.FEED_A, CR01H.FLOW_A, names, CR_SOLVENT)),
        "Feed C": fresh_feed(CR01H.make_feed(
            DB_PATH, CR01H.FEED_B, CR01H.FLOW_B, names, CR_SOLVENT)),
        "CR01H out": entries["CR01H"],
        "A push": fresh_feed(CR01H.make_feed(
            DB_PATH, CR02H.FEED_A_PUSH, CR02H.FLOW_A_PUSH, names, CR_SOLVENT)),
        "CR02H out": entries["CR02H"],
        "F01H cake": cake_1,
        "Filtrate 1": filtrate_1,
        # W02H, one block: D02H's discharge to F02H stays inside it.
        "W02H wash H": fresh_feed(D02H.h_stream(names)),
        "W02H cake": wash_h_cake,
        "W02H filtrate": wash_h_filtrate,
        "Feed H": fresh_feed(D01H.h_stream(names)),
        "D01H out": entries["D01H"],
        "R02R makeup": fresh_feed(r02r_makeup(names)),
        "R02R out": entries["R02R"],
        "Feed B": fresh_feed(D01R.antisolvent_stream(names)),
        "D01R out": entries["D01R"],
        "F01R cake": cake_2,
        "Filtrate 2": filtrate_2,
        # W02R, one block: D02R's discharge to F02R stays inside it.
        "W02R wash B": fresh_feed(D02R.b_stream(names)),
        "W02R cake": wash_r_cake,
        "W02R filtrate": wash_r_filtrate,
        "Feed L": fresh_feed(D03R.l_stream(names)),
        "D03R product": entries["D03R"],
    }


PLANT_IN = ("Feed A", "Feed C", "A push", "W02H wash H", "Feed H",
            "R02R makeup", "Feed B", "W02R wash B", "Feed L")
PLANT_OUT = ("Filtrate 1", "W02H filtrate", "Filtrate 2", "W02R filtrate",
             "D03R product")
UNIT_BALANCES = {
    "CR01H": (("Feed A", "Feed C"), ("CR01H out",)),
    "CR02H": (("CR01H out", "A push"), ("CR02H out",)),
    "F01H": (("CR02H out",), ("F01H cake", "Filtrate 1")),
    "W02H": (("F01H cake", "W02H wash H"), ("W02H cake", "W02H filtrate")),
    "D01H": (("W02H cake", "Feed H"), ("D01H out",)),
    "R02R": (("D01H out", "R02R makeup"), ("R02R out",)),
    "D01R": (("R02R out", "Feed B"), ("D01R out",)),
    "F01R": (("D01R out",), ("F01R cake", "Filtrate 2")),
    "W02R": (("F01R cake", "W02R wash B"), ("W02R cake", "W02R filtrate")),
    "D03R": (("W02R cake", "Feed L"), ("D03R product",)),
}


def outlet_history(vessel, species) -> SimpleNamespace:
    """What a vessel discharges vs time [kg/s], per species."""
    time = np.asarray(vessel.result.time, dtype=float)
    names = SlurryShape(vessel).names
    dissolved = {s: np.zeros(time.size) for s in names}
    crystals = {s: np.zeros(time.size) for s in species}

    for i in range(time.size):
        out = outlet_at(vessel, i)
        for j, name in enumerate(names):
            dissolved[name][i] = out.liquid_mass_j[j]
        for name, mass in zip(species, out.solid_mass):
            crystals[name][i] = mass

    return SimpleNamespace(time=time, dissolved=dissolved, crystals=crystals)


def settling(vessel, species) -> float:
    """Largest relative change in a discharge over the last 10 % of the run."""
    history = outlet_history(vessel, species)
    mark = int(0.9 * (history.time.size - 1))
    series = list(history.dissolved.values()) + list(history.crystals.values())
    scale = max(float(s[-1]) for s in series)
    return max(abs(float(s[-1] - s[mark])) / max(abs(float(s[-1])), 1e-3 * scale)
               for s in series if max(abs(s[-1]), abs(s[mark])) > 1e-3 * scale)


def check_filter(cv) -> float:
    """Filter link vs the real ContinuousFilter, at CR02H's final outlet."""
    filt = F01H.build_filter()
    filt.Inlets = [flow_from_vessel(cv.members["CR02H"])]
    filt.solve()
    cake, _ = filter_streams(outlet_at(cv.members["CR02H"]),
                             F01H.build_filter(), ("F",))
    real = filt.cake_flow.liquid_mass_flow + filt.cake_flow.solid_mass_flow
    return abs(total(cake) - real) / real if real > 0 else 0.0


# ---------------------------------------------------------------- reporting
def print_report(cv, wall: float):
    table = stream_table(cv)
    names = species_names()
    stats = cv.tear_statistics

    print(f"\nConcurrent solve: {cv.solver_state_collection.dim} states, "
          f"{len(cv.result.time)} saved points, t = "
          f"{cv.result.time[-1] / 3600:.1f} h, {wall / 60:.1f} min wall, "
          f"linear solver {cv.integrator.linear_solver}, tears "
          f"{len(cv.tears)} ({stats['evaluations']} iterated)")

    print("\nStreams at the final time [g/s]:")
    print(f"  {'stream':<14}{'T [K]':>8}{'total':>10}{'crystals':>10}  "
          "main components")
    for label, entry in table.items():
        main = sorted(zip(entry.liquid, names), reverse=True)[:4]
        text = ", ".join(f"{n} {m * 1e3:.4g}" for m, n in main if m > 1e-9)
        solid = ", ".join(f"{n} {m * 1e3:.4g}" for n, m in entry.solids.items()
                          if m > 1e-9)
        print(f"  {label:<14}{entry.temp:>8.2f}{total(entry) * 1e3:>10.4f}"
              f"{sum(entry.solids.values()) * 1e3:>10.4f}  {text}"
              + (f" | solid {solid}" if solid else ""))

    print("\nMass closure [g/s], (in - out) at the final time:")
    for unit, (ins, outs) in UNIT_BALANCES.items():
        mass_in = sum(total(table[s]) for s in ins)
        mass_out = sum(total(table[s]) for s in outs)
        print(f"  {unit:<8}in {mass_in * 1e3:>9.4f}  out {mass_out * 1e3:>9.4f}"
              f"  gap {(mass_in - mass_out) * 1e3:>9.5f} "
              f"({(mass_in - mass_out) / mass_in:>7.2%})")
    plant_in = sum(total(table[s]) for s in PLANT_IN)
    plant_out = sum(total(table[s]) for s in PLANT_OUT)
    print(f"  {'Plant':<8}in {plant_in * 1e3:>9.4f}  out {plant_out * 1e3:>9.4f}"
          f"  gap {(plant_in - plant_out) * 1e3:>9.5f} "
          f"({(plant_in - plant_out) / plant_in:>7.2%})")
    print(f"  F01H link vs ContinuousFilter: {check_filter(cv):.2e} relative")

    print("\nWash blocks (cake in -> cake out):")
    for block, solvent, cake_in in (("W02H", "H", "F01H cake"),
                                    ("W02R", "B", "F01R cake")):
        dissolver = WASH_BLOCKS[block][0]
        fed = table[cake_in]
        washed = table[f"{block} cake"]
        left = sum(from_outlet(outlet_at(cv.members[dissolver]),
                               UNIT_CRYSTALS[dissolver]).solids.values())
        crystals_in = sum(fed.solids.values())
        j_wash = names.index(solvent)
        liquor_in = float(fed.liquid.sum() - fed.liquid[j_wash])
        liquor_out = float(washed.liquid.sum() - washed.liquid[j_wash])
        print(f"  {block}: {dissolver} dissolves "
              f"{1 - left / max(crystals_in, 1e-30):.2%} of the "
              f"{crystals_in * 1e3:.4f} g/s of crystals; non-{solvent} liquor "
              f"in the cake {liquor_in * 1e3:.4f} -> {liquor_out * 1e3:.4f} g/s "
              f"({1 - liquor_out / max(liquor_in, 1e-30):.1%} washed out)")

    print("\nKey results:")
    j_f = names.index("F")
    cake = table["W02H cake"]
    d01h_out = table["D01H out"]
    r02r_out = table["R02R out"]
    fed_f = sum(cake.solids.values())
    left = d01h_out.solids.get("F", 0.0)
    print(f"  D01H dissolves {1 - left / max(fed_f, 1e-30):.2%} of the F "
          f"crystals fed ({fed_f * 1e3:.4f} g/s); the rest dissolve on "
          "entering R02R.")
    f_in = total_f = d01h_out.liquid[j_f] + left
    print(f"  F reaching R02R {f_in * 1e3:.4f} g/s, against "
          f"{R02R.total_feed_flow() * R02R.molar_to_mass_fraction(R02R.FEEDS[0][3], DB_PATH)['F'] * 1e3:.4f} g/s in its "
          f"standalone feed; {1 - r02r_out.liquid[j_f] / max(total_f, 1e-30):.1%} converted.")
    for species in D03R.CRYSTAL_SPECIES:
        j = names.index(species)
        fed = table["W02R cake"].solids[species]
        out = table["D03R product"]
        solid = out.solids[species]
        share = out.liquid[j] / (out.liquid[j] + solid) if out.liquid[j] + solid else 0
        print(f"  D03R {species}: crystals fed {fed * 1e3:.4f} g/s, leaving "
              f"{solid * 1e3:.4f} g/s, {share:.1%} of the {species} in the "
              "product is dissolved.")

    design_f = CAKE_LIQUID_DESIGN["F"]
    actual_f = table["CR02H out"].liquid[j_f]
    actual_t = table["CR02H out"].temp
    if (abs(actual_f / design_f - 1.0) > CAKE_DESIGN_TOL
            or abs(actual_t - CAKE_TEMPERATURE_DESIGN) > 1.0):
        print(f"  WARNING: CR02H's liquid F {actual_f:.3e} kg/s at "
              f"{actual_t:.2f} K has moved off CAKE_LIQUID_DESIGN "
              f"({design_f:.3e} kg/s, {CAKE_TEMPERATURE_DESIGN} K), which "
              "sets D02H's and D01H's S0 and temperature. Update it.")

    print("\nChange over the last 10 % of the run (settled below ~1 %):")
    for name, unit in cv.members.items():
        print(f"  {name:<8}{settling(unit, UNIT_CRYSTALS[name]):>9.2%}")


def plot_profiles(cv):
    """What each vessel discharges vs time, one panel per vessel."""
    import matplotlib.pyplot as plt

    show = {"CR01H": ("F",), "CR02H": ("F",), "D02H": ("F",), "D01H": ("F",),
            "R02R": ("F", "D", "E", "J"), "D01R": ("D", "E", "J"),
            "D02R": ("D", "E", "J"), "D03R": ("D", "E", "J")}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True)

    for ax, (name, species) in zip(axes.flat, show.items()):
        history = outlet_history(cv.members[name], UNIT_CRYSTALS[name])
        hours = history.time / 3600.0
        for s in species:
            ax.plot(hours, history.dissolved[s] * 1e3,
                    color=SPECIES_COLORS[s], lw=2)
            if s in history.crystals:
                ax.plot(hours, history.crystals[s] * 1e3, ls="--",
                        color=SPECIES_COLORS[s], lw=2)
        ax.set_title(PANEL_TITLES.get(name, name), loc="left",
                     color=TEXT_PRIMARY)
        ax.set_ylabel("Discharge [g/s]")
        ax.grid(True, color=GRID_COLOR, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Time [h]")

    handles = [plt.Line2D([], [], color=c, lw=2, label=s)
               for s, c in SPECIES_COLORS.items()]
    handles += [plt.Line2D([], [], color=TEXT_MUTED, lw=2, label="dissolved"),
                plt.Line2D([], [], color=TEXT_MUTED, lw=2, ls="--",
                           label="crystals")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_streams(cv):
    """Every stream at the final time, as a table."""
    import matplotlib.pyplot as plt

    table = stream_table(cv)
    names = species_names()
    # Only species that matter somewhere get a column.
    shown = [n for j, n in enumerate(names)
             if any(e.liquid[j] > 1e-7 or e.solids.get(n, 0.0) > 1e-7
                    for e in table.values())]

    def cell(value):
        return "0" if value < 5e-8 else f"{value * 1e3:.4g}"

    columns = ["Stream", "T [K]", "Total"] + [f"{n}" for n in shown] + ["Crystals"]
    rows = []
    for label, entry in table.items():
        rows.append([label, f"{entry.temp:.1f}", cell(total(entry))]
                    + [cell(entry.liquid[names.index(n)]) for n in shown]
                    + [cell(sum(entry.solids.values()))])

    fig, ax = plt.subplots(figsize=(14, 0.32 * len(rows) + 1.2))
    ax.axis("off")
    grid = ax.table(cellText=rows, colLabels=columns, cellLoc="right",
                    loc="center")
    grid.auto_set_font_size(False)
    grid.set_fontsize(9)
    grid.scale(1, 1.4)
    for (row, col), box in grid.get_celld().items():
        box.set_edgecolor(GRID_COLOR)
        if col == 0:
            box.set_text_props(ha="left")
        if row == 0:
            box.set_text_props(weight="bold")
            box.set_facecolor("#f0efec")
    ax.set_title("Streams at the final time [g/s]. Species columns are "
                 "dissolved; crystals are separate.", pad=10, color=TEXT_PRIMARY)
    fig.tight_layout()
    return fig


def save_figures(cv, out_dir: Path = RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {"Flowsheet_profiles": plot_profiles(cv),
               "Flowsheet_streams": plot_streams(cv)}
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return figures


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

    start = clock.time()
    flowsheet = run_flowsheet(args.runtime, args.points)
    print_report(flowsheet, clock.time() - start)
    save_figures(flowsheet)
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
