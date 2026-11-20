"""Branching and merging flowsheets for PharmaPy 0.3.3, plus a flowsheet chart.

PharmaPy 0.3.3's ``SimulationExec`` only runs straight lines. Three things
stop a flowsheet that splits into parallel lines and joins again:

1. ``convert_str_flowsheet`` only reads ``A --> B --> C``, and the dict form
   needs every key to be a single unit.
2. ``SolveFlowsheet`` connects each unit to the NEXT unit in run order
   (``execution_names[ind + 1]``), not to its successors in the graph. For
   ``R01 -> (R02a, R02b) -> R03`` that wires R01 -> R02a, R02a -> R02b and
   R02b -> R03.
3. ``Connection.PassPhases`` sets ``destination.Inlet``, which replaces every
   inlet the destination had. A unit fed by two lines keeps only the last one,
   and nothing splits the flow, so each branch gets the whole upstream flow.

``ConcurrentVessel`` itself handles splits and merges already: several
``StreamLink``s into one member each add an inlet, and ``fraction`` splits an
outlet. What it lacked was a way to build those links from the same flowsheet
definition.

This module works with PharmaPy as installed, without changing it:

``FlowsheetGraph``
    Reads the flowsheet. Keys and values can be a unit name or a tuple of
    names; a tuple key is a merge, a tuple value a split::

        {'R01': ('R02a', 'R02b'), ('R02a', 'R02b'): 'R03'}
        [{'R01': ('R02a', 'R02b')}, {('R02a', 'R02b'): 'R03'}]   # same
        'R01 --> (R02a, R02b) --> R03'                            # same
        {'R01': {'R02a': 0.3, 'R02b': 0.7}, ('R02a', 'R02b'): 'R03'}

    A split with no fractions given divides the outlet equally.

``ParallelSimExec``
    A ``SimulationExec`` that takes any of those forms, connects every edge of
    the graph, gives a merging unit one inlet per upstream line and scales
    each branch's flow by its fraction. A plain chain runs exactly as
    ``SimulationExec`` runs it.

``concurrent_from_flowsheet``
    Builds a ``ConcurrentVessel`` with one ``StreamLink`` per edge of the same
    definition.

``plot_flowsheet``
    Draws the units of a ``ParallelSimExec`` (or of a bare definition): one
    box per unit with its class and operating mode, an arrow per stream
    labelled with its split and final flow, and every ConcurrentVessel drawn
    as a dashed frame around its members and their links.
"""

import copy
import re
import warnings
from collections import OrderedDict

import numpy as np

from PharmaPy.Connections import Connection
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Results import SimulationResult
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Commons import check_steady_state
from PharmaPy.CheckModule import check_modeling_objects

# Inlet states that are flows, and so are divided by a split. Everything else
# a Connection hands over (concentrations, temperature, a population as a
# number density per m3) is intensive and arrives unchanged in every branch.
FLOW_STATES = ('vol_flow', 'mass_flow', 'mole_flow')

FRACTION_TOL = 1e-9


# =====================================================================
# Reading a flowsheet definition
# =====================================================================

def _as_names(item, where):
    """A unit name or a tuple/list/set of names, as a list of names."""
    if isinstance(item, str):
        names = [item.strip()]
    elif isinstance(item, (tuple, list, set, frozenset)):
        names = [str(name).strip() for name in item]
    else:
        raise TypeError(
            f"{where}: expected a unit name or a tuple of names, got "
            f"{item!r}.")

    for name in names:
        if not name.isidentifier():
            raise ValueError(
                f"{where}: {name!r} is not a valid unit name. Units become "
                "attributes of the flowsheet (sim.R01), so name them like "
                "Python identifiers.")

    if len(set(names)) != len(names):
        raise ValueError(f"{where}: {item!r} names a unit twice.")

    return names


def _parse_string(text):
    """'A --> (B, C) --> D; E --> D' into a list of (sources, sinks) pairs."""
    pairs = []

    for line in re.split(r'[;\n]', text):
        line = line.strip()
        if not line:
            continue

        groups = []
        for part in line.split('-->'):
            part = part.strip()
            if part.startswith('(') and part.endswith(')'):
                part = part[1:-1]
            elif '(' in part or ')' in part:
                raise ValueError(f"Unbalanced parentheses in {part!r}.")

            names = [name.strip() for name in part.split(',') if name.strip()]
            if not names:
                raise ValueError(f"Empty step in flowsheet line {line!r}.")

            groups.append(tuple(names))

        if len(groups) == 1:
            pairs.append((groups[0], ()))
        for upstream, downstream in zip(groups[:-1], groups[1:]):
            pairs.append((upstream, downstream))

    return pairs


class FlowsheetGraph:
    """
    A flowsheet as units and directed streams, with a split fraction on each.

    Parameters
    ----------
    flowsheet : dict, list of dict, or str
        The definition; see the module docstring. In a dict, a key is the
        unit(s) a stream leaves and the value the unit(s) it reaches: a
        name, a tuple of names, a ``{name: fraction}`` dict, or an empty
        tuple/list for a unit that feeds nothing. A tuple key with a tuple
        value connects every key unit to every value unit.
    splits : dict, optional
        ``{source: {sink: fraction}}`` or ``{(source, sink): fraction}``.
        Overrides fractions given in the definition.

    A source's fractions must add up to at most 1. Whatever is left over is
    not sent to any unit in the flowsheet (a purge or product draw). When
    only some of a source's streams have a fraction, the rest of its outlet
    is shared equally among the others.
    """

    def __init__(self, flowsheet, splits=None):
        self.nodes = []
        self.edges = []
        self._given = {}

        if isinstance(flowsheet, FlowsheetGraph):
            self.nodes = list(flowsheet.nodes)
            self.edges = list(flowsheet.edges)
            self._given = dict(flowsheet._given)
        elif isinstance(flowsheet, str):
            for upstream, downstream in _parse_string(flowsheet):
                self._connect(upstream, downstream, where=repr(flowsheet))
        elif isinstance(flowsheet, dict):
            self._read_dict(flowsheet)
        elif isinstance(flowsheet, (list, tuple)):
            for num, piece in enumerate(flowsheet):
                if not isinstance(piece, dict):
                    raise TypeError(
                        f"Flowsheet list item {num} is a "
                        f"{type(piece).__name__}; each item must be a dict.")
                self._read_dict(piece)
        else:
            raise TypeError(
                "flowsheet must be a dict, a list of dicts or a string, got "
                f"{type(flowsheet).__name__}.")

        if splits:
            for key, value in splits.items():
                if isinstance(key, tuple) and len(key) == 2 \
                        and not isinstance(value, dict):
                    self._set_fraction(key[0], key[1], value)
                else:
                    for sink, fraction in dict(value).items():
                        self._set_fraction(key, sink, fraction)

        if not self.nodes:
            raise ValueError("The flowsheet has no units.")

        self.fraction = self._resolve_fractions()
        self.order = self._topological_order()

    # ---------------------------------------------------------- building
    def _add_node(self, name):
        if name not in self.nodes:
            self.nodes.append(name)

    def _read_dict(self, mapping):
        for key, value in mapping.items():
            where = f"flowsheet entry {key!r}: {value!r}"

            if isinstance(value, dict):
                sources = _as_names(key, where)
                sinks = _as_names(tuple(value), where)
                self._connect(sources, sinks, where)
                for source in sources:
                    for sink, fraction in value.items():
                        if fraction is not None:
                            self._set_fraction(source, sink, fraction)
            else:
                if value is None:
                    value = ()
                self._connect(_as_names(key, where),
                              _as_names(value, where), where)

    def _connect(self, sources, sinks, where):
        sources = _as_names(sources, where)
        sinks = _as_names(sinks, where) if sinks else []

        for name in list(sources) + list(sinks):
            self._add_node(name)

        for source in sources:
            for sink in sinks:
                if source == sink:
                    raise ValueError(f"{where}: {source} cannot feed itself.")
                if (source, sink) not in self.edges:
                    self.edges.append((source, sink))

    def _set_fraction(self, source, sink, fraction):
        if (source, sink) not in self.edges:
            raise KeyError(
                f"Split given for {source} -> {sink}, which is not a stream "
                "of the flowsheet.")

        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"Split {source} -> {sink} must be in (0, 1], got {fraction}.")

        self._given[(source, sink)] = fraction

    def _resolve_fractions(self):
        fraction = {}

        for source in self.nodes:
            out = self.successors(source)
            if not out:
                continue

            given = {sink: self._given[(source, sink)] for sink in out
                     if (source, sink) in self._given}
            free = [sink for sink in out if sink not in given]
            taken = sum(given.values())

            if taken > 1.0 + FRACTION_TOL:
                raise ValueError(
                    f"Splits from {source} add up to {taken:.6g}; at most all "
                    "of its outlet can be sent on.")

            if free:
                share = (1.0 - taken) / len(free)
                if share <= FRACTION_TOL:
                    raise ValueError(
                        f"Splits from {source} already take its whole outlet, "
                        f"leaving nothing for {free}.")
                given.update({sink: share for sink in free})

            for sink in out:
                fraction[(source, sink)] = given[sink]

        return fraction

    def _topological_order(self):
        """Upstream first; ties in the order units were first named.

        SimulationExec's own topological_bfs pops from a set, so two
        parallel units can swap places from one run to the next. This is
        deterministic.
        """
        in_degree = {node: len(self.predecessors(node)) for node in self.nodes}
        ready = [node for node in self.nodes if in_degree[node] == 0]
        order = []

        while ready:
            node = ready.pop(0)
            order.append(node)
            for sink in self.successors(node):
                in_degree[sink] -= 1
                if in_degree[sink] == 0:
                    ready.append(sink)
            ready.sort(key=self.nodes.index)

        if len(order) < len(self.nodes):
            stuck = [node for node in self.nodes if node not in order]
            raise PharmaPyNonImplementedError(
                "Provided flowsheet contains recycle stream(s) through "
                f"{stuck}. Put the recycle loop inside a ConcurrentVessel and "
                "use that as one unit of the flowsheet.")

        return order

    # ---------------------------------------------------------- queries
    def successors(self, node):
        return [sink for source, sink in self.edges if source == node]

    def predecessors(self, node):
        return [source for source, sink in self.edges if sink == node]

    def product_fraction(self, node):
        """Share of a unit's outlet that no stream of the flowsheet takes."""
        return 1.0 - sum(self.fraction[(node, sink)]
                         for sink in self.successors(node))

    def as_dict(self):
        """{unit: [successors]}, the form SimulationExec stores as .graph."""
        return OrderedDict((node, self.successors(node)) for node in self.nodes)

    def layers(self):
        """Units by longest distance from a feed unit (for drawing)."""
        depth = {}
        for node in self.order:
            preds = self.predecessors(node)
            depth[node] = 1 + max(depth[p] for p in preds) if preds else 0

        out = [[] for _ in range(max(depth.values()) + 1)]
        for node in self.order:
            out[depth[node]].append(node)
        return out

    def __repr__(self):
        streams = ', '.join(
            f"{s}->{t}" + (f" ({self.fraction[(s, t)]:.3g})"
                           if self.fraction[(s, t)] < 1 - FRACTION_TOL else '')
            for s, t in self.edges)
        return f"FlowsheetGraph({streams or ', '.join(self.nodes)})"


# =====================================================================
# Sequential solution: SimulationExec that follows the graph
# =====================================================================

def _is_concurrent(uo):
    return type(uo).__name__ == 'ConcurrentVessel'


def _inlet_holder(uo):
    """The vessel whose inlet_connections a flowsheet feed lands in."""
    if _is_concurrent(uo):
        return uo.members[uo.entry]
    return uo if hasattr(uo, 'inlet_connections') else None


def _scaled_flows(y_inlet, fraction):
    return {name: (np.asarray(values) * fraction if name in FLOW_STATES
                   else values)
            for name, values in y_inlet.items()}


class ParallelSimExec(SimulationExec):
    """
    SimulationExec for flowsheets with parallel lines.

    Parameters
    ----------
    pure_path : str
        Compound database, as for SimulationExec.
    flowsheet : dict, list of dict, str or FlowsheetGraph
        See FlowsheetGraph.
    splits : dict, optional
        Split fractions; see FlowsheetGraph.

    Units are assigned as attributes and solved with ``SolveFlowsheet``,
    exactly as with SimulationExec. Each unit is solved over its whole run
    after every unit feeding it, and receives each upstream outlet
    trajectory on an inlet of its own.

    Notes
    -----
    - A unit fed by more than one stream must be built on MultiPhaseVessel
      (the refactored units) or be a ConcurrentVessel: only those hold
      several inlets. A ConcurrentVessel receives them all on its entry
      member.
    - A split needs a continuous source: a batch unit discharges one charge,
      not a flow.
    - By default a unit's upstream streams replace the inlet it was built
      with, as SimulationExec does. List it in ``keep_feeds`` when solving to
      keep its own feeds as well (a fresh reagent stream, say).
    """

    def __init__(self, pure_path, flowsheet, splits=None):
        self.flowsheet = FlowsheetGraph(flowsheet, splits=splits)

        # SimulationExec.__init__ only accepts a str or a dict of
        # single-unit keys, so hand it a chain and then replace what it
        # derived with the graph's.
        super().__init__(pure_path, flowsheet=' --> '.join(
            self.flowsheet.nodes))

        self.graph = self.flowsheet.as_dict()
        self.execution_names = list(self.flowsheet.order)
        self.in_degree = {node: 0 for node in self.flowsheet.nodes}
        self.splits = dict(self.flowsheet.fraction)
        self.connections = {}

    # ---------------------------------------------------------- wiring
    def _transfer(self, source_name, sink_name):
        """Run one Connection with the source's flow scaled by its split."""
        source = getattr(self, source_name)
        sink = getattr(self, sink_name)
        fraction = self.splits[(source_name, sink_name)]

        connection = Connection(source_uo=source, destination_uo=sink)
        connection.FeedConnection()
        connection.ConvertUnits()

        matter = connection.Matter
        unscaled = getattr(matter, 'y_inlet', None)

        if fraction < 1.0 - FRACTION_TOL:
            if not source.is_continuous:
                raise PharmaPyNonImplementedError(
                    f"{source_name} -> {sink_name} takes {fraction:.3g} of "
                    f"{source_name}, which is a "
                    f"{getattr(source, 'oper_mode', 'batch')} unit. Only a "
                    "continuous outlet can be split.")
            if unscaled is None:
                raise PharmaPyNonImplementedError(
                    f"{source_name} -> {sink_name}: PharmaPy did not build an "
                    "inlet trajectory for this pair, so the split cannot be "
                    "applied.")
            matter.y_inlet = _scaled_flows(unscaled, fraction)

        try:
            connection.PassPhases()
        finally:
            # Matter is the source's own Outlet. Leave it as Connection
            # would, so the next branch scales from the full flow.
            if unscaled is not None:
                matter.y_inlet = unscaled

        connection.fraction = fraction
        self.connections[f"{source_name}->{sink_name}"] = connection

    def _wire_inlets(self, name, keep):
        sources = self.flowsheet.predecessors(name)
        unit = getattr(self, name)

        for source in sources:
            if getattr(getattr(self, source), 'outputs', None) is None:
                raise RuntimeError(
                    f"{name} is fed by {source}, which has not been solved. "
                    "Include it in pick_units or solve it first.")

        simple = (len(sources) == 1 and not keep
                  and self.splits[(sources[0], name)] >= 1 - FRACTION_TOL)

        if simple:
            # Exactly what SimulationExec does for a chain.
            self._transfer(sources[0], name)
            return

        holder = _inlet_holder(unit)

        if holder is None or (len(sources) > 1
                              and Connection._inlet_mode(unit) == 'Batch'):
            raise PharmaPyNonImplementedError(
                f"{name} ({type(unit).__name__}) is fed by {sources}, but it "
                "can only take one inlet. Use a refactored "
                "(MultiPhaseVessel-based) continuous or semibatch unit, or a "
                "ConcurrentVessel, for a unit where lines merge.")

        before = list(holder.inlet_connections)
        received = []

        for source in sources:
            holder.inlet_connections = list(before)
            self._transfer(source, name)
            received.extend(
                conn for conn in holder.inlet_connections
                if not any(conn is old for old in before))

        connections = list(before)

        if keep:
            connections.extend(received)
        elif _is_concurrent(unit):
            # Linked inlets keep their positions (StreamLink.inlet_index);
            # the feed goes where ConcurrentVessel.Inlet would put it.
            holder.inlet_connections = list(before)
            index = unit._entry_inlet_index()
            if index < len(connections):
                connections[index] = received[0]
                connections.extend(received[1:])
            else:
                connections.extend(received)
        else:
            connections = received

        holder.inlet_connections = connections

        if _is_concurrent(unit):
            unit._eval_key = None

    # ---------------------------------------------------------- solving
    def SolveFlowsheet(self, kwargs_run=None, pick_units=None, verbose=True,
                       steady_state_di=None, tolerances_ss=None, ss_time=0,
                       keep_feeds=()):
        """
        Solve every unit in graph order.

        Takes SimulationExec.SolveFlowsheet's arguments, plus
        ``keep_feeds``: units that keep the inlets they were built with in
        addition to the streams from upstream.
        """
        kwargs_run = kwargs_run or {}
        steady_state_di = steady_state_di or {}
        pick_units = self.execution_names if pick_units is None else pick_units
        keep_feeds = set(keep_feeds)

        unknown = (set(pick_units) | keep_feeds | set(kwargs_run)) \
            - set(self.execution_names)
        if unknown:
            raise KeyError(f"{sorted(unknown)} are not units of the flowsheet "
                           f"{self.execution_names}.")

        time_processing = {}
        self.connections = {}

        for name in self.execution_names:
            if name not in pick_units:
                continue

            instance = getattr(self, name)
            self.uos_instances[name] = instance

            if self.flowsheet.predecessors(name):
                self._wire_inlets(name, keep=name in keep_feeds)

            check_modeling_objects(instance, name)

            if verbose:
                print('\n{0}\nRunning {1}\n{0}\n'.format('-' * 30, name))

            kwargs_uo = dict(kwargs_run.get(name, {}))

            if name in steady_state_di:
                kw_ss = steady_state_di[name]
                tau = instance._get_tau() if hasattr(instance, '_get_tau') \
                    else 0
                ss_time += tau

                if instance.__class__.__name__ != 'Mixer':
                    kw_ss.setdefault('time_stop', ss_time)
                    kw_ss.setdefault('tau', tau)
                    instance.state_event_list.append(
                        {'callable': check_steady_state, 'num_conditions': 1,
                         'event_name': 'steady_state', 'kwargs': kw_ss})
                    kwargs_uo['any_event'] = False

            instance.solve_unit(**kwargs_uo)

            if instance.__module__ != 'PharmaPy.Containers':
                instance.flatten_states()

            if verbose:
                print('\nDone!\n')

            time_prof = getattr(instance.result, 'time', None)
            if time_prof is not None:
                time_prof = np.ravel(time_prof)
                time_processing[name] = time_prof[-1] - time_prof[0]

        self.time_processing = time_processing
        self.result = SimulationResult(self)

    def plot_flowsheet(self, **kwargs):
        """Draw the flowsheet; see plot_flowsheet."""
        return plot_flowsheet(self, **kwargs)


# =====================================================================
# Coupled solution: the same definition as one ConcurrentVessel
# =====================================================================

def concurrent_from_flowsheet(flowsheet, members, splits=None,
                              placeholder_inlets=None, **cv_kwargs):
    """
    Build a ConcurrentVessel from a flowsheet definition.

    Parameters
    ----------
    flowsheet : dict, list of dict, str or FlowsheetGraph
        See FlowsheetGraph. Every unit must be in ``members``.
    members : dict
        {name: MultiPhaseVessel}, with each member's own feeds configured.
    splits : dict, optional
        Split fractions; see FlowsheetGraph.
    placeholder_inlets : dict, optional
        {member: inlet index}. The FIRST stream into that member replaces
        the inlet at this index instead of being added beside it -- for a
        member built with a dummy feed only because a vessel needs one.
    **cv_kwargs
        Passed to ConcurrentVessel (integrator, tear_mode, entry, exit ...).
        ``order`` defaults to the graph's upstream-first order.

    Returns
    -------
    ConcurrentVessel
    """
    from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink

    graph = FlowsheetGraph(flowsheet, splits=splits)

    missing = [name for name in graph.nodes if name not in members]
    extra = [name for name in members if name not in graph.nodes]
    if missing or extra:
        raise KeyError(f"Flowsheet units and members differ: missing "
                       f"{missing}, not in the flowsheet {extra}.")

    placeholders = dict(placeholder_inlets or {})
    links = []
    for source, sink in graph.edges:
        links.append(StreamLink(source, sink,
                                fraction=graph.fraction[(source, sink)],
                                inlet=placeholders.pop(sink, None)))

    cv_kwargs.setdefault('order', graph.order)
    ordered = OrderedDict((name, members[name]) for name in graph.order)

    vessel = ConcurrentVessel(ordered, links=links, **cv_kwargs)
    vessel.flowsheet = graph
    return vessel


# =====================================================================
# Drawing
# =====================================================================

MODE_COLORS = {
    # Fill, edge. Muted so the arrows and labels carry the eye.
    'Continuous': ('#dbe9f6', '#2f6ea6'),
    'Semibatch': ('#fbe8cf', '#b0701e'),
    'Batch': ('#e4ecd9', '#4d7a2a'),
    None: ('#eceef0', '#6b7280'),
}


def _oper_mode(uo):
    if uo is None:
        return None
    try:
        return uo.oper_mode
    except Exception:
        return None


def _final_flow(uo, fraction=1.0):
    """Final outlet volumetric flow [m3/s] times a split, or None."""
    try:
        outputs = uo.outputs
    except Exception:
        return None
    if not outputs or 'vol_flow' not in outputs:
        return None
    flow = float(np.ravel(outputs['vol_flow'])[-1]) * fraction
    # A placeholder feed (a vessel must have one) is not worth a label.
    return flow if abs(flow) > 1e-12 else None


def _format_flow(m3_s):
    l_h = m3_s * 3.6e6
    return f"{l_h:.3g} L/h"


def _expanded_graph(sim, graph):
    """Nodes and edges with every ConcurrentVessel opened up into members.

    Returns (nodes, edges, frames). A node is (key, label, unit); an edge is
    (from_key, to_key, label); frames map a ConcurrentVessel's name to the
    keys of its members.
    """
    nodes, frames = OrderedDict(), OrderedDict()
    entry_key, exit_key = {}, {}
    edges = []

    def unit_of(name):
        return getattr(sim, name, None) if sim is not None else None

    for name in graph.order:
        uo = unit_of(name)

        if uo is not None and _is_concurrent(uo):
            keys = []
            order = uo.order or uo._evaluation_order()
            for member in order:
                key = f"{name}.{member}"
                nodes[key] = (member, uo.members[member])
                keys.append(key)
            frames[name] = keys
            entry_key[name] = f"{name}.{uo.entry}"
            exit_key[name] = f"{name}.{uo.exit}" if uo.exit else keys[-1]

            for link in uo.links:
                label = []
                if link.fraction < 1 - FRACTION_TOL:
                    label.append(f"{link.fraction:.2g}")
                flow = _final_flow(uo.members[link.source], link.fraction)
                if flow is not None:
                    label.append(_format_flow(flow))
                # A link back to a member evaluated earlier is a recycle
                # (ConcurrentVessel marks it is_tear only once compiled).
                tear = link.is_tear or \
                    order.index(link.sink) <= order.index(link.source)
                edges.append((f"{name}.{link.source}", f"{name}.{link.sink}",
                              '\n'.join(label), tear))
        else:
            nodes[name] = (name, uo)
            entry_key[name] = exit_key[name] = name

    for source, sink in graph.edges:
        fraction = graph.fraction[(source, sink)]
        label = []
        if fraction < 1 - FRACTION_TOL:
            label.append(f"{fraction:.2g}")
        flow = _final_flow(unit_of(source), fraction)
        if flow is not None:
            label.append(_format_flow(flow))
        edges.append((exit_key[source], entry_key[sink], '\n'.join(label),
                      False))

    return nodes, edges, frames


def _layout(nodes, edges):
    """Columns by longest path; rows ordered to keep parallel lines apart."""
    preds = {key: [] for key in nodes}
    succs = {key: [] for key in nodes}
    for source, sink, _, tear in edges:
        if not tear:
            preds[sink].append(source)
            succs[source].append(sink)

    depth = {}
    remaining = list(nodes)
    while remaining:
        progressed = False
        for key in list(remaining):
            if all(p in depth for p in preds[key]):
                depth[key] = 1 + max((depth[p] for p in preds[key]),
                                     default=-1)
                remaining.remove(key)
                progressed = True
        if not progressed:  # a loop not marked as a tear; break it
            key = remaining.pop(0)
            depth[key] = 1 + max((depth[p] for p in preds[key] if p in depth),
                                 default=-1)

    columns = {}
    for key in nodes:
        columns.setdefault(depth[key], []).append(key)

    rows = {}
    for col in sorted(columns):
        keys = columns[col]
        # Barycentre of the units feeding each one, in their own column.
        def barycentre(key):
            placed = [rows[p] for p in preds[key] if p in rows]
            return np.mean(placed) if placed else 0.0

        keys.sort(key=lambda k: (barycentre(k), list(nodes).index(k)))
        n = len(keys)
        for i, key in enumerate(keys):
            rows[key] = (n - 1) / 2.0 - i

    return {key: (depth[key], rows[key]) for key in nodes}


def _nested_layout(graph, nodes, edges, frames, col_gap, row_gap):
    """Positions for every drawn unit, ConcurrentVessels kept together.

    The flowsheet's own units are put in columns by longest path from a
    feed. A ConcurrentVessel is one block in that grid, as many columns wide
    and rows tall as its own members need, and its members are laid out
    inside the block from its links alone.
    """
    blocks = {}
    for name in graph.order:
        keys = frames.get(name)
        if keys is None:
            blocks[name] = {'cols': 1, 'rows': 1.0, 'inner': {name: (0, 0.0)}}
            continue

        inner_edges = [edge for edge in edges
                       if edge[0] in keys and edge[1] in keys]
        inner = _layout(OrderedDict((k, nodes[k]) for k in keys), inner_edges)
        cols = max(c for c, _ in inner.values()) + 1
        rows = max(sum(1 for c, _ in inner.values() if c == col)
                   for col in range(cols))
        has_tear = any(edge[3] for edge in inner_edges)
        blocks[name] = {'cols': cols,
                        'rows': rows + 0.45 + (0.55 if has_tear else 0.0),
                        'inner': inner, 'tear': has_tear}

    layers = graph.layers()
    widths = [max(blocks[name]['cols'] for name in layer) for layer in layers]
    starts = np.concatenate(([0.0], np.cumsum(widths)[:-1])) * col_gap

    centre_y, xy = {}, {}
    gap = 0.25

    for depth, layer in enumerate(layers):
        def barycentre(name):
            preds = graph.predecessors(name)
            return np.mean([centre_y[p] for p in preds]) if preds else 0.0

        layer = sorted(layer, key=lambda n: (-barycentre(n),
                                             graph.order.index(n)))
        total = sum(blocks[n]['rows'] for n in layer) + gap * (len(layer) - 1)
        top = total / 2.0

        for name in layer:
            block = blocks[name]
            mid = top - block['rows'] / 2.0
            top -= block['rows'] + gap
            centre_y[name] = mid * row_gap

            # Members sit below the frame's title, above any recycle arc.
            shift = 0.0
            if name in frames:
                shift = -0.2 + (0.3 if block['tear'] else 0.0)

            x0 = starts[depth] + (widths[depth] - block['cols']) * col_gap / 2
            for key, (col, row) in block['inner'].items():
                xy[key] = (x0 + col * col_gap, (mid + shift + row) * row_gap)

    return xy


def plot_flowsheet(source, ax=None, splits=None, title=None, flows=True,
                   name='CV', col_gap=3.2, row_gap=1.6, box_w=1.9, box_h=0.95):
    """
    Draw a flowsheet's unit operations and streams.

    Parameters
    ----------
    source : ParallelSimExec, SimulationExec, ConcurrentVessel, or a
             flowsheet definition (dict, list of dict, str, FlowsheetGraph)
        With a flowsheet object, each box shows the unit's class and
        operating mode, ConcurrentVessels are drawn open with their members
        inside a dashed frame, and once solved every stream shows its final
        flow. A bare definition draws names and splits only.
    ax : matplotlib Axes, optional
    splits : dict, optional
        Split fractions, for a bare definition.
    title : str, optional
    flows : bool
        Label streams with their final volumetric flow when known.
    name : str
        Label for a ConcurrentVessel drawn on its own.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    sim = None
    if isinstance(source, ParallelSimExec):
        sim, graph = source, source.flowsheet
    elif isinstance(source, SimulationExec):
        sim, graph = source, FlowsheetGraph(dict(source.graph))
    elif _is_concurrent(source):
        class _Holder:
            pass
        sim = _Holder()
        setattr(sim, name, source)
        graph = FlowsheetGraph({name: ()})
    else:
        graph = FlowsheetGraph(source, splits=splits)

    nodes, edges, frames = _expanded_graph(sim, graph)
    if not flows:
        edges = [(s, t, '\n'.join(part for part in label.split('\n')
                                  if 'L/h' not in part), tear)
                 for s, t, label, tear in edges]

    xy = _nested_layout(graph, nodes, edges, frames, col_gap, row_gap)

    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    n_cols = (max(xs) - min(xs)) / col_gap + 1
    n_rows = (max(ys) - min(ys)) / row_gap + 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5.0, 2.6 * n_cols),
                                        max(2.6, 1.3 * n_rows + 1.4)))
    else:
        fig = ax.figure

    modes_seen = set()

    # ConcurrentVessel frames first, underneath everything else.
    for cv_name, keys in frames.items():
        xs = [xy[k][0] for k in keys]
        ys = [xy[k][1] for k in keys]
        pad_x, pad_y = box_w / 2 + 0.35, box_h / 2 + 0.3
        x0, x1 = min(xs) - pad_x, max(xs) + pad_x
        y0, y1 = min(ys) - pad_y, max(ys) + pad_y + 0.35
        if any(tear and a in keys for a, _, _, tear in edges):
            y0 -= 0.75   # room for the recycle arc and its label
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle='round,pad=0.02,rounding_size=0.25',
            fill=False, linestyle=(0, (5, 3)), linewidth=1.3,
            edgecolor='#5b6470', zorder=1))
        ax.text(x0 + 0.18, y1 - 0.12, f"{cv_name}  (ConcurrentVessel)",
                ha='left', va='top', fontsize=8.5, color='#3d444d',
                fontweight='bold', zorder=5)

    for key, (label, uo) in nodes.items():
        x, y = xy[key]
        mode = _oper_mode(uo)
        modes_seen.add(mode if mode in MODE_COLORS else None)
        face, edge = MODE_COLORS.get(mode, MODE_COLORS[None])

        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.02,rounding_size=0.12',
            facecolor=face, edgecolor=edge, linewidth=1.4, zorder=3))

        ax.text(x, y + (0.13 if uo is not None else 0), label, ha='center',
                va='center', fontsize=10.5, fontweight='bold',
                color='#1f2328', zorder=4)
        if uo is not None:
            ax.text(x, y - 0.2, type(uo).__name__, ha='center', va='center',
                    fontsize=7.5, color='#3d444d', zorder=4)

    for source_key, sink_key, label, tear in edges:
        (x0, y0), (x1, y1) = xy[source_key], xy[sink_key]
        start = (x0 + box_w / 2, y0)
        end = (x1 - box_w / 2, y1)
        rad = 0.0
        if tear or x1 <= x0:
            start, end = (x0, y0 - box_h / 2), (x1, y1 - box_h / 2)
            # Bow below the boxes whichever way the arrow runs.
            rad = -0.3 if x1 < x0 else 0.3

        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle='-|>', mutation_scale=13,
            connectionstyle=f'arc3,rad={rad}', linewidth=1.3,
            color='#c2410c' if tear else '#3d444d', zorder=2,
            shrinkA=1, shrinkB=1))

        if label:
            if rad:
                lx = (start[0] + end[0]) / 2
                ly = (min(start[1], end[1])
                      - 0.5 * abs(rad) * abs(end[0] - start[0]) - 0.12)
            else:
                lx = (start[0] + end[0]) / 2
                ly = (start[1] + end[1]) / 2
            ax.text(lx, ly, label, ha='center', va='center', fontsize=7.5,
                    color='#1f2328', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                              edgecolor='none', alpha=0.9))

    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    ax.set_xlim(min(xs) - box_w, max(xs) + box_w)
    low_pad = 1.3 if any(tear or xy[t][0] <= xy[s][0]
                         for s, t, _, tear in edges) else 0.35
    ax.set_ylim(min(ys) - box_h / 2 - 0.35 - low_pad,
                max(ys) + box_h / 2 + 0.75)
    ax.set_aspect('equal')
    ax.axis('off')

    if sim is not None:
        handles = [Patch(facecolor=MODE_COLORS[m][0],
                         edgecolor=MODE_COLORS[m][1], label=m or 'other')
                   for m in ('Continuous', 'Semibatch', 'Batch', None)
                   if m in modes_seen]
        if handles:
            ax.legend(handles=handles, loc='upper center',
                      bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                      frameon=False, fontsize=8)

    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig
