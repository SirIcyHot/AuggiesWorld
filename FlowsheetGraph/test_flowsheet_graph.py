"""Tests for branching and merging flowsheets (flowsheet_graph.py).

Run from this folder:  python -m pytest test_flowsheet_graph.py -q

The reactors come from example_parallel_lines.py and run on the scipy
backend for 30 min of process time, so the whole file takes seconds.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.SimExec import SimulationExec

from example_parallel_lines import PATH, feed, reactor
from flowsheet_graph import (FlowsheetGraph, ParallelSimExec,
                             concurrent_from_flowsheet, plot_flowsheet)

RUNTIME = 1800.0
DIAMOND = {'R01': ('R02a', 'R02b'), ('R02a', 'R02b'): 'R03'}


def run_args(sim):
    return {name: {'runtime': RUNTIME} for name in sim.execution_names}


def end(unit, key):
    return np.asarray(getattr(unit.result, key), dtype=float)[-1]


def end_flow(unit):
    return float(np.ravel(unit.result.outlet_vol_flow)[-1])


def rel_gap(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def inlet_flow_at_end(unit, index):
    """Volumetric flow the upstream trajectory delivers on one inlet."""
    stream = unit.inlet_connections[index].stream
    phases = getattr(stream, 'Phases', None) or [stream]
    for phase in [stream] + list(phases):
        y_inlet = getattr(phase, 'y_inlet', None)
        if y_inlet and 'vol_flow' in y_inlet:
            return float(np.ravel(y_inlet['vol_flow'])[-1])
    raise AssertionError('inlet carries no upstream trajectory')


# ================================================================ parsing
class TestDefinition:

    def test_tuple_keys_list_and_string_agree(self):
        forms = [
            DIAMOND,
            [{'R01': ('R02a', 'R02b')}, {('R02a', 'R02b'): 'R03'}],
            'R01 --> (R02a, R02b) --> R03',
            {'R01': ['R02a', 'R02b'], 'R02a': 'R03', 'R02b': ['R03'],
             'R03': []},
        ]
        graphs = [FlowsheetGraph(form) for form in forms]
        for graph in graphs:
            assert set(graph.edges) == {('R01', 'R02a'), ('R01', 'R02b'),
                                        ('R02a', 'R03'), ('R02b', 'R03')}
            assert graph.order == ['R01', 'R02a', 'R02b', 'R03']
            assert graph.fraction[('R01', 'R02a')] == pytest.approx(0.5)
            assert graph.fraction[('R02a', 'R03')] == 1.0

    def test_chain_matches_simulationexec_graph(self):
        graph = FlowsheetGraph('R01 --> CR01 --> F01')
        assert dict(graph.as_dict()) == {'R01': ['CR01'], 'CR01': ['F01'],
                                         'F01': []}

    def test_fractions_in_definition_and_splits(self):
        graph = FlowsheetGraph({'R01': {'R02a': 0.3, 'R02b': 0.7}})
        assert graph.fraction[('R01', 'R02b')] == pytest.approx(0.7)

        graph = FlowsheetGraph(DIAMOND, splits={'R01': {'R02a': 0.2}})
        assert graph.fraction[('R01', 'R02b')] == pytest.approx(0.8)

        graph = FlowsheetGraph(DIAMOND, splits={('R01', 'R02a'): 0.2,
                                                ('R01', 'R02b'): 0.3})
        assert graph.product_fraction('R01') == pytest.approx(0.5)

    def test_three_way_split_and_merge(self):
        graph = FlowsheetGraph('A --> (B, C, D) --> E')
        assert graph.fraction[('A', 'C')] == pytest.approx(1 / 3)
        assert graph.predecessors('E') == ['B', 'C', 'D']
        assert graph.layers() == [['A'], ['B', 'C', 'D'], ['E']]

    def test_many_to_many(self):
        graph = FlowsheetGraph({('A', 'B'): ('C', 'D')})
        assert len(graph.edges) == 4
        assert graph.fraction[('A', 'C')] == pytest.approx(0.5)

    @pytest.mark.parametrize('spec, error', [
        ({'R01': 'R02', 'R02': 'R01'}, PharmaPyNonImplementedError),
        ({'R01': 'R01'}, ValueError),
        ({'R01': {'R02a': 0.8, 'R02b': 0.4}}, ValueError),
        ({'R01': {'R02': 1.5}}, ValueError),
        ({'R 01': 'R02'}, ValueError),
        ({1: 'R02'}, TypeError),
        ('R01 --> (R02a, R02b --> R03', ValueError),
    ])
    def test_bad_definitions(self, spec, error):
        with pytest.raises(error):
            FlowsheetGraph(spec)

    def test_split_on_missing_stream(self):
        with pytest.raises(KeyError):
            FlowsheetGraph(DIAMOND, splits={'R01': {'R03': 0.5}})


# ======================================================= sequential solve
class TestParallelSimExec:

    def test_chain_is_unchanged(self):
        """A plain line gives what SimulationExec gives."""
        results = []
        for cls in (SimulationExec, ParallelSimExec):
            sim = cls(PATH, 'R01 --> R02')
            sim.R01 = reactor(feed())
            sim.R02 = reactor()
            sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': RUNTIME},
                                           'R02': {'runtime': RUNTIME}},
                               verbose=False)
            results.append(sim)

        old, new = results
        assert rel_gap(end(new.R02, 'mole_conc_liquid0'),
                       end(old.R02, 'mole_conc_liquid0')) < 1e-12
        assert end_flow(new.R02) == pytest.approx(end_flow(old.R02),
                                                  rel=1e-12)

    def test_branch_feeds_both_lines_from_the_split(self):
        """R01 feeds R02a and R02b, not R01 -> R02a -> R02b."""
        sim = ParallelSimExec(PATH, {'R01': {'R02a': 0.25, 'R02b': 0.75}})
        sim.R01 = reactor(feed())
        sim.R02a = reactor()
        sim.R02b = reactor()
        sim.SolveFlowsheet(kwargs_run=run_args(sim), verbose=False)

        q01 = end_flow(sim.R01)
        assert inlet_flow_at_end(sim.R02a, 0) == pytest.approx(0.25 * q01)
        assert inlet_flow_at_end(sim.R02b, 0) == pytest.approx(0.75 * q01)
        # The source's own outlet is left unscaled for the next branch.
        assert np.ravel(sim.R01.Outlet.y_inlet['vol_flow'])[-1] \
            == pytest.approx(q01)

    def test_merge_matches_concurrent_vessel(self):
        splits = {'R01': {'R02a': 0.3}}

        sim = ParallelSimExec(PATH, DIAMOND, splits=splits)
        sim.R01 = reactor(feed())
        sim.R02a = reactor()
        sim.R02b = reactor()
        sim.R03 = reactor()
        sim.SolveFlowsheet(kwargs_run=run_args(sim), verbose=False)

        assert len(sim.R03.inlet_connections) == 2
        assert sorted(sim.connections) == ['R01->R02a', 'R01->R02b',
                                           'R02a->R03', 'R02b->R03']

        cv = concurrent_from_flowsheet(
            DIAMOND,
            {'R01': reactor(feed()), 'R02a': reactor(), 'R02b': reactor(),
             'R03': reactor()},
            splits=splits,
            placeholder_inlets={'R02a': 0, 'R02b': 0, 'R03': 0},
            integrator=ScipyBackend())
        cv.solve_unit(runtime=RUNTIME)

        assert [link.fraction for link in cv.links] == \
            pytest.approx([0.3, 0.7, 1.0, 1.0])
        for name in sim.execution_names:
            assert rel_gap(end(getattr(sim, name), 'mole_conc_liquid0'),
                           end(cv[name], 'mole_conc_liquid0')) < 1e-6, name
            assert end_flow(getattr(sim, name)) == \
                pytest.approx(end_flow(cv[name]), rel=1e-6), name

    def test_keep_feeds_adds_upstream_beside_own_feed(self):
        sim = ParallelSimExec(PATH, DIAMOND)
        sim.R01 = reactor(feed())
        sim.R02a = reactor()
        sim.R02b = reactor()
        sim.R03 = reactor(feed(flow=0.002))
        sim.SolveFlowsheet(kwargs_run=run_args(sim), verbose=False,
                           keep_feeds=['R03'])

        assert len(sim.R03.inlet_connections) == 3
        q_in = (end_flow(sim.R02a) + end_flow(sim.R02b)
                + inlet_flow_or_static(sim.R03, 0))
        assert end_flow(sim.R03) > end_flow(sim.R02a) + end_flow(sim.R02b)
        assert q_in > 0

    def test_lines_merge_into_a_concurrent_vessel(self):
        """Two lines feed a ConcurrentVessel's entry member."""
        r10, r11 = reactor(), reactor()
        cv = ConcurrentVessel({'R10': r10, 'R11': r11},
                              links=[StreamLink('R10', 'R11', inlet=0)],
                              integrator=ScipyBackend())

        sim = ParallelSimExec(PATH, {('R01a', 'R01b'): 'CV'})
        sim.R01a = reactor(feed())
        sim.R01b = reactor(feed(flow=0.003))
        sim.CV = cv
        sim.SolveFlowsheet(kwargs_run=run_args(sim), verbose=False)

        # The placeholder is replaced by one line, the other is added.
        assert len(r10.inlet_connections) == 2
        assert inlet_flow_at_end(r10, 0) == pytest.approx(end_flow(sim.R01a))
        assert inlet_flow_at_end(r10, 1) == pytest.approx(end_flow(sim.R01b))
        # R11 is still fed by its link.
        assert len(r11.inlet_connections) == 1
        assert end_flow(r11) == pytest.approx(end_flow(r10), rel=0.05)

    def test_merge_into_old_unit_is_refused(self):
        from PharmaPy.Reactors import CSTR

        sim = ParallelSimExec(PATH, DIAMOND)
        sim.R01 = reactor(feed())
        sim.R02a = reactor()
        sim.R02b = reactor()
        sim.R03 = CSTR()
        with pytest.raises(PharmaPyNonImplementedError, match='one inlet'):
            sim.SolveFlowsheet(kwargs_run=run_args(sim), verbose=False,
                               pick_units=['R01', 'R02a', 'R02b', 'R03'])

    def test_unknown_unit_names(self):
        sim = ParallelSimExec(PATH, DIAMOND)
        with pytest.raises(KeyError):
            sim.SolveFlowsheet(kwargs_run={'R99': {}}, verbose=False)


def inlet_flow_or_static(unit, index):
    stream = unit.inlet_connections[index].stream
    try:
        return inlet_flow_at_end(unit, index)
    except AssertionError:
        return float(np.ravel(stream.vol_flow)[-1])


# ================================================================ drawing
class TestPlot:

    def test_definition_only(self):
        fig = plot_flowsheet('R01 --> (R02a, R02b, R02c) --> R03')
        texts = {t.get_text() for t in fig.axes[0].texts}
        assert {'R01', 'R02a', 'R02b', 'R02c', 'R03'} <= texts
        assert '0.33' in texts

    def test_solved_flowsheet_and_vessel(self, tmp_path):
        sim = ParallelSimExec(PATH, DIAMOND)
        sim.R01 = reactor(feed())
        sim.R02a = reactor()
        sim.R02b = reactor()
        sim.R03 = reactor()
        sim.SolveFlowsheet(kwargs_run=run_args(sim), verbose=False)

        fig = sim.plot_flowsheet(title='diamond')
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert 'ContinuousReactor' in texts
        assert any('L/h' in text for text in texts)
        fig.savefig(tmp_path / 'diamond.png')

        cv = ConcurrentVessel(
            {'R10': reactor(feed()), 'R11': reactor()},
            links=[StreamLink('R10', 'R11', inlet=0),
                   StreamLink('R11', 'R10', fraction=0.4)],
            integrator=ScipyBackend())
        fig = plot_flowsheet(cv, name='LOOP')
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert 'LOOP  (ConcurrentVessel)' in texts
        assert '0.4' in texts
