"""Parallel lines that split and rejoin, solved sequentially and coupled.

    R01 --> (R02a, R02b) --> R03

R01 is fed; its outlet is split between two parallel reactors, R02a and
R02b, whose outlets merge into R03. The same flowsheet definition is solved
two ways:

1. ParallelSimExec: one unit after another, each fed the recorded outlet
   trajectories of the units upstream of it.
2. One ConcurrentVessel built from the definition with
   concurrent_from_flowsheet: all four reactors on one integrator.

There is no recycle, so both solve the same equations and should agree up to
the interpolation of recorded trajectories. Figures go to results/:

    flowsheet_parallel_simexec.png      the sequential flowsheet
    flowsheet_parallel_concurrent.png   the same units as a ConcurrentVessel
    flowsheet_parallel_compare.png      R03 outlet, sequential vs coupled
    flowsheet_cv_as_line.png            a ConcurrentVessel (with a recycle)
                                        as one of two parallel lines

Run from this folder:

    python example_parallel_lines.py [--split 0.5] [--runtime 3600] [--no-show]
"""

import argparse
import os

import numpy as np

from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ProcessControl_Refactored import ContinuousVesselController
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream

from flowsheet_graph import (ParallelSimExec, concurrent_from_flowsheet,
                             plot_flowsheet)

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, os.pardir, 'tests', 'Flowsheet', 'data',
                    'compound_database.json')
RESULTS = os.path.join(HERE, 'results')

FLOWSHEET = {'R01': ('R02a', 'R02b'), ('R02a', 'R02b'): 'R03'}

TEMP = 313.15                                   # [K]
FEED_FLOW = 0.006                               # [kg/s]
FEED_FRAC = np.array([.15, .15, 0, 0, .70])     # A, B, C, D, solvent
CONC_INIT = np.array([.33, .33, 0, 0, 0])       # [mol/L]
VOL_INIT = 0.06                                 # [m3]


def feed(flow=FEED_FLOW, frac=FEED_FRAC):
    return LiquidStream(PATH, temp=TEMP, mass_flow=flow, mass_frac=frac,
                        name_solv='solvent')


def placeholder_feed():
    """A vessel needs an inlet; this one is replaced by the upstream line."""
    return feed(flow=1e-12, frac=np.array([0, 0, 0, 0, 1.0]))


def reactor(inlet=None, integrator=None):
    """Isothermal ContinuousReactor running A + B -> C, C + A -> D."""
    unit = ContinuousReactor(
        integrator=integrator or ScipyBackend(), h_conv=1e4, diam=0.4,
        controller=ContinuousVesselController(temp_func=lambda t: TEMP))
    unit.Phases = LiquidPhase(PATH, temp=TEMP, mole_conc=CONC_INIT.copy(),
                              vol=VOL_INIT, name_solv='solvent')
    unit.RxnKinetics = RxnKinetics(
        path=PATH, rxn_list=['A + B --> C', 'C + A --> D'],
        k_params=np.array([2.654e4, 5.3e2]), ea_params=np.array([4e4, 3e4]))
    unit.Inlet = inlet if inlet is not None else placeholder_feed()
    return unit


def solve_sequential(runtime, splits=None, verbose=False):
    sim = ParallelSimExec(PATH, FLOWSHEET, splits=splits)
    sim.R01 = reactor(feed())
    sim.R02a = reactor()
    sim.R02b = reactor()
    sim.R03 = reactor()
    sim.SolveFlowsheet(kwargs_run={name: {'runtime': runtime}
                                   for name in sim.execution_names},
                       verbose=verbose)
    return sim


def solve_concurrent(runtime, splits=None):
    members = {'R01': reactor(feed()), 'R02a': reactor(), 'R02b': reactor(),
               'R03': reactor()}
    cv = concurrent_from_flowsheet(
        FLOWSHEET, members, splits=splits,
        # The first line into each downstream reactor replaces its
        # placeholder feed; the second line into R03 is added beside it.
        placeholder_inlets={'R02a': 0, 'R02b': 0, 'R03': 0},
        integrator=ScipyBackend())
    cv.solve_unit(runtime=runtime)
    return cv


def solve_cv_as_line(runtime):
    """R01 --> (R02x, CV) --> R04, where CV holds R02a, R02b -> R03 with
    40 % of R03 recycled to R02a. The CV is one unit of the flowsheet: the
    split arrives on its entry member (R02a) and its exit member (R03)
    feeds R04 alongside R02x."""
    from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink

    cv = ConcurrentVessel(
        {'R02a': reactor(), 'R02b': reactor(feed(flow=0.002)),
         'R03': reactor()},
        links=[StreamLink('R02a', 'R03', inlet=0), StreamLink('R02b', 'R03'),
               StreamLink('R03', 'R02a', fraction=0.4)],
        integrator=ScipyBackend())

    sim = ParallelSimExec(PATH, 'R01 --> (R02x, CV) --> R04')
    sim.R01 = reactor(feed())
    sim.R02x = reactor()
    sim.CV = cv
    sim.R04 = reactor()
    sim.SolveFlowsheet(kwargs_run={name: {'runtime': runtime}
                                   for name in sim.execution_names},
                       verbose=False)
    return sim


def final(unit, key):
    return np.ravel(np.asarray(getattr(unit.result, key), dtype=float))[-1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--split', type=float, default=0.3,
                        help="share of R01's outlet sent to R02a")
    parser.add_argument('--runtime', type=float, default=3600.0)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    splits = {'R01': {'R02a': args.split}}
    os.makedirs(RESULTS, exist_ok=True)

    sim = solve_sequential(args.runtime, splits)
    cv = solve_concurrent(args.runtime, splits)

    print(sim.flowsheet)
    print('Run order:', sim.execution_names)
    print('R03 inlets (sequential):', len(sim.R03.inlet_connections))
    print('R03 inlets (concurrent):', len(cv.R03.inlet_connections))
    print('\nFinal outlet, sequential vs ConcurrentVessel')
    for name in sim.execution_names:
        q_seq = final(getattr(sim, name), 'outlet_vol_flow')[0]
        q_cv = final(cv[name], 'outlet_vol_flow')[0]
        c_cv = np.asarray(cv[name].result.mole_conc_liquid0)[-1]
        c_seq = np.asarray(getattr(sim, name).result.mole_conc_liquid0)[-1]
        gap = np.abs(c_seq - c_cv).max() / np.abs(c_cv).max()
        print(f"  {name:5s} flow {q_seq * 3.6e6:8.4f} vs {q_cv * 3.6e6:8.4f}"
              f" L/h   conc gap {gap:.2e}")

    fig = plot_flowsheet(sim, title='ParallelSimExec: R01 --> (R02a, R02b) '
                                    '--> R03')
    fig.savefig(os.path.join(RESULTS, 'flowsheet_parallel_simexec.png'),
                dpi=150)

    fig = plot_flowsheet(cv, title='The same units in one ConcurrentVessel')
    fig.savefig(os.path.join(RESULTS, 'flowsheet_parallel_concurrent.png'),
                dpi=150)

    sim_cv = solve_cv_as_line(args.runtime)
    fig = plot_flowsheet(sim_cv, title='A ConcurrentVessel as one of two '
                                       'parallel lines')
    fig.savefig(os.path.join(RESULTS, 'flowsheet_cv_as_line.png'), dpi=150)

    species = ['A', 'B', 'C', 'D']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    t_seq = np.asarray(sim.R03.result.time) / 60
    t_cv = np.asarray(cv.R03.result.time) / 60
    c_seq = np.asarray(sim.R03.result.mole_conc_liquid0)
    c_cv = np.asarray(cv.R03.result.mole_conc_liquid0)
    for i, comp in enumerate(species):
        line, = axes[0].plot(t_seq, c_seq[:, i], label=comp)
        axes[0].plot(t_cv, c_cv[:, i], '--', color=line.get_color())
    axes[0].set(xlabel='time [min]', ylabel='R03 concentration [mol/L]',
                title='R03: solid = sequential, dashed = concurrent')
    axes[0].legend(frameon=False)
    for name in sim.execution_names:
        unit = getattr(sim, name)
        axes[1].plot(np.asarray(unit.result.time) / 60,
                     np.ravel(unit.result.outlet_vol_flow) * 3.6e6,
                     label=name)
    axes[1].set(xlabel='time [min]', ylabel='outlet flow [L/h]',
                title='Outlet flows (sequential)')
    axes[1].legend(frameon=False, loc='center right')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'flowsheet_parallel_compare.png'),
                dpi=150)

    print(f'\nFigures saved to {RESULTS}')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
