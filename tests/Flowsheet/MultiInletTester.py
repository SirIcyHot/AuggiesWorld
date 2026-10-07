"""Several feed streams into one refactored ContinuousReactor.

Uses PharmaPy as installed, without modifying it, and asks two questions:

    Stage A  Isolated reactor, several feeds   -- do N separate inlets behave
             exactly like one premixed inlet? (material, then energy)
    Stage B  Two upstream reactors -> one      -- wired by hand, does the
             shared reactor receive both upstream trajectories?
    Stage C  Same layout through SimulationExec -- reports what PharmaPy's own
             flowsheet wiring delivers (INFO only, never fails the run)

Run it like FlowsheetTester.py:

    python MultiInletTester.py            # Assimulo backend
    python MultiInletTester.py scipy      # scipy backend

The compound database is looked up next to this file (data/ or alongside),
then in the PHARMAPY_COMPOUNDS environment variable.
"""

import faulthandler
import os
import sys
import traceback

import numpy as np
from scipy.optimize import brentq

from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.ProcessControl_Refactored import ContinuousVesselController
from PharmaPy.IntegratorBackends import AssimuloBackend, ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Connections import Connection
from PharmaPy.SimExec import SimulationExec

faulthandler.enable(file=sys.stderr, all_threads=True)

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_database():
    candidates = [
        os.path.join(HERE, 'data', 'compound_database.json'),
        os.path.join(HERE, 'compound_database.json'),
        os.environ.get('PHARMAPY_COMPOUNDS', ''),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise SystemExit(
        'compound_database.json not found. Put it in %s\\data or set '
        'PHARMAPY_COMPOUNDS to its path.' % HERE)


PATH = _find_database()

BACKEND = (sys.argv[1] if len(sys.argv) > 1
           else os.environ.get('PHARMAPY_BACKEND', 'assimulo')).lower()
BACKENDS = {'assimulo': AssimuloBackend, 'scipy': ScipyBackend}
if BACKEND not in BACKENDS:
    raise SystemExit('Unknown backend %r. Choose one of %s.'
                     % (BACKEND, sorted(BACKENDS)))

# Same chemistry and vessel as FlowsheetTester.py.
RXNS = ['A + B --> C', 'C + A --> D']
K_VALS = np.array([2.654e4, 5.3e2])
EA_VALS = np.array([4.0e4, 3.0e4])
TEMP_INIT = 313.15
CONC_INIT = np.array([0.33, 0.33, 0, 0, 0])
VOL_INIT = 0.06
H_CONV = 10000.0
VESSEL_DIAM = 0.4
RUNTIME = 1800.0

# Feeds for stage A: different flows, compositions and temperatures.
FRACS = [np.array([.30, .00, 0, 0, .70]),
         np.array([.00, .30, 0, 0, .70]),
         np.array([.15, .15, 0, 0, .70]),
         np.array([.05, .25, 0, 0, .70])]
FLOWS = [0.004, 0.006, 0.003, 0.005]           # kg/s
TEMPS = [293.15, 313.15, 333.15, 303.15]       # K

# Pass thresholds.
TOL_MATCH = 1e-6        # N feeds vs premixed feed, relative
TOL_FLOW = 1e-2         # shared reactor outflow vs summed upstream outflow


# ---------------------------------------------------------------- builders
def make_integrator():
    return BACKENDS[BACKEND](options={'maxh': 60})


def rxn_kinetics():
    return RxnKinetics(path=PATH, rxn_list=RXNS, k_params=K_VALS,
                       ea_params=EA_VALS)


def initial_liquid():
    return LiquidPhase(PATH, temp=TEMP_INIT, mole_conc=CONC_INIT.copy(),
                       vol=VOL_INIT, name_solv='solvent')


def mass_feed(flow, frac, temp):
    return LiquidStream(PATH, temp=temp, mass_flow=flow, mass_frac=frac,
                        name_solv='solvent')


def conc_feed(mole_conc, vol_flow=1e-5):
    return LiquidStream(PATH, temp=TEMP_INIT, mole_conc=mole_conc,
                        vol_flow=vol_flow, name_solv='solvent')


def reactor(inlet=None, adiabatic=False):
    """A ContinuousReactor; isothermal via its controller unless adiabatic."""
    if adiabatic:
        unit = ContinuousReactor(integrator=make_integrator(), h_conv=H_CONV,
                                 diam=VESSEL_DIAM, adiabatic=True)
    else:
        unit = ContinuousReactor(
            integrator=make_integrator(), h_conv=H_CONV, diam=VESSEL_DIAM,
            controller=ContinuousVesselController(
                temp_func=lambda t: TEMP_INIT))
    unit.Phases = initial_liquid()
    unit.RxnKinetics = rxn_kinetics()
    if inlet is not None:
        unit.Inlet = inlet
    return unit


def premixed_feed(n, temps):
    """One stream equal to the first n feeds mixed adiabatically."""
    q = np.array(FLOWS[:n])
    w = np.array(FRACS[:n])
    total = q.sum()
    w_mix = q @ w / total

    if len(set(temps)) == 1:
        return mass_feed(total, w_mix, temps[0])

    def enthalpy(stream, temp, frac):
        return float(np.ravel(stream.getEnthalpy(temp, mass_frac=frac))[0])

    h_in = sum(qi * enthalpy(mass_feed(qi, wi, ti), ti, wi)
               for qi, wi, ti in zip(q, w, temps))
    probe = mass_feed(total, w_mix, temps[0])
    temp_mix = brentq(lambda t: total * enthalpy(probe, t, w_mix) - h_in,
                      200.0, 450.0)
    return mass_feed(total, w_mix, temp_mix)


def end(unit, key):
    return np.ravel(np.asarray(getattr(unit.result, key), dtype=float)[-1])


def rel_gap(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def outflow(unit):
    return float(end(unit, 'outlet_vol_flow')[0])


def feed_from_upstream(destination, *sources):
    """Give `destination` one inlet per solved upstream unit.

    PharmaPy's Connection stamps the upstream trajectory onto the stream it
    hands over, then assigns destination.Inlet -- which replaces any inlet
    already there. So each source is transferred on its own, the inlet it
    produced is kept, and all of them are installed together at the end.
    """
    kept = []
    for source in sources:
        Connection(source_uo=source, destination_uo=destination).transfer_data()
        kept.extend(destination.inlet_connections)
    destination.inlet_connections = kept


# ---------------------------------------------------------------- stages
def stage_a_isolated():
    """N separate feeds must reproduce one premixed feed."""
    notes = []
    for adiabatic, temps_for in ((False, lambda n: [TEMP_INIT] * n),
                                 (True, lambda n: TEMPS[:n])):
        label = 'adiabatic' if adiabatic else 'isothermal'
        for n in (2, 3, 4):
            temps = temps_for(n)
            split = reactor([mass_feed(FLOWS[i], FRACS[i], temps[i])
                             for i in range(n)], adiabatic=adiabatic)
            mixed = reactor(premixed_feed(n, temps), adiabatic=adiabatic)
            split.solve_unit(runtime=RUNTIME, verbose=False)
            mixed.solve_unit(runtime=RUNTIME, verbose=False)

            if len(split.inlet_connections) != n:
                raise AssertionError('%s N=%d: reactor kept %d inlets'
                                     % (label, n, len(split.inlet_connections)))

            gaps = {key: rel_gap(end(split, key), end(mixed, key))
                    for key in ('mass_j_liquid0', 'mole_conc_liquid0',
                                'Total_m_in_vessel', 'outlet_vol_flow',
                                'global_temp')}
            worst = max(gaps, key=gaps.get)
            if gaps[worst] > TOL_MATCH:
                raise AssertionError(
                    '%s N=%d: %s differs from premixed feed by %.2e'
                    % (label, n, worst, gaps[worst]))

            temp_span = float(np.ptp(np.asarray(split.result.global_temp,
                                                dtype=float)))
            notes.append('%s N=%d %.0e (T span %.2f K)'
                         % (label, n, gaps[worst], temp_span))
    return '; '.join(notes)


def stage_b_manual_two_into_one():
    """Two upstream reactors wired into a third by hand."""
    r01 = reactor(conc_feed(np.array([1.0, 0, 0, 0, 0])))   # A only
    r02 = reactor(conc_feed(np.array([0, 1.0, 0, 0, 0])))   # B only
    r03 = reactor()
    r01.solve_unit(runtime=RUNTIME, verbose=False)
    r02.solve_unit(runtime=RUNTIME, verbose=False)

    feed_from_upstream(r03, r01, r02)
    if len(r03.inlet_connections) != 2:
        raise AssertionError('R03 has %d inlets, expected 2'
                             % len(r03.inlet_connections))

    r03.solve_unit(runtime=RUNTIME, verbose=False)

    # R03's two inlets must carry different streams: R01 ends A-rich and
    # R02 B-rich, so a mix-up shows as identical inlet compositions.
    inlet_conc = [np.ravel(np.asarray(list(c.stream)[0].mole_conc,
                                      dtype=float))[-5:]
                  for c in r03.inlet_connections]
    if rel_gap(inlet_conc[0], inlet_conc[1]) < 1e-6:
        raise AssertionError('both R03 inlets carry the same stream')

    fed = outflow(r01) + outflow(r02)
    drift = abs(outflow(r03) - fed) / fed
    if drift > TOL_FLOW:
        raise AssertionError('R03 outflow %.3e vs R01+R02 %.3e (rel %.2e)'
                             % (outflow(r03), fed, drift))

    return ('R03 got 2 inlets; outflow %.3e vs R01+R02 %.3e (rel %.1e)'
            % (outflow(r03), fed, drift))


def stage_c_simulationexec():
    """What SimulationExec delivers for the same layout. INFO, not a test."""
    flst = SimulationExec(PATH, flowsheet={'R01': ['R03'], 'R02': ['R03'],
                                           'R03': []})
    flst.R01 = reactor(conc_feed(np.array([1.0, 0, 0, 0, 0])))
    flst.R02 = reactor(conc_feed(np.array([0, 1.0, 0, 0, 0])))
    flst.R03 = reactor()
    flst.SolveFlowsheet(kwargs_run={name: {'runtime': RUNTIME,
                                           'verbose': False}
                                    for name in ('R01', 'R02', 'R03')},
                        verbose=False)

    # Each upstream unit run on its own feed alone, to show whether the
    # flowsheet replaced that feed with the other unit's outlet.
    overwritten = []
    for name, conc in (('R01', [1.0, 0, 0, 0, 0]), ('R02', [0, 1.0, 0, 0, 0])):
        alone = reactor(conc_feed(np.array(conc)))
        alone.solve_unit(runtime=RUNTIME, verbose=False)
        gap = rel_gap(end(getattr(flst, name), 'mole_conc_liquid0'),
                      end(alone, 'mole_conc_liquid0'))
        if gap > TOL_MATCH:
            overwritten.append('%s (off its own-feed run by %.1e)'
                               % (name, gap))

    fed = outflow(flst.R01) + outflow(flst.R02)
    return ('run order %s; R03 inlets %d; R03 outflow %.3e vs R01+R02 '
            '%.3e; feed replaced by upstream outlet: %s'
            % (' -> '.join(flst.execution_names),
               len(flst.R03.inlet_connections), outflow(flst.R03), fed,
               ', '.join(overwritten) or 'none'))


STAGES = (
    ('A  isolated reactor, 2-4 feeds vs premixed   ', stage_a_isolated, True),
    ('B  R01 + R02 -> R03, wired by hand            ',
     stage_b_manual_two_into_one, True),
    ('C  R01 + R02 -> R03 via SimulationExec (info) ',
     stage_c_simulationexec, False),
)


def main():
    results = {}
    for label, fn, counts in STAGES:
        print('\n===== running stage %s =====' % label.strip(), flush=True)
        try:
            detail = fn()
            status = 'PASS' if counts else 'INFO'
        except Exception as exc:
            traceback.print_exc()
            detail = '%s: %s' % (type(exc).__name__, exc)
            status = 'FAIL' if counts else 'INFO'
        results[label] = (status, detail)
        print(status, detail, flush=True)

    print('\n' + '=' * 78)
    print('MULTI-INLET RESULTS  (backend: %s)' % BACKEND)
    print('=' * 78)
    for label, (status, detail) in results.items():
        print('  [%s] %s %s' % (status, label, detail))
    return results


if __name__ == '__main__':
    main()
