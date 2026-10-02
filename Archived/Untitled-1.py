# -*- coding: utf-8 -*-
"""
r02_ncstr_base_case.py
=======================
The PharmaPy "base case": an N-CSTR-in-series R02 train, N adjustable, built
DIRECTLY from the reaction network / feed / kinetics / thermal machinery in
r02_ncstr_process_scale.py -- with NO SLSQP optimizer in the loop.
 
r02_ncstr_process_scale.py answers "what (T_i, tau_i, split_i) per stage
MINIMIZES total volume subject to thermal/inventory/conversion constraints".
That is a design search, and it is slow and easy to misread (a run that
fails to converge is not the same statement as "N is infeasible").
 
This module answers a smaller, more mechanical question: "here are N CSTRs
at conditions I specify -- build them and solve the train." No optimizer,
no constraint search. You choose N (and, optionally, per-stage T/tau/split),
this module builds the design dict optimize_train() would have produced and
hands it to the SAME build_pharmapy_train / run_pharmapy_train / report_*
functions r02_ncstr_process_scale.py already uses and has validated. There
is exactly one reaction network, one feed, one kinetics table, one thermal
model in this codebase -- both modules import it from
r02_ncstr_process_scale.py, so there is nothing to keep in sync by hand.
 
WHAT "BASE CASE" MEANS HERE
----------------------------
Every stage gets the same temperature and the same residence time
(tau_total / N) unless you override with explicit per-stage arrays -- i.e.
the textbook "N identical CSTRs in series", not a graded train. This is
the natural reference point to compare against the optimizer's graded
answer, and it is also just the fastest way to explore "what happens as I
add stages" by hand.
 
Substrate feed splitting follows r02_ncstr_process_scale.FEED_SPLIT_MODE by
default ('all_stage1' -- everything into stage 1). If you want the train to
actually share the exotherm across stages, pass split_mode='equal'; see the
"SUBSTRATE FEED SPLITTING" note in r02_ncstr_process_scale.py -- splitting
RESIDENCE TIME alone does not split the heat on this reaction network,
splitting the SUBSTRATE STREAM does.
 
Production basis, kinetics, feed composition, stoichiometry mode, thermal
parameters, temperature bounds (including the AcONO2 decomposition cap) all
come from r02_ncstr_process_scale.py's module-level constants. Change them
there; this module reads them, it does not redefine them.
 
USAGE
-----
    python3 r02_ncstr_base_case.py --N 4
    python3 r02_ncstr_base_case.py --N 4 --T 60 --tau 900
    python3 r02_ncstr_base_case.py --N 4 --T 55 --tau 1200 --split-mode equal
    python3 r02_ncstr_base_case.py --N 4 --nonisothermal
    python3 r02_ncstr_base_case.py --N 4 --nosim          # sizing only, no PharmaPy
    python3 r02_ncstr_base_case.py --temps-C 50,55,60,65 --taus-s 200,200,250,250
    python3 r02_ncstr_base_case.py --scan-N 1 8           # algebraic N=1..8 scan
 
Also importable:
    from r02_ncstr_base_case import build_base_case_design, run_base_case
    design = build_base_case_design(N=4, T_C=60.0, tau_total_s=900.0)
    design, results = run_base_case(N=4, isothermal=True)
"""
 
import numpy as np
 
import r02_ncstr_process_scale as ncstr
from r02_ncstr_process_scale import (
    # network / feed / sizing -- the single source of truth
    build_network, build_feed_streams, resolve_splits, size_train,
    _t_bounds,
    # PharmaPy layer
    build_pharmapy_train, run_pharmapy_train, pharmapy_available,
    # reporting
    report_design, report_pharmapy_run,
    # scan reused as-is for the pure-algebraic N sweep
    scan_n,
    # defaults / constants, so this file never redefines a condition
    DEFAULT_SUBSTRATE, DEFAULT_INLINE, FEED_SPLIT_MODE,
    N_MAX_SEARCH,
)
 
 
# =============================================================================
# 1.  BUILD THE DESIGN  --  no optimizer, just N stages at stated conditions
# =============================================================================
 
def build_base_case_design(N, T_C=60.0, tau_total_s=900.0,
                           substrate=DEFAULT_SUBSTRATE,
                           inline=DEFAULT_INLINE,
                           split_mode=None,
                           temps_C=None, taus_s=None,
                           weights=None,
                           verbose=True):
    """Construct the `design` dict for an N-CSTR train at directly-specified
    conditions. Same shape/keys as optimize_train()'s return value, so it is
    a drop-in for build_pharmapy_train / run_pharmapy_train / report_design.
 
    Two ways to specify the train:
      * scalar T_C, scalar tau_total_s  -> N IDENTICAL stages, each at T_C,
        each with tau = tau_total_s / N. This is "the base case".
      * temps_C / taus_s (length-N arrays) -> a graded train, e.g. cold/short
        at the front (highest substrate concentration, highest volumetric
        heat release) warming up downstream. N is then inferred from the
        array length and any passed-in N is checked against it.
 
    substrate/inline/split_mode/weights are passed straight through to
    r02_ncstr_process_scale's build_network / build_feed_streams /
    resolve_splits -- see that module for what each governs (in particular:
    STOICH_MODE and FEED_SPLIT_MODE live there, not here).
    """
    if temps_C is not None or taus_s is not None:
        if temps_C is None or taus_s is None:
            raise ValueError("pass both temps_C and taus_s together, or "
                             "neither (and use T_C/tau_total_s instead)")
        temps_C = np.asarray(temps_C, dtype=float)
        taus_s = np.asarray(taus_s, dtype=float)
        if len(temps_C) != len(taus_s):
            raise ValueError(f"temps_C (len {len(temps_C)}) and taus_s "
                             f"(len {len(taus_s)}) must be the same length")
        N_eff = len(temps_C)
        if N is not None and N != N_eff:
            raise ValueError(f"N={N} but temps_C/taus_s imply N={N_eff}; "
                             f"pass one or the other")
        N = N_eff
        temps = temps_C + 273.15
        taus = taus_s
    else:
        if N is None or N < 1:
            raise ValueError("N must be a positive integer (or supply "
                             "temps_C/taus_s directly)")
        temps = np.full(N, T_C + 273.15)
        taus = np.full(N, tau_total_s / N)
 
    split_mode = FEED_SPLIT_MODE if split_mode is None else split_mode
 
    net = build_network(substrate, inline)
    carrier, sub, phi, names = build_feed_streams(substrate, inline,
                                                   verbose=verbose)
 
    t_lo, t_hi = _t_bounds(inline)
    out_of_range = (temps < t_lo) | (temps > t_hi)
    if verbose and np.any(out_of_range):
        stages = ', '.join(str(i + 1) for i in np.where(out_of_range)[0])
        print(f"  [base case] WARNING: stage(s) {stages} run outside "
              f"[{t_lo-273.15:.0f}, {t_hi-273.15:.0f}] C -- that upper bound "
              f"already folds in\n  the AcONO2 decomposition cap "
              f"({ncstr.ACONO2_T_CAP_C:.0f} C) when applicable. The network "
              f"does not model AcONO2\n  decomposition chemistry, so results "
              f"above the cap understate the true heat release.")
 
    if taus.min() < ncstr.TAU_MIN_S and verbose:
        print(f"  [base case] WARNING: a stage tau ({taus.min():.1f} s) is "
              f"below TAU_MIN_S={ncstr.TAU_MIN_S:.0f} s -- the ideal-CSTR "
              f"assumption\n  (instantaneous mixing) is doing work it "
              f"hasn't earned down there.")
 
    splits = resolve_splits(N, split_mode, weights=weights)
 
    sized = size_train(carrier, sub, phi, splits, taus, temps, net, names,
                       substrate)
 
    return dict(N=N, success=bool(sized['feasible']), x=None,
               temps=temps, taus=taus, splits=splits, sized=sized,
               names=names, net=net, carrier=carrier, sub_stream=sub,
               phi=phi, substrate=substrate, inline=inline,
               split_mode=split_mode)
 
 
# =============================================================================
# 2.  RUN THE DESIGN  --  algebraic report, then (optionally) real PharmaPy
# =============================================================================
 
def run_base_case(N=None, T_C=60.0, tau_total_s=900.0, isothermal=True,
                  run_pharmapy=True, verbose=True, **design_kwargs):
    """Build the base-case design, print the algebraic sizing report (section
    4/5/6 of r02_ncstr_process_scale.py -- no PharmaPy needed for this part),
    then, if requested and importable, build and solve the real N-CSTR
    PharmaPy train and print that report too.
 
    Returns (design, results). results is None if the design was infeasible,
    PharmaPy was unavailable, or run_pharmapy=False.
    """
    design = build_base_case_design(N, T_C=T_C, tau_total_s=tau_total_s,
                                    verbose=verbose, **design_kwargs)
    sz = design['sized']
    if not sz['feasible']:
        print(f"  N={design['N']}: INFEASIBLE ({sz.get('reason', '?')}) -- "
              f"no PharmaPy train built.")
        return design, None
 
    if verbose:
        report_design(design)
 
    if not run_pharmapy:
        return design, None
    if not pharmapy_available():
        print("  PharmaPy could not be imported in this interpreter -- "
              "algebraic sizing above is unaffected. Activate the "
              "environment holding the Auggie-dev branch and re-run for "
              "the PharmaPy train.")
        return design, None
 
    results = {}
    for iso in ((isothermal,) if isinstance(isothermal, bool)
                else (True, False)):
        tag = 'ISOTHERMAL' if iso else 'NON-ISOTHERMAL'
        print(f"\n  ---- PharmaPy {tag} train, N={design['N']} ----")
        try:
            res = run_pharmapy_train(design, isothermal=iso)
            report_pharmapy_run(res, design, iso)
            results[tag] = res
        except Exception as e:
            print(f"  *** {tag} run FAILED: {e}")
            results[tag] = None
    return design, results
 
 
# =============================================================================
# 3.  N SWEEP  --  algebraic-only, reuses scan_n() as-is
# =============================================================================
# scan_n() in r02_ncstr_process_scale.py already IS "the base case swept over
# N": equal T, equal tau split, N=1..n_max, no PharmaPy. Not reimplemented
# here on purpose -- see that function for q_frac (where the exotherm lands)
# and u_ht_req (what the jacket needs) per N.
 
def sweep_N(n_max=N_MAX_SEARCH, substrate=DEFAULT_SUBSTRATE,
           inline=DEFAULT_INLINE, tau_total_s=900.0, T_C=60.0,
           split_mode=None):
    return scan_n(n_max=n_max, substrate=substrate,
                 use_nitric_acid_inline=inline, tau_total_s=tau_total_s,
                 temp_C=T_C, split_mode=split_mode, verbose=True)
 
 
# =============================================================================
# 4.  CLI
# =============================================================================
 
def _parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip() != '']
 
 
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Adjustable-N CSTR-in-series base case for R02, built "
                    "from r02_ncstr_process_scale.py's network/feed/sizing.")
    p.add_argument('--N', type=int, default=None,
                   help="number of identical CSTRs in series")
    p.add_argument('--T', dest='T_C', type=float, default=60.0,
                   help="stage temperature [C], applied to every stage "
                        "(default 60, the Jadhav T_ref)")
    p.add_argument('--tau', dest='tau_total_s', type=float, default=900.0,
                   help="TOTAL train residence time [s], split evenly "
                        "across N stages (default 900 = 15 min)")
    p.add_argument('--temps-C', type=str, default=None,
                   help="comma-separated per-stage temperatures [C]; "
                        "overrides --N/--T for a graded train")
    p.add_argument('--taus-s', type=str, default=None,
                   help="comma-separated per-stage residence times [s]; "
                        "required together with --temps-C")
    p.add_argument('--substrate', choices=['Hexamine', 'HDN'],
                   default=DEFAULT_SUBSTRATE)
    p.add_argument('--inline', action='store_true', default=DEFAULT_INLINE,
                   help="in-line AcONO2 formation route (default: "
                        "pre-formed)")
    p.add_argument('--split-mode', choices=['all_stage1', 'equal'],
                   default=None,
                   help=f"substrate feed split across stages (default: "
                        f"module default '{FEED_SPLIT_MODE}')")
    iso_grp = p.add_mutually_exclusive_group()
    iso_grp.add_argument('--isothermal', dest='iso', action='store_true',
                         default=True)
    iso_grp.add_argument('--nonisothermal', dest='iso', action='store_false')
    iso_grp.add_argument('--both', dest='iso_both', action='store_true',
                         default=False,
                         help="run both isothermal and non-isothermal")
    p.add_argument('--nosim', action='store_true',
                   help="algebraic sizing report only, skip PharmaPy")
    p.add_argument('--scan-N', dest='scan_N', type=int, nargs=2,
                   metavar=('N_MIN', 'N_MAX'), default=None,
                   help="pure algebraic N=N_MIN..N_MAX scan (no PharmaPy) "
                        "instead of building one train")
    args = p.parse_args()
 
    if args.scan_N is not None:
        # scan_n's n_max counts from 1; a N_MIN/N_MAX pair is provided for
        # convenience but the underlying scan always starts at N=1.
        sweep_N(n_max=args.scan_N[1], substrate=args.substrate,
               inline=args.inline, tau_total_s=args.tau_total_s,
               T_C=args.T_C, split_mode=args.split_mode)
        return
 
    kwargs = dict(substrate=args.substrate, inline=args.inline,
                 split_mode=args.split_mode)
    if args.temps_C is not None or args.taus_s is not None:
        if args.temps_C is None or args.taus_s is None:
            p.error("--temps-C and --taus-s must be given together")
        kwargs['temps_C'] = _parse_float_list(args.temps_C)
        kwargs['taus_s'] = _parse_float_list(args.taus_s)
        N = None
    else:
        N = args.N if args.N is not None else 1
 
    run_base_case(N=N, T_C=args.T_C, tau_total_s=args.tau_total_s,
                 isothermal=(True, False) if args.iso_both else args.iso,
                 run_pharmapy=not args.nosim, **kwargs)
 
 
if __name__ == "__main__":
    main()