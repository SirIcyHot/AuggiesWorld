# -*- coding: utf-8 -*-
"""
r02_ncstr_process_scale.py
===========================
Process-scale sizing and optimization of the R02 nitration step as a TRAIN OF
N CSTRs IN SERIES, at a production basis of 1500 kg/day of nitramine product,
on the acetyl-nitrate (AcONO2) route -- pre-formed or in-line.
 
Derived from r02_acono2_flowsheet.py (reaction network, feed basis, params_f
ordering discipline) and r02_flowsheet_fixed.py (SIGNED Arrhenius activation
energies). Uses the 14-species compound database.
 
WHAT THIS MODULE ANSWERS
------------------------
  1. What volumetric feed rate Q delivers 1500 kg/day of product?
  2. How many CSTRs (N) does the train need before the heat is manageable?
  3. What (T_i, tau_i) per stage minimizes total reactor volume while keeping
     every stage thermally stable (Ug <= Ur) and under the per-vessel
     inventory cap?
  4. Does the optimized train hold up when actually solved in PharmaPy,
     isothermally first and then non-isothermally?
 
Stages are NOT constrained to be identical -- each stage carries its own
temperature and its own residence time/volume. That is the entire point: the
first stage sees the highest reactant concentration and therefore the highest
volumetric heat release, so it generally wants to be small and cold, while
later stages can be larger and hotter to push conversion.
 
WHY N CSTRs FOR HEAT (THE SCALING ARGUMENT)
-------------------------------------------
Heat generation scales with VOLUME. Jacket heat-transfer area scales with
V^(2/3). Split one vessel of volume V into N vessels of volume V/N and the
total area goes as N * (V/N)^(2/3) = N^(1/3) * V^(2/3) -- i.e. splitting the
duty across N stages buys you N^(1/3) more area for the same total holdup,
plus it lets each stage sit at its own temperature and its own coolant
approach. That is the physical justification for the N-scan in section 7.
 
For energetic materials there is a SECOND, independent driver that often
binds before heat does: per-vessel explosive inventory. MAX_VESSEL_VOL_M3
below is a hard constraint, not an economic preference, and the reported N
is the max of (thermal N, inventory N).
 
THERMAL CRITERION -- Ug vs Ur (van Heerden)
--------------------------------------------
For each stage, at its steady state:
 
    Q_gen(T) = V_L * sum_j (-DH_j) * r_j(C_ss(T), T)          [W]
    Q_rem(T) = mdot*Cp*(T - T_in) + u_ht*A*(T - T_c)          [W]
 
    Ug = dQ_gen/dT   evaluated ALONG THE STEADY-STATE MANIFOLD
                     (mass balance re-solved at T +/- dT, so the stabilizing
                      dC_ss/dT feedback is included -- not just the bare
                      Arrhenius derivative at frozen composition)
    Ur = dQ_rem/dT = mdot*Cp + u_ht*A                          [W/K]
 
Steady state exists where Q_gen = Q_rem; it is STABLE where Ug < Ur. This
module enforces Ur >= STABILITY_MARGIN * Ug on every stage, AND separately
enforces that the duty actually closes with coolant no colder than
T_COOLANT_MIN_C. Both are collapsed into one reported number per stage,
u_ht_required = max(duty-required, slope-required), which is then compared
against U_HT_AVAILABLE. See section 6.
 
READ THIS BEFORE BELIEVING THE Ug NUMBERS
------------------------------------------
With the SIGNED Jadhav activation energies (+/-/-, see section 2), two of the
three nitration reactions are anti-Arrhenius: their rates FALL as T rises, so
they contribute NEGATIVELY to Ug. Only the RDX pathway pushes Ug up. The
network is therefore substantially self-stabilizing in the slope sense, and
you should expect the DUTY check -- not the Ug<Ur slope check -- to be the
binding constraint at most design points. The module reports which one binds
per stage (`binds` column) precisely so this doesn't get lost. If you find
Ug<Ur never binds anywhere, that is a real result of the +/-/- kinetics, not
a bug in the constraint.
 
That conclusion is only as good as DH_RDX/DH_HMX/DH_SIDE, which are still a
single lumped Jadhav calorimetric value applied identically to all three
pathways (see section 2). The moment fit_thermal_params.py returns
pathway-resolved DH values -- particularly if the Side pathway is confirmed
more exothermic than RDX, as expected -- the Ug balance shifts and this
should be re-run.
 
STATUS / PROVISIONAL
--------------------
  * Kinetics are Jadhav's nitric-acid-basis fit with the nitric acid order/Ea
    slot reassigned to AcONO2 -- PROVISIONAL, exactly as in
    r02_acono2_flowsheet.py. Refit against the EasyMax Set-2 DOE before
    trusting absolute volumes.
  * DH_RDX/HMX/SIDE are the single lumped Jadhav value. DH_PREFORM is real
    (Tsvetkov 1989 via NIST).
  * STOICH_MODE defaults to 'legacy' (1:1 hexamine:AcONO2) purely for
    continuity with r02_acono2_flowsheet.py. THIS IS CHEMICALLY WRONG AND IT
    MATTERS AT PROCESS SCALE -- it sets your AcONO2 consumption and your
    product-per-hexamine ceiling, i.e. your raw material bill. See section 2.
 
USAGE
-----
    python3 r02_ncstr_process_scale.py check      # species/DB sanity table
    python3 r02_ncstr_process_scale.py scan       # N-scan, find minimum N
    python3 r02_ncstr_process_scale.py optimize   # optimize the chosen N
    python3 r02_ncstr_process_scale.py isothermal # PharmaPy train, isothermal
    python3 r02_ncstr_process_scale.py nonisothermal
    python3 r02_ncstr_process_scale.py sweep      # T-range sweep of the train
    python3 r02_ncstr_process_scale.py all
 
Sections 1-8 are pure numpy/scipy and import NO PharmaPy -- same
dependency-light discipline as acoh_dilution_analysis.py, so the N-scan and
the optimizer stay fast. PharmaPy is imported lazily in section 9, which is
where the real CSTR objects get built and solved.
"""
 
import os
import json
import time
import numpy as np
from scipy.optimize import least_squares, minimize
 
# =============================================================================
# 0.  SCALE BASIS AND DESIGN SETTINGS
# =============================================================================
 
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "compound_databases", "fmcompound_database.json")
if not os.path.exists(DB_PATH):           # fall back to a sibling-file DB
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "compound_database.json")
 
# ---- Production basis --------------------------------------------------
PRODUCT_KG_PER_DAY = 1500.0    # kg/day of PRODUCT_BASIS, wet-free crystal basis
PRODUCT_BASIS      = 'RDX+HMX'  # 'RDX' or 'RDX+HMX'
                                # 'RDX+HMX' = total crystallizable nitramine
                                # leaving R02 (what CR01/F01 actually see).
                                # 'RDX' = spec on RDX alone; makes the plant
                                # bigger by 1/sel_RDX. Nitramines (linear
                                # side products) are NEVER counted as product.
ONSTREAM_FRACTION  = 1.0        # de-rate here for planned downtime if wanted
 
MW_G_MOL = {'RDX': 222.117, 'HMX': 296.156}   # g/mol
 
# ---- Temperature operating range ---------------------------------------
T_MIN_C = 45.0     # K bound, lower -- below this the train gets impractically
                   # long (and Jadhav has no data down here)
T_MAX_C = 75.0     # K bound, upper -- EXPLORATION bound, see cap below
T_REF_KIN_C = 60.0  # Jadhav T_ref. FIXED. Never sweeps with reactor T --
                    # setting temp_ref = T_op makes inv_temp = 0 identically
                    # and kills all temperature sensitivity.
 
# AcONO2 thermal decomposition onset is reported above ~60 C (Andreozzi et
# al., J. Hazard. Mater. 2002). Any stage running above this on the PRE-FORMED
# route is carrying a decomposing nitrating agent whose exotherm is NOT in
# this reaction network at all. The optimizer respects the cap; it does not
# model what happens if you ignore it.
ACONO2_T_CAP_C = 65.0
ENFORCE_ACONO2_CAP = True
 
# ---- Residence-time bounds per stage -----------------------------------
TAU_MIN_S = 5.0         # s. Deliberately LOW. Under the current (provisional)
                        # k values the substrate is consumed in ~25 s at feed
                        # concentration, so a 60 s floor would force stage 1 to
                        # full conversion and the optimizer could never explore
                        # splitting the exotherm at all. Below ~10-30 s the
                        # ideal-CSTR assumption is doing real work it hasn't
                        # earned (mixing time is no longer negligible) -- treat
                        # any optimum that lands on this bound as a signal to
                        # re-examine the kinetics, not as a design.
TAU_MAX_S = 3600.0      # 60 min -- inventory/holdup practicality
 
# ---- Heat transfer -----------------------------------------------------
# PharmaPy: u_ht = 1/(1/h_conv_reactor + 1/h_conv_jacket). Both default 1000
# -> u_ht = 500 W/m2/K, which is the value acoh_dilution_analysis.py assumes.
H_CONV_REACTOR   = 1000.0   # W/m2/K, process-side film
H_CONV_JACKET    = 1000.0   # W/m2/K, utility-side film
U_HT_AVAILABLE   = 1.0 / (1.0 / H_CONV_REACTOR + 1.0 / H_CONV_JACKET)   # 500
U_HT_CEILING     = 3000.0   # W/m2/K -- realistic ceiling (SiC / high-perf
                            # compact exchanger). Same ceiling as
                            # fit_thermal_params.py's H_CONV_MAX.
T_COOLANT_MIN_C  = 10.0     # chilled water. Drop to -10 for glycol, but then
                            # revisit freeze-out of AcOH (mp 16.6 C!) on the
                            # wall -- a real fouling risk on this system.
COOLANT_DELTA_T_K = 10.0    # design coolant temperature rise across a jacket
STABILITY_MARGIN = 2.0      # required Ur / Ug. 1.0 = marginal stability;
                            # 2.0 is a conventional design margin.
 
VOL_OFFSET = 0.75           # liquid fill fraction; mirrors Reactors.py
 
# ---- Hard process-safety constraint ------------------------------------
# Per-vessel energetic inventory cap. For an RDX/HMX slurry this is set by
# your facility's siting/quantity-distance basis, NOT by economics. Placeholder
# -- REPLACE WITH YOUR ACTUAL LICENSED LIMIT before this design means anything.
MAX_VESSEL_VOL_M3 = 0.50    # m3 liquid (500 L)
 
# ---- Conversion target -------------------------------------------------
X_HEXAMINE_MIN = 0.85       # fractional substrate conversion out of the train
 
# ---- Mixture physical properties (fast layer only) ---------------------
# The PharmaPy layer (section 9) reads real properties from the compound
# database. These constants are for the scipy screening layer only, and are
# an acetic-acid-solvent basis: the feed is ~5.9 mol/L AcOH and AcOH dominates
# both rho and Cp. Cross-check against fmcompound_database.json if you start
# using absolute Q_gen numbers from the fast layer for anything load-bearing.
RHO_MIX_KG_M3 = 1050.0      # ~AcOH (1049) with dissolved solids
CP_MIX_J_KG_K = 2050.0      # AcOH: 123 J/mol/K / 0.06005 kg/mol = 2048
 
R_GAS = 8.314462618         # J/mol/K
C_FLOOR = 1e-6              # mol/L. Numerical floor on rate-law species --
                            # fractional exponents produce NaN the instant a
                            # solver perturbs a zero concentration slightly
                            # negative. Same floor as the flowsheet modules.
 
# =============================================================================
# 1.  SPECIES / DATABASE HELPERS  (no hardcoded positional indices, ever)
# =============================================================================
 
def _load_name_species(db_path=DB_PATH):
    with open(db_path) as f:
        db = json.load(f)
    return list(db.keys())
 
 
def build_conc_array(conc_dict, db_path=DB_PATH):
    """Full-length mole_conc array [mol/L] from {species: conc}, positions
    resolved from the LIVE database key order."""
    names = _load_name_species(db_path)
    arr = np.zeros(len(names))
    for name, val in conc_dict.items():
        if name not in names:
            raise KeyError(f"'{name}' not in {db_path}. Add it before "
                           f"building this feed.")
        arr[names.index(name)] = val
    return arr, names
 
 
def sanity_check_ordering(db_path=DB_PATH):
    """Print the live DB index of every species this module touches. Run once
    before trusting anything else in here."""
    names = _load_name_species(db_path)
    used = ['Nitric_Acid', 'Water', 'Hexamine', 'RDX', 'HMX', 'HDN',
            'Acetic_Anhydride', 'Acetic_Acid', 'Ammonium_Nitrate',
            'Nitramines', 'AcONO2']
    print(f"Database: {db_path}")
    print(f"  {len(names)} species found "
          f"({'OK -- 14 as expected' if len(names) == 14 else 'NOT 14 -- check'})")
    print(f"  {'Species':<20}{'DB index':>10}")
    for s in used:
        idx = names.index(s) if s in names else 'MISSING'
        flag = '' if s in names else '   <-- add to compound database'
        print(f"  {s:<20}{str(idx):>10}{flag}")
    return names
 
 
# =============================================================================
# 2.  REACTION NETWORK -- SINGLE SOURCE OF TRUTH
# =============================================================================
# Both the fast scipy layer (sections 4-8) and the PharmaPy layer (section 9)
# are built from the structures below. Do not define the network twice.
#
# ---- Rate constants: k(T_ref), Jadhav Table 3 @ 60 C, L/mol/min -> /s -----
K0_RDX  = 0.070 / 60    # (L/mol)^(sum_a - 1)/s at T_ref
K0_HMX  = 0.038 / 60
K0_SIDE = 0.200 / 60
 
# ---- Activation energies: SIGNED (+/-/-) ---------------------------------
# From r02_flowsheet_fixed.py, which refit Jadhav Table 3 by two-point
# Arrhenius. Jadhav REPORTS all three positive; the reported values are in
# error. Table 3's own k values for HMX and Side DECREASE with temperature,
# and the paper's conclusion text says so explicitly ("Rate of formation of
# RDX is increasing with temperature while decreasing with HMX and side
# products"). The two-point refit reproduces Table 3 to within 0.4%. In
# PharmaPy's reparametrized form
#       k(T) = k(T_ref) * exp(Ea/R * (1/T_ref - 1/T))
# genuine anti-Arrhenius behavior REQUIRES Ea < 0.
EA_RDX  =   78024.1     # J/mol   POSITIVE  (~18.65 kcal/mol; Jadhav: 18.71)
EA_HMX  =  -52623.6     # J/mol   NEGATIVE  (Jadhav magnitude: 12.95)
EA_SIDE = -101436.6     # J/mol   NEGATIVE  (Jadhav magnitude: 25.20)
 
# ---- Heats of reaction ---------------------------------------------------
# Jadhav calorimetric value, hexamine basis, applied identically to all three
# parallel pathways because each consumes 1 mol hexamine as written. This is a
# LUMP, not three measurements. Selectivity therefore has NO effect on total
# heat release in this model -- which is almost certainly wrong (the linear
# nitramine byproduct pathway is expected to be MORE exothermic than HMX), and
# it is the single biggest soft spot in every thermal number this module
# prints. Replace from fit_thermal_params.py output.
_DH_JADHAV = -148.78 * 4184     # J/mol = -622,496 J/mol hexamine
DH_RDX = DH_HMX = DH_SIDE = _DH_JADHAV
 
# ---- Pre-formation reaction (in-line route only) -------------------------
K0_PREFORM = 5.0        # L/mol/s  PLACEHOLDER -- deliberately fast
EA_PREFORM = 30000.0    # J/mol    PLACEHOLDER -- low barrier
DH_PREFORM = -16700.0   # J/mol    REAL: Tsvetkov et al. 1989 via NIST WebBook
                        # (reverse of AcONO2 + AcOH -> Ac2O + HNO3, +16.7)
 
# ---- Reaction orders (Jadhav Table 3; nitric acid slot -> AcONO2) --------
# NOTE: these are KINETIC concentration dependences fitted to a power law.
# They are NOT stoichiometric coefficients and must not be reused as such --
# stoichiometry is set separately below.
ORDERS_RDX  = dict(substrate=0.29, acono2=1.46)
ORDERS_HMX  = dict(substrate=0.21, acono2=1.60)
ORDERS_SIDE = dict(substrate=0.50, acono2=0.97)
ORDERS_PREFORM = dict(nitric_acid=1.0, ac2o=1.0)
 
# ---- STOICHIOMETRY -- READ THIS ------------------------------------------
# 'legacy'   : 1 substrate + 1 AcONO2 --> 1 product + 1 AcOH.
#              Exactly what r02_acono2_flowsheet.py uses. Chemically WRONG
#              (RDX carries three nitro groups; it cannot come from one
#              AcONO2), and it caps product at 1 mol per mol hexamine.
#              Kept as the default ONLY so this module's numbers are directly
#              comparable to the bench-scale module you already trust.
#
# 'bachmann' : atom-balanced on the classical Bachmann equation. Starting from
#                 Hex + 4 HNO3 + 2 NH4NO3 + 6 Ac2O --> 2 RDX + 12 AcOH
#              and substituting HNO3 + Ac2O --> AcONO2 + AcOH gives
#                 Hex + 4 AcONO2 + 2 NH4NO3 + 2 Ac2O --> 2 RDX + 8 AcOH
#              (C 22=22, H 44=44, N 12=12, O 28=28 -- verified.)
#              Ammonium_Nitrate appears as a stoichiometric N source here.
#              That directly CONTRADICTS Zheng's proton-shuttle reading, which
#              is why r02_acono2_flowsheet.py dropped it. Its kinetic order
#              stays 0.0 either way (consistent with both Jadhav and Zheng);
#              only its consumption changes. Set 3 of the EasyMax DOE (the
#              ammonium nitrate add-back arms) is the experiment that decides
#              which of these two modes is right.
#
# AT PROCESS SCALE THIS IS NOT COSMETIC. 'legacy' consumes 0.51 mol/L AcONO2
# against a 3.84 mol/L charge (87% of your nitrating agent leaves unreacted)
# and caps you at 1 product/hexamine. 'bachmann' consumes 2.04 mol/L and gives
# 2 product/hexamine, roughly HALVING the hexamine feed for the same 1500
# kg/day. That is a raw-material-bill-sized difference. Do not ship a design
# on 'legacy' -- ship on whichever the DOE supports.
STOICH_MODE = 'legacy'
 
 
def _nu_nitration(product, substrate, mode=None):
    """Stoichiometry dict {species: coeff} for one nitration pathway.
    Negative = consumed. HDN releases its 2 dinitrate counterions as
    Nitric_Acid on reacting."""
    mode = STOICH_MODE if mode is None else mode
    if mode == 'legacy':
        nu = {substrate: -1.0, 'AcONO2': -1.0, product: 1.0, 'Acetic_Acid': 1.0}
    elif mode == 'bachmann':
        nu = {substrate: -1.0, 'AcONO2': -4.0, 'Ammonium_Nitrate': -2.0,
              'Acetic_Anhydride': -2.0, product: 2.0, 'Acetic_Acid': 8.0}
    else:
        raise ValueError(f"STOICH_MODE must be 'legacy' or 'bachmann', "
                         f"got {mode!r}")
    if substrate == 'HDN':
        nu['Nitric_Acid'] = nu.get('Nitric_Acid', 0.0) + 2.0
    return nu
 
 
def build_network(substrate='Hexamine', use_nitric_acid_inline=False,
                  mode=None):
    """Return the reaction network as a list of dicts -- the single structure
    both the scipy layer and the PharmaPy layer are built from.
 
    Each entry: name, nu {species: coeff}, orders {species: exponent},
                k0 [k(T_ref)], ea [J/mol, SIGNED], dh [J/mol].
    """
    if substrate not in ('Hexamine', 'HDN'):
        raise ValueError("substrate must be 'Hexamine' or 'HDN'")
 
    net = []
    for pname, prod, orders, k0, ea, dh in [
        ('RDX',  'RDX',        ORDERS_RDX,  K0_RDX,  EA_RDX,  DH_RDX),
        ('HMX',  'HMX',        ORDERS_HMX,  K0_HMX,  EA_HMX,  DH_HMX),
        ('Side', 'Nitramines', ORDERS_SIDE, K0_SIDE, EA_SIDE, DH_SIDE),
    ]:
        nu = _nu_nitration(prod, substrate, mode)
        order_dict = {substrate: orders['substrate'],
                      'AcONO2': orders['acono2']}
        # Any species that is CONSUMED must carry an order slot in PharmaPy
        # (order_map = stoich_matrix < 0). Kinetically silent consumed
        # species get order 0.0 -- consumed stoichiometrically, absent from
        # the rate law. This is what keeps 'bachmann' mode legal.
        for sp, coeff in nu.items():
            if coeff < 0 and sp not in order_dict:
                order_dict[sp] = 0.0
        net.append(dict(name=pname, nu=nu, orders=order_dict,
                        k0=k0, ea=ea, dh=dh))
 
    if use_nitric_acid_inline:
        net.append(dict(
            name='Preform',
            nu={'Nitric_Acid': -1.0, 'Acetic_Anhydride': -1.0,
                'AcONO2': 1.0, 'Acetic_Acid': 1.0},
            orders={'Nitric_Acid': ORDERS_PREFORM['nitric_acid'],
                    'Acetic_Anhydride': ORDERS_PREFORM['ac2o']},
            k0=K0_PREFORM, ea=EA_PREFORM, dh=DH_PREFORM))
    return net
 
 
def _rxn_string(nu):
    """PharmaPy reaction string from a stoichiometry dict. PharmaPy's
    disect_rxns regex '^\\d+(\\.\\d+)?(/\\d+)?\\s?' parses both integer and
    decimal leading coefficients, so '4 AcONO2' and '1.5 AcONO2' are both
    legal."""
    def side(sign):
        toks = []
        for sp, c in nu.items():
            if sign * c > 0:
                mag = abs(c)
                toks.append(sp if abs(mag - 1.0) < 1e-12 else f"{mag:g} {sp}")
        return ' + '.join(toks)
    return f"{side(-1)} --> {side(+1)}"
 
 
def build_params_f(net, db_path=DB_PATH):
    """params_f in the order RxnKinetics actually consumes it: per reaction,
    reactant orders sorted by DB index -- NOT by the order they appear in the
    reaction string.
 
    PharmaPy flattens params_f into `orders[stoich_matrix < 0]`, row-major,
    after permuting partic_species into DB order (Commons.get_permutation_
    indexes). So the column order is DB order, full stop. This is the
    generalized fix for the HMX/Side params_f order-swap bug -- nothing here
    is hardcoded, so it survives you reordering the JSON.
 
    Returns a list of lists (NOT an ndarray -- RxnKinetics wants lists).
    """
    names = _load_name_species(db_path)
    params_f = []
    for rxn in net:
        reactants = [sp for sp, c in rxn['nu'].items() if c < 0]
        missing = set(reactants) - set(rxn['orders'])
        if missing:
            raise ValueError(f"reaction '{rxn['name']}': consumed species "
                             f"{missing} have no order assigned. Every "
                             f"species with negative stoich needs one "
                             f"(use 0.0 for kinetically silent).")
        reactants.sort(key=lambda s: names.index(s))
        params_f.append([float(rxn['orders'][s]) for s in reactants])
    return params_f
 
 
# =============================================================================
# 3.  FEED  (mol/L -- PharmaPy internal units. NEVER mol/m3.)
# =============================================================================
# Same basis as r02_acono2_flowsheet.py / EasyMax DOE Set 2.
 
NITRATE_TARGET = 3.841   # mol/L nitrating dose (as AcONO2, or as HNO3 inline)
AC2O_TOTAL     = 5.276   # mol/L
AC2O_EXCESS    = AC2O_TOTAL - NITRATE_TARGET     # 1.435
ACOH_TOTAL     = 5.938   # mol/L
ACOH_TOPUP     = ACOH_TOTAL - NITRATE_TARGET     # 2.097
SUBSTRATE_CONC = 0.509   # mol/L Hexamine or HDN
AMMONIUM_NITRATE_DOSE = 1.11   # mol/L, AN:Hex = 2.18 (Jadhav's fixed ratio --
                               # the ratio at which his k values are embedded)
 
ACOH_PURE_CONC = 17.47   # mol/L, neat acetic acid (1049 g/L / 60.05 g/mol)
 
T_FEED_C = 30.0
 
# ---- SUBSTRATE FEED SPLITTING -- the lever that actually matters ---------
# Splitting residence time across N stages does NOT split the exotherm. Under
# these kinetics the substrate is consumed almost entirely in stage 1 at any
# practical tau, so an equal-tau train leaves stage 1 carrying ~100% of the
# heat with LESS jacket area than a single vessel -- strictly worse. Run
# `scan_n(...)` and read the q_frac column; it is unambiguous.
#
# The lever is splitting the SUBSTRATE stream itself: feed the AcONO2/AcOH
# carrier into stage 1 and inject hexamine (or HDN) into each stage. This
# does two things at once:
#   * splits the exotherm, because heat follows substrate consumption, and
#   * IMPROVES selectivity, because the substrate reaction orders run
#     Side 0.50 > RDX 0.29 > HMX 0.21, so a LOW residual substrate
#     concentration preferentially starves the side pathway. Keeping every
#     stage substrate-lean is good for both heat and selectivity at once.
#     (This is also why a PFR would be the wrong answer here -- it would hold
#     substrate concentration HIGH along the front end and feed the side
#     pathway. Backmixing is your friend on this network.)
#
# This maps onto the BAE patent topology, where the substrate and nitrating
# lines are already separate (Lines B+C meet before Line A), so a distributed
# substrate injection is a piping change, not a new unit operation.
#
# Stream basis: an overall blend of `carrier` and `substrate` streams that
# reproduces the DOE feed exactly. build_feed_streams() back-computes the
# carrier from the overall feed, so nothing is invented -- but it does need
# SUBSTRATE_STREAM_CONC (a solubility question, see below) and it will raise
# if the implied carrier composition is not physical.
FEED_SPLIT_MODE = 'all_stage1'   # 'all_stage1' | 'equal' | 'optimized'
 
SUBSTRATE_STREAM_CONC = 2.0      # mol/L substrate in the AcOH stream from MIX.
                                 # PLACEHOLDER -- this is a SOLUBILITY limit
                                 # and you should measure it. It sets phi (the
                                 # substrate stream's volume fraction) and
                                 # hence how concentrated the carrier has to
                                 # be. Raise it and the carrier gets easier;
                                 # lower it and the carrier eventually goes
                                 # infeasible (build_feed_streams will say so).
SUBSTRATE_STREAM_ACOH_FRAC = 0.85  # fraction of neat AcOH molar density left
                                   # in the substrate stream once the
                                   # substrate occupies its own volume.
 
 
# Rough molar volumes [L/mol] for a volume-closure sanity check on the
# back-computed streams. These are pure-component values at ~25 C and ignore
# excess volume of mixing, so treat the check as "is this stream roughly
# physical", never as a density model.
V_MOLAR_L_PER_MOL = {
    'Hexamine': 0.105, 'HDN': 0.166, 'Acetic_Acid': 0.0573,
    'Acetic_Anhydride': 0.0945, 'AcONO2': 0.085, 'Nitric_Acid': 0.0424,
    'Water': 0.0180, 'Ammonium_Nitrate': 0.0466, 'RDX': 0.123,
    'HMX': 0.156, 'Nitramines': 0.120, 'DPT': 0.130, 'PHX': 0.130,
    'INT1': 0.120,
}
 
 
def _volume_closure(conc, names):
    """Sum of C_i * v_molar_i for a stream. Should land near 1.0 L/L."""
    return float(sum(conc[i] * V_MOLAR_L_PER_MOL.get(n, 0.0)
                     for i, n in enumerate(names)))
 
 
def build_feed_streams(substrate='Hexamine', use_nitric_acid_inline=False,
                       db_path=DB_PATH, verbose=False, **kw):
    """Split the overall DOE feed into a CARRIER stream (nitrating agent +
    solvent) and a SUBSTRATE stream (hexamine/HDN in AcOH).
 
    Returns (carrier_conc, substrate_conc, phi, names) with
    phi = Q_substrate / Q_total, such that
 
        phi * substrate_conc + (1 - phi) * carrier_conc == overall feed
 
    holds componentwise BY CONSTRUCTION. The carrier is back-computed from the
    overall feed, not invented, so the blend always reproduces build_feed()
    exactly and the split is a pure repartitioning of a feed you already
    trust.
 
    The AcOH content of the substrate stream is capped automatically at what
    the overall feed can actually supply. This matters for the in-line route:
    there AcOH is generated in situ from Ac2O + HNO3, so the FEED carries very
    little of it, and a substrate stream at 0.85 * neat AcOH would demand more
    AcOH than exists. Rather than fail, the stream is thinned to the feasible
    maximum -- and then flagged, because a thinned stream may not physically
    close on volume (see below).
    """
    overall, names = build_feed(substrate, use_nitric_acid_inline,
                                db_path=db_path, **kw)
    phi = SUBSTRATE_CONC / SUBSTRATE_STREAM_CONC
    if not (0.0 < phi < 1.0):
        raise ValueError(
            f"SUBSTRATE_STREAM_CONC={SUBSTRATE_STREAM_CONC} gives substrate "
            f"stream volume fraction phi={phi:.3f}. Must be in (0,1): the "
            f"stream cannot be more dilute than the blend it has to make.")
 
    i_acoh = names.index('Acetic_Acid')
    acoh_want = ACOH_PURE_CONC * SUBSTRATE_STREAM_ACOH_FRAC
    acoh_max = overall[i_acoh] / phi            # all the feed's AcOH, no more
    acoh_sub = min(acoh_want, acoh_max)
    thinned = acoh_sub < acoh_want - 1e-9
 
    sub = np.zeros(len(names))
    sub[names.index(substrate)] = SUBSTRATE_STREAM_CONC
    sub[i_acoh] = acoh_sub
 
    carrier = (overall - phi * sub) / (1.0 - phi)
    bad = {names[i]: float(carrier[i]) for i in np.where(carrier < -1e-9)[0]}
    if bad:
        raise ValueError(
            f"Implied carrier stream is negative in {bad}. The substrate "
            f"stream carries more of those species than the overall feed "
            f"contains. Lower SUBSTRATE_STREAM_CONC or revisit the feed.")
    carrier = np.maximum(carrier, 0.0)
 
    v_sub, v_car = _volume_closure(sub, names), _volume_closure(carrier, names)
    if verbose or thinned or not (0.80 <= v_sub <= 1.15):
        if thinned:
            print(f"  [feed] substrate stream AcOH thinned to "
                  f"{acoh_sub:.2f} mol/L (wanted {acoh_want:.2f}): the "
                  f"{'in-line' if use_nitric_acid_inline else 'pre-formed'} "
                  f"feed only carries {overall[i_acoh]:.2f} mol/L AcOH total.")
        print(f"  [feed] volume closure: substrate {v_sub:.2f} L/L, "
              f"carrier {v_car:.2f} L/L (want ~1.0)")
        if v_sub < 0.80:
            print(f"  [feed] *** The substrate stream does not close on "
                  f"volume -- {1-v_sub:.2f} L/L is unaccounted for. As "
                  f"written it is\n         NOT a physically realizable "
                  f"solution: {SUBSTRATE_CONC/phi:.2f} mol/L "
                  f"{substrate} needs a solvent and this feed has no spare\n"
                  f"         AcOH to give it. For the in-line route the "
                  f"substrate would have to be carried in Ac2O, or fed as a "
                  f"slurry/melt,\n         or phi raised. This is a real "
                  f"flowsheet decision, not a tuning constant -- the split "
                  f"numbers below are\n         arithmetically consistent "
                  f"but do not yet describe a stream you can pump. ***")
 
    return carrier, sub, phi, names
 
 
def resolve_splits(N, mode=None, weights=None):
    """Fraction of the substrate stream injected into each stage; sums to 1."""
    mode = FEED_SPLIT_MODE if mode is None else mode
    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=float), 1e-9, None)
        return w / w.sum()
    if mode == 'all_stage1':
        s = np.zeros(N); s[0] = 1.0; return s
    if mode in ('equal', 'optimized'):
        return np.full(N, 1.0 / N)
    raise ValueError(f"FEED_SPLIT_MODE must be 'all_stage1' | 'equal' | "
                     f"'optimized', got {mode!r}")
 
 
 
def build_feed(substrate='Hexamine', use_nitric_acid_inline=False,
               include_ammonium_nitrate=None, dilution_factor=1.0,
               db_path=DB_PATH):
    """Feed mole_conc [mol/L] for one configuration.
 
    include_ammonium_nitrate defaults to True iff STOICH_MODE=='bachmann'
    (where AN is a stoichiometric N source and the train physically cannot run
    without it). On 'legacy' it defaults False, matching
    r02_acono2_flowsheet.py.
 
    dilution_factor f: reactive species scale by f; AcOH takes up the balance
    as a volume-weighted blend with neat AcOH -- NOT a naive uniform scale.
    f=1.0 is the undiluted DOE feed.
    """
    if include_ammonium_nitrate is None:
        include_ammonium_nitrate = (STOICH_MODE == 'bachmann')
 
    f = float(dilution_factor)
    conc = {substrate: SUBSTRATE_CONC * f, 'Water': 0.0}
 
    if use_nitric_acid_inline:
        conc['Nitric_Acid']      = NITRATE_TARGET * f
        conc['AcONO2']           = 0.0
        conc['Acetic_Anhydride'] = AC2O_TOTAL * f
        acoh_orig = ACOH_TOPUP
    else:
        conc['Nitric_Acid']      = 0.0
        conc['AcONO2']           = NITRATE_TARGET * f
        conc['Acetic_Anhydride'] = AC2O_EXCESS * f
        acoh_orig = ACOH_TOTAL
 
    conc['Acetic_Acid'] = acoh_orig * f + ACOH_PURE_CONC * (1.0 - f)
    conc['Ammonium_Nitrate'] = (AMMONIUM_NITRATE_DOSE * f
                                if include_ammonium_nitrate else 0.0)
    for p in ('RDX', 'HMX', 'Nitramines'):
        conc[p] = 0.0
    return build_conc_array(conc, db_path)
 
 
# =============================================================================
# 4.  FAST ALGEBRAIC STEADY-STATE SOLVER  (scipy only -- no PharmaPy)
# =============================================================================
# Solved in REACTION EXTENTS, not concentrations: 3-4 unknowns instead of 14,
# far better conditioned, and non-negativity is enforceable by bounds.
#
#   C = C_in + sum_j nu_j * xi_j          xi_j = tau * r_j   [mol/L]
#   residual_j(xi) = xi_j - tau * r_j(C(xi), T) = 0
 
_MATRIX_CACHE = {}
 
 
def _net_key(net, names):
    """Cheap hashable signature. The network is a list of dicts, so it can't be
    an lru_cache key directly, but name+order structure fully determines both
    the stoichiometric matrix and the order terms."""
    return (tuple((r['name'], tuple(sorted(r['nu'].items())),
                   tuple(sorted(r['orders'].items()))) for r in net),
            tuple(names))
 
 
_MATRIX_CACHE = {}
 
 
def _net_key(net, names):
    """Cheap hashable signature for a network. `net` is a list of dicts so it
    can't key an lru_cache directly, but the stoichiometry and orders fully
    determine every structural matrix built from it."""
    return (tuple((r['name'], tuple(sorted(r['nu'].items())),
                   tuple(sorted(r['orders'].items()))) for r in net),
            tuple(names))
 
 
def _stoich_matrix(net, names):
    """(num_rxn x num_species) stoichiometric matrix in DB column order.
 
    Cached: the optimizer rebuilt this on every single objective and
    constraint evaluation, which meant re-running names.index() inside a
    double loop a few hundred thousand times per SLSQP start.
    """
    key = ('S',) + _net_key(net, names)
    if key not in _MATRIX_CACHE:
        S = np.zeros((len(net), len(names)))
        for j, rxn in enumerate(net):
            for sp, c in rxn['nu'].items():
                S[j, names.index(sp)] = c
        _MATRIX_CACHE[key] = S
    return _MATRIX_CACHE[key]
 
 
def _order_matrix(net, names):
    """(num_rxn x num_species) rate-law exponents in DB column order."""
    key = ('A',) + _net_key(net, names)
    if key not in _MATRIX_CACHE:
        A = np.zeros((len(net), len(names)))
        for j, rxn in enumerate(net):
            for sp, a in rxn['orders'].items():
                A[j, names.index(sp)] = a
        _MATRIX_CACHE[key] = A
    return _MATRIX_CACHE[key]
 
 
def k_of_T(net, temp, temp_ref=None):
    """k(T) in PharmaPy's reparametrized form:
           k(T) = k(T_ref) * exp(Ea/R * (1/T_ref - 1/T))
    SIGNED Ea: Ea>0 rises with T, Ea<0 falls with T (anti-Arrhenius).
    temp_ref is FIXED at the Jadhav reference and must never track T."""
    tref = (T_REF_KIN_C + 273.15) if temp_ref is None else temp_ref
    k0 = np.array([r['k0'] for r in net])
    ea = np.array([r['ea'] for r in net])
    return k0 * np.exp(ea / R_GAS * (1.0 / tref - 1.0 / temp))
 
 
def _order_terms(net, names, A=None):
    """Per reaction, the (species_index, exponent) pairs with NONZERO
    exponent. Order-zero species (ammonium nitrate; acetic anhydride under
    'bachmann') are consumed stoichiometrically but absent from the rate law,
    so they must be skipped, not raised to the power 0."""
    key = ('T',) + _net_key(net, names)
    if key not in _MATRIX_CACHE:
        Am = _order_matrix(net, names) if A is None else A
        _MATRIX_CACHE[key] = [[(int(i), float(Am[j, i]))
                               for i in np.nonzero(Am[j])[0]]
                              for j in range(len(net))]
    return _MATRIX_CACHE[key]
 
 
def rates_at(conc, temp, net, names, A=None, terms=None):
    """Rate of every reaction [mol/L/s] at (conc, temp).
 
    Concentrations are clipped at ZERO, not at C_FLOOR. Clipping at zero is
    what makes the rate vanish when a reactant is exhausted, which is what
    keeps the steady-state fixed point non-negative all by itself. A 1e-6
    floor here would instead manufacture a small positive rate out of an
    empty reactant and drive C negative -- which is exactly what made every
    N>=2 stage report "infeasible" on the first pass.
 
    (The 1e-6 floor is still the right fix on the PharmaPy side: there it
    guards against CVode's finite-difference Jacobian perturbing a zero
    concentration slightly NEGATIVE, where a fractional exponent gives NaN.
    Different failure, different fix. C_FLOOR is kept for that use.)
 
    A direct power product is used rather than exp(A @ log c): the log form
    hits 0 * -inf = NaN for any order-zero species sitting at zero.
    """
    terms = _order_terms(net, names, A) if terms is None else terms
    c = np.clip(np.asarray(conc, dtype=float), 0.0, None)
    k = k_of_T(net, temp)
    out = np.empty(len(net))
    for j, tj in enumerate(terms):
        p = 1.0
        for i, a in tj:
            p *= c[i] ** a          # 0.0**0.29 = 0.0 exactly; no NaN
        out[j] = k[j] * p
    return out
 
 
def _newton_extents(conc_in, tau_s, temp, net, names, S, terms, xi0,
                    tol=1e-13, itmax=60):
    """Projected damped Newton on the square system F(xi) = xi - tau*r(C(xi)).
 
    Three equations, three unknowns, analytic Jacobian. least_squares was
    spending ~44 trust-region iterations and an SVD apiece on this, which is
    how a 3x3 root-find ended up dominating a process-scale optimizer. Newton
    gets it in a handful of 3x3 linear solves.
 
    Non-negativity is handled by projection (xi >= 0) rather than by a bounded
    solver: the extents are physically non-negative and the projection is
    exact on this box. Returns (xi, converged); a False sends the caller to
    the least_squares fallback rather than letting a bad root through.
    """
    n = len(net)
    eye = np.eye(n)
    xi = np.maximum(np.asarray(xi0, dtype=float), 0.0)
 
    def F_of(x):
        c = conc_in + S.T @ x
        return x - tau_s * rates_at(c, temp, net, names, terms=terms), c
 
    F, c = F_of(xi)
    for _ in range(itmax):
        nrm = np.max(np.abs(F))
        if nrm < tol:
            return xi, True
        drdc = rates_jac(c, temp, net, names, terms=terms)
        J = eye - tau_s * (drdc @ S.T)
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return xi, False
        if not np.all(np.isfinite(step)):
            return xi, False
        # backtrack until the residual actually decreases -- Newton alone will
        # happily overshoot into a region where a fractional power sends the
        # rate somewhere useless.
        lam, ok = 1.0, False
        for _ in range(40):
            xi_try = np.maximum(xi + lam * step, 0.0)
            F_try, c_try = F_of(xi_try)
            if np.max(np.abs(F_try)) < nrm:
                xi, F, c, ok = xi_try, F_try, c_try, True
                break
            lam *= 0.5
        if not ok:
            # Can't decrease. If we're already at a decent residual this is
            # the projection pinning us on the boundary, which is a genuine
            # solution; otherwise give up and let the fallback try.
            return xi, nrm < 1e-9 * max(1.0, tau_s)
    return xi, np.max(np.abs(F)) < 1e-9 * max(1.0, tau_s)
 
 
def solve_stage_ss(conc_in, tau_s, temp, net, names, S=None, terms=None,
                   xi0=None):
    """Steady state of ONE CSTR. Returns (conc_out, rates, ok).
 
    Fast path is a projected damped Newton; least_squares is kept as a
    fallback for the cases Newton gives up on. Both are checked against the
    same residual and non-negativity criteria, so the fallback cannot smuggle
    in a worse answer than the fast path would have.
    """
    S = _stoich_matrix(net, names) if S is None else S
    terms = _order_terms(net, names) if terms is None else terms
    conc_in = np.asarray(conc_in, dtype=float)
    n_r = len(net)
 
    def conc_of(xi):
        return conc_in + S.T @ xi
 
    def resid(xi):
        return xi - tau_s * rates_at(conc_of(xi), temp, net, names,
                                     terms=terms)
 
    # xi0 = 0 exactly, NOT 1e-8. When a reactant is exhausted the true extent
    # IS zero, and seeding at 1e-8 made least_squares report `gtol` success
    # while sitting on the seed -- the same false-convergence-at-seed failure
    # that froze fit_thermal_params.py. The seed then leaked back through
    # C = C_in + S.T @ xi as a -3e-8 concentration and every downstream stage
    # got flagged infeasible. gtol is disabled in the fallback for the same
    # reason.
    if xi0 is None:
        xi0 = np.zeros(n_r)
    xi0 = np.maximum(np.asarray(xi0, dtype=float), 0.0)
 
    scale = max(1.0, float(np.max(np.abs(conc_in))))
 
    def accept(xi):
        c = conc_of(xi)
        return (np.all(np.isfinite(c)) and np.all(c > -1e-6 * scale)
                and np.max(np.abs(resid(xi))) < 1e-9 * max(1.0, tau_s))
 
    xi, converged = _newton_extents(conc_in, tau_s, temp, net, names, S,
                                    terms, xi0)
    if not (converged and accept(xi)):
        def jac(x):
            return np.eye(n_r) - tau_s * (
                rates_jac(conc_of(x), temp, net, names, terms=terms) @ S.T)
        sol = least_squares(resid, xi0, jac=jac,
                            bounds=(np.zeros(n_r), np.full(n_r, np.inf)),
                            xtol=1e-12, ftol=1e-12, gtol=None, max_nfev=200)
        if accept(sol.x) or not accept(xi):
            xi = sol.x
 
    conc_out = np.maximum(conc_of(xi), 0.0)
    ok = accept(xi)
    return conc_out, rates_at(conc_out, temp, net, names, terms=terms), ok
 
 
def solve_train_ss(carrier, sub_stream, phi, splits, taus, temps, net, names):
    """Steady state of the whole N-CSTR train with distributed substrate feed.
 
    Volumetric basis is normalized: total feed = 1. The carrier enters stage 1;
    a fraction splits[i] of the substrate stream is injected into stage i.
    Stage i therefore runs at throughput q_i = (1-phi) + phi*sum(splits[:i+1]),
    and its inlet is the volume-weighted blend of the previous outlet with that
    side injection.
 
    Everything here is independent of the ABSOLUTE flow rate -- concentrations
    depend only on tau, not on Q. That is what lets section 5 back-solve Q from
    the 1500 kg/day target in one shot with no iteration.
 
    Returns per-stage concentrations, rates, and the normalized throughput
    q_i of each stage.
    """
    S, terms = _stoich_matrix(net, names), _order_terms(net, names)
    splits = np.asarray(splits, dtype=float)
 
    q = 1.0 - phi
    conc = np.asarray(carrier, dtype=float).copy()
    concs, rates, qs, ok_all, xi = [], [], [], True, None
 
    for i, (tau, T) in enumerate(zip(taus, temps)):
        q_side = phi * splits[i]
        q_new = q + q_side
        if q_new <= 0:
            return dict(ok=False, conc=np.zeros((len(taus), len(names))),
                        rates=np.zeros((len(taus), len(net))),
                        q_stage=np.ones(len(taus)))
        conc_mix = (q * conc + q_side * np.asarray(sub_stream)) / q_new
 
        conc, r, ok = solve_stage_ss(conc_mix, tau, T, net, names, S, terms, xi)
        xi = tau * r
        q = q_new
        concs.append(conc.copy()); rates.append(r.copy()); qs.append(q)
        ok_all = ok_all and ok
 
    return dict(conc=np.array(concs), rates=np.array(rates),
                q_stage=np.array(qs), ok=ok_all)
 
 
# =============================================================================
# 5.  SCALE-UP:  1500 kg/day  ->  Q  ->  per-stage volumes
# =============================================================================
# The key simplification: for FIXED (tau_i, T_i), outlet composition does NOT
# depend on Q -- only on tau. So there is no iteration here. Solve the train
# once on a per-litre basis, read the outlet product concentration, then set
#     Q = (target mass rate) / (product mass concentration out)
#     V_i = tau_i * Q
# Q falls straight out. That is why the objective in section 8 can be total
# volume with production held exactly at target by construction.
 
def target_mass_rate_kg_s():
    return PRODUCT_KG_PER_DAY / 86400.0 / max(ONSTREAM_FRACTION, 1e-9)
 
 
def rates_jac(conc, temp, net, names, terms=None, rates=None):
    """d(rate_j)/d(conc_i), shape (n_rxn, n_species). Analytic.
 
    For a power-law rate r_j = k_j * prod_i C_i^a_ji,
        dr_j/dC_i = r_j * a_ji / C_i
    which is exact and costs nothing once r_j is in hand.
 
    Supplying this to least_squares removes its finite-difference Jacobian,
    which profiling showed was ~84% of the entire optimizer runtime: SLSQP
    was finite-differencing an outer problem whose every evaluation contained
    an inner least_squares that was ALSO finite-differencing a 3x3 system.
 
    At C_i = 0 with a fractional exponent the true derivative is +inf. We
    return 0 there instead. That is a deliberate regularization: r_j is
    already 0 at that point, the extent is pinned at its lower bound, and
    handing the solver an infinite slope on the boundary is how you get the
    NaN/freeze behaviour rather than a converged answer.
    """
    terms = _order_terms(net, names) if terms is None else terms
    c = np.clip(np.asarray(conc, dtype=float), 0.0, None)
    r = rates_at(c, temp, net, names, terms=terms) if rates is None else rates
    J = np.zeros((len(net), len(names)))
    for j, tj in enumerate(terms):
        if r[j] == 0.0:
            continue
        for i, a in tj:
            if c[i] > 0.0:
                J[j, i] = r[j] * a / c[i]
    return J
 
 
def product_mass_conc(conc_out, names, basis=None):
    """Product mass concentration [kg/m3] of the train outlet.
    Nitramines (linear side products) are never product."""
    basis = PRODUCT_BASIS if basis is None else basis
    species = ['RDX'] if basis == 'RDX' else ['RDX', 'HMX']
    # mol/L * g/mol = g/L = kg/m3
    return sum(conc_out[names.index(s)] * MW_G_MOL[s] for s in species)
 
 
def size_train(carrier, sub_stream, phi, splits, taus, temps, net, names,
               substrate='Hexamine'):
    """Full process-scale sizing of one candidate train.
 
    Because outlet composition is independent of absolute flow, Q falls out in
    one step:  Q_total = (target mass rate) / (product mass conc out), then
    V_i = tau_i * q_i * Q_total. No iteration, no fixed point.
    """
    taus = np.asarray(taus, dtype=float)
    temps = np.asarray(temps, dtype=float)
    splits = np.asarray(splits, dtype=float)
 
    ss = solve_train_ss(carrier, sub_stream, phi, splits, taus, temps, net,
                        names)
    if not ss['ok']:
        return dict(feasible=False, reason='stage solver failed',
                    vol_total=np.inf, ss=ss)
    conc_out = ss['conc'][-1]
 
    cp_mass = product_mass_conc(conc_out, names)          # kg/m3
    if cp_mass < 1e-9:
        return dict(feasible=False, reason='no product', vol_total=np.inf,
                    ss=ss)
 
    q_total = target_mass_rate_kg_s() / cp_mass           # m3/s, train outlet
    q_stage = ss['q_stage'] * q_total                     # m3/s per stage
    vols = taus * q_stage                                 # m3
    areas = np.array([area_ht_m2(v) for v in vols])       # m2
 
    # Substrate conversion is measured against everything FED, not against the
    # stage-1 inlet -- with a distributed feed those are different numbers and
    # conflating them silently inflates conversion.
    i_sub = names.index(substrate)
    sub_fed = phi * sub_stream[i_sub] + (1.0 - phi) * carrier[i_sub]
    sub_out = conc_out[i_sub] * ss['q_stage'][-1]
    conv = (sub_fed - sub_out) / sub_fed if sub_fed > 0 else np.nan
 
    i_r, i_h, i_s = (names.index(x) for x in ('RDX', 'HMX', 'Nitramines'))
    tot = conc_out[i_r] + conc_out[i_h] + conc_out[i_s]
    sel = ((conc_out[i_r] / tot, conc_out[i_h] / tot, conc_out[i_s] / tot)
           if tot > 1e-12 else (0.0, 0.0, 0.0))
 
    # Per-stage inlet: blend of previous outlet with that stage's side feed.
    conc_in_stages, q_prev, c_prev = [], (1.0 - phi), np.asarray(carrier)
    for i in range(len(taus)):
        q_side = phi * splits[i]
        q_new = q_prev + q_side
        conc_in_stages.append((q_prev * c_prev + q_side * sub_stream) / q_new)
        q_prev, c_prev = q_new, ss['conc'][i]
 
    temps_in = np.concatenate(([T_FEED_C + 273.15], temps[:-1]))
    thermal = [stage_thermal(conc_in_stages[i], ss['conc'][i], ss['rates'][i],
                             taus[i], temps[i], temps_in[i], vols[i], areas[i],
                             q_stage[i], net, names)
               for i in range(len(taus))]
 
    q_tot_w = sum(t['q_gen_w'] for t in thermal)
    for t in thermal:
        t['q_frac'] = (t['q_gen_w'] / q_tot_w) if q_tot_w > 1e-9 else 0.0
 
    return dict(feasible=True, ok=ss['ok'], ss=ss, conc_out=conc_out,
                q_vol=q_total, q_stage=q_stage, splits=splits, phi=phi,
                taus=taus, temps=temps, vols=vols, areas=areas,
                vol_total=float(vols.sum()), area_total=float(areas.sum()),
                conversion=float(conv), selectivity=sel,
                prod_mass_conc=cp_mass, thermal=thermal,
                q_gen_total=float(q_tot_w),
                q_frac_max=float(max(t['q_frac'] for t in thermal)),
                u_ht_req_max=float(max(t['u_ht_req'] for t in thermal)),
                vol_max=float(vols.max()))
 
 
# =============================================================================
# 6.  THERMAL:  geometry, duty, and the Ug vs Ur criterion
# =============================================================================
 
def area_ht_m2(vol_m3, vol_offset=VOL_OFFSET):
    """Jacket heat-transfer area [m2] for a liquid volume vol_m3.
    Mirrors Reactors.py exactly (_BaseReactor.heat_transfer / CSTR.solve_unit):
        vol_tank = vol / vol_offset ; diam = (4/pi * vol_tank)^(1/3)
        area = 4/diam * vol + pi/4 * diam^2
    Note area ~ V^(2/3) while heat release ~ V -- this ratio IS the N-CSTR
    argument."""
    vol_tank = vol_m3 / vol_offset
    diam = (4.0 / np.pi * vol_tank) ** (1.0 / 3.0)
    return 4.0 / diam * vol_m3 + np.pi / 4.0 * diam ** 2
 
 
def q_gen_w(rates, vol_m3, net):
    """Instantaneous heat generation [W]: V_L * sum_j (-DH_j) * r_j."""
    dh = np.array([r['dh'] for r in net])
    return float(-(dh * np.asarray(rates)).sum() * vol_m3 * 1000.0)
 
 
def ug_manifold(conc_out, rates, tau_s, temp, net, names, terms=None, S=None):
    """dQ_gen/dT ALONG THE STEADY-STATE MANIFOLD, analytically. Returns W/K.
 
    The manifold constraint is  xi = tau * r(C_in + S.T @ xi). Differentiating
    it with respect to T at fixed (C_in, tau):
 
        dxi/dT = tau * [ dr/dT|_C + (dr/dC) S.T dxi/dT ]
        =>  ( I - tau (dr/dC) S.T ) dxi/dT = tau dr/dT|_C
                ^^^^^^^^^^^^^^^^^^
                this is exactly the residual Jacobian solve_stage_ss already
                uses -- the stabilizing composition feedback comes for free.
 
    Then dC/dT = S.T dxi/dT and the TOTAL rate derivative is
 
        dr/dT|_manifold = dr/dT|_C + (dr/dC) dC/dT
 
    with the partial dr_j/dT|_C = r_j * Ea_j / (R T^2). SIGNED Ea does the
    right thing automatically: Ea<0 makes the partial negative, which is the
    anti-Arrhenius contribution that drags Ug down.
 
    This replaces a +/- dT re-solve of the whole mass balance (2 extra
    nonlinear solves per stage per evaluation, and the dominant cost in the
    optimizer). It is also strictly more accurate -- no dT truncation error,
    and no risk of the re-solve landing on a different branch.
    """
    terms = _order_terms(net, names) if terms is None else terms
    S = _stoich_matrix(net, names) if S is None else S
    r = np.asarray(rates, dtype=float)
    ea = np.array([rx['ea'] for rx in net])
    dh = np.array([rx['dh'] for rx in net])
 
    drdT_partial = r * ea / (R_GAS * temp ** 2)          # 1/L/s/K
    drdc = rates_jac(conc_out, temp, net, names, terms=terms, rates=r)
    J = np.eye(len(net)) - tau_s * (drdc @ S.T)
    try:
        dxi_dT = np.linalg.solve(J, tau_s * drdT_partial)
    except np.linalg.LinAlgError:
        # Singular J means the SS is at a turning point -- the manifold has no
        # single-valued slope there. Fall back to the frozen-composition
        # derivative, which is the conservative (larger) Ug.
        dxi_dT = np.zeros(len(net))
    dr_dT = drdT_partial + drdc @ (S.T @ dxi_dT)
    # q_gen_w convention: Q = -sum(dh_j * r_j) * vol * 1000
    return float(-(dh * dr_dT).sum() * 1000.0)
 
 
def stage_thermal(conc_in, conc_out, rates, tau_s, temp, temp_in, vol_m3,
                  area_m2, q_vol, net, names, dT=0.25):
    """Full thermal analysis of one stage at its steady state.
 
    Ug is computed ALONG THE STEADY-STATE MANIFOLD (see ug_manifold), so the
    stabilizing dC_ss/dT feedback is captured. The bare Arrhenius derivative
    at frozen composition would overstate Ug -- sometimes badly.
    """
    S, terms = _stoich_matrix(net, names), _order_terms(net, names)
 
    qgen = q_gen_w(rates, vol_m3, net)
    mdot = q_vol * RHO_MIX_KG_M3                        # kg/s
    flow_cap = mdot * CP_MIX_J_KG_K                     # W/K -- sensible sink
    q_flow = flow_cap * (temp - temp_in)                # W removed by feed heating
 
    # ---- duty: what the jacket must actually pull -----------------------
    q_jacket = qgen - q_flow                            # W (may be negative:
                                                        # cold feed alone can
                                                        # over-remove)
    dT_drive_max = temp - (T_COOLANT_MIN_C + 273.15)
    if q_jacket <= 0:
        u_req_duty = 0.0                                # no cooling needed
        t_coolant_req = np.nan
    elif dT_drive_max <= 0:
        u_req_duty = np.inf
        t_coolant_req = np.nan
    else:
        u_req_duty = q_jacket / (area_m2 * dT_drive_max)
        # coolant temperature the CURRENT u_ht would actually demand
        t_coolant_req = temp - q_jacket / (U_HT_AVAILABLE * area_m2)
 
    # ---- Ug: dQ_gen/dT along the steady-state manifold -------------------
    ug = vol_m3 * ug_manifold(conc_out, rates, tau_s, temp, net, names,
                              terms, S)                 # W/K
 
    # ---- Ur and the slope requirement ------------------------------------
    ur = flow_cap + U_HT_AVAILABLE * area_m2            # W/K
    if ug <= 0:
        # Anti-Arrhenius net: heat generation FALLS with T. Unconditionally
        # stable in the van Heerden sense -- no u_ht is required to satisfy
        # the slope criterion.
        u_req_slope = 0.0
    else:
        u_req_slope = max(0.0,
                          (STABILITY_MARGIN * ug - flow_cap) / area_m2)
 
    u_req = max(u_req_duty, u_req_slope)
    binds = ('duty' if u_req_duty >= u_req_slope else 'Ug<Ur')
    if not np.isfinite(u_req):
        binds = 'infeasible'
 
    return dict(temp=float(temp), temp_in=float(temp_in), tau_s=float(tau_s),
                vol_m3=float(vol_m3), area_m2=float(area_m2),
                q_gen_w=qgen, q_flow_w=float(q_flow), q_jacket_w=float(q_jacket),
                ug=float(ug), ur=float(ur),
                ur_over_ug=(float(ur / ug) if ug > 0 else np.inf),
                stable=(ug <= 0 or ur >= STABILITY_MARGIN * ug),
                u_ht_req_duty=float(u_req_duty),
                u_ht_req_slope=float(u_req_slope),
                u_ht_req=float(u_req), binds=binds,
                t_coolant_req_C=(float(t_coolant_req - 273.15)
                                 if np.isfinite(t_coolant_req) else np.nan),
                coolant_kg_s=(max(q_jacket, 0.0) / (4180.0 * COOLANT_DELTA_T_K)))
 
 
# =============================================================================
# 7.  N-SCAN:  how many CSTRs does the heat (and the inventory cap) demand?
# =============================================================================
 
def scan_n(n_max=8, substrate='Hexamine', use_nitric_acid_inline=False,
           tau_total_s=900.0, temp_C=60.0, split_mode=None, verbose=True):
    """Equal-tau-split scan at fixed total residence time and temperature.
 
    Read the q_frac column before anything else. It is the fraction of the
    total exotherm landing in the WORST stage. If q_frac stays near 1.0 as N
    grows, the train is not splitting the heat at all -- it is just building
    smaller vessels around the same fireball, and u_req will get WORSE with N,
    not better. That is the expected result for split_mode='all_stage1' under
    these kinetics; compare against 'equal' to see the difference.
    """
    net = build_network(substrate, use_nitric_acid_inline)
    carrier, sub, phi, names = build_feed_streams(substrate,
                                                  use_nitric_acid_inline)
    rows = []
    for N in range(1, n_max + 1):
        splits = resolve_splits(N, split_mode)
        taus = np.full(N, tau_total_s / N)
        temps = np.full(N, temp_C + 273.15)
        sized = size_train(carrier, sub, phi, splits, taus, temps, net, names,
                           substrate)
        if not sized['feasible']:
            rows.append(dict(N=N, feasible=False,
                             reason=sized.get('reason', '?')))
            continue
        u_req = sized['u_ht_req_max']
        rows.append(dict(
            N=N, feasible=True, vol_total=sized['vol_total'],
            vol_max=sized['vol_max'], area_total=sized['area_total'],
            q_gen_total=sized['q_gen_total'], q_frac_max=sized['q_frac_max'],
            u_ht_req=u_req, conv=sized['conversion'],
            sel_rdx=sized['selectivity'][0],
            ok_u=(u_req <= U_HT_AVAILABLE),
            ok_u_ceiling=(u_req <= U_HT_CEILING),
            ok_inv=(sized['vol_max'] <= MAX_VESSEL_VOL_M3), sized=sized))
 
    if verbose:
        mode = split_mode or FEED_SPLIT_MODE
        print(f"\nN-SCAN  ({substrate}, "
              f"{'in-line' if use_nitric_acid_inline else 'pre-formed'} "
              f"AcONO2, equal tau split, tau_tot={tau_total_s/60:.0f} min, "
              f"T={temp_C:.0f}C)")
        print(f"  substrate feed split = '{mode}'  |  target "
              f"{PRODUCT_KG_PER_DAY:.0f} kg/day {PRODUCT_BASIS}  |  "
              f"STOICH_MODE={STOICH_MODE}")
        print("  " + "-" * 100)
        print(f"  {'N':>2} {'V_tot[L]':>9} {'V_max[L]':>9} {'A_tot[m2]':>10} "
              f"{'Qgen[kW]':>9} {'q_frac':>7} {'u_req':>8} {'u<=500':>7} "
              f"{'u<=3000':>8} {'V<=cap':>7} {'X_sub':>7} {'RDX%':>6}")
        print("  " + "-" * 100)
        for r in rows:
            if not r['feasible']:
                print(f"  {r['N']:>2}  -- infeasible ({r.get('reason','?')})")
                continue
            print(f"  {r['N']:>2} {r['vol_total']*1000:>9.1f} "
                  f"{r['vol_max']*1000:>9.1f} {r['area_total']:>10.2f} "
                  f"{r['q_gen_total']/1000:>9.1f} {r['q_frac_max']:>7.2f} "
                  f"{r['u_ht_req']:>8.0f} {'yes' if r['ok_u'] else 'NO':>7} "
                  f"{'yes' if r['ok_u_ceiling'] else 'NO':>8} "
                  f"{'yes' if r['ok_inv'] else 'NO':>7} "
                  f"{r['conv']*100:>6.1f}% {r['sel_rdx']*100:>5.1f}%")
        print("  " + "-" * 100)
        n_th = _first_true(rows, 'ok_u')
        n_thc = _first_true(rows, 'ok_u_ceiling')
        n_inv = _first_true(rows, 'ok_inv')
        print(f"  Minimum N on thermal duty @ u_ht={U_HT_AVAILABLE:.0f}: "
              f"{n_th or 'not reached'}")
        print(f"  Minimum N on thermal duty @ u_ht={U_HT_CEILING:.0f} "
              f"(ceiling): {n_thc or 'not reached'}")
        print(f"  Minimum N on inventory cap "
              f"({MAX_VESSEL_VOL_M3*1000:.0f} L/vessel): {n_inv or 'not reached'}")
        if n_th and n_inv:
            print(f"  ==> at FIXED tau_tot and T, N >= {max(n_th, n_inv)} "
                  f"(binding: "
                  f"{'inventory' if n_inv >= n_th else 'heat'})")
        print("  NOT a design. tau and T are frozen here and the split is even,")
        print("  so this OVERSTATES the N you need -- section 4/8 optimizes both")
        print("  and clears the same constraints at lower N. Read this table for")
        print("  WHERE THE HEAT GOES; take N from the inventory scan.")
        feas = [r for r in rows if r['feasible']]
        if feas and min(r['q_frac_max'] for r in feas) > 0.9:
            print("\n  *** The worst stage carries >90% of the exotherm at EVERY N. ***")
            print("  Splitting residence time is not splitting the heat -- the")
            print("  substrate is consumed in stage 1 no matter how you slice tau,")
            print("  so all N buys you is a smaller vessel around the same duty.")
            print("  Re-run with split_mode='equal' to distribute the SUBSTRATE")
            print("  feed instead. That is the variable that moves the heat.")
    return rows
 
 
def _first_true(rows, key):
    for r in rows:
        if r.get('feasible') and r.get(key):
            return r['N']
    return None
 
 
# =============================================================================
# 8.  PER-STAGE OPTIMIZATION  (T_i, tau_i)  --  stages need not be identical
# =============================================================================
# Decision vector x = [T_1..T_N, tau_1..tau_N].
#
# Objective: TOTAL REACTOR VOLUME required to make 1500 kg/day. Because
# production is pinned to target by construction (section 5), V_total =
# n_target * sum(tau_i) / C_product_out -- so minimizing it is exactly
# minimizing "reactor volume per unit product", which correctly trades
# residence time against selectivity and conversion. No arbitrary weights.
#
# Constraints (all inequality, >= 0):
#   g1_i : U_HT_AVAILABLE - u_ht_req_i          per stage, thermal (duty+Ug<Ur)
#   g2_i : MAX_VESSEL_VOL_M3 - V_i              per stage, inventory cap
#   g3   : conversion - X_HEXAMINE_MIN          train outlet
#
# SLSQP with numeric gradients, multistart. Direct scipy, not PharmaPy's
# ParameterEstimation -- that tooling fits kinetic parameters from
# concentration data and has no notion of a design constraint.
 
def _unpack(x, N):
    return np.asarray(x[:N], dtype=float), np.asarray(x[N:], dtype=float)
 
 
def _t_bounds(use_nitric_acid_inline):
    hi = T_MAX_C
    if ENFORCE_ACONO2_CAP and not use_nitric_acid_inline:
        hi = min(hi, ACONO2_T_CAP_C)
    return T_MIN_C + 273.15, hi + 273.15
 
 
def optimize_train(N, substrate='Hexamine', use_nitric_acid_inline=False,
                   split_mode=None, n_starts=6, verbose=True, seed=0,
                   maxiter=150):
    """Optimize (T_i, tau_i[, split_i]) for an N-stage train.
 
    Decision vector x = [T_1..T_N, tau_1..tau_N] and, when split_mode is
    'optimized', a further [w_1..w_N] of unnormalized substrate-split weights.
    The weights are normalized internally rather than carrying a sum-to-one
    equality constraint -- SLSQP handles the reduced problem far better, and a
    scale-invariant objective doesn't care about the redundant degree of
    freedom.
    """
    split_mode = FEED_SPLIT_MODE if split_mode is None else split_mode
    net = build_network(substrate, use_nitric_acid_inline)
    carrier, sub, phi, names = build_feed_streams(substrate,
                                                  use_nitric_acid_inline)
    t_lo, t_hi = _t_bounds(use_nitric_acid_inline)
    fit_splits = (split_mode == 'optimized') and N > 1
    nvar = 3 * N if fit_splits else 2 * N
 
    cache = {}
 
    def evaluate(x):
        key = tuple(np.round(x, 8))
        if key not in cache:
            temps, taus = np.asarray(x[:N]), np.asarray(x[N:2 * N])
            splits = (resolve_splits(N, weights=x[2 * N:3 * N]) if fit_splits
                      else resolve_splits(N, split_mode))
            cache[key] = size_train(carrier, sub, phi, splits, taus, temps,
                                    net, names, substrate)
            if len(cache) > 30000:
                cache.clear()
        return cache[key]
 
    def objective(x):
        sz = evaluate(x)
        return 1e6 if not sz['feasible'] else sz['vol_total'] * 1000.0  # litres
 
    def con_thermal(x):
        sz = evaluate(x)
        if not sz['feasible']:
            return -np.ones(N)
        return np.array([U_HT_AVAILABLE - t['u_ht_req'] for t in sz['thermal']])
 
    def con_inventory(x):
        sz = evaluate(x)
        return -np.ones(N) if not sz['feasible'] else \
            MAX_VESSEL_VOL_M3 - sz['vols']
 
    def con_conversion(x):
        sz = evaluate(x)
        return np.array([-1.0]) if not sz['feasible'] else \
            np.array([sz['conversion'] - X_HEXAMINE_MIN])
 
    cons = [dict(type='ineq', fun=con_thermal),
            dict(type='ineq', fun=con_inventory),
            dict(type='ineq', fun=con_conversion)]
    bounds = [(t_lo, t_hi)] * N + [(TAU_MIN_S, TAU_MAX_S)] * N
    if fit_splits:
        bounds += [(1e-3, 1.0)] * N
 
    rng = np.random.default_rng(seed)
    starts = []
    for tau_tot in (900.0, 300.0, 1800.0):
        # Physically-motivated seed: cold and short up front (stage 1 sees the
        # highest substrate concentration and therefore the highest volumetric
        # heat release), ramping warmer and longer downstream.
        temps0 = (np.linspace(t_lo, min(t_hi, t_lo + 15.0), N) if N > 1
                  else np.array([0.5 * (t_lo + t_hi)]))
        w = np.linspace(0.6, 1.4, N) if N > 1 else np.array([1.0])
        taus0 = np.clip(w / w.sum() * tau_tot, TAU_MIN_S, TAU_MAX_S)
        x0 = np.concatenate([temps0, taus0])
        if fit_splits:
            x0 = np.concatenate([x0, np.full(N, 1.0 / N)])
        starts.append(x0)
    while len(starts) < n_starts:
        x0 = np.concatenate([rng.uniform(t_lo, t_hi, N),
                             rng.uniform(TAU_MIN_S, min(TAU_MAX_S, 1800.0), N)])
        if fit_splits:
            x0 = np.concatenate([x0, rng.uniform(0.05, 1.0, N)])
        starts.append(x0)
    # n_starts is a CAP, not a floor. The physically-motivated seeds are
    # ordered best-first (mid-range tau, cold-to-hot ramp), so truncating
    # keeps the ones worth having.
    starts = starts[:max(1, n_starts)]
 
    best, best_f = None, np.inf
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options=dict(maxiter=maxiter, ftol=1e-6, eps=1e-4))
        except Exception:
            continue
        if not np.isfinite(res.fun) or not evaluate(res.x)['feasible']:
            continue
        viol = -min(np.min(con_thermal(res.x)) / max(U_HT_AVAILABLE, 1.0),
                    np.min(con_inventory(res.x)) / MAX_VESSEL_VOL_M3,
                    np.min(con_conversion(res.x)))
        if viol > 1e-4:
            continue
        if res.fun < best_f:
            best_f, best = res.fun, res
 
    if best is None:
        if verbose:
            print(f"\n  N={N}: NO FEASIBLE DESIGN.")
            print(f"    constraints: u_ht<={U_HT_AVAILABLE:.0f} W/m2/K, "
                  f"V<={MAX_VESSEL_VOL_M3*1000:.0f} L/vessel, "
                  f"X>={X_HEXAMINE_MIN:.2f}, T in "
                  f"[{t_lo-273.15:.0f},{t_hi-273.15:.0f}] C, split='{split_mode}'")
            print("    Relax one and re-run to find out which is binding -- "
                  "u_ht is\n    usually the honest culprit, and the honest "
                  "answer to it is a\n    different exchanger, not a bigger N.")
        return dict(N=N, success=False, sized=None)
 
    temps, taus = np.asarray(best.x[:N]), np.asarray(best.x[N:2 * N])
    splits = (resolve_splits(N, weights=best.x[2 * N:3 * N]) if fit_splits
              else resolve_splits(N, split_mode))
    sized = size_train(carrier, sub, phi, splits, taus, temps, net, names,
                       substrate)
    out = dict(N=N, success=True, x=best.x, temps=temps, taus=taus,
               splits=splits, sized=sized, names=names, net=net,
               carrier=carrier, sub_stream=sub, phi=phi, substrate=substrate,
               inline=use_nitric_acid_inline, split_mode=split_mode)
    if verbose:
        report_design(out)
    return out
 
 
# =============================================================================
# 9.  PHARMAPY LAYER:  build and solve the real N-CSTR train
# =============================================================================
# PharmaPy is imported HERE, lazily, so sections 1-8 stay dependency-light.
#
# Chaining: solve stage i, hand reactor.Outlet (a LiquidStream carrying the
# converged composition, temperature, and vol_flow) straight into stage i+1's
# .Inlet. Assignment order is .Phases -> .Kinetics -> .Inlet -> .Utility;
# .Phases must precede .Kinetics.
 
def _import_pharmapy():
    from PharmaPy.Kinetics import RxnKinetics
    from PharmaPy.Reactors import CSTR
    from PharmaPy.Phases import LiquidPhase
    from PharmaPy.Streams import LiquidStream
    from PharmaPy.Utilities import CoolingWater
    return RxnKinetics, CSTR, LiquidPhase, LiquidStream, CoolingWater
 
 
def build_kinetics(net, db_path=DB_PATH, temp_ref=None):
    """RxnKinetics for the network. A FRESH object per stage -- RxnKinetics
    mutates internal state in set_params, so sharing one across stages is
    asking for a silent cross-contamination bug.
 
    temp_ref is FIXED at T_REF_KIN_C for every stage regardless of that
    stage's operating temperature. If temp_ref tracked T_op then
    inv_temp = 1/temp_ref - 1/T = 0 identically and every Ea -- including the
    negative ones this whole design turns on -- would drop out of the model.
    """
    RxnKinetics = _import_pharmapy()[0]
    tref = (T_REF_KIN_C + 273.15) if temp_ref is None else temp_ref
    return RxnKinetics(
        path       = db_path,
        rxn_list   = [_rxn_string(r['nu']) for r in net],
        k_params   = [r['k0'] for r in net],
        ea_params  = [r['ea'] for r in net],      # SIGNED: +/-/-
        delta_hrxn = [r['dh'] for r in net],
        tref_hrxn  = tref,
        temp_ref   = tref,
        params_f   = build_params_f(net, db_path),   # list of lists, DB order
    )
 
 
def _blend_stream(LiquidStream, db_path, conc_a, q_a, temp_a,
                  conc_b, q_b, temp_b):
    """Volume-weighted blend of two liquid streams into one LiquidStream.
    Stands in for a MIX unit at each substrate injection point.
 
    The blend temperature is a flow-weighted mean, i.e. it assumes both
    streams share a heat capacity -- fine here (both AcOH-dominated), not fine
    in general. It also ignores heat of mixing, which for hexamine dissolving
    into an AcONO2/AcOH carrier is not obviously negligible and is worth a
    calorimetry point of its own.
    """
    q = q_a + q_b
    conc = (q_a * np.asarray(conc_a) + q_b * np.asarray(conc_b)) / q
    temp = (q_a * temp_a + q_b * temp_b) / q
    return LiquidStream(path_thermo=db_path, temp=temp, mole_conc=conc,
                        vol_flow=q)
 
 
def build_pharmapy_train(design, isothermal=True, db_path=DB_PATH):
    """Build the N-CSTR train as real PharmaPy CSTR objects from a design dict
    (the output of optimize_train). Returns (reactors, side_streams,
    carrier_stream), all unsolved -- wiring happens in run_pharmapy_train,
    because stage i's Inlet doesn't exist until stage i-1 has been solved and
    produced an Outlet."""
    _, CSTR, LiquidPhase, LiquidStream, CoolingWater = _import_pharmapy()
 
    net = design['net']
    temps, splits = design['temps'], design['splits']
    sz = design['sized']
    q_total, phi = sz['q_vol'], design['phi']
    t_feed = T_FEED_C + 273.15
 
    # Initial charge = neat AcOH, NOT the feed at 50%. Charging a vessel with
    # reactants produces an instantaneous t=0 exotherm that has never once
    # appeared in an EasyMax run -- in reality you start on solvent and ramp
    # feed in. Same fix that killed the phantom t=0 blowup.
    acoh_init, _ = build_conc_array({'Acetic_Acid': ACOH_PURE_CONC}, db_path)
 
    carrier_stream = LiquidStream(path_thermo=db_path, temp=t_feed,
                                  mole_conc=design['carrier'],
                                  vol_flow=(1.0 - phi) * q_total)
    side_streams = []
    for i in range(design['N']):
        q_side = phi * splits[i] * q_total
        side_streams.append(
            None if q_side <= 1e-15 else
            LiquidStream(path_thermo=db_path, temp=t_feed,
                         mole_conc=design['sub_stream'], vol_flow=q_side))
 
    reactors = []
    for i in range(design['N']):
        th = sz['thermal'][i]
        # Jacket inlet from the section-6 design point. If a stage needs no net
        # cooling (cold feed alone over-removes), the jacket must run WARM --
        # clamping it to the coolant supply would quench the stage below its
        # design T, and the non-isothermal run would then "disagree" with the
        # sizing for a reason that is pure artifact.
        t_jacket = (th['t_coolant_req_C'] + 273.15
                    if np.isfinite(th['t_coolant_req_C']) else temps[i])
        t_jacket = max(t_jacket, T_COOLANT_MIN_C + 273.15)
        m_cool = max(th['coolant_kg_s'], 1e-3)   # never 0: the jacket ODE
                                                 # divides by coolant holdup/flow
 
        reactor = CSTR(isothermal=isothermal, h_conv=H_CONV_REACTOR,
                       ht_mode='jacket', reset_states=True)
        # .Phases BEFORE .Kinetics -- assignment order is load-bearing.
        reactor.Phases = LiquidPhase(path_thermo=db_path, temp=temps[i],
                                     mole_conc=acoh_init, vol=sz['vols'][i])
        reactor.Kinetics = build_kinetics(net, db_path)   # fresh object/stage
        reactor.Utility = CoolingWater(mass_flow=m_cool, temp_in=t_jacket,
                                       h_conv=H_CONV_JACKET)
        # The Utility setter computes u_ht = 1/(1/h_conv + 1/utility.h_conv) at
        # assignment time. Touch h_conv afterwards and u_ht goes stale silently
        # -- recompute explicitly, same as fit_thermal_params.py has to.
        reactor.u_ht = 1.0 / (1.0 / reactor.h_conv +
                              1.0 / reactor.Utility.h_conv)
        reactors.append(reactor)
    return reactors, side_streams, carrier_stream
 
 
def run_pharmapy_train(design, isothermal=True, n_tau=12.0, db_path=DB_PATH,
                       verbose=False):
    """Solve the train stage by stage, threading reactor.Outlet (blended with
    that stage's substrate side-feed) into the next stage's .Inlet.
 
    RUN ISOTHERMAL FIRST. It isolates stoichiometry and the mass balance from
    the energy balance. If the isothermal train doesn't reproduce the
    section-5 algebraic sizing, a non-isothermal run only stacks a second
    unknown on top of the first.
    """
    _, _, _, LiquidStream, _ = _import_pharmapy()
    reactors, side_streams, carrier = build_pharmapy_train(design, isothermal,
                                                           db_path)
    names = design['names']
    out = []
    inlet, q_prev, t_prev = carrier, carrier.vol_flow, T_FEED_C + 273.15
    conc_prev = np.asarray(design['carrier'])
 
    for i, reactor in enumerate(reactors):
        side = side_streams[i]
        if side is not None:
            inlet = _blend_stream(LiquidStream, db_path, conc_prev, q_prev,
                                  t_prev, design['sub_stream'], side.vol_flow,
                                  T_FEED_C + 273.15)
            q_prev = q_prev + side.vol_flow
        reactor.Inlet = inlet
 
        try:
            time, states = reactor.solve_unit(
                runtime=n_tau * float(design['taus'][i]), verbose=verbose,
                sundials_opts={'rtol': 1e-6, 'atol': 1e-8, 'maxsteps': 50000})
        except Exception as e:
            print(f"  stage {i+1}: solve FAILED -- {e}")
            out.append(dict(stage=i + 1, ok=False, error=str(e)))
            break
 
        res = reactor.result
        temp_prof = np.atleast_1d(np.asarray(res.temp, dtype=float))
        peak_T = float(np.nanmax(temp_prof))
        # A runaway can produce finite-but-astronomical temperatures that sail
        # straight through np.isfinite(). Check against a PHYSICAL ceiling --
        # the same trap that froze the thermal optimizer at its seed.
        blew_up = (not np.all(np.isfinite(temp_prof))) or peak_T > 500.0
 
        conc_out = np.asarray(states)[-1][:len(names)]
        out.append(dict(stage=i + 1, ok=not blew_up, reactor=reactor,
                        time=np.asarray(time), states=np.asarray(states),
                        temp=temp_prof, peak_T_C=peak_T - 273.15,
                        design_T_C=float(design['temps'][i] - 273.15),
                        q_rxn=np.asarray(res.q_rxn),
                        q_ht=np.asarray(res.q_ht), conc_out=conc_out))
        if blew_up:
            print(f"  stage {i+1}: THERMAL RUNAWAY -- peak T = "
                  f"{peak_T - 273.15:.0f} C. Truncating here rather than "
                  f"propagating\n    a fictional composition downstream.")
            break
 
        inlet = reactor.Outlet
        conc_prev, t_prev = conc_out, float(temp_prof[-1])
    return out
 
 
def report_pharmapy_run(results, design, isothermal):
    names = design['names']
    tag = 'ISOTHERMAL' if isothermal else 'NON-ISOTHERMAL'
    print(f"\n  PharmaPy {tag} train -- {design['N']} stages")
    print("  " + "-" * 76)
    print(f"  {'stg':>3} {'T_des[C]':>9} {'T_peak[C]':>10} {'q_rxn[kW]':>10} "
          f"{'q_ht[kW]':>9} {'RDX[M]':>8} {'HMX[M]':>8} {'ok':>4}")
    print("  " + "-" * 76)
    for r in results:
        if not r.get('ok') and 'error' in r:
            print(f"  {r['stage']:>3}  solve error")
            continue
        c = r['conc_out']
        print(f"  {r['stage']:>3} {r['design_T_C']:>9.1f} {r['peak_T_C']:>10.1f} "
              f"{np.abs(r['q_rxn']).max()/1000:>10.1f} "
              f"{np.abs(r['q_ht']).max()/1000:>9.1f} "
              f"{c[names.index('RDX')]:>8.4f} {c[names.index('HMX')]:>8.4f} "
              f"{'yes' if r['ok'] else 'NO':>4}")
    print("  " + "-" * 76)
    if results and results[-1].get('ok'):
        c = results[-1]['conc_out']
        mass = product_mass_conc(c, names)
        rate = mass * design['sized']['q_vol'] * 86400.0
        print(f"  Train outlet {PRODUCT_BASIS}: {mass:.1f} kg/m3 "
              f"-> {rate:.0f} kg/day  (target {PRODUCT_KG_PER_DAY:.0f})")
        err = abs(rate - PRODUCT_KG_PER_DAY) / PRODUCT_KG_PER_DAY * 100
        print(f"  Deviation from the section-5 algebraic sizing: {err:.1f}%")
        if err > 5:
            print("  >5% means the fast algebraic layer and PharmaPy disagree. "
                  "Trust\n  PharmaPy and find out why before going further.")
 
 
# =============================================================================
# 10.  REPORTING
# =============================================================================
 
def report_design(design):
    sz = design['sized']
    if not sz or not sz.get('feasible'):
        print("  infeasible design"); return
    print("\n" + "=" * 96)
    print(f"PROCESS-SCALE R02 TRAIN  --  N = {design['N']}  |  "
          f"{design['substrate']}  |  "
          f"{'in-line' if design['inline'] else 'pre-formed'} AcONO2  |  "
          f"split='{design.get('split_mode', FEED_SPLIT_MODE)}'")
    print(f"target {PRODUCT_KG_PER_DAY:.0f} kg/day {PRODUCT_BASIS}  |  "
          f"STOICH_MODE={STOICH_MODE}  |  u_ht={U_HT_AVAILABLE:.0f} W/m2/K  |  "
          f"Ea = +/-/-")
    print("=" * 96)
    print(f"  Total feed Q         : {sz['q_vol']*3600:.3f} m3/h "
          f"({sz['q_vol']*1e6:.1f} mL/s)")
    print(f"  Total reactor volume : {sz['vol_total']*1000:.1f} L "
          f"(largest vessel {sz['vol_max']*1000:.1f} L / cap "
          f"{MAX_VESSEL_VOL_M3*1000:.0f} L)")
    print(f"  Total jacket area    : {sz['area_total']:.2f} m2")
    print(f"  Total tau            : {sz['taus'].sum()/60:.2f} min")
    print(f"  Substrate conversion : {sz['conversion']*100:.1f}%  "
          f"(on TOTAL substrate fed, not stage-1 inlet)")
    print(f"  Selectivity          : RDX {sz['selectivity'][0]*100:.1f}% | "
          f"HMX {sz['selectivity'][1]*100:.1f}% | "
          f"Side {sz['selectivity'][2]*100:.1f}%")
    print(f"  Total heat release   : {sz['q_gen_total']/1000:.1f} kW   "
          f"<-- set by production rate x DH. N cannot change this;")
    print(f"                         only where it lands and how much area "
          f"you have to take it out.")
    print("-" * 96)
    print(f"  {'stg':>3} {'T[C]':>6} {'tau[s]':>8} {'V[L]':>8} {'A[m2]':>7} "
          f"{'sub%':>6} {'Qgen[kW]':>9} {'q_frac':>7} {'Ug[W/K]':>9} "
          f"{'Ur[W/K]':>9} {'u_req':>7} {'binds':>7} {'Tc[C]':>7}")
    print("-" * 96)
    for i, t in enumerate(sz['thermal']):
        print(f"  {i+1:>3} {t['temp']-273.15:>6.1f} {t['tau_s']:>8.1f} "
              f"{t['vol_m3']*1000:>8.1f} {t['area_m2']:>7.3f} "
              f"{sz['splits'][i]*100:>5.1f}% {t['q_gen_w']/1000:>9.2f} "
              f"{t['q_frac']:>7.2f} {t['ug']:>9.1f} {t['ur']:>9.1f} "
              f"{t['u_ht_req']:>7.0f} {t['binds']:>7} "
              f"{t['t_coolant_req_C']:>7.1f}")
    print("-" * 96)
    print("  sub%  = share of the substrate stream injected into that stage")
    print("  q_frac= share of the total exotherm landing in that stage")
    print("  binds = which criterion sets u_req: 'duty' or 'Ug<Ur'")
    n_slope = sum(1 for t in sz['thermal'] if t['binds'] == 'Ug<Ur')
    print(f"\n  Stages where Ug<Ur binds: {n_slope} of {design['N']}")
    if n_slope == 0:
        print("  Every stage is DUTY-limited, not slope-limited. That is the")
        print("  expected signature of Ea = +/-/-: two of three pathways are")
        print("  anti-Arrhenius, so they drag Ug negative and the network is")
        print("  self-stabilizing in the van Heerden sense. It is NOT evidence")
        print("  the reactor is safe -- it means the binding question is 'can")
        print("  the jacket carry the duty', not 'will it run away'. And it")
        print("  rests entirely on the lumped DH: if the side pathway turns out")
        print("  more exothermic than RDX, this conclusion moves. Re-run when")
        print("  fit_thermal_params.py gives you pathway-resolved DH.")
    if any(not t['stable'] for t in sz['thermal']):
        print(f"  *** WARNING: a stage violates Ur >= {STABILITY_MARGIN}*Ug ***")
    if sz['q_frac_max'] > 0.9 and design['N'] > 1:
        print(f"  *** One stage carries {sz['q_frac_max']*100:.0f}% of the "
              f"exotherm. The train is not\n      sharing the load -- see "
              f"FEED_SPLIT_MODE. ***")
    if np.any(np.asarray(sz['taus']) <= TAU_MIN_S * 1.01):
        print(f"  *** A stage is pinned at TAU_MIN_S={TAU_MIN_S:.0f} s. The "
              f"optimum is on a\n      bound, not an interior optimum -- and "
              f"at that tau the ideal-CSTR\n      assumption is doing work it "
              f"hasn't earned. Treat as a signal about\n      the k "
              f"magnitudes, not as a design. ***")
    print("=" * 96)
 
 
def sweep_temperature(N=3, substrate='Hexamine', use_nitric_acid_inline=False,
                      temps_C=(45, 50, 55, 60, 65, 70, 75), tau_total_s=900.0,
                      split_mode=None):
    """Isothermal train swept across the operating temperature range, every
    stage at the same T. This is where the +/-/- Ea signature should show
    itself: RDX selectivity climbing monotonically with T while HMX and Side
    both fall. If it doesn't, the Ea signs aren't doing what you think."""
    net = build_network(substrate, use_nitric_acid_inline)
    carrier, sub, phi, names = build_feed_streams(substrate,
                                                  use_nitric_acid_inline)
    t_cap = (ACONO2_T_CAP_C if (ENFORCE_ACONO2_CAP and not use_nitric_acid_inline)
             else np.inf)
    print(f"\nTEMPERATURE SWEEP  (N={N}, equal tau split, "
          f"tau_tot={tau_total_s/60:.0f} min, split='{split_mode or FEED_SPLIT_MODE}')")
    print("  " + "-" * 88)
    print(f"  {'T[C]':>6} {'V_tot[L]':>9} {'Q[m3/h]':>9} {'X_sub':>7} "
          f"{'RDX%':>7} {'HMX%':>7} {'Side%':>7} {'Qgen[kW]':>9} "
          f"{'u_req':>7} {'note':>12}")
    print("  " + "-" * 88)
    rows = []
    for T_C in temps_C:
        splits = resolve_splits(N, split_mode)
        sz = size_train(carrier, sub, phi, splits, np.full(N, tau_total_s / N),
                        np.full(N, T_C + 273.15), net, names, substrate)
        if not sz['feasible']:
            print(f"  {T_C:>6.0f}   infeasible ({sz.get('reason','?')})"); continue
        note = 'AcONO2 cap!' if T_C > t_cap else ''
        print(f"  {T_C:>6.0f} {sz['vol_total']*1000:>9.1f} "
              f"{sz['q_vol']*3600:>9.3f} {sz['conversion']*100:>6.1f}% "
              f"{sz['selectivity'][0]*100:>6.1f}% "
              f"{sz['selectivity'][1]*100:>6.1f}% "
              f"{sz['selectivity'][2]*100:>6.1f}% "
              f"{sz['q_gen_total']/1000:>9.1f} {sz['u_ht_req_max']:>7.0f} "
              f"{note:>12}")
        rows.append(dict(T_C=T_C, sized=sz))
    print("  " + "-" * 88)
    print(f"  'AcONO2 cap!' = above the ~{ACONO2_T_CAP_C:.0f} C decomposition "
          f"onset (Andreozzi 2002).")
    print("  The AcONO2 decomposition exotherm is NOT in this network, so those")
    print("  rows understate the heat -- they are not a licence to run there.")
    return rows
 
 
# =============================================================================
# 11.  ENTRY POINT
# =============================================================================
 
# =============================================================================
# 11.  ENTRY POINT  --  just run the file
# =============================================================================
# `python3 r02_ncstr_process_scale.py`  runs the whole pipeline end to end and
# picks N for you. Everything below is also importable and individually
# callable if you want one piece; the optional CLI mode argument is documented
# at the bottom.
 
DEFAULT_SUBSTRATE = 'Hexamine'
DEFAULT_INLINE = False       # False = pre-formed AcONO2, True = in-line
N_MAX_SEARCH = 8
 
 
def pharmapy_available():
    try:
        _import_pharmapy()
        return True
    except Exception:
        return False
 
 
def find_min_feasible_N(substrate=DEFAULT_SUBSTRATE, inline=DEFAULT_INLINE,
                        n_max=N_MAX_SEARCH, split_mode='optimized',
                        n_starts=2, maxiter=100):
    """Smallest N with a fully feasible optimized design, searching upward.
 
    This -- not scan_n -- is the honest answer to "what N do I need". scan_n
    holds T and tau fixed and splits them evenly, so it UNDERSTATES what a
    train can do: it said N=6 while the optimizer clears the same constraints
    at N=4 once it is allowed to tune stage temperature and feed split. The
    scan is a diagnostic for WHERE THE HEAT GOES, not a sizing tool.
    """
    print("\n" + "=" * 96)
    print(f"SEARCHING FOR MINIMUM FEASIBLE N   (optimizing T_i, tau_i and "
          f"feed split_i at each N)")
    print(f"  must satisfy: u_ht <= {U_HT_AVAILABLE:.0f} W/m2/K | V <= "
          f"{MAX_VESSEL_VOL_M3*1000:.0f} L/vessel | X >= {X_HEXAMINE_MIN:.2f}")
    print("=" * 96)
    for N in range(1, n_max + 1):
        d = optimize_train(N, substrate, inline, split_mode=split_mode,
                           n_starts=n_starts, maxiter=maxiter, verbose=False)
        if d['success']:
            sz = d['sized']
            print(f"  N={N}: FEASIBLE  -- V_tot {sz['vol_total']*1000:6.1f} L, "
                  f"u_req {sz['u_ht_req_max']:4.0f}, "
                  f"X {sz['conversion']*100:.1f}%, "
                  f"RDX {sz['selectivity'][0]*100:.1f}%")
            print(f"\n  ==> MINIMUM FEASIBLE N = {N}")
            return N, d
        print(f"  N={N}: infeasible")
    print(f"\n  ==> NO feasible N up to {n_max}. The binding constraint is "
          f"almost certainly u_ht.\n      More stages cannot fix an exchanger "
          f"problem -- see the closing notes.")
    return None, None
 
 
def scan_inventory_cap(substrate=DEFAULT_SUBSTRATE, inline=DEFAULT_INLINE,
                      caps_L=(500, 250, 100, 50, 25, 10), n_max=16,
                      n_starts=1, maxiter=80):
    """Minimum feasible N as a function of the per-vessel inventory cap.
 
    THIS is the table that answers "how many reactors do I need", because on
    this network heat does not set N. The optimizer clears the thermal
    constraint at small N by simply building a longer-tau vessel: tau buys
    jacket area (A ~ V^(2/3)) faster than it buys duty, and the cold feed
    absorbs a chunk of the exotherm as sensible heat before the jacket ever
    sees it. What forces a train is the limit on how much energetic material
    you will accept in one vessel -- a licensing and safety number, not a
    chemistry one.
 
    MAX_VESSEL_VOL_M3 is a placeholder. Find your real limit, read this table
    at that row, and that is your N. Everything else here is downstream of
    that single input.
 
    Caps are walked in DECREASING order and each search resumes at the
    previous cap's answer. N_min is monotonically non-increasing in the cap,
    so this is exact, not a heuristic -- it just skips the N values already
    known to be infeasible.
    """
    global MAX_VESSEL_VOL_M3
    saved = MAX_VESSEL_VOL_M3
    caps = sorted(caps_L, reverse=True)
    print("\n" + "=" * 96)
    print("MINIMUM N vs PER-VESSEL INVENTORY CAP")
    print("=" * 96)
    print(f"  {'cap [L]':>9} {'N_min':>6} {'V_tot[L]':>9} {'V_max[L]':>9} "
          f"{'A_tot[m2]':>10} {'u_req':>7} {'RDX%':>7} {'Side%':>7} "
          f"{'tau_tot[min]':>13}")
    print("  " + "-" * 92)
    rows, n_from = [], 1
    try:
        for cap_L in caps:
            MAX_VESSEL_VOL_M3 = cap_L / 1000.0
            found = None
            for N in range(n_from, n_max + 1):
                d = optimize_train(N, substrate, inline,
                                   split_mode='optimized', n_starts=n_starts,
                                   maxiter=maxiter, verbose=False)
                if d['success']:
                    found, n_from = d, N
                    break
            if found is None:
                print(f"  {cap_L:>9.0f} {'--':>6}   no feasible N <= {n_max}")
                rows.append(dict(cap_L=cap_L, N=None, design=None))
                continue
            sz = found['sized']
            print(f"  {cap_L:>9.0f} {found['N']:>6} {sz['vol_total']*1000:>9.1f} "
                  f"{sz['vol_max']*1000:>9.1f} {sz['area_total']:>10.2f} "
                  f"{sz['u_ht_req_max']:>7.0f} "
                  f"{sz['selectivity'][0]*100:>6.1f}% "
                  f"{sz['selectivity'][2]*100:>6.1f}% "
                  f"{sz['taus'].sum()/60:>13.2f}")
            rows.append(dict(cap_L=cap_L, N=found['N'], design=found))
    finally:
        MAX_VESSEL_VOL_M3 = saved
    print("  " + "-" * 92)
    print(f"  MAX_VESSEL_VOL_M3 is currently {saved*1000:.0f} L -- a "
          f"PLACEHOLDER, not a real limit.")
    print("  Replace it with your licensed number and read N off this table.")
    print("\n  Read the V_tot and RDX% columns together. Tightening the cap")
    print("  does NOT cost total inventory -- it REDUCES it, and improves")
    print("  selectivity at the same time. A train is not a concession you make")
    print("  to a safety limit; on this network it is strictly better on both")
    print("  counts. The single vessel only looks competitive because it has to")
    print("  inflate tau to buy itself jacket area, and that inflated tau is")
    print("  pure inventory you are holding for no chemical reason.")
    print("  What a train actually costs you is capital and control complexity,")
    print("  which is not modelled here.")
    return rows
 
 
def main(substrate=DEFAULT_SUBSTRATE, inline=DEFAULT_INLINE,
         run_pharmapy=True):
    """Full pipeline. Every step is guarded: a failure in one section reports
    itself and the run continues, so you always get the sections that did work
    rather than a traceback and nothing."""
    t0 = time.time()
    print("=" * 96)
    print("R02 PROCESS-SCALE N-CSTR TRAIN  --  AcONO2 ROUTE")
    print("=" * 96)
    print(f"  target          : {PRODUCT_KG_PER_DAY:.0f} kg/day {PRODUCT_BASIS}")
    print(f"  substrate       : {substrate}")
    print(f"  route           : {'in-line' if inline else 'pre-formed'} AcONO2")
    print(f"  stoichiometry   : STOICH_MODE={STOICH_MODE}")
    print(f"  activation E    : +/-/- (signed; HMX and Side anti-Arrhenius)")
    print(f"  T window        : {T_MIN_C:.0f}-{T_MAX_C:.0f} C, "
          f"kinetic T_ref fixed at {T_REF_KIN_C:.0f} C")
    if ENFORCE_ACONO2_CAP and not inline:
        print(f"  AcONO2 T cap    : {ACONO2_T_CAP_C:.0f} C "
              f"(Andreozzi 2002 decomposition onset)")
    print(f"  available u_ht  : {U_HT_AVAILABLE:.0f} W/m2/K "
          f"(ceiling {U_HT_CEILING:.0f}, SiC)")
    print("\n  Kinetics are PROVISIONAL and DH is a single lumped value.")
    print("  Read the closing notes before you quote any number from this run.")
 
    # ---- 1. structural checks ------------------------------------------
    print("\n\n" + "#" * 96)
    print("# 1. STRUCTURAL CHECKS  --  does PharmaPy see what I think it sees?")
    print("#" * 96)
    try:
        sanity_check_ordering()
        net = build_network(substrate, inline)
        print("\n  Reaction network as PharmaPy will see it:")
        for r, pf in zip(net, build_params_f(net)):
            print(f"    {r['name']:<8} {_rxn_string(r['nu'])}")
            print(f"    {'':<8}   k0={r['k0']:.4e}  Ea={r['ea']:>+10.1f} J/mol"
                  f"  DH={r['dh']:>+10.1f} J/mol  params_f={pf}")
        carrier, sub, phi, names = build_feed_streams(substrate, inline)
        blend = phi * sub + (1 - phi) * carrier
        overall, _ = build_feed(substrate, inline)
        print(f"\n  Feed decomposition (phi = Q_substrate/Q_total = {phi:.4f}):")
        print(f"    {'Species':<20}{'carrier':>10}{'substrate':>11}"
              f"{'blend':>10}{'DOE feed':>10}")
        for i, n in enumerate(names):
            if max(carrier[i], sub[i]) > 1e-9:
                print(f"    {n:<20}{carrier[i]:>10.3f}{sub[i]:>11.3f}"
                      f"{blend[i]:>10.3f}{overall[i]:>10.3f}")
        err = float(np.max(np.abs(blend - overall)))
        print(f"    blend vs DOE feed max error = {err:.2e} mol/L  "
              f"{'OK' if err < 1e-9 else '*** MISMATCH ***'}")
    except Exception as e:
        print(f"  *** structural checks FAILED: {e}")
        return
 
    # ---- 2. where does the heat go? -------------------------------------
    print("\n\n" + "#" * 96)
    print("# 2. WHERE DOES THE HEAT GO?  --  the reason N alone is not the "
          "answer")
    print("#" * 96)
    print("  Two scans at identical N, tau and T. The ONLY difference is "
          "whether the\n  substrate is all fed to stage 1 or spread across "
          "the train. Watch q_frac\n  (share of the exotherm in the worst "
          "stage) and u_req.")
    scans = {}
    for sm in ('all_stage1', 'equal'):
        try:
            scans[sm] = scan_n(n_max=N_MAX_SEARCH, substrate=substrate,
                               use_nitric_acid_inline=inline, split_mode=sm)
        except Exception as e:
            print(f"  *** scan '{sm}' FAILED: {e}")
 
    # ---- 3. temperature sweep -------------------------------------------
    print("\n\n" + "#" * 96)
    print("# 3. ISOTHERMAL TEMPERATURE SWEEP  --  the +/-/- Ea signature")
    print("#" * 96)
    print("  Every stage held at the same T. RDX should climb monotonically "
          "while HMX\n  and Side both fall -- that IS the signed-Ea "
          "signature. If it doesn't, the\n  Ea signs are not doing what you "
          "think they are.")
    try:
        sweep_temperature(N=4, substrate=substrate,
                          use_nitric_acid_inline=inline, split_mode='equal')
    except Exception as e:
        print(f"  *** sweep FAILED: {e}")
 
    # ---- 4. pick N ------------------------------------------------------
    print("\n\n" + "#" * 96)
    print("# 4. NON-ISOTHERMAL DESIGN  --  each stage gets its own T, tau and "
          "feed split")
    print("#" * 96)
    design = None
    try:
        N_min, design = find_min_feasible_N(substrate, inline)
        if design is not None:
            design_full = optimize_train(N_min, substrate, inline,
                                         split_mode='optimized', n_starts=3,
                                         maxiter=200, verbose=False)
            if design_full['success'] and (design_full['sized']['vol_total']
                                           < design['sized']['vol_total']):
                design = design_full
            report_design(design)
    except Exception as e:
        print(f"  *** optimization FAILED: {e}")
 
    # ---- 5. what actually sets N ----------------------------------------
    print("\n\n" + "#" * 96)
    print("# 5. WHAT ACTUALLY SETS N  --  inventory, not heat")
    print("#" * 96)
    print("  Section 4 cleared the thermal constraint at low N by building a")
    print("  longer-tau vessel: tau buys jacket area (A ~ V^(2/3)) faster than")
    print("  it buys duty, and the cold feed absorbs a chunk of the exotherm as")
    print("  sensible heat before the jacket ever sees it. So heat does NOT set")
    print("  N here. The cap on energetic material per vessel does -- and that")
    print("  is a number only you can supply. Find your row.")
    try:
        scan_inventory_cap(substrate, inline)
    except Exception as e:
        print(f"  *** inventory scan FAILED: {e}")
 
    # ---- 6. PharmaPy ----------------------------------------------------
    print("\n\n" + "#" * 96)
    print("# 6. PHARMAPY SIMULATION  --  isothermal first, then "
          "non-isothermal")
    print("#" * 96)
    if design is None:
        print("  Skipped: no feasible design to simulate.")
    elif not run_pharmapy:
        print("  Skipped: run_pharmapy=False.")
    elif not pharmapy_available():
        print("  Skipped: PharmaPy could not be imported in this interpreter.")
        print("  Everything above is pure numpy/scipy and is unaffected.")
        print("  Activate the environment holding the Auggie-dev branch and "
              "re-run to\n  get sections 6a/6b.")
    else:
        for iso in (True, False):
            tag = 'ISOTHERMAL' if iso else 'NON-ISOTHERMAL'
            print(f"\n  ---- 6{'a' if iso else 'b'}. {tag} ----")
            if iso:
                print("  Isothermal first, on purpose: it isolates "
                      "stoichiometry and the mass\n  balance from the energy "
                      "balance. If this does not reproduce the section-4\n  "
                      "algebraic sizing, a non-isothermal run only stacks a "
                      "second unknown on\n  top of the first.")
            try:
                res = run_pharmapy_train(design, isothermal=iso)
                report_pharmapy_run(res, design, iso)
            except Exception as e:
                print(f"  *** {tag} run FAILED: {e}")
                if not iso:
                    print("  A non-isothermal failure here is often physical "
                          "(runaway), not a bug.")
 
    # ---- 6. closing notes -----------------------------------------------
    print("\n\n" + "#" * 96)
    print("# 7. WHAT THIS RUN DOES AND DOES NOT ESTABLISH")
    print("#" * 96)
    if design is not None:
        sz = design['sized']
        print(f"  DESIGN: N={design['N']} | V_tot="
              f"{sz['vol_total']*1000:.1f} L | A_tot={sz['area_total']:.2f} m2 "
              f"| Q={sz['q_vol']*3600:.3f} m3/h")
        print(f"          RDX {sz['selectivity'][0]*100:.1f}% / HMX "
              f"{sz['selectivity'][1]*100:.1f}% / Side "
              f"{sz['selectivity'][2]*100:.1f}% | "
              f"{sz['q_gen_total']/1000:.1f} kW total")
        print(f"          substrate split across stages: "
              f"{np.round(sz['splits']*100, 1)} %")
    print("""
  ESTABLISHED (subject to the caveats below):
    * Heat does not set N. The thermal constraint clears at N=1 by inflating
      tau until the vessel has enough jacket area, because A ~ V^(2/3) buys
      area faster than volume buys duty, and the cold feed absorbs ~25% of
      the exotherm as sensible heat before the jacket sees it. The
      PER-VESSEL INVENTORY CAP sets N. That is a licensing number you have to
      supply -- section 5 gives you N as a function of it.
    * A train is not a concession to that cap. Tightening the cap LOWERS
      total inventory (138 -> 59 L from a 500 L to a 10 L cap) and RAISES RDX
      selectivity (52 -> 56%) at identical u_req. The single vessel is only
      competitive because it holds tau it does not chemically need.
    * Splitting RESIDENCE TIME does not split the exotherm on this network.
      The substrate is consumed in stage 1 at any practical tau, so an
      equal-tau train wraps a smaller vessel around the same duty and u_req
      gets WORSE with N. Section 2 shows this directly.
    * Splitting the SUBSTRATE FEED does split it, ~1/N per stage, and it also
      improves selectivity: substrate orders run Side 0.50 > RDX 0.29 >
      HMX 0.21, so keeping every stage substrate-lean starves the side
      pathway. This is also why a PFR is the wrong instinct here -- it holds
      substrate concentration high at the front end. Backmixing helps you.
    * Ug < Ur is not the binding constraint anywhere. With two of three
      pathways anti-Arrhenius, Ug is driven to ~0 and every stage is
      DUTY-limited. The question is 'can the jacket carry the load', not
      'will it run away'.
    * Total heat is set by production rate x DH. N changes only where it
      lands and how much area you get. It cannot change the total.
 
  NOT ESTABLISHED -- decide these before quoting anything:
    * DH is ONE lumped Jadhav value shared by all three pathways. This is the
      biggest soft spot in the whole module. You already know the linear
      byproduct pathways are MORE exothermic than HMX; once
      fit_thermal_params.py returns pathway-resolved DH, the selectivity/heat
      coupling changes sign of argument and the Ug conclusion may move.
    * STOICH_MODE='legacy' is 1:1 and chemically wrong. At 1500 kg/day it
      sets your raw-material bill. 'bachmann' is implemented and atom-balanced
      but makes ammonium nitrate a stoichiometric N source, contradicting
      Zheng's proton-shuttle result. Your Set 3 add-back arms decide it.
    * The k values are Jadhav's, extracted from the CSTR performance equation
      at his inlet concentrations. If a stage pins at TAU_MIN_S, that is this
      problem surfacing, not a design.
    * SUBSTRATE_STREAM_CONC is a SOLUBILITY guess and it sets phi and hence
      the whole carrier composition. Measure it.
    * MAX_VESSEL_VOL_M3 is a placeholder and it is THE load-bearing input of
      this whole module -- it is what sets N. Replace it with your licensed
      limit and re-read the section 5 table.
    * AcONO2 decomposition is NOT in the network, so anything above
      ~65 C understates the heat.""")
    print(f"\n  [run completed in {time.time() - t0:.1f} s]")
    print("#" * 96)
 
 
if __name__ == "__main__":
    import sys
    MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
 
    if MODE in ("all", "main"):
        main()
    elif MODE in ("-h", "--help", "help"):
        print(__doc__ or "")
        print("Usage:\n"
              "  python3 r02_ncstr_process_scale.py           # everything\n"
              "  python3 r02_ncstr_process_scale.py check     # structure "
              "only\n"
              "  python3 r02_ncstr_process_scale.py scan      # heat "
              "distribution vs N\n"
              "  python3 r02_ncstr_process_scale.py sweep     # isothermal T "
              "sweep\n"
              "  python3 r02_ncstr_process_scale.py optimize  # find + report "
              "min feasible N\n"
              "  python3 r02_ncstr_process_scale.py inventory # N vs "
              "inventory cap  <-- the real answer\n"
              "  python3 r02_ncstr_process_scale.py nosim     # everything "
              "except PharmaPy")
    elif MODE == "nosim":
        main(run_pharmapy=False)
    elif MODE == "check":
        sanity_check_ordering()
        net = build_network(DEFAULT_SUBSTRATE, DEFAULT_INLINE)
        for r, pf in zip(net, build_params_f(net)):
            print(f"  {r['name']:<8} {_rxn_string(r['nu'])}  Ea={r['ea']:>+9.1f}"
                  f"  params_f={pf}")
    elif MODE == "scan":
        for sm in ('all_stage1', 'equal'):
            scan_n(n_max=N_MAX_SEARCH, substrate=DEFAULT_SUBSTRATE,
                   use_nitric_acid_inline=DEFAULT_INLINE, split_mode=sm)
    elif MODE == "sweep":
        sweep_temperature(N=4, substrate=DEFAULT_SUBSTRATE,
                          use_nitric_acid_inline=DEFAULT_INLINE,
                          split_mode='equal')
    elif MODE == "inventory":
        scan_inventory_cap(DEFAULT_SUBSTRATE, DEFAULT_INLINE)
    elif MODE == "optimize":
        N_min, d = find_min_feasible_N(DEFAULT_SUBSTRATE, DEFAULT_INLINE)
        if d:
            report_design(d)
    elif MODE in ("isothermal", "nonisothermal"):
        N_min, d = find_min_feasible_N(DEFAULT_SUBSTRATE, DEFAULT_INLINE)
        if d:
            iso = (MODE == "isothermal")
            report_pharmapy_run(run_pharmapy_train(d, isothermal=iso), d, iso)
    else:
        print(f"Unknown mode '{MODE}'. Try --help.")