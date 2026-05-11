"""Multi-unit process simulation:
1. CSTR1 -> 2. Surge1 -> 3. Crash tank -> 4. Continuous filter -> 5. CSTR2 -> 6. Surge2.

Units used in this script:
- Flow rates: L/s
- Volumes: L
- Concentrations: mol/L
- Temperature: K
- Solids inventories: mol (total in unit) or mol/L when normalised
- Heat transfer: J/(s*K)
- Heat of reaction: J/mol

CSTR2 kinetic model – Method 1: multi-step mechanistic surrogate
----------------------------------------------------------------
The lumped single power-law has been replaced with THREE explicit steps:

  Step 1  (activation / initiation):
      A + FWNA → [I]
      r1 = k1(T) · C_A^α1 · C_FWNA^β1
      k1 = k0_1 · exp(-Ea1 / RT)
      Physical meaning: FWNA (fuming HNO3) attacks A to form a reactive
      nitrated intermediate [I].  FWNA is converted to NA (nitric acid) 1:1.

  Step 2  (product formation):
      [I] + AAh → f_lb1·B1(l) + f_lb2·B2(l) + f_sb1·B1(s) + f_sb2·B2(s) + NA
      r2 = k2(T) · C_I^γ · C_AAh^δ
      k2 = k0_2 · exp(-Ea2 / RT)
      Physical meaning: The intermediate reacts with the acid (AAh) to form
      the desired solid/liquid B products.  AAh is consumed 1:1.

  Impurity (lumped, parallel to Step 1):
      A + AA → Imp
      r_imp = k_imp(T) · C_A^m · C_AA^n
      k_imp = k0_imp · exp(-Ea_imp / RT)
      Physical meaning: AA (anhydrous/acetic acid co-solvent) reacts with A
      via a slower, higher-Ea side pathway producing a lumped impurity pool.
      Because Ea_imp > Ea_main, impurity selectivity worsens at high T.

CSTR2 state vector (12 components):
  [C_A, C_FWNA, C_I, C_AAh, C_AA, C_NA, C_B1, C_B2, C_Imp, T, S_B1, S_B2]
"""

# Created by Augustus Sison on 2024-06-05
# CSTR2 kinetics updated to Method 1 multi-step surrogate

# TODO: replace the static density values with temperature-dependent density correlations for each unit.
# TODO: send the waste stream to extractive distillation and convert it to reusable liq N later.
# TODO: crash tank needs more robust handling of supersaturation and solids inventory management.
# TODO: temperature-dependent crystallization kinetics should affect solids formation rate (currently ignored).
# TODO: filter model is very simplified - could be improved with more realistic semi-batch scheduling
#       or a continuous filtration model with dynamic stream splits based on solids loading and filter capacity.
# TODO: Stream split fractions in the filter are currently hardcoded -
#       could be made dynamic based on concentrations, solids loading, or other factors.
# TODO: Unit operations dependent on surge tank active or not, not dependent on previous reactor
#       residence time and travel time.
# TODO: Fit reaction orders α1, β1, γ, δ, m, n from isothermal concentration-time experiments.
# TODO: Fit k0_1, Ea1, k0_2, Ea2, k0_imp, Ea_imp from temperature-series data.
# TODO: Step-2 stoichiometry coefficients (f_lb1, f_lb2, f_sb1, f_sb2) sum constraint
#       (f_lb1+f_lb2+f_sb1+f_sb2 = 1) is currently soft – add a hard normalisation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd

# =============================================================================
# Parameters – upstream units (unchanged)
# =============================================================================
params_cstr1 = {
    "F": 1.0, "V": 10.0,
    "C_Hf": 2.0, "C_Nf": 2.0, "C_Wf": 5.0, "C_Af": 0.0,
    "Tf": 350.0, "Tc": 300.0,
    "rho_ref": 1000.0, "Cp": 4.18,
    "species_rho": {"H": 980.0, "N": 1020.0, "W": 1000.0, "A": 1100.0, "solid_A": 1500.0},
    "UA": 5000.0,
    "k0": 1e6, "Ea": 50000.0, "R": 8.314,
    "dH": -100000.0,
}

params_surge1 = {
    "F_in": params_cstr1["F"],
    "V": 10.0, "V0": 0.0,
    "t_open": 0.8 * 10.0 / params_cstr1["F"],
    "rho_ref": 995.0, "Cp": params_cstr1["Cp"],
    "species_rho": params_cstr1["species_rho"],
    "UA": 2000.0, "Tc": 298.15,
}

params_crash = {
    "F_in": params_surge1["F_in"], "F_N": 0.2, "V": 5.0,
    "Tc": 290.0, "UA": 3000.0, "rho_ref": 1100.0, "Cp": params_cstr1["Cp"],
    "species_rho": params_cstr1["species_rho"],
    "k_p": 0.5, "C_A_eq": 0.05,
    "C_N_ext": 5.0, "T_N": 293.15,
    "M_A_gmol": 100.0, "T0": 293.15, "dH_precip": -5000.0,
}

params_filter = {
    "n_filters": 3,
    "t_start": params_surge1["t_open"],
    "t_fill": 1.0, "t_vacuum": 1.0, "t_redissolve": 1.0,
    "V0": 0.8, "V_max": 2.0,
    "F_feed": params_crash["F_in"] + params_crash["F_N"],
    "AA_ratio": 1.25, "trace_A_fraction": 0.05,
    "f_N_main_base": 0.02, "f_W_main_base": 0.01, "f_A_main_base": 0.15,
    "alpha_solid": 0.2, "solid_scale": 1.0,
}
#old
params_filter["t_cycle"] = (
    params_filter["t_fill"] + params_filter["t_vacuum"] + params_filter["t_redissolve"]
)

# Consistent colour scheme
COLOR_SCHEME = {
    "C_H": "#1f77b4", "C_N": "#ff7f0e", "C_W": "#2ca02c", "C_A": "#d62728",
    "C_AAh": "#e377c2", "S": "#9467bd", "T": "#8c564b", "AA": "#17becf",
    "C_I": "#bcbd22",   # yellow-green  – intermediate [I]
    "C_NA": "#ff7f0e",  # orange        – NA (nitric acid product)
    "C_Imp": "#7f7f7f", # grey          – lumped impurity
}

# =============================================================================
# Shared helper functions (unchanged)
# =============================================================================
def mixture_density_liquid(composition, species_rho):
    total_concentration = sum(composition.values())
    if total_concentration <= 0.0:
        return 1.0
    specific_volume = 0.0
    for species, concentration in composition.items():
        rho_i = species_rho[species]
        specific_volume += (concentration / total_concentration) / rho_i
    return 1.0 / max(specific_volume, 1e-12)

def liquid_composition_from_molar(c_h, c_n, c_w, c_a):
    return {"H": c_h, "N": c_n, "W": c_w, "A": c_a}

def crash_mixture_density(c_h, c_n, c_w, c_a, solid_moles_a, p, tank_volume_l):
    liquid_composition = liquid_composition_from_molar(c_h, c_n, c_w, c_a)
    liquid_density = mixture_density_liquid(liquid_composition, p["species_rho"])
    liquid_mass = liquid_density * (tank_volume_l / 1000.0)
    solid_mass = solid_moles_a * (p["M_A_gmol"] / 1000.0)
    return (liquid_mass + solid_mass) / max(tank_volume_l / 1000.0, 1e-12)

def tail_mean(series, n_tail):
    return float(np.mean(series[-n_tail:]))

# =============================================================================
# CSTR1 model
# =============================================================================
def cstr1_model(t, y, p):
    C_H, C_N, C_W, C_A, T = y
    F, V = p["F"], p["V"]
    k = p["k0"] * np.exp(-p["Ea"] / (p["R"] * T))
    r = k * C_H * C_N * C_W
    dC_H_dt = (F/V)*(p["C_Hf"] - C_H) - r
    dC_N_dt = (F/V)*(p["C_Nf"] - C_N) - r
    dC_W_dt = (F/V)*(p["C_Wf"] - C_W) - r
    dC_A_dt = (F/V)*(p["C_Af"] - C_A) + r
    dT_dt = (F/V)*(p["Tf"] - T) - (p["dH"]/(p["rho_ref"]*p["Cp"]))*r \
            - (p["UA"]/(p["rho_ref"]*p["Cp"]*V))*(T - p["Tc"])
    return [dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt]

def cstr1_density(c_h, c_n, c_w, c_a, p):
    return mixture_density_liquid(liquid_composition_from_molar(c_h, c_n, c_w, c_a), p["species_rho"])

# =============================================================================
# Surge1 model
# =============================================================================
def surge1_tank_model(t, y, p, inlet_interp):
    V, C_H, C_N, C_W, C_A, T = y
    C_H_in = inlet_interp["C_H"](t); C_N_in = inlet_interp["C_N"](t)
    C_W_in = inlet_interp["C_W"](t); C_A_in = inlet_interp["C_A"](t)
    T_in = inlet_interp["T"](t)
    F_in = p["F_in"]
    F_out = F_in if t >= p["t_open"] else 0.0
    dV_dt = F_in - F_out
    if V <= 1e-9:
        return [dV_dt, 0.0, 0.0, 0.0, 0.0, 0.0]
    dC_H_dt = (F_in/V)*(C_H_in - C_H)
    dC_N_dt = (F_in/V)*(C_N_in - C_N)
    dC_W_dt = (F_in/V)*(C_W_in - C_W)
    dC_A_dt = (F_in/V)*(C_A_in - C_A)
    dT_dt = (F_in/V)*(T_in - T) - (p["UA"]/(p["rho_ref"]*p["Cp"]*V))*(T - p["Tc"])
    return [dV_dt, dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt]

def surge1_density(c_h, c_n, c_w, c_a, p):
    return mixture_density_liquid(liquid_composition_from_molar(c_h, c_n, c_w, c_a), p["species_rho"])

# =============================================================================
# Crash tank model (unchanged)
# =============================================================================
def crash_tank_model(t, y, p, inlet_interp, t_open):
    C_H, C_N, C_W, C_A, T, S = y
    surge_active = t >= t_open
    if not surge_active:
        r_p = p["k_p"] * max(C_A - p["C_A_eq"], 0.0)
        return [0.0, 0.0, 0.0, -r_p, 0.0, 0.0]
    F_in = p["F_in"]; F_N = p["F_N"]
    F_tot = F_in + F_N
    C_H_in = inlet_interp["C_H"](t); C_N_in = inlet_interp["C_N"](t)
    C_W_in = inlet_interp["C_W"](t); C_A_in = inlet_interp["C_A"](t)
    T_in = inlet_interp["T"](t)
    C_Hf = (F_in*C_H_in) / F_tot
    C_Nf = (F_in*C_N_in + F_N*p["C_N_ext"]) / F_tot
    C_Wf = (F_in*C_W_in) / F_tot
    C_Af = (F_in*C_A_in) / F_tot
    Tf   = (F_in*T_in + F_N*p["T_N"]) / F_tot
    r_p = p["k_p"] * max(C_A - p["C_A_eq"], 0.0)
    dC_H_dt = (F_tot/p["V"])*(C_Hf - C_H)
    dC_N_dt = (F_tot/p["V"])*(C_Nf - C_N)
    dC_W_dt = (F_tot/p["V"])*(C_Wf - C_W)
    dC_A_dt = (F_tot/p["V"])*(C_Af - C_A) - r_p
    solid_out = (F_tot * max(S, 0.0) / max(p["V"], 1e-12)) if surge_active else 0.0
    dS_dt = r_p*p["V"] - solid_out
    rho_mix = crash_mixture_density(C_H, C_N, C_W, C_A, S, p, p["V"])
    dT_dt = (F_tot/p["V"])*(Tf - T) \
            - (p["UA"]/(rho_mix*p["Cp"]*p["V"]))*(T - p["Tc"]) \
            - (p.get("dH_precip", 0.0)*r_p)/(rho_mix*p["Cp"])
    return [dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt, dS_dt]

def crash_liquid_density(c_h, c_n, c_w, c_a, p):
    return mixture_density_liquid(liquid_composition_from_molar(c_h, c_n, c_w, c_a), p["species_rho"])

# =============================================================================
# Filter
# =============================================================================
def continuous_filter_split(t, p, inlet_interp):
    surge_active = t >= p["t_start"]
    if not surge_active:
        return {k: 0.0 for k in [
            "S_out_to_main","C_H_main","C_N_main","C_W_main","C_A_main",
            "C_H_waste","C_N_waste","C_W_waste","C_A_waste","AA_main","total_A_main"
        ]} | {"phase": "inactive"}
    C_H_in = float(inlet_interp["C_H"](t)); C_N_in = float(inlet_interp["C_N"](t))
    C_W_in = float(inlet_interp["C_W"](t)); C_A_in = float(inlet_interp["C_A"](t))
    S_in   = float(inlet_interp["S"](t))
    solid_A_main = max(S_in, 0.0)
    s_frac = S_in / (S_in + p.get("solid_scale", 1.0)) if S_in >= 0.0 else 0.0
    f_N_main = float(np.clip(p.get("f_N_main_base",0.02) + p.get("alpha_solid",0.1)*s_frac, 0.0, 0.9))
    f_W_main = float(np.clip(p.get("f_W_main_base",0.01) + 0.5*p.get("alpha_solid",0.1)*s_frac, 0.0, 0.9))
    f_A_main = float(np.clip(p.get("f_A_main_base",0.15)*(1.0 - 0.5*s_frac), 0.0, 1.0))
    C_H_main = 0.0
    C_N_main = f_N_main*C_N_in; C_W_main = f_W_main*C_W_in; C_A_main = f_A_main*C_A_in
    C_H_waste = C_H_in
    C_N_waste = max(0.0, 1.0-f_N_main)*C_N_in
    C_W_waste = max(0.0, 1.0-f_W_main)*C_W_in
    C_A_waste = max(0.0, 1.0-f_A_main)*C_A_in
    AA_main = p["AA_ratio"]*solid_A_main
    return {
        "phase": "continuous", "S_out_to_main": solid_A_main,
        "C_H_main": C_H_main, "C_N_main": C_N_main, "C_W_main": C_W_main, "C_A_main": C_A_main,
        "C_H_waste": C_H_waste, "C_N_waste": C_N_waste, "C_W_waste": C_W_waste, "C_A_waste": C_A_waste,
        "AA_main": AA_main, "total_A_main": solid_A_main + C_A_main,
    }

# =============================================================================
# CSTR2  –  Method 1: multi-step mechanistic surrogate
# =============================================================================
#
# Stoichiometric assignments:
#   Step 1:   A  +  FWNA  →  [I]  +  NA
#             - A consumed 1:1 with FWNA
#             - intermediate [I] forms 1:1
#             - FWNA is converted to NA (nitric acid) 1:1
#
#   Step 2:   [I]  +  AAh  →  products (B1, B2 in liquid and solid phases)
#             - [I] consumed 1:1 with AAh
#             - products split by stoichiometric fractions f_lb1, f_lb2 (liquid)
#               and f_sb1, f_sb2 (solid)
#             - no NA produced in step 2 (acid consumed, not regenerated)
#
#   Impurity: A  +  AA  →  Imp          (parallel side reaction)
#             - lumped, higher Ea than main path
#             - AA acts as both solvent and (slow) acylating agent
#
# Energy balance accounts for ΔH of each step independently.
#
params_cstr2 = {
    "F": 0.8,       # L/s total feed to CSTR2
    "V": 8.0,       # L

    # ------- Feed concentrations (mol/L) -------
    # These can be overridden by upstream filter interpolants at runtime.
    "C_A_in":     0.05,   # unreacted A arriving from filter main stream
    "C_FWNA_in":  0.30,   # fuming white nitric acid (fresh feed to CSTR2)
    "C_AAh_in":   0.50,   # acid form (protonated)
    "C_AA_in":    0.20,   # anhydrous / co-solvent form
    "C_NA_in":    0.00,   # nitric acid (product) – zero in fresh feed
    "Tf":         298.15, # feed temperature (K)
    "Tc":         295.0,  # jacket temperature (K)

    # ------- Physical properties -------
    "rho_ref": 1050.0,    # kg/m³ (approximate)
    "Cp":      4.18,      # J/(g·K)
    "UA":      1500.0,    # J/(s·K)
    "R":       8.314,     # J/(mol·K)

    # ------- Step 1 kinetics: A + FWNA → [I] + NA -------
    # Reaction orders (to be fitted from isothermal experiments)
    "alpha1":  1.0,   # order in A
    "beta1":   1.0,   # order in FWNA
    "k0_1":    1e4,   # pre-exponential (L^(α1+β1-1) mol^(1-α1-β1) s⁻¹)
    "Ea1":     30000.0,  # J/mol  (moderate barrier – fast initiation)
    "dH1":    -40000.0,  # J/mol  (exothermic activation)

    # ------- Step 2 kinetics: [I] + AAh → B products -------
    "gamma":   1.0,   # order in [I]
    "delta":   1.0,   # order in AAh
    "k0_2":    5e3,   # pre-exponential
    "Ea2":     25000.0,  # J/mol  (lower barrier than step 1 → step 2 fast once I forms)
    "dH2":    -66000.0,  # J/mol  (strongly exothermic product formation)

    # ------- Impurity kinetics: A + AA → Imp (lumped) -------
    "m_imp":   1.0,   # order in A
    "n_imp":   1.0,   # order in AA
    "k0_imp":  5e1,   # pre-exponential (much smaller than main path)
    "Ea_imp":  55000.0,  # J/mol  (higher Ea → worsens at high T, sets T operating window)
    "dH_imp": -15000.0,  # J/mol

    # ------- Product split fractions for Step 2 -------
    # Liquid products (mol product per mol [I] reacted)
    "f_lb1": 0.30,  # B1 liquid fraction
    "f_lb2": 0.20,  # B2 liquid fraction
    # Solid products (mol product per mol [I] reacted)
    "f_sb1": 0.25,  # B1 solid fraction
    "f_sb2": 0.15,  # B2 solid fraction
    # Note: f_lb1+f_lb2+f_sb1+f_sb2 = 0.90 (remaining 10% lumped in side products / not tracked)
}


def cstr2_model(t, y, p, inlet_interp=None, t_feed_start=None):
    """
    CSTR2 – multi-step mechanistic surrogate.

    State vector (12 components):
      C_A    – substrate A (mol/L)
      C_FWNA – fuming white nitric acid (mol/L)
      C_I    – reactive intermediate [I] (mol/L)
      C_AAh  – protonated acid (mol/L)
      C_AA   – anhydrous acid / co-solvent (mol/L)
      C_NA   – nitric acid product NA (mol/L)
      C_B1   – liquid product B1 (mol/L)
      C_B2   – liquid product B2 (mol/L)
      C_Imp  – lumped impurity (mol/L)
      T      – reactor temperature (K)
      S_B1   – solid B1 inventory (mol, total in reactor)
      S_B2   – solid B2 inventory (mol, total in reactor)
    """
    C_A, C_FWNA, C_I, C_AAh, C_AA, C_NA, C_B1, C_B2, C_Imp, T, S_B1, S_B2 = y

    F, V, R = p["F"], p["V"], p["R"]
    feed_active = (t_feed_start is not None and t >= t_feed_start) or (t_feed_start is None)

    # ---- Feed concentrations: prefer upstream interpolant, fall back to params ----
    def _get(key, fallback_key):
        if inlet_interp is not None and key in inlet_interp:
            return float(inlet_interp[key](t))
        return p.get(fallback_key, 0.0)

    C_A_f    = _get("C_A",   "C_A_in")
    # FWNA and AAh only feed when previous stream enters (feed_active)
    C_FWNA_f = _get("C_FWNA","C_FWNA_in") if feed_active else 0.0
    C_AAh_f  = _get("C_AAh", "C_AAh_in") if feed_active else 0.0
    C_AA_f   = _get("AA",    "C_AA_in")      # filter uses key "AA" for recovered acid
    C_NA_f   = _get("C_NA",  "C_NA_in")

    # ---- Guard against negative concentrations (numerical noise) ----
    C_A_r    = max(C_A,    0.0)
    C_FWNA_r = max(C_FWNA, 0.0)
    C_I_r    = max(C_I,    0.0)
    C_AAh_r  = max(C_AAh,  0.0)
    C_AA_r   = max(C_AA,   0.0)
    T_safe   = max(T, 1.0)

    # ---- Rate constants (Arrhenius) ----
    k1   = p["k0_1"]   * np.exp(-p["Ea1"]   / (R * T_safe))
    k2   = p["k0_2"]   * np.exp(-p["Ea2"]   / (R * T_safe))
    k_imp = p["k0_imp"] * np.exp(-p["Ea_imp"] / (R * T_safe))

    # ---- Reaction rates (mol/L/s) ----
    r1   = k1   * (C_A_r   ** p["alpha1"]) * (C_FWNA_r ** p["beta1"])
    r2   = k2   * (C_I_r   ** p["gamma"])  * (C_AAh_r  ** p["delta"])
    r_imp = k_imp * (C_A_r  ** p["m_imp"]) * (C_AA_r   ** p["n_imp"])

    # ---- Convective term shorthand ----
    tau_inv = F / V   # 1/τ (s⁻¹)

    # ---- Mass balances ----
    dC_A_dt    = tau_inv*(C_A_f    - C_A)    - r1 - r_imp         # A consumed in step 1 and impurity
    dC_FWNA_dt = tau_inv*(C_FWNA_f - C_FWNA) - r1                 # FWNA consumed 1:1 in step 1
    dC_I_dt    = tau_inv*(0.0      - C_I)    + r1  - r2            # [I] formed in step 1, consumed in step 2
    dC_AAh_dt  = tau_inv*(C_AAh_f  - C_AAh)  - r2                  # AAh consumed 1:1 in step 2
    dC_AA_dt   = tau_inv*(C_AA_f   - C_AA)   - r_imp               # AA consumed in impurity path
    dC_NA_dt   = tau_inv*(C_NA_f   - C_NA)   + r1                  # NA produced 1:1 from FWNA in step 1
    dC_B1_dt   = tau_inv*(0.0      - C_B1)   + p["f_lb1"]*r2       # liquid B1
    dC_B2_dt   = tau_inv*(0.0      - C_B2)   + p["f_lb2"]*r2       # liquid B2
    dC_Imp_dt  = tau_inv*(0.0      - C_Imp)  + r_imp               # lumped impurity

    # ---- Solid inventories (mol in reactor) ----
    # Solids form from step 2 and leave with the outlet stream proportionally
    solid_out_B1 = tau_inv * S_B1
    solid_out_B2 = tau_inv * S_B2
    dS_B1_dt = p["f_sb1"] * r2 * V - solid_out_B1
    dS_B2_dt = p["f_sb2"] * r2 * V - solid_out_B2

    # ---- Energy balance ----
    rho, Cp = p["rho_ref"], p["Cp"]
    # Contribution from each step: –ΔHᵢ·rᵢ / (ρ·Cp)
    Q_rxn = -(p["dH1"]*r1 + p["dH2"]*r2 + p["dH_imp"]*r_imp) / (rho * Cp)
    Q_cool = -(p["UA"] / (rho * Cp * V)) * (T - p["Tc"])
    dT_dt = tau_inv*(p["Tf"] - T) + Q_rxn + Q_cool

    return [dC_A_dt, dC_FWNA_dt, dC_I_dt, dC_AAh_dt, dC_AA_dt,
            dC_NA_dt, dC_B1_dt, dC_B2_dt, dC_Imp_dt,
            dT_dt, dS_B1_dt, dS_B2_dt]


# =============================================================================
# Solve upstream units
# =============================================================================
t_span = (0, 120)
t_eval = np.linspace(*t_span, 500)

# CSTR1
y0_cstr1 = [0.0, 0.0, 0.0, 0.0, 350.0]
sol_cstr1 = solve_ivp(
    lambda t, y: cstr1_model(t, y, params_cstr1),
    t_span, y0_cstr1, t_eval=t_eval, method="RK45"
)
C_H_cstr1, C_N_cstr1, C_W_cstr1, C_A_cstr1, T_cstr1 = sol_cstr1.y
t = sol_cstr1.t
X_A_cstr1 = C_A_cstr1 / params_cstr1["C_Hf"]
rho_cstr1 = np.array([cstr1_density(ch, cn, cw, ca, params_cstr1)
                      for ch, cn, cw, ca in zip(C_H_cstr1, C_N_cstr1, C_W_cstr1, C_A_cstr1)])
print(f"Surge1 tank outlet opens at t = {params_surge1['t_open']:.2f} s")

interp_cstr1 = {k: interp1d(t, v, kind="cubic", bounds_error=False, fill_value="extrapolate")
                for k, v in zip(["C_H","C_N","C_W","C_A","T"],
                                [C_H_cstr1, C_N_cstr1, C_W_cstr1, C_A_cstr1, T_cstr1])}

# Surge1
y0_surge1 = [params_surge1["V0"], C_H_cstr1[0], C_N_cstr1[0], C_W_cstr1[0], C_A_cstr1[0], T_cstr1[0]]
sol_surge1 = solve_ivp(
    lambda tt, yy: surge1_tank_model(tt, yy, params_surge1, interp_cstr1),
    t_span, y0_surge1, t_eval=t_eval, method="RK45"
)
V_surge1, C_H_surge1, C_N_surge1, C_W_surge1, C_A_surge1, T_surge1 = sol_surge1.y

interp_surge1 = {k: interp1d(sol_surge1.t, v, kind="cubic", bounds_error=False, fill_value="extrapolate")
                 for k, v in zip(["C_H","C_N","C_W","C_A","T"],
                                 [C_H_surge1, C_N_surge1, C_W_surge1, C_A_surge1, T_surge1])}

# Crash tank
y0_crash = [0.0, 0.0, 0.0, 0.0, params_crash.get("T0", 293.15), 0.0]
sol_crash = solve_ivp(
    lambda tt, yy: crash_tank_model(tt, yy, params_crash, interp_surge1, params_surge1["t_open"]),
    t_span, y0_crash, t_eval=t_eval, method="RK45"
)
C_H_crash, C_N_crash, C_W_crash, C_A_crash, T_crash, S_crash = sol_crash.y
Tc_crash = np.full_like(sol_crash.t, params_crash["Tc"])

V_m3 = params_crash["V"] / 1000.0
M_A_kg = params_crash["M_A_gmol"] / 1000.0
rho_crash_liquid = np.array([crash_liquid_density(ch, cn, cw, ca, params_crash)
                              for ch, cn, cw, ca in zip(C_H_crash, C_N_crash, C_W_crash, C_A_crash)])
mass_liquid_kg  = rho_crash_liquid * V_m3
mass_solid_kg   = S_crash * M_A_kg
solid_content_pct = 100.0 * mass_solid_kg / (mass_solid_kg + mass_liquid_kg + 1e-12)
crash_solid_out_rate = np.where(
    sol_crash.t >= params_surge1["t_open"],
    (params_crash["F_in"] + params_crash["F_N"]) * (S_crash / params_crash["V"]),
    0.0,
)

interp_crash = {k: interp1d(sol_crash.t, v, kind="cubic", bounds_error=False, fill_value="extrapolate")
                for k, v in zip(["C_H","C_N","C_W","C_A","S"],
                                [C_H_crash, C_N_crash, C_W_crash, C_A_crash, S_crash])}

# Filter
filter_results = {k: [] for k in [
    "S_out_to_main","C_H_main","C_N_main","C_W_main","C_A_main",
    "C_H_waste","C_N_waste","C_W_waste","C_A_waste","AA_main","total_A_main"
]}
for tt in sol_crash.t:
    fs = continuous_filter_split(tt, params_filter, interp_crash)
    for k in filter_results:
        filter_results[k].append(fs[k])
for k in filter_results:
    filter_results[k] = np.array(filter_results[k])

filter_S_main_series       = filter_results["S_out_to_main"]
filter_C_A_main_series     = filter_results["C_A_main"]
filter_C_N_main_series     = filter_results["C_N_main"]
filter_C_W_main_series     = filter_results["C_W_main"]
filter_C_H_main_series     = filter_results["C_H_main"]
filter_AA_main_series      = filter_results["AA_main"]
filter_total_A_main_series = filter_results["total_A_main"]
filter_C_H_waste_series    = filter_results["C_H_waste"]
filter_C_N_waste_series    = filter_results["C_N_waste"]
filter_C_W_waste_series    = filter_results["C_W_waste"]
filter_C_A_waste_series    = filter_results["C_A_waste"]

interp_filter_main = {
    "C_A":  interp1d(sol_crash.t, filter_C_A_main_series, kind="cubic", bounds_error=False, fill_value="extrapolate"),
    "AA":   interp1d(sol_crash.t, filter_AA_main_series,  kind="cubic", bounds_error=False, fill_value="extrapolate"),
    "C_N":  interp1d(sol_crash.t, filter_C_N_main_series, kind="cubic", bounds_error=False, fill_value="extrapolate"),
    "S_A":  interp1d(sol_crash.t, filter_S_main_series,   kind="cubic", bounds_error=False, fill_value="extrapolate"),
    # FWNA and AAh come from fresh feed defined in params_cstr2 (no upstream source)
}

# =============================================================================
# Solve CSTR2 (Method 1 multi-step model)
# =============================================================================
y0_cstr2 = [
    0.0,   # C_A
    0.0,   # C_FWNA
    0.0,   # C_I  (intermediate – starts at zero)
    0.0,   # C_AAh
    0.0,   # C_AA
    0.0,   # C_NA
    0.0,   # C_B1
    0.0,   # C_B2
    0.0,   # C_Imp
    params_cstr2["Tf"],  # T
    0.0,   # S_B1
    0.0,   # S_B2
]

# Calculate filter residence time: τ = (n_filters * V_avg) / F_feed
V_avg_filter = (params_filter["V0"] + params_filter["V_max"]) / 2.0
tau_filter = (params_filter["n_filters"] * V_avg_filter) / params_filter["F_feed"]
t_feed_cstr2_start = params_filter["t_start"] + tau_filter

sol_cstr2 = solve_ivp(
    lambda t, y: cstr2_model(t, y, params_cstr2, inlet_interp=interp_filter_main, t_feed_start=t_feed_cstr2_start),
    t_span, y0_cstr2, t_eval=t_eval, method="RK45",
    rtol=1e-6, atol=1e-9,
)

# Unpack CSTR2 results
(C_A_cstr2, C_FWNA_cstr2, C_I_cstr2, C_AAh_cstr2, C_AA_cstr2,
 C_NA_cstr2, C_B1_cstr2, C_B2_cstr2, C_Imp_cstr2,
 T_cstr2, S_B1_cstr2, S_B2_cstr2) = sol_cstr2.y

# Derived quantities
C_A_in_ref = max(params_cstr2["C_A_in"], 1e-12)
X_A_cstr2         = 1.0 - (C_A_cstr2 / C_A_in_ref)
combined_B_cstr2  = C_B1_cstr2 + C_B2_cstr2 + (S_B1_cstr2 + S_B2_cstr2) / params_cstr2["V"]
X_B_cstr2         = combined_B_cstr2 / C_A_in_ref

# Selectivity: rate of B formation vs total A consumption
# (evaluated pointwise as concentration ratios; not a true instantaneous selectivity unless at SS)
with np.errstate(divide="ignore", invalid="ignore"):
    selectivity_B = np.where(
        combined_B_cstr2 + C_Imp_cstr2 > 1e-12,
        combined_B_cstr2 / (combined_B_cstr2 + C_Imp_cstr2),
        np.nan,
    )

# =============================================================================
# Surge2 model – updated for new CSTR2 state vector
# =============================================================================
params_surge2 = {
    "F_in":    params_cstr2["F"],
    "V":       200.0,   # large final holding tank
    "V0":      0.0,
    "t_open":  0.0,
    "rho_ref": params_cstr2["rho_ref"],
    "Cp":      params_cstr2["Cp"],
    "UA":      1000.0,
    "Tc":      295.0,
}


def surge2_tank_model(t, y, p, inlet_interp):
    """
    Surge2 – tracks the same 12 species as CSTR2 plus volume (13 states total).
    States: V, C_A, C_FWNA, C_I, C_AAh, C_AA, C_NA, C_B1, C_B2, C_Imp, T, S_B1, S_B2
    """
    V, C_A, C_FWNA, C_I, C_AAh, C_AA, C_NA, C_B1, C_B2, C_Imp, T, S_B1, S_B2 = y

    def _in(key):
        return float(inlet_interp[key](t))

    F_in  = p["F_in"]
    F_out = 0.0   # no outlet during simulated horizon
    dV_dt = F_in - F_out
    V_eff = max(V, 1e-9)
    if V <= 1e-9:
        return [dV_dt] + [0.0]*12

    tau_in = F_in / V_eff

    dC_A_dt    = tau_in*(_in("C_A")    - C_A)
    dC_FWNA_dt = tau_in*(_in("C_FWNA") - C_FWNA)
    dC_I_dt    = tau_in*(_in("C_I")    - C_I)
    dC_AAh_dt  = tau_in*(_in("C_AAh")  - C_AAh)
    dC_AA_dt   = tau_in*(_in("C_AA")   - C_AA)
    dC_NA_dt   = tau_in*(_in("C_NA")   - C_NA)
    dC_B1_dt   = tau_in*(_in("C_B1")   - C_B1)
    dC_B2_dt   = tau_in*(_in("C_B2")   - C_B2)
    dC_Imp_dt  = tau_in*(_in("C_Imp")  - C_Imp)
    dT_dt      = tau_in*(_in("T")      - T) \
                 - (p["UA"]/(p["rho_ref"]*p["Cp"]*V_eff))*(T - p["Tc"])

    # Solids: approximate as dilution by the growing tank
    S_B1_in_rate = F_in * (_in("S_B1") / max(params_cstr2["V"], 1e-9))
    S_B2_in_rate = F_in * (_in("S_B2") / max(params_cstr2["V"], 1e-9))
    solid_out_B1 = (F_out / V_eff) * S_B1
    solid_out_B2 = (F_out / V_eff) * S_B2
    dS_B1_dt = S_B1_in_rate - solid_out_B1
    dS_B2_dt = S_B2_in_rate - solid_out_B2

    return [dV_dt, dC_A_dt, dC_FWNA_dt, dC_I_dt, dC_AAh_dt, dC_AA_dt,
            dC_NA_dt, dC_B1_dt, dC_B2_dt, dC_Imp_dt, dT_dt, dS_B1_dt, dS_B2_dt]


# Prepare CSTR2 outlet interpolants for surge2
_cstr2_keys = ["C_A","C_FWNA","C_I","C_AAh","C_AA","C_NA","C_B1","C_B2","C_Imp","T","S_B1","S_B2"]
_cstr2_vals = [C_A_cstr2, C_FWNA_cstr2, C_I_cstr2, C_AAh_cstr2, C_AA_cstr2,
               C_NA_cstr2, C_B1_cstr2, C_B2_cstr2, C_Imp_cstr2, T_cstr2, S_B1_cstr2, S_B2_cstr2]
interp_cstr2 = {k: interp1d(sol_cstr2.t, v, kind="cubic", bounds_error=False, fill_value="extrapolate")
                for k, v in zip(_cstr2_keys, _cstr2_vals)}

y0_surge2 = [params_surge2["V0"]] + [0.0]*9 + [params_cstr2["Tf"]] + [0.0, 0.0]

sol_surge2 = solve_ivp(
    lambda tt, yy: surge2_tank_model(tt, yy, params_surge2, interp_cstr2),
    t_span, y0_surge2, t_eval=t_eval, method="RK45",
)

(V_surge2, C_A_surge2, C_FWNA_surge2, C_I_surge2, C_AAh_surge2, C_AA_surge2,
 C_NA_surge2, C_B1_surge2, C_B2_surge2, C_Imp_surge2, T_surge2, S_B1_surge2, S_B2_surge2) = sol_surge2.y

# =============================================================================
# Plots
# =============================================================================

# -- Figure 1: CSTR1 --
fig1, ax1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
ax1[0].plot(t, T_cstr1, label="CSTR1 Reactor T", color=COLOR_SCHEME["T"])
ax1[0].plot(t, np.full_like(t, params_cstr1["Tc"]), "--", color="gray", lw=1, label="Jacket Tc")
ax1[0].set_ylabel("Temperature (K)"); ax1[0].set_title("CSTR1: Temperature vs Time")
ax1[0].grid(True, alpha=0.3); ax1[0].legend()
ax1[1].plot(t, C_H_cstr1, label="C_H", color=COLOR_SCHEME["C_H"])
ax1[1].plot(t, C_N_cstr1, label="C_N", color=COLOR_SCHEME["C_N"])
ax1[1].plot(t, C_W_cstr1, label="C_W", color=COLOR_SCHEME["C_W"])
ax1[1].plot(t, C_A_cstr1, label="C_A", color=COLOR_SCHEME["C_A"])
ax1[1].set_ylabel("Concentration (mol/L)"); ax1[1].set_title("CSTR1: Concentrations vs Time")
ax1[1].grid(True, alpha=0.3); ax1[1].legend()
ax1[2].plot(t, X_A_cstr1, color="black", label="Conversion of A")
ax1[2].set_xlabel("Time (s)"); ax1[2].set_ylabel("Conversion")
ax1[2].set_title("CSTR1: A Conversion vs Time"); ax1[2].set_ylim(0, 1.05)
ax1[2].grid(True, alpha=0.3); ax1[2].legend()
plt.tight_layout()

# -- Figure 2: Surge1 --
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
ax2[0].plot(sol_surge1.t, V_surge1, color="tab:blue", label="Surge1 volume")
ax2[0].axhline(params_surge1["V"], color="red", ls="--", lw=1, label="Max volume")
ax2[0].set_ylabel("Volume (L)"); ax2[0].set_title("Surge1 Tank: Volume vs Time")
ax2[0].grid(True, alpha=0.3); ax2[0].legend()
ax2[1].plot(sol_surge1.t, C_A_surge1, label="C_A", color=COLOR_SCHEME["C_A"])
ax2[1].plot(sol_surge1.t, C_H_surge1, label="C_H", color=COLOR_SCHEME["C_H"])
ax2[1].plot(sol_surge1.t, C_N_surge1, label="C_N", color=COLOR_SCHEME["C_N"], linestyle=":")
ax2[1].plot(sol_surge1.t, C_W_surge1, label="C_W", color=COLOR_SCHEME["C_W"])
ax2[1].set_ylabel("Concentration (mol/L)"); ax2[1].set_title("Surge1 Tank: Concentrations vs Time")
ax2[1].grid(True, alpha=0.3); ax2[1].legend()
ax2[2].plot(sol_surge1.t, T_surge1, label="T (surge1)", color=COLOR_SCHEME["T"], lw=2)
ax2[2].plot(sol_surge1.t, np.full_like(sol_surge1.t, params_surge1["Tc"]), "--", color="gray", lw=1, label="Jacket Tc")
ax2[2].set_xlabel("Time (s)"); ax2[2].set_ylabel("Temperature (K)")
ax2[2].set_title("Surge1 Tank: Temperature vs Time"); ax2[2].grid(True, alpha=0.3); ax2[2].legend()
plt.tight_layout()

# -- Figure 3: Crash tank --
fig3, ax3 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
ax3[0].plot(sol_crash.t, solid_content_pct, color="brown", label="Solid content (%)")
ax3[0].set_ylabel("Solid content (%)"); ax3[0].set_title("Crash Tank: Solid content vs Time")
ax3[0].grid(True, alpha=0.3); ax3[0].legend()
ax3_0b = ax3[0].twinx()
ax3_0b.plot(sol_crash.t, crash_solid_out_rate, color="black", ls="--", label="Solids out to filter")
ax3_0b.set_ylabel("Solids out rate (mol/s)"); ax3_0b.legend(loc="upper right")
ax3[1].plot(sol_crash.t, C_A_crash, label="C_A", lw=2, color=COLOR_SCHEME["C_A"])
ax3[1].plot(sol_crash.t, C_N_crash, label="C_N", lw=2, color=COLOR_SCHEME["C_N"])
ax3[1].plot(sol_crash.t, C_H_crash, label="C_H", alpha=0.7, color=COLOR_SCHEME["C_H"])
ax3[1].plot(sol_crash.t, C_W_crash, label="C_W", alpha=0.7, color=COLOR_SCHEME["C_W"])
ax3[1].set_ylabel("Concentration (mol/L)"); ax3[1].set_title("Crash Tank: Concentrations vs Time")
ax3[1].grid(True, alpha=0.3); ax3[1].legend()
ax3[2].plot(sol_crash.t, T_crash, label="T (crash)")
ax3[2].plot(sol_crash.t, Tc_crash, "--", label="Jacket Tc")
ax3[2].set_xlabel("Time (s)"); ax3[2].set_ylabel("Temperature (K)")
ax3[2].set_title("Crash Tank: Temperature vs Time"); ax3[2].grid(True, alpha=0.3); ax3[2].legend()
plt.tight_layout()

# -- Figure 4: Filter --
fig4, ax4 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
ax4[0].plot(sol_crash.t, filter_AA_main_series,      label="AA recovered (main)",      lw=2, color="blue")
ax4[0].plot(sol_crash.t, filter_S_main_series,       label="Solid A to main (mol)",    lw=2, color="red")
ax4[0].plot(sol_crash.t, filter_total_A_main_series, label="Total A in main", lw=1, ls="--", color="darkred")
ax4[0].set_ylabel("Amount (mol or mol/L)"); ax4[0].set_title("Filter: Main Stream"); ax4[0].grid(True, alpha=0.3); ax4[0].legend()
ax4[1].plot(sol_crash.t, filter_C_N_main_series, label="C_N (main)", color=COLOR_SCHEME["C_N"])
ax4[1].plot(sol_crash.t, filter_C_W_main_series, label="C_W (main)", color=COLOR_SCHEME["C_W"])
ax4[1].plot(sol_crash.t, filter_C_A_main_series, label="C_A (main)", color=COLOR_SCHEME["C_A"])
ax4[1].set_ylabel("Concentration (mol/L)"); ax4[1].set_title("Filter: Main Stream Trace Concentrations")
ax4[1].grid(True, alpha=0.3); ax4[1].legend()
ax4[2].plot(sol_crash.t, filter_C_H_waste_series, label="C_H (waste)", color=COLOR_SCHEME["C_H"])
ax4[2].plot(sol_crash.t, filter_C_N_waste_series, label="C_N (waste)", color=COLOR_SCHEME["C_N"])
ax4[2].plot(sol_crash.t, filter_C_W_waste_series, label="C_W (waste)", color=COLOR_SCHEME["C_W"])
ax4[2].plot(sol_crash.t, filter_C_A_waste_series, label="C_A (waste)", color=COLOR_SCHEME["C_A"])
ax4[2].set_xlabel("Time (s)"); ax4[2].set_ylabel("Concentration (mol/L)")
ax4[2].set_title("Filter: Waste Stream Composition"); ax4[2].grid(True, alpha=0.3); ax4[2].legend()
plt.tight_layout()

# -- Figure 5: CSTR2 – Method 1 kinetics --
fig5, axes5 = plt.subplots(4, 1, figsize=(11, 16), sharex=True)

# Panel 1: Temperature
axes5[0].plot(sol_cstr2.t, T_cstr2, label="CSTR2 T", color=COLOR_SCHEME["T"], lw=2)
axes5[0].plot(sol_cstr2.t, np.full_like(sol_cstr2.t, params_cstr2["Tc"]), "--", color="gray", lw=1, label="Jacket Tc")
axes5[0].set_ylabel("Temperature (K)")
axes5[0].set_title("CSTR2 (Method 1 – Multi-Step Surrogate): Temperature")
axes5[0].grid(True, alpha=0.3); axes5[0].legend()

# Panel 2: Reactants + intermediate
axes5[1].plot(sol_cstr2.t, C_A_cstr2,    label="C_A (substrate)",  color=COLOR_SCHEME["C_A"],   lw=2)
axes5[1].plot(sol_cstr2.t, C_FWNA_cstr2, label="C_FWNA",           color=COLOR_SCHEME["C_W"],   lw=2)
axes5[1].plot(sol_cstr2.t, C_I_cstr2,    label="C_[I] intermediate",color=COLOR_SCHEME["C_I"],  lw=2, ls="--")
axes5[1].plot(sol_cstr2.t, C_AAh_cstr2,  label="C_AAh",             color=COLOR_SCHEME["C_AAh"],lw=1.5)
axes5[1].plot(sol_cstr2.t, C_AA_cstr2,   label="C_AA",              color=COLOR_SCHEME["AA"],   lw=1.5)
axes5[1].set_ylabel("Concentration (mol/L)")
axes5[1].set_title("CSTR2: Reactants and Intermediate [I]")
axes5[1].grid(True, alpha=0.3); axes5[1].legend()

# Panel 3: Products + impurity
axes5[2].plot(sol_cstr2.t, C_B1_cstr2,  label="C_B1 (liq)",       color="#006400", lw=2)
axes5[2].plot(sol_cstr2.t, C_B2_cstr2,  label="C_B2 (liq)",       color="#228B22", lw=2)
axes5[2].plot(sol_cstr2.t, C_NA_cstr2,  label="C_NA (nitric acid)",color=COLOR_SCHEME["C_NA"],  lw=1.5, ls="-.")
axes5[2].plot(sol_cstr2.t, C_Imp_cstr2, label="C_Imp (lumped)",    color=COLOR_SCHEME["C_Imp"], lw=1.5, ls=":")
ax5_twin = axes5[2].twinx()
ax5_twin.plot(sol_cstr2.t, S_B1_cstr2 / params_cstr2["V"], label="S_B1 (mol/L)", color="#8B0000", ls=":")
ax5_twin.plot(sol_cstr2.t, S_B2_cstr2 / params_cstr2["V"], label="S_B2 (mol/L)", color="#FF4500", ls=":")
ax5_twin.set_ylabel("Solids (mol/L)"); ax5_twin.legend(loc="upper right")
axes5[2].set_ylabel("Concentration (mol/L)")
axes5[2].set_title("CSTR2: Products, NA, and Lumped Impurity")
axes5[2].grid(True, alpha=0.3); axes5[2].legend(loc="upper left")

# Panel 4: Conversion + selectivity
axes5[3].plot(sol_cstr2.t, X_A_cstr2,        color="black",     label="Conversion of A (X_A)", lw=2)
axes5[3].plot(sol_cstr2.t, X_B_cstr2,        color="darkgreen", label="Combined B1+B2 yield", lw=2, ls="--")
axes5[3].plot(sol_cstr2.t, selectivity_B,    color="purple",    label="Selectivity S = B/(B+Imp)", lw=1.5, ls="-.")
axes5[3].set_xlabel("Time (s)"); axes5[3].set_ylabel("Fraction")
axes5[3].set_title("CSTR2: Conversion, Yield, and B-Selectivity vs Time")
axes5[3].set_ylim(-0.05, 1.15); axes5[3].grid(True, alpha=0.3); axes5[3].legend()

plt.tight_layout()

# -- Figure 6: Surge2 --
fig6, ax6 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
ax6[0].plot(sol_surge2.t, V_surge2, color="tab:blue", label="Surge2 volume")
ax6[0].axhline(params_surge2["V"], color="red", ls="--", lw=1, label="Max volume")
ax6[0].set_ylabel("Volume (L)"); ax6[0].set_title("Surge2 Tank: Volume vs Time")
ax6[0].grid(True, alpha=0.3); ax6[0].legend()
ax6[1].plot(sol_surge2.t, C_A_surge2,   label="C_A",   color=COLOR_SCHEME["C_A"])
ax6[1].plot(sol_surge2.t, C_B1_surge2,  label="C_B1",  color="#006400")
ax6[1].plot(sol_surge2.t, C_B2_surge2,  label="C_B2",  color="#228B22")
ax6[1].plot(sol_surge2.t, C_NA_surge2,  label="C_NA",  color=COLOR_SCHEME["C_NA"], ls="-.")
ax6[1].plot(sol_surge2.t, C_Imp_surge2, label="C_Imp", color=COLOR_SCHEME["C_Imp"], ls=":")
ax6b = ax6[1].twinx()
ax6b.plot(sol_surge2.t, S_B1_surge2 / np.maximum(V_surge2, 1e-9), label="S_B1 (mol/L)", color="#8B0000", ls=":")
ax6b.plot(sol_surge2.t, S_B2_surge2 / np.maximum(V_surge2, 1e-9), label="S_B2 (mol/L)", color="#FF4500", ls=":")
ax6b.set_ylabel("Solids (mol/L)"); ax6b.legend(loc="upper right")
ax6[1].set_ylabel("Concentration (mol/L)"); ax6[1].set_title("Surge2: Concentrations vs Time")
ax6[1].grid(True, alpha=0.3); ax6[1].legend(loc="upper left")
ax6[2].plot(sol_surge2.t, T_surge2, label="T (surge2)", color=COLOR_SCHEME["T"])
ax6[2].plot(sol_surge2.t, np.full_like(sol_surge2.t, params_surge2["Tc"]), "--", color="gray", lw=1, label="Jacket Tc")
ax6[2].set_xlabel("Time (s)"); ax6[2].set_ylabel("Temperature (K)")
ax6[2].set_title("Surge2: Temperature vs Time"); ax6[2].grid(True, alpha=0.3); ax6[2].legend()
plt.tight_layout()

# =============================================================================
# Steady-state summary tables
# =============================================================================
n_ss = max(1, int(0.2 * len(sol_cstr1.t)))

def ss(arr): return tail_mean(arr, n_ss)

# Upstream flowsheet (unchanged columns)
flowsheet_data = {
    "Stream / Unit":   ["CSTR1 Outlet","Surge1 Outlet","Crash Outlet","Filter Main","Filter Waste"],
    "C_H (mol/L)":     [f"{ss(C_H_cstr1):.4f}", f"{ss(C_H_surge1):.4f}", f"{ss(C_H_crash):.4f}", f"{ss(filter_C_H_main_series):.4f}", f"{ss(filter_C_H_waste_series):.4f}"],
    "C_N (mol/L)":     [f"{ss(C_N_cstr1):.4f}", f"{ss(C_N_surge1):.4f}", f"{ss(C_N_crash):.4f}", f"{ss(filter_C_N_main_series):.4f}", f"{ss(filter_C_N_waste_series):.4f}"],
    "C_W (mol/L)":     [f"{ss(C_W_cstr1):.4f}", f"{ss(C_W_surge1):.4f}", f"{ss(C_W_crash):.4f}", f"{ss(filter_C_W_main_series):.4f}", f"{ss(filter_C_W_waste_series):.4f}"],
    "C_A (mol/L)":     [f"{ss(C_A_cstr1):.4f}", f"{ss(C_A_surge1):.4f}", f"{ss(C_A_crash):.4f}", f"{ss(filter_C_A_main_series):.4f}", f"{ss(filter_C_A_waste_series):.4f}"],
    "S_A (mol solid)": ["—","—", f"{ss(S_crash):.4f}", f"{ss(filter_S_main_series):.4f}","—"],
    "T (K)":           [f"{ss(T_cstr1):.2f}", f"{ss(T_surge1):.2f}", f"{ss(T_crash):.2f}","—","—"],
    "AA (mol)":        ["—","—","—", f"{ss(filter_AA_main_series):.4f}","—"],
}
df1 = pd.DataFrame(flowsheet_data)

# CSTR2 / Surge2 flowsheet (new columns for Method 1 species)
flowsheet2_data = {
    "Stream / Unit":      ["CSTR2 Outlet","Surge2 Outlet"],
    "C_A (mol/L)":        [f"{ss(C_A_cstr2):.4f}",    f"{ss(C_A_surge2):.4f}"],
    "C_FWNA (mol/L)":     [f"{ss(C_FWNA_cstr2):.4f}", f"{ss(C_FWNA_surge2):.4f}"],
    "C_[I] (mol/L)":      [f"{ss(C_I_cstr2):.6f}",    f"{ss(C_I_surge2):.6f}"],
    "C_NA (mol/L)":       [f"{ss(C_NA_cstr2):.4f}",   f"{ss(C_NA_surge2):.4f}"],
    "C_B1 liq (mol/L)":   [f"{ss(C_B1_cstr2):.4f}",   f"{ss(C_B1_surge2):.4f}"],
    "C_B2 liq (mol/L)":   [f"{ss(C_B2_cstr2):.4f}",   f"{ss(C_B2_surge2):.4f}"],
    "S_B1 (mol)":         [f"{ss(S_B1_cstr2):.4f}",   f"{ss(S_B1_surge2):.4f}"],
    "S_B2 (mol)":         [f"{ss(S_B2_cstr2):.4f}",   f"{ss(S_B2_surge2):.4f}"],
    "C_Imp (mol/L)":      [f"{ss(C_Imp_cstr2):.5f}",  f"{ss(C_Imp_surge2):.5f}"],
    "T (K)":              [f"{ss(T_cstr2):.2f}",       f"{ss(T_surge2):.2f}"],
    "X_A":                [f"{ss(X_A_cstr2):.4f}",     "—"],
    "Selectivity_B":      [f"{float(np.nanmean(selectivity_B[-n_ss:])):.4f}", "—"],
}
df2 = pd.DataFrame(flowsheet2_data)

W = 130
print("\n" + "="*W)
print("STEADY-STATE FLOWSHEET – Upstream (averaged over final 20% of simulation)")
print("="*W)
print(df1.to_string(index=False))
print("="*W)

print("\n" + "="*W)
print("STEADY-STATE FLOWSHEET – CSTR2 / Surge2  [Method 1: Multi-Step Surrogate]")
print("="*W)
print(df2.to_string(index=False))
print("="*W)

print(f"""
CSTR2 KINETIC MODEL SUMMARY (Method 1 – Multi-Step Mechanistic Surrogate)
--------------------------------------------------------------------------
  Step 1  A + FWNA  →  [I] + NA
          r1  = k1(T)·C_A^{params_cstr2['alpha1']}·C_FWNA^{params_cstr2['beta1']}
          k0_1={params_cstr2['k0_1']:.2e}  Ea1={params_cstr2['Ea1']/1000:.1f} kJ/mol  ΔH1={params_cstr2['dH1']/1000:.1f} kJ/mol

  Step 2  [I] + AAh  →  B1(l/s) + B2(l/s)
          r2  = k2(T)·C_[I]^{params_cstr2['gamma']}·C_AAh^{params_cstr2['delta']}
          k0_2={params_cstr2['k0_2']:.2e}  Ea2={params_cstr2['Ea2']/1000:.1f} kJ/mol  ΔH2={params_cstr2['dH2']/1000:.1f} kJ/mol
          Product splits: f_lb1={params_cstr2['f_lb1']} f_lb2={params_cstr2['f_lb2']} f_sb1={params_cstr2['f_sb1']} f_sb2={params_cstr2['f_sb2']}

  Impurity (lumped, parallel):  A + AA  →  Imp
          r_i = k_i(T)·C_A^{params_cstr2['m_imp']}·C_AA^{params_cstr2['n_imp']}
          k0_i={params_cstr2['k0_imp']:.2e}  Ea_i={params_cstr2['Ea_imp']/1000:.1f} kJ/mol  ΔH_i={params_cstr2['dH_imp']/1000:.1f} kJ/mol

  Ea_imp ({params_cstr2['Ea_imp']/1000:.0f} kJ/mol) > Ea1 ({params_cstr2['Ea1']/1000:.0f} kJ/mol) > Ea2 ({params_cstr2['Ea2']/1000:.0f} kJ/mol)
  → Impurity selectivity worsens faster than main reaction with rising T.
  → Optimal T operating window: maximize r2 / r_imp ratio.
""")

plt.show()