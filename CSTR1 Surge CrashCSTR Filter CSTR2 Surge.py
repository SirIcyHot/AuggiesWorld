#Created by Augustus Sison on 2024-06-05


# TODO: replace the static density values with temperature-dependent density correlations for each unit.
# TODO: send the waste stream to extractive distillation and convert it to reusable liq N later.
# TODO: crash tank needs more robust handling of supersaturation and solids inventory management.
# TODO: temperature-dependent crystallization kinetics should affect solids formation rate (currently ignored).
# TODO: filter model is very simplified - could be improved with more realistic semi-batch scheduling or a continuous filtration model with dynamic stream splits based on solids loading and filter capacity.
# TODO: Stream split fractions in the filter are currently hardcoded - could be made dynamic based on concentrations, solids loading, or other factors.
# TODO: Unit operations dependent on surge tank active or not, not dependent on previous reactor residence time and travel time (currently surge tank just starts filling at t_open but could have more complex logic).


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd

# -----------------------------
# Parameters
# -----------------------------
params_cstr1 = {
    # Reactor
    "F": 1.0,        # volumetric flow rate (L/s)
    "V": 10.0,       # reactor volume (L)

    # Feed concentrations (mol/L)
    "C_Hf": 2.0,
    "C_Nf": 2.0,
    "C_Wf": 5.0,
    "C_Af": 0.0,

    # Temperatures (K)
    "Tf": 350.0,
    "Tc": 300.0,

    # Physical properties
    "rho_ref": 1000.0,   # kg/m^3 bulk reference density
    "Cp": 4.18,      # J/(g*K) or adjust units consistently
    "species_rho": {
        "H": 980.0,
        "N": 1020.0,
        "W": 1000.0,
        "A": 1100.0,
        "solid_A": 1500.0,
    },

    # Heat transfer
    "UA": 5000.0,    # J/(s*K)

    # Kinetics
    "k0": 1e6,       # pre-exponential factor
    "Ea": 50000.0,   # J/mol
    "R": 8.314,      # J/mol-K

    # Reaction enthalpy
    "dH": -100000.0   # J/mol (negative = exothermic)
}

# -----------------------------
# Surge1 tank parameters
# -----------------------------
params_surge1 = {
    "F_in": params_cstr1["F"],
    "V": 10.0,
    "V0": 0.0,
    "t_open": 0.8 * 10.0 / params_cstr1["F"],
    "rho_ref": 995.0,
    "Cp": params_cstr1["Cp"],
    "species_rho": params_cstr1["species_rho"],
    "UA": 2000.0,
    "Tc": 298.15,
}

# -----------------------------
# Crash tank parameters
# -----------------------------
params_crash = {
    "F_in": params_surge1["F_in"],   # inlet flow from surge1 tank (L/s)
    "F_N": 0.2,                      # additional N feed (L/s)
    "V": 5.0,                        # crash tank volume (L)
    "Tc": 290.0,                     # jacket temperature (K)
    "UA": 3000.0,
    "rho_ref": 1100.0,
    "Cp": params_cstr1["Cp"],
    "species_rho": params_cstr1["species_rho"],
    # precipitation kinetics
    "k_p": 0.5,            # 1/s per concentration unit (phenomenological)
    "C_A_eq": 0.05,        # saturation concentration (mol/L)
    # external N feed concentration (pure N)
    "C_N_ext": 5.0,
    "T_N": 293.15,
    # molar mass for A (g/mol) used for solid mass calc
    "M_A_gmol": 100.0,
    # ambient/start temperature for crash tank (K)
    "T0": 293.15,
    # heat of precipitation (J/mol) (negative = exothermic)
    "dH_precip": -5000.0,
}

# -----------------------------
# Semi-batch filter parameters
# -----------------------------
params_filter = {
    "n_filters": 3,
    "t_start": params_surge1["t_open"],
    "t_fill": 1.0,
    "t_vacuum": 1.0,
    "t_redissolve": 1.0,
    "V0": 0.8,
    "V_max": 2.0,
    "F_feed": params_crash["F_in"] + params_crash["F_N"],
    "AA_ratio": 1.25,
    "trace_A_fraction": 0.05,
    # dynamic split parameters
    "f_N_main_base": 0.02,
    "f_W_main_base": 0.01,
    "f_A_main_base": 0.15,
    "alpha_solid": 0.2,
    "solid_scale": 1.0,
}
params_filter["t_cycle"] = (
    params_filter["t_fill"] + params_filter["t_vacuum"] + params_filter["t_redissolve"]
)

# Consistent color scheme for all plots
COLOR_SCHEME = {
    'C_H': '#1f77b4',  # blue
    'C_N': '#ff7f0e',  # orange
    'C_W': '#2ca02c',  # green
    'C_A': '#d62728',  # red
    'C_AAh': '#e377c2', # magenta
    'S': '#9467bd',    # purple (solids)
    'T': '#8c564b',    # brown
    'AA': '#17becf',   # cyan
}


def mixture_density_liquid(composition, species_rho):
    """Approximate bulk liquid density using a concentration-weighted specific-volume blend."""
    total_concentration = sum(composition.values())
    if total_concentration <= 0.0:
        return 1.0

    specific_volume = 0.0
    for species, concentration in composition.items():
        rho_i = species_rho[species]
        specific_volume += (concentration / total_concentration) / rho_i

    return 1.0 / max(specific_volume, 1e-12)


def liquid_composition_from_molar(c_h, c_n, c_w, c_a):
    """Map liquid component concentrations to the composition used for density blending."""
    return {
        "H": c_h,
        "N": c_n,
        "W": c_w,
        "A": c_a,
    }


def crash_mixture_density(c_h, c_n, c_w, c_a, solid_moles_a, p, tank_volume_l):
    liquid_composition = liquid_composition_from_molar(c_h, c_n, c_w, c_a)
    liquid_density = mixture_density_liquid(liquid_composition, p["species_rho"])
    liquid_mass = liquid_density * (tank_volume_l / 1000.0)
    solid_mass = solid_moles_a * (p["M_A_gmol"] / 1000.0)
    bulk_density = (liquid_mass + solid_mass) / max(tank_volume_l / 1000.0, 1e-12)

    return bulk_density

# -----------------------------
# CSTR1 model
# -----------------------------
def cstr1_model(t, y, p):
    C_H, C_N, C_W, C_A, T = y

    # Unpack parameters
    F, V = p["F"], p["V"]
    Tf, Tc = p["Tf"], p["Tc"]
    rho, Cp = p["rho_ref"], p["Cp"]
    UA = p["UA"]
    k0, Ea, R = p["k0"], p["Ea"], p["R"]
    dH = p["dH"]

    C_Hf, C_Nf, C_Wf, C_Af = p["C_Hf"], p["C_Nf"], p["C_Wf"], p["C_Af"]

    # Reaction rate
    k = k0 * np.exp(-Ea / (R * T))
    r = k * C_H * C_N * C_W

    # Mass balances
    dC_H_dt = (F/V)*(C_Hf - C_H) - r
    dC_N_dt = (F/V)*(C_Nf - C_N) - r
    dC_W_dt = (F/V)*(C_Wf - C_W) - r
    dC_A_dt = (F/V)*(C_Af - C_A) + r

    # Energy balance
    dT_dt = (F/V)*(Tf - T) \
            - (dH/(rho * Cp)) * r \
            - (UA/(rho * Cp * V)) * (T - Tc)

    return [dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt]


def cstr1_density(c_h, c_n, c_w, c_a, p):
    composition = liquid_composition_from_molar(c_h, c_n, c_w, c_a)
    return mixture_density_liquid(composition, p["species_rho"])


# -----------------------------
# Surge1 tank model
# -----------------------------
def surge1_tank_model(t, y, p, inlet_interp):
    V, C_H, C_N, C_W, C_A, T = y

    C_H_in = inlet_interp["C_H"](t)
    C_N_in = inlet_interp["C_N"](t)
    C_W_in = inlet_interp["C_W"](t)
    C_A_in = inlet_interp["C_A"](t)
    T_in = inlet_interp["T"](t)

    F_in = p["F_in"]
    F_out = F_in if t >= p["t_open"] else 0.0

    dV_dt = F_in - F_out
    
    # Avoid unstable dynamics at near-zero holdup; compositions are undefined at V~0.
    if V <= 1e-9:
        return [dV_dt, 0.0, 0.0, 0.0, 0.0, 0.0]
    dC_H_dt = (F_in / V) * (C_H_in - C_H)
    dC_N_dt = (F_in / V) * (C_N_in - C_N)
    dC_W_dt = (F_in / V) * (C_W_in - C_W)
    dC_A_dt = (F_in / V) * (C_A_in - C_A)
    
    # Energy balance with jacket cooling
    rho = p["rho_ref"]
    dT_dt = (F_in / V) * (T_in - T) - (p["UA"] / (rho * p["Cp"] * V)) * (T - p["Tc"])

    return [dV_dt, dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt]


def surge1_density(c_h, c_n, c_w, c_a, p):
    composition = liquid_composition_from_molar(c_h, c_n, c_w, c_a)
    return mixture_density_liquid(composition, p["species_rho"])


# -----------------------------
# Crash tank model
# -----------------------------
def crash_tank_model(t, y, p, inlet_interp, t_open):
    C_H, C_N, C_W, C_A, T, S = y

    surge_active = t >= t_open

    if not surge_active:
        F_in = 0.0
        F_N = 0.0
    else:
        F_in = p["F_in"]
        F_N = p["F_N"]

    F_tot = F_in + F_N

    if F_tot <= 1e-12:
        r_p = p["k_p"] * max(C_A - p["C_A_eq"], 0.0)
        return [0.0, 0.0, 0.0, -r_p, 0.0, 0.0]

    C_H_in = inlet_interp["C_H"](t)
    C_N_in = inlet_interp["C_N"](t)
    C_W_in = inlet_interp["C_W"](t)
    C_A_in = inlet_interp["C_A"](t)
    T_in = inlet_interp["T"](t)

    # mixed inlet concentrations (weighted by flow)
    C_Hf = (F_in * C_H_in + F_N * 0.0) / F_tot
    C_Nf = (F_in * C_N_in + F_N * p["C_N_ext"]) / F_tot
    C_Wf = (F_in * C_W_in + F_N * 0.0) / F_tot
    C_Af = (F_in * C_A_in + F_N * 0.0) / F_tot
    Tf = (F_in * T_in + F_N * p["T_N"]) / F_tot

    # precipitation (simple phenomenological rate)
    r_p = p["k_p"] * max(C_A - p["C_A_eq"], 0.0)

    dC_H_dt = (F_tot / p["V"]) * (C_Hf - C_H)
    dC_N_dt = (F_tot / p["V"]) * (C_Nf - C_N)
    dC_W_dt = (F_tot / p["V"]) * (C_Wf - C_W)
    dC_A_dt = (F_tot / p["V"]) * (C_Af - C_A) - r_p

    # solids leave with the stream to the filter once the outlet is active
    solid_out = 0.0
    if surge_active:
        solid_conc = max(S, 0.0) / max(p["V"], 1e-12)
        solid_out = F_tot * solid_conc

    # solid accumulation (moles of A solid)
    dS_dt = r_p * p["V"] - solid_out

    # energy balance (TODO no heat of precipitation included here, and no jacket?)
    rho_mix = crash_mixture_density(C_H, C_N, C_W, C_A, S, p, p["V"])
    # include heat from precipitation (r_p is mol/L/s; multiply by V to get mol/s if needed)
    # Q_precip (J/s) = -dH_precip * r_p * V; term in dT_dt = Q_precip / (rho_mix * Cp * V) = -(dH_precip * r_p) / (rho_mix * Cp)
    dT_dt = (F_tot / p["V"]) * (Tf - T) - (p["UA"] / (rho_mix * p["Cp"] * p["V"])) * (T - p["Tc"]) - (p.get("dH_precip", 0.0) * r_p) / (rho_mix * p["Cp"])

    return [dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt, dS_dt]


def crash_liquid_density(c_h, c_n, c_w, c_a, p):
    composition = liquid_composition_from_molar(c_h, c_n, c_w, c_a)
    return mixture_density_liquid(composition, p["species_rho"])


def filter_cycle_state(t, p):
    if t < p["t_start"]:
        return 0, "idle", 0.0

    elapsed = t - p["t_start"]
    cycle_index = int(elapsed // p["t_cycle"])
    phase_time = elapsed % p["t_cycle"]
    filter_number = (cycle_index % p["n_filters"]) + 1

    if phase_time < p["t_fill"]:
        return filter_number, "fill", phase_time
    if phase_time < p["t_fill"] + p["t_vacuum"]:
        return filter_number, "vacuum", phase_time - p["t_fill"]
    return filter_number, "redissolve", phase_time - p["t_fill"] - p["t_vacuum"]


# def filter_cycle_state(t, p):
#     """OLD: Rotating semi-batch filter schedule - replaced with continuous filtration."""
#     if t < p["t_start"]:
#         return 0, "idle", 0.0
#     elapsed = t - p["t_start"]
#     cycle_index = int(elapsed // p["t_cycle"])
#     phase_time = elapsed % p["t_cycle"]
#     filter_number = (cycle_index % p["n_filters"]) + 1
#     if phase_time < p["t_fill"]:
#         return filter_number, "fill", phase_time
#     if phase_time < p["t_fill"] + p["t_vacuum"]:
#         return filter_number, "vacuum", phase_time - p["t_fill"]
#     return filter_number, "redissolve", phase_time - p["t_fill"] - p["t_vacuum"]


def continuous_filter_split(t, p, inlet_interp):
    """Simplified continuous filtration: separate solids and replace liquids with AA.
    Main stream carries dissolved A in AA plus trace N, W, A.
    Waste stream carries remaining liquids with low A.
    """
    surge_active = t >= p["t_start"]
    
    if not surge_active:
        return {
            "phase": "inactive",
            "S_out_to_main": 0.0,
            "C_H_main": 0.0,
            "C_N_main": 0.0,
            "C_W_main": 0.0,
            "C_A_main": 0.0,
            "C_H_waste": 0.0,
            "C_N_waste": 0.0,
            "C_W_waste": 0.0,
            "C_A_waste": 0.0,
            "AA_main": 0.0,
            "total_A_main": 0.0,
        }
    
    C_H_in = float(inlet_interp["C_H"](t))
    C_N_in = float(inlet_interp["C_N"](t))
    C_W_in = float(inlet_interp["C_W"](t))
    C_A_in = float(inlet_interp["C_A"](t))
    S_in = float(inlet_interp["S"](t))
    
    # Solids are separated and go to main stream (dissolved in AA)
    solid_A_main = max(S_in, 0.0)

    # Dynamic split fractions based on solids loading (S_in)
    s_frac = S_in / (S_in + p.get("solid_scale", 1.0)) if S_in >= 0.0 else 0.0
    f_N_main = float(np.clip(p.get("f_N_main_base", 0.02) + p.get("alpha_solid", 0.1) * s_frac, 0.0, 0.9))
    f_W_main = float(np.clip(p.get("f_W_main_base", 0.01) + 0.5 * p.get("alpha_solid", 0.1) * s_frac, 0.0, 0.9))
    # as solids increase, less liquid A remains in main (more becomes solid/AA)
    f_A_main = float(np.clip(p.get("f_A_main_base", 0.15) * (1.0 - 0.5 * s_frac), 0.0, 1.0))

    # Trace H in main is negligible after AA dissolution
    C_H_main = 0.0
    C_N_main = f_N_main * C_N_in
    C_W_main = f_W_main * C_W_in
    C_A_main = f_A_main * C_A_in  # residual liquid A in main stream

    # Waste stream carries the remainder
    C_H_waste = C_H_in
    C_N_waste = max(0.0, (1.0 - f_N_main)) * C_N_in
    C_W_waste = max(0.0, (1.0 - f_W_main)) * C_W_in
    C_A_waste = max(0.0, (1.0 - f_A_main)) * C_A_in  # remainder of liquid A goes to waste
    
    AA_main = p["AA_ratio"] * solid_A_main
    total_A_main = solid_A_main + C_A_main
    
    return {
        "phase": "continuous",
        "S_out_to_main": solid_A_main,
        "C_H_main": C_H_main,
        "C_N_main": C_N_main,
        "C_W_main": C_W_main,
        "C_A_main": C_A_main,
        "C_H_waste": C_H_waste,
        "C_N_waste": C_N_waste,
        "C_W_waste": C_W_waste,
        "C_A_waste": C_A_waste,
        "AA_main": AA_main,
        "total_A_main": total_A_main,
    }


# OLD: def filter_stream_split(t, p, inlet_interp): [complex rotating semi-batch logic - now replaced]

# -----------------------------
# Initial Conditions 
# -----------------------------
y0_cstr1 = [
    0.0,   # C_H (start empty)
    0.0,   # C_N (start empty)
    0.0,   # C_W (start empty)
    0.0,   # C_A (start empty)
    350.0  # T (start at feed temperature)
]

# -----------------------------
# Time Span
# -----------------------------
t_span = (0, 120)
t_eval = np.linspace(*t_span, 500)

# -----------------------------
# Solve CSTR1
# -----------------------------
sol_cstr1 = solve_ivp(
    fun=lambda t, y: cstr1_model(t, y, params_cstr1),
    t_span=t_span,
    y0=y0_cstr1,
    t_eval=t_eval,
    method='RK45'
)

# -----------------------------
# Extract CSTR1 results
# -----------------------------
C_H_cstr1, C_N_cstr1, C_W_cstr1, C_A_cstr1, T_cstr1 = sol_cstr1.y
t = sol_cstr1.t
Tc_cstr1 = np.full_like(t, params_cstr1["Tc"])
# Use A formation relative to the initial H feed so the conversion plot tracks product formation.
X_A_cstr1 = C_A_cstr1 / params_cstr1["C_Hf"]
rho_cstr1 = np.array([
    cstr1_density(ch, cn, cw, ca, params_cstr1)
    for ch, cn, cw, ca in zip(C_H_cstr1, C_N_cstr1, C_W_cstr1, C_A_cstr1)
])

# Example: print final values
print(f"Surge1 tank outlet opens at t = {params_surge1['t_open']:.2f} s")

# -----------------------------
# Prepare surge1 tank inlet inputs (interpolate CSTR1 outlet)
# -----------------------------
interp_cstr1 = {
    "C_H": interp1d(t, C_H_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_N": interp1d(t, C_N_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_W": interp1d(t, C_W_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_A": interp1d(t, C_A_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "T": interp1d(t, T_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
}

# -----------------------------
# Solve surge1 tank #TODO unit ops based on surge tank active of not, not dependent on previous reactor residence time and travel time
# -----------------------------
y0_surge1 = [
    params_surge1["V0"],
    C_H_cstr1[0],
    C_N_cstr1[0],
    C_W_cstr1[0],
    C_A_cstr1[0],
    T_cstr1[0],
]

sol_surge1 = solve_ivp(
    fun=lambda tt, yy: surge1_tank_model(tt, yy, params_surge1, interp_cstr1),
    t_span=t_span,
    y0=y0_surge1,
    t_eval=t_eval,
    method='RK45'
)

# Extract surge1 tank results
V_surge1, C_H_surge1, C_N_surge1, C_W_surge1, C_A_surge1, T_surge1 = sol_surge1.y
rho_surge1 = np.array([
    surge1_density(ch, cn, cw, ca, params_surge1)
    for ch, cn, cw, ca in zip(C_H_surge1, C_N_surge1, C_W_surge1, C_A_surge1)
])

# Prepare crash tank inputs from the surge1 tank outlet
interp_surge1 = {
    "C_H": interp1d(sol_surge1.t, C_H_surge1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_N": interp1d(sol_surge1.t, C_N_surge1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_W": interp1d(sol_surge1.t, C_W_surge1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_A": interp1d(sol_surge1.t, C_A_surge1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "T": interp1d(sol_surge1.t, T_surge1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
}

# -----------------------------
# Solve crash tank
# -----------------------------
y0_crash = [0.0, 0.0, 0.0, 0.0, params_crash.get("T0", 293.15), 0.0]

sol_crash = solve_ivp(
    fun=lambda tt, yy: crash_tank_model(tt, yy, params_crash, interp_surge1, params_surge1["t_open"]),
    t_span=t_span,
    y0=y0_crash,
    t_eval=t_eval,
    method='RK45'
)

# Extract crash tank results
C_H_crash, C_N_crash, C_W_crash, C_A_crash, T_crash, S_crash = sol_crash.y
Tc_crash = np.full_like(sol_crash.t, params_crash["Tc"])

# Compute solid loading/content% by mass
V_m3 = params_crash['V'] / 1000.0
M_A_kg_per_mol = params_crash['M_A_gmol'] / 1000.0
rho_crash_liquid = np.array([
    crash_liquid_density(ch, cn, cw, ca, params_crash)
    for ch, cn, cw, ca in zip(C_H_crash, C_N_crash, C_W_crash, C_A_crash)
])
mass_liquid_kg = rho_crash_liquid * V_m3
mass_solid_kg = S_crash * M_A_kg_per_mol
solid_content_pct = 100.0 * mass_solid_kg / (mass_solid_kg + mass_liquid_kg + 1e-12)

# Crash tank solids follow the same stream to the filter so the filter sees the solids load.
crash_solid_out_rate = np.where(
    sol_crash.t >= params_surge1["t_open"],
    (params_crash["F_in"] + params_crash["F_N"]) * (S_crash / params_crash["V"]),
    0.0,
)

# Prepare crash tank outlet inputs for the filter
interp_crash = {
    "C_H": interp1d(sol_crash.t, C_H_crash, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_N": interp1d(sol_crash.t, C_N_crash, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_W": interp1d(sol_crash.t, C_W_crash, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_A": interp1d(sol_crash.t, C_A_crash, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "S": interp1d(sol_crash.t, S_crash, kind='cubic', bounds_error=False, fill_value='extrapolate'),
}

# Evaluate the simplified continuous filter
filter_S_main_series = []
filter_C_H_main_series = []
filter_C_N_main_series = []
filter_C_W_main_series = []
filter_C_A_main_series = []
filter_C_H_waste_series = []
filter_C_N_waste_series = []
filter_C_W_waste_series = []
filter_C_A_waste_series = []
filter_AA_main_series = []
filter_total_A_main_series = []

for tt in sol_crash.t:
    filter_state = continuous_filter_split(tt, params_filter, interp_crash)
    filter_S_main_series.append(filter_state["S_out_to_main"])
    filter_C_H_main_series.append(filter_state["C_H_main"])
    filter_C_N_main_series.append(filter_state["C_N_main"])
    filter_C_W_main_series.append(filter_state["C_W_main"])
    filter_C_A_main_series.append(filter_state["C_A_main"])
    filter_C_H_waste_series.append(filter_state["C_H_waste"])
    filter_C_N_waste_series.append(filter_state["C_N_waste"])
    filter_C_W_waste_series.append(filter_state["C_W_waste"])
    filter_C_A_waste_series.append(filter_state["C_A_waste"])
    filter_AA_main_series.append(filter_state["AA_main"])
    filter_total_A_main_series.append(filter_state["total_A_main"])

filter_S_main_series = np.array(filter_S_main_series)
filter_C_H_main_series = np.array(filter_C_H_main_series)
filter_C_N_main_series = np.array(filter_C_N_main_series)
filter_C_W_main_series = np.array(filter_C_W_main_series)
filter_C_A_main_series = np.array(filter_C_A_main_series)
filter_C_H_waste_series = np.array(filter_C_H_waste_series)
filter_C_N_waste_series = np.array(filter_C_N_waste_series)
filter_C_W_waste_series = np.array(filter_C_W_waste_series)
filter_C_A_waste_series = np.array(filter_C_A_waste_series)
filter_AA_main_series = np.array(filter_AA_main_series)
filter_total_A_main_series = np.array(filter_total_A_main_series)

# Prepare filter main outlet interpolants (used as CSTR2 inlet)
interp_filter_main = {
    "C_A": interp1d(sol_crash.t, filter_C_A_main_series, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "AA": interp1d(sol_crash.t, filter_AA_main_series, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_N": interp1d(sol_crash.t, filter_C_N_main_series, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "S_A": interp1d(sol_crash.t, filter_S_main_series, kind='cubic', bounds_error=False, fill_value='extrapolate'),
}

# -----------------------------
# Plots for CSTR1 (conversion, temperature, concentrations)
# -----------------------------
fig1, ax1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax1[0].plot(t, T_cstr1, label="CSTR1 Reactor T", color=COLOR_SCHEME['T'])
# TODO: make the jacket temperature dynamic and replace this constant trace with Tc(t).
ax1[0].plot(t, Tc_cstr1, "--", label="CSTR1 Jacket Tc", color='gray', linewidth=1)
ax1[0].set_ylabel("Temperature (K)")
ax1[0].set_title("CSTR1: Temperature vs Time")
ax1[0].grid(True, alpha=0.3)
ax1[0].legend()

ax1[1].plot(t, C_H_cstr1, label="C_H (CSTR1)", color=COLOR_SCHEME['C_H'])
ax1[1].plot(t, C_N_cstr1, label="C_N (CSTR1)", color=COLOR_SCHEME['C_N'])
ax1[1].plot(t, C_W_cstr1, label="C_W (CSTR1)", color=COLOR_SCHEME['C_W'])
ax1[1].plot(t, C_A_cstr1, label="C_A (CSTR1)", color=COLOR_SCHEME['C_A'])
ax1[1].set_ylabel("Concentration (mol/L)")
ax1[1].set_title("CSTR1: Concentrations vs Time")
ax1[1].grid(True, alpha=0.3)
ax1[1].legend()

ax1[2].plot(t, X_A_cstr1, color="black", label="Conversion of A (CSTR1)")
ax1[2].set_xlabel("Time (s)")
ax1[2].set_ylabel("Conversion")
ax1[2].set_title("CSTR1: A Conversion vs Time")
ax1[2].set_ylim(0, 1.05)
ax1[2].grid(True, alpha=0.3)
ax1[2].legend()

plt.tight_layout()

# -----------------------------
# Plots for surge1 tank
# -----------------------------
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax2[0].plot(sol_surge1.t, V_surge1, color='tab:blue', label='Surge1 volume')
ax2[0].axhline(params_surge1["V"], color='red', linestyle='--', linewidth=1, label='Surge1 max volume')
ax2[0].set_ylabel('Volume (L)')
ax2[0].set_title('Surge1 Tank: Volume vs Time')
ax2[0].grid(True, alpha=0.3)
ax2[0].legend()

ax2[1].plot(sol_surge1.t, C_A_surge1, label='C_A (surge1)', color=COLOR_SCHEME['C_A'])
ax2[1].plot(sol_surge1.t, C_H_surge1, label='C_H (surge1)', color=COLOR_SCHEME['C_H'])
ax2[1].plot(sol_surge1.t, C_N_surge1, label='C_N (surge1)', color=COLOR_SCHEME['C_N'], linestyle='-')
ax2[1].plot(sol_surge1.t, C_W_surge1, label='C_W (surge1)', color=COLOR_SCHEME['C_W'])
ax2[1].set_ylabel('Concentration (mol/L)')
ax2[1].set_title('Surge1 Tank: Concentrations vs Time')
ax2[1].grid(True, alpha=0.3)
ax2[1].legend()

ax2[2].plot(sol_surge1.t, T_surge1, label='T (surge1)', color=COLOR_SCHEME['T'], linewidth=2)
Tc_surge1 = np.full_like(sol_surge1.t, params_surge1["Tc"])
ax2[2].plot(sol_surge1.t, Tc_surge1, '--', label='T_c (jacket)', color='gray', linewidth=1)
ax2[2].set_xlabel('Time (s)')
ax2[2].set_ylabel('Temperature (K)')
ax2[2].set_title('Surge1 Tank: Temperature vs Time (with jacket cooling)')
ax2[2].grid(True, alpha=0.3)
ax2[2].legend()

plt.tight_layout()

# -----------------------------
# Plots for crash tank
# -----------------------------
fig3, ax3 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax3[0].plot(sol_crash.t, solid_content_pct, color='brown', label='Solid content (%)')
ax3[0].set_ylabel('Solid content (%)')
ax3[0].set_title('Crash Tank: Solid content and solids out vs Time')
ax3[0].grid(True, alpha=0.3)
ax3[0].legend()

ax3_0b = ax3[0].twinx()
ax3_0b.plot(sol_crash.t, crash_solid_out_rate, color='black', linestyle='--', label='Solids out to filter')
ax3_0b.set_ylabel('Solids out rate (mol/s)')
ax3_0b.legend(loc='upper right')

ax3[1].plot(sol_crash.t, C_A_crash, label='C_A (crash)', linewidth=2, color=COLOR_SCHEME['C_A'])
ax3[1].plot(sol_crash.t, C_N_crash, label='C_N (crash)', linewidth=2, linestyle='-', color=COLOR_SCHEME['C_N'])
ax3[1].plot(sol_crash.t, C_H_crash, label='C_H (crash)', alpha=0.7, color=COLOR_SCHEME['C_H'])
ax3[1].plot(sol_crash.t, C_W_crash, label='C_W (crash)', alpha=0.7, color=COLOR_SCHEME['C_W'])
ax3[1].set_ylabel('Concentration (mol/L)')
ax3[1].set_title('Crash Tank: Concentrations vs Time (all species)')
ax3[1].grid(True, alpha=0.3)
ax3[1].legend()

ax3[2].plot(sol_crash.t, T_crash, label='T (crash)')
ax3[2].plot(sol_crash.t, Tc_crash, '--', label='T_c (jacket)')
ax3[2].set_xlabel('Time (s)')
ax3[2].set_ylabel('Temperature (K)')
ax3[2].set_title('Crash Tank: Temperature vs Time')
ax3[2].grid(True, alpha=0.3)
ax3[2].legend()

plt.tight_layout()

# # OLD: Semi-batch rotating filter plots - replaced with continuous filtration
# fig4, ax4 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
# ax4[0].plot(sol_crash.t, filter_volume_series, color='tab:purple', label='Filter hold-up volume')
# ax4[0].axhline(params_filter["V_max"], color='red', linestyle='--', linewidth=1, label='Max filter volume')
# ax4[0].set_ylabel('Volume (L)')
# ax4[0].set_title('Semi-Batch Filter: Hold-up Volume vs Time')
# ax4[0].grid(True, alpha=0.3)
# ax4[0].legend()
# ax4[1].plot(sol_crash.t, filter_liquid_A_series, label='Liquid A in filter')
# ax4[1].plot(sol_crash.t, filter_main_A_series, label='Main stream A in AA')
# ax4[1].plot(sol_crash.t, filter_waste_A_series, label='Waste stream A')
# ax4[1].set_ylabel('Concentration / split basis')
# ax4[1].set_title('Semi-Batch Filter: Material Split vs Time')
# ax4[1].grid(True, alpha=0.3)
# ax4[1].legend()
# ax4[2].plot(sol_crash.t, filter_main_flow_series, label='Main stream flow')
# ax4[2].plot(sol_crash.t, filter_waste_flow_series, label='Waste stream flow')
# ax4[2].plot(sol_crash.t, filter_AA_makeup_series, label='AA make-up flow')
# ax4[2].set_xlabel('Time (s)')
# ax4[2].set_ylabel('Flow basis')
# ax4[2].set_title('Semi-Batch Filter: Stream Rotation vs Time')
# ax4[2].grid(True, alpha=0.3)
# ax4[2].legend()

# -------- NEW: Simplified continuous filter plots --------
fig4, ax4 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Main stream composition (solids dissolved in AA with trace liquids)
ax4[0].plot(sol_crash.t, filter_AA_main_series, label='AA recovered (main stream)', linewidth=2, color='blue')
ax4[0].plot(sol_crash.t, filter_S_main_series, label='Solid A to main (moles)', linewidth=2, color='red')
ax4[0].plot(sol_crash.t, filter_total_A_main_series, label='Total A in main stream', linewidth=1, linestyle='--', color='darkred')
ax4[0].set_ylabel('Amount (moles or mol/L)')
ax4[0].set_title('Continuous Filter: Main Stream Composition (AA recovery + Solid A + Trace Liquids)')
ax4[0].grid(True, alpha=0.3)
ax4[0].legend()

# Main stream trace concentrations (N, W, A remain)
ax4[1].plot(sol_crash.t, filter_C_N_main_series, label='C_N (main, trace)', linewidth=1.5, color=COLOR_SCHEME['C_N'])
ax4[1].plot(sol_crash.t, filter_C_W_main_series, label='C_W (main, trace)', linewidth=1.5, color=COLOR_SCHEME['C_W'])
ax4[1].plot(sol_crash.t, filter_C_A_main_series, label='C_A (main, residual)', linewidth=1.5, color=COLOR_SCHEME['C_A'])
ax4[1].set_ylabel('Concentration (mol/L)')
ax4[1].set_title('Continuous Filter: Main Stream Trace Concentrations')
ax4[1].grid(True, alpha=0.3)
ax4[1].legend()

# Waste stream composition (most liquids exit here)
ax4[2].plot(sol_crash.t, filter_C_H_waste_series, label='C_H (waste)', linewidth=1.5, color=COLOR_SCHEME['C_H'])
ax4[2].plot(sol_crash.t, filter_C_N_waste_series, label='C_N (waste)', linewidth=1.5, color=COLOR_SCHEME['C_N'])
ax4[2].plot(sol_crash.t, filter_C_W_waste_series, label='C_W (waste)', linewidth=1.5, color=COLOR_SCHEME['C_W'])
ax4[2].plot(sol_crash.t, filter_C_A_waste_series, label='C_A (waste, majority)', linewidth=1.5, color=COLOR_SCHEME['C_A'])
ax4[2].set_xlabel('Time (s)')
ax4[2].set_ylabel('Concentration (mol/L)')
ax4[2].set_title('Continuous Filter: Waste Stream Composition (all species)')
ax4[2].grid(True, alpha=0.3)
ax4[2].legend()

plt.tight_layout()

# -------- Steady-state flowsheet table --------
# Average the last 20% of the simulation to estimate steady state
n_ss = max(1, int(0.2 * len(sol_cstr1.t)))

# CSTR1 outlet (steady state)
c_h_cstr1_ss = np.mean(C_H_cstr1[-n_ss:])
c_n_cstr1_ss = np.mean(C_N_cstr1[-n_ss:])
c_w_cstr1_ss = np.mean(C_W_cstr1[-n_ss:])
c_a_cstr1_ss = np.mean(C_A_cstr1[-n_ss:])
t_cstr1_ss = np.mean(T_cstr1[-n_ss:])

# Surge tank outlet (steady state)
c_h_surge1_ss = np.mean(C_H_surge1[-n_ss:])
c_n_surge1_ss = np.mean(C_N_surge1[-n_ss:])
c_w_surge1_ss = np.mean(C_W_surge1[-n_ss:])
c_a_surge1_ss = np.mean(C_A_surge1[-n_ss:])
t_surge1_ss = np.mean(T_surge1[-n_ss:])

# Crash tank outlet (steady state)
c_h_crash_ss = np.mean(C_H_crash[-n_ss:])
c_n_crash_ss = np.mean(C_N_crash[-n_ss:])
c_w_crash_ss = np.mean(C_W_crash[-n_ss:])
c_a_crash_ss = np.mean(C_A_crash[-n_ss:])
s_crash_ss = np.mean(S_crash[-n_ss:])
t_crash_ss = np.mean(T_crash[-n_ss:])

# Filter outlet (main stream, steady state)
filt_c_h_main_ss = np.mean(filter_C_H_main_series[-n_ss:])
filt_c_n_main_ss = np.mean(filter_C_N_main_series[-n_ss:])
filt_c_w_main_ss = np.mean(filter_C_W_main_series[-n_ss:])
filt_c_a_main_ss = np.mean(filter_C_A_main_series[-n_ss:])
filt_s_main_ss = np.mean(filter_S_main_series[-n_ss:])
filt_aa_main_ss = np.mean(filter_AA_main_series[-n_ss:])

# Filter outlet (waste stream, steady state)
filt_c_h_waste_ss = np.mean(filter_C_H_waste_series[-n_ss:])
filt_c_n_waste_ss = np.mean(filter_C_N_waste_series[-n_ss:])
filt_c_w_waste_ss = np.mean(filter_C_W_waste_series[-n_ss:])
filt_c_a_waste_ss = np.mean(filter_C_A_waste_series[-n_ss:])

# Build flowsheet table
flowsheet_data = {
    'Stream / Unit': [
        'CSTR1 Outlet',
        'Surge1 Tank Outlet',
        'Crash Tank Outlet',
        'Filter Main Outlet',
        'Filter Waste Outlet',
    ],
    'C_H (mol/L)': [
        f"{c_h_cstr1_ss:.4f}",
        f"{c_h_surge1_ss:.4f}",
        f"{c_h_crash_ss:.4f}",
        f"{filt_c_h_main_ss:.4f}",
        f"{filt_c_h_waste_ss:.4f}",
    ],
    'C_N (mol/L)': [
        f"{c_n_cstr1_ss:.4f}",
        f"{c_n_surge1_ss:.4f}",
        f"{c_n_crash_ss:.4f}",
        f"{filt_c_n_main_ss:.4f}",
        f"{filt_c_n_waste_ss:.4f}",
    ],
    'C_W (mol/L)': [
        f"{c_w_cstr1_ss:.4f}",
        f"{c_w_surge1_ss:.4f}",
        f"{c_w_crash_ss:.4f}",
        f"{filt_c_w_main_ss:.4f}",
        f"{filt_c_w_waste_ss:.4f}",
    ],
    'C_A (mol/L)': [
        f"{c_a_cstr1_ss:.4f}",
        f"{c_a_surge1_ss:.4f}",
        f"{c_a_crash_ss:.4f}",
        f"{filt_c_a_main_ss:.4f}",
        f"{filt_c_a_waste_ss:.4f}",
    ],
    'S (mol A solid)': [
        '—',
        '—',
        f"{s_crash_ss:.4f}",
        f"{filt_s_main_ss:.4f}",
        '—',
    ],
    'T (K)': [
        f"{t_cstr1_ss:.2f}",
        f"{t_surge1_ss:.2f}",
        f"{t_crash_ss:.2f}",
        '—',
        '—',
    ],
    'AA (mol)': [
        '—',
        '—',
        '—',
        f"{filt_aa_main_ss:.4f}",
        '—',
    ],
}

df_flowsheet = pd.DataFrame(flowsheet_data)

print("\n" + "="*130)
print("STEADY-STATE FLOWSHEET (averaged over final 20% of simulation)")
print("="*130)
print(df_flowsheet.to_string(index=False))
print("="*130)
print("\nNOTES:")
print(f"  • CSTR1: Initial reaction producing A from H, N, W reactants (exothermic, jacket cooled to {params_cstr1['Tc']} K)")
print(f"  • Surge1: Passive buffer, jacket cooled to {params_surge1['Tc']} K, outlet opens at t={params_surge1['t_open']:.1f}s")
print(f"  • Crash: N feed added ({params_crash['C_N_ext']} mol/L external), A precipitates as solid, jacket cooled to {params_crash['Tc']} K")
print(f"  • Filter: Continuous separation - solids to main (with AA), liquids split (residual A in main, majority to waste)")
print(f"  • Main stream: {100*filt_c_a_main_ss/max(filt_c_a_main_ss+filt_c_a_waste_ss, 1e-9):.0f}% of liquid A (residual) + {filt_s_main_ss:.4f} mol solids + {filt_aa_main_ss:.4f} mol AA")
print(f"  • Waste stream: {100*filt_c_a_waste_ss/max(filt_c_a_main_ss+filt_c_a_waste_ss, 1e-9):.0f}% of liquid A (for distillation to recover N)")
print("="*130 + "\n")

# (Moved: CSTR2 conversion reporting is printed after CSTR2 & Surge2 results are computed.)

# -----------------------------
# CSTR2: complex nitration reactor (A + AAh + AA + FWNA -> N + B1 + B2 + nitramines)
# -----------------------------
params_cstr2 = {
    "F": 0.8,    # L/s total feed to CSTR2
    "V": 8.0,    # L
    # feed compositions (these could be linked to upstream streams via interp)
    "C_A_in": 0.05,    # mol/L (unreacted A entering CSTR2)
    "C_AAh_feed": 0.5, # mol/L
    "C_AA_feed": 0.2,  # mol/L
    "C_FWNA_feed": 0.3,# mol/L
    "C_N_feed": 0.0,
    "Tf": 298.15,
    "Tc": 295.0,
    "rho_ref": 1050.0,
    "Cp": 4.18,
    "UA": 1500.0,
    # kinetics (phenomenological lumped rate)
    "k2": 1e3,
    "Ea2": 20000.0,
    "R": 8.314,
    "dH2": -66000.0,  # J/mol (+10% exothermic magnitude)
    # product split fractions (sum of solids+liquid+nitramines = 1)
    "f_sb1": 0.30,
    "f_sb2": 0.20,
    "f_lb1": 0.25,
    "f_lb2": 0.15,
    "f_nit": 0.10,
}


def cstr2_model(t, y, p, inlet_interp=None):
    # States: C_A, C_AAh, C_AA, C_FWNA, C_N, C_B1, C_B2, C_Nit, T, S_B1, S_B2
    C_A, C_AAh, C_AA, C_FWNA, C_N, C_B1, C_B2, C_Nit, T, S_B1, S_B2 = y

    F, V = p["F"], p["V"]
    # feed concentrations from inlet_interp (filter main) when available, otherwise use params
    if inlet_interp is not None:
        C_A_in = float(inlet_interp.get("C_A", lambda tt: p.get("C_A_in", 0.0))(t))
        # treat AA/AAh feed from AA recovered (interp key 'AA') if present
        AA_val = float(inlet_interp.get("AA", lambda tt: p.get("C_AA_feed", 0.0))(t))
        C_AAh_f = AA_val
        C_AA_f = AA_val
        C_FWNA_f = float(inlet_interp.get("C_FWNA", lambda tt: p.get("C_FWNA_feed", 0.0))(t))
        C_Nf = float(inlet_interp.get("C_N", lambda tt: p.get("C_N_feed", 0.0))(t))
    else:
        C_A_in = p["C_A_in"]
        C_AAh_f = p["C_AAh_feed"]
        C_AA_f = p["C_AA_feed"]
        C_FWNA_f = p["C_FWNA_feed"]
        C_Nf = p["C_N_feed"]

    # lumped reaction rate (phenomenological)
    k2 = p["k2"] * np.exp(-p["Ea2"] / (p["R"] * max(T, 1e-6)))
    r2 = k2 * max(C_A, 0.0) * max(C_AAh, 0.0) * max(C_AA, 0.0) * max(C_FWNA, 0.0)

    # mass balances (convective + reaction)
    dC_A_dt = (F / V) * (C_A_in - C_A) - r2
    dC_AAh_dt = (F / V) * (C_AAh_f - C_AAh) - r2
    dC_AA_dt = (F / V) * (C_AA_f - C_AA) - r2
    dC_FWNA_dt = (F / V) * (C_FWNA_f - C_FWNA) - r2

    # nitrogen production and accumulation
    dC_N_dt = (F / V) * (C_Nf - C_N) + 1.0 * r2

    # liquid products
    dC_B1_dt = - (F / V) * C_B1 + p["f_lb1"] * r2
    dC_B2_dt = - (F / V) * C_B2 + p["f_lb2"] * r2
    dC_Nit_dt = - (F / V) * C_Nit + p["f_nit"] * r2

    # solids accumulation and removal with outlet (simple inventory)
    solid_out_B1 = (F / V) * S_B1
    solid_out_B2 = (F / V) * S_B2
    dS_B1_dt = p["f_sb1"] * r2 * V - solid_out_B1
    dS_B2_dt = p["f_sb2"] * r2 * V - solid_out_B2

    # energy balance
    rho = p["rho_ref"]
    Cp = p["Cp"]
    dT_dt = (F / V) * (p["Tf"] - T) - (p["dH2"] / (rho * Cp)) * r2 - (p["UA"] / (rho * Cp * V)) * (T - p["Tc"])

    return [dC_A_dt, dC_AAh_dt, dC_AA_dt, dC_FWNA_dt, dC_N_dt, dC_B1_dt, dC_B2_dt, dC_Nit_dt, dT_dt, dS_B1_dt, dS_B2_dt]


# -----------------------------
# Solve CSTR2
# -----------------------------
y0_cstr2 = [
    0.0,   # C_A (empty at t=0)
    0.0,   # C_AAh (empty at t=0)
    0.0,   # C_AA (empty at t=0)
    0.0,   # C_FWNA (empty at t=0)
    0.0,   # C_N (empty at t=0)
    0.0,   # C_B1
    0.0,   # C_B2
    0.0,   # C_Nit
    params_cstr2["Tf"],  # initialized at feed temperature
    0.0,   # S_B1
    0.0,   # S_B2
]

sol_cstr2 = solve_ivp(
    fun=lambda t, y: cstr2_model(t, y, params_cstr2, inlet_interp=interp_filter_main),
    t_span=t_span,
    y0=y0_cstr2,
    t_eval=t_eval,
    method='RK45'
)

# Extract CSTR2 results
C_A_cstr2, C_AAh_cstr2, C_AA_cstr2, C_FWNA_cstr2, C_N_cstr2, C_B1_cstr2, C_B2_cstr2, C_Nit_cstr2, T_cstr2, S_B1_cstr2, S_B2_cstr2 = sol_cstr2.y

# Combined B (liquid + solid expressed on concentration basis):
# solids S_B1, S_B2 are in moles; convert to mol/L by dividing by reactor volume V
combined_B_cstr2 = C_B1_cstr2 + C_B2_cstr2 + (S_B1_cstr2 + S_B2_cstr2) / params_cstr2["V"]

# Correct A conversion definition: fraction of A consumed (1 - C_A/C_A_in)
X_A_cstr2 = 1.0 - (C_A_cstr2 / np.maximum(params_cstr2.get('C_A_in', 1e-12), 1e-12))

# Combined B conversion relative to input A concentration (mol/L basis)
X_B_combined_cstr2 = combined_B_cstr2 / np.maximum(params_cstr2.get('C_A_in', 1e-12), 1e-12)

# -----------------------------
# Surge2: final resting tank for CSTR2 outlet
# -----------------------------
params_surge2 = {
    "F_in": params_cstr2["F"],
    "V": 200.0,   # large volume final surge
    "V0": 0.0,
    "t_open": 0.0,
    "rho_ref": params_cstr2["rho_ref"],
    "Cp": params_cstr2["Cp"],
    "UA": 1000.0,
    "Tc": 295.0,
}


def surge2_tank_model(t, y, p, inlet_interp):
    # States: V, C_A, C_AAh, C_AA, C_FWNA, C_N, C_B1, C_B2, C_Nit, T, S_B1, S_B2
    V, C_A, C_AAh, C_AA, C_FWNA, C_N, C_B1, C_B2, C_Nit, T, S_B1, S_B2 = y

    C_A_in = inlet_interp["C_A"](t)
    C_AAh_in = inlet_interp["C_AAh"](t)
    C_AA_in = inlet_interp["C_AA"](t)
    C_FWNA_in = inlet_interp["C_FWNA"](t)
    C_N_in = inlet_interp["C_N"](t)
    C_B1_in = inlet_interp["C_B1"](t)
    C_B2_in = inlet_interp["C_B2"](t)
    C_Nit_in = inlet_interp["C_Nit"](t)
    T_in = inlet_interp["T"](t)

    F_in = p["F_in"]
    # Final resting tank: no outlet during the simulated horizon.
    F_out = 0.0

    dV_dt = F_in - F_out
    V_eff = max(V, 1e-9)
    if V <= 1e-9:
        return [dV_dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    dC_A_dt = (F_in / V_eff) * (C_A_in - C_A)
    dC_AAh_dt = (F_in / V_eff) * (C_AAh_in - C_AAh)
    dC_AA_dt = (F_in / V_eff) * (C_AA_in - C_AA)
    dC_FWNA_dt = (F_in / V_eff) * (C_FWNA_in - C_FWNA)
    dC_N_dt = (F_in / V_eff) * (C_N_in - C_N)
    dC_B1_dt = (F_in / V_eff) * (C_B1_in - C_B1)
    dC_B2_dt = (F_in / V_eff) * (C_B2_in - C_B2)
    dC_Nit_dt = (F_in / V_eff) * (C_Nit_in - C_Nit)
    dT_dt = (F_in / V_eff) * (T_in - T) - (p["UA"] / (p["rho_ref"] * p["Cp"] * V_eff)) * (T - p["Tc"])

    # solids inlet (mol) are provided as S_B1 and S_B2 from CSTR2 outlet; convert to mol/L addition rate = S_in_flow / V_eff
    S_B1_in = float(inlet_interp.get("S_B1", lambda tt: 0.0)(t))
    S_B2_in = float(inlet_interp.get("S_B2", lambda tt: 0.0)(t))
    # treat inlet solids as mol/s entering via F_in scaled by fraction (approx): convert S_B*_in (mol) arriving per upstream outlet snapshot -> approximate rate = (F_in/V_eff)*S_B*_in
    dS_B1_dt = (F_in / V_eff) * (S_B1_in / max(1.0, p.get("V", V_eff)) - S_B1 / V_eff) * V_eff
    dS_B2_dt = (F_in / V_eff) * (S_B2_in / max(1.0, p.get("V", V_eff)) - S_B2 / V_eff) * V_eff

    return [dV_dt, dC_A_dt, dC_AAh_dt, dC_AA_dt, dC_FWNA_dt, dC_N_dt, dC_B1_dt, dC_B2_dt, dC_Nit_dt, dT_dt, dS_B1_dt, dS_B2_dt]


# Prepare CSTR2 outlet interpolation for surge2
interp_cstr2 = {
    "C_A": interp1d(sol_cstr2.t, C_A_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_AAh": interp1d(sol_cstr2.t, C_AAh_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_AA": interp1d(sol_cstr2.t, C_AA_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_FWNA": interp1d(sol_cstr2.t, C_FWNA_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_N": interp1d(sol_cstr2.t, C_N_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_B1": interp1d(sol_cstr2.t, C_B1_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_B2": interp1d(sol_cstr2.t, C_B2_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_Nit": interp1d(sol_cstr2.t, C_Nit_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "S_B1": interp1d(sol_cstr2.t, S_B1_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "S_B2": interp1d(sol_cstr2.t, S_B2_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "T": interp1d(sol_cstr2.t, T_cstr2, kind='cubic', bounds_error=False, fill_value='extrapolate'),
}

# Solve surge2
y0_surge2 = [
    params_surge2["V0"],  # empty volume at t=0
    0.0,  # C_A
    0.0,  # C_AAh
    0.0,  # C_AA
    0.0,  # C_FWNA
    0.0,  # C_N
    0.0,  # C_B1
    0.0,  # C_B2
    0.0,  # C_Nit
    params_cstr2["Tf"],
    0.0,  # S_B1
    0.0,  # S_B2
]

sol_surge2 = solve_ivp(
    fun=lambda tt, yy: surge2_tank_model(tt, yy, params_surge2, interp_cstr2),
    t_span=t_span,
    y0=y0_surge2,
    t_eval=t_eval,
    method='RK45'
)

# Extract surge2 results (including solids)
V_surge2, C_A_surge2, C_AAh_surge2, C_AA_surge2, C_FWNA_surge2, C_N_surge2, C_B1_surge2, C_B2_surge2, C_Nit_surge2, T_surge2, S_B1_surge2, S_B2_surge2 = sol_surge2.y

# -----------------------------
# Plots for CSTR2 (similar to CSTR1)
# -----------------------------
fig5, ax5 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax5[0].plot(sol_cstr2.t, T_cstr2, label='CSTR2 Reactor T', color=COLOR_SCHEME['T'])
ax5[0].plot(sol_cstr2.t, np.full_like(sol_cstr2.t, params_cstr2['Tc']), '--', label='CSTR2 Jacket Tc', color='gray')
ax5[0].set_ylabel('Temperature (K)')
ax5[0].set_title('CSTR2: Temperature vs Time')
ax5[0].grid(True, alpha=0.3)
ax5[0].legend()

ax5[1].plot(sol_cstr2.t, C_A_cstr2, label='C_A', color=COLOR_SCHEME['C_A'])
ax5[1].plot(sol_cstr2.t, C_AAh_cstr2, label='C_AAh', color=COLOR_SCHEME['C_AAh'])
ax5[1].plot(sol_cstr2.t, C_AA_cstr2, label='C_AA', color=COLOR_SCHEME['AA'])
ax5[1].plot(sol_cstr2.t, C_FWNA_cstr2, label='C_FWNA', color=COLOR_SCHEME['C_W'])
ax5[1].plot(sol_cstr2.t, C_N_cstr2, label='C_N', color=COLOR_SCHEME['C_N'], linestyle='-')
ax5[1].plot(sol_cstr2.t, C_B1_cstr2, label='C_B1 (liq)', color='#006400')
ax5[1].plot(sol_cstr2.t, C_B2_cstr2, label='C_B2 (liq)', color='#228B22')
ax5[1].set_ylabel('Concentration (mol/L)')
ax5[1].set_title('CSTR2: Key Concentrations vs Time')
ax5[1].grid(True, alpha=0.3)
ax5[1].legend()

ax5[2].plot(sol_cstr2.t, X_A_cstr2, color='black', label='Conversion of A (CSTR2)')
ax5[2].plot(sol_cstr2.t, X_B_combined_cstr2, color='darkgreen', linestyle='--', label='Combined B1+B2 conversion (liq+sol)')
ax5[2].set_xlabel('Time (s)')
ax5[2].set_ylabel('Conversion')
ax5[2].set_title('CSTR2: Conversion vs Time (A and combined B)')
ax5[2].set_ylim(0, 1.05)
ax5[2].grid(True, alpha=0.3)
ax5[2].legend()

plt.tight_layout()

# plot solids on twin axis for CSTR2
ax5b = ax5[1].twinx()
ax5b.plot(sol_cstr2.t, S_B1_cstr2 / params_cstr2['V'], label='S_B1 (solids, mol/L)', color='#8B0000', linestyle=':')
ax5b.plot(sol_cstr2.t, S_B2_cstr2 / params_cstr2['V'], label='S_B2 (solids, mol/L)', color='#FF4500', linestyle=':')
ax5b.set_ylabel('Solids (mol/L)')
ax5b.legend(loc='upper right')

# -----------------------------
# Plots for surge2 (final tank)
# -----------------------------
fig6, ax6 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax6[0].plot(sol_surge2.t, V_surge2, color='tab:blue', label='Surge2 volume')
ax6[0].axhline(params_surge2['V'], color='red', linestyle='--', linewidth=1, label='Surge2 max volume')
ax6[0].set_ylabel('Volume (L)')
ax6[0].set_title('Surge2 Tank: Volume vs Time')
ax6[0].grid(True, alpha=0.3)
ax6[0].legend()

ax6[1].plot(sol_surge2.t, C_A_surge2, label='C_A (surge2)', color=COLOR_SCHEME['C_A'])
ax6[1].plot(sol_surge2.t, C_AAh_surge2, label='C_AAh (surge2)', color=COLOR_SCHEME['C_AAh'])
ax6[1].plot(sol_surge2.t, C_AA_surge2, label='C_AA (surge2)', color=COLOR_SCHEME['AA'])
ax6[1].plot(sol_surge2.t, C_N_surge2, label='C_N (surge2)', color=COLOR_SCHEME['C_N'], linestyle='-')
ax6[1].plot(sol_surge2.t, C_B1_surge2, label='C_B1 (surge2, liq)', color='#006400')
ax6[1].plot(sol_surge2.t, C_B2_surge2, label='C_B2 (surge2, liq)', color='#228B22')
ax6[1].set_ylabel('Concentration (mol/L)')
ax6[1].set_title('Surge2: Concentrations vs Time')
ax6[1].grid(True, alpha=0.3)
ax6[1].legend()

ax6[2].plot(sol_surge2.t, T_surge2, label='T (surge2)', color=COLOR_SCHEME['T'])
ax6[2].plot(sol_surge2.t, np.full_like(sol_surge2.t, params_surge2['Tc']), '--', label='T_c (jacket)', color='gray')
ax6[2].set_xlabel('Time (s)')
ax6[2].set_ylabel('Temperature (K)')
ax6[2].set_title('Surge2: Temperature vs Time')
ax6[2].grid(True, alpha=0.3)
ax6[2].legend()

# solids in surge2 on twin axis
ax6b = ax6[1].twinx()
ax6b.plot(sol_surge2.t, S_B1_surge2 / np.maximum(V_surge2, 1e-9), label='S_B1 (surge2, mol/L)', color='#8B0000', linestyle=':')
ax6b.plot(sol_surge2.t, S_B2_surge2 / np.maximum(V_surge2, 1e-9), label='S_B2 (surge2, mol/L)', color='#FF4500', linestyle=':')
ax6b.set_ylabel('Solids (mol/L)')
ax6b.legend(loc='upper right')

plt.tight_layout()

# Print final conversion summary for CSTR2 before showing plots
final_X_A_cstr2 = X_A_cstr2[-1]
final_X_B_combined_cstr2 = X_B_combined_cstr2[-1]
print(f"CSTR2 final conversion of A (1 - C_A/C_A_in): {final_X_A_cstr2:.4f}")
print(f"CSTR2 final combined B1+B2 conversion (liquid+solid) relative to A_in: {final_X_B_combined_cstr2:.4f}")
print("Note: CSTR2 uses only its internal lumped reaction r2; reaction 1 (CSTR1) is not modeled inside CSTR2.")

# -----------------------------
# Extended flowsheet including CSTR2 and Surge2
# -----------------------------
combined_B_cstr2_ss = np.mean(combined_B_cstr2[-n_ss:])
combined_B_surge2 = C_B1_surge2 + C_B2_surge2
combined_B_surge2_ss = np.mean(combined_B_surge2[-n_ss:])

flowsheet2_data = {
    'Stream / Unit': [
        'CSTR2 Outlet',
        'Surge2 Tank Outlet',
    ],
    'C_A (mol/L)': [
        f"{np.mean(C_A_cstr2[-n_ss:]):.4f}",
        f"{np.mean(C_A_surge2[-n_ss:]):.4f}",
    ],
    'Combined_B (mol/L)': [
        f"{combined_B_cstr2_ss:.4f}",
        f"{combined_B_surge2_ss:.4f}",
    ],
    'Combined_B_conv (rel to A_in)': [
        f"{np.mean(X_B_combined_cstr2[-n_ss:]):.4f}",
        f"{np.mean((combined_B_surge2 / np.maximum(params_cstr2.get('C_A_in',1e-12), 1e-12))[-n_ss:]):.4f}",
    ],
    'T (K)': [
        f"{np.mean(T_cstr2[-n_ss:]):.2f}",
        f"{np.mean(T_surge2[-n_ss:]):.2f}",
    ],
}

df_flowsheet2 = pd.DataFrame(flowsheet2_data)
print('\n' + '='*100)
print('CSTR2 / Surge2 SUMMARY (averaged over final 20% of simulation)')
print('='*100)
print(df_flowsheet2.to_string(index=False))
print('='*100 + "\n")

plt.show()