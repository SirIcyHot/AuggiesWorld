import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

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
    "dH": -50000.0   # J/mol (negative = exothermic)
}

# -----------------------------
# Surge tank parameters
# -----------------------------
params_surge = {
    "F_in": params_cstr1["F"],
    "V": 10.0,
    "V0": 2.0,
    "t_open": 0.8 * (10.0 - 2.0) / params_cstr1["F"],
    "rho_ref": 995.0,
    "Cp": params_cstr1["Cp"],
    "species_rho": params_cstr1["species_rho"],
}

# -----------------------------
# Crash tank parameters
# -----------------------------
params_crash = {
    "F_in": params_surge["F_in"],   # inlet flow from surge tank (L/s)
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
    "M_A_gmol": 100.0
}

# TODO: replace the static density values with temperature-dependent density correlations for each unit.


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
# Surge tank model
# -----------------------------
def surge_tank_model(t, y, p, inlet_interp):
    V, C_H, C_N, C_W, C_A, T = y

    C_H_in = inlet_interp["C_H"](t)
    C_N_in = inlet_interp["C_N"](t)
    C_W_in = inlet_interp["C_W"](t)
    C_A_in = inlet_interp["C_A"](t)
    T_in = inlet_interp["T"](t)

    F_in = p["F_in"]
    F_out = F_in if t >= p["t_open"] else 0.0

    if V <= 1e-9:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    dV_dt = F_in - F_out
    dC_H_dt = (F_in / V) * (C_H_in - C_H)
    dC_N_dt = (F_in / V) * (C_N_in - C_N)
    dC_W_dt = (F_in / V) * (C_W_in - C_W)
    dC_A_dt = (F_in / V) * (C_A_in - C_A)
    dT_dt = (F_in / V) * (T_in - T)

    return [dV_dt, dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt]


def surge_density(c_h, c_n, c_w, c_a, p):
    composition = liquid_composition_from_molar(c_h, c_n, c_w, c_a)
    return mixture_density_liquid(composition, p["species_rho"])


# -----------------------------
# Crash tank model
# -----------------------------
def crash_tank_model(t, y, p, inlet_interp, t_open):
    C_H, C_N, C_W, C_A, T, S = y

    if t < t_open:
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

    # solid accumulation (moles of A solid)
    dS_dt = r_p * p["V"]

    # energy balance (no heat of precipitation included here)
    rho_mix = crash_mixture_density(C_H, C_N, C_W, C_A, S, p, p["V"])
    dT_dt = (F_tot / p["V"]) * (Tf - T) - (p["UA"] / (rho_mix * p["Cp"] * p["V"])) * (T - p["Tc"])

    return [dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt, dS_dt]


def crash_liquid_density(c_h, c_n, c_w, c_a, p):
    composition = liquid_composition_from_molar(c_h, c_n, c_w, c_a)
    return mixture_density_liquid(composition, p["species_rho"])

# -----------------------------
# Initial Conditions
# -----------------------------
y0_cstr1 = [
    1.0,   # C_H
    1.0,   # C_N
    4.0,   # C_W
    0.0,   # C_A
    330.0  # T
]

# -----------------------------
# Time Span
# -----------------------------
t_span = (0, 50)
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
X_H_cstr1 = (params_cstr1["C_Hf"] - C_H_cstr1) / params_cstr1["C_Hf"]
rho_cstr1 = np.array([
    cstr1_density(ch, cn, cw, ca, params_cstr1)
    for ch, cn, cw, ca in zip(C_H_cstr1, C_N_cstr1, C_W_cstr1, C_A_cstr1)
])

# Example: print final values
print("Final CSTR1 concentrations and temperature:")
print(f"C_H = {C_H_cstr1[-1]:.3f}")
print(f"C_N = {C_N_cstr1[-1]:.3f}")
print(f"C_W = {C_W_cstr1[-1]:.3f}")
print(f"C_A = {C_A_cstr1[-1]:.3f}")
print(f"T   = {T_cstr1[-1]:.2f} K")
print(f"Surge tank outlet opens at t = {params_surge['t_open']:.2f} s")

# -----------------------------
# Prepare surge tank inlet inputs (interpolate CSTR1 outlet)
# -----------------------------
interp_cstr1 = {
    "C_H": interp1d(t, C_H_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_N": interp1d(t, C_N_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_W": interp1d(t, C_W_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_A": interp1d(t, C_A_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "T": interp1d(t, T_cstr1, kind='cubic', bounds_error=False, fill_value='extrapolate'),
}

# -----------------------------
# Solve surge tank
# -----------------------------
y0_surge = [
    params_surge["V0"],
    C_H_cstr1[0],
    C_N_cstr1[0],
    C_W_cstr1[0],
    C_A_cstr1[0],
    T_cstr1[0],
]

sol_surge = solve_ivp(
    fun=lambda tt, yy: surge_tank_model(tt, yy, params_surge, interp_cstr1),
    t_span=t_span,
    y0=y0_surge,
    t_eval=t_eval,
    method='RK45'
)

# Extract surge tank results
V_surge, C_H_surge, C_N_surge, C_W_surge, C_A_surge, T_surge = sol_surge.y
rho_surge = np.array([
    surge_density(ch, cn, cw, ca, params_surge)
    for ch, cn, cw, ca in zip(C_H_surge, C_N_surge, C_W_surge, C_A_surge)
])

# Prepare crash tank inputs from the surge tank outlet
interp_surge = {
    "C_H": interp1d(sol_surge.t, C_H_surge, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_N": interp1d(sol_surge.t, C_N_surge, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_W": interp1d(sol_surge.t, C_W_surge, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "C_A": interp1d(sol_surge.t, C_A_surge, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    "T": interp1d(sol_surge.t, T_surge, kind='cubic', bounds_error=False, fill_value='extrapolate'),
}

# -----------------------------
# Solve crash tank
# -----------------------------
y0_crash = [0.0, 0.0, 0.0, 0.0, 293.15, 0.0]

sol_crash = solve_ivp(
    fun=lambda tt, yy: crash_tank_model(tt, yy, params_crash, interp_surge, params_surge["t_open"]),
    t_span=t_span,
    y0=y0_crash,
    t_eval=t_eval,
    method='RK45'
)

# Extract crash tank results
C_H_crash, C_N_crash, C_W_crash, C_A_crash, T_crash, S_crash = sol_crash.y

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

# -----------------------------
# Plots for CSTR1 (conversion, temperature, concentrations)
# -----------------------------
fig1, ax1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax1[0].plot(t, T_cstr1, label="CSTR1 Reactor T")
# TODO: make the jacket temperature dynamic and replace this constant trace with Tc(t).
ax1[0].plot(t, Tc_cstr1, "--", label="CSTR1 Jacket Tc")
ax1[0].set_ylabel("Temperature (K)")
ax1[0].set_title("CSTR1: Temperature vs Time")
ax1[0].grid(True, alpha=0.3)
ax1[0].legend()

ax1[1].plot(t, C_H_cstr1, label="C_H (CSTR1)")
ax1[1].plot(t, C_N_cstr1, label="C_N (CSTR1)")
ax1[1].plot(t, C_W_cstr1, label="C_W (CSTR1)")
ax1[1].plot(t, C_A_cstr1, label="C_A (CSTR1)")
ax1[1].set_ylabel("Concentration (mol/L)")
ax1[1].set_title("CSTR1: Concentrations vs Time")
ax1[1].grid(True, alpha=0.3)
ax1[1].legend()

ax1[2].plot(t, X_H_cstr1, color="black", label="Conversion of H (CSTR1)")
ax1[2].set_xlabel("Time (s)")
ax1[2].set_ylabel("Conversion")
ax1[2].set_title("CSTR1: Conversion vs Time")
ax1[2].set_ylim(0, 1.05)
ax1[2].grid(True, alpha=0.3)
ax1[2].legend()

plt.tight_layout()

# -----------------------------
# Plots for surge tank
# -----------------------------
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax2[0].plot(sol_surge.t, V_surge, color='tab:blue', label='Surge volume')
ax2[0].axhline(params_surge["V"], color='red', linestyle='--', linewidth=1, label='Surge max volume')
ax2[0].set_ylabel('Volume (L)')
ax2[0].set_title('Surge Tank: Volume vs Time')
ax2[0].grid(True, alpha=0.3)
ax2[0].legend()

ax2[1].plot(sol_surge.t, C_A_surge, label='C_A (surge)')
ax2[1].plot(sol_surge.t, C_H_surge, label='C_H (surge)')
ax2[1].plot(sol_surge.t, C_N_surge, label='C_N (surge)')
ax2[1].plot(sol_surge.t, C_W_surge, label='C_W (surge)')
ax2[1].set_ylabel('Concentration (mol/L)')
ax2[1].set_title('Surge Tank: Concentrations vs Time')
ax2[1].grid(True, alpha=0.3)
ax2[1].legend()

ax2[2].plot(sol_surge.t, T_surge, label='T (surge)')
ax2[2].set_xlabel('Time (s)')
ax2[2].set_ylabel('Temperature (K)')
ax2[2].set_title('Surge Tank: Temperature vs Time')
ax2[2].grid(True, alpha=0.3)
ax2[2].legend()

plt.tight_layout()

# -----------------------------
# Plots for crash tank
# -----------------------------
fig3, ax3 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax3[0].plot(sol_crash.t, solid_content_pct, color='brown', label='Solid content (%)')
ax3[0].set_ylabel('Solid content (%)')
ax3[0].set_title('Crash Tank: Solid content vs Time')
ax3[0].grid(True, alpha=0.3)
ax3[0].legend()

ax3[1].plot(sol_crash.t, C_A_crash, label='C_A (crash)')
ax3[1].set_ylabel('Concentration (mol/L)')
ax3[1].set_title('Crash Tank: Liquid A concentration vs Time')
ax3[1].grid(True, alpha=0.3)
ax3[1].legend()

ax3[2].plot(sol_crash.t, T_crash, label='T (crash)')
ax3[2].set_xlabel('Time (s)')
ax3[2].set_ylabel('Temperature (K)')
ax3[2].set_title('Crash Tank: Temperature vs Time')
ax3[2].grid(True, alpha=0.3)
ax3[2].legend()

plt.tight_layout()
plt.show()