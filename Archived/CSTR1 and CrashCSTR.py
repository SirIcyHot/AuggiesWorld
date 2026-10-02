import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# -----------------------------
# Parameters
# -----------------------------
params = {
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
    "rho": 1000.0,   # kg/m^3
    "Cp": 4.18,      # J/(g*K) or adjust units consistently

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
# Crash tank (CSTR2) parameters
# -----------------------------
params_crash = {
    "F_in": params["F"],   # inlet flow from CSTR1 (L/s)
    "F_N": 0.2,             # additional N feed (L/s)
    "V": 5.0,               # crash tank volume (L)
    "Tc": 290.0,           # jacket temperature (K)
    "UA": 3000.0,
    "rho": params["rho"],
    "Cp": params["Cp"],
    # precipitation kinetics
    "k_p": 0.5,            # 1/s per concentration unit (phenomenological)
    "C_A_eq": 0.05,        # saturation concentration (mol/L)
    # external N feed concentration (pure N)
    "C_N_ext": 5.0,
    "T_N": 293.15,
    # molar mass for A (g/mol) used for solid mass calc
    "M_A_gmol": 100.0
}

# -----------------------------
# ODE System
# -----------------------------
def cstr_model(t, y, p):
    C_H, C_N, C_W, C_A, T = y

    # Unpack parameters
    F, V = p["F"], p["V"]
    Tf, Tc = p["Tf"], p["Tc"]
    rho, Cp = p["rho"], p["Cp"]
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


# Named wrapper for CSTR1 (keeps original behavior)
def cstr1_model(t, y):
    return cstr_model(t, y, params)


# Crash tank model (CSTR2) — includes precipitation to solid A and jacket
def crash_model(t, y, p, out_interp):
    C_H, C_N, C_W, C_A, T, S = y

    # inlet from CSTR1 (time-varying)
    C_H_in = out_interp['C_H'](t)
    C_N_in = out_interp['C_N'](t)
    C_W_in = out_interp['C_W'](t)
    C_A_in = out_interp['C_A'](t)
    T_in   = out_interp['T'](t)

    F1 = p['F_in']
    F_N = p['F_N']
    F_tot = F1 + F_N

    # mixed inlet concentrations (weighted by flow)
    C_Hf = (F1*C_H_in + F_N*0.0) / F_tot
    C_Nf = (F1*C_N_in + F_N*p['C_N_ext']) / F_tot
    C_Wf = (F1*C_W_in + F_N*0.0) / F_tot
    C_Af = (F1*C_A_in + F_N*0.0) / F_tot
    Tf   = (F1*T_in + F_N*p['T_N']) / F_tot

    # precipitation (simple phenomenological rate)
    k_p = p['k_p']
    C_A_eq = p['C_A_eq']
    r_p = k_p * max(C_A - C_A_eq, 0.0)

    dC_H_dt = (F_tot/p['V']) * (C_Hf - C_H)
    dC_N_dt = (F_tot/p['V']) * (C_Nf - C_N)
    dC_W_dt = (F_tot/p['V']) * (C_Wf - C_W)
    dC_A_dt = (F_tot/p['V']) * (C_Af - C_A) - r_p

    # solid accumulation (moles of A solid)
    dS_dt = r_p * p['V']

    # energy balance (no heat of precipitation included here)
    dT_dt = (F_tot/p['V'])*(Tf - T) - (p['UA']/(p['rho']*p['Cp']*p['V']))*(T - p['Tc'])

    return [dC_H_dt, dC_N_dt, dC_W_dt, dC_A_dt, dT_dt, dS_dt]

# -----------------------------
# Initial Conditions
# -----------------------------
y0 = [
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
# Solve ODE
# -----------------------------
sol = solve_ivp(
    fun=lambda t, y: cstr_model(t, y, params),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45'
)

# -----------------------------
# Extract Results
# -----------------------------
C_H, C_N, C_W, C_A, T = sol.y
t = sol.t
Tc = np.full_like(t, params["Tc"])
X_H = (params["C_Hf"] - C_H) / params["C_Hf"]

# Example: print final values
print("Final concentrations and temperature:")
print(f"C_H = {C_H[-1]:.3f}")
print(f"C_N = {C_N[-1]:.3f}")
print(f"C_W = {C_W[-1]:.3f}")
print(f"C_A = {C_A[-1]:.3f}")
print(f"T   = {T[-1]:.2f} K")

# -----------------------------
# Prepare crash tank inputs (interpolate CSTR1 outlet)
# -----------------------------
interp_funcs = {
    'C_H': interp1d(t, C_H, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    'C_N': interp1d(t, C_N, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    'C_W': interp1d(t, C_W, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    'C_A': interp1d(t, C_A, kind='cubic', bounds_error=False, fill_value='extrapolate'),
    'T':   interp1d(t, T,   kind='cubic', bounds_error=False, fill_value='extrapolate')
}

# Initial conditions for crash tank: use initial outlet of CSTR1
y0_crash = [C_H[0], C_N[0], C_W[0], C_A[0], T[0], 0.0]

# Solve crash tank using interpolated inlet from CSTR1
sol2 = solve_ivp(
    fun=lambda tt, yy: crash_model(tt, yy, params_crash, interp_funcs),
    t_span=t_span,
    y0=y0_crash,
    t_eval=t_eval,
    method='RK45'
)

# Extract crash tank results
C_H2, C_N2, C_W2, C_A2, T2, S = sol2.y

# Compute solid loading/content% by mass
V_m3 = params_crash['V'] / 1000.0
mass_liquid_kg = params_crash['rho'] * V_m3
M_A_kg_per_mol = params_crash['M_A_gmol'] / 1000.0
mass_solid_kg = S * M_A_kg_per_mol
solid_loading_pct = 100.0 * mass_solid_kg / (mass_solid_kg + mass_liquid_kg + 1e-12)

# -----------------------------
# Plots for CSTR1 (conversion, temperature, concentrations)
# -----------------------------
fig1, ax1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax1[0].plot(t, T, label="CSTR1 Reactor T")
# TODO: make the jacket temperature dynamic and replace this constant trace with Tc(t).
ax1[0].plot(t, Tc, "--", label="CSTR1 Jacket Tc")
ax1[0].set_ylabel("Temperature (K)")
ax1[0].set_title("CSTR1: Temperature vs Time")
ax1[0].grid(True, alpha=0.3)
ax1[0].legend()

ax1[1].plot(t, C_H, label="C_H (CSTR1)")
ax1[1].plot(t, C_N, label="C_N (CSTR1)")
ax1[1].plot(t, C_W, label="C_W (CSTR1)")
ax1[1].plot(t, C_A, label="C_A (CSTR1)")
ax1[1].set_ylabel("Concentration (mol/L)")
ax1[1].set_title("CSTR1: Concentrations vs Time")
ax1[1].grid(True, alpha=0.3)
ax1[1].legend()

ax1[2].plot(t, X_H, color="black", label="Conversion of H (CSTR1)")
ax1[2].set_xlabel("Time (s)")
ax1[2].set_ylabel("Conversion")
ax1[2].set_title("CSTR1: Conversion vs Time")
ax1[2].set_ylim(0, 1.05)
ax1[2].grid(True, alpha=0.3)
ax1[2].legend()

plt.tight_layout()

# -----------------------------
# Plots for Crash Tank (CSTR2): solid loading, concentration A, temperature
# -----------------------------
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax2[0].plot(sol2.t, solid_loading_pct, color='brown', label='Solid loading (%)')
ax2[0].set_ylabel('Solid loading (%)')
ax2[0].set_title('Crash Tank: Solid loading vs Time')
ax2[0].grid(True, alpha=0.3)
ax2[0].legend()

ax2[1].plot(sol2.t, C_A2, label='C_A (liquid, CSTR2)')
ax2[1].set_ylabel('Concentration (mol/L)')
ax2[1].set_title('Crash Tank: Liquid A concentration vs Time')
ax2[1].grid(True, alpha=0.3)
ax2[1].legend()

ax2[2].plot(sol2.t, T2, label='T (CSTR2)')
ax2[2].set_xlabel('Time (s)')
ax2[2].set_ylabel('Temperature (K)')
ax2[2].set_title('Crash Tank: Temperature vs Time')
ax2[2].grid(True, alpha=0.3)
ax2[2].legend()

plt.tight_layout()
plt.show()