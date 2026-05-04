import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
# Plots
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(t, T, label="Reactor temperature, T")
# TODO: make the jacket temperature dynamic and replace this constant trace with Tc(t).
axes[0].plot(t, Tc, "--", label="Jacket temperature, Tc")
axes[0].set_ylabel("Temperature (K)")
axes[0].set_title("Temperature vs Time")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(t, C_H, label="C_H")
axes[1].plot(t, C_N, label="C_N")
axes[1].plot(t, C_W, label="C_W")
axes[1].plot(t, C_A, label="C_A")
axes[1].set_ylabel("Concentration (mol/L)")
axes[1].set_title("Concentrations vs Time")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(t, X_H, color="black", label="Conversion of H")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Conversion")
axes[2].set_title("Conversion vs Time")
axes[2].set_ylim(0, 1.05)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()