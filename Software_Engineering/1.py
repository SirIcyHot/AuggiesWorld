"""
Toy parameter estimation: fit k1, k2 of a CSTR (A -> B -> C) to noisy data.

Structure:
  1. model()      - the ODE right-hand side (what you already know)
  2. simulate()   - integrate with BDF (stand-in for CVode)
  3. residuals()  - simulated minus measured, as a VECTOR
  4. gauss_newton - a hand-written optimizer, so nothing is a black box
  5. scipy's least_squares on the same problem, for comparison
  6. plots: the cost landscape with the optimizer's path, and the final fit
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# ---------------------------------------------------------------- 1. model
TAU = 5.0      # residence time V/q [min]
CA0 = 1.0      # feed concentration of A [mol/L]


def model(t, c, k1, k2):
    ca, cb = c
    dca = (CA0 - ca) / TAU - k1 * ca
    dcb = -cb / TAU + k1 * ca - k2 * cb
    return [dca, dcb]

# ---------------------------------------------------------------- 2. simulate


T_MEAS = np.linspace(0.5, 20, 25)


def simulate(k1, k2):
    sol = solve_ivp(
        model,
        (0, T_MEAS[-1]),
        [0.0, 0.0],
        args=(k1, k2),
        method="BDF",
        t_eval=T_MEAS,
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.y  # shape (2, n_times): rows are CA, CB

# "Experimental" data: true parameters plus measurement noise


K_TRUE = np.array([0.5, 0.2])
rng = np.random.default_rng(0)
DATA = simulate(*K_TRUE) + rng.normal(0, 0.01, (2, T_MEAS.size))

# ---------------------------------------------------------------- 3. residuals
# The optimizer works on p = log(k). Rate constants are positive and can
# span orders of magnitude, so log-space keeps them positive and well scaled.


def residuals(p):
    k1, k2 = np.exp(p)
    return (simulate(k1, k2) - DATA).ravel()  # one long vector


def cost(p):
    r = residuals(p)
    return 0.5 * r @ r                           # f(p) = 1/2 * sum(r^2)

# ------------------------------------------------------ 4. hand-written
# Gauss-Newton


def fd_jacobian(p, h=1e-6):
    """Compute J by forward differences (one simulation per parameter).

    J[i, j] = d r_i / d p_j.
    """
    r0 = residuals(p)
    J = np.empty((r0.size, p.size))
    for j in range(p.size):
        dp = np.zeros_like(p)
        dp[j] = h
        J[:, j] = (residuals(p + dp) - r0) / h
    return J, r0


def gauss_newton(p0, tol=1e-10, max_iter=50):
    p = p0.copy()
    lam = 1e-3                        # Levenberg-Marquardt damping
    path = [p.copy()]
    print(f"{'iter':>4} {'k1':>9} {'k2':>9} {'cost':>12} {'|grad|':>10}")
    for it in range(max_iter):
        J, r = fd_jacobian(p)
        f = 0.5 * r @ r
        grad = J.T @ r                # gradient of f
        H = J.T @ J                   # Gauss-Newton approximate Hessian
        print(f"{it:4d} {np.exp(p[0]):9.5f} {np.exp(p[1]):9.5f} "
              f"{f:12.4e} {np.linalg.norm(grad):10.2e}")
        if np.linalg.norm(grad) < tol:
            break                     # grad f = 0  ->  we're at the bottom
        # Try a step; if it doesn't lower the cost, damp harder and retry
        # (same spirit as CVode rejecting a step and shrinking h)
        while True:
            dp = np.linalg.solve(H + lam * np.diag(np.diag(H)), -grad)
            if cost(p + dp) < f:
                p = p + dp
                lam = max(lam / 10, 1e-12)
                break
            lam *= 10
        path.append(p.copy())
    return p, np.array(path)


if __name__ == "__main__":
    p0 = np.log([2.0, 0.02])          # deliberately bad initial guess
    print("True k:", K_TRUE)
    print("\n--- hand-written Gauss-Newton ---")
    p_gn, path = gauss_newton(p0)
    print("Fitted k:", np.exp(p_gn))

    # ------------------------------------------------------------ 5. scipy
    print("\n--- scipy.optimize.least_squares ---")
    sol = least_squares(residuals, p0, method="trf", x_scale="jac")
    print("Fitted k:", np.exp(sol.x), f"  ({sol.nfev} residual evaluations)")

    # ------------------------------------------------------------ 6. plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    lk1 = np.linspace(np.log(0.1), np.log(3.0), 45)
    lk2 = np.linspace(np.log(0.01), np.log(1.0), 45)
    F = np.array([[cost(np.array([a, b])) for a in lk1] for b in lk2])
    K1, K2 = np.meshgrid(np.exp(lk1), np.exp(lk2))
    cs = ax1.contourf(K1, K2, np.log10(F), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax1, label="log10(cost)")
    ax1.plot(
        *np.exp(path).T,
        "o-",
        color="white",
        label="optimizer path",
    )
    ax1.plot(*K_TRUE, "r*", ms=15, label="true k")
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="k1 [1/min]",
        ylabel="k2 [1/min]",
        title="Cost landscape: the bowl",
    )
    ax1.legend()

    t_fine = np.linspace(0, 20, 200)
    k_fit = np.exp(p_gn)
    fine = solve_ivp(model, (0, 20), [0, 0], args=tuple(k_fit),
                     method="BDF", t_eval=t_fine, rtol=1e-10).y
    for i, name in enumerate(["CA", "CB"]):
        ax2.plot(T_MEAS, DATA[i], "o", label=f"{name} data")
        ax2.plot(t_fine, fine[i], "-", label=f"{name} fit")
    ax2.set(xlabel="time [min]", ylabel="conc [mol/L]", title="Fit vs data")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("cstr_fit.png", dpi=120)
    print("\nSaved cstr_fit.png")
    plt.show()
