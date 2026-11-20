"""State-space (LTI) control of R02R, linearized at its steady state.

Used by ``R02R_state_space.ipynb``; this file does the heavy lifting and the
PharmaPy runs, and caches them in ``results/state_space/``.

The plant
---------
R02R (``R02R.build_reactor()`` unchanged) in a one-member ConcurrentVessel,
with PharmaPy's PI level loop on the outlet as part of the plant::

    u = jacket (coolant) inlet temperature   [K]   manipulated
    d = feed temperature                      [K]   disturbance
    y = vessel temperature                    [K]   measured

The state is PharmaPy's own solver state, 14 entries: the liquid mass of
each of the 12 database species, the temperature, and the level loop's
integral. Linearizing ``dx/dt = f(x, u, d)`` at the steady state
``f(x0, u0, d0) = 0`` gives the LTI model::

    dx/dt = A x + B u + E d,     y = C x

in deviation variables (x - x0, u - u0, d - d0). Four species (B, C, K, L)
are never fed or made, so their masses stay at zero: they form an
uncontrollable, unobservable part that the controller drops (``keep``).

What is computed
----------------
1. ``steady_state.npz``: a 10 h run from R02R's charge, then Newton on
   PharmaPy's right-hand side so f(x0) is zero to round-off, not just small.
2. ``linearization.npz``: A, B, E by central differences of
   ``MultiPhaseVessel.unit_model`` at x0.
3. ``open_loop.npz``: a nonlinear run from x0 with a +2 K jacket step and a
   +20 K feed-temperature step, to check the LTI model against.
4. ``closed_loop.npz``: the same scenario for three controllers, all running
   inside PharmaPy on the nonlinear model: PharmaPy's PI jacket loop, and the
   LQG controller designed here (``LQGController``: LQR state feedback with
   integral action, on the state a Kalman filter estimates from T alone),
   once with a plain filter (``lqg``) and once with the feed temperature
   estimated as an extra state (``lqgd``).

Run it with::

    python R02R_state_space.py            # everything (~20 s)
    python R02R_state_space.py --closed-loop-only   # reuse 1-3
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import R02R  # noqa: E402
import R02R_control as control  # noqa: E402
from PharmaPy.DataClasses import (  # noqa: E402
    OperatingKey, PhaseRef, StateKey, StateVariable)
from PharmaPy.ProcessControl_Refactored import (  # noqa: E402
    JacketPIDContinuousVesselController,
    PIDContinuousVesselVolume,
)

OUT_DIR = HERE / "results" / "state_space"
SS_FILE = OUT_DIR / "steady_state.npz"
LIN_FILE = OUT_DIR / "linearization.npz"
OPEN_FILE = OUT_DIR / "open_loop.npz"
CLOSED_FILE = OUT_DIR / "closed_loop.npz"

U0 = R02R.COOLANT_TEMPERATURE      # K, jacket inlet at the operating point
D0 = R02R.FEED_TEMPERATURE         # K, feed temperature
SETTLE_TIME = 10 * 3600.0          # s, ~7 residence times
DT_SAVE = 20.0                     # s

# Open-loop check: jacket +2 K at T_U, feed +20 K at T_D.
OPEN_DU, OPEN_T_U = 2.0, 0.25 * 3600
OPEN_DD, OPEN_T_D = 20.0, 3.0 * 3600
OPEN_RUNTIME = 6.0 * 3600

# Closed-loop scenario: set point -2 K at T_R, feed +20 K at T_D.
# The feed is 2 % of the vessel per minute, so only a large feed-temperature
# swing moves T visibly (DC gain 0.043 K/K).
CL_DR, CL_T_R = -2.0, 0.25 * 3600
CL_DD, CL_T_D = 20.0, 2.5 * 3600
CL_RUNTIME = 5.0 * 3600
JACKET_LIMITS = control.JACKET_LIMITS

# LQG design (scaled coordinates, see ``scaling``). Q weights the
# temperature error and its integral, R the jacket move.
LQ_WEIGHTS = dict(q_temp=1.0, q_integral=1.0 / 60.0**2, r_input=0.05)
# Kalman filter: feed temperature wanders by ~SIGMA_D, every state has a
# small unmodelled drift, and the thermometer reads to ~SIGMA_Y.
KF_NOISE = dict(sigma_d=2.0, sigma_model=1e-3, sigma_y=0.05)
# With the disturbance estimated: the feed temperature is a random walk of
# this intensity [K/sqrt(s)], which sets how fast its estimate follows.
SIGMA_D_WALK = 0.05

FEED_TEMPERATURE = OperatingKey("temp", connection=0,
                                phaseref=PhaseRef("liquid", 0), port="inlet")
TEMP_KEY = StateKey("global_temp")


# ------------------------------------------------------------ controllers
class FeedTemperatureMixin:
    """Feed temperature ``d_func(t) + dd`` as an inlet override."""

    d_func = staticmethod(lambda t: D0)
    dd = 0.0

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):
        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)
        if resolved_inlets is None:
            self.operating_conditions[FEED_TEMPERATURE] = (
                self.d_func(time) + self.dd)


class OpenLoopPlant(FeedTemperatureMixin, PIDContinuousVesselVolume):
    """PI level loop; jacket inlet ``u_func(t) + du``, open loop."""

    def __init__(self, u_func=lambda t: U0, d_func=lambda t: D0, **level):
        super().__init__(**level)
        self.u_func, self.d_func = u_func, d_func
        self.du = 0.0

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):
        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)
        if resolved_inlets is None:
            self.operating_conditions[unit.UTILITY_TEMPERATURE] = (
                self.u_func(time) + self.du)


class JacketPI(FeedTemperatureMixin, JacketPIDContinuousVesselController):
    """PharmaPy's PI jacket loop (default tuning), with the feed override."""

    def __init__(self, setpoint, d_func=lambda t: D0, **kwargs):
        super().__init__(setpoint, **kwargs)
        self.d_func = d_func


class LQGController(FeedTemperatureMixin, PIDContinuousVesselVolume):
    """LQR state feedback with integral action, on a Kalman-filter estimate.

    In deviation variables, on the observer model (Ao, Bo, Co)::

        u  = u0 - K xhat - Kz z                      (clipped to limits)
        dxhat/dt = Ao xhat + Bo (u_applied - u0) + L (y - y0 - Co xhat)
        dz/dt    = (r - y) + (u_applied - u) / (-Kz Tt)

    The observer model is the reduced plant model (A, B, C), or, with
    ``design_lqg(..., disturbance=True)``, that model plus the feed
    temperature as an extra, constant state, so the filter estimates the
    unmeasured disturbance too (K is zero on that entry).

    ``xhat`` (the estimated state) and ``z`` (the integral of the tracking
    error) are solver states, so CVode integrates the observer with the
    plant. The last term of dz/dt is back-calculation anti-windup: while the
    jacket sits at a limit it pulls z back until the law asks for what is
    applied. The level is still held by PharmaPy's PI loop.
    """

    def __init__(self, design, setpoint, d_func=lambda t: D0,
                 jacket_limits=JACKET_LIMITS, tracking_time=60.0, **level):
        super().__init__(**level)
        # Prefixed: DefaultContinuousVesselVolume already uses self.K (its
        # level gain), and a plain self.K would silently replace it.
        self.lqg_A, self.lqg_B, self.lqg_C = design.Ao, design.Bo, design.Co
        self.lqg_K, self.lqg_Kz, self.lqg_L = (design.Ko, float(design.Kz),
                                               design.L)
        self.lqg_u0, self.lqg_y0 = design.u0, design.y0
        self.r_func, self.d_func = setpoint, d_func
        self.jacket_limits = jacket_limits
        self.tt = tracking_time
        n = self.lqg_A.shape[0]
        self.xhat_key, self.z_key = StateKey("lqg_xhat"), StateKey("lqg_z")
        self.lqg_xhat = np.zeros(n)
        self.lqg_z = 0.0
        self._lqg_states = (
            StateVariable(name="lqg_xhat", dim=n, units="", state_type="diff",
                          limit_negative_inventory=False),
            StateVariable(name="lqg_z", dim=1, units="K s",
                          state_type="diff", limit_negative_inventory=False))

    @property
    def solver_states(self):
        return tuple(super().solver_states) + self._lqg_states

    def _state(self, completed_state):
        xhat = np.ravel(completed_state.get(self.xhat_key, self.lqg_xhat))
        z = float(np.ravel(completed_state.get(self.z_key, self.lqg_z))[0])
        return xhat, z

    def jacket(self, completed_state):
        """(law, applied) jacket inlet temperature [K]."""
        xhat, z = self._state(completed_state)
        law = (self.lqg_u0 - float(self.lqg_K @ xhat)
               - self.lqg_Kz * z)
        low, high = self.jacket_limits
        return law, float(np.clip(law, low, high))

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):
        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)
        if resolved_inlets is None:
            self.operating_conditions[unit.UTILITY_TEMPERATURE] = \
                self.jacket(completed_state)[1]

    def get_solver_state_rates(self, time, completed_state, unit,
                               resolved_inlets=None, resolved_outlets=None,
                               operating_conditions=None):
        rates = dict(super().get_solver_state_rates(
            time, completed_state, unit, resolved_inlets, resolved_outlets,
            operating_conditions))
        xhat, _ = self._state(completed_state)
        law, applied = self.jacket(completed_state)
        y = float(np.ravel(completed_state[TEMP_KEY])[0])
        innovation = y - self.lqg_y0 - float(self.lqg_C @ xhat)
        rates[self.xhat_key] = (self.lqg_A @ xhat
                                + self.lqg_B * (applied - self.lqg_u0)
                                + self.lqg_L * innovation)
        rates[self.z_key] = np.array([
            (self.r_func(time) - y) + (applied - law) / (-self.lqg_Kz * self.tt)])
        return rates


# ------------------------------------------------------------ plant
# Explicit CVode tolerances: R02R runs ~35x faster on these than on
# Assimulo's defaults. Tighter than R02R_control's (1e-6 / 1e-8), whose
# integration error (~4 mK) would hide the LTI model's own (~1 mK).
SOLVER_OPTIONS = dict(rtol=1e-8, atol=1e-10)


def wrap(vessel):
    return control.wrap("R02R", vessel, SOLVER_OPTIONS)


def build_at(x, controller, names):
    """R02R starting at solver state ``x`` (masses, T, level integral)
    instead of its charge, with ``controller`` installed."""
    masses, temp = x[:len(names)], float(x[len(names)])
    saved = (R02R.REACTOR_MASS, R02R.INITIAL_REACTOR, R02R.INITIAL_TEMPERATURE)
    try:
        R02R.REACTOR_MASS = float(masses.sum())
        R02R.INITIAL_REACTOR = {n: float(m / masses.sum())
                                for n, m in zip(names, masses)}
        R02R.INITIAL_TEMPERATURE = temp
        vessel = R02R.build_reactor()
    finally:
        (R02R.REACTOR_MASS, R02R.INITIAL_REACTOR,
         R02R.INITIAL_TEMPERATURE) = saved
    controller.level_integral = float(x[len(names) + 1])
    return control.install_controller(vessel, controller)


def state_labels(vessel):
    labels = []
    collection = vessel.solver_state_collection
    for key in collection.states:
        dim = int(collection.states[key].dim)
        name = key.name
        if name == "mass_j":
            labels += [f"m_{s}" for s in vessel.name_species]
        else:
            labels += [f"{name}[{i}]" for i in range(dim)] if dim > 1 \
                else [name]
    return labels


def packed_state(vessel, index=-1):
    collection = vessel.solver_state_collection
    return np.concatenate([
        np.ravel(np.asarray(getattr(vessel.result,
                                    collection.format_key(k)))[index])
        for k in collection.states])


def rhs(vessel, t, x):
    # unit_model reuses its output buffer: copy it.
    return np.array(vessel.unit_model(t, x), dtype=float, copy=True).ravel()


def jacobian(vessel, x, t=0.0, rel=1e-6):
    n = x.size
    J = np.empty((n, n))
    scale = np.maximum(np.abs(x), 1e-6 * max(np.abs(x).max(), 1.0))
    for j in range(n):
        h = rel * scale[j]
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (rhs(vessel, t, xp) - rhs(vessel, t, xm)) / (2 * h)
    return J


def input_column(vessel, x, attr, h=1e-2, t=0.0):
    ctrl = vessel.controller
    setattr(ctrl, attr, +h)
    fp = rhs(vessel, t, x)
    setattr(ctrl, attr, -h)
    fm = rhs(vessel, t, x)
    setattr(ctrl, attr, 0.0)
    return (fp - fm) / (2 * h)


def find_steady_state():
    """10 h from the charge, then Newton on f(x) = 0."""
    plant = OpenLoopPlant()
    cv = wrap(control.install_controller(R02R.build_reactor(), plant))
    wall = control.solve(cv, SETTLE_TIME, int(SETTLE_TIME / 300) + 1)
    vessel = cv["R02R"]
    names = list(vessel.name_species)
    x = packed_state(vessel)
    target_volume = float(plant.target_volume)
    f_run = rhs(vessel, 0.0, x)

    # Newton. Species that are never present stay exactly at zero.
    history = [np.linalg.norm(f_run)]
    for _ in range(6):
        J = jacobian(vessel, x)
        step = np.linalg.solve(J, -rhs(vessel, 0.0, x))
        x = x + step
        history.append(np.linalg.norm(rhs(vessel, 0.0, x)))
        if history[-1] < 1e-13:
            break
    # Species that are never present come out at round-off (~1e-20 kg,
    # either sign); PharmaPy refuses a negative mass fraction, so zero them.
    tiny = np.abs(x[:len(names)]) < 1e-12
    x[:len(names)][tiny] = 0.0
    history[-1] = np.linalg.norm(rhs(vessel, 0.0, x))
    np.savez_compressed(
        SS_FILE, x0=x, x_run=packed_state(vessel), f_run=f_run,
        newton=np.array(history), names=np.array(names),
        labels=np.array(state_labels(vessel)), target_volume=target_volume,
        time=np.asarray(vessel.result.time),
        temp=np.ravel(vessel.result.global_temp),
        mass=np.asarray(vessel.result.mass_j_liquid0), wall=wall)
    print(f"steady state: run {wall:.0f} s, |f| {history[0]:.2e} -> "
          f"{history[-1]:.2e} after {len(history) - 1} Newton steps, "
          f"T0 = {x[len(names)]:.4f} K")


def linearize():
    ss = np.load(SS_FILE)
    x0, names = ss["x0"], list(ss["names"])
    plant = OpenLoopPlant(target_volume=float(ss["target_volume"]))
    vessel = wrap(build_at(x0, plant, names))["R02R"]
    vessel.compile_structure()
    start = clock.time()
    A = jacobian(vessel, x0)
    B = input_column(vessel, x0, "du")[:, None]
    E = input_column(vessel, x0, "dd")[:, None]
    labels = state_labels(vessel)
    f0 = rhs(vessel, 0.0, x0)
    np.savez_compressed(LIN_FILE, A=A, B=B, E=E, x0=x0, f0=f0, u0=U0, d0=D0,
                        labels=np.array(labels), names=np.array(names),
                        i_temp=labels.index("global_temp"),
                        target_volume=float(ss["target_volume"]),
                        wall=clock.time() - start)
    print(f"linearization: {len(x0)} states in {clock.time() - start:.1f} s, "
          f"|f(x0)| = {np.linalg.norm(f0):.2e}")


# ------------------------------------------------------------ model and design
def load_model():
    """The LTI model, the reduced (controller) model and the output maps."""
    lin = np.load(LIN_FILE)
    A, B, E, x0 = lin["A"], lin["B"], lin["E"], lin["x0"]
    labels, names = list(lin["labels"]), list(lin["names"])
    n_sp = len(names)
    i_temp = int(lin["i_temp"])
    C = np.zeros((1, A.shape[0]))
    C[0, i_temp] = 1.0
    # Species that are never present: zero mass, and nothing else drives
    # them (their row of A is diagonal, so they only wash out).
    off = np.abs(A - np.diag(np.diag(A)))
    absent = [j for j in range(n_sp)
              if abs(x0[j]) < 1e-9 and off[j].max() <= 1e-12 * np.abs(A).max()
              and B[j, 0] == 0.0 and E[j, 0] == 0.0]
    keep = [i for i in range(A.shape[0]) if i not in absent]

    # Mass fraction w_j = m_j / M, linearized: dw_j = (dm_j - w_j dM) / M.
    mass = x0[:n_sp]
    total = mass.sum()
    Cw = np.zeros((n_sp, A.shape[0]))
    Cw[:, :n_sp] = (np.eye(n_sp) - np.outer(mass / total, np.ones(n_sp))) \
        / total
    return SimpleNamespace(
        A=A, B=B, E=E, C=C, Cw=Cw, x0=x0, u0=float(lin["u0"]),
        d0=float(lin["d0"]), labels=labels, names=names, i_temp=i_temp,
        absent=absent, keep=keep, target_volume=float(lin["target_volume"]),
        f0=lin["f0"])


def reduced(model):
    k = model.keep
    return SimpleNamespace(
        A=model.A[np.ix_(k, k)], B=model.B[k], E=model.E[k], C=model.C[:, k],
        Cw=model.Cw[:, k], x0=model.x0[k],
        labels=[model.labels[i] for i in k])


def scaling(red):
    """S = diag(|steady-state change per 1 K of jacket|), floored. In
    x = S xs every scaled state moves by O(1) for a 1 K jacket step, which
    keeps Gramians, Riccati equations and pole placement well conditioned."""
    gain = np.abs(np.linalg.solve(red.A, -red.B).ravel())
    return np.maximum(gain, 1e-3 * gain.max())


def design_lqg(model, weights=LQ_WEIGHTS, noise=KF_NOISE, disturbance=False,
               sigma_walk=SIGMA_D_WALK):
    """LQI state feedback + steady-state Kalman filter, physical units.

    With ``disturbance``, the filter's model carries the feed temperature as
    an extra state (dd/dt = noise), so it estimates d instead of absorbing
    it as a biased state estimate."""
    red = reduced(model)
    s = scaling(red)
    Si = np.diag(1 / s)
    As, Bs, Es, Cs = Si @ red.A @ np.diag(s), Si @ red.B, Si @ red.E, \
        red.C @ np.diag(s)
    n = As.shape[0]

    # LQI: augment z' = r - y = -C x.
    Aa = np.block([[As, np.zeros((n, 1))], [-Cs, np.zeros((1, 1))]])
    Ba = np.vstack([Bs, [[0.0]]])
    Q = np.zeros((n + 1, n + 1))
    Q[:n, :n] = weights["q_temp"] * Cs.T @ Cs
    Q[n, n] = weights["q_integral"]
    R = np.array([[weights["r_input"]]])
    P = linalg.solve_continuous_are(Aa, Ba, Q, R)
    Ka = np.linalg.solve(R, Ba.T @ P)
    Ks, Kz = Ka[:, :n], Ka[0, n]

    # Kalman filter on the scaled model.
    V = np.array([[noise["sigma_y"]**2]])
    if disturbance:
        Ao = np.block([[As, Es], [np.zeros((1, n + 1))]])
        Co = np.hstack([Cs, [[0.0]]])
        W = np.zeros((n + 1, n + 1))
        W[:n, :n] = noise["sigma_model"]**2 * np.eye(n)
        W[n, n] = sigma_walk**2
        so = np.r_[s, 1.0]
    else:
        Ao, Co = As, Cs
        W = (noise["sigma_d"]**2 * Es @ Es.T
             + noise["sigma_model"]**2 * np.eye(n))
        so = s
    Pf = linalg.solve_continuous_are(Ao.T, Co.T, W, V)
    Ls = Pf @ Co.T @ np.linalg.inv(V)
    no = Ao.shape[0]
    Ko = np.zeros(no)
    Ko[:n] = (Ks @ Si).ravel()

    return SimpleNamespace(
        A=red.A, B=red.B.ravel(), C=red.C.ravel(), E=red.E.ravel(),
        K=(Ks @ Si).ravel(), Kz=Kz, L=(np.diag(so) @ Ls).ravel(),
        Ao=np.diag(so) @ Ao @ np.diag(1 / so),
        Bo=np.r_[red.B.ravel(), [0.0] * (no - n)],
        Co=(Co @ np.diag(1 / so)).ravel(), Ko=Ko, disturbance=disturbance,
        x0=red.x0, u0=model.u0, y0=float(model.x0[model.i_temp]),
        scale=s, Ka_scaled=Ka, Ls_scaled=Ls, Pf_scaled=Pf, red=red)


def lqg_closed_loop(design):
    """Linear closed loop (A, B, C, D): states [x, xhat, z], inputs [r, d],
    outputs [y, u - u0]. Unclipped, as the nonlinear run is unless at a
    limit. D is zero: the set point reaches u only through z."""
    A, B, C, E = design.A, design.B[:, None], design.C[None, :], \
        design.E[:, None]
    Ao, Bo, Co = design.Ao, design.Bo[:, None], design.Co[None, :]
    K, L, Kz = design.Ko[None, :], design.L[:, None], design.Kz
    n, no = A.shape[0], Ao.shape[0]
    Z = np.zeros
    # u = -K xhat - Kz z;  xhat' = Ao xhat + Bo u + L (C x - Co xhat)
    Acl = np.block([
        [A, -B @ K, -B * Kz],
        [L @ C, Ao - L @ Co - Bo @ K, -Bo * Kz],
        [-C, Z((1, no)), Z((1, 1))]])
    Bcl = np.block([[Z((n, 1)), E], [Z((no, 1)), Z((no, 1))],
                    [np.ones((1, 1)), Z((1, 1))]])
    Ccl = np.block([[C, Z((1, no)), Z((1, 1))],
                    [Z((1, n)), -K, -Kz * np.ones((1, 1))]])
    return Acl, Bcl, Ccl, np.zeros((2, 2))


def pi_closed_loop(design, kc, ti):
    """PharmaPy's PI jacket loop on the linear plant. Its law, in deviation
    variables: u = r + Kc (r - y + I / Ti), I' = r - y. States [x, I],
    inputs [r, d], outputs [y, u - u0]. D carries the set point straight
    into u, (1 + Kc) r: the feedforward kick."""
    A, B, C, E = design.A, design.B[:, None], design.C[None, :], \
        design.E[:, None]
    n = A.shape[0]
    Acl = np.block([[A - kc * B @ C, kc / ti * B], [-C, np.zeros((1, 1))]])
    Bcl = np.block([[(1 + kc) * B, E], [np.ones((1, 1)), np.zeros((1, 1))]])
    Ccl = np.block([[C, np.zeros((1, 1))],
                    [-kc * C, kc / ti * np.ones((1, 1))]])
    Dcl = np.array([[0.0, 0.0], [1.0 + kc, 0.0]])
    return Acl, Bcl, Ccl, Dcl


# ------------------------------------------------------------ nonlinear runs
def step(t0, size, base):
    return lambda t: base + (size if t >= t0 else 0.0)


def record(vessel, extra=None):
    res = vessel.result
    mass = np.asarray(res.mass_j_liquid0, dtype=float)
    out = dict(time=np.asarray(res.time, dtype=float),
               temp=np.ravel(res.global_temp).astype(float), mass=mass,
               frac=mass / mass.sum(axis=1, keepdims=True),
               volume=np.ravel(res.vessel_vol).astype(float))
    out.update(extra or {})
    return out


def run_open_loop():
    model = load_model()
    u_func, d_func = step(OPEN_T_U, OPEN_DU, U0), step(OPEN_T_D, OPEN_DD, D0)
    plant = OpenLoopPlant(u_func, d_func, target_volume=model.target_volume)
    cv = wrap(build_at(model.x0, plant, model.names))
    wall = control.solve(cv, OPEN_RUNTIME, int(OPEN_RUNTIME / DT_SAVE) + 1)
    data = record(cv["R02R"])
    data.update(u=np.array([u_func(t) for t in data["time"]]),
                d=np.array([d_func(t) for t in data["time"]]), wall=wall)
    np.savez_compressed(OPEN_FILE, **data)
    print(f"open loop: {wall:.0f} s, saved {OPEN_FILE.name}")


def run_closed_loop():
    model = load_model()
    y0 = float(model.x0[model.i_temp])
    r_func, d_func = step(CL_T_R, CL_DR, y0), step(CL_T_D, CL_DD, D0)
    out = {}

    # PharmaPy's PI jacket loop.
    pi = JacketPI(r_func, d_func, target_volume=model.target_volume,
                  temperature=dict(jacket_limits=JACKET_LIMITS))
    cv = wrap(build_at(model.x0, pi, model.names))
    # Start the integral where it holds u0 at the operating point:
    # u = r + Kc I / Ti = u0  ->  I = (u0 - y0) Ti / Kc.
    # One right-hand side at x0 lets the loop read Ti = m cp / (U A).
    vessel = cv["R02R"]
    vessel.compile_structure()
    rhs(vessel, 0.0, np.append(model.x0, 0.0))
    kc, ti, _, _ = pi.temperature_tuning()
    pi.temperature_integral = (U0 - y0) * ti / kc
    pi.level_integral = float(model.x0[-1])
    wall = control.solve(cv, CL_RUNTIME, int(CL_RUNTIME / DT_SAVE) + 1)
    rec = record(vessel)
    key = pi.temperature_loop.integral_key
    integral = np.ravel(getattr(vessel.result, key.name))
    rec["u"] = np.array([pi.jacket_temperature(
        t, {TEMP_KEY: T, key: I}, vessel)[1]
        for t, T, I in zip(rec["time"], rec["temp"], integral)])
    out.update({f"pi_{k}": v for k, v in rec.items()},
               pi_kc=kc, pi_ti=ti, pi_wall=wall)
    print(f"closed loop, PI (Kc {kc:g}, Ti {ti:.1f} s): {wall:.0f} s")

    # LQG, with the plain Kalman filter (tag lqg) and with the feed
    # temperature estimated as an extra state (tag lqgd).
    for tag, disturbance in (("lqg", False), ("lqgd", True)):
        design = design_lqg(model, disturbance=disturbance)
        lqg = LQGController(design, r_func, d_func,
                            target_volume=model.target_volume)
        cv = wrap(build_at(model.x0, lqg, model.names))
        vessel = cv["R02R"]
        wall = control.solve(cv, CL_RUNTIME, int(CL_RUNTIME / DT_SAVE) + 1)
        rec = record(vessel)
        xhat = np.asarray(vessel.result.lqg_xhat, dtype=float)
        z = np.ravel(vessel.result.lqg_z)
        rec["u"] = np.array([lqg.jacket({lqg.xhat_key: xh, lqg.z_key: zz})[1]
                             for xh, zz in zip(xhat, z)])
        out.update({f"{tag}_{k}": v for k, v in rec.items()})
        out.update({f"{tag}_xhat": xhat, f"{tag}_z": z, f"{tag}_wall": wall,
                    f"{tag}_K": design.K, f"{tag}_Kz": design.Kz,
                    f"{tag}_L": design.L})
        print(f"closed loop, LQG ({tag}): {wall:.0f} s")
    out.update(r=np.array([r_func(t) for t in rec["time"]]),
               d=np.array([d_func(t) for t in rec["time"]]),
               keep=np.array(model.keep))
    np.savez_compressed(CLOSED_FILE, **out)
    print(f"saved {CLOSED_FILE.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--closed-loop-only", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.closed_loop_only:
        find_steady_state()
        linearize()
        run_open_loop()
    run_closed_loop()
