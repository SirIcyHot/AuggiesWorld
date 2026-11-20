"""CR02R and D01R as ONE ConcurrentVessel, CR02R's slurry feeding D01R live.

The system is::

    D01H slurry (design copy) --+
                                +--> CR02R ==slurry==> D01R --> slurry
    CR02R makeup ---------------+                       ^
                                                        |
                                           FEED_B (antisolvent)

Both vessels come from their own scripts' builders, with those scripts'
settings: ``CR02R.build_reactor()`` (fed its standalone design copy of
D01H's discharge, as in ``CR02R.py``) and ``D01R.build_crystallizer()``.
They are members of one ``PharmaPy.ConcurrentVessel``, so one integrator
advances both through every time step and D01R reads CR02R's outlet as it
is at that instant, not a finished trajectory or a steady stream. That is
what makes this a control study and not just two runs in a row: a move on
CR02R reaches D01R through the link, inside the same solve.

The link is ``Flowsheet.py``'s ``SlurryInlet``, unchanged (imported, not
copied). CR02R holds F, D, E and J crystals and D01R only D, E and J, so
each D01R solid gets CR02R's crystals of its own species and CR02R's
leftover F crystals dissolve in the link (a trace; the report prints it).
D01R is wrapped by ``with_true_slurry_feed`` as in the flowsheet, so the
crystals it is fed count at their real mass flow in its energy balance.

Where this differs from the standalone scripts
----------------------------------------------
- D01R is fed CR02R's slurry (liquid with D, E and J crystals) instead of
  R02R's all-liquid outlet, so ~20 % of its D and E arrives already solid.
- CR02R's products and D01R share one size grid. By default it is the
  full flowsheet's 50 um R-side grid (``Flowsheet.R_CRYSTAL_SIZE_GRID``),
  which the flowsheet tested against 20 um; ``--fine`` uses the standalone
  20 um grid (slower). F stays on CR02R's standalone 20 um grid, which its
  made-up 1 mm design feed needs (see ``CR02R.F_SIZE_GRID``).

Control
-------
CR02R gets one of two controllers (``--mode``), and at ``STEP_TIME``,
after both vessels have settled, a -2 K move:

``jacket``  PharmaPy's PI jacket loop (``JacketPIDContinuousVesselController``,
            default tuning) holding CR02R at 60 C; the set point steps to 58 C.
``open``    PharmaPy's default P level loop, jacket inlet fixed at 60 C as in
            ``CR02R.py``; the jacket inlet steps to 58 C (open loop).

D01R keeps its own default controller (P level loop; D01R is isothermal).
The figure shows the move travelling: CR02R's temperature and crystal
shares, the crystals crossing the link, and D01R's response.

Run it with::

    python CR02R_D01R.py --no-show            # 16 h, PI jacket loop
    python CR02R_D01R.py --mode open --no-show
    python CR02R_D01R.py --runtime 600        # a short run, to check wiring

Results go to ``results/`` next to this file.
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent          # Project work/CR02R_D01R
PROJECT = HERE.parent
RESULTS_DIR = HERE / "results"

for folder in ("Flowsheet", "R02R", "D01R", "CR02R"):
    sys.path.insert(0, str(PROJECT / folder))

import CR02R  # noqa: E402
import D01R  # noqa: E402

# Flowsheet.py sets its coarse size grids on these modules when imported.
# Keep CR02R's standalone F grid; the product grid is chosen in build().
_STANDALONE_GRIDS = dict(F=CR02R.F_SIZE_GRID, R=D01R.CRYSTAL_SIZE_GRID)

import Flowsheet as fs  # noqa: E402
import R02R_control as control  # noqa: E402
from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink  # noqa: E402
from PharmaPy.IntegratorBackends import AssimuloBackend  # noqa: E402
from PharmaPy.ProcessControl_Refactored import (  # noqa: E402
    DefaultContinuousVesselVolume,
    JacketPIDContinuousVesselController,
)

CR02R.F_SIZE_GRID = _STANDALONE_GRIDS["F"]

# ------------------------------------------------------------ settings
RUNTIME = 16 * 3600.0          # s; D01R settles ~10 h in, after CR02R
POINTS = 241                   # saved points, every 4 min
STEP_TIME = 10 * 3600.0        # s
TEMP_BASE = 60 + 273.15        # K, CR02R's set point / jacket inlet
TEMP_STEP = -2.0               # K
SOLVER_OPTIONS = dict(fs.SOLVER_OPTIONS)  # CVode + GMRES, maxkrylov 25
MODES = ("jacket", "open")

SPECIES_COLORS = fs.SPECIES_COLORS
TEXT_PRIMARY, TEXT_MUTED, GRID_COLOR = (fs.TEXT_PRIMARY, fs.TEXT_MUTED,
                                        fs.GRID_COLOR)


def moved(t):
    """CR02R's set point (jacket mode) or jacket inlet (open mode) [K]."""
    return TEMP_BASE + (TEMP_STEP if t >= STEP_TIME else 0.0)


class SteppedJacket(DefaultContinuousVesselVolume):
    """PharmaPy's default P level loop, with the jacket inlet at moved(t)."""

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):
        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)
        if resolved_inlets is None:
            self.operating_conditions[unit.UTILITY_TEMPERATURE] = moved(time)


def make_controller(mode):
    if mode == "jacket":
        return JacketPIDContinuousVesselController(
            moved, temperature=dict(jacket_limits=control.JACKET_LIMITS))
    if mode == "open":
        return SteppedJacket()
    raise ValueError(f"mode must be one of {MODES}")


# ------------------------------------------------------------ builder
def build(mode="jacket", fine=False) -> ConcurrentVessel:
    grid = _STANDALONE_GRIDS["R"] if fine else fs.R_CRYSTAL_SIZE_GRID
    CR02R.PRODUCT_SIZE_GRID = grid
    D01R.CRYSTAL_SIZE_GRID = grid
    names = CR02R.species_names()

    cr02r = control.install_controller(CR02R.build_reactor(),
                                       make_controller(mode))
    d01r = fs.with_true_slurry_feed(
        D01R.build_crystallizer(fs.product_template(names)))

    cv = ConcurrentVessel({"CR02R": cr02r, "D01R": d01r},
                          integrator=AssimuloBackend(
                              options=dict(SOLVER_OPTIONS)))
    fs.attach(cv, StreamLink("CR02R", "D01R", inlet=0, name="CR02R->D01R"),
              fs.SlurryInlet(cr02r, D01R.SOLID_SPECIES),
              fs.product_template(names))
    return cv


def n_states(vessel):
    return int(sum(s.dim for s in vessel.solver_state_collection.states
                   .values()))


# ------------------------------------------------------------ results
def link_history(cr02r):
    """What CR02R discharges into the link at every saved point [kg/s]."""
    t = np.asarray(cr02r.result.time, dtype=float)
    outs = [CR02R.outlet(cr02r, i) for i in range(t.size)]
    return SimpleNamespace(
        time=t,
        crystals={s: np.array([o.crystals[s] for o in outs])
                  for s in CR02R.SOLID_SPECIES},
        dissolved={s: np.array([o.dissolved[s] for o in outs])
                   for s in CR02R.SOLID_SPECIES})


def jacket_history(cr02r, mode):
    t = np.asarray(cr02r.result.time, dtype=float)
    if mode == "open":
        return np.array([moved(ti) for ti in t])
    h = control.controller_history(cr02r)
    return h.jacket


def settled(values, time, window=0.1):
    """Relative change over the last ``window`` of the time before the step."""
    before = time < STEP_TIME if time[-1] > STEP_TIME else np.ones_like(
        time, dtype=bool)
    t, v = time[before], values[before]
    recent = t >= t[-1] * (1 - window)
    ref = max(abs(v[-1]), 1e-30)
    return float(np.ptp(v[recent]) / ref)


def report(cv, mode, wall):
    cr02r, d01r = cv["CR02R"], cv["D01R"]
    print(f"CR02R + D01R as ONE ConcurrentVessel ({mode}): "
          f"{n_states(cr02r)} + {n_states(d01r)} states, wall "
          f"{wall / 60:.1f} min")
    prof = CR02R.history(cr02r)
    d = D01R.history(d01r)
    link = link_history(cr02r)
    t = prof.time
    idx = [int(np.searchsorted(t, STEP_TIME)) - 1] if t[-1] > STEP_TIME \
        else []
    idx.append(t.size - 1)
    for i in idx:
        label = ("before the step" if i != t.size - 1 else "end of run")
        print(f"\n  t = {t[i] / 3600:.1f} h ({label})")
        print(f"    CR02R  T {prof.temp[i] - 273.15:6.2f} C; crystallized "
              + ", ".join(f"{s} {prof.fraction[s][i]:.1%}"
                          for s in CR02R.CRYSTAL_SPECIES))
        print("    link   crystals "
              + ", ".join(f"{s} {link.crystals[s][i] * 1e3:.4f}"
                          for s in CR02R.SOLID_SPECIES) + " g/s")
        print("    D01R   crystallized "
              + ", ".join(f"{s} {d.fraction_crystallized[s][i]:.2%}"
                          for s in D01R.SOLID_SPECIES)
              + "; crystals held "
              + ", ".join(f"{s} {d.crystal_mass[s][i]:.3f}"
                          for s in D01R.SOLID_SPECIES) + " kg")
    coupling = cv.links[0].coupling if hasattr(cv, "links") else None
    if coupling is not None:
        print(f"\n  F crystals dissolved in the link (last evaluation): "
              f"{coupling.dissolved * 1e3:.2e} g/s")
    print("\n  settling before the step (spread over the last 10 %): "
          f"CR02R T {settled(prof.temp, t):.1e}, D01R D crystals "
          f"{settled(d.crystal_mass['D'], t):.1e}, E crystals "
          f"{settled(d.crystal_mass['E'], t):.1e}")
    return prof, d, link


# ------------------------------------------------------------ figure
def _style(ax):
    ax.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)


def plot(cv, mode, prof, d, link):
    import matplotlib.pyplot as plt

    cr02r = cv["CR02R"]
    hours = prof.time / 3600
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    (ax_t, ax_f, ax_l), (ax_d, ax_m, ax_c) = axes

    ax_t.plot(hours, prof.temp - 273.15, color=TEXT_PRIMARY, lw=2,
              label="CR02R")
    jacket = jacket_history(cr02r, mode)
    if jacket is not None:
        ax_t.plot(hours, np.asarray(jacket) - 273.15, color="#2a78d6", lw=1.5,
                  label="jacket inlet")
    if mode == "jacket":
        ax_t.plot(hours, [moved(t) - 273.15 for t in prof.time],
                  color=TEXT_MUTED, lw=1, ls="--", label="set point")
    ax_t.set_ylabel("Temperature [C]")
    ax_t.set_title("CR02R temperature", fontsize=11)
    ax_t.legend(frameon=False, fontsize=8)

    for s in CR02R.CRYSTAL_SPECIES:
        ax_f.plot(hours, prof.fraction[s] * 100, color=SPECIES_COLORS[s],
                  lw=2, label=s)
    ax_f.set_ylabel("Crystal share of discharge [%]")
    ax_f.set_title("CR02R fraction crystallized", fontsize=11)
    ax_f.legend(frameon=False, fontsize=8)

    for s in CR02R.SOLID_SPECIES:
        ax_l.plot(hours, link.crystals[s] * 1e3, color=SPECIES_COLORS[s],
                  lw=2, label=s)
    ax_l.set_ylabel("Crystal flow [g/s]")
    ax_l.set_title("Crystals crossing the CR02R -> D01R link", fontsize=11)
    ax_l.legend(frameon=False, fontsize=8)

    for s in D01R.SOLID_SPECIES:
        ax_d.plot(hours, d.fraction_crystallized[s] * 100,
                  color=SPECIES_COLORS[s], lw=2, label=s)
    ax_d.set_ylabel("Solid share in the tank [%]")
    ax_d.set_title("D01R fraction crystallized", fontsize=11)
    ax_d.legend(frameon=False, fontsize=8)

    for s in D01R.SOLID_SPECIES:
        ax_m.plot(hours, d.crystal_mass[s], color=SPECIES_COLORS[s], lw=2,
                  label=s)
    ax_m.set_ylabel("Crystals held [kg]")
    ax_m.set_title("D01R crystal inventory", fontsize=11)
    ax_m.legend(frameon=False, fontsize=8)

    for s in D01R.SOLID_SPECIES:
        ax_c.plot(hours, d.conc[s], color=SPECIES_COLORS[s], lw=2,
                  label=f"{s} liquid")
    ax_c.set_yscale("log")
    ax_c.set_ylabel("Liquid concentration [mol/L]")
    ax_c.set_title("D01R mother liquor", fontsize=11)
    ax_c.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        _style(ax)
        if prof.time[-1] > STEP_TIME:
            ax.axvline(STEP_TIME / 3600, color=TEXT_MUTED, lw=0.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("Time [h]")
    what = ("set point" if mode == "jacket" else "jacket inlet")
    fig.suptitle(f"CR02R + D01R in one ConcurrentVessel: CR02R {what} "
                 f"{TEMP_STEP:+.0f} K at {STEP_TIME / 3600:.0f} h "
                 f"({'PI jacket loop' if mode == 'jacket' else 'open loop'})",
                 color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def zoom(cv, mode, prof, d, link):
    """The move alone: every response as a change from its pre-step value,
    over the hours after the step."""
    import matplotlib.pyplot as plt

    t = prof.time
    if t[-1] <= STEP_TIME:
        return None
    i0 = int(np.searchsorted(t, STEP_TIME)) - 1
    after = slice(i0, None)
    h = (t[after] - STEP_TIME) / 3600
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax_t, ax_f, ax_d = axes
    ax_t.plot(h, prof.temp[after] - prof.temp[i0], color=TEXT_PRIMARY, lw=2)
    ax_t.set_ylabel("Delta T [K]")
    ax_t.set_title("CR02R temperature", fontsize=11)
    for s in CR02R.CRYSTAL_SPECIES:
        ax_f.plot(h, (prof.fraction[s][after] - prof.fraction[s][i0]) * 100,
                  color=SPECIES_COLORS[s], lw=2, label=s)
        ax_d.plot(h, (link.crystals[s][after] - link.crystals[s][i0]) * 1e3,
                  color=SPECIES_COLORS[s], lw=2, ls="--",
                  label=f"{s} into D01R")
    for s in D01R.CRYSTAL_SPECIES:
        m = d.crystal_mass[s]
        ax_d.plot(h, (m[after] - m[i0]) / max(m[i0], 1e-30) * 100,
                  color=SPECIES_COLORS[s], lw=2, label=f"{s} held in D01R [%]")
    ax_f.set_ylabel("Delta crystal share [percentage points]")
    ax_f.set_title("CR02R fraction crystallized", fontsize=11)
    ax_d.set_ylabel("Change  (flow [g/s], held [%])")
    ax_d.set_title("Through the link into D01R", fontsize=11)
    for ax in axes:
        _style(ax)
        ax.axhline(0, color=TEXT_MUTED, lw=0.8)
        ax.set_xlabel("Hours after the step")
        if ax is not ax_t:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Response to the CR02R move, relative to just before it",
                 color=TEXT_PRIMARY)
    fig.tight_layout()
    return fig


def save(cv, mode, prof, d, link, wall, out_dir=RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fig in ((f"CR02R_D01R_{mode}", plot(cv, mode, prof, d, link)),
                      (f"CR02R_D01R_{mode}_step",
                       zoom(cv, mode, prof, d, link))):
        if fig is not None:
            path = out_dir / f"{name}.png"
            fig.savefig(path, dpi=150)
            saved.append(path)
    np.savez_compressed(
        out_dir / f"CR02R_D01R_{mode}.npz", time=prof.time, temp=prof.temp,
        jacket=np.asarray(jacket_history(cv["CR02R"], mode), dtype=float),
        **{f"cr02r_frac_{s}": prof.fraction[s] for s in CR02R.CRYSTAL_SPECIES},
        **{f"link_crystals_{s}": link.crystals[s]
           for s in CR02R.SOLID_SPECIES},
        **{f"d01r_frac_{s}": d.fraction_crystallized[s]
           for s in D01R.SOLID_SPECIES},
        **{f"d01r_mass_{s}": d.crystal_mass[s] for s in D01R.SOLID_SPECIES},
        **{f"d01r_conc_{s}": d.conc[s] for s in D01R.SOLID_SPECIES},
        wall=wall, step_time=STEP_TIME, step=TEMP_STEP)
    return saved


def main():
    global STEP_TIME
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="jacket")
    parser.add_argument("--runtime", type=float, default=RUNTIME)
    parser.add_argument("--points", type=int, default=POINTS)
    parser.add_argument("--step-time", type=float, default=STEP_TIME)
    parser.add_argument("--fine", action="store_true",
                        help="standalone 20 um product grid (slower)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    STEP_TIME = args.step_time

    cv = build(args.mode, args.fine)
    start = clock.time()
    cv.solve_unit(time_grid=np.linspace(0.0, args.runtime, args.points),
                  verbose=False)
    wall = clock.time() - start
    prof, d, link = report(cv, args.mode, wall)
    for path in save(cv, args.mode, prof, d, link, wall):
        print(f"  saved {path}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return cv


if __name__ == "__main__":
    main()
