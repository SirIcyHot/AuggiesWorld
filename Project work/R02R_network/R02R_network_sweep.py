"""Temperature x residence-time sweep of the full-stoichiometry network CSTR.

Each grid point solves ``R02R_network_full`` to steady state, isothermal at
temperature T, with residence time tau set by the main-feed flow
(tau = REACTOR_MASS / main-feed mass flow; the water stream stays at 2 wt%
of the A fed, so it scales with the feed). The run time is 8 tau.

Recorded per point: molar yield per mol Z1 fed of E, D, Z5 and INT1ONO2.

Caveats that matter for reading the map:

* The reactor is held isothermal: no exotherm, no boiling. H (acetic acid)
  boils at ~118 C at 1 atm, so points above that assume a pressurised
  vessel.
* K (acetyl nitrate) is thermally unstable and is reported to decompose
  violently when heated well above room temperature. The model has no
  decomposition step, so the high-temperature region is a kinetic
  what-if, not an operating recommendation.
* Rate constants are Eyring from E+ZPE barriers (upper bounds for
  multi-molecule steps); K formation and hydrolysis use guessed Arrhenius
  rates extrapolated from 25 C.

Run it with::

    python R02R_network_sweep.py            # sweep, save figure + CSV, show
    python R02R_network_sweep.py --no-show  # sweep and save only
    python R02R_network_sweep.py --from-csv # redraw from the saved CSV

Writes ``results/R02R_network_sweep.png`` and
``results/R02R_network_sweep.csv``.
"""

from __future__ import annotations

import argparse

import numpy as np

import R02R_network as net
import R02R_network_full as full

TEMPS_C = np.arange(40.0, 200.0 + 1e-9, 10.0)
TAUS_H = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
PRODUCTS = ("E", "D", "Z5", "INT1ONO2")
BOILING_POINT_H_C = 118.0  # acetic acid at 1 atm

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#3b3a37"
TEXT_MUTED = "#6b6a66"
GRID = "#e5e4e0"
# Sequential blue ramp (light -> dark), steps 100..700.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95", "#0d366b"]
# Categorical slots for the line panel, in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1",
          "#d6457a")


def run_point(temp_c: float, tau_h: float) -> dict[str, float]:
    flow = net.REACTOR_MASS / (tau_h * 3600.0)
    full.MAIN_FEED = ("Feed", flow, *net.FEEDS[0][2:])
    reactor = full.run_reactor(temp=temp_c + 273.15,
                               runtime=8 * tau_h * 3600.0)
    balances = net.replay_balances(reactor)
    yields = full.molar_yields(reactor, balances, products=PRODUCTS)
    yields["gap"] = net.steady_state_gap(balances)
    return yields


def run_sweep() -> dict[str, np.ndarray]:
    original_feed = full.MAIN_FEED
    grids = {p: np.zeros((len(TAUS_H), len(TEMPS_C)))
             for p in (*PRODUCTS, "gap")}
    try:
        for i, tau_h in enumerate(TAUS_H):
            for j, temp_c in enumerate(TEMPS_C):
                point = run_point(temp_c, tau_h)
                for key, value in point.items():
                    grids[key][i, j] = value
            print(f"  tau {tau_h:>5g} h: E " + " ".join(
                f"{v:.2g}" for v in grids["E"][i]))
    finally:
        full.MAIN_FEED = original_feed
    return grids


def save_csv(grids, path):
    with open(path, "w") as f:
        f.write("tau_h,temp_C," + ",".join(f"yield_{p}" for p in PRODUCTS)
                + ",steady_state_gap\n")
        for i, tau_h in enumerate(TAUS_H):
            for j, temp_c in enumerate(TEMPS_C):
                f.write(f"{tau_h:g},{temp_c:g},"
                        + ",".join(f"{grids[p][i, j]:.6g}" for p in PRODUCTS)
                        + f",{grids['gap'][i, j]:.3g}\n")


def load_csv(path) -> dict[str, np.ndarray]:
    """Grids back from ``save_csv`` output, on TAUS_H x TEMPS_C."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    grids = {p: np.zeros((len(TAUS_H), len(TEMPS_C)))
             for p in (*PRODUCTS, "gap")}
    for row in data:
        i = int(np.argmin(np.abs(TAUS_H - row["tau_h"])))
        j = int(np.argmin(np.abs(TEMPS_C - row["temp_C"])))
        for p in PRODUCTS:
            grids[p][i, j] = row[f"yield_{p}"]
        grids["gap"][i, j] = row["steady_state_gap"]
    return grids


def plot(grids):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    fig = plt.figure(figsize=(16, 11), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.38,
                          wspace=0.22)

    # --- E yield heatmap (linear 0-1, binned so cells read as levels).
    bounds = [0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    cmap = ListedColormap(BLUE_RAMP)
    norm = BoundaryNorm(bounds, cmap.N)
    ax = fig.add_subplot(gs[0, 0])
    _heatmap(ax, grids["E"], cmap, norm,
             title="Yield of E (mol per mol Z1 fed)")
    cbar = fig.colorbar(ax.images[0], ax=ax, ticks=bounds, shrink=0.9,
                        pad=0.02)
    cbar.ax.set_yticklabels([_pct(b) for b in bounds])
    cbar.outline.set_visible(False)

    # --- D yield heatmap on log bins (it never gets large).
    d_bounds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    d_norm = BoundaryNorm(d_bounds, cmap.N)
    ax_d = fig.add_subplot(gs[0, 1])
    _heatmap(ax_d, np.maximum(grids["D"], 1e-12), cmap, d_norm,
             title="Yield of D (mol per mol Z1 fed)")
    cbar = fig.colorbar(ax_d.images[0], ax=ax_d, ticks=d_bounds, shrink=0.9,
                        pad=0.02)
    cbar.ax.set_yticklabels([f"$10^{{{int(np.log10(b))}}}$"
                             for b in d_bounds])
    cbar.outline.set_visible(False)

    # --- E vs T at a few residence times.
    ax_e = fig.add_subplot(gs[1, 0])
    picks = [0.25, 1, 4, 16, 64]
    for color, tau_h in zip(SERIES, picks):
        i = list(TAUS_H).index(tau_h)
        ax_e.plot(TEMPS_C, grids["E"][i], "o-", color=color, lw=2, ms=6,
                  mec=SURFACE, mew=1.5, label=f"{_tau(tau_h)} min")
    ax_e.set_ylabel("Yield of E")
    ax_e.set_title("E vs temperature", loc="left", fontsize=11)
    ax_e.legend(frameon=False, loc="upper left", title="Residence time τ")
    _line_axes(ax_e)

    # --- Where the Z1 goes at tau = 1 h (stacked fractions).
    ax_s = fig.add_subplot(gs[1, 1])
    i = list(TAUS_H).index(1)
    shares = [grids["Z5"][i], grids["E"][i], grids["INT1ONO2"][i]]
    labels = ["Z5", "E", "INT1ONO2"]
    ax_s.stackplot(TEMPS_C, *shares, colors=SERIES[:3], labels=labels,
                   edgecolor=SURFACE, linewidth=1.5)
    ax_s.set_ylim(0, 1.0)
    ax_s.set_ylabel("Fraction of Z1 fed")
    ax_s.set_title("Where the Z1 ends up at τ = 60 min", loc="left",
                   fontsize=11)
    ax_s.legend(frameon=False, loc="center left", reverse=True)
    _line_axes(ax_s)

    fig.suptitle(
        "R02R network (full stoichiometry): temperature × residence-time "
        "sweep, isothermal CSTR", x=0.06, ha="left", fontsize=13)
    fig.text(0.06, 0.015,
             f"H boils at ~{BOILING_POINT_H_C:g} °C at 1 atm (points above "
             "assume a pressurised vessel). K is thermally "
             "unstable and has no decomposition step in the model, so the "
             "hot region is a kinetic what-if.",
             fontsize=9, color=TEXT_MUTED)
    return fig


def _heatmap(ax, data, cmap, norm, title):
    """Plain yield map: colour only, values read from the colour bar."""
    ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=norm,
              extent=(TEMPS_C[0] - 5, TEMPS_C[-1] + 5, -0.5,
                      len(TAUS_H) - 0.5))
    ax.set_yticks(range(len(TAUS_H)))
    ax.set_yticklabels([_tau(t) for t in TAUS_H])
    ax.set_xticks(TEMPS_C[::2])
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Residence time τ [min]")
    ax.set_title(title, loc="left", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _line_axes(ax):
    ax.set_xlabel("Temperature [°C]")
    ax.set_xlim(TEMPS_C[0], TEMPS_C[-1])
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _pct(v):
    if v >= 0.1:
        return f"{v * 100:.0f}%"
    if v >= 0.001:
        return f"{v * 100:.1f}%"
    return "<0.1%" if v > 0 else "0"


def _tau(h):
    """Residence time in minutes, as a plain number."""
    return f"{h * 60:g}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--from-csv", action="store_true",
                        help="redraw from results/R02R_network_sweep.csv "
                             "instead of re-running the sweep")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    csv_path = net.RESULTS_DIR / "R02R_network_sweep.csv"
    if args.from_csv:
        grids = load_csv(csv_path)
    else:
        print(f"Sweeping {len(TEMPS_C)} temperatures x {len(TAUS_H)} "
              "residence times:")
        grids = run_sweep()
        print(f"Largest steady-state gap: {grids['gap'].max():.2%} of "
              "inlet flow")
        net.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_csv(grids, csv_path)
    fig = plot(grids)
    fig.savefig(net.RESULTS_DIR / "R02R_network_sweep.png", dpi=150,
                bbox_inches="tight")
    print(f"Saved to {net.RESULTS_DIR}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
