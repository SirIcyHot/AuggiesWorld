"""What-if: how much faster would which steps need to be to make D and E?

Each scenario multiplies the Eyring rate constant of selected steps by a
factor f (``R02R_network.RATE_MULTIPLIERS``), solves the isothermal CSTR to
steady state in PharmaPy, and records the molar yield of D and E::

    yield_i = (outlet molar flow of i) / (inlet molar flow of Z1)

A factor f is equivalent to lowering that step's barrier by R*T*ln(f), which
is 6.4 kJ/mol per factor of 10 at 60 C. The top axis shows that conversion.

Run it with::

    python R02R_network_rate_whatif.py            # save and show
    python R02R_network_rate_whatif.py --no-show  # save only

Writes ``results/R02R_network_rate_whatif.png``.
"""

from __future__ import annotations

import argparse

import numpy as np

import R02R_network as net
from PharmaPy.ThermoModule import ParseDatabase

FACTORS = 10.0 ** np.arange(0, 10.5, 1.0)
SCENARIOS = {  # label -> steps multiplied
    "Z5→E": ["Z5 --> E"],
    "Z2→INT5": ["Z2 --> INT5"],
    "INT5→D": ["INT5 --> D"],
    "Z2→INT5 and INT5→D": ["Z2 --> INT5", "INT5 --> D"],
    "all steps": ["all"],
}
# Categorical slots 1-5 in fixed order (blue, orange, aqua, yellow, violet).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1")
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#6b6a66"
GRID = "#e5e4e0"


def yields(mw: dict[str, float]) -> tuple[float, float]:
    """Steady-state molar yields of D and E per mol Z1 fed."""
    reactor = net.run_reactor()
    balances = net.replay_balances(reactor)
    index = {name: j for j, name in enumerate(reactor.name_species)}
    z1_in = balances.inlet[-1][index["Z1"]] / mw["Z1"]
    return tuple(max(balances.outlet[-1][index[p]], 0.0) / mw[p] / z1_in
                 for p in ("D", "E"))


def run_sweep() -> dict[str, np.ndarray]:
    database = ParseDatabase(str(net.DEFAULT_DATABASE_PATH), to_arrays=False)
    mw = dict(zip(database["name_species"], database["mw"]))
    results = {}
    for label, steps in SCENARIOS.items():
        rows = []
        for factor in FACTORS:
            net.RATE_MULTIPLIERS.clear()
            net.RATE_MULTIPLIERS.update({step: factor for step in steps})
            rows.append(yields(mw))
        results[label] = np.array(rows)
        print(f"  {label:<20} D max {results[label][:, 0].max():.3g}  "
              f"E max {results[label][:, 1].max():.3g}")
    net.RATE_MULTIPLIERS.clear()
    return results


def plot(results: dict[str, np.ndarray]):
    import matplotlib.pyplot as plt

    rt_ln10 = net.GAS_CONSTANT * net.OPERATING_TEMPERATURE * np.log(10) / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, (col, product) in zip(axes, enumerate(("D", "E"))):
        for color, (label, data) in zip(COLORS, results.items()):
            ax.plot(FACTORS, data[:, col], "o-", color=color, lw=2, ms=6,
                    mec="white", mew=1.5, label=label)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Factor f on the listed steps' k  (and the equivalent barrier decrease)")
        ax.set_title(f"Yield of {product}", loc="left", fontsize=11)
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ticks = FACTORS[::2]
        ax.set_xticks(ticks)
        drops = [f"−{rt_ln10 * np.log10(f):.0f} kJ/mol" if f > 1
                 else "0 kJ/mol" for f in ticks]
        ax.set_xticklabels([f"$10^{{{int(np.log10(f))}}}$\n{drop}"
                            for f, drop in zip(ticks, drops)])
        ax.minorticks_off()
    axes[0].set_ylabel("mol out per mol Z1 fed")
    axes[0].legend(title="Steps sped up", frameon=False, loc="upper left")
    axes[0].text(1.3, 0.12, "Z5→E, Z2→INT5, INT5→D alone, and all steps:\n"
                 "D yield stays below 0.1% (lines on zero)", fontsize=8.5,
                 color=TEXT_MUTED)
    axes[1].text(1.3, 0.06, "Z2→INT5, INT5→D and both together:\n"
                 "E yield stays below 0.001% (lines on zero)", fontsize=8.5,
                 color=TEXT_MUTED)
    axes[1].axhline(2 / 3, color=TEXT_MUTED, lw=1, ls="--")
    axes[1].text(1.3, 2 / 3 + 0.015, "⅔: the share that reaches Z2 "
                 "(⅓ is stuck as INT1ONO2)", fontsize=8.5, color=TEXT_MUTED)
    fig.suptitle(f"R02R network what-if at "
                 f"{net.OPERATING_TEMPERATURE - 273.15:g} °C, "
                 f"τ = {net.REACTOR_MASS / net.total_feed_flow() / 60:.0f} min",
                 x=0.06, ha="left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figure without opening a window")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    print("Steady-state yields per scenario:")
    fig = plot(run_sweep())
    net.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = net.RESULTS_DIR / "R02R_network_rate_whatif.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
