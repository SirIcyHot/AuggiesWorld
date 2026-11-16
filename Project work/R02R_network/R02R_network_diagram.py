"""Diagram of why D and E barely form in the R02R network CSTR.

Two panels, both from one run of ``R02R_network.py`` at its operating
temperature:

* **Network map.** Each node is a species, labelled with its steady-state
  concentration. Each arrow is a step: its colour and width show k*tau, how
  many times the step could turn over in one residence time, and its label is
  the reported Ea. A step only converts a meaningful share of what reaches it
  when k*tau is about 1 or more.
* **Speed vs barrier.** k*tau for every step against its Ea, with the Eyring
  curve and the Ea at which k*tau = 1. Steps right of that line wash out
  before they react.

Run it with::

    python R02R_network_diagram.py            # save and show
    python R02R_network_diagram.py --no-show  # save only

Writes ``results/R02R_network_diagram.png``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import R02R_network as net
from PharmaPy.ThermoModule import ParseDatabase

# Node positions (x, y), laid out left to right along the network.
_LAYOUT = {
    "Z1": (0.0, 1.2), "F": (0.0, -1.2),
    "INT1": (1.6, 0.0),
    "INT1ONO2": (3.3, 2.0), "INT1OH": (3.3, 0.0), "INT1AcO": (3.3, -2.0),
    "Z2": (5.0, 0.0),
    "INT3": (5.0, -3.6), "INT4": (6.7, -3.6), "Z3": (8.4, -3.6),
    "INT5": (6.7, 2.0), "Z4": (8.4, 3.0), "D": (8.4, 1.0),
    "Z5": (6.7, -1.0), "Z6": (8.4, 0.0), "INT2OH": (8.4, -2.2),
    "E": (10.3, -1.0), "Z7": (10.1, -3.5), "INT2ONO2": (11.7, -2.6),
}
X_STRETCH = 1.3  # horizontal spacing, so short arrows stay readable
POSITIONS = {name: (x * X_STRETCH, y) for name, (x, y) in _LAYOUT.items()}
PRODUCTS = ("D", "E")

# k*tau bins: (lower bound, label, colour, line width, dashed). Sequential
# blue for "how fast"; the dead bin drops to muted gray so it recedes.
KTAU_BINS = [
    (1.0, "fast: k·τ ≥ 1 (mostly converts)", "#104281", 3.2, False),
    (1e-3, "slow: 10⁻³ ≤ k·τ < 1", "#3987e5", 2.2, False),
    (1e-6, "very slow: 10⁻⁶ ≤ k·τ < 10⁻³", "#9ec5f4", 1.6, False),
    (0.0, "negligible: k·τ < 10⁻⁶", "#a3a29d", 1.2, True),
]
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#3b3a37"
TEXT_MUTED = "#6b6a66"
GRID = "#e5e4e0"
NODE_MAJOR = "#cde2fb"
NODE_MINOR = "#f0efec"
ACCENT = "#d6457a"  # annotation callouts for the two bottlenecks


def ktau_style(ktau: float):
    for lower, label, color, width, dashed in KTAU_BINS:
        if ktau >= lower:
            return label, color, width, dashed
    return KTAU_BINS[-1][1:]


def steady_state_conc(reactor) -> dict[str, float]:
    """Final molar concentration [mol/L] of every species."""
    database = ParseDatabase(str(net.DEFAULT_DATABASE_PATH), to_arrays=False)
    mw = dict(zip(database["name_species"], database["mw"]))
    mass = np.asarray(reactor.result.mass_j_liquid0, dtype=float)[-1]
    vol = float(np.ravel(reactor.result.vessel_vol)[-1])  # m3
    return {name: max(m, 0.0) / mw[name] / vol  # kmol/m3 = mol/L
            for name, m in zip(reactor.name_species, mass)}


def fmt_conc(c: float) -> str:
    if c < 1e-14:
        return "≈ 0"
    if c >= 0.01:
        return f"{c:.2f} mol/L"
    exponent = int(np.floor(np.log10(c)))
    mantissa = c / 10 ** exponent
    return f"{mantissa:.1f}×10$^{{{exponent}}}$ mol/L"


def _clip_to_boxes(start, end, width, height, pad=0.08):
    """Segment between two node centres, trimmed to the box edges."""
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    half_w, half_h = width / 2 + pad, height / 2 + pad
    t = min(half_w / abs(dx) if dx else np.inf,
            half_h / abs(dy) if dy else np.inf)
    return (x0 + t * dx, y0 + t * dy), (x1 - t * dx, y1 - t * dy)


def draw_network(ax, conc: dict[str, float], ktau: np.ndarray):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    node_w, node_h = 1.25, 0.62

    for (reactant, product, ea), kt in zip(net.STEPS, ktau):
        _, color, width, dashed = ktau_style(kt)
        (x0, y0), (x1, y1) = _clip_to_boxes(POSITIONS[reactant],
                                            POSITIONS[product],
                                            node_w, node_h)
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=14, lw=width, color=color,
            linestyle=(0, (4, 3)) if dashed else "-",
            shrinkA=0, shrinkB=0, zorder=1,
        )
        ax.add_patch(arrow)
        # Ea label on the visible segment's midpoint, on a surface chip.
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(xm, ym, f"{ea:g}", fontsize=8, color=TEXT_SECONDARY,
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

    for name, (x, y) in POSITIONS.items():
        c = conc.get(name, 0.0)
        major = c >= 0.01
        box = FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=NODE_MAJOR if major else NODE_MINOR,
            ec=TEXT_PRIMARY if name in PRODUCTS else "#c9c8c3",
            lw=2.0 if name in PRODUCTS else 0.8, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.11, name, ha="center", va="center", fontsize=10,
                weight="bold", color=TEXT_PRIMARY, zorder=4)
        ax.text(x, y - 0.15, fmt_conc(c), ha="center", va="center",
                fontsize=7.5, color=TEXT_SECONDARY, zorder=4)

    # Bottleneck callouts.
    ax.annotate(
        "D bottleneck: selectivity at Z2\n"
        "Z2→Z5 (73.3) beats Z2→INT5 (118.9)\n"
        "by ~10⁷, then INT5→D converts 0.1%",
        xy=(7.6, 1.0), xytext=(3.4, 3.55), fontsize=8.5, color=TEXT_PRIMARY,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=ACCENT, lw=1.2))
    ax.annotate(
        "E bottleneck: speed at Z5\n"
        "Z5→E (140.9): k·τ ≈ 2×10⁻⁶,\n"
        "so Z5 washes out unreacted",
        xy=(11.3, -1.0), xytext=(13.6, 1.0), fontsize=8.5, color=TEXT_PRIMARY,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=ACCENT, lw=1.2))

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color, lw=width,
                      ls=(0, (4, 3)) if dashed else "-", label=label)
               for _, label, color, width, dashed in KTAU_BINS]
    handles.append(Line2D([], [], ls="none", label="Arrow label = Ea [kJ/mol]"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5,
              title="Step speed vs residence time", title_fontsize=9,
              bbox_to_anchor=(-0.01, -0.02))

    ax.set_xlim(-1.0, 17.0)
    ax.set_ylim(-4.6, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_ktau_vs_ea(ax, ktau: np.ndarray, temp: float, tau: float):
    ea = net.ACTIVATION_ENERGIES / 1e3
    k_raw = net.eyring_rate_constants(temp)
    capped = net.RATE_CONSTANTS < k_raw * (1 - 1e-12)

    grid = np.linspace(0, 260, 400)
    prefactor = net.BOLTZMANN * temp / net.PLANCK
    curve = prefactor * np.exp(-grid * 1e3 / (net.GAS_CONSTANT * temp)) * tau
    ea_star = net.GAS_CONSTANT * temp * np.log(prefactor * tau) / 1e3

    ax.axhspan(1, 1e20, color="#f0efec", zorder=0)
    ax.plot(grid, curve, color=TEXT_MUTED, lw=1.5, zorder=1,
            label="Eyring k·τ (uncapped)")
    ax.axhline(1.0, color=TEXT_PRIMARY, lw=1.0, zorder=1)
    ax.axvline(ea_star, color=TEXT_PRIMARY, lw=1.0, ls="--", zorder=1)
    ax.text(ea_star + 3, 1e14, f"k·τ = 1 at Ea ≈ {ea_star:.0f} kJ/mol",
            fontsize=9, color=TEXT_PRIMARY, va="center")
    ax.text(255, 3e1, "above k·τ = 1: converts within one residence time",
            fontsize=8.5, color=TEXT_SECONDARY, va="bottom", ha="right")
    ax.text(255, 3e-1, "below: washes out before reacting", fontsize=8.5,
            color=TEXT_SECONDARY, va="top", ha="right")

    colors = [ktau_style(kt)[1] for kt in ktau]
    ax.scatter(ea[~capped], np.maximum(ktau[~capped], 1e-25), s=64,
               c=[c for c, m in zip(colors, ~capped) if m],
               edgecolors=SURFACE, linewidths=2, zorder=3,
               label="step, k used")
    ax.scatter(ea[capped], ktau[capped], s=64, marker="s",
               c=[c for c, m in zip(colors, capped) if m],
               edgecolors=SURFACE, linewidths=2, zorder=3,
               label=f"step, capped at K_MAX (Eyring k·τ is higher)")

    # Text offsets in points; each label gets a thin leader to its marker.
    label_offsets = {"Z2→Z5": (0, 34), "INT5→D": (24, 26),
                     "Z2→INT5": (-30, -34), "Z5→E": (-10, -40),
                     "INT1ONO2→Z2": (34, 14)}
    for (reactant, product, _), x, y in zip(net.STEPS, ea, ktau):
        key = f"{reactant}→{product}"
        if key in label_offsets:
            ax.annotate(key, (x, y), xytext=label_offsets[key],
                        textcoords="offset points", fontsize=8,
                        color=TEXT_PRIMARY, ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=TEXT_MUTED,
                                        lw=0.8, shrinkA=2, shrinkB=5))

    ax.set_yscale("log")
    ax.set_ylim(1e-26, 1e17)
    ax.set_xlim(-5, 260)
    ax.set_xlabel("Reported Ea [kJ/mol]")
    ax.set_ylabel("k·τ  (turnovers per residence time)")
    ax.set_title(f"Speed vs barrier at {temp - 273.15:g} °C, τ = {tau / 60:.0f} min",
                 fontsize=11, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def make_figure(reactor):
    import matplotlib.pyplot as plt

    tau = net.REACTOR_MASS / net.total_feed_flow()
    temp = net.OPERATING_TEMPERATURE
    ktau = net.RATE_CONSTANTS * tau
    conc = steady_state_conc(reactor)

    fig = plt.figure(figsize=(16, 13), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.75, 1], hspace=0.12)
    ax_net = fig.add_subplot(gs[0])
    ax_k = fig.add_subplot(gs[1])

    draw_network(ax_net, conc, ktau)
    ax_net.set_title("R02R network at steady state: species concentration and "
                     "step speed", fontsize=12, loc="left")
    draw_ktau_vs_ea(ax_k, ktau, temp, tau)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figure without opening a window")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    database = Path(os.environ.get("PHARMAPY_DATABASE",
                                   net.DEFAULT_DATABASE_PATH))
    reactor = net.run_reactor(database)
    fig = make_figure(reactor)
    net.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = net.RESULTS_DIR / "R02R_network_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
