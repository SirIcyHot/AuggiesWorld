"""Flowsheet map: stream phases, stream contents and each unit op's class.

The process flowsheet is::

    CR01H -> CR02H -> F01H -> D01H -> R02R -> D01R -> F01R -> D02R

This script solves every unit through its own module, using the same
functions those modules' own ``__main__`` blocks use, and records what each
stream holds in each phase at the final (steady) time. Nothing about a stream
is typed in here, and each unit's class is read from the solved object, so the
figure follows the code as it changes.

Run it with the ``PharmaPy`` conda environment::

    python flowsheet_visual.py            # draw from the cached solve, or solve
    python flowsheet_visual.py --solve    # re-solve first (after a code change)
    python flowsheet_visual.py --no-show  # save the figure only

Files, in ``results/`` next to this script:

    flowsheet_streams.json   the solved streams and unit classes (the cache)
    flowsheet.png            the map: units, classes, stream phases, wiring
    flowsheet_streams.png    every stream by species and phase, as a table

Whether the three links that cross between modules are really connected is
checked from the solved flows (``link_gap``). The map's layout itself is drawn
for the current wiring, so move the stand-in boxes if the wiring changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CACHE_PATH = RESULTS_DIR / "flowsheet_streams.json"
FIGURE_PATH = RESULTS_DIR / "flowsheet.png"
TABLE_PATH = RESULTS_DIR / "flowsheet_streams.png"

# The folders holding the unit modules. "R02R" is the copy D01R and F01R
# import (A + F + G + I); R02R_onemethod and R02R.py are older variants.
UNIT_DIRS = ("R01H", "F01H", "D01H", "R02R", "D01R", "F01R", "D02R")

# Anything below this flow [kg/s] is solver roundoff, not a stream component.
FLOW_FLOOR = 1e-10


# ---------------------------------------------------------------- solving
def named_flows(names, values) -> dict[str, float]:
    """{species: kg/s} for the species actually present."""
    return {name: float(value) for name, value in zip(names, values)
            if abs(float(value)) > FLOW_FLOOR}


def stream(names, temp, liquid=None, solid=None) -> dict:
    """One stream record. ``liquid``: array by species; ``solid``: {name: kg/s}."""
    liquid = {} if liquid is None else named_flows(names, liquid)
    solid = {name: float(value) for name, value in (solid or {}).items()
             if abs(float(value)) > FLOW_FLOOR}
    return {"temp": float(temp), "liquid": liquid, "solid": solid}


def flow_stream(flow) -> dict:
    """A stream record from a ``continuous_filter.SlurryFlow``."""
    names = list(flow.name_species)
    solid = {}
    if flow.has_solids:
        solid = named_flows(names, flow.solid_mass_flow
                            * np.asarray(flow.solid_mass_frac))
    return stream(names, flow.temp, flow.liquid_mass_j, solid)


def describe_class(unit) -> dict:
    """The class of a solved unit, read from the object itself."""
    cls = type(unit)
    return {"name": cls.__name__, "module": cls.__module__,
            "bases": [base.__name__ for base in cls.__mro__[1:-1]]}


def describe_phases(unit) -> dict:
    """What the vessel holds: its liquid count and each solid's species."""
    phases = list(unit.Phases)
    solids = [phase for phase in phases if phase.phase_family == "solid"]
    names = list(phases[0].name_species)
    return {
        "liquids": sum(p.phase_family == "liquid" for p in phases),
        "solids": ["+".join(n for n, w in zip(names, solid.mass_frac) if w > 0)
                   for solid in solids],
    }


def final_temp(unit) -> float:
    return float(np.ravel(unit.result.global_temp)[-1])


def solve_front_end(names) -> tuple[dict, dict]:
    """CR01H -> CR02H."""
    import CR01H
    import CR02H
    from R01H import FEED_TEMPERATURE

    cr01 = CR01H.run_reactor()
    cr01_bal = CR01H.replay_balances(cr01)
    cr02 = CR02H.run_reactor(feed=CR02H.cr01_feed())
    cr02_bal = CR02H.replay_balances(cr02)

    cr01_out_temp = final_temp(cr01)
    streams = {
        "feed_a": stream(names, FEED_TEMPERATURE, cr01_bal.feeds[-1][0]),
        "feed_c": stream(names, FEED_TEMPERATURE, cr01_bal.feeds[-1][1]),
        "cr01_out": stream(names, cr01_out_temp, cr01_bal.outlet[-1],
                           {"F": cr01_bal.crystal["outlet"][-1]}),
        "cr02_liquid_feed": stream(names, cr01_out_temp, cr02_bal.feeds[-1][0]),
        "a_push": stream(names, FEED_TEMPERATURE, cr02_bal.feeds[-1][1]),
        "cr02_out": stream(names, final_temp(cr02), cr02_bal.outlet[-1],
                           {"F": cr02_bal.crystal["outlet"][-1]}),
    }
    units = {
        "CR01H": {"class": describe_class(cr01),
                  "phases": describe_phases(cr01),
                  "reactions": list(CR01H.REACTIONS),
                  "temp": cr01_out_temp,
                  "steady_gap": CR01H.steady_state_gap(cr01_bal)},
        "CR02H": {"class": describe_class(cr02),
                  "phases": describe_phases(cr02),
                  "reactions": list(CR02H.REACTIONS),
                  "temp": final_temp(cr02),
                  "steady_gap": CR02H.steady_state_gap(cr02_bal),
                  "feed_solids": bool(CR02H.FEED_SOLIDS)},
    }
    return streams, units


def solve_h_train(names) -> tuple[dict, dict]:
    """R01H -> F01H, and R01H -> F01H -> D01H, as those modules wire them."""
    import F01H
    import D01H

    r01, r01_out, _, filt = F01H.run_flowsheet()
    cake, cake_flow, tank = D01H.run_flowsheet()
    bal = D01H.final_balance(tank)
    temp = D01H.dissolver_temperature(cake)

    streams = {
        "r01h_out": flow_stream(r01_out),
        "f01h_feed": flow_stream(filt.feed),
        "f01h_cake": flow_stream(filt.cake_flow),
        "f01h_filtrate": flow_stream(filt.filtrate_flow),
        "h_feed": stream(names, D01H.H_TEMPERATURE, bal.feeds[-1]),
        "d01h_out": stream(names, temp, bal.outlet,
                           {"F": bal.crystal["outlet"]}),
    }
    units = {
        "R01H": {"class": describe_class(r01)},
        "F01H": {"class": describe_class(filt),
                 "solids_loss": F01H.SOLIDS_LOSS,
                 "cake_moisture": F01H.CAKE_MOISTURE,
                 "solid_fraction": F01H.crystalline_fraction(filt.feed)},
        "D01H": {"class": describe_class(tank),
                 "phases": describe_phases(tank),
                 "temp": float(temp),
                 "steady_gap": D01H.steady_state_gap(bal),
                 "dissolved": {"F": D01H.fraction_dissolved(bal)}},
    }
    return streams, units


def solve_r_train(names) -> tuple[dict, dict]:
    """R02R, R02R -> D01R, R02R -> F01R, R02R -> F01R -> D02R, as wired."""
    import R02R
    import D01R
    import F01R
    import D02R

    r02r = R02R.run_reactor()
    r02r_bal = R02R.replay_balances(r02r)

    d01r_feed, d01r = D01R.run_flowsheet()
    d01r_bal = D01R.final_balance(d01r)

    _, r02r_flow, _, filt = F01R.run_flowsheet()

    d02r_cake, d02r_cake_flow, d02r = D02R.run_flowsheet()
    d02r_bal = D02R.final_balance(d02r)
    d02r_temp = D02R.dissolver_temperature(d02r_cake)
    species = D01R.CRYSTAL_SPECIES
    feed_phases = d02r_bal.feed_phases

    streams = {
        "r02r_feed": stream(names, R02R.FEED_TEMPERATURE,
                            r02r_bal.feeds[-1][0]),
        "r02r_out": stream(names, final_temp(r02r), r02r_bal.outlet[-1]),
        "b_feed": stream(names, D01R.ANTISOLVENT_TEMPERATURE,
                         d01r_bal.feeds[1]),
        "d01r_out": stream(names, D01R.CRYSTALLIZER_TEMPERATURE,
                           d01r_bal.outlet,
                           {s: d01r_bal.crystal[s]["outlet"] for s in species}),
        "f01r_feed": flow_stream(filt.feed),
        "f01r_cake": flow_stream(filt.cake_flow),
        "f01r_filtrate": flow_stream(filt.filtrate_flow),
        "l_feed": stream(names, D02R.L_TEMPERATURE,
                         d02r_bal.feeds[len(feed_phases) - 1]),
        "d02r_out": stream(names, d02r_temp, d02r_bal.outlet,
                           {s: d02r_bal.crystal[s]["outlet"] for s in species}),
    }
    profiles = D01R.history(d01r)
    units = {
        "R02R": {"class": describe_class(r02r),
                 "phases": describe_phases(r02r),
                 "reactions": list(R02R.REACTIONS),
                 "temp": final_temp(r02r),
                 "steady_gap": R02R.steady_state_gap(r02r_bal)},
        "D01R": {"class": describe_class(d01r),
                 "phases": describe_phases(d01r),
                 "temp": float(D01R.CRYSTALLIZER_TEMPERATURE),
                 "steady_gap": D01R.steady_state_gap(d01r_bal),
                 "crystallized": {
                     s: float(profiles.fraction_crystallized[s][-1])
                     for s in species}},
        "F01R": {"class": describe_class(filt),
                 "solids_loss": F01R.SOLIDS_LOSS,
                 "cake_moisture": F01R.CAKE_MOISTURE,
                 "solid_fraction": F01R.crystalline_fraction(filt.feed)},
        "D02R": {"class": describe_class(d02r),
                 "phases": describe_phases(d02r),
                 "temp": float(d02r_temp),
                 "steady_gap": D02R.steady_state_gap(d02r_bal),
                 "dissolved": D02R.fraction_dissolved(d02r_bal,
                                                      d02r_cake_flow)},
    }
    return streams, units


def solve_flowsheet() -> dict:
    """Solve every unit and return the streams and classes as plain data."""
    sys.path[:0] = [str(HERE / folder) for folder in UNIT_DIRS]
    sys.path.insert(0, str(HERE.parent / "CustomUnitOps"))

    import R01H

    names = list(R01H.database_species(R01H.DEFAULT_DATABASE_PATH))
    streams, units = {}, {}
    for label, solver in (("CR01H -> CR02H", solve_front_end),
                          ("R01H -> F01H -> D01H", solve_h_train),
                          ("R02R, D01R, F01R, D02R", solve_r_train)):
        start = time.perf_counter()
        print(f"Solving {label}...", flush=True)
        new_streams, new_units = solver(names)
        streams.update(new_streams)
        units.update(new_units)
        print(f"  done in {time.perf_counter() - start:.0f} s", flush=True)
    return {"streams": streams, "units": units}


def load_or_solve(force: bool) -> dict:
    if CACHE_PATH.exists() and not force:
        print(f"Using cached solve: {CACHE_PATH} (--solve to refresh)")
        return json.loads(CACHE_PATH.read_text())
    data = solve_flowsheet()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=1))
    return data


# ---------------------------------------------------------------- drawing
# Phase colors are the blue and orange the unit scripts already plot with.
# Phase is never color alone: slurry is a double line and every card names it.
LIQUID_COLOR = "#2a78d6"
SOLID_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID_COLOR = "#e5e4e0"
BOX_FILL = "#f7f6f3"
SURFACE = "#ffffff"
UNWIRED_COLOR = "#7a7974"

BOX_W, BOX_H = 3.2, 2.7
CARD_W, CARD_H = 3.0, 1.22
ROW_1, ROW_2 = 9.3, 3.1  # y of each row's centre
COLUMNS = (3.5, 10.2, 16.6, 22.9)  # x of each column's centre

ROLES = {
    "CR01H": "reactive crystallizer",
    "CR02H": "reactive crystallizer, A push",
    "F01H": "filter",
    "D01H": "dissolver (H)",
    "R02R": "jacketed CSTR",
    "D01R": "antisolvent crystallizer (B)",
    "F01R": "filter",
    "D02R": "dissolver (L)",
}

# Stream ids and names, in flowsheet order. These also set the table's rows.
STREAMS = (
    ("feed_a", "S1", "Feed A"),
    ("feed_c", "S2", "Feed C"),
    ("cr01_out", "S3", "CR01H outlet"),
    ("cr02_liquid_feed", "S4", "CR01H liquid to CR02H"),
    ("a_push", "S5", "A push"),
    ("cr02_out", "S6", "CR02H outlet"),
    ("r01h_out", "S7", "R01H outlet"),
    ("f01h_feed", "S8", "F01H feed"),
    ("f01h_filtrate", "S9", "F01H filtrate"),
    ("f01h_cake", "S10", "F01H cake"),
    ("h_feed", "S11", "Fresh H"),
    ("d01h_out", "S12", "D01H outlet"),
    ("r02r_feed", "S13", "R02R feed"),
    ("r02r_out", "S14", "R02R outlet"),
    ("b_feed", "S15", "Antisolvent B"),
    ("d01r_out", "S16", "D01R outlet"),
    ("f01r_feed", "S17", "F01R feed"),
    ("f01r_filtrate", "S18", "F01R filtrate"),
    ("f01r_cake", "S19", "F01R cake"),
    ("l_feed", "S20", "Fresh L"),
    ("d02r_out", "S21", "D02R outlet"),
)
STREAM_INFO = {key: (sid, name) for key, sid, name in STREAMS}


def phase_totals(record: dict) -> tuple[float, float]:
    """(liquid, solid) mass flow of a stream [kg/s]."""
    return sum(record["liquid"].values()), sum(record["solid"].values())


def phase_label(record: dict) -> str:
    liquid, solid = phase_totals(record)
    if solid <= 0:
        return "liquid"
    if liquid <= 0:
        return "solid"
    return "wet cake" if solid > liquid else "slurry"


def species_totals(record: dict) -> dict[str, float]:
    """{species: kg/s} over both phases."""
    totals = dict(record["liquid"])
    for name, value in record["solid"].items():
        totals[name] = totals.get(name, 0.0) + value
    return totals


def link_gap(streams: dict, upstream: str, downstream: str) -> float:
    """Largest species mismatch between two streams, relative to the larger
    stream's total flow. Near zero means the first really feeds the second."""
    a, b = species_totals(streams[upstream]), species_totals(streams[downstream])
    scale = max(sum(a.values()), sum(b.values()))
    return max(abs(a.get(n, 0.0) - b.get(n, 0.0)) for n in set(a) | set(b)) / scale


def percent_list(values: dict) -> str:
    return "  ".join(f"{name} {min(value, 1.0):.0%}"
                     for name, value in values.items())


def pipe(ax, points, phase, dashed=False, head=True):
    """A stream line through ``points``: blue liquid, double line for slurry.

    A slurry line must be a single straight segment.
    """
    xs, ys = map(np.array, zip(*points))
    style = (0, (5, 3)) if dashed else "-"
    if phase in ("slurry", "wet cake"):
        (x0, x1), (y0, y1) = xs, ys
        length = np.hypot(x1 - x0, y1 - y0)
        nx, ny = -(y1 - y0) / length * 0.055, (x1 - x0) / length * 0.055
        ax.plot(xs + nx, ys + ny, color=LIQUID_COLOR, lw=2.2, ls=style,
                solid_capstyle="butt", zorder=2)
        ax.plot(xs - nx, ys - ny, color=SOLID_COLOR, lw=2.2, ls=style,
                solid_capstyle="butt", zorder=2)
        head_color = TEXT_PRIMARY
    else:
        color = {"liquid": LIQUID_COLOR, "solid": SOLID_COLOR,
                 "unwired": UNWIRED_COLOR}[phase]
        ax.plot(xs, ys, color=color, lw=2.2, ls=style, zorder=2,
                solid_joinstyle="round")
        head_color = color
    if head:
        ax.annotate("", xy=points[-1], xytext=points[-2], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=head_color, lw=0,
                                    mutation_scale=17, shrinkA=0, shrinkB=0))


def card(ax, streams, key, x, y):
    """A stream's card, lower-left corner at (x, y): name, phase split, content."""
    from matplotlib.patches import Rectangle

    record = streams[key]
    sid, name = STREAM_INFO[key]
    liquid, solid = phase_totals(record)
    total = liquid + solid

    ax.add_patch(Rectangle((x, y), CARD_W, CARD_H, fc=SURFACE, ec=GRID_COLOR,
                           lw=1, zorder=4))
    left, top = x + 0.08, y + CARD_H
    ax.text(left, top - 0.17, f"{sid}  {name}", fontsize=9.5, weight="bold",
            color=TEXT_PRIMARY, va="center", zorder=5)
    ax.text(left, top - 0.40,
            f"{phase_label(record)} · {total * 1e3:.3g} g/s · "
            f"{record['temp']:.0f} K",
            fontsize=8.5, color=TEXT_MUTED, va="center", zorder=5)

    # Phase split by mass, with a small surface gap between the two fills.
    bar_w = CARD_W - 0.16
    gap = 0.03 if liquid > 0 and solid > 0 else 0.0
    liquid_w = (bar_w - gap) * liquid / total
    solid_w = (bar_w - gap) * solid / total
    ax.add_patch(Rectangle((left, top - 0.63), liquid_w, 0.09,
                           fc=LIQUID_COLOR, lw=0, zorder=5))
    ax.add_patch(Rectangle((left + liquid_w + gap, top - 0.63), solid_w, 0.09,
                           fc=SOLID_COLOR, lw=0, zorder=5))

    def contents(flows, limit=5):
        ranked = sorted(flows.items(), key=lambda item: -item[1])
        shown = [f"{n} {v * 1e3:.2g}" for n, v in ranked[:limit]
                 if v > 0.005 * sum(flows.values())]
        return "  ".join(shown) if shown else "-"

    ax.text(left, top - 0.81, "liq  " + contents(record["liquid"]),
            fontsize=8.5, color=TEXT_PRIMARY, va="center", zorder=5)
    ax.text(left, top - 1.03, "sol  " + contents(record["solid"]),
            fontsize=8.5, color=TEXT_PRIMARY, va="center", zorder=5)


def unit_box(ax, data, name, x, y):
    """A unit op centred on (x, y), its class read from the solved object."""
    from matplotlib.patches import FancyBboxPatch

    unit = data["units"][name]
    cls = unit["class"]
    ax.add_patch(FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0,rounding_size=0.08",
        fc=BOX_FILL, ec=TEXT_MUTED, lw=1.3, zorder=3))
    left = x - BOX_W / 2 + 0.12
    top = y + BOX_H / 2

    def line(offset, text, **style):
        style = {"fontsize": 9, "color": TEXT_PRIMARY, "va": "center",
                 "zorder": 5, **style}
        ax.text(left, top - offset, text, **style)

    line(0.24, name, fontsize=14, weight="bold")
    line(0.53, ROLES[name], fontsize=8.5, color=TEXT_MUTED)
    line(0.86, "class", fontsize=7.5, color=TEXT_MUTED)
    line(1.06, cls["name"], fontsize=8.6, weight="bold", family="monospace")
    line(1.27, cls["module"], fontsize=7.5, color=TEXT_MUTED,
         family="monospace")

    facts = []
    if "phases" in unit:
        solids = unit["phases"]["solids"]
        facts.append("holds: liquid" + (" + solid " + ", ".join(solids)
                                        if solids else " only"))
        facts.append(f"T {unit['temp']:.0f} K")
    else:
        facts.append(f"solids loss {unit['solids_loss']:.0%}, "
                     f"cake LOD {unit['cake_moisture']:.0%}")
        facts.append("steady-state split, no holdup")
    if "reactions" in unit:
        reactions = unit["reactions"]
        facts[-1] += ";  " + ("F + A + I + G → D/E/J + H" if len(reactions) > 1
                              else reactions[0].replace("-->", "→"))
    if "crystallized" in unit:
        facts.append("crystallized: " + percent_list(unit["crystallized"]))
    if "dissolved" in unit:
        facts.append("dissolved: " + percent_list(unit["dissolved"]))
    for i, fact in enumerate(facts):
        line(1.56 + 0.26 * i, fact, fontsize=8.6,
             weight="bold" if fact.startswith("dissolved") else "normal")


def note(ax, x, y, text, ha="center", warn=False, **style):
    style = {"fontsize": 8.8, "color": TEXT_PRIMARY, "va": "center",
             "zorder": 6, **style}
    if warn:
        style["bbox"] = dict(fc="#fff4d6", ec="#c98a00", lw=0.8,
                             boxstyle="round,pad=0.25")
    ax.text(x, y, text, ha=ha, **style)


def draw_map(data: dict):
    """The flowsheet: units with their classes, streams with their phases."""
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    streams = data["streams"]
    fig, ax = plt.subplots(figsize=(22, 13.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.75, 14.25)
    ax.axis("off")

    c1, c2, c3, c4 = COLUMNS
    half_w, half_h = BOX_W / 2, BOX_H / 2
    top_1, bottom_1 = ROW_1 + half_h, ROW_1 - half_h
    top_2, bottom_2 = ROW_2 + half_h, ROW_2 - half_h

    def total(key):
        return sum(phase_totals(streams[key])) * 1e3

    def phase_of(key):
        return phase_label(streams[key])

    wired = {
        "cr02": link_gap(streams, "cr02_out", "f01h_feed") < 0.01,
        "d01h": link_gap(streams, "d01h_out", "r02r_feed") < 0.01,
        "d01r": link_gap(streams, "d01r_out", "f01r_feed") < 0.01,
    }
    num_unwired = sum(not value for value in wired.values())

    for name, x, y in (("CR01H", c1, ROW_1), ("CR02H", c2, ROW_1),
                       ("F01H", c3, ROW_1), ("D01H", c4, ROW_1),
                       ("R02R", c1, ROW_2), ("D01R", c2, ROW_2),
                       ("F01R", c3, ROW_2), ("D02R", c4, ROW_2)):
        unit_box(ax, data, name, x, y)

    # ---- CR01H: two feeds in from above.
    card(ax, streams, "feed_a", 0.25, 12.55)
    card(ax, streams, "feed_c", 3.4, 12.55)
    pipe(ax, [(2.6, 12.55), (2.6, top_1)], "liquid")
    pipe(ax, [(4.4, 12.55), (4.4, top_1)], "liquid")

    # ---- CR01H -> CR02H: the slurry leaves, only its liquid goes on.
    split_x = c1 + half_w + 1.05
    pipe(ax, [(c1 + half_w, ROW_1), (split_x, ROW_1)], phase_of("cr01_out"),
         head=False)
    pipe(ax, [(split_x, ROW_1), (c2 - half_w, ROW_1)], "liquid")
    pipe(ax, [(split_x, ROW_1), (split_x, ROW_1 - 0.85)], "solid", head=False)
    ax.plot([split_x], [ROW_1 - 0.85], marker="x", ms=11, mew=2.4,
            color=SOLID_COLOR, zorder=3)
    card(ax, streams, "cr01_out", 5.3, 10.0)
    note(ax, split_x + 1.0, ROW_1 + 0.2, "S4", weight="bold")
    note(ax, split_x + 0.7, ROW_1 - 1.4,
         f"CR01H crystals dropped: "
         f"{streams['cr01_out']['solid']['F'] * 1e3:.3g} g/s F\n"
         "FEED_SOLIDS = False", warn=True)

    # ---- CR02H: A push in from above.
    card(ax, streams, "a_push", c2 - CARD_W / 2, 12.55)
    pipe(ax, [(c2, 12.55), (c2, top_1)], "liquid")

    # ---- CR02H -> F01H: intended link.
    out_end = c2 + half_w + 0.6
    pipe(ax, [(c2 + half_w, ROW_1), (out_end, ROW_1)], phase_of("cr02_out"),
         head=False)
    pipe(ax, [(out_end, ROW_1), (c3 - half_w, ROW_1)],
         phase_of("cr02_out") if wired["cr02"] else "unwired",
         dashed=not wired["cr02"])
    card(ax, streams, "cr02_out", 11.9, 9.65)
    if not wired["cr02"]:
        note(ax, 13.4, ROW_1 - 0.6,
             "NOT WIRED: F01H reads R01H\n"
             f"F01H feed {total('f01h_feed'):.3g} g/s; "
             f"CR02H out {total('cr02_out'):.3g} g/s", warn=True)

    # ---- R01H stands in as F01H's source.
    ax.add_patch(FancyBboxPatch(
        (c3 - 1.2, 12.75), 2.4, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
        fc=SURFACE, ec=UNWIRED_COLOR, lw=1.2, ls=(0, (4, 2)), zorder=3))
    note(ax, c3, 13.38, "R01H  (stand-in source)", fontsize=9, weight="bold")
    note(ax, c3, 13.0, data["units"]["R01H"]["class"]["name"], fontsize=8.6,
         family="monospace")
    card(ax, streams, "r01h_out", 12.2, 12.55)
    pill_y = 11.75
    pipe(ax, [(c3, 12.75), (c3, pill_y + 0.17)], "liquid")
    note(ax, c3, pill_y, "crystallize_f(): arithmetic split", warn=True)
    pipe(ax, [(c3, pill_y - 0.17), (c3, top_1)], phase_of("f01h_feed"))
    card(ax, streams, "f01h_feed", c3 + 1.25, 10.75)

    # ---- F01H: cake on, filtrate off.
    pipe(ax, [(c3, bottom_1), (c3, 7.62)], phase_of("f01h_filtrate"))
    card(ax, streams, "f01h_filtrate", c3 - CARD_W / 2 + 0.2, 6.4)
    pipe(ax, [(c3 + half_w, ROW_1), (c4 - half_w, ROW_1)],
         phase_of("f01h_cake"))
    card(ax, streams, "f01h_cake", 18.25, 8.0)

    # ---- D01H: H in from above, slurry out.
    card(ax, streams, "h_feed", c4 - CARD_W / 2 + 0.3, 12.55)
    pipe(ax, [(c4, 12.55), (c4, top_1)], "liquid")
    route_y = 6.15
    pipe(ax, [(c4, bottom_1), (c4, 7.17)], phase_of("d01h_out"))
    card(ax, streams, "d01h_out", c4 - CARD_W / 2 + 0.3, 5.95)

    # ---- D01H -> R02R: intended link, the long way round.
    if not wired["d01h"]:
        pipe(ax, [(c4 - CARD_W / 2 + 0.3, route_y), (4.4, route_y),
                  (4.4, top_2)], "unwired", dashed=True)
        fed_f = species_totals(streams["r02r_feed"])["F"] * 1e3
        out_f = species_totals(streams["d01h_out"])["F"] * 1e3
        note(ax, 11.3, route_y,
             "NOT WIRED: R02R has its own fixed feed\n"
             f"R02R feed: F {fed_f:.3g} g/s in {total('r02r_feed'):.3g} g/s;  "
             f"D01H out: F {out_f:.2g} g/s in {total('d01h_out'):.3g} g/s",
             warn=True)

    # ---- R02R: fresh feed, liquid out.
    card(ax, streams, "r02r_feed", 0.25, 5.2)
    pipe(ax, [(2.4, 5.2), (2.4, top_2)], "liquid")
    pipe(ax, [(c1 + half_w, ROW_2), (c2 - half_w, ROW_2)], "liquid")
    card(ax, streams, "r02r_out", 5.2, 1.6)

    # ---- R02R -> F01R: the outlet actually used, passing D01R by.
    bypass_y = 5.15
    bypass_x = c1 + half_w + 0.45
    pipe(ax, [(bypass_x, ROW_2), (bypass_x, bypass_y), (c3 - 1.4, bypass_y)],
         "liquid", head=False)
    note(ax, c3, bypass_y, "crystallize(): 100 % D, E, J", warn=True)
    pipe(ax, [(c3, bypass_y - 0.17), (c3, top_2)], phase_of("f01r_feed"))
    note(ax, c2, bypass_y + 0.2, "S14 also goes straight to F01R, bypassing D01R",
         color=TEXT_MUTED)
    card(ax, streams, "f01r_feed", c3 + 1.25, 4.55)

    # ---- D01R: B in from below, slurry out.
    card(ax, streams, "b_feed", c2 - CARD_W / 2, -0.35)
    pipe(ax, [(c2, -0.35 + CARD_H), (c2, bottom_2)], "liquid")

    # ---- D01R -> F01R: intended link.
    out_end = c2 + half_w + 0.6
    pipe(ax, [(c2 + half_w, ROW_2), (out_end, ROW_2)], phase_of("d01r_out"),
         head=False)
    pipe(ax, [(out_end, ROW_2), (c3 - half_w, ROW_2)],
         phase_of("d01r_out") if wired["d01r"] else "unwired",
         dashed=not wired["d01r"])
    card(ax, streams, "d01r_out", 11.9, 3.3)
    if not wired["d01r"]:
        note(ax, 13.4, ROW_2 - 0.75,
             "NOT WIRED: F01R reads R02R\n"
             f"F01R feed: {streams['f01r_feed']['temp']:.0f} K, no B\n"
             f"D01R out: {streams['d01r_out']['temp']:.0f} K, "
             f"B {streams['d01r_out']['liquid']['B'] * 1e3:.3g} g/s", warn=True)

    # ---- F01R: cake on, filtrate off.
    pipe(ax, [(c3, bottom_2), (c3, -0.35 + CARD_H)], phase_of("f01r_filtrate"))
    card(ax, streams, "f01r_filtrate", c3 - CARD_W / 2, -0.35)
    pipe(ax, [(c3 + half_w, ROW_2), (c4 - half_w, ROW_2)],
         phase_of("f01r_cake"))
    card(ax, streams, "f01r_cake", 18.25, 1.8)

    # ---- D02R: L in from above, product out.
    card(ax, streams, "l_feed", c4 - CARD_W / 2 + 0.3, 4.65)
    pipe(ax, [(c4, 4.65), (c4, top_2)], "liquid")
    pipe(ax, [(c4, bottom_2), (c4, -0.35 + CARD_H)], phase_of("d02r_out"))
    card(ax, streams, "d02r_out", c4 - CARD_W / 2 + 0.3, -0.35)

    # ---- title and legend
    fig.suptitle(
        f"Flowsheet as coded: {num_unwired} of 7 unit links are not wired, "
        f"and D02R dissolves {percent_list(data['units']['D02R']['dissolved'])}",
        fontsize=17, weight="bold", color=TEXT_PRIMARY, x=0.012, ha="left",
        y=0.99)
    ax.text(0.25, 14.05,
            "Each unit solved through its own module, steady-state flows. "
            "Card contents in g/s. Class names read from the solved objects.",
            fontsize=10.5, color=TEXT_MUTED, va="center")
    handles = [
        Line2D([], [], color=LIQUID_COLOR, lw=2.6),
        Line2D([], [], color=SOLID_COLOR, lw=2.6),
        (Line2D([], [], color=LIQUID_COLOR, lw=2.2),
         Line2D([], [], color=SOLID_COLOR, lw=2.2)),
        Line2D([], [], color=UNWIRED_COLOR, lw=2.2, ls=(0, (5, 3))),
    ]
    ax.legend(handles, ["Liquid", "Solid only (dropped crystals)",
                        "Slurry or wet cake (two lines)",
                        "Intended link, not wired"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              loc="lower left", bbox_to_anchor=(0.0, -0.005), ncol=4,
              frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.96, bottom=0.005)
    return fig


def draw_table(data: dict):
    """Every stream by species and phase, shaded by flow on a log scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    streams = data["streams"]
    liquids = sorted({n for s in streams.values() for n in s["liquid"]})
    solids = sorted({n for s in streams.values() for n in s["solid"]})
    columns = (["Stream", "Phase", "T [K]", "Total"]
               + [f"{n} (l)" for n in liquids] + [f"{n} (s)" for n in solids])

    def shade(color, value):
        """Tint a color by flow: 0.001 g/s faint, 10 g/s strongest."""
        level = np.clip((np.log10(value * 1e3) + 3.0) / 4.0, 0.0, 1.0)
        alpha = 0.08 + 0.55 * level
        return tuple(alpha * c + (1 - alpha) for c in to_rgb(color))

    def cell(value):
        return f"{value * 1e3:.3g}" if value > 5e-7 else ""

    rows, colors = [], []
    for key, sid, name in STREAMS:
        record = streams[key]
        liquid = [record["liquid"].get(n, 0.0) for n in liquids]
        solid = [record["solid"].get(n, 0.0) for n in solids]
        rows.append([f"{sid}  {name}", phase_label(record),
                     f"{record['temp']:.0f}",
                     f"{sum(phase_totals(record)) * 1e3:.3g}"]
                    + [cell(v) for v in liquid + solid])
        colors.append(
            [SURFACE] * 4
            + [shade(LIQUID_COLOR, v) if v > 5e-7 else SURFACE for v in liquid]
            + [shade(SOLID_COLOR, v) if v > 5e-7 else SURFACE for v in solid])

    fig, ax = plt.subplots(figsize=(17, 0.34 * len(rows) + 0.7))
    ax.axis("off")
    first = [0.17, 0.065, 0.045, 0.05]
    other = (1.0 - sum(first)) / (len(columns) - len(first))
    table = ax.table(cellText=rows, cellColours=colors, colLabels=columns,
                     colWidths=first + [other] * (len(columns) - len(first)),
                     cellLoc="right", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, col), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor(GRID_COLOR)
        if col < 2:
            cell_obj.set_text_props(ha="left")
        if row == 0:
            cell_obj.set_text_props(weight="bold")
            cell_obj.set_facecolor("#f0efec")
    ax.set_title("Stream table [g/s]: blue is dissolved (l), orange is "
                 "crystals (s); darker means more flow", fontsize=13,
                 weight="bold", loc="left", pad=10)
    fig.tight_layout()
    return fig


def print_summary(data: dict):
    """Each unit's class, and the mismatch across each intended link."""
    print(f"{'unit':<7}{'class':<33}{'module':<35}steady-state gap")
    for name, unit in data["units"].items():
        gap = unit.get("steady_gap")
        print(f"{name:<7}{unit['class']['name']:<33}"
              f"{unit['class']['module']:<35}"
              + (f"{gap:.2%}" if gap is not None else "-"))
    print("\nLink mismatch, relative to the larger stream (near 0 = wired):")
    for upstream, downstream in (("cr02_out", "f01h_feed"),
                                 ("d01h_out", "r02r_feed"),
                                 ("d01r_out", "f01r_feed")):
        gap = link_gap(data["streams"], upstream, downstream)
        print(f"  {upstream:<10} -> {downstream:<10} {gap:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solve", action="store_true",
                        help="re-solve even if a cached solve exists")
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    flowsheet = load_or_solve(args.solve)
    print_summary(flowsheet)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    draw_map(flowsheet).savefig(FIGURE_PATH, dpi=130)
    draw_table(flowsheet).savefig(TABLE_PATH, dpi=130, bbox_inches="tight")
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
