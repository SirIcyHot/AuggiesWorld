"""Flowsheet map: stream phases, stream contents and each unit op's class.

The process, as ``Flowsheet.py`` couples it in one ConcurrentVessel::

    CR01H -> CR02H -> F01H -> W02H -> D01H -> R02R -> D01R -> F01R -> W02R
          -> D03R

Each wash block is two unit ops, drawn as one block: W02H is the D02H
dissolver (H) and the F02H filter; W02R is the D02R dissolver (B) and the
F02R filter.

This script solves ``Flowsheet.build_flowsheet`` and draws what it finds.
Stream contents come from ``Flowsheet.stream_table`` at the final time. Each
class is read from the solved object, and each link's coupling from the link
itself, so the figure follows the code as it changes. Only the layout, the
role wording and the stream names the layout expects are typed in here.

Run it from the activated environment, or LAPACK fails to load::

    conda activate PharmaPy
    python flowsheet_visual.py            # draw from the cached solve, or solve
    python flowsheet_visual.py --solve    # re-solve first (after a code change)
    python flowsheet_visual.py --no-show  # save the figures only

Files, in ``results/`` next to this script:

    flowsheet_map_cache.json    the solved streams and classes
    flowsheet_map.png           units, classes and stream phases
    flowsheet_stream_table.png  every stream by species and phase
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CACHE_PATH = RESULTS_DIR / "flowsheet_map_cache.json"
MAP_PATH = RESULTS_DIR / "flowsheet_map.png"
TABLE_PATH = RESULTS_DIR / "flowsheet_stream_table.png"

# Anything below this flow [kg/s] is solver roundoff, not a stream component.
FLOW_FLOOR = 1e-10


# ---------------------------------------------------------------- solving
def describe_class(unit) -> dict:
    """A vessel's PharmaPy class, plus any project subclass on top of it.

    ``name`` is the first PharmaPy class in the MRO, i.e. what the unit is at
    heart (R02R's BalanceRecordingReactor is reported as ContinuousReactor).
    ``subclass`` is a project class between the two, if any, and ``mixins``
    the Flowsheet mixins that ``with_true_slurry_feed`` adds.
    """
    mro = type(unit).__mro__
    base = next((c for c in mro if c.__module__.startswith("PharmaPy")),
                type(unit))
    own = [c for c in mro[:mro.index(base)] if c.__module__ != "Flowsheet"]
    mixins = [c.__name__ for c in mro
              if c.__module__ == "Flowsheet" and c.__name__ != base.__name__
              and c not in own]
    return {"name": base.__name__, "module": base.__module__,
            "subclass": ({"name": own[0].__name__, "module": own[0].__module__}
                         if own else None),
            "mixins": [m for m in mixins if m != type(unit).__name__]}


def describe_phases(unit) -> list[str]:
    """Each solid phase's species."""
    phases = list(unit.Phases)
    names = list(phases[0].name_species)
    return ["+".join(n for n, w in zip(names, phase.mass_frac) if w > 0)
            for phase in phases if phase.phase_family == "solid"]


def describe_link(link) -> dict:
    coupling = getattr(link, "coupling", None)
    separator = getattr(coupling, "separator", None)
    out = {"name": getattr(link, "name", None), "source": link.source,
           "sink": link.sink,
           "coupling": type(coupling).__name__,
           "coupling_bases": [c.__name__ for c in type(coupling).__mro__[1:-1]]}
    if separator is not None:
        out["separator"] = {
            "class": type(separator).__name__,
            "module": type(separator).__module__,
            "solids_loss": float(separator.solids_loss),
            "cake_moisture": float(separator.cake_moisture),
        }
    return out


def solve_flowsheet() -> dict:
    """Solve the coupled flowsheet and return what the figures need."""
    import Flowsheet as fs

    cv = fs.run_flowsheet()
    names = fs.species_names()

    streams = {}
    for label, entry in fs.stream_table(cv).items():
        streams[label] = {
            "temp": float(entry.temp),
            "liquid": {n: float(v) for n, v in zip(names, entry.liquid)
                       if abs(float(v)) > FLOW_FLOOR},
            "solid": {n: float(v) for n, v in entry.solids.items()
                      if abs(float(v)) > FLOW_FLOOR},
        }

    # Dissolution rate constants k_d [um/s] of the units that dissolve.
    dissolution = {"D01R": fs.D01R.DISSOLUTION, "D02R": fs.D02R.DISSOLUTION,
                   "D03R": fs.D03R.DISSOLUTION}

    units = {}
    for name, unit in cv.members.items():
        kinetics = getattr(unit, "RxnKinetics", None)
        units[name] = {
            "class": describe_class(unit),
            "solids": describe_phases(unit),
            "reactions": list(getattr(kinetics, "rxn_list", None) or []),
            "settling": float(fs.settling(unit, fs.UNIT_CRYSTALS[name])),
            "k_d": sorted({float(v[0]) for v in dissolution.get(name, {})
                           .values()}),
        }

    return {
        "streams": streams,
        "units": units,
        "links": [describe_link(link) for link in cv.links],
        "balances": {unit: [list(ins), list(outs)]
                     for unit, (ins, outs) in fs.UNIT_BALANCES.items()},
        "system": {"class": type(cv).__name__,
                   "module": type(cv).__module__,
                   "states": int(cv.solver_state_collection.dim),
                   "integrator": type(cv.integrator).__name__,
                   "time_h": float(cv.result.time[-1]) / 3600.0},
    }


def load_or_solve(force: bool) -> dict:
    if CACHE_PATH.exists() and not force:
        data = json.loads(CACHE_PATH.read_text())
        if "W02H cake" in data["streams"]:
            print(f"Using cached solve: {CACHE_PATH} (--solve to refresh)")
            return data
        print("The cached solve predates the wash blocks; solving again.")
    data = solve_flowsheet()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=1))
    return data


# ---------------------------------------------------------------- reporting
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


def dissolved_share(record: dict) -> dict[str, float]:
    """For each crystallizing species, the share of it that is dissolved."""
    out = {}
    for name, solid in record["solid"].items():
        liquid = record["liquid"].get(name, 0.0)
        out[name] = liquid / (liquid + solid)
    for name in ("D", "E", "J"):
        if name not in out and name in record["liquid"]:
            out[name] = 1.0
    return dict(sorted(out.items()))


def percent_list(values: dict) -> str:
    return "  ".join(f"{name} {value:.0%}" for name, value in values.items())


# Wash block -> (cake in, wash cake out, filtrate out).
WASH_STREAMS = {"W02H": ("F01H cake", "W02H cake", "W02H filtrate"),
                "W02R": ("F01R cake", "W02R cake", "W02R filtrate")}


def wash_dissolved(streams: dict, block: str) -> float:
    """Share of the crystals fed to a wash block that leave it dissolved."""
    fed, cake, filtrate = (streams[s] for s in WASH_STREAMS[block])
    crystals_in = sum(fed["solid"].values())
    crystals_out = sum(cake["solid"].values()) + sum(filtrate["solid"].values())
    return 1.0 - crystals_out / crystals_in if crystals_in > 0 else 0.0


# What the process is meant to do, checked against the solve.
CR01H_CRYSTAL_LIMIT = 1e-3  # share of CR01H's F that may leave as crystals
D03R_DISSOLVED_TARGET = 0.95  # "almost completely" dissolved
WASH_DISSOLVED_LIMIT = 0.10  # a wash should dissolve very little
SETTLED_LIMIT = 0.01  # change over the last 10 % of the run

# Where each vessel is drawn: the washes' dissolvers sit inside their block.
DRAWN_IN = {"D02H": "W02H", "D02R": "W02R"}

ASSUMPTIONS = [
    "A1  The solid F that D01H passes on as slurry dissolves quickly and "
    "completely in R02R's liquid, then reacts. R02R has no solid phase, so "
    "the D01H -> R02R link turns crystal mass into dissolved F, conserving "
    "mass.",
    "A2  F01H, F02H, F01R and F02R are steady-state splits evaluated inside "
    "the links (ContinuousFilter settings, no holdup), not vessels. F02H "
    "and F02R use the settings of F01H and F01R.",
    "A3  The solubility S0 of D02H and D01H is fixed at CR02H's design "
    "mother liquor (Flowsheet.CAKE_LIQUID_DESIGN).",
    "A4  R02R's outlet is all liquid. D, E and J would start to come out of "
    "solution in R02R, but the model assumes no solids form there; all "
    "crystallization happens in D01R.",
    "A5  W02H and W02R are each two unit ops, a dissolver vessel (D02H, "
    "D02R) and a filter link (F02H, F02R), drawn as one block.",
]


def find_warnings(data: dict) -> dict[str, str]:
    """Unit -> warning, where the solve departs from the intended process."""
    streams = data["streams"]
    out = {}

    cr01 = streams["CR01H out"]
    crystals = cr01["solid"].get("F", 0.0)
    f_total = crystals + cr01["liquid"].get("F", 0.0)
    if f_total > 0 and crystals / f_total > CR01H_CRYSTAL_LIMIT:
        out["CR01H"] = (f"CR01H crystallizes {crystals * 1e3:.3g} g/s of F "
                        f"({crystals / f_total:.0%} of its F); it should "
                        "stay liquid.")

    short = {n: v for n, v in dissolved_share(streams["D03R product"]).items()
             if v < D03R_DISSOLVED_TARGET}
    if short:
        out["D03R"] = ("D03R leaves " + percent_list(short) + " dissolved; L "
                       "should dissolve D, E, J almost completely (retune "
                       "SOLVENT_K_L or L_FLOW).")

    for block in WASH_STREAMS:
        share = wash_dissolved(streams, block)
        if share > WASH_DISSOLVED_LIMIT:
            out[block] = (f"{block} dissolves {share:.0%} of the crystals fed "
                          "to it; a wash should dissolve very little.")

    for name, unit in data["units"].items():
        where = DRAWN_IN.get(name, name)
        if unit["settling"] > SETTLED_LIMIT and where not in out:
            out[where] = (f"{name} not at steady state: still moving "
                          f"{unit['settling']:.1%} over the last 10 % of the run.")
    return out


def print_summary(data: dict):
    system = data["system"]
    print(f"{system['class']} ({system['module']}): {system['states']} states, "
          f"{system['integrator']}, t = {system['time_h']:.1f} h\n")
    print(f"{'unit':<7}{'class':<33}{'module':<36}settling")
    for name, unit in data["units"].items():
        cls = unit["class"]
        sub = cls.get("subclass")
        print(f"{name:<7}{cls['name']:<33}{cls['module']:<36}"
              f"{unit['settling']:.2%}"
              + (f"   (as {sub['module']}.{sub['name']})" if sub else ""))
    print("\nLinks:")
    for link in data["links"]:
        separator = link.get("separator", {}).get("class", "")
        print(f"  {link['source']:>6} -> {link['sink']:<6} "
              f"{link['coupling']} {separator}")
    print("\nMass closure, (in - out) / in:")
    streams = data["streams"]
    for unit, (ins, outs) in data["balances"].items():
        mass_in = sum(sum(phase_totals(streams[s])) for s in ins)
        mass_out = sum(sum(phase_totals(streams[s])) for s in outs)
        print(f"  {unit:<7}{mass_in * 1e3:>9.4f} g/s in  "
              f"{mass_out * 1e3:>9.4f} g/s out  "
              f"{(mass_in - mass_out) / mass_in:>7.2%}")
    for block in WASH_STREAMS:
        print(f"{block} dissolves {wash_dissolved(streams, block):.2%} of "
              "the crystals fed to it")
    print("\nDissolved share in the D03R product: "
          + percent_list(dissolved_share(streams["D03R product"])))
    warnings = find_warnings(data)
    print("\nWarnings:" if warnings else "\nWarnings: none")
    for text in warnings.values():
        print("  " + text)


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
NOTE_FILL = "#fff4d6"  # assumptions
NOTE_EDGE = "#c98a00"
WARN_FILL = "#fdeceb"  # warnings: always with the word WARNING, never colour alone
WARN_EDGE = "#c62f2f"

# Trebuchet MS for text (humanist, ships with Windows, true bold, lining
# figures with a distinct 0/O and 1/l, so F01H reads as F01H) and Consolas
# for code names. Each is a fallback chain: glyphs a face lacks and machines
# without it fall through to DejaVu.
FONT = ["Trebuchet MS", "DejaVu Sans"]
MONO = ["Consolas", "DejaVu Sans Mono"]


def use_fonts():
    import matplotlib as mpl
    mpl.rcParams["font.family"] = FONT
    mpl.rcParams["font.monospace"] = MONO

BOX_W, BOX_H = 3.2, 2.7
CARD_W, CARD_H = 3.0, 1.22
ROW_1, ROW_2 = 8.9, 2.7  # y of each row's centre
COLUMNS = (3.5, 10.0, 16.5, 23.0, 29.5)  # x of each column's centre
X_MAX = 32.0

ROLES = {
    "CR01H": "reactive crystallizer",
    "CR02H": "reactive crystallizer, A push",
    "F01H": "filter",
    "W02H": "wash: D02H dissolver (H) + F02H filter",
    "D01H": "dissolver (H)",
    "R02R": "jacketed CSTR, liquid only",
    "D01R": "antisolvent crystallizer (B)",
    "F01R": "filter",
    "W02R": "wash: D02R dissolver (B) + F02R filter",
    "D03R": "dissolver (L)",
}

# Wash block -> (its dissolver vessel, its filter link).
WASH_BLOCKS = {"W02H": ("D02H", "F02H"), "W02R": ("D02R", "F02R")}


def pipe(ax, points, phase, head=True):
    """A stream through ``points``: one blue line for liquid, blue and orange
    side by side for anything carrying crystals. Turns must be right angles."""
    points = np.asarray(points, dtype=float)
    if phase == "liquid":
        ax.plot(*points.T, color=LIQUID_COLOR, lw=2.2, zorder=2,
                solid_joinstyle="round")
        head_color = LIQUID_COLOR
    else:
        steps = np.diff(points, axis=0)
        normals = np.column_stack((-steps[:, 1], steps[:, 0]))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        # Each vertex moves along both neighbouring normals, which keeps the
        # two lines parallel through a right-angle turn.
        offsets = np.vstack((normals[:1], normals[:-1] + normals[1:],
                             normals[-1:])) * 0.055
        ax.plot(*(points + offsets).T, color=LIQUID_COLOR, lw=2.2, zorder=2,
                solid_capstyle="butt")
        ax.plot(*(points - offsets).T, color=SOLID_COLOR, lw=2.2, zorder=2,
                solid_capstyle="butt")
        head_color = TEXT_PRIMARY
    if head:
        ax.annotate("", xy=points[-1], xytext=points[-2], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=head_color, lw=0,
                                    mutation_scale=17, shrinkA=0, shrinkB=0))


def card(ax, streams, label, x, y):
    """A stream's card, lower-left corner at (x, y): name, phase split, content."""
    from matplotlib.patches import Rectangle

    record = streams[label]
    liquid, solid = phase_totals(record)
    total = liquid + solid

    ax.add_patch(Rectangle((x, y), CARD_W, CARD_H, fc=SURFACE, ec=GRID_COLOR,
                           lw=1, zorder=4))
    left, top = x + 0.08, y + CARD_H
    ax.text(left, top - 0.17, label, fontsize=9.5, weight="bold",
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


def box(ax, name, x, y, class_lines, facts, dashed=False):
    """A unit centred on (x, y). Dashed means it holds no states of its own."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0,rounding_size=0.08",
        fc=SURFACE if dashed else BOX_FILL, ec=TEXT_MUTED, lw=1.3,
        ls=(0, (4, 2)) if dashed else "-", zorder=3))
    left = x - BOX_W / 2 + 0.12
    top = y + BOX_H / 2

    def line(offset, text, **style):
        style = {"fontsize": 8.6, "color": TEXT_PRIMARY, "va": "center",
                 "zorder": 5, **style}
        ax.text(left, top - offset, text, **style)

    line(0.24, name, fontsize=14, weight="bold")
    line(0.53, ROLES[name], fontsize=8.5, color=TEXT_MUTED)
    line(0.84, "class", fontsize=7.5, color=TEXT_MUTED)
    offset = 1.03
    for i, (text, kind) in enumerate(class_lines):
        if kind == "class":
            line(offset, text, weight="bold", family="monospace")
        else:
            line(offset, text, fontsize=7.5, color=TEXT_MUTED,
                 family="monospace")
        offset += 0.2
    offset += 0.08
    for fact in facts:
        line(offset, fact, weight="bold" if fact.startswith("product") else
             "normal")
        offset += 0.25


def vessel_box(ax, data, name, x, y, extra=()):
    unit = data["units"][name]
    cls = unit["class"]
    class_lines = [(cls["name"], "class"), (cls["module"], "module")]
    if cls.get("subclass"):
        sub = cls["subclass"]
        class_lines.append((f"run as {sub['module']}.{sub['name']}", "module"))
    if cls["mixins"]:
        class_lines.append(("+ " + ", ".join(cls["mixins"]), "module"))
    solids = unit["solids"]
    if len(solids) > 1:
        holds = (f"holds: liquid + {len(solids)} solid phases "
                 + " | ".join(solids))
    elif solids:
        holds = f"holds: liquid + solid {solids[0]}"
    else:
        holds = "holds: liquid only"
    facts = [holds]
    reactions = unit["reactions"]
    if len(reactions) > 1:
        facts.append("F + A + I + G → D / E / J + H")
    elif reactions:
        facts.append(reactions[0].replace("-->", "→"))
    if unit.get("k_d"):
        facts.append("dissolution k_d "
                     + ", ".join(f"{k:g}" for k in unit["k_d"]) + " µm/s")
    facts += list(extra)
    box(ax, name, x, y, class_lines, facts)


def filter_box(ax, data, name, x, y):
    """A filter: no vessel, only an algebraic step inside a link."""
    link = next(item for item in data["links"] if item["name"] == name)
    separator = link["separator"]
    class_lines = [
        (link["coupling"], "class"),
        ("Flowsheet (link coupling, no states)", "module"),
        (f"settings: {separator['class']}", "module"),
    ]
    facts = [f"{link['source']} → {link['sink']} link",
             f"solids loss {separator['solids_loss']:.0%}, "
             f"cake LOD {separator['cake_moisture']:.0%}"]
    box(ax, name, x, y, class_lines, facts, dashed=True)


def wash_box(ax, data, block, x, y):
    """A wash block: its dissolver vessel and its filter link, as one box."""
    vessel, filter_name = WASH_BLOCKS[block]
    unit = data["units"][vessel]
    cls = unit["class"]
    link = next(item for item in data["links"] if item["name"] == filter_name)
    separator = link["separator"]
    class_lines = [(f"{vessel} {cls['name']}", "class"),
                   (cls["module"], "module")]
    if cls["mixins"]:
        class_lines.append(("+ " + ", ".join(cls["mixins"]), "module"))
    class_lines += [
        (f"{filter_name} {link['coupling']}", "class"),
        (f"link: solids loss {separator['solids_loss']:.0%}, "
         f"cake LOD {separator['cake_moisture']:.0%}", "module"),
    ]
    solids = unit["solids"]
    holds = (f"{vessel} holds liquid + solid " + " | ".join(solids)
             if solids else f"{vessel} holds liquid only")
    facts = [holds,
             f"product: dissolves {wash_dissolved(data['streams'], block):.1%}"
             " of crystals fed"]
    box(ax, block, x, y, class_lines, facts)


def note(ax, x, y, text, warn=False, **style):
    style = {"fontsize": 8.8, "color": TEXT_PRIMARY, "va": "center",
             "ha": "center", "zorder": 6,
             "bbox": dict(fc=WARN_FILL if warn else NOTE_FILL,
                          ec=WARN_EDGE if warn else NOTE_EDGE,
                          lw=1.2 if warn else 0.8,
                          boxstyle="round,pad=0.3"), **style}
    ax.text(x, y, text, **style)


def draw_map(data: dict):
    """The flowsheet: units with their classes, streams with their phases."""
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    streams = data["streams"]
    system = data["system"]
    warnings = find_warnings(data)
    fig, ax = plt.subplots(figsize=(27.5, 15.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-4.5, 13.1)
    ax.axis("off")

    c1, c2, c3, c4, c5 = COLUMNS
    half_w, half_h = BOX_W / 2, BOX_H / 2
    top_1, bottom_1 = ROW_1 + half_h, ROW_1 - half_h
    top_2, bottom_2 = ROW_2 + half_h, ROW_2 - half_h
    top_cards, low_cards = 11.1, -0.35
    above_row_1, above_row_2 = ROW_1 + 0.2, ROW_2 + 0.2
    mid_cards, feed_row_2 = 6.05, 4.35
    route_y = 5.75

    def phase_of(label):
        return phase_label(streams[label])

    def gap_card(label, column, y):
        """A card above the pipe from ``column`` to the next column."""
        card(ax, streams, label, column + half_w + 0.15, y)

    def centred_card(label, column, y):
        card(ax, streams, label, column - CARD_W / 2, y)

    def across(column, label, row):
        """The pipe from ``column`` to the next column, with its card."""
        nxt = COLUMNS[COLUMNS.index(column) + 1]
        pipe(ax, [(column + half_w, row), (nxt - half_w, row)],
             phase_of(label))
        gap_card(label, column, row + 0.2)

    product = streams["D03R product"]
    vessel_box(ax, data, "CR01H", c1, ROW_1)
    vessel_box(ax, data, "CR02H", c2, ROW_1)
    filter_box(ax, data, "F01H", c3, ROW_1)
    wash_box(ax, data, "W02H", c4, ROW_1)
    vessel_box(ax, data, "D01H", c5, ROW_1)
    vessel_box(ax, data, "R02R", c1, ROW_2,
               ["ASSUMPTION A4: no solids form here"])
    vessel_box(ax, data, "D01R", c2, ROW_2)
    filter_box(ax, data, "F01R", c3, ROW_2)
    wash_box(ax, data, "W02R", c4, ROW_2)
    vessel_box(ax, data, "D03R", c5, ROW_2,
               ["product dissolved  "
                + percent_list(dissolved_share(product))])

    # A badge on each unit with a warning; the text is in the panel below.
    tags = {name: f"W{i}" for i, name in enumerate(warnings, start=1)}
    centres = {"CR01H": (c1, ROW_1), "CR02H": (c2, ROW_1), "F01H": (c3, ROW_1),
               "W02H": (c4, ROW_1), "D01H": (c5, ROW_1), "R02R": (c1, ROW_2),
               "D01R": (c2, ROW_2), "F01R": (c3, ROW_2), "W02R": (c4, ROW_2),
               "D03R": (c5, ROW_2)}
    for name, tag in tags.items():
        x, y = centres[name]
        note(ax, x + half_w - 0.62, y + half_h - 0.24, f"WARNING {tag}",
             warn=True, fontsize=8.5, weight="bold", color=WARN_EDGE)

    # ---- row 1
    card(ax, streams, "Feed A", 0.25, top_cards)
    card(ax, streams, "Feed C", 3.4, top_cards)
    pipe(ax, [(2.6, top_cards), (2.6, top_1)], "liquid")
    pipe(ax, [(4.4, top_cards), (4.4, top_1)], "liquid")
    across(c1, "CR01H out", ROW_1)

    centred_card("A push", c2, top_cards)
    pipe(ax, [(c2, top_cards), (c2, top_1)], "liquid")
    across(c2, "CR02H out", ROW_1)

    pipe(ax, [(c3, bottom_1), (c3, mid_cards + CARD_H)],
         phase_of("Filtrate 1"))
    centred_card("Filtrate 1", c3, mid_cards)
    across(c3, "F01H cake", ROW_1)

    centred_card("W02H wash H", c4, top_cards)
    pipe(ax, [(c4, top_cards), (c4, top_1)], "liquid")
    pipe(ax, [(c4, bottom_1), (c4, mid_cards + CARD_H)],
         phase_of("W02H filtrate"))
    centred_card("W02H filtrate", c4, mid_cards)
    across(c4, "W02H cake", ROW_1)

    centred_card("Feed H", c5, top_cards)
    pipe(ax, [(c5, top_cards), (c5, top_1)], "liquid")
    pipe(ax, [(c5, bottom_1), (c5, mid_cards + CARD_H)], phase_of("D01H out"))
    centred_card("D01H out", c5, mid_cards)

    # ---- D01H -> R02R: crystals become dissolved F inside the link (A1).
    # The route runs under the filtrate cards, then down into R02R.
    link = next(item for item in data["links"] if item["sink"] == "R02R")
    pill_x = 9.2
    pipe(ax, [(c5, mid_cards), (c5, route_y), (pill_x + 2.0, route_y)],
         phase_of("D01H out"), head=False)
    pipe(ax, [(pill_x - 2.0, route_y), (4.4, route_y), (4.4, top_2)], "liquid")
    note(ax, pill_x, route_y,
         "ASSUMPTION A1: solid F dissolves quickly, then reacts\n"
         f"({link['coupling']}: slurry in, liquid out to R02R)")

    # ---- row 2
    card(ax, streams, "R02R makeup", 0.25, feed_row_2)
    pipe(ax, [(2.4, feed_row_2), (2.4, top_2)], "liquid")
    pipe(ax, [(c1 + half_w, ROW_2), (c2 - half_w, ROW_2)], "liquid")
    gap_card("R02R out", c1, above_row_2)

    centred_card("Feed B", c2, low_cards)
    pipe(ax, [(c2, low_cards + CARD_H), (c2, bottom_2)], "liquid")
    across(c2, "D01R out", ROW_2)

    pipe(ax, [(c3, bottom_2), (c3, low_cards + CARD_H)],
         phase_of("Filtrate 2"))
    centred_card("Filtrate 2", c3, low_cards)
    across(c3, "F01R cake", ROW_2)

    centred_card("W02R wash B", c4, feed_row_2)
    pipe(ax, [(c4, feed_row_2), (c4, top_2)], "liquid")
    pipe(ax, [(c4, bottom_2), (c4, low_cards + CARD_H)],
         phase_of("W02R filtrate"))
    centred_card("W02R filtrate", c4, low_cards)
    across(c4, "W02R cake", ROW_2)

    centred_card("Feed L", c5, feed_row_2)
    pipe(ax, [(c5, feed_row_2), (c5, top_2)], "liquid")
    pipe(ax, [(c5, bottom_2), (c5, low_cards + CARD_H)],
         phase_of("D03R product"))
    centred_card("D03R product", c5, low_cards)

    # ---- title and legend
    fig.suptitle(
        "Flowsheet phases and unit classes: D03R product is "
        + percent_list(dissolved_share(product)) + " dissolved",
        fontsize=17, weight="bold", color=TEXT_PRIMARY, x=0.012, ha="left",
        y=0.99)
    ax.text(0.25, 12.8,
            f"All vessels solved together in one {system['class']} "
            f"({system['states']} states, {system['integrator']}), flows at "
            f"t = {system['time_h']:.0f} h. Card contents in g/s. "
            "Classes read from the solved objects.",
            fontsize=10.5, color=TEXT_MUTED, va="center")
    handles = [
        Line2D([], [], color=LIQUID_COLOR, lw=2.6),
        (Line2D([], [], color=LIQUID_COLOR, lw=2.2),
         Line2D([], [], color=SOLID_COLOR, lw=2.2)),
        Line2D([], [], color=TEXT_MUTED, lw=1.3, ls=(0, (4, 2))),
        Line2D([], [], color=TEXT_MUTED, lw=1.3),
    ]
    ax.legend(handles, ["Liquid stream",
                        "Slurry or wet cake (blue + orange lines)",
                        "Dashed box: algebraic step in a link, no states",
                        "Solid box: vessel with its own states"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              loc="upper left", bbox_to_anchor=(0.1, -0.55),
              bbox_transform=ax.transData, ncol=4, frameon=False,
              fontsize=10.5)

    # ---- assumptions and warnings
    def panel(x, title, lines, warn):
        ax.text(x, -1.5, title, fontsize=11.5, weight="bold", va="top",
                color=WARN_EDGE if warn else TEXT_PRIMARY)
        y = -1.95
        for text in lines:
            wrapped = textwrap.fill(text, 112, subsequent_indent="      ")
            ax.text(x, y, wrapped, fontsize=10, va="top", color=TEXT_PRIMARY,
                    linespacing=1.3)
            y -= 0.25 * (wrapped.count("\n") + 1) + 0.12

    panel(0.25, "Assumptions", ASSUMPTIONS, warn=False)
    panel(16.25, "WARNINGS" if warnings else "Warnings: none",
          [f"{tags[name]}  {text}" for name, text in warnings.items()],
          warn=bool(warnings))
    fig.subplots_adjust(left=0.01, right=0.995, top=0.965, bottom=0.005)
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
    for label, record in streams.items():
        liquid = [record["liquid"].get(n, 0.0) for n in liquids]
        solid = [record["solid"].get(n, 0.0) for n in solids]
        rows.append([label, phase_label(record), f"{record['temp']:.0f}",
                     f"{sum(phase_totals(record)) * 1e3:.3g}"]
                    + [cell(v) for v in liquid + solid])
        colors.append(
            [SURFACE] * 4
            + [shade(LIQUID_COLOR, v) if v > 5e-7 else SURFACE for v in liquid]
            + [shade(SOLID_COLOR, v) if v > 5e-7 else SURFACE for v in solid])

    fig, ax = plt.subplots(figsize=(17, 0.34 * len(rows) + 0.7))
    ax.axis("off")
    first = [0.12, 0.065, 0.045, 0.05]
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
    use_fonts()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    draw_map(flowsheet).savefig(MAP_PATH, dpi=130)
    draw_table(flowsheet).savefig(TABLE_PATH, dpi=130, bbox_inches="tight")
    print(f"\nFigures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
