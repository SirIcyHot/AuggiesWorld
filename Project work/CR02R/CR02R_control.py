"""CR02R as a one-member ConcurrentVessel, for control studies.

The vessel is ``CR02R.build_reactor()`` unchanged (DESIGN_FEED plus makeup,
R02R's reactions and jacket, F/D/E/J population balances); this file only
chooses its controller, wraps it in a ConcurrentVessel and solves it.

Everything control-related is shared with ``R02R/R02R_control.py`` -- the
four modes (``open``, ``pi``, ``jacket``, ``imposed``), the set point step,
the jacket limits, ``install_controller`` and the figures -- so the two
units are tested with exactly the same loops. See that file's docstring for
what each native PharmaPy controller does.

What is different here is the process the loops act on. CR02R is a
reactive crystallizer, so:

- the temperature set point also moves the solubilities (~3 %/K), so a
  temperature step is a step in how much D, E and J crystallize;
- the slurry outlet carries crystals, so the outflow has to differ from the
  inflow at steady state and the P-only level loop (``open``) leaves a
  volume offset; the PI loops remove it.

The population balances make this ~100x slower than R02R per mode, so the
default is one mode (``jacket``); ``--compare`` runs all four.

Run it with::

    python CR02R_control.py --no-show
    python CR02R_control.py --compare --no-show
    python CR02R_control.py --mode imposed --runtime 7200

Figures go to ``results/control/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "R02R"))

import CR02R  # noqa: E402
import R02R_control as control  # noqa: E402

RESULTS_DIR = HERE / "results" / "control"

# CR02R.build_reactor uses Assimulo's default CVode settings; keep them.
SOLVER_OPTIONS = None


def build(mode: str):
    vessel = control.install_controller(CR02R.build_reactor(),
                                        control.make_controller(mode))
    return control.wrap("CR02R", vessel, SOLVER_OPTIONS)


if __name__ == "__main__":
    control.main("CR02R", build, __doc__, default_mode="jacket",
                 out_dir=RESULTS_DIR)
