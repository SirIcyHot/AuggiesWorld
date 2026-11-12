"""Tests for D02R, the B wash (block W02R) on the F01R cake.

The fixture solves the R02R -> D01R -> F01R -> D02R chain once.

Units: mass flow [kg/s], concentration [mol/L], time [s].
"""

import warnings

import numpy as np
import pytest

# D02R puts F01R, D01R, R02R and CustomUnitOps on the path, so it is first.
import D02R


@pytest.fixture(scope="module")
def run():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cake, cake_flow, tank = D02R.run_flowsheet()
        return (cake, cake_flow, tank, D02R.history(tank, cake_flow),
                D02R.final_balance(tank))


def test_solubility_and_kinetics_are_d01rs():
    assert D02R.SOLUBILITY_S0 is D02R.D01R.SOLUBILITY_S0
    assert D02R.ANTISOLVENT_K_B == D02R.D01R.ANTISOLVENT_K_B
    assert D02R.DISSOLUTION is D02R.D01R.DISSOLUTION
    assert D02R.SOLVENT_NAME == D02R.D01R.ANTISOLVENT_NAME == "B"


def test_reaches_steady_state(run):
    *_, balance = run
    assert D02R.steady_state_gap(balance) < D02R.STEADY_STATE_TOL


def test_wash_dissolves_very_little(run):
    _, cake_flow, _, _, balance = run
    for species, share in D02R.fraction_dissolved(balance, cake_flow).items():
        assert share < 0.02, species


def test_liquid_is_mostly_b(run):
    _, _, tank, profiles, _ = run
    names = D02R.species_names()
    conc = np.asarray(tank.result.mole_conc_liquid0)[-1]
    mw = np.asarray(tank.Phases.Liquids[0].mw)
    mass = conc * mw
    assert mass[names.index("B")] / mass.sum() > 0.5


def test_mass_balance_closes(run):
    *_, balance = run
    total = D02R.balance_rows(balance)[-1]
    _, cake_in, b_in, _, out, accumulated = total
    assert abs(cake_in + b_in - out - accumulated) < 1e-3 * (cake_in + b_in)
