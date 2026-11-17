"""Tests for CR02R, R02R as a reactive crystallizer.

The fixture solves CR02R standalone once, on DESIGN_FEED.

Units: mass flow [kg/s], solubility [kg/kg H], temperature [K].
"""

import warnings

import numpy as np
import pytest

import CR02R


@pytest.fixture(scope="module")
def reactor():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CR02R.run_reactor()


def test_solids_follow_solid_species(reactor):
    names = CR02R.species_names()
    for k, species in enumerate(CR02R.SOLID_SPECIES):
        pbm = CR02R.population_balance(reactor, k)
        assert names[pbm.target_ind[0]] == species
        np.testing.assert_array_equal(pbm.x_grid, CR02R.size_grid(species))


def test_design_feed_carries_the_f_crystals():
    feed = CR02R.design_feed()
    assert [s.phase_family for s in feed.Streams] == ["liquid", "solid"]
    crystals = float(np.sum(feed.Streams[1].mass_flow))
    assert crystals == pytest.approx(CR02R.DESIGN_FEED["f_crystals"],
                                     rel=1e-3)


def test_makeup_has_no_f():
    names = CR02R.species_names()
    makeup = CR02R.makeup_stream(names)
    assert makeup.mass_frac[names.index("F")] == 0.0


@pytest.mark.parametrize("species, temp", [("D", 333.0), ("E", 320.0),
                                           ("J", 340.0), ("F", 300.0)])
def test_apelblat_solubility(species, temp):
    a, b, c = CR02R.SOLUBILITY_APELBLAT[species]
    assert float(CR02R.solubility(species, temp)) == pytest.approx(
        np.exp(a + b / temp + c * np.log(temp)), rel=1e-12)


def test_kinetics_use_the_apelblat_fit(reactor):
    """PharmaPy's own solubility equals CR02R's, on the kg/kg H basis."""
    temp = float(np.ravel(reactor.result.global_temp)[-1])
    for k, species in enumerate(CR02R.SOLID_SPECIES):
        kinetics = CR02R.population_balance(reactor, k).mechanism_kinetics
        assert kinetics.solubility_basis == "mass_per_mass_solvent"
        assert float(kinetics.get_solubility(temp)) == pytest.approx(
            float(CR02R.solubility(species, temp)), rel=1e-12)


def test_about_20_percent_of_each_product_crystallizes(reactor):
    for species, share in CR02R.fraction_crystallized(
            CR02R.outlet(reactor)).items():
        assert 0.15 < share < 0.25, species


def test_f_crystals_dissolve(reactor):
    out = CR02R.outlet(reactor)
    assert out.crystals["F"] < 0.01 * CR02R.DESIGN_FEED["f_crystals"]


def test_products_match_the_f_fed(reactor):
    """Each product is made 1:1 from F, so the products' F equivalent equals
    the F fed (within 1 %). On a too-coarse F grid dissolution makes extra
    dissolved F and this comes out ~16 % high."""
    out = CR02R.outlet(reactor)
    names = CR02R.species_names()
    mw = dict(zip(names, np.asarray(reactor.Phases.Liquids[0].mw, float)))
    made = sum((out.dissolved[s] + out.crystals[s]) / mw[s]
               for s in CR02R.CRYSTAL_SPECIES) * mw["F"]
    fed = (CR02R.DESIGN_FEED["f_crystals"]
           + CR02R.DESIGN_FEED["liquid"]["F"])
    left = out.dissolved["F"] + out.crystals["F"]
    assert made == pytest.approx(fed - left, rel=0.01)


def test_steady_mass_balance(reactor):
    """Within 2 %: R02R alone, with the same reactions, ends ~1.2 % short
    (its R02R_mass_balance figure), and CR02R matches it."""
    fed, out = CR02R.mass_closure(reactor)
    assert out == pytest.approx(fed, rel=0.02)


def test_crystals_stay_on_the_grid(reactor):
    """Under 1 % of each product's crystal mass sits in the top 5 cells."""
    res = reactor.result
    for k, species in enumerate(CR02R.SOLID_SPECIES):
        if species == "F":
            continue
        pbm = CR02R.population_balance(reactor, k)
        x = np.asarray(pbm.x_grid)
        mass = np.asarray(getattr(res, f"distrib_solid{k}"))[-1] * x**3 * pbm.dx
        assert mass[-5:].sum() < 0.01 * mass.sum(), species


def test_figures_are_written(reactor, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    CR02R.save_figures(reactor, out_dir=tmp_path)
    for name in ("CR02R_profiles", "CR02R_csd", "CR02R_settings"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
