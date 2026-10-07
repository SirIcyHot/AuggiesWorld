"""Regressions for the crystal size distribution basis conversions.

A size distribution can be written three ways, and PharmaPy uses all three at
different boundaries:

* **quantity** - a number density, or a volume/mass fraction *shape*
* **extent**   - absolute counts, or per m3 of slurry

Intensive number density is the standard inside the refactored stack; the other
forms are input and output formats converted at the boundary. The legacy
``SolidPhase`` stores the extensive number density and accepts a fraction shape
on construction, which is why handing the same array to both stacks gives two
different physical distributions.

These tests pin the two things that follow: that the shared conversions still
reproduce the legacy arithmetic exactly, and that a distribution stated once
physically lands identically on both stacks.
"""

import os

import numpy as np
import pytest

from PharmaPy.Distributions import (mass_fraction_to_number,
                                    number_to_volume_fraction, to_extensive,
                                    to_intensive, volume_fraction_to_number)
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases import SolidPhase as LegacySolid
from PharmaPy.Phases_Refactored import SolidPhase as RefactoredSolid

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

MASSFRAC_SOLID = [0, 0, 1, 0, 0]

VOL_LIQUID = 2750e-6  # [m**3]
MASS_SOLID = VOL_LIQUID * 2.4e-2 * 1e3  # [kg]


def legacy_solid(x_grid, shape, mass=MASS_SOLID):
    """An old SolidPhase, which reads `distrib` as a volume-fraction shape."""

    return LegacySolid(DATA_PATH, mass=mass, x_distrib=x_grid, distrib=shape,
                       mass_frac=MASSFRAC_SOLID)


def refactored_solid(x_grid, shape, vol_slurry, density, mass=MASS_SOLID):
    """The same physical solid, stated through the basis flag."""

    solid = RefactoredSolid(DATA_PATH, mass=mass, mass_frac=MASSFRAC_SOLID)
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components="C", solvent_name="solvent", x_grid=x_grid,
        distrib_init=shape, distrib_basis="vol_perc", basis_mass=mass,
        vol_slurry=vol_slurry, density=density,
    )

    return solid


def mechanism_of(solid):
    mechanisms = solid.mechanisms
    return mechanisms[0] if isinstance(mechanisms, (list, tuple)) \
        else mechanisms


# ---------------------------------------------------------------------------
# Extent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vol_slurry", [2.75e-3, 1.0, 1234.5])
def test_extent_round_trip(vol_slurry):
    """Extensive and intensive are one division apart, and it inverts."""

    distrib = np.geomspace(1e3, 1e9, 60)

    np.testing.assert_allclose(
        to_extensive(to_intensive(distrib, vol_slurry), vol_slurry),
        distrib, rtol=1e-15, atol=0,
    )


def test_extent_conversion_refuses_a_missing_slurry_volume():
    """A solid on its own has no slurry volume, and silence would be wrong."""

    for volume in (0.0, -1.0):
        with pytest.raises(ValueError, match="slurry volume"):
            to_intensive(np.ones(4), volume)

        with pytest.raises(ValueError, match="slurry volume"):
            to_extensive(np.ones(4), volume)


# ---------------------------------------------------------------------------
# Quantity - the shared copy must still be the legacy arithmetic
# ---------------------------------------------------------------------------


def test_shared_conversions_reproduce_the_legacy_expressions():
    """`Phases.convert_distribution` now delegates; it must not have moved.

    Compared against the original expressions written out inline, so this
    fails if either copy drifts rather than only if they drift apart.
    """

    x_grid = np.arange(1, 501, dtype=float)
    shape = np.ones_like(x_grid)

    solid = legacy_solid(x_grid, shape)
    density, dx, kv = solid.getDensity(), solid.dx, solid.kv
    fraction = shape / shape.sum()

    expected = (MASS_SOLID / density) * fraction / kv / x_grid**3 / dx * 1e18
    assert np.array_equal(
        volume_fraction_to_number(x_grid, dx, fraction, MASS_SOLID, density,
                                  kv=kv),
        expected,
    )

    # The construction path lands on the same numbers.
    assert np.array_equal(solid.distrib, expected)

    mom_three = solid.getMoments(distrib=solid.distrib, mom_num=3)
    expected_back = solid.distrib * dx * x_grid**3 * kv / mom_three / 1e18
    assert np.array_equal(
        number_to_volume_fraction(x_grid, dx, solid.distrib, mom_three, kv=kv),
        expected_back,
    )

    expected_mass = MASS_SOLID * fraction / x_grid**3 / kv * 1e18
    assert np.array_equal(
        mass_fraction_to_number(x_grid, fraction, MASS_SOLID, kv=kv),
        expected_mass,
    )


def test_volume_fraction_needs_a_mass():
    with pytest.raises(ValueError, match="mass"):
        volume_fraction_to_number(np.arange(1., 5.), 1.0, np.ones(4), 0, 1200.)


# ---------------------------------------------------------------------------
# The two stacks, given one physical distribution
# ---------------------------------------------------------------------------


def test_one_physical_distribution_lands_identically_on_both_stacks():
    """The point of the basis flag.

    Handing both stacks the same raw array does NOT do this - the legacy one
    reads it as a volume-fraction shape and the refactored one as a number
    density. Stated through the flag, they agree to machine precision.
    """

    x_grid = np.arange(1, 501, dtype=float)
    shape = np.ones_like(x_grid)

    legacy = legacy_solid(x_grid, shape)
    vol_slurry = VOL_LIQUID + legacy.vol

    refactored = refactored_solid(x_grid, shape, vol_slurry,
                                  legacy.getDensity())

    # The refactored solid stores the intensive form, so compare on the
    # legacy stack's extensive basis.
    np.testing.assert_allclose(
        mechanism_of(refactored).to_extensive(vol_slurry=vol_slurry),
        legacy.distrib, rtol=1e-12, atol=0,
    )


def test_moment_gap_is_quadrature_and_converges():
    """What is left after the basis conversion, and why it is not zero.

    The refactored mechanism integrates with the rectangle rule because the
    FVM stores cell averages; `SolidPhase.getMoments` uses the trapezoid. The
    rectangle rule was deliberate - the trapezoid half-weighted the first
    cell, where nucleation injects, so the population gained mu_3 at twice the
    rate mass transfer removed solute.

    So the moments differ by O(dx), and on a 1/x**3 distribution the gap is
    large because the integrand is sharply peaked at the first node. What
    makes it quadrature rather than an error is that it SHRINKS with the
    grid; a basis mistake would not.
    """

    gaps = []

    for num_nodes in (500, 1000, 2000, 4000):

        x_grid = np.linspace(1, 500, num_nodes)
        shape = np.ones_like(x_grid)

        legacy = legacy_solid(x_grid, shape)
        vol_slurry = VOL_LIQUID + legacy.vol
        refactored = refactored_solid(x_grid, shape, vol_slurry,
                                      legacy.getDensity())

        moments = np.asarray(mechanism_of(refactored).moments) * vol_slurry

        gaps.append(
            float(np.max(np.abs(moments - legacy.moments)
                         / np.abs(legacy.moments)))
        )

    gaps = np.array(gaps)

    # Monotone, and roughly halving with each doubling of the grid.
    assert np.all(np.diff(gaps) < 0), gaps
    assert gaps[-1] < 0.6 * gaps[0], gaps


# ---------------------------------------------------------------------------
# The legacy bridge
# ---------------------------------------------------------------------------


def test_legacy_bridge_requires_the_slurry_volume():
    """It used to hand absolute counts into an intensive slot in silence."""

    x_grid = np.arange(1, 51, dtype=float)
    legacy = legacy_solid(x_grid, np.ones_like(x_grid))
    owner = RefactoredSolid(DATA_PATH, mass=MASS_SOLID,
                            mass_frac=MASSFRAC_SOLID)

    with pytest.raises(ValueError, match="slurry volume"):
        OneDFVMMechanism.from_legacy_phase(
            legacy, owner, target_components="C", solvent_name="solvent")


def test_legacy_bridge_converts_extent():
    """Given the volume, the bridge lands on the intensive equivalent."""

    x_grid = np.arange(1, 51, dtype=float)
    legacy = legacy_solid(x_grid, np.ones_like(x_grid))
    vol_slurry = VOL_LIQUID + legacy.vol

    owner = RefactoredSolid(DATA_PATH, mass=MASS_SOLID,
                            mass_frac=MASSFRAC_SOLID)

    built = OneDFVMMechanism.from_legacy_phase(
        legacy, owner, target_components="C", solvent_name="solvent",
        vol_slurry=vol_slurry)

    np.testing.assert_allclose(
        np.asarray(built.distrib) * vol_slurry, legacy.distrib,
        rtol=1e-12, atol=0,
    )
