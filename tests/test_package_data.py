"""Runtime data files ship inside the PharmaPy package.

A pip install copies only the package directory, so every file PharmaPy reads
at runtime has to resolve relative to the package rather than the repository
root. Run against a non-editable install, this also checks that the
package-data globs in pyproject.toml pick the files up.
"""

import pathlib

import pytest

import PharmaPy
from PharmaPy import CheckModule

pytestmark = pytest.mark.unit

PACKAGE_DATA = pathlib.Path(PharmaPy.__file__).parent / "data"


@pytest.mark.parametrize("relative_path", [
    "minimum_modeling_objects.json",
    "evaporator/props_nitrogen.json",
    "thermodynamics/unifac_interaction_params.csv",
    "thermodynamics/unifac_rk_qk.csv",
])
def test_runtime_data_ships_inside_the_package(relative_path):
    assert (PACKAGE_DATA / relative_path).is_file()


def test_modeling_object_checks_read_the_packaged_file():
    assert CheckModule.root == PACKAGE_DATA
