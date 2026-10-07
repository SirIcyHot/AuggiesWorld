"""Import-boundary regressions for the optional Assimulo solver stack.

Covers both the legacy unit-operation modules and the MultiPhaseVessel
refactor modules. Assimulo is blocked in a subprocess rather than skipped, so
these tests exercise the dependency-free path even where Assimulo is
installed.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType

import pytest

import PharmaPy._assimulo as assimulo_backend

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
AFFECTED_MODULES = (
    "PharmaPy.Containers",
    "PharmaPy.Crystallizers",
    "PharmaPy.Distillation",
    "PharmaPy.Drying_Model",
    "PharmaPy.DynamicExtraction",
    "PharmaPy.Evaporators",
    "PharmaPy.Reactors",
    "PharmaPy.SolidLiquidSep",
    "PharmaPy.ThreePhaseSettler",
    # MultiPhaseVessel refactor modules.
    "PharmaPy.Crystallizers_Refactored",
    "PharmaPy.DataClasses",
    "PharmaPy.IntegratorBackends",
    "PharmaPy.Mechanisms",
    "PharmaPy.MixedPhases_Refactored",
    "PharmaPy.MultiPhaseVessel",
    "PharmaPy.Phases_Refactored",
    "PharmaPy.ProcessControl_Refactored",
    "PharmaPy.Reactors_Refactored",
    "PharmaPy.Streams_Refactored",
)
# Modules that raise TerminateSimulation from solver event handlers and must
# therefore share one exception class with PharmaPy.Commons.
TERMINATE_SIMULATION_CONSUMERS = (
    "PharmaPy.Evaporators",
    "PharmaPy.SolidLiquidSep",
)
LAZY_CONSTRUCTORS = (
    "CVode",
    "IDA",
    "Explicit_Problem",
    "Implicit_Problem",
)

# Block Assimulo through a sys.meta_path finder rather than by patching
# builtins.__import__. importlib.import_module, which PharmaPy._assimulo uses to
# load the backend, resolves through importlib._bootstrap._gcd_import and never
# consults builtins.__import__, so a builtins patch lets the real Assimulo load
# whenever it is installed. A meta-path finder participates in that resolution
# and so blocks both statement imports and importlib.import_module.
#
# ModuleNotFoundError carries name="assimulo" because _load_assimulo_symbol
# distinguishes a missing install from a broken one on exactly that attribute.
IMPORT_BLOCKER = textwrap.dedent("""
    import sys

    class _AssimuloBlocker:
        def find_spec(self, name, path=None, target=None):
            if name == "assimulo" or name.startswith("assimulo."):
                raise ModuleNotFoundError(
                    "Assimulo import blocked by regression test",
                    name="assimulo",
                )
            return None

    sys.meta_path.insert(0, _AssimuloBlocker())
    """)


def _run_without_assimulo(script, tmp_path):
    """Run Python source while rejecting imports of Assimulo.

    Parameters
    ----------
    script : str
        Python source to execute after installing the import blocker.
    tmp_path : pathlib.Path
        Temporary directory used for the Matplotlib configuration cache.

    Returns
    -------
    subprocess.CompletedProcess
        Completed child-process result with captured text output.
    """
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, "-c", f"{IMPORT_BLOCKER}\n{script}"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_model_modules_import_without_assimulo(tmp_path):
    """Every solver-backed model module imports without the optional backend."""
    script = textwrap.dedent(f"""
        import importlib

        for module_name in {AFFECTED_MODULES!r}:
            importlib.import_module(module_name)
        """)
    result = _run_without_assimulo(script, tmp_path)

    assert result.returncode == 0, result.stderr


def test_terminate_simulation_fallback_is_raisable(tmp_path):
    """The Assimulo-free TerminateSimulation is a usable exception class.

    Event handlers in PharmaPy.Commons raise TerminateSimulation to stop an
    integration, so this name cannot use the lazy-constructor pattern applied
    to the solvers: a factory function is neither raisable nor valid in an
    ``except`` clause. Assert the exception protocol rather than mere
    importability, which a plain function would also satisfy.
    """
    script = textwrap.dedent("""
        from PharmaPy.Commons import TerminateSimulation

        if not isinstance(TerminateSimulation, type):
            raise AssertionError(
                f"expected a class, got {TerminateSimulation!r}"
            )
        if not issubclass(TerminateSimulation, BaseException):
            raise AssertionError(
                f"expected an exception class, got {TerminateSimulation!r}"
            )

        sentinel = "state event reached"
        try:
            raise TerminateSimulation(sentinel)
        except TerminateSimulation as exc:
            if str(exc) != sentinel:
                raise AssertionError(
                    f"payload not preserved: {exc!s}"
                ) from exc
        """)
    result = _run_without_assimulo(script, tmp_path)

    assert result.returncode == 0, result.stderr


def test_terminate_simulation_is_shared_across_modules(tmp_path):
    """Model modules reuse the Commons TerminateSimulation object.

    A per-module fallback definition would still import cleanly while making
    ``except TerminateSimulation`` in one module blind to the exception raised
    by another, so compare object identity rather than the name alone.
    """
    script = textwrap.dedent(f"""
        import importlib

        from PharmaPy.Commons import TerminateSimulation

        for module_name in {TERMINATE_SIMULATION_CONSUMERS!r}:
            module = importlib.import_module(module_name)
            if module.TerminateSimulation is not TerminateSimulation:
                raise AssertionError(
                    f"{{module_name}} does not share the Commons exception"
                )
        """)
    result = _run_without_assimulo(script, tmp_path)

    assert result.returncode == 0, result.stderr


def test_refactored_vessel_solves_on_default_scipy_backend(tmp_path):
    """A vessel built without an integrator solves with scipy alone.

    This is the path a plain ``pip install`` user takes: no Assimulo, no
    ``integrator=`` argument. Asserting the backend type as well as a finished
    solve pins the default itself, not merely that some backend ran.
    """
    script = textwrap.dedent("""
        import os

        import numpy as np

        from PharmaPy.IntegratorBackends import ScipyBackend
        from PharmaPy.Kinetics import RxnKinetics
        from PharmaPy.Phases_Refactored import LiquidPhase
        from PharmaPy.Reactors_Refactored import BatchReactor
        from PharmaPy.Utilities import CoolingWater

        path = os.path.join("tests", "Flowsheet", "data",
                            "compound_database.json")

        vessel = BatchReactor(h_conv=1e4, diam=0.01)
        vessel.Phases = LiquidPhase(path, mass=1.0,
                                    mass_frac=[0.4, 0.6, 0.0, 0.0, 0.0])
        vessel.Utility = CoolingWater(mass_flow=100.0, temp_in=273.55)
        vessel.RxnKinetics = RxnKinetics(
            path=path,
            rxn_list=["A + B --> C", "C + A --> D"],
            k_params=np.array([1e-2, 1e-2]),
            ea_params=np.array([1e2, 1e2]),
        )

        if not isinstance(vessel.integrator, ScipyBackend):
            raise AssertionError(
                f"default integrator is {type(vessel.integrator).__name__}"
            )

        time, states = vessel.solve_unit(runtime=10.0)

        if not np.isclose(time[-1], 10.0):
            raise AssertionError(f"solve stopped at t={time[-1]}")
        if not np.all(np.isfinite(states)):
            raise AssertionError("solve returned non-finite states")
        """)
    result = _run_without_assimulo(script, tmp_path)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("symbol_name", LAZY_CONSTRUCTORS)
def test_solver_construction_reports_missing_assimulo(symbol_name, tmp_path):
    """Lazy solver construction gives an actionable dependency error."""
    script = textwrap.dedent(f"""
        from PharmaPy._assimulo import {symbol_name}

        try:
            {symbol_name}(object())
        except ImportError as exc:
            expected = (
                "Assimulo is required for solver-backed PharmaPy simulations"
            )
            if expected not in str(exc):
                raise AssertionError(str(exc)) from exc
            # Both the missing-install and the incompatible-install branch embed
            # the message above, so assert the discriminator too: a missing
            # install is reported from ModuleNotFoundError, while an installed
            # backend lacking the symbol is reported from AttributeError and
            # adds "does not provide". Without this the test would still pass
            # while exercising the branch it is not named for.
            if not isinstance(exc.__cause__, ModuleNotFoundError):
                raise AssertionError(
                    "expected a missing-install error, got cause "
                    f"{{type(exc.__cause__).__name__}}: {{exc}}"
                ) from exc
        else:
            raise AssertionError(
                "{symbol_name} construction unexpectedly succeeded"
            )
        """)
    result = _run_without_assimulo(script, tmp_path)

    assert result.returncode == 0, result.stderr


def test_missing_assimulo_symbol_reports_qualified_name(monkeypatch):
    """An incompatible install identifies the missing constructor."""
    stub_module = ModuleType("assimulo.solvers")

    def load_stub(module_name):
        assert module_name == "assimulo.solvers"
        return stub_module

    monkeypatch.setattr(assimulo_backend, "import_module", load_stub)

    with pytest.raises(
        ImportError,
        match=r"assimulo\.solvers\.CVode",
    ) as exc_info:
        assimulo_backend.CVode()

    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_broken_assimulo_install_has_distinct_error(monkeypatch):
    """A backend loader failure is not misreported as a missing install."""
    loader_error = "SUNDIALS shared library could not be loaded"

    def fail_import(module_name):
        assert module_name == "assimulo.solvers"
        raise ImportError(loader_error)

    monkeypatch.setattr(assimulo_backend, "import_module", fail_import)

    with pytest.raises(
        ImportError,
        match="Assimulo is installed but could not be imported",
    ) as exc_info:
        assimulo_backend.CVode()

    assert loader_error in str(exc_info.value.__cause__)
