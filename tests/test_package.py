"""Public API, independence, and installed typing checks."""
import doctest
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import laprust


def test_only_two_public_solvers():
    assert laprust.__all__ == ["lapjv", "lapjv_batch"]
    assert [name for name in dir(laprust) if not name.startswith("_")] == ["lapjv", "lapjv_batch"]
    assert laprust.lapjv.__module__ == "laprust._lapjv"
    assert list(inspect.signature(laprust.lapjv).parameters) == ["cost", "extend_cost", "cost_limit", "return_cost", "precision", "output"]
    assert list(inspect.signature(laprust.lapjv_batch).parameters) == ["costs", "extend_cost", "cost_limit", "return_cost", "n_threads", "precision", "output"]


def test_installed_typing():
    directory = Path(laprust.__file__).parent
    assert (directory / "py.typed").is_file()
    assert (directory / "_lapjv.pyi").is_file()
    assert (directory / "__init__.pyi").is_file()


def test_documented_examples():
    assert doctest.testmod(importlib.import_module("laprust._lapjv")).failed == 0


def test_no_other_solver_package_required():
    code = """
import importlib.abc
import sys
class BlockOtherPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('lap', 'lapcpp'):
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockOtherPackages())
import laprust
assert laprust.lapjv([[2., 1.], [1., 2.]])[0] == 2.
assert laprust.lapjv_batch([[[2., 1.], [1., 2.]]])[0].tolist() == [2.]
"""
    subprocess.run([sys.executable, "-c", code], check=True)
