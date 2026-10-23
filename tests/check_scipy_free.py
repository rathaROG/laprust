"""Run the portable core suite with SciPy imports forbidden, even if installed."""
import importlib.abc
import os
from pathlib import Path
import sys


class NoScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "scipy":
            raise AssertionError("The portable core tests must not depend on SciPy")


if __name__ == "__main__":
    assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
    sys.meta_path.insert(0, NoScipy())
    import pytest
    directory = Path(__file__).resolve().parent
    os.chdir(directory)  # Subprocess probes must import the installed wheel too.
    raise SystemExit(pytest.main(["-q", *[str(directory / filename) for filename in (
        "test_core.py", "test_edge_cases.py", "test_batch_paths.py", "test_package.py")]]))
