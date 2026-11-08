# Build, packaging, and benchmark-tool checks

Solver behavior, native error handling, cross-package solver comparisons, and
public API typing checks are in [tests/](../../tests/README.md).
This directory contains build configuration, installation, distribution, and
benchmark-tool checks. It has its own pytest configuration and dependencies.

## Routine checks

Run from the repository root:

```sh
python -m pip install -r .github/tests/requirements.txt
python -m pytest -q .github/tests -m "not editable"
```

- `test_setup.py` checks compiler flags and build configuration without a native compiler.
- `test_benchmarks.py` checks benchmark inputs, API calls, reference results,
  timing boundaries, and ranking output using stand-ins for the installed solvers.
  It needs NumPy and SciPy, but no installed solver package.

For platforms without SciPy wheels, run only the compiler-option tests here:

```sh
python -m pytest -q .github/tests/test_setup.py
```

The portable solver suite runs separately with `python tests/check_scipy_free.py`.

## Distributions and runtime compatibility

1. Build the source archive and wheel.
2. Install the wheel.
3. Run these checks from the repository root:

```sh
python .github/tests/check_dist.py
python .github/tests/check_wheel.py
```

`check_dist.py` verifies metadata, namespaces, licenses, and source-archive
contents. Source archives include the solver suite and these CI checks.
Full CI runs can therefore validate wheels built from source archives.
`check_wheel.py` checks the installed package and rejects imports from the source
checkout. With NumPy and the wheel installed, run the command with the path shown above.
`install_numpy_compat.py` installs the NumPy/SciPy versions selected by CI and
checks the built wheel against that runtime.

## Isolated editable installs

These checks require access to the package index, a native C++17 compiler, Rust, and Cargo.
Run from the repository root:

```sh
python -m pytest -q .github/tests/test_editable_install.py --run-editable
```

These checks copy the current source into a temporary checkout.
They create a fresh virtual environment without system site packages.
They check default, strict, and compatibility modes, visible source edits,
native imports, standalone dependencies, type stubs, wheel replacement, and
editable reinstallation. They do not modify the working checkout or the
environment from which you run them.
Without `--run-editable`, pytest skips this group.

The [tests workflow](../workflows/tests.yaml) runs solver and build/tool checks
in separate steps, including the older-NumPy jobs. The
[integration workflow](../workflows/integration.yaml) runs the editable group
separately from cross-package solver checks.

Publish and prepublish run only
the NumPy-only installed-wheel smoke check for each cibuildwheel target.
Manual prepublish runs default to Python 3.9 and 3.15, matching LAPX.
To build every supported Python version, select `full`.
Publishing always requests
the full Python 3.9-3.15 wheel set. Release builds run in two Python groups,
with separate jobs for each architecture and Linux wheel type.

Integration and native/sanitizer workflows run on pull requests or manual
dispatch, separately from publishing, matching LAPX's release workflow.
