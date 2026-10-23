# Development

Use Python 3.9+ and a C++17 compiler (MSVC Build Tools on Windows, GCC or Clang
on Linux, Xcode Command Line Tools on macOS). `laprust` additionally needs Rust
1.83+ and Cargo; stable Rust is used in CI.

Windows builds use MSVC's `/std:c++17` language mode for the C++ wrapper,
as lapcpp does. Linux and macOS builds use `-std=c++17`; the Rust core uses
the edition declared in `Cargo.toml`.

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -e ".[dev]"
cd tests
python -m pytest -q -m "not cross_package"
cd ..
python -m pytest -q .github/tests -m "not editable"
python -m pyright tests/typing/return_types.py
python -m build
python .github/tests/check_dist.py
```

`python -m build` builds the wheel from the source archive by default. To verify
an installed wheel, uninstall any editable installation, install the wheel in a
fresh environment, and run `python .github/tests/check_wheel.py`. Run the solver tests
from the `tests` directory to avoid importing the source package accidentally.

For `laprust`, also run:

```sh
cargo fmt --check --manifest-path src/_lapjv/Cargo.toml
cargo test --locked --all-features --manifest-path src/_lapjv/Cargo.toml
cargo clippy --locked --all-targets --all-features --manifest-path src/_lapjv/Cargo.toml -- -D warnings
```

The Cython wrapper lives under `src/_lapjv`; generated `_lapjv.cpp`, binaries,
Cargo targets, build directories and benchmark output are ignored. Preserve
the source{d}/lapjv and LAPX license notices when changing native source.

Build controls use the `LAPRUST_` prefix:

- `LAPRUST_BASEOPTS=0` disables added base optimizations and LTO.
- `LAPRUST_LTO=0` disables LTO, including inherited compiler defaults.
- `LAPRUST_NATIVE=1` enables host CPU tuning where supported; leave it unset for portable wheels.

Strict floating-point behavior is always retained. There is no fast-math option.
Rust target architecture must match Python's extension architecture. macOS
builds use one architecture at a time; universal2 is not supported for Rust.

## Solver development and LAPX

laprust 1.0.0 derives from LAPX 0.11.1's `lapjvr` family. lapcpp and laprust
are the lead repositories for C++ and Rust solver development, respectively.

Develop solver fixes, improvements, and regression tests in the corresponding
standalone repository first. Keep the common numerical and API contracts aligned
between lapcpp and laprust, then port relevant changes into LAPX's corresponding
solver families. Preserve the LAPX and original upstream attribution in `NOTICE`.

## Workflows

- `tests.yaml`: source-to-wheel builds, separate solver and build/tool tests, typing, a SciPy-free
  core run, and older NumPy runtime compatibility.
- `native.yaml`: native resource/failure tests, Cython exception translation,
  and Linux address/undefined-behavior/thread sanitizer checks.
- `integration.yaml`: required cross-package concurrency and isolated editable
  installs (default/strict/compat), on Windows/Linux/macOS and Python 3.9/3.15.
- `prepublish.yaml`: manual cibuildwheel smoke builds for CPython 3.9 and 3.15;
  select `full` to build all Python versions (3.9-3.15). Reusable calls default
  to the full set. Builds cover
  Windows AMD64/ARM64, Linux manylinux x86_64/aarch64/armv7l, Linux musllinux
  x86_64/aarch64, and macOS x86_64/arm64, for combinations supported by
  cibuildwheel. musllinux armv7l remains excluded.
- `publish.yaml`: tags `v*`, version check, the full wheel matrix with installed-wheel
  smoke checks, then PyPI Trusted Publishing, matching LAPX. Integration,
  native and sanitizer suites run separately on PRs or manual dispatch.
- `benchmark_single.yaml`, `benchmark_batch.yaml`, `benchmark_tracking.yaml`:
  manual runs of the five-candidate comparison, with JSON artifacts.

Set up the PyPI Trusted Publisher for this repository, `publish.yaml`, environment
`pypi`, before publishing. Tag `v1.0.0` only when releasing version 1.0.0. The
manual prepublish workflow creates artifacts without uploading them to PyPI.
Benchmark workflows accept counterpart and LAPX repository refs. LAPX defaults
to the `v0.11.1` release; record all
three revisions when reporting comparisons. For unreleased local changes,
install the two standalone checkouts and LAPX 0.11.1 from PyPI, as described in
[benchmark methodology](benchmarks/README.md).

ARMv7 manylinux builds on Python 3.12-3.15 use
`.github/constraints/numpy-armv7-build.txt` for both build and test dependencies,
matching LAPX's GCC 9-compatible NumPy range (`>=2,<2.5`). The override retains
the Rust/Cargo path and does not change the installed package's NumPy requirement.
Every runnable wheel target uses the NumPy-only installed-wheel smoke check in
`.github/tests/check_wheel.py`; full solver and build/tool suites run in
`tests.yaml`. musllinux armv7l is excluded because the current native Rust
toolchain setup cannot build that target, matching LAPX's Rust exclusion.

Wheel jobs separate Windows architectures, manylinux/musllinux targets,
and macOS architectures. Full builds also split Python 3.9-3.11 and
3.12-3.15 into parallel groups, so each job builds at most four wheels.
Windows and macOS reuse cached cibuildwheel downloads between runs.
Release compiler optimizations and platform coverage are unchanged.

See the [solver coverage inventory](tests/README.md) for the LAPX case mapping
and cross-package checks, and [build and packaging checks](.github/tests/README.md)
for isolated installation checks, benchmark-tool tests, and CI scope.

See [native failure and sanitizer checks](tests/README.md#portable-and-native-failure-checks)
for local commands and test-only fault injection details.
