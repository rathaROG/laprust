# Solver test coverage

First, install `.[test]` or a wheel plus NumPy, SciPy, pytest, and pytest-timeout.
Run tests from `tests/` so Python imports the installed wheel instead of the
source package.

```sh
cd tests
python -m pytest -q -m "not cross_package"
```

Cross-package solver checks require an explicit opt-in because they need both
packages. Before running these checks, install both sibling repositories.
Run these commands from `tests/`:

```sh
python -m pip install ../ ../../lapcpp
python -m pytest -q test_cross_package.py --run-cross-package
# Run the complete solver suite, including cross-package checks:
python -m pytest -q --run-cross-package
```

An explicitly requested cross-package run fails if a package is missing.
It never silently skips a missing solver.

Build configuration, isolated editable installs, distribution checks, and
benchmark-tool tests are in [.github/tests/](../.github/tests/README.md).
CI runs those checks separately on pull requests or manual dispatch.

## Coverage relative to LAPX

| LAPX coverage | Coverage here |
| --- | --- |
| `test_lapjvr.py` and its fixtures | `test_lapjv.py` / `lapjv_cases.py` retain all 37 test functions (600 parameterized cases). Package/API names are adapted. |
| Historical `test_lapjv.py` fixtures | `test_regressions.py` covers augmentation/TrackMate cases, sliced/noncontiguous inputs, rectangles and explicit padding, finite limits, isolated forbidden-edge matches, and epsilon data. It also checks seeded known optima at sizes 100, 1000 and 4000. |
| `test_arr_loop.py`, reported LAPX/upstream issues | Tests retain the original dense ARR-loop, negative-cost, infinity and million-cost blocking cases. They cover both orientations and precisions, for single and batch calls. |
| `test_large_cost_consistency.py` | Tests check fixed billion-cost optima with integer and floating input, rectangular orientations, both result layouts, and totals on/off. They cover batches and threads, forbidden edges, signed large costs and finite limits. |
| Input/native/wrapper stability and dense smoke tests | `test_lapjv.py` + `test_edge_cases.py` use independent exhaustive and SciPy objectives. They cover required/rejected matching, signatures, invalid inputs/options, empty inputs, original-value totals and precision boundaries. They check readonly/strided/unaligned/endian/broadcast inputs, result-buffer independence/lifetime, overflow, deterministic failures and recovery. |
| Batch/thread tests | `test_lapjv.py` checks native worker-count normalization, changing automatic CPU counts, sequential/concurrent calls, multiple pools and sizes, ordering, input preservation and failure recovery. `test_batch_paths.py` checks large batches of small problems split across workers. A failure among those problems must report the lowest index. Result construction covers batch sizes 7–33 with required, fully matched and partly rejected limits. All results are checked against separate single solves. |
| `test_lapjv_rh.py` | `test_cross_package.py` checks both actual packages, repeated-row completion cases, and the original seeded C++/Rust equivalence cases. It covers alternating and concurrent single/batch calls, errors isolated across packages, both import orders and interpreter shutdown. |
| Public API, package independence, and typing | `test_package.py` and `typing/return_types.py` check these properties. |
| Build and installation checks | See [.github/tests/](../.github/tests/README.md). |
| Benchmark reference correctness | [.github/tests/test_benchmarks.py](../.github/tests/test_benchmarks.py) checks dedicated five-candidate API contracts, original JVS precision, tracking post-filtering, assignment/objective/total checks, timing boundaries, and calibrated calls with per-solver warm-up. It also checks pinned input seeds, workload bounds, verdict thresholds and aggregate ranking. |

These packages expose only `lapjv` and `lapjv_batch`. Sparse `lapmod` formats,
legacy mapping outputs, other LAPX algorithms and their private helpers do
not apply. Infeasible required assignments raise `ValueError` here, rather
than returning the infinite totals of legacy LAPX algorithms. Cost matrices
containing NaN or negative infinity are outside the documented contract.
Tests verify that the solver rejects invalid `cost_limit` values.

Numerical checks use exhaustive enumeration, SciPy, and historical known
optima as independent references. Single/batch or cross-package agreement
alone is not evidence of optimality. Tied optimum assignments are compared
by objective unless exact equality is the behavior being tested. Tests do
not assert solver timing. This is a functional coverage inventory, not a
claim of measured 100% line or branch coverage.

## CI

- `tests.yaml` builds a wheel from the source archive.
  It runs solver and build/tool checks in separate steps on Python 3.9-3.15.
  The workflow covers Windows, Linux and macOS, x86-64 and ARM64, where the
  Python/runner combination is supported. Windows ARM64 starts at Python 3.11.
  The workflow also checks typing and distribution contents.
  It runs the portable core suite with SciPy imports explicitly forbidden.
- NumPy compatibility jobs test NumPy-2-built wheels against NumPy 1.21.6 on
  Python 3.9 and NumPy 1.26.4 on Python 3.11.
- `integration.yaml` runs cross-package and isolated editable checks on
  Windows, Linux and macOS, using Python 3.9 and 3.15. The counterpart source
  ref is configurable for manual/reusable runs. The workflow logs both revisions.
- Wheel builds cover Windows AMD64/ARM64, macOS x86_64/arm64, Linux manylinux
  x86_64/aarch64/armv7l, and Linux musllinux x86_64/aarch64, where supported by
  cibuildwheel. Each wheel runs the NumPy-only installed-wheel smoke check in
  `.github/tests/check_wheel.py`, matching LAPX's lightweight wheel validation.
  The full solver, portable, and build/tool suites run separately in `tests.yaml`.

  Manual prepublish defaults to Python 3.9 and 3.15, matching LAPX. Select
  `full` for a complete rehearsal. Publishing always builds Python 3.9-3.15.
  Architectures and Linux wheel types run in separate jobs. Full builds split
  Python 3.9-3.11 and 3.12-3.15 into parallel groups, with unique artifacts.
  Windows and macOS cache downloaded cibuildwheel tools between runs.

  ARMv7 manylinux builds on Python 3.12-3.15 constrain NumPy to `>=2,<2.5`
  during both builds and tests for compatibility with the GCC 9 image.
  `publish.yaml` requires the version check and all wheel builds/smoke checks
  to succeed. It then publishes the wheels and source archive. Integration,
  native checks and sanitizers run in their separate PR/manual workflows.
  musllinux armv7l remains excluded for laprust.


## Portable and native failure checks

`python tests/check_scipy_free.py` exercises required/optional square and
rectangular assignments against exhaustive enumeration and an independent
bitmask reference. It covers precisions, return layouts, forbidden edges,
concurrent batches, recovery, empty results with/without totals, and batch byte
size guards. It blocks SciPy imports even when SciPy is installed. On 32-bit
hosts, the suite skips only the two tests requiring dimensions above NumPy's own
int32 limit. Byte-size overflow checks still run.

`python tests/check_native_errors.py` compiles the actual Cython adapter against
a test-only C ABI in a temporary directory. It checks native infeasibility,
allocation, internal, dimension and unknown status codes.
It also checks Python exception types, batch error indices, both
precisions/output layouts/totals modes, and recovery.
It needs NumPy, Cython, setuptools and a C++ compiler.
This stub never enters an installed wheel.

The native core tests separately inject workspace allocation,
worker startup and internal failures and verify recovery. Fault injection is
excluded from ordinary wheel builds. It does not exhaust host resources.

`native.yaml` runs native tests and the adapter checks on Windows, Linux and
macOS, x86-64 and ARM64. Linux sanitizer jobs run AddressSanitizer and
ThreadSanitizer separately. Any finding fails the job. The Rust jobs instrument
the core, dependencies and standard library with nightly Rust and `rust-src`.
Sanitizers exercise native test executables, not the Python interpreter.
The sanitizer configuration follows
[Clang's documentation](https://clang.llvm.org/docs/AddressSanitizer.html) and
[the Rust sanitizer guide](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html).

The historical fixture data/builders retain their original BSD-2-Clause
license in [LICENSE](LICENSE). In particular, `cost_eps.csv.gz` is the original
epsilon-regression dataset, included in source distributions.

Run the Rust native tests from the repository root:

```sh
cargo test --locked --manifest-path src/_lapjv/Cargo.toml
cargo test --locked --all-features --manifest-path src/_lapjv/Cargo.toml
```

The C ABI tests also verify panic containment in serial/parallel calls, the
lowest infeasible batch index, and dimension rejection before pointer access.
See `native.yaml` for the Linux nightly sanitizer commands.

## Float32 working-arithmetic regressions

`test_float32_working_precision.py` checks tiny exact optima beside large costs
in rectangular required and finite-limit solves. It covers both orientations,
precision selections (including genuine float64 controls), indices/pairs,
totals on/off, readonly/strided batches, worker counts 1/2, and a fixture that
requires augmenting search. Independent Fraction enumeration checks represented
costs and rounded half-penalties. Tests check returned totals separately on the
original input.

Quantization-sensitive fixtures ensure widening does not change
the public cost-conversion contract. These are correctness checks, not timings.

The helpers in `test_lapjv.py` also model selected cost precision and rounded
half-limits explicitly. Reduced-cost overflow beyond float32 is now solvable
with float64 working state. Tests retain error/recovery coverage for genuine
float64 overflow. Widening is not an arbitrary-range exactness guarantee.
