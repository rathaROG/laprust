# Solver test coverage

Run tests from `tests/` so an installed wheel is not shadowed by the source
package. Install `.[test]` first, or install a wheel plus NumPy, SciPy, pytest,
and pytest-timeout.

```sh
cd tests
python -m pytest -q -m "not cross_package"
```

Cross-package solver checks are an explicit opt-in because they require both
packages. Install both sibling repositories before running them from `tests/`:

```sh
python -m pip install ../ ../../lapcpp
python -m pytest -q test_cross_package.py --run-cross-package
# Run the complete solver suite, including cross-package checks:
python -m pytest -q --run-cross-package
```

An explicitly requested cross-package run fails if a package is missing.
It never silently skips a missing solver.

Build configuration, isolated editable installs, distribution checks, and
benchmark-tool tests live in [.github/tests/](../.github/tests/README.md).
CI runs those checks separately on pull requests or manual dispatch.

## Coverage relative to LAPX

| LAPX coverage | Coverage here |
| --- | --- |
| `test_lapjvr.py` and its fixtures | All 37 test functions (600 parameterized cases) retained in `test_lapjv.py` / `lapjv_cases.py`, with the package/API names adapted. |
| Historical `test_lapjv.py` fixtures | `test_regressions.py`: augmentation/TrackMate cases, sliced/noncontiguous inputs, rectangles and explicit padding, finite limits, isolated forbidden-edge matches, epsilon data, seeded known optima at sizes 100, 1000 and 4000. |
| `test_arr_loop.py`, reported LAPX/upstream issues | Original dense ARR-loop, negative-cost, infinity and million-cost blocking cases, both orientations and precisions, single and batch. |
| `test_large_cost_consistency.py` | Fixed billion-cost optima, integer and floating input, rectangular orientations, both result layouts, totals on/off, batches and threads, forbidden edges, signed large costs and finite limits. |
| Input/native/wrapper stability and dense smoke tests | `test_lapjv.py` + `test_edge_cases.py`: independent exhaustive and SciPy objectives, required/rejected matching, signatures, invalid inputs/options, empty inputs, original-value totals, precision boundaries, readonly/strided/unaligned/endian/broadcast inputs, result-buffer independence/lifetime, overflow, deterministic failures and recovery. |
| Batch/thread tests | Native worker-count normalization, changing automatic CPU counts, sequential/concurrent calls, multiple pools and sizes, ordering, input preservation and failure recovery in `test_lapjv.py`. `test_batch_paths.py`: large batches of small problems split across workers, a failure among them reporting the lowest index, and result construction at batch sizes 7–33 with required, fully matched and partly rejected limits, all against separate single solves. |
| `test_lapjv_rh.py` | `test_cross_package.py`: both actual packages, repeated-row completion cases, original seeded C++/Rust equivalence cases, alternating and concurrent single/batch calls, errors isolated across packages, both import orders and interpreter shutdown. |
| Public API, package independence, and typing | `test_package.py`, `typing/return_types.py`. |
| Build and installation checks | See [.github/tests/](../.github/tests/README.md). |
| Benchmark reference correctness | [.github/tests/test_benchmarks.py](../.github/tests/test_benchmarks.py): dedicated five-candidate API contracts, original JVS precision, tracking post-filtering, assignment/objective/total checks, timing boundaries, calibrated calls with per-solver warm-up, pinned input seeds, workload bounds, verdict thresholds and aggregate ranking. |

These packages expose only `lapjv` and `lapjv_batch`. Sparse `lapmod` formats,
legacy mapping outputs, other LAPX algorithms and their private helpers do
not apply. Infeasible required assignments raise `ValueError` here, rather
than returning the infinite totals of legacy LAPX algorithms. Cost matrices
containing NaN or negative infinity are outside the documented contract;
invalid `cost_limit` values are tested and rejected.

Numerical checks use exhaustive enumeration, SciPy, and historical known
optima as independent references. Single/batch or cross-package agreement
alone is not evidence of optimality. Tied optimum assignments are compared
by objective unless exact equality is the behavior being tested. Tests do
not assert solver timing. This is a functional coverage inventory, not a
claim of measured 100% line or branch coverage.

## CI

- `tests.yaml` builds a wheel from the source archive and runs solver and
  build/tool checks in separate steps on Python 3.9-3.15 across Windows, Linux and macOS, x86-64 and ARM64
  where the Python/runner combination is supported. Windows ARM64 starts at
  Python 3.11. It also checks typing, distribution contents, and runs the
  portable core suite with SciPy imports explicitly forbidden.
- NumPy compatibility jobs test NumPy-2-built wheels against NumPy 1.21.6 on
  Python 3.9 and NumPy 1.26.4 on Python 3.11.
- `integration.yaml` runs cross-package and isolated editable checks on
  Windows, Linux and macOS, using Python 3.9 and 3.15. The counterpart source
  ref is configurable for manual/reusable runs and both revisions are logged.
- Wheel builds cover Windows AMD64/ARM64, macOS x86_64/arm64, Linux manylinux
  x86_64/aarch64/armv7l, and Linux musllinux x86_64/aarch64, where supported by
  cibuildwheel. Each wheel runs the NumPy-only installed-wheel smoke check in
  `.github/tests/check_wheel.py`, matching LAPX's lightweight wheel validation.
  The full solver, portable, and build/tool suites run separately in `tests.yaml`.
  Manual prepublish defaults to Python 3.9 and 3.15, matching LAPX. Select
  `full` for a complete rehearsal; publishing always builds Python 3.9-3.15.
  Architectures and Linux wheel types run in separate jobs. Full builds split
  Python 3.9-3.11 and 3.12-3.15 into parallel groups, with unique artifacts.
  Windows and macOS cache downloaded cibuildwheel tools between runs.
  ARMv7 manylinux builds on Python 3.12-3.15 constrain NumPy to `>=2,<2.5`
  during both builds and tests for compatibility with the GCC 9 image.
  `publish.yaml` requires the version check and all wheel builds/smoke checks
  to succeed, then publishes the wheels and source archive. Integration,
  native checks and sanitizers run in their separate PR/manual workflows.
  musllinux armv7l remains excluded for laprust.


## Portable and native failure checks

`python tests/check_scipy_free.py` exercises required/optional square and
rectangular assignments against exhaustive enumeration and an independent
bitmask reference. It covers precisions, return layouts, forbidden edges,
concurrent batches, recovery, empty results with/without totals, and batch byte
size guards. It blocks SciPy imports even when SciPy is installed. On 32-bit
hosts only the two tests requiring dimensions above NumPy's own int32 limit
are skipped; byte-size overflow checks still run.

`python tests/check_native_errors.py` compiles the actual Cython adapter against
a test-only C ABI in a temporary directory. It checks native infeasibility,
allocation, internal, dimension and unknown status codes, Python exception types,
batch error indices, both precisions/output layouts/totals modes, and recovery.
It needs NumPy, Cython, setuptools and a C++ compiler. This stub never enters an
installed wheel. The native core tests separately inject workspace allocation,
worker startup and internal failures and verify recovery. Fault injection is
compiled out of ordinary wheel builds; it does not exhaust host resources.

`native.yaml` runs native tests and the adapter checks on Windows, Linux and
macOS, x86-64 and ARM64. Linux sanitizer jobs run AddressSanitizer (plus
UndefinedBehaviorSanitizer for C++) and ThreadSanitizer separately; any finding
fails the job. The Rust jobs instrument the core, dependencies and standard
library using nightly Rust and `rust-src`. Sanitizers exercise the native test
executables, not the Python interpreter. The sanitizer configuration follows
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
