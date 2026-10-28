# laprust

**laprust** solves dense linear assignment problems with a native
Rust implementation and a NumPy API. Python **3.9 or newer** is required.
It exposes two solvers: `lapjv` and `lapjv_batch`.

laprust 1.0.0 derives from the `lapjvr` solver in LAPX 0.11.1.
Future solver development is led by lapcpp for C++ and laprust for Rust;
LAPX will incorporate their fixes and improvements.

It solves rectangular problems directly and handles unmatched rows and columns
implicitly when a finite `cost_limit` is supplied, without allocating an expanded
square cost matrix.
The square path uses Jonker–Volgenant initialization, reduction, and shortest
augmenting paths. This package is independent of LAPX and imports as `laprust`.

## Installation

Once available on PyPI:

```sh
python -m pip install laprust
```

A source build requires a C++17 compiler and, for `laprust`, Rust/Cargo 1.83+
with a compatible target. Build dependencies (NumPy, Cython, setuptools) are
installed automatically by pip. NumPy is the only Python runtime dependency.
An installed wheel needs no compiler or Rust toolchain.

## Quick start

```python
import numpy as np
import laprust

cost = np.array([[3., 1.], [1., 4.]])
total, rows, cols = laprust.lapjv(cost)
# total = 2.0; rows = [0, 1]; cols = [1, 0]

costs = np.stack([cost, cost])  # (batch, rows, columns)
totals, rows, cols = laprust.lapjv_batch(costs, n_threads=2)
# totals = [2., 2.]; rows and cols are lists of arrays, one per problem

rectangular = np.array([[.1, .8, .9], [.9, .7, .8]], dtype=np.float32)
total, rows, cols = laprust.lapjv(rectangular, cost_limit=.5)
# Only (0, 0) is selected; total is about .1, excluding rejection penalties.
```

## API

```python
lapjv(cost, extend_cost=False, cost_limit=np.inf, return_cost=True,
      *, precision="native", output="indices")
lapjv_batch(costs, extend_cost=False, cost_limit=np.inf, return_cost=True,
            n_threads=0, *, precision="native", output="indices")
```

| Option | Behavior |
| --- | --- |
| `cost` / `costs` | Real numeric arrays with shape `(N, M)` / `(B, N, M)`. |
| `extend_cost=True` | Allows rectangular inputs when the limit is infinite; no square padding is used. |
| `cost_limit` | A finite value permits unmatched endpoints, each charged `cost_limit / 2` rounded to the selected cost dtype. Finite limits also permit rectangles and apply to square inputs. |
| `return_cost` | Defaults to `True`; totals sum selected original costs in float64 and exclude rejection penalties. `False` omits totals and skips the sum. |
| `precision` | `"native"` preserves float32/float64; other real numeric inputs use float64. Explicit `"float32"` / `"float64"` select working costs. |
| `output` | `"indices"` returns aligned row/column arrays; `"pairs"` returns `(K, 2)` arrays. |
| `n_threads` | Batch only. `0` / `None` selects automatically; negative values use one worker; positive values are capped by batch size. |

Rows are sorted and all returned indices are int64. They identify the selected
pairs; unmatched items are absent. Batch results contain lists because each
problem may select a different number of pairs. Equal-cost optima may return
different assignments across solvers or precisions.

All solver paths retain the selected float32 or float64 cost storage and use
float64 potentials, distances, and reduced-cost arithmetic. Finite-limit
half-penalties are still rounded in the selected cost dtype. This improves
robustness when small assignment differences coexist with large finite costs;
extreme dynamic ranges and input conversion can still hide differences.

With a finite limit `L`, define `h = cost_dtype(L/2)`, where `cost_dtype` is
the cost storage dtype selected by `precision`. The optimized objective is
`sum(selected represented costs) + (N + M - 2*K) * h`, where `N` and `M`
are the input dimensions and `K` is the number of selected pairs.
Represented costs are the input values after conversion to that cost dtype.

The reported `total` separately sums the selected **original-input values** in
float64 and excludes rejection penalties. It can differ from the represented-cost
sum when conversion rounds input values. Joint rejection is different from
solving a full assignment and then filtering high-cost pairs.
See [API details](docs/api.md) for numerical and memory behavior.

## Development and checks

```sh
python -m pip install -e ".[dev]"
cd tests
python -m pytest -q -m "not cross_package"
cd ..
python -m pytest -q .github/tests -m "not editable"
python -m build
```

The test suite covers SciPy and exhaustive-reference comparisons, rectangular
and finite-limit cases, float32/float64, precision conversion, forbidden edges,
empty inputs, invalid arguments, repeated workspaces, batch thread counts,
concurrent calls, package independence, and type annotations.
Historical LAPX regressions, cross-package concurrency, and isolated editable
installs are also covered. See the [solver test inventory](tests/README.md),
[build and packaging checks](.github/tests/README.md), and
[CONTRIBUTING.md](CONTRIBUTING.md) for commands and coverage.

## Benchmarks

All three benchmark scripts compare **SciPy (baseline), lapcpp, laprust,
LAPX lapjvxa and LAPX lapjvsa**. The LAPX candidates represent the JVX and JVS
families using pair outputs, including their batch APIs. Once lapcpp and laprust
are available on PyPI, install the dependencies below. Clone or
download this repository for the scripts and run them from the repository root.

```sh
python -m pip install --upgrade lapcpp laprust "lapx==0.11.1" scipy
python benchmarks/benchmark_single.py --quick --repeats 3
python benchmarks/benchmark_batch.py --quick --threads 4 --repeats 3
python benchmarks/benchmark_tracking.py --quick --repeats 3
```

Use `--json benchmark-results/run.json` to save measurements and environment
details. Each solver is warmed right before its own sample, and calls repeat
until a sample lasts at least 10 ms; `--reference-timing` restores the pinned
LAPX timing. See [benchmark methodology](benchmarks/README.md) for full workloads,
reference LAPX settings, dedicated solver functions, tracking post-filtering, and
timing boundaries. Tracking never passes its threshold as solver `cost_limit`.

## Paper

The solver, its implicit rejection, and its performance against SciPy and other
Jonker–Volgenant implementations are described in *Native Rectangular Linear
Assignment with Implicit Rejection: A Jonker–Volgenant Solver without Square
Expansion*, where laprust is labeled JVRUST. Read the
[PDF](https://github.com/rathaROG/lap-paper/blob/main/output/pdf/native_rectangular_assignment.pdf);
its scripts, raw results, and reproduction steps are in the
[lap-paper](https://github.com/rathaROG/lap-paper) repository.

## Automation and license

GitHub workflows test supported Python versions on Windows, Linux and macOS,
build and test source archives and wheels, and run the three benchmark modes.
Wheel builds are also available manually. A version tag triggers the release
workflow; PyPI publishing uses the repository's `pypi` environment and requires
its Trusted Publisher configuration. No releases are published by local builds.

MIT licensed. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[native-source license](src/_lapjv/LICENSE) for attribution.
