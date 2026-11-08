# laprust

[![PyPI version](https://img.shields.io/pypi/v/laprust?logo=pypi&logoColor=white)](https://pypi.org/project/laprust/)
[![Python versions](https://img.shields.io/pypi/pyversions/laprust?logo=python&logoColor=gold)](https://pypi.org/project/laprust/)
[![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-gold)](https://pypi.org/project/laprust/#files)

[![Total downloads](https://static.pepy.tech/badge/laprust)](https://pepy.tech/projects/laprust)
[![Monthly downloads](https://static.pepy.tech/badge/laprust/month)](https://pepy.tech/projects/laprust)
[![License: MIT](https://img.shields.io/github/license/rathaROG/laprust)](https://github.com/rathaROG/laprust/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2610.35948-b31b1b)](https://arxiv.org/abs/2610.35948)

**laprust** solves dense linear assignment problems with a native
Rust implementation and a NumPy API. Python **3.9 or newer** is required.
It exposes two solvers: `lapjv` and `lapjv_batch`.

laprust solves rectangular problems directly. With a finite `cost_limit`, it
handles unmatched rows and columns implicitly. Neither operation allocates an
expanded square cost matrix.
The square path uses Jonker–Volgenant initialization, reduction, and shortest
augmenting paths. This package is independent of LAPX and imports as `laprust`.

The original laprust 1.0.0 release was derived from the `lapjvr` family in
[LAPX 0.11.1](https://github.com/rathaROG/lapx/tree/v0.11.1).
[lapcpp](https://github.com/rathaROG/lapcpp) leads native solver development for C++.
laprust leads native solver development for Rust. LAPX incorporates their fixes
and improvements.
See the [changelog](https://github.com/rathaROG/laprust/blob/main/CHANGELOG.md) for releases.

## Installation

Install from [PyPI](https://pypi.org/project/laprust/):

```sh
python -m pip install laprust
```

Prebuilt wheels cover Python 3.9–3.15 on Windows, Linux and macOS. See the
[available files](https://pypi.org/project/laprust/#files) for platform and
architecture coverage. NumPy is the only Python runtime dependency.

A source build of `laprust` requires a C++17 compiler and Rust/Cargo 1.83+ with a
compatible target. Pip installs the build dependencies (NumPy, Cython and setuptools)
automatically. Installing a compatible wheel requires no compiler or Rust
toolchain. See [contributing](https://github.com/rathaROG/laprust/blob/main/CONTRIBUTING.md) for source builds.

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
| `cost` / `costs` | These real numeric arrays have shape `(N, M)` / `(B, N, M)`. |
| `extend_cost=True` | This option permits rectangular inputs when the limit is infinite. The solver does not use square padding. |
| `cost_limit` | A finite value permits unmatched rows and columns. Each unmatched row or column incurs `cost_limit / 2`, rounded to the selected cost dtype. Finite limits also permit rectangular inputs and apply to square inputs. |
| `return_cost` | The default is `True`. Totals sum selected original costs in float64 and exclude rejection penalties. `False` omits totals and skips the sum. |
| `precision` | `"native"` preserves float32/float64. Other real numeric inputs use float64. Explicit `"float32"` / `"float64"` select the working cost dtype. |
| `output` | `"indices"` returns aligned row/column arrays. `"pairs"` returns `(K, 2)` arrays. |
| `n_threads` | This option applies only to batches. `0` / `None` selects the worker count automatically. Negative values use one worker. Positive values are capped by batch size. |

All solver paths use float64 potentials, distances and reduced-cost arithmetic,
while matrix storage follows `precision`. Finite-limit half-penalties retain
rounding in that selected cost dtype before search. Float64 working arithmetic
reduces cancellation with float32 costs without widening the whole matrix.
Extreme cost ranges and cost conversion can still obscure small objective
differences. Float64 working arithmetic does not guarantee exactness for
arbitrary cost ranges.

Rows are sorted and all returned indices are int64. They identify the selected
pairs. Unmatched rows and columns are absent. Batch results contain lists because
each problem may select a different number of pairs. Equal-cost optima may return
different assignments across solvers or precisions.

With a finite limit `L`, define `h = cost_dtype(L/2)`, where `cost_dtype` is
the cost storage dtype selected by `precision`. The optimized objective is
`sum(selected represented costs) + (N + M - 2*K) * h`.
Here, `N` and `M` are the input dimensions. `K` is the number of selected pairs.
Represented costs are the input values after conversion to that cost dtype.

The reported `total` separately sums the selected **original-input values** in
float64 and excludes rejection penalties. It can differ from the represented-cost
sum when conversion rounds input values. Joint rejection is different from
solving a full assignment and then filtering high-cost pairs.
See [API details](https://github.com/rathaROG/laprust/blob/main/docs/api.md) for numerical and memory behavior.

## Development and checks

```sh
python -m pip install -e ".[dev]"
cd tests
python -m pytest -q -m "not cross_package"
cd ..
python -m pytest -q .github/tests -m "not editable"
python -m build
```

The test suite covers these areas:

- Comparisons use SciPy and exhaustive references.
- Solver cases include rectangular and finite-limit problems, float32/float64,
  precision conversion, forbidden edges, empty inputs, and invalid arguments.
- Execution checks cover repeated workspaces, batch thread counts, concurrent
  calls, and cross-package concurrency.
- Package checks cover independence, type annotations, and isolated editable
  installs.
- Regression checks include historical LAPX cases.

See the [solver test inventory](https://github.com/rathaROG/laprust/blob/main/tests/README.md),
[build and packaging checks](https://github.com/rathaROG/laprust/blob/main/.github/tests/README.md), and
[CONTRIBUTING.md](https://github.com/rathaROG/laprust/blob/main/CONTRIBUTING.md) for commands and coverage.

## Benchmarks

All three benchmark scripts compare **SciPy (baseline), lapcpp, laprust,
LAPX lapjvxa and LAPX lapjvsa**. Every mode requires **LAPX 0.13.0 or newer**.
The LAPX candidates represent the JVX and JVS families using pair outputs,
including their batch APIs. Clone or download this repository for the scripts.
Run the scripts from the repository root with the released packages:

```sh
python -m pip install --upgrade lapcpp laprust "lapx>=0.13.0" scipy packaging
python benchmarks/benchmark_single.py --quick --repeats 3
python benchmarks/benchmark_batch.py --quick --threads 4 --repeats 3
python benchmarks/benchmark_tracking.py --quick --repeats 3
```

To benchmark this checkout, install `python -m pip install -e ".[benchmark]"`
instead. This command installs all comparison packages with the same LAPX
minimum version.

Use `--json benchmark-results/run.json` to save measurements and environment
details. Each solver receives a warm-up call immediately before its own sample.
Calibration chooses a repeat count that targets at least 10 ms per sample.
Measured durations can vary. `--reference-timing` uses the LAPX 0.13.0 benchmark
timing protocol.

See [benchmark methodology](https://github.com/rathaROG/laprust/blob/main/benchmarks/README.md) for full workloads,
reference LAPX settings, dedicated solver functions, tracking post-filtering, and
timing boundaries. Tracking never passes its threshold as solver `cost_limit`.

These package benchmarks use uniform random costs and JVS's default float32
working precision. The paper also covers structured, geometric, finite-limit,
memory and application experiments. To reproduce those experiments, use the
separate [paper benchmark guide](https://github.com/rathaROG/lap-paper/blob/main/benchmark/README.md).
That guide specifies its own workloads, precision settings and pinned environment.

## Paper

The paper *Native Rectangular Linear Assignment with Implicit Rejection:
A Jonker–Volgenant Solver without Square Expansion* describes the solver and
its implicit rejection. It compares performance against SciPy and other
Jonker–Volgenant implementations. The paper labels laprust **JVRUST**.

Read the [current repository PDF](https://github.com/rathaROG/lap-paper/blob/main/output/pdf/native_rectangular_assignment.pdf)
or the [arXiv preprint](https://arxiv.org/abs/2610.35948).
The revised repository manuscript evaluates lapcpp/laprust **1.2.0** and
LAPX **0.13.0** across four Windows/Linux and Python environments.
The [lap-paper repository](https://github.com/rathaROG/lap-paper) includes the
scripts, [frozen evidence](https://github.com/rathaROG/lap-paper/blob/main/benchmark/evidence/README.md)
and reproduction steps. Its
[artifact guide](https://github.com/rathaROG/lap-paper/blob/main/benchmark/paper-artifacts/README.md)
explains how to regenerate figures and tables from the saved results.

## Automation and license

GitHub workflows test supported Python versions on Windows, Linux and macOS.
The workflows also build and test source archives and wheels. Separate
workflows run the three benchmark modes.
Wheel builds are also available manually. A version tag triggers the release
workflow. PyPI publishing uses the repository's `pypi` environment and requires
its Trusted Publisher configuration. Local builds do not publish releases.

laprust uses the MIT license. See [LICENSE](https://github.com/rathaROG/laprust/blob/main/LICENSE),
[NOTICE](https://github.com/rathaROG/laprust/blob/main/NOTICE), and
[native-source license](https://github.com/rathaROG/laprust/blob/main/src/_lapjv/LICENSE) for attribution.
