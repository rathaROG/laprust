# Five-candidate benchmarks

See the [initialization policy](initialization-policy.md) for the 1.2.0
reduction rules and performance tradeoffs.

Compare single assignments, native batches and tracking against SciPy. Each
candidate has a dedicated solver call. There are no fallback solvers.

| Candidate | Single / tracking API | Batch API | Working precision |
| --- | --- | --- | --- |
| SciPy (baseline) | `scipy.optimize.linear_sum_assignment` | Serial loop | float64 |
| lapcpp | `lapcpp.lapjv` | `lapcpp.lapjv_batch` | Input dtype |
| laprust | `laprust.lapjv` | `laprust.lapjv_batch` | Input dtype |
| LAPX JVX | `lap.lapjvxa` | `lap.lapjvxa_batch` | float64 |
| LAPX JVS | `lap.lapjvsa` | `lap.lapjvsa_batch` | Default float32 |

The candidates use different working precisions. JVS uses float32 even with
float64 inputs. The LAPX candidates include only the XA/SA variants. The
benchmarks do not force JVS to use float64.

## Installation and commands

Clone or download [laprust](https://github.com/rathaROG/laprust) to get the benchmark
scripts. Run these commands from the repository root to install the published
packages and run the benchmarks. The `lapx` package imports as `lap`.

```sh
python -m pip install --upgrade lapcpp laprust "lapx>=0.13.0" scipy packaging
python benchmarks/benchmark_single.py --json benchmark-results/single.json
python benchmarks/benchmark_batch.py --threads 4 --json benchmark-results/batch.json
python benchmarks/benchmark_tracking.py --json benchmark-results/tracking.json
```

To benchmark a local laprust change, run `python -m pip install -e ".[benchmark]"`
instead of the installation command above. All three scripts require LAPX
0.13.0 or newer and record the installed version. Before timing, the scripts
check the version and compare the installed distribution with the imported
module. They stop if the version is too old or the versions differ.

The separate `lap` package uses the same import name as `lapx`. If the version
check detects a conflicting installation, use a fresh virtual environment.

| Option | Effect / default |
| --- | --- |
| `--quick` | Use smaller workloads for smoke checks and CI. |
| `--dtype float32\|float64\|both` | Select the input dtype. The default is float64. |
| `--seed N` | Set the seed for reproducible inputs. The default is 2026. |
| `--repeats N` | Set the timing rounds. The default is 3 for single and tracking, or 2 for batch. |
| `--fast-rounds N` | Set the rounds for cases whose slowest call is under 1 ms. The default is 7. |
| `--min-time S` | Set the target seconds per timed sample. Each solver calibrates its call count. The default is 0.01. |
| `--calls N` | Set a fixed call count per sample. The default, 0, selects calibration. |
| `--reference-timing` | Use LAPX v0.13.0 timing. Use 1 call per sample for batches or dimensions above 64. Otherwise, use 25 calls. |
| `--verbose` | Print all environment metadata to the console. |
| `--threads N` | Set batch workers: 0 selects automatically, and 1 uses serial native workers. SciPy stays serial. |
| `--cost` | Include selected-cost totals for every batch candidate. |
| `--threshold T ...` | Set tracking thresholds. The default is `.05 .1 .5 1 1e9`, or `.05 .5` in quick mode. |

By default, each solver gets one untimed warm-up call right before its own
sample. A separate calibration call in the first round estimates the call count
needed to target `--min-time`. Later rounds reuse this count, so the measured
sample duration can vary.

`--reference-timing` uses the LAPX v0.13.0 timing settings. In each round, every
solver gets one warm-up call before any timed calls start.

Single and tracking cases with dimensions >=5000 always use one round. To set
custom tracking thresholds, use an option such as `--threshold .05 .1 .5`.
When measuring performance, avoid concurrent benchmarks or heavy CPU work.

## What is timed

- **Single:** The benchmark returns assignment pairs without totals. lapcpp and
  laprust use `precision="native", output="pairs", return_cost=False`. XA/SA
  return pairs directly with `return_cost=False`.
- **Batch:** The benchmark compares native batch APIs with a serial SciPy loop.
  The measurement includes worker management and optional `--cost` calculations.
- **Tracking:** The benchmark solves full assignment, removes pairs whose
  original cost exceeds the threshold, and constructs unmatched row and column
  lists. Native solvers use `extend_cost=True, return_cost=False`. The benchmark
  applies the threshold after solving. It does not pass the threshold as solver
  `cost_limit`, because joint rejection solves a different objective. Tracking
  has no `--cost`, `--cost-limit` or `--mode` option.

Single and batch benchmarks enable `extend_cost` only for rectangular inputs.
The benchmarks add no padding or pair sorting.

Inputs are uniform random costs in [0, 1). Every round shares a fresh input
across candidates and shuffles their order. Seeds are `[seed + round_index, N, M]`
for single/tracking and `[seed + round_index, B, N, M]` for batch.
Timing excludes input generation, warm-up and calibration calls, dispatch
selection, and validation. Timing includes solver function calls, required
SciPy pair conversion, and tracking post-processing. Timing includes Python
function-call overhead equally for all candidates.

## Validation and results

Single and tracking benchmarks require the same pairs as SciPy, regardless of
pair order. Tracking also checks thresholds and unmatched lists. Batch checks
feasibility, full cardinality, and selected-cost sums against SciPy with
`rtol=1e-6` and `atol=1e-8`. If totals are requested, batch also checks the
returned totals. **Any discrepancy stops the run.**

With other seeds or dtypes, tied or near-tied optima can fail strict pair checks.
This is especially relevant to float32 JVS.

Single and batch benchmarks print a ranked table for each case. Tracking groups
results by threshold and dtype. Times are medians in milliseconds, with checks
and speed ratios against SciPy. Verdicts are ⭐ baseline, 🏆 faster, 🐌 slower
and 🤝 within 1% of SciPy. The 1% band is a display rule and does not establish
statistical significance.

`OVERALL RANKING` orders solvers by **sum of per-case medians**. Large cases
dominate this sum. The ranking includes case counts and placing counts
(🥇, 🥈, 🥉, 🚩, 🏳️). Medal counts do not determine rank. Exact timing ties use
candidate order.

Each dtype/shape/threshold combination counts as one case. Repeated calls and
individual matrices within a batch do not count as separate cases.

JSON preserves every sample, shapes, thresholds, checks, package versions,
module paths, LAPX installation provenance, API calls and settings. Console
formatting does not change saved measurements.

## Workloads and reference

Counts below are per candidate, per dtype:

| Mode | Full run | `--quick` |
| --- | --- | --- |
| Single | 39 square/rectangular shapes, up to 5000x5000 | 14 shapes |
| Tracking | Same 39 shapes x 5 thresholds = 195 cases | 14 shapes x 2 thresholds = 28 cases |
| Batch | 65 workloads, dimensions up to 1000, batch sizes 1–128 | 15 workloads |

Quick workloads are subsets of the full runs. Batch sizes vary by shape to
expose worker overhead while keeping inputs at <=4 million entries. Dimensions
2000–5000 appear only in single and tracking benchmarks. See `cases()` in
[_common.py](_common.py) for every shape and batch count. That module handles
inputs, timing, checks and reporting. The three benchmark scripts contain the
solver calls.

The methodology follows the [LAPX v0.13.0 benchmarks](https://github.com/rathaROG/lapx/tree/v0.13.0/benchmarks).
These scripts add workloads, five candidates, dtype options, JSON output and
rankings.
The manual GitHub workflows compare lapcpp and laprust source checkouts against
the published `lapx>=0.13.0` package. They record the two checkout revisions and
installed LAPX version. JSON also records package versions and module paths.

These scripts compare packages. They do not reproduce the paper's full
protocol. The [paper benchmark guide](https://github.com/rathaROG/lap-paper/blob/main/benchmark/README.md)
covers the assignment campaign, cached tracking/replay and LSAPE study. The
paper configures JVS as float64. These scripts retain the LAPX benchmark's
default float32 JVS path. To reproduce the paper's tables and figures, use its
pinned dependencies and frozen evidence.
