# Five-candidate benchmarks

See the [initialization policy](initialization-policy.md) for the 1.2.0
reduction rules and performance tradeoffs.

Compare single assignments, native batches and tracking against SciPy. Each
candidate has a dedicated solver call; there are no fallback solvers.

| Candidate | Single / tracking API | Batch API | Working precision |
| --- | --- | --- | --- |
| SciPy (baseline) | `scipy.optimize.linear_sum_assignment` | Serial loop | float64 |
| lapcpp | `lapcpp.lapjv` | `lapcpp.lapjv_batch` | Input dtype |
| laprust | `laprust.lapjv` | `laprust.lapjv_batch` | Input dtype |
| LAPX JVX | `lap.lapjvxa` | `lap.lapjvxa_batch` | float64 |
| LAPX JVS | `lap.lapjvsa` | `lap.lapjvsa_batch` | Default float32 |

**This is not an equal-working-precision comparison.** JVS uses float32 even
with float64 inputs. Only the XA/SA variants are included, with no forced-float64
JVS candidate.

## Installation and commands

Once lapcpp and laprust are on PyPI, install the dependencies below
(`lapx` imports as `lap`). Clone or download either [lapcpp](https://github.com/rathaROG/lapcpp)
or [laprust](https://github.com/rathaROG/laprust) for the identical benchmark
scripts, then run from that repository's root:

```sh
python -m pip install --upgrade lapcpp laprust "lapx==0.11.1" scipy
python benchmarks/benchmark_single.py --json benchmark-results/single.json
python benchmarks/benchmark_batch.py --threads 4 --json benchmark-results/batch.json
python benchmarks/benchmark_tracking.py --json benchmark-results/tracking.json
```

| Option | Effect / default |
| --- | --- |
| `--quick` | Smaller smoke/CI workloads |
| `--dtype float32\|float64\|both` | Input dtype; default float64 |
| `--seed N` | Reproducible inputs; default 2026 |
| `--repeats N` | Timing rounds; default 3 for single/tracking, 2 for batch |
| `--fast-rounds N` | Rounds for cases whose slowest call is under 1 ms; default 7 |
| `--min-time S` | Minimum seconds per timed sample; calls are calibrated per solver; default 0.01 |
| `--calls N` | Fixed calls per sample instead of calibration; default 0 (calibrate) |
| `--reference-timing` | LAPX v0.11.1 timing: 1 call per sample in batches and above 64 columns, 25 otherwise |
| `--verbose` | Full environment metadata in the console |
| `--threads N` | Batch only: 0 = automatic, 1 = serial native workers; SciPy stays serial |
| `--cost` | Batch only: include selected-cost totals for every candidate |
| `--threshold T ...` | Tracking only: default `.05 .1 .5 1 1e9`; quick mode `.05 .5` |

Each solver gets one untimed warm-up call right before its own sample, and
calls repeat until the sample lasts `--min-time`, so microsecond cases are not
single cold calls. `--reference-timing` keeps the LAPX v0.11.1 settings only for
comparison with its published numbers.

Single/tracking dimensions >=5000 always use one round. For custom tracking
thresholds, use e.g. `--threshold .05 .1 .5`. Avoid concurrent benchmarks or heavy
CPU work when measuring performance.

## What is timed

- **Single:** assignment pairs without totals. lapcpp/laprust use
  `precision="native", output="pairs", return_cost=False`; XA/SA return pairs
  directly with `return_cost=False`. Single/batch enable `extend_cost` only for
  rectangular inputs; no padding or pair sorting is added.
- **Batch:** native batch APIs versus a serial SciPy loop. Worker management and
  optional `--cost` calculations are included in the measurement.
- **Tracking:** full assignment, removal of pairs whose original cost exceeds
  the threshold, then construction of unmatched row/column lists. Native solvers
  use `extend_cost=True, return_cost=False`. **The threshold is a post-filter,
  not a solver `cost_limit`:** joint rejection solves a different objective.
  Tracking has no `--cost`, `--cost-limit` or `--mode` option.

Inputs are uniform random costs in [0, 1). Every round shares a fresh input
across candidates and shuffles their order. Seeds are `[seed + round_index, N, M]`
for single/tracking and `[seed + round_index, B, N, M]` for batch.
Input generation, warm-up and calibration calls, dispatch selection and validation are outside timing;
solver function calls, required SciPy pair conversion and tracking post-processing
are inside. Python function-call overhead is included equally for all candidates.

## Validation and results

Single/tracking require unordered pair equality with SciPy; tracking also checks
thresholds and unmatched lists. Batch checks feasibility, full cardinality and
selected-cost sums against SciPy (`rtol=1e-6`, `atol=1e-8`), plus returned totals
when requested. **Any discrepancy stops the run.** Tied or near-tied optima can
fail strict pair checks with other seeds/dtypes, especially float32 JVS.

Single/batch print ranked per-case tables; tracking groups results by threshold
and dtype. Times are medians in milliseconds, with checks and speed ratios
against SciPy. Verdicts are ⭐ baseline, 🏆 faster, 🐌 slower and 🤝 within 1%
of SciPy; the 1% band is a display rule, not statistical significance.

`OVERALL RANKING` orders solvers by **sum of per-case medians**; large cases
dominate. It includes case counts and placing counts (🥇, 🥈, 🥉, 🚩, 🏳️),
but medal counts do not determine rank. Exact timing ties use candidate order.
Cases count each dtype/shape/threshold combination, not repeated calls or
individual matrices within a batch.

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
expose worker overhead while keeping inputs at <=4 million entries; dimensions
2000–5000 appear only in single/tracking. See `cases()` in
[_common.py](_common.py) for every shape and batch count. That module handles
inputs, timing, checks and reporting; solver calls live in the three scripts.

Methodology follows the [LAPX v0.11.1 benchmarks](https://github.com/rathaROG/lapx/tree/v0.11.1/benchmarks),
with expanded workloads, five candidates, optional dtypes and JSON/ranking output.
The manual GitHub workflows compare source checkouts, default `lapx_ref` to
`v0.11.1`, allow overrides and record all three package revisions.
