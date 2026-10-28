# Changelog

## Unreleased

- Keep float32 cost storage while using float64 potentials, distances, and
  reduced-cost arithmetic in rectangular and finite-limit solving, matching the
  existing required-square working precision. This fixes lost small assignment
  differences beside large finite costs without widening the full cost matrix.
- Preserve cost conversion, represented half-limit rounding, and totals from
  original selected costs. Float64 working arithmetic is not an arbitrary-range
  exactness guarantee.
- Add exact float32 initialization and augmenting-search regressions, including
  primal/dual certificates and required/implicit-rejection paths.

## 1.1.0 - 2026-10-26

- Bound row-reduction initialization on large rectangular inputs to improve
  congested assignments, using separate budgets for required and finite-limit
  solving while preserving the public API and assignment objective.
- Keep the rectangular core synchronized with LAPX's Rust and hybrid solvers.
- Performance gains depend on the workload; some small-input slowdowns remain.

## 1.0.0 - 2026-10-23

First stable release.

### Package and API

- Standalone `laprust` distribution and import namespace for Python 3.9+.
- `lapjv` and `lapjv_batch` derived from the LAPX 0.11.1 `lapjvr` solver.
- Future solver development is led by lapcpp (C++) and laprust (Rust), with
  LAPX incorporating their fixes and improvements.
- Native rectangular assignment and implicit finite-limit rejection.
- Float32/float64 working costs, index or pair output, and reusable batch workers.
- Type annotations (`py.typed`).

### Solver and performance

- Rectangular and finite-limit search: reuse the root row's reduced costs for the
  first relaxation instead of resetting distances and predecessors separately.
- Batch results: convert counts once, and build per-problem views by row iteration
  when every problem is known to be fully matched (batches of 8 or more).
- Windows builds compile the C++ wrapper with MSVC `/std:c++17`, as lapcpp does.
- Stricter native checks: a path walk-back that never reaches its root and a final
  matching with `y[x[i]] != i` report an internal error instead of returning.

### Testing and CI

- Numerical/API tests, historical LAPX regression fixtures, edge/ownership checks,
  and exhaustive references.
- SciPy-free ARM numerical tests, complete empty-result and batch-size boundaries.
- Native allocation/worker failure injection, adapter exception/recovery checks,
  and native sanitizer jobs in separate PR/manual workflows.
- Required cross-package concurrency and isolated editable-install tests in CI.
- Full standalone wheel testing and older NumPy runtime compatibility jobs.

### Benchmarks

- Five-candidate benchmarks: dedicated SciPy, lapcpp, laprust and pair-output LAPX
  JVX/JVS functions, following LAPX v0.11.1 settings and uniform input generation.
- Single/tracking use the same 39 matrix shapes, including 4000x3000 and other large
  rectangles; batch uses 65 tapered workloads with medium rectangles and small-batch
  overhead cases. Every mode includes tiny and small rectangular workloads.
- Tracking always solves first and post-filters every candidate; no solver cost_limit
  or rejection expansion, with the original unmatched-list construction and
  threshold sweep.
- Calls per sample are calibrated (at least 10 ms) and each solver is warmed right
  before its own sample; sub-millisecond cases use 7 rounds. `--reference-timing`
  keeps the pinned LAPX timing for comparison.
- Benchmark verdicts and ranked test-count/total-time summaries in text and JSON.
