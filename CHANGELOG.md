# Changelog

## 1.2.1 - 2026-11-08

- Require LAPX 0.13.0 or newer for all package benchmarks.
- Record and check the installed LAPX version.
- Include lapcpp in the benchmark installation extra.
- Refresh installation, development and paper links.
- Distinguish the package benchmarks from the paper's reproducible experiment
  suite.
- Add package, license, paper and total/monthly download badges.
- Add documentation and support links to the package metadata.

## 1.2.0 - 2026-10-28

- Use a common four-scan row-reduction allowance throughout rectangular and
  finite-limit initialization.
- Prefer equally cheap free columns throughout rectangular and finite-limit
  initialization.
- Stop lowering column potentials after a stalled displacement chain in required
  rectangular matching. Keep the bounded reduction allowance for finite-limit
  matching, where interrupting long chains can increase augmenting-search work.
- Preserve the assignment objective, float32 fixes, and required-square solver.
  Performance depends on the workload. See the
  [initialization policy and tradeoffs](benchmarks/initialization-policy.md).

## 1.1.1 - 2026-10-28

- Keep float32 cost storage while using float64 potentials, distances, and
  reduced-cost arithmetic in rectangular and finite-limit solving.
  This matches the existing required-square working precision.
  This fixes lost small assignment differences beside large finite costs without
  widening the full cost matrix.
- Preserve cost conversion, represented half-limit rounding, and totals from
  original selected costs. Float64 working arithmetic is not an arbitrary-range
  exactness guarantee.
- Add exact float32 initialization and augmenting-search regressions, including
  primal/dual certificates and required/implicit-rejection paths.

## 1.1.0 - 2026-10-26

- Bound row-reduction initialization on large rectangular inputs to improve
  congested assignments. Use separate budgets for required and finite-limit
  solving. Preserve the public API and assignment objective.
- Keep the rectangular core synchronized with LAPX's Rust and hybrid solvers.
- Performance gains depend on the workload. Some small-input slowdowns remain.

## 1.0.0 - 2026-10-23

This is the first stable release.

### Package and API

- Provide a standalone `laprust` distribution and import namespace for Python 3.9+.
- Derive `lapjv` and `lapjv_batch` from the LAPX 0.11.1 `lapjvr` solver.
- lapcpp (C++) and laprust (Rust) lead future solver development.
  LAPX incorporates their fixes and improvements.
- Support native rectangular assignment and implicit finite-limit rejection.
- Support float32/float64 working costs and index or pair output.
- Reuse batch workers.
- Provide type annotations (`py.typed`).

### Solver and performance

- Reuse the root row's reduced costs for the first relaxation in rectangular
  and finite-limit search. This replaces separate resets of distances and predecessors.
- Convert batch result counts once. For batches of 8 or more, build per-problem
  views by row iteration when every problem is known to be fully matched.
- Windows builds compile the C++ wrapper with MSVC `/std:c++17`, as lapcpp does.
- If a path walk-back never reaches its root, report an internal error instead
  of returning a result.
- If a final matching has `y[x[i]] != i`, report an internal error instead of
  returning a result.

### Testing and CI

- Add numerical/API tests, historical LAPX regression fixtures, edge/ownership
  checks, and exhaustive references.
- Add SciPy-free ARM numerical tests.
- Cover all empty-result and batch-size boundaries.
- Inject native allocation/worker failures in tests.
- Check adapter exceptions and recovery.
- Add native sanitizer jobs in separate PR/manual workflows.
- Require cross-package concurrency and isolated editable-install tests in CI.
- Run full tests on standalone wheels.
- Add compatibility jobs for older NumPy runtimes.

### Benchmarks

- Compare five candidates through dedicated SciPy, lapcpp, laprust and pair-output
  LAPX JVX/JVS functions. Use LAPX v0.11.1 settings and uniform input generation.
- Use the same 39 matrix shapes for single/tracking benchmarks, including
  4000x3000 and other large rectangles.
- Use 65 tapered batch workloads with medium rectangles and small-batch
  overhead cases. Every mode includes tiny and small rectangular workloads.
- In tracking benchmarks, solve first and post-filter every candidate.
  Do not pass a solver cost_limit or expand the matrix for rejection.
  Preserve the original unmatched-list construction and threshold sweep.
- Calibrate calls per sample to target at least 10 ms.
  Warm each solver immediately before its own sample. Use 7 rounds for
  sub-millisecond cases. `--reference-timing` keeps the pinned LAPX timing for
  comparison.
- Report benchmark verdicts and ranked test-count/total-time summaries in text
  and JSON.
