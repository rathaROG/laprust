# API and numerical behavior

`laprust.lapjv` and `laprust.lapjv_batch` keep the options and return contract of
the LAPX 0.11.1 `lapjvr` family they were derived from. See their Python
docstrings for the full parameter reference and executable examples.

## Return formats

| Call options | Single return | Batch return |
| --- | --- | --- |
| Defaults | `(total, rows, cols)` | `(totals, rows_list, cols_list)` |
| `return_cost=False` | `(rows, cols)` | `(rows_list, cols_list)` |
| `output="pairs"` | `(total, pairs)` | `(totals, pairs_list)` |
| Both | `pairs` | `pairs_list` |

`total` is a Python float; `totals` is a float64 array. Indices are int64, sorted
by row, and refer to the original matrix. Empty matrix dimensions produce empty
assignments and zero totals. An empty batch returns empty lists and totals.

## Matching and rejection

An infinite limit requires `min(N, M)` assignments. Nonempty rectangles need
`extend_cost=True`. Positive infinity in a cost matrix forbids that edge; an
infeasible required assignment raises `ValueError`.

A finite `cost_limit=L` jointly optimizes assignments and rejection, with a
penalty of `L/2` per unmatched row and per unmatched column. Rectangular inputs
are accepted regardless of `extend_cost`; square inputs use the same objective.
Negative finite limits are supported. The returned total sums selected original
costs and excludes all rejection penalties. At equality, multiple optima can
have different cardinalities.

Inputs must contain real values with no NaN or negative infinity. Those values
are not globally scanned and their results are undefined. A NaN or negative
infinite `cost_limit` is rejected. Infeasibility, dimension overflow and reduced
cost overflow raise `ValueError`; allocation failure raises `MemoryError`.
Batch native failures report the lowest failing batch index and return no
partial result.

## Precision and storage

`precision="native"` preserves float32 and float64 inputs; other numeric types
are converted to float64. Explicit conversion to float32 rejects finite values
or limits outside the float32 range. Integer inputs are converted to float64,
so integers outside its exact range can lose precision.

The required square path keeps potentials and distances in float64 even with
float32 costs. Rectangular and finite-limit paths use the selected working
precision; rejection penalties are represented in that precision. The returned
total always sums the original selected costs in float64 before any requested
float32 conversion. Changing precision can change the optimal assignment of
the rounded working problem.

Suitable aligned, contiguous, native-endian arrays are borrowed. Layout,
endianness or dtype conversion can require an input copy. Native rectangles
avoid square padding; tall matrices may need a transposed copy. Scratch storage
is linear in the dimensions, in addition to costs, output and any conversion or
transpose buffers. Workspaces and a bounded number of thread pools are reused;
large temporary transpose buffers are released after each call.

## Concurrency

Native solving releases the GIL. Do not mutate or resize the input or an alias
until its call returns. Independent calls may run concurrently. Batch order is
preserved. Automatic thread selection may execute small workloads serially;
explicit positive worker counts are limited by batch size. Avoid oversubscribing
cores with simultaneous large batches. Use process spawning rather than forking
after worker pools have initialized.
