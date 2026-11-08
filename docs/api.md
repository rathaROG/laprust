# API and numerical behavior

`laprust.lapjv` and `laprust.lapjv_batch` derive from the LAPX 0.11.1 `lapjvr`
family. They preserve its options and return contract. See their Python
docstrings for the full parameter reference and executable examples.

## Return formats

| Call options | Single return | Batch return |
| --- | --- | --- |
| Defaults | `(total, rows, cols)` | `(totals, rows_list, cols_list)` |
| `return_cost=False` | `(rows, cols)` | `(rows_list, cols_list)` |
| `output="pairs"` | `(total, pairs)` | `(totals, pairs_list)` |
| Both | `pairs` | `pairs_list` |

`total` is a Python float. `totals` is a float64 array. Indices are int64, sorted
by row, and refer to the original matrix. Empty matrix dimensions produce empty
assignments and zero totals.

An empty batch returns empty lists and totals.

## Matching and rejection

An infinite limit requires `min(N, M)` assignments. Nonempty rectangles need
`extend_cost=True`. Positive infinity in a cost matrix forbids that edge.
An infeasible required assignment raises `ValueError`.

A finite `cost_limit=L` jointly optimizes assignments and rejection.
The penalty is `L/2` per unmatched row and per unmatched column.
Rectangular inputs are accepted regardless of `extend_cost`.
Square inputs use the same objective.
Negative finite limits are supported. The returned total sums selected original
costs and excludes all rejection penalties.

Equal-cost optima can contain different numbers of pairs.

Inputs must contain real values with no NaN or negative infinity. The solver
does not scan the entire input for NaN or negative infinity. Results are
undefined if the input contains either value. A NaN or negative
infinite `cost_limit` is rejected. Infeasibility, dimension overflow and reduced
cost overflow raise `ValueError`. Allocation failure raises `MemoryError`.

If native solving fails for a batch, the error reports the lowest failing batch
index. The call returns no partial result.

## Precision and storage

`precision="native"` preserves float32 and float64 inputs. Other numeric types
are converted to float64. Explicit conversion to float32 rejects finite values
or limits outside the float32 range. Integer inputs are converted to float64,
so integers outside its exact range can lose precision.

Other real dtypes first pass through NumPy's float64 conversion without an
equivalent explicit overflow check. Finite values from wider floating types
can become positive infinity. The solver treats those edges as forbidden.
If conversion would lose relevant values, prepare the input before calling the
solver:

- Choose a supported dtype.
- Scale the costs.

All paths keep potentials, distances and reduced-cost arithmetic in float64,
while matrix storage follows the selected cost precision. For finite limits,
each half-penalty is rounded in that cost dtype. The represented halves are
combined in that dtype to form the row-rejection penalty before search uses
float64. The returned total always
sums the original selected costs in float64. It uses their values before any
requested float32 conversion.
Changing precision can change the optimal assignment of the rounded working
problem.

Float64 working arithmetic reduces cancellation with float32 costs without
widening the whole matrix. Float64 arithmetic cannot restore costs rounded by
conversion. Sufficiently extreme magnitude differences can still obscure small
objective differences. Float64 arithmetic does not guarantee exactness for
arbitrary cost ranges.

Suitable aligned, contiguous, native-endian arrays are borrowed. Layout,
endianness or dtype conversion can require an input copy. Native rectangles
avoid square padding. Tall matrices may need a transposed copy. Scratch storage
is linear in the dimensions, in addition to costs, output and any conversion or
transpose buffers.

The solver reuses workspaces and a bounded number of thread pools. The solver
releases large temporary transpose buffers after each call.

## Concurrency

Native solving releases the GIL. Until the call returns, do not mutate or
resize the input or an alias. Independent calls may run concurrently. Batch
order is preserved.

Automatic thread selection may execute small workloads serially. Explicit
positive worker counts are limited by batch size. Avoid oversubscribing cores
with simultaneous large batches. After worker pools have initialized, use
process spawning rather than forking.
