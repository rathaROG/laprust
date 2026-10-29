# Rectangular initialization policy

The 1.2.0 rectangular initializer uses `B = min(columns, 4)` in both
row-reduction passes. Each pass allows at most `B` times its initial free-row
count in row scans. It prefers a free column among equal reduced-cost minima;
the implicit unmatched choice still wins finite-limit ties.

For required rectangular assignments, `max(rows, 8)` consecutive displacements
without acquiring a free column disable further price lowering. Acquiring a
free column resets the streak until the cutoff is reached. The current scan
completes normally, and the cutoff persists across both passes. Unresolved rows
then proceed to augmenting search.

Finite-limit assignments, including optional square problems, use the same
budget and tie rule without the stall window. Required square assignments
keep their existing solver. The assignment objective, precision rules, totals
and public API are preserved; equally optimal pairs may differ on tied inputs.

The window limits unproductive displacement chains in required matching.
Optional chains can still make useful progress, so cutting them off early can
increase augmenting-search work. Performance depends on the input: free-column
tie scans can also add overhead on heavily tied costs.
