# Rectangular initialization policy

The 1.2.0 rectangular initializer uses `B = min(columns, 4)` in both
row-reduction passes. Each pass allows at most `B` times its initial free-row
count in row scans. It prefers a free column among equal reduced-cost minima.
The implicit unmatched choice still wins finite-limit ties.

For required rectangular assignments, the cutoff is `max(rows, 8)` consecutive
displacements without acquiring a free column. This cutoff disables further
price lowering. Before the cutoff, acquiring a free column resets the count of
consecutive displacements. The current scan completes normally. The cutoff
persists across both passes. Unresolved rows then proceed to augmenting search.

Finite-limit assignments, including optional square problems, use the same
budget and tie rule without the cutoff. Required square assignments keep their
existing solver. The assignment objective, precision rules, totals and public
API remain unchanged. Equally optimal pairs may differ on tied inputs.

The window limits unproductive displacement chains in required matching.
Optional chains can still make useful progress, so cutting them off early can
increase augmenting-search work. Performance depends on the input. Scans for
free columns among tied costs can also add overhead on heavily tied costs.
