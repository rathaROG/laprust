# Copyright (c) 2026 Ratha SIV | MIT License
"""Batch scheduling and result-construction paths.

Rayon splits large batches of small problems across workers, and batches known
to be fully matched build their per-problem results by row iteration. Every such
path must return exactly what solving each problem separately returns.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def native():
    import laprust
    return laprust


def split(result, output, return_cost):
    """Return (totals or None, per-problem arrays) for single or batch results."""
    total = result[0] if return_cost else None
    arrays = result[1:] if return_cost else result
    if output == "pairs":
        arrays = (arrays[0],) if return_cost else (arrays,)
    return total, arrays


def assert_matches_single_solves(native, costs, batch, output, return_cost, **options):
    totals, arrays = split(batch, output, return_cost)
    for index, cost in enumerate(costs):
        single = native.lapjv(cost, output=output, return_cost=return_cost, **options)
        total, expected = split(single, output, return_cost)
        for column, wanted in zip(arrays, expected):
            actual = column[index]
            assert actual.dtype == np.int64
            np.testing.assert_array_equal(actual, wanted)
        if return_cost:
            assert totals[index] == total


@pytest.mark.parametrize("shape", [(2000, 5, 5), (500, 8, 12), (300, 12, 8)])
@pytest.mark.parametrize("threads", [2, 4])
@pytest.mark.parametrize("limit", [np.inf, .5])
@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_chunked_parallel_batches_match_single_solves(native, shape, threads, limit, output):
    # Hundreds of small problems per worker exercise the parallel split and ordering.
    costs = np.random.default_rng(shape[0] + threads).random(shape)
    options = dict(extend_cost=True, cost_limit=limit)
    batch = native.lapjv_batch(costs, n_threads=threads, output=output, **options)
    assert_matches_single_solves(native, costs, batch, output, True, **options)
    serial = native.lapjv_batch(costs, n_threads=1, output=output, **options)
    for actual, expected in zip(split(batch, output, True)[1], split(serial, output, True)[1]):
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_failure_inside_a_chunk_reports_lowest_index_and_recovers(native, threads):
    costs = np.random.default_rng(70).random((2000, 5, 5))
    bad = costs.copy()
    for index in (300, 70, 1999):  # Rows with no permitted column are infeasible.
        bad[index, 2, :] = np.inf
    with pytest.raises(ValueError, match="batch index 70$"):
        native.lapjv_batch(bad, n_threads=threads)
    # Workers and their workspaces remain usable after the failure.
    batch = native.lapjv_batch(costs, n_threads=threads)
    assert_matches_single_solves(native, costs, batch, "indices", True)


@pytest.mark.parametrize("size", [7, 8, 31, 32, 33])
@pytest.mark.parametrize("limit", [np.inf, 1e9, .05])
@pytest.mark.parametrize("output", ["indices", "pairs"])
@pytest.mark.parametrize("return_cost", [False, True])
def test_result_unpacking_boundaries(native, size, limit, output, return_cost):
    # An infinite limit and a very large finite limit match every row; a tight
    # limit leaves rows unmatched, so each batch size meets both result paths.
    costs = np.random.default_rng(size).random((size, 6, 9))
    options = dict(extend_cost=True, cost_limit=limit)
    batch = native.lapjv_batch(costs, n_threads=2, output=output, return_cost=return_cost, **options)
    assert_matches_single_solves(native, costs, batch, output, return_cost, **options)
    _, arrays = split(batch, output, return_cost)
    counts = [len(item) for item in arrays[0]]
    assert len(counts) == size
    assert (min(counts) < 6) == (limit == .05)
    # Each problem's result is its own view: writing one leaves the others intact.
    first, second = arrays[0][0], arrays[0][1]
    before = second.copy()
    first[...] = -1
    np.testing.assert_array_equal(second, before)
