"""LAPX historical regressions, adapted to aligned indices and native rejection.

Historical matrices/builders are preserved in regression_cases.py under the
original BSD license (tests/LICENSE). Tied optima are checked by objective,
not by freezing a legacy solver's arbitrary assignment order.
"""
import numpy as np
import pytest

import regression_cases as cases
from assignment_checks import normalize, scipy_optimum, validate


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["slice", "rectangle", "padded", "finite", "isolated"])
def test_original_extension_slices_limits_and_isolated_match(solver, dtype, case):
    cost = cases.get_dense_8x8_int()[0].astype(dtype)
    limit = np.inf
    if case == "slice":
        cost, expected = cost[:3, :3], 8.
    elif case == "rectangle":
        cost, expected = cost[:2, :4], 3.
    elif case == "padded":
        cost = np.concatenate([cost[:2, :4], np.array([[1001, 1001, 1001, 2001],
                                                      [2001, 1001, 1001, 1001]], dtype=dtype)])
        expected = 2005.
    elif case == "finite":
        cost, expected, limit = cost[:3, :3], 3., 4.99
    else:
        cost = np.full((4, 4), np.inf, dtype=dtype)
        cost[:3, :3] = [[1000, 4, 1], [1, 1000, 3], [5, 1, 1000]]
        cost[3, 3], expected = 0, 3.
    for value in (cost, cost.T):
        optimum = scipy_optimum(value, limit)
        for output in ("indices", "pairs"):
            total, pairs = normalize(solver.lapjv(value, extend_cost=True, cost_limit=limit,
                                                output=output), output)
            assert total == expected
            validate(value, total, pairs, limit, optimum=optimum)
            totals, assignments = normalize(solver.lapjv_batch(np.stack([value] * 2),
                extend_cost=True, cost_limit=limit, output=output, n_threads=2), output, batch=True)
            for total, pairs in zip(totals, assignments):
                assert total == expected
                validate(value, total, pairs, limit, optimum=optimum)


@pytest.mark.parametrize("case", cases.SQUARE_CASES + cases.SPARSE_CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_historical_square_augmentation(solver, case, dtype, output):
    cost, expected = case
    cost = cost.astype(dtype)
    original = cost.copy()
    cost.setflags(write=False)
    for return_cost in (True, False):
        total, pairs = normalize(solver.lapjv(cost, output=output, return_cost=return_cost), output, return_cost)
        validate(cost, total, pairs, optimum=expected[0])
        totals, assignments = normalize(solver.lapjv_batch(np.stack([cost] * 2), n_threads=2,
                                        output=output, return_cost=return_cost), output, return_cost, True)
        for i, pairs in enumerate(assignments):
            validate(cost, None if totals is None else totals[i], pairs, optimum=expected[0])
    np.testing.assert_array_equal(cost, original)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_arr_loop_regression(solver, transpose, dtype):
    cost = cases.arr_loop_cost().astype(dtype)
    if transpose:
        cost = cost.T
    optimum = scipy_optimum(cost)
    if dtype == np.float64:
        assert optimum == pytest.approx(.8455356917416, rel=1e-10)
    for output in ("indices", "pairs"):
        total, pairs = normalize(solver.lapjv(cost, extend_cost=True, output=output), output)
        validate(cost, total, pairs, optimum=optimum, atol=1e-7 if dtype == np.float32 else 1e-12)
        for threads in (1, 2):
            totals, assignments = normalize(solver.lapjv_batch(np.stack([cost] * 2), extend_cost=True,
                                            output=output, n_threads=threads), output, batch=True)
            for total, pairs in zip(totals, assignments):
                validate(cost, total, pairs, optimum=optimum, atol=1e-7 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["negative", "infinity", "blocking"])
def test_reported_rectangular_issues(solver, case, dtype, transpose):
    if case == "negative":
        cost = np.array([[2, 4, 6, 8], [1, 2, 4, 8]], dtype=dtype) - 30
        expected = -56.
    elif case == "infinity":
        cost = np.array([[np.inf, 11, 8], [8, np.inf, 7]], dtype=dtype)
        expected = 16.
    else:
        cost = np.full((11, 12), 1e6, dtype=dtype)
        for i in range(11):
            cost[i, i] = .01 * (i + 1)
        cost[0, 0] = .0057
        expected = sum(float(cost[i, i]) for i in range(11))
    if transpose:
        cost = cost.T
    for output in ("indices", "pairs"):
        total, pairs = normalize(solver.lapjv(cost, extend_cost=True, output=output), output)
        validate(cost, total, pairs, optimum=expected)
        if case == "blocking":
            np.testing.assert_array_equal(pairs, np.column_stack([np.arange(11)] * 2))
        totals, assignments = normalize(solver.lapjv_batch(np.stack([cost] * 3), extend_cost=True,
                                        n_threads=2, output=output), output, batch=True)
        for total, pairs in zip(totals, assignments):
            validate(cost, total, pairs, optimum=expected)


@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_epsilon_dataset(solver, output):
    cost, historical_optimum = cases.get_dense_eps()
    optimum = scipy_optimum(cost)
    assert optimum == historical_optimum
    total, pairs = normalize(solver.lapjv(cost, output=output), output)
    validate(cost, total, pairs, optimum=optimum, atol=3e-11)
    assert total == historical_optimum
    totals, assignments = normalize(solver.lapjv_batch(np.stack([cost] * 2), output=output,
                                    n_threads=2), output, batch=True)
    for total, pairs in zip(totals, assignments):
        validate(cost, total, pairs, optimum=optimum, atol=3e-11)


@pytest.mark.parametrize("builder", [
    cases.get_dense_100x100_int, cases.get_dense_100x100_int_hard,
    cases.get_sparse_100x100_int, cases.get_dense_1kx1k_int,
    cases.get_dense_1kx1k_int_hard, cases.get_sparse_1kx1k_int,
    cases.get_sparse_4kx4k_int,
], ids=lambda f: f.__name__.removeprefix("get_"))
def test_historical_seeded_known_optima(solver, builder):
    values = builder()
    cost, optimum = values[0].astype(np.float64), values[-1]
    if len(values) == 3:
        # The original dense tests use INT_MAX for masked edges, not sparse input.
        cost[~values[1]] = np.iinfo(np.int32).max
    cost.setflags(write=False)
    total, pairs = solver.lapjv(cost, output="pairs")
    validate(cost, total, pairs, optimum=optimum, atol=0)
    if len(cost) <= 1000:
        totals, assignments = solver.lapjv_batch(np.stack([cost] * 2), n_threads=2, output="pairs")
        for total, pairs in zip(totals, assignments):
            validate(cost, total, pairs, optimum=optimum, atol=0)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3)])
@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_billion_costs_known_optima_and_batch_order(solver, dtype, shape, output):
    problems = [cases.large_cost_case(shape, dtype, case) for case in range(3)]
    costs = np.stack([p[0] for p in problems])
    before = costs.copy()
    costs.setflags(write=False)
    for return_cost in (True, False):
        for cost, optimum, expected in problems:
            total, pairs = normalize(solver.lapjv(cost, extend_cost=True, output=output,
                                                return_cost=return_cost), output, return_cost)
            validate(cost, total, pairs, optimum=optimum, atol=0)
            np.testing.assert_array_equal(pairs, expected)
        for threads in (1, 2):
            totals, assignments = normalize(solver.lapjv_batch(costs, extend_cost=True,
                return_cost=return_cost, output=output, n_threads=threads), output, return_cost, True)
            for i, (cost, optimum, expected) in enumerate(problems):
                validate(cost, None if totals is None else totals[i], assignments[i], optimum=optimum, atol=0)
                np.testing.assert_array_equal(assignments[i], expected)
    np.testing.assert_array_equal(costs, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3)])
def test_billion_costs_with_forbidden_edges(solver, dtype, shape):
    cost, optimum, expected = cases.large_cost_case(shape, dtype, 1, forbidden=True)
    total, pairs = solver.lapjv(cost, extend_cost=True, output="pairs")
    validate(cost, total, pairs, optimum=optimum, atol=0)
    np.testing.assert_array_equal(pairs, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(128, 128), (96, 128), (128, 96)])
def test_large_signed_scaled_costs(solver, dtype, shape):
    cost = np.random.RandomState(608).randint(-16, 17, size=shape).astype(dtype) * dtype(2**24)
    optimum = scipy_optimum(cost)
    for precision in ("native", "float32", "float64"):
        total, pairs = solver.lapjv(cost, extend_cost=True, precision=precision, output="pairs")
        validate(cost, total, pairs, optimum=optimum, atol=0)
        totals, assignments = solver.lapjv_batch(np.stack([cost] * 2), extend_cost=True,
                                               precision=precision, output="pairs", n_threads=2)
        for total, pairs in zip(totals, assignments):
            validate(cost, total, pairs, optimum=optimum, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_large_rejection_limit_and_reversed_batch(solver, dtype, output):
    cost = np.array([[1e9, 8e9], [8e9, 4e9]], dtype=dtype)
    for return_cost in (True, False):
        for threads in (1, 2):
            totals, assignments = normalize(solver.lapjv_batch(np.stack([cost, cost[:, ::-1]]),
                cost_limit=2e9, n_threads=threads, output=output, return_cost=return_cost),
                output, return_cost, True)
            np.testing.assert_array_equal(assignments[0], [[0, 0]])
            np.testing.assert_array_equal(assignments[1], [[0, 1]])
            if totals is not None:
                np.testing.assert_array_equal(totals, [1e9, 1e9])
