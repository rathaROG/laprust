"""Small independent numerical references for every wheel, including ARMv7.

Keep this module and assignment_checks importable without SciPy. These cases
exercise all native solve paths and return layouts, not just successful imports.
"""
from itertools import product
import numpy as np
import pytest
from assignment_checks import exhaustive_optimum, small_optimum, normalize, validate


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (3, 3), (3, 4), (4, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("limit", [np.inf, -.5, 0., .5])
def test_independent_optima_all_return_modes(solver, shape, dtype, limit):
    rng = np.random.default_rng(2081)
    costs = rng.integers(-8, 17, (4,) + shape).astype(dtype) / 8
    costs[0] = 0  # Ties must be compared by objective, not assignment identity.
    costs[1, 0, -1] = np.inf
    # Keep a full feasible diagonal for the required-matching problem.
    for cost in costs[1:]:
        np.fill_diagonal(cost, -.25)
    costs.setflags(write=False)
    expected = [exhaustive_optimum(a, limit) for a in costs]
    assert [small_optimum(a, limit) for a in costs] == expected
    for precision, output, return_cost in product(
            ("native", "float32", "float64"), ("indices", "pairs"), (False, True)):
        options = dict(extend_cost=True, cost_limit=limit, precision=precision,
                       output=output, return_cost=return_cost)
        for cost, optimum in zip(costs, expected):
            total, pairs = normalize(solver.lapjv(cost, **options), output, return_cost)
            validate(cost, total, pairs, limit, optimum=optimum, atol=0)
        for threads in (1, 2):
            totals, assignments = normalize(solver.lapjv_batch(costs, n_threads=threads, **options),
                                            output, return_cost, True)
            assert len(assignments) == len(costs)
            for i, pairs in enumerate(assignments):
                validate(costs[i], None if totals is None else totals[i], pairs, limit,
                         optimum=expected[i], atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (4, 3)])
def test_infeasible_optional_and_parallel_recovery(solver, dtype, shape):
    good = np.zeros((5,) + shape, dtype=dtype)
    bad = good.copy()
    bad[1] = bad[4] = np.inf
    for threads in (1, 2, 0):
        with pytest.raises(ValueError, match="batch index 1"):
            solver.lapjv_batch(bad, extend_cost=True, n_threads=threads)
        totals, assignments = solver.lapjv_batch(bad, cost_limit=.5, n_threads=threads, output="pairs")
        for cost, total, pairs in zip(bad, totals, assignments):
            validate(cost, total, pairs, .5, optimum=small_optimum(cost, .5), atol=0)
        np.testing.assert_array_equal(solver.lapjv_batch(good, extend_cost=True, n_threads=threads)[0], 0)
