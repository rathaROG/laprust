"""Independent reference and validity checks for public assignment results."""
from itertools import combinations, permutations
import numpy as np


def normalize(result, output="indices", return_cost=True, batch=False):
    total = result[0] if return_cost else None
    values = (result[1] if output == "pairs" else result[1:]) if return_cost else result
    if batch:
        pairs = values if output == "pairs" else [np.column_stack(p) for p in zip(*values)]
        return total, pairs
    return total, values if output == "pairs" else np.column_stack(values)


def selected_total(cost, pairs):
    total = 0.0
    for row, col in pairs:
        total += float(cost[row, col])
    return total


def objective(cost, pairs, limit=np.inf):
    total = selected_total(cost, pairs)
    if np.isfinite(limit):
        half = float(cost.dtype.type(limit / 2))
        total += (sum(cost.shape) - 2 * len(pairs)) * half
    return total


def scipy_optimum(cost, limit=np.inf):
    from scipy.optimize import linear_sum_assignment
    n, m = cost.shape
    work = cost.astype(np.float64)
    if np.isfinite(limit):
        expanded = np.zeros((n + m, n + m))
        expanded[:n, :m] = work
        expanded[:n, m:] = expanded[n:, :m] = float(cost.dtype.type(limit / 2))
        work = expanded
    rows, cols = linear_sum_assignment(work)
    return float(work[rows, cols].sum(dtype=np.float64))


def small_optimum(cost, limit=np.inf):
    """Independent bitmask dynamic program for small matrices, without SciPy."""
    n, m = cost.shape
    if n > m:
        return small_optimum(cost.T, limit)
    if m > 12:
        raise ValueError("The small-problem reference is limited to 12 columns")
    states = {0: 0.0}
    optional = np.isfinite(limit)
    for row in cost:
        following = dict(states) if optional else {}
        for used, total in states.items():
            for col, value in enumerate(row):
                if not used & (1 << col) and np.isfinite(value):
                    mask = used | (1 << col)
                    following[mask] = min(following.get(mask, np.inf), total + float(value))
        states = following
    half = float(cost.dtype.type(limit / 2)) if optional else 0.0
    return min((value + (n + m - 2 * bin(mask).count("1")) * half
                for mask, value in states.items()), default=np.inf)


def exhaustive_optimum(cost, limit=np.inf):
    """Enumerate matchings, including every rejection cardinality when finite."""
    n, m = cost.shape
    best = np.inf
    counts = range(min(n, m) + 1) if np.isfinite(limit) else [min(n, m)]
    for count in counts:
        for rows in combinations(range(n), count):
            for cols in permutations(range(m), count):
                pairs = np.column_stack((rows, cols)).astype(np.int64)
                if np.isfinite(cost[tuple(pairs.T)]).all():
                    best = min(best, objective(cost, pairs, limit))
    return best


def validate(cost, total, pairs, limit=np.inf, optimum=None, atol=1e-10):
    assert pairs.shape[1:] == (2,) and pairs.ndim == 2
    assert pairs.dtype == np.int64
    assert (pairs >= 0).all()
    assert (pairs[:, 0] < cost.shape[0]).all() and (pairs[:, 1] < cost.shape[1]).all()
    assert len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1])) == len(pairs)
    assert np.all(pairs[1:, 0] > pairs[:-1, 0])
    assert np.isfinite(cost[tuple(pairs.T)]).all()
    if not np.isfinite(limit):
        assert len(pairs) == min(cost.shape)
    if total is not None:
        assert total == selected_total(cost, pairs)
    if optimum is not None:
        np.testing.assert_allclose(objective(cost, pairs, limit), optimum, rtol=0, atol=atol)


def assert_same(actual, expected):
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for got, wanted in zip(actual, expected):
            assert_same(got, wanted)
    else:
        np.testing.assert_array_equal(actual, expected)
