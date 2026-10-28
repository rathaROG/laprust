"""Exact regressions for cancellation in rectangular and finite-limit solving.

Input entries are exact binary fractions. Small exhaustive references distinguish
the represented optimization objective from totals on the original input.
"""
from fractions import Fraction
from itertools import permutations
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.fixture
def native_precision_family(solver):
    return SimpleNamespace(single=solver.lapjv, batch=solver.lapjv_batch)


def _exact_cost(cost, pairs):
    return sum((Fraction.from_float(float(cost[i, j])) for i, j in pairs), Fraction())


def _required_optimum(cost):
    oriented = cost if cost.shape[0] <= cost.shape[1] else cost.T
    return min(
        sum((Fraction.from_float(float(oriented[i, j])) for i, j in enumerate(cols)), Fraction())
        for cols in permutations(range(oriented.shape[1]), oriented.shape[0])
    )


def _assert_required(cost, pairs):
    assert pairs.shape == (min(cost.shape), 2)
    assert pairs.dtype == np.int64
    rows, cols = pairs.T
    assert np.all((rows >= 0) & (rows < cost.shape[0]))
    assert np.all((cols >= 0) & (cols < cost.shape[1]))
    assert len(np.unique(rows)) == len(rows)
    assert len(np.unique(cols)) == len(cols)
    assert _exact_cost(cost, pairs) == _required_optimum(cost)


def _half_penalty(limit, working_dtype):
    # The public native contract rounds the half-limit in the working cost
    # type before combining its two halves. Widening scratch must retain it.
    return Fraction.from_float(float(working_dtype(float(limit) / 2)))


def _optional_objective(cost, pairs, half):
    return _exact_cost(cost, pairs) + (sum(cost.shape) - 2 * len(pairs)) * half


def _optional_optimum(cost, half):
    # Enumerate every partial matching independently of a dummy construction
    # or another assignment implementation. Inputs here have at most 4 rows.
    best = None

    def visit(row, used, selected):
        nonlocal best
        if row == cost.shape[0]:
            value = _optional_objective(cost, selected, half)
            best = value if best is None else min(best, value)
            return
        visit(row + 1, used, selected)
        for col in range(cost.shape[1]):
            if col not in used and np.isfinite(cost[row, col]):
                visit(row + 1, used | {col}, selected + [(row, col)])

    visit(0, set(), [])
    return best


def _native_pairs(result, output, return_cost):
    if return_cost:
        result = result[1] if output == 'pairs' else result[1:]
    return result if output == 'pairs' else np.column_stack(result)


def _assert_native_optimum(cost, pairs, limit, working_dtype):
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert pairs.dtype == np.int64
    rows, cols = pairs.T
    assert np.all((rows >= 0) & (rows < cost.shape[0]))
    assert np.all((cols >= 0) & (cols < cost.shape[1]))
    assert len(np.unique(rows)) == len(rows)
    assert len(np.unique(cols)) == len(cols)
    assert np.all(rows[1:] > rows[:-1])
    assert np.isfinite(cost[rows, cols]).all()
    if np.isfinite(limit):
        half = _half_penalty(limit, working_dtype)
        assert _optional_objective(cost, pairs, half) == _optional_optimum(cost, half)
    else:
        _assert_required(cost, pairs)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('output', ['indices', 'pairs'])
@pytest.mark.parametrize('precision,input_dtype', [
    ('native', np.float32), ('float32', np.float64), ('float64', np.float32),
    ('native', np.float64), ('float64', np.float64),
])
def test_native_rectangular_small_differences_beside_large_finite_costs(
        native_precision_family, transpose, output, precision, input_dtype):
    high, low = 181406.1875, 1 / 4096
    cost = np.array([[high, 2 * low, high], [high, low, high]], dtype=input_dtype)
    if transpose:
        cost = cost.T
    original = cost.copy()
    cost.setflags(write=False)
    working_dtype = (np.float64 if precision == 'float64' or
                     (precision == 'native' and input_dtype == np.float64) else np.float32)
    # Both limits force full cardinality on this example. The finite call also
    # exercises the implicit rejection path, which has its own reduced scans.
    for limit in (np.inf, 4 * high):
        for return_cost in (False, True):
            result = native_precision_family.single(
                cost, extend_cost=True, cost_limit=limit, precision=precision,
                output=output, return_cost=return_cost)
            pairs = _native_pairs(result, output, return_cost)
            _assert_native_optimum(cost, pairs, limit, working_dtype)
            assert len(pairs) == min(cost.shape)
            if return_cost:
                assert result[0] == float(_exact_cost(cost, pairs))
    np.testing.assert_array_equal(cost, original)


@pytest.mark.parametrize('input_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('output', ['indices', 'pairs'])
@pytest.mark.parametrize('transpose', [False, True])
def test_cancellation_in_grouped_long_row_scan(
        native_precision_family, input_dtype, output, transpose):
    # More than 16 columns selects the grouped optional minimum scan. Extra
    # high-cost columns preserve the exact H+s optimum of the two-row fixture.
    high, small = 181406.1875, 1 / 4096
    cost = np.full((2, 17), high, dtype=input_dtype)
    cost[:, 1] = [2 * small, small]
    if transpose:
        cost = cost.T
    original = cost.copy()
    cost.setflags(write=False)
    for limit in (np.inf, 4 * high):
        for return_cost in (False, True):
            result = native_precision_family.single(
                cost, extend_cost=True, cost_limit=limit,
                output=output, return_cost=return_cost)
            pairs = _native_pairs(result, output, return_cost)
            _assert_native_optimum(cost, pairs, limit, input_dtype)
            assert len(pairs) == 2
            assert _exact_cost(cost, pairs) == Fraction.from_float(high) + Fraction.from_float(small)
            if return_cost:
                assert result[0] == high + small
        costs = np.stack((cost, cost, cost))
        costs.setflags(write=False)
        batch = native_precision_family.batch(
            costs, extend_cost=True, cost_limit=limit, output=output, n_threads=2)
        assignments = batch[1] if output == 'pairs' else [
            np.column_stack((rows, cols)) for rows, cols in zip(*batch[1:])]
        for total, pairs in zip(batch[0], assignments):
            _assert_native_optimum(cost, pairs, limit, input_dtype)
            assert total == high + small
    np.testing.assert_array_equal(cost, original)



@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('output', ['indices', 'pairs'])
@pytest.mark.parametrize('n_threads', [1, 2])
@pytest.mark.parametrize('input_dtype', [np.float32, np.float64])
def test_native_rectangular_precision_batches_keep_small_optima(
        native_precision_family, transpose, output, n_threads, input_dtype):
    high, low = 181406.1875, 1 / 4096
    base = np.array([[high, 2 * low, high], [high, low, high]], dtype=input_dtype)
    costs = np.stack((base, base[::-1], base[:, ::-1]))
    if transpose:
        costs = costs.transpose(0, 2, 1)  # Retain a strided batch view.
    original = costs.copy()
    costs.setflags(write=False)
    for limit in (np.inf, 4 * high):
        for return_cost in (False, True):
            result = native_precision_family.batch(
                costs, extend_cost=True, cost_limit=limit, output=output,
                return_cost=return_cost, n_threads=n_threads)
            totals = result[0] if return_cost else None
            if output == 'pairs':
                assignments = result[1] if return_cost else result
            else:
                rows, cols = result[1:] if return_cost else result
                assert len(rows) == len(cols) == len(costs)
                assignments = [np.column_stack((r, c)) for r, c in zip(rows, cols)]
            assert len(assignments) == len(costs)
            if return_cost:
                assert totals.shape == (len(costs),)
            for index, (cost, pairs) in enumerate(zip(costs, assignments)):
                _assert_native_optimum(cost, pairs, limit, input_dtype)
                assert len(pairs) == min(cost.shape)
                if return_cost:
                    assert totals[index] == float(_exact_cost(cost, pairs))
    np.testing.assert_array_equal(costs, original)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('output', ['indices', 'pairs'])
@pytest.mark.parametrize('high,small', [(2**24, 1 / 8), (181406.1875, 1 / 4096)])
def test_native_small_differences_survive_augmenting_search(
        native_precision_family, transpose, output, high, small):
    # Three rows compete for two cheap columns, leaving an unresolved row
    # after reduction. The later path relaxation must preserve H-2s.
    cost = np.array([[2 * small, 2 * small, high, high],
                     [small, small, high, high],
                     [0, 0, high, high]], dtype=np.float32)
    if transpose:
        cost = cost.T
    original = cost.copy()
    cost.setflags(write=False)
    for limit in (np.inf, 2 * high):
        for return_cost in (False, True):
            result = native_precision_family.single(
                cost, extend_cost=True, cost_limit=limit, output=output,
                return_cost=return_cost)
            pairs = _native_pairs(result, output, return_cost)
            _assert_native_optimum(cost, pairs, limit, np.float32)
            assert len(pairs) == 3
            assert _exact_cost(cost, pairs) == Fraction.from_float(high) + Fraction.from_float(small)
            if return_cost:
                assert result[0] == high + small
    np.testing.assert_array_equal(cost, original)


@pytest.mark.parametrize('precision', ['native', 'float32', 'float64'])
def test_native_widened_arithmetic_preserves_rounded_half_limit(
        native_precision_family, precision):
    # Float32 rounds limit/2 up to .5, giving a unit penalty. Its two-edge
    # optimum then differs strictly from the optimum under the raw limit.
    # This guards against silently recomputing the penalty in widened precision.
    cost = np.array([[2**-27, .5], [.5, 2]], dtype=np.float32)
    limit = 1 - 2**-25
    working_dtype = np.float64 if precision == 'float64' else np.float32
    half = _half_penalty(limit, working_dtype)
    original = cost.copy()
    cost.setflags(write=False)
    for return_cost in (False, True):
        result = native_precision_family.single(
            cost, cost_limit=limit, precision=precision,
            output='pairs', return_cost=return_cost)
        pairs = _native_pairs(result, 'pairs', return_cost)
        _assert_native_optimum(cost, pairs, limit, working_dtype)
        assert len(pairs) == (1 if working_dtype == np.float64 else 2)
        assert _optional_objective(cost, pairs, half) == _optional_optimum(cost, half)
        if return_cost:
            assert result[0] == float(_exact_cost(cost, pairs))
    np.testing.assert_array_equal(cost, original)


@pytest.mark.parametrize('output', ['indices', 'pairs'])
@pytest.mark.parametrize('n_threads', [1, 2])
@pytest.mark.parametrize('precision', ['native', 'float32', 'float64'])
def test_native_rounded_half_limit_batches(
        native_precision_family, output, n_threads, precision):
    base = np.array([[2**-27, .5], [.5, 2]], dtype=np.float32)
    costs = np.stack((base, base[::-1], base[:, ::-1]))
    original = costs.copy()
    costs.setflags(write=False)
    limit = 1 - 2**-25
    working_dtype = np.float64 if precision == 'float64' else np.float32
    for return_cost in (False, True):
        result = native_precision_family.batch(
            costs, cost_limit=limit, precision=precision, output=output,
            return_cost=return_cost, n_threads=n_threads)
        totals = result[0] if return_cost else None
        if output == 'pairs':
            assignments = result[1] if return_cost else result
        else:
            rows, cols = result[1:] if return_cost else result
            assert len(rows) == len(cols) == len(costs)
            assignments = [np.column_stack((r, c)) for r, c in zip(rows, cols)]
        assert len(assignments) == len(costs)
        if return_cost:
            assert totals.shape == (len(costs),)
        for index, (cost, pairs) in enumerate(zip(costs, assignments)):
            _assert_native_optimum(cost, pairs, limit, working_dtype)
            assert len(pairs) == (1 if working_dtype == np.float64 else 2)
            if return_cost:
                assert totals[index] == float(_exact_cost(cost, pairs))
    np.testing.assert_array_equal(costs, original)


def _quantization_changes_optimum(transpose):
    eps = float(np.finfo(np.float32).eps)
    cost = np.array([[1 + 5 * eps / 8, 1., 4.],
                     [1 + 11 * eps / 8, 1 + 5 * eps / 8, 4.]], dtype=np.float64)
    return cost.T if transpose else cost


@pytest.mark.parametrize('output', ['indices', 'pairs'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('limit', [np.inf, 3.])
def test_native_quantized_objective_and_original_input_total(
        native_precision_family, output, transpose, limit):
    cost = _quantization_changes_optimum(transpose)
    original = cost.copy()
    cost.setflags(write=False)
    quantized = cost.astype(np.float32)
    result = native_precision_family.single(
        cost, extend_cost=True, cost_limit=limit, precision='float32', output=output)
    pairs = _native_pairs(result, output, True)
    _assert_native_optimum(quantized, pairs, limit, np.float32)
    assert result[0] == float(_exact_cost(cost, pairs))
    # This input has different strict optima before and after quantization.
    assert _exact_cost(cost, pairs) > _required_optimum(cost)
    for n_threads in (1, 2):
        costs = np.stack((cost, cost, cost))
        costs.setflags(write=False)
        batch = native_precision_family.batch(
            costs, extend_cost=True, cost_limit=limit, precision='float32',
            output=output, n_threads=n_threads)
        assert batch[0].shape == (len(costs),)
        if output == 'pairs':
            assignments = batch[1]
        else:
            rows, cols = batch[1:]
            assert len(rows) == len(cols) == len(costs)
            assignments = [np.column_stack((r, c)) for r, c in zip(rows, cols)]
        assert len(assignments) == len(costs)
        for index, assignment in enumerate(assignments):
            np.testing.assert_array_equal(assignment, pairs)
            assert batch[0][index] == float(_exact_cost(cost, assignment))
    np.testing.assert_array_equal(cost, original)
