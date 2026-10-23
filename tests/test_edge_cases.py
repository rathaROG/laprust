"""Public API boundaries, ownership, and independent small-problem checks."""
import gc
import inspect
import sys
from itertools import product
import numpy as np
import pytest

from assignment_checks import assert_same, exhaustive_optimum, normalize, small_optimum, validate


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("batch_size", [None, 0, 1, 3])
@pytest.mark.parametrize("output", ["indices", "pairs"])
@pytest.mark.parametrize("return_cost", [False, True])
@pytest.mark.parametrize("precision", ["native", "float32", "float64"])
def test_empty_return_contract(solver, shape, batch_size, output, return_cost, precision):
    batch = batch_size is not None
    cost = np.empty((batch_size,) + shape if batch else shape, dtype=np.float32)
    solve = solver.lapjv_batch if batch else solver.lapjv
    result = solve(cost, output=output, return_cost=return_cost, precision=precision)
    if return_cost:
        assert type(result) is tuple
        total = result[0]
        if batch:
            assert total.shape == (batch_size,) and total.dtype == np.float64
            np.testing.assert_array_equal(total, np.zeros(batch_size))
        else:
            assert type(total) is float and total == 0.
        result = result[1] if output == "pairs" else result[1:]
    if output == "indices":
        assert type(result) is tuple and len(result) == 2
        groups = result
    else:
        groups = (result,)
    arrays = []
    for group in groups:
        if batch:
            assert type(group) is list and len(group) == batch_size
            arrays.extend(group)
        else:
            arrays.append(group)
    assert len({id(a) for a in arrays}) == len(arrays)
    for a in arrays:
        assert type(a) is np.ndarray and a.dtype == np.int64
        assert a.shape == ((0, 2) if output == "pairs" else (0,))


@pytest.mark.parametrize("guard", ["input", "output"])
def test_batch_byte_size_guards_before_copy(solver, guard):
    # uint8 zero-stride views fit NumPy's own size limit, while the solver's
    # float64 input/int64 output would overflow. No giant allocation is made.
    width = 1 << min(30, sys.maxsize.bit_length() // 2)
    shape = (1, width, width) if guard == "input" else (sys.maxsize // 16 + 1, 1, 1)
    costs = np.broadcast_to(np.uint8(0), shape)
    with pytest.raises(ValueError, match="Batch " + guard + " is too large"):
        solver.lapjv_batch(costs, extend_cost=True)
    assert solver.lapjv([[0.]])[0] == 0.


@pytest.mark.skipif(sys.maxsize <= 2**31 - 1, reason="NumPy cannot represent dimensions above int32 on a 32-bit host")
@pytest.mark.parametrize("shape", [(1, 2**31, 1), (1, 1, 2**31)])
def test_batch_dimension_guards_before_copy(solver, shape):
    with pytest.raises(ValueError, match="int32"):
        solver.lapjv_batch(np.broadcast_to(np.uint8(0), shape), extend_cost=True)


def result_arrays(result):
    if isinstance(result, np.ndarray):
        return [result]
    if isinstance(result, (tuple, list)):
        return [array for value in result for array in result_arrays(value)]
    return []


@pytest.mark.parametrize("batch", [False, True])
def test_exact_signature_defaults_and_parameter_kinds(solver, batch):
    function = solver.lapjv_batch if batch else solver.lapjv
    expected = [("costs" if batch else "cost", inspect.Parameter.empty),
                ("extend_cost", False), ("cost_limit", np.inf), ("return_cost", True)]
    if batch:
        expected += [("n_threads", 0)]
    expected += [("precision", "native"), ("output", "indices")]
    parameters = inspect.signature(function).parameters
    assert list(parameters) == [name for name, _ in expected]
    for name, default in expected:
        assert parameters[name].default == default
        assert parameters[name].kind == (inspect.Parameter.KEYWORD_ONLY if name in ("precision", "output")
                                         else inspect.Parameter.POSITIONAL_OR_KEYWORD)
    assert function.__name__ == ("lapjv_batch" if batch else "lapjv")
    assert "Parameters\n" in function.__doc__ and "Returns\n" in function.__doc__


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.uint64, np.int32, np.int64,
                                   np.float16, np.float32, np.float64])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_positional_keyword_and_duplicate_arguments(solver, dtype, batch, output):
    a = np.array([[9, 1, 8], [1, 9, 8], [8, 8, 2]], dtype=dtype)
    a = np.stack([a] * 2) if batch else a
    name = "costs" if batch else "cost"
    solve = solver.lapjv_batch if batch else solver.lapjv
    for return_cost in (True, False):
        args = [a, False, np.inf, return_cost] + ([2] if batch else [])
        options = {name: a, "extend_cost": False, "cost_limit": np.inf,
                   "return_cost": return_cost, "precision": "native", "output": output}
        if batch:
            options["n_threads"] = 2
        assert_same(solve(*args, precision="native", output=output), solve(**options))
    with pytest.raises(TypeError):
        solve(a, **{name: a})
    with pytest.raises(TypeError):
        solve(**{("cost" if batch else "costs"): a})
    with pytest.raises(TypeError):
        solve(*([a, False, np.inf, True] + ([1] if batch else []) + ["float32"]))


@pytest.mark.parametrize("dtype", ["O", "U3", "S3", "complex64", "datetime64[D]", "timedelta64[D]"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_unsupported_dtype_even_for_empty_input(solver, dtype, empty, batch):
    shape = ((0 if empty else 2), 2, 2) if batch else ((0 if empty else 2), 2)
    cost = np.zeros(shape, dtype=dtype)
    with pytest.raises(TypeError, match="real numeric"):
        (solver.lapjv_batch if batch else solver.lapjv)(cost, extend_cost=True)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("kwargs", [{"cost_limit": np.nan}, {"cost_limit": -np.inf},
                                    {"precision": "float16"}, {"output": "mappings"}])
def test_options_validated_on_empty_and_nonempty_calls(solver, batch, empty, kwargs):
    cost = np.zeros((0 if empty else 2, 2, 2) if batch else (0 if empty else 2, 2))
    with pytest.raises(ValueError):
        (solver.lapjv_batch if batch else solver.lapjv)(cost, **kwargs)


@pytest.mark.parametrize("count", [np.nan, np.inf, -np.inf, "invalid", [], {}])
@pytest.mark.parametrize("size", [0, 1, 3])
def test_bad_worker_counts_fail_before_batch_work(solver, count, size):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        solver.lapjv_batch(np.zeros((size, 2, 2)), n_threads=count)
    np.testing.assert_array_equal(solver.lapjv_batch(np.zeros((2, 2, 2)), n_threads=2)[0], [0., 0.])


@pytest.mark.parametrize("layout", ["fortran", "negative-stride", "swapped", "unaligned", "broadcast"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_batch_layout_and_result_lifetime(solver, layout, dtype, output):
    costs = np.random.default_rng(121).integers(-8, 33, size=(3, 5, 7)).astype(dtype) / 8
    if layout == "fortran":
        costs = np.asfortranarray(costs)
    elif layout == "negative-stride":
        costs = costs[::-1, ::-1, ::-1]
    elif layout == "swapped":
        costs = costs.astype(costs.dtype.newbyteorder("S"))
    elif layout == "unaligned":
        aligned = costs
        costs = np.ndarray(aligned.shape, dtype=dtype, buffer=bytearray(aligned.nbytes + 1), offset=1)
        costs[:] = aligned
    else:
        costs = np.broadcast_to(costs[:1], costs.shape)
    before = costs.copy()
    costs.setflags(write=False)
    result = solver.lapjv_batch(costs, extend_cost=True, output=output, cost_limit=.5, n_threads=2)
    totals, assignments = normalize(result, output, batch=True)
    # Inspect original returned buffers; column_stack used for validation copies indices.
    arrays = result_arrays(result)
    saved = [array.copy() for array in arrays]
    for i, array in enumerate(arrays):
        assert not np.shares_memory(array, costs)
        for sibling in arrays[i + 1:]:
            assert not np.shares_memory(array, sibling)
    for a, total, pairs in zip(before, totals, assignments):
        # Convert endian metadata for the reference arithmetic only.
        native = a.astype(dtype)
        validate(native, total, pairs, .5, optimum=small_optimum(native, .5), atol=0)
    np.testing.assert_array_equal(costs, before)
    del costs
    gc.collect()
    for n in (1, 64, 2, 31):
        solver.lapjv_batch(np.zeros((4, n, n)), n_threads=3, output=output)
    for array, expected in zip(arrays, saved):
        np.testing.assert_array_equal(array, expected)
    # Mutating an owned result must not corrupt sibling results or the input.
    for i, array in enumerate(arrays):
        if array.size:
            array.flat[0] = -999
            for j, sibling in enumerate(arrays):
                if i != j:
                    np.testing.assert_array_equal(sibling, saved[j])
            array[:] = saved[i]


@pytest.mark.parametrize("output", ["indices", "pairs"])
def test_single_result_survives_input_mutation_and_workspace_reuse(solver, output):
    cost = np.array([[9., 1.], [2., 9.]])
    result = solver.lapjv(cost, output=output)
    total = result[0]
    arrays = result_arrays(result)
    saved = [array.copy() for array in arrays]
    for i, array in enumerate(arrays):
        assert not np.shares_memory(array, cost)
        for sibling in arrays[i + 1:]:
            assert not np.shares_memory(array, sibling)
    cost.fill(100)
    del cost
    gc.collect()
    for size in (32, 1, 100, 2):
        solver.lapjv(np.zeros((size, size)), output=output)
    assert total == 3.
    for array, expected in zip(arrays, saved):
        np.testing.assert_array_equal(array, expected)


@pytest.mark.parametrize("shape", [(4, 4), (4, 5), (5, 4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_required_optima_by_full_enumeration(solver, shape, dtype):
    cost = np.random.default_rng(12345).integers(-8, 32, size=shape).astype(dtype) / 8
    optimum = exhaustive_optimum(cost)
    for return_cost, output in product((True, False), ("indices", "pairs")):
        total, pairs = normalize(solver.lapjv(cost, extend_cost=True, return_cost=return_cost,
                                           output=output), output, return_cost)
        validate(cost, total, pairs, optimum=optimum, atol=0)
        totals, assignments = normalize(solver.lapjv_batch(np.stack([cost] * 2), extend_cost=True,
            n_threads=2, return_cost=return_cost, output=output), output, return_cost, True)
        for i, pairs in enumerate(assignments):
            validate(cost, None if totals is None else totals[i], pairs, optimum=optimum, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (3, 3)])
def test_rejection_enumeration_permutations_and_objective_preserving_shift(solver, dtype, shape):
    rng = np.random.default_rng(931)
    # Exact binary fractions include ties, negative limits and forbidden edges.
    for limit in (-.5, -0., .5, 2.):
        cost = rng.integers(-8, 17, size=shape).astype(dtype) / 8
        cost[0, -1] = np.inf
        variants = [(cost, limit), (cost + dtype(4), limit + 4), (cost.T, limit)]
        rows, cols = rng.permutation(shape[0]), rng.permutation(shape[1])
        variants.append((cost[np.ix_(rows, cols)], limit))
        original_optimum = exhaustive_optimum(cost, limit)
        for index, (value, bound) in enumerate(variants):
            optimum = exhaustive_optimum(value, bound)
            assert optimum == original_optimum + (2 * sum(shape) if index == 1 else 0)
            total, pairs = solver.lapjv(value, cost_limit=bound, output="pairs")
            validate(value, total, pairs, bound, optimum=optimum, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("variant", ["hall-zero", "hall-weighted", "column", "row", "all"])
def test_historical_infeasible_graphs_and_recovery(solver, dtype, variant):
    if variant.startswith("hall"):
        cost = np.full((5, 5), np.inf, dtype=dtype)
        cost[np.ix_([0, 4], [0, 1, 2])] = [[0, 0, 0], [0, 0, 0]]
        cost[1:4, 3:] = 0
        if variant == "hall-weighted":
            cost[np.ix_([0, 4], [0, 1, 2])] = [[19, 22, 16], [11, 14, 13]]
            cost[1:4, 3:] = [[4, 13], [3, 14], [10, 12]]
    else:
        if variant == "column":
            cost = np.array([[0, np.inf, 0, 0, np.inf], [np.inf, np.inf, 0, 0, 0],
                             [np.inf, np.inf, np.inf, 0, np.inf],
                             [np.inf, np.inf, np.inf, 0, 0], [0, np.inf, 0, np.inf, np.inf]], dtype=dtype)
        elif variant == "row":
            cost = np.array([[0, 0, 0, 0, np.inf], [np.inf, np.inf, 0, 0, 0],
                             [np.inf] * 5, [np.inf, np.inf, np.inf, 0, 0],
                             [0, 0, 0, np.inf, np.inf]], dtype=dtype)
        else:
            cost = np.full((5, 5), np.inf, dtype=dtype)
    with pytest.raises(ValueError, match="infeasible"):
        solver.lapjv(cost)
    batch = np.zeros((4, 5, 5), dtype=dtype)
    batch[1] = batch[3] = cost
    with pytest.raises(ValueError, match="batch index 1"):
        solver.lapjv_batch(batch, n_threads=3)
    # Optional rejection remains feasible; compare its full objective with SciPy.
    total, pairs = solver.lapjv(cost, cost_limit=1., output="pairs")
    validate(cost, total, pairs, 1., optimum=small_optimum(cost, 1.), atol=0)
    np.testing.assert_array_equal(solver.lapjv_batch(np.zeros_like(batch), n_threads=3)[0], np.zeros(4))
