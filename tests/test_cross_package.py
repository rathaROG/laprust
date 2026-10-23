"""Mixed C++/Rust use in one interpreter, including concurrent worker pools."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import inspect
import subprocess
import sys
import numpy as np
import pytest

from assignment_checks import assert_same, normalize, scipy_optimum, validate

pytestmark = pytest.mark.cross_package


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_original_cpp_rust_seeded_equivalence(both_packages, dtype):
    """Retain the experiment's exact cross-language cases and pool-size changes."""
    cpp, rust = both_packages
    rng = np.random.default_rng(20261021)
    jobs = []
    for shape in ((1, 1), (17, 17), (129, 129), (129, 137), (137, 129), (20, 500), (500, 20)):
        for limit in (np.inf, .2, -.3, 0.):
            for tied in (False, True):
                costs = (rng.random((4,) + shape) * 2 - 1).astype(dtype)
                if tied:
                    costs = np.round(costs * 4) / 4
                costs[:, 0, -1] = np.inf
                if shape == (1, 1):
                    costs[:, 0, 0] = 0
                costs.setflags(write=False)
                expected = rust.lapjv_batch(costs, extend_cost=True, cost_limit=limit,
                                           output="pairs", n_threads=1)
                jobs.append((costs, limit, expected))

    def solve(job):
        costs, limit, expected = job
        for threads in (1, 0, 2, 3, 4):
            for package in (cpp, rust):
                actual = package.lapjv_batch(costs, extend_cost=True, cost_limit=limit,
                                            output="pairs", n_threads=threads)
                assert_same(actual, expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(solve, jobs))


def test_public_contracts_match_without_forwarding(both_packages):
    cpp, rust = both_packages
    for name in ("lapjv", "lapjv_batch"):
        a, b = getattr(cpp, name), getattr(rust, name)
        assert a is not b
        assert inspect.signature(a) == inspect.signature(b)
        assert a.__module__ == "lapcpp._lapjv" and b.__module__ == "laprust._lapjv"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("output", ["indices", "pairs"])
@pytest.mark.parametrize("return_cost", [False, True])
def test_alternating_and_concurrent_packages(both_packages, dtype, output, return_cost):
    rng = np.random.default_rng(20261021)
    problems = []
    for shape, limit in [((96, 96), np.inf), ((33, 27), np.inf), ((27, 33), np.inf),
                         ((31, 31), .25), ((20, 40), -.25)]:
        costs = rng.integers(-8, 33, size=(3,) + shape).astype(dtype) / 8
        if shape == (96, 96):
            costs[:, :70] = costs[:, :1]  # Force square completion and tied rows.
        costs.setflags(write=False)
        expected = {}
        for module in both_packages:
            result = module.lapjv_batch(costs, extend_cost=True, cost_limit=limit,
                n_threads=1, output=output, return_cost=return_cost)
            totals, assignments = normalize(result, output, return_cost, True)
            for i, (cost, pairs) in enumerate(zip(costs, assignments)):
                validate(cost, None if totals is None else totals[i], pairs, limit,
                         optimum=scipy_optimum(cost, limit), atol=0)
            expected[module.__name__] = result
        # Precision and traversal order are preserved by the C++ port.
        assert_same(expected["lapcpp"], expected["laprust"])
        problems.append((costs, limit, expected))

    jobs = [(module, problem, threads) for threads in (1, 0, 2, 3)
            for problem in problems for module in both_packages]
    barrier = Barrier(4)

    def run(job, synchronize=False):
        module, (costs, limit, expected), threads = job
        if synchronize:
            barrier.wait(timeout=30)
        result = module.lapjv_batch(costs, extend_cost=True, cost_limit=limit,
                                   n_threads=threads, output=output, return_cost=return_cost)
        assert_same(result, expected[module.__name__])
        # Also alternate the single APIs in the same worker threads.
        single = module.lapjv(costs[0], extend_cost=True, cost_limit=limit,
                             output=output, return_cost=return_cost)
        total, pairs = normalize(single, output, return_cost)
        validate(costs[0], total, pairs, limit, optimum=scipy_optimum(costs[0], limit), atol=0)

    for job in jobs:
        run(job)
    assert len(jobs) % 4 == 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run, job, True) for job in reversed(jobs)]
        for future in futures:
            future.result(timeout=120)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_concurrent_failure_isolation_and_pool_recovery(both_packages, dtype):
    good = np.random.default_rng(92).random((5, 33, 27)).astype(dtype)
    bad = good.copy()
    bad[2] = bad[4] = np.inf
    good.setflags(write=False)
    bad.setflags(write=False)
    wanted = [m.lapjv_batch(good, extend_cost=True, output="pairs") for m in both_packages]
    barrier = Barrier(4)

    def run(index, fail):
        module = both_packages[index]
        barrier.wait(timeout=30)
        if fail:
            with pytest.raises(ValueError, match="batch index 2"):
                module.lapjv_batch(bad, extend_cost=True, n_threads=3)
        else:
            assert_same(module.lapjv_batch(good, extend_cost=True, n_threads=2, output="pairs"), wanted[index])
        assert_same(module.lapjv_batch(good, extend_cost=True, n_threads=3, output="pairs"), wanted[index])

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run, index, fail) for index in (0, 1) for fail in (False, True)]
        for future in futures:
            future.result(timeout=120)


@pytest.mark.parametrize("order", [("lapcpp", "laprust"), ("laprust", "lapcpp")])
def test_import_orders_and_pool_shutdown(both_packages, order, tmp_path):
    script = '''
import importlib
import importlib.abc
import sys
import numpy as np
class BlockLap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'lap':
            raise AssertionError('Unexpected LAPX dependency')
sys.meta_path.insert(0, BlockLap())
modules = [importlib.import_module(name) for name in sys.argv[1:]]
for module in modules:
    assert module.__all__ == ['lapjv', 'lapjv_batch']
    for shape in ((8,8),(8,12),(12,8)):
        data = np.random.default_rng(17).random((5,)+shape)
        for limit in (np.inf, .25):
            for threads in (1,2,3,4,2):
                module.lapjv_batch(data, extend_cost=True, cost_limit=limit, n_threads=threads)
assert modules[0].lapjv is not modules[1].lapjv
print('clean shutdown')
'''
    result = subprocess.run([sys.executable, "-c", script, *order], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean shutdown" in result.stdout
