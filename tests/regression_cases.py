# Derived from LAPX/gatagat-lap test data. BSD-2-Clause; see tests/LICENSE.
import os
from gzip import GzipFile
import numpy as np


def make_hard(cost, lo, hi, random_state):
    hard = cost.copy()
    for row in range(hard.shape[0]):
        hard[row, :] += random_state.randint(lo, hi)
    for col in range(hard.shape[1]):
        hard[:, col] += random_state.randint(lo, hi)
    return hard

def get_dense_8x8_int():
    cost = np.array([[1000, 2, 11, 10, 8, 7, 6, 5],
                     [6, 1000, 1, 8, 8, 4, 6, 7],
                     [5, 12, 1000, 11, 8, 12, 3, 11],
                     [11, 9, 10, 1000, 1, 9, 8, 10],
                     [11, 11, 9, 4, 1000, 2, 10, 9],
                     [12, 8, 5, 2, 11, 1000, 11, 9],
                     [10, 11, 12, 10, 9, 12, 1000, 3],
                     [10, 10, 10, 10, 6, 3, 1, 1000]])
    opt = 17.
    return cost, opt

def get_dense_int(sz, rng, hard=True, seed=1299821):
    # Preserve the original seeded matrices without changing global RNG state.
    random_state = np.random.RandomState(seed)
    cost = random_state.randint(1, rng+1, size=(sz, sz))
    if hard is True:
        cost = make_hard(cost, 0, rng, random_state)
    return cost

def get_sparse_int(sz, rng, sparsity, hard=True, seed=1299821):
    random_state = np.random.RandomState(seed)
    cost = random_state.randint(1, rng+1, size=(sz, sz))
    if hard is True:
        cost = make_hard(cost, 0, rng, random_state)
    mask = random_state.rand(sz, sz)
    thresh = np.percentile(
            mask.flat, max(0, (sparsity - sz/float(sz*sz)) * 100.))
    mask = mask < thresh
    # Make sure there exists a solution.
    row = random_state.permutation(sz)
    col = random_state.permutation(sz)
    mask[row, col] = True
    return cost, mask

def get_dense_100x100_int():
    cost = get_dense_int(100, 100, hard=False, seed=1299821)
    opt = 198.
    return cost, opt

def get_dense_100x100_int_hard():
    cost = get_dense_int(100, 100, hard=True, seed=1299821)
    opt = 11399.
    return cost, opt

def get_sparse_100x100_int():
    cost, mask = get_sparse_int(100, 100, 0.04, seed=1299821)
    opt = 11406
    return cost, np.logical_not(mask), opt

def get_dense_1kx1k_int():
    cost = get_dense_int(1000, 100, hard=False, seed=1299821)
    opt = 1000.
    return cost, opt

def get_dense_1kx1k_int_hard():
    cost = get_dense_int(1000, 100, hard=True, seed=1299821)
    opt = 101078.0
    return cost, opt

def get_sparse_1kx1k_int():
    cost, mask = get_sparse_int(1000, 100, 0.01, seed=1299821)
    opt = 101078
    return cost, np.logical_not(mask), opt

def get_sparse_4kx4k_int():
    cost, mask = get_sparse_int(4000, 100, 0.004, seed=1299821)
    opt = 402541
    return cost, np.logical_not(mask), opt

def get_dense_eps():
    from pytest import approx
    datadir = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(datadir, 'cost_eps.csv.gz')
    with GzipFile(filename) as data:
        cost = np.genfromtxt(data, delimiter=",")
    opt = approx(224.8899507294651, 0.0000000000001)
    return cost, opt

SQUARE_CASES = [
    (np.array([[1000, 2, 11, 10, 8, 7, 6, 5],
               [6, 1000, 1, 8, 8, 4, 6, 7],
               [5, 12, 1000, 11, 8, 12, 3, 11],
               [11, 9, 10, 1000, 1, 9, 8, 10],
               [11, 11, 9, 4, 1000, 2, 10, 9],
               [12, 8, 5, 2, 11, 1000, 11, 9],
               [10, 11, 12, 10, 9, 12, 1000, 3],
               [10, 10, 10, 10, 6, 3, 1, 1000]]),
     (17.0, [1, 2, 0, 4, 5, 3, 7, 6], [2, 0, 1, 5, 3, 4, 7, 6])),
    # Solved in column reduction.
    (np.array([[1000, 4, 1],
               [1, 1000, 3],
               [5, 1, 1000]]),
     (3., [2, 0, 1], [1, 2, 0])),
    # Solved in augmenting row reduction.
    (np.array([[5, 1000, 3],
               [1000, 2, 2],
               [1, 5, 1000]]),
     (6., [2, 1, 0], [2, 1, 0])),
    # Needs augmentating row reduction - only a single row previously assigned.
    (np.array([[1000, 1000+1, 1000],
               [1000, 1000, 1000+1],
               [1, 2, 3]]),
     (1000+1000+1., [2, 1, 0], [2, 1, 0])),
    # Triggers the trackmate bug
    # Solution is ambiguous, [1, 0, 2] gives the same cost, depends on whether
    # in column reduction columns are iterated over from largest to smallest or
    # the other way around.
    (np.array([[10, 10, 13],
               [4, 8, 8],
               [8, 5, 8]]),
     (13+4+5, [2, 0, 1], [1, 2, 0])),
    (np.array([[11, 10,  6],
               [10, 11, 11],
               [11, 12, 15]]),
     (6+10+12, [2, 0, 1], [1, 2, 0])),
    (np.array([[12, 4, 9],
               [16, 15, 14],
               [19, 13, 17]]),
     (4+16+17, [1, 0, 2], [1, 0, 2])),
    (np.array([[2, 5, 7],
               [7, 10, 12],
               [1, 5, 9]]),
     (7+10+1, [2, 1, 0], [2, 1, 0])),
    # This triggered error in augmentation.
    (np.array([[10, 6, 14, 1],
               [17, 18, 17, 15],
               [14, 17, 15, 8],
               [11, 13, 11, 4]]),
     (6+17+14+4, [1, 2, 0, 3], [2, 0, 1, 3])),
    # Test matrix from centrosome
    (np.array([[10, 10, 13],
               [4, 8, 8],
               [8, 5, 8]]),
     (22., [2, 0, 1], [1, 2, 0])),
    # Test matrix from centrosome
    (np.array([[2, 5, 7],
               [7, 10, 12],
               [1, 5, 9]]),
     (18., [2, 1, 0], [2, 1, 0])),
    ]

SPARSE_CASES = [
    (np.array([[11.,  20.,  np.inf,  np.inf,  np.inf],
               [12.,  np.inf,  12.,  np.inf,  np.inf],
               [np.inf,  11.,  10.,  15.,   9.],
               [15.,  np.inf,  np.inf,  22.,  np.inf],
               [13.,  np.inf,  np.inf,  np.inf,  15.]], dtype=float),
     (11+12+11+22+15, [0, 2, 1, 3, 4], [0, 2, 1, 3, 4])),
    ]

def arr_loop_cost():
    shape = (7, 3)
    cc = np.array([
        2.593883482138951146e-01, 3.080381437461217620e-01,
        1.976243020727339317e-01, 2.462740976049606068e-01,
        4.203993396282833528e-01, 4.286184525458427985e-01,
        1.706431415909629434e-01, 2.192929371231896185e-01,
        2.117769622802734286e-01, 2.604267578125001315e-01])
    ii = np.array([0, 0, 1, 1, 2, 2, 5, 5, 6, 6])
    jj = np.array([0, 1, 0, 1, 1, 2, 0, 1, 0, 1])
    cost = np.empty(shape)
    cost[:] = 1000.
    cost[ii, jj] = cc
    return cost


def large_cost_case(shape, dtype, case=0, forbidden=False):
    # These values are exactly representable in all four dtypes, including
    # float32. The unique optimum separates assignment errors from rounding.
    low, high = [(1_000_000_000, 2_000_000_000),
                 (500_000_000, 2_000_000_000), (-1_000_000_000, 0)][case]
    cost = np.full(shape, np.inf if forbidden else high, dtype=dtype)
    k = min(shape)
    rows = np.roll(np.arange(shape[0]), case)[:k]
    cols = np.roll(np.arange(shape[1]), -case)[:k]
    cost[rows, cols] = low
    pairs = np.column_stack((rows, cols))
    pairs = pairs[np.argsort(rows)]
    return cost, k * float(low), pairs
