"""Time full assignment, post-filtering and unmatched lists, as in pinned LAPX."""
from functools import partial

import lap
import lapcpp
import laprust
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from _common import (assert_same_pairs, cases, checked_cost, common_parser, environment,
                     finish, input_dtypes, measure_case, parse_args, print_header, print_tracking_records)


def _decorate_return(n_rows, n_cols, matches):
    """Same output construction as LAPX v0.11.1's tracking benchmark."""
    matches = np.asarray(matches)
    if matches.size == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = matches.astype(int)
    if matches.size == 0:
        unmatched_rows = list(np.arange(n_rows))
        unmatched_cols = list(np.arange(n_cols))
    else:
        unmatched_rows = list(np.setdiff1d(np.arange(n_rows), matches[:, 0]))
        unmatched_cols = list(np.setdiff1d(np.arange(n_cols), matches[:, 1]))
    return matches, unmatched_rows, unmatched_cols


def solve_scipy(cost, threshold):
    rows, cols = linear_sum_assignment(cost)
    mask = cost[rows, cols] <= threshold
    matches = np.column_stack((rows[mask], cols[mask]))
    return _decorate_return(cost.shape[0], cost.shape[1], matches)


def solve_lapcpp(cost, threshold):
    matches = lapcpp.lapjv(cost, extend_cost=True, return_cost=False,
                          precision="native", output="pairs")
    mask = cost[matches[:, 0], matches[:, 1]] <= threshold
    return _decorate_return(cost.shape[0], cost.shape[1], matches[mask])


def solve_laprust(cost, threshold):
    matches = laprust.lapjv(cost, extend_cost=True, return_cost=False,
                           precision="native", output="pairs")
    mask = cost[matches[:, 0], matches[:, 1]] <= threshold
    return _decorate_return(cost.shape[0], cost.shape[1], matches[mask])


def solve_lapjvxa(cost, threshold):
    matches = lap.lapjvxa(cost, extend_cost=True, return_cost=False)
    mask = cost[matches[:, 0], matches[:, 1]] <= threshold
    return _decorate_return(cost.shape[0], cost.shape[1], matches[mask])


def solve_lapjvsa(cost, threshold):
    matches = lap.lapjvsa(cost, extend_cost=True, return_cost=False)
    mask = cost[matches[:, 0], matches[:, 1]] <= threshold
    return _decorate_return(cost.shape[0], cost.shape[1], matches[mask])


def check_tracking(cost, result, baseline, *, threshold):
    matches, unmatched_rows, unmatched_cols = result
    checked_cost(cost, matches, required=False)
    assert (cost[matches[:, 0], matches[:, 1]] <= threshold).all()
    assert_same_pairs(matches, baseline[0])
    assert isinstance(unmatched_rows, list) and isinstance(unmatched_cols, list)
    np.testing.assert_array_equal(unmatched_rows, np.setdiff1d(np.arange(cost.shape[0]), matches[:, 0]))
    np.testing.assert_array_equal(unmatched_cols, np.setdiff1d(np.arange(cost.shape[1]), matches[:, 1]))
    assert unmatched_rows == baseline[1] and unmatched_cols == baseline[2]


def main():
    parser = common_parser(__doc__, repeats=3)
    parser.add_argument("--threshold", nargs="+", type=float,
                        help="post-filter thresholds; default .05 .1 .5 1 1e9 (quick: .05 .5)")
    args = parse_args(parser)
    if args.threshold is None:
        args.threshold = [.05, .5] if args.quick else [.05, .1, .5, 1., 1e9]
    if not all(np.isfinite(value) for value in args.threshold):
        parser.error("thresholds must be finite")
    metadata = environment("tracking", args, {"scipy": scipy, "lapcpp": lapcpp, "laprust": laprust, "lapx": lap},
                           {"scipy": "scipy.optimize.linear_sum_assignment + post-filter",
                            "lapcpp": "lapcpp.lapjv + post-filter", "laprust": "laprust.lapjv + post-filter",
                            "lapjvxa": "lap.lapjvxa + post-filter", "lapjvsa": "lap.lapjvsa + post-filter"})
    print_header(metadata)
    records = []
    for threshold in args.threshold:
        solvers = {"scipy": partial(solve_scipy, threshold=threshold),
                   "lapcpp": partial(solve_lapcpp, threshold=threshold),
                   "laprust": partial(solve_laprust, threshold=threshold),
                   "lapjvxa": partial(solve_lapjvxa, threshold=threshold),
                   "lapjvsa": partial(solve_lapjvsa, threshold=threshold)}
        check = partial(check_tracking, threshold=threshold)
        for dtype in input_dtypes(args):
            group = []
            for shape in cases("tracking", args.quick):
                result = measure_case(solvers, shape, dtype, args, check, threshold=threshold)
                group.extend(result)
            print_tracking_records(group)
            records.extend(group)
    return finish(metadata, records, args.json)


if __name__ == "__main__":
    main()
