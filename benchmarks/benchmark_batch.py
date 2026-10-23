"""Time a serial SciPy loop against the four dedicated native batch APIs."""
from functools import partial

import lap
import lapcpp
import laprust
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from _common import (cases, check_batch, common_parser, environment, finish,
                     input_dtypes, measure_case, parse_args, print_header, print_records)


def solve_scipy(costs, *, return_cost=False):
    totals = np.empty(len(costs), dtype=np.float64) if return_cost else None
    assignments = []
    for index, cost in enumerate(costs):
        rows, cols = linear_sum_assignment(cost)
        if totals is not None:
            totals[index] = cost[rows, cols].sum(dtype=np.float64)
        assignments.append(np.stack([rows, cols], axis=1))
    return (totals, assignments) if totals is not None else assignments


def solve_lapcpp(costs, *, n_threads=0, return_cost=False):
    return lapcpp.lapjv_batch(costs, extend_cost=costs.shape[1] != costs.shape[2],
                             n_threads=n_threads, return_cost=return_cost,
                             precision="native", output="pairs")


def solve_laprust(costs, *, n_threads=0, return_cost=False):
    return laprust.lapjv_batch(costs, extend_cost=costs.shape[1] != costs.shape[2],
                              n_threads=n_threads, return_cost=return_cost,
                              precision="native", output="pairs")


def solve_lapjvxa(costs, *, n_threads=0, return_cost=False):
    return lap.lapjvxa_batch(costs, extend_cost=costs.shape[1] != costs.shape[2],
                            n_threads=n_threads, return_cost=return_cost)


def solve_lapjvsa(costs, *, n_threads=0, return_cost=False):
    return lap.lapjvsa_batch(costs, extend_cost=costs.shape[1] != costs.shape[2],
                            n_threads=n_threads, return_cost=return_cost)


def main():
    parser = common_parser(__doc__, repeats=2)
    parser.add_argument("--threads", type=int, default=0, help="native batch workers; 0 uses each library's automatic policy")
    parser.add_argument("--cost", action="store_true", help="include total-cost calculation for every candidate")
    args = parse_args(parser)
    if args.threads < 0:
        parser.error("threads must be nonnegative")
    solvers = {"scipy": partial(solve_scipy, return_cost=args.cost),
               "lapcpp": partial(solve_lapcpp, n_threads=args.threads, return_cost=args.cost),
               "laprust": partial(solve_laprust, n_threads=args.threads, return_cost=args.cost),
               "lapjvxa": partial(solve_lapjvxa, n_threads=args.threads, return_cost=args.cost),
               "lapjvsa": partial(solve_lapjvsa, n_threads=args.threads, return_cost=args.cost)}
    metadata = environment("batch", args, {"scipy": scipy, "lapcpp": lapcpp, "laprust": laprust, "lapx": lap},
                           {"scipy": "serial scipy.optimize.linear_sum_assignment loop",
                            "lapcpp": "lapcpp.lapjv_batch", "laprust": "laprust.lapjv_batch",
                            "lapjvxa": "lap.lapjvxa_batch", "lapjvsa": "lap.lapjvsa_batch"})
    print_header(metadata)
    check = partial(check_batch, return_cost=args.cost)
    records = []
    for dtype in input_dtypes(args):
        for shape in cases("batch", args.quick):
            result = measure_case(solvers, shape, dtype, args, check, batch=True)
            print_records(result, batch=True)
            records.extend(result)
    return finish(metadata, records, args.json)


if __name__ == "__main__":
    main()
