"""Time assignment pairs using the settings from the pinned LAPX benchmark."""
import lap
import lapcpp
import laprust
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from _common import (cases, check_single, common_parser, environment, finish,
                     input_dtypes, measure_case, parse_args, print_header, print_records)


def solve_scipy(cost):
    rows, cols = linear_sum_assignment(cost)
    return np.column_stack((rows, cols))


def solve_lapcpp(cost):
    return lapcpp.lapjv(cost, extend_cost=cost.shape[0] != cost.shape[1],
                       return_cost=False, precision="native", output="pairs")


def solve_laprust(cost):
    return laprust.lapjv(cost, extend_cost=cost.shape[0] != cost.shape[1],
                        return_cost=False, precision="native", output="pairs")


def solve_lapjvxa(cost):
    return lap.lapjvxa(cost, extend_cost=cost.shape[0] != cost.shape[1], return_cost=False)


def solve_lapjvsa(cost):
    # Keep LAPX's default float32 working precision, as in the reference.
    return lap.lapjvsa(cost, extend_cost=cost.shape[0] != cost.shape[1], return_cost=False)


SOLVERS = {"scipy": solve_scipy, "lapcpp": solve_lapcpp, "laprust": solve_laprust,
           "lapjvxa": solve_lapjvxa, "lapjvsa": solve_lapjvsa}


def main():
    args = parse_args(common_parser(__doc__, repeats=3))
    metadata = environment("single", args, {"scipy": scipy, "lapcpp": lapcpp, "laprust": laprust, "lapx": lap},
                           {"scipy": "scipy.optimize.linear_sum_assignment",
                            "lapcpp": "lapcpp.lapjv", "laprust": "laprust.lapjv",
                            "lapjvxa": "lap.lapjvxa", "lapjvsa": "lap.lapjvsa"})
    print_header(metadata)
    records = []
    for dtype in input_dtypes(args):
        for shape in cases("single", args.quick):
            result = measure_case(SOLVERS, shape, dtype, args, check_single)
            print_records(result)
            records.extend(result)
    return finish(metadata, records, args.json)


if __name__ == "__main__":
    main()
