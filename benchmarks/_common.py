"""Timing, input generation, validation and reporting shared by the benchmarks.

Solver calls belong in the three entry points. Inputs, workloads and checks
follow the LAPX v0.11.1 release; its timing is
available with --reference-timing.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from unicodedata import combining, east_asian_width

import numpy as np

REFERENCE_RELEASE = "v0.11.1"
CANDIDATES = ("scipy", "lapcpp", "laprust", "lapjvxa", "lapjvsa")
POSITION_VERDICTS = {0: "🥇", 1: "🥈", 2: "🥉", 3: "🚩", 4: "🏳️",
                     5: "🥴", 6: "🗿", 7: "🙅‍♂️"}


def common_parser(description, repeats):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="small smoke/CI workloads")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=repeats)
    parser.add_argument("--calls", type=int, default=0,
                        help="fixed calls per timed sample; 0 calibrates calls to --min-time "
                             "(or uses the reference defaults with --reference-timing)")
    parser.add_argument("--min-time", type=float, default=0.01,
                        help="calibrated timing: minimum seconds per sample; default 0.01")
    parser.add_argument("--fast-rounds", type=int, default=7,
                        help="calibrated timing: rounds for cases whose slowest call is under 1 ms; default 7")
    parser.add_argument("--reference-timing", action="store_true",
                        help="LAPX v0.11.1 timing: 1 call per sample above 64 columns or in batches, "
                             "25 otherwise, every solver warmed once before the timed calls")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="float64",
                        help="input dtype; float64 matches the reference benchmark")
    parser.add_argument("--verbose", action="store_true", help="print full environment metadata")
    parser.add_argument("--json", type=Path, help="save samples, settings, checks and summaries")
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.seed < 0 or args.repeats < 1 or args.calls < 0:
        parser.error("seed/calls must be nonnegative and repeats must be positive")
    if not args.min_time > 0 or args.fast_rounds < 1:
        parser.error("min-time and fast-rounds must be positive")
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8")
    return args


def input_dtypes(args):
    return ("float32", "float64") if args.dtype == "both" else (args.dtype,)


def make_costs(shape, seed, round_index, dtype):
    # Identical seeds and uniform costs to the pinned single/tracking/batch code.
    return np.random.default_rng([seed + round_index, *shape]).random(shape).astype(dtype, copy=False)


def checked_cost(cost, pairs, *, required=True):
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert np.issubdtype(pairs.dtype, np.integer)
    assert (pairs >= 0).all()
    assert (pairs[:, 0] < cost.shape[0]).all() and (pairs[:, 1] < cost.shape[1]).all()
    assert len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1])) == len(pairs)
    if required:
        assert len(pairs) == min(cost.shape)
    selected = cost[tuple(pairs.T)]
    assert np.isfinite(selected).all()
    return selected.sum(dtype=np.float64)


def assert_same_pairs(actual, expected):
    # The reference compares sets. Sorting/copying for checks is never timed.
    np.testing.assert_array_equal(actual[np.lexsort((actual[:, 1], actual[:, 0]))],
                                  expected[np.lexsort((expected[:, 1], expected[:, 0]))])


def check_single(cost, result, baseline):
    checked_cost(cost, result)
    assert_same_pairs(result, baseline)


def check_batch(costs, result, baseline, *, return_cost=False):
    pairs = result[1] if return_cost else result
    expected = baseline[1] if return_cost else baseline
    assert len(pairs) == len(expected) == len(costs)
    totals = np.array([checked_cost(cost, p) for cost, p in zip(costs, pairs)])
    optimum = np.array([checked_cost(cost, p) for cost, p in zip(costs, expected)])
    # Same tolerances as checked_assignment_costs/measure in the pinned batch file.
    np.testing.assert_allclose(totals, optimum, rtol=1e-6, atol=1e-8)
    if return_cost:
        assert result[0].shape == (len(costs),)
        np.testing.assert_allclose(result[0], totals, rtol=1e-6, atol=1e-8)


def measure_case(solvers, shape, dtype, args, check, *, batch=False, threshold=None):
    """Time already-bound dedicated functions; check outputs after all timings.

    Calibrated timing (default) warms each solver immediately before its own
    sample and repeats calls until a sample lasts at least --min-time, so
    microsecond cases are not single cold calls. --reference-timing keeps the
    pinned LAPX settings for comparison with its published numbers.
    """
    names = list(solvers)
    assert tuple(names) == CANDIDATES
    input_shape = shape if batch else shape[1:]
    reference = getattr(args, "reference_timing", False)
    if args.calls or reference:
        fixed = args.calls or (1 if batch or max(shape[1:]) > 64 else 25)
        calls = dict.fromkeys(names, fixed)
    else:
        calls = dict.fromkeys(names)  # Calibrated per solver in the first round.
    repeats = 1 if max(shape[1:]) >= 5000 else args.repeats
    order_rng = np.random.default_rng([args.seed, *input_shape, 1])
    samples = {name: [] for name in names}
    round_index = 0
    while round_index < repeats:
        costs = make_costs(input_shape, args.seed, round_index, dtype)
        if reference:
            warm = {name: solve(costs) for name, solve in solvers.items()}
            for name in names:
                check(costs, warm[name], warm["scipy"])
            del warm
        outputs = {}
        for index in order_rng.permutation(len(names)):
            name = names[index]
            solve = solvers[name]  # Resolve dispatch before starting the clock.
            if not reference:
                solve(costs)  # Untimed warm-up right before this solver's sample.
                if calls[name] is None:
                    start = time.perf_counter()
                    solve(costs)
                    once = time.perf_counter() - start
                    calls[name] = min(1_000_000, max(1, int(np.ceil(args.min_time / max(once, 1e-9)))))
            start = time.perf_counter()
            for _ in range(calls[name]):
                output = solve(costs)
            elapsed = (time.perf_counter() - start) * 1000 / calls[name]
            samples[name].append(elapsed)
            outputs[name] = output
        for name in names:
            try:
                check(costs, outputs[name], outputs["scipy"])
            except AssertionError as exc:
                raise AssertionError(f"{name}: failed check for {shape}, {dtype}, threshold={threshold}") from exc
        if round_index == 0 and not reference and repeats > 1:
            # Sub-millisecond cases are cheap; more rounds steady their medians.
            if max(samples[name][0] for name in names) < 1.0:
                repeats = max(repeats, getattr(args, "fast_rounds", repeats))
        round_index += 1
    baseline_ms = float(np.median(samples["scipy"]))
    records = []
    for name in names:
        median = float(np.median(samples[name]))
        record = dict(dtype=dtype, shape=list(shape), candidate=name, median_ms=median,
                      samples_ms=samples[name], speedup=baseline_ms / median,
                      calls_per_round=calls[name], check="PASS", verdict=verdict(name, median, baseline_ms))
        if threshold is not None:
            record["threshold"] = threshold
        records.append(record)
    return records


def environment(kind, args, modules, solver_calls):
    provenance = importlib.metadata.distribution("lapx").read_text("direct_url.json")
    return dict(schema_version=3, reference_release=REFERENCE_RELEASE, kind=kind,
                input_model="uniform [0, 1)", numpy=np.__version__, python=sys.version,
                platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                versions={name: module.__version__ for name, module in modules.items()},
                modules={name: str(module.__file__) for name, module in modules.items()},
                lapx_source=json.loads(provenance) if provenance else None,
                solver_calls=solver_calls,
                working_precision={"scipy": "float64", "lapcpp": "native input dtype",
                                   "laprust": "native input dtype", "lapjvxa": "float64",
                                   "lapjvsa": "LAPX default float32"},
                settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})


def print_header(metadata):
    kind = metadata["kind"]
    settings = metadata["settings"]
    print(f"\n{kind.capitalize()} assignment benchmark")
    versions = " | ".join(f"{name} {version}" for name, version in metadata["versions"].items())
    print(f"Python {metadata['python'].split()[0]} | NumPy {metadata['numpy']} | {versions}")
    print(f"Inputs: {metadata['input_model']} | dtype: {settings['dtype']} | seed: {settings['seed']}")
    if settings.get("reference_timing"):
        print(f"Timing: LAPX reference | rounds: {settings['repeats']} | "
              f"calls/round: {settings['calls'] or 'auto'} | median time in ms")
    else:
        calls = settings['calls'] or f"calibrated to >= {settings.get('min_time', .01) * 1000:g} ms per sample"
        print(f"Timing: warm-up before each sample | rounds: {settings['repeats']} "
              f"({settings.get('fast_rounds', 7)} for sub-millisecond cases) | calls: {calls} | median time in ms")
    print("JVS: default float32 | lapcpp/laprust: native precision | JVX/SciPy: float64")
    if kind == "batch":
        workers = settings['threads'] or 'auto'
        print(f"Native workers: {workers} | SciPy: serial loop | return_cost={settings['cost']}")
    elif kind == "tracking":
        print("Full assignment, then post-filter and construct unmatched lists.")
    print(f"LAPX reference: {metadata['reference_release']}")
    if settings.get("verbose"):
        print("\nFull environment metadata:")
        print(json.dumps(metadata, indent=2))


def display_width(text):
    """Terminal columns for our ASCII labels, emoji and joined rank symbols."""
    width = cluster_width = 0
    joined = False
    for char in text:
        if char == "\u200d":
            joined = True
        elif char == "\ufe0f":
            # Emoji presentation (e.g. the white flag) occupies two columns.
            width += max(0, 2 - cluster_width)
            cluster_width = max(2, cluster_width)
        elif combining(char) or char == "\ufe0e":
            continue
        else:
            char_width = 2 if east_asian_width(char) in ("W", "F") else 1
            if joined:
                width += max(0, char_width - cluster_width)
                cluster_width = max(cluster_width, char_width)
                joined = False
            else:
                width += char_width
                cluster_width = char_width
    return width


def _pad(text, width, align="<"):
    padding = " " * max(0, width - display_width(text))
    return padding + text if align == ">" else text + padding


def _print_table(header, rows, alignments, separator="  "):
    widths = [max(display_width(cell) for cell in column) for column in zip(header, *rows)]

    def line(cells):
        return separator.join(_pad(cell, width, align) for cell, width, align in
                              zip(cells, widths, alignments)).rstrip()

    rule = "-" * (sum(widths) + len(separator) * (len(header) - 1))
    print()  # Keep a blank line immediately before every table header.
    print(line(header))
    print(rule)
    for row in rows:
        print(line(row))
    print(rule)


def ranked_records(records):
    # Stable canonical order resolves exact timing ties, independently of input order.
    return sorted(records, key=lambda row: (row["median_ms"], CANDIDATES.index(row["candidate"])))


def _check_label(name, passed=True):
    if not passed:
        return "❌ FAIL"
    return "⭐ BASE" if name == "scipy" else "✅ PASS"


def _comparison(record, baseline_ms, position):
    elapsed = record["median_ms"]
    if record["check"] != "PASS":
        relative = "-"
    elif record["candidate"] == "scipy":
        relative = "baseline"
    elif elapsed <= 0 or baseline_ms <= 0:
        relative = "n/a"
    elif verdict(record["candidate"], elapsed, baseline_ms) == "🤝":
        relative = "within 1%"
    elif elapsed < baseline_ms:
        relative = f"{baseline_ms / elapsed:.2f}x faster"
    else:
        relative = f"{elapsed / baseline_ms:.2f}x slower"
    return f"{POSITION_VERDICTS[position - 1]} {relative}"


def print_records(records, *, batch=False):
    """A ranked LAPX-style table for one single or batch case."""
    first = records[0]
    b, n, m = first["shape"]
    title = f"Batch {b} x {n} x {m}" if batch else f"Matrix {n} x {m}"
    rounds = len(first["samples_ms"])
    low = min(row["calls_per_round"] for row in records)
    high = max(row["calls_per_round"] for row in records)
    calls = f"{low}" if low == high else f"{low}-{high}"
    print(f"\n{title} | {first['dtype']} | rounds: {rounds} | calls/round: {calls}")
    baseline_ms = next(row["median_ms"] for row in records if row["candidate"] == "scipy")
    rows = []
    for position, record in enumerate(ranked_records(records), 1):
        passed = record["check"] == "PASS"
        rows.append((record["candidate"], _check_label(record["candidate"], passed),
                     f"{record['median_ms']:.6f}", _comparison(record, baseline_ms, position),
                     record["verdict"] if passed else "-"))
    _print_table(("Solver", "Check", "Time (ms)", "vs SciPy", "Verdict"), rows,
                 ("<", "<", ">", "<", "<"))


def _ordinal(position):
    suffix = "th" if 10 <= position % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th")
    return f"{position}{suffix}"


def _case_groups(records):
    groups = {}
    for record in records:
        key = (record["dtype"], tuple(record["shape"]), record.get("threshold"))
        groups.setdefault(key, []).append(record)
    return groups.values()


def print_tracking_records(records):
    """One compact matrix per dtype/threshold, with all solvers side by side."""
    groups = {}
    for record in records:
        groups.setdefault((record["dtype"], record["threshold"]), []).append(record)
    for (dtype, threshold), group in groups.items():
        print(f"\nTracking | {dtype} | threshold {threshold:g} | time in ms")
        print("Cells: time, check, position, verdict.  ✓ PASS | ⭐ baseline | 🏆 faster | 🤝 within 1% | 🐌 slower")
        time_width = max(len(f"{record['median_ms']:.6f}") for record in group)
        rows = []
        for case in _case_groups(group):
            positions = {row["candidate"]: rank for rank, row in enumerate(ranked_records(case), 1)}
            by_name = {row["candidate"]: row for row in case}
            _, n, m = case[0]["shape"]
            cells = [f"{n}x{m}"]
            for name in CANDIDATES:
                record = by_name[name]
                passed = record["check"] == "PASS"
                check = "✓" if passed else "✗"
                result = record["verdict"] if passed else "-"
                cells.append(f"{record['median_ms']:>{time_width}.6f} {check} {_ordinal(positions[name])} {result}")
            rows.append(tuple(cells))
        _print_table(("Size", *CANDIDATES), rows, ("<",) * (len(CANDIDATES) + 1), separator=" | ")


def finish(metadata, records, destination):
    summary = summarize(records)
    print_summary(summary, records)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(dict(metadata=metadata, results=records, summary=summary),
                                         indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return records


def cases(kind, quick):
    small = [(4, 4), (8, 8), (16, 16), (24, 24),
             (8, 12), (12, 8), (16, 24), (24, 16), (24, 32), (32, 24)]
    if kind == "batch":
        if quick:
            return ([(8, n, m) for n, m in small] +
                    [(8, 32, 32), (8, 32, 48), (8, 48, 32), (64, 24, 32), (128, 24, 32)])
        # Sweep worker overhead on small problems, then taper larger batches.
        # Keep dimensions <= 1000 and at most 4 million entries per input.
        groups = [(n, m, (8, 128)) for n, m in small] + [
            (24, 32, (64,)), (32, 24, (64,)),
            (32, 32, (1, 2, 4, 8, 16, 32, 64, 128)),
            (32, 48, (8, 32)), (48, 32, (8, 32)),
            (64, 64, (32, 64, 128)),
            (64, 96, (8, 32)), (96, 64, (8, 32)),
            (16, 128, (8, 32)), (128, 16, (8, 32)),
            (128, 128, (16, 32)),
            (128, 192, (8, 16)), (192, 128, (8, 16)),
            (256, 256, (8, 16)), (512, 512, (4, 8)), (1000, 1000, (2, 4)),
            (64, 128, (32,)), (128, 64, (32,)),
            (250, 500, (8,)), (500, 250, (8,)),
            (500, 1000, (2, 4)), (1000, 500, (2, 4)),
        ]
        return [(b, n, m) for n, m, batches in groups for b in batches]
    # Single and tracking share the same sizes, including the large cases.
    shapes = small + ([(32, 32), (64, 96), (96, 64), (128, 128)] if quick else [
        (32, 32), (32, 48), (48, 32),
        (64, 64), (64, 96), (96, 64), (16, 128), (128, 16),
        (128, 128), (128, 192), (192, 128),
        (256, 256), (250, 500), (500, 250), (512, 512),
        (500, 1000), (1000, 500), (1000, 1000),
        (1000, 2000), (2000, 1000), (2000, 2000),
        (2000, 3000), (3000, 2000), (3000, 3000),
        (3000, 4000), (4000, 3000), (4000, 5000), (5000, 4000), (5000, 5000),
    ])
    return [(1, n, m) for n, m in shapes]

def verdict(candidate, median_ms, baseline_ms):
    if candidate == "scipy":
        return "⭐"
    if baseline_ms * .99 <= median_ms <= baseline_ms * 1.01:
        return "🤝"
    return "🏆" if median_ms < baseline_ms else "🐌"

def summarize(records):
    summary = []
    for candidate in CANDIDATES:
        timings = [record["median_ms"] for record in records if record["candidate"] == candidate]
        if timings:
            summary.append(dict(candidate=candidate, tests=len(timings), total_ms=sum(timings)))
    summary.sort(key=lambda row: row["total_ms"])
    for position, row in enumerate(summary):
        row.update(position=position + 1, verdict=POSITION_VERDICTS[position])
    return summary

def print_summary(summary, records=None):
    counts = {name: [0] * len(CANDIDATES) for name in CANDIDATES}
    passed = dict.fromkeys(CANDIDATES, True)
    if records is not None:
        for case in _case_groups(records):
            for position, record in enumerate(ranked_records(case)):
                name = record["candidate"]
                counts[name][position] += 1
                passed[name] &= record["check"] == "PASS"
    # Give each placing a fixed slot across every solver, including zero
    # counts. Right-align counts within each placing's slot.
    positions = [position for position in range(len(CANDIDATES))
                 if any(counts[row["candidate"]][position] for row in summary)]
    digits = {position: max(len(str(counts[row["candidate"]][position])) for row in summary)
              for position in positions}
    print("\nOVERALL RANKING")
    print("Total time = sum of per-test medians; lower is faster. Tests include each dtype/shape/threshold case.")
    rows = []
    for row in summary:
        name = row["candidate"]
        slots = []
        for position in positions:
            emoji = POSITION_VERDICTS[position]
            count = counts[name][position]
            width = digits[position]
            slots.append(f"{count:>{width}}x{emoji}")
        placements = " ".join(slots) or "-"
        rows.append((name, _check_label(name, passed[name]), str(row["tests"]), f"{row['total_ms']:.6f}",
                     placements, row["verdict"]))
    _print_table(("Solver", "Check", "Tests", "Total (ms)", "Case positions", "Verdict"), rows,
                 ("<", "<", ">", ">", "<", "<"))
