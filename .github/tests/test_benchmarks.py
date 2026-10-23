"""Guard the public fast-path calls, timing boundaries and benchmark checks.

Native API stand-ins keep these tests runnable in either package's standalone
CI. The benchmark workflows also execute all five real installed candidates.
"""
from functools import partial
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
spec = importlib.util.spec_from_file_location("benchmark_common", BENCHMARKS / "_common.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture
def scripts(monkeypatch):
    calls = []

    def native_api(name, batch=False):
        def api(cost, **options):
            calls.append((name, cost, options))
            matrices = cost if batch else cost[None]
            pairs = [np.column_stack(linear_sum_assignment(a))[::-1] for a in matrices]
            # Deliberately return a different order to detect normalization in wrappers.
            if not options["return_cost"]:
                return pairs if batch else pairs[0]
            totals = np.array([a[tuple(p.T)].sum(dtype=np.float64) for a, p in zip(matrices, pairs)])
            return (totals, pairs) if batch else (totals[0], pairs[0])
        return api

    for name in ("lapcpp", "laprust"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(
            lapjv=native_api(name), lapjv_batch=native_api(name, batch=True)))
    monkeypatch.setitem(sys.modules, "lap", SimpleNamespace(
        lapjvxa=native_api("lapjvxa"), lapjvsa=native_api("lapjvsa"),
        lapjvxa_batch=native_api("lapjvxa", batch=True), lapjvsa_batch=native_api("lapjvsa", batch=True)))
    monkeypatch.setitem(sys.modules, "_common", bench)
    loaded = {}
    for kind in ("single", "batch", "tracking"):
        spec = importlib.util.spec_from_file_location("benchmark_" + kind, BENCHMARKS / ("benchmark_" + kind + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[kind] = module
    return loaded, calls


@pytest.mark.parametrize("candidate", bench.CANDIDATES)
@pytest.mark.parametrize("shape", [(4, 4), (4, 6), (6, 4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_uses_unmodified_input_and_default_fast_path(scripts, candidate, shape, dtype):
    modules, calls = scripts
    cost = bench.make_costs(shape, 2026, 0, dtype)
    result = getattr(modules["single"], "solve_" + candidate)(cost)
    baseline = np.column_stack(linear_sum_assignment(cost))
    bench.check_single(cost, result, baseline)
    if candidate == "scipy":
        assert not calls
        return
    name, received, options = calls.pop()
    assert name == candidate and received is cost and not calls
    expected = dict(extend_cost=shape[0] != shape[1], return_cost=False)
    if candidate in ("lapcpp", "laprust"):
        expected.update(precision="native", output="pairs")
    # In particular, no finite cost_limit and no prefer_float32 override for JVS.
    assert options == expected
    np.testing.assert_array_equal(result, baseline[::-1])


@pytest.mark.parametrize("candidate", bench.CANDIDATES)
@pytest.mark.parametrize("shape", [(3, 4, 4), (3, 4, 6), (3, 6, 4)])
@pytest.mark.parametrize("return_cost", [False, True])
def test_batch_calls_public_batch_api_once(scripts, candidate, shape, return_cost):
    modules, calls = scripts
    costs = bench.make_costs(shape, 2026, 1, np.float64)
    options = dict(return_cost=return_cost)
    if candidate != "scipy":
        options["n_threads"] = 3
    result = getattr(modules["batch"], "solve_" + candidate)(costs, **options)
    baseline = modules["batch"].solve_scipy(costs, return_cost=return_cost)
    bench.check_batch(costs, result, baseline, return_cost=return_cost)
    if candidate == "scipy":
        assert not calls
        return
    name, received, options = calls.pop()
    assert name == candidate and received is costs and not calls
    expected = dict(extend_cost=shape[1] != shape[2], return_cost=return_cost, n_threads=3)
    if candidate in ("lapcpp", "laprust"):
        expected.update(precision="native", output="pairs")
    assert options == expected


@pytest.mark.parametrize("candidate", bench.CANDIDATES)
@pytest.mark.parametrize("threshold", [0., .4, 1.])
@pytest.mark.parametrize("transpose", [False, True])
def test_tracking_postfilters_full_assignment_and_keeps_unmatched_lists(scripts, candidate, threshold, transpose):
    modules, calls = scripts
    # A joint rejection solve at .4 could retain .1; post-filtering retains the
    # two .4 edges of the full optimum. This distinguishes the two objectives.
    cost = np.array([[.1, .4, 9.], [.4, .9, 9.], [9., 9., .8], [9., 9., 9.]])
    if transpose:
        cost = cost.T.copy()
    result = getattr(modules["tracking"], "solve_" + candidate)(cost, threshold)
    full = np.array([[0, 1], [1, 0], [2, 2]])
    expected = full[cost[tuple(full.T)] <= threshold]
    unmatched_rows = sorted(set(range(cost.shape[0])) - set(expected[:, 0]))
    unmatched_cols = sorted(set(range(cost.shape[1])) - set(expected[:, 1]))
    modules["tracking"].check_tracking(cost, result, (expected, unmatched_rows, unmatched_cols), threshold=threshold)
    assert result[0].dtype == np.dtype(int)
    if candidate == "scipy":
        assert not calls
        return
    name, received, options = calls.pop()
    assert name == candidate and received is cost and not calls
    expected_options = dict(extend_cost=True, return_cost=False)
    if candidate in ("lapcpp", "laprust"):
        expected_options.update(precision="native", output="pairs")
    assert options == expected_options


def test_checks_detect_invalid_suboptimal_and_wrong_total_results(scripts):
    cost = np.array([[0., 1.], [1., 0.]])
    optimum = np.array([[0, 0], [1, 1]])
    wrong = np.array([[0, 1], [1, 0]])
    for pairs in (wrong, optimum[:1], np.array([[0, 0], [1, 0]]), np.array([[0, -1], [1, 1]])):
        with pytest.raises(AssertionError):
            bench.check_single(cost, pairs, optimum)
        with pytest.raises(AssertionError):
            bench.check_batch(cost[None], [pairs], [optimum])
    with pytest.raises(AssertionError):
        bench.check_batch(cost[None], (np.array([3.]), [optimum]), (np.array([0.]), [optimum]), return_cost=True)
    # Batch checks the original objective, so equivalent tied optima are valid.
    bench.check_batch(np.zeros((1, 2, 2)), [wrong], [optimum])
    tracking = scripts[0]["tracking"]
    with pytest.raises(AssertionError):
        tracking.check_tracking(cost, (optimum, [0], []), (optimum, [], []), threshold=.5)
    with pytest.raises(AssertionError):
        tracking.check_tracking(cost, (wrong, [], []), (wrong, [], []), threshold=.5)


def test_inputs_and_default_precision_match_pinned_reference():
    for shape in ((4, 6), (8, 4, 6)):
        for round_index in (0, 1):
            expected = np.random.default_rng([2026 + round_index, *shape]).random(shape)
            np.testing.assert_array_equal(bench.make_costs(shape, 2026, round_index, "float64"), expected)
    args = bench.common_parser("test", repeats=3).parse_args([])
    assert args.dtype == "float64" and args.seed == 2026 and args.repeats == 3


def test_timing_excludes_generation_warmup_checks_and_dispatch(monkeypatch):
    clock = [0.]
    in_timer = [False]
    counts = dict.fromkeys(bench.CANDIDATES, 0)

    def now():
        in_timer[0] = not in_timer[0]
        return clock[0]

    def make_costs(*args):
        assert not in_timer[0]
        clock[0] += 1000
        return np.zeros((4, 4))

    def check(*args):
        assert not in_timer[0]
        clock[0] += 100

    def solve(name, costs):
        counts[name] += 1
        clock[0] += .125
        return np.column_stack((np.arange(4), np.arange(4)))

    class Solvers(dict):
        def __getitem__(self, key):
            assert not in_timer[0]
            return super().__getitem__(key)

    monkeypatch.setattr(bench.time, "perf_counter", now)
    monkeypatch.setattr(bench, "make_costs", make_costs)
    solvers = Solvers((name, partial(solve, name)) for name in bench.CANDIDATES)
    args = SimpleNamespace(seed=2026, repeats=2, calls=3)
    records = bench.measure_case(solvers, (1, 4, 4), "float64", args, check)
    assert counts == dict.fromkeys(bench.CANDIDATES, 8)  # One warmup + three timed calls per round.
    assert all(record["samples_ms"] == [125., 125.] for record in records)
    assert all(record["median_ms"] == 125. for record in records)


@pytest.mark.parametrize("seconds, calls, rounds", [(.001, 10, 2), (.0005, 20, 5)])
def test_calibrated_timing_warms_each_solver_and_repeats_calls(monkeypatch, seconds, calls, rounds):
    clock = [0.]
    events = []

    def solve(name, costs):
        events.append(name)
        clock[0] += seconds
        return np.column_stack((np.arange(4), np.arange(4)))

    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    solvers = {name: partial(solve, name) for name in bench.CANDIDATES}
    args = SimpleNamespace(seed=2026, repeats=2, calls=0, min_time=.01, fast_rounds=5, reference_timing=False)
    records = bench.measure_case(solvers, (1, 4, 4), "float64", args, lambda *args: None)
    # Round one: warm-up, calibration call, timed calls. Later rounds: warm-up, timed calls.
    # Sub-millisecond cases get the fast-round count.
    per_solver = 2 + calls + (rounds - 1) * (1 + calls)
    assert all(events.count(name) == per_solver for name in bench.CANDIDATES)
    # Each sample's calls run back to back, right after that solver's warm-up.
    first = events[:calls + 2]
    assert first == [first[0]] * (calls + 2)
    for record in records:
        assert record["calls_per_round"] == calls
        assert record["samples_ms"] == pytest.approx([seconds * 1000] * rounds)


def test_workload_sizes_preserve_requested_small_and_tapered_batches():
    for kind in ("single", "tracking", "batch"):
        for quick in (False, True):
            shapes = bench.cases(kind, quick)
            assert len(shapes) == len(set(shapes))
            for size in ((4, 4), (8, 8), (24, 32), (32, 24)):
                assert any(shape[1:] == size for shape in shapes)
    batch = bench.cases("batch", False)
    assert len(batch) == 65 and max(b for b, _, _ in batch) == 128
    assert (4, 1000, 1000) in batch and (128, 1000, 1000) not in batch
    assert max(b * n * m for b, n, m in batch) <= 4_000_000
    assert max(max(n, m) for _, n, m in batch) <= 1000
    assert all((b, 32, 32) in batch for b in (1, 2, 4))
    for kind in ("single", "tracking", "batch"):
        assert set(bench.cases(kind, True)) <= set(bench.cases(kind, False))
        shapes = {(n, m) for _, n, m in bench.cases(kind, False)}
        for shape in ((32, 48), (64, 96), (128, 192), (16, 128)):
            assert shape in shapes and shape[::-1] in shapes
    single = bench.cases("single", False)
    assert single == bench.cases("tracking", False) and len(single) == 39
    for n, m in ((2000, 2000), (3000, 3000), (5000, 5000),
                 (500, 1000), (2000, 3000), (3000, 4000), (4000, 5000)):
        assert (1, n, m) in single and (1, m, n) in single


def test_verdicts_and_summary_rank_actual_total_times(capsys):
    assert bench.verdict("scipy", 100, 100) == "⭐"
    for candidate in bench.CANDIDATES[1:]:
        assert bench.verdict(candidate, 98, 100) == "🏆"
        assert bench.verdict(candidate, 102, 100) == "🐌"
        for tied in (99, 100, 101):
            assert bench.verdict(candidate, tied, 100) == "🤝"
    # SciPy wins more individual cases, but totals must determine the rank.
    records = [dict(candidate=name, median_ms=ms) for name, times in
               (("scipy", [1, 1, 100]), ("lapcpp", [2, 2, 4]), ("laprust", [3, 3, 5]))
               for ms in times]
    summary = bench.summarize(records)
    assert summary == [
        dict(candidate="lapcpp", tests=3, total_ms=8, position=1, verdict="🥇"),
        dict(candidate="laprust", tests=3, total_ms=11, position=2, verdict="🥈"),
        dict(candidate="scipy", tests=3, total_ms=102, position=3, verdict="🥉"),
    ]
    bench.print_summary(summary)
    lines = capsys.readouterr().out.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Solver"))
    assert lines[header - 1] == ""
    assert lines[header].endswith("Verdict")


def test_five_candidate_rankings_and_partial_summary():
    records = [dict(candidate=name, median_ms=index + 1) for index, name in enumerate(bench.CANDIDATES)]
    assert [row["verdict"] for row in bench.summarize(records)] == ["🥇", "🥈", "🥉", "🚩", "🏳️"]
    assert [row["candidate"] for row in bench.summarize(records[:2])] == ["scipy", "lapcpp"]


def _reported_case(times, *, shape=(1, 24, 32), dtype="float64", threshold=None):
    baseline = times[0]
    records = []
    for name, ms in zip(bench.CANDIDATES, times):
        record = dict(candidate=name, median_ms=ms, shape=list(shape), dtype=dtype,
                      samples_ms=[ms, ms, ms], calls_per_round=25, check="PASS",
                      verdict=bench.verdict(name, ms, baseline))
        if threshold is not None:
            record["threshold"] = threshold
        records.append(record)
    return records


def test_ranked_table_labels_baseline_ratios_ties_and_keeps_raw_results(capsys):
    from copy import deepcopy
    records = _reported_case([10, 2, 9.95, 20, 5])
    saved = deepcopy(records)
    bench.print_records(records)
    output = capsys.readouterr().out
    lines = output.splitlines()
    rows = [line for line in lines if line.split() and line.split()[0] in bench.CANDIDATES]
    assert [line.split()[0] for line in rows] == ["lapcpp", "lapjvsa", "laprust", "scipy", "lapjvxa"]
    assert "🥇 5.00x faster" in rows[0]
    assert "🥉 within 1%" in rows[2] and rows[2].endswith("🤝")
    assert "⭐ BASE" in rows[3] and "🚩 baseline" in rows[3] and rows[3].endswith("⭐")
    assert "🏳️ 2.00x slower" in rows[4] and rows[4].endswith("🐌")
    header = next(i for i, line in enumerate(lines) if line.startswith("Solver"))
    assert lines[header - 1] == "" and lines[header].endswith("Verdict")
    assert "Matrix 24 x 32 | float64" in output
    assert records == saved


def test_batch_table_shows_positions_for_baseline_and_slower_results(capsys):
    records = _reported_case([1, 2, 3, 4, 5], shape=(128, 24, 32))
    records[-1]["check"] = "FAIL"
    bench.print_records(records, batch=True)
    output = capsys.readouterr().out
    assert "Batch 128 x 24 x 32" in output
    for relative in ("🥇 baseline", "🥈 2.00x slower", "🥉 3.00x slower", "🚩 4.00x slower", "🏳️ -"):
        assert relative in output
    failed = next(line for line in output.splitlines() if line.startswith("lapjvsa"))
    assert "❌ FAIL" in failed and "PASS" not in failed and failed.endswith("-")


def test_tracking_tables_group_thresholds_and_dtypes_and_align_columns(capsys):
    records = (_reported_case([1, 2, 3, 4, 5], threshold=.05) +
               _reported_case([20, 3, 2, 4, 5], shape=(1, 1000, 1000), threshold=.05) +
               _reported_case([2, 1, 3, 4, 5], threshold=.5) +
               _reported_case([2, 1, 3, 4, 5], threshold=.05, dtype="float32"))
    bench.print_tracking_records(records)
    output = capsys.readouterr().out
    assert output.count("Tracking |") == 3
    assert "Tracking | float64 | threshold 0.05" in output
    assert "Tracking | float64 | threshold 0.5" in output
    assert "Tracking | float32 | threshold 0.05" in output
    first = output.split("Tracking |")[1]
    data_lines = [line for line in first.splitlines() if line.startswith(("24x32", "1000x1000"))]
    assert len(data_lines) == 2 and "1st ⭐" in data_lines[0] and "1st 🏆" in data_lines[1]
    # Column boundaries stay aligned despite emoji and different time magnitudes.
    boundaries = [[bench.display_width(line[:i]) for i, char in enumerate(line) if char == "|"] for line in data_lines]
    assert boundaries[0] == boundaries[1]
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Size"):
            assert lines[i - 1] == ""


def test_overall_ranking_counts_case_positions_but_ranks_by_total(capsys):
    records = (_reported_case([1, 2, 3, 4, 5], shape=(1, 4, 4)) +
               _reported_case([100, 4, 5, 6, 7], shape=(1, 8, 8)))
    bench.print_summary(bench.summarize(records), records)
    output = capsys.readouterr().out
    rows = [line for line in output.splitlines() if line.split() and line.split()[0] in bench.CANDIDATES]
    assert rows[0].startswith("lapcpp") and rows[0].endswith("  🥇")
    assert "1x🥇 1x🥈 0x🥉 0x🚩 0x🏳️" in rows[0] and "6.000000" in rows[0]
    assert rows[-1].startswith("scipy") and "1x🥇" in rows[-1] and "1x🏳️" in rows[-1]
    for position in bench.POSITION_VERDICTS.values():
        token = "x" + position
        columns = {bench.display_width(line[:line.index(token)]) for line in rows if token in line}
        assert len(columns) <= 1  # A placing stays in the same column even when other ranks are absent.
    assert "OVERALL RANKING" in output and "Case positions" in output


def test_terminal_width_accounts_for_rank_emoji_and_variation_selectors():
    for emoji in bench.POSITION_VERDICTS.values():
        assert bench.display_width(emoji) == 2
    assert bench.display_width("✅ PASS") == 7
    assert bench.display_width("12x🏳️") == 5
    assert bench.display_width("✓") == 1


def test_header_keeps_full_metadata_available_without_default_dump(capsys):
    metadata = dict(kind="single", python="3.11.0", numpy="2.0.0", versions={"lapcpp": "1.0.0"},
                    input_model="uniform [0, 1)", reference_release=bench.REFERENCE_RELEASE,
                    settings=dict(dtype="float64", seed=2026, repeats=3, calls=0, verbose=False),
                    modules={"lapcpp": "example/loaded/package"})
    bench.print_header(metadata)
    output = capsys.readouterr().out
    assert "Single assignment benchmark" in output and "float64" in output
    assert "example/loaded/package" not in output
    metadata["settings"]["verbose"] = True
    bench.print_header(metadata)
    assert '"lapcpp": "example/loaded/package"' in capsys.readouterr().out
