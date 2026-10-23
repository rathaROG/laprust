"""Pass workflow inputs as validated arguments, without shell interpolation."""
import os
from pathlib import Path
import subprocess
import sys

kind = sys.argv[1]
if kind not in ("single", "batch", "tracking"):
    raise ValueError("unknown benchmark")
repeats = int(os.environ.get("BENCH_REPEATS", "5"))
if not 1 <= repeats <= 100:
    raise ValueError("BENCH_REPEATS must be between 1 and 100")
root = Path(__file__).resolve().parents[1]
command = [sys.executable, str(root / "benchmarks" / ("benchmark_" + kind + ".py")),
           "--quick", "--repeats", str(repeats), "--json",
           str(root / "benchmark-results" / (kind + ".json"))]
subprocess.run(command, check=True)
