"""Install the CI-selected NumPy runtime without rebuilding the NumPy-2 wheel."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, "-m", "pip", "install", "numpy==" + os.environ["NUMPY_VERSION"],
                "scipy==" + os.environ["SCIPY_VERSION"], "-r", str(root / ".github/tests/requirements.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                "--find-links=" + str(root / "dist"), "laprust"], check=True)
subprocess.run([sys.executable, str(root / ".github/tests/check_wheel.py")], check=True)
