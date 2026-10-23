"""Check built distributions without importing the native package."""
import ast
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "laprust"
tree = ast.parse((ROOT / PACKAGE / "__init__.py").read_text(encoding="utf-8"))
version = next(ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets))
wheels = list((ROOT / "dist").glob(f"{PACKAGE}-{version}-*.whl"))
assert wheels, "Build the source archive and wheel first: python -m build"
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = BytesParser().parsebytes(archive.read(next(n for n in names if n.endswith("/METADATA"))))
        assert metadata["Name"] == PACKAGE and metadata["Version"] == version
        assert metadata["Requires-Python"] == ">=3.9"
        runtime = [r for r in metadata.get_all("Requires-Dist", []) if "extra ==" not in r]
        assert runtime == ["numpy>=1.21.6"], runtime
        assert all(n.startswith((PACKAGE + "/", PACKAGE + ".libs/",
                                 f"{PACKAGE}-{version}.dist-info/")) for n in names)
        extensions = [n for n in names if n.startswith(PACKAGE + "/_lapjv.")
                      and n.endswith((".pyd", ".so"))]
        assert len(extensions) == 1, extensions
        for suffix in ("/licenses/LICENSE", "/licenses/NOTICE", "/licenses/src/_lapjv/LICENSE", "/licenses/tests/LICENSE"):
            assert any(n.endswith(suffix) for n in names), suffix
        for filename in ("__init__.pyi", "_lapjv.pyi", "py.typed"):
            assert PACKAGE + "/" + filename in names

with tarfile.open(ROOT / "dist" / f"{PACKAGE}-{version}.tar.gz") as archive:
    prefix = f"{PACKAGE}-{version}/"
    names = {n[len(prefix):] for n in archive.getnames() if n.startswith(prefix)}
    assert not any("/target/" in n or n.endswith((".pyd", ".so", ".pyc")) for n in names)
    assert not any(n.startswith(("build/", "dist/", ".git/")) for n in names)
    required = {"setup.py", "pyproject.toml", "src/_lapjv/_lapjv.pyx", "src/_lapjv/lapjv.h",
                "tests/test_lapjv.py", ".github/tests/check_wheel.py", ".github/tests/check_dist.py",
                "tests/typing/return_types.py", "benchmarks/_common.py",
                "tests/cost_eps.csv.gz", "tests/LICENSE", "tests/regression_cases.py",
                "tests/README.md", "tests/conftest.py", "tests/assignment_checks.py",
                ".github/tests/test_benchmarks.py", ".github/tests/install_numpy_compat.py",
                "tests/test_regressions.py", "tests/test_edge_cases.py", "tests/test_batch_paths.py",
                "tests/test_core.py", "tests/check_scipy_free.py", "tests/check_native_errors.py",
                ".github/tests/test_setup.py", ".github/tests/conftest.py",
                ".github/tests/pytest.ini", ".github/tests/requirements.txt",
                ".github/tests/README.md", ".github/workflows/native.yaml",
                "tests/test_cross_package.py", ".github/tests/test_editable_install.py",
                ".github/workflows/integration.yaml",
                ".github/workflows/tests.yaml",
                ".github/constraints/numpy-armv7-build.txt"}
    required |= ({"src/_lapjv/solver.cpp", "src/_lapjv/completion.cpp", "tests/native/CMakeLists.txt",
                  "tests/native/test_solver.cpp", "tests/native/faults.h"} if PACKAGE == "lapcpp" else
                 {"src/_lapjv/Cargo.lock", "src/_lapjv/Cargo.toml", "src/_lapjv/src/ffi.rs",
                  "src/_lapjv/THIRD_PARTY_NOTICES", "src/_lapjv/src/ffi_tests.rs",
                  "src/_lapjv/src/test_faults.rs"})
    assert required <= names, required - names
    moved = ("test_setup.py", "test_editable_install.py", "test_benchmarks.py",
             "check_dist.py", "check_wheel.py", "install_numpy_compat.py")
    assert not any("tests/" + name in names for name in moved)
print(PACKAGE, version, "distribution metadata, namespaces, licenses and source contents: PASS")
