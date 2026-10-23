"""Build the real Cython adapter against a controlled failing C ABI.

Run separately from installed-wheel tests: this requires Cython and a compiler.
The stub is confined to a temporary extension and never enters a wheel. Native
core tests separately inject real allocation/startup failures and test recovery.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PACKAGE = "laprust"
ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix=PACKAGE + "-error-adapter-") as directory:
        work = Path(directory)
        shutil.copyfile(ROOT / "src/_lapjv/_lapjv.pyx", work / "_error_probe.pyx")
        shutil.copyfile(ROOT / "src/_lapjv/lapjv.h", work / "lapjv.h")
        (work / "failure.cpp").write_text('''
#include "lapjv.h"
extern "C" int PACKAGE_run(const void*, int, const void* original, int original32,
    size_t b, size_t rows, size_t cols, double, size_t, int, int pairs, int,
    int64_t* output, size_t* counts, double* totals, size_t* failed) {
    const int status = int(original32 ? static_cast<const float*>(original)[0]
                                     : static_cast<const double*>(original)[0]);
    *failed = b > 1 ? 2 : 0;
    if (status) return status;
    const size_t k = rows < cols ? rows : cols;
    for (size_t i=0; i<b; ++i) {
        counts[i]=1; totals[i]=0;
        output[2*k*i]=0; output[2*k*i+(pairs ? 1 : k)]=0;
    }
    return 0;
}
'''.replace("PACKAGE", PACKAGE), encoding="utf-8")
        (work / "setup.py").write_text('''
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
setup(name="native-error-probe", ext_modules=cythonize([
    Extension("_error_probe", ["_error_probe.pyx", "failure.cpp"], language="c++",
              include_dirs=[np.get_include(), "."])
], compiler_directives={"language_level": 3}))
''', encoding="utf-8")
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=work, check=True)
        # Use a subprocess so Windows releases the DLL before temporary cleanup.
        probe = r'''
from itertools import product
import numpy as np
import _error_probe as native
checks=0
for batch, dtype, output, return_cost in product((False,True), (np.float32,np.float64), ("indices","pairs"), (False,True)):
    solve=native.lapjv_batch if batch else native.lapjv
    for status, expected in ((1,ValueError),(2,MemoryError),(3,RuntimeError),(4,ValueError),(99,RuntimeError)):
        a=np.full((4,2,2) if batch else (2,2), status, dtype=dtype)
        try: solve(a, output=output, return_cost=return_cost)
        except expected as error:
            assert ("at batch index 2" in str(error)) == batch, str(error)
        else: raise AssertionError((status,expected))
        recovered=solve(np.zeros_like(a), output=output, return_cost=return_cost)
        values=recovered[1] if return_cost and output=="pairs" else recovered[1:] if return_cost else recovered
        if output=="pairs":
            groups=values if batch else [values]
            for pairs in groups: np.testing.assert_array_equal(pairs,[[0,0]])
        else:
            for group in values:
                for array in (group if batch else [group]): np.testing.assert_array_equal(array,[0])
        if return_cost: np.testing.assert_array_equal(recovered[0], np.zeros(4) if batch else 0.)
        checks+=1
print(f"{checks} native status/exception translation and recovery checks passed")
'''
        subprocess.run([sys.executable, "-c", probe], cwd=work, check=True)


if __name__ == "__main__":
    main()
