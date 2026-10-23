"""Real PEP 660 installs and wheel transitions, using a disposable checkout/venv."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import pytest

PACKAGE = "laprust"
ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.editable, pytest.mark.timeout(600)]


def command(python, args, cwd, env, timeout=300):
    result = subprocess.run([str(python), *args], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installation(tmp_path_factory):
    directory = tmp_path_factory.mktemp(PACKAGE + "-editable-")
    project = directory / "checkout"
    ignored = shutil.ignore_patterns(".git", ".venv", "build", "dist", "wheelhouse", "target",
                                    "*.egg-info", "*.pyd", "*.so", "__pycache__", ".pytest_cache",
                                    "benchmark-results", "_lapjv.cpp")
    shutil.copytree(ROOT, project, ignore=ignored)
    environment = directory / "venv"
    # No system-site-packages: an old wheel elsewhere must not satisfy this test.
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    outside = directory / "outside"
    outside.mkdir()
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    assert command(python, ["-I", "-c", "import sys; assert sys.prefix != sys.base_prefix"], outside, env) == ""
    # Older Python releases seed pip without the --config-settings option.
    command(python, ["-m", "pip", "install", "--upgrade", "pip>=22.1"], outside, env)
    return project, environment, python, outside, env


PROBE = r'''
import importlib
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
import numpy as np
package, checkout, environment, mode, marker = sys.argv[1:]
class BlockOtherPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'lap', 'lapcpp', 'laprust'} - {package}:
            raise AssertionError('Unexpected solver dependency: ' + fullname)
sys.meta_path.insert(0, BlockOtherPackages())
module = importlib.import_module(package)
extension = importlib.import_module(package + '._lapjv')
checkout, environment = Path(checkout).resolve(), Path(environment).resolve()
origin = Path(module.__file__).resolve()
native = Path(extension.__file__).resolve()
distribution = importlib.metadata.distribution(package)
assert distribution.version == module.__version__
url = json.loads(distribution.read_text('direct_url.json'))
if mode == 'editable':
    assert url['dir_info']['editable'] is True
    assert Path(unquote(urlparse(url['url']).path).lstrip('/') if sys.platform == 'win32'
                else unquote(urlparse(url['url']).path)).resolve() == checkout
    assert checkout in origin.parents and checkout in native.parents, (origin, native)
else:
    assert not url.get('dir_info', {}).get('editable', False)
    assert environment in origin.parents and environment in native.parents, (origin, native)
    assert checkout not in origin.parents
assert module.__all__ == ['lapjv', 'lapjv_batch']
assert module.lapjv is extension.lapjv
assert getattr(module, '_editable_probe', None) == (marker or None)
for filename in ('__init__.pyi', '_lapjv.pyi', 'py.typed'):
    assert origin.with_name(filename).is_file(), filename
for dtype in (np.float32, np.float64):
    for cost, limit in [(np.array([[3.,1.],[1.,4.]],dtype=dtype), np.inf),
                        (np.array([[.1,.8,.9],[.9,.7,.8]],dtype=dtype), .5)]:
        expected_pairs = np.array([[0,1],[1,0]]) if not np.isfinite(limit) else np.array([[0,0]])
        for oriented, expected in ((cost, expected_pairs),
                                   (cost.T, expected_pairs[:,::-1][np.argsort(expected_pairs[:,1])])):
            for output in ('indices','pairs'):
                total, *rest = module.lapjv(oriented, extend_cost=True, cost_limit=limit, output=output)
                pairs = rest[0] if output == 'pairs' else np.column_stack(rest)
                np.testing.assert_array_equal(pairs, expected)
                np.testing.assert_allclose(total, oriented[tuple(pairs.T)].sum(dtype=np.float64))
                totals, pairs_list = module.lapjv_batch(np.stack([oriented]*3), extend_cost=True,
                    cost_limit=limit, n_threads=2, output='pairs')
                np.testing.assert_array_equal(totals, [total]*3)
                for actual in pairs_list: np.testing.assert_array_equal(actual, expected)
print(mode, package, origin)
'''


@pytest.mark.parametrize("mode", ["default", "strict", "compat"])
def test_editable_source_changes_and_wheel_roundtrip(installation, mode):
    project, environment, python, outside, env = installation
    settings = [] if mode == "default" else ["--config-settings", "editable_mode=" + mode]
    command(python, ["-m", "pip", "install", "--force-reinstall", "-e", str(project), *settings], outside, env)
    probe = ["-I", "-c", PROBE, PACKAGE, str(project), str(environment)]
    command(python, probe + ["editable", ""], outside, env)

    initializer = project / PACKAGE / "__init__.py"
    original = initializer.read_text(encoding="utf-8")
    try:
        # In-place writes also exercise strict-mode hardlinks. Never edit the real checkout.
        initializer.write_text(original + '\n_editable_probe = "source-edit-visible"\n', encoding="utf-8")
        command(python, probe + ["editable", "source-edit-visible"], outside, env)
    finally:
        initializer.write_text(original, encoding="utf-8")
    command(python, probe + ["editable", ""], outside, env)

    wheelhouse = project.parent / ("wheel-" + mode)
    command(python, ["-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheelhouse), str(project)], outside, env)
    wheel, = wheelhouse.glob(PACKAGE + "-*.whl")
    command(python, ["-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)], outside, env)
    command(python, probe + ["wheel", ""], outside, env)
    # A subsequent editable reinstall must work after a regular wheel install.
    command(python, ["-m", "pip", "install", "--no-deps", "-e", str(project), *settings], outside, env)
    command(python, probe + ["editable", ""], outside, env)
