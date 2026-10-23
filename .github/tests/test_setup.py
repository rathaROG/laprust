"""Check build configuration and compiler option selection without a native compiler."""

import ast
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError, LinkError


class ProbeCompiler:
    shared_lib_extension = ".so"

    def __init__(self, compiler_type, compile_failures=(), link_failures=()):
        self.compiler_type = compiler_type
        self.compile_failures = compile_failures
        self.link_failures = link_failures
        self.compiles = []
        self.links = []
        self.temp_dirs = []

    def compile(self, sources, output_dir=None, extra_postargs=(), **kwargs):
        self.compiles.extend(extra_postargs)
        self.temp_dirs.append(Path(output_dir or Path.cwd()))
        if any(flag in self.compile_failures for flag in extra_postargs):
            raise CompileError("Unsupported compiler flag")
        obj = Path(output_dir or Path(sources[0]).parent) / "flagcheck.obj"
        obj.write_bytes(b"object")
        return [str(obj)]

    def link_shared_object(self, objects, output_filename, extra_postargs=(), **kwargs):
        assert all(Path(obj).is_file() for obj in objects)
        assert kwargs["target_lang"] == "c++"
        self.links.extend(extra_postargs)
        if any(flag in self.link_failures for flag in extra_postargs):
            raise LinkError("LTO linker support is unavailable")
        Path(output_filename).write_bytes(b"library")


@pytest.fixture
def build_command(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    namespace = runpy.run_path(str(root / "setup.py"))
    for name in ("LAPRUST_BASEOPTS", "LAPRUST_LTO", "LAPRUST_NATIVE"):
        monkeypatch.delenv(name, raising=False)
    # Exercise option selection and the probes, without building LAPRUST itself.
    monkeypatch.setattr(build_ext, "build_extensions", lambda self: None)
    cmd = namespace["BuildExt"](Distribution())
    cmd.ensure_finalized()
    cmd.extensions = [Extension("laprust.probe", ["probe.cpp"])]
    return cmd


@pytest.mark.parametrize(
    "compiler_type,compile_failures,link_failures,expected_compile,expected_link",
    [
        ("msvc", (), (), "/GL", "/LTCG"),
        ("msvc", ("/GL",), (), "/GL-", "/LTCG:OFF"),
        ("msvc", (), ("/LTCG",), "/GL-", "/LTCG:OFF"),
        ("unix", (), (), "-flto=thin", "-flto=thin"),
        ("unix", ("-flto=thin",), (), "-flto", "-flto"),
        ("unix", (), ("-flto=thin",), "-flto", "-flto"),
        ("unix", (), ("-flto=thin", "-flto"), "-fno-lto", "-fno-lto"),
    ],
)
def test_lto_requires_compile_and_link_support(
    build_command, compiler_type, compile_failures, link_failures,
    expected_compile, expected_link,
):
    compiler = ProbeCompiler(compiler_type, compile_failures, link_failures)
    build_command.compiler = compiler
    build_command.build_extensions()
    ext = build_command.extensions[0]
    assert expected_compile in ext.extra_compile_args
    assert ext.extra_link_args == [expected_link]
    assert ("/O2" if compiler_type == "msvc" else "-O3") in ext.extra_compile_args
    if expected_link not in ("/LTCG:OFF", "-fno-lto"):
        assert expected_link in compiler.links
    assert all(not path.exists() for path in compiler.temp_dirs)


@pytest.mark.parametrize("compiler_type", ["msvc", "unix"])
@pytest.mark.parametrize("switch", ["LAPRUST_LTO", "LAPRUST_BASEOPTS"])
def test_disabled_lto_overrides_existing_flags(build_command, monkeypatch, compiler_type, switch):
    monkeypatch.setenv(switch, "0")
    compiler = ProbeCompiler(compiler_type)
    build_command.compiler = compiler
    ext = build_command.extensions[0]
    # Existing arguments, like compiler defaults, must precede the off switches.
    compile_on, link_on, compile_off, link_off = (
        ("/GL", "/LTCG", "/GL-", "/LTCG:OFF") if compiler_type == "msvc"
        else ("-flto", "-flto", "-fno-lto", "-fno-lto")
    )
    ext.extra_compile_args = [compile_on]
    ext.extra_link_args = [link_on]
    build_command.build_extensions()
    assert ext.extra_compile_args[0] == compile_on
    assert ext.extra_compile_args[-1] == compile_off
    assert ext.extra_link_args == [link_on, link_off]
    assert not compiler.links
    assert not {"/GL", "-flto", "-flto=thin"}.intersection(compiler.compiles)
    assert all(not path.exists() for path in compiler.temp_dirs)


@pytest.mark.parametrize('platform,target', [
    ('win-amd64', 'x86_64-pc-windows-msvc'),
    ('win-arm64', 'aarch64-pc-windows-msvc'),
    ('win32', 'i686-pc-windows-msvc'),
])
def test_rust_archive_matches_windows_extension_architecture(platform, target):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'setup.py'))
    choose = namespace['windows_rust_target']
    assert choose(platform) == choose(platform, target) == target
    with pytest.raises(RuntimeError, match='must match'):
        choose(platform, 'wrong-target')

@pytest.fixture
def macos_rust_build(monkeypatch):
    import os
    import platform
    import sys
    import sysconfig
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'setup.py'))
    # The build helper writes directly to os.environ; isolate those writes too.
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    for name in ('CARGO_BUILD_TARGET', 'ARCHFLAGS', '_PYTHON_HOST_PLATFORM',
                 'MACOSX_DEPLOYMENT_TARGET'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(sysconfig, 'get_config_var', lambda _: '10.9')
    return namespace['configure_macos_rust_build']

@pytest.mark.parametrize('architecture,flags,expected', [
    ('x86_64', '-arch x86_64', 'x86_64-apple-darwin'),
    ('x86_64', '-arch arm64', 'aarch64-apple-darwin'),
    ('arm64', '-arch x86_64', 'x86_64-apple-darwin'),
    ('arm64', '-arch arm64 -arch x86_64', None),
])
def test_macos_single_architecture_rust_target(
    monkeypatch, macos_rust_build, architecture, flags, expected,
):
    import os
    import platform
    monkeypatch.setattr(platform, 'machine', lambda: architecture)
    monkeypatch.setenv('ARCHFLAGS', flags)
    if expected is None:
        with pytest.raises(RuntimeError, match='universal2'):
            macos_rust_build()
    else:
        macos_rust_build()
        assert os.environ['CARGO_BUILD_TARGET'] == expected
        target_arch = 'arm64' if expected.startswith('aarch64') else 'x86_64'
        assert os.environ['_PYTHON_HOST_PLATFORM'].endswith('-' + target_arch)
        assert os.environ['MACOSX_DEPLOYMENT_TARGET'] == ('11.0' if target_arch == 'arm64' else '10.12')

@pytest.mark.parametrize('architecture,host_platform,deployment,expected', [
    ('x86_64', 'macosx-10.9-x86_64', '10.12', '10.12'),
    ('x86_64', 'macosx-10.9-x86_64', '10.13', '10.13'),
    ('x86_64', 'macosx-10.9-x86_64', '10.15', '10.15'),
    ('x86_64', 'macosx-10.9-x86_64', None, '10.12'),
    ('x86_64', 'macosx-10.12-x86_64', '10.12', '10.12'),
    ('arm64', 'macosx-11.0-arm64', '10.12', '11.0'),
    ('arm64', 'macosx-11.0-arm64', '12.0', '12.0'),
])
def test_macos_wheel_platform_tracks_deployment_target(
    monkeypatch, macos_rust_build, architecture, host_platform, deployment, expected,
):
    import os
    # cibuildwheel's platform tag keeps its default OS version even when the
    # deployment target is raised by the project or a newer Python version.
    monkeypatch.setenv('ARCHFLAGS', '-arch ' + architecture)
    monkeypatch.setenv('_PYTHON_HOST_PLATFORM', host_platform)
    if deployment is not None:
        monkeypatch.setenv('MACOSX_DEPLOYMENT_TARGET', deployment)
    macos_rust_build()
    assert os.environ['MACOSX_DEPLOYMENT_TARGET'] == expected
    assert os.environ['_PYTHON_HOST_PLATFORM'] == 'macosx-{}-{}'.format(expected, architecture)
    assert os.environ['CARGO_BUILD_TARGET'] == (
        'x86_64-apple-darwin' if architecture == 'x86_64' else 'aarch64-apple-darwin')
    assert os.environ['ARCHFLAGS'] == '-arch ' + architecture
    # Repeated setup hooks must accept the normalized platform as well.
    macos_rust_build()
    assert os.environ['_PYTHON_HOST_PLATFORM'] == 'macosx-{}-{}'.format(expected, architecture)

@pytest.mark.parametrize('architecture,host_platform', [
    ('x86_64', 'macosx-11.0-arm64'),
    ('arm64', 'macosx-10.12-x86_64'),
    ('x86_64', 'macosx-10.9-universal2'),
    ('x86_64', 'linux-x86_64'),
])
def test_macos_wheel_platform_rejects_incompatible_architecture(
    monkeypatch, macos_rust_build, architecture, host_platform,
):
    monkeypatch.setenv('ARCHFLAGS', '-arch ' + architecture)
    monkeypatch.setenv('_PYTHON_HOST_PLATFORM', host_platform)
    with pytest.raises(RuntimeError, match='wheel platform must match'):
        macos_rust_build()

def test_macos_rust_target_rejects_incompatible_architecture(monkeypatch, macos_rust_build):
    monkeypatch.setenv('ARCHFLAGS', '-arch x86_64')
    monkeypatch.setenv('CARGO_BUILD_TARGET', 'aarch64-apple-darwin')
    with pytest.raises(RuntimeError, match='Rust target must match'):
        macos_rust_build()
