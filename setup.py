# Copyright (c) 2026 Ratha SIV | MIT License
from pathlib import Path
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

def windows_rust_target(platform_name, configured=None):
    """Match Cargo to setuptools, including cibuildwheel ARM64 cross builds."""
    target = {'win-amd64': 'x86_64-pc-windows-msvc',
              'win-arm64': 'aarch64-pc-windows-msvc',
              'win32': 'i686-pc-windows-msvc'}.get(platform_name)
    if target is None or (configured and configured != target):
        raise RuntimeError('Rust target must match the Python extension platform: ' + platform_name)
    return target

def configure_macos_rust_build():
    """Keep the Python extension and Rust archive on the same architecture."""
    import os, platform, re, sys, sysconfig
    if sys.platform != 'darwin':
        return
    flags = os.environ.get('ARCHFLAGS', '-arch ' + platform.machine())
    architectures = set(re.findall(r'-arch\s+(\S+)', flags))
    if len(architectures) != 1 or not architectures <= {'x86_64', 'arm64'}:
        raise RuntimeError('Build laprust separately for each macOS architecture; universal2 is unsupported')
    architecture = architectures.pop()
    target_triple = {'x86_64': 'x86_64-apple-darwin', 'arm64': 'aarch64-apple-darwin'}[architecture]
    if os.environ.get('CARGO_BUILD_TARGET', target_triple) != target_triple:
        raise RuntimeError('Rust target must match the macOS extension architecture')
    os.environ['CARGO_BUILD_TARGET'] = target_triple
    os.environ['ARCHFLAGS'] = flags
    minimum = (11, 0) if architecture == 'arm64' else (10, 12)
    configured = os.environ.get('MACOSX_DEPLOYMENT_TARGET') or sysconfig.get_config_var(
        'MACOSX_DEPLOYMENT_TARGET') or '0.0'
    target = max(minimum, tuple(int(part) for part in configured.split('.')))
    deployment_target = '.'.join(str(part) for part in target)
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = deployment_target
    platform_name = 'macosx-{}-{}'.format(deployment_target, architecture)
    host_platform = os.environ.get('_PYTHON_HOST_PLATFORM', platform_name)
    # cibuildwheel keeps a baseline OS version in this tag (10.9 on Intel),
    # even when MACOSX_DEPLOYMENT_TARGET is newer. Validate the architecture,
    # then align the wheel tag with the deployment target used by C++ and Rust.
    if not re.fullmatch(r'macosx-\d+(?:\.\d+)*-' + architecture, host_platform):
        raise RuntimeError('laprust wheel platform must match the macOS extension architecture')
    os.environ['_PYTHON_HOST_PLATFORM'] = platform_name

class BuildExt(build_ext):
    """Add portable optimization flags for the compiler and linker.

    Environment variables control these options:
    - LAPRUST_BASEOPTS=0 disables the base flags that LAPRUST adds and link-time
      optimization (LTO). Other compiler defaults may still apply.
    - LAPRUST_NATIVE=1 enables -march=native -mtune=native.
    - LAPRUST_LTO=0 disables LTO, including inherited LTO defaults.

    LAPRUST_BASEOPTS=0 does not guarantee an unoptimized or debug build.
    Native tuning is disabled by default. Strict floating-point arithmetic is always used.
    """
    def has_flag(self, flag, link_flag=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'flagcheck.cpp')
            with open(fname, 'w') as f:
                f.write("int main(){return 0;}")
            try:
                objects = self.compiler.compile(
                    [fname], output_dir=tmpdir, extra_postargs=[flag], debug=self.debug)
                if link_flag is not None:
                    self.compiler.link_shared_object(
                        objects,
                        os.path.join(tmpdir, 'flagcheck' + self.compiler.shared_lib_extension),
                        extra_postargs=[link_flag], target_lang='c++', debug=self.debug)
            except Exception:
                return False
        return True

    def build_extensions(self):
        import os, sys

        rust_extensions = [ext for ext in self.extensions if ext.name == 'laprust._lapjv']
        if rust_extensions:
            import subprocess
            from pathlib import Path
            manifest = Path(__file__).resolve().parent / 'src/_lapjv/Cargo.toml'
            target_dir = Path(self.build_temp).resolve() / 'laprust-core'
            rust_env = os.environ.copy()
            if sys.platform == 'win32':
                rust_env['CARGO_BUILD_TARGET'] = windows_rust_target(
                    self.plat_name, rust_env.get('CARGO_BUILD_TARGET'))
            if os.environ.get('LAPRUST_BASEOPTS', '1').strip() == '0':
                rust_env['CARGO_PROFILE_RELEASE_OPT_LEVEL'] = '0'
                rust_env['CARGO_PROFILE_RELEASE_LTO'] = 'false'
            elif os.environ.get('LAPRUST_LTO', '1').strip() == '0':
                rust_env['CARGO_PROFILE_RELEASE_LTO'] = 'false'
            if os.environ.get('LAPRUST_NATIVE', '').strip() == '1':
                rust_env['RUSTFLAGS'] = rust_env.get('RUSTFLAGS', '') + ' -C target-cpu=native'
            try:
                if sys.platform.startswith('linux'):
                    version = subprocess.check_output(['rustc', '-vV'], text=True, env=rust_env)
                    host = next(line.split(': ', 1)[1] for line in version.splitlines()
                                if line.startswith('host: '))
                    if 'musl' in rust_env.get('CARGO_BUILD_TARGET', host):
                        # A Python shared library must use the process's libc.
                        rust_env['RUSTFLAGS'] = rust_env.get('RUSTFLAGS', '') + (
                            ' -C target-feature=-crt-static -C relocation-model=pic')
                subprocess.run(['cargo', 'build', '--release', '--locked',
                                '--manifest-path', str(manifest), '--target-dir', str(target_dir)],
                               check=True, env=rust_env)
            except FileNotFoundError as exc:
                raise RuntimeError('Building laprust from source requires Rust and Cargo (rustup.rs).') from exc
            rust_target = rust_env.get('CARGO_BUILD_TARGET')
            library_dir = target_dir / rust_target / 'release' if rust_target else target_dir / 'release'
            library = library_dir / ('laprust_core.lib' if sys.platform == 'win32' else 'liblaprust_core.a')
            for ext in rust_extensions:
                ext.extra_objects = list(ext.extra_objects or []) + [str(library)]
                ext.depends = list(ext.depends or []) + [str(library)]
                ext.libraries = list(ext.libraries or []) + (
                    ['ws2_32', 'userenv', 'advapi32', 'ntdll', 'bcrypt'] if sys.platform == 'win32'
                    else (['pthread', 'dl', 'm', 'rt', 'util'] if sys.platform.startswith('linux') else ['pthread', 'm']))

        base_enabled = (os.environ.get('LAPRUST_BASEOPTS', '1').strip() != '0')
        env_native   = (os.environ.get('LAPRUST_NATIVE',   '').strip() == '1')
        env_lto_on   = (os.environ.get('LAPRUST_LTO',      '1').strip() == '1')

        print("\n[+] LAPRUST -> Build options")
        print(f"    - LAPRUST_BASEOPTS  :  {'Enabled' if base_enabled else 'Disabled'}")
        print(f"    - LAPRUST_NATIVE    :  {'Enabled' if env_native else 'Disabled'}")
        print(f"    - LTO LAPRUST_LTO   :  {'Enabled' if (base_enabled and env_lto_on) else 'Disabled'}\n")

        ctype = self.compiler.compiler_type
        is_msvc = (ctype == 'msvc')

        compile_opts = []
        link_opts = []
        lto_enabled = False

        if is_msvc:
            # Base optimizations on MSVC
            if base_enabled:
                compile_opts += ['/O2', '/DNDEBUG']
                # Respect LAPRUST_LTO on MSVC when base opts are enabled
                if env_lto_on and self.has_flag('/GL', '/LTCG'):
                    compile_opts += ['/GL']
                    link_opts += ['/LTCG']
                    lto_enabled = True
        else:
            # Base optimizations on GCC/Clang
            if base_enabled:
                compile_opts += ['-O3', '-DNDEBUG']
                if sys.version_info >= (3, 9) and self.has_flag('-fvisibility=hidden'):
                    compile_opts += ['-fvisibility=hidden']
                if self.has_flag('-fno-math-errno'):
                    compile_opts += ['-fno-math-errno']
                # Link-time optimization (prefer ThinLTO when available)
                if env_lto_on:
                    if self.has_flag('-flto=thin', '-flto=thin'):
                        compile_opts += ['-flto=thin']; link_opts += ['-flto=thin']
                        lto_enabled = True
                    elif self.has_flag('-flto', '-flto'):
                        compile_opts += ['-flto']; link_opts += ['-flto']
                        lto_enabled = True
                # Minor call overhead reduction on Linux/glibc (if supported)
                if sys.platform.startswith('linux') and self.has_flag('-fno-plt'):
                    compile_opts += ['-fno-plt']
            # Optional native tuning (opt-in; avoid for portable wheels)
            if env_native:
                if self.has_flag('-march=native'):
                    compile_opts += ['-march=native']
                if self.has_flag('-mtune=native'):
                    compile_opts += ['-mtune=native']

        # Override inherited LTO defaults when disabled or when the probe fails.
        if not lto_enabled:
            compile_opts += ['/GL-' if is_msvc else '-fno-lto']
            link_opts += ['/LTCG:OFF' if is_msvc else '-fno-lto']

        # Apply to all extensions (always)
        for ext in self.extensions:
            prev_cargs = list(getattr(ext, 'extra_compile_args', []) or [])
            prev_largs = list(getattr(ext, 'extra_link_args', []) or [])
            ext.extra_compile_args = prev_cargs + compile_opts
            ext.extra_link_args = prev_largs + link_opts
            if ext.name == 'laprust._lapjv':
                # Conversion and forbidden-edge checks require strict arithmetic.
                ext.extra_compile_args += ['/fp:precise' if is_msvc else '-fno-fast-math']
                if ext.name == 'laprust._lapjv' and sys.platform.startswith('linux'):
                    # Keep the bundled Rust runtime private to this extension.
                    ext.extra_link_args += ['-Wl,--exclude-libs,ALL']

        # Show final compiler/linker args for debugging
        for ext in self.extensions:
            print(f"\n[+] LAPRUST -> Extension << {ext.name} >>")
            print(f"    - extra_compile_args : {ext.extra_compile_args}")
            print(f"    - extra_link_args    : {ext.extra_link_args}")
            if ext == self.extensions[-1]:
                print()

        super().build_extensions()


def main_setup():
    import numpy as np
    from Cython.Build import cythonize
    configure_macos_rust_build()
    extension = Extension(
        "laprust._lapjv", sources=['src/_lapjv/_lapjv.pyx'], language="c++",
        include_dirs=[np.get_include(), "src/_lapjv"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17", "-pthread", "-ffp-contract=off"],
        extra_link_args=[] if sys.platform == "win32" else ["-pthread"],
        depends=["src/_lapjv/lapjv.h"],
    )
    version = next(line.split('"')[1] for line in Path("laprust/__init__.py").read_text().splitlines() if line.startswith("__version__"))
    setup(
        name="laprust", version=version,
        description="Native Rust linear assignment solvers with implicit rejection and batch solving.",
        long_description=Path("README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        author="Ratha SIV", maintainer="rathaROG", license="MIT",
        url="https://github.com/rathaROG/laprust",
        license_files=['LICENSE', 'NOTICE', 'tests/LICENSE', 'src/_lapjv/LICENSE', 'src/_lapjv/THIRD_PARTY_NOTICES'],
        packages=["laprust"], package_data={"laprust": ["*.pyi", "py.typed"]},
        include_package_data=True,
        python_requires=">=3.9", install_requires=["numpy>=1.21.6"],
        extras_require={"test": ["pytest>=7", "scipy>=1.9", "pytest-timeout"],
                        "benchmark": ["scipy>=1.9", "lapx==0.11.1"], "dev": ["build", "setuptools>=77", "wheel", "pytest>=7", "scipy>=1.9", "pytest-timeout", "pyright"]},
        classifiers=["Development Status :: 5 - Production/Stable",
                     "Environment :: Console",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Education",
                     "Intended Audience :: Science/Research",
                     "Programming Language :: Python :: 3",
                     "Programming Language :: Python :: 3.9",
                     "Programming Language :: Python :: 3.10",
                     "Programming Language :: Python :: 3.11",
                     "Programming Language :: Python :: 3.12",
                     "Programming Language :: Python :: 3.13",
                     "Programming Language :: Python :: 3.14",
                     "Programming Language :: Python :: 3.15",
                     "Topic :: Education",
                     "Topic :: Education :: Testing",
                     "Topic :: Scientific/Engineering",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Software Development",
                     "Topic :: Software Development :: Libraries",
                     "Operating System :: Microsoft :: Windows",
                     "Operating System :: POSIX",
                     "Operating System :: Unix",
                     "Operating System :: MacOS",],
        ext_modules=cythonize([extension], compiler_directives={
            "language_level": 3, "boundscheck": False, "wraparound": False,
            "nonecheck": False, "initializedcheck": False, "cdivision": True,
            "infer_types": True, "profile": False, "linetrace": False}),
        cmdclass={"build_ext": BuildExt},
    )


if __name__ == "__main__":
    """Use :py:mod:`build` to build the package without changes to the current environment.

    >>> pip install build
    >>> python -m build --sdist
    >>> python -m build --wheel

    Source builds need stable Rust and Cargo (https://rustup.rs) in addition to a C++
    compiler. Cargo builds the Rust core in release mode with thin LTO and one codegen
    unit, and the Cython wrapper links it as a static library.

    The build applies safe base optimizations automatically:
    - /O2 on MSVC or -O3 on GCC and Clang enables compiler optimizations.
    - -DNDEBUG disables debug assertions.
    - Link-time optimization (LTO) applies when the compiler and linker support it.
    - Strict floating-point arithmetic (/fp:precise or -fno-fast-math) is always used;
      there is no fast-math option.
    - The Windows C++ wrapper uses /std:c++17.

    Environment variables control additional options for both the C++ wrapper and Cargo:
    - LAPRUST_BASEOPTS=0 disables the base flags that LAPRUST adds and LTO, and builds
      the Rust core with opt-level 0. This produces a much slower solver.
      Other compiler defaults may still apply.
    - LAPRUST_NATIVE=1 enables -march=native -mtune=native on GCC and Clang, and adds
      -C target-cpu=native to the Rust core on every platform.
    - LAPRUST_LTO=0 disables LTO for the wrapper and the Rust core, including inherited
      LTO defaults.

    LAPRUST_BASEOPTS=0 does not guarantee an unoptimized or debug build.
    Native tuning is disabled by default. Keep it disabled for portable release wheels.

    To build with native tuning on Linux or macOS, use this command:
    >>> LAPRUST_NATIVE=1 python -m build --wheel

    To build without LTO in the Windows command prompt (CMD), use this command:
    >>> set "LAPRUST_LTO=0" && python -m build --wheel

    The build applies Cython compiler directives by default for Cython modules.
    Examples include boundscheck=False, wraparound=False, and cdivision=True.
    """
    main_setup()
