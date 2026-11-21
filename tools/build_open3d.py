"""Build the pinned Windows 2Cut2DGS wheel without requiring PowerShell."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from setup_environment import ROOT, PYTHON_CHECK, install_open3d_wheel, project_path, run


def build_directory_check(build, source, venv, expected):
    """Keep CMake's package cleanup away from source/venv and unrelated builds."""
    if any(path == build or path.is_relative_to(build) for path in (ROOT, source, venv)):
        raise ValueError('BuildPath must be a dedicated build directory, not a parent of the project or venv.')
    if any(build.is_relative_to(path) for path in (source, venv, ROOT/'.git')):
        raise ValueError('BuildPath must be outside the submodule, venv and .git directories.')
    cache = build/'CMakeCache.txt'
    if cache.is_file():
        values = {}
        for line in cache.read_text(encoding='utf-8-sig').splitlines():
            match = re.match(r'^([^:#]+):[^=]+=(.*)$', line)
            if match:
                values[match[1]] = match[2]
        for name, value in expected.items():
            if name in values and os.path.normcase(values[name]) != os.path.normcase(str(value)):
                raise ValueError('Build cache belongs to another source, Python or compiler. Choose a new --build-path.')
    elif build.exists() and (not build.is_dir() or any(build.iterdir())):
        raise ValueError('BuildPath is nonempty without a CMake cache. Choose an empty build directory.')


def toolchain(cmake_exe):
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.is_file():
        raise RuntimeError('Install Visual Studio with Desktop development with C++ (MSVC and Windows SDK), '
                           'or use setup.bat --open3d-wheel with a prebuilt 2Cut2DGS wheel.')
    instances = json.loads(run(vswhere, '-latest', '-products', '*', '-requires',
                              'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-format', 'json', '-utf8', capture=True))
    if not instances:
        raise RuntimeError('Visual Studio MSVC x64/x86 build tools are missing.')
    vs = instances[0]
    major = int(vs['installationVersion'].split('.')[0])
    generators = {16: 'Visual Studio 16 2019', 17: 'Visual Studio 17 2022', 18: 'Visual Studio 18 2026'}
    if major not in generators:
        raise RuntimeError(f'Unsupported Visual Studio version: {major}')
    generator = generators[major]
    instance = Path(vs['installationPath'])
    cmake = cmake_exe or shutil.which('cmake') or str(instance/'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
    capabilities = json.loads(run(cmake, '-E', 'capabilities', capture=True))
    version = capabilities['version']
    if (version['major'], version['minor']) < (3, 24) or generator not in {g['name'] for g in capabilities['generators']}:
        raise RuntimeError(f'Install CMake >= 3.24 supporting {generator}, or specify --cmake-exe.')
    return cmake, generator, instance


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the pinned Open3D submodule and install 2Cut2DGS into a Python 3.12 x64 venv. '
                                    'Requires Visual Studio C++ tools, Windows SDK, CMake >= 3.24, Git and internet access. '
                                    'Close applications using this venv before installation.')
    parser.add_argument('--venv-path', default='.venv')
    parser.add_argument('--build-path', default='build/open3d', help='dedicated output directory; use a short path if needed')
    parser.add_argument('--cmake-exe')
    parser.add_argument('--jobs', type=int, choices=range(1, 257), metavar='N', default=min(4, os.cpu_count() or 1))
    parser.add_argument('--check-only', action='store_true', help='check prerequisites without building or installing')
    args = parser.parse_args(argv)
    if os.name != 'nt':
        raise RuntimeError('This script supports Windows x64 only.')
    venv = project_path(args.venv_path)
    build = project_path(args.build_path)
    source = ROOT/'Open3D'
    python = venv/'Scripts/python.exe'
    if not python.is_file() or not (venv/'pyvenv.cfg').is_file():
        raise RuntimeError('No valid venv found. Run setup.bat first (use the same --venv-path).')
    run(python, '-B', '-c', PYTHON_CHECK + '\nassert sys.prefix != sys.base_prefix, "A virtual environment is required"')
    build_directory_check(build, source, venv, {})
    cmake, generator, instance = toolchain(args.cmake_exe)
    build_directory_check(build, source, venv, {
        'CMAKE_HOME_DIRECTORY': source, 'Python3_EXECUTABLE': python,
        'CMAKE_GENERATOR': generator, 'CMAKE_GENERATOR_INSTANCE': instance})
    git = shutil.which('git')
    if not git:
        raise RuntimeError('Git is required to obtain the pinned Open3D submodule.')
    if not (source/'.git').exists():
        if args.check_only:
            raise RuntimeError('Initialize the submodule first: git submodule update --init Open3D')
        run(git, '-C', ROOT, 'submodule', 'update', '--init', '--', 'Open3D')
    entry = run(git, '-C', ROOT, 'ls-files', '--stage', '--', 'Open3D', capture=True)
    match = re.fullmatch(r'160000 ([0-9a-f]+) 0\s+Open3D', entry)
    if not match:
        raise RuntimeError('Open3D is not a registered submodule in the index.')
    commit = match[1]
    actual = run(git, '-C', source, 'rev-parse', 'HEAD', capture=True)
    dirty = run(git, '-C', source, 'status', '--porcelain', '--untracked-files=no', capture=True)
    untracked = run(git, '-C', source, 'ls-files', '--others', '--exclude-standard', '-z', capture=True)
    # Local prose documents do not affect the build. Keep checking all tracked
    # changes and untracked code/configuration, including files under docs/.
    untracked_build_files = [name for name in untracked.split('\0') if name and not (
        name.startswith('docs/') and Path(name).suffix.lower() in {'.rst', '.md'})]
    if actual != commit or dirty or untracked_build_files:
        raise RuntimeError('Open3D differs from the recorded commit or has local source changes. '
                           'Preserve those changes and restore the pinned submodule before building.')
    print(f'Source: {source} ({commit})\nPython: {python}\nBuild: {build} / {generator} / Release / {args.jobs} jobs', flush=True)
    if args.check_only:
        print('Prerequisites passed. No build or installation performed.')
        return
    run(python, '-m', 'pip', 'install', '-r', source/'python/requirements_build.txt')
    configure = ['-S', source, '-B', build, '-G', generator, '-A', 'x64',
                 f'-DCMAKE_GENERATOR_INSTANCE={instance}', f'-DPython3_EXECUTABLE={python}',
                 # pybind11 contains UTF-8 literals; Japanese Windows otherwise
                 # interprets them as CP932 and fails with MSVC error C2001.
                 '-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /W3 /GR /EHsc /utf-8',
                 '-DDEVELOPER_BUILD=ON', '-DPYPI_PACKAGE_NAME=open3d',
                 '-DBUILD_SHARED_LIBS=ON', '-DSTATIC_WINDOWS_RUNTIME=OFF', '-DBUILD_PYTHON_MODULE=ON',
                 '-DBUILD_GUI=ON', '-DBUILD_CUDA_MODULE=OFF', '-DBUILD_SYCL_MODULE=OFF',
                 '-DBUILD_ISPC_MODULE=ON', '-DBUILD_WEBRTC=OFF', '-DBUILD_JUPYTER_EXTENSION=OFF',
                 '-DBUILD_EXAMPLES=OFF', '-DBUILD_UNIT_TESTS=OFF', '-DBUILD_BENCHMARKS=OFF',
                 '-DBUILD_PYTORCH_OPS=OFF', '-DBUILD_TENSORFLOW_OPS=OFF', '-DBUNDLE_OPEN3D_ML=OFF', '-DWITH_STUBGEN=OFF']
    run(cmake, *configure)
    started = time.time()
    run(cmake, '--build', build, '--config', 'Release', '--target', 'pip-package', '--parallel', str(args.jobs))
    wheels = [p for p in (build/'lib/python_package/pip_package').glob('open3d-*-cp312-cp312-win_amd64.whl')
              if p.stat().st_mtime >= started - 2]
    if len(wheels) != 1:
        raise RuntimeError('Expected exactly one newly built Python 3.12 x64 Open3D wheel.')
    install_open3d_wheel(python, wheels[0])
    run(python, '-m', 'pip', 'check')
    print(f'Build and installation completed. Installed wheel: {wheels[0]}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Open3D build/install failed: {exc}', file=sys.stderr)
        sys.exit(1)
