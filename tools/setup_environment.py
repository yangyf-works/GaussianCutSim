"""Windows venv setup and 2Cut2DGS installation invoked by setup.bat."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON_CHECK = """
import struct, sys
if sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
    raise SystemExit('GaussianCutSim requires 64-bit Python 3.12.')
"""


def run(executable, *arguments, capture=False):
    """Use native argument quoting and propagate failures without shell evaluation."""
    result = subprocess.run([str(executable), *map(str, arguments)], check=True,
                            stdout=subprocess.PIPE if capture else None, text=True,
                            encoding='utf-8' if capture else None)
    return result.stdout.strip() if capture else None


def project_path(value):
    if not value.strip():
        raise ValueError('Paths must not be empty.')
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_open3d():
    """Reject the official 3DGS package: this distribution requires 2Cut2DGS."""
    sys.path.insert(0, str(ROOT))
    import open3d as o3d
    from cutsim.rendering.open3d_renderer import require_gaussian_api
    if not Path(o3d.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Open3D was imported from outside the target venv.')
    require_gaussian_api(o3d)
    print('Open3D:', o3d.__version__, '(2Cut2DGS / extended surface GS)', flush=True)


def has_extended_open3d(python):
    probe = subprocess.run([str(python), '-B', str(Path(__file__).resolve()), '--check-open3d'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return probe.returncode == 0


def install_open3d_wheel(python, wheel):
    """Resolve dependencies, then replace even a same-version official wheel."""
    check_extended_wheel(wheel)
    run(python, '-m', 'pip', 'install', wheel)
    run(python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', wheel)
    run(python, '-B', Path(__file__).resolve(), '--check-open3d')


def check_extended_wheel(wheel):
    """Reject official wheels before pip can replace the target environment.

    Windows pybind modules retain these binding names. This is a compatibility
    precheck; the installed module must also pass the actual API probe afterward.
    """
    required = (b'gaussian_splat_surface_lighting', b'gaussian_splat_surface_shininess',
                b'gaussian_splat_opaque_surface', b'normalize_gaussian_splat_clipping')
    try:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.startswith('open3d/') and 'pybind' in Path(name).name and name.endswith('.pyd'):
                    module = archive.read(name)
                    if all(symbol in module for symbol in required):
                        return
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError('Not a valid Open3D wheel.') from exc
    raise ValueError('This wheel does not contain the required 2Cut2DGS bindings. '
                     'Standard Open3D will not be installed.')


def ensure_open3d(python, directory, wheel, args):
    if wheel:
        install_open3d_wheel(python, wheel)
    elif has_extended_open3d(python):
        print('Preserving the existing 2Cut2DGS installation.', flush=True)
    else:
        print('2Cut2DGS is missing. Building the pinned Open3D submodule.\n'
              'Requires Visual Studio C++ tools and CMake; no PowerShell is used.\n'
              'To use a prebuilt wheel instead: setup.bat --open3d-wheel "path/to/extended.whl"', flush=True)
        build_args = ['--venv-path', directory, '--build-path', args.build_path, '--jobs', str(args.jobs)]
        if args.cmake_exe:
            build_args += ['--cmake-exe', args.cmake_exe]
        run(python, '-B', ROOT/'tools/build_open3d.py', *build_args)


def validate_runtime():
    """Executed by the target venv, including a real CUDA compilation and launch."""
    sys.path.insert(0, str(ROOT))
    import numpy
    import scipy
    import numba
    import PySide6
    from PySide6 import QtCore, QtWidgets
    print('Python:', sys.version.split()[0])
    validate_open3d()
    try:
        from cutsim.cuda.profile_dexel_cuda import runtime
        rt = runtime()
        with rt.context():
            cp = rt.cp
            kernel = cp.RawKernel('''
            #include <math_constants.h>
            extern "C" __global__ void setup_probe(double* x) {
                x[0] = CUDART_INF; x[1] = 7.0;
            }
            ''', 'setup_probe')
            result = cp.empty(2, cp.float64)
            kernel((1,), (1,), (result,))
            values = cp.asnumpy(result)
            assert numpy.isposinf(values[0]) and values[1] == 7.0
        print('CUDA kernel compilation and execution: OK -', rt.name)
    except Exception as exc:
        raise RuntimeError('CUDA validation failed: ' + str(exc) + '\n'
                           'Check the NVIDIA driver and GPU. If --use-system-cuda was used, '
                           'check the CUDA 12 Toolkit, or rerun without that option.') from exc
    from cutsim.core.machining_data import simulation_input
    program, _ = simulation_input()
    print('Default machining data:', len(program), 'cutting blocks')
    print('Environment validation passed.')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Create/update a Windows Python 3.12 x64 venv with 2Cut2DGS. '
                    'Builds the Open3D submodule unless an extended build is installed or supplied. No PowerShell required.',
        epilog='Examples: setup.bat | setup.bat --check-only | '
               'setup.bat --open3d-wheel "D:\\packages\\open3d.whl"')
    parser.add_argument('--python-exe', '-PythonExe', help='Python 3.12 x64 executable used to create the venv')
    parser.add_argument('--venv-path', '-VenvPath', default='.venv', help='venv path, relative to the repository')
    parser.add_argument('--open3d-wheel', '-Open3DWheel', help='local extended Open3D wheel (relative to current directory)')
    parser.add_argument('--use-system-cuda', '-UseSystemCuda', action='store_true', help='use an installed CUDA 12 Toolkit')
    parser.add_argument('--check-only', '-CheckOnly', action='store_true', help='validate existing environment without package installation')
    parser.add_argument('--build-path', default='build/open3d', help='Open3D build directory, relative to repository')
    parser.add_argument('--cmake-exe', help='optional cmake.exe path for the Open3D build')
    parser.add_argument('--jobs', type=int, choices=range(1, 257), metavar='N', default=min(4, os.cpu_count() or 1), help='parallel build jobs (default: at most 4)')
    parser.add_argument('--validate-runtime', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--check-open3d', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if os.name != 'nt':
        raise RuntimeError('This setup script supports Windows x64 only.')
    if args.validate_runtime:
        validate_runtime()
        return
    if args.check_open3d:
        validate_open3d()
        return
    if args.check_only and args.open3d_wheel:
        raise ValueError('--check-only cannot install --open3d-wheel.')
    wheel = Path(args.open3d_wheel).resolve() if args.open3d_wheel else None
    if wheel and (not wheel.is_file() or wheel.suffix.lower() != '.whl'):
        raise ValueError('Open3DWheel must point to an existing .whl file.')
    if wheel:
        check_extended_wheel(wheel)
    requirements = ROOT / 'requirements.txt'
    if not requirements.is_file():
        raise FileNotFoundError('Missing distribution file: requirements.txt')
    directory = project_path(args.venv_path)
    python = directory / 'Scripts/python.exe'
    if not python.is_file():
        if args.check_only:
            raise RuntimeError('No environment found. Run setup.bat first.')
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise RuntimeError('The target directory is not an intact venv. Choose a new --venv-path; existing files will not be removed.')
        bootstrap = (shutil.which(args.python_exe) or args.python_exe) if args.python_exe else sys.executable
        run(bootstrap, '-B', '-c', PYTHON_CHECK)
        print('Creating environment:', directory, flush=True)
        run(bootstrap, '-m', 'venv', directory)
    if not (directory / 'pyvenv.cfg').is_file():
        raise RuntimeError('The target is not a Python virtual environment (pyvenv.cfg is missing).')
    run(python, '-B', '-c', PYTHON_CHECK)
    run(python, '-B', '-c', 'import sys; assert sys.prefix != sys.base_prefix, "A virtual environment is required"')
    if not args.check_only:
        run(python, '-m', 'pip', 'install', '--upgrade', 'pip')
        install = ['-m', 'pip', 'install', '-r', requirements]
        if not args.use_system_cuda:
            # requirements.txt pins the version for both the base package and extras.
            install.append('cupy-cuda12x[ctk]')
        run(python, *install)
        ensure_open3d(python, directory, wheel, args)
    run(python, '-m', 'pip', 'check')
    run(python, '-B', Path(__file__).resolve(), '--validate-runtime')
    print('\nReady. Start the application from cmd.exe with:')
    print(f'"{python}" "{ROOT / "run_app.py"}"')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        sys.exit(1)
