@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem No PowerShell or venv activation is required. Options: setup.bat --help
rem Requires 2Cut2DGS: preserves an extended build or builds the Open3D submodule.
rem For a prebuilt wheel, use --open3d-wheel "path\to\extended.whl".
rem Set GS_SETUP_PYTHON to an executable path if Python is not on PATH.
if defined GS_SETUP_PYTHON goto explicit_python
if exist "%~dp0.venv\Scripts\python.exe" goto existing_python
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 goto launcher_python
python -c "import sys" >nul 2>&1
if not errorlevel 1 goto path_python
echo Python was not found. Install Python 3.12 x64 or set GS_SETUP_PYTHON. 1>&2
exit /b 1

:explicit_python
"%GS_SETUP_PYTHON%" -B "%~dp0tools\setup_environment.py" %*
exit /b %errorlevel%

:existing_python
"%~dp0.venv\Scripts\python.exe" -B "%~dp0tools\setup_environment.py" %*
exit /b %errorlevel%

:launcher_python
py -3.12 -B "%~dp0tools\setup_environment.py" %*
exit /b %errorlevel%

:path_python
python -B "%~dp0tools\setup_environment.py" %*
exit /b %errorlevel%
