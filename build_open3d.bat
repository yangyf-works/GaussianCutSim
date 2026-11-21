@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Build 2Cut2DGS without PowerShell. Options: build_open3d.bat --help
if defined GS_SETUP_PYTHON goto explicit_python
if exist "%~dp0.venv\Scripts\python.exe" goto existing_python
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 goto launcher_python
python -c "import sys" >nul 2>&1
if not errorlevel 1 goto path_python
echo Python was not found. Run setup.bat first or set GS_SETUP_PYTHON. 1>&2
exit /b 1

:explicit_python
"%GS_SETUP_PYTHON%" -B "%~dp0tools\build_open3d.py" %*
exit /b %errorlevel%

:existing_python
"%~dp0.venv\Scripts\python.exe" -B "%~dp0tools\build_open3d.py" %*
exit /b %errorlevel%

:launcher_python
py -3.12 -B "%~dp0tools\build_open3d.py" %*
exit /b %errorlevel%

:path_python
python -B "%~dp0tools\build_open3d.py" %*
exit /b %errorlevel%
