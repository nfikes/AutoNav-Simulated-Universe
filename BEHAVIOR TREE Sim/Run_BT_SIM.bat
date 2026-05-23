@echo off
REM Windows launcher panel. Creates the venv and installs requirements
REM with `uv` on first run, then opens the launcher panel. To force a
REM requirements reinstall (e.g. after editing requirements.txt), delete
REM simulated_world\.venv\.requirements.stamp before running.
setlocal
set "ROOT=%~dp0"
set "SIM=%ROOT%simulated_world"
set "VENV=%SIM%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "REQ=%SIM%\requirements.txt"
set "STAMP=%VENV%\.requirements.stamp"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 'uv' not found on PATH.
    echo Install from https://astral.sh/uv/
    pause
    exit /b 1
)

if not exist "%PY%" (
    echo [setup] Creating Python 3.12 venv at %VENV% ...
    uv venv --python 3.12 "%VENV%"
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

if not exist "%STAMP%" (
    echo [setup] Installing requirements from requirements.txt ...
    set "VIRTUAL_ENV=%VENV%"
    uv pip install -r "%REQ%"
    if errorlevel 1 (
        echo [ERROR] uv pip install failed.
        pause
        exit /b 1
    )
    echo done > "%STAMP%"
)

cd /d "%SIM%"
"%PY%" "%SIM%\launcher.py" %*
if errorlevel 1 pause
