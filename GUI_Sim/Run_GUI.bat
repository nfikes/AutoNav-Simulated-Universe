@echo off
setlocal
set "ROOT=%~dp0"
set "SIM=%ROOT%simulated_world"
set "PY=%SIM%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python venv not found at:
    echo   %PY%
    echo Run uv venv inside simulated_world\ to recreate it.
    pause
    exit /b 1
)

cd /d "%SIM%"
"%PY%" "%SIM%\runner.py"
if errorlevel 1 pause
