@echo off
REM Bypass the launcher panel and run lidar_sim_gui.py with any args passed in.
REM Example: Run_LIDAR_SIM_direct.bat --threshold 20 --rays 1000
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
"%PY%" "%SIM%\lidar_sim_gui.py" %*
if errorlevel 1 pause
