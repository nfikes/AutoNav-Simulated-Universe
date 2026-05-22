@echo off
setlocal enabledelayedexpansion
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

REM File picker: CSV (extracted playback) OR .db3 / metadata.yaml (raw rosbag2)
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Add-Type -AssemblyName System.Windows.Forms;" ^
  "$f = New-Object System.Windows.Forms.OpenFileDialog;" ^
  "$f.InitialDirectory = (@('%ROOT%data\example-playback-csv','%ROOT%data','%ROOT%') | Where-Object { Test-Path $_ } | Select-Object -First 1);" ^
  "$f.Filter = 'Bake input (*.csv;*.db3;metadata.yaml)|*.csv;*.db3;metadata.yaml|Playback CSV (*.csv)|*.csv|ROS bag (*.db3)|*.db3|Bag manifest (metadata.yaml)|metadata.yaml|All files (*.*)|*.*';" ^
  "$f.Title = 'Pick a playback CSV or a rosbag2 .db3 / metadata.yaml';" ^
  "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.FileName }"`) do (
    set "PICK=%%I"
)

if not defined PICK (
    echo No file selected, exiting.
    pause
    exit /b 1
)

REM If user picked a .db3 or metadata.yaml, the baker wants the *parent directory*.
set "SRC=!PICK!"
for %%F in ("!PICK!") do (
    set "EXT=%%~xF"
    set "NAME=%%~nxF"
    set "PARENT=%%~dpF"
)
if "!PARENT:~-1!"=="\" set "PARENT=!PARENT:~0,-1!"

if /i "!EXT!"==".db3" set "SRC=!PARENT!"
if /i "!NAME!"=="metadata.yaml" set "SRC=!PARENT!"

echo.
echo Picked:  !PICK!
echo Passing: !SRC!
echo Output:  %ROOT%data\Baked_Playbacks\^<stem^>_baked.mp4
echo.

cd /d "%SIM%"
"%PY%" "%SIM%\bake_offscreen.py" "!SRC!"

echo.
if errorlevel 1 (
    echo [Bake failed]
) else (
    echo [Done]
)
pause
