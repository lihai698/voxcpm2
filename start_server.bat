@echo off
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

if "%VOXCPM_PYTHON%"=="" (
    set "VOXCPM_PYTHON=E:\ziyuan\VoxCPM-2.0.2-20260505\jian27\python.exe"
)
if "%VOXCPM_HOST%"=="" (
    set "VOXCPM_HOST=127.0.0.1"
)
if "%VOXCPM_PORT%"=="" (
    set "VOXCPM_PORT=8808"
)

set "MODEL_DIR=%PROJECT_DIR%pretrained_models\VoxCPM2"
set "PYTHONPATH=%PROJECT_DIR%src"
set "VOXCPM_URL=http://%VOXCPM_HOST%:%VOXCPM_PORT%"

title VoxCPM2 Server %VOXCPM_PORT%

echo Starting VoxCPM2...
echo Project: %PROJECT_DIR%
echo Python: %VOXCPM_PYTHON%
echo Model:  %MODEL_DIR%
echo URL:    %VOXCPM_URL%
echo.

if not exist "%VOXCPM_PYTHON%" (
    echo [ERROR] Python not found: %VOXCPM_PYTHON%
    echo Set VOXCPM_PYTHON to your Python executable path and try again.
    pause
    exit /b 1
)

set "EXISTING_STATUS=0"
for /f %%S in ('powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri '%VOXCPM_URL%/config' -UseBasicParsing -TimeoutSec 2).StatusCode } catch { 0 }"') do set "EXISTING_STATUS=%%S"
if "%EXISTING_STATUS%"=="200" (
    echo VoxCPM2 is already running.
    echo Opening %VOXCPM_URL%
    start "" "%VOXCPM_URL%"
    pause
    exit /b 0
)

"%VOXCPM_PYTHON%" scripts\preflight.py --model-dir "%MODEL_DIR%" --host "%VOXCPM_HOST%" --port %VOXCPM_PORT%
if errorlevel 1 (
    echo.
    echo Startup checks failed. Fix the issue above and try again.
    pause
    exit /b 1
)

echo.
echo Open %VOXCPM_URL% in your browser.
echo.
"%VOXCPM_PYTHON%" app.py --model-id "%MODEL_DIR%" --port %VOXCPM_PORT% --host "%VOXCPM_HOST%"
pause
