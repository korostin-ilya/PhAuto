@echo off
setlocal
cd /d "%~dp0\.."

echo ============================================================================
echo [1/2] Checking AWS SSO session and updating .env via Python...
echo ============================================================================

set "PYTHON_EXE="

if exist "%~dp0\..\.venv\Scripts\python.exe" (
    "%~dp0\..\.venv\Scripts\python.exe" -V >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=%~dp0\..\.venv\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" -V >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
)

if "%PYTHON_EXE%"=="" (
    python -V >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if "%PYTHON_EXE%"=="" (
    py -V >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=py"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python interpreter not found in .venv, AppData, PATH, or py launcher.
    pause
    exit /b 1
)

echo Found Python: %PYTHON_EXE%

"%PYTHON_EXE%" scripts\auth_aws.py
if errorlevel 1 (
    echo [ERROR] AWS SSO authentication or .env export failed.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo [2/2] Starting async S3 download and panorama pipeline in Docker...
echo ============================================================================
docker compose run --rm --entrypoint python3 pano-pipeline -u scripts/async_pipeline.py %*
pause
