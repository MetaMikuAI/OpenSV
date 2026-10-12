@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
    echo uv is required. Install it from https://docs.astral.sh/uv/ and retry.
    exit /b 1
)
if not exist "%~dp0opensv.exe" (
    echo Bundled opensv.exe is missing. Extract the complete archive.
    exit /b 1
)
uv run --project "%~dp0training" --locked --no-group cpu --extra cuda opensv-gui --engine "%~dp0opensv.exe" %*
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
