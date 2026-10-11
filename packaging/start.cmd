@echo off
cd /d "%~dp0"
uv run --project training --locked opensv-gui
pause
