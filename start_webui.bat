@echo off
title Microduck RL WebUI
cd /d "%~dp0"
echo ========================================================
echo Starting Microduck RL WebUI ...
echo ========================================================

if exist "..\microduck_rl\.venv\Scripts\python.exe" (
    "..\microduck_rl\.venv\Scripts\python.exe" start_webui.py
) else (
    uv run python start_webui.py
)
pause
