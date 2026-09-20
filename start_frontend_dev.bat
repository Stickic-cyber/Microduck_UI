@echo off
title Microduck RL WebUI - Frontend Dev Server
cd /d "%~dp0frontend"
echo ========================================================
echo Starting Vite Dev Server on http://localhost:3000 ...
echo Proxying /api and /ws to http://localhost:8000
echo ========================================================
set PATH=D:\Program Files\nodejs;%PATH%
npm run dev
pause
