@echo off
rem Roche Scientific AI - desktop launcher
rem Starts the assistant server (if it isn't running yet) and opens the
rem chatbot in its own app window (no browser tabs / address bar).
rem Tip: right-click this file -> Send to -> Desktop (create shortcut).

cd /d "%~dp0"

rem Is the server already up?
curl -s -o nul -m 2 http://127.0.0.1:8000/ >nul 2>&1
if errorlevel 1 (
  echo Starting Roche Scientific AI server...
  start "Roche Scientific AI Server" /min "%~dp0.venv\Scripts\python.exe" "%~dp0src\api.py"
)

rem Wait until it answers (first start ingests documents, can take a minute)
set /a tries=0
:wait
curl -s -o nul -m 2 http://127.0.0.1:8000/ >nul 2>&1
if not errorlevel 1 goto open
set /a tries+=1
if %tries% geq 90 (
  echo Server did not start in time. Check the logs folder for errors.
  pause
  exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait

:open
start "" msedge --app=http://127.0.0.1:8000/
exit /b 0
