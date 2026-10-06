@echo off
cd /d "%~dp0"
echo Starting Photography Agent MVP...
.\venv\Scripts\python.exe main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error.
    pause
)
