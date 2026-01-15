@echo off
REM Start racecard file watcher (Windows CMD)
REM Monitors data/raw/ for new racecards and auto-generates predictions

echo.
echo ========================================
echo   Racecard File Watcher
echo ========================================
echo.
echo Monitoring: data\raw\
echo Pattern: racecards_YYYY-MM-DD.json
echo.
echo Press Ctrl+C to stop watching
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run watcher
python scripts\watch_racecards.py

pause
