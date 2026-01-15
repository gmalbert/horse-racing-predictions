@echo off
REM Quick batch prediction generator for Windows
REM Activates venv and runs batch prediction script

echo.
echo ========================================
echo   Batch Prediction Generator
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run batch script
python scripts\batch_generate_predictions.py %*

REM Keep window open if run by double-clicking
if "%1"=="" pause
