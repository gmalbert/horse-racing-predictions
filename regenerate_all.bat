@echo off
REM ============================================================================
REM  Horse Racing Predictions - Full Pipeline Regeneration
REM ============================================================================
REM  
REM  This script runs the complete data pipeline to regenerate cleaned data,
REM  features, and model after appending new race data.
REM  
REM  Pipeline steps:
REM  1. Phase 1: Data cleaning and filtering
REM  2. Phase 2: Race profitability scoring
REM  3. Enhanced form features (6 features)
REM  4. Connections form V2 features (13 features)
REM  5. Phase 3: Model training (91 features)
REM  6. Generate predictions for today
REM  
REM  Usage:
REM    regenerate_all.bat
REM  
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo   HORSE RACING PREDICTIONS - FULL PIPELINE REGENERATION
echo ========================================================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo [INFO] Activating virtual environment...
    if exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate.bat
        echo [OK] Virtual environment activated
    ) else (
        echo [ERROR] Virtual environment not found. Please activate .venv first.
        exit /b 1
    )
)

echo [INFO] Started at: %date% %time%
echo [INFO] Working directory: %cd%
echo.

REM Step 1: Phase 1 Data Cleaning
echo ========================================================================
echo   STEP 1/6: Phase 1 - Data Cleaning ^& Filtering
echo ========================================================================
echo.
python scripts/phase1_data_cleaning.py
if errorlevel 1 (
    echo [ERROR] Phase 1 failed
    exit /b 1
)
echo [OK] Phase 1 completed
echo.

REM Step 2: Phase 2 Race Scoring
echo ========================================================================
echo   STEP 2/6: Phase 2 - Race Profitability Scoring
echo ========================================================================
echo.
python scripts/phase2_score_races.py
if errorlevel 1 (
    echo [ERROR] Phase 2 failed
    exit /b 1
)
echo [OK] Phase 2 completed
echo.

REM Step 3: Enhanced Form Features
echo ========================================================================
echo   STEP 3/6: Enhanced Form Features (6 new features)
echo ========================================================================
echo.
python scripts/add_enhanced_form_features.py
if errorlevel 1 (
    echo [ERROR] Enhanced form features failed
    exit /b 1
)
echo [OK] Enhanced form features completed
echo.

REM Step 4: Connections Form V2 Features
echo ========================================================================
echo   STEP 4/6: Connections Form V2 (13 new features)
echo ========================================================================
echo [INFO] This step may take 15-20 minutes...
echo.
python scripts/add_connections_form_v2.py
if errorlevel 1 (
    echo [ERROR] Connections form V2 failed
    exit /b 1
)
echo [OK] Connections form V2 completed
echo.

REM Step 5: Phase 3 Model Training
echo ========================================================================
echo   STEP 5/6: Phase 3 - Model Training (91 features)
echo ========================================================================
echo.
python scripts/phase3_build_horse_model.py
if errorlevel 1 (
    echo [ERROR] Model training failed
    exit /b 1
)
echo [OK] Model training completed
echo.

REM Step 6: Generate Predictions
echo ========================================================================
echo   STEP 6/6: Generate Predictions
echo ========================================================================
echo [INFO] Generating predictions for today...
echo.
python scripts/predict_todays_races.py
if errorlevel 1 (
    echo [ERROR] Prediction generation failed
    exit /b 1
)
echo [OK] Predictions generated
echo.

REM Summary
echo.
echo ========================================================================
echo   PIPELINE COMPLETED SUCCESSFULLY
echo ========================================================================
echo.
echo [OK] All pipeline steps completed successfully!
echo [INFO] Finished at: %date% %time%
echo.

endlocal
