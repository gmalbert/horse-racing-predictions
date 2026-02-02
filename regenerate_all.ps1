#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Regenerate All Features and Model Pipeline
    
.DESCRIPTION
    Runs the complete data pipeline to regenerate cleaned data, features, and model.
    This should be run after appending new race data to all_gb_races.parquet.
    
    Pipeline steps:
    1. Phase 1: Data cleaning and filtering
    2. Phase 2: Race profitability scoring
    3. Enhanced form features (6 features)
    4. Connections form V2 features (13 features)
    5. Phase 3: Model training (91 features)
    6. Generate predictions for today
    
.PARAMETER SkipPredictions
    Skip the final prediction generation step
    
.PARAMETER DateForPredictions
    Specific date to generate predictions for (YYYY-MM-DD). Default is today.
    
.EXAMPLE
    .\regenerate_all.ps1
    
.EXAMPLE
    .\regenerate_all.ps1 -SkipPredictions
    
.EXAMPLE
    .\regenerate_all.ps1 -DateForPredictions "2026-02-15"
#>

param(
    [switch]$SkipPredictions,
    [string]$DateForPredictions = ""
)

# Color output functions
function Write-Step {
    param([string]$Message)
    Write-Host "`n$('='*70)" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host $('='*70) -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Message {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Yellow
}

# Error handling
$ErrorActionPreference = "Stop"

# Track timing
$TotalStartTime = Get-Date

Write-Host @"

╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    🏇 HORSE RACING PREDICTIONS - FULL PIPELINE REGENERATION      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Magenta

Write-Info "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Info "Working directory: $PWD"

# Verify virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Info "Virtual environment not detected. Attempting to activate..."
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        & .venv\Scripts\Activate.ps1
        Write-Success "Virtual environment activated"
    } else {
        Write-Error-Message "Virtual environment not found. Please activate .venv first."
        exit 1
    }
}

# Step 1: Phase 1 Data Cleaning
Write-Step "STEP 1/10: Phase 1 - Data Cleaning & Filtering"
$Step1Start = Get-Date
try {
    python scripts/phase1_data_cleaning.py
    $Step1Duration = (Get-Date) - $Step1Start
    Write-Success "Phase 1 completed in $($Step1Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "Phase 1 failed: $_"
    exit 1
}

# Step 2: Phase 2 Race Scoring
Write-Step "STEP 2/10: Phase 2 - Race Profitability Scoring"
$Step2Start = Get-Date
try {
    python scripts/phase2_score_races.py
    $Step2Duration = (Get-Date) - $Step2Start
    Write-Success "Phase 2 completed in $($Step2Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "Phase 2 failed: $_"
    exit 1
}

# Step 3: Enhanced Form Features
Write-Step "STEP 3/10: Enhanced Form Features (6 new features)"
$Step3Start = Get-Date
try {
    python scripts/add_enhanced_form_features.py
    $Step3Duration = (Get-Date) - $Step3Start
    Write-Success "Enhanced form features completed in $($Step3Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "Enhanced form features failed: $_"
    exit 1
}

# Step 4: Connections Form V2 Features
Write-Step "STEP 4/10: Connections Form V2 (13 new features)"
$Step4Start = Get-Date
Write-Info "This step may take 15-20 minutes..."
try {
    python scripts/add_connections_form_v2.py
    $Step4Duration = (Get-Date) - $Step4Start
    Write-Success "Connections form V2 completed in $($Step4Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "Connections form V2 failed: $_"
    exit 1
}

# Step 5: Pedigree Features
Write-Step "STEP 5/10: Pedigree Features (12 new features)"
$Step5Start = Get-Date
Write-Info "This step may take 10-15 minutes..."
try {
    python scripts/add_pedigree_features.py
    $Step5Duration = (Get-Date) - $Step5Start
    Write-Success "Pedigree features completed in $($Step5Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "Pedigree features failed: $_"
    exit 1
}

# Step 6: Going Preference Features
Write-Step "STEP 6/10: Going Preference Features (6 new features)"
$Step6Start = Get-Date
try {
    python scripts/add_going_preference_features.py
    $Step6Duration = (Get-Date) - $Step6Start
    Write-Success "Going preference features completed in $($Step6Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "Going preference features failed: $_"
    exit 1
}

# Step 7: OR Context Features
Write-Step "STEP 7/10: OR Context Features (13 new features)"
$Step7Start = Get-Date
try {
    python scripts/add_or_context_features.py
    $Step7Duration = (Get-Date) - $Step7Start
    Write-Success "OR context features completed in $($Step7Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "OR context features failed: $_"
    exit 1
}

# Step 8: Phase 3 Model Training
Write-Step "STEP 8/10: Phase 3 - Model Training (122 features)"
$Step8Start = Get-Date
try {
    python scripts/phase3_build_horse_model.py
    $Step8Duration = (Get-Date) - $Step8Start
    Write-Success "Model training completed in $($Step7Duration.TotalMinutes.ToString('0.00')) minutes"
} catch {
    Write-Error-Message "Model training failed: $_"
    exit 1
}

# Step 9: Generate Predictions (optional)
if (-not $SkipPredictions) {
    Write-Step "STEP 9/10: Generate Predictions"
    $Step9Start = Get-Date
    try {
        if ($DateForPredictions) {
            Write-Info "Generating predictions for: $DateForPredictions"
            python scripts/predict_todays_races.py --date $DateForPredictions
        } else {
            Write-Info "Generating predictions for today"
            python scripts/predict_todays_races.py
        }
        $Step9Duration = (Get-Date) - $Step9Start
        Write-Success "Predictions generated in $($Step9Duration.TotalMinutes.ToString('0.00')) minutes"
    } catch {
        Write-Error-Message "Prediction generation failed: $_"
        exit 1
    }
} else {
    Write-Info "Skipping prediction generation (--SkipPredictions flag set)"
}

# Summary
$TotalDuration = (Get-Date) - $TotalStartTime

Write-Host @"

╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                    ✅ PIPELINE COMPLETED SUCCESSFULLY             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "⏱️  TIMING SUMMARY:" -ForegroundColor Cyan
Write-Host "   Step 1 (Data Cleaning):        $($Step1Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
Write-Host "   Step 2 (Race Scoring):         $($Step2Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
Write-Host "   Step 3 (Enhanced Form):        $($Step3Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
Write-Host "   Step 4 (Connections V2):       $($Step4Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
Write-Host "   Step 5 (Pedigree Features):    $($Step5Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
Write-Host "   Step 6 (Going Preferences):    $($Step6Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
Write-Host "   Step 7 (OR Context):           $($Step7Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
Write-Host "   Step 8 (Model Training):       $($Step8Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
if (-not $SkipPredictions) {
    Write-Host "   Step 9 (Predictions):          $($Step9Duration.TotalMinutes.ToString('0.00')) min" -ForegroundColor White
}
Write-Host "   ──────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "   TOTAL:                         $($TotalDuration.TotalMinutes.ToString('0.00')) min" -ForegroundColor Yellow
Write-Host ""

Write-Success "All pipeline steps completed successfully!"
Write-Info "Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
