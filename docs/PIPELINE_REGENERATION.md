# Pipeline Regeneration Scripts

## Overview

Two convenience scripts are provided to automatically run the complete data pipeline after appending new race data:

- **`regenerate_all.ps1`** - PowerShell script (recommended)
- **`regenerate_all.bat`** - Batch file alternative

## What They Do

These scripts run the 6-step pipeline to regenerate all features and the ML model:

1. **Phase 1**: Data cleaning and filtering
2. **Phase 2**: Race profitability scoring
3. **Enhanced Form**: Add 6 new form features
4. **Connections V2**: Add 13 new connections features (15-20 min)
5. **Phase 3**: Train ML model (91 features total)
6. **Predictions**: Generate predictions for today

## Usage

### PowerShell Script (Recommended)

```powershell
# Run full pipeline
.\regenerate_all.ps1

# Skip predictions step
.\regenerate_all.ps1 -SkipPredictions

# Generate predictions for specific date
.\regenerate_all.ps1 -DateForPredictions "2026-02-15"
```

### Batch File

```cmd
regenerate_all.bat
```

## Features

### PowerShell Script
- ✅ **Colored progress output** with step indicators
- ✅ **Timing** for each step and total duration
- ✅ **Error handling** with clear failure messages
- ✅ **Auto-activation** of virtual environment
- ✅ **Optional parameters** for customization

### Batch File
- ✅ **Simple Windows CMD** compatibility
- ✅ **Same pipeline** as PowerShell version
- ✅ **Basic error handling**

## Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║    🏇 HORSE RACING PREDICTIONS - FULL PIPELINE REGENERATION      ║
╚══════════════════════════════════════════════════════════════════╝

STEP 1/6: Phase 1 - Data Cleaning & Filtering
✓ Phase 1 completed in 2.34 minutes

STEP 2/6: Phase 2 - Race Profitability Scoring
✓ Phase 2 completed in 3.12 minutes

STEP 3/6: Enhanced Form Features (6 new features)
✓ Enhanced form features completed in 0.52 minutes

STEP 4/6: Connections Form V2 (13 new features)
ℹ This step may take 15-20 minutes...
✓ Connections form V2 completed in 16.23 minutes

STEP 5/6: Phase 3 - Model Training (91 features)
✓ Model training completed in 1.45 minutes

STEP 6/6: Generate Predictions
✓ Predictions generated in 0.18 minutes

⏱️  TIMING SUMMARY:
   Step 1 (Data Cleaning):        2.34 min
   Step 2 (Race Scoring):          3.12 min
   Step 3 (Enhanced Form):         0.52 min
   Step 4 (Connections V2):        16.23 min
   Step 5 (Model Training):        1.45 min
   Step 6 (Predictions):           0.18 min
   ──────────────────────────────────────
   TOTAL:                          23.84 min

✅ PIPELINE COMPLETED SUCCESSFULLY
```

## When to Use

Run these scripts **after** appending new race data with `append_new_race_data.py`:

```powershell
# 1. Append new data
python scripts/append_new_race_data.py new_races.csv

# 2. Regenerate everything
.\regenerate_all.ps1
```

## Time Estimate

- **Total runtime**: ~20-25 minutes
- **Bottleneck**: Connections V2 features (15-20 minutes)
- **Fastest step**: Predictions generation (10-20 seconds)

## Requirements

- **Virtual environment**: `.venv/` must exist
- **Python dependencies**: All packages in `requirements.txt`
- **Data**: `data/processed/all_gb_races.parquet` must exist

## Error Handling

Both scripts will:
- Stop on first error with clear message
- Show which step failed
- Preserve partial progress for debugging

## See Also

- [Appending New Data](APPENDING_NEW_DATA.md) - How to add new race data
- [Data Pipeline](../README.md#data-pipeline) - Full pipeline documentation
- [Feature Engineering V2](FEATURE_ENGINEERING_V2.md) - Feature details</content>
<parameter name="filePath">c:\Users\gmalb\Downloads\horse-racing-predictions\PIPELINE_REGENERATION.md