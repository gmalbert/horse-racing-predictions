# Appending New Race Data

This guide explains how to add new race data to the historical dataset using the automated append script.

## Quick Start

```powershell
# 1. Dry run first to preview changes
python scripts/append_new_race_data.py path/to/new_data.csv --dry-run

# 2. Actually append the data
python scripts/append_new_race_data.py path/to/new_data.csv
```

## The Append Script

**Location:** `scripts/append_new_race_data.py`

**Purpose:** Safely append new racing data to `data/processed/all_gb_races.parquet` with automatic deduplication and validation.

## Features

✅ **Automatic deduplication** - removes records that already exist (based on race_id + horse_id)  
✅ **Data type consistency** - ensures new data matches historical data types  
✅ **Missing column handling** - adds any missing columns with NaN values  
✅ **Automatic backup** - creates timestamped backup before appending  
✅ **Dry-run mode** - preview changes without modifying data  
✅ **Date range validation** - reports date coverage  

## Usage Examples

### Basic Usage

```powershell
# Append new data with automatic backup
python scripts/append_new_race_data.py data/raw/new_races_2026.csv
```

### Dry Run (Recommended First Step)

```powershell
# Preview what would be added without modifying files
python scripts/append_new_race_data.py data/raw/new_races_2026.csv --dry-run
```

Expected output:
```
============================================================
📈 SUMMARY
============================================================
Historical data:  630,911 rows
New data loaded:  8,913 rows
Duplicates removed: 260 rows
New data added:   8,653 rows
Total after append: 639,564 rows
Date range:       2015-01-01 to 2026-01-31
============================================================
```

### Skip Backup (Not Recommended)

```powershell
# Append without creating backup (use with caution)
python scripts/append_new_race_data.py data/raw/new_races.csv --no-backup
```

### Custom Output File

```powershell
# Save to a different file
python scripts/append_new_race_data.py new_data.csv --output data/processed/all_gb_races_updated.parquet
```

## Expected CSV Format

The new data CSV should match the existing historical data structure (51 columns):

```csv
date,region,course_id,course,course_detail,race_id,off,race_name,type,class,pattern,rating_band,age_band,sex_rest,dist,dist_f,dist_m,dist_y,going,surface,ran,num,pos,draw,ovr_btn,btn,horse_id,horse,age,sex,lbs,hg,time,secs,dec,jockey_id,jockey,trainer_id,trainer,prize,or,rpr,sire_id,sire,dam_id,dam,damsire_id,damsire,owner_id,owner,silk_url
2026-01-15,GB,393,Lingfield (AW),,912345,14:30,Example Race,Flat,Class 4,...
```

## Workflow: Adding New Data

### Step 1: Obtain New Data

Place your new race data CSV in an accessible location:
```powershell
# Example: from external data scrape
c:\Users\gmalb\Downloads\horse-racing-data-scrape\data\region\gb\flat\2025_12_15_2026_01_31.csv
```

### Step 2: Preview Changes (Dry Run)

Always run dry-run first to verify:
```powershell
python scripts/append_new_race_data.py "c:\Users\gmalb\Downloads\horse-racing-data-scrape\data\region\gb\flat\2025_12_15_2026_01_31.csv" --dry-run
```

Review the output:
- ✓ Check duplicates removed count is reasonable
- ✓ Verify new data added count matches expectations
- ✓ Confirm date range extends as expected

### Step 3: Execute Append

If dry-run looks good, run without `--dry-run`:
```powershell
python scripts/append_new_race_data.py "c:\Users\gmalb\Downloads\horse-racing-data-scrape\data\region\gb\flat\2025_12_15_2026_01_31.csv"
```

The script will:
1. Create automatic backup: `data/processed/backups/all_gb_races_backup_YYYYMMDD_HHMMSS.parquet`
2. Append new records
3. Save updated file to `data/processed/all_gb_races.parquet`

### Step 4: Regenerate Features

After appending new data, you need to regenerate the processed datasets.

**Option A: Use the automated regeneration script (recommended)**

```powershell
# Run the full pipeline (all 6 steps)
.\regenerate_all.ps1

# Or use the batch file
.\regenerate_all.bat

# Skip predictions step
.\regenerate_all.ps1 -SkipPredictions

# Generate predictions for specific date
.\regenerate_all.ps1 -DateForPredictions "2026-02-15"
```

**Option B: Run steps manually**

```powershell
# Regenerate cleaned data and scores
python scripts/phase1_data_cleaning.py
python scripts/phase2_score_races.py
python scripts/add_enhanced_form_features.py
python scripts/add_connections_form_v2.py
python scripts/phase3_build_horse_model.py
python scripts/predict_todays_races.py
```

**Option C: Wait for automation**

The weekly GitHub Actions workflow will automatically run steps 2-6 every Monday at 07:00 UTC.

## Backups

Backups are stored in: `data/processed/backups/`

Example backup file: `all_gb_races_backup_20260201_143022.parquet`

### Restoring from Backup

If you need to restore:
```powershell
# Copy backup back to main file
copy data/processed/backups/all_gb_races_backup_20260201_143022.parquet data/processed/all_gb_races.parquet
```

## Deduplication Logic

The script deduplicates using a composite key:
```
race_id + horse_id
```

This ensures:
- Same horse in same race = duplicate (skipped)
- Same horse in different race = new record (added)
- Different horse in same race = new record (added)

## Data Validation

The script performs these validation steps:

1. **Column Matching**: Ensures new CSV has same 51 columns
2. **Data Type Consistency**: Converts new data types to match historical data
3. **Missing Columns**: Adds any missing columns with NaN values
4. **Date Parsing**: Converts date strings to datetime objects
5. **Sorting**: Ensures final dataset is sorted by date

## Troubleshooting

### "No new data to append"

All records were duplicates. This is normal if you're re-running with the same data.

### "Column mismatch"

Your CSV is missing columns or has extra columns. Ensure your CSV matches the expected 51-column format.

### "Data type conversion error"

The script will report which column failed to convert. Check that column's data format in your CSV.

## Integration with Data Pipeline

After appending new data, the full pipeline is:

```
1. append_new_race_data.py       ← Add raw race results
2. phase1_data_cleaning.py       ← Clean and filter data
3. phase2_score_races.py         ← Score race quality
4. add_enhanced_form_features.py ← Add form features (6 new)
5. add_connections_form_v2.py    ← Add connections (13 new)
6. phase3_build_horse_model.py   ← Retrain model
7. predict_todays_races.py       ← Generate predictions
```

The weekly GitHub Actions workflow handles steps 2-6 automatically.

## Best Practices

✅ **Always dry-run first** - preview changes before committing  
✅ **Check for duplicates** - review the "Duplicates removed" count  
✅ **Keep backups** - don't use `--no-backup` unless necessary  
✅ **Verify date ranges** - ensure new data extends the dataset correctly  
✅ **Regenerate features** - run feature engineering scripts after appending  

## Example: Full Workflow

```powershell
# 1. Preview the append
python scripts/append_new_race_data.py "c:\path\to\new_data.csv" --dry-run

# 2. Actually append
python scripts/append_new_race_data.py "c:\path\to\new_data.csv"

# 3. Regenerate everything (automated script)
.\regenerate_all.ps1

# OR run steps manually:
# python scripts/phase1_data_cleaning.py
# python scripts/phase2_score_races.py
# python scripts/add_enhanced_form_features.py
# python scripts/add_connections_form_v2.py
# python scripts/phase3_build_horse_model.py
# python scripts/predict_todays_races.py
```

## Automated Regeneration Scripts

Two convenience scripts are provided to run the full pipeline:

### PowerShell Script (`regenerate_all.ps1`)

**Features:**
- ✅ Colored progress output
- ✅ Timing for each step
- ✅ Error handling with clear messages
- ✅ Optional parameters for customization

**Usage:**
```powershell
# Basic usage - run all steps
.\regenerate_all.ps1

# Skip prediction generation
.\regenerate_all.ps1 -SkipPredictions

# Generate predictions for specific date
.\regenerate_all.ps1 -DateForPredictions "2026-02-15"
```

**Example output:**
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
```

### Batch Script (`regenerate_all.bat`)

Simpler batch file alternative for Windows command prompt.

**Usage:**
```cmd
regenerate_all.bat
```

## See Also

- [Data Pipeline Documentation](../README.md#data-pipeline)
- [Feature Engineering V2](FEATURE_ENGINEERING_V2.md)
- [Model Training](../scripts/phase3_build_horse_model.py)
