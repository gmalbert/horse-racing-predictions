# Batch Prediction Automation

## Overview

Automatically generate predictions for racecards in `data/raw/` without manual intervention.

## 🔥 **NEW: Automatic File Watcher** (Recommended)

The watcher runs continuously and automatically processes new racecards as soon as they're copied to `data/raw/`.

### Start the Watcher

**PowerShell:**
```powershell
.\start_watcher.ps1
```

**CMD:**
```cmd
start_watcher.bat
```

**Direct Python:**
```bash
python scripts/watch_racecards.py
```

### How It Works

1. **Monitors** `data/raw/` folder in real-time
2. **Detects** when `racecards_YYYY-MM-DD.json` files are created/updated
3. **Automatically runs** predictions for new dates
4. **Skips** dates that already have predictions
5. **Runs continuously** until you press Ctrl+C

### Example Output

```
======================================================================
  🔭 RACECARD FILE WATCHER
======================================================================

  Monitoring: data/raw
  Pattern:    racecards_YYYY-MM-DD.json
  Auto-run:   Enabled

  Press Ctrl+C to stop watching

======================================================================

✓ Watching for new racecards...

[14:32:15] 📥 New racecard detected: racecards_2026-01-17.json

======================================================================
  🔮 Auto-generating predictions for 2026-01-17
======================================================================
  ✓ Success: Predictions ready for 2026-01-17
  📁 Output: data/processed/predictions_2026-01-17.csv
======================================================================
```

---

## Manual Batch Processing

If you prefer to process all racecards at once instead of watching:

### Option 1: One-Command (PowerShell)
```powershell
.\run_batch_predictions.ps1
```

### Option 2: One-Command (CMD)
```cmd
run_batch_predictions.bat
```

### Option 3: Direct Python
```bash
python scripts/batch_generate_predictions.py
```

## What It Does

1. **Scans** `data/raw/` for all `racecards_YYYY-MM-DD.json` files
2. **Checks** which dates already have predictions in `data/processed/`
3. **Generates** predictions only for missing dates
4. **Outputs** `predictions_YYYY-MM-DD.csv` for each date

## Usage Examples

### Preview what will be processed (dry run)
```bash
python scripts/batch_generate_predictions.py --dry-run
```

### Process all racecards (skips existing)
```bash
python scripts/batch_generate_predictions.py
```

### Regenerate all predictions (force)
```bash
python scripts/batch_generate_predictions.py --force
```

### Process specific date range
```bash
python scripts/batch_generate_predictions.py --start-date 2026-01-01 --end-date 2026-01-31
```

## Workflow for New Racecards

### Automatic (Recommended)

1. **Start the watcher once**
   ```powershell
   .\start_watcher.ps1
   ```

2. **Copy racecards** to `data/raw/`
   - Format: `racecards_YYYY-MM-DD.json`
   - Predictions generate automatically!

3. **View in Streamlit**
   ```bash
   streamlit run predictions.py
   ```

### Manual

1. **Copy racecards** to `data/raw/`
   - Format: `racecards_YYYY-MM-DD.json`

2. **Run batch script**
   ```bash
   python scripts/batch_generate_predictions.py
   ```

3. **View in Streamlit**
   ```bash
   streamlit run predictions.py
   ```

## Example Output

```
======================================================================
  BATCH PREDICTION GENERATOR
======================================================================

Found 21 racecard file(s)

----------------------------------------------------------------------
SUMMARY:
----------------------------------------------------------------------
  Total racecards:           21
  Already have predictions:  19
  Need predictions:          2

Date            Status               Racecard File
----------------------------------------------------------------------
2025-12-17      NEW                  racecards_2025-12-17.json
2025-12-18      NEW                  racecards_2025-12-18.json
2026-01-01      SKIP (exists)        racecards_2026-01-01.json
...

======================================================================
  PROCESSING PREDICTIONS
======================================================================

[1/2] Processing 2025-12-17...
  ✓ Success: Predictions generated for 2025-12-17

[2/2] Processing 2025-12-18...
  ✓ Success: Predictions generated for 2025-12-18

======================================================================
  FINAL SUMMARY
======================================================================
  ✓ Successful: 2/2

🎉 All predictions generated successfully!
```

## Command Line Options

### Batch Script Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview what will be processed without running |
| `--force` | Regenerate predictions even if they already exist |
| `--start-date YYYY-MM-DD` | Only process racecards from this date onwards |
| `--end-date YYYY-MM-DD` | Only process racecards up to this date |

### Watcher Script Options

| Option | Description |
|--------|-------------|
| `--once` | Process existing files once and exit (don't watch) |
| `--no-auto` | Just notify about new files, don't auto-process |

## Files Created

For each racecard date, generates:
- `data/processed/predictions_YYYY-MM-DD.csv` - Win/place/show probabilities with odds

## Troubleshooting

### No racecards found
- Ensure files are named: `racecards_YYYY-MM-DD.json`
- Check they're in `data/raw/` directory

### Prediction fails
- Verify historical data exists: `data/processed/race_scores_with_betting_tiers.parquet`
- Ensure models are trained: `models/horse_win_predictor.pkl`
- Check racecard format matches expected structure

### Skip regenerating
- By default, existing predictions are skipped
- Use `--force` to regenerate all predictions

## Integration with UI

After running batch predictions, the Streamlit app will automatically detect the new prediction files. Simply refresh the app and select the date in the "Today & Tomorrow" tab.
