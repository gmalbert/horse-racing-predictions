# GitHub Actions Cache Corruption Fix

## Problem

The weekly model training workflow fails with:

```
❌ Cleaned data is invalid, regenerating...
pyarrow.lib.ArrowInvalid: Could not open Parquet input source '<Buffer>': 
Parquet magic bytes not found in footer. Either the file is corrupted or this is not a parquet file.
```

**Root Cause:**
- GitHub Actions cache can become corrupted during upload/download
- Parquet files (~74MB) may be incompletely cached
- Cache expiry (7 days without access) can lose data
- Network interruptions during cache operations

## Quick Fix

The workflow now **auto-detects and handles** cache corruption:

### Automatic Behavior (v2 - Current)

✅ **Validates cache before use**
- Checks if `all_gb_races.parquet` exists and is valid
- Checks if `all_gb_races_cleaned.parquet` exists and is valid
- Removes corrupted files automatically

✅ **Graceful degradation**
- If base data is missing: Skip workflow run with clear message
- If cleaned data is corrupted: Regenerates from base data
- Never crashes with unhelpful error messages

✅ **Clear user feedback**
```
╔════════════════════════════════════════════════════════════╗
║  ⚠️  CACHE MISS: Historical data not available             ║
╚════════════════════════════════════════════════════════════╝

To rebuild the cache:
  1. Run data aggregation locally to generate all_gb_races.parquet
  2. Push updated data files to repository
  3. Re-run this workflow
```

## Manual Cache Restoration

If the workflow skips due to missing cache:

### Option 1: Local Data Upload (Recommended)

```bash
# 1. Ensure you have the latest data locally
cd horse-racing-predictions
ls -lh data/processed/all_gb_races.parquet
# Should show ~74MB file

# 2. Commit and push (will be cached on next workflow run)
git add data/processed/all_gb_races.parquet
git commit -m "chore: restore historical race data cache"
git push
```

### Option 2: Regenerate from Scratch

If you don't have the file locally:

```bash
# 1. Run your data aggregation process
python scripts/aggregate_all_races.py  # Or your data source script

# 2. Verify the file
python -c "import pandas as pd; df = pd.read_parquet('data/processed/all_gb_races.parquet'); print(f'Loaded {len(df):,} rows')"
# Should output: Loaded 639,xxx rows

# 3. Commit and push
git add data/processed/all_gb_races.parquet
git commit -m "chore: regenerate historical race data"
git push
```

### Option 3: Invalidate and Rebuild Cache

Force GitHub Actions to rebuild cache from Git LFS (if available):

```bash
# Change the cache key in the workflow file
# Edit .github/workflows/weekly_model_training.yml
# Line ~28: change 'horse-racing-data-v2' to 'horse-racing-data-v3'

# Commit the change
git add .github/workflows/weekly_model_training.yml
git commit -m "chore: invalidate corrupted cache"
git push

# Next workflow run will pull fresh data from Git LFS (if within bandwidth quota)
```

## Prevention

### Best Practices

1. **Regular cache refresh**: The daily predictions workflow keeps cache alive
2. **Monitor cache size**: Check Actions → Caches in GitHub repository
3. **Keep local backups**: Maintain `all_gb_races.parquet` in local backups
4. **Validate before push**:
   ```bash
   python -c "import pandas as pd; pd.read_parquet('data/processed/all_gb_races.parquet')"
   ```

### Cache Limits

- **Size**: 10 GB per repository (currently using ~300MB)
- **Expiry**: 7 days without access
- **Key rotation**: Change key version to force rebuild

## Workflow Changes (v2)

### Before (v1 - Crashed on corruption)
```yaml
- name: Validate cached data
  run: |
    python -c "import pandas as pd; pd.read_parquet('data/processed/all_gb_races_cleaned.parquet')"
    # ❌ Crashes if file is corrupted
```

### After (v2 - Graceful handling)
```yaml
- name: Validate cached data
  id: validate_cache
  run: |
    # Validates files, removes corrupted ones
    # Sets has_data=false if base data missing
    # Regenerates cleaned data if needed
    # Provides clear user guidance
```

## Troubleshooting

### Error: "Workflow skipped: Cache corruption detected"

**Cause**: Base data file (`all_gb_races.parquet`) is missing or corrupted

**Fix**: Follow "Manual Cache Restoration" above

### Error: "Cleaned data is corrupted, removing..."

**Cause**: Derived file is corrupted but base data is OK

**Fix**: Workflow auto-regenerates cleaned data (no action needed)

### Workflow runs but produces no predictions

**Cause**: All steps skipped due to cache validation failure

**Fix**: Check workflow logs for validation step output, follow guidance

## Related Documentation

- **Cache strategy**: `docs/GITHUB_ACTIONS_LFS_FIX.md`
- **Data pipeline**: `docs/PIPELINE_REGENERATION.md`
- **Data appending**: `docs/APPENDING_NEW_DATA.md`

## Changes Made

**Files Modified:**
- `.github/workflows/weekly_model_training.yml` - Enhanced cache validation
- `scripts/phase1_data_cleaning.py` - Better error handling for missing files

**Features Added:**
- Parquet file validation before use
- Corrupted file auto-removal
- Graceful workflow degradation
- Clear user guidance messages
- Skip workflow if base data unavailable
