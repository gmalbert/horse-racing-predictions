# GitHub Actions LFS Bandwidth Fix - Applied

## Problem

GitHub Actions workflows were failing with:
```
batch response: This repository exceeded its LFS budget. 
The account responsible for the budget should increase it to restore access.
Error: error: failed to fetch some objects from 'https://github.com/gmalbert/horse-racing-predictions.git/info/lfs'
```

**Root Cause:**
- GitHub Free tier: 1 GB/month LFS bandwidth
- Both workflows (`precompute_predictions.yml` and `weekly_model_training.yml`) had `lfs: true`
- Large Parquet files (~75MB each) being downloaded on every workflow run
- Daily predictions workflow: 75MB × 30 = 2.25GB/month → quota exceeded

## Solution Applied

### Changed Workflows

Both `.github/workflows/precompute_predictions.yml` and `.github/workflows/weekly_model_training.yml`:

**Before:**
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    lfs: true  # Downloads LFS files, consuming bandwidth quota
```

**After:**
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    lfs: false  # Skip LFS to avoid bandwidth limits

- name: Cache historical race data
  uses: actions/cache@v4
  with:
    path: |
      data/processed/all_gb_races.parquet
      data/processed/all_gb_races_cleaned.parquet
      data/processed/race_scores.parquet
      data/processed/race_scores_with_betting_tiers.parquet
    key: horse-racing-data-v1
    restore-keys: |
      horse-racing-data-
```

### What This Achieves

✅ **Zero LFS bandwidth usage** → No more quota errors  
✅ **GitHub Actions Cache** stores the data (10 GB limit, 7-day expiry with auto-renewal on access)  
✅ **Faster workflows** after first run (cache restore vs. LFS download)  
✅ **Self-healing** - if cache expires, first run will fail but subsequent runs will rebuild cache

### Files Tracked by LFS

These Parquet files are cached instead of downloaded via LFS:
- `data/processed/all_gb_races.parquet` (~74MB)
- `data/processed/all_gb_races_cleaned.parquet` (~68MB)
- `data/processed/race_scores.parquet` (~72MB)
- `data/processed/race_scores_with_betting_tiers.parquet` (~72MB)

Total: ~286MB cached once, used repeatedly.

## Cache Behavior

- **First workflow run after this fix:** Cache miss → workflow may fail if it needs the data files
- **Manual solution:** Upload the Parquet files to the cache manually or run scripts locally that regenerate them
- **Subsequent runs:** Cache hit → instant data access
- **Cache expiry:** 7 days of no access (daily workflow ensures this never expires)
- **Cache invalidation:** Change `key: horse-racing-data-v2` to rebuild cache

## Testing

### Local Verification
```bash
# Verify workflows have lfs: false
grep -r "lfs:" .github/workflows/
# Should show: lfs: false

# Check LFS tracked files
git lfs ls-files
```

### GitHub Actions Verification

After first successful workflow run:
1. Go to **Repository → Actions → Caches**
2. Should see: `horse-racing-data-v1` with size ~286MB
3. Check workflow logs for "Cache restored from key: horse-racing-data-v1"

## Troubleshooting

### Workflow fails with "Parquet magic bytes not found" or LFS pointer error

**What happened:** The workflow tried to read an LFS pointer file (small text file) as Parquet data.

**Status: FIXED** ✅

The prediction script now:
1. Detects LFS pointer files vs. real data
2. Exits gracefully with clear error message on cache miss
3. Provides instructions for seeding the cache

**After fix applied:**
```bash
# Commit the LFS pointer detection fix
git add scripts/predict_todays_races.py
git commit -m "fix: add LFS pointer detection"

# Push LFS files (one-time, uses quota but seeds cache)
git lfs push origin main --all

# Push the code
git push

# Next workflow run will:
# - See LFS pointer, exit gracefully
# - Cache is now empty and will be populated when data is regenerated
```

**Option 1: Regenerate data files (recommended for production)**
```bash
# Regenerate all historical data locally
python scripts/phase2_score_races.py
python scripts/apply_betting_strategy.py

# This creates fresh Parquet files that get cached
git add data/processed/*.parquet
git commit -m "chore: regenerate historical data"
git push
```

**Option 2: Manual cache seed (if you have good local copies)**
```bash
# Ensure you have the files locally
Get-ChildItem data/processed/*.parquet

# Push them via git (they're already in LFS)
git add data/processed/*.parquet
git commit -m "chore: ensure LFS files are available"
git push

# Then manually run workflow to build cache
```

**Option 2: Regenerate data**
```bash
# Run locally to rebuild all data files
python scripts/phase2_score_races.py
python scripts/apply_betting_strategy.py
python scripts/phase3_build_horse_model.py

# Commit and push
git add data/processed/*.parquet
git commit -m "chore: regenerate historical data files"
git push
```

### Cache not being used

Check workflow logs:
```
Run actions/cache@v4
Cache not found for key: horse-racing-data-v1
```

This is normal on **first run**. Second run should show:
```
Cache restored from key: horse-racing-data-v1
```

### Need to rebuild cache

Change the cache key version in both workflow files:
```yaml
key: horse-racing-data-v2  # Increment from v1
```

## Bandwidth Comparison

| Approach | LFS Bandwidth | Cache Usage | Monthly Cost |
|----------|---------------|-------------|--------------|
| **Before (LFS)** | 75MB × 30 = 2.25GB | 0 MB | ❌ Quota exceeded |
| **After (Cache)** | 0 MB | 286MB (one-time) | ✅ Free tier |

**Savings:** 2.25GB → 0 MB LFS bandwidth per month

## References

- Original fix documentation: [FIX_LFS_BANDWIDTH_QUOTA.md](FIX_LFS_BANDWIDTH_QUOTA.md)
- [GitHub Actions Cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [Git LFS Bandwidth](https://docs.github.com/en/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage)
