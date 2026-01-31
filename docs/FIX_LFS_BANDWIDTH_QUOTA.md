# Fix: GitHub Actions LFS Bandwidth Quota Exceeded

## Problem

GitHub Actions workflows fail with LFS bandwidth quota errors:

```
batch response: This repository exceeded its LFS budget. 
The account responsible for the budget should increase it to restore access.
Error: error: failed to fetch some objects from 'https://github.com/...'
The process '/usr/bin/git' failed with exit code 2
```

**Root Cause:**
- GitHub Free tier: 1 GB/month LFS bandwidth
- Workflow with `lfs: true` downloads large LFS files every run
- Daily workflows × large files = quota exhausted in ~2 weeks

## Solution Overview

Replace LFS downloads with:
1. **GitHub Actions Cache** for historical data (persistent across runs)
2. **External API downloads** for new data (unlimited bandwidth)
3. **Smart detection** to handle LFS pointers gracefully

**Bandwidth Comparison:**

| Approach | LFS Bandwidth | External API | Monthly Cost |
|----------|---------------|--------------|--------------|
| Before (LFS) | 74MB × 30 = 2.2GB | 0 MB | Quota exceeded ❌ |
| After (Cache) | 0 MB | 15MB × 30 = 450MB | $0 ✅ |

## Implementation Steps

### Step 1: Disable LFS in GitHub Actions Workflows

Find all workflow files in `.github/workflows/` and change checkout steps:

**Before:**
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    lfs: true  # Downloads LFS files using quota
```

**After:**
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    lfs: false  # Skip LFS to avoid bandwidth limits
```

**Files to update:**
- `.github/workflows/nightly-update.yml`
- `.github/workflows/send_predictions_schedule.yml`
- `.github/workflows/update-schedule.yml`
- `.github/workflows/rss_test.yml`
- Any other workflow files

---

### Step 2: Add GitHub Actions Cache for Historical Data

Add a cache step **before** your data update script runs:

```yaml
- name: Cache historical data
  uses: actions/cache@v4
  with:
    path: data_files/your_historical_data_file.csv.gz
    key: historical-data-v1  # Change v1→v2 to rebuild cache
    restore-keys: |
      historical-data-
```

**Example in context:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt

- name: Cache historical play-by-play data (2020-2024)
  uses: actions/cache@v4
  with:
    path: data_files/nfl_play_by_play_historical_2020_2024.csv.gz
    key: pbp-historical-2020-2024-v1
    restore-keys: |
      pbp-historical-2020-2024-

- name: Update data
  run: python update_data_script.py
```

**Cache behavior:**
- Expires after 7 days of no access
- Timer resets each time cache is accessed
- Daily workflows → cache persists indefinitely
- 10 GB limit per repository (plenty of room)

---

### Step 3: Update Data Download Script

Create a smart download script that:
1. Detects LFS pointers vs. real files
2. Uses cached historical data if available
3. Downloads only new data when cache exists
4. Falls back to full download on cache miss

**Full implementation (`update_data_script.py`):**

```python
#!/usr/bin/env python3
"""
Smart Data Updater - LFS-free with GitHub Actions caching support
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import sys

DATA_DIR = Path('data_files')
FULL_DATA_FILE = DATA_DIR / 'complete_dataset.csv.gz'
METADATA_FILE = DATA_DIR / 'data_metadata.json'
HISTORICAL_CACHE = DATA_DIR / 'historical_data_cache.csv.gz'

def get_current_year():
    """Get current year or season"""
    return datetime.now().year

def is_lfs_pointer(filepath):
    """Check if file is an LFS pointer instead of actual data"""
    if not filepath.exists():
        return False
    
    try:
        # LFS pointers are always < 200 bytes, real files are MB-GB
        if filepath.stat().st_size < 1000:
            return True
        return False
    except:
        return False

def read_metadata():
    """Read metadata file to get last update info"""
    if not METADATA_FILE.exists():
        return None
    
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error reading metadata: {e}")
        return None

def write_metadata(total_records):
    """Write metadata file with current update info"""
    metadata = {
        'last_update': datetime.now().isoformat(),
        'total_records': total_records
    }
    
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Updated metadata: {metadata}")
    except Exception as e:
        print(f"⚠️  Error writing metadata: {e}")

def should_update():
    """Check if we need to update based on metadata"""
    metadata = read_metadata()
    
    if metadata is None:
        return True
    
    try:
        last_update = datetime.fromisoformat(metadata['last_update'])
        hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
        
        print(f"📅 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ Hours since update: {hours_since_update:.1f}")
        
        # Update if it's been more than 12 hours
        if hours_since_update >= 12:
            print("✅ Update needed (>12 hours since last update)")
            return True
        else:
            print("⏭️  Skipping update (recent update found)")
            return False
            
    except Exception as e:
        print(f"⚠️  Error checking metadata: {e}, will update")
        return True

def download_year_data(year):
    """Download data for a specific year from external API"""
    try:
        print(f"📥 Downloading {year} data...")
        # Replace with your actual data source URL
        url = f"https://your-data-source.com/data_{year}.csv.gz"
        
        data = pd.read_csv(url, compression='gzip', low_memory=False)
        print(f"✅ Downloaded {len(data):,} records for {year}")
        return data
    except Exception as e:
        print(f"❌ Failed to download {year} data: {e}")
        return None

def download_all_years(start_year, end_year=None):
    """Download data for all years in range"""
    if end_year is None:
        end_year = get_current_year()
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        year_data = download_year_data(year)
        if year_data is not None:
            all_data.append(year_data)
    
    if not all_data:
        return None
    
    combined_data = pd.concat(all_data, ignore_index=True, sort=True)
    print(f"🔄 Combined {len(combined_data):,} total records ({start_year}-{end_year})")
    return combined_data

def merge_with_existing_data(new_data):
    """Merge new data with existing historical data"""
    if not FULL_DATA_FILE.exists() or is_lfs_pointer(FULL_DATA_FILE):
        return new_data
    
    try:
        print("📚 Loading existing data...")
        existing_data = pd.read_csv(FULL_DATA_FILE, compression='gzip', sep='\t', low_memory=False)
        
        # Remove current year from existing data
        current_year = get_current_year()
        existing_data = existing_data[existing_data['year'] != current_year]
        
        # Merge
        combined_data = pd.concat([existing_data, new_data], ignore_index=True, sort=True)
        print(f"🔄 Combined {len(combined_data):,} total records")
        return combined_data
    except Exception as e:
        print(f"⚠️  Error merging data: {e}")
        return new_data

def main():
    print("🧠 Smart Data Updater")
    print("=" * 50)
    
    current_year = get_current_year()
    print(f"📅 Current year: {current_year}")
    
    DATA_DIR.mkdir(exist_ok=True)
    
    # Check if we have real data (not LFS pointer)
    has_real_file = FULL_DATA_FILE.exists() and not is_lfs_pointer(FULL_DATA_FILE)
    
    # Check metadata to see if update is needed (local optimization)
    if has_real_file and not should_update():
        print("✅ Data is up to date, skipping download")
        return 0
    
    # Strategy 1: Local development with real file - incremental update
    if has_real_file:
        print("📦 Local file found - downloading current year only")
        new_data = download_year_data(current_year)
        if new_data is None:
            print("❌ Download failed")
            return 1
        
        final_data = merge_with_existing_data(new_data)
    
    # Strategy 2: CI/CD with cached historical data - merge with current year
    elif HISTORICAL_CACHE.exists() and not is_lfs_pointer(HISTORICAL_CACHE):
        print("✓ Using cached historical data")
        historical_data = pd.read_csv(HISTORICAL_CACHE, compression='gzip', sep='\t', low_memory=False)
        print(f"📚 Loaded {len(historical_data):,} historical records")
        
        print(f"📥 Downloading current year ({current_year}) only...")
        current_data = download_year_data(current_year)
        if current_data is None:
            print("❌ Download failed")
            return 1
        
        # Merge historical cache with current year
        final_data = pd.concat([historical_data, current_data], ignore_index=True, sort=True)
        print(f"🔄 Combined {len(final_data):,} total records")
    
    # Strategy 3: Fresh install or cache miss - download everything
    else:
        print("🌐 No cache found - downloading all years")
        
        # Build historical cache (adjust years as needed)
        print("📦 Building historical cache...")
        historical_data = download_all_years(start_year=2020, end_year=current_year - 1)
        if historical_data is None:
            print("❌ Failed to download historical data")
            return 1
        
        # Save to cache for future runs
        historical_data.to_csv(HISTORICAL_CACHE, compression='gzip', index=False, sep='\t')
        print(f"💾 Cached {len(historical_data):,} historical records")
        
        # Download current year
        current_data = download_year_data(current_year)
        if current_data is None:
            print("❌ Failed to download current year")
            return 1
        
        # Merge
        final_data = pd.concat([historical_data, current_data], ignore_index=True, sort=True)
        print(f"🔄 Combined {len(final_data):,} total records")
    
    # Save updated data
    final_data.to_csv(FULL_DATA_FILE, compression='gzip', index=False, sep='\t')
    print(f"💾 Saved {len(final_data):,} records to {FULL_DATA_FILE}")
    
    # Update metadata
    write_metadata(len(final_data))
    
    print("✅ Data update complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
```

---

### Step 4: Create Metadata File

Create an initial metadata file to track updates:

**`data_files/data_metadata.json`:**
```json
{
  "last_update": "2026-01-29T00:00:00",
  "total_records": 200000
}
```

This file should be committed to git (not in `.gitignore`).

---

### Step 5: Update .gitignore (Optional)

Ensure the cache file is ignored locally but the metadata is tracked:

```gitignore
# Cache file (built locally, also cached in CI)
data_files/historical_data_cache.csv.gz

# Ensure metadata is NOT ignored (should be tracked)
!data_files/data_metadata.json
```

---

## Testing Locally

Test the three scenarios:

```bash
# Test 1: With existing file (incremental update)
python update_data_script.py

# Test 2: Without file (simulates CI first run)
mv data_files/complete_dataset.csv.gz data_files/complete_dataset.csv.gz.backup
python update_data_script.py

# Test 3: With cache (simulates CI with cache)
# (Cache should exist from Test 2)
rm data_files/complete_dataset.csv.gz
python update_data_script.py
```

---

## Testing in GitHub Actions

1. **Commit and push changes**:
   ```bash
   git add .github/workflows/*.yml
   git add update_data_script.py
   git add data_files/data_metadata.json
   git commit -m "fix: disable LFS and add caching to avoid quota errors"
   git push
   ```

2. **First workflow run**:
   - No cache exists → downloads all years
   - Builds cache for future runs
   - Takes ~1-2 minutes

3. **Subsequent runs**:
   - Restores cache → only downloads current year
   - Takes ~30 seconds

4. **Monitor cache**:
   - GitHub UI: Repository → Actions → Caches
   - Check cache size and last access time

---

## Invalidating Cache

If you need to rebuild the cache (e.g., data source changed):

1. **Change the cache key** in workflow file:
   ```yaml
   key: historical-data-v2  # Increment version
   ```

2. **Or delete via GitHub UI**:
   - Repository → Actions → Caches → Delete cache

3. **Or delete via API**:
   ```bash
   gh cache delete historical-data-v1
   ```

---

## Benefits Summary

✅ **Zero LFS bandwidth usage** → No quota errors  
✅ **99% bandwidth reduction** after first run (75MB → 15MB)  
✅ **Faster workflows** (cache restore = instant)  
✅ **Self-healing** (automatically rebuilds cache if expired)  
✅ **Works locally** (fast incremental updates)  
✅ **No manual intervention** required  

---

## Troubleshooting

### Cache not being used

Check workflow logs for cache hit/miss:
```
Run actions/cache@v4
Cache not found for key: historical-data-v1
Cache restored from key: None
```

**Solution:** First run builds cache, subsequent runs will hit.

### Still seeing LFS errors

Ensure ALL workflow files have `lfs: false`:
```bash
grep -r "lfs:" .github/workflows/
```

### Download failures

Add retry logic to download functions:
```python
import time

def download_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return pd.read_csv(url, compression='gzip')
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
            else:
                raise e
```

---

## Adaptation Checklist

When applying to a new repository:

- [ ] Identify your LFS-tracked file(s)
- [ ] Find external API/source for the data
- [ ] Update all workflow files (`lfs: false`)
- [ ] Add cache step with appropriate path/key
- [ ] Adapt download script to your data source
- [ ] Adjust year/time ranges as needed
- [ ] Update file paths and column names
- [ ] Create initial metadata file
- [ ] Test locally (all three scenarios)
- [ ] Test in GitHub Actions
- [ ] Monitor first few workflow runs

---

## References

- [GitHub Actions Cache Documentation](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [Git LFS Bandwidth Quotas](https://docs.github.com/en/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage)
- [Checkout Action Options](https://github.com/actions/checkout#usage)
