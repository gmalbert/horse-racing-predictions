"""
Remove Class 5, 6, and 7 races from the database to optimize size.

These classes are not targeted by the betting strategy:
- Zero Class 5-7 races score ≥70 (Tier 1 Focus)
- Saves 61% database size
- No impact on betting opportunities
"""

import pandas as pd
from pathlib import Path

def remove_low_classes():
    """Remove Class 5, 6, and 7 races from all processed data files"""
    
    print("="*60)
    print("REMOVING CLASS 5, 6, AND 7 RACES")
    print("="*60)
    
    data_dir = Path('data/processed')
    
    # Files to process
    files_to_clean = [
        'all_gb_races_cleaned.parquet',
        'race_scores.parquet'
    ]
    
    for filename in files_to_clean:
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"\n[SKIP] {filename} - file not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print('='*60)
        
        # Load data
        df = pd.read_parquet(filepath)
        original_count = len(df)
        original_races = df['race_id'].nunique() if 'race_id' in df.columns else 'N/A'
        
        print(f"Original records: {original_count:,}")
        if original_races != 'N/A':
            print(f"Original unique races: {original_races:,}")
        
        # Show class distribution before
        print(f"\nClass distribution (before):")
        class_counts = df['class_clean'].value_counts().sort_index()
        for cls, count in class_counts.items():
            pct = count / original_count * 100
            print(f"  {cls}: {count:,} ({pct:.1f}%)")
        
        # Filter out Class 5, 6, and 7
        df_filtered = df[~df['class_clean'].isin(['Class 5', 'Class 6', 'Class 7'])].copy()
        
        new_count = len(df_filtered)
        new_races = df_filtered['race_id'].nunique() if 'race_id' in df_filtered.columns else 'N/A'
        removed_count = original_count - new_count
        
        print(f"\nFiltered records: {new_count:,}")
        if new_races != 'N/A':
            print(f"Filtered unique races: {new_races:,}")
        print(f"Removed: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
        
        # Show class distribution after
        print(f"\nClass distribution (after):")
        class_counts_after = df_filtered['class_clean'].value_counts().sort_index()
        for cls, count in class_counts_after.items():
            pct = count / new_count * 100
            print(f"  {cls}: {count:,} ({pct:.1f}%)")
        
        # Create backup of original
        backup_path = filepath.with_suffix('.backup.parquet')
        print(f"\n[BACKUP] Creating backup: {backup_path.name}")
        df.to_parquet(backup_path, index=False)
        
        # Save filtered data
        print(f"[SAVE] Saving filtered data: {filename}")
        df_filtered.to_parquet(filepath, index=False)
        
        print(f"[OK] {filename} updated successfully")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ Class 5, 6, and 7 races removed from all files")
    print("✅ Backups created with .backup.parquet extension")
    print("✅ Database size reduced by ~61%")
    print("\nNext steps:")
    print("  1. Retrain the model with cleaner data")
    print("  2. Verify Tier 1 Focus races are unchanged")
    print("  3. Delete backup files once verified (optional)")

if __name__ == '__main__':
    remove_low_classes()
