#!/usr/bin/env python3
"""
Append New Race Data to Historical Dataset

This script appends new racing data from CSV files to the existing historical Parquet dataset.
It handles:
- Deduplication based on race_id + horse_id
- Data type consistency
- Date range validation
- Automatic backup before appending

Usage:
  python scripts/append_new_race_data.py <csv_file>
  python scripts/append_new_race_data.py data/raw/new_races_2026.csv
  python scripts/append_new_race_data.py data/raw/new_races_2026.csv --dry-run
  
Options:
  --dry-run: Show what would be added without actually modifying the data
  --backup: Create a dated backup before appending (default: True)
  --no-backup: Skip backup creation
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# Paths
DATA_DIR = Path('data/processed')
HISTORICAL_FILE = DATA_DIR / 'all_gb_races.parquet'
BACKUP_DIR = DATA_DIR / 'backups'


def create_backup(file_path):
    """Create a dated backup of the historical data file"""
    if not file_path.exists():
        print(f"⚠️  No existing file to backup: {file_path}")
        return None
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = BACKUP_DIR / f"{file_path.stem}_backup_{timestamp}.parquet"
    
    print(f"📦 Creating backup: {backup_file.name}")
    shutil.copy2(file_path, backup_file)
    print(f"   ✓ Backup created: {backup_file}")
    return backup_file


def load_historical_data(file_path):
    """Load existing historical race data"""
    if not file_path.exists():
        print(f"⚠️  No existing historical data found at {file_path}")
        print(f"   Creating new dataset from scratch")
        return None
    
    print(f"📂 Loading historical data: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def load_new_data(csv_file):
    """Load new race data from CSV"""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    print(f"\n📥 Loading new data: {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def standardize_data_types(df, reference_df):
    """Ensure new data has same data types as historical data"""
    if reference_df is None:
        return df
    
    print(f"\n🔧 Standardizing data types...")
    changes = 0
    
    for col in df.columns:
        if col in reference_df.columns:
            ref_dtype = reference_df[col].dtype
            new_dtype = df[col].dtype
            
            if ref_dtype != new_dtype:
                try:
                    df[col] = df[col].astype(ref_dtype)
                    changes += 1
                    print(f"   ✓ {col}: {new_dtype} → {ref_dtype}")
                except Exception as e:
                    print(f"   ⚠️  Could not convert {col}: {e}")
    
    if changes == 0:
        print(f"   ✓ All data types match")
    else:
        print(f"   ✓ Converted {changes} columns")
    
    return df


def add_missing_columns(df, reference_df):
    """Add any columns present in historical data but missing in new data"""
    if reference_df is None:
        return df
    
    missing_cols = set(reference_df.columns) - set(df.columns)
    if missing_cols:
        print(f"\n➕ Adding {len(missing_cols)} missing columns:")
        for col in missing_cols:
            df[col] = np.nan
            print(f"   ✓ Added: {col} (filled with NaN)")
    
    # Reorder columns to match historical data
    df = df[reference_df.columns]
    return df


def deduplicate_data(historical_df, new_df):
    """Remove records from new_df that already exist in historical_df"""
    if historical_df is None:
        print(f"\n✓ No existing data to deduplicate against")
        return new_df, 0
    
    print(f"\n🔍 Checking for duplicates...")
    
    # Create composite key for deduplication
    if 'race_id' in new_df.columns and 'horse_id' in new_df.columns:
        historical_df['_key'] = historical_df['race_id'].astype(str) + '_' + historical_df['horse_id'].astype(str)
        new_df['_key'] = new_df['race_id'].astype(str) + '_' + new_df['horse_id'].astype(str)
        
        before_count = len(new_df)
        new_df = new_df[~new_df['_key'].isin(historical_df['_key'])]
        duplicates = before_count - len(new_df)
        
        # Drop temporary key column
        new_df = new_df.drop(columns=['_key'])
        
        if duplicates > 0:
            print(f"   ⚠️  Removed {duplicates:,} duplicate records (already in historical data)")
        else:
            print(f"   ✓ No duplicates found")
        
        return new_df, duplicates
    else:
        print(f"   ⚠️  Cannot deduplicate: missing race_id or horse_id columns")
        return new_df, 0


def append_data(historical_df, new_df):
    """Append new data to historical data"""
    if historical_df is None:
        print(f"\n✓ Using new data as initial dataset")
        return new_df
    
    print(f"\n📊 Appending data...")
    combined_df = pd.concat([historical_df, new_df], ignore_index=True)
    print(f"   ✓ Combined: {len(combined_df):,} total rows")
    
    # Sort by date
    combined_df = combined_df.sort_values('date', ignore_index=True)
    print(f"   ✓ Sorted by date: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df


def save_data(df, file_path):
    """Save combined data to Parquet"""
    print(f"\n💾 Saving to: {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False, compression='snappy')
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved {len(df):,} rows ({file_size_mb:.2f} MB)")


def print_summary(historical_df, new_df, duplicates, combined_df):
    """Print summary of the append operation"""
    print(f"\n" + "="*60)
    print(f"📈 SUMMARY")
    print(f"="*60)
    
    if historical_df is not None:
        print(f"Historical data:  {len(historical_df):,} rows")
    else:
        print(f"Historical data:  0 rows (new dataset)")
    
    print(f"New data loaded:  {len(new_df) + duplicates:,} rows")
    
    if duplicates > 0:
        print(f"Duplicates removed: {duplicates:,} rows")
    
    new_added = len(new_df)
    print(f"New data added:   {new_added:,} rows")
    
    if combined_df is not None:
        print(f"Total after append: {len(combined_df):,} rows")
        print(f"Date range:       {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    print(f"="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Append new race data to historical dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('csv_file', help='Path to CSV file with new race data')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be added without modifying data')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--output', help='Output file path (default: data/processed/all_gb_races.parquet)')
    
    args = parser.parse_args()
    
    # Determine output file
    output_file = Path(args.output) if args.output else HISTORICAL_FILE
    
    print(f"{'='*60}")
    print(f"🏇 APPEND NEW RACE DATA")
    print(f"{'='*60}")
    
    # Load existing historical data
    historical_df = load_historical_data(output_file)
    
    # Load new data from CSV
    new_df = load_new_data(args.csv_file)
    
    # Standardize data types to match historical data
    new_df = standardize_data_types(new_df, historical_df)
    
    # Add any missing columns
    new_df = add_missing_columns(new_df, historical_df)
    
    # Remove duplicates
    new_df, duplicates = deduplicate_data(historical_df, new_df)
    
    if len(new_df) == 0:
        print(f"\n⚠️  No new data to append (all records were duplicates)")
        return
    
    # Append data
    combined_df = append_data(historical_df, new_df)
    
    # Print summary
    print_summary(historical_df, new_df, duplicates, combined_df)
    
    # Save or dry-run
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No files modified")
        print(f"   Would have saved to: {output_file}")
    else:
        # Create backup unless disabled
        if not args.no_backup and historical_df is not None:
            create_backup(output_file)
        
        # Save combined data
        save_data(combined_df, output_file)
        
        print(f"\n✅ SUCCESS - Data appended successfully!")


if __name__ == '__main__':
    main()
