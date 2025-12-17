"""
Combine CSV files from the GB racing data into a single file.
"""

import pandas as pd
from pathlib import Path

# Source directory
SOURCE_DIR = Path("data/raw/gb")
OUTPUT_FILE = Path("data/processed/all_gb_races.csv")

def combine_csv_files():
    """Combine all CSV files in the flat directory into one file."""
    
    # Get all CSV files
    csv_files = sorted(SOURCE_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in {SOURCE_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        print(f"Reading {file.name}...")
        df = pd.read_csv(file)
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n[OK] Combined total rows: {len(combined_df)}")
    print(f"[OK] Columns: {list(combined_df.columns)}")
    
    # Save to output file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved to {OUTPUT_FILE}")
    
    # Show summary
    print(f"\nSummary:")
    print(f"  Total races: {len(combined_df)}")
    if 'course' in combined_df.columns:
        print(f"  Unique courses: {combined_df['course'].nunique()}")
    if 'date' in combined_df.columns:
        print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")


if __name__ == "__main__":
    combine_csv_files()
