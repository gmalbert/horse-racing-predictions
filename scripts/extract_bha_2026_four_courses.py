#!/usr/bin/env python3
"""Extract 2026 BHA fixture rows (all courses, Class 1-4 only) and save CSV.

Writes: data/processed/bha_2026_all_courses_class1-4.csv
"""
from pathlib import Path
import sys
import pandas as pd

INPUT = Path("data/raw/2026_Fixture_List.xlsx")
OUT = Path("data/processed/bha_2026_all_courses_class1-4.csv")

def find_class_column(df: pd.DataFrame):
    """Find the column containing race class information"""
    candidates = ["Class", "Race Class", "RaceClass", "Grade"]
    for c in df.columns:
        for k in candidates:
            if c.strip().lower() == k.strip().lower():
                return c
    return None


def main():
    if not INPUT.exists():
        print(f"Input not found: {INPUT}")
        sys.exit(2)

    df = pd.read_excel(INPUT, sheet_name=0)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Find class column
    class_col = find_class_column(df)
    
    if class_col is None:
        print("Warning: Could not detect a class column. Including all classes.")
        print("Available columns:", df.columns.tolist())
        out_df = df.copy()
    else:
        # Filter to Class 1-4 only (exclude Class 5, 6, 7)
        print(f"Found class column: {class_col}")
        print(f"Original rows: {len(df)}")
        
        # Convert to string and extract class number
        df['class_str'] = df[class_col].astype(str).str.strip()
        
        # Filter out Class 5, 6, 7
        excluded_pattern = r'\b(?:Class\s*[567]|[567])\b'
        mask = ~df['class_str'].str.contains(excluded_pattern, case=False, na=False, regex=True)
        out_df = df[mask].copy()
        
        # Drop temporary column
        out_df = out_df.drop('class_str', axis=1)
        
        print(f"Filtered rows (Class 1-4 only): {len(out_df)}")
        print(f"Removed: {len(df) - len(out_df)} Class 5-7 races")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)

    print(f"\nWrote {len(out_df)} rows to {OUT}")


if __name__ == "__main__":
    main()
