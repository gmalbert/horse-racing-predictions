#!/usr/bin/env python3
"""Extract 2026 BHA fixture rows for four target courses and save CSV.

Writes: data/processed/bha_2026_ascot_newmarket_doncaster_york.csv
"""
from pathlib import Path
import sys
import pandas as pd

INPUT = Path("data/raw/2026_Fixture_List.xlsx")
OUT = Path("data/processed/bha_2026_ascot_newmarket_doncaster_york.csv")
TARGETS = ["Ascot", "Newmarket", "Doncaster", "York"]

def find_course_column(df: pd.DataFrame):
    candidates = [
        "Course",
        "Venue",
        "Track",
        "Course Name",
        "Racecourse",
        "Location",
        "Racecourse Name",
    ]
    for c in df.columns:
        for k in candidates:
            if c.strip().lower() == k.strip().lower():
                return c

    # Fallback: find a column that contains at least one target name
    for c in df.columns:
        try:
            sample = df[c].dropna().astype(str).head(200).str.lower()
        except Exception:
            continue
        if sample.apply(lambda x: any(t.lower() in x for t in TARGETS)).any():
            return c
    return None


def main():
    if not INPUT.exists():
        print(f"Input not found: {INPUT}")
        sys.exit(2)

    df = pd.read_excel(INPUT, sheet_name=0)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    course_col = find_course_column(df)
    if course_col is None:
        print("Could not detect a course/track column. Columns:\n", df.columns.tolist())
        sys.exit(3)

    pattern = r"\b(?:" + "|".join([p.replace("(", "\\(").replace(")", "\\)") for p in TARGETS]) + r")\b"
    mask = df[course_col].astype(str).str.contains(pattern, case=False, na=False, regex=True)
    out_df = df[mask].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)

    print(f"Wrote {len(out_df)} rows to {OUT}")


if __name__ == "__main__":
    main()
