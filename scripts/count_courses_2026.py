#!/usr/bin/env python3
"""Count occurrences of each course in the 2026 BHA fixture Excel and save CSV.

Outputs: data/processed/bha_2026_course_counts.csv
Prints top 30 course counts to stdout.
"""
from pathlib import Path
import sys
import pandas as pd

INPUT = Path("data/raw/2026_Fixture_List.xlsx")
OUT = Path("data/processed/bha_2026_course_counts.csv")
TARGET_DISPLAY = 30

def find_course_column(df: pd.DataFrame):
    candidates = ["Course", "Venue", "Track", "Course Name", "Racecourse", "Location"]
    for c in df.columns:
        if any(c.strip().lower() == k.strip().lower() for k in candidates):
            return c
    # fallback: choose column with most string-like values
    for c in df.columns:
        try:
            s = df[c].dropna().astype(str).head(200)
        except Exception:
            continue
        if s.str.contains(r"[A-Za-z]", regex=True).any():
            return c
    return None


def main():
    if not INPUT.exists():
        print(f"Input not found: {INPUT}")
        sys.exit(2)

    df = pd.read_excel(INPUT, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    course_col = find_course_column(df)
    if course_col is None:
        print("Could not detect a course column. Columns:\n", df.columns.tolist())
        sys.exit(3)

    counts = df[course_col].astype(str).str.strip().value_counts(dropna=True)
    out = counts.reset_index()
    out.columns = ["Course","Count"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Wrote {len(out)} unique courses to {OUT}\n")
    print(out.head(TARGET_DISPLAY).to_string(index=False))


if __name__ == "__main__":
    main()
