#!/usr/bin/env python3
"""Compute per-course summary: total rows, number of distinct years, and latest year.

Outputs: data/processed/bha_2026_course_summary.csv
"""
from pathlib import Path
import sys
import pandas as pd

INPUT = Path("data/raw/2026_Fixture_List.xlsx")
OUT = Path("data/processed/bha_2026_course_summary.csv")

def find_course_column(df: pd.DataFrame):
    candidates = ["Course", "Venue", "Track", "Course Name", "Racecourse", "Location"]
    for c in df.columns:
        if any(c.strip().lower() == k.strip().lower() for k in candidates):
            return c
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

    # Attempt to find a date column to extract year
    date_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "day")):
            date_col = c
            break
    if date_col is None:
        # fallback: find first datetime-like column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break

    # If we have a date column, parse years; otherwise try to extract year from any column strings
    if date_col is not None:
        df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    else:
        # fallback: search for 4-digit year in any column text
        df["_year"] = None
        for c in df.columns:
            try:
                years = df[c].astype(str).str.extract(r"(20\d{2})")[0]
            except Exception:
                continue
            df["_year"] = df["_year"].fillna(years)
        df["_year"] = pd.to_numeric(df["_year"], errors="coerce")

    grp = df.groupby(df[course_col].astype(str).str.strip())
    rows = []
    for name, g in grp:
        yrs = g["_year"].dropna().astype(int)
        years_of_data = int(yrs.nunique()) if not yrs.empty else 0
        latest = int(yrs.max()) if not yrs.empty else None
        rows.append({"Course": name, "Rows": int(g.shape[0]), "YearsOfData": years_of_data, "LatestYear": latest})

    summary = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)

    print(f"Wrote {len(summary)} course summary rows to {OUT}\n")
    print(summary.sort_values(["Rows"], ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
