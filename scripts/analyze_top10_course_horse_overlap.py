#!/usr/bin/env python3
"""Analyze whether horses run on multiple of the top-10 courses.

Outputs:
 - data/processed/top10_horse_course_overlap.csv  (per-horse counts)
 - data/processed/top10_course_horse_stats.csv    (per-course summary)

Prints summary statistics to stdout.
"""
from pathlib import Path
import pandas as pd
import sys

COURSE_COUNTS = Path("data/processed/bha_2026_course_counts.csv")
RACES = Path("data/processed/all_gb_races.csv")
OUT_HORSE = Path("data/processed/top10_horse_course_overlap.csv")
OUT_COURSE = Path("data/processed/top10_course_horse_stats.csv")

def main():
    if not COURSE_COUNTS.exists():
        print(f"Missing course counts: {COURSE_COUNTS}")
        sys.exit(2)
    if not RACES.exists():
        print(f"Missing races CSV: {RACES}")
        sys.exit(2)

    counts = pd.read_csv(COURSE_COUNTS)
    top10 = counts.head(10)["Course"].astype(str).str.strip().tolist()
    print("Top-10 courses:", top10)

    df = pd.read_csv(RACES, usecols=["course", "horse", "date"], parse_dates=["date"], low_memory=False)
    df["course"] = df["course"].astype(str).str.strip()
    df["horse"] = df["horse"].astype(str).str.strip()

    df_top = df[df["course"].isin(top10)].copy()

    # For each horse, how many distinct top10 courses did it run at?
    horse_course_counts = (
        df_top.groupby("horse")["course"].nunique().reset_index().rename(columns={"course": "num_top10_courses"})
    )

    # Also list which courses per horse (comma-separated)
    horse_courses = (
        df_top.groupby("horse")["course"].unique().apply(lambda arr: ", ".join(sorted(arr))).reset_index().rename(columns={0: "courses"})
    )

    # Merge
    merged = horse_course_counts.merge(horse_courses, on="horse")
    OUT_HORSE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_HORSE, index=False)

    # Summary stats
    total_horses = merged.shape[0]
    multi_course = (merged["num_top10_courses"] > 1).sum()
    pct_multi = multi_course / total_horses * 100 if total_horses>0 else 0

    print(f"Horses that ran at least once on the top-10 courses: {total_horses}")
    print(f"Horses that ran on multiple of the top-10 courses: {multi_course} ({pct_multi:.1f}%)")

    # Per-course: of horses that ran at course X, what fraction also ran at >1 top10 course?
    course_stats = []
    for c in top10:
        horses_at_c = set(df_top.loc[df_top["course"]==c, "horse"]) 
        if not horses_at_c:
            course_stats.append({"Course": c, "Horses": 0, "MultiCourseHorses": 0, "PctMulti": 0.0})
            continue
        horses_at_c_df = merged[merged["horse"].isin(horses_at_c)]
        hc = horses_at_c_df.shape[0]
        hm = (horses_at_c_df["num_top10_courses"]>1).sum()
        pct = hm/hc*100 if hc>0 else 0
        course_stats.append({"Course": c, "Horses": hc, "MultiCourseHorses": int(hm), "PctMulti": round(pct,1)})

    course_stats_df = pd.DataFrame(course_stats).sort_values("Horses", ascending=False)
    course_stats_df.to_csv(OUT_COURSE, index=False)

    print("\nPer-course multi-course percentages:")
    print(course_stats_df.to_string(index=False))

    print(f"\nWrote per-horse overlap to {OUT_HORSE} and per-course stats to {OUT_COURSE}")


if __name__ == "__main__":
    main()
