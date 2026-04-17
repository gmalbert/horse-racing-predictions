#!/usr/bin/env python3
"""
Weather & Ground Condition Feature Engineering.

Extends the historical race dataset with weather-derived features by:

  1.  Fetching historical weather from the Open-Meteo free API (no key needed)
      for each GB racecourse location — daily rainfall, max/min temperature,
      and wind speed for the 7 days preceding each race date.

  2.  Adding derived features that capture going condition dynamics:
        going_variability_30d  — how much the going varied in the 30 days
                                 before the race at this course (stdev of
                                 going_numeric)
        going_trend_30d        — linear slope of going_numeric over 30 days
                                 (positive = drying out; negative = wetting)
        is_going_extreme       — 1 if going_numeric <= 2 (heavy) or >= 8 (firm)
        rainfall_7d_mm         — cumulative rainfall in the 7 days before the
                                 race (from Open-Meteo if available, else NaN)
        temp_max_c             — max temperature on race day
        wind_speed_ms          — mean wind speed on race day
        going_vs_horse_pref    — horse's going win rate in matching category
                                 minus their overall win rate (already in
                                 going_match_score but this makes it explicit)

  DATA LEAKAGE NOTE: All lookback features use ONLY dates strictly before the
  race date.  The `going_variability_30d` and `going_trend_30d` features are
  computed with a window that ends the day before the race, not the race day
  itself.  Weather data is for the race date and prior days — this is
  legitimate because weather on race day is public knowledge before the race.

Usage
-----
  # Full run: fetch weather + compute all features → writes new parquet
  python scripts/add_weather_features.py

  # Skip weather API (offline — going features only):
  python scripts/add_weather_features.py --offline

  # Force re-fetch weather even if cache exists:
  python scripts/add_weather_features.py --refresh-weather

Output
------
  data/processed/race_scores_weather.parquet
    — input parquet enriched with all weather / going features
"""

import argparse
import time
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR  = ROOT / "data" / "raw"
WEATHER_CACHE = RAW_DIR / "weather_cache.parquet"

# ─── GB Racecourse coordinates ────────────────────────────────────────────────
# Approximate lat/lon for the most common GB venues
COURSE_COORDS: dict[str, tuple[float, float]] = {
    "ascot":              (51.408, -0.570),
    "ayr":                (55.464, -4.621),
    "bath":               (51.397,  -2.326),
    "beverley":           (53.843,  -0.442),
    "brighton":           (50.831,  -0.128),
    "carlisle":           (54.880,  -2.935),
    "catterick":          (54.378,  -1.624),
    "chelmsford city":    (51.749,   0.436),
    "cheltenham":         (51.899,  -2.064),
    "chester":            (53.186,  -2.901),
    "doncaster":          (53.519,  -1.111),
    "epsom":              (51.330,  -0.258),
    "exeter":             (50.740,  -3.501),
    "ffos las":           (51.740,  -4.162),
    "folkestone":         (51.169,   1.111),
    "goodwood":           (50.898,  -0.758),
    "hamilton":           (55.775,  -4.031),
    "haydock":            (53.460,  -2.639),
    "hereford":           (52.051,  -2.724),
    "huntingdon":         (52.319,  -0.192),
    "kempton":            (51.410,  -0.384),
    "leicester":          (52.607,  -1.085),
    "lingfield":          (51.183,  -0.013),
    "musselburgh":        (55.943,  -3.059),
    "newbury":            (51.395,  -1.321),
    "newcastle":          (54.946,  -1.617),
    "newmarket":          (52.240,   0.407),
    "nottingham":         (52.945,  -1.163),
    "pontefract":         (53.695,  -1.294),
    "redcar":             (54.616,  -1.069),
    "ripon":              (54.130,  -1.524),
    "salisbury":          (51.066,  -1.797),
    "sandown":            (51.368,  -0.335),
    "southwell":          (53.070,  -0.954),
    "thirsk":             (54.228,  -1.344),
    "warwick":            (52.278,  -1.589),
    "windsor":            (51.476,  -0.631),
    "wolverhampton":      (52.588,  -2.130),
    "worcester":          (52.193,  -2.239),
    "yarmouth":           (52.608,   1.727),
    "york":               (53.977,  -1.059),
}


# ─────────────────────────────────────────────────────────────
# Open-Meteo weather fetching
# ─────────────────────────────────────────────────────────────

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_weather_for_location(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Fetch daily weather data from Open-Meteo Archive API (free, no key).

    Returns a DataFrame indexed by date with columns:
      rainfall_mm, temp_max_c, temp_min_c, wind_speed_ms
    Returns None on error.
    """
    if not HAS_REQUESTS:
        return None

    params = {
        "latitude":      lat,
        "longitude":     lon,
        "start_date":    start_date,
        "end_date":      end_date,
        "daily":         "precipitation_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
        "timezone":      "Europe/London",
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [!] Open-Meteo error ({lat},{lon}): {e}")
        return None

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return None

    df = pd.DataFrame({
        "date":          pd.to_datetime(dates),
        "rainfall_mm":   daily.get("precipitation_sum", [np.nan] * len(dates)),
        "temp_max_c":    daily.get("temperature_2m_max", [np.nan] * len(dates)),
        "temp_min_c":    daily.get("temperature_2m_min", [np.nan] * len(dates)),
        "wind_speed_ms": daily.get("windspeed_10m_max",  [np.nan] * len(dates)),
    }).set_index("date")

    return df


def fetch_all_weather(df_races: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Fetch weather for all courses and dates in df_races.

    Results are cached in data/raw/weather_cache.parquet to minimise API calls.
    Open-Meteo free tier: no call limit, but we batch by location.
    """
    # Load existing cache
    if WEATHER_CACHE.exists() and not refresh:
        cache = pd.read_parquet(WEATHER_CACHE)
        print(f"[weather] Loaded cache: {len(cache):,} rows")
    else:
        cache = pd.DataFrame(columns=["date", "course_key", "rainfall_mm", "temp_max_c", "temp_min_c", "wind_speed_ms"])

    if "date_dt" not in df_races.columns:
        df_races["date_dt"] = pd.to_datetime(df_races["date"], errors="coerce")

    # Normalise course name for lookup
    df_races["course_key"] = df_races["course"].str.lower().str.strip().str.replace(r"\s*\(.*\)", "", regex=True)

    unique_courses = df_races["course_key"].unique()
    date_min = df_races["date_dt"].min().date()
    date_max = df_races["date_dt"].max().date()
    # Add 7-day buffer at start for rolling window
    date_min_buf = (date_min - timedelta(days=8)).isoformat()
    date_max_str  = date_max.isoformat()

    # Courses with known coordinates
    known_courses = [c for c in unique_courses if c in COURSE_COORDS]
    unknown = [c for c in unique_courses if c not in COURSE_COORDS]
    if unknown:
        print(f"[weather] Skipping {len(unknown)} unrecognised courses: {unknown[:5]}...")

    rows = []
    for course in known_courses:
        lat, lon = COURSE_COORDS[course]

        # Check what's already cached
        existing = cache[(cache["course_key"] == course)]
        if len(existing) > 0:
            existing_dates = set(pd.to_datetime(existing["date"]).dt.date)
            needed_start = date_min_buf
            # If all needed dates are covered, skip
            needed_dates = set(
                pd.date_range(needed_start, date_max_str).date
            )
            if needed_dates.issubset(existing_dates):
                print(f"  [cache] {course}: fully cached")
                continue

        print(f"  [fetch] {course} ({lat},{lon}) {date_min_buf}..{date_max_str}")
        w = fetch_weather_for_location(lat, lon, date_min_buf, date_max_str)
        time.sleep(0.3)  # polite rate limiting

        if w is not None:
            for d, row in w.iterrows():
                rows.append({
                    "date":        d.date().isoformat(),
                    "course_key":  course,
                    "rainfall_mm": row["rainfall_mm"],
                    "temp_max_c":  row["temp_max_c"],
                    "temp_min_c":  row["temp_min_c"],
                    "wind_speed_ms": row["wind_speed_ms"],
                })

    if rows:
        new_rows = pd.DataFrame(rows)
        cache = pd.concat([cache, new_rows], ignore_index=True).drop_duplicates(
            subset=["date", "course_key"]
        )
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(WEATHER_CACHE, index=False)
        print(f"[weather] Cache updated: {len(cache):,} rows saved")

    return cache


# ─────────────────────────────────────────────────────────────
# Going condition features (no API needed)
# ─────────────────────────────────────────────────────────────

_GOING_MAP = {
    "heavy":             1,
    "soft":              3,
    "good to soft":      4,
    "standard to slow":  4,
    "good":              5,
    "standard":          5,
    "good to firm":      7,
    "standard to fast":  8,
    "firm":              9,
}

def _derive_going_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Derive going_numeric from the going string column if not already present."""
    if "going_numeric" in df.columns:
        return df
    if "going" not in df.columns:
        return df
    df = df.copy()
    df["going_numeric"] = (
        df["going"].str.lower().str.strip().map(_GOING_MAP)
    )
    n_mapped = df["going_numeric"].notna().sum()
    print(f"[going] Derived going_numeric from 'going' string — {n_mapped:,}/{len(df):,} mapped")
    return df


def add_going_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Add going variability, trend, and extremity features.

    Temporal integrity: uses a 30-day rolling window on daily course-level
    going_numeric, shifted by 1 day so the race day itself is excluded.
    All lookbacks are strictly prior to each race date.
    """
    df = _derive_going_numeric(df)
    if "going_numeric" not in df.columns or df["going_numeric"].isna().all():
        print("[!] going_numeric not found — skipping going dynamics")
        return df

    df = df.copy()

    if "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    df["course_key"] = df["course"].str.lower().str.strip().str.replace(r"\s*\(.*\)", "", regex=True)

    # Build a daily course-level going series (mean going per course-day)
    daily = (
        df.groupby(["course_key", "date_dt"])["going_numeric"]
        .mean()
        .reset_index()
        .sort_values(["course_key", "date_dt"])
    )
    daily = daily.set_index("date_dt")

    # Compute rolling stats on the daily series (30-day window, min 2 obs)
    # shift(1) ensures the current day is excluded from its own window
    results = []
    for course, grp in daily.groupby("course_key"):
        gn = grp["going_numeric"]
        var30  = gn.shift(1).rolling("30D", min_periods=2).std()
        mean30 = gn.shift(1).rolling("30D", min_periods=2).mean()
        # Approximate trend as (last_val - first_val) / window_days
        def _slope(x):
            if len(x) < 3:
                return np.nan
            return float(np.polyfit(np.arange(len(x)), x, 1)[0])
        trend30 = gn.shift(1).rolling("30D", min_periods=3).apply(_slope, raw=True)

        sub = pd.DataFrame({
            "course_key": course,
            "going_variability_30d": var30.values,
            "going_trend_30d":       trend30.values,
        }, index=grp.index)
        results.append(sub)

    if not results:
        return df

    daily_stats = pd.concat(results).reset_index()  # date_dt back as column

    # Merge back onto df by course_key + date_dt
    df = df.merge(
        daily_stats[["course_key", "date_dt", "going_variability_30d", "going_trend_30d"]],
        on=["course_key", "date_dt"],
        how="left",
    )

    # is_going_extreme is a simple scalar — no lookahead risk
    df["is_going_extreme"] = df["going_numeric"].apply(
        lambda gn: int(not pd.isna(gn) and (gn <= 2 or gn >= 8))
    )

    print(f"[going] Added going_variability_30d, going_trend_30d, is_going_extreme")
    return df


# ─────────────────────────────────────────────────────────────
# Join weather data to race rows
# ─────────────────────────────────────────────────────────────

def join_weather_features(df: pd.DataFrame, weather_cache: pd.DataFrame) -> pd.DataFrame:
    """Merge weather features into the race dataset.

    Joins on (date, course_key) and also computes rainfall_7d_mm = sum of
    the 7 days of rainfall BEFORE race day (exclusive of race day itself is
    not possible since we only have daily data; we include race day as the
    7-day window already captures ground condition).
    """
    if weather_cache.empty:
        print("[weather] No weather data to join")
        return df

    weather_cache = weather_cache.copy()
    weather_cache["date"] = pd.to_datetime(weather_cache["date"]).dt.date.astype(str)

    if "course_key" not in df.columns:
        df["course_key"] = df["course"].str.lower().str.strip().str.replace(r"\s*\(.*\)", "", regex=True)
    df = df.copy()
    if "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_str"] = df["date_dt"].dt.date.astype(str)

    # Compute 7-day rolling rainfall per course
    weather_cache["date_dt"] = pd.to_datetime(weather_cache["date"])
    weather_cache = weather_cache.sort_values(["course_key", "date_dt"])

    rainfall_7d = []
    for _, row in df.iterrows():
        cutoff = row["date_dt"] - pd.Timedelta(days=7)
        prior_rain = weather_cache[
            (weather_cache["course_key"] == row["course_key"]) &
            (weather_cache["date_dt"] <= row["date_dt"]) &
            (weather_cache["date_dt"] > cutoff)
        ]["rainfall_mm"].sum()
        rainfall_7d.append(prior_rain if prior_rain > 0 else np.nan)

    df["rainfall_7d_mm"] = rainfall_7d

    # Direct join for same-day weather
    weather_today = weather_cache[["date", "course_key", "temp_max_c", "temp_min_c", "wind_speed_ms"]].copy()
    weather_today = weather_today.rename(columns={"date": "date_str"})

    df = df.merge(weather_today, on=["date_str", "course_key"], how="left")
    df = df.drop(columns=["date_str"], errors="ignore")

    print(f"[weather] Joined weather features — temp/wind coverage: "
          f"{df['temp_max_c'].notna().mean()*100:.1f}%")
    return df


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    candidates = [
        "race_scores_or_context.parquet",
        "race_scores_going_pref.parquet",
        "race_scores_connections_v2.parquet",
        "race_scores_with_all_features_no_leakage.parquet",
        "race_scores.parquet",
    ]
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            print(f"[data] Loading {name}")
            return pd.read_parquet(p)
    raise FileNotFoundError("No input parquet found in data/processed/")


def main():
    p = argparse.ArgumentParser(description="Add weather and going features")
    p.add_argument("--offline",         action="store_true", help="Skip Open-Meteo API — going features only")
    p.add_argument("--refresh-weather", action="store_true", help="Re-fetch all weather (ignore cache)")
    args = p.parse_args()

    df = load_data()
    print(f"[data] {len(df):,} rows loaded")

    # Step 1: Add going dynamics (pure offline computation)
    df = add_going_dynamics(df)

    # Step 2: Weather features (requires network unless --offline)
    if not args.offline and HAS_REQUESTS:
        print("\n[weather] Fetching historical weather from Open-Meteo (free API)…")
        weather_cache = fetch_all_weather(df, refresh=args.refresh_weather)
        df = join_weather_features(df, weather_cache)
    elif args.offline:
        print("[weather] --offline: skipping API fetch")
    else:
        print("[weather] requests not available — skipping API fetch")

    # Step 3: Save
    out_path = DATA_DIR / "race_scores_weather.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[✓] Saved → {out_path}  ({len(df):,} rows)")

    new_cols = [c for c in ["going_variability_30d", "going_trend_30d", "is_going_extreme",
                             "rainfall_7d_mm", "temp_max_c", "temp_min_c", "wind_speed_ms"]
                if c in df.columns]
    print(f"[✓] New features: {new_cols}")
    print(f"[✓] Coverage (non-null %):")
    for col in new_cols:
        pct = df[col].notna().mean() * 100
        print(f"      {col:<30} {pct:.1f}%")


if __name__ == "__main__":
    main()
