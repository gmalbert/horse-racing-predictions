"""
Export precomputed JSON artifacts for the React frontend.

The React site (Cloudflare Pages) serves ONLY precomputed values. This script
reads the same data files as the Streamlit app and writes static JSON to
``web/frontend/public/data/``. Every value the Streamlit app computed on the
fly (ranks, exacta/trifecta probabilities, value-bet edges, US race scores,
trainer/jockey form windows, calibration bins, segment ROI, ...) is computed
here at export time.

Run after the prediction/odds pipelines:

    python scripts/export_web_data.py                 # export everything
    python scripts/export_web_data.py --history 7     # days of GB/US history
    python scripts/export_web_data.py --skip-stats    # skip trainer/jockey windows
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "web" / "frontend" / "public" / "data"

STATS_WINDOWS = (14, 30, 90, 180, 365)

# -- US race scoring constants (ported from pages/us_racing.py) ---------------
TRACK_TIER = {
    "Churchill Downs": 20, "Saratoga": 20, "Belmont Park": 20,
    "Keeneland": 20, "Santa Anita Park": 20, "Del Mar": 20,
    "Gulfstream Park": 12, "Aqueduct": 12, "Oaklawn Park": 12,
    "Fair Grounds": 12, "Pimlico": 12, "Laurel Park": 12,
    "Monmouth Park": 10, "Parx Racing": 10, "Tampa Bay Downs": 8,
    "Turfway Park": 8, "Horseshoe Indianapolis": 6, "Prairie Meadows": 6,
}
GRADE_BONUS = {"G1": 40, "G2": 25, "G3": 15}
PURSE_BONUS = [(1_000_000, 30), (500_000, 20), (200_000, 14), (100_000, 8), (50_000, 4)]
HARNESS_FRAGMENTS = frozenset([
    "yonkers", "harrington", "plainridge", "scioto", "meadowlands",
    "pocono", "tioga", "batavia", "northfield", "hawthorne raceway",
    "pompano", "dover", "cal expo", "hoosier park", "miami valley", "lebanon",
])


# -- helpers ------------------------------------------------------------------

def _clean(obj):
    """Recursively convert numpy/pandas types and NaN to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if obj is pd.NaT or obj is pd.NA:
        return None
    return obj


def write_json(name: str, payload) -> Path:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean(payload), separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    size_kb = path.stat().st_size / 1024
    print(f"  OK {name:<42} {size_kb:>8.1f} KB")
    return path


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 2:
            return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ! could not read {path.name}: {exc}")
    return pd.DataFrame()


def distance_band(dist_f):
    if pd.isna(dist_f):
        return None
    try:
        d = float(dist_f)
    except (TypeError, ValueError):
        return None
    if d < 6:
        return "Sprint (<6f)"
    if d <= 8:
        return "Mile (6-8f)"
    if d <= 12:
        return "Middle (8-12f)"
    return "Staying (>12f)"


# -- race-level enrichment (shared by GB and US) ------------------------------

def enrich_race(grp: pd.DataFrame) -> dict:
    """Precompute everything the race-detail view needs for one race."""
    grp = grp.copy()
    grp["win_rank"] = grp["win_probability"].rank(ascending=False, method="min").astype(int)
    grp["place_rank"] = grp["place_probability"].rank(ascending=False, method="min").astype(int)
    grp["show_rank"] = grp["show_probability"].rank(ascending=False, method="min").astype(int)
    grp["top_2_prob"] = grp["win_probability"] + grp["place_probability"]
    grp["top_3_prob"] = (
        grp["win_probability"] + grp["place_probability"] + grp["show_probability"]
    )
    grp = grp.sort_values("win_probability", ascending=False)

    top3 = grp.head(3)
    exacta = trifecta = fair_trifecta_odds = None
    if len(top3) >= 3:
        p1, p2, p3 = top3["win_probability"].iloc[:3]
        if p1 < 0.99:
            p_second = p2 / (1 - p1)
            exacta = p1 * p_second
            if (p1 + p2) < 0.99:
                trifecta = p1 * p_second * (p3 / (1 - p1 - p2))
                if trifecta and trifecta > 0:
                    fair_trifecta_odds = (1 / trifecta) - 1

    fav = grp.iloc[0]
    fav_win = float(fav["win_probability"])
    fav_top2 = min(float(fav["place_probability"]), 0.95)  # already cumulative
    fav_top3 = min(float(fav["show_probability"]), 0.98)

    favorites = {
        "win": {"horse": fav["horse"], "prob": fav_win},
        "place": {
            "horse": grp.loc[grp["place_probability"].idxmax(), "horse"],
            "prob": float(grp["place_probability"].max()),
        },
        "show": {
            "horse": grp.loc[grp["show_probability"].idxmax(), "horse"],
            "prob": float(grp["show_probability"].max()),
        },
        "favorite_cumulative": {
            "horse": fav["horse"],
            "win": fav_win,
            "top_2": fav_top2,
            "top_3": fav_top3,
            "top_2_increment": fav_top2 - fav_win,
            "top_3_increment": fav_top3 - fav_top2,
        },
    }

    meta_cols = [
        "course", "race_time", "race_time_gmt", "race_name", "race_class",
        "distance_f", "distance_str", "going", "surface", "race_type", "date",
    ]
    meta = {c: (grp[c].iloc[0] if c in grp.columns else None) for c in meta_cols}
    meta["runners"] = len(grp)

    horse_cols = [
        "horse", "jockey", "trainer", "win_probability", "win_odds_fractional",
        "place_probability", "place_odds_fractional", "show_probability",
        "show_odds_fractional", "top_2_prob", "top_3_prob",
        "win_rank", "place_rank", "show_rank",
        "age", "weight_lbs", "ofr", "form", "draw", "ml_odds", "sire", "dam",
        "market_odds", "implied_prob", "edge", "ev_per_unit", "best_bookmaker",
        "odds_trend", "is_value_bet", "is_strong_value",
        "betfair_back_odds", "betfair_value_edge",
        "finish_position", "sp_decimal", "actual_result",
    ]
    horses = grp[[c for c in horse_cols if c in grp.columns]].to_dict("records")

    return {
        **meta,
        "horses": horses,
        "favorites": favorites,
        "exacta_prob": exacta,
        "trifecta_prob": trifecta,
        "fair_trifecta_odds": fair_trifecta_odds,
        "predicted_123": top3["horse"].tolist() if len(top3) >= 3 else [],
    }


def race_key(df: pd.DataFrame) -> pd.Series:
    race_name = df["race_name"] if "race_name" in df.columns else pd.Series([""] * len(df), index=df.index)
    return (
        df["course"].astype("string").fillna("") + "|"
        + df["race_time"].astype("string").fillna("") + "|"
        + race_name.astype("string").fillna("")
    )


def build_day_payload(df: pd.DataFrame, date_str: str) -> dict:
    """Build the full per-date payload: summary, best bet, value bets, races."""
    df = df.copy()
    dist_col = df["distance_f"] if "distance_f" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    df["distance_band"] = dist_col.apply(distance_band)

    races = [enrich_race(grp) for _, grp in df.groupby(race_key(df), sort=True)]
    races.sort(key=lambda r: (str(r.get("race_time") or ""), str(r.get("course") or "")))

    has_edge = "edge" in df.columns and df["edge"].notna().any()
    best_bet = None
    value_bets = []
    if has_edge:
        edged = df[df["edge"].notna()].sort_values("edge", ascending=False)
        if not edged.empty:
            best_bet = edged.iloc[0].to_dict()
        vb_cols = [
            "date", "race_time", "course", "horse", "win_probability",
            "market_odds", "implied_prob", "edge", "ev_per_unit",
            "best_bookmaker", "odds_trend", "is_value_bet", "is_strong_value",
        ]
        flagged = edged[edged["is_value_bet"].eq(True)] if "is_value_bet" in edged else edged.iloc[:0]
        value_bets = flagged[[c for c in vb_cols if c in flagged.columns]].to_dict("records")

    top_cols = [
        "date", "race_time", "course", "horse", "jockey", "win_probability",
        "win_odds_fractional", "place_probability", "show_probability",
        "market_odds", "edge", "odds_trend", "race_class", "distance_f",
        "distance_band", "ofr",
    ]
    top25 = (
        df.sort_values("win_probability", ascending=False)
        [[c for c in top_cols if c in df.columns]].head(25).to_dict("records")
    )

    def _uniq(col):
        if col not in df.columns:
            return []
        return sorted(str(x) for x in df[col].dropna().unique())

    filters = {
        "classes": _uniq("race_class"),
        "going": _uniq("going"),
        "race_types": _uniq("race_type"),
        "surfaces": _uniq("surface"),
        "courses": _uniq("course"),
    }

    return {
        "date": date_str,
        "summary": {
            "horses": int(len(df)),
            "races": len(races),
            "courses": int(df["course"].nunique()),
            "has_odds": bool(has_edge),
        },
        "best_bet": best_bet,
        "value_bets": value_bets,
        "top25": top25,
        "races": races,
        "filters": filters,
    }


# -- exporters ----------------------------------------------------------------

def export_schedule(today: pd.Timestamp) -> dict:
    path = PROC_DIR / "bha_2026_all_courses_class1-4.csv"
    df = read_csv_safe(path)
    if df.empty:
        return {"count": 0}
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    upcoming = df[df["Date"] >= today].sort_values("Date").head(300)
    return {
        "generated_from": path.name,
        "total_upcoming": int((df["Date"] >= today).sum()),
        "courses": int(upcoming["Course"].nunique()),
        "turf_count": int((upcoming["Surface"] == "Turf").sum()),
        "span_days": int((upcoming["Date"].max() - today).days) if not upcoming.empty else 0,
        "fixtures": upcoming[["Date", "Course", "Time", "Type", "Surface"]].to_dict("records"),
    }


def export_fixtures(today: pd.Timestamp) -> dict:
    df = read_csv_safe(PROC_DIR / "scored_fixtures_calendar.csv")
    if df.empty:
        return {"count": 0}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    upcoming = df[df["date"] >= today].copy()

    tiers = upcoming["race_tier"].fillna("")
    summary = {
        "total": int(len(upcoming)),
        "tier1": int(tiers.str.startswith("Tier 1").sum()),
        "tier2": int(tiers.str.startswith("Tier 2").sum()),
        "avg_score": float(upcoming["race_score"].mean()) if len(upcoming) else 0,
    }

    # Precompute score histogram per tier (20 bins, 0-100)
    bins = np.linspace(0, 100, 21)
    histogram = {"bin_edges": bins.tolist(), "series": {}}
    for tier_name, grp in upcoming.groupby(tiers):
        counts, _ = np.histogram(grp["race_score"].dropna(), bins=bins)
        histogram["series"][tier_name or "Unknown"] = counts.tolist()

    top_courses = (
        upcoming.groupby("course")["race_score"]
        .agg(total_races="size", avg_score="mean", max_score="max")
        .reset_index()
    )
    t1 = upcoming[tiers.str.startswith("Tier 1")].groupby("course").size()
    top_courses["tier1_count"] = top_courses["course"].map(t1).fillna(0).astype(int)
    top_courses["avg_score"] = top_courses["avg_score"].round(1)
    top_courses["max_score"] = top_courses["max_score"].round(1)
    top_courses = top_courses.sort_values("max_score", ascending=False).head(15)

    cols = ["date", "course", "class", "prize", "race_score", "race_tier", "weekday", "surface"]
    keep = [c for c in cols if c in upcoming.columns]

    t1_races = (
        upcoming[tiers.str.startswith("Tier 1")]
        .sort_values(["date", "race_score"], ascending=[True, False])
        .head(50)[keep]
    )

    return {
        "summary": summary,
        "histogram": histogram,
        "top_courses": top_courses.to_dict("records"),
        "fixtures": upcoming.sort_values("date")[keep].to_dict("records"),
        "courses": sorted(upcoming["course"].dropna().unique().tolist()),
        "tiers": sorted(tiers.unique().tolist()),
        "top_tier1_races": t1_races.to_dict("records"),
    }


def _dates_in_window(pattern: str, start: pd.Timestamp, end: pd.Timestamp, fallback_latest: int = 3) -> list:
    """Dates with files inside [start, end]; if empty, the most recent N available."""
    window_dates = []
    day = start
    while day <= end:
        date_str = day.strftime("%Y-%m-%d")
        if (PROC_DIR / pattern.format(d=date_str)).exists():
            window_dates.append(date_str)
        day += timedelta(days=1)
    if window_dates:
        return window_dates
    # Stale pipeline fallback: surface the latest available dates so the site
    # still renders (banner in the UI warns that data is not current).
    all_dates = sorted(
        p.stem.rsplit("_", 1)[-1] for p in PROC_DIR.glob(pattern.format(d="*"))
    )
    return all_dates[-fallback_latest:]


def export_predictions(history_days: int, today: pd.Timestamp):
    """Export GB + US per-date prediction payloads. Returns (gb_dates, us_dates)."""
    gb_dates, us_dates = [], []
    start = today - timedelta(days=history_days)
    end = today + timedelta(days=2)

    for kind, pattern in (("gb", "predictions_{d}.csv"), ("us", "us_predictions_{d}.csv")):
        for date_str in _dates_in_window(pattern, start, end):
            df = read_csv_safe(PROC_DIR / pattern.format(d=date_str))
            if not df.empty and "win_probability" in df.columns:
                payload = build_day_payload(df, date_str)
                if kind == "us":
                    payload["model_provenance"] = {
                        "source_model": df["source_model"].iloc[0] if "source_model" in df.columns else None,
                        "model_artifact": df["model_artifact"].iloc[0] if "model_artifact" in df.columns else None,
                        "model_feature_count": int(df["model_feature_count"].iloc[0]) if "model_feature_count" in df.columns and pd.notna(df["model_feature_count"].iloc[0]) else None,
                    }
                    write_json(f"us_predictions/{date_str}.json", payload)
                    us_dates.append(date_str)
                else:
                    write_json(f"predictions/{date_str}.json", payload)
                    gb_dates.append(date_str)
    return gb_dates, us_dates


def export_us_schedule(today: pd.Timestamp) -> dict:
    """US upcoming schedule from us_entries_{date}.csv (today + next 2 days,
    falling back to latest available dates when the pipeline is stale)."""
    dates = _dates_in_window("us_entries_{d}.csv", today, today + timedelta(days=2), fallback_latest=3)
    days = []
    for date_str in dates:
        df = read_csv_safe(PROC_DIR / f"us_entries_{date_str}.csv")
        if df.empty:
            continue
        scratched = df["scratched"].astype(str).str.lower().isin(["true", "1", "yes"]) if "scratched" in df else pd.Series([False] * len(df))
        races = df.groupby(["track_name", "race_number"]).size() if "race_number" in df else df.groupby("track_name").size()
        track_tbl = (
            df.groupby(["track_code", "track_name"])
            .agg(races=("race_number", "nunique") if "race_number" in df else ("runner_name", "size"),
                 entries=("runner_name", "size"))
            .reset_index()
            if "track_code" in df else None
        )
        rt_cols = [c for c in ("track_name", "race_number", "race_time", "race_name", "surface", "distance") if c in df.columns]
        race_times = df[rt_cols].drop_duplicates(subset=[c for c in ("track_name", "race_number") if c in df.columns]).head(60)
        days.append({
            "date": date_str,
            "tracks": int(df["track_name"].nunique()) if "track_name" in df else 0,
            "races": int(len(races)),
            "entries": int(len(df)),
            "scratches": int(scratched.sum()),
            "tracks_table": track_tbl.to_dict("records") if track_tbl is not None else [],
            "race_times": race_times.to_dict("records"),
        })
    return {"days": days}


def export_us_stakes(today: pd.Timestamp) -> dict:
    path = RAW_DIR / f"abr_stakes_{today.year}.json"
    if not path.exists():
        return {"count": 0}
    data = json.loads(path.read_text(encoding="utf-8"))
    stakes = data.get("stakes", [])
    df = pd.DataFrame(stakes)
    if df.empty:
        return {"count": 0}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    upcoming = df[df["date"] >= today]
    grades = upcoming["grade"].astype(str) if "grade" in upcoming else pd.Series(dtype=str)
    return {
        "year": data.get("year"),
        "fetched_at": data.get("fetched_at"),
        "total": int(len(df)),
        "upcoming": int(len(upcoming)),
        "g1_remaining": int(grades.str.contains("1", na=False).sum()),
        "g2g3_remaining": int(grades.str.contains("2|3", na=False).sum()),
        "stakes": df.to_dict("records"),
    }


def detect_race_type(course: str, race_name: str, race_class: str) -> str:
    combined = f"{course} {race_name} {race_class}".lower()
    for fragment in HARNESS_FRAGMENTS:
        if fragment in combined:
            return "Harness"
    if any(k in combined for k in ("pace", "trot", "standardbred")):
        return "Harness"
    if any(k in combined for k in ("quarter horse", "350y", "400y")):
        return "QuarterHorse"
    return "Thoroughbred"


def export_us_top_races(us_dates: list, today: pd.Timestamp) -> dict:
    """Precompute the US top-predictive-race scoring (ported from us_racing.py)."""
    frames = []
    for d in us_dates:
        df = read_csv_safe(PROC_DIR / f"us_predictions_{d}.csv")
        if not df.empty:
            frames.append(df)
    if not frames:
        return {"count": 0}
    df = pd.concat(frames, ignore_index=True)

    grade_lookup = {}
    stakes_path = RAW_DIR / f"abr_stakes_{today.year}.json"
    if stakes_path.exists():
        data = json.loads(stakes_path.read_text(encoding="utf-8"))
        grade_lookup = {
            s["race_name"].lower().strip(): s["grade"]
            for s in data.get("stakes", [])
            if s.get("grade") and s.get("race_name")
        }

    has_race_type = "race_type" in df.columns
    has_purse = "purse" in df.columns
    for c in ("race_class", "surface", "distance_str"):
        if c not in df.columns:
            df[c] = ""
    group_cols = ["date", "course", "race_time", "race_name", "race_class", "surface", "distance_str"]

    rows = []
    for keys, grp in df.groupby(group_cols, sort=False):
        date, course, race_time, race_name, race_class, surface, distance_str = keys
        probs = grp["win_probability"].sort_values(ascending=False).values
        top_prob = float(probs[0]) if len(probs) else 0.0
        margin = top_prob - float(probs[1]) if len(probs) > 1 else top_prob
        n = len(grp)
        field_score = 1.0 if 8 <= n <= 12 else (0.3 if n < 5 else (0.6 if n > 16 else 0.8))

        rc = str(race_class or "")
        grade = next((g for g in ("G1", "G2", "G3") if g in rc), None)
        if grade is None:
            grade = grade_lookup.get(str(race_name or "").lower().strip())
        purse_val = float(grp["purse"].iloc[0]) if has_purse and pd.notna(grp["purse"].iloc[0]) else 0.0
        purse_bonus = next((b for t, b in PURSE_BONUS if purse_val >= t), 0)
        race_type = str(grp["race_type"].iloc[0]) if has_race_type else detect_race_type(str(course), str(race_name), rc)
        penalty = -15 if race_type in ("Harness", "QuarterHorse") else 0

        score = (top_prob * 40 + margin * 30 + field_score * 10
                 + GRADE_BONUS.get(grade or "", 0) + TRACK_TIER.get(str(course), 0)
                 + purse_bonus + penalty)
        top3 = grp.sort_values("win_probability", ascending=False).head(3)
        spread = (float(top3["win_probability"].iloc[0]) - float(top3["win_probability"].iloc[-1])) if len(top3) >= 3 else 0.0
        rows.append({
            "date": str(date), "course": course, "race_time": race_time,
            "race_name": race_name, "race_class": race_class, "race_type": race_type,
            "surface": surface, "distance": distance_str, "grade": grade or "-",
            "purse": purse_val, "runners": n,
            "top_horse": top3["horse"].iloc[0] if len(top3) else "-",
            "top_odds": top3["win_odds_fractional"].iloc[0] if len(top3) else "-",
            "top_win_pct": round(top_prob * 100, 1), "margin_pct": round(margin * 100, 1),
            "confidence": "High" if spread < 0.12 else ("Medium" if spread < 0.22 else "Low"),
            "score": round(score, 1),
        })

    if not rows:
        return {"count": 0}
    result = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return {
        "summary": {
            "races_ranked": int(len(result)),
            "graded": int((result["grade"] != "-").sum()),
            "tracks": int(result["course"].nunique()),
            "avg_top_win_pct": float(result["top_win_pct"].mean()),
            "avg_margin_pct": float(result["margin_pct"].mean()),
        },
        "scoring_methodology": {
            "top_win_prob": "x40", "margin": "x30", "field_score": "x10",
            "grade_bonus": GRADE_BONUS, "purse_bonus": {str(t): b for t, b in PURSE_BONUS},
            "harness_qh_penalty": -15,
        },
        "races": result.to_dict("records"),
    }


def export_model_insights() -> dict:
    payload: dict = {}
    fi = read_csv_safe(MODELS_DIR / "feature_importance.csv")
    if not fi.empty:
        payload["feature_importance"] = fi.head(15).to_dict("records")
        payload["n_features"] = int(len(fi))
    meta_path = MODELS_DIR / "model_metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as fh:
            md = pickle.load(fh)
        if isinstance(md, dict):
            payload["metadata"] = {k: v for k, v in md.items() if k != "feature_columns"}
    cal_path = MODELS_DIR / "calibration_metrics.json"
    if cal_path.exists():
        payload["calibration"] = json.loads(cal_path.read_text(encoding="utf-8"))
    payload["calibrated_model_exists"] = (MODELS_DIR / "horse_win_predictor_calibrated.pkl").exists()
    diag_path = PROC_DIR / "model_diagnostics.json"
    if diag_path.exists():
        payload["diagnostics"] = json.loads(diag_path.read_text(encoding="utf-8"))
    return payload


def export_us_model_insights() -> dict:
    payload: dict = {
        "us_model_exists": (MODELS_DIR / "us_horse_model.json").exists() or (MODELS_DIR / "us_horse_model.pkl").exists(),
    }
    us_meta = MODELS_DIR / "us_model_metadata.json"
    if us_meta.exists():
        payload["us_metadata"] = json.loads(us_meta.read_text(encoding="utf-8"))
    for name in ("us_races_cleaned.parquet", "all_us_races_cleaned.parquet"):
        p = PROC_DIR / name
        if p.exists():
            try:
                import pyarrow.parquet as pq
                payload["us_training_rows"] = int(pq.read_metadata(p).num_rows)
                payload["us_training_file"] = name
                break
            except Exception:
                pass

    acc_files = sorted(PROC_DIR.glob("us_accuracy_*.csv"))[-30:]
    if acc_files:
        acc = pd.concat([read_csv_safe(f) for f in acc_files], ignore_index=True)
        if not acc.empty:
            per_track = (
                acc.groupby("course")
                .agg(races=("top1_correct", "size"), top1=("top1_correct", "mean"), top3=("top3_correct", "mean"))
                .reset_index().sort_values("races", ascending=False)
            )
            payload["accuracy"] = {
                "days_tracked": int(acc["date"].nunique()) if "date" in acc else len(acc_files),
                "races_evaluated": int(len(acc)),
                "top1_accuracy": float(acc["top1_correct"].mean()) if "top1_correct" in acc else None,
                "top3_accuracy": float(acc["top3_correct"].mean()) if "top3_correct" in acc else None,
                "per_track": per_track.head(20).to_dict("records"),
                "recent": acc.tail(50).to_dict("records"),
            }
    return payload


def _stats_source() -> pd.DataFrame:
    import pyarrow.parquet as pq
    candidates = []
    for name in (
        "race_scores_engineered.parquet", "race_scores_or_context.parquet",
        "race_scores_connections_v2.parquet", "race_scores.parquet",
    ):
        path = PROC_DIR / name
        if path.exists():
            wanted = ["date", "trainer", "jockey", "pos_clean", "course_clean", "going"]
            try:
                available = set(pq.read_schema(path).names)
            except (OSError, ValueError) as exc:
                print(f'  Skipping unreadable stats source {name}: {exc}')
                continue
            if not {'date', 'trainer', 'jockey'}.issubset(available):
                continue
            if not ({'pos_clean', 'pos'} & available) or not ({'course_clean', 'course'} & available):
                continue
            cols = [c for c in wanted if c in available]
            if "course_clean" not in cols and "course" in available:
                cols.append("course")
            if "pos_clean" not in cols and "pos" in available:
                cols.append("pos")
            df = pd.read_parquet(path, columns=cols)
            if "course_clean" not in df.columns and "course" in df.columns:
                df["course_clean"] = df["course"]
            if "pos_clean" not in df.columns and "pos" in df.columns:
                df["pos_clean"] = df["pos"]
            latest = pd.to_datetime(df['date'], errors='coerce').max()
            if pd.notna(latest):
                candidates.append((latest, len(df), name, df))
    if not candidates:
        return pd.DataFrame()
    latest, _, name, df = max(candidates, key=lambda item: item[:2])
    print(f'  Stats source: {name}, results through {latest:%Y-%m-%d}')
    return df


def _form_table(df: pd.DataFrame, key: str) -> list:
    d = df.dropna(subset=[key])
    g = d.groupby(key)
    out = pd.DataFrame({
        "runs": g.size(),
        "wins": g.apply(lambda x: int((x["pos_clean"] == 1).sum()), include_groups=False),
        "places": g.apply(lambda x: int((x["pos_clean"] <= 3).sum()), include_groups=False),
    }).reset_index()
    out["win_pct"] = (out["wins"] / out["runs"] * 100).round(1)
    out["place_pct"] = (out["places"] / out["runs"] * 100).round(1)
    return out.sort_values("wins", ascending=False).to_dict("records")


def _pair_table(df: pd.DataFrame, k1: str, k2: str, min_runs: int, limit: int) -> list:
    d = df.dropna(subset=[k1, k2])
    g = d.groupby([k1, k2])
    out = pd.DataFrame({
        "runs": g.size(),
        "wins": g.apply(lambda x: int((x["pos_clean"] == 1).sum()), include_groups=False),
        "places": g.apply(lambda x: int((x["pos_clean"] <= 3).sum()), include_groups=False),
    }).reset_index()
    out = out[out["runs"] >= min_runs]
    out["win_pct"] = (out["wins"] / out["runs"] * 100).round(1)
    out["place_pct"] = (out["places"] / out["runs"] * 100).round(1)
    return out.sort_values(["win_pct", "runs"], ascending=False).head(limit).to_dict("records")


def export_entity_stats(df: pd.DataFrame, entity: str, window: int, ref_date: pd.Timestamp) -> dict:
    """entity = 'trainer' | 'jockey'. Precompute every table the stats pages show."""
    other = "jockey" if entity == "trainer" else "trainer"
    cutoff = ref_date - timedelta(days=window)
    d = df[(df["date"] >= cutoff) & (df["date"] <= ref_date)].copy()
    d["pos_clean"] = pd.to_numeric(d["pos_clean"], errors="coerce")
    d["course"] = d["course_clean"].fillna("")
    d["going_n"] = d["going"].fillna("Unknown").astype(str).str.strip().str.title()

    recent_cutoff = ref_date - timedelta(days=14)
    recent = _form_table(d[d["date"] >= recent_cutoff], entity)

    leaderboard = _form_table(d, entity)
    specialists = _pair_table(d, entity, "course", 5, 200)
    partnerships = _pair_table(d, entity, other, 10, 100)

    # Going patterns for the busiest entities (kept small for JSON)
    top_entities = [r[entity] for r in leaderboard[:25]]
    going_patterns = {}
    for name in top_entities:
        sub = d[d[entity] == name]
        g = sub.groupby("going_n")
        tbl = pd.DataFrame({
            "runs": g.size(),
            "wins": g.apply(lambda x: int((x["pos_clean"] == 1).sum()), include_groups=False),
            "places": g.apply(lambda x: int((x["pos_clean"] <= 3).sum()), include_groups=False),
        }).reset_index()
        if not tbl.empty:
            tbl["win_pct"] = (tbl["wins"] / tbl["runs"] * 100).round(1)
            going_patterns[name] = tbl.to_dict("records")

    payload = {
        "entity": entity,
        "window_days": window,
        "as_of": ref_date.strftime("%Y-%m-%d"),
        "leaderboard": leaderboard[:150],
        "recent_form_14d": recent[:50],
        "course_specialists": specialists,
        "partnerships": partnerships,
        "going_patterns": going_patterns,
        "courses": sorted(x for x in d["course"].unique() if x),
    }

    if entity == "jockey":
        # Precompute 30-race rolling win-rate series for the top jockeys
        trends = {}
        for name in [r["jockey"] for r in leaderboard[:20]]:
            sub = d[d["jockey"] == name].sort_values("date")
            if len(sub) >= 10:
                wins = (sub["pos_clean"] == 1).astype(float)
                roll = wins.rolling(30, min_periods=5).mean() * 100
                trends[name] = {
                    "dates": sub["date"].dt.strftime("%Y-%m-%d").tolist()[-120:],
                    "rolling_win_pct": roll.round(1).tolist()[-120:],
                    "last30_win_pct": round(float(wins.tail(30).mean() * 100), 1),
                    "last30_wins": int(wins.tail(30).sum()),
                }
        payload["form_trends"] = trends
    return payload


def export_backtest() -> dict:
    payload: dict = {}
    for cand in (PROC_DIR / "backtest_results.json", MODELS_DIR / "backtest_results.json"):
        if cand.exists():
            payload["results"] = json.loads(cand.read_text(encoding="utf-8"))
            break
    folds = (payload.get("results") or {}).get("folds") or []
    if folds:
        aucs = [f["auc"] for f in folds if f.get("auc") is not None]
        payload["summary"] = {
            "n_folds": len(folds),
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_min": float(np.min(aucs)) if aucs else None,
            "auc_max": float(np.max(aucs)) if aucs else None,
            "top1_mean": float(np.mean([f.get("top1_accuracy", 0) for f in folds])),
            "top3_mean": float(np.mean([f.get("top3_accuracy", 0) for f in folds])),
            "level_roi_pct_mean": float(np.mean([f.get("level_roi_pct", 0) for f in folds])),
        }

    # Accuracy / ROI / calibration from historical predictions with results
    frames = []
    for f in sorted(PROC_DIR.glob("predictions_*.csv")):
        df = read_csv_safe(f)
        if not df.empty and "actual_result" in df.columns and df["actual_result"].notna().any():
            frames.append(df)
    if frames:
        hist = pd.concat(frames, ignore_index=True)
        hist = hist[hist["actual_result"].notna()].copy()
        hist["actual_result"] = pd.to_numeric(hist["actual_result"], errors="coerce")
        hist["win_probability"] = pd.to_numeric(hist["win_probability"], errors="coerce")
        hist = hist.dropna(subset=["win_probability", "actual_result"])
        if "distance_band" not in hist.columns:
            dist_col = hist["distance_f"] if "distance_f" in hist.columns else pd.Series([np.nan] * len(hist), index=hist.index)
            hist["distance_band"] = dist_col.apply(distance_band)
        if "date" in hist.columns:
            hist["month"] = pd.to_datetime(hist["date"], errors="coerce").dt.strftime("%Y-%m")

        segments = {}
        for seg in ("race_class", "distance_band", "going", "month"):
            if seg not in hist.columns:
                continue
            sub = hist.dropna(subset=[seg])
            if sub.empty:
                continue
            g = sub.groupby(sub[seg].astype(str))
            seg_tbl = pd.DataFrame({
                "count": g.size(),
                "accuracy": g.apply(lambda x: float(((x["win_probability"] > 0.5) == (x["actual_result"] == 1)).mean()), include_groups=False),
                "avg_prob": g["win_probability"].mean(),
                "actual_win_rate": g["actual_result"].mean(),
            }).reset_index().rename(columns={seg: "segment"})
            roi_tbl = None
            if "market_odds" in sub.columns:
                m = sub.dropna(subset=["market_odds"])
                if not m.empty:
                    profit = np.where(m["actual_result"] == 1, m["market_odds"] - 1, -1.0)
                    roi = pd.DataFrame({"segment": m[seg].astype(str), "profit": profit})
                    roi_g = roi.groupby("segment").agg(bets=("profit", "size"), total_profit=("profit", "sum")).reset_index()
                    roi_g["roi_pct"] = (roi_g["total_profit"] / roi_g["bets"] * 100).round(1)
                    roi_tbl = roi_g.to_dict("records")
            segments[seg] = {"accuracy": seg_tbl.to_dict("records"), "roi": roi_tbl}

        # Calibration bins (10 equal-width bins, min 10 samples)
        hist["bin"] = pd.cut(hist["win_probability"], bins=10)
        cal = hist.groupby("bin", observed=True).agg(
            predicted_mean=("win_probability", "mean"),
            actual_rate=("actual_result", "mean"),
            count=("actual_result", "size"),
        ).reset_index()
        cal = cal[cal["count"] >= 10]
        cal["bin_label"] = cal["bin"].astype(str)

        payload["historical"] = {
            "n_predictions": int(len(hist)),
            "date_range": [str(hist["date"].min()), str(hist["date"].max())] if "date" in hist else None,
            "segments": segments,
            "calibration": cal.drop(columns=["bin"]).to_dict("records"),
        }
    return payload


def export_explorer_overview() -> dict:
    path = PROC_DIR / "race_scores.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= "2021-01-01"]
    df["pos_n"] = pd.to_numeric(df["pos"], errors="coerce")
    df['_won'] = df['pos_n'].eq(1)
    df['_top3'] = df['pos_n'].le(3)

    def leaders(key, top=20):
        g = df.dropna(subset=[key]).groupby(key)
        out = pd.DataFrame({
            "races": g.size(),
            "avg_finish": g["pos_n"].mean().round(2),
            "wins": g['_won'].sum(),
            "top3": g['_top3'].sum(),
        }).reset_index()
        out["win_rate_pct"] = (out["wins"] / out["races"] * 100).round(1)
        return out.sort_values("wins", ascending=False).head(top).to_dict("records")

    prize = pd.to_numeric(df["prize"], errors="coerce") if "prize" in df else None
    return {
        "as_of": df["date"].max().strftime("%Y-%m-%d"),
        "summary": {
            "total_rows": int(len(df)),
            "horses": int(df["horse"].nunique()),
            "jockeys": int(df["jockey"].nunique()),
            "trainers": int(df["trainer"].nunique()),
            "courses": int(df["course"].nunique()),
            "avg_finish": float(df["pos_n"].mean()),
            "total_prize": float(prize.sum()) if prize is not None else None,
        },
        "top_horses": leaders("horse"),
        "top_jockeys": leaders("jockey"),
        "top_courses": leaders("course"),
        "distance_distribution": df["dist"].value_counts().head(10).reset_index()
            .rename(columns={"dist": "distance", "count": "races"}).to_dict("records")
            if "dist" in df.columns else [],
        "filter_domains": {
            "years": sorted(df["date"].dt.year.dropna().unique().astype(int).tolist(), reverse=True),
            "courses": sorted(df["course"].dropna().unique().tolist()),
            "classes": sorted(df["class"].dropna().unique().tolist()) if "class" in df else [],
            "going": sorted(df["going"].dropna().unique().tolist()) if "going" in df else [],
            "surfaces": sorted(df["surface"].dropna().unique().tolist()) if "surface" in df else [],
            "age_bands": sorted(df["age_band"].dropna().unique().tolist()) if "age_band" in df else [],
            "tiers": sorted(df["race_tier"].dropna().unique().tolist()) if "race_tier" in df else [],
        },
    }


def export_betting_watchlist() -> dict:
    df = read_csv_safe(PROC_DIR / "betting_watchlist.csv")
    if df.empty:
        return {"count": 0}
    cols = ["date", "course", "class", "prize", "race_score", "weekday", "betting_tier", "race_tier"]
    keep = [c for c in cols if c in df.columns]
    tier_col = df["betting_tier"].astype(str) if "betting_tier" in df else pd.Series([""] * len(df))
    return {
        "total": int(len(df)),
        "tier1": df[tier_col.str.contains("1", na=False)][keep].to_dict("records"),
        "tier2": df[tier_col.str.contains("2", na=False)][keep].to_dict("records"),
    }


# -- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=14, help="days of prediction history to export")
    parser.add_argument("--skip-stats", action="store_true", help="skip trainer/jockey window stats")
    args = parser.parse_args()

    now = datetime.now().astimezone().replace(tzinfo=None)
    today = pd.Timestamp(now.date())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting precomputed web data -> {OUT_DIR}")

    manifest: dict = {"generated_at": now.isoformat(), "stats_windows": list(STATS_WINDOWS)}

    print("\n[1/9] Upcoming schedule")
    write_json("schedule_upcoming.json", export_schedule(today))

    print("\n[2/9] Scored fixtures")
    write_json("fixtures_scored.json", export_fixtures(today))

    print("\n[3/9] Predictions (GB + US)")
    gb_dates, us_dates = export_predictions(args.history, today)
    manifest["gb_prediction_dates"] = gb_dates
    manifest["us_prediction_dates"] = us_dates

    print("\n[4/9] US stakes + schedule + top races")
    write_json("us_stakes.json", export_us_stakes(today))
    write_json("us_schedule.json", export_us_schedule(today))
    write_json("us_top_races.json", export_us_top_races(us_dates, today))

    print("\n[5/9] Model insights")
    write_json("model_insights.json", export_model_insights())
    write_json("us_model_insights.json", export_us_model_insights())

    print("\n[6/9] Trainer / jockey stats")
    if not args.skip_stats:
        src = _stats_source()
        if not src.empty:
            src["date"] = pd.to_datetime(src["date"], errors="coerce")
            ref = src["date"].max()
            manifest["stats_as_of"] = ref.strftime("%Y-%m-%d")
            for window in STATS_WINDOWS:
                write_json(f"trainer_stats_{window}.json", export_entity_stats(src, "trainer", window, ref))
                write_json(f"jockey_stats_{window}.json", export_entity_stats(src, "jockey", window, ref))
        else:
            print("  ! no stats source parquet found")
    else:
        print("  skipped")
        prior_stats = OUT_DIR / "trainer_stats_90.json"
        if prior_stats.exists():
            manifest["stats_as_of"] = json.loads(prior_stats.read_text(encoding="utf-8"))["as_of"]

    print("\n[7/9] Backtest")
    write_json("backtest.json", export_backtest())

    print("\n[8/9] Explorer overview + watchlist")
    write_json("explorer_overview.json", export_explorer_overview())
    write_json("betting_watchlist.json", export_betting_watchlist())

    print("\n[9/9] Manifest")
    write_json("manifest.json", manifest)
    print("\nDone.")


if __name__ == "__main__":
    main()
