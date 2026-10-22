"""Server-side filtering + aggregation over the historical race parquet.

This backs the Data Explorer page - the one view that cannot be fully
precomputed (user-driven filter combinations over ~650K rows).
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pandas as pd

from .. import config

_LOCK = threading.Lock()
_DF: pd.DataFrame | None = None

COLUMNS = [
    "date", "course", "race_name", "horse", "pos", "jockey", "trainer",
    "dist", "going", "time", "or", "prize", "class", "pattern", "surface",
    "race_tier", "race_score", "ran", "age_band", "dist_f",
]

RAW_COLUMNS = [
    "date", "course", "race_name", "horse", "pos", "jockey", "trainer",
    "dist", "going", "time", "or", "prize", "class", "race_score", "race_tier",
]


def _load() -> pd.DataFrame:
    global _DF
    with _LOCK:
        if _DF is not None:
            return _DF
        import pyarrow.parquet as pq

        path = config.DATA_DIR / config.EXPLORER_PARQUET
        available = set(pq.read_schema(path).names)
        cols = [c for c in COLUMNS if c in available]
        df = pd.read_parquet(path, columns=cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df[df["date"] >= config.EXPLORER_START_DATE]
        df["pos_n"] = pd.to_numeric(df["pos"], errors="coerce")
        df["or_n"] = pd.to_numeric(df.get("or"), errors="coerce")
        df["prize_n"] = pd.to_numeric(df.get("prize"), errors="coerce")
        df["score_n"] = pd.to_numeric(df.get("race_score"), errors="coerce")
        df["ran_n"] = pd.to_numeric(df.get("ran"), errors="coerce")
        df["dist_f_n"] = pd.to_numeric(df.get("dist_f"), errors="coerce")
        df["year"] = df["date"].dt.year
        _DF = df
        return df


def get_filter_domains() -> dict[str, Any]:
    df = _load()
    def bounds(column, default):
        values = df[column].dropna()
        return [float(values.min()), float(values.max())] if len(values) else default

    return {
        "years": sorted(df["year"].dropna().unique().astype(int).tolist(), reverse=True),
        "courses": sorted(df["course"].dropna().unique().tolist()),
        "classes": sorted(df["class"].dropna().astype(str).unique().tolist()) if "class" in df else [],
        "going": sorted(df["going"].dropna().astype(str).unique().tolist()) if "going" in df else [],
        "surfaces": sorted(df["surface"].dropna().astype(str).unique().tolist()) if "surface" in df else [],
        "age_bands": sorted(df["age_band"].dropna().astype(str).unique().tolist()) if "age_band" in df else [],
        "tiers": sorted(df["race_tier"].dropna().astype(str).unique().tolist()) if "race_tier" in df else [],
        "finish_positions": sorted(int(x) for x in df["pos_n"].dropna().unique() if x <= 30),
        "or_range": bounds("or_n", [0, 200]),
        "score_range": bounds("score_n", [0, 100]),
        "field_size_range": bounds("ran_n", [1, 40]),
    }


def apply_filters(df: pd.DataFrame, f: "ExplorerFilters") -> pd.DataFrame:
    if f.years:
        df = df[df["year"].isin(f.years)]
    if f.courses:
        df = df[df["course"].isin(f.courses)]
    if f.horse_search:
        df = df[df["horse"].astype(str).str.contains(f.horse_search, case=False, na=False, regex=False)]
    if f.going:
        df = df[df["going"].astype(str).isin(f.going)]
    if f.finish_positions:
        df = df[df["pos_n"].isin(f.finish_positions)]
    if f.top3_only:
        df = df[df["pos_n"] <= 3]
    if f.or_min is not None:
        df = df[df["or_n"] >= f.or_min]
    if f.or_max is not None:
        df = df[df["or_n"] <= f.or_max]
    if f.tiers:
        df = df[df["race_tier"].astype(str).isin(f.tiers)]
    if f.score_min is not None:
        df = df[df["score_n"] >= f.score_min]
    if f.score_max is not None:
        df = df[df["score_n"] <= f.score_max]
    if f.classes:
        df = df[df["class"].astype(str).isin(f.classes)]
    if f.min_prize is not None:
        df = df[df["prize_n"] >= f.min_prize]
    if f.field_size_min is not None:
        df = df[df["ran_n"] >= f.field_size_min]
    if f.field_size_max is not None:
        df = df[df["ran_n"] <= f.field_size_max]
    if f.surfaces:
        df = df[df["surface"].astype(str).isin(f.surfaces)]
    if f.distance_band and f.distance_band != "All":
        band_ranges = {"Sprint (5-7f)": (4.5, 7.5), "Mile (7-9f)": (7.5, 9.5), "Middle (9-12f)": (9.5, 12.5), "Long (12f+)": (12.5, 100)}
        lo, hi = band_ranges[f.distance_band]
        df = df[(df["dist_f_n"] >= lo) & (df["dist_f_n"] < hi)]
    if f.pattern_only:
        df = df[df["pattern"].fillna("").astype(str).str.strip().ne("")] if "pattern" in df else df.iloc[:0]
    if f.age_bands:
        df = df[df["age_band"].astype(str).isin(f.age_bands)]
    return df


def _leaders(df: pd.DataFrame, key: str, limit: int) -> list[dict]:
    d = df.dropna(subset=[key])
    if d.empty:
        return []
    out = d.assign(_win=d["pos_n"].eq(1), _top3=d["pos_n"].le(3)).groupby(key).agg(
        races=("pos_n", "size"), avg_finish=("pos_n", "mean"),
        wins=("_win", "sum"), top3=("_top3", "sum"), total_prize=("prize_n", "sum"),
    ).reset_index()
    out["avg_finish"] = out["avg_finish"].round(2)
    out["total_prize"] = out["total_prize"].round(0)
    out["win_rate_pct"] = (out["wins"] / out["races"] * 100).round(1)
    return out.sort_values("wins", ascending=False).head(limit).to_dict("records")


def _clean_records(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, float) and np.isnan(v):
                clean[k] = None
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.strftime("%Y-%m-%d")
            elif v is pd.NA or v is pd.NaT:
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def query(f: "ExplorerFilters") -> dict[str, Any]:
    df = apply_filters(_load(), f)

    result: dict[str, Any] = {
        "total_rows": int(len(df)),
        "overall": {
            "horses": int(df["horse"].nunique()),
            "jockeys": int(df["jockey"].nunique()),
            "trainers": int(df["trainer"].nunique()),
            "courses": int(df["course"].nunique()),
            "avg_finish": float(df["pos_n"].mean()) if df["pos_n"].notna().any() else None,
            "total_prize": float(df["prize_n"].sum()) if len(df) else None,
        },
    }

    if f.view in ("horses", "all"):
        result["horses"] = _clean_records(_leaders(df, "horse", 100))
    if f.view in ("courses", "all"):
        result["courses"] = _clean_records(_leaders(df, "course", 100))
    if f.view in ("jockeys", "all"):
        result["jockeys"] = _clean_records(_leaders(df, "jockey", 100))
    if f.view in ("raw", "all"):
        page_size = min(f.page_size, config.MAX_PAGE_SIZE)
        start = (f.page - 1) * page_size
        cols = [c for c in RAW_COLUMNS if c in df.columns]
        rows = (
            df.sort_values("date", ascending=False)[cols]
            .iloc[start:start + page_size]
        )
        result["rows"] = _clean_records(rows.to_dict("records"))
        result["page"] = f.page
        result["page_size"] = page_size
        result["total_pages"] = int(np.ceil(len(df) / page_size)) if len(df) else 0
    return result


from pydantic import BaseModel, ConfigDict, Field  # noqa: E402
from typing import Literal


class ExplorerFilters(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    years: list[int] | None = None
    courses: list[str] | None = None
    going: list[str] | None = None
    horse_search: str | None = None
    finish_positions: list[int] | None = None
    top3_only: bool = False
    or_min: float | None = None
    or_max: float | None = None
    tiers: list[str] | None = None
    score_min: float | None = None
    score_max: float | None = None
    classes: list[str] | None = None
    min_prize: float | None = None
    field_size_min: float | None = None
    field_size_max: float | None = None
    surfaces: list[str] | None = None
    distance_band: Literal["All", "Sprint (5-7f)", "Mile (7-9f)", "Middle (9-12f)", "Long (12f+)"] | None = None
    pattern_only: bool = False
    age_bands: list[str] | None = None
    view: Literal["horses", "courses", "jockeys", "raw", "all"] = "all"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=config.MAX_PAGE_SIZE)
