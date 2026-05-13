"""
build_us_training_data.py — Consolidate all available US result sources into
                            data/processed/us_races_cleaned.parquet.

Sources consumed (priority order — later sources fill gaps):
  1. TVG GraphQL results  …  data/raw/tvg_results_YYYY-MM-DD.json
  2. T1-TB Playwright scraper  …  data/processed/us_t1_tb_canonical_*.csv
  3. Equibase static-chart HTML  …  data/processed/equibase_results_YYYY-MM-DD.csv

For canonical CSVs (T1-TB scraper) finish positions are inferred from payoffs:
    win_payoff present  →  position = 1
    place_payoff only   →  position = 2
    show_payoff only    →  position = 3
    "Also ran" in notes →  position = 4, 5, 6 … (program order from notes)

Usage:
    python scripts/build_us_training_data.py
    python scripts/build_us_training_data.py --dry-run
    python scripts/build_us_training_data.py --force    # rebuild from scratch
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR    = REPO_ROOT / "data" / "raw"
PROC_DIR   = REPO_ROOT / "data" / "processed"
OUT_PATH   = PROC_DIR / "us_races_cleaned.parquet"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] build_us_training: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("build_us_training")


# ---------------------------------------------------------------------------
# Source 1: TVG raw JSON  (data/raw/tvg_results_YYYY-MM-DD.json)
# ---------------------------------------------------------------------------

def _decimal_odds(odds_obj: dict | None) -> float | None:
    if not isinstance(odds_obj, dict):
        return None
    num = odds_obj.get("numerator")
    den = odds_obj.get("denominator")
    if den and den != 0:
        try:
            return round(float(num) / float(den) + 1, 3)
        except (TypeError, ValueError):
            pass
    return None


def load_tvg_json(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("tvg_results_*.json"))
    if not files:
        logger.info("TVG JSON: no files found")
        return pd.DataFrame()

    rows: list[dict] = []
    for fp in files:
        try:
            races = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read %s: %s", fp.name, exc)
            continue

        for race in races:
            track    = race.get("track") or {}
            loc      = track.get("location") or {}
            race_id  = race.get("id", "")
            date_str = race.get("date", "")
            course   = track.get("name") or track.get("code") or ""
            surface  = (race.get("surface") or {}).get("name") or "Dirt"
            distance = (race.get("distance") or {}).get("value") or ""
            r_class  = (race.get("raceClass") or {}).get("name") or ""
            r_type   = (race.get("type") or {}).get("name") or ""
            r_name   = race.get("description") or ""
            purse    = race.get("purse")
            fsize    = race.get("numRunners")
            all_runners = (race.get("results") or {}).get("allRunners") or []

            for runner in all_runners:
                if runner.get("scratched"):
                    continue
                pos_raw = runner.get("finishPosition")
                status  = runner.get("finishStatus") or ""
                if pos_raw is None and not status:
                    continue
                try:
                    position = int(pos_raw) if pos_raw is not None else None
                except (TypeError, ValueError):
                    position = None

                rows.append({
                    "horse":       runner.get("runnerName") or "",
                    "date":        date_str,
                    "course":      course,
                    "track_code":  track.get("code") or "",
                    "surface":     surface,
                    "distance":    distance,
                    "race_class":  r_class,
                    "race_type":   r_type,
                    "race_name":   r_name,
                    "position":    str(position) if position is not None else (status or None),
                    "finish_status": status,
                    "race_id":     race_id,
                    "race_number": race.get("number"),
                    "field_size":  int(fsize) if fsize else len(all_runners),
                    "purse":       purse,
                    "claiming_price": race.get("claimingPrice"),
                    "draw":        runner.get("runnerNumber"),
                    "win_payoff":  runner.get("winPayoff"),
                    "place_payoff": runner.get("placePayoff"),
                    "show_payoff": runner.get("showPayoff"),
                    "is_favorite": runner.get("favorite"),
                    "dec":         _decimal_odds(runner.get("currentOdds")),
                    "city":        loc.get("city"),
                    "state":       loc.get("state"),
                    "country":     loc.get("country"),
                    "source":      "tvg",
                })

    df = pd.DataFrame(rows)
    logger.info("TVG JSON: %d files → %d runner rows", len(files), len(df))
    return df


# ---------------------------------------------------------------------------
# Source 2: T1-TB canonical CSVs  (data/processed/us_t1_tb_canonical_*.csv)
# ---------------------------------------------------------------------------

_ALSO_RAN_RE = re.compile(r"(?:(?:^|(?<=\s))(\d+)\s*[-–]\s*([^0-9\n]+?)(?=\s+\d+\s*[-–]|\s*$))", re.DOTALL)


def _infer_position(row: pd.Series) -> str | None:
    """Infer finish position from payoff columns."""
    if pd.notna(row.get("win_payoff")):
        return "1"
    if pd.notna(row.get("place_payoff")):
        return "2"
    if pd.notna(row.get("show_payoff")):
        return "3"
    return None


def _parse_also_rans(notes: str) -> list[tuple[str, str]]:
    """
    Parse 'Also ran: 1 - Horse A 2 - Horse B ...' and return
    list of (program_number, horse_name) tuples.
    """
    if not notes or "also ran" not in notes.lower():
        return []
    text = re.sub(r"(?i)also\s+ran\s*[:–-]?\s*", "", notes).strip()
    results = []
    for m in re.finditer(r"(\d+)\s*[-–]\s*([^\d\n]+?)(?=\s+\d+\s*[-–]|\s*$)", text.strip()):
        prog = m.group(1).strip()
        name = m.group(2).strip().rstrip(",;")
        if name:
            results.append((prog, name))
    return results


def load_canonical_csvs(proc_dir: Path = PROC_DIR) -> pd.DataFrame:
    files = sorted(proc_dir.glob("us_t1_tb_canonical_*.csv"))
    if not files:
        logger.info("Canonical CSVs: no files found")
        return pd.DataFrame()

    all_frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            logger.warning("Could not read %s: %s", fp.name, exc)
            continue
        if df.empty or "event_type" not in df.columns:
            continue
        df = df[df["event_type"] == "results"].copy()
        if df.empty:
            continue
        all_frames.append(df)

    if not all_frames:
        logger.info("Canonical CSVs: no results rows")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Deduplicate rows (canonical_all is a superset of per-date files)
    key_cols = ["date", "track_code", "race_number", "runner_name"]
    key_cols = [c for c in key_cols if c in combined.columns]
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols)

    rows: list[dict] = []
    for _, row in combined.iterrows():
        pos = _infer_position(row)
        notes = str(row.get("notes") or "")

        base = {
            "horse":       str(row.get("runner_name") or ""),
            "date":        str(row.get("date") or ""),
            "course":      str(row.get("track_name") or ""),
            "track_code":  str(row.get("track_code") or ""),
            "surface":     str(row.get("surface") or "Dirt"),
            "distance":    str(row.get("distance") or ""),
            "race_class":  str(row.get("race_class") or ""),
            "race_name":   str(row.get("race_name") or ""),
            "race_number": row.get("race_number"),
            "position":    pos,
            "win_payoff":  row.get("win_payoff"),
            "place_payoff": row.get("place_payoff"),
            "show_payoff": row.get("show_payoff"),
            "draw":        row.get("runner_number"),
            "source":      "t1_tb_canonical",
        }

        if pos is not None:
            rows.append(base)

        # Add also-ran horses at positions 4+
        also_rans = _parse_also_rans(notes)
        for idx, (prog, name) in enumerate(also_rans):
            also_row = dict(base)
            also_row["horse"]    = name
            also_row["draw"]     = prog
            also_row["position"] = str(4 + idx)
            also_row["win_payoff"] = None
            also_row["place_payoff"] = None
            also_row["show_payoff"] = None
            rows.append(also_row)

    df_out = pd.DataFrame(rows)
    logger.info("Canonical CSVs: %d files → %d runner rows", len(files), len(df_out))
    return df_out


# ---------------------------------------------------------------------------
# Source 3: Equibase static chart CSVs
# ---------------------------------------------------------------------------

def load_equibase_csvs(proc_dir: Path = PROC_DIR) -> pd.DataFrame:
    files = sorted(proc_dir.glob("equibase_results_*.csv"))
    if not files:
        logger.info("Equibase CSVs: no files found")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
            # Rename columns to match our schema
            df = df.rename(columns={
                "race_date": "date",
                "horse":     "horse",
                "position":  "position",
            })
            df["source"] = "equibase_html"
            frames.append(df)
        except Exception as exc:
            logger.warning("Could not read %s: %s", fp.name, exc)

    if not frames:
        return pd.DataFrame()

    df_out = pd.concat(frames, ignore_index=True)
    logger.info("Equibase CSVs: %d files → %d runner rows", len(files), len(df_out))
    return df_out


# ---------------------------------------------------------------------------
# Merge + deduplicate
# ---------------------------------------------------------------------------

def _dedup_key(df: pd.DataFrame) -> pd.Series:
    """Composite key for deduplication: date + track + race# + horse."""
    parts = []
    for col in ("date", "track_code", "race_number", "horse"):
        parts.append(df[col].astype(str) if col in df.columns else pd.Series("", index=df.index))
    return parts[0] + "|" + parts[1] + "|" + parts[2] + "|" + parts[3]


def merge_sources(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all source frames, align columns, deduplicate."""
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    # Drop rows with no horse name or no date
    combined = combined[combined["horse"].str.strip().astype(bool)]
    combined = combined[combined["date"].notna()]

    # Deduplicate — keep first occurrence (TVG has priority as it's loaded first)
    key = _dedup_key(combined)
    combined = combined[~key.duplicated(keep="first")].reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate all US result sources into us_races_cleaned.parquet"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing")
    parser.add_argument("--force",   action="store_true",
                        help="Rebuild from scratch (ignore existing parquet)")
    args = parser.parse_args()

    print("=" * 65)
    print("BUILD US TRAINING DATA  (consolidate → us_races_cleaned.parquet)")
    print("=" * 65)

    df_tvg   = load_tvg_json()
    df_canon = load_canonical_csvs()
    df_eqb   = load_equibase_csvs()

    # TVG first so its records win on dedup
    combined = merge_sources(df_tvg, df_canon, df_eqb)

    if combined.empty:
        print("\nNo data found in any source. Run fetch_tvg_results.py first.")
        return

    # Summary
    winners = combined[combined["position"].isin(["1", 1])].shape[0]
    tracks  = combined["track_code"].nunique() if "track_code" in combined else "?"
    sources = combined["source"].value_counts().to_dict() if "source" in combined else {}
    date_range = f"{combined['date'].min().date()} → {combined['date'].max().date()}"

    print(f"\nRows: {len(combined):,}")
    print(f"Winners (position=1): {winners:,}")
    print(f"Tracks: {tracks}")
    print(f"Date range: {date_range}")
    print(f"By source: {sources}")

    if args.dry_run:
        print("\n[dry-run] Would write to us_races_cleaned.parquet. Exiting.")
        return

    # If not force, merge with any existing parquet
    if OUT_PATH.exists() and not args.force:
        existing = pd.read_parquet(OUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        logger.info("Existing parquet: %d rows", len(existing))
        combined = merge_sources(combined, existing)
        logger.info("After merge with existing: %d rows", len(combined))

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH.name}  ({len(combined):,} rows)")
    print("\nNext: python scripts/train_us_model.py --min-rows 1000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
