"""
Generate horse racing predictions for US races.

Reads:   data/raw/us_racecards_YYYY-MM-DD.json
Writes:  data/processed/us_predictions_YYYY-MM-DD.csv

Feature-building reuses the UK XGBoost model trained on 47 core features,
adapting US-specific fields (class, going, distance, surface) to the same
numeric scale the model expects.  When a US-specific model is available at
models/us_horse_model.pkl it is used automatically instead.

Usage:
    python scripts/predict_us_races.py --date 2026-05-09
    python scripts/predict_us_races.py              # defaults to today
"""

import os
import sys
import json
import pickle
import argparse
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).parent))
from odds_converter import probability_to_fractional_odds
from us_distance_parser import parse_us_distance, get_distance_band_us
from us_class_mapper import map_us_class_to_numeric
from us_going_mapper import map_us_going_to_numeric, get_surface_going_key

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROC = BASE_DIR / 'data' / 'processed'

warnings.filterwarnings('ignore')

# ── Premium US track tiers ────────────────────────────────────────────────────
US_COURSE_TIERS: dict[str, str] = {
    # Premium
    'churchill downs': 'Premium', 'belmont park': 'Premium',
    'santa anita': 'Premium', 'santa anita park': 'Premium',
    'keeneland': 'Premium', 'saratoga': 'Premium',
    # Major
    'del mar': 'Major', 'gulfstream': 'Major', 'gulfstream park': 'Major',
    'oaklawn': 'Major', 'oaklawn park': 'Major',
    'aqueduct': 'Major', 'pimlico': 'Major',
    'monmouth': 'Major', 'monmouth park': 'Major',
    'lone star': 'Major', 'lone star park': 'Major',
    'fair grounds': 'Major', 'fair grounds race course': 'Major',
    'woodbine': 'Major',
    'los alamitos': 'Major',
}

# ── Feature columns expected by the UK base model ────────────────────────────
UK_FEATURE_COLS = [
    'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
    'cd_runs', 'cd_win_rate', 'class_num', 'class_step', 'or_numeric',
    'or_change', 'or_trend_3', 'avg_last_3_pos', 'wins_last_3', 'days_since_last',
    'field_size', 'is_turf', 'going_numeric', 'race_score', 'draw', 'draw_pct',
    'draw_group_win_rate', 'weight_lbs', 'weight_vs_avg', 'is_top_weight',
    'weight_change', 'age', 'is_peak_age', 'is_3yo', 'is_veteran', 'age_vs_avg',
    'avg_btn_last_3', 'unlucky_last', 'has_blinkers', 'has_visor',
    'first_time_blinkers', 'gear_changed', 'is_handicap', 'is_maiden', 'is_pattern',
    'prize_log', 'is_sprint', 'is_mile', 'is_middle', 'is_staying',
    'jockey_career_runs', 'jockey_course_runs', 'jockey_trainer_runs',
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_country_suffix(name: str) -> str:
    """Remove trailing country codes like (IRE), (USA), (FR)."""
    if not name or not isinstance(name, str):
        return ''
    return re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', name).strip()


def _score_us_race(race: dict, furlongs: float | None) -> float:
    """Score a US race 0-100 for predictability / quality."""
    score = 0.0
    course_name = (race.get('course') or '').lower()
    tier = US_COURSE_TIERS.get(course_name)

    if tier == 'Premium':
        score += 15
    elif tier == 'Major':
        score += 5

    race_class = race.get('race_class') or race.get('class') or ''
    if re.search(r'grade\s*(i|1)', race_class, re.I) and not re.search(r'grade\s*(ii|iii|2|3)', race_class, re.I):
        score += 25
    elif re.search(r'grade\s*(ii|2)', race_class, re.I) and not re.search(r'grade\s*iii', race_class, re.I):
        score += 20
    elif re.search(r'grade\s*(iii|3)', race_class, re.I):
        score += 15
    elif re.search(r'listed', race_class, re.I):
        score += 12
    elif re.search(r'allowance', race_class, re.I):
        score += 8
    elif re.search(r'stakes', race_class, re.I):
        score += 10

    if furlongs is not None:
        if 8.0 <= furlongs <= 10.0:
            score += 10
        elif 6.0 <= furlongs <= 7.5:
            score += 5

    field_size = len(race.get('runners') or [])
    if 8 <= field_size <= 12:
        score += 10
    elif field_size > 16:
        score -= 5

    surface = (race.get('surface') or '').lower()
    if surface == 'dirt':
        score += 5

    return min(score, 100.0)


def _parse_prize(prize_raw) -> float:
    """Convert prize string/number to float (USD)."""
    if prize_raw is None:
        return 0.0
    s = str(prize_raw).replace('$', '').replace(',', '').replace(' ', '')
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_weight(weight_raw) -> float:
    """Convert weight string (e.g., '126', '9-0') to pounds."""
    if weight_raw is None:
        return 0.0
    s = str(weight_raw).strip().lower().replace('lbs', '').replace('lb', '').strip()
    # stone-pounds format "9-0"
    m = re.match(r'^(\d+)-(\d+)$', s)
    if m:
        return int(m.group(1)) * 14 + int(m.group(2))
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_age(age_raw, age_sex_raw='') -> int | None:
    """Extract a horse age from either a dedicated age field or NYRA's age/sex string."""
    for candidate in (age_raw, age_sex_raw):
        if candidate is None:
            continue
        match = re.search(r'(\d+)', str(candidate))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _fractional_odds_to_probability(odds_raw: str | None) -> float | None:
    """Convert fractional odds like '3/1' or '9/5' into implied probability."""
    if not odds_raw:
        return None
    s = str(odds_raw).strip().lower()
    if not s:
        return None
    if s in {'evens', 'even', '1/1', '1-1'}:
        return 0.5
    match = re.match(r'^(\d+)\s*[/\-]\s*(\d+)$', s)
    if not match:
        return None
    num = int(match.group(1))
    den = int(match.group(2))
    if num <= 0 or den <= 0:
        return None
    return den / (num + den)


# ── Feature building ──────────────────────────────────────────────────────────

def _build_race_context_features(race: dict, furlongs: float | None) -> dict:
    """Return race-level features from the racecard dict."""
    surface_raw = race.get('surface') or ''
    going_raw = race.get('going') or ''
    race_class_raw = race.get('race_class') or race.get('class') or ''

    is_turf = 1 if 'turf' in surface_raw.lower() else 0
    going_numeric = map_us_going_to_numeric(going_raw)
    class_num = map_us_class_to_numeric(race_class_raw)
    race_score = _score_us_race(race, furlongs)

    dist_f = furlongs or 8.0
    is_sprint  = 1 if dist_f < 7 else 0
    is_mile    = 1 if 7 <= dist_f < 9 else 0
    is_middle  = 1 if 9 <= dist_f < 12 else 0
    is_staying = 1 if dist_f >= 12 else 0

    prize_val = _parse_prize(race.get('prize'))
    prize_log = float(np.log1p(prize_val)) if prize_val >= 0 else 9.0

    race_type_lower = (race.get('type') or '').lower()
    is_handicap = 1 if 'handicap' in race_type_lower or 'claiming' in race_type_lower else 0
    is_maiden   = 1 if 'maiden' in race_type_lower or 'maiden' in race_class_raw.lower() else 0
    is_pattern  = 1 if re.search(r'grade\s*[123i]', race_class_raw, re.I) else 0

    field_size = len(race.get('runners') or [])

    return dict(
        field_size=field_size,
        is_turf=is_turf,
        going_numeric=going_numeric,
        race_score=race_score,
        class_num=class_num,
        is_sprint=is_sprint, is_mile=is_mile, is_middle=is_middle, is_staying=is_staying,
        prize_log=prize_log,
        is_handicap=is_handicap, is_maiden=is_maiden, is_pattern=is_pattern,
    )


def _build_horse_features(runner: dict, race_ctx: dict,
                           horse_history: pd.DataFrame,
                           all_weights: list[float]) -> dict:
    """Build the full 47-feature vector for one horse."""
    features = {col: 0 for col in UK_FEATURE_COLS}

    # ── Race-level context ────────────────────────────────────────────────────
    features.update(race_ctx)

    # ── Historical career stats ───────────────────────────────────────────────
    if len(horse_history) > 0:
        hist = horse_history.copy()
        hist['pos_num'] = pd.to_numeric(hist.get('pos', hist.get('position', None)), errors='coerce')
        hist['btn_num'] = pd.to_numeric(hist.get('btn', 0), errors='coerce').fillna(0)
        hist['or_num']  = pd.to_numeric(hist.get('or_numeric', hist.get('or', 0)), errors='coerce').fillna(0)

        n = len(hist)
        wins   = (hist['pos_num'] == 1).sum()
        places = (hist['pos_num'] <= 3).sum()

        features['career_runs']       = n
        features['career_win_rate']   = wins / n
        features['career_place_rate'] = places / n
        features['career_earnings']   = float(hist.get('prize_clean', pd.Series([0]*n)).sum())

        last3 = hist.head(3)
        features['avg_last_3_pos'] = float(last3['pos_num'].mean()) if len(last3) > 0 else 8.0
        features['wins_last_3']    = int((last3['pos_num'] == 1).sum())
        features['avg_btn_last_3'] = float(last3['btn_num'].mean())

        most_recent = hist.iloc[0]
        if n > 1:
            features['or_change'] = float(most_recent['or_num']) - float(hist.iloc[1]['or_num'])
        features['or_trend_3'] = float(last3['or_num'].mean()) if len(last3) > 0 else 0.0

        if 'date_dt' in hist.columns:
            features['days_since_last'] = (pd.Timestamp.now() - pd.to_datetime(hist['date_dt'].iloc[0])).days
        elif 'date' in hist.columns:
            try:
                features['days_since_last'] = (pd.Timestamp.now() - pd.to_datetime(hist['date'].iloc[0])).days
            except Exception:
                features['days_since_last'] = 30

        # Class step
        if 'class_num' in hist.columns:
            prev_class = float(hist['class_num'].iloc[0]) if pd.notna(hist['class_num'].iloc[0]) else features['class_num']
            features['class_step'] = features['class_num'] - prev_class

        # Unlucky flag
        features['unlucky_last'] = 0

    else:
        # No history — estimate from OR
        or_val = 0.0
        or_raw = runner.get('ofr') or runner.get('or') or runner.get('official_rating')
        if or_raw and str(or_raw) not in ('', '-', 'None'):
            try:
                or_val = float(or_raw)
            except ValueError:
                pass

        base_win = max(0.02, min(0.30, (or_val - 70) / 280)) if or_val > 0 else 0.08
        features['career_runs']       = 10
        features['career_win_rate']   = base_win
        features['career_place_rate'] = min(0.65, base_win * 3.0)
        features['career_earnings']   = base_win * 10 * 5000
        features['avg_last_3_pos']    = 6.0
        features['days_since_last']   = 30

    # ── Runner-specific features ──────────────────────────────────────────────

    # OR
    or_raw = runner.get('ofr') or runner.get('or') or runner.get('official_rating')
    if or_raw and str(or_raw) not in ('', '-', 'None'):
        try:
            features['or_numeric'] = float(or_raw)
        except ValueError:
            pass

    # Age
    age_raw = _extract_age(runner.get('age'), runner.get('age_sex'))
    if age_raw is not None:
        try:
            age = int(age_raw)
            features['age']         = age
            features['is_peak_age'] = 1 if 3 <= age <= 5 else 0
            features['is_3yo']      = 1 if age == 3 else 0
            features['is_veteran']  = 1 if age >= 7 else 0
        except (ValueError, TypeError):
            features['age'] = 4

    # Weight
    weight_lbs = _parse_weight(runner.get('weight') or runner.get('lbs'))
    features['weight_lbs'] = weight_lbs
    if all_weights:
        avg_w = float(np.mean(all_weights))
        features['weight_vs_avg'] = weight_lbs - avg_w
        features['is_top_weight'] = 1 if weight_lbs == max(all_weights) else 0
        features['age_vs_avg']    = features['age'] - 4  # placeholder

    # Draw
    draw_raw = runner.get('draw') or runner.get('stall') or runner.get('number')
    if draw_raw and str(draw_raw) not in ('', '-', 'None'):
        try:
            features['draw'] = int(draw_raw)
        except (ValueError, TypeError):
            pass

    # Equipment (blinkers / visor)
    headgear = (runner.get('headgear') or runner.get('equipment') or runner.get('equip') or '').lower()
    features['has_blinkers'] = 1 if 'blinker' in headgear else 0
    features['has_visor']    = 1 if 'visor' in headgear else 0

    # Jockey placeholder stats (no US jockey DB yet)
    features['jockey_career_runs'] = 200
    features['jockey_course_runs'] = 20
    features['jockey_trainer_runs'] = 30

    # draw_pct / draw_group_win_rate — not calculable without track data
    features['draw_pct']           = 0.5
    features['draw_group_win_rate'] = 0.1
    features['weight_change']       = 0.0
    features['first_time_blinkers'] = 0
    features['gear_changed']        = 0

    return features


def _calibrate_probability(raw: float, field_size: int, k: float = 3.5) -> float:
    """Shrinkage calibration: pull raw probs toward prior = 1/field_size."""
    prior = 1.0 / max(field_size, 1)
    return (raw + prior * k) / (1 + k)


# ── Core prediction ───────────────────────────────────────────────────────────

def predict_us_race(race: dict, historical_df: pd.DataFrame,
                    model, feature_cols: list[str],
                    prediction_date: str) -> list[dict]:
    """Return prediction dicts for all runners in a single US race."""
    runners = race.get('runners') or []
    if not runners:
        return []

    # Parse distance
    dist_raw = race.get('distance') or race.get('distance_f') or ''
    furlongs = parse_us_distance(str(dist_raw)) if dist_raw else None

    # Race-level context features (same for all horses)
    race_ctx = _build_race_context_features(race, furlongs)
    field_size = race_ctx['field_size'] or len(runners)

    # Gather all weights for relative features
    all_weights = []
    for r in runners:
        w = _parse_weight(r.get('weight') or r.get('lbs'))
        if w > 0:
            all_weights.append(w)

    raw_preds = []

    for runner in runners:
        horse_name = runner.get('name') or runner.get('horse') or 'Unknown'
        horse_clean = _strip_country_suffix(horse_name).lower()

        # Find historical records for this horse
        if len(historical_df) > 0 and 'horse_clean' in historical_df.columns:
            history = historical_df[historical_df['horse_clean'] == horse_clean].sort_values(
                'date_dt' if 'date_dt' in historical_df.columns else 'date', ascending=False
            )
        else:
            history = pd.DataFrame()

        feats = _build_horse_features(runner, race_ctx, history, all_weights)

        # Build feature matrix in model's expected order
        X = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0)
        X = X.fillna(0)

        try:
            raw_prob = float(model.predict_proba(X)[0, 1])
        except Exception:
            raw_prob = 1.0 / field_size

        raw_preds.append({
            'horse': horse_name,
            'jockey': runner.get('jockey') or '',
            'trainer': runner.get('trainer') or '',
            'ofr': runner.get('ofr') or runner.get('or') or '-',
            'age': _extract_age(runner.get('age'), runner.get('age_sex')) or '',
            'weight_lbs': _parse_weight(runner.get('weight') or runner.get('lbs')),
            'form': runner.get('form') or '',
            'draw': runner.get('draw') or runner.get('stall') or runner.get('number') or '',
            'ml_odds': runner.get('ml_odds') or '',
            'raw_win_probability': raw_prob,
        })

    market_probs = [_fractional_odds_to_probability(pred.get('ml_odds')) for pred in raw_preds]
    valid_market = [prob for prob in market_probs if prob is not None and prob > 0]
    raw_values = np.array([pred['raw_win_probability'] for pred in raw_preds], dtype=float)

    if valid_market:
        market_total = float(sum(valid_market))
        normalized_market = [
            (prob / market_total) if prob is not None and prob > 0 else None
            for prob in market_probs
        ]
        raw_spread = float(raw_values.max() - raw_values.min()) if len(raw_values) else 0.0
        use_market_fallback = raw_spread < 0.01

        for pred, market_prob in zip(raw_preds, normalized_market):
            if market_prob is None:
                continue
            if use_market_fallback:
                pred['raw_win_probability'] = market_prob
            else:
                pred['raw_win_probability'] = pred['raw_win_probability'] * 0.65 + market_prob * 0.35

    # ── Calibration ────────────────────────────────────────────────────────────
    results = []
    for pred in raw_preds:
        cal_win = _calibrate_probability(pred['raw_win_probability'], field_size)

        boost_place = cal_win * (1 - cal_win) * 0.8
        boost_show  = cal_win * (1 - cal_win) * 0.3
        place_prob  = min(max(cal_win + boost_place, cal_win), 0.98)
        show_prob   = min(max(place_prob + boost_show, place_prob), 0.99)

        pred.update({
            'win_probability':     cal_win,
            'win_odds_fractional': probability_to_fractional_odds(cal_win),
            'win_odds_decimal':    round(1.0 / cal_win, 2) if cal_win > 0 else 99.0,
            'place_probability':     place_prob,
            'place_odds_fractional': probability_to_fractional_odds(place_prob),
            'show_probability':      show_prob,
            'show_odds_fractional':  probability_to_fractional_odds(show_prob),
        })
        del pred['raw_win_probability']
        results.append(pred)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """Return (model, feature_cols, model_label, model_artifact)."""
    us_model_path = MODELS_DIR / 'us_horse_model.pkl'
    uk_model_path = MODELS_DIR / 'horse_win_predictor.pkl'
    feature_path  = MODELS_DIR / 'feature_columns.txt'

    model_path = us_model_path if us_model_path.exists() else uk_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")

    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)

    label = 'US' if model_path == us_model_path else 'UK (base)'
    print(f"[OK] Loaded {label} model from {model_path.name}")

    # Feature columns
    if feature_path.exists():
        with open(feature_path) as fh:
            cols = [line.strip() for line in fh if line.strip()]
        feature_cols = cols if cols else UK_FEATURE_COLS
    else:
        feature_cols = UK_FEATURE_COLS

    return model, feature_cols, label, model_path.name


def load_us_historical() -> pd.DataFrame:
    """Load US historical race data if available; else return empty frame."""
    candidates = [
        DATA_PROC / 'us_races_cleaned.parquet',
        DATA_PROC / 'all_us_races_cleaned.parquet',
        DATA_PROC / 'race_scores_connections_v2.parquet',  # UK fallback for shape
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"[OK] Loaded historical data from {path.name}  ({len(df):,} rows)")
            if 'horse_clean' not in df.columns:
                df['horse_clean'] = df['horse'].apply(
                    lambda x: _strip_country_suffix(str(x)).lower() if pd.notna(x) else ''
                )
            if 'date_dt' not in df.columns:
                df['date_dt'] = pd.to_datetime(df.get('date', pd.NaT), errors='coerce')
            return df

    print("[WARN] No US (or UK fallback) historical data found — using defaults for all horses")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for US races')
    parser.add_argument('--date', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='Race date YYYY-MM-DD (default: today)')
    args = parser.parse_args()
    date_str = args.date

    print(f"\n{'='*60}")
    print(f"  US RACE PREDICTIONS  —  {date_str}")
    print(f"{'='*60}\n")

    # ── Load model ──────────────────────────────────────────────────────────
    try:
        model, feature_cols, model_label, model_artifact = load_model()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # ── Load racecards ──────────────────────────────────────────────────────
    racecard_file = DATA_RAW / f'us_racecards_{date_str}.json'
    if not racecard_file.exists():
        nyra_file = DATA_RAW / f'nyra_entries_{date_str}.json'
        if nyra_file.exists():
            print(f"[INFO] us_racecards not found; using nyra_entries_{date_str}.json")
            racecard_file = nyra_file
        else:
            print(f"[ERROR] No US racecards found for {date_str}")
            print(f"        Run: python scripts/fetch_nyra_entries.py --date {date_str}")
            sys.exit(1)

    with open(racecard_file, encoding='utf-8') as fh:
        raw = json.load(fh)

    # Normalise to a flat list of race dicts
    races: list[dict] = []
    if isinstance(raw, dict):
        candidates_keys = ['racecards', 'races', 'results']
        for key in candidates_keys:
            if key in raw and isinstance(raw[key], list):
                races = raw[key]
                break
        else:
            # Deep flatten nested dict structure
            for v1 in raw.values():
                if isinstance(v1, list):
                    races.extend(v1)
                elif isinstance(v1, dict):
                    for v2 in v1.values():
                        if isinstance(v2, dict) and 'runners' in v2:
                            races.append(v2)
                        elif isinstance(v2, list):
                            races.extend(v2)
    elif isinstance(raw, list):
        races = raw

    if not races:
        print("[WARN] Racecard file contained no races — nothing to predict")
        sys.exit(0)

    print(f"[OK] Loaded {len(races)} US races\n")

    # ── Load historical data ────────────────────────────────────────────────
    historical_df = load_us_historical()

    # ── Generate predictions ────────────────────────────────────────────────
    all_predictions: list[dict] = []

    for idx, race in enumerate(races, 1):
        course     = race.get('course') or race.get('track_name') or race.get('track') or 'Unknown'
        race_time  = race.get('time') or race.get('off') or race.get('race_time') or ''
        race_name  = race.get('race_name') or race.get('name') or race.get('race') or ''
        race_class = race.get('race_class') or race.get('class') or ''
        surface    = race.get('surface') or ''
        going      = race.get('going') or ''
        dist_raw   = race.get('distance') or race.get('distance_f') or ''
        furlongs   = parse_us_distance(str(dist_raw)) if dist_raw else None
        runners_n  = len(race.get('runners') or [])

        print(f"[{idx:3d}/{len(races)}] {race_time:8s}  {course:25s}  "
              f"{surface:10s}  {dist_raw:10s}  ({runners_n} runners)")

        preds = predict_us_race(race, historical_df, model, feature_cols, date_str)

        for pred in preds:
            pred.update({
                'region':       'US',
                'date':         date_str,
                'source_model': model_label,
                'model_artifact': model_artifact,
                'model_feature_count': len(feature_cols),
                'course':       course,
                'race_time':    race_time,
                'race_name':    race_name,
                'race_class':   race_class,
                'surface':      surface,
                'going':        going,
                'distance_f':   furlongs,
                'distance_str': dist_raw,
                'distance_band': get_distance_band_us(furlongs),
            })

        all_predictions.extend(preds)

        # Top 3 preview
        top3 = sorted(preds, key=lambda x: x['win_probability'], reverse=True)[:3]
        for i, p in enumerate(top3, 1):
            print(f"         {i}. {p['horse']:28s}  Win: {p['win_probability']:.1%}")

    if not all_predictions:
        print("[WARN] No predictions generated")
        sys.exit(0)

    # ── Save output ─────────────────────────────────────────────────────────
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    output_file = DATA_PROC / f'us_predictions_{date_str}.csv'
    df_out = pd.DataFrame(all_predictions)
    df_out.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"  SAVED -> {output_file}")
    print(f"  Total horses: {len(df_out)}")
    print(f"  Win prob  min={df_out['win_probability'].min():.2%}  "
          f"max={df_out['win_probability'].max():.2%}  "
          f"mean={df_out['win_probability'].mean():.2%}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
