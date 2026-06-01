# Horse Racing Predictions — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Live Racecard Auto-Refresh (GitHub Actions)

**Why:** Today's racecards must be fetched manually via `scripts/fetch_racecards.py`. Adding a GitHub Actions workflow that automatically fetches racecards and generates predictions each morning would make the app self-updating without manual intervention.

**How:**
1. Create `.github/workflows/daily_predictions.yml` triggered at `07:00 UTC` on weekdays
2. Steps: checkout → activate venv → run `scripts/fetch_racecards.py --date {today}` → run `scripts/predict_todays_races.py` → commit updated `data/processed/predictions_{today}.csv` to main
3. Use the existing workflow in `.github/workflows/` as a template (model training already runs weekly)
4. Add a "Last updated" timestamp to the Streamlit sidebar sourced from the predictions file `mtime`

**Complexity:** Low

---

## Feature 2: Jockey Form Feature

**Why:** Jockey win percentage is one of the top-3 most predictive features in horse racing. A jockey on a hot streak (10%+ win rate over last 30 days) provides meaningful signal beyond the horse's historical performance. The Racing API provides jockey-level stats.

**How:**
1. In `scripts/phase2_score_races.py`, fetch jockey recent form using The Racing API's jockey endpoint
2. Compute: `jockey_win_pct_l30d` (last 30 days), `jockey_win_pct_going_l30d` (last 30 days on this going type)
3. Add to the feature vector in `scripts/phase3_build_horse_model.py`
4. Use `shift(1)` — only include jockey form from races prior to the current one

**Complexity:** Low

---

## Feature 3: Class Rise/Drop Feature

**Why:** Horses that drop in class (from Group 1 to Group 3, or from Class 1 to Class 3) frequently outperform market expectations. The current feature set includes race class but likely does not encode the delta from the horse's last race.

**How:**
1. In `scripts/build_engineered_dataset.py`, sort each horse's race history by date
2. Compute `class_delta = current_race_class_numeric − last_race_class_numeric` (negative = class drop = favorable)
3. Add `class_delta` and `is_class_drop` binary to the feature vector
4. Verify leakage-free: class of current race is known before race, class of last race is from prior race

**Complexity:** Low

---

## Feature 4: Post Position / Gate Draw Bias by Track

**Why:** Certain tracks (Chester, tight/flat circuits) heavily favor low draw numbers; others (Ascot, wide/galloping) are more neutral. Historical win rate by starting position at each racecourse is a track-specific signal that can add meaningful edge.

**How:**
1. From historical Parquet data, compute `win_pct_by_gate_by_track`: for each track, the win percentage per starting stall (positions 1–20)
2. Create a `post_position_bias` feature: the historical win rate at this gate at this track (rolling 5-year window)
3. Add to the feature vector — most impactful for turf races at tight tracks
4. Validate: SHAP importance should be elevated for tracks like Chester vs flat, straight Newmarket

**Complexity:** Medium

---

## Feature 5: Exacta / Multi-Race Parlay Builder

**Why:** Single-race win predictions are the foundation, but exacta (1-2 finish prediction) and multi-race parlays offer higher returns for users who trust the model. The underlying win probabilities are already computed — this is a UI and computation layer.

**How:**
1. In `predictions.py`, add a "Parlay Builder" tab
2. For each race, compute exacta probability: `P(A wins) × P(B wins | A wins)` using the conditional renormalization method (same as the horse racing instructions document describes)
3. Allow users to select 2–3 races and the top-2 predicted finishers per race → compute combined parlay probability
4. Display implied fair odds vs available market odds (if odds integration from Feature set is live)
5. Apply Kelly sizing: `kelly_fraction = edge / (implied_odds - 1)` (half-Kelly)

**Complexity:** Low
