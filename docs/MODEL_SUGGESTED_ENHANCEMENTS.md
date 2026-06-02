# Horse Racing Predictions — Model Suggested Enhancements

## Priority 1: Improve Ensemble AUC (Current: 0.6892)

### Pedigree Features
- Sire win rate on each going type (e.g., "Son of Galileo on Soft: 31% win rate"). Dam's sire contribution.
- Going-specific sire win rate is already partially implemented; ensure it's in the full feature set.

### Fractional Times / Pace Features
- Early pace (time to first furlong) predicts which horses will burn out. Add `pace_first_2f_secs` where available.
- "Pace pressure" flag: number of front-runners in the field pushes `exp_pace_collapse = True`.

### Trainer Form Refinement
- `trainer_win_pct_l14` (14-day rolling, per trainer) captures training camp hot streaks better than long-term average.
- Add `trainer_course_win_pct` (career win % at this specific course).

### Class Drop / Rise
- `class_change` (positive = drop in class, negative = rise). Class drops are among the strongest short-price betting signals.

## Priority 2: Going & Distance Interaction

### Going × Distance Matrix
- A horse that wins on Soft at 1 mile may be unproven on Good at 1.5 miles. Add a `going_distance_win_pct` interaction feature.

### Going Preference Score
- `going_pref` = (wins on preferred going / total starts on preferred going) vs. (wins on other going / starts on other going). Binary: `is_on_preferred_going`.

## Priority 3: Jockey Features

### Jockey × Trainer Partnership
- Win % when the specific jockey–trainer pair team up. Top partnerships outperform their individual stats.

### Jockey Course Win %
- Some jockeys are specialists at specific tracks. Add `jockey_course_win_pct` alongside existing jockey features.

## Priority 4: Calibration & Bankroll

- Run Brier score tracking on historical predictions.
- Implement fractional Kelly sizing (0.25 Kelly) as the default bet size in the bankroll manager.
- Add CLV tracking: compare model's implied odds to closing SP (starting price).
