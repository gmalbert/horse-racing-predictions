# Quick Start - Model v2.0 (72 Features)

## ✅ What's Complete

- [x] Model retrained with 72 features (was 47)
- [x] ROC AUC improved to 0.702 (was 0.671)
- [x] Data leakage fixed in pedigree features
- [x] Prediction scripts updated with all new features
- [x] Test successful on 2025-12-28 (21 races, 182 horses)

## 🚀 Ready to Use

### Generate Predictions for Today
```bash
python scripts/predict_todays_races.py
```

### Generate Predictions for Specific Date
```bash
python scripts/predict_todays_races.py --date 2025-01-15
```

### Batch Generate All Missing Predictions
```bash
python scripts/batch_generate_predictions.py
```

## 📊 What Changed

### New Features (25 total)

**Pedigree (6)**:
- Sire win/place rates
- Surface, class, distance-specific sire performance

**Pace (9)**:
- Running style classification (LEADER/PRESSER/CLOSER/MIDPACK)
- Distance specialists (sprint/staying)
- Race pace scenarios (leader count, closer count, style advantage)

**Recent Form (10)**:
- Jockey 14d/30d win rates
- Trainer 14d/30d win rates
- "Hot" connection flags (>25% recent win rate)

### Model Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Features | 47 | 72 | +25 |
| ROC AUC | 0.671 | 0.702 | +0.031 |
| Data Leakage | Yes | No | Fixed |

**Top 5 Features**:
1. field_size (0.0805)
2. **sprint_specialist** (0.0586) ← NEW
3. **staying_specialist** (0.0502) ← NEW
4. avg_last_3_pos (0.0297)
5. is_top_weight (0.0262)

## 📁 Output Format

### Predictions CSV Columns (59 total)

**Race Info**: course, race_time, race_name, race_class, distance_f, surface, going, field_size

**Horse Info**: horse, jockey, trainer, age, weight_lbs, ofr, last_run, form

**Predictions**: 
- win_probability, place_probability, show_probability
- win_odds_decimal, win_odds_fractional
- place_odds_decimal, place_odds_fractional
- show_odds_decimal, show_odds_fractional

**Features** (all 72 model features included for transparency):
- Career stats: career_runs, career_win_rate, career_place_rate, career_earnings
- Course/Distance: cd_runs, cd_win_rate
- Class: class_num, class_step
- Rating: or_numeric, or_change, or_trend_3
- Recent: avg_last_3_pos, wins_last_3, days_since_last
- Jockey: jockey_career_runs, jockey_career_win_rate, etc.
- **NEW**: race_leader_count, race_closer_count, style_advantage, age_vs_avg

## 🎯 Next Steps (Optional)

### Update Streamlit UI
The UI (`predictions.py`) currently shows old model predictions. To update:
1. Regenerate predictions for recent dates
2. UI will automatically load new CSV files
3. Consider adding feature displays (jockey form, pace scenario)

### Backtest Performance
Test model on historical races to measure real-world accuracy:
```bash
# Generate predictions for Oct-Dec 2024
python scripts/batch_generate_predictions.py --start-date 2024-10-01 --end-date 2024-12-31

# Then manually compare to actual results
```

### Review Top Predictions
```python
import pandas as pd

df = pd.read_csv('data/processed/predictions_2025-12-28.csv')

# Highest win probabilities
print(df.nlargest(10, 'win_probability')[['horse', 'course', 'race_time', 'win_probability']])

# Jockeys in form
print(df[df['jockey_form_30d'] > 0.25][['horse', 'jockey', 'jockey_form_30d']])

# Pace advantages
print(df[df['style_advantage'] == 1][['horse', 'course', 'pace_style_leader', 'race_leader_count']])
```

## 📖 Documentation

- **PREDICTION_UPDATES_COMPLETE.md** - Full technical documentation
- **MODEL_RETRAINING_COMPLETE.md** - Model training details
- **DATA_LEAKAGE_AUDIT.md** - Data leakage fixes

## ⚠️ Important Notes

1. **Historical Data Required**: Script loads 245K races from `data/processed/race_scores_with_all_features_no_leakage.parquet`
2. **Temporal Integrity**: All features use proper date filtering (no future data leakage)
3. **Coverage**: Some features require minimum data (e.g., 5+ races for pace classification)
4. **Defaults**: Missing data handled gracefully with sensible defaults

## 🐛 Troubleshooting

**"Model feature count mismatch"**
→ Ensure using latest model: `models/horse_win_predictor.json` (72 features)

**"Low predictions for all horses"**
→ Check historical data loaded: Should see "Loaded 245,298 historical records"

**"All pace features are 0"**
→ Expected for horses with <5 career races

**"Jockey/trainer form all zeros"**
→ Expected if no rides/runners in last 30 days

## 📞 Support

For issues or questions:
1. Check **PREDICTION_UPDATES_COMPLETE.md** for technical details
2. Review console output for error messages
3. Verify file paths and data availability

---

**Status**: ✅ Production Ready  
**Version**: Model 2.0 (72 features)  
**Last Updated**: January 2025
