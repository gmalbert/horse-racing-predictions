# Immediate Action Plan (Week 1-2)

This document tracks the current production-ready actions for the Horse Racing Predictions app.
It is intended to help developers and operators confirm the minimal must-do steps, status, and next
commands required to keep the prediction pipeline healthy.

## Current Status

- ✅ Core app navigation migrated to Streamlit `st.navigation()`
- ✅ Main UI updated to display enhanced prediction features and diagnostics
- ✅ Model insights tab includes calibration status, feature importance, and diagnostics
- ✅ `scripts/diagnose_model.py` is available for failure analysis
- ✅ `scripts/calibrate_model.py` is available for probability calibration
- ⚠️ Backtesting and production deployment are still pending

## Immediate Actions

1. Run model calibration

```bash
python scripts/calibrate_model.py
```

2. Generate predictions for a sample date

```bash
python scripts/predict_todays_races.py --date 2026-02-02
```

3. View results in the Streamlit UI

```bash
streamlit run predictions.py
```

4. If diagnostics are needed, run the model diagnosis script

```bash
python scripts/diagnose_model.py
```

## Key Focus Areas

- Calibration should produce `models/calibration_metrics.json` and optionally `models/horse_win_predictor_calibrated.pkl`
- Predictions should generate `data/processed/predictions_YYYY-MM-DD.csv`
- The UI should surface model calibration metrics and explain top picks
- Track any feature mismatch issues between the trained model and current data schema

## Known Limitations

- Model retraining remains the authoritative fix if feature coverage changes
- Backtesting on historical races is still required to confirm real-world impact
- Production deployment is not yet completed

## Next Steps

- Run a focused backtest on the latest available historic data
- Audit the pipeline end-to-end after any feature or data schema changes
- Document any remaining issues in `docs/OUTSTANDING_FEATURES.md`

## Notes

This file was repaired and simplified to reflect the current repository state.
The previous content was corrupted and has been replaced with a concise immediate plan.
