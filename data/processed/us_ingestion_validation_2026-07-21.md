# US Ingestion Validation Report

- Date: 2026-07-21
- Status: **PASS**
- Generated At (UTC): 2026-07-21T07:08:18.753222+00:00

## Metrics

- Total races: 0
- Total runners: 0
- Unique tracks: 0
- ML odds coverage: 0.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-07-21.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-07-21.json
- [PASS] min_total_races: 0 >= 8 (empty racecards allowed)
- [PASS] min_total_runners: 0 >= 50 (empty racecards allowed)
- [PASS] min_ml_odds_coverage: 0.0% >= 50.0% (empty racecards allowed)
- [PASS] coverage_drift_guard: Insufficient history for drift baseline; check skipped
- [PASS] source_report_total_matches_racecards: report_total=0, racecards_total=0
