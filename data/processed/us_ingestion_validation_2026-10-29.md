# US Ingestion Validation Report

- Date: 2026-10-29
- Status: **PASS**
- Generated At (UTC): 2026-10-28T12:51:52.477157+00:00

## Metrics

- Total races: 102
- Total runners: 867
- Unique tracks: 11
- ML odds coverage: 97.8%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-10-29.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-10-29.json
- [PASS] min_total_races: 102 >= 8
- [PASS] min_total_runners: 867 >= 50
- [PASS] min_ml_odds_coverage: 97.8% >= 50.0%
- [PASS] coverage_drift_guard: today=102 vs median=52.0, min_allowed=26.0
- [PASS] harness_source_coverage: 46 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=102, racecards_total=102
