# US Ingestion Validation Report

- Date: 2026-11-11
- Status: **PASS**
- Generated At (UTC): 2026-11-10T14:51:19.441552+00:00

## Metrics

- Total races: 81
- Total runners: 694
- Unique tracks: 9
- ML odds coverage: 97.7%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-11.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-11.json
- [PASS] min_total_races: 81 >= 8
- [PASS] min_total_runners: 694 >= 50
- [PASS] min_ml_odds_coverage: 97.7% >= 50.0%
- [PASS] coverage_drift_guard: today=81 vs median=94.5, min_allowed=47.2
- [PASS] harness_source_coverage: 25 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=81, racecards_total=81
