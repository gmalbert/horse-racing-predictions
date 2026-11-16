# US Ingestion Validation Report

- Date: 2026-11-16
- Status: **PASS**
- Generated At (UTC): 2026-11-16T15:40:23.766686+00:00

## Metrics

- Total races: 76
- Total runners: 639
- Unique tracks: 7
- ML odds coverage: 97.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-16.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-16.json
- [PASS] min_total_races: 76 >= 8
- [PASS] min_total_runners: 639 >= 50
- [PASS] min_ml_odds_coverage: 97.0% >= 50.0%
- [PASS] coverage_drift_guard: today=76 vs median=88.0, min_allowed=44.0
- [PASS] harness_source_coverage: 49 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=76, racecards_total=76
