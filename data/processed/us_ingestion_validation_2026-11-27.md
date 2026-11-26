# US Ingestion Validation Report

- Date: 2026-11-27
- Status: **PASS**
- Generated At (UTC): 2026-11-26T16:57:49.970770+00:00

## Metrics

- Total races: 141
- Total runners: 1249
- Unique tracks: 14
- ML odds coverage: 99.9%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-27.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-27.json
- [PASS] min_total_races: 141 >= 8
- [PASS] min_total_runners: 1249 >= 50
- [PASS] min_ml_odds_coverage: 99.9% >= 50.0%
- [PASS] coverage_drift_guard: today=141 vs median=81.5, min_allowed=40.8
- [PASS] harness_source_coverage: 29 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=141, racecards_total=141
