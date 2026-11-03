# US Ingestion Validation Report

- Date: 2026-11-03
- Status: **PASS**
- Generated At (UTC): 2026-11-03T14:11:39.785945+00:00

## Metrics

- Total races: 64
- Total runners: 545
- Unique tracks: 6
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-03.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-03.json
- [PASS] min_total_races: 64 >= 8
- [PASS] min_total_runners: 545 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=64 vs median=88.5, min_allowed=44.2
- [PASS] harness_source_coverage: 37 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=64, racecards_total=64
