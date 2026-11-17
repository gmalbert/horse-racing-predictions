# US Ingestion Validation Report

- Date: 2026-11-18
- Status: **PASS**
- Generated At (UTC): 2026-11-17T15:59:45.885461+00:00

## Metrics

- Total races: 91
- Total runners: 790
- Unique tracks: 10
- ML odds coverage: 98.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-18.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-18.json
- [PASS] min_total_races: 91 >= 8
- [PASS] min_total_runners: 790 >= 50
- [PASS] min_ml_odds_coverage: 98.0% >= 50.0%
- [PASS] coverage_drift_guard: today=91 vs median=88.0, min_allowed=44.0
- [PASS] harness_source_coverage: 25 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=91, racecards_total=91
