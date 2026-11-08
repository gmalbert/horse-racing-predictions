# US Ingestion Validation Report

- Date: 2026-11-09
- Status: **PASS**
- Generated At (UTC): 2026-11-08T13:24:38.489896+00:00

## Metrics

- Total races: 52
- Total runners: 454
- Unique tracks: 5
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-09.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-09.json
- [PASS] min_total_races: 52 >= 8
- [PASS] min_total_runners: 454 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=52 vs median=94.5, min_allowed=47.2
- [PASS] harness_source_coverage: 25 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=52, racecards_total=52
