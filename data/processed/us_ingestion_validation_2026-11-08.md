# US Ingestion Validation Report

- Date: 2026-11-08
- Status: **PASS**
- Generated At (UTC): 2026-11-08T13:24:38.404783+00:00

## Metrics

- Total races: 94
- Total runners: 815
- Unique tracks: 9
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-08.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-08.json
- [PASS] min_total_races: 94 >= 8
- [PASS] min_total_runners: 815 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=94 vs median=95.0, min_allowed=47.5
- [PASS] harness_source_coverage: 30 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=94, racecards_total=94
