# US Ingestion Validation Report

- Date: 2026-11-01
- Status: **PASS**
- Generated At (UTC): 2026-11-01T13:23:33.631159+00:00

## Metrics

- Total races: 82
- Total runners: 764
- Unique tracks: 8
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-01.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-01.json
- [PASS] min_total_races: 82 >= 8
- [PASS] min_total_runners: 764 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=82 vs median=98.5, min_allowed=49.2
- [PASS] harness_source_coverage: 15 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=82, racecards_total=82
