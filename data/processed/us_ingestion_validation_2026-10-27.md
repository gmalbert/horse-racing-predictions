# US Ingestion Validation Report

- Date: 2026-10-27
- Status: **PASS**
- Generated At (UTC): 2026-10-26T15:10:22.839923+00:00

## Metrics

- Total races: 52
- Total runners: 437
- Unique tracks: 5
- ML odds coverage: 97.3%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-10-27.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-10-27.json
- [PASS] min_total_races: 52 >= 8
- [PASS] min_total_runners: 437 >= 50
- [PASS] min_ml_odds_coverage: 97.3% >= 50.0%
- [PASS] coverage_drift_guard: today=52 vs median=50.0, min_allowed=25.0
- [PASS] harness_source_coverage: 35 harness races; collection failures: SCD
- [PASS] source_report_total_matches_racecards: report_total=52, racecards_total=52
