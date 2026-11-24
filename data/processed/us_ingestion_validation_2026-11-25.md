# US Ingestion Validation Report

- Date: 2026-11-25
- Status: **PASS**
- Generated At (UTC): 2026-11-24T17:40:37.813099+00:00

## Metrics

- Total races: 83
- Total runners: 709
- Unique tracks: 9
- ML odds coverage: 97.7%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-25.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-25.json
- [PASS] min_total_races: 83 >= 8
- [PASS] min_total_runners: 709 >= 50
- [PASS] min_ml_odds_coverage: 97.7% >= 50.0%
- [PASS] coverage_drift_guard: today=83 vs median=86.5, min_allowed=43.2
- [PASS] harness_source_coverage: 16 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=83, racecards_total=83
