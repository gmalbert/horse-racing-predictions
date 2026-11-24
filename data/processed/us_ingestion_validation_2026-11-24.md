# US Ingestion Validation Report

- Date: 2026-11-24
- Status: **PASS**
- Generated At (UTC): 2026-11-24T17:40:37.721545+00:00

## Metrics

- Total races: 81
- Total runners: 674
- Unique tracks: 8
- ML odds coverage: 96.7%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-24.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-24.json
- [PASS] min_total_races: 81 >= 8
- [PASS] min_total_runners: 674 >= 50
- [PASS] min_ml_odds_coverage: 96.7% >= 50.0%
- [PASS] coverage_drift_guard: today=81 vs median=86.5, min_allowed=43.2
- [PASS] harness_source_coverage: 39 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=81, racecards_total=81
