# US Ingestion Validation Report

- Date: 2026-11-23
- Status: **PASS**
- Generated At (UTC): 2026-11-23T18:10:56.022794+00:00

## Metrics

- Total races: 68
- Total runners: 522
- Unique tracks: 7
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-23.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-23.json
- [PASS] min_total_races: 68 >= 8
- [PASS] min_total_runners: 522 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=68 vs median=86.5, min_allowed=43.2
- [PASS] harness_source_coverage: 41 harness races; collection failures: MDL
- [PASS] source_report_total_matches_racecards: report_total=68, racecards_total=68
