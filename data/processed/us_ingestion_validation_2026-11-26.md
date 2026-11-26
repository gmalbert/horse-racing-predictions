# US Ingestion Validation Report

- Date: 2026-11-26
- Status: **PARTIAL**
- Generated At (UTC): 2026-11-26T16:57:49.884536+00:00

## Metrics

- Total races: 39
- Total runners: 298
- Unique tracks: 4
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-26.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-26.json
- [PASS] min_total_races: 39 >= 8
- [PASS] min_total_runners: 298 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [FAIL] coverage_drift_guard: today=39 vs median=86.5, min_allowed=43.2
- [PASS] harness_source_coverage: 0 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=39, racecards_total=39

## Alerts

- Coverage drift detected: race volume significantly below recent baseline.
