# US Ingestion Validation Report

- Date: 2026-11-10
- Status: **PASS**
- Generated At (UTC): 2026-11-09T16:15:19.736549+00:00

## Metrics

- Total races: 63
- Total runners: 520
- Unique tracks: 6
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-11-10.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-11-10.json
- [PASS] min_total_races: 63 >= 8
- [PASS] min_total_runners: 520 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=63 vs median=94.5, min_allowed=47.2
- [PASS] harness_source_coverage: 36 harness races; collection failures: none
- [PASS] source_report_total_matches_racecards: report_total=63, racecards_total=63
