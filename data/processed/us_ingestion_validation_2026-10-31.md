# US Ingestion Validation Report

- Date: 2026-10-31
- Status: **PASS**
- Generated At (UTC): 2026-10-30T12:56:39.783971+00:00

## Metrics

- Total races: 133
- Total runners: 1186
- Unique tracks: 13
- ML odds coverage: 100.0%

## Checks

- [PASS] racecards_file_exists: data/raw/us_racecards_2026-10-31.json
- [PASS] source_report_exists: data/raw/us_source_report_2026-10-31.json
- [PASS] min_total_races: 133 >= 8
- [PASS] min_total_runners: 1186 >= 50
- [PASS] min_ml_odds_coverage: 100.0% >= 50.0%
- [PASS] coverage_drift_guard: today=133 vs median=95.0, min_allowed=47.5
- [PASS] harness_source_coverage: 25 harness races; collection failures: SCD
- [PASS] source_report_total_matches_racecards: report_total=133, racecards_total=133
