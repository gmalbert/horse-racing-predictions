# US Odds Coverage Audit

- Date: 2026-10-26
- Generated At (UTC): 2026-10-26T15:10:22.904516+00:00

## Source Coverage

- us_racecards: available=True, races=50, coverage=95.9%
- nyra_entries: available=True, races=0, coverage=0.0%
- oddsportal: available=False, races=0, coverage=0.0%

## Fallback Hierarchy

- 1) us_racecards_<date>.json (TVG/graphql ml_odds)
- 2) nyra_entries_<date>.json (NYRA ml_odds where available)
- 3) oddsportal_us_<date>.json (major stakes only)
- 4) last-known odds snapshot from previous day (stale flag required)
