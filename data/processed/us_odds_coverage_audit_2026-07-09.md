# US Odds Coverage Audit

- Date: 2026-07-09
- Generated At (UTC): 2026-07-09T07:26:19.049984+00:00

## Source Coverage

- us_racecards: available=True, races=0, coverage=0.0%
- nyra_entries: available=True, races=0, coverage=0.0%
- oddsportal: available=False, races=0, coverage=0.0%

## Fallback Hierarchy

- 1) us_racecards_<date>.json (TVG/graphql ml_odds)
- 2) nyra_entries_<date>.json (NYRA ml_odds where available)
- 3) oddsportal_us_<date>.json (major stakes only)
- 4) last-known odds snapshot from previous day (stale flag required)

## Alerts

- Primary source coverage low: 0.0% < 50.0%.
