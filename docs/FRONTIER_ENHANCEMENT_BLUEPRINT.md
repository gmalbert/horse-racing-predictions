# Frontier Enhancement Blueprint

The repository already has unusually deep documentation on leakage, calibration, architecture, US/UK sources, odds, betting, feature engineering, and scraping. The proposals below target microstructure, race-shape simulation, and decision-aware exotic construction without repeating that work.

## Sectional race-shape model

Represent each horse's latent pace by race segment and running-style policy. Simulate interaction: multiple leaders change the pace distribution for every runner, so independent finish models are structurally wrong.

```python
def simulate_race(horses, segments, rng):
    state = initialize_positions(horses)
    for segment in segments:
        pressure = front_pressure(state)
        state = advance(state, segment, pressure, correlated_draws(horses, rng))
    return rank_finish(state)
```

Use GPS/sectionals where licensed and chart-call proxies elsewhere, with an explicit source-quality feature.

## Tote microstructure

Model probable final odds from pool trajectory, bet type, takeout, time to post, and late-money patterns. Evaluate recommendations against executable odds ranges, not the last displayed price. Track coupled pools and scratches/rule changes.

## Network and uncertainty features

- Horse-trainer-jockey-owner-breeder graph with time-bounded edges.
- Trainer intent/regime features (layoff pattern, class move, equipment/medication change) with strong shrinkage.
- Track-bias latent state updated only from races observed so far that day.
- Shipping, quarantine, surface switch, and weather interactions.
- Prediction sets/abstention for maiden, lightly raced, foreign, and low-coverage runners.

## Product additions

- Race-shape animation and pace-pressure scenarios.
- Odds-range recommendation (“bet only at 6.0+”) with pool-latency warning.
- Exotic-ticket optimizer maximizing utility under budget and correlated outcomes.
- Scratch/recompute audit trail.
- Evidence and source-quality drawer for every runner.

## Gates

Use strict race-time snapshots and track/day-forward splits. Report multiclass log loss, calibration by field size/odds, ranking metrics, final-odds forecast error, CLV, takeout-adjusted ROI, and drawdown. Simulate exotic tickets from the full joint finish distribution and compare against simple dutching/boxing baselines. Audit every feature for post-time leakage.
