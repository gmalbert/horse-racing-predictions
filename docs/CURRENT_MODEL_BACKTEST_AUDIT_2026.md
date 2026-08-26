# Current-model backtest audit (2026)

## Verdict

The ensemble artifact reports 14,210 evaluated races: AUC 0.6892, log loss 0.3278, Brier 0.0946, top-1 accuracy 25.16%, and top-3 accuracy 57.09%. It improves substantially over the saved solo XGBoost calibration metrics, while a separate US artifact reports test AUC 0.6105. These are encouraging ranking/calibration diagnostics, but the files do not establish a gap-separated walk-forward last-year test or a settled market-odds ledger. Calibration gains may be optimistic if calibrator selection shares time with evaluation.

## Changes justified by the result

1. Preserve the calibrated ensemble as challenger, but validate with race-date walk-forward splits and a gap; keep all horses from one race in one fold.
2. Normalize probabilities to sum to one within each field and evaluate race-level log loss, Brier, top-k, and calibration by field size/odds band.
3. Separate UK/Irish and US models or use jurisdiction hierarchy; the US AUC gap suggests domain shift.
4. Add market residual modeling and close-price benchmarking. AUC does not establish value after takeout/overround.

## Betting strategy decision

- **Win:** paper-trade overlays only after de-vigging the entire field.
- **Place/show/each-way:** derive joint order probabilities and apply bookmaker place/dead-heat terms.
- **Exacta/trifecta/multis:** require Plackett-Luce or Monte Carlo order simulation and pool takeout; no independence shortcuts.
- **Lay betting:** include commission and liability limits.
- **Staking:** capped fractional Kelly at race level only after validation; cap total exposure across mutually exclusive runners.

## Release gate

Gap-separated forward season, 5,000+ races, positive CLV and net ROI after takeout, calibration by field/price/jurisdiction, and bootstrap intervals clustered by race day.
