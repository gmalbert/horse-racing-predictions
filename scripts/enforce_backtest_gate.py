#!/usr/bin/env python3
"""Enforce quality gates from walk-forward backtest output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def _mean(values):
    return statistics.mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce backtest quality gate")
    parser.add_argument(
        "--results-file",
        default="data/processed/backtest_results.json",
        help="Path to backtest results JSON",
    )
    parser.add_argument("--min-folds", type=int, default=4)
    parser.add_argument("--min-mean-auc", type=float, default=0.58)
    parser.add_argument("--min-mean-top1", type=float, default=0.10)
    parser.add_argument("--max-mean-level-drawdown", type=float, default=500.0)
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"[FAIL] Missing backtest results file: {results_path}")
        return 1

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    folds = payload.get("folds") or []

    print("=" * 60)
    print("BACKTEST QUALITY GATE")
    print("=" * 60)
    print(f"Results file: {results_path}")
    print(f"Fold count:   {len(folds)}")

    if len(folds) < args.min_folds:
        print(f"[FAIL] Fold count {len(folds)} is below minimum {args.min_folds}")
        return 1

    aucs = [float(f.get("auc", 0.0)) for f in folds]
    top1 = [float(f.get("top1_accuracy", 0.0)) for f in folds]
    drawdowns = [float(f.get("level_max_dd", 0.0)) for f in folds]

    mean_auc = _mean(aucs)
    mean_top1 = _mean(top1)
    mean_dd = _mean(drawdowns)

    print(f"Mean AUC:     {mean_auc:.4f}")
    print(f"Mean Top-1:   {mean_top1:.4f}")
    print(f"Mean Max DD:  {mean_dd:.4f}")

    failures = []
    if mean_auc < args.min_mean_auc:
        failures.append(
            f"Mean AUC {mean_auc:.4f} below threshold {args.min_mean_auc:.4f}"
        )
    if mean_top1 < args.min_mean_top1:
        failures.append(
            f"Mean Top-1 {mean_top1:.4f} below threshold {args.min_mean_top1:.4f}"
        )
    if mean_dd > args.max_mean_level_drawdown:
        failures.append(
            f"Mean drawdown {mean_dd:.4f} above threshold {args.max_mean_level_drawdown:.4f}"
        )

    if failures:
        print("\n[FAIL] Backtest quality gate failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\n[OK] Backtest quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
