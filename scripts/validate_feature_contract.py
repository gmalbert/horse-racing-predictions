#!/usr/bin/env python3
"""Validate training/inference feature contract.

Checks that expected model feature columns are present in the best available
processed dataset used by prediction scripts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd


def load_expected_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature contract file: {path}")
    cols = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cols:
        raise ValueError(f"No features found in: {path}")
    return cols


def choose_best_dataset(expected: list[str], dataset_candidates: list[Path]):
    """Pick the available dataset with the highest feature coverage."""
    available = [p for p in dataset_candidates if p.exists()]
    if not available:
        raise FileNotFoundError(
            "No dataset found. Checked: " + ", ".join(str(p) for p in dataset_candidates)
        )

    best = None
    for path in available:
        df = pd.read_parquet(path)
        present = [c for c in expected if c in df.columns]
        missing = [c for c in expected if c not in df.columns]
        score = len(present)
        candidate = {
            "path": path,
            "df": df,
            "present": present,
            "missing": missing,
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate model feature contract")
    parser.add_argument(
        "--feature-file",
        default="models/feature_columns.txt",
        help="Path to expected feature columns text file",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Parquet dataset candidate to validate (repeatable)",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.05,
        help="Fail if missing feature ratio exceeds this threshold",
    )
    args = parser.parse_args()

    feature_file = Path(args.feature_file)
    dataset_candidates = args.dataset or [
        "data/processed/race_scores_engineered.parquet",
        "data/processed/race_scores_with_all_features_no_leakage.parquet",
        "data/processed/race_scores_connections_v2.parquet",
    ]
    dataset_candidates = [Path(p) for p in dataset_candidates]

    expected = load_expected_features(feature_file)
    best = choose_best_dataset(expected, dataset_candidates)
    dataset_file = best["path"]
    present = best["present"]
    missing = best["missing"]

    missing_ratio = len(missing) / len(expected)

    print("=" * 60)
    print("FEATURE CONTRACT VALIDATION")
    print("=" * 60)
    print(f"Feature file: {feature_file}")
    print(f"Dataset:      {dataset_file}")
    print(f"Expected:     {len(expected)}")
    print(f"Present:      {len(present)}")
    print(f"Missing:      {len(missing)} ({missing_ratio:.1%})")

    if missing:
        print("\nMissing features (first 50):")
        for name in missing[:50]:
            print(f"  - {name}")

    if missing_ratio > args.max_missing_ratio:
        print(
            f"\n[FAIL] Missing ratio {missing_ratio:.1%} exceeds threshold "
            f"{args.max_missing_ratio:.1%}"
        )
        return 1

    print("\n[OK] Feature contract is within threshold")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
