from __future__ import annotations

import pathlib
import sys

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import backtest_walk_forward as btwf


def test_attach_market_odds_adds_diagnostics_and_matches(tmp_path):
    df = pd.DataFrame(
        {
            "date": ["2026-05-11", "2026-05-11", "2026-05-12"],
            "horse": ["Alpha", "Bravo (IRE)", "Charlie"],
            "pos_clean": [1, 2, 3],
        }
    )

    odds_df = pd.DataFrame(
        {
            "date": ["2026-05-11", "2026-05-11", "2026-05-11", "2026-05-12"],
            "horse": ["Alpha", "Alpha", "Bravo", "Nobody"],
            "market_odds": [2.1, 2.3, 3.4, 8.0],
        }
    )
    odds_path = tmp_path / "market_odds.csv"
    odds_df.to_csv(odds_path, index=False)

    out = btwf.attach_market_odds(df, str(odds_path))

    assert "market_odds" in out.columns
    assert out["market_odds"].notna().sum() == 2

    diagnostics = out.attrs.get("market_odds_diagnostics")
    assert diagnostics is not None
    assert diagnostics["source"] == "market_odds.csv"
    assert diagnostics["total_rows"] == 3
    assert diagnostics["matched_rows"] == 2
    assert abs(diagnostics["coverage"] - (2 / 3)) < 1e-6
    assert diagnostics["source_duplicate_key_rows"] == 1


def test_attach_market_odds_missing_required_columns_returns_input(tmp_path):
    df = pd.DataFrame(
        {
            "date": ["2026-05-11"],
            "horse": ["Alpha"],
            "pos_clean": [1],
        }
    )
    bad_odds = pd.DataFrame({"date": ["2026-05-11"], "horse": ["Alpha"]})
    odds_path = tmp_path / "bad_odds.csv"
    bad_odds.to_csv(odds_path, index=False)

    out = btwf.attach_market_odds(df, str(odds_path))

    assert "market_odds" not in out.columns
    assert out.equals(df)


def test_attach_market_odds_name_normalization_handles_diacritics_and_punctuation(tmp_path):
    df = pd.DataFrame(
        {
            "date": ["2026-05-11", "2026-05-11"],
            "horse": ["Senorita's Dream (IRE)", "Blue  Sky"],
            "pos_clean": [1, 2],
        }
    )

    odds_df = pd.DataFrame(
        {
            "date": ["2026-05-11", "2026-05-11"],
            "horse": ["Senoritas Dream", "Blue-Sky"],
            "market_odds": [5.5, 3.2],
        }
    )
    odds_path = tmp_path / "norm_odds.csv"
    odds_df.to_csv(odds_path, index=False)

    out = btwf.attach_market_odds(df, str(odds_path))

    assert out["market_odds"].notna().sum() == 2
    assert sorted(out["market_odds"].tolist()) == [3.2, 5.5]
