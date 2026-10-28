import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scripts import export_best_bets as feed

TODAY = date(2026, 10, 28)
ROW = {"horse": "Runner", "date": str(TODAY), "course": "Track", "race_time": "1:10p", "win_probability": .3, "win_odds_fractional": "2/1"}


class HorseFeedTests(unittest.TestCase):
    def export_rows(self, rows, filename="us_predictions_2026-10-28.csv"):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame(rows).to_csv(root / filename, index=False)
            with patch.object(feed, "DATA_PROC", root), patch.object(feed, "DATA_FILES", root):
                feed.export(TODAY)
            return json.loads((root / "best_bets_today.json").read_text())

    def test_current_us_predictions_and_explicit_morning_line_are_published(self):
        result = self.export_rows([{**ROW, "market_odds_fractional": "5/1", "odds_source": "us_racecards.ml_odds"}])
        bet = result["bets"][0]
        self.assertEqual((bet["edge"], bet["tier"], bet["odds"]), (.1333, "Elite", 500))
        self.assertEqual(bet["price_type"], "morning_line")

    def test_model_derived_odds_are_not_market_prices(self):
        result = self.export_rows([ROW])
        self.assertEqual(result["bets"], [])
        self.assertEqual(result["meta"]["status"], "pipeline_pending")

    def test_past_file_and_past_row_are_not_relabelled(self):
        result = self.export_rows([{**ROW, "market_odds": 10}], "predictions_2026-05-19.csv")
        self.assertEqual(result["bets"], [])
        result = self.export_rows([{**ROW, "date": "2026-05-19", "market_odds": 10}])
        self.assertEqual(result["bets"], [])

    def test_bookmaker_price_wins_over_model_fair_odds_and_race_is_capped(self):
        rows = [{**ROW, "horse": f"Runner {i}", "market_odds": 10} for i in range(5)]
        result = self.export_rows(rows)
        self.assertEqual(len(result["bets"]), 3)
        self.assertTrue(all(b["odds"] == 900 for b in result["bets"]))

    def test_invalid_probability_is_not_clamped_into_a_pick(self):
        result = self.export_rows([{**ROW, "win_probability": float("nan"), "market_odds": 10}])
        self.assertEqual(result["bets"], [])


if __name__ == "__main__":
    unittest.main()
