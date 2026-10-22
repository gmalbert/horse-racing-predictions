"""API contracts and money accounting, isolated from real betting records."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import config
from app.main import app
from app.services import bankroll_service, explorer_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "bankroll_state.json")
    monkeypatch.setattr(config, "BETTING_HISTORY_FILE", tmp_path / "betting_history.csv")
    monkeypatch.setattr(explorer_service, "_DF", None)
    return TestClient(app)


def bet(result="Pending", **overrides):
    return {"bet_date": "2026-10-21", "horse_name": "Test Horse", "stake": 10,
            "market_odds": 3, "result": result, **overrides}


@pytest.mark.parametrize("result,balance,profit", [("Won", 1020, 20), ("Lost", 990, -10)])
def test_direct_result_matches_pending_settlement(client, result, balance, profit):
    direct = client.post("/api/bets", json=bet(result)).json()
    assert direct["state"]["current_bankroll"] == balance
    client.put("/api/bankroll", json={"current_bankroll": 1000})
    pending = client.post("/api/bets", json=bet()).json()
    assert pending["state"]["current_bankroll"] == 990
    settled = client.post(f"/api/bets/{pending['index']}/settle", json={"result": result})
    assert settled.status_code == 200
    assert settled.json()["state"]["current_bankroll"] == balance
    assert settled.json()["profit"] == profit
    assert client.post(f"/api/bets/{pending['index']}/settle", json={"result": result}).status_code == 400
    history = client.get("/api/bets").json()
    assert len(history["bets"]) == 2
    assert history["bets"][0]["race_id"] == ""
    assert client.get("/api/performance").json()["totals"]["pnl"] == profit * 2


@pytest.mark.parametrize("overrides", [
    {"result": "oops"}, {"stake": 0}, {"market_odds": 1},
    {"win_probability": 1.1}, {"bet_date": "../missing"}, {"horse_name": "  "},
])
def test_bad_bets_do_not_write(client, overrides):
    assert client.post("/api/bets", json=bet(**overrides)).status_code == 422
    assert not config.BETTING_HISTORY_FILE.exists()


def test_zero_bankroll_and_persistence(client):
    assert client.put("/api/bankroll", json={"current_bankroll": 0}).status_code == 200
    assert TestClient(app).get("/api/bankroll").json()["current_bankroll"] == 0


def test_parallel_logging_preserves_every_write(client):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: bankroll_service.log_bet(bet(horse_name=f"Horse {i}")), range(20)))
    result = client.get("/api/bets").json()
    assert len(result["bets"]) == 20
    assert result["state"]["current_bankroll"] == 800
    assert result["state"]["total_bets"] == 20


def write_history():
    pd.DataFrame([
        {"date": "2026-01-01", "horse": "A[", "course": "Ascot", "pos": "1", "jockey": "J", "trainer": "T", "going": "Good", "dist_f": 7, "pattern": "G1"},
        {"date": "2026-01-02", "horse": "B", "course": "York", "pos": "PU", "jockey": "K", "trainer": "U", "going": "Soft", "dist_f": 8, "pattern": ""},
    ]).to_parquet(config.DATA_DIR / config.EXPLORER_PARQUET)


def test_explorer_filters_literal_search_and_empty_statistics(client):
    write_history()
    assert client.get("/api/explorer/filters").status_code == 200
    response = client.post("/api/explorer/query", json={"view": "raw", "horse_search": "["})
    assert response.status_code == 200
    assert response.json()["total_rows"] == 1
    assert client.post("/api/explorer/query", json={"going": ["Soft"], "view": "raw"}).json()["rows"][0]["horse"] == "B"
    assert client.post("/api/explorer/query", json={"pattern_only": True}).json()["total_rows"] == 1
    assert client.post("/api/explorer/query", json={"horse_search": "missing"}).json()["overall"]["avg_finish"] is None
    assert client.post("/api/explorer/query", json={"going": ["Soft"]}).json()["overall"]["avg_finish"] is None


@pytest.mark.parametrize("body", [{"page": 0}, {"page_size": 0}, {"page_size": 201}, {"view": "unknown"}])
def test_explorer_rejects_invalid_pagination(client, body):
    assert client.post("/api/explorer/query", json=body).status_code == 422


def test_missing_dataset_is_service_unavailable(client):
    assert client.get("/api/explorer/filters").status_code == 503
    assert client.get("/api/health").status_code == 200


def test_kelly_requires_valid_date_and_positive_edge(client):
    assert client.get("/api/kelly?date=bad").status_code == 422
    assert bankroll_service.kelly_stake_fraction(0.1, 2) == 0
    assert client.get("/api/kelly?date=2026-10-21").json()["found"] is False


def test_export_preserves_value_flags_and_distance_boundaries():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from export_web_data import build_day_payload, distance_band
    frame = pd.DataFrame([
        {"date": "2026-10-21", "course": "Ascot", "race_time": "12:00", "horse": "Value", "win_probability": .4, "place_probability": .3, "show_probability": .2, "edge": .1, "is_value_bet": True},
        {"date": "2026-10-21", "course": "Ascot", "race_time": "12:00", "horse": "Not Value", "win_probability": .3, "place_probability": .2, "show_probability": .1, "edge": -.1, "is_value_bet": False},
    ])
    payload = build_day_payload(frame, "2026-10-21")
    assert [row["horse"] for row in payload["value_bets"]] == ["Value"]
    assert payload["summary"]["horses"] == 2
    assert distance_band(8) == "Mile (6-8f)"
    assert distance_band(12) == "Middle (8-12f)"
