from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from scripts import fetch_racecards_api


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"racecards": [{"race_id": "rac_123", "course": "Test", "runners": []}]}


def test_free_plan_uses_documented_endpoint_and_day_parameter(monkeypatch):
    captured = {}
    monkeypatch.setattr(fetch_racecards_api, "get_credentials", lambda: ("user", "password"))

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(fetch_racecards_api.requests, "get", fake_get)
    today = datetime.now(ZoneInfo("America/New_York")).date()

    races = fetch_racecards_api.fetch_racecards_from_api(str(today), region="GB")

    assert races[0]["race_id"] == "rac_123"
    assert captured["url"] == "https://api.theracingapi.com/v1/racecards/free"
    assert captured["params"] == {"day": "today", "region_codes": "GB"}
    assert captured["auth"] == ("user", "password")


def test_rate_limit_retries_using_retry_after(monkeypatch):
    responses = [
        type("RateLimited", (), {"status_code": 429, "headers": {"Retry-After": "2"}})(),
        _Response(),
    ]
    waits = []
    monkeypatch.setattr(fetch_racecards_api, "get_credentials", lambda: ("user", "password"))
    monkeypatch.setattr(fetch_racecards_api.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(fetch_racecards_api.time, "sleep", waits.append)
    today = datetime.now(ZoneInfo("America/New_York")).date()

    races = fetch_racecards_api.fetch_racecards_from_api(str(today))

    assert races[0]["race_id"] == "rac_123"
    assert waits == [2]


def test_free_plan_rejects_dates_outside_today_and_tomorrow(monkeypatch):
    today = datetime.now(ZoneInfo("America/New_York")).date()

    try:
        fetch_racecards_api.fetch_racecards_from_api(str(today + timedelta(days=2)))
    except ValueError as error:
        assert "today and tomorrow" in str(error)
    else:
        raise AssertionError("expected an unsupported-date error")


def test_transform_preserves_api_ids_and_basic_runner_fields():
    payload = [{
        "race_id": "rac_123",
        "course": "Test Course",
        "region": "GB",
        "off_time": "1:50",
        "type": "Flat",
        "runners": [{"horse": "Example Horse", "lbs": "140", "speed_rating": "97"}],
    }]

    race = fetch_racecards_api.transform_to_standard_format(payload, "2026-08-26")["GB"]["Test Course"]["1:50"]

    assert race["race_id"] == "rac_123"
    assert race["race_type"] == "Flat"
    assert race["runners"][0]["name"] == "Example Horse"
    assert race["runners"][0]["lbs"] == "140"
    assert race["runners"][0]["ts"] == "97"
