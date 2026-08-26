import json

import pytest

from scripts.fetch_racecards import build_racecard_from_html
from scripts.fetch_racecards_browser import (
    _runner_collection,
    _state_race_links,
    _write_validated_snapshot,
    target_dates,
)


def test_target_dates_accepts_explicit_iso_date():
    assert target_dates("2026-08-25", None, "America/New_York") == ["2026-08-25"]


def test_runner_collection_finds_nested_browser_json():
    payload = {
        "data": {
            "race": {
                "cardRunners": {
                    "one": {
                        "horse": {"id": 123, "name": "Test Runner", "age": 4},
                        "jockey": {"id": 45, "name": "J Doe"},
                        "trainer": {"id": 67, "name": "A Trainer"},
                        "number": 2,
                    }
                }
            }
        }
    }

    runners = _runner_collection(payload)

    assert runners[0]["horseName"] == "Test Runner"
    assert runners[0]["horseUid"] == 123
    assert runners[0]["jockeyName"] == "J Doe"
    assert runners[0]["trainerStylename"] == "A Trainer"
    assert runners[0]["startNumber"] == 2


def test_state_race_links_finds_unmounted_meetings_and_filters_date():
    payload = {
        "raceCards": {
            "meetings": [
                {"races": [
                    {"raceUrl": "/racecards/16/musselburgh/2026-08-25/925679"},
                    {"raceUrl": "/racecards/31/lingfield/2026-08-25/925685"},
                    {"raceUrl": "/racecards/31/lingfield/2026-08-26/925700"},
                ]}
            ]
        }
    }

    assert _state_race_links(payload, "2026-08-25") == [
        ("925679", "/racecards/16/musselburgh/2026-08-25/925679"),
        ("925685", "/racecards/31/lingfield/2026-08-25/925685"),
    ]


def test_browser_payload_builds_canonical_racecard():
    page_html = b"""
    <html><body>
      <h1 class="RC-courseHeader__name">Kempton</h1>
      <span class="RC-header__raceInstanceTitle">Evening Handicap</span>
      <span class="RC-header__raceDistance">(1m)</span>
      <span class="RC-header__raceClass">(Class 4)</span>
      <div class="RC-headerBox__going">Going: Standard</div>
    </body></html>
    """
    runners = [{
        "horseName": "Example Horse",
        "horseAge": 4,
        "startNumber": 3,
        "raceDatetime": "2026-08-25T18:30:00",
        "courseUid": 1079,
        "raceTypeCode": "F",
    }]

    race = build_racecard_from_html(
        race_id="925679",
        href="/racecards/1079/kempton/2026-08-25/925679",
        date="2026-08-25",
        page_html=page_html,
        runners_list=runners,
    )

    assert race["race_id"] == 925679
    assert race["course"] == "Kempton"
    assert race["off_time"] == "18:30"
    assert race["going"] == "Standard"
    assert race["surface"] == "AW"
    assert race["field_size"] == 1
    assert race["runners"][0]["name"] == "Example Horse"


def test_embedded_race_data_wins_over_partial_dom_fallback():
    next_data = {
        "props": {
            "pageProps": {
                "initialState": {
                    "racePage": {
                        "data": {
                            "race": {
                                "raceTitle": "Complete Browser Race",
                                "raceType": "F",
                                "localMeetingRaceDateTime": "2026-08-25T14:15:00+01:00",
                                "courseId": 16,
                                "meetingName": "Musselburgh",
                                "countryCode": "GB",
                                "distanceFurlongs": 7,
                                "distanceYards": 1540,
                                "going": "Good",
                                "numberOfRunners": 2,
                            },
                            "runners": [
                                {"horseId": 1, "horseName": "First Horse", "startNumber": 1},
                                {"horseId": 2, "horseName": "Second Horse", "startNumber": 2},
                            ],
                        }
                    }
                }
            }
        }
    }
    page_html = (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(next_data)
        + "</script></body></html>"
    ).encode()

    race = build_racecard_from_html(
        race_id="925679",
        href="/racecards/16/musselburgh/2026-08-25/925679",
        date="2026-08-25",
        page_html=page_html,
        runners_list=[{"horseName": "Partial DOM Horse"}],
        race_meta={"courseName": "14:15\nMusselburgh"},
    )

    assert race["course"] == "Musselburgh"
    assert race["race_name"] == "Complete Browser Race"
    assert [runner["name"] for runner in race["runners"]] == ["First Horse", "Second Horse"]


def test_snapshot_validation_preserves_existing_file_on_empty_fetch(tmp_path):
    output_path = tmp_path / "racecards_2026-08-25.json"
    output_path.write_text(json.dumps({"existing": True}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to replace snapshot"):
        _write_validated_snapshot("2026-08-25", {}, tmp_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"existing": True}
