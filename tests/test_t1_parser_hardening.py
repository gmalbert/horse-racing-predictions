"""Parser hardening tests for T1 TB/QH/H track-site canonical parsers.

These tests use offline synthetic fixtures to validate in-progress track parsing
behavior and avoid live network dependencies.
"""

from __future__ import annotations

import json
import pathlib
import sys
import asyncio
import pytest

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
FIXTURES = pathlib.Path(__file__).parent / "fixtures"

sys.path.insert(0, str(SCRIPTS))

import parse_t1_tb_tracksite as tb_parser
import parse_t1_qh_tracksite as qh_parser
import parse_t1_h_tracksite as h_parser
import us_track_collector as collector
from us_track_sources import ALL_TRACKS


SAMPLES = json.loads((FIXTURES / "t1_tracksite_samples.json").read_text(encoding="utf-8"))
REAL_SNAPSHOTS = json.loads((FIXTURES / "t1_real_capture_snapshots.json").read_text(encoding="utf-8"))


def test_tam_results_parser_extracts_runner_rows():
    rows = tb_parser._parse_tam_results(SAMPLES["tb_tam_track"])

    assert rows, "Expected TAM parser to return rows"
    assert any(r.get("runner_name") == "Blue Sky" for r in rows)
    assert any(r.get("race_number") == 1 for r in rows)
    assert any(r.get("race_number") == 2 for r in rows)


def test_tb_summary_parser_uses_summary_metadata_and_pdf_title():
    rows = tb_parser._parse_summary_event(SAMPLES["tb_summary_track"], SAMPLES["pdf_info_tb"])

    assert len(rows) == 1
    row = rows[0]
    assert row["event_type"] == "entries_summary"
    assert row["race_time"] in {"1:00 PM", "1:05 PM"}
    assert "pdf_title=Lone Star Entries" in (row.get("notes") or "")


def test_qh_summary_parser_uses_pdf_fallback_and_notes():
    rows = qh_parser._parse_summary_event(SAMPLES["qh_summary_track"], SAMPLES["pdf_info_qh"])

    assert len(rows) == 1
    row = rows[0]
    assert row["race_time"] in {"12:30 PM", "12:45 PM"}
    assert row["event_type"] == "entries_summary"
    assert "race_count_hint=9" in (row.get("notes") or "")


def test_h_summary_parser_uses_pdf_time_when_missing():
    rows = h_parser._parse_summary_event(SAMPLES["h_summary_track"], SAMPLES["pdf_info_h"])

    assert len(rows) == 1
    row = rows[0]
    assert row["race_time"] == "7:15 PM"
    assert "pdf_needs_ocr=true" in (row.get("notes") or "")


def test_qh_parse_date_writes_canonical_outputs(tmp_path):
    # Patch module paths to isolated temp workspace
    qh_parser.TRACKSITE_DIR = tmp_path / "tracksite"
    qh_parser.PROCESSED_DIR = tmp_path / "processed"
    qh_parser.TARGET_TRACKS = ["ZIA"]

    qh_parser.TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = "2026-05-11"
    (qh_parser.TRACKSITE_DIR / f"ZIA_{date_str}.json").write_text(
        json.dumps(SAMPLES["qh_summary_track"], indent=2), encoding="utf-8"
    )
    (qh_parser.TRACKSITE_DIR / f"pdf_extracted_{date_str}.json").write_text(
        json.dumps({"pdfs": [SAMPLES["pdf_info_qh"]]}, indent=2), encoding="utf-8"
    )

    csv_out, json_out, rows = qh_parser.parse_date(date_str, append=False)

    assert rows == 1
    assert csv_out.exists()
    assert json_out.exists()

    df = pd.read_csv(csv_out)
    assert len(df) == 1
    assert df.iloc[0]["track_code"] == "ZIA"


def test_tb_parse_date_with_real_snapshots(tmp_path):
    tb_parser.TRACKSITE_DIR = tmp_path / "tracksite_tb"
    tb_parser.PROCESSED_DIR = tmp_path / "processed_tb"
    tb_parser.TARGET_TRACKS = ["TAM", "HOU"]

    tb_parser.TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = "2026-05-09"
    (tb_parser.TRACKSITE_DIR / f"TAM_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_tam_2026_05_09"], indent=2), encoding="utf-8"
    )
    (tb_parser.TRACKSITE_DIR / f"HOU_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_hou_2026_05_09"], indent=2), encoding="utf-8"
    )

    csv_out, json_out, rows = tb_parser.parse_date(date_str, append=False)

    assert rows >= 4
    assert csv_out.exists()
    assert json_out.exists()

    df = pd.read_csv(csv_out)
    assert "results" in set(df["event_type"].tolist())
    assert "entries_summary" in set(df["event_type"].tolist())
    assert "Maxitas" in set(df["runner_name"].dropna().tolist())


def test_tb_parse_date_with_additional_real_tracks(tmp_path):
    tb_parser.TRACKSITE_DIR = tmp_path / "tracksite_tb_extra"
    tb_parser.PROCESSED_DIR = tmp_path / "processed_tb_extra"
    tb_parser.TARGET_TRACKS = ["CBY", "CT", "ELP"]

    tb_parser.TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = "2026-05-11"
    (tb_parser.TRACKSITE_DIR / f"CBY_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_cby_2026_05_11"], indent=2), encoding="utf-8"
    )
    (tb_parser.TRACKSITE_DIR / f"CT_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_ct_2026_05_11"], indent=2), encoding="utf-8"
    )
    (tb_parser.TRACKSITE_DIR / f"ELP_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_elp_2026_05_11"], indent=2), encoding="utf-8"
    )

    csv_out, json_out, rows = tb_parser.parse_date(date_str, append=False)

    assert rows == 3
    assert csv_out.exists()
    assert json_out.exists()

    df = pd.read_csv(csv_out)
    assert set(df["track_code"].tolist()) == {"CBY", "CT", "ELP"}
    assert set(df["event_type"].tolist()) == {"entries_summary"}


def test_tb_parse_date_with_mnr_pen_pid_real_tracks(tmp_path):
    tb_parser.TRACKSITE_DIR = tmp_path / "tracksite_tb_more"
    tb_parser.PROCESSED_DIR = tmp_path / "processed_tb_more"
    tb_parser.TARGET_TRACKS = ["MNR", "PEN", "PID"]

    tb_parser.TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = "2026-05-11"
    (tb_parser.TRACKSITE_DIR / f"MNR_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_mnr_2026_05_11"], indent=2), encoding="utf-8"
    )
    (tb_parser.TRACKSITE_DIR / f"PEN_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_pen_2026_05_11"], indent=2), encoding="utf-8"
    )
    (tb_parser.TRACKSITE_DIR / f"PID_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["tb_pid_2026_05_11"], indent=2), encoding="utf-8"
    )

    csv_out, json_out, rows = tb_parser.parse_date(date_str, append=False)

    assert rows == 3
    assert csv_out.exists()
    assert json_out.exists()

    df = pd.read_csv(csv_out)
    assert set(df["track_code"].tolist()) == {"MNR", "PEN", "PID"}
    assert set(df["event_type"].tolist()) == {"entries_summary"}


def test_h_parse_date_supports_multiple_tracks(tmp_path):
    h_parser.TRACKSITE_DIR = tmp_path / "tracksite_h"
    h_parser.PROCESSED_DIR = tmp_path / "processed_h"
    h_parser.TARGET_TRACKS = ["RSC", "CAL"]

    h_parser.TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = "2026-05-11"
    rsc = dict(SAMPLES["h_summary_track"])
    cal = dict(SAMPLES["h_summary_track"])
    cal["track_code"] = "CAL"
    cal["track_name"] = "Cal Expo"

    (h_parser.TRACKSITE_DIR / f"RSC_{date_str}.json").write_text(
        json.dumps(rsc, indent=2), encoding="utf-8"
    )
    (h_parser.TRACKSITE_DIR / f"CAL_{date_str}.json").write_text(
        json.dumps(cal, indent=2), encoding="utf-8"
    )
    (h_parser.TRACKSITE_DIR / f"pdf_extracted_{date_str}.json").write_text(
        json.dumps({"pdfs": [SAMPLES["pdf_info_h"]]}, indent=2), encoding="utf-8"
    )

    csv_out, _, rows = h_parser.parse_date(date_str, append=False)

    assert rows == 2
    df = pd.read_csv(csv_out)
    assert set(df["track_code"].tolist()) == {"RSC", "CAL"}


def test_qh_parse_date_with_real_snapshots(tmp_path):
    qh_parser.TRACKSITE_DIR = tmp_path / "tracksite_qh"
    qh_parser.PROCESSED_DIR = tmp_path / "processed_qh"
    qh_parser.TARGET_TRACKS = ["ZIA", "LA"]

    qh_parser.TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = "2026-05-10"
    (qh_parser.TRACKSITE_DIR / f"ZIA_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["qh_zia_2026_05_10"], indent=2), encoding="utf-8"
    )
    (qh_parser.TRACKSITE_DIR / f"LA_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["qh_la_2026_05_10"], indent=2), encoding="utf-8"
    )

    csv_out, json_out, rows = qh_parser.parse_date(date_str, append=False)

    assert rows == 2
    assert csv_out.exists()
    assert json_out.exists()

    df = pd.read_csv(csv_out)
    assert set(df["track_code"].tolist()) == {"ZIA", "LA"}
    assert set(df["event_type"].tolist()) == {"entries_summary"}


def test_h_parse_date_with_real_snapshots(tmp_path):
    h_parser.TRACKSITE_DIR = tmp_path / "tracksite_h_real"
    h_parser.PROCESSED_DIR = tmp_path / "processed_h_real"
    h_parser.TARGET_TRACKS = ["PLN", "NTH"]

    h_parser.TRACKSITE_DIR.mkdir(parents=True, exist_ok=True)

    date_str = "2026-05-10"
    (h_parser.TRACKSITE_DIR / f"PLN_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["h_pln_2026_05_10"], indent=2), encoding="utf-8"
    )
    (h_parser.TRACKSITE_DIR / f"NTH_{date_str}.json").write_text(
        json.dumps(REAL_SNAPSHOTS["h_nth_2026_05_10"], indent=2), encoding="utf-8"
    )

    csv_out, json_out, rows = h_parser.parse_date(date_str, append=False)

    assert rows == 2
    assert csv_out.exists()
    assert json_out.exists()

    df = pd.read_csv(csv_out)
    assert set(df["track_code"].tolist()) == {"PLN", "NTH"}
    assert set(df["event_type"].tolist()) == {"entries_summary"}


def test_qh_summary_parser_handles_missing_summary_fields():
    malformed = {
        "date": "2026-05-11",
        "track_code": "LA",
        "track_name": "Los Alamitos",
        "best_page": {
            "url": "https://example.com/entries",
            "summary_fields": {},
        },
    }

    rows = qh_parser._parse_summary_event(malformed, None)
    assert len(rows) == 1
    assert rows[0]["track_code"] == "LA"
    assert rows[0]["event_type"] == "entries_summary"


def test_tb_tam_parser_handles_empty_text_sample():
    empty_tam = {
        "date": "2026-05-11",
        "track_code": "TAM",
        "track_name": "Tampa Bay Downs",
        "selected": {"final_url": "https://example.com/results"},
        "text_sample": "",
    }
    rows = tb_parser._parse_tam_results(empty_tam)
    assert rows == []


def test_real_capture_tb_tam_snapshot_parses_runner_rows():
    snap = REAL_SNAPSHOTS["tb_tam_2026_05_11"]
    rows = tb_parser._parse_tam_results(snap)

    assert rows
    assert any(r.get("runner_name") == "Wicked Legacy" for r in rows)
    assert any(str(r.get("race_number")) == "1" for r in rows)


def test_real_capture_tb_ls_snapshot_summary_parses():
    snap = REAL_SNAPSHOTS["tb_ls_2026_05_10"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "LS"
    assert row["event_type"] == "entries_summary"
    assert row["race_time"] == "1:35pm"
    assert "race_count_hint=9" in (row.get("notes") or "")


def test_real_capture_tb_prx_snapshot_summary_mentions_results():
    snap = REAL_SNAPSHOTS["tb_prx_2026_05_10"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "PRX"
    assert row["source_url"] == "https://www.parxracing.com/racing/entries"
    assert "mentions_results=true" in (row.get("notes") or "")


def test_real_capture_tb_tam_0509_snapshot_parses_runner_rows():
    snap = REAL_SNAPSHOTS["tb_tam_2026_05_09"]
    rows = tb_parser._parse_tam_results(snap)

    assert rows
    assert any(r.get("runner_name") == "Maxitas" for r in rows)
    assert any(str(r.get("race_number")) == "1" for r in rows)


def test_real_capture_tb_hou_0509_snapshot_summary_parses():
    snap = REAL_SNAPSHOTS["tb_hou_2026_05_09"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "HOU"
    assert row["source_url"] == "https://www.shrp.com/racing"
    assert "mentions_entries=true" in (row.get("notes") or "")


def test_real_capture_tb_cby_snapshot_summary_parses():
    snap = REAL_SNAPSHOTS["tb_cby_2026_05_11"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "CBY"
    assert row["event_type"] == "entries_summary"
    assert "mentions_results=true" in (row.get("notes") or "")


def test_real_capture_tb_ct_snapshot_summary_handles_no_mentions():
    snap = REAL_SNAPSHOTS["tb_ct_2026_05_11"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "CT"
    assert row["event_type"] == "entries_summary"
    assert row["notes"] in {None, ""}


def test_real_capture_tb_elp_snapshot_summary_handles_404_page():
    snap = REAL_SNAPSHOTS["tb_elp_2026_05_11"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "ELP"
    assert row["source_url"] == "https://www.ellisparkracing.com/racing/entries"
    assert row["event_type"] == "entries_summary"


def test_real_capture_tb_mnr_snapshot_summary_handles_blocked_page():
    snap = REAL_SNAPSHOTS["tb_mnr_2026_05_11"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "MNR"
    assert row["source_url"] == "https://www.cnty.com/mountaineer/racing/entries-results"
    assert row["notes"] in {None, ""}


def test_real_capture_tb_pen_snapshot_summary_mentions_results_only():
    snap = REAL_SNAPSHOTS["tb_pen_2026_05_11"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "PEN"
    assert "mentions_results=true" in (row.get("notes") or "")
    assert "mentions_entries=true" not in (row.get("notes") or "")


def test_real_capture_tb_pid_snapshot_summary_mentions_entries_and_results():
    snap = REAL_SNAPSHOTS["tb_pid_2026_05_11"]
    rows = tb_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "PID"
    assert "mentions_entries=true" in (row.get("notes") or "")
    assert "mentions_results=true" in (row.get("notes") or "")


def test_real_capture_qh_rud_snapshot_summary_parses():
    snap = REAL_SNAPSHOTS["qh_rud_2026_05_11"]
    rows = qh_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "RUD"
    assert row["event_type"] == "entries_summary"
    assert "has_entries_section=true" in (row.get("notes") or "")


def test_real_capture_qh_zia_access_denied_still_parses_summary():
    snap = REAL_SNAPSHOTS["qh_zia_2026_05_10"]
    rows = qh_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "ZIA"
    assert row["source_url"] == "https://www.ziaparkracing.com/racing/entries"
    assert row["event_type"] == "entries_summary"


def test_real_capture_qh_la_null_best_page_parses_summary():
    snap = REAL_SNAPSHOTS["qh_la_2026_05_10"]
    rows = qh_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "LA"
    assert row["source_url"] is None
    assert row["event_type"] == "entries_summary"


def test_real_capture_qh_sun_0509_null_best_page_parses_summary():
    snap = REAL_SNAPSHOTS["qh_sun_2026_05_09"]
    rows = qh_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "SUN"
    assert row["source_url"] is None
    assert row["event_type"] == "entries_summary"


def test_real_capture_h_rsc_snapshot_summary_parses():
    snap = REAL_SNAPSHOTS["h_rsc_2026_05_11"]
    rows = h_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "RSC"
    assert row["event_type"] == "entries_summary"
    assert row["source_url"] == "https://www.rosecroft.com/racing/entries-results"


def test_real_capture_h_pln_fallback_page_parses_summary():
    snap = REAL_SNAPSHOTS["h_pln_2026_05_10"]
    rows = h_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "PLN"
    assert row["source_url"] == "https://www.plainridgepark.com/racing"
    assert row["event_type"] == "entries_summary"


def test_real_capture_h_nth_null_best_page_parses_summary():
    snap = REAL_SNAPSHOTS["h_nth_2026_05_10"]
    rows = h_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "NTH"
    assert row["source_url"] is None
    assert row["event_type"] == "entries_summary"


def test_real_capture_h_run_0509_null_best_page_parses_summary():
    snap = REAL_SNAPSHOTS["h_run_2026_05_09"]
    rows = h_parser._parse_summary_event(snap, None)

    assert len(rows) == 1
    row = rows[0]
    assert row["track_code"] == "RUN"
    assert row["source_url"] is None
    assert row["event_type"] == "entries_summary"


# Minimal versions of the actual Equibase/USTA markup verified on 2026-10-25.
EQB_CARD = """<h2 class="track-info">Parx Racing October 26, 2026</h2>
<div class="entryRace"><h4>Race 3 POST Time - 1:20 PM ET</h4>
<table><thead><tr><th>P#</th><th>PP</th><th>Horse</th><th>Jockey</th>
<th>Trainer</th><th>M/L</th><th>Wgt</th></tr></thead><tbody>
<tr><td>1A</td><td>2</td><td>Test Horse (KY)</td><td>A Rider</td><td>B Trainer</td><td>5/2</td><td>122</td></tr>
<tr class="scratched"><td>2</td><td>3</td><td>Scratched Horse</td><td>A</td><td>B</td><td>3/1</td><td>122</td></tr>
</tbody></table><div class="race-info marginTop20"><h3>Claiming</h3>
<h5>$21,000<span></span>6 F<span></span>Dirt<span></span>1:20 PM ET</h5></div></div>"""
USTA_CARD = """<h4>PLAINRIDGE PARK Monday, October 26, 2026</h4>
<div class="card"><h4 class="card-header"><div class="race-status entries"></div>RACE 1 2:00 PM</h4>
<table><thead><tr><th>PP</th><th>Horse</th><th>Driver</th><th>Trainer</th><th>Odds</th></tr></thead>
<tbody><tr><td>1</td><td>Test Pacer</td><td>A Driver</td><td>B Trainer</td><td>9-2</td></tr>
<tr><td>2</td><td>Another Pacer</td><td>C Driver</td><td>D Trainer</td><td>5-1</td></tr></tbody></table></div>"""


def test_dated_equibase_card_preserves_runners_and_metadata():
    races, status = collector.parse_entries(EQB_CARD, 'https://tvg.equibase.com/static/entry/PRX102626USA-EQB.html',
                                           'PRX', ALL_TRACKS['PRX'], '2026-10-26')
    assert status == 'ok' and len(races) == 1
    assert races[0]['race_number'] == 3
    assert races[0]['distance'] == '6 F'
    assert races[0]['surface'] == 'Dirt'
    assert races[0]['runners'] == [{'horse': 'Test Horse', 'number': '1A', 'draw': '2',
                                  'jockey': 'A Rider', 'driver': '', 'trainer': 'B Trainer',
                                  'ml_odds': '5/2', 'weight': '122'}]


def test_usta_card_preserves_drivers_and_odds():
    races, status = collector.parse_entries(USTA_CARD, 'https://m.ustrotting.com/entriesandresults/racecardbytrack.cfm?race_card_id=1',
                                           'PLN', ALL_TRACKS['PLN'], '2026-10-26')
    assert status == 'ok'
    assert len(races[0]['runners']) == 2
    assert races[0]['runners'][0]['driver'] == 'A Driver'
    assert races[0]['runners'][0]['ml_odds'] == '9-2'


@pytest.mark.parametrize('html,reason', [
    (EQB_CARD.replace('October 26', 'October 25'), 'date_unverified'),
    (EQB_CARD.replace('Parx Racing', 'Mountaineer'), 'track_mismatch'),
    ('<select><option>October 26, 2026</option></select>'+EQB_CARD.replace('October 26', 'October 25'), 'date_unverified'),
    ('<h2 class="track-info">Parx Racing October 26, 2026</h2>', 'not_listed'),
    (EQB_CARD.replace('entryRace', 'changedMarkup'), 'no_runner_tables'),
])
def test_card_provenance_and_markup_failures(html, reason):
    races, status = collector.parse_entries(html, 'https://tvg.equibase.com/static/entry/PRX102626USA-EQB.html',
                                           'PRX', ALL_TRACKS['PRX'], '2026-10-26')
    assert races == [] and status == reason


def test_completed_usta_results_are_not_entries():
    races, status = collector.parse_entries(USTA_CARD.replace('race-status entries', 'race-status results'),
                                           'https://m.ustrotting.com/entriesandresults/racecardbytrack.cfm?race_card_id=1',
                                           'PLN', ALL_TRACKS['PLN'], '2026-10-26')
    assert races == [] and status == 'no_runner_tables'


def test_discovery_prioritizes_dated_entries_and_limits_hosts():
    html = """<a href="/results">Results</a><a href="/news/race-1">News</a>
    <a href="https://track.example.evil.test/entries">Entries</a>
    <a href="/entries">Entries</a><a href="https://tvg.equibase.com/static/entry/PRX102626USA-EQB.html">Entries</a>"""
    links = collector.discover_links(html, 'https://track.example/racing/', '2026-10-26')
    assert links == ['https://tvg.equibase.com/static/entry/PRX102626USA-EQB.html', 'https://track.example/entries']
    assert collector.soup_text('<style>Race 8</style><script>Race 9</script><p>Race 1</p>') == 'Race 1'


@pytest.mark.parametrize('error,status,text,kind', [
    ('net::ERR_NAME_NOT_RESOLVED', None, '', 'dns'),
    ('net::ERR_CERT_COMMON_NAME_INVALID', None, '', 'tls'),
    ('net::ERR_HTTP2_PROTOCOL_ERROR', None, '', 'http2'),
    ('Navigation interrupted by another navigation', None, '', 'navigation'),
    ('Timeout 18000ms exceeded', None, '', 'timeout'),
    ('', 403, '', 'blocked'), ('', 200, 'Pardon Our Interruption', 'blocked'),
    ('', 404, '', 'not_found'),
])
def test_source_error_classification(error, status, text, kind):
    assert collector.error_kind(error, status, text) == kind


def test_transport_retry_uses_a_new_page_and_preserves_full_errors(monkeypatch):
    long_error = 'navigation interrupted by another navigation to ' + 'https://track.example/redirect/'*10
    def broken_http(url):
        raise OSError('connection closed')
    monkeypatch.setattr(collector, '_http_get', broken_http)
    pages = []
    class Page:
        closed = False
        async def goto(self, *args, **kwargs):
            raise RuntimeError(long_error)
        async def close(self):
            self.closed = True
    class Context:
        async def new_page(self):
            page = Page()
            pages.append(page)
            return page
    async def run():
        context = Context()
        return [await collector.fetch_page(context, 'https://track.example/entries') for _ in range(2)]
    results = asyncio.run(run())
    assert len(pages) == 2 and pages[0] is not pages[1]
    assert all(p.closed for p in pages)
    assert all(r['attempts'][-1]['error'] == long_error for r in results)


def test_blocked_track_recovers_from_independent_provider(monkeypatch):
    seen = []
    async def fetch(context, url):
        seen.append(url)
        html = EQB_CARD if url.startswith('https://www.equibase.com/') else 'Access Denied'
        error = None if html == EQB_CARD else 'blocked'
        return {'url': url, 'status_code': 200 if not error else 403, 'html': html,
                'text': collector.soup_text(html), 'error_kind': error,
                'attempts': [{'url': url, 'error_kind': error}]}
    monkeypatch.setattr(collector, 'fetch_page', fetch)
    record = asyncio.run(collector.collect_track(None, 'PRX', ALL_TRACKS['PRX'], '2026-10-26'))
    assert record['status'] == 'ok' and len(record['racecards']) == 1
    assert len(seen) == 2
    assert record['attempts'][0]['error_kind'] == 'blocked'


def test_usta_selects_requested_date_instead_of_latest_card(monkeypatch):
    index = """<h3>PLAINRIDGE PARK</h3><a href="racecardbytrack.cfm?race_card_id=99">October 27, 2026</a>
    <a href="racecardbytrack.cfm?race_card_id=1">Monday, October 26, 2026</a>"""
    seen = []
    async def fetch(context, url):
        seen.append(url)
        html = index if 'tracklist' in url else USTA_CARD
        return {'url': url, 'status_code': 200, 'html': html, 'text': collector.soup_text(html),
                'error_kind': None, 'attempts': [{'url': url}]}
    monkeypatch.setattr(collector, 'fetch_page', fetch)
    record = asyncio.run(collector.collect_track(None, 'PLN', ALL_TRACKS['PLN'], '2026-10-26'))
    assert record['status'] == 'ok'
    assert seen[-1].endswith('race_card_id=1')
    assert not any('race_card_id=99' in url for url in seen)


@pytest.mark.parametrize('parser,code', [(qh_parser, 'ZIA'), (h_parser, 'PLN'), (tb_parser, 'PRX')])
def test_empty_history_and_multiple_runners_round_trip(tmp_path, monkeypatch, parser, code):
    tracksite, processed = tmp_path/'tracksite', tmp_path/'processed'
    tracksite.mkdir(); processed.mkdir()
    monkeypatch.setattr(parser, 'TRACKSITE_DIR', tracksite)
    monkeypatch.setattr(parser, 'PROCESSED_DIR', processed)
    monkeypatch.setattr(parser, 'TARGET_TRACKS', [code])
    group = {'ZIA': 'qh', 'PLN': 'h', 'PRX': 'tb'}[code]
    (processed/f'us_t1_{group}_canonical_all.csv').write_text('\n')
    csv, js, count = parser.parse_date('2026-10-26', append=True)
    assert count == 0 and pd.read_csv(csv).empty
    races, _ = collector.parse_entries(USTA_CARD,
        'https://m.ustrotting.com/entriesandresults/racecardbytrack.cfm?race_card_id=1',
        'PLN', ALL_TRACKS['PLN'], '2026-10-26')
    record = {'date': '2026-10-26', 'track_code': code, 'track_name': ALL_TRACKS[code]['name'],
              'status': 'ok', 'racecards': races}
    (tracksite/f'{code}_2026-10-26.json').write_text(json.dumps(record))
    _, js, count = parser.parse_date('2026-10-26', append=True)
    assert count == 2
    rows = json.loads(js.read_text())
    assert rows[0]['ml_odds'] == '9-2'
    assert len(pd.read_csv(processed/f'us_t1_{group}_canonical_all.csv')) == 2
    record['status'] = 'collection_error'
    (tracksite/f'{code}_2026-10-26.json').write_text(json.dumps(record))
    _, _, count = parser.parse_date('2026-10-26')
    assert count == 0


@pytest.mark.parametrize('known_day', [True, False])
def test_usta_absent_track_requires_a_current_provider_date(monkeypatch, known_day):
    index = '<select id="slctRaceDate"><option value="'+('2026-10-26' if known_day else '2026-09-01')+'"></option></select>'
    index += '<select id="slctTrackCode"><option value="PRc">PLAINRIDGE PARK</option></select>'
    async def fetch(context, url):
        return {'url': url, 'status_code': 200, 'html': index if 'tracklist' in url else '',
                'text': '', 'error_kind': None, 'attempts': [{'url': url}]}
    monkeypatch.setattr(collector, 'fetch_page', fetch)
    record = asyncio.run(collector.collect_track(None, 'RUN', ALL_TRACKS['RUN'], '2026-10-26'))
    assert record['status'] == ('not_listed' if known_day else 'collection_error')


def test_collector_reports_merge_real_cards_without_result_rows(tmp_path, monkeypatch):
    import fetch_us_all_sources as sources
    races, _ = collector.parse_entries(EQB_CARD, 'https://tvg.equibase.com/static/entry/PRX102626USA-EQB.html',
                                       'PRX', ALL_TRACKS['PRX'], '2026-10-26')
    record = {'date': '2026-10-26', 'track_code': 'PRX', 'track_name': 'Parx', 'status': 'ok',
              'racecards': races, 'attempts': [], 'html': '', 'text_sample': ''}
    collector.save_group('tb', '2026-10-26', [record], root=tmp_path)
    monkeypatch.setattr(sources, 'RAW_DIR', tmp_path/'data/raw')
    monkeypatch.setattr(sources, 'PROC_DIR', tmp_path/'data/processed')
    monkeypatch.setattr(sources, '_run_t1_pipeline', lambda *args: {'returncode': 0})
    monkeypatch.setattr(sources, '_run_nyra', lambda *args: {'returncode': 0})
    monkeypatch.setattr(sys, 'argv', ['collector', '--date', '2026-10-26', '--force', '--no-playwright'])
    sources.main()
    payload = json.loads((sources.RAW_DIR/'us_racecards_2026-10-26.json').read_text())
    assert payload['racecards'] == races
    assert payload['racecards'][0]['runners'][0]['ml_odds'] == '5/2'
    sources.PROC_DIR.mkdir(parents=True)
    rows = collector.canonical_entry_rows(record, tb_parser._base_row)
    rows += [dict(rows[0], event_type='results'), dict(rows[0], date='2026-10-25')]
    (sources.PROC_DIR/'us_t1_tb_canonical_2026-10-26.json').write_text(json.dumps(rows))
    canonical = sources._collect_t1_canonical_runner_races('2026-10-26')
    assert len(canonical) == 1 and len(canonical[0]['runners']) == 1
    assert canonical[0]['runners'][0]['ml_odds'] == '5/2'


def test_subprocess_timeout_keeps_byte_diagnostics(monkeypatch):
    import fetch_us_all_sources as sources
    import subprocess
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(['collector'], 1, output=b'output', stderr=b'failure')
    monkeypatch.setattr(sources.subprocess, 'run', fail)
    result = sources._run(['collector'], timeout_seconds=1)
    assert result['returncode'] == 124
    assert result['stdout_tail'] == 'output' and result['stderr_tail'] == 'failure'


@pytest.mark.parametrize('failure_type', [RuntimeError, asyncio.TimeoutError])
def test_one_track_exception_does_not_cancel_other_tracks(monkeypatch, failure_type):
    import playwright.async_api
    contexts = []
    class Context:
        closed = False
        async def close(self):
            self.closed = True
    class Browser:
        async def new_context(self, **kwargs):
            context = Context()
            contexts.append(context)
            return context
        async def close(self):
            pass
    class Chromium:
        async def launch(self, **kwargs):
            assert '--disable-http2' in kwargs['args']
            return Browser()
    class API:
        chromium = Chromium()
    class Manager:
        async def __aenter__(self):
            return API()
        async def __aexit__(self, *args):
            pass
    async def collect(context, code, cfg, day, *, attempts):
        attempts.append({'url': cfg['url'], 'phase': 'provider request'})
        if code == 'CT':
            raise failure_type('collector failed')
        return {'track_code': code, 'status': 'ok', 'racecards': []}
    monkeypatch.setattr(playwright.async_api, 'async_playwright', Manager)
    monkeypatch.setattr(collector, 'collect_track', collect)
    records = asyncio.run(collector.pull_tracks({c: ALL_TRACKS[c] for c in ('CT','PRX')}, '2026-10-26'))
    assert [r['status'] for r in records] == ['collection_error', 'ok']
    assert records[0]['attempts'][0]['phase'] == 'provider request'
    assert records[0]['attempts'][-1]['error_kind'] == ('timeout' if failure_type is asyncio.TimeoutError else 'collector')
    assert all(c.closed for c in contexts)
