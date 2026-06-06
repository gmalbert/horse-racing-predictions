"""Parser hardening tests for T1 TB/QH/H track-site canonical parsers.

These tests use offline synthetic fixtures to validate in-progress track parsing
behavior and avoid live network dependencies.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
FIXTURES = pathlib.Path(__file__).parent / "fixtures"

sys.path.insert(0, str(SCRIPTS))

import parse_t1_tb_tracksite as tb_parser
import parse_t1_qh_tracksite as qh_parser
import parse_t1_h_tracksite as h_parser


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
