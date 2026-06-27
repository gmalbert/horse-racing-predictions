"""
US Horse Racing — dedicated Streamlit page.

Tabs:
  1. 📅 Upcoming Schedule    — TVG track/race calendar (today + next 2 days)
  2. 🎲 Today & Tomorrow     — Fetch entries, generate predictions, view results
  3. 📅 Predicted Fixtures   — Coming soon (US graded-stakes calendar)
  4. 🎯 Top Predictive Races — Coming soon (top US stakes ranking)
  5. 📊 Model Insights       — Calibration & feature importance
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
import os
import re

import pandas as pd
import streamlit as st

from shared.utils import (
    BASE_DIR, LOGO_FILE, MODEL_FILE,
    get_now_local, load_model, get_dataframe_height, safe_st_call,
)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Theme (identical to main page) ────────────────────────────────────────────
_THEME_CSS = """
<style>
:root { --primary: #3c5a99; --bg: #eef3fb; --card: #dfe8f5; --text: #152a50;
        --button-text: #152a50; --accent: #6c8dd8; --border: #a7b7d6; }
[data-testid="stAppViewContainer"] { background-color: var(--bg) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background-color: var(--card) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stHeader"] { background-color: var(--bg) !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { color: var(--text) !important; }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] { border-bottom-color: var(--accent) !important; color: var(--accent) !important; }
[data-testid="stButton"] > button { background-color: var(--primary) !important; color: var(--button-text, #fff) !important; border: none !important; border-radius: 6px !important; }
[data-testid="stButton"] > button:hover { filter: brightness(1.15) !important; }
[data-testid="metric-container"] { background-color: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; padding: 10px !important; min-height: 114px !important; }
[data-testid="metric-container"] label { color: var(--accent) !important; }
[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.2 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    line-height: 1.1 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"],
[data-testid="metric-container"] [data-testid="stMetricDelta"] * {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
}
[data-testid="stExpander"] summary { background-color: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 6px !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 6px !important; }
[data-baseweb="select"] { background-color: var(--card) !important; border-color: var(--border) !important; }
[data-baseweb="input"] { background-color: var(--card) !important; }
[data-testid="stSidebarNav"] a { color: var(--text) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { color: var(--accent) !important; font-weight: 700 !important; }
[data-testid="stAlert"] { border-left-color: var(--accent) !important; background-color: var(--card) !important; color: var(--text) !important; }
@media (max-width: 980px) {
    [data-testid="metric-container"] {
        min-height: 126px !important;
        padding: 12px !important;
    }
}
</style>
"""

RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _apply_theme():
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def _render_sidebar_logo():
    if LOGO_FILE.exists():
        encoded = base64.b64encode(LOGO_FILE.read_bytes()).decode()
        st.sidebar.markdown(
            "<div style='display:flex; justify-content:center; padding: 12px 0;'>"
            f"<img src='data:image/png;base64,{encoded}' width='150' style='max-width:150px; height:auto;'/>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown("---")


def _show_file_status(path: Path, label: str):
    """Small green/grey status for a data file."""
    if path.exists():
        mtime = pd.Timestamp.fromtimestamp(path.stat().st_mtime)
        st.success(f"✅ {label}\n{mtime.strftime('%H:%M:%S')}")
    else:
        st.info(f"⬜ {label} — not yet available")


def _safe_tail(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def _render_subprocess_error(
    title: str,
    cmd: list[str],
    result: subprocess.CompletedProcess[str] | None = None,
    exc: Exception | None = None,
):
    """Render detailed subprocess diagnostics for cloud-friendly debugging."""
    st.error(f"❌ {title}")

    cmd_str = subprocess.list2cmdline(cmd)
    lines = [
        f"cwd: {BASE_DIR}",
        f"command: {cmd_str}",
    ]

    if result is not None:
        lines.append(f"returncode: {result.returncode}")

    if exc is not None:
        lines.append(f"exception: {type(exc).__name__}: {exc}")
        if isinstance(exc, subprocess.TimeoutExpired):
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            err = exc.stderr if isinstance(exc.stderr, str) else ""
            if out:
                lines.append("\n--- stdout (tail) ---\n" + _safe_tail(out))
            if err:
                lines.append("\n--- stderr (tail) ---\n" + _safe_tail(err))
    elif result is not None:
        out_tail = _safe_tail(result.stdout)
        err_tail = _safe_tail(result.stderr)
        if out_tail:
            lines.append("\n--- stdout (tail) ---\n" + out_tail)
        if err_tail:
            lines.append("\n--- stderr (tail) ---\n" + err_tail)
        if not out_tail and not err_tail:
            lines.append("\n(no stdout/stderr captured)")

    details = "\n".join(lines)

    log_dir = BASE_DIR / "tmp" / "error_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "error"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{slug}_{timestamp}.log"
    log_file.write_text(details, encoding="utf-8")

    with st.expander("Error details"):
        st.code(details, language="text")
        st.caption(f"Saved diagnostic log: {log_file}")


# ── Pipeline actions ───────────────────────────────────────────────────────────

def _ingest_us_entries(date_str: str, label: str = ""):
    """Run ingest_us_entries.py (TVG fetch + write CSV + merge racecards JSON)."""
    cmd = [sys.executable, "scripts/ingest_us_entries.py", "--date", date_str]
    with st.spinner(f"📡 Fetching US entries from TVG for {label}{date_str}… (~5s)"):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=120,
            )
        except Exception as exc:
            _render_subprocess_error("TVG fetch failed", cmd, exc=exc)
            return
        if result.returncode == 0:
            st.success(f"✅ US entries fetched for {date_str}!")
            st.rerun()
        else:
            _render_subprocess_error("TVG fetch failed", cmd, result=result)


def _generate_us_predictions(date_str: str, label: str = ""):
    """Run predict_us_races.py for *date_str*."""
    rc = RAW_DIR / f"us_racecards_{date_str}.json"
    if not rc.exists():
        st.error(f"❌ No racecards for {date_str} — fetch entries first.")
        return
    cmd = [sys.executable, "scripts/predict_us_races.py", "--date", date_str]
    with st.spinner(f"🤖 Generating US predictions for {label}{date_str}…"):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300,
            )
        except Exception as exc:
            _render_subprocess_error("Prediction generation failed", cmd, exc=exc)
            return
        if result.returncode == 0:
            st.success("✅ US predictions generated!")
            st.balloons()
            st.rerun()
        else:
            _render_subprocess_error("Prediction generation failed", cmd, result=result)


def _betfair_fetch_odds(date_str: str, label: str = ""):
    cmd = [
        sys.executable,
        "scripts/fetch_betfair_us_odds.py",
        "--date",
        date_str,
        "--overlay",
    ]
    with st.spinner(f"💹 Fetching Betfair odds for {label}{date_str}…"):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=120,
            )
        except Exception as exc:
            _render_subprocess_error("Betfair fetch failed", cmd, exc=exc)
            return
        if result.returncode == 0:
            st.success(f"✅ Betfair odds fetched for {date_str}!")
            st.rerun()
        else:
            _render_subprocess_error("Betfair fetch failed", cmd, result=result)


# ── Main page ──────────────────────────────────────────────────────────────────

def us_racing_page():
    _apply_theme()
    _render_sidebar_logo()
    st.title("🇺🇸 US Horse Racing")

    tab_schedule, tab_today, tab_fixtures, tab_top, tab_insights = st.tabs([
        "📅 Upcoming Schedule",
        "🎲 Today & Tomorrow",
        "📅 Predicted Fixtures",
        "🎯 Top Predictive Races",
        "📊 Model Insights",
    ])

    with tab_schedule:
        _tab_schedule()

    with tab_today:
        _tab_today_tomorrow()

    with tab_fixtures:
        _tab_fixtures()

    with tab_top:
        _tab_top_races()

    with tab_insights:
        _tab_model_insights()

    from footer import add_betting_oracle_footer
    add_betting_oracle_footer()


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Upcoming Schedule
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_schedule():
    """Show TVG track/race schedule for today and the next 2 days."""
    st.subheader("📅 Upcoming US Race Schedule")
    st.caption(
        "Data sourced from TVG GraphQL API — covers all US thoroughbred, harness, "
        "and quarter-horse tracks. TVG typically publishes entries 1–2 days ahead."
    )

    tz_name = os.environ.get("APP_TIMEZONE")
    now_local = get_now_local(tz_name)
    dates = [
        (now_local + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(3)
    ]
    labels = ["Today", "Tomorrow", "Day After"]

    for date_str, label in zip(dates, labels):
        entries_csv = PROC_DIR / f"us_entries_{date_str}.csv"
        rc_json     = RAW_DIR  / f"us_racecards_{date_str}.json"

        with st.expander(f"**{label} — {date_str}**", expanded=(label == "Today")):
            c_fetch, c_status = st.columns([3, 2])
            with c_fetch:
                if st.button(f"📡 Fetch {label}'s Entries", key=f"sched_fetch_{date_str}"):
                    _ingest_us_entries(date_str, f"{label.lower()}'s ")
            with c_status:
                if entries_csv.exists():
                    mtime = pd.Timestamp.fromtimestamp(entries_csv.stat().st_mtime)
                    st.success(f"✅ Entries cached — {mtime.strftime('%H:%M:%S')}")
                else:
                    st.info("⬜ No cached entries yet")

            if entries_csv.exists():
                try:
                    df = pd.read_csv(entries_csv)
                    # Summary metrics
                    mr1c1, mr1c2 = st.columns(2)
                    mr2c1, mr2c2 = st.columns(2)
                    mr1c1.metric("Tracks",  df["track_code"].nunique())
                    mr1c2.metric("Races",   df["race_number"].nunique() if "race_number" in df.columns else "—")
                    mr2c1.metric("Entries", len(df))
                    scratches = int(df["scratched"].sum()) if "scratched" in df.columns else 0
                    mr2c2.metric("Scratches", scratches)

                    # Track grid
                    st.markdown("##### Tracks racing today")
                    track_summary = (
                        df.groupby(["track_code", "track_name"])
                        .agg(
                            Races=("race_number", "nunique"),
                            Entries=("runner_name", "count"),
                            Surface=("surface", lambda x: x.mode().iloc[0] if not x.mode().empty else "—"),
                        )
                        .reset_index()
                        .rename(columns={"track_code": "Code", "track_name": "Track"})
                        .sort_values("Code")
                    )
                    safe_st_call(
                        st.dataframe, track_summary,
                        hide_index=True, width="stretch",
                        height=get_dataframe_height(track_summary, max_height=400),
                    )

                    # Race timeline
                    if "race_time" in df.columns and "race_number" in df.columns:
                        with st.expander("🕐 Race times by track", expanded=False):
                            timeline = (
                                df.drop_duplicates(["track_code", "race_number"])
                                  [["track_code", "race_number", "race_time", "race_name", "surface", "distance"]]
                                  .sort_values(["track_code", "race_number"])
                                  .rename(columns={
                                      "track_code": "Track", "race_number": "Race #",
                                      "race_time": "Post Time", "race_name": "Name",
                                      "surface": "Surface", "distance": "Distance",
                                  })
                            )
                            safe_st_call(
                                st.dataframe, timeline,
                                hide_index=True, width="stretch",
                                height=get_dataframe_height(timeline, max_height=500),
                            )
                except Exception as exc:
                    st.warning(f"Could not parse entries file: {exc}")
            else:
                st.info(
                    f"Fetch {label.lower()}'s entries above to see the schedule. "
                    "TVG entries are typically available by 09:00 ET on race day."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Today & Tomorrow
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_today_tomorrow():
    """Fetch US entries, generate predictions, display results."""
    st.subheader("🎲 US Predictions — Today & Tomorrow")
    st.caption(
        "Entries from TVG GraphQL API • Win/Place/Show probabilities from XGBoost model "
        "• Morning-line odds shown alongside model fair-value odds"
    )

    tz_name = os.environ.get("APP_TIMEZONE")
    now_local = get_now_local(tz_name)
    today_str    = now_local.strftime("%Y-%m-%d")
    tomorrow_str = (now_local + timedelta(days=1)).strftime("%Y-%m-%d")

    today_pf = PROC_DIR / f"us_predictions_{today_str}.csv"
    tmrw_pf  = PROC_DIR / f"us_predictions_{tomorrow_str}.csv"

    today_rc  = RAW_DIR / f"us_racecards_{today_str}.json"
    tmrw_rc   = RAW_DIR / f"us_racecards_{tomorrow_str}.json"
    today_bf  = RAW_DIR / f"betfair_us_odds_{today_str}.json"
    tmrw_bf   = RAW_DIR / f"betfair_us_odds_{tomorrow_str}.json"

    has_betfair = bool(os.environ.get("BETFAIR_USERNAME"))

    # ── Step 1: Fetch entries ────────────────────────────────────────────────
    today_needs_data    = not today_pf.exists()
    tomorrow_needs_data = not tmrw_pf.exists()

    if today_needs_data or tomorrow_needs_data:
        st.markdown("### Step 1: Fetch US Entries (TVG)")
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            if today_needs_data:
                if st.button("📡 Fetch Today's Entries", key="us2_fetch_today"):
                    _ingest_us_entries(today_str, "today's ")
            _show_file_status(today_rc, f"Racecards {today_str}")
        with c2:
            if tomorrow_needs_data:
                if st.button("📡 Fetch Tomorrow's Entries", key="us2_fetch_tmrw"):
                    _ingest_us_entries(tomorrow_str, "tomorrow's ")
            _show_file_status(tmrw_rc, f"Racecards {tomorrow_str}")
        with c3:
            st.info(
                "TVG entries are fetched live and cached in `data/raw/`. "
                "Re-fetch to pick up late scratches or added runners."
            )
        st.markdown("---")

        # ── Step 2: Generate predictions ────────────────────────────────────
        st.markdown("### Step 2: Generate Predictions")
        g1, g2, g3 = st.columns([2, 2, 1])
        with g1:
            if today_needs_data:
                if st.button("🔄 Generate Today's Predictions", type="primary", key="us2_gen_today"):
                    _generate_us_predictions(today_str, "today's ")
        with g2:
            if tomorrow_needs_data:
                if st.button("🔄 Generate Tomorrow's Predictions", type="primary", key="us2_gen_tmrw"):
                    _generate_us_predictions(tomorrow_str, "tomorrow's ")
        with g3:
            if st.button("🔃 Refresh", key="us2_refresh"):
                st.rerun()
        st.markdown("---")

    # ── Optional: Betfair overlay ────────────────────────────────────────────
    if has_betfair:
        st.markdown("### Betfair Exchange Odds (optional)")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("💹 Fetch Today's Betfair Odds", key="us2_bf_today"):
                _betfair_fetch_odds(today_str, "today's ")
            _show_file_status(today_bf, f"Betfair odds {today_str}")
        with b2:
            if st.button("💹 Fetch Tomorrow's Betfair Odds", key="us2_bf_tmrw"):
                _betfair_fetch_odds(tomorrow_str, "tomorrow's ")
            _show_file_status(tmrw_bf, f"Betfair odds {tomorrow_str}")
        st.markdown("---")
    else:
        with st.expander("💹 Betfair Exchange odds (configure to enable)", expanded=False):
            st.info(
                "Set `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, and `BETFAIR_APP_KEY` in `.env` "
                "to enable live Betfair odds overlay. "
                "Note: Betfair Exchange is geo-restricted — UK/Ireland/EU access required."
            )

    # ── Load predictions ─────────────────────────────────────────────────────
    frames = []
    for date_str, day_label, pf in [
        (today_str, "Today", today_pf),
        (tomorrow_str, "Tomorrow", tmrw_pf),
    ]:
        if pf.exists():
            df = pd.read_csv(pf)
            df["day_label"] = day_label
            df["date"] = date_str
            frames.append(df)

    if not frames:
        st.info("📅 No US predictions yet. Use the steps above to fetch and generate.")
        return

    preds = pd.concat(frames, ignore_index=True)

    # ── Day tabs ─────────────────────────────────────────────────────────────
    day_tab_today, day_tab_tmrw = st.tabs(["📅 Today", "📅 Tomorrow"])
    for day_label, tab_obj in [("Today", day_tab_today), ("Tomorrow", day_tab_tmrw)]:
        with tab_obj:
            day_df = preds[preds["day_label"] == day_label].copy()
            if day_df.empty:
                st.info(
                    f"No predictions for {day_label.lower()}. "
                    "Use the steps above to fetch entries and generate predictions."
                )
                continue
            _display_day_predictions(day_df, day_label, key_prefix=f"us_{day_label.lower()}")


def _display_day_predictions(day_df: pd.DataFrame, day_label: str, key_prefix: str):
    """Display all predictions for one day — summary, top picks, race-by-race."""
    date_val = day_df["date"].iloc[0] if "date" in day_df.columns else ""
    courses  = day_df["course"].nunique() if "course" in day_df.columns else 0
    races    = (
        day_df[["race_time", "course"]].fillna("").drop_duplicates().shape[0]
    )
    overlay_msg = ""
    if "betfair_back_odds" in day_df.columns:
        n_bf = int(day_df["betfair_back_odds"].notna().sum())
        overlay_msg = f"  |  💹 {n_bf} Betfair odds overlaid"

    st.success(
        f"✅ {day_label} ({date_val}): {len(day_df)} horses across "
        f"{races} races at {courses} track(s){overlay_msg}"
    )

    # Model provenance
    if "source_model" in day_df.columns and day_df["source_model"].notna().any():
        model_name = str(day_df["source_model"].dropna().iloc[0])
        artifact   = str(day_df["model_artifact"].dropna().iloc[0]) if "model_artifact" in day_df.columns else "—"
        feat_count = None
        if "model_feature_count" in day_df.columns and day_df["model_feature_count"].notna().any():
            try:
                feat_count = int(float(day_df["model_feature_count"].dropna().iloc[0]))
            except Exception:
                pass
        feat_msg = f" | Features: {feat_count}" if feat_count else ""
        st.info(f"🧠 Model: {model_name} ({artifact}){feat_msg}")

    st.markdown("---")

    # Track breakdown
    with st.expander("📊 Track breakdown", expanded=False):
        track_grp = (
            day_df.groupby("course")
            .agg(Races=("race_time", "nunique"), Horses=("horse", "count"))
            .reset_index()
            .rename(columns={"course": "Track"})
            .sort_values("Track")
        )
        safe_st_call(st.dataframe, track_grp, hide_index=True, width="stretch",
                     height=get_dataframe_height(track_grp, max_height=350))

    # Top picks
    st.markdown("##### 🏆 Top Picks")
    _display_top_picks_table(day_df)

    st.markdown("---")

    # Race-by-race
    _display_race_by_race(day_df, key_prefix)


def _display_top_picks_table(preds: pd.DataFrame):
    """Top 25 horses sorted by win probability."""
    wanted = [
        "day_label", "date", "race_time", "course", "race_name",
        "horse", "jockey", "win_probability", "place_probability",
        "show_probability", "win_odds_fractional", "win_odds_decimal",
        "ml_odds", "surface", "distance_str", "race_class",
    ]
    display_cols = [c for c in wanted if c in preds.columns]

    top = (
        preds.sort_values(["date", "win_probability"], ascending=[True, False])
             .head(25)
    )[display_cols].copy()

    for col in ["win_probability", "place_probability", "show_probability"]:
        if col in top.columns:
            top[col] = top[col].apply(lambda x: f"{x:.1%}")

    rename = {
        "day_label": "Day", "date": "Date", "race_time": "Time",
        "course": "Track", "race_name": "Event", "horse": "Horse",
        "jockey": "Jockey", "win_probability": "Win %",
        "place_probability": "Place %", "show_probability": "Show %",
        "win_odds_fractional": "Fair Win Odds", "win_odds_decimal": "Decimal",
        "ml_odds": "ML Odds", "surface": "Surface",
        "distance_str": "Distance", "race_class": "Class",
    }
    top.rename(columns={k: v for k, v in rename.items() if k in top.columns}, inplace=True)
    safe_st_call(st.dataframe, top, hide_index=True, width="stretch",
                 height=get_dataframe_height(top))


def _display_race_by_race(preds: pd.DataFrame, key_prefix: str = "us"):
    """Race selector + full detailed race breakdown."""
    st.markdown("##### 📋 Race-by-Race")

    group_cols = [c for c in ["date", "day_label", "race_time", "course", "race_name"]
                  if c in preds.columns]
    races = preds.groupby(group_cols, dropna=False).size().reset_index()[group_cols]

    if races.empty:
        st.info("No races to display.")
        return

    race_options = []
    for _, row in races.iterrows():
        time_val   = row.get("race_time", "")
        course_val = row.get("course", "")
        day_val    = row.get("day_label", "")
        name_val   = row.get("race_name", "")
        label = f"{day_val} | {time_val} | {course_val}"
        if name_val and str(name_val) != "nan":
            label += f" — {str(name_val)[:40]}"
        race_options.append((label, row))

    selected_label = st.selectbox(
        "Select a race:",
        [r[0] for r in race_options],
        key=f"{key_prefix}_race_select",
    )
    selected_row = next(r[1] for r in race_options if r[0] == selected_label)

    mask = pd.Series(True, index=preds.index)
    for col in group_cols:
        if col in selected_row.index:
            val = selected_row[col]
            if pd.isna(val):
                mask &= preds[col].isna()
            else:
                mask &= preds[col] == val

    race_df = preds[mask].copy().sort_values("win_probability", ascending=False)

    if race_df.empty:
        st.warning("No runners found for this race.")
        return

    _display_race_detail(race_df, key_prefix)


def _display_race_detail(race_df: pd.DataFrame, key_prefix: str):
    """Full race breakdown — info header, top picks, exacta/trifecta, horse table, value calc."""
    # ── Race header ──────────────────────────────────────────────────────────
    surface    = race_df["surface"].iloc[0]    if "surface"    in race_df.columns else "—"
    going      = race_df["going"].iloc[0]      if "going"      in race_df.columns else "—"
    dist_str   = race_df["distance_str"].iloc[0] if "distance_str" in race_df.columns else ""
    dist_band  = race_df["distance_band"].iloc[0] if "distance_band" in race_df.columns else ""
    race_class = race_df["race_class"].iloc[0] if "race_class" in race_df.columns else "—"

    hr1, hr2 = st.columns(2)
    hr3, hr4 = st.columns(2)
    hr1.metric("Surface",  surface or "—")
    hr2.metric("Going",    going   or "—")
    hr3.metric("Distance", f"{dist_str} ({dist_band})" if dist_str else dist_band or "—")
    hr4.metric("Class",    f"{race_class} ({len(race_df)} runners)")

    st.markdown("##### 🏆 Top Picks")
    col1, col2 = st.columns(2)
    col3 = st.container()
    top3 = race_df.head(3)
    col1.metric("Win favourite",   top3.iloc[0]["horse"] if len(top3) > 0 else "—",
                f"{top3.iloc[0]['win_probability']:.1%}" if len(top3) > 0 else "")
    col2.metric("Place favourite", top3.iloc[1]["horse"] if len(top3) > 1 else "—",
                f"{top3.iloc[1]['win_probability']:.1%}" if len(top3) > 1 else "")
    col3.metric("Each-way pick",   top3.iloc[2]["horse"] if len(top3) > 2 else "—",
                f"{top3.iloc[2]['win_probability']:.1%}" if len(top3) > 2 else "")

    # ── Exacta / Trifecta ────────────────────────────────────────────────────
    if len(top3) >= 3:
        p1 = float(top3.iloc[0]["win_probability"])
        p2 = float(top3.iloc[1]["win_probability"])
        p3 = float(top3.iloc[2]["win_probability"])

        exacta_prob   = p1 * (p2 / (1 - p1)) if p1 < 0.99 else 0
        tri_denom     = 1 - p1 - p2
        trifecta_prob = exacta_prob * (p3 / tri_denom) if tri_denom > 0.01 else 0

        st.markdown("---")
        st.markdown("##### 🎯 Exacta / Trifecta Probabilities")
        e1, e2 = st.columns(2)
        e3 = st.container()
        e1.metric("🥇🥈 Exacta (1-2 in order)", f"{exacta_prob:.1%}",
                  help=f"{top3.iloc[0]['horse']} → {top3.iloc[1]['horse']}")
        e2.metric("🥇🥈🥉 Trifecta (1-2-3)", f"{trifecta_prob:.1%}",
                  help=f"{top3.iloc[0]['horse']} → {top3.iloc[1]['horse']} → {top3.iloc[2]['horse']}")
        if trifecta_prob > 0:
            e3.metric("💰 Fair Trifecta Odds", f"{(1/trifecta_prob)-1:.1f}/1")
        else:
            e3.metric("💰 Fair Trifecta Odds", "N/A")

        st.caption(
            f"🎯 Predicted 1-2-3: **{top3.iloc[0]['horse']}** → "
            f"**{top3.iloc[1]['horse']}** → **{top3.iloc[2]['horse']}**"
        )

        # Cumulative probabilities for the win favourite
        st.markdown("##### 🏆 Win Favourite — Cumulative Probabilities")
        prob_win   = p1
        prob_top2  = min(float(top3.iloc[0].get("place_probability", prob_win * 1.5)), 0.95)
        prob_top3  = min(float(top3.iloc[0].get("show_probability",  prob_top2 * 1.2)), 0.98)
        inc_place  = prob_top2 - prob_win
        inc_show   = prob_top3 - prob_top2
        cp1, cp2 = st.columns(2)
        cp3 = st.container()
        cp1.metric("🥇 Win (1st)",        f"{prob_win:.1%}")
        cp2.metric("🥇🥈 Win or Place",   f"{prob_top2:.1%} (+{inc_place:.1%})")
        cp3.metric("🥇🥈🥉 Win/Place/Show", f"{prob_top3:.1%} (+{inc_show:.1%})")

    # ── All horses table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 🐎 All Runners")
    _display_all_horses_table(race_df)

    # ── Value bet calculator ─────────────────────────────────────────────────
    _display_value_bet_calculator(race_df, key_prefix)


def _display_all_horses_table(race_df: pd.DataFrame):
    """Ranked runners table with model odds and optional market overlay."""
    base_cols = [
        "horse", "jockey", "trainer", "age", "weight_lbs", "form", "draw", "ml_odds",
        "win_probability", "win_odds_fractional", "win_odds_decimal",
        "place_probability", "place_odds_fractional",
        "show_probability", "show_odds_fractional",
        "sire", "dam",
    ]
    display = race_df[[c for c in base_cols if c in race_df.columns]].copy()

    display["top_2_prob"] = race_df["win_probability"] + race_df["place_probability"]
    display["top_3_prob"] = race_df["win_probability"] + race_df["place_probability"] + race_df["show_probability"]
    display["win_rank"] = race_df["win_probability"].rank(ascending=False, method="min").astype(int)

    # Optional market overlay columns
    has_mkt   = "betfair_back_odds" in race_df.columns
    has_edge  = "betfair_value_edge" in race_df.columns
    if has_mkt:
        display["betfair_back_odds"] = race_df["betfair_back_odds"]
    if has_edge:
        display["betfair_value_edge"] = race_df["betfair_value_edge"]

    for col in ["win_probability", "place_probability", "show_probability"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.1%}")
    display["top_2_prob"] = display["top_2_prob"].apply(lambda x: f"{x:.1%}")
    display["top_3_prob"] = display["top_3_prob"].apply(lambda x: f"{x:.1%}")
    if has_edge and "betfair_value_edge" in display.columns:
        display["betfair_value_edge"] = display["betfair_value_edge"].apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "—"
        )

    ordered = [
        "win_rank", "horse", "jockey", "trainer",
        "win_probability", "win_odds_fractional", "win_odds_decimal",
        "place_probability", "place_odds_fractional",
        "show_probability", "show_odds_fractional",
        "top_2_prob", "top_3_prob",
        "ml_odds", "betfair_back_odds", "betfair_value_edge",
        "age", "weight_lbs", "form", "draw", "sire", "dam",
    ]
    display = display[[c for c in ordered if c in display.columns]].sort_values("win_rank")

    rename = {
        "win_rank": "#", "horse": "Horse", "jockey": "Jockey", "trainer": "Trainer",
        "win_probability": "Win %", "win_odds_fractional": "Fair Win Odds",
        "win_odds_decimal": "Dec. Odds",
        "place_probability": "Place %", "place_odds_fractional": "Fair Place Odds",
        "show_probability": "Show %",  "show_odds_fractional": "Fair Show Odds",
        "top_2_prob": "Top 2 %", "top_3_prob": "Top 3 %",
        "ml_odds": "ML Odds", "betfair_back_odds": "BF Back",
        "betfair_value_edge": "BF Edge",
        "age": "Age", "weight_lbs": "Wt (lbs)", "form": "Form",
        "draw": "Gate", "sire": "Sire", "dam": "Dam",
    }
    display.rename(columns={k: v for k, v in rename.items() if k in display.columns}, inplace=True)
    safe_st_call(st.dataframe, display, hide_index=True, width="stretch",
                 height=get_dataframe_height(display))
    st.caption(
        "💡 **US Betting tip**: Compare Fair Win Odds with ADW prices at TwinSpires, "
        "FanDuel Racing, or DraftKings. Positive BF Edge = model favours the horse vs market."
    )


def _display_value_bet_calculator(race_df: pd.DataFrame, key_prefix: str):
    """Value bet calculator pre-populated with Betfair odds when available."""
    st.markdown("---")
    has_bf = "betfair_back_odds" in race_df.columns
    expander_label = (
        "🧮 Live Betfair Odds: Value Analysis (pre-populated)"
        if has_bf else
        "🧮 Value Bet Calculator (enter ADW / tote odds)"
    )

    with st.expander(expander_label, expanded=has_bf):
        if not has_bf:
            st.caption(
                "Enter the decimal odds available at TwinSpires, FanDuel Racing, "
                "or DraftKings to check whether there is value vs the model price."
            )

        horse_list = race_df["horse"].tolist()
        sel_horse  = st.selectbox("Select horse", horse_list, key=f"{key_prefix}_vb_horse")
        horse_row  = race_df[race_df["horse"] == sel_horse].iloc[0]

        model_prob = float(horse_row["win_probability"])
        model_dec  = 1 / model_prob if model_prob > 0 else 99.0
        model_frac = horse_row.get("win_odds_fractional", "—")

        default_odds = float(horse_row["betfair_back_odds"]) if (
            has_bf and pd.notna(horse_row.get("betfair_back_odds"))
        ) else round(model_dec, 1)

        bookie_odds = st.number_input(
            "Bookmaker / ADW decimal odds",
            min_value=1.01, max_value=999.0,
            value=default_odds, step=0.1,
            key=f"{key_prefix}_vb_odds",
            help="Pre-filled from Betfair if overlaid, otherwise enter manually."
                 if has_bf else "Enter the decimal odds (e.g. 4.0 = 3/1).",
        )

        bookie_implied = 1 / bookie_odds
        edge = model_prob - bookie_implied

        st.markdown("---")
        v1, v2 = st.columns(2)
        v3, v4 = st.columns(2)
        v1.metric("Model Win %",      f"{model_prob:.1%}")
        v2.metric("Model Fair Odds",  f"{model_frac} ({model_dec:.2f})")
        v3.metric("Bookie Implied %", f"{bookie_implied:.1%}")
        v4.metric("Edge", f"{edge:+.1%}", delta=f"{edge:+.1%}",
                  delta_color="normal" if edge > 0 else "inverse")

        st.markdown("---")
        if edge >= 0.05:
            st.success(f"✅ **VALUE BET!** Edge: {edge:+.1%}  — Back **{sel_horse}**")
        elif edge >= 0.02:
            st.info(f"⚖️ **Marginal value** — Edge: {edge:+.1%}")
        elif edge >= -0.02:
            st.warning(f"📊 **Fair odds** — Edge: {edge:+.1%}")
        else:
            st.error(f"❌ **No value** — Edge: {edge:+.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Predicted Fixtures (stub)
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_fixtures():
    """
    Predicted Fixtures tab — two panels:
      • ABR graded-stakes calendar (full year, G1/G2/G3)
      • Equibase entries for the next 7 days (runner-level detail)
    """
    st.subheader("📅 US Predicted Fixtures")
    st.caption(
        "Graded-stakes calendar from **America's Best Racing** (full year · G1/G2/G3). "
        "Entry-level detail from **Equibase** (free static HTML, up to 7 days ahead). "
        "No subscription required."
    )

    tz_name = os.environ.get("APP_TIMEZONE")
    now_local = get_now_local(tz_name)
    year = now_local.year
    today_str = now_local.strftime("%Y-%m-%d")

    # ── Section 1: ABR Graded-Stakes Calendar ────────────────────────────────
    st.markdown("### 🏆 Graded-Stakes Calendar (America's Best Racing)")

    abr_json  = RAW_DIR  / f"abr_stakes_{year}.json"
    abr_csv   = PROC_DIR / f"abr_stakes_{year}.csv"

    c1, c2 = st.columns([3, 2])
    with c1:
        col_fetch, col_force = st.columns([2, 1])
        with col_fetch:
            if st.button("📥 Fetch Stakes Calendar", key="abr_fetch"):
                _fetch_abr_stakes(year, force=False)
        with col_force:
            if st.button("🔄 Re-fetch", key="abr_force", help="Force re-fetch even if cached"):
                _fetch_abr_stakes(year, force=True)
    with c2:
        if abr_json.exists():
            mtime = pd.Timestamp.fromtimestamp(abr_json.stat().st_mtime)
            st.success(f"✅ Cached — {mtime.strftime('%d %b %Y %H:%M')}")
        else:
            st.info("⬜ Not yet fetched")

    if abr_json.exists():
        try:
            with abr_json.open(encoding="utf-8") as f:
                abr_data = json.load(f)
            stakes = abr_data.get("stakes", [])
            if stakes:
                df_abr = pd.DataFrame(stakes)

                # Ensure date column is sortable
                df_abr["date"] = pd.to_datetime(df_abr["date"], errors="coerce")
                df_abr = df_abr.sort_values(["date", "grade"], na_position="last")

                # Summary metrics
                m1, m2 = st.columns(2)
                m3, m4 = st.columns(2)
                future = df_abr[df_abr["date"] >= pd.Timestamp(today_str)]
                m1.metric("Total Stakes", len(df_abr))
                m2.metric("Upcoming", len(future))
                m3.metric("G1 Remaining", int((future["grade"] == "G1").sum()))
                m4.metric("G2/G3 Remaining",
                          int(((future["grade"] == "G2") | (future["grade"] == "G3")).sum()))

                # Grade filter
                grade_opts = ["All"] + sorted(df_abr["grade"].dropna().unique().tolist())
                time_opts  = ["All upcoming", "Next 30 days", "Next 90 days", "Full year"]
                fa, fb = st.columns([2, 2])
                with fa:
                    sel_grade = st.selectbox("Filter by grade", grade_opts, key="abr_grade")
                with fb:
                    sel_window = st.selectbox("Time window", time_opts, key="abr_window")

                view = df_abr.copy()
                if sel_grade != "All":
                    view = view[view["grade"] == sel_grade]
                if sel_window == "Next 30 days":
                    cutoff = pd.Timestamp(today_str) + pd.Timedelta(days=30)
                    view = view[(view["date"] >= pd.Timestamp(today_str)) & (view["date"] <= cutoff)]
                elif sel_window == "Next 90 days":
                    cutoff = pd.Timestamp(today_str) + pd.Timedelta(days=90)
                    view = view[(view["date"] >= pd.Timestamp(today_str)) & (view["date"] <= cutoff)]
                elif sel_window == "All upcoming":
                    view = view[view["date"] >= pd.Timestamp(today_str)]

                # Format for display
                disp = view[["date", "race_name", "track", "grade", "purse", "distance", "surface"]].copy()
                disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
                disp["purse"] = disp["purse"].apply(
                    lambda x: f"${int(x):,}" if pd.notna(x) and x > 0 else "—"
                )
                disp = disp.rename(columns={
                    "date": "Date", "race_name": "Race", "track": "Track",
                    "grade": "Grade", "purse": "Purse", "distance": "Distance",
                    "surface": "Surface",
                })
                safe_st_call(
                    st.dataframe, disp, hide_index=True, width="stretch",
                    height=get_dataframe_height(disp, max_height=500),
                )
            else:
                st.warning("Cache file exists but contains no stakes. Try re-fetching.")
        except Exception as exc:
            st.warning(f"Could not load ABR stakes data: {exc}")
    else:
        st.info(
            "Click **Fetch Stakes Calendar** to download the full-year graded-stakes schedule "
            "from America's Best Racing. No account or API key required."
        )

    st.markdown("---")

    # ── Section 2: Equibase Upcoming Entries ────────────────────────────────
    st.markdown("### 📋 Upcoming Entries (Equibase — up to 7 days)")
    st.caption(
        "Equibase publishes free static entry pages for each track. "
        "Coverage: Churchill Downs, Belmont, Saratoga, Keeneland, Santa Anita, Del Mar, "
        "Gulfstream, Oaklawn, Fair Grounds, Monmouth, Parx, and more."
    )

    with st.expander("⚠️ Bot-protection note", expanded=False):
        st.info(
            "Equibase uses Imperva bot-protection which may block plain HTTP requests. "
            "If a track returns 0 entries, install Playwright for browser-based fetching: "
            "`pip install playwright && playwright install chromium`. "
            "The scraper will automatically fall back to Playwright when available."
        )

    upcoming_dates = [
        (now_local + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(7)
    ]
    date_labels = ["Today", "Tomorrow"] + [
        (now_local + timedelta(days=d)).strftime("%a %d %b")
        for d in range(2, 7)
    ]

    eq_fetch_col, eq_all_col = st.columns([3, 2])
    with eq_fetch_col:
        if st.button("📥 Fetch All 7 Days (Equibase)", key="eq_fetch_all"):
            for d_str, d_label in zip(upcoming_dates, date_labels):
                _fetch_equibase_entries(d_str, d_label)
    with eq_all_col:
        cached = sum(1 for d in upcoming_dates
                     if (RAW_DIR / f"equibase_entries_{d}.json").exists())
        st.info(f"📦 {cached}/7 days cached")

    # Per-day expanders
    for d_str, d_label in zip(upcoming_dates, date_labels):
        eq_json = RAW_DIR  / f"equibase_entries_{d_str}.json"
        eq_csv  = PROC_DIR / f"equibase_entries_{d_str}.csv"

        with st.expander(f"**{d_label} — {d_str}**", expanded=(d_str == today_str)):
            btn_col, stat_col = st.columns([3, 2])
            with btn_col:
                if st.button(f"📥 Fetch {d_label}", key=f"eq_fetch_{d_str}"):
                    _fetch_equibase_entries(d_str, d_label)
            with stat_col:
                if eq_json.exists():
                    mtime = pd.Timestamp.fromtimestamp(eq_json.stat().st_mtime)
                    st.success(f"✅ Cached {mtime.strftime('%H:%M:%S')}")
                else:
                    st.info("⬜ Not cached")

            if eq_json.exists():
                try:
                    with eq_json.open(encoding="utf-8") as f:
                        eq_data = json.load(f)
                    entries = eq_data.get("entries", [])
                    if entries:
                        df_eq = pd.DataFrame(entries)
                        # Summary
                        s1, s2 = st.columns(2)
                        s3, s4 = st.columns(2)
                        s1.metric("Tracks", df_eq["track_code"].nunique() if "track_code" in df_eq.columns else "—")
                        s2.metric("Races",  df_eq["race_number"].nunique() if "race_number" in df_eq.columns else "—")
                        s3.metric("Entries", len(df_eq))
                        scratches = int(df_eq["scratched"].sum()) if "scratched" in df_eq.columns else 0
                        s4.metric("Scratches", scratches)

                        # Stakes highlight — races where purse > 0 or grade-like race_class
                        if "purse" in df_eq.columns or "race_class" in df_eq.columns:
                            stakes_races = df_eq[
                                df_eq.get("race_class", pd.Series(dtype=str)).str.contains(
                                    r"G[123]|Stakes|Graded", na=False, regex=True
                                ) |
                                (df_eq.get("purse", pd.Series(dtype=str)).str.replace(r"[,$]", "", regex=True)
                                 .apply(lambda x: int(x) if str(x).isdigit() else 0) > 50_000)
                            ].drop_duplicates(subset=["track_code", "race_number"])
                            if not stakes_races.empty:
                                st.markdown("**🏆 Stakes / Rich handicaps this day:**")
                                stakes_disp = stakes_races[
                                    [c for c in ["track_code", "track_name", "race_number",
                                                 "race_time", "race_name", "race_class",
                                                 "purse", "distance", "surface"]
                                     if c in stakes_races.columns]
                                ].sort_values(["track_code", "race_number"])
                                safe_st_call(
                                    st.dataframe, stakes_disp,
                                    hide_index=True, width="stretch",
                                    height=get_dataframe_height(stakes_disp, max_height=300),
                                )

                        # Full entries table
                        with st.expander("All entries", expanded=False):
                            show_cols = [c for c in [
                                "track_code", "track_name", "race_number", "race_time",
                                "race_name", "race_class", "surface", "distance", "purse",
                                "program_number", "runner_name", "jockey", "trainer",
                                "ml_odds", "scratched",
                            ] if c in df_eq.columns]
                            safe_st_call(
                                st.dataframe, df_eq[show_cols],
                                hide_index=True, width="stretch",
                                height=get_dataframe_height(df_eq, max_height=600),
                            )
                    else:
                        ok = eq_data.get("tracks_ok", [])
                        failed = eq_data.get("tracks_failed", [])
                        st.warning(
                            f"Cache exists but no entries returned. "
                            f"Tracks OK: {ok or 'none'}. "
                            f"Failed (Imperva block): {failed or 'none'}."
                        )
                except Exception as exc:
                    st.warning(f"Could not load Equibase entries: {exc}")


def _fetch_abr_stakes(year: int, force: bool = False):
    """Run fetch_abr_stakes.py and show result."""
    force_flag = ["--force"] if force else []
    label = "Re-fetching" if force else "Fetching"
    cmd = [sys.executable, "scripts/fetch_abr_stakes.py", "--year", str(year)] + force_flag
    with st.spinner(f"📥 {label} ABR stakes calendar for {year}… (uses browser rendering, ~15 s)"):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=90,
            )
        except Exception as exc:
            _render_subprocess_error("ABR fetch failed", cmd, exc=exc)
            return
    if result.returncode == 0:
        st.success(f"✅ ABR stakes calendar for {year} fetched!")
        st.rerun()
    else:
        _render_subprocess_error("ABR fetch failed", cmd, result=result)


def _fetch_equibase_entries(date_str: str, label: str = ""):
    """Run fetch_equibase_entries.py for a single date and show result."""
    cmd = [sys.executable, "scripts/fetch_equibase_entries.py", "--date", date_str]
    with st.spinner(
        f"📥 Fetching Equibase entries for {label} {date_str}… "
        "(up to 3 min — Equibase uses bot-protection that slows requests)"
    ):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300,
            )
        except Exception as exc:
            _render_subprocess_error(f"Equibase fetch failed for {date_str}", cmd, exc=exc)
            return
    if result.returncode == 0:
        st.success(f"✅ Equibase entries fetched for {date_str}!")
        st.rerun()
    else:
        _render_subprocess_error(f"Equibase fetch failed for {date_str}", cmd, result=result)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Top Predictive Races
# ═══════════════════════════════════════════════════════════════════════════════

# Track tier bonus points (Tier 1 = premium racecourses)
_TRACK_TIER: dict[str, int] = {
    # Tier 1
    "Churchill Downs": 20, "Saratoga": 20, "Belmont Park": 20,
    "Keeneland": 20, "Santa Anita Park": 20, "Del Mar": 20,
    # Tier 2
    "Gulfstream Park": 12, "Aqueduct": 12, "Oaklawn Park": 12,
    "Fair Grounds": 12, "Pimlico": 12, "Laurel Park": 12,
    "Monmouth Park": 10, "Parx Racing": 10, "Tampa Bay Downs": 8,
    "Turfway Park": 8, "Horseshoe Indianapolis": 6, "Prairie Meadows": 6,
}
_GRADE_BONUS: dict[str, int] = {"G1": 40, "G2": 25, "G3": 15}

# Purse bonus tiers (USD)
_PURSE_BONUS: list[tuple[float, int]] = [
    (1_000_000, 30),
    (500_000,   20),
    (200_000,   14),
    (100_000,    8),
    (50_000,     4),
]

# Harness-track name fragments (these tracks are standardbred, not thoroughbred)
_HARNESS_COURSES: frozenset[str] = frozenset([
    "yonkers", "harrington", "plainridge", "scioto", "meadowlands",
    "pocono", "tioga", "batavia", "northfield", "hawthorne raceway",
    "pompano", "dover", "cal expo", "hoosier park", "miami valley",
    "lebanon",
])


def _load_us_predictions(date_strs: list[str]) -> pd.DataFrame:
    """Load one or more prediction CSVs and concatenate."""
    frames = []
    for d in date_strs:
        path = PROC_DIR / f"us_predictions_{d}.csv"
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _build_abr_grade_lookup(year: int) -> dict[str, str]:
    """Return {normalised_race_name: grade} from ABR stakes JSON."""
    path = RAW_DIR / f"abr_stakes_{year}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            s["race_name"].lower().strip(): s["grade"]
            for s in data.get("stakes", [])
            if s.get("grade") and s.get("race_name")
        }
    except Exception:
        return {}


def _detect_race_type_ui(course: str, race_name: str, race_class: str) -> str:
    """Classify race type from course/name without needing the race_type column."""
    combined = f"{course} {race_name} {race_class}".lower()
    for fragment in _HARNESS_COURSES:
        if fragment in combined:
            return "Harness"
    if any(k in combined for k in ("pace", "trot", "standardbred")):
        return "Harness"
    if any(k in combined for k in ("quarter horse", "350y", "400y")):
        return "QuarterHorse"
    return "Thoroughbred"


def _purse_bonus(purse: float) -> int:
    """Return score bonus based on purse value."""
    for threshold, bonus in _PURSE_BONUS:
        if purse >= threshold:
            return bonus
    return 0


def _score_races(df: pd.DataFrame, grade_lookup: dict[str, str]) -> pd.DataFrame:
    """
    Aggregate prediction rows to race level and compute a predictability score.

    Score = top_win_prob*40 + margin*30 + field_score*10 + grade_bonus + tier_bonus + purse_bonus
      - top_win_prob: model confidence in the winner (0-1)
      - margin: gap between #1 and #2 predicted probabilities (separation)
      - field_score: 0-1 scaled by how close field size is to optimal (8-12)
      - grade_bonus: 40/25/15 for G1/G2/G3 (from race_class col or ABR lookup)
      - tier_bonus: 0-20 based on track prestige
      - purse_bonus: 0-30 based on purse value
    Harness and QuarterHorse races receive a -15 penalty (UK model less applicable).
    """
    # Use race_class column or ABR lookup to find grade
    def _grade(row):
        rc = str(row.get("race_class", "") or "")
        for g in ("G1", "G2", "G3"):
            if g in rc:
                return g
        return grade_lookup.get(str(row.get("race_name", "") or "").lower().strip(), None)

    group_cols = ["date", "course", "race_time", "race_name", "race_class", "surface", "distance_str"]
    # race_type / purse columns may not exist in older prediction files
    has_race_type = "race_type" in df.columns
    has_purse = "purse" in df.columns

    race_groups = df.groupby(group_cols, sort=False)

    rows = []
    for keys, grp in race_groups:
        date, course, race_time, race_name, race_class, surface, distance_str = keys
        sorted_probs = grp["win_probability"].sort_values(ascending=False).values
        top_prob = float(sorted_probs[0]) if len(sorted_probs) > 0 else 0.0
        second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = top_prob - second_prob
        field_size = len(grp)
        # Optimal field size 8-12; penalise <5 or >16
        if 8 <= field_size <= 12:
            field_score = 1.0
        elif field_size < 5:
            field_score = 0.3
        elif field_size > 16:
            field_score = 0.6
        else:
            field_score = 0.8

        sample_row = grp.iloc[0].to_dict()
        sample_row["race_name"] = race_name
        grade = _grade(sample_row)
        grade_bonus = _GRADE_BONUS.get(grade or "", 0)
        tier_bonus = _TRACK_TIER.get(str(course), 0)

        # Purse bonus
        purse_val = float(grp["purse"].iloc[0]) if has_purse and pd.notna(grp["purse"].iloc[0]) else 0.0
        pb = _purse_bonus(purse_val)

        # Race type (for display and filter)
        if has_race_type:
            race_type = str(grp["race_type"].iloc[0] or "Thoroughbred")
        else:
            race_type = _detect_race_type_ui(str(course), str(race_name), str(race_class))

        # Harness/QH penalty — UK model not applicable
        type_penalty = -15 if race_type in ("Harness", "QuarterHorse") else 0

        score = (top_prob * 40 + margin * 30 + field_score * 10
                 + grade_bonus + tier_bonus + pb + type_penalty)

        top3 = grp.sort_values("win_probability", ascending=False).head(3)
        top_horse = top3.iloc[0]["horse"] if len(top3) > 0 else "—"
        top_horse_odds = top3.iloc[0]["win_odds_fractional"] if len(top3) > 0 else "—"

        rows.append({
            "date": date,
            "course": course,
            "race_time": race_time,
            "race_name": race_name,
            "race_class": race_class,
            "race_type": race_type,
            "surface": surface,
            "distance": distance_str,
            "grade": grade or "—",
            "purse": purse_val,
            "runners": field_size,
            "top_horse": top_horse,
            "top_odds": top_horse_odds,
            "top_win%": round(top_prob * 100, 1),
            "margin%": round(margin * 100, 1),
            "score": round(score, 1),
            "_grp": grp,  # keep reference for drilldown
        })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def _tab_top_races():
    st.subheader("🎯 Top Predictive US Races")
    st.caption(
        "Races ranked by a composite **predictability score**: model confidence in the winner, "
        "separation from the field, field size, track prestige, and race grade."
    )

    tz_name = os.environ.get("APP_TIMEZONE")
    now_local = get_now_local(tz_name)
    today_str = now_local.strftime("%Y-%m-%d")
    year = now_local.year

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
    with c1:
        window_opts = {
            "Today": [today_str],
            "Today + Tomorrow": [
                today_str,
                (now_local + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            ],
            "Next 7 days": [
                (now_local + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
                for d in range(7)
            ],
        }
        sel_window = st.selectbox("Date range", list(window_opts.keys()), key="tr_window")
    with c2:
        grade_filter = st.selectbox(
            "Grade", ["All", "G1 only", "G1 + G2", "G1–G3 (graded only)"],
            key="tr_grade",
        )
    with c3:
        race_type_filter = st.selectbox(
            "Race type",
            ["Thoroughbred only", "All types", "Harness only", "QuarterHorse only"],
            key="tr_racetype",
        )
    with c4:
        surface_filter = st.selectbox(
            "Surface", ["All", "Dirt", "Turf", "Synthetic"],
            key="tr_surface",
        )
    with c5:
        top_n = st.selectbox("Top N", [10, 20, 50], key="tr_topn")

    date_strs = window_opts[sel_window]
    df_all = _load_us_predictions(date_strs)

    if df_all.empty:
        st.warning(
            "No US predictions found for the selected date range. "
            "Go to **Today & Tomorrow** tab to generate predictions first."
        )
        return

    grade_lookup = _build_abr_grade_lookup(year)
    scored = _score_races(df_all, grade_lookup)

    # Apply filters
    if grade_filter == "G1 only":
        scored = scored[scored["grade"] == "G1"]
    elif grade_filter == "G1 + G2":
        scored = scored[scored["grade"].isin(["G1", "G2"])]
    elif grade_filter == "G1–G3 (graded only)":
        scored = scored[scored["grade"].isin(["G1", "G2", "G3"])]

    if race_type_filter == "Thoroughbred only":
        scored = scored[scored["race_type"] == "Thoroughbred"]
    elif race_type_filter == "Harness only":
        scored = scored[scored["race_type"] == "Harness"]
    elif race_type_filter == "QuarterHorse only":
        scored = scored[scored["race_type"] == "QuarterHorse"]

    if surface_filter != "All":
        scored = scored[scored["surface"].str.lower().str.contains(surface_filter.lower(), na=False)]

    if scored.empty:
        st.info("No races match the selected filters.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m5 = st.container()
    m1.metric("Races ranked", len(scored))
    m2.metric("Graded stakes", int((scored["grade"].isin(["G1","G2","G3"])).sum()))
    m3.metric("Tracks", scored["course"].nunique())
    m4.metric("Avg top win%", f"{scored['top_win%'].mean():.1f}%")
    m5.metric("Avg separation", f"{scored['margin%'].mean():.1f}%")

    st.markdown("---")

    # ── Ranked table ──────────────────────────────────────────────────────────
    display_top = scored.head(top_n)
    disp_cols = [c for c in ["rank", "date", "course", "race_time", "race_name",
                              "race_type", "grade", "surface", "distance",
                              "runners", "purse", "top_horse", "top_odds",
                              "top_win%", "margin%", "score"]
                 if c in display_top.columns]
    tbl = display_top[disp_cols].copy()
    tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
    if "purse" in tbl.columns:
        tbl["purse"] = tbl["purse"].apply(
            lambda x: f"${int(x):,}" if pd.notna(x) and float(x) > 0 else "—"
        )
    safe_st_call(
        st.dataframe, tbl, hide_index=True, width="stretch",
        height=get_dataframe_height(tbl, max_height=520),
    )

    # ── Score breakdown legend ────────────────────────────────────────────────
    with st.expander("ℹ️ How the score is calculated", expanded=False):
        st.markdown(
            """
| Component | Max pts | Description |
|-----------|---------|-------------|
| Top win % | 40 | Model confidence in the predicted winner (×40) |
| Separation | 30 | Gap between #1 and #2 predicted probs (×30) |
| Field size | 10 | Optimal 8–12 runners = 10 pts; <5 = 3 pts |
| Grade bonus | 40 | G1=40 / G2=25 / G3=15 |
| Track tier | 20 | Premier tracks (Churchill, Saratoga, Keeneland…) = 20 pts |
| Purse bonus | 30 | ≥$1M=30 / ≥$500k=20 / ≥$200k=14 / ≥$100k=8 / ≥$50k=4 |
| Race type penalty | −15 | Harness and QuarterHorse races (UK model less applicable) |

A G1 at Churchill Downs (purse $1M) where the model has 35% confidence in the
winner with 15% gap to #2 would score: 14 + 4.5 + 10 + 40 + 20 + 30 = **118.5**.
            """
        )

    # ── Per-race drilldown ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Race drilldown")

    race_labels = [
        f"#{int(r['rank'])}  {r['course']} {str(r['date'].strftime('%m/%d') if hasattr(r['date'],'strftime') else r['date'])}  {r['race_name']} ({r['grade']})"
        for _, r in display_top.iterrows()
    ]
    sel_race_label = st.selectbox("Select a race to inspect", race_labels, key="tr_drilldown")
    sel_idx = race_labels.index(sel_race_label)
    sel_row = display_top.iloc[sel_idx]

    grp = sel_row["_grp"]
    grp_sorted = grp.sort_values("win_probability", ascending=False).reset_index(drop=True)

    dr1, dr2 = st.columns(2)
    dr3, dr4 = st.columns(2)
    dr1.metric("Track", sel_row["course"])
    dr2.metric("Race", sel_row["race_name"])
    dr3.metric("Grade", sel_row["grade"])
    dr4.metric("Score", sel_row["score"])

    # ── Confidence spread (win/place/show model variance as proxy) ───────────
    if all(c in grp_sorted.columns for c in ["win_probability", "place_probability", "show_probability"]):
        top_horse_row = grp_sorted.iloc[0]
        w  = float(top_horse_row["win_probability"])
        pl = float(top_horse_row["place_probability"])
        sh = float(top_horse_row["show_probability"])
        spread = sh - w  # range from win to show prob
        conf_label = "High" if spread < 0.12 else ("Medium" if spread < 0.22 else "Low")
        cf1, cf2 = st.columns(2)
        cf3, cf4 = st.columns(2)
        cf1.metric("Top pick win%", f"{w:.1%}")
        cf2.metric("Top pick place%", f"{pl:.1%}")
        cf3.metric("Top pick show%", f"{sh:.1%}")
        cf4.metric("Confidence", conf_label,
                   help="High = tight win/show spread (<12pp); Low = wide spread (>22pp)")

    # ── Market odds edge (Betfair / OddsPortal data if available) ─────────────
    betfair_path = RAW_DIR / f"betfair_us_odds_{sel_row['date'].strftime('%Y-%m-%d') if hasattr(sel_row['date'], 'strftime') else str(sel_row['date'])[:10]}.json"
    oddsportal_path = RAW_DIR / f"oddsportal_us_{sel_row['date'].strftime('%Y-%m-%d') if hasattr(sel_row['date'], 'strftime') else str(sel_row['date'])[:10]}.json"

    market_data: dict[str, float] = {}  # horse_name_lower -> decimal back odds
    for odds_path in (betfair_path, oddsportal_path):
        if odds_path.exists():
            try:
                raw_odds = json.loads(odds_path.read_text(encoding="utf-8"))
                runners_odds = raw_odds if isinstance(raw_odds, list) else raw_odds.get("runners", raw_odds.get("markets", []))
                for r in runners_odds:
                    name = str(r.get("runner_name") or r.get("horse") or r.get("name") or "").lower().strip()
                    back = r.get("back_price") or r.get("best_back") or r.get("decimal_odds")
                    if name and back:
                        try:
                            market_data[name] = float(back)
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass
        if market_data:
            break

    if market_data:
        st.markdown("**📈 Market odds vs model (edge)**")
        edge_rows = []
        for _, hr in grp_sorted.iterrows():
            horse_key = str(hr["horse"]).lower().strip()
            back_dec = market_data.get(horse_key)
            if back_dec and back_dec > 1.0:
                market_impl = 1.0 / back_dec
                model_win   = float(hr["win_probability"])
                edge        = model_win - market_impl
                edge_rows.append({
                    "horse": hr["horse"],
                    "model win%": f"{model_win:.1%}",
                    "market implied%": f"{market_impl:.1%}",
                    "back odds": f"{back_dec:.2f}",
                    "edge": f"{edge:+.1%}",
                    "signal": "✅ Value" if edge >= 0.05 else ("⚖️ Fair" if edge >= -0.02 else "❌ Short"),
                })
        if edge_rows:
            safe_st_call(
                st.dataframe, pd.DataFrame(edge_rows), hide_index=True, width="stretch",
                height=get_dataframe_height(pd.DataFrame(edge_rows), max_height=250),
            )
    else:
        st.caption(
            "_No market odds cached for this date. "
            "Run `scripts/fetch_betfair_us_odds.py` or `scripts/fetch_oddsportal_us.py` "
            "to enable value-bet edge display._"
        )

    st.markdown("---")
    horse_cols = [c for c in [
        "horse", "jockey", "trainer", "age", "weight_lbs",
        "ml_odds", "win_probability", "win_odds_fractional",
        "place_probability", "place_odds_fractional",
        "show_probability", "show_odds_fractional",
        "form", "draw",
    ] if c in grp_sorted.columns]

    horse_disp = grp_sorted[horse_cols].copy()
    horse_disp["win_probability"] = (horse_disp["win_probability"] * 100).round(1).astype(str) + "%"
    if "place_probability" in horse_disp.columns:
        horse_disp["place_probability"] = (horse_disp["place_probability"] * 100).round(1).astype(str) + "%"
    if "show_probability" in horse_disp.columns:
        horse_disp["show_probability"] = (horse_disp["show_probability"] * 100).round(1).astype(str) + "%"

    safe_st_call(
        st.dataframe, horse_disp, hide_index=True, width="stretch",
        height=get_dataframe_height(horse_disp, max_height=400),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Model Insights
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_model_insights():
    st.subheader("📊 Model Insights")
    st.info(
        "US predictions currently use the same XGBoost model trained on UK/Irish "
        "race data. A US-specific model will be trained once sufficient historical "
        "US results have been collected (target: ~50 k races)."
    )

    model, metadata, feature_importance = load_model()

    if model is None:
        st.warning("⚠️ No trained model found.")
        return

    st.success("✅ Model loaded")

    if metadata:
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Type", metadata.get("model_type", "Unknown"))
        col2.metric("Features",   metadata.get("n_features", 0))
        col3.metric("Trained",    str(metadata.get("trained_date", "Unknown"))[:10])

    # Calibration
    st.markdown("---")
    st.markdown("### 📊 Model Calibration")
    cal_model  = BASE_DIR / "models" / "horse_win_predictor_calibrated.pkl"
    cal_metrics_f = BASE_DIR / "models" / "calibration_metrics.json"

    if cal_model.exists():
        st.success("✅ Calibrated model available")
        if cal_metrics_f.exists():
            try:
                cal = json.loads(cal_metrics_f.read_text())
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c1.metric("Brier Score",      f"{cal['metrics']['brier_score_calibrated']:.4f}")
                c2.metric("Calibration Gain", f"+{cal['metrics']['brier_improvement_pct']:.1f}%")
                c3.metric("Cal. Samples",     f"{cal['n_calibration_samples']:,}")
                c4.metric("Calibrated",       str(cal["calibration_date"])[:10])

                with st.expander("📈 Calibration curve", expanded=False):
                    cal_plot = BASE_DIR / "models" / "calibration_plot.png"
                    if cal_plot.exists():
                        st.image(str(cal_plot), width="stretch")
                        st.caption("Points should lie on the diagonal for perfect calibration.")
            except Exception as exc:
                st.warning(f"Could not load calibration metrics: {exc}")
    else:
        st.info("ℹ️ Model not yet calibrated.")

    # Feature importance
    st.markdown("---")
    st.markdown("### 🎯 Feature Importance (top 15)")
    if feature_importance is not None and not feature_importance.empty:
        if "rank" not in feature_importance.columns:
            feature_importance = (
                feature_importance.sort_values("importance", ascending=False)
                                  .reset_index(drop=True)
            )
            feature_importance["rank"] = range(1, len(feature_importance) + 1)

        top15 = feature_importance.head(15)
        if HAS_PLOTLY:
            fig = go.Figure(go.Bar(
                x=top15["importance"],
                y=[f"#{int(r['rank'])} {r['feature']}" for _, r in top15.iterrows()],
                orientation="h",
                marker=dict(color=top15["importance"], colorscale="Viridis", showscale=True),
            ))
            fig.update_layout(
                title="Feature Importance (XGBoost)",
                xaxis_title="Importance Score",
                height=500,
                yaxis={"categoryorder": "total ascending"},
            )
            safe_st_call(st.plotly_chart, fig, width="stretch")
        else:
            st.bar_chart(top15.set_index("feature")["importance"])
    else:
        st.info("Feature importance not available.")

    # Diagnostics
    st.markdown("---")
    st.markdown("### 🔍 Latest Prediction Diagnostics (US)")
    diag_files = sorted(PROC_DIR.glob("us_predictions_*.csv"), reverse=True)
    if diag_files:
        latest = diag_files[0]
        try:
            df_diag = pd.read_csv(latest)
            date_tag = latest.stem.replace("us_predictions_", "")
            d1, d2 = st.columns(2)
            d3, d4 = st.columns(2)
            d1.metric("Date",         date_tag)
            d2.metric("Tracks",       df_diag["course"].nunique() if "course" in df_diag.columns else "—")
            d3.metric("Races",        df_diag[["race_time", "course"]].drop_duplicates().shape[0])
            d4.metric("Horses",       len(df_diag))
            if "win_probability" in df_diag.columns:
                e1, e2, e3 = st.columns(3)
                e1.metric("Mean Win Prob", f"{df_diag['win_probability'].mean():.1%}")
                e2.metric("Max Win Prob",  f"{df_diag['win_probability'].max():.1%}")
                top_per_race = (
                    df_diag.groupby(["course", "race_time"])["win_probability"].max()
                )
                e3.metric("Avg Top Pick", f"{top_per_race.mean():.1%}")
        except Exception as exc:
            st.warning(f"Could not load diagnostics: {exc}")
    else:
        st.info("No US prediction files found. Generate predictions to see diagnostics.")

    # ── US historical accuracy (from accuracy report CSVs) ────────────────────
    st.markdown("---")
    st.markdown("### 🎯 US Prediction Accuracy (historical)")

    accuracy_files = sorted(PROC_DIR.glob("us_accuracy_*.csv"), reverse=True)
    if accuracy_files:
        frames = []
        for f in accuracy_files[:30]:  # last 30 days max
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                pass

        if frames:
            acc_all = pd.concat(frames, ignore_index=True)

            top1_races = acc_all[acc_all["predicted_rank"] == 1]
            top1_acc   = top1_races["top1_correct"].mean() if len(top1_races) > 0 else None
            top3_acc   = (acc_all[acc_all["predicted_rank"] <= 3]["top3_correct"].mean()
                          if "top3_correct" in acc_all.columns else None)

            a1, a2 = st.columns(2)
            a3, a4 = st.columns(2)
            a1.metric("Days tracked", len(accuracy_files[:30]))
            a2.metric("Races evaluated", top1_races.groupby(["course", "race_time"]).ngroups
                       if "race_time" in top1_races.columns else len(top1_races))
            a3.metric("Top-1 accuracy",
                      f"{top1_acc:.1%}" if top1_acc is not None else "—")
            a4.metric("Top-3 accuracy",
                      f"{top3_acc:.1%}" if top3_acc is not None else "—")

            # Per-track breakdown
            if "course" in acc_all.columns and top1_acc is not None:
                track_acc = (
                    top1_races.groupby("course")["top1_correct"]
                    .agg(races="count", top1="mean")
                    .reset_index()
                    .sort_values("top1", ascending=False)
                )
                track_acc["top1"] = (track_acc["top1"] * 100).round(1).astype(str) + "%"
                with st.expander("Per-track Top-1 accuracy", expanded=False):
                    safe_st_call(
                        st.dataframe, track_acc, hide_index=True, width="stretch",
                        height=get_dataframe_height(track_acc, max_height=300),
                    )

            with st.expander("📅 Recent accuracy detail (last 7 days)", expanded=False):
                recent = acc_all[acc_all["predicted_rank"] == 1].tail(200)
                show_cols = [c for c in ["date", "course", "race_time", "race_name",
                                          "horse", "actual_position", "top1_correct",
                                          "win_probability", "win_odds_fractional"]
                              if c in recent.columns]
                safe_st_call(
                    st.dataframe, recent[show_cols], hide_index=True, width="stretch",
                    height=get_dataframe_height(recent, max_height=400),
                )
    else:
        st.info(
            "No accuracy data yet. Run `scripts/fetch_equibase_results.py --date YYYY-MM-DD` "
            "after each race day to build the accuracy history."
        )

    # ── US model readiness ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🇺🇸 US-Specific Model Readiness")

    us_model_paths = [
        BASE_DIR / "models" / "us_horse_model.json",
        BASE_DIR / "models" / "us_horse_model.pkl",
    ]
    us_model_path = next((p for p in us_model_paths if p.exists()), None)
    us_data_paths = [
        PROC_DIR / "us_races_cleaned.parquet",
        PROC_DIR / "all_us_races_cleaned.parquet",
    ]
    us_data_exists = [p for p in us_data_paths if p.exists()]

    r1, r2 = st.columns(2)
    with r1:
        if us_model_path is not None:
            st.success(f"✅ US model (`{us_model_path.name}`) is present")
        else:
            st.warning("⚠️ No US-specific model yet — using UK base model")
            st.caption(
                "Train with: `python scripts/train_us_model.py` "
                "(requires ~50k US race results in `data/processed/us_races_cleaned.parquet`)"
            )
    with r2:
        if us_data_exists:
            for p in us_data_exists:
                try:
                    n = len(pd.read_parquet(p))
                    pct = min(100, int(n / 500))
                    st.info(f"📦 `{p.name}`: {n:,} rows ({pct}% of 50k target)")
                except Exception:
                    st.info(f"📦 `{p.name}` exists")
        else:
            st.info(
                "No US historical race data yet. Collect results via "
                "`fetch_equibase_results.py` over time."
            )


# ── Entry point ────────────────────────────────────────────────────────────────
us_racing_page()
