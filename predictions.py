"""
Horse Racing Predictions - Main Page (Lightweight)

Displays today's and tomorrow's race predictions with model insights.
Uses precomputed CSVs for fast loading.
"""
import pandas as pd
import streamlit as st
import subprocess
import sys
from datetime import timedelta
import os

# Import shared utilities
from shared.utils import (
    BASE_DIR, LOGO_FILE, MODEL_FILE,
    get_now_local, load_model, get_dataframe_height, safe_st_call,
    SCORED_FIXTURES_FILE
)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def main():
    st.set_page_config(
        page_title="Horse Racing Predictions",
        page_icon="🏇",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Display logo
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=200)
    
    # Navigation hint
    st.info("📊 **Looking for data exploration?** Check out the **Data Explorer** page in the sidebar (≡ menu icon)", icon="ℹ️")
    
    # Top Tier 1 races expander
    with st.expander("🎯 Top Predictive Races (Tier 1 Focus)", expanded=False):
        if SCORED_FIXTURES_FILE.exists():
            try:
                fixtures_scored = pd.read_csv(SCORED_FIXTURES_FILE)
                fixtures_tier1 = fixtures_scored[fixtures_scored['race_tier'] == 'Tier 1: Focus'].copy()
                
                if len(fixtures_tier1) > 0:
                    fixtures_tier1['date'] = pd.to_datetime(fixtures_tier1['date'])
                    today = pd.Timestamp.today().normalize()
                    fixtures_tier1 = fixtures_tier1[fixtures_tier1['date'] >= today]
                    
                    if len(fixtures_tier1) > 0:
                        fixtures_tier1 = fixtures_tier1.sort_values(['date', 'race_score'], ascending=[True, False]).head(50)
                        
                        display_df = fixtures_tier1[['date', 'course', 'class', 'prize', 'race_score']].copy()
                        display_df['date'] = display_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "Invalid")
                        display_df['prize'] = display_df['prize'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "")
                        display_df['race_score'] = display_df['race_score'].round(1)
                        display_df.columns = ['Date', 'Course', 'Class', 'Prize', 'Score']
                        
                        st.info(f"📊 Showing next {len(display_df)} upcoming Tier 1 Focus races (score ≥70)")
                        height = get_dataframe_height(display_df, max_height=400)
                        st.dataframe(display_df, hide_index=True, height=height)
                    else:
                        st.warning("No upcoming Tier 1 Focus races found")
                else:
                    st.warning("No upcoming Tier 1 Focus races found")
            except Exception as e:
                st.warning(f"Could not load predicted fixtures: {e}")
        else:
            st.warning("Race scoring data not available. Run Phase 2 scoring first.")
    
    # Upcoming schedule expander
    fixtures_file = BASE_DIR / "data" / "processed" / "bha_2026_all_courses_class1-4.csv"
    if fixtures_file.exists():
        try:
            fixtures = pd.read_csv(fixtures_file)
            if "Date" in fixtures.columns:
                fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
                fixtures = fixtures.sort_values("Date")
                today = pd.Timestamp.today().normalize()
                upcoming = fixtures[fixtures["Date"] >= today].copy()

            with st.expander("📅 Upcoming Schedule (Class 1-4 Races)", expanded=False):
                st.caption("Complete fixture calendar for premium races")

                if not upcoming.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Fixtures", f"{len(upcoming):,}")
                    with col2:
                        unique_courses = int(upcoming["Course"].nunique()) if "Course" in upcoming.columns else 0
                        st.metric("🏇 Courses", unique_courses)
                    with col3:
                        if "Surface" in upcoming.columns:
                            turf_count = int((upcoming["Surface"] == "Turf").sum())
                            st.metric("🌱 Turf Races", f"{turf_count:,}")
                    with col4:
                        try:
                            min_date = upcoming["Date"].min()
                            max_date = upcoming["Date"].max()
                            if pd.isna(min_date) or pd.isna(max_date):
                                span = "N/A"
                            else:
                                date_range = max_date - min_date
                                span = f"{int(getattr(date_range, 'days', 0))} days"
                        except Exception:
                            span = "N/A"
                        st.metric("📆 Calendar Span", span)

                    st.markdown("---")

                    show_cols = [c for c in ["Date", "Course", "Time", "Type", "Surface"] if c in upcoming.columns]
                    fixtures_display = upcoming[show_cols].head(200).copy()

                    if "Date" in fixtures_display.columns:
                        fixtures_display["Date"] = fixtures_display["Date"].apply(lambda x: x.strftime("%a %d %b %Y") if pd.notna(x) else "Invalid Date")

                    st.markdown(f"##### Next {len(fixtures_display)} Upcoming Fixtures")
                    height = get_dataframe_height(fixtures_display, max_height=400)
                    safe_st_call(st.dataframe, fixtures_display, hide_index=True, height=height, width='stretch')

                    st.caption("💡 **Tip:** Use the 'Data Explorer' page to see more details")
                else:
                    st.info("✨ No upcoming fixtures in calendar")
        except Exception:
            pass
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎲 Today & Tomorrow", "📅 Predicted Fixtures", "📊 Model Insights"])
    
    with tab1:
        display_predictions_tab()
    
    with tab2:
        display_predicted_fixtures_tab()
    
    with tab3:
        display_model_insights()


def display_predictions_tab():
    """Display today's and tomorrow's race predictions"""
    st.subheader("🎲 Today & Tomorrow's Race Predictions")
    
    # Get today's and tomorrow's dates
    tz_name = os.environ.get('APP_TIMEZONE')
    now_local = get_now_local(tz_name)
    today_str = now_local.strftime('%Y-%m-%d')
    tomorrow_str = (now_local + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Files for today and tomorrow
    today_predictions_file = BASE_DIR / "data" / "processed" / f"predictions_{today_str}.csv"
    today_racecards_file = BASE_DIR / "data" / "raw" / f"racecards_{today_str}.json"
    tomorrow_predictions_file = BASE_DIR / "data" / "processed" / f"predictions_{tomorrow_str}.csv"
    tomorrow_racecards_file = BASE_DIR / "data" / "raw" / f"racecards_{tomorrow_str}.json"
    
    # Check which days need data
    today_needs_data = not today_predictions_file.exists()
    tomorrow_needs_data = not tomorrow_predictions_file.exists()
    
    # Only show fetch/generate UI if at least one day needs data
    if today_needs_data or tomorrow_needs_data:
        display_fetch_generate_ui(today_str, tomorrow_str, today_needs_data, tomorrow_needs_data,
                                   today_racecards_file, tomorrow_racecards_file,
                                   today_predictions_file, tomorrow_predictions_file)
    
    # Load and display predictions
    display_race_predictions(today_str, tomorrow_str, today_predictions_file, tomorrow_predictions_file)


def display_fetch_generate_ui(today_str, tomorrow_str, today_needs_data, tomorrow_needs_data,
                               today_racecards_file, tomorrow_racecards_file,
                               today_predictions_file, tomorrow_predictions_file):
    """Display UI for fetching racecards and generating predictions"""
    num_days_needing_data = sum([today_needs_data, tomorrow_needs_data])
    
    # Step 1: Fetch Racecards
    st.markdown("### Step 1: Fetch Racecards")
    
    if num_days_needing_data == 2:
        col1a, col1b, col1c = st.columns([2, 2, 2])
    elif today_needs_data:
        col1a, col1c = st.columns([3, 2])
        col1b = None
    else:
        col1b, col1c = st.columns([3, 2])
        col1a = None
    
    if today_needs_data and col1a:
        with col1a:
            if st.button("📥 Fetch Today's Racecards", type="secondary"):
                fetch_racecards(today_str)
    
    if tomorrow_needs_data and col1b:
        with col1b:
            if st.button("📥 Fetch Tomorrow's Racecards", type="secondary"):
                fetch_racecards(tomorrow_str, "tomorrow's")
    
    with col1c:
        if today_needs_data:
            show_racecard_status(today_racecards_file, "Today's")
        if tomorrow_needs_data:
            show_racecard_status(tomorrow_racecards_file, "Tomorrow's")
    
    st.markdown("---")
    
    # Step 2: Generate Predictions
    st.markdown("### Step 2: Generate Machine Learning Predictions")
    
    if num_days_needing_data == 2:
        col2a, col2b, col2c = st.columns([2, 2, 1])
    elif today_needs_data:
        col2a, col2c = st.columns([3, 2])
        col2b = None
    else:
        col2b, col2c = st.columns([3, 2])
        col2a = None
    
    if today_needs_data and col2a:
        with col2a:
            if st.button("🔄 Generate Today's Predictions", type="primary"):
                generate_predictions(today_str, today_racecards_file)
    
    if tomorrow_needs_data and col2b:
        with col2b:
            if st.button("🔄 Generate Tomorrow's Predictions", type="primary"):
                generate_predictions(tomorrow_str, tomorrow_racecards_file, "tomorrow's")
    
    with col2c:
        if today_needs_data:
            show_prediction_status(today_predictions_file, "Today")
        if tomorrow_needs_data:
            show_prediction_status(tomorrow_predictions_file, "Tomorrow")
        
        if st.button("🔃 Refresh"):
            st.rerun()
    
    st.markdown("---")


def fetch_racecards(date_str, label=""):
    """Fetch racecards for a given date"""
    label_text = label if label else "racecards"
    with st.spinner(f"📡 Fetching {label_text} from external source..."):
        try:
            result = subprocess.run(
                [sys.executable, "scripts/fetch_racecards.py", "--date", date_str],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                st.success(f"✅ {label_text.capitalize()} fetched successfully!")
                if result.stdout:
                    with st.expander("📋 Fetch Details"):
                        st.code(result.stdout, language="text")
                st.rerun()
            else:
                st.error(f"❌ Failed to fetch {label_text}")
                with st.expander("View Error Details"):
                    st.code(result.stderr, language="text")
        except subprocess.TimeoutExpired:
            st.error(f"❌ {label_text.capitalize()} fetch timed out (>2 minutes)")
        except Exception as e:
            st.error(f"❌ Error: {e}")


def generate_predictions(date_str, racecards_file, label=""):
    """Generate predictions for a given date"""
    label_text = label if label else ""
    if not racecards_file.exists():
        st.error(f"❌ Racecards not found for {date_str}")
        st.info(f"Please click 'Fetch {label_text.capitalize()} Racecards' button above first")
    else:
        with st.spinner(f"🤖 Running Machine Learning predictions {label_text}..."):
            try:
                cmd = [sys.executable, "scripts/predict_todays_races.py"]
                if label:
                    cmd.extend(["--date", date_str])
                
                result = subprocess.run(
                    cmd,
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success(f"✅ {label_text.capitalize()} predictions generated successfully!")
                    st.balloons()
                    st.info("🔄 Auto-refreshing in 3 seconds...")
                    
                    if result.stdout:
                        output_lines = result.stdout.split('\n')
                        summary_lines = [line for line in output_lines if 'Total horses' in line or 'Total races' in line or 'SAVED' in line]
                        if summary_lines:
                            with st.expander("📊 Generation Summary"):
                                st.code('\n'.join(summary_lines), language="text")
                    
                    import time
                    time.sleep(3)
                    st.rerun()
                else:
                    st.error(f"❌ {label_text.capitalize()} prediction generation failed")
                    with st.expander("View Error Details"):
                        st.code(result.stderr, language="text")
            except subprocess.TimeoutExpired:
                st.error(f"❌ {label_text.capitalize()} prediction generation timed out")
            except Exception as e:
                st.error(f"❌ Error: {e}")


def show_racecard_status(racecards_file, label):
    """Show status of racecards file"""
    if racecards_file.exists():
        file_time = pd.Timestamp.fromtimestamp(racecards_file.stat().st_mtime)
        time_str = file_time.strftime('%I:%M %p').lstrip('0')
        st.success(f"✅ {label} racecards\nFetched at {time_str}")
    else:
        status_type = "warning" if "Today" in label else "info"
        msg = f"⚠️ No racecards for {label.lower()}"
        if status_type == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def show_prediction_status(predictions_file, label):
    """Show status of predictions file"""
    if predictions_file.exists():
        file_time = pd.Timestamp.fromtimestamp(predictions_file.stat().st_mtime)
        st.success(f"✅ {label}\n{file_time.strftime('%H:%M:%S')}")
    else:
        status_type = "warning" if label == "Today" else "info"
        msg = f"⚠️ No predictions for {label.lower()}"
        if status_type == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def display_race_predictions(today_str, tomorrow_str, today_predictions_file, tomorrow_predictions_file):
    """Load and display race predictions for today and tomorrow"""
    all_predictions = []
    
    if today_predictions_file.exists():
        today_df = pd.read_csv(today_predictions_file)
        today_df['date'] = today_str
        today_df['day_label'] = 'Today'
        all_predictions.append(today_df)
    
    if tomorrow_predictions_file.exists():
        tomorrow_df = pd.read_csv(tomorrow_predictions_file)
        tomorrow_df['date'] = tomorrow_str
        tomorrow_df['day_label'] = 'Tomorrow'
        all_predictions.append(tomorrow_df)
    
    if not all_predictions:
        st.info("📅 No predictions available. Use the buttons above to fetch racecards and generate predictions.")
        return
    
    # Combine all predictions
    predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Show summary by day
    for day_label in predictions['day_label'].unique():
        day_df = predictions[predictions['day_label'] == day_label]
        day_date = day_df['date'].iloc[0]
        st.success(f"✅ {day_label} ({day_date}): {len(day_df)} horses from {len(day_df['course'].unique())} races")
    
    st.markdown("---")
    
    # Top predictions
    display_top_predictions(predictions)
    
    st.markdown("---")
    
    # Race-by-race breakdown
    display_race_by_race(predictions)


def display_top_predictions(predictions):
    """Display top 25 predictions per day"""
    st.markdown("##### 🏆 Top 25 Predictions Per Day")
    
    has_odds = 'bookmaker_odds' in predictions.columns
    
    if has_odds:
        display_cols = ['day_label', 'date', 'race_time', 'course', 'horse', 'jockey', 'win_probability', 'place_probability', 'show_probability', 'bookmaker_odds', 'race_class', 'distance_f', 'ofr']
    else:
        display_cols = ['day_label', 'date', 'race_time', 'course', 'horse', 'jockey', 'win_probability', 'place_probability', 'show_probability', 'race_class', 'distance_f', 'ofr']
    
    top_per_day = (
        predictions
        .sort_values(['date', 'win_probability'], ascending=[True, False])
        .groupby('date', as_index=False)
        .head(25)
    )[display_cols].copy()

    top_per_day['win_probability'] = top_per_day['win_probability'].apply(lambda x: f"{x:.1%}")
    top_per_day['place_probability'] = top_per_day['place_probability'].apply(lambda x: f"{x:.1%}")
    top_per_day['show_probability'] = top_per_day['show_probability'].apply(lambda x: f"{x:.1%}")
    
    if has_odds:
        top_per_day['bookmaker_odds'] = top_per_day['bookmaker_odds'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')
        top_per_day.columns = ['Day', 'Date', 'Time', 'Course', 'Horse', 'Jockey', 'Win %', 'Place %', 'Show %', 'Odds', 'Class', 'Distance', 'OR']
    else:
        top_per_day.columns = ['Day', 'Date', 'Time', 'Course', 'Horse', 'Jockey', 'Win %', 'Place %', 'Show %', 'Class', 'Distance', 'OR']
    
    height = get_dataframe_height(top_per_day)
    safe_st_call(st.dataframe, top_per_day, hide_index=True, width='stretch', height=height)
    
    if 'race_time_gmt' in predictions.columns:
        st.caption("⏰ Times shown in **US Eastern Time (ET)** | GMT times available in detailed view")


def display_race_by_race(predictions):
    """Display race-by-race predictions with detailed analysis"""
    st.markdown("##### 📋 Race-by-Race Predictions")
    
    races = predictions.groupby(['date', 'day_label', 'race_time', 'course', 'race_name'], observed=False).size().reset_index()[['date', 'day_label', 'race_time', 'course', 'race_name']]
    
    race_options = [f"{row['day_label']} ({row['date']}) - {row['race_time']} - {row['course']} - {row['race_name'][:40]}" for _, row in races.iterrows()]
    
    selected_race_idx = st.selectbox(
        "Select a race to see detailed predictions:",
        range(len(race_options)),
        format_func=lambda i: race_options[i]
    )
    
    selected_race_info = races.iloc[selected_race_idx]
    
    race_preds = predictions[
        (predictions['date'] == selected_race_info['date']) &
        (predictions['race_time'] == selected_race_info['race_time']) &
        (predictions['course'] == selected_race_info['course'])
    ].copy()
    
    race_preds = race_preds.sort_values('win_probability', ascending=False)
    
    # Race details
    display_race_details(selected_race_info, race_preds)
    
    # Top picks
    display_top_picks(race_preds)
    
    # Exacta/Trifecta probabilities
    display_exacta_trifecta(race_preds)
    
    # All horses predictions table
    display_all_horses_table(race_preds)
    
    # Value bet calculator
    display_value_bet_calculator(race_preds, selected_race_idx)


def display_race_details(selected_race_info, race_preds):
    """Display race details"""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Day", selected_race_info['day_label'])
    with col2:
        st.metric("Date", selected_race_info['date'])
    with col3:
        st.metric("Course", selected_race_info['course'])
    with col4:
        st.metric("Time", selected_race_info['race_time'])
    with col5:
        race_class = race_preds.iloc[0]['race_class']
        st.metric("Class & Runners", f"Class {race_class} ({len(race_preds)})")


def display_top_picks(race_preds):
    """Display top picks for win, place, and show"""
    st.markdown("##### 🏆 Top Picks")
    col1, col2, col3 = st.columns(3)
    
    top_win = race_preds.nlargest(1, 'win_probability').iloc[0]
    top_place = race_preds.nlargest(1, 'place_probability').iloc[0]
    top_show = race_preds.nlargest(1, 'show_probability').iloc[0]
    
    with col1:
        st.metric("Most Likely to WIN", top_win['horse'], f"{top_win['win_probability']:.1%}")
    with col2:
        st.metric("Most Likely to PLACE", top_place['horse'], f"{top_place['place_probability']:.1%}")
    with col3:
        st.metric("Most Likely to SHOW", top_show['horse'], f"{top_show['show_probability']:.1%}")


def display_exacta_trifecta(race_preds):
    """Display Exacta and Trifecta probabilities"""
    st.markdown("---")
    st.markdown("##### 🎯 Exacta/Trifecta Probabilities")
    
    top_3 = race_preds.nlargest(3, 'win_probability')
    
    if len(top_3) >= 3:
        p1 = top_3.iloc[0]['win_probability']
        p2 = top_3.iloc[1]['win_probability']
        p3 = top_3.iloc[2]['win_probability']
        
        if p1 < 0.99:
            p_second_given_first = p2 / (1 - p1)
            if (p1 + p2) < 0.99:
                p_third_given_first_second = p3 / (1 - p1 - p2)
                trifecta_prob = p1 * p_second_given_first * p_third_given_first_second
            else:
                trifecta_prob = 0
        else:
            trifecta_prob = 0
        
        exacta_prob = p1 * p_second_given_first if p1 < 0.99 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "🥇🥈 Exacta (1-2 in order)", 
                f"{exacta_prob:.1%}",
                help=f"{top_3.iloc[0]['horse']} (1st) → {top_3.iloc[1]['horse']} (2nd)"
            )
        with col2:
            st.metric(
                "🥇🥈🥉 Trifecta (1-2-3 in order)", 
                f"{trifecta_prob:.1%}",
                help=f"{top_3.iloc[0]['horse']} (1st) → {top_3.iloc[1]['horse']} (2nd) → {top_3.iloc[2]['horse']} (3rd)"
            )
        with col3:
            if trifecta_prob > 0:
                trifecta_odds = (1 / trifecta_prob) - 1
                st.metric(
                    "💰 Fair Trifecta Odds",
                    f"{trifecta_odds:.1f}/1",
                    help="Fair odds based on model probabilities"
                )
            else:
                st.metric("💰 Fair Trifecta Odds", "N/A")
        
        st.caption(f"🎯 **Predicted 1-2-3:** {top_3.iloc[0]['horse']} → {top_3.iloc[1]['horse']} → {top_3.iloc[2]['horse']}")
        
        # Cumulative probabilities
        st.markdown("##### 🏆 Top Selection Probabilities")
        st.caption(f"Likelihood that **{top_3.iloc[0]['horse']}** (the favorite) finishes in the money")
        
        top_horse_win_prob = top_3.iloc[0]['win_probability']
        top_horse_place_prob = top_3.iloc[0]['place_probability'] if 'place_probability' in top_3.iloc[0] else 0
        top_horse_show_prob = top_3.iloc[0]['show_probability'] if 'show_probability' in top_3.iloc[0] else 0
        
        prob_top_1 = top_horse_win_prob
        prob_top_2 = top_horse_win_prob + top_horse_place_prob
        prob_top_3 = top_horse_win_prob + top_horse_place_prob + top_horse_show_prob
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🥇 Win (1st)", f"{prob_top_1:.1%}")
        with col2:
            st.metric("🥇🥈 Win or Place", f"{prob_top_2:.1%} (+{top_horse_place_prob:.1%})")
        with col3:
            st.metric("🥇🥈🥉 Win, Place, or Show", f"{prob_top_3:.1%} (+{top_horse_show_prob:.1%})")


def display_all_horses_table(race_preds):
    """Display table of all horses in the race with predictions"""
    st.markdown("##### 🐎 All Horse Predictions")
    st.caption("📊 Form shows recent race finishes (read right to left: rightmost = most recent race)")
    st.caption("💰 Model Odds show the fair value based on probabilities")
    
    display_cols = ['horse', 'jockey', 'win_probability', 'win_odds_fractional', 'place_probability', 'place_odds_fractional', 'show_probability', 'show_odds_fractional', 'age', 'weight_lbs', 'ofr', 'form']
    display_df = race_preds[display_cols].copy()
    
    display_df['top_2_prob'] = race_preds['win_probability'] + race_preds['place_probability']
    display_df['top_3_prob'] = race_preds['win_probability'] + race_preds['place_probability'] + race_preds['show_probability']
    
    display_df['win_rank'] = race_preds['win_probability'].rank(ascending=False, method='min').astype(int)
    display_df['place_rank'] = race_preds['place_probability'].rank(ascending=False, method='min').astype(int)
    display_df['show_rank'] = race_preds['show_probability'].rank(ascending=False, method='min').astype(int)
    
    display_df['win_probability'] = display_df['win_probability'].apply(lambda x: f"{x:.1%}")
    display_df['place_probability'] = display_df['place_probability'].apply(lambda x: f"{x:.1%}")
    display_df['show_probability'] = display_df['show_probability'].apply(lambda x: f"{x:.1%}")
    display_df['top_2_prob'] = display_df['top_2_prob'].apply(lambda x: f"{x:.1%}")
    display_df['top_3_prob'] = display_df['top_3_prob'].apply(lambda x: f"{x:.1%}")
    
    display_df = display_df[[
        'horse', 'jockey',
        'win_rank', 'win_probability', 'win_odds_fractional',
        'place_rank', 'place_probability', 'place_odds_fractional',
        'show_rank', 'show_probability', 'show_odds_fractional',
        'top_2_prob', 'top_3_prob',
        'age', 'weight_lbs', 'ofr', 'form'
    ]]
    display_df.columns = [
        'Horse', 'Jockey',
        'Win Rank', 'Win %', 'Win Odds',
        'Place Rank', 'Place %', 'Place Odds',
        'Show Rank', 'Show %', 'Show Odds',
        'Top 2 %', 'Top 3 %',
        'Age', 'Weight', 'OR', 'Recent Form'
    ]
    
    display_df = display_df.sort_values('Win Rank')
    
    st.dataframe(display_df, hide_index=True, width=800)


def display_value_bet_calculator(race_preds, selected_race_idx):
    """Display value bet calculator"""
    st.markdown("---")
    st.markdown("##### 💰 Value Bet Calculator")
    st.caption("Compare model odds to bookmaker odds to identify value betting opportunities")
    
    with st.expander("🧮 Calculate Value Bet (Enter Bookmaker Odds)", expanded=False):
        st.markdown("**How to use:**")
        st.markdown("1. Look up bookmaker's odds for a horse")
        st.markdown("2. Enter the decimal odds below")
        st.markdown("3. See if there's value compared to model's fair odds")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            horses_list = race_preds['horse'].tolist()
            selected_horse = st.selectbox(
                "Select Horse",
                horses_list,
                key=f"vb_horse_{selected_race_idx}"
            )
        
        with col2:
            bookmaker_odds_input = st.number_input(
                "Bookmaker Decimal Odds",
                min_value=1.01,
                max_value=1000.0,
                value=3.0,
                step=0.1,
                key=f"vb_odds_{selected_race_idx}"
            )
        
        horse_data = race_preds[race_preds['horse'] == selected_horse].iloc[0]
        model_prob = horse_data['win_probability']
        model_decimal_odds = 1 / model_prob
        model_fractional = horse_data['win_odds_fractional']
        
        bookmaker_implied_prob = 1 / bookmaker_odds_input
        edge = model_prob - bookmaker_implied_prob
        edge_pct = edge * 100
        
        st.markdown("---")
        st.markdown(f"**Analysis for {selected_horse}:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Win %", f"{model_prob:.1%}")
        with col2:
            st.metric("Model Odds", f"{model_fractional} ({model_decimal_odds:.2f})")
        with col3:
            st.metric("Bookmaker Implied %", f"{bookmaker_implied_prob:.1%}")
        with col4:
            delta_color = "normal" if edge > 0 else "inverse"
            st.metric("Edge", f"{edge_pct:+.1f}%", delta=f"{edge_pct:+.1f}%", delta_color=delta_color)
        
        st.markdown("---")
        
        if edge >= 0.05:
            st.success(f"✅ **VALUE BET!** Edge: {edge_pct:+.1f}%")
            st.markdown(f"**Recommendation:** BACK {selected_horse}")
        elif edge >= 0.02:
            st.info(f"⚖️ **MARGINAL VALUE** Edge: {edge_pct:+.1f}%")
        elif edge >= -0.02:
            st.warning(f"📊 **FAIR ODDS** Edge: {edge_pct:+.1f}%")
        else:
            st.error(f"❌ **NO VALUE** Edge: {edge_pct:+.1f}%")


def display_model_insights():
    """Display ML model insights and feature importance"""
    st.subheader("Machine Learning Model")
    
    model, metadata, feature_importance = load_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Train the model first.")
        
        if st.button("🚀 Train Model Now", type="primary"):
            with st.spinner("Training model... This may take 2-3 minutes."):
                try:
                    result = subprocess.run(
                        [sys.executable, "scripts/phase3_build_horse_model.py"],
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Model trained successfully!")
                        st.text("Training Output:")
                        st.code(result.stdout[-2000:], language="text")
                        st.info("Refresh the page to load the new model.")
                    else:
                        st.error(f"❌ Training failed with error code {result.returncode}")
                        st.code(result.stderr, language="text")
                except Exception as e:
                    st.error(f"Error running training script: {e}")
    else:
        st.success("✅ Model loaded successfully")
        
        if metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", metadata.get('model_type', 'Unknown'))
            with col2:
                st.metric("Features", metadata.get('n_features', 0))
            with col3:
                st.metric("Trained", metadata.get('trained_date', 'Unknown')[:10])
        
        st.markdown("---")
        
        if feature_importance is not None and not feature_importance.empty:
            if 'rank' not in feature_importance.columns:
                feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
                feature_importance['rank'] = range(1, len(feature_importance) + 1)
            
            top_features = feature_importance.head(15)
            
            if HAS_PLOTLY:
                fig = go.Figure(go.Bar(
                    x=top_features['importance'],
                    y=[f"#{int(row['rank'])} {row['feature']}" for _, row in top_features.iterrows()],
                    orientation='h',
                    marker=dict(
                        color=top_features['importance'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title="Feature Importance (XGBoost) - Ranked",
                    xaxis_title="Importance Score",
                    yaxis_title="Rank & Feature",
                    height=500,
                    yaxis={'categoryorder':'total ascending'}
                )
                
                safe_st_call(st.plotly_chart, fig, width='stretch')
            else:
                st.bar_chart(top_features.set_index('feature')['importance'])


def display_predicted_fixtures_tab():
    """Display predicted fixtures for 2025-2026"""
    st.subheader("📅 Predicted Fixtures (2025-2026)")
    
    # Load scored fixtures
    if SCORED_FIXTURES_FILE.exists():
        try:
            fixtures_scored = pd.read_csv(SCORED_FIXTURES_FILE)
            fixtures_scored['date'] = pd.to_datetime(fixtures_scored['date'])
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Fixtures", f"{len(fixtures_scored):,}")
            with col2:
                tier1_count = (fixtures_scored['race_tier'] == 'Tier 1: Focus').sum()
                st.metric("Tier 1 Focus", tier1_count)
            with col3:
                tier2_count = (fixtures_scored['race_tier'] == 'Tier 2: Value').sum()
                st.metric("Tier 2 Value", tier2_count)
            with col4:
                avg_score = fixtures_scored['race_score'].mean()
                st.metric("Avg Score", f"{avg_score:.1f}")
            
            st.markdown("---")
            
            # Explanation
            with st.expander("ℹ️ About Predicted Fixtures", expanded=False):
                st.markdown("""
                **How predictions work:**
                
                1. **Course Profiles**: Analyzed 245,298 historical races to build statistical profiles for 37 courses
                2. **Characteristic Prediction**: For each fixture, predicted:
                   - **Class**: Weekend races get better class (Class 2 vs weekday Class 3)
                   - **Prize Money**: 75th percentile for weekends, median for weekdays
                   - **Field Size**: Larger fields predicted for weekend races
                   - **Going**: Seasonal adjustments (winter=softer, summer=firmer)
                   - **Distance**: Course-specific median distances
                3. **Scoring**: Applied Phase 2 race scoring algorithm to predicted characteristics
                
                **Prediction Quality:**
                - Based on 10+ years of historical data per course
                - Weekend/weekday distinction improves accuracy
                - Seasonal going adjustments match UK weather patterns
                - Scores are estimates - actual races may vary
                
                **Top Predicted Courses:**
                - **Ascot**: 14 Tier 1 races (score 85.4)
                - **York**: 8 Tier 1 races (score 82+)
                - **Goodwood**: 7 Tier 1 races (score 74+)
                """)
            
            # Score distribution chart
            st.subheader("Score Distribution")
            if HAS_PLOTLY:
                try:
                    import plotly.express as px
                    fig = px.histogram(
                        fixtures_scored, 
                        x='race_score',
                        nbins=30,
                        title='Distribution of Predicted Race Scores',
                        labels={'race_score': 'Race Score', 'count': 'Number of Races'},
                        color='race_tier',
                        color_discrete_map={
                            'Tier 1: Focus': '#2ecc71',
                            'Tier 2: Value': '#f39c12',
                            'Tier 3: Avoid': '#e74c3c'
                        }
                    )
                    fig.update_layout(height=400)
                    safe_st_call(st.plotly_chart, fig, width='stretch')
                except:
                    st.info("Install plotly to see score distribution chart")
            else:
                st.info("Install plotly to see score distribution chart")
            
            # Course breakdown
            st.subheader("Top Courses by Score")
            course_stats = fixtures_scored.groupby('course', observed=False).agg({
                'race_score': ['count', 'mean', 'max'],
                'race_tier': lambda x: (x == 'Tier 1: Focus').sum()
            }).round(1)
            course_stats.columns = ['Total Races', 'Avg Score', 'Max Score', 'Tier 1 Count']
            course_stats = course_stats.sort_values('Max Score', ascending=False).head(15)
            height = get_dataframe_height(course_stats)
            st.dataframe(course_stats, height=height)
            
            # Filterable table of all fixtures
            st.subheader("All Predicted Fixtures")
            
            # Filters in columns
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                tier_filter = st.multiselect(
                    "Filter by Tier",
                    options=['Tier 1: Focus', 'Tier 2: Value', 'Tier 3: Avoid'],
                    default=['Tier 1: Focus', 'Tier 2: Value'],
                    key="pred_tier_filter"
                )
            with fcol2:
                course_filter = st.multiselect(
                    "Filter by Course",
                    options=sorted(fixtures_scored['course'].unique()),
                    key="pred_course_filter"
                )
            with fcol3:
                min_score = st.number_input(
                    "Min Score",
                    min_value=0,
                    max_value=100,
                    value=0,
                    key="pred_min_score"
                )
            
            # Apply filters
            filtered_fixtures = fixtures_scored.copy()
            if tier_filter:
                filtered_fixtures = filtered_fixtures[filtered_fixtures['race_tier'].isin(tier_filter)]
            if course_filter:
                filtered_fixtures = filtered_fixtures[filtered_fixtures['course'].isin(course_filter)]
            filtered_fixtures = filtered_fixtures[filtered_fixtures['race_score'] >= min_score]
            
            # Sort by date
            filtered_fixtures = filtered_fixtures.sort_values('date')
            
            # Display table
            display_cols = ['date', 'course', 'class', 'prize', 'race_score', 'race_tier', 'weekday', 'surface']
            display_fixtures = filtered_fixtures[display_cols].copy()
            display_fixtures['date'] = display_fixtures['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "Invalid")
            display_fixtures['prize'] = display_fixtures['prize'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "")
            display_fixtures['race_score'] = display_fixtures['race_score'].round(1)
            display_fixtures.columns = ['Date', 'Course', 'Class', 'Prize', 'Score', 'Tier', 'Day', 'Surface']
            
            st.info(f"Showing {len(display_fixtures):,} of {len(fixtures_scored):,} predicted fixtures")
            height = get_dataframe_height(display_fixtures, max_height=600)
            st.dataframe(display_fixtures, hide_index=True, height=height)
            
        except Exception as e:
            st.error(f"Error loading predicted fixtures: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning(f"Predicted fixtures file not found. Run `python scripts/score_fixture_calendar.py` to generate predictions.")
        st.info(f"Expected file: {SCORED_FIXTURES_FILE}")


if __name__ == "__main__":
    main()
