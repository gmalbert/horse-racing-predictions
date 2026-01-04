"""
Horse Racing Predictions - Streamlit App

Displays UK horse race results with filtering capabilities.
Includes ML model for win probability predictions.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import pickle
import subprocess
import sys
from datetime import datetime, timedelta
import os

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
try:
    from dotenv import load_dotenv
    load_dotenv()  # load variables from .env into environment (APP_TIMEZONE etc.)
except Exception:
    # dotenv is optional; if not present we rely on the environment
    pass


def get_now_local(tz_name: str | None = None) -> datetime:
    """Return a timezone-aware 'now' datetime for the given IANA tz name.

    If tz_name is None, the function will consult the `APP_TIMEZONE`
    environment variable. If ZoneInfo is unavailable or the tz name is
    invalid, falls back to the system local timezone.
    """
    if tz_name is None:
        tz_name = os.environ.get('APP_TIMEZONE')

    if tz_name and ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = datetime.now().astimezone().tzinfo
    else:
        tz = datetime.now().astimezone().tzinfo

    return datetime.now(tz)


BASE_DIR = Path(__file__).parent
PARQUET_FILE = BASE_DIR / "data" / "processed" / "race_scores.parquet"  # Using cleaned data with scores
CSV_FILE = BASE_DIR / "data" / "processed" / "all_gb_races.csv"
LOGO_FILE = BASE_DIR / "data" / "logo.png"
MODEL_FILE = BASE_DIR / "models" / "horse_win_predictor.pkl"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "models" / "feature_importance.csv"
METADATA_FILE = BASE_DIR / "models" / "model_metadata.pkl"
SCORED_FIXTURES_FILE = BASE_DIR / "data" / "processed" / "scored_fixtures_calendar.csv"


@st.cache_data
def load_model():
    """Load trained ML model and metadata"""
    if not MODEL_FILE.exists():
        return None, None, None
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        metadata = None
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
        
        feature_importance = None
        if FEATURE_IMPORTANCE_FILE.exists():
            feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
        
        return model, metadata, feature_importance
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load and cache the UK horse races dataset (Parquet preferred).

    The function will prefer `all_gb_races.parquet` if present for faster
    loading and smaller storage. Falls back to CSV if Parquet is missing.
    """
    if PARQUET_FILE.exists():
        df = pd.read_parquet(PARQUET_FILE)
    elif CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
    else:
        raise FileNotFoundError(f"Dataset not found: {PARQUET_FILE} or {CSV_FILE}")

    if 'date' in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    # Rename columns to be more readable
    column_rename = {
        "date": "Date",
        "course": "Course",
        "race_name": "Race Name",
        "horse": "Horse",
        "pos": "Finish Position",
        "jockey": "Jockey",
        "trainer": "Trainer",
        "dist": "Distance",
        "going": "Going",
        "time": "Time",
        "age": "Age",
        "sex": "Sex",
        "draw": "Draw",
        "or": "Official Rating",
        "rpr": "RPR",
        "sire": "Sire",
        "dam": "Dam",
        "owner": "Owner",
        "prize": "Prize",
        "class": "Class",
        "type": "Type",
        "off": "Off Time"
    }
    df = df.rename(columns=column_rename)
    
    return df

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

def main():
    st.set_page_config(
        page_title="Horse Racing Predictions",
        page_icon="🏇",
        layout="wide"
    )

    # Display logo
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=200)
    
    # st.title("🏇 Equine Edge")
    # st.markdown("---")

    # Load data
    df = load_data()

    # Top predictive races expander - includes both historical AND upcoming predicted races
    with st.expander("🎯 Top Predictive Races (Tier 1 Focus)", expanded=False):
        # Combine historical scores with predicted fixture scores
        all_tier1_races = pd.DataFrame()
        
        # 1. Load historical Tier 1 races
        if 'race_tier' in df.columns and 'race_score' in df.columns:
            historical_tier1 = df[df['race_tier'] == 'Tier 1: Focus'].copy()
            if 'Date' in historical_tier1.columns:
                historical_tier1['Date'] = pd.to_datetime(historical_tier1['Date'])
            historical_tier1['source'] = 'Historical'
            all_tier1_races = historical_tier1
        
        # 2. Load predicted fixture scores
        if SCORED_FIXTURES_FILE.exists():
            try:
                fixtures_scored = pd.read_csv(SCORED_FIXTURES_FILE)
                fixtures_tier1 = fixtures_scored[fixtures_scored['race_tier'] == 'Tier 1: Focus'].copy()
                if len(fixtures_tier1) > 0:
                    # Rename columns to match historical data format
                    col_mapping = {
                        'date': 'Date',
                        'course': 'Course',
                        'class': 'Class',
                        'prize': 'Prize',
                        'dist_f': 'Distance'
                    }
                    fixtures_tier1 = fixtures_tier1.rename(columns=col_mapping)
                    if 'Date' in fixtures_tier1.columns:
                        fixtures_tier1['Date'] = pd.to_datetime(fixtures_tier1['Date'])
                    fixtures_tier1['source'] = 'Predicted 🔮'
                    fixtures_tier1['Race Name'] = 'Predicted Race'
                    
                    # Combine with historical
                    all_tier1_races = pd.concat([all_tier1_races, fixtures_tier1], ignore_index=True)
            except Exception as e:
                st.warning(f"Could not load predicted fixtures: {e}")
        
        if len(all_tier1_races) > 0:
            # Filter to future races only
            today = pd.Timestamp.today().normalize()
            all_tier1_races = all_tier1_races[all_tier1_races['Date'] >= today]
            
            # Group by unique race and get the best score
            race_cols = ['Date', 'Course', 'Race Name', 'Class', 'Distance', 'Prize', 'race_score', 'source']
            available_cols = [c for c in race_cols if c in all_tier1_races.columns]
            
            if len(all_tier1_races) > 0:
                # Get unique races (deduplicate by race_id if available, otherwise by date+course)
                if 'race_id' in all_tier1_races.columns:
                    tier1_unique = all_tier1_races.groupby('race_id')[available_cols].first().reset_index(drop=True)
                else:
                    tier1_unique = all_tier1_races[available_cols].drop_duplicates(['Date', 'Course'])
                
                # Sort by date ascending (soonest first), then by score descending
                tier1_unique = tier1_unique.sort_values(['Date', 'race_score'], ascending=[True, False]).head(50)
                
                # Format display
                display_df = tier1_unique.copy()
                if 'Date' in display_df.columns:
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                if 'Prize' in display_df.columns:
                    display_df['Prize'] = display_df['Prize'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "")
                if 'Distance' in display_df.columns:
                    display_df['Distance'] = display_df['Distance'].apply(lambda x: f"{x:.0f}f" if pd.notna(x) else "")
                if 'race_score' in display_df.columns:
                    display_df['Score'] = display_df['race_score'].round(1)
                    display_df = display_df.drop('race_score', axis=1)
                
                # Count predicted vs historical
                predicted_count = (tier1_unique['source'] == 'Predicted 🔮').sum() if 'source' in tier1_unique.columns else 0
                st.info(f"📊 Showing next {len(display_df)} upcoming Tier 1 Focus races (score ≥70) - {predicted_count} predicted, {len(display_df)-predicted_count} with actual data")
                height = get_dataframe_height(display_df, max_height=400)
                st.dataframe(display_df, hide_index=True, height=height)
            else:
                st.warning("No upcoming Tier 1 Focus races found")
        else:
            st.warning("Race scoring data not available. Run Phase 2 scoring first.")
    
    # Upcoming schedule (fixtures) - show in an expander
    fixtures_file = BASE_DIR / "data" / "processed" / "bha_2026_all_courses_class1-4.csv"
    if fixtures_file.exists():
        try:
            fixtures = pd.read_csv(fixtures_file)
            # Parse Date column if present and sort chronologically
            if "Date" in fixtures.columns:
                fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
                fixtures = fixtures.sort_values("Date")
                
                # Filter to upcoming races only
                today = pd.Timestamp.today().normalize()
                upcoming = fixtures[fixtures["Date"] >= today].copy()

            with st.expander("📅 Upcoming Schedule (Class 1-4 Races)", expanded=False):
                st.caption("Complete fixture calendar for premium races")
                
                if len(upcoming) > 0:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Fixtures", f"{len(upcoming):,}")
                    with col2:
                        unique_courses = upcoming["Course"].nunique() if "Course" in upcoming.columns else 0
                        st.metric("🏇 Courses", unique_courses)
                    with col3:
                        if "Surface" in upcoming.columns:
                            turf_count = (upcoming["Surface"] == "Turf").sum()
                            st.metric("🌱 Turf Races", f"{turf_count:,}")
                    with col4:
                        date_range = upcoming["Date"].max() - upcoming["Date"].min()
                        st.metric("📆 Calendar Span", f"{date_range.days} days")
                    
                    st.markdown("---")
                    
                    # Display table
                    show_cols = [c for c in ["Date", "Course", "Time", "Type", "Surface"] if c in upcoming.columns]
                    fixtures_display = upcoming[show_cols].head(200).copy()
                    
                    # Format date column
                    if "Date" in fixtures_display.columns:
                        fixtures_display["Date"] = fixtures_display["Date"].dt.strftime("%a %d %b %Y")
                    
                    st.markdown(f"##### Next {len(fixtures_display)} Upcoming Fixtures")
                    height = get_dataframe_height(fixtures_display, max_height=400)
                    st.dataframe(fixtures_display, hide_index=True, height=height, width='stretch')
                    
                    st.caption("💡 **Tip:** Use the 'Predicted Fixtures' tab to see profitability scores for these races")
                else:
                    st.info("✨ No upcoming fixtures in calendar")
        except Exception as e:
            st.warning(f"Could not load upcoming schedule: {e}")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Year filter
    df["year"] = df["Date"].dt.year
    years = sorted(df["year"].dropna().unique(), reverse=True)
    selected_years = st.sidebar.multiselect(
        "Year",
        options=years,
        default=None,
        placeholder="All years"
    )

    # Course filter
    courses = sorted(df["Course"].dropna().unique())
    selected_courses = st.sidebar.multiselect(
        "Course",
        options=courses,
        default=None,
        placeholder="All courses"
    )

    # Horse name filter
    horse_name = st.sidebar.text_input("Horse Name (contains)", "")

    # Finish order filter - convert to int for proper sorting
    # Create a numeric finish-position column for reliable filtering
    df["Finish Position Numeric"] = pd.to_numeric(df["Finish Position"], errors="coerce")
    positions = sorted([int(p) for p in df["Finish Position Numeric"].dropna().unique()])
    selected_positions = st.sidebar.multiselect(
        "Finish Position",
        options=positions,
        default=None,
        placeholder="All positions"
    )

    # Checkbox to show only top-3 finishes
    top3_only = st.sidebar.checkbox("Top 3 only", value=False, help="Show only horses finishing 1-3")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Filters")

    # 1. Official Rating (OR) range slider
    df["OR Numeric"] = pd.to_numeric(df["Official Rating"], errors="coerce")
    or_min = int(df["OR Numeric"].min()) if df["OR Numeric"].notna().any() else 0
    or_max = int(df["OR Numeric"].max()) if df["OR Numeric"].notna().any() else 150
    or_range = st.sidebar.slider(
        "Official Rating (OR)",
        min_value=or_min,
        max_value=or_max,
        value=(or_min, or_max),
        help="Filter by horse's official rating"
    )

    # 2. Race Tier filter (from Phase 2 scorer)
    if 'race_tier' in df.columns:
        tiers = sorted([t for t in df['race_tier'].dropna().unique() if t])
        selected_tiers = st.sidebar.multiselect(
            "🎯 Race Tier (Profitability)",
            options=tiers,
            default=None,
            placeholder="All tiers",
            help="Tier 1 Focus = best betting value (score ≥70), Tier 2 Value (50-69), Tier 3 Avoid (<50)"
        )
    else:
        selected_tiers = None
    
    # 3. Race Score range (from Phase 2 scorer)
    if 'race_score' in df.columns:
        score_min = int(df['race_score'].min()) if df['race_score'].notna().any() else 0
        score_max = int(df['race_score'].max()) if df['race_score'].notna().any() else 100
        score_range = st.sidebar.slider(
            "Race Score",
            min_value=score_min,
            max_value=score_max,
            value=(score_min, score_max),
            help="Race profitability score (0-100) - higher = better betting opportunity"
        )
    else:
        score_range = None

    # 4. Race Class multiselect
    if "Class" in df.columns:
        classes = sorted([c for c in df["Class"].dropna().unique() if c])
        selected_classes = st.sidebar.multiselect(
            "Race Class",
            options=classes,
            default=None,
            placeholder="All classes",
            help="Class 1 = highest quality"
        )
    else:
        selected_classes = None

    # 5. Prize Money threshold
    df["Prize Numeric"] = pd.to_numeric(df["Prize"], errors="coerce")
    if df["Prize Numeric"].notna().any():
        prize_min = st.sidebar.number_input(
            "Min Prize Money (£)",
            min_value=0,
            value=0,
            step=1000,
            help="Show races with prize ≥ this amount"
        )
    else:
        prize_min = 0

    # 6. Field Size filter
    if "ran" in df.columns:
        df["Field Size"] = pd.to_numeric(df["ran"], errors="coerce")
        field_min = int(df["Field Size"].min()) if df["Field Size"].notna().any() else 0
        field_max = int(df["Field Size"].max()) if df["Field Size"].notna().any() else 30
        field_range = st.sidebar.slider(
            "Field Size (runners)",
            min_value=field_min,
            max_value=field_max,
            value=(field_min, field_max),
            help="Number of runners in race"
        )
    else:
        field_range = None

    # 7. Surface filter
    if "Surface" in df.columns:
        surfaces = sorted([s for s in df["Surface"].dropna().unique() if s])
        selected_surfaces = st.sidebar.multiselect(
            "Surface",
            options=surfaces,
            default=None,
            placeholder="All surfaces",
            help="Turf vs All-Weather"
        )
    else:
        selected_surfaces = None

    # 8. Distance bands
    df["Distance F Numeric"] = pd.to_numeric(df["Distance"].str.extract(r'(\d+)f')[0], errors="coerce") if "Distance" in df.columns else pd.to_numeric(df.get("dist_f"), errors="coerce")
    distance_bands = [
        ("Sprint", 5, 7),
        ("Mile", 7, 9),
        ("Middle", 9, 12),
        ("Long", 12, 99)
    ]
    selected_distance_band = st.sidebar.selectbox(
        "Distance Band",
        options=["All"] + [band[0] for band in distance_bands],
        help="Sprint (5-7f), Mile (7-9f), Middle (9-12f), Long (12f+)"
    )

    # 9. Pattern races checkbox
    if "Pattern" in df.columns:
        pattern_only = st.sidebar.checkbox(
            "Pattern Races Only",
            value=False,
            help="Show only Group/Listed races"
        )
    else:
        pattern_only = False

    # 10. Age band filter
    if "Age Band" in df.columns or "age_band" in df.columns:
        age_col = "Age Band" if "Age Band" in df.columns else "age_band"
        age_bands = sorted([a for a in df[age_col].dropna().unique() if a])
        selected_age_bands = st.sidebar.multiselect(
            "Age Band",
            options=age_bands,
            default=None,
            placeholder="All age bands",
            help="2yo, 3yo, 3yo+, 4yo+, etc."
        )
    else:
        selected_age_bands = None

    # Apply filters
    filtered_df = df.copy()

    if selected_years:
        filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]

    if selected_courses:
        filtered_df = filtered_df[filtered_df["Course"].isin(selected_courses)]

    if horse_name:
        filtered_df = filtered_df[
            filtered_df["Horse"].str.contains(horse_name, case=False, na=False)
        ]

    if selected_positions:
        filtered_df = filtered_df[filtered_df["Finish Position Numeric"].isin(selected_positions)]

    if top3_only:
        filtered_df = filtered_df[filtered_df["Finish Position Numeric"] <= 3]

    # Apply advanced filters
    # 1. OR range
    filtered_df = filtered_df[
        (filtered_df["OR Numeric"].isna()) | 
        ((filtered_df["OR Numeric"] >= or_range[0]) & (filtered_df["OR Numeric"] <= or_range[1]))
    ]

    # 2. Race Tier
    if selected_tiers:
        filtered_df = filtered_df[filtered_df['race_tier'].isin(selected_tiers)]
    
    # 3. Race Score
    if score_range:
        filtered_df = filtered_df[
            (filtered_df['race_score'].isna()) |
            ((filtered_df['race_score'] >= score_range[0]) & (filtered_df['race_score'] <= score_range[1]))
        ]

    # 4. Race Class
    if selected_classes:
        filtered_df = filtered_df[filtered_df["Class"].isin(selected_classes)]

    # 5. Prize Money
    if prize_min > 0:
        filtered_df = filtered_df[filtered_df["Prize Numeric"] >= prize_min]

    # 6. Field Size
    if field_range:
        filtered_df = filtered_df[
            (filtered_df["Field Size"].isna()) |
            ((filtered_df["Field Size"] >= field_range[0]) & (filtered_df["Field Size"] <= field_range[1]))
        ]

    # 7. Surface
    if selected_surfaces:
        filtered_df = filtered_df[filtered_df["Surface"].isin(selected_surfaces)]

    # 8. Distance bands
    if selected_distance_band != "All":
        band = next((b for b in distance_bands if b[0] == selected_distance_band), None)
        if band:
            filtered_df = filtered_df[
                (filtered_df["Distance F Numeric"] >= band[1]) & 
                (filtered_df["Distance F Numeric"] < band[2])
            ]

    # 9. Pattern races
    if pattern_only:
        filtered_df = filtered_df[filtered_df["Pattern"].notna() & (filtered_df["Pattern"] != "")]

    # 10. Age band
    if selected_age_bands:
        age_col = "Age Band" if "Age Band" in filtered_df.columns else "age_band"
        filtered_df = filtered_df[filtered_df[age_col].isin(selected_age_bands)]

    # Sort by date descending
    filtered_df = filtered_df.sort_values("Date", ascending=False)
    
    # Results display moved to the "Raw Data" tab at the bottom of the page.

    # Show summary stats in sidebar
    # st.sidebar.markdown("---")
    st.sidebar.subheader("Summary")
    st.sidebar.metric("Total Races in Dataset", f"{len(df):,}")
    st.sidebar.metric("Filtered Races", f"{len(filtered_df):,}")
    
    # Detailed statistics section on main page
    # st.markdown("---")
    # st.header("📊 Data Summary")
    
    # Create tabs for different summary views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["🏇 Horses", "🏟️ Courses", "👤 Jockeys", "📈 Overall", "🔮 ML Model", "🗃️ Raw Data", "📅 Predicted Fixtures", "🎯 Betting Watchlist", "🎲 Today & Tomorrow"])
    
    with tab1:
        st.subheader("Horse Performance")
        
        # For all summary tabs, use ALL filtered data (not limited by num_results)
        analysis_df = df.copy()
        analysis_df["Finish Position Numeric"] = pd.to_numeric(analysis_df["Finish Position"], errors='coerce')
        
        if selected_years:
            analysis_df = analysis_df[analysis_df["year"].isin(selected_years)]
        if selected_courses:
            analysis_df = analysis_df[analysis_df["Course"].isin(selected_courses)]
        if horse_name:
            analysis_df = analysis_df[analysis_df["Horse"].str.contains(horse_name, case=False, na=False)]
        if selected_positions:
            analysis_df = analysis_df[analysis_df["Finish Position Numeric"].isin(selected_positions)]
        
        horse_stats = analysis_df.groupby("Horse").agg({
            "Finish Position Numeric": [
                "count", 
                "mean", 
                lambda x: (x == 1).sum(), 
                lambda x: (x == 2).sum(),
                lambda x: (x == 3).sum(),
                lambda x: (x <= 3).sum()
            ],
            "Prize": "sum"
        }).reset_index()
        
        horse_stats.columns = ["Horse", "Total Races", "Avg Finish", "Wins", "Place", "Show", "Top 3 Finishes", "Total Prize"]
        horse_stats["Win Rate %"] = (horse_stats["Wins"] / horse_stats["Total Races"] * 100).round(1)
        horse_stats["Avg Finish"] = horse_stats["Avg Finish"].round(2)
        
        # Format columns
        horse_stats["Total Races"] = horse_stats["Total Races"].apply(lambda x: f"{int(x):,}")
        horse_stats["Total Prize"] = horse_stats["Total Prize"].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "£0")
        
        horse_stats = horse_stats.sort_values("Wins", ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top Horse (by wins)", horse_stats.iloc[0]["Horse"] if len(horse_stats) > 0 else "N/A")
        with col2:
            st.metric("Most Wins", int(horse_stats.iloc[0]["Wins"]) if len(horse_stats) > 0 else 0)
        
        height = get_dataframe_height(horse_stats)
        st.dataframe(
            horse_stats,
            # width="stretch",

            hide_index=True,
            height=height
        )
    
    with tab2:
        st.subheader("Course Statistics")
        
        # For course stats, apply all filters (including course) but use ALL matching data, not limited by num_results
        course_analysis_df = df.copy()
        course_analysis_df["Finish Position Numeric"] = pd.to_numeric(course_analysis_df["Finish Position"], errors='coerce')
        
        if selected_years:
            course_analysis_df = course_analysis_df[course_analysis_df["year"].isin(selected_years)]
        if selected_courses:
            course_analysis_df = course_analysis_df[course_analysis_df["Course"].isin(selected_courses)]
        if horse_name:
            course_analysis_df = course_analysis_df[course_analysis_df["Horse"].str.contains(horse_name, case=False, na=False)]
        if selected_positions:
            course_analysis_df = course_analysis_df[course_analysis_df["Finish Position Numeric"].isin(selected_positions)]
        
        course_stats = course_analysis_df.groupby("Course").agg({
            "Horse": "count"
        }).reset_index()
        
        course_stats.columns = ["Course", "Total Races"]
        course_stats = course_stats.sort_values("Total Races", ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Most Active Course", course_stats.iloc[0]["Course"] if len(course_stats) > 0 else "N/A")
        with col2:
            st.metric("Races at Top Course", f"{int(course_stats.iloc[0]['Total Races']):,}" if len(course_stats) > 0 else "0")
        
        # Format Total Races with commas for display
        course_stats["Total Races"] = course_stats["Total Races"].apply(lambda x: f"{int(x):,}")
        
        height = get_dataframe_height(course_stats)
        st.dataframe(
            course_stats,
            # width="stretch",
            hide_index=True,
            height=height
        )
    
    with tab3:
        st.subheader("Jockey Performance")
        
        jockey_stats = analysis_df.groupby("Jockey").agg({
            "Finish Position Numeric": ["count", "mean", lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        
        jockey_stats.columns = ["Jockey", "Total Races", "Avg Finish", "Wins", "Top 3 Finishes"]
        jockey_stats["Win Rate %"] = (jockey_stats["Wins"] / jockey_stats["Total Races"] * 100).round(1)
        jockey_stats["Avg Finish"] = jockey_stats["Avg Finish"].round(2)
        
        # Format Total Races with commas
        jockey_stats["Total Races"] = jockey_stats["Total Races"].apply(lambda x: f"{int(x):,}")
        
        jockey_stats = jockey_stats.sort_values("Wins", ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top Jockey (by wins)", jockey_stats.iloc[0]["Jockey"] if len(jockey_stats) > 0 else "N/A")
        with col2:
            st.metric("Most Wins", int(jockey_stats.iloc[0]["Wins"]) if len(jockey_stats) > 0 else 0)
        
        height = get_dataframe_height(jockey_stats)
        st.dataframe(
            jockey_stats,
            # width="stretch",
            hide_index=True,
            height=height
        )
    
    with tab4:
        st.subheader("Overall Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Horses", f"{analysis_df['Horse'].nunique():,}")
            st.metric("Total Jockeys", f"{analysis_df['Jockey'].nunique():,}")
        
        with col2:
            st.metric("Total Trainers", analysis_df["Trainer"].nunique())
            st.metric("Unique Courses", analysis_df["Course"].nunique())
        
        with col3:
            avg_position = analysis_df["Finish Position Numeric"].mean()
            st.metric("Avg Finish Position", f"{avg_position:.2f}" if not pd.isna(avg_position) else "N/A")
            total_prize = analysis_df["Prize"].sum()
            st.metric("Total Prize Money", f"£{total_prize:,.0f}" if not pd.isna(total_prize) else "N/A")
        
        # Distance distribution
        st.subheader("Race Distance Distribution")
        distance_counts = analysis_df["Distance"].value_counts().head(10).reset_index()
        distance_counts.columns = ["Distance", "Count"]
        
        # Format Count with commas
        distance_counts["Count"] = distance_counts["Count"].apply(lambda x: f"{int(x):,}")
        
        height = get_dataframe_height(distance_counts)
        st.dataframe(
            distance_counts,
            # width="stretch",
            hide_index=True,
            height=height
        )

    with tab5:
        st.subheader("Machine Learning Model")
        
        # Load model
        model, metadata, feature_importance = load_model()
        
        if model is None:
            st.warning("⚠️ No trained model found. Train the model first.")
            st.info("The model predicts horse win probability using 18+ features including career stats, recent form, and race context.")
            
            if st.button("🚀 Train Model Now", type="primary"):
                with st.spinner("Training model... This may take 2-3 minutes."):
                    try:
                        # Run training script
                        result = subprocess.run(
                            [sys.executable, "scripts/phase3_build_horse_model.py"],
                            cwd=str(BASE_DIR),
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Model trained successfully!")
                            st.text("Training Output:")
                            st.code(result.stdout[-2000:], language="text")  # Show last 2000 chars
                            st.info("Refresh the page to load the new model.")
                        else:
                            st.error(f"❌ Training failed with error code {result.returncode}")
                            st.code(result.stderr, language="text")
                    except Exception as e:
                        st.error(f"Error running training script: {e}")
        else:
            # Model loaded successfully
            st.success("✅ Model loaded successfully")
            
            # Show metadata
            if metadata:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Type", metadata.get('model_type', 'Unknown'))
                with col2:
                    st.metric("Features", metadata.get('n_features', 0))
                with col3:
                    st.metric("Trained", metadata.get('trained_date', 'Unknown')[:10])
            
            st.markdown("---")
            
            # Feature importance visualization
            if feature_importance is not None:
                st.subheader("📊 Top 15 Most Important Features")
                
                # Get top 15
                top_features = feature_importance.head(15)
                
                # Create bar chart using plotly
                try:
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker=dict(
                            color=top_features['importance'],
                            colorscale='Viridis',
                            showscale=True
                        )
                    ))
                    
                    fig.update_layout(
                        title="Feature Importance (XGBoost)",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        height=500,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                except ImportError:
                    # Fallback to simple bar chart
                    st.bar_chart(top_features.set_index('feature')['importance'])
                
                # Show feature descriptions
                with st.expander("📖 Feature Descriptions"):
                    feature_descriptions = {
                        'avg_last_3_pos': 'Average finishing position in last 3 races (lower = better recent form)',
                        'field_size': 'Number of runners in the race',
                        'class_num': 'Race class (1 = highest quality, 7 = lowest)',
                        'or_change': 'Change in Official Rating since last race (positive = improving)',
                        'career_place_rate': 'Career percentage of top-3 finishes',
                        'class_step': 'Change in race class (negative = stepping up)',
                        'wins_last_3': 'Number of wins in last 3 races',
                        'or_numeric': 'Official Rating (horse ability score)',
                        'career_runs': 'Total career races',
                        'days_since_last': 'Days since last race',
                        'cd_win_rate': 'Win rate at this course/distance combination',
                        'career_win_rate': 'Career win percentage',
                        'race_score': 'Race profitability score (from Phase 2 scorer)',
                        'or_trend_3': '3-race average Official Rating trend',
                        'going_numeric': 'Going/ground condition (1=Firm to 7=Heavy)',
                        'is_turf': 'Whether race is on turf (1) vs all-weather (0)',
                        'career_earnings': 'Total career prize money won',
                        'cd_runs': 'Number of previous runs at this course/distance'
                    }
                    
                    for feat in top_features['feature']:
                        desc = feature_descriptions.get(feat, 'No description available')
                        st.markdown(f"**{feat}**: {desc}")
                
                # Show full feature importance table
                with st.expander("📋 Full Feature Importance Table"):
                    st.dataframe(
                        feature_importance,
                        hide_index=True,
                        height=400
                    )

                    # Top 10 features detailed table (leak-free)
                    with st.expander("📘 Top 10 Features (Leak-free)"):
                        st.markdown(
                            """
    | Feature | Calculation | Description |
    |---|---|---|
    | `field_size` | `ran` (numeric) | Number of runners declared in the race (pre-race feature) |
    | `career_place_rate` | `groupby('horse')['top3'].cumsum().shift(1) / career_runs` | Career percentage of top-3 finishes computed from prior races only |
    | `is_veteran` | `age >= 8` | Binary flag for horses aged 8 or older (possible decline/specialist) |
    | `avg_last_3_pos` | mean of `pos` from last 3 races using `.shift(1)` | Recent form: average finishing position in the three most recent completed races (lower = better) |
    | `or_change` | `or_numeric - prev_or` (uses `.shift(1)`) | Change in Official Rating since previous race (improvement/decline) |
    | `is_pattern` | `pattern.notna()` | Flag indicating Group/Listed (stakes) races — a race-level property |
    | `or_numeric` | numeric conversion of `or` | Official Rating assigned to the horse before the race (published) |
    | `class_num` | numeric extracted from `class_clean` | Race class (1 = highest quality) — same for all runners in a race |
    | `class_step` | `class_num - prev_class` (uses `.shift(1)`) | Movement in class since the horse's previous run (stepping up/down) |
    | `age_vs_avg` | `age - race_mean_age` (grouped by race) | Horse age relative to the race average (captures maturity advantage/disadvantage) |

    Notes:
    - All historical features use `.shift(1)` or equivalent temporal ordering to avoid lookahead leakage.
    - `prize_log` now uses total race prize pool (same value for all horses) to avoid leakage.
    """
                        )
            
            st.markdown("---")
            
            # Retrain button
            st.subheader("🔄 Retrain Model")
            st.info("Retrain the model with latest data or different parameters.")
            
            if st.button("🔄 Retrain Model", type="secondary"):
                with st.spinner("Retraining model... This may take 2-3 minutes."):
                    try:
                        result = subprocess.run(
                            [sys.executable, "scripts/phase3_build_horse_model.py"],
                            cwd=str(BASE_DIR),
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Model retrained successfully!")
                            st.text("Training Output:")
                            st.code(result.stdout[-2000:], language="text")
                            st.info("Clear cache and refresh to load the new model.")
                            
                            # Add cache clear button
                            if st.button("Clear Cache & Refresh"):
                                st.cache_data.clear()
                                st.rerun()
                        else:
                            st.error(f"❌ Retraining failed with error code {result.returncode}")
                            st.code(result.stderr, language="text")
                    except Exception as e:
                        st.error(f"Error running training script: {e}")
    
    with tab6:
        st.subheader("Raw Results")

        # Show number-of-results selector and the filtered results (this uses the
        # full `filtered_df` computed above, but displays a limited subset when
        # the user selects a cap).
        total_filtered = len(filtered_df)
        num_results_options = [25, 50, 75, 100, "All"]
        num_results = st.selectbox(
            "Number of Results to Display",
            options=num_results_options,
            index=1,
            key="num_results_display"
        )

        if num_results != "All":
            display_df = filtered_df.head(num_results).copy()
        else:
            display_df = filtered_df.copy()

        # Format Date column for display only
        if "Date" in display_df.columns:
            display_df["Date"] = display_df["Date"].apply(
                lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) and x.hour == 0 and x.minute == 0 and x.second == 0
                else (x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else "")
            )

        results_text = f"top {num_results}" if num_results != "All" else "all"
        st.subheader(f"Results ({len(display_df):,} of {total_filtered:,} races shown, {results_text} by date)")

        display_columns = [
            "Date", "Course", "Race Name", "Horse", "Finish Position", 
            "Jockey", "Trainer", "Distance", "Going", "Time"
        ]
        display_columns = [col for col in display_columns if col in display_df.columns]

        height = get_dataframe_height(display_df)
        st.dataframe(
            display_df[display_columns],
            hide_index=True,
            height=height,
            # width="stretch"
        )
    
    with tab7:
        st.subheader("Predicted Fixtures (2025-2026)")
        
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
                    st.plotly_chart(fig, width='stretch')
                except ImportError:
                    st.info("Install plotly to see score distribution chart")
                
                # Course breakdown
                st.subheader("Top Courses by Score")
                course_stats = fixtures_scored.groupby('course').agg({
                    'race_score': ['count', 'mean', 'max'],
                    'race_tier': lambda x: (x == 'Tier 1: Focus').sum()
                }).round(1)
                course_stats.columns = ['Total Races', 'Avg Score', 'Max Score', 'Tier 1 Count']
                course_stats = course_stats.sort_values('Max Score', ascending=False).head(15)
                height = get_dataframe_height(course_stats)
                st.dataframe(course_stats, height=height, width='stretch')
                
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
                display_fixtures['date'] = display_fixtures['date'].dt.strftime('%Y-%m-%d')
                display_fixtures['prize'] = display_fixtures['prize'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "")
                display_fixtures['race_score'] = display_fixtures['race_score'].round(1)
                display_fixtures.columns = ['Date', 'Course', 'Class', 'Prize', 'Score', 'Tier', 'Day', 'Surface']
                
                st.info(f"Showing {len(display_fixtures):,} of {len(fixtures_scored):,} predicted fixtures")
                height = get_dataframe_height(display_fixtures, max_height=600)
                st.dataframe(display_fixtures, hide_index=True, height=height, width='stretch')
                
            except Exception as e:
                st.error(f"Error loading predicted fixtures: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning(f"Predicted fixtures file not found. Run `python scripts/score_fixture_calendar.py` to generate predictions.")
            st.info(f"Expected file: {SCORED_FIXTURES_FILE}")
    
    with tab8:
        st.subheader("Betting Watchlist (Strategy-Based)")
        
        # Load betting watchlist
        watchlist_file = BASE_DIR / "data" / "processed" / "betting_watchlist.csv"
        if watchlist_file.exists():
            try:
                watchlist = pd.read_csv(watchlist_file)
                watchlist['date'] = pd.to_datetime(watchlist['date'])
                
                # Overview
                st.markdown("""
                **Betting Strategy Tiers** (from BETTING_STRATEGY.md):
                - **Tier 1: Focus** - Highest ROI potential (Class 1-2, premium courses, mile-middle distance)
                - **Tier 2: Value** - Medium risk/reward (Class 3-4 handicaps, competitive fields)
                - **Tier 3: Avoid** - Low predictability (Class 5-7, weak fields, amateur races)
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Watchlist Races", len(watchlist))
                with col2:
                    tier1_count = (watchlist['betting_tier'] == 'Tier 1: Focus').sum()
                    st.metric("Tier 1 Focus", tier1_count)
                with col3:
                    tier2_count = (watchlist['betting_tier'] == 'Tier 2: Value').sum()
                    st.metric("Tier 2 Value", tier2_count)
                
                st.markdown("---")
                
                # Tier breakdown
                if len(watchlist) > 0:
                    # Show Tier 1 races
                    tier1_races = watchlist[watchlist['betting_tier'] == 'Tier 1: Focus'].copy()
                    if len(tier1_races) > 0:
                        st.subheader(f"🎯 Tier 1: Focus Races ({len(tier1_races)})")
                        st.info("**Strategy**: Best betting opportunities. Class 1-2 races at premium courses (Ascot, Newmarket, York, Doncaster) with 7-12f distance.")
                        
                        # Display Tier 1 races
                        display_cols = ['date', 'course', 'class', 'prize', 'race_score', 'weekday']
                        available = [c for c in display_cols if c in tier1_races.columns]
                        
                        tier1_display = tier1_races[available].copy()
                        tier1_display['date'] = tier1_display['date'].dt.strftime('%Y-%m-%d')
                        if 'prize' in tier1_display.columns:
                            tier1_display['prize'] = tier1_display['prize'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "")
                        if 'race_score' in tier1_display.columns:
                            tier1_display['race_score'] = tier1_display['race_score'].round(1)
                        
                        tier1_display.columns = [c.title() for c in tier1_display.columns]
                        st.dataframe(tier1_display, hide_index=True, width='stretch')
                    
                    # Show Tier 2 races
                    tier2_races = watchlist[watchlist['betting_tier'] == 'Tier 2: Value'].copy()
                    if len(tier2_races) > 0:
                        st.subheader(f"💰 Tier 2: Value Races ({len(tier2_races)})")
                        st.info("**Strategy**: Medium risk/reward. Class 3-4 handicaps with 10-16 runners. Look for well-handicapped horses or 2nd/3rd favorites.")
                        
                        # Display Tier 2 races
                        display_cols = ['date', 'course', 'class', 'prize', 'race_score', 'weekday']
                        available = [c for c in display_cols if c in tier2_races.columns]
                        
                        tier2_display = tier2_races[available].copy()
                        tier2_display['date'] = tier2_display['date'].dt.strftime('%Y-%m-%d')
                        if 'prize' in tier2_display.columns:
                            tier2_display['prize'] = tier2_display['prize'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "")
                        if 'race_score' in tier2_display.columns:
                            tier2_display['race_score'] = tier2_display['race_score'].round(1)
                        
                        tier2_display.columns = [c.title() for c in tier2_display.columns]
                        with st.expander("Show Tier 2 Races", expanded=False):
                            st.dataframe(tier2_display, hide_index=True, width='stretch')
                    
                    # Betting workflow guidance
                    st.markdown("---")
                    st.subheader("📋 Betting Workflow")
                    st.markdown("""
                    **Before Race (48h):**
                    
                    1. Review watchlist races above
                    2. Wait for racecard publication (usually 48h before race)
                    3. Fetch horse entries, jockeys, trainers via API
                    
                    **Day Before Race (24h):**
                    
                    4. Run ML model to predict win probabilities for each horse
                    5. Compare model probabilities to bookmaker odds
                    6. Identify value bets (model prob > market prob + 5% edge)
                    7. Calculate Kelly Criterion bet sizing
                    
                    **Race Day:**
                    
                    8. Place bets 12-24h before race (best odds window)
                    9. Track results and update bankroll
                    10. Record bets for performance analysis
                    """)
                    
                    # Historical performance (if available)
                    historical_file = BASE_DIR / "data" / "processed" / "race_scores_with_betting_tiers.parquet"
                    if historical_file.exists():
                        with st.expander("📊 Historical Tier Performance", expanded=False):
                            df_hist = pd.read_parquet(historical_file)
                            
                            # Calculate stats by tier
                            tier_stats = df_hist.groupby('betting_tier').agg({
                                'race_id': 'count',
                                'race_score': 'mean'
                            }).round(1)
                            tier_stats.columns = ['Total Races', 'Avg Score']
                            tier_stats = tier_stats.sort_values('Avg Score', ascending=False)
                            
                            st.write("**Race counts and average scores by betting tier:**")
                            st.dataframe(tier_stats, hide_index=True, width='stretch')
                            
                            st.caption("Note: Historical tiers use strict criteria. Upcoming predictions use relaxed criteria due to limited data.")
                
                else:
                    st.info("No races in betting watchlist. Run `python scripts/apply_betting_strategy.py` to generate watchlist.")
                    
            except Exception as e:
                st.error(f"Error loading betting watchlist: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("Betting watchlist not found. Run the betting strategy classifier:")
            st.code("python scripts/apply_betting_strategy.py", language="bash")
            st.info(f"Expected file: {watchlist_file}")
    
    with tab9:
        st.subheader("🎲 Today & Tomorrow's Race Predictions")
        
        # Get today's and tomorrow's dates using timezone-aware local time.
        # If the server is UTC/GMT but you want a specific local timezone (e.g., for your region),
        # set the `APP_TIMEZONE` environment variable to a valid IANA timezone string (e.g., 'Europe/London' or 'America/New_York').
        tz_name = os.environ.get('APP_TIMEZONE')
        now_local = get_now_local(tz_name)
        today_str = now_local.strftime('%Y-%m-%d')
        tomorrow_str = (now_local + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Files for today
        today_predictions_file = BASE_DIR / "data" / "processed" / f"predictions_{today_str}.csv"
        today_racecards_file = BASE_DIR / "data" / "raw" / f"racecards_{today_str}.json"
        
        # Files for tomorrow
        tomorrow_predictions_file = BASE_DIR / "data" / "processed" / f"predictions_{tomorrow_str}.csv"
        tomorrow_racecards_file = BASE_DIR / "data" / "raw" / f"racecards_{tomorrow_str}.json"
        
        # Check which days need data
        today_needs_data = not today_predictions_file.exists()
        tomorrow_needs_data = not tomorrow_predictions_file.exists()
        
        # Only show fetch/generate UI if at least one day needs data
        if today_needs_data or tomorrow_needs_data:
            # Determine how many columns we need
            num_days_needing_data = sum([today_needs_data, tomorrow_needs_data])
            
            # Step 1: Fetch Racecards (only for days that need them)
            st.markdown("### Step 1: Fetch Racecards")
            
            if num_days_needing_data == 2:
                col1a, col1b, col1c = st.columns([2, 2, 2])
            elif today_needs_data:
                # Only today needs data
                col1a, col1c = st.columns([3, 2])
                col1b = None
            else:
                # Only tomorrow needs data
                col1b, col1c = st.columns([3, 2])
                col1a = None
            
            if today_needs_data and col1a:
                with col1a:
                    if st.button("📥 Fetch Today's Racecards", type="secondary", width='stretch'):
                        with st.spinner("📡 Fetching racecards from external source... Please wait..."):
                            try:
                                # Run the fetch racecards script
                                result = subprocess.run(
                                    [sys.executable, "scripts/fetch_racecards.py", "--date", today_str],
                                    cwd=str(BASE_DIR),
                                    capture_output=True,
                                    text=True,
                                    timeout=120  # 2 minute timeout
                                )
                                
                                if result.returncode == 0:
                                    st.success("✅ Racecards fetched successfully!")
                                    # Show output
                                    if result.stdout:
                                        with st.expander("📋 Fetch Details"):
                                            st.code(result.stdout, language="text")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to fetch racecards")
                                    with st.expander("View Error Details"):
                                        st.code(result.stderr, language="text")
                                        if result.stdout:
                                            st.code(result.stdout, language="text")
                            
                            except subprocess.TimeoutExpired:
                                st.error("❌ Racecard fetch timed out (>2 minutes)")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                                import traceback
                                with st.expander("View Traceback"):
                                    st.code(traceback.format_exc())
            
            if tomorrow_needs_data and col1b:
                with col1b:
                    if st.button("📥 Fetch Tomorrow's Racecards", type="secondary", width='stretch'):
                        with st.spinner("📡 Fetching tomorrow's racecards... Please wait..."):
                            try:
                                # Run the fetch racecards script for tomorrow
                                result = subprocess.run(
                                    [sys.executable, "scripts/fetch_racecards.py", "--date", tomorrow_str],
                                    cwd=str(BASE_DIR),
                                    capture_output=True,
                                    text=True,
                                    timeout=120  # 2 minute timeout
                                )
                                
                                if result.returncode == 0:
                                    st.success("✅ Tomorrow's racecards fetched successfully!")
                                    # Show output
                                    if result.stdout:
                                        with st.expander("📋 Fetch Details"):
                                            st.code(result.stdout, language="text")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to fetch tomorrow's racecards")
                                    with st.expander("View Error Details"):
                                        st.code(result.stderr, language="text")
                                        if result.stdout:
                                            st.code(result.stdout, language="text")
                            
                            except subprocess.TimeoutExpired:
                                st.error("❌ Tomorrow's racecard fetch timed out (>2 minutes)")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                                import traceback
                                with st.expander("View Traceback"):
                                    st.code(traceback.format_exc())
            
            with col1c:
                # Show racecards status for days that need data
                if today_needs_data:
                    if today_racecards_file.exists():
                        file_time = pd.Timestamp.fromtimestamp(today_racecards_file.stat().st_mtime)
                        time_str = file_time.strftime('%I:%M %p').lstrip('0')
                        st.success(f"✅ Today's racecards\nFetched at {time_str}")
                    else:
                        st.warning("⚠️ No racecards for today")
                
                if tomorrow_needs_data:
                    if tomorrow_racecards_file.exists():
                        file_time = pd.Timestamp.fromtimestamp(tomorrow_racecards_file.stat().st_mtime)
                        time_str = file_time.strftime('%I:%M %p').lstrip('0')
                        st.success(f"✅ Tomorrow's racecards\nFetched at {time_str}")
                    else:
                        st.info("⚠️ No racecards for tomorrow")
            
            st.markdown("---")
            
            # Step 2: Generate Predictions (only for days that need them)
            st.markdown("### Step 2: Generate ML Predictions")
            
            if num_days_needing_data == 2:
                col2a, col2b, col2c = st.columns([2, 2, 1])
            elif today_needs_data:
                # Only today needs data
                col2a, col2c = st.columns([3, 2])
                col2b = None
            else:
                # Only tomorrow needs data
                col2b, col2c = st.columns([3, 2])
                col2a = None
            
            if today_needs_data and col2a:
                with col2a:
                    if st.button("🔄 Generate Today's Predictions", type="primary", width='stretch'):
                        # Check if racecards exist
                        if not today_racecards_file.exists():
                            st.error(f"❌ Racecards not found for {today_str}")
                            st.info(f"Please click '📥 Fetch Today's Racecards' button above first")
                        else:
                            with st.spinner("🤖 Running ML predictions... This may take 1-2 minutes..."):
                                try:
                                    # Run the prediction script
                                    result = subprocess.run(
                                        [sys.executable, "scripts/predict_todays_races.py"],
                                        cwd=str(BASE_DIR),
                                        capture_output=True,
                                        text=True,
                                        timeout=300  # 5 minute timeout
                                    )
                                    
                                    if result.returncode == 0:
                                        st.success("✅ Predictions generated successfully!")
                                        st.balloons()
                                        st.info("🔄 Auto-refreshing in 3 seconds to display new predictions...")
                                        
                                        # Show brief output summary
                                        if result.stdout:
                                            # Extract summary info from output
                                            output_lines = result.stdout.split('\n')
                                            summary_lines = [line for line in output_lines if 'Total horses' in line or 'Total races' in line or 'SAVED' in line]
                                            if summary_lines:
                                                with st.expander("📊 Generation Summary"):
                                                    st.code('\n'.join(summary_lines), language="text")
                                        
                                        # Brief delay so user sees the success message
                                        import time
                                        time.sleep(3)
                                        
                                        # Rerun to show new predictions
                                        st.rerun()
                                    else:
                                        st.error("❌ Prediction generation failed")
                                        with st.expander("View Error Details"):
                                            st.code(result.stderr, language="text")
                                            if result.stdout:
                                                st.code(result.stdout, language="text")
                                
                                except subprocess.TimeoutExpired:
                                    st.error("❌ Prediction generation timed out (>5 minutes)")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                                    import traceback
                                    with st.expander("View Traceback"):
                                        st.code(traceback.format_exc())
            
            if tomorrow_needs_data and col2b:
                with col2b:
                    if st.button("🔄 Generate Tomorrow's Predictions", type="primary", width='stretch'):
                        # Check if racecards exist
                        if not tomorrow_racecards_file.exists():
                            st.error(f"❌ Racecards not found for {tomorrow_str}")
                            st.info(f"Please click '📥 Fetch Tomorrow's Racecards' button above first")
                        else:
                            with st.spinner("🤖 Running ML predictions for tomorrow... This may take 1-2 minutes..."):
                                try:
                                    # Run the prediction script with tomorrow's date
                                    result = subprocess.run(
                                        [sys.executable, "scripts/predict_todays_races.py", "--date", tomorrow_str],
                                        cwd=str(BASE_DIR),
                                        capture_output=True,
                                        text=True,
                                        timeout=300  # 5 minute timeout
                                    )
                                    
                                    if result.returncode == 0:
                                        st.success("✅ Tomorrow's predictions generated successfully!")
                                        st.balloons()
                                        st.info("🔄 Auto-refreshing in 3 seconds to display new predictions...")
                                        
                                        # Show brief output summary
                                        if result.stdout:
                                            # Extract summary info from output
                                            output_lines = result.stdout.split('\n')
                                            summary_lines = [line for line in output_lines if 'Total horses' in line or 'Total races' in line or 'SAVED' in line]
                                            if summary_lines:
                                                with st.expander("📊 Generation Summary"):
                                                    st.code('\n'.join(summary_lines), language="text")
                                        
                                        # Brief delay so user sees the success message
                                        import time
                                        time.sleep(3)
                                        
                                        # Rerun to show new predictions
                                        st.rerun()
                                    else:
                                        st.error("❌ Tomorrow's prediction generation failed")
                                        with st.expander("View Error Details"):
                                            st.code(result.stderr, language="text")
                                            if result.stdout:
                                                st.code(result.stdout, language="text")
                                
                                except subprocess.TimeoutExpired:
                                    st.error("❌ Tomorrow's prediction generation timed out (>5 minutes)")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                                    import traceback
                                    with st.expander("View Traceback"):
                                        st.code(traceback.format_exc())
                
            with col2c:
                # Show status for days that need predictions
                if today_needs_data:
                    if today_predictions_file.exists():
                        file_time = pd.Timestamp.fromtimestamp(today_predictions_file.stat().st_mtime)
                        st.success(f"✅ Today\n{file_time.strftime('%H:%M:%S')}")
                    else:
                        st.warning("⚠️ No predictions for today")
                
                if tomorrow_needs_data:
                    if tomorrow_predictions_file.exists():
                        file_time = pd.Timestamp.fromtimestamp(tomorrow_predictions_file.stat().st_mtime)
                        st.success(f"✅ Tomorrow\n{file_time.strftime('%H:%M:%S')}")
                    else:
                        st.info("⚠️ No tomorrow")
                
                # Refresh button
                if st.button("🔃 Refresh", width='stretch'):
                    st.rerun()
            
            st.markdown("---")
        
        # Load and combine predictions from both days
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
        else:
            # Combine all predictions
            predictions = pd.concat(all_predictions, ignore_index=True)
            
            # st.markdown(f"## 📅 Predictions for Today & Tomorrow")
            
            # Show summary by day
            for day_label in predictions['day_label'].unique():
                day_df = predictions[predictions['day_label'] == day_label]
                day_date = day_df['date'].iloc[0]
                st.success(f"✅ {day_label} ({day_date}): {len(day_df)} horses from {len(day_df['course'].unique())} races")
            
            st.markdown("---")
            
            # Top predictions: top 25 per day, sorted by date (asc) then win % (desc)
            st.markdown("##### 🏆 Top 25 Predictions Per Day")
            
            # Check if odds column exists
            has_odds = 'bookmaker_odds' in predictions.columns
            
            if has_odds:
                display_cols = ['day_label', 'date', 'race_time', 'course', 'horse', 'jockey', 'win_probability', 'place_probability', 'show_probability', 'bookmaker_odds', 'race_class', 'distance_f', 'ofr']
            else:
                display_cols = ['day_label', 'date', 'race_time', 'course', 'horse', 'jockey', 'win_probability', 'place_probability', 'show_probability', 'race_class', 'distance_f', 'ofr']
            
            # Group by date and take top 25 per date by win probability
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
            st.dataframe(top_per_day, hide_index=True, width='stretch', height=height)
            
            # if not has_odds:
                # st.info("💡 **Add live odds:** Run `python scripts/fetch_odds.py --date " + today_str + "` to fetch bookmaker odds and enable value bet detection")
            
            # Show timezone info
            if 'race_time_gmt' in predictions.columns:
                st.caption("⏰ Times shown in **US Eastern Time (ET)** | GMT times available in detailed view")
            
            st.markdown("---")
            
            # Race-by-race breakdown
            st.markdown("##### 📋 Race-by-Race Predictions")
            
            # Group by race
            races = predictions.groupby(['date', 'day_label', 'race_time', 'course', 'race_name']).size().reset_index()[['date', 'day_label', 'race_time', 'course', 'race_name']]
            
            # Race selector
            race_options = [f"{row['day_label']} ({row['date']}) - {row['race_time']} - {row['course']} - {row['race_name'][:40]}" for _, row in races.iterrows()]
            
            selected_race_idx = st.selectbox(
                "Select a race to see detailed predictions:",
                range(len(race_options)),
                format_func=lambda i: race_options[i]
            )
            
            selected_race_info = races.iloc[selected_race_idx]
            
            # Filter predictions for selected race
            race_preds = predictions[
                (predictions['date'] == selected_race_info['date']) &
                (predictions['race_time'] == selected_race_info['race_time']) &
                (predictions['course'] == selected_race_info['course'])
            ].copy()
            
            # Sort by win probability
            race_preds = race_preds.sort_values('win_probability', ascending=False)
            
            # Race details
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
            
            # Top picks summary
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
            
            # Trifecta probability (top 3 finishing in exact order 1-2-3)
            st.markdown("---")
            st.markdown("##### 🎯 Exacta/Trifecta Probabilities")
            
            # Get top 3 horses by win probability
            top_3 = race_preds.nlargest(3, 'win_probability')
            
            if len(top_3) >= 3:
                p1 = top_3.iloc[0]['win_probability']
                p2 = top_3.iloc[1]['win_probability']
                p3 = top_3.iloc[2]['win_probability']
                
                # Calculate trifecta probability (adjusted for conditional probabilities)
                # P(A wins AND B second AND C third) = P(A wins) * P(B wins | A already won) * P(C wins | A,B already won)
                # Approximate using renormalization: P(B second | A won) ≈ p_B / (1 - p_A)
                if p1 < 0.99:  # Avoid division by zero
                    p_second_given_first = p2 / (1 - p1)
                    if (p1 + p2) < 0.99:  # Avoid division by zero
                        p_third_given_first_second = p3 / (1 - p1 - p2)
                        trifecta_prob = p1 * p_second_given_first * p_third_given_first_second
                    else:
                        trifecta_prob = 0
                else:
                    trifecta_prob = 0
                
                # Calculate exacta (top 2 in exact order)
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
                    # Calculate odds for trifecta bet
                    if trifecta_prob > 0:
                        trifecta_odds = (1 / trifecta_prob) - 1
                        st.metric(
                            "💰 Fair Trifecta Odds",
                            f"{trifecta_odds:.1f}/1",
                            help="Fair odds based on model probabilities"
                        )
                    else:
                        st.metric("💰 Fair Trifecta Odds", "N/A")
                
                # Show the predicted top 3
                st.caption(f"🎯 **Predicted 1-2-3:** {top_3.iloc[0]['horse']} → {top_3.iloc[1]['horse']} → {top_3.iloc[2]['horse']}")
                
                # Top horse finishing in top 2 or top 3
                st.markdown("##### 🏆 Top Selection Probabilities")
                st.caption(f"Likelihood that **{top_3.iloc[0]['horse']}** (the favorite) finishes in the money")
                
                # Calculate cumulative probabilities for top horse
                # Note: place_probability = P(2nd), show_probability = P(3rd)
                # So we need to add them to get cumulative probabilities
                
                top_horse_win_prob = top_3.iloc[0]['win_probability']
                top_horse_place_prob = top_3.iloc[0]['place_probability'] if 'place_probability' in top_3.iloc[0] else 0
                top_horse_show_prob = top_3.iloc[0]['show_probability'] if 'show_probability' in top_3.iloc[0] else 0
                
                # Cumulative probabilities
                prob_top_1 = top_horse_win_prob
                prob_top_2 = top_horse_win_prob + top_horse_place_prob  # P(1st) + P(2nd)
                prob_top_3 = top_horse_win_prob + top_horse_place_prob + top_horse_show_prob  # P(1st) + P(2nd) + P(3rd)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "🥇 Win (1st)", 
                        f"{prob_top_1:.1%}",
                        help=f"{top_3.iloc[0]['horse']} finishes 1st"
                    )
                with col2:
                    st.metric(
                        "🥇🥈 Win or Place (1st or 2nd)", 
                        f"{prob_top_2:.1%} (+{top_horse_place_prob:.1%})",
                        help=f"{top_3.iloc[0]['horse']} finishes in top 2"
                    )
                with col3:
                    st.metric(
                        "🥇🥈🥉 Win, Place, or Show (top 3)", 
                        f"{prob_top_3:.1%} (+{top_horse_show_prob:.1%})",
                        help=f"{top_3.iloc[0]['horse']} finishes in top 3"
                    )
            
            # Predictions table
            st.markdown("##### 🐎 All Horse Predictions")
            st.caption("📊 Form shows recent race finishes (read right to left: rightmost = most recent race). Lower numbers = better finishes. 1 = Won, 2 = 2nd, 3 = 3rd, etc.")
            st.caption("💰 Model Odds show the fair value based on probabilities. Compare to bookmaker odds to find value bets!")
            
            display_cols = ['horse', 'jockey', 'win_probability', 'win_odds_fractional', 'place_probability', 'place_odds_fractional', 'show_probability', 'show_odds_fractional', 'age', 'weight_lbs', 'ofr', 'form']
            display_df = race_preds[display_cols].copy()
            
            # Calculate cumulative probabilities for each horse
            display_df['top_2_prob'] = race_preds['win_probability'] + race_preds['place_probability']
            display_df['top_3_prob'] = race_preds['win_probability'] + race_preds['place_probability'] + race_preds['show_probability']
            
            # Add rankings for each category before formatting
            display_df['win_rank'] = race_preds['win_probability'].rank(ascending=False, method='min').astype(int)
            display_df['place_rank'] = race_preds['place_probability'].rank(ascending=False, method='min').astype(int)
            display_df['show_rank'] = race_preds['show_probability'].rank(ascending=False, method='min').astype(int)
            
            # Format probabilities
            display_df['win_probability'] = display_df['win_probability'].apply(lambda x: f"{x:.1%}")
            display_df['place_probability'] = display_df['place_probability'].apply(lambda x: f"{x:.1%}")
            display_df['show_probability'] = display_df['show_probability'].apply(lambda x: f"{x:.1%}")
            display_df['top_2_prob'] = display_df['top_2_prob'].apply(lambda x: f"{x:.1%}")
            display_df['top_3_prob'] = display_df['top_3_prob'].apply(lambda x: f"{x:.1%}")
            
            # Reorder and rename columns
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
            
            # Sort by win rank
            display_df = display_df.sort_values('Win Rank')
            
            st.dataframe(display_df, hide_index=True, width='stretch')
            
            # Handicap Analysis for this specific race
            st.markdown("---")
            st.markdown("##### 🎯 Handicap Analysis - This Race")
            
            # Calculate race-specific metrics
            min_weight = race_preds['weight_lbs'].min()
            max_weight = race_preds['weight_lbs'].max()
            avg_weight = race_preds['weight_lbs'].mean()
            weight_spread = max_weight - min_weight
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Weight Range", f"{min_weight:.0f} - {max_weight:.0f} lbs")
            with col2:
                st.metric("Average Weight", f"{avg_weight:.1f} lbs")
            with col3:
                st.metric("Weight Spread", f"{weight_spread:.0f} lbs")
            with col4:
                # Check if tightly handicapped (5+ horses within 3 lbs)
                weights_grouped = race_preds['weight_lbs'].value_counts()
                tightly_handicapped = any(count >= 5 for count in weights_grouped.values) or weight_spread <= 10
                st.metric("Handicap Type", "Tight" if tightly_handicapped else "Spread")
            
            # Identify handicap angles for this race
            handicap_angles = []
            
            # 1. Class droppers in this race
            if 'class_step' in race_preds.columns:
                class_droppers_race = race_preds[race_preds['class_step'] < -0.5]
                if len(class_droppers_race) > 0:
                    for _, horse in class_droppers_race.iterrows():
                        handicap_angles.append({
                            'Horse': horse['horse'],
                            'Angle': '⬇️ Class Dropper',
                            'Win %': f"{horse['win_probability']:.1%}",
                            'Details': f"Down {abs(horse['class_step']):.1f} classes",
                            'Action': 'BET' if horse['win_probability'] > 0.5 else 'WATCH'
                        })
            
            # 2. Well-in horses (OR improved)
            if 'or_change' in race_preds.columns:
                well_in_race = race_preds[race_preds['or_change'] > 3]
                if len(well_in_race) > 0:
                    for _, horse in well_in_race.iterrows():
                        handicap_angles.append({
                            'Horse': horse['horse'],
                            'Angle': '📈 Well-In (OR ↑)',
                            'Win %': f"{horse['win_probability']:.1%}",
                            'Details': f"OR +{horse['or_change']:.0f} to {horse['ofr']}",
                            'Action': 'BET' if horse['win_probability'] > 0.4 else 'WATCH'
                        })
            
            # 3. Bottom weight horses
            bottom_weight_horses = race_preds[race_preds['weight_lbs'] == min_weight]
            for _, horse in bottom_weight_horses.iterrows():
                if horse['win_probability'] > 0.2:
                    handicap_angles.append({
                        'Horse': horse['horse'],
                        'Angle': '⚖️ Bottom Weight',
                        'Win %': f"{horse['win_probability']:.1%}",
                        'Details': f"{horse['weight_lbs']:.0f} lbs (lightest)",
                        'Action': 'VALUE' if horse['win_probability'] > 0.25 else 'LONGSHOT'
                    })
            
            # 4. Top weight horses (potential traps)
            top_weight_horses = race_preds[race_preds['weight_lbs'] == max_weight]
            for _, horse in top_weight_horses.iterrows():
                handicap_angles.append({
                    'Horse': horse['horse'],
                    'Angle': '⚠️ Top Weight' if horse['win_probability'] < 0.3 else '💪 Top Weight (strong)',
                    'Win %': f"{horse['win_probability']:.1%}",
                    'Details': f"{horse['weight_lbs']:.0f} lbs (heaviest)",
                    'Action': 'AVOID' if horse['win_probability'] < 0.25 else 'CONSIDER'
                })
            
            # 5. Young horses with good weights
            if 'age' in race_preds.columns:
                young_advantage = race_preds[
                    (race_preds['age'] <= 4) & 
                    (race_preds['weight_lbs'] < avg_weight) &
                    (race_preds['win_probability'] > 0.3)
                ]
                for _, horse in young_advantage.iterrows():
                    handicap_angles.append({
                        'Horse': horse['horse'],
                        'Angle': '🌟 Young + Light Weight',
                        'Win %': f"{horse['win_probability']:.1%}",
                        'Details': f"Age {horse['age']}, {horse['weight_lbs']:.0f} lbs",
                        'Action': 'BET' if horse['win_probability'] > 0.4 else 'VALUE'
                    })
            
            if handicap_angles:
                angles_df = pd.DataFrame(handicap_angles)
                
                # Color code by action
                st.dataframe(
                    angles_df,
                    hide_index=True,
                    width='stretch',
                    height=min(300, len(angles_df) * 35 + 38)
                )
                
                # Interpretation guide
                with st.expander("📖 How to Use Handicap Angles"):
                    st.markdown("""
                    **Action Meanings:**
                    - **BET**: Strong opportunity based on handicap advantage + model probability
                    - **VALUE**: Potential value if bookmaker odds are generous
                    - **WATCH**: Interesting angle but needs confirmation (check odds/form)
                    - **LONGSHOT**: Low probability but handicap suggests possible upset
                    - **AVOID**: Top weight with insufficient win probability
                    - **CONSIDER**: Top weight but strong enough to overcome burden
                    
                    **Key Handicap Principles:**
                    - **Class Droppers**: Horse competed at higher level, now facing easier competition
                    - **Well-In**: Official Rating increased but weight hasn't caught up yet
                    - **Bottom Weight**: Handicapper rates them weakest, but model disagrees
                    - **Top Weight**: Carrying extra weight - only back if very strong
                    - **Young + Light**: Natural advantages that compound
                    """)
            else:
                st.info("No specific handicap angles identified for this race")
            
            st.markdown("---")
            
            # Value Bet Calculator
            st.markdown("---")
            st.markdown("##### 💰 Value Bet Calculator")
            st.caption("Compare model odds to bookmaker odds to identify value betting opportunities")
            
            # Interactive calculator
            with st.expander("🧮 Calculate Value Bet (Enter Bookmaker Odds)", expanded=False):
                st.markdown("**How to use:**")
                st.markdown("1. Look up bookmaker's odds for a horse (e.g., Bet365, William Hill)")
                st.markdown("2. Enter the decimal odds below (e.g., 4.5)")
                st.markdown("3. See if there's value compared to model's fair odds")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Select horse
                    horses_list = race_preds['horse'].tolist()
                    selected_horse = st.selectbox(
                        "Select Horse",
                        horses_list,
                        key=f"vb_horse_{selected_race_idx}"
                    )
                
                with col2:
                    # Enter bookmaker odds
                    bookmaker_odds_input = st.number_input(
                        "Bookmaker Decimal Odds",
                        min_value=1.01,
                        max_value=1000.0,
                        value=3.0,
                        step=0.1,
                        key=f"vb_odds_{selected_race_idx}"
                    )
                
                # Calculate value bet
                horse_data = race_preds[race_preds['horse'] == selected_horse].iloc[0]
                model_prob = horse_data['win_probability']
                model_decimal_odds = 1 / model_prob
                model_fractional = horse_data['win_odds_fractional']
                
                bookmaker_implied_prob = 1 / bookmaker_odds_input
                edge = model_prob - bookmaker_implied_prob
                edge_pct = edge * 100
                
                # Display results
                st.markdown("---")
                st.markdown(f"**Analysis for {selected_horse}:**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Model Win %",
                        f"{model_prob:.1%}",
                        help="Model's predicted win probability"
                    )
                
                with col2:
                    st.metric(
                        "Model Odds",
                        f"{model_fractional} ({model_decimal_odds:.2f})",
                        help="Fair odds based on model probability"
                    )
                
                with col3:
                    st.metric(
                        "Bookmaker Implied %",
                        f"{bookmaker_implied_prob:.1%}",
                        help="Bookmaker's implied probability"
                    )
                
                with col4:
                    delta_color = "normal" if edge > 0 else "inverse"
                    st.metric(
                        "Edge",
                        f"{edge_pct:+.1f}%",
                        delta=f"{edge_pct:+.1f}%",
                        delta_color=delta_color,
                        help="Positive edge = value bet opportunity"
                    )
                
                # Value bet recommendation
                st.markdown("---")
                
                if edge >= 0.05:  # 5%+ edge
                    st.success(f"✅ **VALUE BET!** Edge: {edge_pct:+.1f}%")
                    st.markdown(f"**Recommendation:** BACK {selected_horse}")
                    st.markdown(f"- Model says: {model_prob:.1%} chance ({model_fractional})")
                    st.markdown(f"- Bookmaker offers: {bookmaker_odds_input:.2f} decimal ({bookmaker_implied_prob:.1%} implied)")
                    st.markdown(f"- **You have a {edge_pct:.1f}% edge over the bookmaker**")
                elif edge >= 0.02:  # 2-5% edge
                    st.info(f"⚖️ **MARGINAL VALUE** Edge: {edge_pct:+.1f}%")
                    st.markdown(f"Small edge detected. Consider bet size and variance.")
                elif edge >= -0.02:  # Close to fair
                    st.warning(f"📊 **FAIR ODDS** Edge: {edge_pct:+.1f}%")
                    st.markdown(f"Bookmaker odds closely match model prediction. No clear edge.")
                else:  # Negative edge
                    st.error(f"❌ **NO VALUE** Edge: {edge_pct:+.1f}%")
                    st.markdown(f"**Recommendation:** AVOID - Bookmaker odds are worse than model's fair value")
                    st.markdown(f"- Model says: {model_prob:.1%} ({model_fractional})")
                    st.markdown(f"- Bookmaker offers: {bookmaker_odds_input:.2f} ({bookmaker_implied_prob:.1%} implied)")
            
            # Quick reference table for all horses
            with st.expander("📊 Quick Reference - All Horses' Fair Odds"):
                reference_df = race_preds[['horse', 'win_probability', 'win_odds_decimal', 'win_odds_fractional']].copy()
                reference_df['win_probability'] = reference_df['win_probability'].apply(lambda x: f"{x:.1%}")
                reference_df['win_odds_decimal'] = reference_df['win_odds_decimal'].apply(lambda x: f"{x:.2f}")
                reference_df.columns = ['Horse', 'Model Win %', 'Fair Decimal Odds', 'Fair Fractional Odds']
                
                st.dataframe(reference_df, hide_index=True, width='stretch')
                
                st.markdown("**💡 Value Betting Rule:**")
                st.markdown("- ✅ BET if Bookmaker Odds **>** Fair Odds (positive edge)")
                st.markdown("- ❌ AVOID if Bookmaker Odds **<** Fair Odds (negative edge)")
                st.markdown("- Example: Fair odds 3.5, Bookmaker offers 4.5 → **VALUE BET!**")
            
            # Feature importance for this prediction
            with st.expander("🔍 Key Factors (Top Horse)"):
                top_horse = race_preds.iloc[0]
                
                st.markdown(f"**{top_horse['horse']}** - Predicted Win: {top_horse['win_probability']:.1%}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Career Win Rate", f"{top_horse['career_win_rate']:.1%}")
                    st.metric("Career Runs", int(top_horse['career_runs']))
                with col2:
                    st.metric("CD Win Rate", f"{top_horse['cd_win_rate']:.1%}")
                    st.metric("CD Runs", int(top_horse['cd_runs']))
                with col3:
                    st.metric("Recent Form (Avg Pos)", f"{top_horse['avg_last_3_pos']:.1f}")
                    st.metric("Wins Last 3", int(top_horse['wins_last_3']))
            
            # st.markdown("---")
            
            # # Instructions for fetching odds
            # with st.expander("🔧 How to Add Live Odds"):
            #     st.markdown("""
            #     **To enable value bet detection, integrate odds data:**
                
            #     1. **Use The Odds API** - https://the-odds-api.com
            #        - Free tier: 500 requests/month
            #        - Covers major bookmakers
                
            #     2. **Or scrape odds from:**
            #        - Oddschecker.com
            #        - Betfair Exchange
            #        - Racing Post
                
            #     3. **Store odds in CSV** alongside predictions
                
            #     4. **Calculate value automatically:**
            #        ```python
            #        value = model_prob - (1 / decimal_odds)
            #        if value > 0.05:  # 5% edge minimum
            #            # This is a value bet
            #        ```
            #     """)
        
        # If no predictions exist for either day, show watchlist
        if not today_predictions_file.exists() and not tomorrow_predictions_file.exists():
            st.warning(f"⏳ No predictions found for today ({today_str}) or tomorrow ({tomorrow_str})")
            st.info("💡 Use the buttons above to fetch racecards and generate predictions.")
            
            # Show future watchlist
            st.markdown("---")
            st.markdown("### 📅 Upcoming High-Value Races")
            st.caption("Target races identified by profitability scoring system")
            
            watchlist_file = BASE_DIR / "data" / "processed" / "betting_watchlist.csv"
            
            if watchlist_file.exists():
                watchlist = pd.read_csv(watchlist_file)
                watchlist['date'] = pd.to_datetime(watchlist['date'])
                
                future = watchlist[watchlist['date'] > pd.Timestamp.now()].copy()
                
                if len(future) > 0:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Races", len(future))
                    with col2:
                        avg_score = future['race_score'].mean()
                        st.metric("⭐ Avg Score", f"{avg_score:.1f}/100")
                    with col3:
                        total_prize = future['prize'].sum()
                        st.metric("💰 Total Prize Money", f"£{total_prize:,.0f}")
                    with col4:
                        unique_courses = future['course'].nunique()
                        st.metric("🏇 Courses", unique_courses)
                    
                    st.markdown("---")
                    
                    # Top 10 races
                    st.markdown("##### 🎯 Top 10 Target Races (By Score)")
                    future_sorted = future.nlargest(10, 'race_score')
                    
                    display_cols = ['date', 'course', 'class', 'dist_f', 'prize', 'race_score']
                    display_df = future_sorted[display_cols].copy()
                    
                    # Format columns
                    display_df['date'] = display_df['date'].dt.strftime('%a %d %b %Y')
                    display_df['prize'] = display_df['prize'].apply(lambda x: f"£{x:,.0f}")
                    display_df['dist_f'] = display_df['dist_f'].apply(lambda x: f"{x}f")
                    display_df['race_score'] = display_df['race_score'].apply(lambda x: f"{x:.0f}")
                    display_df.columns = ['Date', 'Course', 'Class', 'Distance', 'Prize', 'Score']
                    
                    st.dataframe(display_df, hide_index=True, width='stretch')
                    
                    st.markdown("---")
                    st.caption("⏰ **Tip:** Fetch racecards 24-48 hours before race date for best results")
                    st.caption("💡 Race scores are based on class quality, prize money, course tier, and race patterns")
                else:
                    st.info("✨ No upcoming races in watchlist")
            else:
                st.info("💡 Run `python scripts/apply_betting_strategy.py` to generate race watchlist")
            
            # Handicap Opportunities Summary (shown at bottom for reference)
            st.markdown("---")
            st.markdown("##### 🎯 Handicap Betting Opportunities")
            st.caption("Smart handicap plays identified by the model")
            
            # Explanation expander
            with st.expander("📖 What Are Handicap Angles?"):
                st.markdown("""
                **Handicap racing** uses a weight system to level the playing field. Better horses carry more weight, weaker horses carry less. 
                This creates specific betting opportunities when the handicapper gets it wrong or circumstances change:
                
                **⬇️ Class Droppers**
                - Horse competed in higher class races (e.g., Class 2) and now drops to lower class (e.g., Class 4)
                - Usually carrying similar weight but facing easier competition
                - **Edge**: Experience advantage + physical ability gap
                - **Look for**: Win probability > 50% with class drop of 1+ levels
                
                **📈 Well-In Horses (Rising Official Rating)**
                - Official Rating (OR) has increased by 3+ points recently
                - Weight assignment hasn't caught up with improved form yet
                - **Edge**: Horse is better than the handicapper thinks
                - **Look for**: OR change > +3 with win probability > 40%
                
                **⚖️ Bottom Weight Opportunities**
                - Lightest horse in the race (handicapper rates them weakest)
                - But model gives them a reasonable chance (>20% win probability)
                - **Edge**: Handicapper underestimating ability, weight advantage
                - **Look for**: Minimum weight in race + improving recent form
                
                **⚠️ Top Weight Traps (Avoid These)**
                - Heaviest horse in race carrying maximum burden
                - Low win probability (<30%) despite high rating
                - **Trap**: Public over-bets "best handicapped horse"
                - **Look for**: Maximum weight + low model probability = DON'T BET
                
                **💡 Betting Strategy:**
                - Class Droppers + Well-In horses = **Priority bets** when odds are fair
                - Bottom Weights = **Value longshots** if odds > 6.0
                - Top Weights = **Avoid** unless win probability > 40%
                """)
            
            # Calculate handicap opportunities from today/tomorrow predictions if available
            if today_predictions_file.exists() or tomorrow_predictions_file.exists():
                # Load both prediction files
                all_pred_dfs = []
                if today_predictions_file.exists():
                    all_pred_dfs.append(pd.read_csv(today_predictions_file))
                if tomorrow_predictions_file.exists():
                    all_pred_dfs.append(pd.read_csv(tomorrow_predictions_file))
                
                if all_pred_dfs:
                    predictions_for_handicap = pd.concat(all_pred_dfs, ignore_index=True)
                    
                    # Calculate handicap opportunities
                    handicap_opportunities = []
                    
                    # 1. Class Droppers (stepping down in class with high win %)
                    if 'class_step' in predictions_for_handicap.columns:
                        class_droppers = predictions_for_handicap[
                            (predictions_for_handicap['class_step'] < -0.5) & 
                            (predictions_for_handicap['win_probability'] > 0.5)
                        ].copy()
                        for _, horse in class_droppers.iterrows():
                            handicap_opportunities.append({
                                'date': horse['date'],
                                'day_label': horse['day_label'],
                                'race_time': horse['race_time'],
                                'course': horse['course'],
                                'horse': horse['horse'],
                                'opportunity': '⬇️ Class Dropper',
                                'win_prob': horse['win_probability'],
                                'details': f"Class {horse['race_class']} (down from Class {int(horse['race_class'] + abs(horse['class_step']))})"
                            })
                    
                    # 2. Well-In Horses (OR increased, high win %)
                    if 'or_change' in predictions_for_handicap.columns:
                        well_in = predictions_for_handicap[
                            (predictions_for_handicap['or_change'] > 3) & 
                            (predictions_for_handicap['win_probability'] > 0.4)
                        ].copy()
                        for _, horse in well_in.iterrows():
                            handicap_opportunities.append({
                                'date': horse['date'],
                                'day_label': horse['day_label'],
                                'race_time': horse['race_time'],
                                'course': horse['course'],
                                'horse': horse['horse'],
                                'opportunity': '📈 Well-In',
                                'win_prob': horse['win_probability'],
                                'details': f"OR +{horse['or_change']:.0f} (now {horse['ofr']})"
                            })
                    
                    # 3. Bottom-Weight Value (lightest in race with decent win %)
                    for race_key, race_df in predictions_for_handicap.groupby(['date', 'course', 'race_time']):
                        min_weight = race_df['weight_lbs'].min()
                        bottom_weight_horses = race_df[
                            (race_df['weight_lbs'] == min_weight) & 
                            (race_df['win_probability'] > 0.2)
                        ]
                        for _, horse in bottom_weight_horses.iterrows():
                            handicap_opportunities.append({
                                'date': horse['date'],
                                'day_label': horse['day_label'],
                                'race_time': horse['race_time'],
                                'course': horse['course'],
                                'horse': horse['horse'],
                                'opportunity': '⚖️ Bottom Weight',
                                'win_prob': horse['win_probability'],
                                'details': f"{horse['weight_lbs']:.0f} lbs (lightest)"
                            })
                    
                    # 4. Top-Weight Traps (heaviest in race with low win %)
                    for race_key, race_df in predictions_for_handicap.groupby(['date', 'course', 'race_time']):
                        max_weight = race_df['weight_lbs'].max()
                        top_weight_traps = race_df[
                            (race_df['weight_lbs'] == max_weight) & 
                            (race_df['win_probability'] < 0.3)
                        ]
                        for _, horse in top_weight_traps.iterrows():
                            handicap_opportunities.append({
                                'date': horse['date'],
                                'day_label': horse['day_label'],
                                'race_time': horse['race_time'],
                                'course': horse['course'],
                                'horse': horse['horse'],
                                'opportunity': '⚠️ Top Weight Trap',
                                'win_prob': horse['win_probability'],
                                'details': f"{horse['weight_lbs']:.0f} lbs (heaviest, avoid)"
                            })
                    
                    if handicap_opportunities:
                        opp_df = pd.DataFrame(handicap_opportunities)
                        opp_df = opp_df.sort_values('win_prob', ascending=False)
                        
                        # Format for display
                        display_opp = opp_df.copy()
                        display_opp['win_prob'] = display_opp['win_prob'].apply(lambda x: f"{x:.1%}")
                        display_opp.columns = ['Date', 'Day', 'Race Time', 'Course', 'Horse', 'Type', 'Win %', 'Details']
                        
                        st.dataframe(
                            display_opp,
                            hide_index=True,
                            width = 'stretch',
                            height=min(400, len(display_opp) * 35 + 38)
                        )
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            class_drop_count = len([o for o in handicap_opportunities if 'Class Dropper' in o['opportunity']])
                            st.metric("⬇️ Class Droppers", class_drop_count)
                        with col2:
                            well_in_count = len([o for o in handicap_opportunities if 'Well-In' in o['opportunity']])
                            st.metric("📈 Well-In Horses", well_in_count)
                        with col3:
                            bottom_wt_count = len([o for o in handicap_opportunities if 'Bottom Weight' in o['opportunity']])
                            st.metric("⚖️ Bottom Weights", bottom_wt_count)
                        with col4:
                            trap_count = len([o for o in handicap_opportunities if 'Trap' in o['opportunity']])
                            st.metric("⚠️ Traps to Avoid", trap_count)
                    else:
                        st.info("No significant handicap opportunities identified today")
            else:
                st.info("No prediction data available to analyze handicap opportunities")


if __name__ == "__main__":
    main()
