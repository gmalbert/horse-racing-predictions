"""
Historical Data Explorer - Horse Racing Predictions

Provides comprehensive data exploration and analysis tools for historical race data.
Includes horse performance, course statistics, jockey analysis, and betting strategies.
"""
import pandas as pd
import streamlit as st
from pathlib import Path

# Import shared utilities
from shared.utils import (
    load_data, get_dataframe_height, safe_st_call, BASE_DIR,
    SCORED_FIXTURES_FILE, HAS_PLOTLY
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def main():
    st.set_page_config(
        page_title="Historical Data Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Historical Data Explorer")
    
    # Navigation hint back to main page
    st.info("💡 Use the sidebar to navigate back to the main Predictions page", icon="ℹ️")
    
    # Helper to show filter instruction
    def show_filter_hint():
        st.info("📊 **Filters available:** Adjust filters in the left sidebar to narrow down results", icon="ℹ️")
    
    # Load data
    df = load_data()
    
    # Inform user about data scope
    st.info("📊 **Recent Data Only**: To keep the app fast, we're showing the last 5 years of races. Don't worry—the AI model learned from all past races for smart predictions!", icon="ℹ️")
    
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

    # Show summary stats in sidebar
    st.sidebar.subheader("Summary")
    st.sidebar.metric("Total Races in Dataset", f"{len(df):,}")
    st.sidebar.metric("Filtered Races", f"{len(filtered_df):,}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏇 Horses",
        "🏟️ Courses",
        "👤 Jockeys",
        "📈 Overall",
        "🗃️ Raw Data",
        "🎯 Betting Watchlist"
    ])
    
    with tab1:
        display_horses_tab(df, filtered_df, selected_years, selected_courses, horse_name, selected_positions, show_filter_hint)
    
    with tab2:
        display_courses_tab(df, filtered_df, selected_years, selected_courses, horse_name, selected_positions, show_filter_hint)
    
    with tab3:
        display_jockeys_tab(filtered_df, show_filter_hint)
    
    with tab4:
        display_overall_tab(filtered_df, show_filter_hint)
    
    with tab5:
        display_raw_data_tab(filtered_df, show_filter_hint)
    
    with tab6:

        display_betting_watchlist_tab()


def display_horses_tab(df, filtered_df, selected_years, selected_courses, horse_name, selected_positions, show_filter_hint):
    """Display horse performance statistics"""
    show_filter_hint()
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
    
    horse_stats = analysis_df.groupby("Horse", observed=False).agg({
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
        hide_index=True,
        height=height
    )


def display_courses_tab(df, filtered_df, selected_years, selected_courses, horse_name, selected_positions, show_filter_hint):
    """Display course statistics"""
    show_filter_hint()
    st.subheader("Course Statistics")
    
    # For course stats, apply all filters but use ALL matching data
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
    
    course_stats = course_analysis_df.groupby("Course", observed=False).agg({
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
        hide_index=True,
        height=height
    )


def display_jockeys_tab(filtered_df, show_filter_hint):
    """Display jockey performance statistics"""
    show_filter_hint()
    st.subheader("Jockey Performance")
    
    analysis_df = filtered_df.copy()
    analysis_df["Finish Position Numeric"] = pd.to_numeric(analysis_df["Finish Position"], errors='coerce')
    
    jockey_stats = analysis_df.groupby("Jockey", observed=False).agg({
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
        hide_index=True,
        height=height
    )


def display_overall_tab(filtered_df, show_filter_hint):
    """Display overall statistics"""
    show_filter_hint()
    st.subheader("Overall Statistics")
    
    analysis_df = filtered_df.copy()
    analysis_df["Finish Position Numeric"] = pd.to_numeric(analysis_df["Finish Position"], errors='coerce')
    
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
        hide_index=True,
        height=height
    )


def display_raw_data_tab(filtered_df, show_filter_hint):
    """Display raw results table"""
    show_filter_hint()
    st.subheader("Raw Results")

    # Show number-of-results selector
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
        height=height
    )


def display_betting_watchlist_tab():
    """Display betting watchlist based on strategy tiers"""
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
                    st.dataframe(tier1_display, hide_index=True)
                
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
                        tier_stats = df_hist.groupby('betting_tier', observed=False).agg({
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


if __name__ == "__main__":
    main()
