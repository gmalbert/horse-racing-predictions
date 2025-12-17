"""
Horse Racing Predictions - Streamlit App

Displays UK horse race results with filtering capabilities.
"""
import pandas as pd
import streamlit as st
from pathlib import Path


BASE_DIR = Path(__file__).parent
PARQUET_FILE = BASE_DIR / "data" / "processed" / "all_gb_races.parquet"
CSV_FILE = BASE_DIR / "data" / "processed" / "all_gb_races.csv"
LOGO_FILE = BASE_DIR / "data" / "logo.png"


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

    # Upcoming schedule (fixtures) - show in an expander
    fixtures_file = BASE_DIR / "data" / "processed" / "bha_2026_ascot_newmarket_doncaster_york.csv"
    if fixtures_file.exists():
        try:
            fixtures = pd.read_csv(fixtures_file)
            # Parse Date column if present and sort chronologically
            if "Date" in fixtures.columns:
                fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
                fixtures = fixtures.sort_values("Date")

            with st.expander("Upcoming Schedule 🗓️", expanded=False):
                show_cols = [c for c in ["Date", "Course", "Time", "Type", "Surface"] if c in fixtures.columns]
                fixtures_display = fixtures.copy()
                if "Date" in fixtures_display.columns:
                    fixtures_display["Date"] = fixtures_display["Date"].dt.strftime("%Y-%m-%d")

                st.dataframe(fixtures_display[show_cols].head(200), width="stretch", hide_index=True)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏇 Horses", "🏟️ Courses", "👤 Jockeys", "📈 Overall", "🗃️ Raw Data"])
    
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
            width="stretch",
            hide_index=True,
            height=height
        )


if __name__ == "__main__":
    main()
