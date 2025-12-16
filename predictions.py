"""
Horse Racing Predictions - Streamlit App

Displays UK horse race results with filtering capabilities.
"""
import pandas as pd
import streamlit as st
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "processed" / "uk_horse_races.csv"
LOGO_FILE = BASE_DIR / "data" / "logo.png"


@st.cache_data
def load_data():
    """Load and cache the UK horse races CSV."""
    df = pd.read_csv(DATA_FILE)
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
    
    st.title("🏇 Equine Edge")
    st.markdown("---")

    # Load data
    df = load_data()

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
    positions = sorted([int(p) for p in df["Finish Position"].dropna().unique() if str(p).isdigit()])
    selected_positions = st.sidebar.multiselect(
        "Finish Position",
        options=positions,
        default=None,
        placeholder="All positions"
    )

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
        filtered_df = filtered_df[filtered_df["Finish Position"].isin(selected_positions)]

    # Sort by date descending
    filtered_df = filtered_df.sort_values("Date", ascending=False)
    
    # Number of results to display - on main page
    total_filtered = len(filtered_df)
    num_results_options = [25, 50, 75, 100, "All"]
    num_results = st.selectbox(
        "Number of Results to Display",
        options=num_results_options,
        index=1  # Default to 50
    )
    
    # Apply result limit
    if num_results != "All":
        filtered_df = filtered_df.head(num_results)

    # Format date column to remove time if it's 00:00:00
    filtered_df["Date"] = filtered_df["Date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if x.hour == 0 and x.minute == 0 and x.second == 0 else x.strftime("%Y-%m-%d %H:%M:%S")
    )

    # Display results
    results_text = f"top {num_results}" if num_results != "All" else "all"
    st.subheader(f"Results ({len(filtered_df):,} of {total_filtered:,} races shown, {results_text} by date)")
    
    # Select key columns to display
    display_columns = [
        "Date", "Course", "Race Name", "Horse", "Finish Position", 
        "Jockey", "Trainer", "Distance", "Going", "Time"
    ]
    
    # Filter to only existing columns
    display_columns = [col for col in display_columns if col in filtered_df.columns]
    
    height = get_dataframe_height(filtered_df)
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True,
        height=height
    )

    # Show summary stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("Summary")
    st.sidebar.metric("Total Races in Dataset", f"{len(df):,}")
    st.sidebar.metric("Filtered Races", f"{len(filtered_df):,}")


if __name__ == "__main__":
    main()
