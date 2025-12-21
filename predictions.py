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


BASE_DIR = Path(__file__).parent
PARQUET_FILE = BASE_DIR / "data" / "processed" / "race_scores.parquet"  # Using cleaned data with scores
CSV_FILE = BASE_DIR / "data" / "processed" / "all_gb_races.csv"
LOGO_FILE = BASE_DIR / "data" / "logo.png"
MODEL_FILE = BASE_DIR / "models" / "horse_win_predictor.pkl"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "models" / "feature_importance.csv"
METADATA_FILE = BASE_DIR / "models" / "model_metadata.pkl"


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

    # Top predictive races expander
    with st.expander("🎯 Top Predictive Races (Tier 1 Focus)", expanded=False):
        if 'race_tier' in df.columns and 'race_score' in df.columns:
            # Get Tier 1 Focus races on or after today
            today = pd.Timestamp.today().normalize()
            tier1_races = df[df['race_tier'] == 'Tier 1: Focus'].copy()
            
            # Filter to future races only
            if 'Date' in tier1_races.columns:
                tier1_races['Date'] = pd.to_datetime(tier1_races['Date'])
                tier1_races = tier1_races[tier1_races['Date'] >= today]
            
            # Group by unique race and get the best score
            race_cols = ['Date', 'Course', 'Race Name', 'Class', 'Distance', 'Prize', 'race_score']
            available_cols = [c for c in race_cols if c in tier1_races.columns]
            
            if len(tier1_races) > 0:
                # Get unique races (deduplicate by race_id if available)
                if 'race_id' in tier1_races.columns:
                    tier1_unique = tier1_races.groupby('race_id')[available_cols].first().reset_index(drop=True)
                else:
                    tier1_unique = tier1_races[available_cols].drop_duplicates()
                
                # Sort by date ascending (soonest first), then by score descending
                tier1_unique = tier1_unique.sort_values(['Date', 'race_score'], ascending=[True, False]).head(50)
                
                # Format display
                display_df = tier1_unique.copy()
                if 'Date' in display_df.columns:
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                if 'Prize' in display_df.columns:
                    display_df['Prize'] = display_df['Prize'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "")
                if 'race_score' in display_df.columns:
                    display_df['Score'] = display_df['race_score'].round(1)
                    display_df = display_df.drop('race_score', axis=1)
                
                st.info(f"📊 Showing next {len(display_df)} upcoming Tier 1 Focus races (score ≥70) - sorted by date")
                height = get_dataframe_height(display_df, max_height=400)
                st.dataframe(display_df, hide_index=True, height=height)
            else:
                st.warning("No upcoming Tier 1 Focus races found in dataset")
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

            with st.expander("Upcoming Schedule 🗓️", expanded=False):
                show_cols = [c for c in ["Date", "Course", "Time", "Type", "Surface"] if c in fixtures.columns]
                fixtures_display = fixtures[show_cols].head(200).copy()
                if "Date" in fixtures_display.columns:
                    fixtures_display["Date"] = fixtures_display["Date"].dt.strftime("%Y-%m-%d")

                height = get_dataframe_height(fixtures_display, max_height=400)
                st.dataframe(fixtures_display, hide_index=True, height=height)
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏇 Horses", "🏟️ Courses", "👤 Jockeys", "📈 Overall", "🔮 ML Model", "🗃️ Raw Data"])
    
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
            width='stretch'
        )


if __name__ == "__main__":
    main()
