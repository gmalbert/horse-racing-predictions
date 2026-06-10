"""
Jockey Statistics Page

Displays detailed jockey analytics:
- Current season win%
- Course win%
- Jockey-trainer partnership tables
- Form trends
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="Jockey Stats", page_icon="🏇", layout="wide")

def load_race_data():
    """Load historical race data for jockey analysis"""
    data_dir = BASE_DIR / "data" / "processed"
    
    datasets = [
        data_dir / "race_scores_engineered.parquet",
        data_dir / "race_scores_or_context.parquet",
        data_dir / "race_scores_connections_v2.parquet",
        data_dir / "race_scores.parquet",
    ]
    
    for dataset in datasets:
        if dataset.exists():
            try:
                df = pd.read_parquet(dataset)
                st.success(f"✅ Loaded {len(df):,} races from {dataset.name}")
                return df
            except Exception as e:
                st.warning(f"Failed to load {dataset.name}: {e}")
                continue
    
    st.error("No race data available")
    return None


def calculate_jockey_stats(df, days=None):
    """Calculate jockey statistics"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if days:
        cutoff = df['date'].max() - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff]
    
    jockey_stats = df.groupby('jockey').agg({
        'pos_clean': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
    }).reset_index()
    
    jockey_stats.columns = ['jockey', 'runs', 'wins', 'places']
    jockey_stats['win_pct'] = (jockey_stats['wins'] / jockey_stats['runs'] * 100).round(1)
    jockey_stats['place_pct'] = (jockey_stats['places'] / jockey_stats['runs'] * 100).round(1)
    
    return jockey_stats.sort_values('wins', ascending=False)


def calculate_jockey_course_stats(df, min_runs=5):
    """Calculate jockey performance at specific courses"""
    df = df.copy()
    
    jockey_course = df.groupby(['jockey', 'course_clean']).agg({
        'pos_clean': ['count', lambda x: (x == 1).sum()]
    }).reset_index()
    
    jockey_course.columns = ['jockey', 'course', 'runs', 'wins']
    jockey_course = jockey_course[jockey_course['runs'] >= min_runs]
    jockey_course['win_pct'] = (jockey_course['wins'] / jockey_course['runs'] * 100).round(1)
    
    return jockey_course.sort_values('win_pct', ascending=False)


def calculate_jockey_trainer_partnerships(df, min_runs=10):
    """Calculate jockey-trainer partnership stats"""
    df = df.copy()
    
    partnerships = df.groupby(['jockey', 'trainer']).agg({
        'pos_clean': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
    }).reset_index()
    
    partnerships.columns = ['jockey', 'trainer', 'runs', 'wins', 'places']
    partnerships = partnerships[partnerships['runs'] >= min_runs]
    partnerships['win_pct'] = (partnerships['wins'] / partnerships['runs'] * 100).round(1)
    partnerships['place_pct'] = (partnerships['places'] / partnerships['runs'] * 100).round(1)
    
    return partnerships.sort_values('wins', ascending=False)


def analyze_jockey_form_trend(df, jockey_name, window=30):
    """Analyze jockey form trend over time"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    jockey_races = df[df['jockey'] == jockey_name].sort_values('date')
    
    # Rolling win rate
    jockey_races['won'] = (jockey_races['pos_clean'] == 1).astype(int)
    jockey_races['rolling_win_rate'] = jockey_races['won'].rolling(window=window, min_periods=5).mean() * 100
    
    return jockey_races[['date', 'rolling_win_rate', 'won']].dropna()


def main():
    st.title("🏇 Jockey Statistics & Analytics")
    st.markdown("Comprehensive jockey performance analysis")
    
    # Load data
    df = load_race_data()
    if df is None:
        return
    
    # Ensure required columns exist
    if 'jockey' not in df.columns:
        st.error("Jockey column not found in dataset")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_range = st.sidebar.slider(
            "Date Range (Days)", 
            min_value=7, 
            max_value=365, 
            value=90,
            step=7
        )
        cutoff = df['date'].max() - pd.Timedelta(days=date_range)
        df_filtered = df[df['date'] >= cutoff]
    else:
        df_filtered = df
    
    st.sidebar.info(f"Analyzing {len(df_filtered):,} races")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Current Season",
        "🏇 Course Records",
        "🤝 Trainer Partnerships",
        "📈 Form Trends"
    ])
    
    with tab1:
        st.subheader("Current Season Statistics")
        
        season_stats = calculate_jockey_stats(df_filtered)
        
        if not season_stats.empty:
            # Top jockeys
            st.markdown("#### 🏆 Leading Jockeys")
            top_jockeys = season_stats.head(30)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Most winners
                fig = px.bar(
                    top_jockeys.head(15),
                    x='jockey',
                    y='wins',
                    title='Most Winners',
                    labels={'wins': 'Wins', 'jockey': 'Jockey'}
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Best strike rate (min 20 runs)
                high_sr = season_stats[season_stats['runs'] >= 20].head(15)
                fig = px.bar(
                    high_sr,
                    x='jockey',
                    y='win_pct',
                    title='Best Strike Rate (20+ runs)',
                    labels={'win_pct': 'Win %', 'jockey': 'Jockey'}
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Full leaderboard
            st.markdown("#### 📋 Full Jockey Leaderboard")
            display_df = season_stats.copy()
            display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
            display_df['place_pct'] = display_df['place_pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, hide_index=True, width="content")
        else:
            st.info("No season data available")
    
    with tab2:
        st.subheader("Course Specialists")
        st.caption("Jockeys with strong course records (min 5 runs)")
        
        course_stats = calculate_jockey_course_stats(df_filtered, min_runs=5)
        
        if not course_stats.empty:
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                courses = ['All'] + sorted(course_stats['course'].unique().tolist())
                selected_course = st.selectbox("Filter by Course", courses)
            
            with col2:
                jockeys = ['All'] + sorted(course_stats['jockey'].unique().tolist())
                selected_jockey = st.selectbox("Filter by Jockey", jockeys)
            
            # Apply filters
            filtered_stats = course_stats.copy()
            if selected_course != 'All':
                filtered_stats = filtered_stats[filtered_stats['course'] == selected_course]
            if selected_jockey != 'All':
                filtered_stats = filtered_stats[filtered_stats['jockey'] == selected_jockey]
            
            # Display
            display_df = filtered_stats.copy()
            display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, hide_index=True, width="content")
            
            # Highlight top specialists
            st.markdown("#### 🌟 Top Course Specialists (30%+ win rate)")
            top_specialists = filtered_stats[filtered_stats['win_pct'] >= 30].head(20)
            if not top_specialists.empty:
                display_df = top_specialists.copy()
                display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, hide_index=True, width="content")
            else:
                st.info("No jockeys meet the threshold")
        else:
            st.info("No course data available")
    
    with tab3:
        st.subheader("Jockey-Trainer Partnerships")
        st.caption("Successful combinations (min 10 runs)")
        
        partnerships = calculate_jockey_trainer_partnerships(df_filtered, min_runs=10)
        
        if not partnerships.empty:
            # Top partnerships
            st.markdown("#### 🤝 Most Successful Partnerships")
            top_partnerships = partnerships.head(40)
            
            # Display
            display_df = top_partnerships.copy()
            display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
            display_df['place_pct'] = display_df['place_pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, hide_index=True, width="content")
            
            # Elite partnerships
            st.markdown("#### 🌟 Elite Partnerships (30%+ win rate, 20+ runs)")
            elite = partnerships[(partnerships['win_pct'] >= 30) & (partnerships['runs'] >= 20)]
            if not elite.empty:
                display_df = elite.copy()
                display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
                display_df['place_pct'] = display_df['place_pct'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, hide_index=True, width="content")
            else:
                st.info("No partnerships meet the elite threshold")
        else:
            st.info("No partnership data available")
    
    with tab4:
        st.subheader("Jockey Form Trends")
        st.caption("Rolling 30-race win rate over time")
        
        # Jockey selector
        jockeys = sorted([j for j in df_filtered['jockey'].unique() if pd.notna(j)])
        selected_jockey = st.selectbox("Select Jockey", jockeys, key="trend_jockey")
        
        if selected_jockey:
            form_trend = analyze_jockey_form_trend(df_filtered, selected_jockey, window=30)
            
            if not form_trend.empty and len(form_trend) > 5:
                # Line chart
                fig = px.line(
                    form_trend,
                    x='date',
                    y='rolling_win_rate',
                    title=f'{selected_jockey} - 30-Race Rolling Win Rate',
                    labels={'rolling_win_rate': 'Win Rate (%)', 'date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent form summary
                recent = form_trend.tail(30)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    recent_wr = recent['won'].mean() * 100
                    st.metric("Last 30 Races Win %", f"{recent_wr:.1f}%")
                
                with col2:
                    recent_wins = int(recent['won'].sum())
                    st.metric("Last 30 Races Wins", recent_wins)
                
                with col3:
                    trend_change = recent['rolling_win_rate'].iloc[-1] - recent['rolling_win_rate'].iloc[0]
                    st.metric("Trend", f"{trend_change:+.1f}%", delta_color="normal")
            else:
                st.info("Insufficient data for form trend analysis")
    
    # Add footer
    from footer import add_betting_oracle_footer
    add_betting_oracle_footer()


if __name__ == "__main__":
    main()
