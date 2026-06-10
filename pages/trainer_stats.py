"""
Trainer Statistics Page

Displays detailed trainer analytics:
- Form last 14 days
- Course specialists
- Yard patterns by going
- Trainer-jockey partnerships
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="Trainer Stats", page_icon="🎯", layout="wide")

def load_race_data():
    """Load historical race data for trainer analysis"""
    data_dir = BASE_DIR / "data" / "processed"
    
    # Try to load the most complete dataset
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


def calculate_trainer_form(df, days=14):
    """Calculate recent trainer form"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    cutoff = df['date'].max() - pd.Timedelta(days=days)
    
    recent = df[df['date'] >= cutoff].copy()
    
    trainer_stats = recent.groupby('trainer').agg({
        'pos_clean': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
    }).reset_index()
    
    trainer_stats.columns = ['trainer', 'runs', 'wins', 'places']
    trainer_stats['win_pct'] = (trainer_stats['wins'] / trainer_stats['runs'] * 100).round(1)
    trainer_stats['place_pct'] = (trainer_stats['places'] / trainer_stats['runs'] * 100).round(1)
    trainer_stats['strike_rate'] = trainer_stats['win_pct']
    
    return trainer_stats.sort_values('wins', ascending=False)


def find_course_specialists(df, min_runs=5):
    """Find trainers who are course specialists"""
    df = df.copy()
    
    trainer_course = df.groupby(['trainer', 'course_clean']).agg({
        'pos_clean': ['count', lambda x: (x == 1).sum()]
    }).reset_index()
    
    trainer_course.columns = ['trainer', 'course', 'runs', 'wins']
    trainer_course = trainer_course[trainer_course['runs'] >= min_runs]
    trainer_course['win_pct'] = (trainer_course['wins'] / trainer_course['runs'] * 100).round(1)
    
    return trainer_course.sort_values('win_pct', ascending=False)


def analyze_going_patterns(df, trainer_name):
    """Analyze trainer performance by going"""
    df = df.copy()
    trainer_races = df[df['trainer'] == trainer_name]
    
    going_stats = trainer_races.groupby('going_clean').agg({
        'pos_clean': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
    }).reset_index()
    
    going_stats.columns = ['going', 'runs', 'wins', 'places']
    going_stats['win_pct'] = (going_stats['wins'] / going_stats['runs'] * 100).round(1)
    going_stats['place_pct'] = (going_stats['places'] / going_stats['runs'] * 100).round(1)
    
    return going_stats.sort_values('runs', ascending=False)


def find_trainer_jockey_partnerships(df, min_runs=10):
    """Find successful trainer-jockey partnerships"""
    df = df.copy()
    
    partnerships = df.groupby(['trainer', 'jockey']).agg({
        'pos_clean': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
    }).reset_index()
    
    partnerships.columns = ['trainer', 'jockey', 'runs', 'wins', 'places']
    partnerships = partnerships[partnerships['runs'] >= min_runs]
    partnerships['win_pct'] = (partnerships['wins'] / partnerships['runs'] * 100).round(1)
    partnerships['place_pct'] = (partnerships['places'] / partnerships['runs'] * 100).round(1)
    
    return partnerships.sort_values('wins', ascending=False)


def main():
    st.title("🎯 Trainer Statistics & Analytics")
    st.markdown("Comprehensive trainer performance analysis")
    
    # Load data
    df = load_race_data()
    if df is None:
        return
    
    # Ensure required columns exist
    if 'trainer' not in df.columns:
        st.error("Trainer column not found in dataset")
        return
    
    # Add sidebar filters
    st.sidebar.header("Filters")
    
    # Date range
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
        "📊 Recent Form",
        "🏇 Course Specialists",
        "🌦️ Going Patterns",
        "🤝 Jockey Partnerships"
    ])
    
    with tab1:
        st.subheader("Recent Trainer Form (Last 14 Days)")
        
        trainer_form = calculate_trainer_form(df_filtered, days=14)
        
        if not trainer_form.empty:
            # Top trainers
            st.markdown("#### 🔥 In-Form Trainers")
            top_trainers = trainer_form.head(20)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of winners
                fig = px.bar(
                    top_trainers.head(15),
                    x='trainer',
                    y='wins',
                    title='Most Winners (Last 14 Days)',
                    labels={'wins': 'Wins', 'trainer': 'Trainer'}
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart of win percentage
                high_sr = top_trainers[top_trainers['runs'] >= 5].head(15)
                fig = px.bar(
                    high_sr,
                    x='trainer',
                    y='win_pct',
                    title='Highest Strike Rate (5+ runs)',
                    labels={'win_pct': 'Win %', 'trainer': 'Trainer'}
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Full table
            st.markdown("#### 📋 Full Trainer Form Table")
            display_df = trainer_form.copy()
            display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
            display_df['place_pct'] = display_df['place_pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, hide_index=True, width="content")
        else:
            st.info("No recent form data available")
    
    with tab2:
        st.subheader("Course Specialists")
        st.caption("Trainers with strong records at specific courses (min 5 runs)")
        
        specialists = find_course_specialists(df_filtered, min_runs=5)
        
        if not specialists.empty:
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                courses = ['All'] + sorted(specialists['course'].unique().tolist())
                selected_course = st.selectbox("Filter by Course", courses)
            
            with col2:
                trainers = ['All'] + sorted(specialists['trainer'].unique().tolist())
                selected_trainer = st.selectbox("Filter by Trainer", trainers)
            
            # Apply filters
            filtered_specialists = specialists.copy()
            if selected_course != 'All':
                filtered_specialists = filtered_specialists[filtered_specialists['course'] == selected_course]
            if selected_trainer != 'All':
                filtered_specialists = filtered_specialists[filtered_specialists['trainer'] == selected_trainer]
            
            # Display
            display_df = filtered_specialists.copy()
            display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, hide_index=True, width="content")
            
            # Highlight top specialists
            st.markdown("#### 🌟 Top Course Specialists (25%+ win rate, 5+ runs)")
            top_specialists = filtered_specialists[filtered_specialists['win_pct'] >= 25].head(20)
            if not top_specialists.empty:
                st.dataframe(top_specialists, hide_index=True, width="content")
            else:
                st.info("No trainers meet the threshold")
        else:
            st.info("No course specialist data available")
    
    with tab3:
        st.subheader("Going Patterns by Trainer")
        st.caption("Analyze how trainers perform on different ground conditions")
        
        # Trainer selector
        trainers = sorted([t for t in df_filtered['trainer'].unique() if pd.notna(t)])
        selected_trainer = st.selectbox("Select Trainer", trainers, key="going_trainer")
        
        if selected_trainer:
            going_stats = analyze_going_patterns(df_filtered, selected_trainer)
            
            if not going_stats.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Win rate by going
                    fig = px.bar(
                        going_stats,
                        x='going',
                        y='win_pct',
                        title=f'{selected_trainer} - Win % by Going',
                        labels={'win_pct': 'Win %', 'going': 'Going'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Runs by going
                    fig = px.pie(
                        going_stats,
                        values='runs',
                        names='going',
                        title=f'{selected_trainer} - Runs by Going'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Full table
                display_df = going_stats.copy()
                display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
                display_df['place_pct'] = display_df['place_pct'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, hide_index=True, width="content")
            else:
                st.info("No going data available for this trainer")
    
    with tab4:
        st.subheader("Trainer-Jockey Partnerships")
        st.caption("Successful combinations (min 10 runs together)")
        
        partnerships = find_trainer_jockey_partnerships(df_filtered, min_runs=10)
        
        if not partnerships.empty:
            # Top partnerships
            st.markdown("#### 🤝 Most Successful Partnerships")
            top_partnerships = partnerships.head(30)
            
            # Display table
            display_df = top_partnerships.copy()
            display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
            display_df['place_pct'] = display_df['place_pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, hide_index=True, width="content")
            
            # Highlight elite partnerships
            st.markdown("#### 🌟 Elite Partnerships (25%+ win rate, 20+ runs)")
            elite = partnerships[(partnerships['win_pct'] >= 25) & (partnerships['runs'] >= 20)]
            if not elite.empty:
                display_df = elite.copy()
                display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.1f}%")
                display_df['place_pct'] = display_df['place_pct'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, hide_index=True, width="content")
            else:
                st.info("No partnerships meet the elite threshold")
        else:
            st.info("No partnership data available")
    
    # Add footer
    from footer import add_betting_oracle_footer
    add_betting_oracle_footer()


if __name__ == "__main__":
    main()
