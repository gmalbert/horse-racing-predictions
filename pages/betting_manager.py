"""
Betting Manager — Kelly Calculator + P&L Tracker

Interactive bankroll management tools:
- Kelly criterion stake calculator
- Bet logging and P&L tracking
- Rolling ROI analysis
- Drawdown alerts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, date, timedelta
import sys
import json
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "processed"
STATE_FILE = DATA_DIR / "bankroll_state.json"
BETTING_HISTORY_FILE = DATA_DIR / "betting_history.csv"

st.set_page_config(page_title="Betting Manager", page_icon="💰", layout="wide")


# ─────────────────────────────────────────────────────────────
# Kelly Criterion Helpers
# ─────────────────────────────────────────────────────────────

def kelly_stake_fraction(
    win_prob: float,
    decimal_odds: float,
    frac: float = 0.25,
    min_frac: float = 0.01,
    max_frac: float = 0.10,
) -> float:
    """Compute fractional-Kelly stake as fraction of bankroll."""
    b = decimal_odds - 1.0
    edge = b * win_prob - (1.0 - win_prob)
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    sized = full_kelly * frac
    return float(np.clip(sized, min_frac, max_frac))


def model_odds_from_prob(win_prob: float) -> float:
    """Convert model P(win) to fair decimal odds."""
    if win_prob <= 0:
        return 999.0
    return round(1.0 / win_prob, 2)


# ─────────────────────────────────────────────────────────────
# Bankroll State Management
# ─────────────────────────────────────────────────────────────

def load_bankroll_state():
    """Load current bankroll state from JSON."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    
    # Default state
    return {
        "initial_bankroll": 1000.0,
        "current_bankroll": 1000.0,
        "peak_bankroll": 1000.0,
        "total_bets": 0,
        "total_wins": 0,
        "last_updated": str(date.today())
    }


def save_bankroll_state(state: dict):
    """Save bankroll state to JSON."""
    state["last_updated"] = str(date.today())
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def load_betting_history():
    """Load betting history from CSV."""
    if BETTING_HISTORY_FILE.exists():
        df = pd.read_csv(BETTING_HISTORY_FILE)
        df['bet_date'] = pd.to_datetime(df['bet_date'])
        return df
    
    # Return empty dataframe with schema
    return pd.DataFrame(columns=[
        'bet_date', 'race_id', 'horse_name', 'win_probability',
        'market_odds', 'stake', 'result', 'profit'
    ])


def save_betting_history(df: pd.DataFrame):
    """Save betting history to CSV."""
    BETTING_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BETTING_HISTORY_FILE, index=False)


# ─────────────────────────────────────────────────────────────
# Prediction Loading
# ─────────────────────────────────────────────────────────────

def load_predictions_for_date(target_date: date):
    """Load predictions CSV for specified date."""
    pred_file = DATA_DIR / f"predictions_{target_date}.csv"
    
    if not pred_file.exists():
        return None
    
    df = pd.read_csv(pred_file)
    return df


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────

def main():
    st.title("💰 Betting Manager")
    st.markdown("Kelly criterion calculator, bet logging, and P&L tracking")
    
    # Sidebar: Bankroll Configuration
    st.sidebar.header("⚙️ Bankroll Settings")
    
    state = load_bankroll_state()
    
    # Editable bankroll
    current_bankroll = st.sidebar.number_input(
        "Current Bankroll (£)",
        min_value=0.0,
        value=float(state["current_bankroll"]),
        step=10.0,
        help="Your current betting bankroll"
    )
    
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Fraction of full Kelly to use (0.25 = quarter Kelly, conservative)"
    )
    
    min_edge = st.sidebar.slider(
        "Minimum Edge (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=1.0,
        help="Only recommend bets with at least this edge"
    )
    
    # Update state if bankroll changed
    if current_bankroll != state["current_bankroll"]:
        state["current_bankroll"] = current_bankroll
        if current_bankroll > state["peak_bankroll"]:
            state["peak_bankroll"] = current_bankroll
        save_bankroll_state(state)
    
    # Drawdown calculation
    peak = state["peak_bankroll"]
    drawdown = ((peak - current_bankroll) / peak * 100) if peak > 0 else 0.0
    
    # Display drawdown alert
    if drawdown >= 20:
        st.sidebar.error(f"⚠️ DRAWDOWN ALERT: {drawdown:.1f}% below peak!")
    elif drawdown >= 10:
        st.sidebar.warning(f"⚠️ Drawdown: {drawdown:.1f}% below peak")
    else:
        st.sidebar.success(f"✅ {drawdown:.1f}% below peak")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Kelly Calculator",
        "📝 Log Bet",
        "📊 P&L Tracker",
        "📈 Performance"
    ])
    
    with tab1:
        st.subheader("Kelly Criterion Stake Calculator")
        st.caption("Recommended stakes for today's best bets")
        
        # Date selector
        target_date = st.date_input(
            "Predictions Date",
            value=date.today(),
            help="Select date to load predictions"
        )
        
        predictions = load_predictions_for_date(target_date)
        
        if predictions is None:
            st.info(f"No predictions available for {target_date}. Run predict_todays_races.py first.")
        else:
            # Filter for value bets with minimum edge
            if 'edge' not in predictions.columns:
                # Calculate edge if not present
                predictions['model_odds'] = predictions['win_probability'].apply(
                    lambda p: 1.0 / p if p > 0 else 999.0
                )
                if 'market_odds' in predictions.columns:
                    predictions['edge'] = (
                        (predictions['market_odds'] / predictions['model_odds'] - 1) * 100
                    )
                else:
                    st.warning("Market odds not available — cannot calculate edge")
                    predictions['edge'] = 0.0
            
            # Filter value bets
            value_bets = predictions[predictions['edge'] >= min_edge].copy()
            
            if len(value_bets) == 0:
                st.info(f"No bets meet the {min_edge}% minimum edge threshold today.")
            else:
                # Calculate Kelly stakes
                value_bets['kelly_frac'] = value_bets.apply(
                    lambda row: kelly_stake_fraction(
                        row['win_probability'],
                        row['market_odds'] if 'market_odds' in row else model_odds_from_prob(row['win_probability']),
                        frac=kelly_fraction
                    ),
                    axis=1
                )
                
                value_bets['stake_pounds'] = (value_bets['kelly_frac'] * current_bankroll).round(2)
                value_bets['expected_profit'] = (
                    value_bets['stake_pounds'] * 
                    (value_bets['win_probability'] * (value_bets.get('market_odds', 1) - 1) - (1 - value_bets['win_probability']))
                ).round(2)
                
                # Sort by expected profit
                value_bets = value_bets.sort_values('expected_profit', ascending=False)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Value Bets", len(value_bets))
                
                with col2:
                    total_stake = value_bets['stake_pounds'].sum()
                    st.metric("Total Outlay", f"£{total_stake:.2f}")
                
                with col3:
                    total_exp_profit = value_bets['expected_profit'].sum()
                    st.metric("Expected Profit", f"£{total_exp_profit:.2f}")
                
                with col4:
                    exp_roi = (total_exp_profit / total_stake * 100) if total_stake > 0 else 0
                    st.metric("Expected ROI", f"{exp_roi:.1f}%")
                
                st.markdown("---")
                
                # Display recommendations table
                display_cols = [
                    'horse_name', 'race_time', 'course', 'win_probability',
                    'market_odds', 'edge', 'stake_pounds', 'expected_profit'
                ]
                
                available_cols = [c for c in display_cols if c in value_bets.columns]
                
                st.dataframe(
                    value_bets[available_cols].head(20),
                    hide_index=True,
                    width='stretch'
                )
    
    with tab2:
        st.subheader("Log a Bet")
        st.caption("Record bets and results to track your P&L")
        
        with st.form("log_bet_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                bet_date = st.date_input("Bet Date", value=date.today())
                horse_name = st.text_input("Horse Name", placeholder="e.g. Arkle")
                race_id = st.text_input("Race ID (optional)", placeholder="e.g. 12345")
                win_prob = st.number_input("Model Win Probability", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
            
            with col2:
                market_odds = st.number_input("Market Odds (decimal)", min_value=1.0, value=3.0, step=0.1)
                stake = st.number_input("Stake (£)", min_value=0.0, value=10.0, step=1.0)
                result = st.selectbox("Result", ["Pending", "Won", "Lost"])
            
            submitted = st.form_submit_button("Log Bet")
            
            if submitted:
                # Calculate profit
                if result == "Won":
                    profit = stake * (market_odds - 1)
                elif result == "Lost":
                    profit = -stake
                else:
                    profit = 0.0
                
                # Load history, append, save
                history = load_betting_history()
                
                new_bet = pd.DataFrame([{
                    'bet_date': bet_date,
                    'race_id': race_id or 'N/A',
                    'horse_name': horse_name,
                    'win_probability': win_prob,
                    'market_odds': market_odds,
                    'stake': stake,
                    'result': result,
                    'profit': profit
                }])
                
                history = pd.concat([history, new_bet], ignore_index=True)
                save_betting_history(history)
                
                # Update bankroll state
                if result != "Pending":
                    state = load_bankroll_state()
                    state["current_bankroll"] += profit
                    state["total_bets"] += 1
                    if result == "Won":
                        state["total_wins"] += 1
                    if state["current_bankroll"] > state["peak_bankroll"]:
                        state["peak_bankroll"] = state["current_bankroll"]
                    save_bankroll_state(state)
                
                st.success(f"✅ Bet logged: {horse_name} @ {market_odds} — {result}")
                st.rerun()
    
    with tab3:
        st.subheader("P&L Tracker")
        st.caption("View and manage your betting history")
        
        history = load_betting_history()
        
        if len(history) == 0:
            st.info("No bets logged yet. Use the 'Log Bet' tab to add bets.")
        else:
            # Summary metrics
            total_bets = len(history)
            won_bets = len(history[history['result'] == 'Won'])
            lost_bets = len(history[history['result'] == 'Lost'])
            pending_bets = len(history[history['result'] == 'Pending'])
            
            total_staked = history['stake'].sum()
            total_profit = history[history['result'] != 'Pending']['profit'].sum()
            
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            strike_rate = (won_bets / (won_bets + lost_bets) * 100) if (won_bets + lost_bets) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Bets", total_bets)
            
            with col2:
                st.metric("Win Rate", f"{strike_rate:.1f}%")
            
            with col3:
                st.metric("Total Profit/Loss", f"£{total_profit:.2f}")
            
            with col4:
                st.metric("ROI", f"{roi:.1f}%")
            
            st.markdown("---")
            
            # Full history table
            st.markdown("#### Betting History")
            
            display_history = history.sort_values('bet_date', ascending=False).copy()
            st.dataframe(
                display_history,
                hide_index=True,
                width='stretch'
            )
            
            # Edit/delete options
            with st.expander("⚙️ Manage Bets"):
                st.markdown("**Update Pending Bets**")
                
                pending = history[history['result'] == 'Pending']
                if len(pending) > 0:
                    for idx, row in pending.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"{row['horse_name']} — £{row['stake']:.2f} @ {row['market_odds']}")
                        
                        with col2:
                            if st.button(f"Won", key=f"won_{idx}"):
                                history.loc[idx, 'result'] = 'Won'
                                history.loc[idx, 'profit'] = row['stake'] * (row['market_odds'] - 1)
                                save_betting_history(history)
                                
                                # Update bankroll
                                state = load_bankroll_state()
                                state["current_bankroll"] += history.loc[idx, 'profit']
                                state["total_wins"] += 1
                                if state["current_bankroll"] > state["peak_bankroll"]:
                                    state["peak_bankroll"] = state["current_bankroll"]
                                save_bankroll_state(state)
                                
                                st.rerun()
                        
                        with col3:
                            if st.button(f"Lost", key=f"lost_{idx}"):
                                history.loc[idx, 'result'] = 'Lost'
                                history.loc[idx, 'profit'] = -row['stake']
                                save_betting_history(history)
                                
                                # Update bankroll
                                state = load_bankroll_state()
                                state["current_bankroll"] += history.loc[idx, 'profit']
                                save_bankroll_state(state)
                                
                                st.rerun()
                else:
                    st.info("No pending bets")
    
    with tab4:
        st.subheader("Performance Analysis")
        st.caption("Rolling ROI, drawdown, and performance trends")
        
        history = load_betting_history()
        
        if len(history) == 0:
            st.info("No betting history available yet.")
        else:
            # Settled bets only
            settled = history[history['result'] != 'Pending'].copy()
            
            if len(settled) == 0:
                st.info("No settled bets yet.")
            else:
                # Cumulative P&L
                settled = settled.sort_values('bet_date')
                settled['cumulative_profit'] = settled['profit'].cumsum()
                settled['cumulative_stake'] = settled['stake'].cumsum()
                settled['cumulative_roi'] = (
                    settled['cumulative_profit'] / settled['cumulative_stake'] * 100
                )
                
                # P&L curve
                fig = px.line(
                    settled,
                    x='bet_date',
                    y='cumulative_profit',
                    title='Cumulative Profit/Loss Over Time',
                    labels={'cumulative_profit': 'Profit (£)', 'bet_date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ROI over time
                fig2 = px.line(
                    settled,
                    x='bet_date',
                    y='cumulative_roi',
                    title='Cumulative ROI % Over Time',
                    labels={'cumulative_roi': 'ROI (%)', 'bet_date': 'Date'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Rolling 30-day performance
                if len(settled) >= 10:
                    st.markdown("#### Rolling 30-Day Performance")
                    
                    # Calculate rolling metrics
                    settled['rolling_profit_30d'] = settled['profit'].rolling(window=30, min_periods=5).sum()
                    settled['rolling_stake_30d'] = settled['stake'].rolling(window=30, min_periods=5).sum()
                    settled['rolling_roi_30d'] = (
                        settled['rolling_profit_30d'] / settled['rolling_stake_30d'] * 100
                    )
                    
                    fig3 = px.line(
                        settled,
                        x='bet_date',
                        y='rolling_roi_30d',
                        title='30-Day Rolling ROI %',
                        labels={'rolling_roi_30d': 'ROI (%)', 'bet_date': 'Date'}
                    )
                    st.plotly_chart(fig3, use_container_width=True)
    
    # Add footer
    from footer import add_betting_oracle_footer
    add_betting_oracle_footer()


if __name__ == "__main__":
    main()
