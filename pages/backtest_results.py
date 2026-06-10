"""
Backtest Results Page

Displays historical model performance:
- Accuracy by race class, distance, going, month
- Calibration curves
- ROI analysis
- Walk-forward backtest results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="Backtest Results", page_icon="📊", layout="wide")

def load_backtest_results():
    """Load backtest results from models directory"""
    backtest_file = BASE_DIR / "models" / "backtest_results.json"
    
    if backtest_file.exists():
        try:
            with open(backtest_file, 'r') as f:
                results = json.load(f)
            st.success(f"✅ Loaded backtest results from {backtest_file.name}")
            return results
        except Exception as e:
            st.warning(f"Failed to load backtest results: {e}")
    else:
        st.info("ℹ️  Run `python scripts/backtest_walk_forward.py` to generate backtest results")
    
    return None


def load_historical_predictions():
    """Load historical predictions for analysis"""
    pred_dir = BASE_DIR / "data" / "processed"
    
    # Look for prediction files
    pred_files = list(pred_dir.glob("predictions_*.csv"))
    
    if not pred_files:
        return None
    
    # Load and combine
    all_preds = []
    for file in sorted(pred_files):
        try:
            df = pd.read_csv(file)
            df['date'] = file.stem.split('_')[-1]
            all_preds.append(df)
        except Exception as e:
            continue
    
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        st.success(f"✅ Loaded {len(combined):,} historical predictions from {len(pred_files)} days")
        return combined
    
    return None


def calculate_accuracy_by_segment(df, segment_col):
    """Calculate model accuracy by segment"""
    if 'actual_result' not in df.columns:
        return None
    
    df = df.copy()
    df['predicted_win'] = (df['win_probability'] > 0.5).astype(int)
    df['actual_win'] = (df['actual_result'] == 1).astype(int)
    df['correct'] = (df['predicted_win'] == df['actual_win']).astype(int)
    
    segment_stats = df.groupby(segment_col).agg({
        'correct': 'sum',
        'predicted_win': 'count'
    }).reset_index()
    
    segment_stats.columns = [segment_col, 'correct', 'total']
    segment_stats['accuracy'] = (segment_stats['correct'] / segment_stats['total'] * 100).round(1)
    
    return segment_stats.sort_values('accuracy', ascending=False)


def calculate_roi_by_segment(df, segment_col):
    """Calculate ROI by segment"""
    if 'actual_result' not in df.columns or 'market_odds' not in df.columns:
        return None
    
    df = df.copy()
    df['stake'] = 1.0  # £1 level stakes
    df['return'] = df.apply(
        lambda x: x['market_odds'] if x['actual_result'] == 1 else 0, axis=1
    )
    df['profit'] = df['return'] - df['stake']
    
    segment_roi = df.groupby(segment_col).agg({
        'stake': 'sum',
        'profit': 'sum',
        'return': 'count'
    }).reset_index()
    
    segment_roi.columns = [segment_col, 'total_stake', 'total_profit', 'bets']
    segment_roi['roi'] = (segment_roi['total_profit'] / segment_roi['total_stake'] * 100).round(1)
    
    return segment_roi.sort_values('roi', ascending=False)


def plot_calibration_curve(df, bins=10):
    """Plot model calibration curve"""
    if 'actual_result' not in df.columns:
        return None
    
    df = df.copy()
    df['prob_bin'] = pd.cut(df['win_probability'], bins=bins)
    
    calibration = df.groupby('prob_bin').agg({
        'win_probability': 'mean',
        'actual_result': ['mean', 'count']
    }).reset_index()
    
    calibration.columns = ['bin', 'predicted', 'actual', 'count']
    calibration = calibration[calibration['count'] >= 10]  # Min 10 samples per bin
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='gray')
    ))
    
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=calibration['predicted'],
        y=calibration['actual'],
        mode='markers+lines',
        name='Model',
        marker=dict(size=calibration['count']/10)
    ))
    
    fig.update_layout(
        title='Model Calibration Curve',
        xaxis_title='Predicted Probability',
        yaxis_title='Actual Win Rate',
        width='stretch'
    )
    
    return fig


def main():
    st.title("📊 Model Backtest Results")
    st.markdown("Historical model performance analysis")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Walk-Forward Results",
        "🎯 Accuracy Analysis",
        "💰 ROI Analysis",
        "📉 Calibration"
    ])
    
    with tab1:
        st.subheader("Walk-Forward Backtest Results")
        st.caption("Model performance on unseen future data")
        
        backtest_results = load_backtest_results()
        
        if backtest_results:
            # Display summary metrics
            if 'summary' in backtest_results:
                summary = backtest_results['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Overall AUC", f"{summary.get('auc', 0):.4f}")
                
                with col2:
                    st.metric("Accuracy", f"{summary.get('accuracy', 0):.1%}")
                
                with col3:
                    st.metric("Precision", f"{summary.get('precision', 0):.1%}")
                
                with col4:
                    st.metric("Recall", f"{summary.get('recall', 0):.1%}")
                
                st.markdown("---")
            
            # Fold-by-fold results
            if 'folds' in backtest_results:
                st.markdown("#### Fold-by-Fold Performance")
                
                folds_df = pd.DataFrame(backtest_results['folds'])
                if not folds_df.empty:
                    # Line chart of AUC over folds
                    fig = px.line(
                        folds_df,
                        x='fold',
                        y='auc',
                        title='AUC by Fold',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Full table
                    st.dataframe(folds_df, hide_index=True, width="content")
        else:
            st.info("No backtest results available. Run backtest_walk_forward.py to generate results.")
    
    with tab2:
        st.subheader("Accuracy by Segment")
        
        historical_preds = load_historical_predictions()
        
        if historical_preds is not None and 'actual_result' in historical_preds.columns:
            # Segment selector
            segments = ['race_class', 'distance_band', 'going', 'month']
            available_segments = [s for s in segments if s in historical_preds.columns]
            
            if available_segments:
                selected_segment = st.selectbox("Analyze by:", available_segments)
                
                accuracy_stats = calculate_accuracy_by_segment(historical_preds, selected_segment)
                
                if accuracy_stats is not None:
                    # Bar chart
                    fig = px.bar(
                        accuracy_stats,
                        x=selected_segment,
                        y='accuracy',
                        title=f'Model Accuracy by {selected_segment}',
                        labels={'accuracy': 'Accuracy (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Full table
                    st.dataframe(accuracy_stats, hide_index=True, width="content")
                else:
                    st.info("Unable to calculate accuracy by segment")
            else:
                st.info("Segment columns not found in predictions")
        else:
            st.info("No historical predictions with actual results available")
    
    with tab3:
        st.subheader("ROI Analysis")
        st.caption("Return on Investment by segment (level stakes)")
        
        historical_preds = load_historical_predictions()
        
        if historical_preds is not None and 'actual_result' in historical_preds.columns:
            if 'market_odds' not in historical_preds.columns:
                st.warning("Market odds not available - cannot calculate ROI")
            else:
                # Segment selector
                segments = ['race_class', 'distance_band', 'going', 'month']
                available_segments = [s for s in segments if s in historical_preds.columns]
                
                if available_segments:
                    selected_segment = st.selectbox("Analyze ROI by:", available_segments, key="roi_segment")
                    
                    roi_stats = calculate_roi_by_segment(historical_preds, selected_segment)
                    
                    if roi_stats is not None:
                        # Bar chart
                        fig = px.bar(
                            roi_stats,
                            x=selected_segment,
                            y='roi',
                            title=f'ROI by {selected_segment}',
                            labels={'roi': 'ROI (%)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            total_bets = roi_stats['bets'].sum()
                            st.metric("Total Bets", f"{total_bets:,}")
                        
                        with col2:
                            total_profit = roi_stats['total_profit'].sum()
                            st.metric("Total Profit/Loss", f"£{total_profit:,.2f}")
                        
                        with col3:
                            overall_roi = (roi_stats['total_profit'].sum() / roi_stats['total_stake'].sum() * 100)
                            st.metric("Overall ROI", f"{overall_roi:.1f}%")
                        
                        # Full table
                        st.dataframe(roi_stats, hide_index=True, width="content")
                    else:
                        st.info("Unable to calculate ROI by segment")
                else:
                    st.info("Segment columns not found in predictions")
        else:
            st.info("No historical predictions with actual results available")
    
    with tab4:
        st.subheader("Model Calibration")
        st.caption("Are predicted probabilities well-calibrated?")
        
        historical_preds = load_historical_predictions()
        
        if historical_preds is not None and 'actual_result' in historical_preds.columns:
            calibration_fig = plot_calibration_curve(historical_preds)
            
            if calibration_fig:
                st.plotly_chart(calibration_fig, use_container_width=True)
                
                st.markdown("""
                **Interpretation**:
                - Points on the diagonal = perfect calibration
                - Points above diagonal = model is underconfident
                - Points below diagonal = model is overconfident
                - Point size = number of predictions in that bin
                """)
            else:
                st.info("Unable to generate calibration curve")
        else:
            st.info("No historical predictions with actual results available")
    
    # Add footer
    from footer import add_betting_oracle_footer
    add_betting_oracle_footer()


if __name__ == "__main__":
    main()
