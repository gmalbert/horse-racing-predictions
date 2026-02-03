#!/usr/bin/env python3
"""scripts/integrate_betfair_bsp.py - Add Betfair SP to predictions."""

import pandas as pd
from pathlib import Path
import argparse

try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    print("WARNING: fuzzywuzzy not installed. Install with: pip install fuzzywuzzy python-Levenshtein")
    FUZZY_AVAILABLE = False

def load_betfair_data(date_str):
    """Load Betfair BSP data for a given date."""
    betfair_dir = Path('data/raw/betfair')
    file_path = betfair_dir / f'bsp_{date_str}.csv'
    
    if not file_path.exists():
        return None
    
    return pd.read_csv(file_path)

def match_horse_names(prediction_name, betfair_names, threshold=85):
    """Fuzzy match horse names between sources."""
    if not FUZZY_AVAILABLE:
        # Fall back to exact match
        if prediction_name in betfair_names:
            return prediction_name
        return None
    
    match, score = process.extractOne(prediction_name, betfair_names, scorer=fuzz.ratio)
    if score >= threshold:
        return match
    return None

def add_bsp_to_predictions(predictions_df, date_str):
    """Add BSP data to predictions."""
    bsp_data = load_betfair_data(date_str)
    
    if bsp_data is None:
        print(f"No BSP data for {date_str}")
        predictions_df['bsp'] = None
        predictions_df['market_prob'] = None
        predictions_df['model_vs_market'] = None
        return predictions_df
    
    print(f"Loaded BSP data for {len(bsp_data)} horses")
    
    # Match and merge
    predictions_df['betfair_horse'] = predictions_df['horse'].apply(
        lambda x: match_horse_names(x, bsp_data['horse'].tolist())
    )
    
    # Count matches
    matched = predictions_df['betfair_horse'].notna().sum()
    total = len(predictions_df)
    print(f"Matched {matched}/{total} horses ({matched/total*100:.1f}%)")
    
    predictions_df = predictions_df.merge(
        bsp_data[['horse', 'bsp']],
        left_on='betfair_horse',
        right_on='horse',
        how='left',
        suffixes=('', '_bf')
    )
    
    # Calculate market probability
    predictions_df['market_prob'] = 1 / predictions_df['bsp']
    predictions_df['model_vs_market'] = predictions_df['win_probability'] - predictions_df['market_prob']
    
    # Add value bet flag (model prob > market prob by 5%+)
    predictions_df['is_value_bet'] = predictions_df['model_vs_market'] > 0.05
    
    return predictions_df

def main():
    """Main function to integrate Betfair BSP into existing predictions."""
    parser = argparse.ArgumentParser(description='Integrate Betfair BSP data into predictions')
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--predictions', help='Path to predictions CSV (default: data/processed/predictions_DATE.csv)')
    parser.add_argument('--output', help='Output path (default: overwrite input)')
    
    args = parser.parse_args()
    
    # Determine paths
    if args.predictions:
        pred_path = Path(args.predictions)
    else:
        pred_path = Path(f'data/processed/predictions_{args.date}.csv')
    
    output_path = Path(args.output) if args.output else pred_path
    
    print("="*60)
    print("BETFAIR BSP INTEGRATION")
    print("="*60)
    print(f"\nDate: {args.date}")
    print(f"Predictions: {pred_path}")
    print(f"Output: {output_path}")
    
    # Load predictions
    print(f"\nLoading predictions from {pred_path}...")
    if not pred_path.exists():
        print(f"ERROR: Predictions file not found: {pred_path}")
        return 1
    
    predictions = pd.read_csv(pred_path)
    print(f"Loaded {len(predictions)} predictions")
    
    # Add BSP data
    print(f"\nIntegrating Betfair BSP data...")
    predictions_with_bsp = add_bsp_to_predictions(predictions, args.date)
    
    # Save
    print(f"\nSaving to {output_path}...")
    predictions_with_bsp.to_csv(output_path, index=False)
    print("✓ Saved")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if predictions_with_bsp['bsp'].notna().any():
        print(f"\nBSP Statistics:")
        print(f"  Min BSP: {predictions_with_bsp['bsp'].min():.2f}")
        print(f"  Max BSP: {predictions_with_bsp['bsp'].max():.2f}")
        print(f"  Mean BSP: {predictions_with_bsp['bsp'].mean():.2f}")
        
        if 'is_value_bet' in predictions_with_bsp.columns:
            value_bets = predictions_with_bsp['is_value_bet'].sum()
            print(f"\nValue Bets Identified: {value_bets}")
            
            if value_bets > 0:
                print("\nTop Value Bets:")
                value_df = predictions_with_bsp[predictions_with_bsp['is_value_bet']].nlargest(
                    10, 'model_vs_market'
                )[['horse', 'course', 'race_time', 'win_probability', 'market_prob', 'model_vs_market', 'bsp']]
                print(value_df.to_string(index=False))
    else:
        print("\nNo BSP data was integrated (missing Betfair data file)")
    
    print("\n" + "="*60)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
