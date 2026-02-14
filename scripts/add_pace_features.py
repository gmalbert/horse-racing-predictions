#!/usr/bin/env python3
"""
Add Pace/Running Style Features
Implements CRITICAL_DATA_GAPS.md Section 2: Pace Analysis

Since we don't have in-running comments, we derive pace style from:
1. Historical early vs late position patterns
2. Draw position preferences
3. Finishing position distributions
"""

import pandas as pd
import numpy as np
from pathlib import Path

def classify_pace_style_from_form(horse_df):
    """
    Classify horse's running style based on historical race patterns.
    
    Logic:
    - LEADER: Consistently finishes in top 3 in sprints, or has low draw in sprints
    - CLOSER: Often finishes better than mid-race position would suggest
    - PRESSER: Consistent finishes, mid-pack runner
    - MIDPACK: Default
    
    Returns: 'LEADER', 'PRESSER', 'MIDPACK', 'CLOSER', or 'UNKNOWN'
    """
    if len(horse_df) < 3:
        return 'UNKNOWN'
    
    # Sprint races (< 8f) with low draw and good finishes = likely front-runner
    sprint_races = horse_df[horse_df['dist_f_clean'] < 8]
    if len(sprint_races) >= 3:
        low_draw_wins = sprint_races[
            (sprint_races['draw'] <= 3) & 
            (sprint_races['pos_clean'] <= 3)
        ]
        if len(low_draw_wins) / len(sprint_races) > 0.4:
            return 'LEADER'
    
    # Consistent top-3 finishes in any race = presser
    top3_rate = (horse_df['pos_clean'] <= 3).mean()
    consistency = horse_df['pos_clean'].std()
    
    if top3_rate > 0.40 and consistency < 3.0:
        return 'PRESSER'
    
    # Large variance in finishes = closer (waits for gaps)
    if consistency > 5.0:
        return 'CLOSER'
    
    # Middle ground
    if top3_rate > 0.25:
        return 'MIDPACK'
    
    return 'UNKNOWN'

def add_pace_features(df):
    """
    Add pace and running style features.
    
    Features added:
    - pace_style: Categorical (LEADER/PRESSER/MIDPACK/CLOSER/UNKNOWN)
    - pace_style_leader: Binary flag (1 if LEADER)
    - pace_style_closer: Binary flag (1 if CLOSER)
    - race_leader_count: Number of likely leaders in race
    - race_closer_count: Number of likely closers in race
    - pace_pressure: Ratio of leaders to field size
    - style_advantage: 1 if pace scenario suits this horse
    - sprint_specialist: 1 if horse excels at sprints
    - staying_specialist: 1 if horse excels at staying trips
    """
    print("\n" + "="*60)
    print("ADDING PACE/RUNNING STYLE FEATURES")
    print("="*60)
    
    df = df.copy()
    df = df.sort_values(['date', 'off']).copy()
    
    # Ensure we have necessary columns
    if 'pos_clean' not in df.columns or 'dist_f_clean' not in df.columns:
        raise ValueError("Need pos_clean and dist_f_clean columns")
    
    # === CLASSIFY RUNNING STYLE ===
    print("\n1. Classifying horse running styles from historical patterns...")
    
    pace_styles = []
    
    for horse in df['horse'].unique():
        horse_data = df[df['horse'] == horse].copy()
        
        # For each race, classify based on PRIOR races only
        for idx in horse_data.index:
            race_date = horse_data.loc[idx, 'date']
            prior_races = horse_data[horse_data['date'] < race_date]
            
            style = classify_pace_style_from_form(prior_races)
            pace_styles.append({
                'index': idx,
                'pace_style': style
            })
    
    # Map back to dataframe
    pace_style_df = pd.DataFrame(pace_styles).set_index('index')
    df['pace_style'] = df.index.map(pace_style_df['pace_style'])
    df['pace_style'] = df['pace_style'].fillna('UNKNOWN')
    
    print(f"   Classified {len(df):,} horses")
    print(f"   Distribution:")
    print(df['pace_style'].value_counts())
    
    # === CREATE BINARY FLAGS ===
    print("\n2. Creating pace style binary indicators...")
    
    df['pace_style_leader'] = (df['pace_style'] == 'LEADER').astype(int)
    df['pace_style_presser'] = (df['pace_style'] == 'PRESSER').astype(int)
    df['pace_style_closer'] = (df['pace_style'] == 'CLOSER').astype(int)
    df['pace_style_midpack'] = (df['pace_style'] == 'MIDPACK').astype(int)
    
    # === RACE-LEVEL PACE ANALYSIS ===
    print("\n3. Calculating race-level pace scenarios...")
    
    # Count each pace type per race
    df['race_id_key'] = df['date'].astype(str) + '_' + df['course_clean'] + '_' + df['off']
    
    race_pace = df.groupby('race_id_key').agg({
        'pace_style_leader': 'sum',
        'pace_style_closer': 'sum',
        'ran': 'first'  # field size
    }).reset_index()
    
    race_pace.columns = ['race_id_key', 'race_leader_count', 'race_closer_count', 'field_size']
    
    # Merge back
    df = df.merge(race_pace, on='race_id_key', how='left', suffixes=('', '_race'))
    
    # Pace pressure (many leaders = fast early pace)
    df['pace_pressure'] = df['race_leader_count'] / df['ran'].clip(lower=1)
    
    # === PACE ADVANTAGE ===
    print("\n4. Calculating pace scenario advantages...")
    
    # Fast early pace (many leaders) favors closers
    # Slow early pace (few leaders) favors front-runners
    
    df['style_advantage'] = 0.0
    
    # High pace pressure + closer = advantage
    df.loc[
        (df['pace_pressure'] > 0.30) & (df['pace_style'] == 'CLOSER'),
        'style_advantage'
    ] = 1.0
    
    # Low pace pressure + leader = advantage
    df.loc[
        (df['pace_pressure'] < 0.15) & (df['pace_style'] == 'LEADER'),
        'style_advantage'
    ] = 1.0
    
    # Moderate pace + presser = advantage
    df.loc[
        (df['pace_pressure'].between(0.15, 0.30)) & (df['pace_style'] == 'PRESSER'),
        'style_advantage'
    ] = 0.5
    
    # === DISTANCE SPECIALIZATION ===
    print("\n5. Calculating distance specialization...")
    
    # Sprint specialist (excels at 5-7f)
    sprint_data = df[df['dist_f_clean'] < 8].groupby('horse').agg({
        'pos_clean': lambda x: (x <= 3).mean() if len(x) >= 3 else None
    }).reset_index()
    sprint_data.columns = ['horse', 'sprint_top3_rate']
    
    df = df.merge(sprint_data, on='horse', how='left')
    df['sprint_specialist'] = (df['sprint_top3_rate'] > 0.35).fillna(False).astype(int)
    
    # Staying specialist (excels at 12f+)
    staying_data = df[df['dist_f_clean'] >= 12].groupby('horse').agg({
        'pos_clean': lambda x: (x <= 3).mean() if len(x) >= 3 else None
    }).reset_index()
    staying_data.columns = ['horse', 'staying_top3_rate']
    
    df = df.merge(staying_data, on='horse', how='left')
    df['staying_specialist'] = (df['staying_top3_rate'] > 0.35).fillna(False).astype(int)
    
    # === DRAW/PACE INTERACTION ===
    print("\n6. Creating draw-pace interaction features...")
    
    # Low draw + leader style in sprint = big advantage
    df['low_draw_leader_sprint'] = (
        (df['draw'] <= 3) & 
        (df['pace_style'] == 'LEADER') & 
        (df['dist_f_clean'] < 8)
    ).astype(int)
    
    # High draw + closer style in sprint = disadvantage (needs to overcome)
    df['high_draw_closer_sprint'] = (
        (df['draw'] > df['ran'] * 0.7) &
        (df['pace_style'] == 'CLOSER') &
        (df['dist_f_clean'] < 8)
    ).astype(int)
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("PACE FEATURES SUMMARY")
    print("="*60)
    
    print(f"\nPace style classification:")
    print(df.groupby('pace_style').size())
    
    print(f"\nPace pressure distribution:")
    print(f"  High pressure (>0.3): {(df['pace_pressure'] > 0.3).sum():,} races")
    print(f"  Moderate (0.15-0.3): {df['pace_pressure'].between(0.15, 0.3).sum():,} races")
    print(f"  Low pressure (<0.15): {(df['pace_pressure'] < 0.15).sum():,} races")
    
    print(f"\nStyle advantages:")
    print(f"  Horses with pace advantage: {(df['style_advantage'] > 0).sum():,}")
    print(f"  Sprint specialists: {df['sprint_specialist'].sum():,}")
    print(f"  Staying specialists: {df['staying_specialist'].sum():,}")
    
    print(f"\nNew features added: 15")
    print(f"  - pace_style (categorical)")
    print(f"  - pace_style_leader, pace_style_presser, pace_style_closer, pace_style_midpack")
    print(f"  - race_leader_count, race_closer_count")
    print(f"  - pace_pressure, style_advantage")
    print(f"  - sprint_specialist, staying_specialist")
    print(f"  - low_draw_leader_sprint, high_draw_closer_sprint")
    
    # Clean up temporary columns
    df = df.drop(columns=['race_id_key'], errors='ignore')
    
    return df

if __name__ == '__main__':
    # Load data
    print("Loading race data...")
    
    # Priority order: OR context > pedigree no-leak > features > pedigree > base
    or_context_path = Path('data/processed/race_scores_or_context.parquet')
    no_leak_pedigree = Path('data/processed/race_scores_with_pedigree_no_leakage.parquet')
    features_path = Path('data/processed/race_scores_with_features.parquet')
    pedigree_path = Path('data/processed/race_scores_with_pedigree.parquet')
    base_path = Path('data/processed/race_scores.parquet')
    
    if or_context_path.exists():
        print(f"  Loading from {or_context_path} (with all prior features)")
        df = pd.read_parquet(or_context_path)
    elif no_leak_pedigree.exists():
        print(f"  Loading from {no_leak_pedigree} (no leakage version)")
        df = pd.read_parquet(no_leak_pedigree)
    elif features_path.exists():
        print(f"  Loading from {features_path}")
        df = pd.read_parquet(features_path)
    elif pedigree_path.exists():
        print(f"  Loading from {pedigree_path}")
        df = pd.read_parquet(pedigree_path)
    else:
        print(f"  Loading from {base_path}")
        df = pd.read_parquet(base_path)
    
    print(f"  Loaded {len(df):,} rows")
    
    # Add pace features
    df_with_pace = add_pace_features(df)
    
    # Save to no-leakage path
    output_path = Path('data/processed/race_scores_with_all_features_no_leakage.parquet')
    print(f"\nSaving to {output_path}...")
    df_with_pace.to_parquet(output_path, index=False)
    
    print("\n✓ COMPLETE: Pace features added successfully")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(df_with_pace):,}")
    print(f"  Columns: {len(df_with_pace.columns)}")
