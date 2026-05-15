import pandas as pd
import numpy as np

def calculate_speed_figures(df):
    """
    Normalizes race times by calculating the track variant.
    df requires: ['track_id', 'date', 'distance', 'surface', 'time']
    """
    # 1. Calculate 'Par Time' (mean time for that distance/surface/track)
    df['par_time'] = df.groupby(['track_id', 'distance', 'surface'])['time'].transform('mean')
    
    # 2. Daily Variant: How much faster/slower the track was today vs Par
    df['daily_variant'] = df.groupby(['track_id', 'date', 'surface'])['time'].transform(
        lambda x: x.mean() - df.loc[x.index, 'par_time'].iloc[0]
    )
    
    # 3. Adjusted Speed Figure
    # Lower is better (seconds), so we subtract the variant from the raw time
    df['adj_speed_fig'] = df['time'] - df['daily_variant']
    
    return df

def normalize_probabilities(probs):
    """
    Ensures win probabilities for a single race sum to 1.0.
    """
    return probs / np.sum(probs)