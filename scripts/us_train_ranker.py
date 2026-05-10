import xgboost as xgb

def train_ranker(train_df):
    """
    train_df needs a 'race_id' to group entries.
    'target' should be finishing position (1 for 1st, 2 for 2nd, etc.)
    Note: XGBoost Ranker minimizes, so 1 (1st place) is the best.
    """
    # Sort by race_id for grouping
    train_df = train_df.sort_values('race_id')
    groups = train_df.groupby('race_id').size().to_list()
    
    X = train_df.drop(['race_id', 'target', 'horse_name'], axis=1)
    y = train_df['target']
    
    model = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:pairwise",
        lambdarank_pair_method="topk",
        learning_rate=0.05,
        n_estimators=500
    )
    
    model.fit(X, y, group=groups)
    return model

def get_predictions(model, race_entries):
    # Predict scores (lower score usually means better rank in this config)
    scores = model.predict(race_entries.drop(['race_id', 'horse_name'], axis=1))
    
    # Convert scores to probabilities using Softmax
    exp_scores = np.exp(-scores)  # Negate because lower rank is better
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities