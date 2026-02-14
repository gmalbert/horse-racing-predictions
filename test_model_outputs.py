"""
Test script to check if the calibrated model produces varied outputs for different inputs
"""
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load models the same way predict_todays_races.py does
print("Loading model...")
try:
    with open('models/horse_win_predictor_calibrated.pkl', 'rb') as f:
        win_model = pickle.load(f)
    print("Successfully loaded calibrated model")
    print(f"Model type: {type(win_model)}")
except Exception as e:
    print(f"Error loading calibrated model: {e}")
    print("Trying base model instead...")
    with open('models/horse_win_predictor.pkl', 'rb') as f:
        win_model = pickle.load(f)
    print("Loaded base (non-calibrated) model")

# Load feature columns
with open('models/feature_columns.txt', 'r') as f:
    feature_cols = [line.strip() for line in f]

print(f"\nModel expects {len(feature_cols)} features\n")

# Create test feature vectors with varying key features
test_cases = []

# Base feature vector (all zeros)
base_features = {col: 0 for col in feature_cols}

# Test case 1: Low OR horse
test1 = base_features.copy()
test1['or_numeric'] = 80
test1['career_win_rate'] = 0.10
test1['jockey_career_win_rate'] = 0.10
test_cases.append(("Low OR (80)", test1))

# Test case 2: High OR horse
test2 = base_features.copy()
test2['or_numeric'] = 150
test2['career_win_rate'] = 0.25
test2['jockey_career_win_rate'] = 0.18
test_cases.append(("High OR (150)", test2))

# Test case 3: Medium OR horse
test3 = base_features.copy()
test3['or_numeric'] = 120
test3['career_win_rate'] = 0.18
test3['jockey_career_win_rate'] = 0.14
test_cases.append(("Medium OR (120)", test3))

# Test case 4: Very low OR horse
test4 = base_features.copy()
test4['or_numeric'] = 50
test4['career_win_rate'] = 0.05
test4['jockey_career_win_rate'] = 0.08
test_cases.append(("Very Low OR (50)", test4))

# Make predictions
print("Testing model outputs:\n")
all_preds = []
for name, features_dict in test_cases:
    # Create DataFrame with correct column order
    features_df = pd.DataFrame([features_dict])[feature_cols]
    
    # Predict
    pred_prob = win_model.predict_proba(features_df)[0, 1]  # Probability of class 1 (win)
    all_preds.append(pred_prob)
    
    print(f"{name:20s} → Win probability: {pred_prob:.6f} ({pred_prob*100:.2f}%)")

print(f"\nNumber of unique predictions: {len(set(all_preds))}")
print(f"Predictions are {'IDENTICAL' if len(set(all_preds)) == 1 else 'VARIED'}")
