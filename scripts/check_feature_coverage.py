"""Check which features are being populated vs expected"""
import json

# Get expected features from model
with open('models/feature_columns.txt', 'r') as f:
    expected_features = [line.strip() for line in f if line.strip()]

print(f"Model expects {len(expected_features)} features\n")

# Features that predict_todays_races.py creates
# Based on reading the code
created_features = [
    'career_runs', 'career_win_rate', 'career_place_rate', 'career_earnings',
    'cd_runs', 'cd_win_rate', 'class_num', 'class_step',
    'or_numeric', 'or_change', 'or_trend_3', 'avg_last_3_pos', 'wins_last_3',
    'days_since_last', 'field_size', 'is_turf', 'going_numeric', 'race_score',
    'draw', 'draw_pct', 'draw_group_win_rate',
    'weight_lbs', 'weight_vs_avg', 'is_top_weight', 'weight_change',
    'age', 'is_peak_age', 'is_3yo', 'is_veteran', 'age_vs_avg',
    'avg_btn_last_3', 'unlucky_last',
    'has_blinkers', 'has_visor', 'first_time_blinkers', 'gear_changed',
    'is_handicap', 'is_maiden', 'is_pattern', 'prize_log',
    'is_sprint', 'is_mile', 'is_middle', 'is_staying',
    'jockey_career_runs', 'jockey_course_runs', 'jockey_trainer_runs',
    # Pedigree features (6)
    'sire_win_rate_v2', 'sire_place_rate_v2', 'sire_turf_win_rate', 'sire_aw_win_rate',
    'sire_surface_pref', 'sire_avg_win_dist', 'sire_sprint_pct', 'sire_stayer_pct',
    'sire_class_avg', 'dam_offspring_count', 'dam_offspring_win_rate', 'damsire_stamina_score',
    # Going features (8)
    'horse_heavy_win_rate', 'horse_soft_win_rate', 'horse_good_win_rate', 'horse_firm_win_rate',
    'going_match_score', 'going_is_preferred',
    'sire_heavy_win_rate', 'sire_soft_win_rate', 'sire_good_win_rate', 'sire_firm_win_rate',
    'sire_going_match_v2',
    # OR context features (17)
    'race_or_max', 'race_or_mean', 'race_or_min', 'race_or_std',
    'or_vs_race_max', 'or_vs_race_avg', 'or_vs_race_min', 'is_highest_rated',
    'or_vs_class_typical', 'is_well_handicapped', 'or_percentile',
    'or_career_high', 'or_at_career_high', 'or_below_career_high',
    'or_improving_3', 'or_volatility', 'or_race_percentile',
    # Pace features (9)
    'pace_style_leader', 'pace_style_presser', 'pace_style_closer', 'pace_style_midpack',
    'race_leader_count', 'race_closer_count', 'style_advantage',
    'sprint_specialist', 'staying_specialist',
    # Recent form features (10)
    'jockey_form_14d', 'jockey_form_30d', 'jockey_in_form', 'jockey_course_form_30d',
    'trainer_form_14d', 'trainer_form_30d', 'trainer_in_form', 'trainer_course_form_30d',
    'jockey_trainer_form_30d', 'connections_in_form',
    # Enhanced form features (19)
    'weighted_pos_avg', 'pos_pct_last_3', 'form_consistency', 'form_trend', 'form_at_class', 'runs_at_class',
    'jockey_form_14d_v2', 'jockey_form_30d_v2', 'jockey_hot_v2',
    'trainer_form_14d_v2', 'trainer_form_30d_v2', 'trainer_hot_v2',
    'combo_form_30d_v2', 'combo_hot_v2',
]

created_features_set = set(created_features)
expected_features_set = set(expected_features)

missing = expected_features_set - created_features_set
extra = created_features_set - expected_features_set

print(f"Features being created: {len(created_features)}")
print(f"Features MISSING from script (will default to 0): {len(missing)}")
print(f"\nMissing features:")
for feat in sorted(missing):
    print(f"  - {feat}")

if extra:
    print(f"\nExtra features in script but NOT in model:")
    for feat in sorted(extra):
        print(f"  + {feat}")

common = expected_features_set & created_features_set
print(f"\nCommon features: {len(common)}")
