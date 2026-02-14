"""Test the exact UI logic from predictions.py"""
import pandas as pd

# Load predictions like the UI does
predictions = []

try:
    df1 = pd.read_csv('data/processed/predictions_2026-02-14.csv')
    df1['day_label'] = 'Today'
    predictions.append(df1)
    print(f"✅ Loaded Today: {len(df1)} horses")
except Exception as e:
    print(f"❌ Failed to load today: {e}")

try:
    df2 = pd.read_csv('data/processed/predictions_2026-02-15.csv')
    df2['day_label'] = 'Tomorrow'
    predictions.append(df2)
    print(f"✅ Loaded Tomorrow: {len(df2)} horses")
except Exception as e:
    print(f"⚠️  No tomorrow predictions: {e}")

if not predictions:
    print("❌ No predictions loaded!")
    exit(1)

predictions = pd.concat(predictions, ignore_index=True)
print(f"\n✅ Combined: {len(predictions)} total horses")

# Check for required odds columns
required_cols = ['win_odds_fractional', 'place_odds_fractional', 'show_odds_fractional']
missing = [col for col in required_cols if col not in predictions.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    exit(1)
else:
    print(f"✅ All odds columns present: {required_cols}")

# Test the exact groupby from display_race_by_race
print("\nTesting display_race_by_race logic...")
races = predictions.groupby(['date', 'day_label', 'race_time', 'course', 'race_name'], observed=False, dropna=False).size().reset_index()[['date', 'day_label', 'race_time', 'course', 'race_name']]

print(f"✅ Found {len(races)} races")

if len(races) == 0:
    print("❌ ERROR: No races found - UI will crash!")
    exit(1)

# Fill NaN race names
races['race_name'] = races['race_name'].fillna('')

# Test race option formatting
race_options = [f"{row['day_label']} ({row['date']}) - {row['race_time']} - {row['course']}" + (f" - {row['race_name'][:40]}" if row['race_name'] else "") for _, row in races.iterrows()]

print(f"\nSample race options:")
for i, opt in enumerate(race_options[:5]):
    print(f"  [{i}] {opt}")

# Test selectbox simulation
selected_race_idx = 0  # Simulate selecting first race
if selected_race_idx is None or not isinstance(selected_race_idx, int) or selected_race_idx >= len(races):
    selected_race_idx = 0

selected_race_info = races.iloc[selected_race_idx]
print(f"\n✅ Selected race {selected_race_idx}:")
print(f"  Date: {selected_race_info['date']}")
print(f"  Time: {selected_race_info['race_time']}")
print(f"  Course: {selected_race_info['course']}")

# Test filtering predictions for selected race
race_preds = predictions[
    (predictions['date'] == selected_race_info['date']) &
    (predictions['race_time'] == selected_race_info['race_time']) &
    (predictions['course'] == selected_race_info['course'])
].copy()

print(f"  Horses: {len(race_preds)}")

if len(race_preds) == 0:
    print("❌ ERROR: No predictions for selected race!")
    exit(1)

# Test display_all_horses_table columns
display_cols = ['horse', 'jockey', 'win_probability', 'win_odds_fractional', 'place_probability', 'place_odds_fractional', 'show_probability', 'show_odds_fractional', 'age', 'weight_lbs', 'ofr', 'form']
try:
    display_df = race_preds[display_cols].copy()
    print(f"\n✅ Display table columns work ({len(display_cols)} columns)")
    print(f"\nSample display data:")
    print(display_df.head(3))
except KeyError as e:
    print(f"❌ ERROR: Missing columns for display: {e}")
    exit(1)

print("\n✅ All UI logic tests passed!")
