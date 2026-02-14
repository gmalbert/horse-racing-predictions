"""Test calibration on actual prediction data"""
import pandas as pd
import numpy as np

def temperature_scaling(prob, temperature=2.5):
    """Apply temperature scaling"""
    if prob <= 0.001:
        return 0.001
    if prob >= 0.999:
        return 0.999
    logit = np.log(prob / (1 - prob))
    scaled_logit = logit / temperature
    return 1 / (1 + np.exp(-scaled_logit))

def shrinkage_calibration(prob, field_size, strength=3):
    """Shrink toward field average"""
    prior = 1.0 / field_size
    return (prob + prior * strength) / (1 + strength)

# Load predictions
df = pd.read_csv('data/processed/predictions_2026-02-14.csv')

# Test different calibration approaches
print("COMPARING CALIBRATION METHODS")
print("="*80)

# Group by race to get field sizes
race_groups = df.groupby(['course', 'race_time'])

results = {}

for name, group in race_groups:
    field_size = len(group)
    course, time = name
    
    # Original (uncalibrated)
    raw_probs = group['win_probability'].values
    
    # Temperature scaling (T=3.5)
    temp_probs = [temperature_scaling(p, 3.5) for p in raw_probs]
    
    # Temperature scaling (T=5.0)
    temp5_probs = [temperature_scaling(p, 5.0) for p in raw_probs]
    
    # Shrinkage (k=3)
    shrink_probs = [shrinkage_calibration(p, field_size, 3) for p in raw_probs]
    
    # Shrinkage (k=5)
    shrink5_probs = [shrinkage_calibration(p, field_size, 5) for p in raw_probs]
    
    if course == 'Sha Tin' and time == '04:45':
        print(f"\nEXAMPLE RACE: {course} {time} ({field_size} horses)")
        print("="*80)
        print(f"{'Horse':<25} {'Raw':>7} {'Temp3.5':>7} {'Temp5.0':>7} {'Shrink3':>7} {'Shrink5':>7}")
        print("-"*80)
        
        for i in range(min(5, len(group))):
            horse = group.iloc[i]['horse']
            print(f"{horse:<25} {raw_probs[i]:>6.1%} {temp_probs[i]:>7.1%} {temp5_probs[i]:>7.1%} {shrink_probs[i]:>7.1%} {shrink5_probs[i]:>7.1%}")

# Overall statistics
print("\n\nOVERALL STATISTICS:")
print("="*80)
print(f"{'Method':<20} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
print("-"*80)

# Calculate for all races
all_raw = df['win_probability'].values
all_temp35 = [temperature_scaling(p, 3.5) for p in all_raw]
all_temp50 = [temperature_scaling(p, 5.0) for p in all_raw]

# For shrinkage, need to apply per-race
all_shrink3 = []
all_shrink5 = []
for _, group in race_groups:
    field_size = len(group)
    for p in group['win_probability']:
        all_shrink3.append(shrinkage_calibration(p, field_size, 3))
        all_shrink5.append(shrinkage_calibration(p, field_size, 5))

methods = {
    'Original (Raw)': all_raw,
    'Temperature T=3.5': all_temp35,
    'Temperature T=5.0': all_temp50,
    'Shrinkage k=3': all_shrink3,
    'Shrinkage k=5': all_shrink5
}

for name, probs in methods.items():
    probs = np.array(probs)
    print(f"{name:<20} {probs.mean():>7.1%} {probs.min():>7.1%} {probs.max():>7.1%} {probs.std():>7.1%}")

print("\n\n✅ RECOMMENDATION:")
print("="*80)
print("Use Temperature Scaling with T=3.5-4.0 OR Shrinkage with k=4")
print("This will give realistic probabilities while preserving relative ordering")
