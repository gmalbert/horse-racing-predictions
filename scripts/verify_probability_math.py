"""Verify cumulative probabilities are mathematically valid"""
import pandas as pd

df = pd.read_csv('data/processed/predictions_2026-02-14.csv')

print("Testing cumulative probability validity...\n")

# Test sample horses
sample = df.head(10)

all_valid = True
for _, row in sample.iterrows():
    win = row['win_probability']
    place = row['place_probability']
    show = row['show_probability']
    
    # Check: win <= place <= show
    valid_order = win <= place <= show
    
    # Check: all <= 1.0
    valid_range = win <= 1.0 and place <= 1.0 and show <= 1.0
    
    # Check: increments are positive
    place_incr = place - win
    show_incr = show - place
    valid_increments = place_incr >= 0 and show_incr >= 0
    
    status = "✅" if (valid_order and valid_range and valid_increments) else "❌"
    
    if not (valid_order and valid_range and valid_increments):
        all_valid = False
        
    print(f"{status} {row['horse'][:25]:25s} | Win: {win:.1%} | Top2: {place:.1%} (+{place_incr:.1%}) | Top3: {show:.1%} (+{show_incr:.1%})")

print("\n" + "="*80)

if all_valid:
    print("✅ ALL PROBABILITIES ARE MATHEMATICALLY VALID")
else:
    print("❌ SOME PROBABILITIES ARE INVALID")

# Test the original problem case
sha_tin = df[df['course'] == 'Sha Tin'].head(14)
if len(sha_tin) > 0:
    print("\n" + "="*80)
    print("SPECIFIC TEST: Sha Tin 04:45 race")
    print("="*80)
    
    for _, row in sha_tin.iterrows():
        win = row['win_probability']
        place = row['place_probability']
        show = row['show_probability']
        place_incr = place - win
        show_incr = show - place
        
        print(f"{row['horse'][:25]:25s} | Win: {win:6.1%} | Top2: {place:6.1%} (+{place_incr:5.1%}) | Top3: {show:6.1%} (+{show_incr:5.1%})")

print("\n✅ Analysis complete!")
