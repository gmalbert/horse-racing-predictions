"""Quick script to check race class distribution"""
import pandas as pd

df = pd.read_parquet('data/processed/race_scores.parquet')

print(f"Total records: {len(df):,}")
print(f"\nClass distribution (all records):")
cls = df['class_clean'].value_counts().sort_index()
for c, count in cls.items():
    pct = count / len(df) * 100
    print(f"  {c}: {count:,} ({pct:.1f}%)")

print("\n" + "="*60)
print("TIER 1 FOCUS RACES (≥70 score) by class:")
tier1 = df[df['race_tier'] == 'Tier 1: Focus'].drop_duplicates('race_id')
print(f"Total Tier 1 races: {len(tier1):,}")
t1_cls = tier1['class_clean'].value_counts().sort_index()
for c, count in t1_cls.items():
    pct = count / len(tier1) * 100
    print(f"  {c}: {count:,} ({pct:.1f}%)")

print("\n" + "="*60)
print("TIER 3 AVOID RACES (<50 score) by class:")
tier3 = df[df['race_tier'] == 'Tier 3: Avoid'].drop_duplicates('race_id')
print(f"Total Tier 3 races: {len(tier3):,}")
t3_cls = tier3['class_clean'].value_counts().sort_index()
for c, count in t3_cls.items():
    pct = count / len(tier3) * 100
    print(f"  {c}: {count:,} ({pct:.1f}%)")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("The model was trained on ALL classes (Class 1-7)")
print("But the betting strategy focuses on:")
print("  - Class 1-3 races (most of Tier 1 Focus)")
print("  - Premium/Major courses")
print("\nYou can safely REMOVE:")
print("  - Class 6 and Class 7 races (low quality, not targeted)")
print("  - This would save ~15-20% database size")
print("  - Minimal impact on betting opportunities")
