#!/usr/bin/env python3
"""Display feature impact analytics summary"""
import json
from pathlib import Path

data = json.load(open('models/feature_impact_analysis.json'))

print('='*70)
print('FEATURE IMPACT ANALYTICS')
print('='*70)

print(f'\nBaseline AUC: {data["baseline_auc"]:.4f}')
print(f'Final AUC: {data["final_auc"]:.4f}')
print(f'Total Improvement: +{data["total_improvement"]:.4f} ({data["relative_improvement_pct"]:.2f}%)')

print('\nModel Progression:')
for m in data['models']:
    print(f'  {m["model_name"]:<30} Features: {m["n_features"]:<3} Test AUC: {m["test_auc"]:.4f}')

print('\n' + '='*70)
