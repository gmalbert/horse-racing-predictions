"""
Probability Calibration Options for Horse Racing Predictions

The model outputs are overconfident (mean 70%, max 98%). 
Real-world horse racing favorites rarely exceed 50-60% win probability.

OPTION 1: Temperature Scaling (Simple, Fast)
-------------------------------------------
Applies a temperature parameter to soften probabilities:
    calibrated = p^T / (p^T + (1-p)^T)

Where T > 1 softens (pushes toward 0.5), T < 1 sharpens

Example with T=2.5:
    Raw 95% -> Calibrated 75%
    Raw 80% -> Calibrated 50%
    Raw 60% -> Calibrated 30%

Pros: Simple, single parameter, preserves ordering
Cons: May need tuning to get right scale


OPTION 2: Platt Scaling (Sigmoid)
----------------------------------
Fits logistic regression on validation data:
    calibrated = 1 / (1 + exp(-A*logit(p) - B))

Requires historical data with actual outcomes

Pros: Theoretically sound, well-calibrated
Cons: Needs ground truth labels for fitting


OPTION 3: Isotonic Regression
------------------------------
Non-parametric calibration that learns monotonic mapping

Pros: Very flexible, no assumptions
Cons: Needs lots of validation data, can overfit


OPTION 4: Beta Calibration
---------------------------
Uses beta distribution to model calibration:
    calibrated = Beta(a*p, b*(1-p))

Pros: Flexible, handles extremes well
Cons: More complex, needs parameter tuning


OPTION 5: Simple Shrinkage (Quick Fix)
---------------------------------------
Pull probabilities toward field average:
    calibrated = (p + prior*k) / (1 + k)

Where prior = 1/field_size, k = shrinkage strength

Example with prior=1/14=7.1%, k=3:
    Raw 95% -> Calibrated 30%
    Raw 80% -> Calibrated 25%
    Raw 60% -> Calibrated 20%

Pros: Very simple, no fitting needed, realistic
Cons: Crude, loses some information


RECOMMENDED APPROACH FOR IMMEDIATE FIX:
========================================
Use Temperature Scaling (T=2.5-3.0) or Simple Shrinkage (k=2-4)

This will:
✅ Compress extreme probabilities toward realistic range
✅ Preserve relative ordering of horses
✅ No training data needed
✅ Fast to implement and run

Then later: Fit Platt/Isotonic on historical validation set
"""

import numpy as np

def temperature_scaling(prob, temperature=2.5):
    """Apply temperature scaling to calibrate probabilities"""
    if prob <= 0.001:
        return 0.001
    if prob >= 0.999:
        return 0.999
    
    logit = np.log(prob / (1 - prob))
    scaled_logit = logit / temperature
    calibrated = 1 / (1 + np.exp(-scaled_logit))
    return calibrated


def shrinkage_calibration(prob, field_size=14, strength=3):
    """Shrink toward field average (1/field_size)"""
    prior = 1.0 / field_size
    return (prob + prior * strength) / (1 + strength)


# Test examples
print("TEMPERATURE SCALING (T=2.5):")
print("="*60)
for p in [0.12, 0.30, 0.50, 0.70, 0.85, 0.95]:
    calibrated = temperature_scaling(p, temperature=2.5)
    print(f"  {p:.0%} -> {calibrated:.1%}")

print("\n\nSIMPLE SHRINKAGE (k=3, 14-horse field):")
print("="*60)
for p in [0.12, 0.30, 0.50, 0.70, 0.85, 0.95]:
    calibrated = shrinkage_calibration(p, field_size=14, strength=3)
    print(f"  {p:.0%} -> {calibrated:.1%}")

print("\n\nRECOMMENDATION:")
print("="*60)
print("Use Temperature Scaling with T=2.5-3.0")
print("This preserves ordering while making probabilities realistic")
print("\nTypical favorites should be 20-40% (not 80-95%)")
print("Strong favorites might reach 50-60% max")
print("Longshots should be 5-15%")
