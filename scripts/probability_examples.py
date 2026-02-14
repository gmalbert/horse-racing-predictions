"""Example calculations showing how the new cumulative probabilities work"""

print("="*80)
print("CUMULATIVE PROBABILITY EXAMPLES")
print("="*80)

examples = [
    ("Strong Favorite", 0.669),
    ("Medium Contender", 0.40),
    ("Outsider", 0.15),
]

for name, win_prob in examples:
    # Calculate using the same formula as the prediction script
    base_place_boost = win_prob * (1 - win_prob) * 0.8
    base_show_boost = win_prob * (1 - win_prob) * 0.3
    
    place_prob = win_prob + base_place_boost
    show_prob = place_prob + base_show_boost
    
    # Apply constraints
    place_prob = max(win_prob, min(place_prob, 0.98))
    show_prob = max(place_prob, min(show_prob, 0.99))
    
    place_incr = place_prob - win_prob
    show_incr = show_prob - place_prob
    
    print(f"\n{name} (Win prob: {win_prob:.1%})")
    print(f"  🥇 Win (1st)           : {win_prob:.1%}")
    print(f"  🥇🥈 Top 2 (1st or 2nd) : {place_prob:.1%} (+{place_incr:.1%})")
    print(f"  🥇🥈🥉 Top 3 (1st/2nd/3rd): {show_prob:.1%} (+{show_incr:.1%})")
    print(f"  Interpretation: {win_prob:.0%} chance to win, {place_prob:.0%} chance for top 2, {show_prob:.0%} chance for top 3")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("✅ Strong favorites (>90% win) have small additional upside for place/show")
print("✅ Medium contenders (~40% win) have significant upside for place/show")
print("✅ Longshots (<20% win) can still have decent top-3 chances")
print("✅ All probabilities satisfy: Win ≤ Place ≤ Show ≤ 100%")
print("\n")
