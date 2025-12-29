#!/usr/bin/env python3
"""
Convert probabilities to betting odds formats.

Provides utilities to convert model probabilities (0.0-1.0) into:
- Decimal odds (European format): 2.50
- Fractional odds (UK format): 3/2
- American odds (Moneyline): +150 or -200

Used to compare model predictions against bookmaker odds for value betting.
"""


def probability_to_decimal_odds(probability):
    """
    Convert probability to decimal odds.
    
    Args:
        probability: Float between 0 and 1 (e.g., 0.25 for 25%)
    
    Returns:
        Float: Decimal odds (e.g., 4.0)
    
    Example:
        25% probability -> 4.0 decimal odds
        50% probability -> 2.0 decimal odds
    """
    if probability <= 0:
        return float('inf')
    if probability >= 1:
        return 1.0
    
    return round(1.0 / probability, 2)


def probability_to_fractional_odds(probability):
    """
    Convert probability to fractional odds (UK format).
    
    Args:
        probability: Float between 0 and 1
    
    Returns:
        String: Fractional odds (e.g., "3/1", "5/2")
    
    Example:
        25% probability -> "3/1"
        33.33% probability -> "2/1"
    """
    if probability <= 0:
        return "999/1"
    if probability >= 1:
        return "1/100"
    
    decimal = 1.0 / probability
    
    # Convert decimal to fraction
    # decimal odds of 4.0 = 3/1 (you win 3 for every 1 staked)
    numerator = decimal - 1
    
    # Try to find a simple fraction (denominator max 2 for very simple odds)
    from fractions import Fraction
    frac = Fraction(float(numerator)).limit_denominator(2)
    
    return f"{frac.numerator}/{frac.denominator}"


def probability_to_american_odds(probability):
    """
    Convert probability to American odds (Moneyline).
    
    Args:
        probability: Float between 0 and 1
    
    Returns:
        String: American odds (e.g., "+300", "-200")
    
    Example:
        25% probability -> "+300"
        66.67% probability -> "-200"
    """
    if probability <= 0:
        return "+99900"
    if probability >= 1:
        return "-10000"
    
    if probability >= 0.5:
        # Favorite (negative odds)
        american = -(probability / (1 - probability)) * 100
        return f"{int(american)}"
    else:
        # Underdog (positive odds)
        american = ((1 - probability) / probability) * 100
        return f"+{int(american)}"


def decimal_odds_to_probability(decimal_odds):
    """
    Convert decimal odds to implied probability.
    
    Args:
        decimal_odds: Float (e.g., 4.0)
    
    Returns:
        Float: Probability between 0 and 1
    
    Example:
        4.0 decimal odds -> 0.25 (25%)
        2.0 decimal odds -> 0.50 (50%)
    """
    if decimal_odds <= 1:
        return 1.0
    
    return 1.0 / decimal_odds


def fractional_odds_to_probability(fractional_odds):
    """
    Convert fractional odds to implied probability.
    
    Args:
        fractional_odds: String (e.g., "3/1", "5/2")
    
    Returns:
        Float: Probability between 0 and 1
    
    Example:
        "3/1" -> 0.25 (25%)
        "5/2" -> 0.2857 (28.57%)
    """
    try:
        parts = fractional_odds.split('/')
        if len(parts) != 2:
            return 0.0
        
        numerator = float(parts[0])
        denominator = float(parts[1])
        
        if denominator == 0:
            return 0.0
        
        return denominator / (numerator + denominator)
    except:
        return 0.0


def calculate_value_bet(model_probability, bookmaker_odds, odds_format='decimal'):
    """
    Calculate if there's a value bet opportunity.
    
    Args:
        model_probability: Your model's probability (0-1)
        bookmaker_odds: Bookmaker's odds
        odds_format: 'decimal', 'fractional', or 'american'
    
    Returns:
        Dict with value bet analysis:
        {
            'is_value': bool,
            'edge': float (percentage),
            'model_odds': str,
            'bookmaker_probability': float
        }
    
    Example:
        Model: 25% (4.0 odds)
        Bookmaker: 6.0 odds (16.67%)
        -> Value bet with 8.33% edge!
    """
    # Convert bookmaker odds to probability
    if odds_format == 'decimal':
        bookie_prob = decimal_odds_to_probability(bookmaker_odds)
    elif odds_format == 'fractional':
        bookie_prob = fractional_odds_to_probability(bookmaker_odds)
    else:
        # For American odds, convert via decimal first
        if bookmaker_odds >= 0:
            decimal = (bookmaker_odds / 100) + 1
        else:
            decimal = (100 / abs(bookmaker_odds)) + 1
        bookie_prob = decimal_odds_to_probability(decimal)
    
    # Calculate edge (model thinks it's more likely than bookmaker)
    edge = model_probability - bookie_prob
    
    # Convert model probability to bookmaker's odds format
    if odds_format == 'decimal':
        model_odds = f"{probability_to_decimal_odds(model_probability)}"
    elif odds_format == 'fractional':
        model_odds = probability_to_fractional_odds(model_probability)
    else:
        model_odds = probability_to_american_odds(model_probability)
    
    return {
        'is_value': edge > 0,
        'edge_percent': round(edge * 100, 2),
        'model_odds': model_odds,
        'bookmaker_probability': round(bookie_prob * 100, 2)
    }


if __name__ == '__main__':
    # Example usage
    print("Probability to Odds Conversion Examples")
    print("=" * 50)
    
    probabilities = [0.10, 0.25, 0.33, 0.50, 0.67, 0.75]
    
    for prob in probabilities:
        decimal = probability_to_decimal_odds(prob)
        fractional = probability_to_fractional_odds(prob)
        american = probability_to_american_odds(prob)
        
        print(f"\n{prob*100:.1f}% probability:")
        print(f"  Decimal:    {decimal}")
        print(f"  Fractional: {fractional}")
        print(f"  American:   {american}")
    
    # Value bet example
    print("\n" + "=" * 50)
    print("Value Bet Example")
    print("=" * 50)
    
    model_prob = 0.25  # Model says 25% chance to win
    bookie_odds = 6.0   # Bookmaker offers 6.0 decimal odds (16.67%)
    
    analysis = calculate_value_bet(model_prob, bookie_odds, 'decimal')
    
    print(f"\nModel probability: {model_prob*100}%")
    print(f"Model implied odds: {analysis['model_odds']}")
    print(f"Bookmaker odds: {bookie_odds}")
    print(f"Bookmaker probability: {analysis['bookmaker_probability']}%")
    print(f"Edge: {analysis['edge_percent']}%")
    print(f"Value bet: {'YES! 🎯' if analysis['is_value'] else 'No'}")
