"""
Map US race class strings to numeric values for modelling.

Scale: 1.0 (best – Grade I) … 7.0 (lowest – Maiden Claiming)
"""

import re

# ── Direct-lookup table ───────────────────────────────────────────────────────
US_CLASS_MAP: dict[str, float] = {
    # Graded stakes
    'Grade I': 1.0,   'Grade 1': 1.0,
    'Grade II': 1.5,  'Grade 2': 1.5,
    'Grade III': 2.0, 'Grade 3': 2.0,

    # Listed / ungraded stakes
    'Listed Stakes': 2.5, 'Listed': 2.5,
    'Ungraded Stakes': 3.0, 'Stakes': 3.0,

    # Allowance
    'Allowance Optional Claiming': 3.5, 'AOC': 3.5,
    'Allowance': 4.0,

    # Maiden Special Weight
    'Maiden Special Weight': 5.0, 'MSW': 5.0,

    # Maiden Claiming — price handled separately below
    'Maiden Claiming': 6.5,
}

# ── Claiming price → numeric ──────────────────────────────────────────────────
_CLAIMING_TIERS: list[tuple[int, float]] = [
    (75_000, 4.0),
    (50_000, 4.5),
    (25_000, 5.0),
    (10_000, 5.5),
    (2_500,  6.0),
]

_MAIDEN_CLAIMING_TIERS: list[tuple[int, float]] = [
    (50_000, 6.0),
    (25_000, 6.5),
]


def extract_claiming_price(class_string: str) -> int | None:
    """Return dollar claiming price from a class string, or None."""
    if not class_string:
        return None
    m = re.search(r'\$?([\d,]+)', class_string)
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def map_us_class_to_numeric(class_string: str) -> float:
    """
    Convert a US race class string to a numeric score.

    Returns a float where 1.0 = best quality, 7.0 = lowest quality.
    Unknown / missing values return 7.0.
    """
    if not class_string:
        return 7.0

    # 1. Direct lookup (exact match)
    if class_string in US_CLASS_MAP:
        return US_CLASS_MAP[class_string]

    # 2. Case-insensitive partial-match on known keys
    lower = class_string.lower()

    if 'grade i' in lower and 'ii' not in lower and 'iii' not in lower:
        return 1.0
    if 'grade iii' in lower or 'grade 3' in lower:
        return 2.0
    if 'grade ii' in lower or 'grade 2' in lower:
        return 1.5
    if 'listed' in lower:
        return 2.5
    if 'ungraded stakes' in lower or ('stakes' in lower and 'claiming' not in lower and 'maiden' not in lower):
        return 3.0
    if 'allowance optional' in lower or 'aoc' in lower:
        return 3.5
    if 'allowance' in lower and 'claiming' not in lower:
        return 4.0
    if 'maiden special weight' in lower or 'msw' in lower:
        return 5.0

    # 3. Claiming — price-tiered
    if 'claiming' in lower:
        price = extract_claiming_price(class_string)

        if 'maiden' in lower:
            if price is None:
                return 6.5
            for threshold, score in _MAIDEN_CLAIMING_TIERS:
                if price >= threshold:
                    return score
            return 7.0

        # Regular claiming
        if price is None:
            return 6.0
        for threshold, score in _CLAIMING_TIERS:
            if price >= threshold:
                return score
        return 6.5

    return 7.0


def get_class_category(class_numeric: float) -> str:
    """Return a human-readable category for a numeric class score."""
    if class_numeric <= 2.0:
        return 'Graded Stakes'
    elif class_numeric <= 3.0:
        return 'Listed/Ungraded Stakes'
    elif class_numeric <= 4.5:
        return 'Allowance/High Claiming'
    elif class_numeric <= 5.5:
        return 'Mid-Level Claiming/MSW'
    else:
        return 'Low-Level Claiming/Maiden'
