"""
Map US track conditions (going) to a numeric scale.

Scale: 1 (fastest/firmest) → 6 (slowest/most testing)
Works for both Dirt and Turf surfaces.
"""

# Primary lookup — try exact match first
US_GOING_MAP: dict[str, int] = {
    # ── Dirt ──
    'Fast': 1,
    'Good': 2,
    'Muddy': 3,
    'Sloppy': 4,
    'Heavy': 5,
    'Frozen': 6,

    # ── Turf ──
    'Firm': 1,
    'Firm (Turf)': 1,
    'Good (Turf)': 2,
    'Yielding': 3,
    'Soft': 4,
    'Soft (Turf)': 4,
    'Heavy (Turf)': 5,
}

# Keyword fallbacks (lower-case contains)
_GOING_KEYWORDS: list[tuple[str, int]] = [
    ('fast', 1),
    ('firm', 1),
    ('good', 2),
    ('muddy', 3),
    ('yielding', 3),
    ('sloppy', 4),
    ('soft', 4),
    ('heavy', 5),
    ('frozen', 6),
]


def map_us_going_to_numeric(going_str: str | None) -> int:
    """
    Convert a US going/track-condition string to a numeric scale.

    Returns an int in [1, 6]; unknown values return 2 (Good — neutral).
    """
    if not going_str:
        return 2

    # Exact lookup
    if going_str in US_GOING_MAP:
        return US_GOING_MAP[going_str]

    # Case-insensitive keyword scan
    lower = going_str.lower()
    for keyword, value in _GOING_KEYWORDS:
        if keyword in lower:
            return value

    return 2  # Default: Good (neutral)


def get_surface_going_key(surface: str | None, going: str | None) -> str:
    """
    Return a combined surface+going key for interaction features.

    Example: ("Dirt", "Fast") → "Dirt_Fast"
    """
    surface_clean = (surface or 'Unknown').strip().title()
    going_clean = (going or 'Unknown').strip().title()
    return f"{surface_clean}_{going_clean}"
