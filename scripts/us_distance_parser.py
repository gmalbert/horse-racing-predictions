"""
Parse US racing distance formats to furlongs (float).

US distances come in various formats:
    "6f"          → 6.0
    "1m"          → 8.0
    "1m 1f"       → 9.0
    "9f 110y"     → 9.5
    "1 1/8 miles" → 9.0
    "1.125 miles" → 9.0
"""

import re


def parse_us_distance(distance_str: str) -> float | None:
    """Convert a US distance string to furlongs. Returns None if unparseable."""
    if not distance_str or not isinstance(distance_str, str):
        return None

    s = distance_str.strip().lower()

    # 1: simple furlongs  "6f" / "6.5f"
    m = re.match(r'^(\d+(?:\.\d+)?)f$', s)
    if m:
        return float(m.group(1))

    # 2: miles only  "1m" / "2m"
    m = re.match(r'^(\d+)m$', s)
    if m:
        return int(m.group(1)) * 8.0

    # 3: miles + furlongs  "1m 1f"
    m = re.match(r'^(\d+)m\s+(\d+)f$', s)
    if m:
        return int(m.group(1)) * 8.0 + int(m.group(2))

    # 4: furlongs + yards  "9f 110y"
    m = re.match(r'^(\d+)f\s+(\d+)y$', s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 220.0

    # 5: fractional miles  "1 1/8 miles"
    m = re.match(r'^(\d+)\s+(\d+)/(\d+)\s+miles?$', s)
    if m:
        total = int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        return total * 8.0

    # 6: compact fractional miles  "1 1/8m"
    m = re.match(r'^(\d+)\s+(\d+)/(\d+)m$', s)
    if m:
        total = int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        return total * 8.0

    # 7: pure fractional miles  "7/8 miles"
    m = re.match(r'^(\d+)/(\d+)\s+miles?$', s)
    if m:
        return (int(m.group(1)) / int(m.group(2))) * 8.0

    # 8: decimal miles  "1.125 miles"
    m = re.match(r'^(\d+\.\d+)\s+miles?$', s)
    if m:
        return float(m.group(1)) * 8.0

    return None


def get_distance_band_us(furlongs: float | None) -> str:
    """Classify a US distance (in furlongs) into a named band."""
    if furlongs is None:
        return 'Unknown'
    if furlongs < 6.5:
        return 'Sprint'
    elif furlongs < 7.5:
        return 'One-Turn Mile'
    elif furlongs < 9.0:
        return 'Classic'
    elif furlongs < 10.5:
        return 'Route'
    else:
        return 'Long'
