"""Collect dated QH entries from verified track/provider sources."""
from us_track_sources import QH_TRACKS as TRACKS
from us_track_collector import run_cli


if __name__ == "__main__":
    run_cli("qh", TRACKS)
