"""Collect dated H entries from verified track/provider sources."""
from us_track_sources import H_TRACKS as TRACKS
from us_track_collector import run_cli


if __name__ == "__main__":
    run_cli("h", TRACKS)
