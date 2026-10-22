"""Configuration via environment variables."""
from __future__ import annotations

import os
from pathlib import Path

# Locate the checkout during local development; container paths come from env.
_REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "web" / "backend").is_dir()),
    Path("/"),
)

# Data directory: on Oracle Cloud, point this at the synced data folder.
DATA_DIR = Path(os.getenv("HRP_DATA_DIR", str(_REPO_ROOT / "data" / "processed")))
RAW_DIR = Path(os.getenv("HRP_RAW_DIR", str(_REPO_ROOT / "data" / "raw")))

# Historical dataset for the explorer
EXPLORER_PARQUET = os.getenv("HRP_EXPLORER_PARQUET", "race_scores.parquet")

# CORS: comma-separated origins, e.g. "https://equine-edge.pages.dev"
CORS_ORIGINS = [o.strip() for o in os.getenv("HRP_CORS_ORIGINS", "*").split(",") if o.strip()]

# Explorer defaults
EXPLORER_START_DATE = os.getenv("HRP_EXPLORER_START_DATE", "2021-01-01")
MAX_PAGE_SIZE = int(os.getenv("HRP_MAX_PAGE_SIZE", "200"))

STATE_DIR = Path(os.getenv("HRP_STATE_DIR", str(DATA_DIR)))
STATE_FILE = STATE_DIR / "bankroll_state.json"
BETTING_HISTORY_FILE = STATE_DIR / "betting_history.csv"
