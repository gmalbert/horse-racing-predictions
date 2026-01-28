"""Demo: load recent rows using pyarrow.dataset to reduce memory footprint.

Usage: import this module and call `load_recent_pyarrow(start_date, columns)`
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_FILE = BASE_DIR / "data" / "processed" / "race_scores.parquet"


def load_recent_pyarrow(start_date: str = '2021-01-01', columns: list | None = None):
    """Load rows with date >= start_date using pyarrow.dataset scanning.

    Returns a pandas.DataFrame. Only loads the requested columns.
    """
    try:
        import pyarrow.dataset as ds
        import pyarrow as pa
    except Exception as e:
        raise RuntimeError("pyarrow is required for this demo: pip install pyarrow") from e

    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_FILE}")

    # Normalize columns list
    if columns is None:
        # default to a small subset used by the UI
        columns = ['date', 'course', 'horse', 'pos', 'jockey', 'trainer', 'rpr', 'or']

    # Build dataset and filter expression
    dataset = ds.dataset(str(PARQUET_FILE), format='parquet')

    # Parse start_date to pyarrow scalar
    try:
        dt = pd.to_datetime(start_date)
        # pyarrow uses timestamps with ns
        scalar = pa.scalar(int(dt.timestamp() * 1e9), type=pa.timestamp('ns'))
        expr = ds.field('date') >= scalar
    except Exception:
        # Fallback: no filter
        expr = None

    # Scan dataset with filter to avoid loading older partitions
    if expr is not None:
        table = dataset.to_table(columns=columns, filter=expr)
    else:
        table = dataset.to_table(columns=columns)

    # Convert to pandas (this will allocate the resulting DataFrame)
    df = table.to_pandas()

    # Ensure date parsed
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


if __name__ == '__main__':
    # simple CLI for quick manual test
    import sys
    sd = sys.argv[1] if len(sys.argv) > 1 else '2021-01-01'
    df = load_recent_pyarrow(sd)
    print(f"Loaded {len(df)} rows from {PARQUET_FILE} since {sd}")
