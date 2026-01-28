"""
Shared Utilities for Horse Racing Predictions

Common functions used across multiple pages of the Streamlit app.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import pickle
import os
from datetime import datetime
import threading
import time
import gc
import csv
import tracemalloc
import psutil

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Constants
BASE_DIR = Path(__file__).parent.parent
PARQUET_FILE = BASE_DIR / "data" / "processed" / "race_scores.parquet"
CSV_FILE = BASE_DIR / "data" / "processed" / "all_gb_races.csv"
LOGO_FILE = BASE_DIR / "data" / "logo.png"
MODEL_FILE = BASE_DIR / "models" / "horse_win_predictor.json"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "models" / "feature_importance.csv"
METADATA_FILE = BASE_DIR / "models" / "model_metadata.pkl"
SCORED_FIXTURES_FILE = BASE_DIR / "data" / "processed" / "scored_fixtures_calendar.csv"


def get_now_local(tz_name: str | None = None) -> datetime:
    """Return a timezone-aware 'now' datetime for the given IANA tz name.

    If tz_name is None, the function will consult the `APP_TIMEZONE`
    environment variable, then Streamlit `secrets` (if present). If ZoneInfo is
    invalid, falls back to the system local timezone.
    """
    if tz_name is None:
        # Prefer explicit environment variable
        tz_name = os.environ.get('APP_TIMEZONE')
        # Fall back to Streamlit secrets
        if not tz_name:
            try:
                tz_name = st.secrets.get('APP_TIMEZONE') if hasattr(st, 'secrets') else None
            except Exception:
                tz_name = None

    if tz_name and ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = datetime.now().astimezone().tzinfo
    else:
        tz = datetime.now().astimezone().tzinfo

    return datetime.now(tz)


@st.cache_data
def load_model():
    """Load trained ML model and metadata"""
    if not MODEL_FILE.exists():
        return None, None, None
    
    try:
        # Load model
        if HAS_XGBOOST:
            model = XGBClassifier()
            model.load_model(str(MODEL_FILE))
        else:
            st.error("XGBoost not available for model loading")
            return None, None, None
        
        metadata = None
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
        
        feature_importance = None
        if FEATURE_IMPORTANCE_FILE.exists():
            feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
        
        return model, metadata, feature_importance
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


@st.cache_data
def load_data(use_pyarrow: bool = False, start_date: str | None = '2021-01-01', columns: list | None = None):
    """Load and cache the UK horse races dataset.

    By default this loads the full Parquet/CSV and applies the repo's
    memory optimizations (same behavior as before). Set `use_pyarrow=True`
    to use `pyarrow.dataset` to read a subset of `columns` and filter by
    `start_date` which can significantly reduce peak memory during load.
    """
    df = None

    # Try pyarrow.dataset path when requested and parquet exists
    if use_pyarrow and PARQUET_FILE.exists():
        try:
            import pyarrow.dataset as ds
            import pyarrow as pa
            # default small column set used by the UI
            if columns is None:
                columns = ['date', 'course', 'race_name', 'horse', 'pos', 'jockey', 'trainer', 'rpr', 'or', 'prize', 'class', 'dist', 'going', 'time']

            dataset = ds.dataset(str(PARQUET_FILE), format='parquet')
            expr = None
            if start_date:
                try:
                    dt = pd.to_datetime(start_date)
                    expr = ds.field('date') >= pa.scalar(int(dt.timestamp() * 1e9), type=pa.timestamp('ns'))
                except Exception:
                    expr = None

            if expr is not None:
                table = dataset.to_table(columns=columns, filter=expr)
            else:
                table = dataset.to_table(columns=columns)

            df = table.to_pandas()
        except Exception:
            df = None

    # Fallback: full read via pandas
    if df is None:
        if PARQUET_FILE.exists():
            df = pd.read_parquet(PARQUET_FILE)
        elif CSV_FILE.exists():
            df = pd.read_csv(CSV_FILE)
        else:
            raise FileNotFoundError(f"Dataset not found: {PARQUET_FILE} or {CSV_FILE}")

    # Ensure date column exists and parsed
    if 'date' in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')

    # Filter to recent data to reduce memory usage (default start_date)
    if start_date and 'date' in df.columns:
        try:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        except Exception:
            pass

    # Optimize memory usage
    low_card_cols = ['course', 'class', 'type', 'pattern', 'age_band', 'rating_band', 'course_detail', 'region', 'going', 'sex']
    for col in low_card_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Rename columns to be more readable
    column_rename = {
        "date": "Date",
        "course": "Course",
        "race_name": "Race Name",
        "horse": "Horse",
        "pos": "Finish Position",
        "jockey": "Jockey",
        "trainer": "Trainer",
        "dist": "Distance",
        "going": "Going",
        "time": "Time",
        "age": "Age",
        "sex": "Sex",
        "draw": "Draw",
        "or": "Official Rating",
        "rpr": "RPR",
        "sire": "Sire",
        "dam": "Dam",
        "owner": "Owner",
        "prize": "Prize",
        "class": "Class",
        "type": "Type",
        "off": "Off Time"
    }
    df = df.rename(columns=column_rename)

    return df


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height


def safe_st_call(func, *args, **kwargs):
    """Call a Streamlit display function but strip string `width` values for
    older Streamlit versions that expect an integer `width`.

    Usage: `safe_st_call(st.dataframe, df, hide_index=True, height=100, width='stretch')`
    """
    if 'width' in kwargs and isinstance(kwargs['width'], str):
        kwargs.pop('width')
    return func(*args, **kwargs)


def start_memory_profiling(path: str | os.PathLike = None, interval: int = 5):
    """Start a background thread that records memory usage periodically.

    Writes CSV rows: timestamp (UTC), rss_mb, vms_mb, tracemalloc_current_kb, tracemalloc_peak_kb, gc_counts
    Controlled by environment variable `APP_MEM_PROFILING=1` or called directly from the app.
    """
    try:
        if path is None:
            path = BASE_DIR / 'tmp' / 'memory_snapshots.csv'
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        stop_event = threading.Event()

        def _loop():
            proc = psutil.Process(os.getpid())
            # Start tracemalloc to capture Python-level memory
            try:
                tracemalloc.start()
            except Exception:
                pass

            header_written = path.exists() and path.stat().st_size > 0
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not header_written:
                    writer.writerow(['timestamp_utc', 'rss_mb', 'vms_mb', 'tracemalloc_current_kb', 'tracemalloc_peak_kb', 'gc_counts'])
                    f.flush()

                while not stop_event.is_set():
                    try:
                        mem = proc.memory_info()
                        rss = mem.rss / (1024 * 1024)
                        vms = mem.vms / (1024 * 1024)
                        try:
                            current, peak = tracemalloc.get_traced_memory()
                            current_kb = current / 1024
                            peak_kb = peak / 1024
                        except Exception:
                            current_kb = peak_kb = 0

                        from datetime import timezone
                        writer.writerow([datetime.now(timezone.utc).isoformat(), f"{rss:.2f}", f"{vms:.2f}", f"{current_kb:.1f}", f"{peak_kb:.1f}", repr(gc.get_count())])
                        f.flush()
                    except Exception:
                        # Keep loop alive even if a single measurement fails
                        pass
                    time.sleep(interval)

        t = threading.Thread(target=_loop, daemon=True, name='mem-profiler')
        t.start()

        return stop_event
    except Exception:
        return None
