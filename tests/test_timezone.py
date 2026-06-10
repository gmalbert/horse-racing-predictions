import sys
import importlib.util
from pathlib import Path
import pytest
from datetime import datetime


# Add project root to sys.path so predictions.py can import from shared/, scripts/, etc.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load the predictions module by file path so pytest can run from the tests directory
spec = importlib.util.spec_from_file_location("predictions", ROOT / "predictions.py")
predictions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predictions)


def test_get_now_local_with_valid_timezone():
    dt = predictions.get_now_local('UTC')
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None


def test_get_now_local_with_invalid_timezone():
    # Invalid timezone should not raise and should return a timezone-aware datetime
    dt = predictions.get_now_local('Invalid/Timezone')
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None


if __name__ == '__main__':
    pytest.main([__file__])
