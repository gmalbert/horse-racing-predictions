import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_feature_contract import choose_best_dataset


def test_choose_best_dataset_skips_invalid_parquet_like_files(tmp_path):
    expected = ["a", "b"]

    valid_path = tmp_path / "valid.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(valid_path)

    invalid_path = tmp_path / "invalid.parquet"
    invalid_path.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")

    best = choose_best_dataset(expected, [invalid_path, valid_path])

    assert best["path"] == valid_path
    assert best["missing"] == []
