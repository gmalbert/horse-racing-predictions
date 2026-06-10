import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

# Add project root to sys.path so tests can import from scripts/, shared/, etc.
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_fixture(name: str):
    p = pathlib.Path(__file__).parent / "fixtures" / name
    return json.loads(p.read_text())


@pytest.fixture
def sample_race_response():
    """Return a parsed JSON object from tests/fixtures/race_sample.json."""
    return _load_fixture("race_sample.json")


@pytest.fixture
def mock_requests_get(monkeypatch, sample_race_response):
    """Monkeypatch `requests.get` to return the saved JSON response.

    Useful for tests that would otherwise call The Racing API.
    """

    def _fake_get(*args, **kwargs):
        return SimpleNamespace(
            status_code=200,
            json=lambda: sample_race_response,
            text=json.dumps(sample_race_response),
        )

    monkeypatch.setattr("requests.get", _fake_get)
    return _fake_get
