import json
import pathlib
from types import SimpleNamespace

import pytest


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
