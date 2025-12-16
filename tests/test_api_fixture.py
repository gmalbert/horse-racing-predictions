import requests


def test_mock_requests_get(mock_requests_get):
    """Simple test demonstrating the mocked `requests.get` returning saved JSON."""
    resp = requests.get("https://api.theracingapi.com/example")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("race_id") == 123
