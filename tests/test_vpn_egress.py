"""VPN routing regression tests; no Proton credentials or external requests."""

import os
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from scripts import run_with_vpn, vpn_egress
from scripts.fetch_racecards_browser import _launch_context
from scripts.scrape_rp_odds import _launch_browser


@pytest.mark.parametrize("url", [
    "http://remote.example:8888", "http://127.0.0.1", "socks5://127.0.0.1:8888",
    "http://user:secret@localhost:8888", "http://localhost:8888/path",
    "http://localhost:8888?direct=true", "http://localhost:99999",
])
def test_rejects_proxy_urls_outside_local_configuration(url):
    with pytest.raises(ValueError):
        vpn_egress.validate_proxy_url(url)


def test_environment_replaces_inherited_proxies_without_mutating_parent(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://old-proxy:80")
    monkeypatch.setenv("no_proxy", "*")
    before = dict(os.environ)
    env = vpn_egress.workflow_environment(vpn_egress.EGRESS_PROXIES["odds"])
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert env[key] == "http://127.0.0.1:18882"
    assert env["no_proxy"] == env["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert env["RACING_POST_PROXY_URL"] == "http://127.0.0.1:18882"
    assert dict(os.environ) == before


def test_ip_probe_explicitly_tunnels_even_with_no_proxy_star(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    response = Mock()
    response.read.return_value = b"8.8.8.8"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    build = Mock(return_value=opener)
    monkeypatch.setattr(vpn_egress, "build_opener", build)
    assert vpn_egress.public_ip("http://127.0.0.1:18881") == "8.8.8.8"
    request = opener.open.call_args.args[0]
    assert request.host == "127.0.0.1:18881"
    assert request._tunnel_host == "api.ipify.org"
    assert request.type == "https"
    assert build.call_args.args[0].proxies == {}


@pytest.mark.parametrize("address", [b"127.0.0.1", b"not an ip"])
def test_ip_probe_rejects_invalid_or_private_addresses(monkeypatch, address):
    response = Mock()
    response.read.return_value = address
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(vpn_egress, "build_opener", lambda *args: opener)
    with pytest.raises(ValueError):
        vpn_egress.public_ip("http://127.0.0.1:18881")


def test_unreachable_proxy_prevents_child_start(monkeypatch):
    monkeypatch.setattr(run_with_vpn, "public_ip", Mock(side_effect=URLError("offline")))
    child = Mock()
    monkeypatch.setattr(run_with_vpn.subprocess, "run", child)
    assert run_with_vpn.main(["--egress", "racecards", "--", "python", "collector.py"]) == 1
    child.assert_not_called()


def test_mismatched_ip_prevents_child_start(monkeypatch):
    monkeypatch.setattr(run_with_vpn, "public_ip", lambda proxy: "8.8.8.8")
    child = Mock()
    monkeypatch.setattr(run_with_vpn.subprocess, "run", child)
    assert run_with_vpn.main(["--egress", "odds", "--expected-ip", "1.1.1.1", "--", "python"]) == 1
    child.assert_not_called()


def test_launcher_passes_arguments_environment_and_exit_code(monkeypatch):
    monkeypatch.setattr(run_with_vpn, "public_ip", lambda proxy: "8.8.8.8")
    child = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(run_with_vpn.subprocess, "run", child)
    command = ["C:/Python With Spaces/python.exe", "collector.py", "--days", "2"]
    assert run_with_vpn.main(["--egress", "odds", "--", *command]) == 7
    assert child.call_args.args[0] == command
    assert child.call_args.kwargs["env"]["RACING_POST_PROXY_URL"] == vpn_egress.EGRESS_PROXIES["odds"]


@pytest.mark.parametrize("addresses,expected", [
    (["8.8.8.8", "1.1.1.1", "9.9.9.9"], 0),
    (["8.8.8.8", "8.8.8.8", "9.9.9.9"], 1),
])
def test_check_all_requires_distinct_ips(monkeypatch, addresses, expected):
    monkeypatch.setattr(run_with_vpn, "public_ip", Mock(side_effect=addresses))
    assert run_with_vpn.main(["--check-all"]) == expected


def test_chromium_fallback_retains_selected_proxy(monkeypatch):
    monkeypatch.setenv("RACING_POST_PROXY_URL", vpn_egress.EGRESS_PROXIES["racecards"])
    browser = Mock()
    playwright = SimpleNamespace(chromium=Mock())
    playwright.chromium.launch.side_effect = [RuntimeError("Chrome unavailable"), browser]
    context, returned_browser = _launch_context(playwright, False, None)
    assert context is browser.new_context.return_value
    assert returned_browser is browser
    for call in playwright.chromium.launch.call_args_list:
        assert call.kwargs["proxy"]["server"] == vpn_egress.EGRESS_PROXIES["racecards"]
    assert "channel" not in playwright.chromium.launch.call_args.kwargs


def test_persistent_profile_uses_selected_proxy(tmp_path):
    playwright = SimpleNamespace(chromium=Mock())
    context, browser = _launch_context(playwright, False, str(tmp_path), vpn_egress.EGRESS_PROXIES["racecards"])
    assert browser is None
    assert context is playwright.chromium.launch_persistent_context.return_value
    assert playwright.chromium.launch_persistent_context.call_args.kwargs["proxy"]["server"] == vpn_egress.EGRESS_PROXIES["racecards"]


def test_odds_browser_fallback_keeps_proxy(monkeypatch):
    monkeypatch.setenv("RACING_POST_PROXY_URL", vpn_egress.EGRESS_PROXIES["odds"])
    playwright = SimpleNamespace(chromium=Mock())
    playwright.chromium.launch.side_effect = [RuntimeError("no Chrome"), Mock()]
    _launch_browser(playwright, False)
    for call in playwright.chromium.launch.call_args_list:
        assert call.kwargs["proxy"]["server"] == vpn_egress.EGRESS_PROXIES["odds"]
        assert "--disable-quic" in call.kwargs["args"]


def test_no_selection_preserves_normal_browser_options(monkeypatch):
    monkeypatch.delenv("RACING_POST_PROXY_URL", raising=False)
    assert vpn_egress.browser_proxy_options() == {}
