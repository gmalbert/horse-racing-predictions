"""Local VPN proxy configuration shared by workflow and browser entry points."""

import ipaddress
import os
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


EGRESS_PROXIES = {
    "racecards": "http://127.0.0.1:18881",
    "odds": "http://127.0.0.1:18882",
    "auxiliary": "http://127.0.0.1:18883",
}


def validate_proxy_url(url: str) -> str:
    """Accept only the unauthenticated, loopback HTTP proxies used here."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or not parsed.port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("VPN proxy must be an HTTP loopback URL with an explicit port")
    return url.rstrip("/")


def browser_proxy_options(proxy_url=None):
    """Playwright does not inherit HTTP_PROXY; configure both launch paths."""
    proxy_url = proxy_url or os.environ.get("RACING_POST_PROXY_URL")
    if not proxy_url:
        return {}
    return {
        "proxy": {"server": validate_proxy_url(proxy_url), "bypass": "<-loopback>"},
        "args": [
            "--disable-quic",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
        ],
    }


def public_ip(proxy_url: str, timeout: float = 20):
    """Probe through the selected proxy, with no direct-connection fallback."""
    proxy_url = validate_proxy_url(proxy_url)
    # An explicitly proxied request avoids inherited NO_PROXY='*' bypasses.
    request = Request("https://api.ipify.org")
    request.set_proxy(urlsplit(proxy_url).netloc, "http")
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        address = ipaddress.ip_address(response.read(128).decode("ascii").strip())
    if not address.is_global:
        raise ValueError("VPN egress check did not return a public IP address")
    return str(address)


def workflow_environment(proxy_url: str):
    """Build child-only environment; leave the parent shell untouched."""
    proxy_url = validate_proxy_url(proxy_url)
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            del env[key]
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[key] = proxy_url
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    env["RACING_POST_PROXY_URL"] = proxy_url
    return env
