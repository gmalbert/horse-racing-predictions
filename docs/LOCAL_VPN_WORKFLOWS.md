# Local workflows through separate Proton VPN exits

The local collectors can select one of three independent WireGuard tunnels.
Each tunnel runs in Gluetun inside Docker and exposes an HTTP CONNECT
proxy on loopback. Windows keeps its current network route and Proton desktop
connection. Both Playwright collectors explicitly use the selected proxy;
Requests-based scripts inherit proxy environment variables from the launcher.

| Egress name | Proxy | Proton configuration |
| --- | --- | --- |
| `racecards` | `http://127.0.0.1:18881` | `local/vpn/racecards.conf` |
| `odds` | `http://127.0.0.1:18882` | `local/vpn/odds.conf` |
| `auxiliary` | `http://127.0.0.1:18883` | `local/vpn/auxiliary.conf` |

## One-time setup

1. Use Docker inside WSL (as with `start.ps1`) or Docker Desktop with Linux
   containers. Docker must be running when scheduled collectors run. WSL's
   localhost forwarding must allow Windows to reach the published proxy ports.
   The Windows Proton app by itself
   does not provide these independent per-workflow proxies.
2. In your Proton account, download three WireGuard configurations for
   **different servers**, subject to your subscription's connection limit.
   Follow [Proton's configuration guide](https://protonvpn.com/support/wireguard-configurations).
3. Create `local/vpn` in this repository and save the downloaded configurations
   using the names in the table. These contain private keys: this directory is
   ignored by Git. Do not paste keys into Compose or commit configuration files.
   Keep the exported keys, addresses, and peer endpoints intact. Separate Docker
   network namespaces allow the exported client addresses to overlap.
4. From the repository root, run:

```powershell
.\start_vpn.ps1
.\.venv\Scripts\python.exe scripts/run_with_vpn.py --check-all
```

`--check-all` requires three reachable, distinct public IPs. Different server
names can still share an exit IP; select another Proton server if this check
fails. `start_vpn.ps1` uses local Docker if available, otherwise Docker in WSL;
use `-Distribution Ubuntu` to select a particular WSL distribution. From a
Docker-enabled terminal, the equivalent is
`docker compose -f compose.vpn.yaml up -d --wait`.
To start just one tunnel, append its service name to that Compose command
and check it using `--egress racecards` (or the relevant name).

## Run the workflows

Racecards (including existing publication options):

```powershell
.\run_racing_post_browser.ps1 -Days 2 -VpnEgress racecards
.\run_racing_post_browser.ps1 -Days 2 -VpnEgress racecards -Push -TriggerPredictions
```

For an existing Task Scheduler action, append `-VpnEgress racecards` to the
arguments that invoke `run_racing_post_browser.ps1`. Continue using the logged-in
Windows user for visible Chrome. Publication and GitHub dispatch remain outside
the VPN child process.

Odds through a different exit:

```powershell
.\.venv\Scripts\python.exe scripts/run_with_vpn.py --egress odds -- .\.venv\Scripts\python.exe scripts/scrape_rp_odds.py --date 2026-10-25
```

Requests-based racecard collector through the third exit:

```powershell
.\.venv\Scripts\python.exe scripts/run_with_vpn.py --egress auxiliary -- .\.venv\Scripts\python.exe scripts/fetch_racecards.py --days 2
```

The generic launcher accepts an executable and its arguments after `--`; use
absolute executable/script paths in scheduled tasks and set the working directory
to this repository. It propagates the child's exit code. Clients that ignore
proxy environment variables need explicit proxy integration before using this
launcher. It does not provide OS-level isolation for arbitrary executables.

The browser collectors also accept `--proxy-url http://127.0.0.1:18881` directly,
or `RACING_POST_PROXY_URL`, but the launcher additionally checks connectivity
before starting. Without a proxy selection, existing behavior is unchanged.
Use a separate Chrome profile directory for concurrently running collectors.

## Checks and failure behavior

```powershell
# Show one exit without starting a collector:
.\.venv\Scripts\python.exe scripts/run_with_vpn.py --egress odds
# Optionally require a previously verified exit address:
.\.venv\Scripts\python.exe scripts/run_with_vpn.py --egress odds --expected-ip YOUR_PUBLIC_IP -- .\.venv\Scripts\python.exe scripts/scrape_rp_odds.py
docker compose -f compose.vpn.yaml ps
docker compose -f compose.vpn.yaml logs --tail 80 odds
docker compose -f compose.vpn.yaml down
```

From Windows without a Docker command, use `start_vpn.ps1 -Action Status`,
`-Action Logs`, or `-Action Down` for the corresponding management commands.

The launcher probes `https://api.ipify.org` through the selected proxy and refuses
to start the command if the check fails or an expected IP does not match. A
reachable proxy is not proof of Proton ownership: use only the provided Compose
services on these ports. Gluetun's firewall remains enabled to block non-VPN
egress from the tunnel container. A failed proxy does not trigger a direct browser
fallback; the bundled Chromium fallback retains the proxy. Credentials are mounted
read-only, and proxy ports bind only to `127.0.0.1`. Other local users/processes
can access these unauthenticated proxies.

VPN exits may still be throttled or blocked. Keep request pacing and backoff in
place. This setup assigns a fixed exit per workflow; it does not automatically
rotate exits on throttling. Re-run the distinct-IP check after changing configs or
restarting tunnels. Existing GitHub-hosted workflows are unaffected.

Implementation references: [Gluetun WireGuard configuration mounts](https://github.com/qdm12/gluetun-wiki/blob/main/setup/options/wireguard.md),
[HTTP proxy settings](https://github.com/qdm12/gluetun-wiki/blob/main/setup/options/http-proxy.md),
and [custom provider](https://github.com/qdm12/gluetun-wiki/blob/main/setup/providers/custom.md).
