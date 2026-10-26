#!/usr/bin/env python3
"""Run a local workflow through a named Proton WireGuard proxy."""

import argparse
import subprocess
import sys

try:
    from .vpn_egress import EGRESS_PROXIES, public_ip, workflow_environment
except ImportError:
    from vpn_egress import EGRESS_PROXIES, public_ip, workflow_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egress", choices=EGRESS_PROXIES)
    parser.add_argument("--check-all", action="store_true", help="Require all three exits to be reachable and distinct")
    parser.add_argument("--expected-ip", help="Optional public IP to require before starting a workflow")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after -- (omit to check only)")
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if args.check_all and (args.egress or command or args.expected_ip):
        parser.error("--check-all cannot be combined with --egress, --expected-ip, or a command")
    if not args.check_all and not args.egress:
        parser.error("select --egress or --check-all")
    try:
        if args.check_all:
            exits = {name: public_ip(proxy) for name, proxy in EGRESS_PROXIES.items()}
            for name, address in exits.items():
                print(f"{name}: {address}", flush=True)
            if len(set(exits.values())) != len(exits):
                raise RuntimeError("VPN exits share a public IP; select different Proton servers")
            return 0
        proxy = EGRESS_PROXIES[args.egress]
        address = public_ip(proxy)
        if args.expected_ip and address != args.expected_ip:
            raise RuntimeError("VPN exit differs from --expected-ip; workflow was not started")
        print(f"VPN egress {args.egress}: {address} via {proxy}", flush=True)
        if command:
            return subprocess.run(command, env=workflow_environment(proxy)).returncode
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"VPN workflow failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
