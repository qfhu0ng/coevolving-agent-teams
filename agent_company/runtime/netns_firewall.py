"""Install a relay-only firewall inside an already-created task network namespace.

This file also runs standalone in the privileged-by-one-capability helper image.
It never runs in the agent process or changes the host network namespace.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import subprocess
from collections.abc import Callable


NETWORK_POLICY = "relay-only-v1"


def validate_endpoint(relay_ip: str | None, relay_port: int | None) -> tuple[str | None, int | None]:
    if (relay_ip is None) != (relay_port is None):
        raise ValueError("Relay IP and port must be supplied together")
    if relay_ip is None:
        return None, None
    address = ipaddress.ip_address(relay_ip)
    if address.is_unspecified or address.is_multicast or getattr(address, "scope_id", None):
        raise ValueError("Relay must have one unicast IP without a scope identifier")
    # Docker's embedded resolver may translate destination ports before filtering.
    if address == ipaddress.ip_address("127.0.0.11"):
        raise ValueError("The Docker embedded DNS address is not a relay endpoint")
    if isinstance(relay_port, bool) or not isinstance(relay_port, int) or not 1 <= relay_port <= 65535:
        raise ValueError("Relay port must be an integer between 1 and 65535")
    return str(address), relay_port


def boundary_facts(relay_ip: str | None, relay_port: int | None) -> dict:
    relay_ip, relay_port = validate_endpoint(relay_ip, relay_port)
    return {"policy": NETWORK_POLICY, "enforced": True,
            "relay_ip": relay_ip, "relay_port": relay_port,
            "external_network_access": False, "runtime_dependency_downloads": False,
            "loopback_addresses": ["127.0.0.1", "::1"], "ipv4": True, "ipv6": True}


def firewall_commands(relay_ip: str | None = None, relay_port: int | None = None) -> list[list[str]]:
    """Both pre-NAT and post-NAT output must match the same narrow allowlist.

    There is deliberately no broad ESTABLISHED rule: a preexisting external
    connection must not survive installation of this boundary. Only reply traffic
    from this task's relay is accepted. The task cannot create arbitrary DNAT rules.
    """
    relay_ip, relay_port = validate_endpoint(relay_ip, relay_port)
    relay_family = ipaddress.ip_address(relay_ip).version if relay_ip else None
    commands: list[list[str]] = []

    def command(binary: str, table: str, *args: str) -> None:
        commands.append([binary, "-w", "5", "-t", table, *args])

    # Drop before flushing or adding any allow rules. A partial failure is fatal;
    # the caller must not start agents if the helper has not completed successfully.
    for binary in ("iptables", "ip6tables"):
        for chain in ("OUTPUT", "INPUT", "FORWARD"):
            command(binary, "filter", "-P", chain, "DROP")
        command(binary, "raw", "-P", "OUTPUT", "DROP")
    for binary in ("iptables", "ip6tables"):
        for chain in ("OUTPUT", "INPUT", "FORWARD"):
            command(binary, "filter", "-F", chain)
        command(binary, "raw", "-F", "OUTPUT")

    for family, binary, loopback in ((4, "iptables", "127.0.0.1/32"),
                                     (6, "ip6tables", "::1/128")):
        for table in ("raw", "filter"):
            if family == 4:
                # All protocols/ports: Docker redirects port 53 to a random port.
                command(binary, table, "-A", "OUTPUT", "-d", "127.0.0.11/32", "-j", "DROP")
            command(binary, table, "-A", "OUTPUT", "-o", "lo", "-d", loopback, "-j", "ACCEPT")
            if relay_family == family:
                command(binary, table, "-A", "OUTPUT", "-d", relay_ip,
                        "-p", "tcp", "--dport", str(relay_port), "-j", "ACCEPT")
        command(binary, "filter", "-A", "INPUT", "-i", "lo", "-s", loopback,
                "-d", loopback, "-j", "ACCEPT")
        if relay_family == family:
            command(binary, "filter", "-A", "INPUT", "-s", relay_ip, "-p", "tcp",
                    "--sport", str(relay_port), "-m", "conntrack", "--ctstate", "ESTABLISHED",
                    "-j", "ACCEPT")
    return commands


def install_firewall(relay_ip: str | None = None, relay_port: int | None = None,
                     *, run: Callable = subprocess.run) -> dict:
    for command in firewall_commands(relay_ip, relay_port):
        run(command, text=True, capture_output=True, check=True, timeout=10)
    return boundary_facts(relay_ip, relay_port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-ip")
    parser.add_argument("--relay-port", type=int)
    arguments = parser.parse_args()
    print(json.dumps(install_firewall(arguments.relay_ip, arguments.relay_port)))


if __name__ == "__main__":
    main()
