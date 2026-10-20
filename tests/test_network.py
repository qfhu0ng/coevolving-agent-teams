"""Firewall construction/lifecycle tests; no Docker daemon or provider required."""
import json
import subprocess

import pytest

from agent_company.runtime.netns_firewall import (
    boundary_facts, firewall_commands, install_firewall, validate_endpoint,
)
from agent_company.runtime.network import NetworkBoundary, NETWORK_HELPER_IMAGE


HELPER_ID = "abc123def456" * 5 + "abcd"


@pytest.mark.parametrize("ip,port", [(None, 80), ("10.0.0.1", None), ("example.com", 80),
                                     ("0.0.0.0", 80), ("::", 80), ("224.0.0.1", 80),
                                     ("fe80::1%eth0", 80), ("127.0.0.11", 80),
                                     ("10.0.0.1", 0), ("10.0.0.1", 65536),
                                     ("10.0.0.1", True), ("10.0.0.1", "80")])
def test_invalid_relay_does_not_execute_anything(ip, port):
    with pytest.raises(ValueError):
        firewall_commands(ip, port)


def test_endpoint_is_canonical_and_no_relay_is_valid():
    assert validate_endpoint(None, None) == (None, None)
    assert validate_endpoint("2001:db8:0::1", 8123) == ("2001:db8::1", 8123)


def test_both_families_drop_before_flush_or_allow_and_forwarding_stays_closed():
    commands = firewall_commands("192.0.2.23", 54321)
    assert len(commands[:8]) == 8
    assert all("-P" in cmd and cmd[-1] == "DROP" for cmd in commands[:8])
    for binary in ("iptables", "ip6tables"):
        for chain in ("INPUT", "OUTPUT", "FORWARD"):
            assert [binary, "-w", "5", "-t", "filter", "-P", chain, "DROP"] in commands
        assert [binary, "-w", "5", "-t", "raw", "-P", "OUTPUT", "DROP"] in commands
    assert not any("FORWARD" in cmd and "ACCEPT" in cmd for cmd in commands)


def test_embedded_dns_is_rejected_before_nat_regardless_of_destination_port():
    commands = firewall_commands()
    for table in ("raw", "filter"):
        drop = ["iptables", "-w", "5", "-t", table, "-A", "OUTPUT",
                "-d", "127.0.0.11/32", "-j", "DROP"]
        assert drop in commands
        assert not any("--dport" in cmd for cmd in commands)
        allow_index = next(i for i, cmd in enumerate(commands)
                           if cmd[:5] == ["iptables", "-w", "5", "-t", table]
                           and cmd[-1] == "ACCEPT")
        assert commands.index(drop) < allow_index
    assert not any("127.0.0.0/8" in cmd for cmd in commands)


@pytest.mark.parametrize("ip,binary", [("192.0.2.23", "iptables"),
                                      ("2001:db8::23", "ip6tables")])
def test_only_one_precise_tcp_relay_is_allowed_and_no_broad_established_rule(ip, binary):
    commands = firewall_commands(ip, 54321)
    outgoing = [cmd for cmd in commands if "--dport" in cmd]
    assert len(outgoing) == 2  # Both pre-NAT and post-NAT enforcement.
    assert all(cmd[0] == binary and cmd[cmd.index("-d") + 1] == ip
               and cmd[cmd.index("--dport") + 1] == "54321"
               and cmd[cmd.index("-p") + 1] == "tcp" for cmd in outgoing)
    incoming = [cmd for cmd in commands if "ESTABLISHED" in cmd]
    assert len(incoming) == 1
    assert "--sport" in incoming[0] and ip in incoming[0] and "INPUT" in incoming[0]
    assert not any("RELATED" in cmd for cmd in commands)


def test_install_stops_on_first_command_failure_and_never_reports_enforced():
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] and kwargs["timeout"] == 10
        if len(calls) == 4:
            raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        install_firewall(run=fail)
    assert len(calls) == 4


def test_success_reports_only_after_all_commands_complete():
    calls = []
    assert install_firewall("192.0.2.5", 8080, run=lambda command, **kw: calls.append(command)) == (
        boundary_facts("192.0.2.5", 8080))
    assert calls == firewall_commands("192.0.2.5", 8080)


def test_prepare_builds_current_source_once_per_instance_using_docker_layer_cache(monkeypatch):
    boundary = NetworkBoundary()
    calls = []

    def docker(self, args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(NetworkBoundary, "_docker", docker)
    boundary.prepare()
    boundary.prepare()
    assert [args[0] for args, _ in calls] == ["build"]
    assert NETWORK_HELPER_IMAGE in calls[0][0]
    assert calls[0][0][-2].endswith("/network_helper.Dockerfile")
    assert "--no-cache" not in calls[0][0]
    # An existing fixed tag is insufficient after firewall source is edited.
    # A new task must ask Docker to reconcile COPY with the current checkout.
    NetworkBoundary().prepare()
    assert [args[0] for args, _ in calls] == ["build", "build"]


def test_prepare_build_failure_does_not_mark_ready(monkeypatch):
    boundary = NetworkBoundary()

    def docker(args, **kwargs):
        if args[0] == "build":
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(boundary, "_docker", docker)
    with pytest.raises(subprocess.CalledProcessError):
        boundary.prepare()
    assert not boundary._prepared


def mocked_boundary(monkeypatch, *, start_error=None, cleanup_error=None, facts=None):
    boundary = NetworkBoundary()
    boundary._prepared = True
    calls = []

    def docker(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "create":
            return subprocess.CompletedProcess(args, 0, HELPER_ID + "\n", "")
        if args[0] == "start":
            if start_error:
                raise start_error
            return subprocess.CompletedProcess(args, 0,
                                               json.dumps(facts or boundary_facts("192.0.2.5", 8080)), "")
        if args[0] == "rm" and cleanup_error:
            raise cleanup_error
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(boundary, "_docker", docker)
    return boundary, calls


def test_helper_has_only_net_admin_and_is_removed_by_returned_id(monkeypatch):
    boundary, calls = mocked_boundary(monkeypatch)
    assert boundary.configure("task-container", "192.0.2.5", 8080) == boundary_facts("192.0.2.5", 8080)
    create = calls[0][0]
    assert create[create.index("--network") + 1] == "container:task-container"
    assert create[create.index("--cap-drop") + 1] == "ALL"
    assert create[create.index("--cap-add") + 1] == "NET_ADMIN"
    assert "--read-only" in create and "no-new-privileges:true" in create
    assert not any(option in create for option in ("--privileged", "--pid", "--mount"))
    assert calls[1][0] == ["start", "--attach", HELPER_ID]
    assert calls[-1][0] == ["rm", "--force", HELPER_ID]


@pytest.mark.parametrize("error", [subprocess.CalledProcessError(1, ["docker", "start"]),
                                   subprocess.TimeoutExpired(["docker", "start"], 90)])
def test_helper_failure_or_timeout_propagates_after_cleanup(monkeypatch, error):
    boundary, calls = mocked_boundary(monkeypatch, start_error=error)
    with pytest.raises(type(error)) as raised:
        boundary.configure("task-container", "192.0.2.5", 8080)
    assert raised.value is error
    assert calls[-1][0] == ["rm", "--force", HELPER_ID]


def test_wrong_helper_confirmation_is_a_failure(monkeypatch):
    boundary, calls = mocked_boundary(monkeypatch, facts={"policy": "open", "enforced": True})
    with pytest.raises(RuntimeError, match="did not confirm"):
        boundary.configure("task-container", "192.0.2.5", 8080)
    assert calls[-1][0][0] == "rm"


def test_cleanup_failure_preserves_original_failure(monkeypatch):
    original = subprocess.CalledProcessError(1, ["docker", "start"])
    boundary, _ = mocked_boundary(monkeypatch, start_error=original, cleanup_error=OSError("daemon lost"))
    with pytest.raises(subprocess.CalledProcessError) as raised:
        boundary.configure("task-container", "192.0.2.5", 8080)
    assert raised.value is original
    assert "cleanup also failed" in " ".join(original.__notes__)


def test_cleanup_failure_after_success_still_fails_closed(monkeypatch):
    boundary, _ = mocked_boundary(monkeypatch, cleanup_error=OSError("daemon lost"))
    with pytest.raises(OSError, match="daemon lost"):
        boundary.configure("task-container", "192.0.2.5", 8080)


@pytest.mark.parametrize("name", ["", "--all", "container; evil", "/var/run/docker.sock", None])
def test_invalid_container_target_cannot_run_commands(monkeypatch, name):
    boundary = NetworkBoundary()
    monkeypatch.setattr(boundary, "_docker", lambda *args, **kw: pytest.fail("must not execute"))
    with pytest.raises(ValueError):
        boundary.configure(name)
