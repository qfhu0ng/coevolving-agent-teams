"""Opt-in real Docker egress checks; no model calls or upstream source downloads.

Set STAGE1_NETWORK_DOCKER=1 and STAGE1_DOCKER_TASK to a locally cached official
RoadmapBench tpl-5 task. STAGE1_NETWORK_DOCKER_OUTPUT optionally saves JSON facts.
Only bounded TCP connection and DNS probes address public endpoints; no HTTP
request, repository clone, or content download is sent to them.
The trusted helper build uses Docker layer caching outside the agent environment.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit
import uuid

import pytest

from agent_company.runtime.network import NETWORK_HELPER_IMAGE
from agent_company.roadmap.task import Task
from agent_company.runtime.workspace import DockerWorkspace


# This program runs inside the task image, with exactly the agent's privileges.
# Keeping probes in Python avoids a dependency on curl, dig, or netcat in images.
PROBE = r'''
import ctypes
import errno
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import threading

request = json.loads(sys.argv[1])

def error(exc):
    return {"error": type(exc).__name__, "errno": getattr(exc, "errno", None),
            "message": str(exc)}

def connect(host, port):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as client:
            client.settimeout(0.7)
            client.connect((host, port))
        return {"connected": True}
    except OSError as exc:
        return {"connected": False, **error(exc)}

def http(host, port):
    client = HTTPConnection(host, port, timeout=3)
    try:
        client.request("GET", "/v1")
        response = client.getresponse()
        return {"status": response.status, "body": response.read(256).decode()}
    finally:
        client.close()

if request["mode"] == "connect":
    result = {name: connect(*target) for name, target in request["targets"].items()}
elif request["mode"] == "http":
    result = http(request["host"], request["port"])
elif request["mode"] == "loopback":
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"loopback-ok")
    class IPv6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6
    result = {}
    for host, server_type in (("127.0.0.1", ThreadingHTTPServer), ("::1", IPv6Server)):
        try:
            server = server_type((host, 0), Handler)
        except OSError as exc:
            if host != "::1" or exc.errno not in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                raise
            result[host] = {"available": False, **error(exc)}
            continue
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            result[host] = {"available": True, **http(host, server.server_port)}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
elif request["mode"] == "dns":
    # A real A query also catches a reachable Docker embedded DNS proxy. Any
    # response, even SERVFAIL, means the supposedly forbidden resolver was reached.
    name = b"".join(bytes([len(part)]) + part.encode()
                    for part in "github.com".split(".")) + b"\0"
    query = struct.pack("!HHHHHH", 0xAC51, 0x0100, 1, 0, 0, 0) + name + struct.pack("!HH", 1, 1)
    result = {}
    for host in ("1.1.1.1", "8.8.8.8", "127.0.0.11"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
                client.settimeout(0.7)
                client.sendto(query, (host, 53))
                reply, peer = client.recvfrom(4096)
            result[host] = {"received": True, "bytes": len(reply), "peer": peer}
        except OSError as exc:
            result[host] = {"received": False, **error(exc)}
elif request["mode"] == "resolve":
    result = {}
    for host in ("github.com", "raw.githubusercontent.com", "codeload.github.com"):
        try:
            answers = socket.getaddrinfo(host + ".", 443, type=socket.SOCK_STREAM)
            result[host] = {"resolved": True, "addresses": sorted({row[4][0] for row in answers})}
        except socket.gaierror as exc:
            result[host] = {"resolved": False, **error(exc)}
elif request["mode"] == "privileges":
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
                  if ":" in line)
    result = {name: status[name].strip() for name in ("CapEff", "CapPrm", "CapBnd", "NoNewPrivs")}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP):
            result["raw_socket"] = {"created": True}
    except OSError as exc:
        result["raw_socket"] = {"created": False, **error(exc)}
    libc = ctypes.CDLL(None, use_errno=True)
    outcome = libc.unshare(0x40000000)  # CLONE_NEWNET, in this disposable probe process only.
    result["unshare_network"] = {"returncode": outcome, "errno": ctypes.get_errno()}
    result["firewall"] = {}
    for command in ("iptables", "ip6tables"):
        executable = shutil.which(command)
        if executable is None:
            result["firewall"][command] = {"available": False}
            continue
        completed = subprocess.run([executable, "-w", "1", "-P", "OUTPUT", "ACCEPT"],
                                   text=True, capture_output=True, timeout=4)
        result["firewall"][command] = {"available": True, "returncode": completed.returncode,
                                        "stderr": completed.stderr[-2000:]}
else:
    raise ValueError("Unknown probe mode")
print(json.dumps(result))
'''


@contextmanager
def _host_server(body):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("0.0.0.0", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _probe(workspace, mode, **arguments):
    completed = workspace.run(
        ["python3", "-c", PROBE, json.dumps({"mode": mode, **arguments})],
        env={"RES_OPTIONS": "attempts:1 timeout:1"}, timeout=25,
    )
    return json.loads(completed.stdout)


def _assert_blocked(facts):
    assert facts and all(not row["connected"] for row in facts.values()), facts


@pytest.mark.skipif(os.environ.get("STAGE1_NETWORK_DOCKER") != "1",
                    reason="Opt in with STAGE1_NETWORK_DOCKER=1; uses real Docker")
def test_task_network_allows_only_loopback_and_configured_relay(tmp_path):
    task_path = os.environ.get("STAGE1_DOCKER_TASK")
    assert task_path, "STAGE1_DOCKER_TASK must point to the locally cached official tpl-5 task"
    task = Task.load(task_path)
    assert task.task_id == "tpl-5.0.0-roadmap", "Use the reviewed official tpl-5 development task"
    assert task.image, "The test requires a prebuilt, locally cached task image"
    # This check cannot pull or build an image. An opted-in run must not silently
    # turn a missing cache into an external download.
    cached = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", task.image],
                            check=True, capture_output=True, text=True, timeout=30)
    cached_helper = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", NETWORK_HELPER_IMAGE],
        check=True, capture_output=True, text=True, timeout=30,
    )
    name = "ac-stage1-" + uuid.uuid4().hex
    workspace = DockerWorkspace(task, platform="linux/amd64", container_name=name)
    facts = {"task_id": task.task_id, "image": task.image, "image_id": cached.stdout.strip(),
             "network_helper_image_id": cached_helper.stdout.strip(),
             "container_name": name, "checks": {}}
    output = Path(os.environ.get("STAGE1_NETWORK_DOCKER_OUTPUT", tmp_path / "network_facts.json"))
    body = ("relay-" + uuid.uuid4().hex).encode()
    with _host_server(body) as relay, _host_server(b"forbidden-port") as other:
        try:
            workspace.start()
            facts["policy_before_relay"] = dict(workspace.network_policy_status)
            assert facts["policy_before_relay"]["enforced"]
            assert facts["policy_before_relay"]["allowed_relay"] is None
            # The shared-network task cannot receive --add-host. Resolve the
            # existing hosts entry in its trusted holder, without opening a port.
            addresses = workspace._docker(
                ["exec", workspace._network_container_id, "getent", "ahostsv4",
                 "host.docker.internal"], timeout=10,
            ).stdout.split()
            assert addresses, "The namespace holder must contain the Docker host-gateway entry"
            host = str(ipaddress.IPv4Address(addresses[0]))
            facts["host_ip"] = host
            checks = facts["checks"]
            checks["before_relay"] = _probe(workspace, "connect", targets={
                "relay": [host, relay.server_port], "other_host_port": [host, other.server_port],
                "public_ip": ["1.1.1.1", 443],
            })
            _assert_blocked(checks["before_relay"])
            checks["loopback"] = _probe(workspace, "loopback")
            assert checks["loopback"]["127.0.0.1"]["body"] == "loopback-ok"
            if checks["loopback"]["::1"]["available"]:
                assert checks["loopback"]["::1"]["body"] == "loopback-ok"

            relay_url = workspace.configure_relay(relay.server_port)
            parsed = urlsplit(relay_url)
            assert parsed.scheme == "http" and parsed.path == "/v1", relay_url
            assert parsed.port == relay.server_port and parsed.hostname == host, relay_url
            ipaddress.ip_address(parsed.hostname)  # No DNS exception is required for the relay.
            facts["relay_url"] = relay_url
            facts["policy_after_relay"] = dict(workspace.network_policy_status)
            assert facts["policy_after_relay"]["enforced"]
            assert facts["policy_after_relay"]["allowed_relay"] == {
                "ip": host, "port": relay.server_port}
            checks["relay_http"] = _probe(workspace, "http", host=host, port=relay.server_port)
            assert checks["relay_http"] == {"status": 200, "body": body.decode()}

            forbidden = {
                "other_host_port": [host, other.server_port],
                "public_ip": ["1.1.1.1", 443],
                "public_dns_tcp": ["8.8.8.8", 53],
                "embedded_dns_tcp": ["127.0.0.11", 53],
                # Fixed literal destinations simulate bypassing name resolution;
                # no claim is made that these remain the domains' current IPs.
                "github_preset_ip": ["140.82.112.3", 443],
                "raw_github_preset_ip": ["185.199.108.133", 443],
                "codeload_preset_ip": ["140.82.112.10", 443],
                "public_ipv6": ["2606:4700:4700::1111", 443],
            }
            checks["forbidden_destinations"] = _probe(workspace, "connect", targets=forbidden)
            _assert_blocked(checks["forbidden_destinations"])
            checks["dns_packets"] = _probe(workspace, "dns")
            assert all(not row["received"] for row in checks["dns_packets"].values()), checks["dns_packets"]
            checks["clone_domain_resolution"] = _probe(workspace, "resolve")
            assert all(not row["resolved"] for row in checks["clone_domain_resolution"].values()), (
                checks["clone_domain_resolution"])
            checks["privileges"] = privileges = _probe(workspace, "privileges")
            forbidden_caps = (1 << 12) | (1 << 13)  # CAP_NET_ADMIN and CAP_NET_RAW.
            for capability_set in ("CapEff", "CapPrm", "CapBnd"):
                assert int(privileges[capability_set], 16) & forbidden_caps == 0, privileges
            assert privileges["NoNewPrivs"] == "1", privileges
            assert not privileges["raw_socket"]["created"], privileges
            assert privileges["unshare_network"]["returncode"] == -1, privileges
            assert all(not row["available"] or row["returncode"] != 0
                       for row in privileges["firewall"].values()), privileges

            # Restart only the task container, leaving the trusted holder alive.
            # Actual socket probes must keep passing without reinstalling rules.
            for phase in ("task_restart", "after_finish_agent"):
                if phase == "task_restart":
                    workspace._docker(["stop", "-t", "0", workspace.container_id], timeout=30)
                    workspace._docker(["start", workspace.container_id], timeout=30)
                else:
                    workspace.finish_agent()
                    assert workspace.agent_finished
                checks[phase] = {
                    "relay": _probe(workspace, "http", host=host, port=relay.server_port),
                    "blocked": _probe(workspace, "connect", targets=forbidden),
                    "dns": _probe(workspace, "dns"),
                }
                assert checks[phase]["relay"] == {"status": 200, "body": body.decode()}
                _assert_blocked(checks[phase]["blocked"])
                assert all(not row["received"] for row in checks[phase]["dns"].values()), checks[phase]
            facts["status"] = "passed"
        except BaseException as exc:
            facts["status"] = "failed"
            facts["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            try:
                workspace.stop()
                remaining = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name=^/{name}", "--format", "{{.Names}}"],
                    check=True, capture_output=True, text=True, timeout=30,
                ).stdout.splitlines()
                facts["remaining_owned_containers"] = remaining
                assert not remaining, f"Workspace stop left owned containers: {remaining}"
            except BaseException as exc:
                facts["status"] = "failed"
                facts["cleanup_failure"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                facts["policy_final"] = dict(getattr(workspace, "network_policy_status", {}))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
