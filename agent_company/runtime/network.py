"""Host-owned Docker network-namespace boundary for single and team rollouts."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
from typing import ClassVar
import uuid

from agent_company.runtime.netns_firewall import NETWORK_POLICY, boundary_facts, validate_endpoint


NETWORK_HELPER_IMAGE = f"agent-company-network-helper:{NETWORK_POLICY}"


@dataclass
class NetworkBoundary:
    image: ClassVar[str] = NETWORK_HELPER_IMAGE
    docker: str = "docker"
    _prepared: bool = field(default=False, init=False)

    def _docker(self, args: list[str], *, timeout: float = 30, check: bool = True):
        return subprocess.run([self.docker, *args], text=True, capture_output=True,
                              check=check, timeout=timeout)

    def prepare(self) -> None:
        """Build the current helper source before starting the task budget.

        Only this trusted image build may install iptables from its package
        repository. Agents never receive a package-download/network exception.
        Docker reuses unchanged layers, but a fixed image tag alone cannot prove
        the copied firewall script matches the current checkout. Build once per
        boundary instance so edits invalidate COPY through Docker's own cache.
        """
        if self._prepared:
            return
        source = Path(__file__).resolve().parent
        self._docker(["build", "--tag", NETWORK_HELPER_IMAGE,
                      "--file", str(source / "network_helper.Dockerfile"), str(source)],
                     timeout=600)
        self._prepared = True

    def configure(self, container_id: str, relay_ip: str | None = None,
                  relay_port: int | None = None) -> dict:
        """Deny all nonlocal networking, optionally except one model relay.

        Call after every task-container start/restart and before starting agents.
        Any helper/create/start/verification/cleanup failure propagates to the
        caller; no permissive fallback is allowed. A helper's capability belongs
        only to the helper, never to the task container or its processes.
        """
        if not isinstance(container_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                                                               container_id):
            raise ValueError("An explicit task container ID or name is required")
        relay_ip, relay_port = validate_endpoint(relay_ip, relay_port)
        self.prepare()
        args = ["create", "--name", "ac-network-helper-" + uuid.uuid4().hex,
                "--network", f"container:{container_id}",
                "--cap-drop", "ALL", "--cap-add", "NET_ADMIN",
                "--security-opt", "no-new-privileges:true", "--read-only",
                "--tmpfs", "/run:rw,nosuid,nodev,noexec,size=1m", "--pids-limit", "32",
                NETWORK_HELPER_IMAGE]
        if relay_ip is not None:
            args += ["--relay-ip", relay_ip, "--relay-port", str(relay_port)]
        helper_id = self._docker(args).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{12,64}", helper_id):
            raise RuntimeError("Docker did not return a valid firewall helper container ID")
        error: BaseException | None = None
        try:
            result = self._docker(["start", "--attach", helper_id], timeout=90)
            facts = json.loads(result.stdout)
            if facts != boundary_facts(relay_ip, relay_port):
                raise RuntimeError("Firewall helper did not confirm the requested boundary")
            return facts
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self._docker(["rm", "--force", helper_id])
            except BaseException as cleanup:
                if error is None:
                    raise
                error.add_note(f"Firewall helper {helper_id} cleanup also failed: "
                               f"{type(cleanup).__name__}")
