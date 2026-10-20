"""One Docker task environment shared by the master and every dynamic worker."""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import time
import uuid
from typing import Mapping, Sequence

from agent_company.roadmap.task import Task
from agent_company.runtime.network import NETWORK_POLICY, NetworkBoundary

# Seconds for each `docker rm -f` attempt of one container (see DockerWorkspace._remove_container).
CONTAINER_REMOVE_TIMEOUTS = (120, 300)


def diagnostic_text(value: str | bytes | None, secrets: Sequence[str] = ()) -> str:
    """Bound diagnostic excerpts and never copy known credentials into them."""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]+",
                  r"\1[REDACTED]", text)
    return text[-16384:]


def exception_diagnostic(exc: BaseException, secrets: Sequence[str] = ()) -> dict:
    """Subprocess exceptions stringify the entire command, including --env secrets."""
    result = {"type": type(exc).__name__}
    if isinstance(exc, subprocess.SubprocessError):
        for name in ("timeout", "returncode"):
            if getattr(exc, name, None) is not None:
                result[name] = getattr(exc, name)
        for name in ("stdout", "stderr"):
            if getattr(exc, name, None):
                result[name] = diagnostic_text(getattr(exc, name), secrets)
    else:
        result["message"] = diagnostic_text(
            (exc.strerror or str(exc)) if isinstance(exc, OSError) else str(exc), secrets)
    return result


@dataclass
class DockerWorkspace:
    task: Task
    mounts: Sequence[tuple[str | Path, str, bool]] = ()
    image_override: str | None = None
    platform: str | None = None
    container_name: str | None = None
    docker: str = "docker"
    worker_root: str = "/workspace/workers"
    runtime_root: str = "/opt/agent-company/run"
    relay_bind_host: str = "0.0.0.0"
    relay_connect_host: str = "host.docker.internal"
    repo_path: str = field(default="", init=False)
    container_id: str | None = field(default=None, init=False)
    agent_finished: bool = field(default=False, init=False)
    diagnostics: list[dict] = field(default_factory=list, init=False)
    diagnostic_secrets: Sequence[str] = field(default=(), init=False, repr=False)
    _phase: str = field(default="not_started", init=False)
    _network: NetworkBoundary | None = field(default=None, init=False, repr=False)
    _network_container_id: str | None = field(default=None, init=False)
    _relay_endpoint: tuple[str, int] | None = field(default=None, init=False)
    _network_enforced: bool = field(default=False, init=False)

    @property
    def network_policy_status(self) -> dict:
        return {"policy": NETWORK_POLICY, "enforced": self._network_enforced,
                "namespace_container_id": self._network_container_id,
                "allowed_relay": ({"ip": self._relay_endpoint[0], "port": self._relay_endpoint[1]}
                                  if self._relay_endpoint else None),
                "dependency_access": "preinstalled-or-offline",
                "verifier_network": "same-restricted-namespace"}

    def _start_network(self) -> None:
        """Keep the firewall's namespace alive across task stop/start.

        The namespace holder has no agent, task files, Docker socket or network
        administration capability. Only a short-lived trusted helper sets rules.
        A task restart must not create a fresh, unfiltered network namespace.
        """
        self._phase = "network_setup"
        self._network = NetworkBoundary(docker=self.docker)
        self._network.prepare()
        name = (self.container_name or "ac-stage1-" + uuid.uuid4().hex) + "-network"
        self._network_container_id = self._docker([
            "create", "--name", name, "--network", "bridge", "--memory", "32m",
            "--cpus", "0.1", "--pids-limit", "8", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--read-only",
            "--add-host", "host.docker.internal:host-gateway",
            "--entrypoint", "python", self._network.image,
            "-c", "import time; time.sleep(2147483647)",
        ], timeout=30).stdout.strip()
        self._docker(["start", self._network_container_id], timeout=30)
        self._apply_network_boundary()

    def _apply_network_boundary(self) -> None:
        if self._network is None or not self._network_container_id:
            raise RuntimeError("Restricted network namespace is not started")
        self._network_enforced = False
        ip, port = self._relay_endpoint or (None, None)
        self._network.configure(self._network_container_id, relay_ip=ip, relay_port=port)
        self._network_enforced = True

    def configure_relay(self, port: int) -> str:
        """Allow exactly this host relay socket; never a subnet or generic proxy."""
        if self.agent_finished or not self.container_id or not self._network_container_id:
            raise RuntimeError("Relay can only be configured in an active agent workspace")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Invalid relay port")
        # Resolve in the trusted holder, not the agent-editable /etc/hosts. The
        # Docker host-gateway entry does not need an external DNS request.
        answer = self._docker(["exec", self._network_container_id, "getent", "ahostsv4",
                               self.relay_connect_host], timeout=10).stdout
        address = str(ipaddress.IPv4Address(answer.split()[0]))
        self._relay_endpoint = (address, port)
        self._apply_network_boundary()
        return f"http://{address}:{port}/v1"

    def _snapshot(self, outcome: str, *, error: BaseException | None = None,
                  stdout_path: Path | None = None, stderr_path: Path | None = None,
                  **facts) -> dict:
        """Best effort only: one bounded read before destructive lifecycle transitions.

        Inspect .State, never .Config (which contains environment variables). An
        uncertain startup name is only observed, not adopted as an owned container.
        """
        record = {"phase": self._phase, "outcome": outcome, "time": time.time(),
                  "container_name": self.container_name, "container_id": self.container_id, **facts}
        record["network"] = self.network_policy_status
        if error is not None:
            record["error"] = exception_diagnostic(error, self.diagnostic_secrets)
        for name, path in (("stdout_tail", stdout_path), ("stderr_tail", stderr_path)):
            if path is not None:
                try:
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 65536))
                        record[name] = diagnostic_text(stream.read(), self.diagnostic_secrets)
                except OSError as exc:
                    record[name + "_error"] = type(exc).__name__
        target = self.container_id or self.container_name
        if target:
            record["inspect_target"] = target
            try:
                response = self._docker(["inspect", "--format", "{{json .State}}", target], timeout=2)
                state = json.loads(response.stdout)
                if not isinstance(state, dict):
                    raise ValueError("Container state is not an object")
                record["container_state"] = {
                    key: diagnostic_text(value, self.diagnostic_secrets) if isinstance(value, str) else value
                    for key, value in state.items()
                    if key in {"Status", "Running", "Paused", "Restarting", "OOMKilled", "Dead",
                               "Pid", "ExitCode", "Error", "StartedAt", "FinishedAt"}
                }
            except Exception as exc:
                record["inspect_error"] = exception_diagnostic(exc, self.diagnostic_secrets)
        self.diagnostics.append(record)
        # A task has only a few lifecycle phases. Bound storage even for repeated cleanup failures.
        del self.diagnostics[:-32]
        return record

    def _docker(self, args: Sequence[str], **kwargs):
        return subprocess.run([self.docker, *map(str, args)], text=True,
                              capture_output=True, check=True, **kwargs)

    def start(self) -> "DockerWorkspace":
        if self.container_id:
            raise RuntimeError("Workspace is already started")
        try:
            return self._start()
        except BaseException as exc:
            self._snapshot("startup_exception", error=exc)
            if self.container_id or self._network_container_id:
                try:
                    self.stop()
                except Exception as cleanup:
                    self.diagnostics[-1]["cleanup_error"] = exception_diagnostic(cleanup, self.diagnostic_secrets)
            raise

    def _start(self) -> "DockerWorkspace":
        if self.container_id:
            raise RuntimeError("Workspace is already started")
        self._phase = "workspace_configuration"
        image = self.image_override or self.task.image
        if not image:
            self._phase = "docker_build"
            # Docker may create a task-specific image; no user image is overwritten.
            result = self._docker(
                ["build", "-q", str(self.task.path / "environment")],
                timeout=self.task.build_timeout_sec,
            )
            image = result.stdout.strip().splitlines()[-1]
        args = ["run", "-d", "--init", "--cpus", str(self.task.cpus),
                "--memory", f"{self.task.memory_mb}m",
                "--cap-drop", "NET_ADMIN", "--cap-drop", "NET_RAW",
                "--security-opt", "no-new-privileges"]
        if self.platform:
            args += ["--platform", self.platform]
        if self.container_name:
            args += ["--name", self.container_name]
        for source, target, readonly in self.mounts:
            source = Path(source).resolve()
            # Mounts are explicit runtime paths, never the benchmark task directory.
            if source == self.task.path or source in self.task.path.parents:
                raise ValueError("A runtime mount must not expose the benchmark tests or oracle")
            if any(hidden == source or hidden in source.parents
                   for hidden in (self.task.path / "tests", self.task.path / "solution")):
                raise ValueError("Hidden evaluation data cannot be mounted during agent execution")
            args += ["--mount", f"type=bind,source={source},target={target}"
                     + (",readonly" if readonly else "")]
        # Install the default-deny rules before starting even the task entrypoint.
        self._start_network()
        args += ["--network", f"container:{self._network_container_id}"]
        # Override task entrypoints and leave its original WORKDIR intact.
        args += ["--entrypoint", "/bin/sh", image, "-c", "exec sleep infinity"]
        self._phase = "docker_run"
        self.container_id = self._docker(args, timeout=self.task.build_timeout_sec).stdout.strip()
        self._phase = "workspace_setup"
        self.repo_path = self.run(["pwd"]).stdout.strip()
        if not self.repo_path.startswith("/") or self.repo_path == "/":
            raise ValueError("Task image needs an explicit repository WORKDIR")
        self.run(["mkdir", "-p", self.worker_root, self.runtime_root])
        self.run(["git", "config", "--global", "--add", "safe.directory", "*"], cwd=self.repo_path)
        check = self.run(["git", "rev-parse", "--verify", "HEAD"], check=False)
        if check.returncode:
            self.run(["git", "init"], cwd=self.repo_path)
            self.run(["git", "add", "-A"], cwd=self.repo_path)
            self.run(["git", "-c", "user.name=Agent Company", "-c",
                      "user.email=agent-company@localhost", "commit", "-qm", "Task initial state"],
                     cwd=self.repo_path)
        self.run(["git", "config", "user.name", "Agent Company"], cwd=self.repo_path)
        self.run(["git", "config", "user.email", "agent-company@localhost"], cwd=self.repo_path)
        self._phase = "workspace_ready"
        return self

    @property
    def task_cwd(self) -> str:
        return self.repo_path

    @property
    def resources(self) -> dict:
        return {"cpus": self.task.cpus, "memory_mb": self.task.memory_mb,
                "shared_by_all_agents": True, "storage_mb_requested": self.task.storage_mb,
                "storage_quota_enforced": False}

    def _exec_args(self, command: Sequence[str], cwd: str | None = None,
                   env: Mapping[str, str] | None = None) -> list[str]:
        if not self.container_id:
            raise RuntimeError("Workspace is not started")
        args = [self.docker, "exec"]
        if cwd or self.repo_path:
            args += ["--workdir", cwd or self.repo_path]
        for key, value in (env or {}).items():
            args += ["--env", f"{key}={value}"]
        return [*args, self.container_id, *map(str, command)]

    def run(self, command: Sequence[str], *, cwd: str | None = None,
            env: Mapping[str, str] | None = None, timeout: float | None = 60,
            check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(self._exec_args(command, cwd, env), text=True,
                              capture_output=True, timeout=timeout, check=check)

    exec = run

    def upload(self, source: str | Path, destination: str, *, contents: bool = False) -> None:
        if not self.container_id:
            raise RuntimeError("Workspace is not started")
        source = Path(source).resolve()
        # Directory contents are merged into an existing destination, like Harbor's tests copy.
        source_text = str(source) + ("/." if contents and source.is_dir() else "")
        self._docker(["cp", source_text, f"{self.container_id}:{destination}"], timeout=120)

    def download(self, source: str, destination: str | Path) -> None:
        if not self.container_id:
            raise RuntimeError("Workspace is not started")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._docker(["cp", f"{self.container_id}:{source}", str(destination)], timeout=120)

    def write_file(self, path: str, content: str) -> None:
        self.run(["mkdir", "-p", str(PurePosixPath(path).parent)])
        with tempfile.TemporaryDirectory(prefix="agent-company-upload-") as temporary:
            file = Path(temporary) / "payload"
            file.write_text(content)
            self.upload(file, path)

    def write_file_atomic(self, path: str, content: str) -> None:
        """Upload beside the target, then rename: readers never see a partial file."""
        temporary_path = path + ".tmp"
        self.write_file(temporary_path, content)
        self.run(["mv", "-f", temporary_path, path])

    def ensure_worker(self, name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Worker name must be a single safe path component")
        path = str(PurePosixPath(self.worker_root) / name)
        existing = self.run(["test", "-d", path], check=False)
        if existing.returncode:
            self.run(["git", "worktree", "add", "--detach", path, "HEAD"], cwd=self.repo_path)
        return path

    def read_head(self, path: str) -> str | None:
        result = self.run(["git", "rev-parse", "HEAD"], cwd=path, check=False, timeout=5)
        return result.stdout.strip() or None if result.returncode == 0 else None

    def _long_command(self, command: Sequence[str], *, env: Mapping[str, str] | None,
                      timeout_seconds: float, stdout_path: str | Path,
                      stderr_path: str | Path) -> dict:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        started = time.monotonic()
        stdout_path, stderr_path = Path(stdout_path), Path(stderr_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        timed_out = False
        # GNU timeout terminates its command process group inside the container.
        args = self._exec_args(["timeout", "--signal=TERM", "--kill-after=5s",
                                f"{timeout_seconds}s", *command], env=env)
        process, diagnostic = None, None
        cleanup_attempted = False
        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                process = subprocess.Popen(args, stdout=stdout, stderr=stderr, text=True)
                try:
                    returncode = process.wait(timeout=timeout_seconds + 15)
                    timed_out = returncode == 124 or (
                        returncode == 137 and time.monotonic() - started >= timeout_seconds
                    )
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout.flush()
                    stderr.flush()
                    diagnostic = self._snapshot("host_wait_timeout", error=exc,
                                                stdout_path=stdout_path, stderr_path=stderr_path)
                    # Killing docker-exec alone leaves Qwen/tools alive. Inspect first,
                    # then stop their container; do not let cleanup mask the timeout.
                    cleanup_attempted = True
                    if self._terminate_exec(process, diagnostic):
                        raise
                    returncode = 124
        except BaseException as exc:
            if diagnostic is None:
                diagnostic = self._snapshot("command_exception", error=exc,
                                            stdout_path=stdout_path, stderr_path=stderr_path)
            if process is not None and not cleanup_attempted:
                self._terminate_exec(process, diagnostic)
            raise
        elapsed_seconds = time.monotonic() - started
        if diagnostic is None:
            diagnostic = self._snapshot("command_exit", stdout_path=stdout_path, stderr_path=stderr_path)
        oom_killed = diagnostic.get("container_state", {}).get("OOMKilled")
        if not isinstance(oom_killed, bool):
            oom_killed = None
        if oom_killed:
            timed_out = False
        diagnostic.update(returncode=returncode, timed_out=timed_out,
                          host_signal=-returncode if returncode < 0 else None)
        return {"returncode": returncode, "timed_out": timed_out, "oom_killed": oom_killed,
                "elapsed_seconds": elapsed_seconds, "diagnostics": diagnostic}

    def _terminate_exec(self, process, diagnostic: dict) -> list[dict]:
        failures = []
        try:
            self._docker(["stop", "-t", "0", self.container_id], timeout=30)
        except Exception as exc:
            failures.append(exception_diagnostic(exc, self.diagnostic_secrets))
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception as exc:
            failures.append(exception_diagnostic(exc, self.diagnostic_secrets))
        if failures:
            diagnostic["termination_errors"] = failures
        return failures

    def run_agent(self, command: Sequence[str], *, env: Mapping[str, str],
                  timeout_seconds: float, stdout_path: str | Path,
                  stderr_path: str | Path) -> dict:
        if self.agent_finished:
            raise RuntimeError("Agent execution cannot resume after final verification began")
        self._phase = "agent"
        self.diagnostic_secrets = (*self.diagnostic_secrets, *(value for key, value in env.items()
            if re.search(r"(?i)(key|token|secret|password|authorization)", key) and value))
        # No per-worker multiplication: this is one deadline for the whole Qwen team.
        timeout_seconds = min(timeout_seconds, self.task.agent_timeout_sec)
        try:
            # Reinstall the exact allowlist before every launch. No unrestricted
            # fallback is permitted if the helper or Docker fails.
            self._apply_network_boundary()
            result = self._long_command(command, env=env, timeout_seconds=timeout_seconds,
                                        stdout_path=stdout_path, stderr_path=stderr_path)
        except BaseException:
            try:
                self.finish_agent()
            except Exception as exc:
                self.diagnostics.append({"phase": "agent_quiesce", "outcome": "cleanup_exception",
                                         "error": exception_diagnostic(exc, self.diagnostic_secrets)})
            raise
        else:
            self.finish_agent()
            return result

    def finish_agent(self) -> None:
        if not self.container_id:
            raise RuntimeError("Workspace is not started")
        if not self.agent_finished:
            self._phase = "agent_quiesce"
            try:
                self._docker(["stop", "-t", "0", self.container_id], timeout=30)
                self._docker(["start", self.container_id], timeout=30)
                self.agent_finished = True
            except BaseException as exc:
                self._snapshot("quiesce_exception", error=exc)
                raise

    def run_verifier(self, command: Sequence[str], *, timeout_seconds: float,
                     stdout_path: str | Path, stderr_path: str | Path) -> dict:
        if not self.agent_finished:
            raise RuntimeError("Terminate agents before loading the hidden verifier")
        self._phase = "verifier"
        try:
            result = self._long_command(command, env=None, timeout_seconds=timeout_seconds,
                                        stdout_path=stdout_path, stderr_path=stderr_path)
        except BaseException:
            try:
                self._docker(["stop", "-t", "0", self.container_id], timeout=30)
            except Exception as exc:
                self.diagnostics.append({"phase": "verifier_quiesce", "outcome": "cleanup_exception",
                                         "error": exception_diagnostic(exc, self.diagnostic_secrets)})
            raise
        else:
            # Prevent verifier children leaking into subsequent work. docker cp still works.
            self._docker(["stop", "-t", "0", self.container_id], timeout=30)
            return result

    def _remove_container(self, container_id: str) -> None:
        """Remove one container, waiting out a busy daemon.

        Right after a large task container is removed, the daemon can still be deleting its
        filesystem, and the next removal waits on it: a 30 s limit failed twice on 2026-10-19
        (batch 2 and the M1 rerun) where the same removal later took 0.27 s. A timed-out removal
        is retried once; it may have finished meanwhile, so a container that no longer exists
        counts as removed.
        """
        for attempt, timeout in enumerate(CONTAINER_REMOVE_TIMEOUTS):
            try:
                self._docker(["rm", "-f", container_id], timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                if attempt + 1 == len(CONTAINER_REMOVE_TIMEOUTS):
                    raise
            except subprocess.CalledProcessError:
                if attempt == 0 or self._container_exists(container_id):
                    raise
                return

    def _container_exists(self, container_id: str) -> bool:
        try:
            self._docker(["container", "inspect", "--format", "{{.Id}}", container_id], timeout=60)
        except subprocess.CalledProcessError as exc:
            if "no such" in f"{exc.stdout or ''}{exc.stderr or ''}".lower():
                return False
            raise
        return True

    def stop(self) -> None:
        if self.container_id:
            self._remove_container(self.container_id)
            self.container_id = None
        # Keep namespace ownership if task cleanup fails: do not accidentally
        # leave a running task behind while throwing away its network identity.
        if self._network_container_id:
            self._remove_container(self._network_container_id)
            self._network_container_id = None
        self._network_enforced = False

    def __enter__(self) -> "DockerWorkspace":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
