"""Workspace adapters and dynamic identities for native Qwen teams.

The registry prepares caller-owned Git worktrees. Qwen still owns agent sessions,
the task board, message delivery and shutdown. Cwd pinning is not an OS sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Protocol


def validate_worker_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("Worker names must start with a letter and contain only letters, digits, _ or - (max 64 characters)")
    if name.lower() in {"master", "leader", "all", "broadcast"}:
        raise ValueError("Reserved worker name")
    return name


class Workspace(Protocol):
    """Paths and commands are interpreted in the Qwen execution environment.

    stdout_path/stderr_path are host artifact paths. A Docker adapter must collect
    its command output there and stop the container-side command on timeout, not
    merely terminate the local docker client. No provider credentials enter env.
    """
    repo_path: str
    worker_root: str
    runtime_root: str
    relay_bind_host: str
    relay_connect_host: str

    def ensure_worker(self, name: str) -> str: ...
    def read_head(self, path: str) -> str | None: ...
    def write_file(self, path: str, contents: str) -> None: ...
    def run_agent(self, command: list[str], env: dict[str, str], timeout_seconds: float,
            stdout_path: Path, stderr_path: Path) -> dict: ...


@dataclass
class LocalWorkspace:
    repo_path: str
    worker_root: str
    runtime_root: str
    relay_bind_host: str = "127.0.0.1"
    relay_connect_host: str = "127.0.0.1"

    def __post_init__(self):
        self.repo_path = str(Path(self.repo_path).resolve())
        self.worker_root = str(Path(self.worker_root).resolve())
        self.runtime_root = str(Path(self.runtime_root).resolve())
        self._lock = threading.RLock()

    def ensure_worker(self, name: str) -> str:
        validate_worker_name(name)
        path = Path(self.worker_root) / name
        with self._lock:
            if path.exists():
                raise ValueError(f"Worker path already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Detached linked worktrees avoid branch-name collisions across runs;
            # workers may commit and explicitly hand those commits to integration.
            subprocess.run(["git", "-C", self.repo_path, "worktree", "add", "--detach", str(path), "HEAD"],
                           check=True, text=True, capture_output=True)
        return str(path)

    def read_head(self, path: str) -> str | None:
        result = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                                check=False, text=True, capture_output=True, timeout=5)
        return result.stdout.strip() or None if result.returncode == 0 else None

    def write_file(self, path: str, contents: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)

    def write_file_atomic(self, path: str, contents: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(contents)
        os.replace(temporary, target)

    def run_agent(self, command, env, timeout_seconds, stdout_path, stderr_path):
        started = time.monotonic()
        timed_out = False
        with Path(stdout_path).open("w") as out, Path(stderr_path).open("w") as err:
            proc = subprocess.Popen(command, cwd=self.repo_path, env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    code = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = proc.wait()
            finally:
                if proc.poll() is None:
                    # Cancellation must not leave a paid worker tree alive.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
        return {"returncode": code, "timed_out": timed_out,
                "elapsed_seconds": round(time.monotonic() - started, 3)}


class WorkerRegistry:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._workers: dict[str, str] = {}
        self._creation_heads: dict[str, str | None] = {}
        self._lock = threading.RLock()

    def register(self, name: str) -> str:
        validate_worker_name(name)
        with self._lock:
            if name not in self._workers:
                self._workers[name] = self.workspace.ensure_worker(name)
                # The native worker has not received its spawn call yet. Record
                # the worktree's actual starting commit now, not after it runs.
                try:
                    self._creation_heads[name] = self.workspace.read_head(self._workers[name])
                except Exception:
                    # Observation failures must neither prevent work nor be
                    # repaired later with a possibly already-modified HEAD.
                    self._creation_heads[name] = None
            return self._workers[name]

    def contains(self, name: str) -> bool:
        with self._lock:
            return name in self._workers

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._workers)

    def metadata(self) -> dict[str, dict]:
        with self._lock:
            return {name: {"working_dir": path, "creation_head": self._creation_heads.get(name)}
                    for name, path in self._workers.items()}
