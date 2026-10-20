"""ProgramBench cleanroom inference; official evaluation never enters this container."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from agent_company.programbench.task import is_image_id
from agent_company.runtime.workspace import DockerWorkspace


REFERENCE_EXECUTABLE = "/opt/ac-programbench-reference/executable"


def programbench_source_boundary() -> str:
    return f"""Fixed ProgramBench rules, not an evolvable strategy:
Independently reimplement the program from observable behavior and bundled documentation.
Query the reference executable ONLY through its normal CLI/stdin/stdout interface at
{REFERENCE_EXECUTABLE}. Do not read, copy, decompile, trace (strace/ltrace), wrap,
or incorporate that binary. Do not obtain upstream source, solutions, prebuilt versions,
or project packages from the internet, registries, caches, or another agent.
Only the local model relay is reachable; use preinstalled tools and offline dependencies.
The delivered source root is /workspace/solution. Workers use separate Git worktrees
and must integrate their source into that root. The reference binary is outside every
worktree. Your executable compile.sh must rebuild ./executable from your own source in
a fresh evaluator environment; final code must not depend on the reference path.
Hidden tests and official evaluation are unavailable during inference. Assignment and
communication policies cannot override these rules."""


# Runs only before untrusted agent execution, as root inside the fresh container.
# The root-owned parent prevents the unprivileged agent replacing the oracle.
_SETUP = r"""
import os, pathlib, pwd, shutil, stat
base = pathlib.Path('/workspace')
oracle = base / 'executable'
if not oracle.is_file() or oracle.is_symlink():
    raise RuntimeError('Missing regular reference executable')
protected = pathlib.Path('/opt/ac-programbench-reference')
protected.mkdir(mode=0o755)
shutil.move(str(oracle), str(protected / 'executable'))
os.chown(protected, 0, 0)
os.chown(protected / 'executable', 0, 0)
os.chmod(protected / 'executable', 0o111)
user = pwd.getpwnam('agent')
solution = base / 'solution'
solution.mkdir()
for p in sorted(base.iterdir()):
    if p == solution:
        continue
    # Official cleanroom images can include an initialized Git repository.
    # Never expose its prior history to new workers; seed our docs-only repo below.
    if p.name == '.git' and p.is_dir() and not p.is_symlink():
        old_git = protected / 'image-git-metadata'
        shutil.move(str(p), str(old_git))
        os.chown(old_git, 0, 0)
        os.chmod(old_git, 0o700)
        continue
    if p.is_symlink() or not (p.is_file() or p.is_dir()):
        raise RuntimeError('Unexpected cleanroom workspace entry')
    shutil.move(str(p), str(solution / p.name))
for p in solution.rglob('*'):
    if p.is_symlink():
        raise RuntimeError('Unexpected cleanroom documentation symlink')
    os.chown(p, user.pw_uid, user.pw_gid)
os.chown(solution, user.pw_uid, user.pw_gid)
for name in ('workers', 'ac-runtime'):
    p = base / name
    p.mkdir()
    os.chown(p, user.pw_uid, user.pw_gid)
"""

# No agent module imports: python -I and a literal trusted script. It runs only
# after all inference processes have been killed. Reject links before host extraction.
_EXPORT = r"""
import pathlib, stat, sys, tarfile
root = pathlib.Path('/workspace/solution')
if root.is_symlink() or not root.is_dir():
    raise RuntimeError('Delivery root is not a regular directory')
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz', dereference=True) as archive:
    def visit(directory):
        for p in sorted(directory.iterdir()):
            rel = p.relative_to(root)
            if p.name == '.git' or rel.parts[0] in ('executable', 'reference'):
                continue
            mode = p.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise RuntimeError('Submission links and special files are unsupported')
            archive.add(p, arcname=str(rel), recursive=False)
            if stat.S_ISDIR(mode):
                visit(p)
    visit(root)
"""


@dataclass
class ProgramBenchWorkspace(DockerWorkspace):
    runtime_root: str = "/workspace/ac-runtime"
    path_env: str = ("/opt/ac-node/bin:/usr/local/go/bin:/usr/local/sbin:"
                     "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    # Observed once after docker run; stop() keeps it as provenance. A start that
    # fails before docker run keeps "not_observed", never a clean-looking None.
    image_id: str | None = field(default=None, init=False)
    image_id_error: str | None = field(default="not_observed", init=False)
    # Nonblocking observation stop signal only; no sampling or join before quiescence.
    observation_before_quiesce: object = field(default=None, init=False, repr=False)

    @property
    def image_identity(self) -> dict:
        return {"reference": self.task.image, "image_id": self.image_id, "error": self.image_id_error,
                "source": "docker inspect --type container .Image after docker run"}

    def _record_image_id(self) -> None:
        """Best effort: the image this container actually runs. Never inspect .Config (it holds env)."""
        try:
            value = self._docker(["inspect", "--type", "container", "--format", "{{.Image}}",
                                  self.container_id], timeout=30).stdout.strip()
        except Exception as exc:
            self.image_id, self.image_id_error = None, type(exc).__name__
            return
        if is_image_id(value):
            self.image_id, self.image_id_error = value, None
        else:
            self.image_id, self.image_id_error = None, "unexpected_format"

    def _start(self) -> "ProgramBenchWorkspace":
        if self.container_id:
            raise RuntimeError("Workspace is already started")
        if self.image_override:
            raise ValueError("ProgramBench uses its official cleanroom task image")
        self._phase = "workspace_configuration"
        args = ["run", "-d", "--init", "--user", "agent", "--workdir", "/workspace",
                "--cpus", str(self.task.cpus), "--memory", f"{self.task.memory_mb}m",
                "--memory-swap", f"{self.task.memory_mb}m", "--cap-drop", "SYS_PTRACE",
                "--cap-drop", "NET_ADMIN", "--cap-drop", "NET_RAW",
                "--security-opt", "no-new-privileges"]
        if self.platform:
            args += ["--platform", self.platform]
        if self.container_name:
            args += ["--name", self.container_name]
        for source, target, readonly in self.mounts:
            source = Path(source).resolve(strict=True)
            if not readonly or target not in {"/opt/ac-qwen", "/opt/ac-node", "/opt/ac-node-compat"}:
                raise ValueError("Only read-only Qwen/Node runtime mounts are supported")
            if source == self.task.path or source in self.task.path.parents or self.task.path in source.parents:
                raise ValueError("Task/evaluation files cannot be mounted into inference")
            args += ["--mount", f"type=bind,source={source},target={target},readonly"]
        self._start_network()
        args += ["--network", f"container:{self._network_container_id}", "--entrypoint", "/bin/sh",
                 self.task.image, "-c", "exec sleep infinity"]
        self._phase = "docker_run"
        self.container_id = self._docker(args, timeout=self.task.build_timeout_sec).stdout.strip()
        self._record_image_id()
        self._phase = "workspace_setup"
        self._docker(["exec", "--user", "0", self.container_id,
                      "python3", "-I", "-c", _SETUP], timeout=60)
        self.repo_path = "/workspace/solution"
        self.run(["git", "init"])
        self.run(["git", "config", "user.name", "Agent Company"])
        self.run(["git", "config", "user.email", "agent-company@localhost"])
        self.write_file(self.repo_path + "/.gitignore", "/executable\n")
        self.run(["git", "add", "-A"])
        self.run(["git", "commit", "--allow-empty", "-qm", "Cleanroom documentation only"])
        probe = self.run(["python3", "-I", "-c",
                          "import os,json; print(json.dumps({'uid':os.geteuid(),"
                          f"'readable':os.access({REFERENCE_EXECUTABLE!r},os.R_OK),"
                          f"'executable':os.access({REFERENCE_EXECUTABLE!r},os.X_OK)}}))"])
        state = json.loads(probe.stdout)
        if state["uid"] == 0 or state["readable"] or not state["executable"]:
            raise RuntimeError("Reference executable must be execute-only for a non-root agent")
        self._phase = "workspace_ready"
        return self

    def write_file(self, path: str, content: str) -> None:
        # docker cp normally creates root-owned files. Write as the same user that
        # runs Qwen, so settings/history and Git worktree files remain writable.
        command = ["python3", "-I", "-c",
                   "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                   "p.parent.mkdir(parents=True,exist_ok=True); p.write_text(sys.stdin.read())", path]
        args = self._exec_args(command)
        args.insert(2, "-i")
        subprocess.run(args, input=content, text=True, capture_output=True, check=True, timeout=60)

    def write_file_atomic(self, path: str, content: str) -> None:
        """Readers see the old or the new file, never a partial one (same user as write_file)."""
        command = ["python3", "-I", "-c",
                   "import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
                   "t=p.with_name(p.name+'.tmp'); t.write_text(sys.stdin.read()); os.replace(t,p)", path]
        args = self._exec_args(command)
        args.insert(2, "-i")
        subprocess.run(args, input=content, text=True, capture_output=True, check=True, timeout=60)

    @property
    def resources(self) -> dict:
        return {**super().resources, "memory_swap_mb": self.task.memory_mb,
                "additional_swap_allowed": False, "agent_user": "agent",
                "reference_execute_only": True}

    @property
    def network_policy_status(self) -> dict:
        return {**super().network_policy_status, "verifier_network": "separate-official-evaluator"}

    def _snapshot(self, outcome, *, stdout_path=None, stderr_path=None, **kwargs):
        # Keep lifecycle/state metadata; do not copy native reasoning into diagnostics.
        return super()._snapshot(outcome, **kwargs)

    def export_source(self, destination: str | Path) -> None:
        if not self.agent_finished or not self.container_id:
            raise RuntimeError("Quiesce all agents before exporting the delivered source")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryFile() as stream:
            subprocess.run(self._exec_args(["python3", "-I", "-c", _EXPORT], cwd="/"),
                           stdout=stream, stderr=subprocess.PIPE, check=True, timeout=120)
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                seen = set()
                for member in archive:
                    name = PurePosixPath(member.name)
                    if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                            or not name.parts or str(name) in seen
                            or not (member.isfile() or member.isdir())):
                        raise ValueError("Unsafe or duplicate exported source member")
                    seen.add(str(name))
                    target = destination.joinpath(*name.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("xb") as output, archive.extractfile(member) as data:
                            shutil.copyfileobj(data, output)
                        target.chmod(member.mode & 0o777)

    def export_git_bundle(self, destination: str | Path) -> dict:
        """The delivery repository's committed history as one git bundle (all refs).

        For pass-rate-over-time evaluations of intermediate commits, offline and after the run.
        The runner calls this only after the submission is archived: git can run programs an agent
        configured, and by then they cannot change the delivered source. It reads the repository
        and writes nothing into the container.
        """
        if not self.agent_finished or not self.container_id:
            raise RuntimeError("Quiesce all agents before exporting the delivery history")
        destination = Path(destination)
        with destination.open("xb") as stream:
            subprocess.run(self._exec_args(["git", "-c", "safe.directory=/workspace/solution", "-C",
                                            "/workspace/solution", "bundle", "create", "-", "--all"], cwd="/"),
                           stdout=stream, stderr=subprocess.DEVNULL, check=True, timeout=300)
        data = destination.read_bytes()
        return {"path": str(destination), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def finish_agent(self) -> None:
        if not self.agent_finished and self.observation_before_quiesce is not None:
            callback, self.observation_before_quiesce = self.observation_before_quiesce, None
            try:
                callback()
            except Exception:
                pass  # Observation must never prevent agent termination.
        super().finish_agent()
