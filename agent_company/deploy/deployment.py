"""One committed ProgramBench deployment and read-only checks of its runtime assets.

This module never starts a rollout, loads provider credentials, builds images, or
pulls images. A failed installation leaves its directory occupied for inspection.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from agent_company.runtime_io import write_json_atomic

from agent_company.deploy.code_version import code_version
from agent_company.evolution.harness_capabilities import native_version
from agent_company.deploy.ssh import clean_environment, ssh
from agent_company.runtime.network import NETWORK_HELPER_IMAGE
from agent_company.programbench.task import CLEANROOM_TAG, ProgramBenchTask, is_image_id

SNAPSHOT_PATHS = ("agent_company", "configs/stage1", "scripts", "tests", "docs", "pyproject.toml")
CLEAN_PATHS = ("agent_company", "configs/stage1", "scripts", "tests", "pyproject.toml")
ARTIFACT_SOURCE = Path("runs/stage1_kimi_programbench_ffmpeg_p1_rerun2_20261003")
IDENTITY_FIELDS = ("manifest_sha256", "code_version", "qwen_version", "node_version", "task_sha256",
                   "evaluator_sha256", "images")


@dataclass(frozen=True)
class Deployment:
    final: Path
    assets: Path
    task_id: str = "jgm__pandoc.5caad90"

    def __post_init__(self):
        object.__setattr__(self, "final", Path(self.final))
        object.__setattr__(self, "assets", Path(self.assets))
        if not self.final.is_absolute() or not self.assets.is_absolute():
            raise ValueError("Deployment and asset directories must be absolute")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", self.task_id):
            raise ValueError("Invalid task identifier")

    @property
    def code(self):
        return self.final / "code"

    @property
    def python(self):
        return self.assets / ".venv/bin/python"

    @property
    def qwen(self):
        return self.assets / "external/qwen-study/package"

    @property
    def node(self):
        return self.assets / "external/stage1-runtime/node-v22.23.2-linux-x64"

    @property
    def docker(self):
        return self.assets / "tools/docker"

    @property
    def evaluator(self):
        return self.final / "evaluator_fix"

    @property
    def task(self):
        return self.assets / "external/programbench-study/src/programbench/data/tasks" / self.task_id


def snapshot(root: Path, commit: str) -> tuple[bytes, dict]:
    """Archive exactly the requested HEAD; docs may be dirty, but no overlays exist."""
    def git(*arguments):
        return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                              capture_output=True, env=clean_environment()).stdout
    head = git("rev-parse", "HEAD").decode().strip()
    if head != commit:
        raise ValueError("HEAD must equal the explicit full commit SHA")
    if git("diff", "--name-only", "-z", "HEAD", "--", *CLEAN_PATHS):
        raise ValueError("Tracked deployment source has uncommitted changes")
    archive = git("archive", "--format=tar", commit, "--", *SNAPSHOT_PATHS)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if member.isfile():
                files[member.name] = hashlib.sha256(source.extractfile(member).read()).hexdigest()
    manifest = "".join(f"{digest}  ./{name}\n" for name, digest in sorted(files.items()))
    return archive, {"commit": head, "files": files, "archive_scope": list(SNAPSHOT_PATHS),
                     "expected_manifest_text": manifest,
                     "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}


def _run(command, *, deployment: Deployment | None = None, timeout=60):
    environment = {**clean_environment(), "PYTHONDONTWRITEBYTECODE": "1"}
    if deployment is not None:
        environment["PYTHONPATH"] = os.pathsep.join((str(deployment.evaluator), str(deployment.code)))
    return subprocess.run([str(part) for part in command], check=True, capture_output=True,
                          text=True, timeout=timeout, env=environment,
                          cwd=deployment.code if deployment is not None else None).stdout


def _tree_sha256(root: Path) -> str:
    """Digest the installed task's file bytes and names, including symbolic-link targets."""
    digest = hashlib.sha256()
    if not root.is_dir():
        raise FileNotFoundError("Task directory missing")
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted([*dirs, *files]):
            path = Path(directory) / name
            if path.is_symlink():
                value = b"link\0" + os.readlink(path).encode()
            elif path.is_file():
                value = b"file\0" + hashlib.sha256(path.read_bytes()).digest()
            else:
                continue
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + value + b"\0")
        dirs.sort()
    return digest.hexdigest()


def required_images(deployment: Deployment) -> list[str]:
    """Official task/evaluator image identities must remain fixed for the series."""
    task = ProgramBenchTask.load(deployment.task,
                                 instruction_path=deployment.code / "configs/stage1/programbench_instruction.md")
    evaluator_image = (f"{os.environ.get('PROGRAMBENCH_DOCKER_ORG', 'programbench')}/"
                       f"{task.task_id.replace('__', '_1776_')}:{CLEANROOM_TAG}")
    return sorted(set([task.image, evaluator_image]))


def runtime_built_images(deployment: Deployment) -> list[str]:
    """Record helper/base IDs, allowing the runtime to rebuild the committed helper."""
    dockerfile = deployment.code / "agent_company/runtime/network_helper.Dockerfile"
    bases = re.findall(r"^FROM\s+(\S+)", dockerfile.read_text(), flags=re.MULTILINE | re.IGNORECASE)
    return sorted(set([NETWORK_HELPER_IMAGE, *bases]))


def server_identity(deployment: Deployment) -> dict:
    """Measure runtime identity with the same source/native readers as the runners."""
    manifest = deployment.final / "code_sha256.txt"
    version = code_version(deployment.code)
    qwen = native_version(deployment.qwen)
    if not isinstance(version, dict) or version.get("kind") != "deployment_manifest":
        raise ValueError("Runner code_version is unknown or is not deployment_manifest")
    if not isinstance(qwen, str) or not qwen.strip() or qwen.strip().lower() == "unknown":
        raise ValueError("Qwen native version is unknown")
    # Docker itself must answer; inspection never pulls missing images.
    docker_version = _run([deployment.docker, "info", "--format", "{{.ServerVersion}}"], deployment=deployment).strip()
    if not docker_version:
        raise ValueError("Docker did not report a server version")
    image_groups = {}
    for name, tags in (("images", required_images(deployment)),
                       ("runtime_built_images", runtime_built_images(deployment))):
        image_groups[name] = {}
        for tag in tags:
            value = _run([deployment.docker, "inspect", "--type", "image", "--format", "{{.Id}}", tag],
                         deployment=deployment).strip()
            if not is_image_id(value):
                raise ValueError(f"Required local image is missing or unknown: {tag}")
            image_groups[name][tag] = value
    node_version = _run([deployment.node / "bin/node", "--version"], deployment=deployment).strip()
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", node_version):
        raise ValueError("Node version is unknown")
    return {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "code_version": version,
            "qwen_version": qwen, "node_version": node_version, "task_sha256": _tree_sha256(deployment.task),
            "evaluator_sha256": _tree_sha256(deployment.evaluator),
            **image_groups, "docker_server_version": docker_version}


def capacity(deployment: Deployment) -> dict:
    """Record all running containers without attribution, mounts, or Config.Env."""
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemTotal", "MemAvailable"}:
            memory[name] = int(value.split()[0])
    if set(memory) != {"MemTotal", "MemAvailable"}:
        raise ValueError("Host memory capacity is unknown")
    template = ('{"name":{{json .Name}},"nano_cpus":{{json .HostConfig.NanoCpus}},'
                '"cpu_quota":{{json .HostConfig.CpuQuota}},"cpu_period":{{json .HostConfig.CpuPeriod}},'
                '"cpuset_cpus":{{json .HostConfig.CpusetCpus}},"memory_bytes":{{json .HostConfig.Memory}},'
                '"memory_swap_bytes":{{json .HostConfig.MemorySwap}}}')
    ids = _run([deployment.docker, "ps", "-q"], deployment=deployment).split()
    running = []
    exited = []
    for identifier in ids:
        try:
            raw = _run([deployment.docker, "inspect", "--type", "container", "--format", template, identifier],
                       deployment=deployment)
        except (OSError, subprocess.SubprocessError):
            # A container can exit and be removed after ps but before inspect.
            current = _run([deployment.docker, "ps", "-q"], deployment=deployment).split()
            if identifier in current:
                raise
            exited.append(identifier)
            continue
        running.append(json.loads(raw))
    return {"nproc": int(_run(["nproc"], deployment=deployment).strip()), "mem_total_kb": memory["MemTotal"],
            "mem_available_kb": memory["MemAvailable"], "loadavg": Path("/proc/loadavg").read_text().strip(),
            "running_containers": running, "exited_during_inspection": exited,
            "container_ownership_scope": "all_running_containers_without_ownership_attribution"}


def _record(path: Path, value: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def record_installed(deployment: Deployment, metadata: dict) -> dict:
    """Finish the install only after byte-for-byte local/remote manifest comparison."""
    actual = (deployment.final / "code_sha256.txt").read_text()
    if actual != metadata["expected_manifest_text"]:
        _record(deployment.final / "INVALID_DEPLOYMENT.json", {"reason": "code_manifest_mismatch"})
        raise ValueError("Deployed source manifest differs; this directory must not be reused")
    identity = server_identity(deployment)
    if identity["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Deployment manifest digest differs")
    value = {"status": "deployed", "commit": metadata["commit"], "final": str(deployment.final),
             "assets": str(deployment.assets), "task_id": deployment.task_id,
             "archive_scope": metadata["archive_scope"], **identity,
             "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _record(deployment.final / "DEPLOYMENT.json", value)
    return value


def deploy(root: Path, deployment: Deployment, commit: str, *, local_record_dir: Path, transport=ssh) -> dict:
    archive, metadata = snapshot(root, commit)
    local_record_dir = Path(local_record_dir)
    if (local_record_dir / "DEPLOYMENT.json").exists():
        raise FileExistsError("Local deployment record already exists")
    final = shlex.quote(str(deployment.final))
    code = shlex.quote(str(deployment.code))
    source = deployment.assets / ARTIFACT_SOURCE
    def q(path):
        return shlex.quote(str(path))
    finish = ("import json; from pathlib import Path; "
              "from agent_company.programbench.evaluator_compat import install; "
              "from agent_company.deploy.deployment import Deployment, record_installed; "
              f"install(Path({str(deployment.evaluator)!r})); "
              f"print(json.dumps(record_installed(Deployment(Path({str(deployment.final)!r}), "
              f"Path({str(deployment.assets)!r}), {deployment.task_id!r}), "
              f"json.loads(Path({str(deployment.final / 'validation/local_candidate.json')!r}).read_text()))))")
    script = f'''set -eu
test ! -e {final} && test ! -L {final}
test -f {q(source / 'evaluator_fix/programbench/eval/eval.py')}
test -f {q(source / 'validation/offline_test_dependencies.zip')}
test -d {q(deployment.task)}
mkdir {final}
mkdir {code} {q(deployment.final / 'validation')} {q(deployment.final / 'paid_runs')} {q(deployment.final / 'profiles')} {q(deployment.final / 'preparations')}
IFS= read -r ARCHIVE
printf '%s' "$ARCHIVE" | base64 -d | tar -x -C {code}
cat > {q(deployment.final / 'validation/local_candidate.json')}
ln -s {q(deployment.assets / 'external')} {q(deployment.code / 'external')}
cp -a {q(source / 'evaluator_fix')} {q(deployment.evaluator)}
cp -a {q(source / 'validation/offline_test_dependencies.zip')} {q(deployment.final / 'validation/')}
(cd {code} && LC_ALL=C find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > {q(deployment.final / 'code_sha256.txt')}
cd {code}
env -u DEEPSEEK_API_KEY -u STAGE1_GLM_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={code} {q(deployment.python)} -c {q(finish)}
'''
    try:
        result = transport("bash -o pipefail -c " + shlex.quote(script),
                           stdin=base64.b64encode(archive).decode() + "\n" + json.dumps(metadata), timeout=300)
        if result.returncode:
            detail = result.stderr.strip()[-800:] or "(no stderr)"
            raise RuntimeError(f"Deployment command failed (exit {result.returncode}); inspect the directory "
                               f"before any further action. Remote stderr (tail):\n{detail}")
        record = json.loads(result.stdout)
        if (record.get("manifest_sha256") != metadata["manifest_sha256"] or record.get("commit") != commit
                or record.get("code_version") != {"kind": "deployment_manifest", "files": dict(sorted(metadata["files"].items()))}):
            raise ValueError("Returned deployment identity differs from the local archive")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        raise RuntimeError("Deployment result unclear; inspect before any further action") from None
    local_record_dir.mkdir(parents=True, exist_ok=True)
    _record(local_record_dir / "DEPLOYMENT.json", record)
    (local_record_dir / "code_sha256.txt").write_text(metadata["expected_manifest_text"], encoding="utf-8")
    return record


def _same_identity(expected: dict, actual: dict):
    if any(expected.get(key) is None or actual.get(key) != expected[key] for key in IDENTITY_FIELDS):
        raise ValueError("Runtime identity no longer matches this deployment")


def verify_current(deployment: Deployment) -> dict:
    """Read-only launch-time recheck; does not create or refresh VERIFY.json."""
    installed = json.loads((deployment.final / "DEPLOYMENT.json").read_text())
    verified = json.loads((deployment.final / "VERIFY.json").read_text())
    if installed.get("status") != "deployed" or verified.get("status") != "verified":
        raise ValueError("Deployment has no successful saved verification")
    current = server_identity(deployment)
    _same_identity(installed, current)
    _same_identity(verified["identity"], current)
    return current


def remote_verify(deployment: Deployment) -> dict:
    installed = json.loads((deployment.final / "DEPLOYMENT.json").read_text())
    current = server_identity(deployment)
    _same_identity(installed, current)
    expression = ("import importlib.util,json; "
                  "s=importlib.util.find_spec('programbench.eval.eval_batch'); "
                  "print(json.dumps({'origin':s.origin if s else None}))")
    found = json.loads(_run([deployment.python, "-c", expression], deployment=deployment))
    origin = found.get("origin")
    if not isinstance(origin, str) or not Path(origin).resolve().is_relative_to(deployment.evaluator.resolve()):
        raise ValueError("Official evaluator does not resolve inside evaluator_fix")
    report = {"status": "verified", "identity": current, "evaluator_origin": origin,
              "capacity": capacity(deployment), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(deployment.final / "VERIFY.json", report)
    return report
