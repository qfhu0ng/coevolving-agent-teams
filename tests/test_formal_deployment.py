"""Local committed fixtures and fake SSH/Docker; no deployment or provider calls."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from agent_company.deploy import deployment as module
from agent_company.deploy import ssh as launch_transport
from agent_company.deploy.code_version import code_version
from agent_company.evolution.loop import TaskResult
from agent_company.runtime.relay import ProviderConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    for name in ("agent_company/module.py", "configs/stage1/profile.json", "scripts/run.py",
                 "tests/test_example.py", "docs/note.md", "pyproject.toml"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed\n")
    git("add", "agent_company", "configs", "scripts", "tests", "docs", "pyproject.toml")
    git("commit", "-qm", "synthetic source")
    return root, git("rev-parse", "HEAD")


def test_snapshot_is_unmodified_commit_archive_and_ignores_dirty_docs(repository):
    root, commit = repository
    (root / "docs/note.md").write_text("uncommitted user literature change\n")
    (root / "agent_company/untracked.py").write_text("not committed\n")
    data, metadata = module.snapshot(root, commit)
    expected = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit,
                               "--", *module.SNAPSHOT_PATHS], check=True, capture_output=True).stdout
    assert data == expected
    with tarfile_open(data) as archive:
        assert archive.extractfile("docs/note.md").read() == b"committed\n"
        assert "agent_company/untracked.py" not in archive.getnames()
    assert metadata["expected_manifest_text"] == "".join(
        f"{digest}  ./{name}\n" for name, digest in sorted(metadata["files"].items()))


def tarfile_open(data):
    import tarfile
    return tarfile.open(fileobj=io.BytesIO(data))


@pytest.mark.parametrize("path", ["agent_company/module.py", "configs/stage1/profile.json", "scripts/run.py",
                                 "tests/test_example.py", "pyproject.toml"])
def test_snapshot_rejects_each_tracked_runtime_change(repository, path):
    root, commit = repository
    (root / path).write_text("changed\n")
    with pytest.raises(ValueError, match="uncommitted"):
        module.snapshot(root, commit)


def test_snapshot_requires_exact_head(repository):
    root, _ = repository
    with pytest.raises(ValueError, match="HEAD"):
        module.snapshot(root, "0" * 40)


def test_transport_removes_credentials_and_preserves_explicit_ssh_parameters(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "SENTINEL_DS")
    monkeypatch.setenv("STAGE1_GLM_API_KEY", "SENTINEL_GLM")
    def run(argv, **kwargs):
        assert "SENTINEL" not in json.dumps(argv)
        assert not any(name in kwargs["env"] for name in launch_transport.CREDENTIAL_ENV_NAMES)
        assert kwargs["input"] == '{"api_key":"SENTINEL_STDIN"}'
        assert argv[-2:] == ["synthetic-host", "public command"]
        assert "UserKnownHostsFile=/synthetic-known" in argv
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    launch_transport.ssh("public command", stdin='{"api_key":"SENTINEL_STDIN"}',
                         host="synthetic-host", key="/synthetic-key", known="/synthetic-known")


def test_credential_reader_is_lazy_and_does_not_echo_parse_failure():
    called = []
    def reader():
        called.append(True)
        return {"api_key": "SENTINEL_DS"}
    assert not called
    assert launch_transport.credential(reader) == "SENTINEL_DS" and called == [True]
    def invalid():
        raise ValueError("PRIVATE_SENTINEL")
    with pytest.raises(ValueError, match="Cannot read") as error:
        launch_transport.credential(invalid)
    assert "PRIVATE_SENTINEL" not in str(error.value)


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    deployment = module.Deployment(tmp_path / "final", tmp_path / "assets", "synthetic__task.abcdef0")
    deployment.code.mkdir(parents=True)
    for name, text in {"agent_company/module.py": "source\n",
                       "agent_company/runtime/network_helper.Dockerfile": "FROM python:3.12-slim\n",
                       "configs/stage1/programbench_instruction.md": "Synthetic task instructions."}.items():
        path = deployment.code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    deployed_script = deployment.code / "scripts/run_programbench_policy_comparison.py"
    deployed_script.parent.mkdir()
    shutil.copyfile(ROOT / "scripts/run_programbench_policy_comparison.py", deployed_script)
    deployment.task.mkdir(parents=True)
    (deployment.task / "task.yaml").write_text("repository: synthetic/task\ncommit: 'abcdef0" + "1" * 33 + "'\n")
    (deployment.task / "tests.json").write_text("opaque synthetic task data")
    deployment.qwen.mkdir(parents=True)
    (deployment.qwen / "package.json").write_text('{"version":"0.23.1"}')
    deployment.evaluator.mkdir()
    files = {p.relative_to(deployment.code).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in deployment.code.rglob("*") if p.is_file()}
    manifest = "".join(f"{digest}  ./{name}\n" for name, digest in sorted(files.items()))
    (deployment.final / "code_sha256.txt").write_text(manifest)
    metadata = {"commit": "a" * 40, "expected_manifest_text": manifest, "archive_scope": list(module.SNAPSHOT_PATHS),
                "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}
    calls = []
    def command(argv, **kwargs):
        calls.append((argv, kwargs))
        assert not any(key in kwargs["env"] for key in launch_transport.CREDENTIAL_ENV_NAMES)
        assert kwargs["env"]["PYTHONPATH"] == str(deployment.evaluator) + ":" + str(deployment.code)
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert kwargs["cwd"] == deployment.code
        if argv[0] == str(deployment.docker):
            if argv[1] == "info":
                return SimpleNamespace(stdout="28.0.0\n")
            assert argv[1:6] == ["inspect", "--type", "image", "--format", "{{.Id}}"]
            return SimpleNamespace(stdout="sha256:" + "d" * 64 + "\n")
        if argv[0] == str(deployment.node / "bin/node"):
            assert argv[1:] == ["--version"]
            return SimpleNamespace(stdout="v22.23.2\n")
        if argv[0] == str(deployment.python):
            assert "find_spec('programbench.eval.eval_batch')" in argv[2]
            assert "run_eval_batch" not in argv[2]
            return SimpleNamespace(stdout=json.dumps({"origin": str(deployment.evaluator / "programbench/eval/eval_batch.py")}))
        pytest.fail(f"Unexpected process: {argv}")
    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(module, "capacity", lambda _: {"synthetic": True})
    return deployment, metadata, calls


def test_verify_uses_actual_runtime_manifest_and_never_reads_a_provider(deployed, monkeypatch):
    deployment, metadata, calls = deployed
    monkeypatch.setenv("DEEPSEEK_API_KEY", "PRIVATE_SENTINEL")
    monkeypatch.setenv("STAGE1_GLM_API_KEY", "PRIVATE_SENTINEL")
    installed = module.record_installed(deployment, metadata)
    assert installed["code_version"] == code_version(deployment.code)
    verified = module.remote_verify(deployment)
    assert verified["identity"]["code_version"] == installed["code_version"]
    assert verified["identity"]["qwen_version"] == "0.23.1"
    assert set(verified["identity"]["images"]) == set(module.required_images(deployment))
    assert set(verified["identity"]["runtime_built_images"]) == {
        "python:3.12-slim", module.NETWORK_HELPER_IMAGE}
    assert not (set(verified["identity"]["images"]) & set(verified["identity"]["runtime_built_images"]))
    before = {p: p.read_bytes() for p in deployment.final.rglob("*.json")}
    assert module.verify_current(deployment) == verified["identity"]
    assert before == {p: p.read_bytes() for p in deployment.final.rglob("*.json")}
    assert module.remote_verify(deployment)["identity"] == verified["identity"]
    assert all("PRIVATE_SENTINEL" not in p.read_text() for p in deployment.final.rglob("*.json"))
    assert all("pull" not in argv and "build" not in argv for argv, _ in calls)


@pytest.mark.parametrize("kind", ["task", "evaluator", "helper", "helper_base"])
def test_current_identity_only_pins_official_images_but_records_helper_changes(deployed, monkeypatch, kind):
    deployment, metadata, _ = deployed
    # Default task/evaluator references coincide; separate the official evaluator
    # registry here to exercise both identity entries independently.
    monkeypatch.setenv("PROGRAMBENCH_DOCKER_ORG", "synthetic-evaluator")
    installed = module.record_installed(deployment, metadata)
    module.remote_verify(deployment)
    task = module.ProgramBenchTask.load(deployment.task,
        instruction_path=deployment.code / "configs/stage1/programbench_instruction.md")
    evaluator = next(tag for tag in installed["images"] if tag != task.image)
    tag = {"task": task.image, "evaluator": evaluator, "helper": module.NETWORK_HELPER_IMAGE,
           "helper_base": "python:3.12-slim"}[kind]
    original = subprocess.run
    def command(argv, **kwargs):
        if argv[:2] == [str(deployment.docker), "inspect"] and argv[-1] == tag:
            return SimpleNamespace(stdout="sha256:" + "e" * 64 + "\n")
        return original(argv, **kwargs)
    monkeypatch.setattr(subprocess, "run", command)
    if kind in {"task", "evaluator"}:
        with pytest.raises(ValueError, match="identity no longer matches"):
            module.verify_current(deployment)
    else:
        current = module.verify_current(deployment)
        assert current["images"] == installed["images"]
        assert current["runtime_built_images"][tag] == "sha256:" + "e" * 64
        assert current["runtime_built_images"][tag] != installed["runtime_built_images"][tag]


@pytest.mark.parametrize("tag", [module.NETWORK_HELPER_IMAGE, "python:3.12-slim"])
def test_current_identity_still_requires_helper_and_base_images_present(deployed, monkeypatch, tag):
    deployment, metadata, calls = deployed
    module.record_installed(deployment, metadata)
    module.remote_verify(deployment)
    original = subprocess.run
    def command(argv, **kwargs):
        if argv[:2] == [str(deployment.docker), "inspect"] and argv[-1] == tag:
            return SimpleNamespace(stdout="")
        return original(argv, **kwargs)
    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises(ValueError, match="Required local image is missing"):
        module.verify_current(deployment)
    assert all("pull" not in argv and "build" not in argv for argv, _ in calls)


@pytest.mark.parametrize("version", [None, "", " ", "unknown", "Unknown", " UNKNOWN ", 231, "0.23.1"])
def test_identity_requires_the_same_known_qwen_version_as_comparison(deployed, version):
    deployment, _, calls = deployed
    (deployment.qwen / "package.json").write_text(json.dumps({"version": version}))
    if version == "0.23.1":
        assert module.server_identity(deployment)["qwen_version"] == version
        assert calls
    else:
        with pytest.raises(ValueError, match="Qwen native version is unknown"):
            module.server_identity(deployment)
        assert calls == []


def test_runtime_run_settings_match_verified_manifest(deployed, monkeypatch):
    deployment, metadata, _ = deployed
    module.record_installed(deployment, metadata)
    verified = module.remote_verify(deployment)
    path = deployment.code / "scripts/run_programbench_policy_comparison.py"
    spec = importlib.util.spec_from_file_location("synthetic_deployed_comparison", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "load_provider", lambda _: ProviderConfig("fake", "http://unused"))
    monkeypatch.setattr(runner, "ProgramBenchRunner", lambda **_: lambda request:
                        TaskResult(status="completed", score=0, metrics={"inference_cleanup_confirmed": True}))
    profile = deployment.final / "profile.json"
    profile.write_text(json.dumps({"benchmark": "ProgramBench", "provider_config": "unused.json",
        "tasks": [{"task_id": deployment.task_id, "path": str(deployment.task),
                   "instructions": "configs/stage1/programbench_instruction.md"}],
        "arms": [{"name": "single", "execution_mode": "single", "policy_version": 0}],
        "inference": {"agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 61440,
                      "memory_swap_mb": 61440, "max_model_calls": None},
        "evaluation": {"docker_cpus": 10}, "parallelism": 1, "evolution_enabled": False}))
    output = deployment.final / "synthetic-run"
    runner.run_comparison(profile, deployment.code, output, qwen_package=deployment.qwen,
                          node_directory=deployment.node, docker="unused", repeat_index=1)
    settings = json.loads((output / "run_settings.json").read_text())
    assert settings["code_version"] == verified["identity"]["code_version"]
    assert settings["qwen_version"] == verified["identity"]["qwen_version"]


@pytest.mark.parametrize("change", ["source", "task", "qwen", "manifest", "evaluator", "missing_verify"])
def test_current_verification_rejects_changed_or_missing_identity(deployed, change):
    deployment, metadata, _ = deployed
    module.record_installed(deployment, metadata)
    module.remote_verify(deployment)
    if change == "source":
        (deployment.code / "agent_company/module.py").write_text("changed")
    elif change == "task":
        (deployment.task / "tests.json").write_text("changed")
    elif change == "qwen":
        (deployment.qwen / "package.json").write_text("{}")
    elif change == "manifest":
        path = deployment.final / "code_sha256.txt"
        path.write_text(path.read_text().replace("  ./", " ./"))
    elif change == "evaluator":
        (deployment.evaluator / "changed.py").write_text("changed evaluator")
    else:
        (deployment.final / "VERIFY.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.verify_current(deployment)


def test_manifest_comparison_is_exact_and_failed_directory_is_marked(deployed):
    deployment, metadata, _ = deployed
    changed = {**metadata, "expected_manifest_text": metadata["expected_manifest_text"].replace("  ./", " ./")}
    with pytest.raises(ValueError, match="must not be reused"):
        module.record_installed(deployment, changed)
    assert (deployment.final / "INVALID_DEPLOYMENT.json").is_file()
    assert not (deployment.final / "DEPLOYMENT.json").exists()


@pytest.mark.parametrize("failure", ["docker_unavailable", "missing_image", "wrong_evaluator"])
def test_verify_refuses_missing_runtime_dependencies(deployed, monkeypatch, failure):
    deployment, metadata, _ = deployed
    module.record_installed(deployment, metadata)
    original = subprocess.run
    def command(argv, **kwargs):
        if failure == "docker_unavailable" and argv[:2] == [str(deployment.docker), "info"]:
            raise subprocess.CalledProcessError(1, argv)
        if failure == "missing_image" and argv[:2] == [str(deployment.docker), "inspect"]:
            return SimpleNamespace(stdout="")
        if failure == "wrong_evaluator" and argv[0] == str(deployment.python):
            return SimpleNamespace(stdout='{"origin":"/unrelated/programbench/eval/eval_batch.py"}')
        return original(argv, **kwargs)
    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        module.remote_verify(deployment)
    assert not (deployment.final / "VERIFY.json").exists()


def capacity_environment(tmp_path, monkeypatch):
    deployment = module.Deployment(tmp_path / "final", tmp_path / "assets")
    deployment.code.mkdir(parents=True)
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemTotal: 300000000 kB\nMemAvailable: 200000000 kB\n"
        if str(path) == "/proc/loadavg":
            return "0.01 0.02 0.03 1/200 300\n"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    return deployment


def test_capacity_records_all_running_containers_without_mounts_or_ownership_claim(tmp_path, monkeypatch):
    deployment = capacity_environment(tmp_path, monkeypatch)
    # ProgramBenchWorkspace mounts shared Qwen/Node assets, not final/paid_runs.
    # The old mount-based ownership check misclassified this actual layout.
    fixtures = {
        "current_runner": {"Mounts": [{"Source": str(deployment.qwen), "Destination": "/opt/ac-qwen"},
                                      {"Source": str(deployment.node), "Destination": "/opt/ac-node"}]},
        "other": {"Mounts": [{"Source": "/private/other/project", "Destination": "/workspace"}]},
        "helper": {"Mounts": []},
    }
    def command(argv, **kwargs):
        assert all(token not in json.dumps(argv) for token in ("Config.Env", "Mounts", "mount_sources"))
        if argv == ["nproc"]:
            return SimpleNamespace(stdout="64\n")
        if argv[1:] == ["ps", "-q"]:
            return SimpleNamespace(stdout=" ".join(fixtures) + "\n")
        assert argv[1:5] == ["inspect", "--type", "container", "--format"]
        name = argv[-1]
        assert name in fixtures
        return SimpleNamespace(stdout=json.dumps({"name": name, "nano_cpus": 20000000000,
            "cpu_quota": 0, "cpu_period": 0, "cpuset_cpus": "", "memory_bytes": 60 * 1024**3,
            "memory_swap_bytes": 60 * 1024**3}))
    monkeypatch.setattr(subprocess, "run", command)
    report = module.capacity(deployment)
    assert report["mem_available_kb"] == 200000000 and report["nproc"] == 64
    assert [row["name"] for row in report["running_containers"]] == list(fixtures)
    assert report["running_containers"][0]["memory_bytes"] == 60 * 1024**3
    assert report["container_ownership_scope"] == "all_running_containers_without_ownership_attribution"
    assert report["exited_during_inspection"] == []
    assert "mount_sources" not in json.dumps(report)
    assert "/private/other/project" not in json.dumps(report)


@pytest.mark.parametrize("remaining", [False, True])
def test_capacity_rechecks_running_containers_after_inspect_failure(tmp_path, monkeypatch, remaining):
    deployment = capacity_environment(tmp_path, monkeypatch)
    scans = []
    def command(argv, **kwargs):
        if argv == ["nproc"]:
            return SimpleNamespace(stdout="64\n")
        if argv[1:] == ["ps", "-q"]:
            scans.append(True)
            return SimpleNamespace(stdout="vanishing live\n" if len(scans) == 1 or remaining else "live\n")
        assert argv[1:5] == ["inspect", "--type", "container", "--format"]
        if argv[-1] == "vanishing":
            raise subprocess.CalledProcessError(1, argv, stderr="No such object")
        assert argv[-1] == "live"
        return SimpleNamespace(stdout='{"name":"live","memory_bytes":1073741824}')
    monkeypatch.setattr(subprocess, "run", command)
    if remaining:
        with pytest.raises(subprocess.CalledProcessError):
            module.capacity(deployment)
    else:
        report = module.capacity(deployment)
        assert report["exited_during_inspection"] == ["vanishing"]
        assert [row["name"] for row in report["running_containers"]] == ["live"]
    assert len(scans) == 2


@pytest.mark.parametrize("failure", [None, "remote", "returned_identity"])
def test_deploy_fake_transport_sends_archive_via_stdin_and_records_exact_result(repository, tmp_path, failure):
    root, commit = repository
    deployment = module.Deployment(tmp_path / "server final", tmp_path / "assets")
    local = tmp_path / "evidence"
    seen = []
    def transport(command, **kwargs):
        seen.append(command)
        shell = shlex.split(command)
        assert shell[:4] == ["bash", "-o", "pipefail", "-c"]
        subprocess.run(["bash", "-n"], input=shell[4], capture_output=True, text=True, check=True)
        compile(shlex.split(shell[4].splitlines()[-1])[-1], "remote_finish", "exec")
        assert "find . -type f" in command and "sort -z" in command
        assert "test ! -e" in command and "mkdir " in command
        assert "env -u DEEPSEEK_API_KEY -u STAGE1_GLM_API_KEY PYTHONDONTWRITEBYTECODE=1" in command
        archive_line, meta = kwargs["stdin"].split("\n", 1)
        metadata = json.loads(meta)
        with tarfile_open(base64.b64decode(archive_line)) as archive:
            assert archive.extractfile("agent_company/module.py").read() == b"committed\n"
        record = {"commit": commit, "manifest_sha256": metadata["manifest_sha256"],
                  "code_version": {"kind": "deployment_manifest", "files": dict(sorted(metadata["files"].items()))}}
        if failure == "returned_identity":
            record["manifest_sha256"] = "wrong"
        return SimpleNamespace(returncode=int(failure == "remote"), stdout=json.dumps(record), stderr="")
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            module.deploy(root, deployment, commit, local_record_dir=local, transport=transport)
        assert not local.exists()
    else:
        record = module.deploy(root, deployment, commit, local_record_dir=local, transport=transport)
        assert json.loads((local / "DEPLOYMENT.json").read_text()) == record
        with pytest.raises(FileExistsError):
            module.deploy(root, deployment, commit, local_record_dir=local, transport=transport)
    assert len(seen) == 1


def test_deploy_remote_failure_reports_bounded_stderr_tail_without_claiming_unclear(repository, tmp_path):
    root, commit = repository
    deployment = module.Deployment(tmp_path / "final", tmp_path / "assets")
    local = tmp_path / "evidence"
    calls = []
    def transport(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=23, stdout="", stderr="OLD DETAIL\n" + "x" * 1500
                               + "\nModuleNotFoundError: No module named 'programbench'\n")
    with pytest.raises(RuntimeError) as error:
        module.deploy(root, deployment, commit, local_record_dir=local, transport=transport)
    assert "failed (exit 23)" in str(error.value)
    assert "ModuleNotFoundError: No module named 'programbench'" in str(error.value)
    assert "OLD DETAIL" not in str(error.value)
    assert "unclear" not in str(error.value)
    assert len(str(error.value)) < 1000
    assert len(calls) == 1 and not local.exists()


@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_deploy_unclear_result_is_distinct_and_never_retries(repository, tmp_path, failure):
    root, commit = repository
    deployment = module.Deployment(tmp_path / "final", tmp_path / "assets")
    calls = []
    def transport(*args, **kwargs):
        calls.append(args)
        if failure == "timeout":
            raise subprocess.TimeoutExpired("ssh", 300)
        return SimpleNamespace(returncode=0, stdout="not-json", stderr="")
    with pytest.raises(RuntimeError, match="result unclear"):
        module.deploy(root, deployment, commit, local_record_dir=tmp_path / "evidence", transport=transport)
    assert len(calls) == 1 and not (tmp_path / "evidence").exists()
