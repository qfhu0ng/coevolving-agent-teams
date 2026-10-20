"""Mock lifecycle and malicious-source tests, without models or hidden tests."""
import io
import json
import os
import pwd
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from agent_company.programbench.workspace import ProgramBenchWorkspace, programbench_source_boundary
from test_programbench import task as task_fixture

task = task_fixture
IMAGE_ID = "sha256:" + "a" * 64


@pytest.fixture
def rig(task, monkeypatch):
    ws = ProgramBenchWorkspace(task, container_name="owned-pb")
    events = []

    def network():
        events.append("network-deny")
        ws._network_container_id = "net"
        ws._network_enforced = True

    def docker(args, **kwargs):
        events.append(args)
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, "container\n", "")
        if args[:3] == ["inspect", "--type", "container"]:
            return subprocess.CompletedProcess(args, 0, IMAGE_ID + "\n", "")
        return subprocess.CompletedProcess(args, 0, "{}", "")

    def run(args, **kwargs):
        events.append(args)
        return subprocess.CompletedProcess(args, 0,
            json.dumps({"uid": 1000, "readable": False, "executable": True}), "")

    monkeypatch.setattr(ws, "_start_network", network)
    monkeypatch.setattr(ws, "_docker", docker)
    monkeypatch.setattr(ws, "run", run)
    monkeypatch.setattr(ws, "write_file", lambda *args: events.append(list(args)))
    return ws, events


def test_nonroot_resource_limits_and_setup_before_git(rig):
    ws, events = rig
    ws.start()
    run = next(e for e in events if isinstance(e, list) and e[0] == "run")
    assert events.index("network-deny") < events.index(run)
    assert run[run.index("--user") + 1] == "agent"
    assert run[run.index("--cpus") + 1] == "20.0"
    assert run[run.index("--memory") + 1] == run[run.index("--memory-swap") + 1] == "61440m"
    assert "SYS_PTRACE" in run and "no-new-privileges" in run
    setup = next(e for e in events if isinstance(e, list) and e[:3] == ["exec", "--user", "0"])
    assert events.index(setup) < events.index(["git", "init"])
    assert ws.repo_path == "/workspace/solution"
    assert ws.network_policy_status["verifier_network"] == "separate-official-evaluator"


def test_image_id_is_read_from_the_started_container_before_setup(rig):
    ws, events = rig
    ws.start()
    run = next(e for e in events if isinstance(e, list) and e[0] == "run")
    inspect = ["inspect", "--type", "container", "--format", "{{.Image}}", "container"]
    setup = next(e for e in events if isinstance(e, list) and e[:3] == ["exec", "--user", "0"])
    assert events.index(run) < events.index(inspect) < events.index(setup)
    assert not any(".Config" in str(part) for e in events if isinstance(e, list) for part in e)
    assert ws.image_identity == {"reference": ws.task.image, "image_id": IMAGE_ID, "error": None,
                                 "source": "docker inspect --type container .Image after docker run"}
    ws.stop()
    assert ws.container_id is None and ws.image_id == IMAGE_ID


@pytest.mark.parametrize("stdout, error", [("{}", "unexpected_format"), ("", "unexpected_format"),
                                           ("sha256:" + "a" * 63, "unexpected_format"),
                                           ("programbench/x:task_cleanroom_v6", "unexpected_format")])
def test_unexpected_image_output_is_unknown_and_startup_continues(rig, monkeypatch, stdout, error):
    ws, events = rig
    original = ws._docker

    def docker(args, **kwargs):
        if args[:3] == ["inspect", "--type", "container"]:
            events.append(args)
            return subprocess.CompletedProcess(args, 0, stdout, "")
        return original(args, **kwargs)

    monkeypatch.setattr(ws, "_docker", docker)
    ws.start()
    assert ws.repo_path == "/workspace/solution"
    assert (ws.image_id, ws.image_id_error) == (None, error)


def test_start_failing_at_docker_run_reports_image_not_observed(rig, monkeypatch):
    ws, events = rig
    assert (ws.image_id, ws.image_id_error) == (None, "not_observed")

    def docker(args, **kwargs):
        events.append(args)
        if args[0] == "run":
            raise subprocess.TimeoutExpired(["docker", *args], 600)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(ws, "_docker", docker)
    with pytest.raises(subprocess.TimeoutExpired):
        ws.start()
    assert not any(e[:2] == ["inspect", "--type"] for e in events if isinstance(e, list))
    assert ws.image_identity["image_id"] is None
    assert ws.image_identity["error"] == "not_observed"


def test_image_inspect_failure_keeps_only_the_error_type(rig, monkeypatch):
    ws, events = rig
    original = ws._docker

    def docker(args, **kwargs):
        if args[:3] == ["inspect", "--type", "container"]:
            raise subprocess.CalledProcessError(1, ["docker", *args], output="PRIVATE_OUT", stderr="PRIVATE_ERR")
        return original(args, **kwargs)

    monkeypatch.setattr(ws, "_docker", docker)
    ws.start()
    assert ws.repo_path == "/workspace/solution"
    assert ws.image_identity["image_id"] is None
    assert ws.image_identity["error"] == "CalledProcessError"
    assert "PRIVATE" not in json.dumps(ws.image_identity)


def test_reference_readable_causes_owned_cleanup(rig, monkeypatch):
    ws, events = rig
    monkeypatch.setattr(ws, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0,
        json.dumps({"uid": 1000, "readable": True, "executable": True}), ""))
    with pytest.raises(RuntimeError, match="execute-only"):
        ws.start()
    assert ["rm", "-f", "container"] in events and ["rm", "-f", "net"] in events
    # The identity observed before the failed setup survives owned cleanup.
    assert ws.container_id is None and ws.image_id == IMAGE_ID


def test_no_export_during_agent_execution(task, tmp_path):
    ws = ProgramBenchWorkspace(task)
    ws.container_id = "container"
    with pytest.raises(RuntimeError, match="Quiesce"):
        ws.export_source(tmp_path / "delivery")
    assert not (tmp_path / "delivery").exists()


@pytest.mark.parametrize("name,kind", [("../outside", tarfile.REGTYPE),
                                      ("escape", tarfile.SYMTYPE), ("escape", tarfile.LNKTYPE)])
def test_export_does_not_extract_hostile_members(task, tmp_path, monkeypatch, name, kind):
    ws = ProgramBenchWorkspace(task)
    ws.container_id, ws.agent_finished = "container", True

    def fake_run(args, stdout, **kwargs):
        with tarfile.open(fileobj=stdout, mode="w:gz") as archive:
            member = tarfile.TarInfo(name)
            member.type, member.linkname = kind, "/etc/passwd"
            archive.addfile(member)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="Unsafe"):
        ws.export_source(tmp_path / "delivery")
    assert not (tmp_path / "outside").exists()


def test_export_keeps_untracked_source_modes(task, tmp_path, monkeypatch):
    ws = ProgramBenchWorkspace(task)
    ws.container_id, ws.agent_finished = "container", True

    def fake_run(args, stdout, **kwargs):
        assert "-I" in args
        with tarfile.open(fileobj=stdout, mode="w:gz") as archive:
            member = tarfile.TarInfo("compile.sh")
            member.size, member.mode = 1, 0o755
            archive.addfile(member, io.BytesIO(b"x"))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ws.export_source(tmp_path / "delivery")
    assert (tmp_path / "delivery/compile.sh").read_text() == "x"
    assert (tmp_path / "delivery/compile.sh").stat().st_mode & 0o111


def test_delivery_history_bundle_only_after_quiesce_and_records_its_hash(task, tmp_path, monkeypatch):
    ws = ProgramBenchWorkspace(task)
    ws.container_id = "container"
    with pytest.raises(RuntimeError, match="Quiesce"):
        ws.export_git_bundle(tmp_path / "history.bundle")
    assert not (tmp_path / "history.bundle").exists()
    ws.agent_finished = True
    calls = []

    def fake_run(args, stdout, **kwargs):
        calls.append(args)
        stdout.write(b"# v2 git bundle\n")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    record = ws.export_git_bundle(tmp_path / "history.bundle")
    assert "exec" in calls[0] and calls[0][-5:] == ["/workspace/solution", "bundle", "create", "-", "--all"]
    assert "status" not in calls[0] and "add" not in calls[0]
    assert record["bytes"] == 16 and len(record["sha256"]) == 64
    with pytest.raises(FileExistsError):
        ws.export_git_bundle(tmp_path / "history.bundle")


def test_benchmark_boundary_is_not_roadmap():
    text = programbench_source_boundary()
    assert "from observable behavior" in text and "baseline repository" not in text
    assert "/opt/ac-programbench-reference/executable" in text


def test_official_image_git_is_isolated_before_seeding_delivery(tmp_path, monkeypatch):
    from agent_company.programbench.workspace import _SETUP
    base, protected = tmp_path / "workspace", tmp_path / "reference"
    base.mkdir()
    (base / "executable").write_text("synthetic oracle")
    (base / "README.mkd").write_text("public documentation")
    (base / ".git").mkdir()
    (base / ".git/history").write_text("must not enter the new source tree")
    monkeypatch.setattr(os, "chown", lambda *a: None)
    monkeypatch.setattr(pwd, "getpwnam", lambda *a: SimpleNamespace(pw_uid=1000, pw_gid=1000))
    exec(_SETUP.replace("'/workspace'", repr(str(base))).replace(
        "'/opt/ac-programbench-reference'", repr(str(protected))), {})
    assert not (base / "solution/.git").exists()
    assert (protected / "image-git-metadata").stat().st_mode & 0o777 == 0o700
    assert (protected / "executable").stat().st_mode & 0o777 == 0o111
    assert (base / "solution/README.mkd").read_text() == "public documentation"


def test_regular_hardlinks_export_as_independent_files(tmp_path):
    from agent_company.programbench.workspace import _EXPORT
    root = tmp_path / "source"
    root.mkdir()
    (root / "one").write_text("source")
    os.link(root / "one", root / "two")
    data = subprocess.check_output([sys.executable, "-I", "-c",
                                   _EXPORT.replace("'/workspace/solution'", repr(str(root)))])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        assert all(member.isfile() for member in archive)
