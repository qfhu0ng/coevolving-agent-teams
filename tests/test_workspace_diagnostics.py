"""Lifecycle diagnostics are collected before quiescence, without Docker/model calls."""
import json
import subprocess

import pytest

from agent_company.runtime.workspace import DockerWorkspace, exception_diagnostic
from test_roadmap import make_task


@pytest.fixture(autouse=True)
def isolate_network_setup(monkeypatch):
    # This module isolates subprocess/lifecycle diagnostics. The network's own
    # failure and ordering behavior is covered in test_network_workspace.
    monkeypatch.setattr(DockerWorkspace, "_start_network", lambda self: None)
    monkeypatch.setattr(DockerWorkspace, "_apply_network_boundary", lambda self: None)


def test_uncertain_start_inspects_only_reserved_name_and_keeps_timeout_bytes(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path), container_name="ac-stage1-exact-reserved")
    workspace.diagnostic_secrets = ("private-key",)
    events = []
    failure = subprocess.TimeoutExpired(["docker", "run", "private-command"], 600,
                                        output=b"Pulling layers\n", stderr=b"token=private-key\n")

    def docker(args, **kwargs):
        events.append((args, kwargs))
        if args[0] == "run":
            raise failure
        assert args == ["inspect", "--format", "{{json .State}}", "ac-stage1-exact-reserved"]
        assert kwargs["timeout"] == 2
        return subprocess.CompletedProcess(args, 0, json.dumps({"Status": "running", "Pid": 17,
                                                               "Env": ["MUST_NOT_SAVE=private"]}), "")

    monkeypatch.setattr(workspace, "_docker", docker)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        workspace.start()
    assert raised.value is failure
    workspace.stop()
    assert [args[0] for args, _ in events] == ["run", "inspect"]
    assert workspace.container_id is None  # An observed name is not proof of ownership.
    event = workspace.diagnostics[0]
    assert event["phase"] == "docker_run" and event["outcome"] == "startup_exception"
    assert event["error"]["stdout"] == "Pulling layers\n"
    assert "[REDACTED]" in event["error"]["stderr"]
    assert event["container_state"] == {"Status": "running", "Pid": 17}
    assert "private-key" not in json.dumps(event)
    assert "private-command" not in json.dumps(event)


@pytest.mark.parametrize("exitcode", [0, 1, 124, -15])
def test_every_cli_exit_snapshots_before_stop_restart(tmp_path, monkeypatch, exitcode):
    workspace = DockerWorkspace(make_task(tmp_path), container_name="exact-name")
    workspace.container_id = "owned-id"
    events = []

    class Process:
        def wait(self, timeout):
            events.append("wait")
            return exitcode

    def launch(args, stdout, stderr, text):
        stderr.write("provider refused private-relay-key\n")
        stdout.write("x" * 20000 + " final output\n")
        return Process()

    def docker(args, **kwargs):
        events.append(args[0])
        if args[0] == "inspect":
            return subprocess.CompletedProcess(args, 0, json.dumps({"Running": True,
                                                                   "OOMKilled": False}), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("agent_company.runtime.workspace.subprocess.Popen", launch)
    monkeypatch.setattr(workspace, "_docker", docker)
    result = workspace.run_agent(["qwen", "private-prompt"], env={"API_KEY": "private-relay-key"},
                                 timeout_seconds=10, stdout_path=tmp_path / "stdout",
                                 stderr_path=tmp_path / "stderr")
    assert events == ["wait", "inspect", "stop", "start"]
    assert workspace.agent_finished
    assert result["returncode"] == exitcode
    assert result["timed_out"] is (exitcode == 124)
    diagnostic = result["diagnostics"]
    assert diagnostic["phase"] == "agent"
    assert diagnostic["host_signal"] == (15 if exitcode == -15 else None)
    assert len(diagnostic["stdout_tail"]) == 16384
    assert diagnostic["stdout_tail"].endswith("final output\n")
    assert "private-relay-key" not in json.dumps(diagnostic)
    assert "private-prompt" not in json.dumps(diagnostic)


def test_inspection_failure_does_not_block_quiescence(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path))
    workspace.container_id = "owned-id"
    events = []

    class Process:
        def wait(self, timeout):
            return 1

    def docker(args, **kwargs):
        events.append(args[0])
        if args[0] == "inspect":
            raise subprocess.TimeoutExpired(args, 2)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("agent_company.runtime.workspace.subprocess.Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(workspace, "_docker", docker)
    result = workspace.run_agent(["qwen"], env={}, timeout_seconds=10,
                                 stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr")
    assert events == ["inspect", "stop", "start"]
    assert result["diagnostics"]["inspect_error"]["type"] == "TimeoutExpired"
    assert result["oom_killed"] is None and workspace.agent_finished


def test_timeout_inspection_precedes_forced_stop_and_does_not_hide_original_error(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path))
    workspace.container_id = "owned-id"
    events = []
    failure = subprocess.TimeoutExpired(["private-command"], 25)

    class Process:
        def wait(self, timeout):
            if timeout == 25:
                raise failure
            events.append("reap")
            return -9

        def kill(self):
            events.append("kill")

    def docker(args, **kwargs):
        events.append(args[0])
        if args[0] == "inspect":
            return subprocess.CompletedProcess(args, 0, '{"Running":true}', "")
        raise OSError("cleanup daemon unavailable")

    monkeypatch.setattr("agent_company.runtime.workspace.subprocess.Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(workspace, "_docker", docker)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        workspace.run_agent(["qwen"], env={}, timeout_seconds=10,
                            stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr")
    assert raised.value is failure
    assert events[:4] == ["inspect", "stop", "kill", "reap"]
    assert not workspace.agent_finished
    event = workspace.diagnostics[0]
    assert event["outcome"] == "host_wait_timeout"
    assert event["error"]["type"] == "TimeoutExpired"
    assert event["termination_errors"][0]["type"] == "OSError"
    assert "private-command" not in json.dumps(workspace.diagnostics)


def test_failed_popen_is_diagnosed_and_quiesced(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path))
    workspace.container_id = "owned-id"
    events = []

    def launch(*args, **kwargs):
        raise FileNotFoundError(2, "No such file", "private-command")

    def docker(args, **kwargs):
        events.append(args[0])
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr("agent_company.runtime.workspace.subprocess.Popen", launch)
    monkeypatch.setattr(workspace, "_docker", docker)
    with pytest.raises(FileNotFoundError):
        workspace.run_agent(["qwen"], env={}, timeout_seconds=10,
                            stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr")
    assert events == ["inspect", "stop", "start"]
    assert workspace.diagnostics[0]["outcome"] == "command_exception"
    assert workspace.agent_finished
    assert "private-command" not in json.dumps(workspace.diagnostics)


def test_subprocess_diagnostic_does_not_stringify_command():
    error = subprocess.CalledProcessError(1, ["docker", "--env", "SECRET=private"],
                                         output="last progress", stderr="x" * 20000)
    detail = exception_diagnostic(error)
    assert detail["returncode"] == 1 and detail["stdout"] == "last progress"
    assert len(detail["stderr"]) == 16384
    assert "SECRET" not in json.dumps(detail)


def test_setup_error_is_not_masked_by_cleanup_failure(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path))
    failure = ValueError("invalid repository directory")
    events = []

    def docker(args, **kwargs):
        events.append(args[0])
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, "owned-id\n", "")
        if args[0] == "inspect":
            return subprocess.CompletedProcess(args, 0, '{"Running":true}', "")
        raise OSError("cleanup unavailable")

    def command(*args, **kwargs):
        raise failure

    monkeypatch.setattr(workspace, "_docker", docker)
    monkeypatch.setattr(workspace, "run", command)
    with pytest.raises(ValueError) as raised:
        workspace.start()
    assert raised.value is failure
    assert events == ["run", "inspect", "rm"]
    assert workspace.diagnostics[0]["phase"] == "workspace_setup"
    assert workspace.diagnostics[0]["cleanup_error"]["message"] == "cleanup unavailable"


def test_starting_started_workspace_does_not_delete_existing_container(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path))
    workspace.container_id = "owned-id"
    monkeypatch.setattr(workspace, "_docker", lambda *a, **kw: pytest.fail("Must not call Docker"))
    with pytest.raises(RuntimeError, match="already started"):
        workspace.start()
    assert workspace.container_id == "owned-id"


def test_verifier_original_failure_survives_cleanup_failure(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path))
    workspace.container_id = "owned-id"
    workspace.agent_finished = True
    failure = ValueError("verifier launch failed")

    def launch(*args, **kwargs):
        raise failure

    def docker(*args, **kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(workspace, "_long_command", launch)
    monkeypatch.setattr(workspace, "_docker", docker)
    with pytest.raises(ValueError) as raised:
        workspace.run_verifier(["verify"], timeout_seconds=10,
                               stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr")
    assert raised.value is failure
    assert workspace.diagnostics[-1]["phase"] == "verifier_quiesce"
