from __future__ import annotations

import json
import subprocess

import pytest

from agent_company.roadmap.task import Task, parse_reward, score
from agent_company.runtime.workspace import DockerWorkspace


def make_task(tmp_path):
    task_path = tmp_path / "tpl-5.0.0-roadmap"
    task_path.mkdir()
    (task_path / "tests").mkdir()
    (task_path / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_path / "instruction.md").write_text("Implement the release requirements.")
    (task_path / "task.toml").write_text('''version = "1"
[metadata]
name = "tpl-5.0.0-roadmap"
[agent]
timeout_sec = 7200
[verifier]
timeout_sec = 1800
[environment]
docker_image = "znpt/roadmapbench-tpl-5.0.0-roadmap"
build_timeout_sec = 600
cpus = 2
memory_mb = 4096
storage_mb = 10240
''')
    return Task.load(task_path)


def test_harbor_manifest_preserves_task_limits_and_explicit_repository(tmp_path):
    task = make_task(tmp_path)
    assert task.agent_timeout_sec == 7200
    assert task.verifier_timeout_sec == 1800
    assert task.cpus == 2
    assert task.memory_mb == 4096
    assert task.repository_id == "tpl"
    assert Task.load(task.path, repository_id="bshoshany/thread-pool").repository_id == "bshoshany/thread-pool"
    assert task.instruction == "Implement the release requirements."
    assert "instruction" not in task.describe()


@pytest.mark.parametrize("details", [
    {"reward": 0.3, "phases_passed": 2, "total_phases": 6},
    {"reward": 0.3, "targets": {"one": {"passed": 1, "total": 3}, "two": True}},
    {"reward": 0.3, "targets_passed": 2, "total_targets": 6},
    0.3,
])
def test_official_scalar_is_not_recomputed_from_heterogeneous_target_details(tmp_path, details):
    (tmp_path / "reward.txt").write_text("0.45454545454545453\n")
    (tmp_path / "reward.json").write_text(json.dumps(details))
    result = parse_reward(tmp_path)
    assert result.reward == 5 / 11
    assert result.details == details
    assert result.source == "reward.txt"


@pytest.mark.parametrize("payload", ["nan", "inf", "-0.1", "1.1"])
def test_invalid_official_scalar_is_not_counted_as_task_failure(tmp_path, payload):
    (tmp_path / "reward.txt").write_text(payload)
    with pytest.raises(ValueError):
        parse_reward(tmp_path)


def test_reward_json_scalar_fallback_and_missing_weighted_score(tmp_path):
    path = tmp_path / "reward.json"
    path.write_text('{"reward": 0.25, "target_scores": [0.5, 0]}')
    assert parse_reward(tmp_path).reward == 0.25
    path.write_text('{"targets_passed": 2, "total_targets": 3}')
    with pytest.raises(ValueError, match="No official scalar"):
        parse_reward(tmp_path)


def test_final_verifier_is_uploaded_only_after_agent_termination(tmp_path):
    task = make_task(tmp_path)
    events = []

    class Workspace:
        def finish_agent(self):
            events.append("terminated")

        def run(self, command):
            events.append(("run", command))

        def upload(self, source, destination, contents):
            assert events[0] == "terminated"
            assert source == task.path / "tests"
            assert destination == "/tests"
            assert contents
            events.append("upload_hidden")

        def run_verifier(self, command, **kwargs):
            assert "upload_hidden" in events
            assert command == ["bash", "/tests/test.sh"]
            assert kwargs["timeout_seconds"] == 1800
            events.append("verify")
            return {"returncode": 0, "timed_out": False, "elapsed_seconds": 1.0}

        def download(self, source, destination):
            assert events[-1] == "verify"
            (destination / "reward.txt").write_text("0.625")
            (destination / "reward.json").write_text('{"reward": 0.625, "phases_passed": 2}')

    result = score(task, Workspace(), tmp_path / "result")
    assert result["reward"] == 0.625
    assert result["status"] == "completed"
    assert json.loads((tmp_path / "result" / "result.json").read_text())["reward"] == 0.625


def test_docker_one_container_resource_limits_and_worker_paths(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    commands = []

    def fake_docker(self, args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "owned-container\n", "")

    def fake_run(self, command, **kwargs):
        commands.append(command)
        if command == ["pwd"]:
            return subprocess.CompletedProcess(command, 0, "/app\n", "")
        return subprocess.CompletedProcess(command, int(command[0] == "test"), "", "")

    monkeypatch.setattr(DockerWorkspace, "_docker", fake_docker)
    monkeypatch.setattr(DockerWorkspace, "run", fake_run)
    monkeypatch.setattr(DockerWorkspace, "_start_network", lambda self: None)
    with DockerWorkspace(task) as workspace:
        assert [workspace.ensure_worker(x) for x in ("one", "two", "three", "four")] == [
            f"/workspace/workers/{x}" for x in ("one", "two", "three", "four")]
        start = commands[0]
        assert start[start.index("--cpus") + 1] == "2.0"
        assert start[start.index("--memory") + 1] == "4096m"
        assert sum(command[0] == "run" for command in commands) == 1
        assert workspace.resources["storage_quota_enforced"] is False
        assert not any("/tests" in str(part) for part in start)
        workspace.finish_agent()
        assert commands[-2:] == [["stop", "-t", "0", "owned-container"], ["start", "owned-container"]]
        with pytest.raises(RuntimeError, match="cannot resume"):
            workspace.run_agent(["qwen"], env={}, timeout_seconds=7200,
                                stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr")


def test_agent_deadline_caps_the_entire_team_and_stops_backgrounds(tmp_path, monkeypatch):
    workspace = DockerWorkspace(make_task(tmp_path))
    workspace.container_id = "owned-container"
    events = []

    def long_command(self, command, **kwargs):
        assert kwargs["timeout_seconds"] == 7200
        events.append("agent")
        return {"returncode": 0, "timed_out": False, "elapsed_seconds": 1.0}

    def finish(self):
        events.append("stop-all-processes")
        self.agent_finished = True

    monkeypatch.setattr(DockerWorkspace, "_long_command", long_command)
    monkeypatch.setattr(DockerWorkspace, "finish_agent", finish)
    monkeypatch.setattr(DockerWorkspace, "_apply_network_boundary", lambda self: None)
    workspace.run_agent(["qwen"], env={}, timeout_seconds=9999,
                        stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr")
    assert events == ["agent", "stop-all-processes"]


def test_worker_name_cannot_escape_workspace(tmp_path):
    workspace = DockerWorkspace(make_task(tmp_path))
    for name in ("../x", "/app", "x/y", "", "."):
        with pytest.raises(ValueError):
            workspace.ensure_worker(name)
