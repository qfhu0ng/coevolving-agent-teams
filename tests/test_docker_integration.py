"""Opt-in real Docker + native Qwen + official verifier, with no paid model.

Set STAGE1_DOCKER_TASK, STAGE1_LINUX_NODE_DIR and optionally QWEN_CODE_PACKAGE.
STAGE1_DOCKER_MEMORY_MB optionally overrides shared task RAM in positive MiB;
when absent, both tests keep the official task memory limit.
The scripted provider checks execution plumbing, not task-solving performance.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_company.evolution.loop import RunRequest, TaskSpec
from agent_company.runtime.relay import ProviderConfig, TokenRates
from agent_company.roadmap.task import Task
from agent_company.roadmap.runner import RoadmapRunner, RoadmapTeamRunner
from agent_company.runtime.workspace import DockerWorkspace
from test_qwen import FourWorkerScenario


def _task_memory_override():
    value = os.environ.get("STAGE1_DOCKER_MEMORY_MB")
    if value is None:
        return None
    try:
        memory_mb = int(value)
    except ValueError:
        pytest.fail("STAGE1_DOCKER_MEMORY_MB must be a positive integer in MiB")
    if memory_mb <= 0:
        pytest.fail("STAGE1_DOCKER_MEMORY_MB must be a positive integer in MiB")
    return memory_mb


def _inspect_task_memory(workspace, task, expected_memory_mb, observed):
    # Inspect exactly one nonsensitive scalar, never the container's Config/env.
    memory_bytes = int(workspace._docker(
        ["inspect", "--format", "{{.HostConfig.Memory}}", workspace.container_id], timeout=10,
    ).stdout.strip())
    observed.append(memory_bytes)
    assert memory_bytes == expected_memory_mb * 1024 * 1024
    assert workspace.task.memory_mb == expected_memory_mb
    assert workspace.task.cpus == task.cpus
    assert workspace.task.agent_timeout_sec == task.agent_timeout_sec
    assert workspace.task.verifier_timeout_sec == task.verifier_timeout_sec
    assert workspace.task.build_timeout_sec == task.build_timeout_sec


def test_full_native_team_delivery_and_official_roadmap_verifier(tmp_path, monkeypatch):
    task_path = os.environ.get("STAGE1_DOCKER_TASK")
    node_path = os.environ.get("STAGE1_LINUX_NODE_DIR")
    if not task_path or not node_path:
        pytest.skip("Opt in with official task path and Linux Node directory")
    original_start = DockerWorkspace.start
    original_run = DockerWorkspace.run_agent

    def start(workspace):
        result = original_start(workspace)
        _inspect_task_memory(workspace, task, expected_memory_mb, observed_memory_bytes)
        return result

    def smoke_run(workspace, command, *, env, timeout_seconds, stdout_path, stderr_path):
        # A scripted fixture should finish in seconds, even though the real
        # benchmark allows two hours. Fail a hung test without changing the
        # production runtime or silently retrying a task.
        if os.environ.get("STAGE1_TRACE_CHILD_PROCESS"):
            env = {**env, "NODE_DEBUG": "child_process"}
        return original_run(workspace, command, env=env,
                            timeout_seconds=min(timeout_seconds, 120),
                            stdout_path=stdout_path, stderr_path=stderr_path)

    monkeypatch.setattr(DockerWorkspace, "start", start)
    monkeypatch.setattr(DockerWorkspace, "run_agent", smoke_run)
    root = Path(__file__).parents[1]
    package = Path(os.environ.get("QWEN_CODE_PACKAGE", root / "external/qwen-study/package"))
    output = Path(os.environ.get("STAGE1_DOCKER_OUTPUT", tmp_path / "run"))
    task = Task.load(task_path, repository_id="thread-pool-development")
    original_task_toml = (task.path / "task.toml").read_bytes()
    memory_override = _task_memory_override()
    expected_memory_mb = task.memory_mb if memory_override is None else memory_override
    observed_memory_bytes = []
    scenario = FourWorkerScenario(SimpleNamespace(repo_path="/app", worker_root="/workspace/workers"))
    provider = ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=40,
                              rates=TokenRates("0", "0", "0"))
    runner = RoadmapTeamRunner(provider, package, Path(node_path), upstream=scenario,
                              task_memory_mb=memory_override)
    request = RunRequest(TaskSpec(task.task_id, task.repository_id, {"path": str(task.path)}),
                         "development", 0, "INTERFACE_CHANGE_POLICY: notify affected teammates.",
                         "Choose teammates for independent work; delegate final integration.", output)
    result = runner(request)
    test_record = {**asdict(result), "docker_memory_limit_bytes": observed_memory_bytes}
    (output / "test_result.json").write_text(json.dumps(test_record, indent=2) + "\n")
    assert (task.path / "task.toml").read_bytes() == original_task_toml
    assert Task.load(task.path).memory_mb == task.memory_mb
    assert result.status == "completed", result.error
    assert observed_memory_bytes == [expected_memory_mb * 1024 * 1024]
    assert result.metrics["team"]["stop_reason"] == "native_cli_completed"
    assert result.metrics["team"]["execution"]["returncode"] == 0
    assert len(result.metrics["team"]["workers"]) == 4
    assert result.metrics["verifier"]["status"] == "completed"
    assert result.score is not None
    assert result.cost_usd == 0
    assert scenario.observations == [
        {"phase": "before_idle", "nonce_visible": False},
        {"phase": "continuation", "nonce_visible": True, "history_visible": True},
    ]
    patch = (output / "delivery.patch").read_text()
    for name in scenario.names:
        assert f"b/{name}.txt" in patch
    assert (output / "native_runtime").is_dir()
    assert result.metrics["resources"]["cpus"] == task.cpus
    assert result.metrics["resources"]["memory_mb"] == expected_memory_mb
    assert result.metrics["resources"]["shared_by_all_agents"]
    assert not result.metrics["team"]["errors"]


def test_full_native_single_delivery_and_official_roadmap_verifier(tmp_path, monkeypatch):
    task_path = os.environ.get("STAGE1_DOCKER_TASK")
    node_path = os.environ.get("STAGE1_LINUX_NODE_DIR")
    if not task_path or not node_path:
        pytest.skip("Opt in with official task path and Linux Node directory")
    from test_qwen import SingleAgentScenario

    root = Path(__file__).parents[1]
    package = Path(os.environ.get("QWEN_CODE_PACKAGE", root / "external/qwen-study/package"))
    output = (Path(os.environ["STAGE1_DOCKER_OUTPUT"]) / "single"
              if os.environ.get("STAGE1_DOCKER_OUTPUT") else tmp_path / "run")
    task = Task.load(task_path, repository_id="thread-pool-development")
    original_task_toml = (task.path / "task.toml").read_bytes()
    memory_override = _task_memory_override()
    expected_memory_mb = task.memory_mb if memory_override is None else memory_override
    observed_memory_bytes = []
    scenario = SingleAgentScenario(SimpleNamespace(repo_path=""))
    events = []
    original_start = DockerWorkspace.start
    original_run = DockerWorkspace.run_agent
    original_finish = DockerWorkspace.finish_agent
    original_upload = DockerWorkspace.upload
    original_verify = DockerWorkspace.run_verifier

    def start(workspace):
        result = original_start(workspace)
        _inspect_task_memory(workspace, task, expected_memory_mb, observed_memory_bytes)
        scenario.ws.repo_path = workspace.repo_path
        # A harmless fixture input in this disposable container. The scripted
        # agent only creates its smoke-test artifact, not roadmap features.
        workspace.run(["test", "!", "-e", "original.txt"])
        workspace.write_file(f"{workspace.repo_path}/original.txt", "initial\n")
        return result

    def smoke_run(workspace, command, *, env, timeout_seconds, stdout_path, stderr_path):
        if os.environ.get("STAGE1_TRACE_CHILD_PROCESS"):
            env = {**env, "NODE_DEBUG": "child_process"}
        return original_run(workspace, command, env=env,
                            timeout_seconds=min(timeout_seconds, 120),
                            stdout_path=stdout_path, stderr_path=stderr_path)

    def finish(workspace):
        first_finish = not workspace.agent_finished
        original_finish(workspace)
        if first_finish:
            assert workspace.agent_finished
            events.append("agents_stopped")

    def upload(workspace, source, destination, *, contents=False):
        if destination == "/tests":
            assert workspace.agent_finished
            assert events == ["agents_stopped"]
            assert Path(source) == task.path / "tests"
        original_upload(workspace, source, destination, contents=contents)
        if destination == "/tests":
            events.append("hidden_tests_uploaded")

    def verify(workspace, command, **kwargs):
        assert workspace.agent_finished
        assert events == ["agents_stopped", "hidden_tests_uploaded"]
        assert command == ["bash", "/tests/test.sh"]
        result = original_verify(workspace, command, **kwargs)
        events.append("official_verifier_finished")
        return result

    monkeypatch.setattr(DockerWorkspace, "start", start)
    monkeypatch.setattr(DockerWorkspace, "run_agent", smoke_run)
    monkeypatch.setattr(DockerWorkspace, "finish_agent", finish)
    monkeypatch.setattr(DockerWorkspace, "upload", upload)
    monkeypatch.setattr(DockerWorkspace, "run_verifier", verify)
    provider = ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=40,
                              rates=TokenRates("0", "0", "0"))
    runner = RoadmapRunner(provider, package, Path(node_path), upstream=scenario,
                          task_memory_mb=memory_override)
    request = RunRequest(TaskSpec(task.task_id, task.repository_id, {"path": str(task.path)}),
                         "development", 0, "", "", output, execution_mode="single")
    result = runner(request)
    test_record = {**asdict(result), "docker_memory_limit_bytes": observed_memory_bytes}
    (output / "test_result.json").write_text(json.dumps(test_record, indent=2) + "\n")

    assert (task.path / "task.toml").read_bytes() == original_task_toml
    assert Task.load(task.path).memory_mb == task.memory_mb
    assert result.status == "completed", result.error
    assert observed_memory_bytes == [expected_memory_mb * 1024 * 1024]
    assert result.metrics["execution_mode"] == "single"
    assert result.metrics["policy_applied"] is False
    assert "team" not in result.metrics and not (output / "team").exists()
    runtime = result.metrics["single"]
    assert runtime["stop_reason"] == "native_cli_completed"
    assert runtime["execution"]["returncode"] == 0
    assert runtime["workers"] == {}
    assert runtime["calls_by_role"] == {"single": 5}
    assert runtime["scripted_provider"] and runtime["usage_complete"] and runtime["cost_complete"]
    assert not runtime["errors"]
    assert scenario.steps == 5
    assert scenario.observations == ["read_original", "write_readback", "shell_verified"]
    assert events == ["agents_stopped", "hidden_tests_uploaded", "official_verifier_finished"]
    assert result.metrics["verifier"]["status"] == "completed"
    assert result.score is not None and 0 <= result.score <= 1
    assert result.cost_usd == result.known_cost_usd == 0
    assert result.usage["total_tokens"] > 0
    assert result.metrics["resources"]["cpus"] == task.cpus
    assert result.metrics["resources"]["memory_mb"] == expected_memory_mb
    assert result.metrics["resources"]["shared_by_all_agents"]
    patch = (output / "delivery.patch").read_text()
    assert "b/single.txt" in patch and "+native-single-delivery" in patch
    assert (output / "native_runtime").is_dir()
    persisted = json.loads((output / "runner_result.json").read_text())
    assert persisted["usage"] == result.usage
    assert persisted["score"] == result.score
    assert persisted["metrics"]["resources"] == result.metrics["resources"]
    usage = json.loads((output / "single/relay/usage.json").read_text())
    assert usage["usage"] == result.usage and usage["cost_complete"]
    assert (output / "verifier/result.json").is_file()
