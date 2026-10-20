"""Shared task-memory overrides, CLI wiring, and resume identity without Docker."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_company.roadmap.cli as cli
from agent_company.evolution.loop import PolicyProposal, PolicyReview, TaskResult
from agent_company.runtime.relay import ProviderConfig
from agent_company.roadmap.task import Task
from agent_company.roadmap.runner import RoadmapRunner
from test_cli import execution_args
from test_runner import rig as rig


@pytest.fixture(autouse=True)
def clear_memory_environment(monkeypatch):
    monkeypatch.delenv("STAGE1_TASK_MEMORY_MB", raising=False)


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("memory", [None, 8192])
def test_runner_uses_one_shared_limit_and_preserves_official_task(rig, mode, memory):
    task_path = Path(rig.request.task.payload["path"])
    official = Task.load(task_path, repository_id=rig.request.task.repo_id)
    original_toml = (task_path / "task.toml").read_bytes()
    runner = replace(rig.runner, task_memory_mb=memory)
    result = runner(replace(rig.request, execution_mode=mode))

    assert result.status == "completed", result.error
    effective = 4096 if memory is None else memory
    workspace = rig.workspaces[0]
    assert workspace.task.memory_mb == effective
    assert workspace.resources == {"cpus": 2, "memory_mb": effective, "shared_by_all_agents": True}
    assert workspace.task.cpus == official.cpus == 2
    assert workspace.task.agent_timeout_sec == official.agent_timeout_sec == 7200
    assert workspace.task.verifier_timeout_sec == official.verifier_timeout_sec == 1800
    assert workspace.task.build_timeout_sec == official.build_timeout_sec == 600
    assert workspace.task.storage_mb == official.storage_mb == 10240
    assert official.memory_mb == 4096
    assert Task.load(task_path).memory_mb == 4096
    assert (task_path / "task.toml").read_bytes() == original_toml
    assert result.metrics["task"]["memory_mb"] == 4096
    assert result.metrics["resources"]["memory_mb"] == effective
    resource_config = {
        "official_memory_mb": 4096, "memory_mb_override": memory,
        "effective_memory_mb": effective, "memory_limit_scope": "task-shared",
    }
    assert result.metrics["resource_config"] == resource_config
    output = rig.request.output_dir
    assert json.loads((output / "workspace.json").read_text())["resource_config"] == resource_config
    assert json.loads((output / "execution.json").read_text())["resource_config"] == resource_config
    assert json.loads((output / "runner_result.json").read_text())["metrics"]["resource_config"] == resource_config
    assert rig.modes == [mode]
    assert workspace.stopped


@pytest.mark.parametrize("value", [True, False, 0, -1, 8192.0, 1.5, "8192"])
def test_direct_runner_rejects_invalid_memory_before_execution(tmp_path, value):
    with pytest.raises(ValueError, match="task_memory_mb.*positive integer"):
        RoadmapRunner(ProviderConfig("fixture", "https://provider.invalid"), tmp_path,
                      tmp_path, task_memory_mb=value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["single", "team"])
def test_run_task_cli_forwards_and_persists_effective_memory(rig, mode):
    provider = rig.request.output_dir.parent / "provider.json"
    provider.write_text(json.dumps({"model": "fixture", "base_url": "https://provider.invalid"}))
    args = [
        "run-task", "--provider-config", str(provider),
        "--qwen-package", str(rig.runner.qwen_package),
        "--node-directory", str(rig.runner.node_directory),
        "--task", rig.request.task.payload["path"], "--repo-id", rig.request.task.repo_id,
        "--output", str(rig.request.output_dir), "--mode", mode, "--task-memory-mb", "8192",
    ]
    assert cli.main(args) == 0
    assert rig.workspaces[0].task.memory_mb == 8192
    assert rig.modes == [mode]
    saved = json.loads((rig.request.output_dir / "task_result.json").read_text())
    assert saved["metrics"]["resource_config"] == {
        "official_memory_mb": 4096, "memory_mb_override": 8192,
        "effective_memory_mb": 8192, "memory_limit_scope": "task-shared",
    }


@pytest.fixture
def offline_cli(monkeypatch):
    """Keep the real CLI/controller; substitute only external task/model execution."""
    seen = SimpleNamespace(constructors=[], requests=[])

    def make_runner(*args, **kwargs):
        seen.constructors.append(kwargs)

        def run(request):
            seen.requests.append((kwargs["task_memory_mb"], request))
            return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5,
                              cost_usd=0.01)

        return run

    monkeypatch.setattr(cli, "RoadmapRunner", make_runner)
    monkeypatch.setattr(cli, "StrategyOptimizer", lambda *a, **k:
                        lambda request: PolicyProposal("communicate changed interfaces",
                                                       "assign interface owners", cost_usd=0.01))
    monkeypatch.setattr(cli, "StrategyReviewer", lambda *a, **k:
                        lambda request: PolicyReview("prefer_candidate", "Fixture advisory review",
                                                     cost_usd=0.01))
    return seen


@pytest.mark.parametrize("command,mode", [("baseline", "single"), ("baseline", "team"), ("run", None)])
@pytest.mark.parametrize("memory", [None, 8192])
def test_cli_baseline_and_evolution_share_memory_across_all_phases(
    tmp_path, offline_cli, command, mode, memory,
):
    args = execution_args(tmp_path, command)
    if command == "run":
        args += ["--replay-selection", "first_n"]  # This fixture tests resource propagation.
    if mode:
        args += ["--mode", mode]
    if memory is not None:
        args += ["--task-memory-mb", str(memory)]
    assert cli.main(args) == 0
    assert offline_cli.constructors[0]["task_memory_mb"] == memory
    assert offline_cli.requests and all(value == memory for value, _ in offline_cli.requests)
    runtime = json.loads((tmp_path / "result/run_settings.json").read_text())["runtime"]
    if memory is None:
        assert "task_memory_mb" not in runtime  # Existing default-memory checkpoints stay compatible.
    else:
        assert runtime["task_memory_mb"] == memory
    requests = [request for _, request in offline_cli.requests]
    if command == "baseline":
        assert len(requests) == 2
        assert all(request.execution_mode == mode for request in requests)
    else:
        assert [request.split for request in requests] == ["train", "train_replay", "test", "test"]
        assert [request.execution_mode for request in requests] == ["team", "team", "single", "team"]


@pytest.mark.parametrize("command", ["baseline", "run"])
@pytest.mark.parametrize("memory", [None, 8192])
def test_cli_resume_reuses_same_memory_and_rejects_changed_limit(
    tmp_path, offline_cli, command, memory,
):
    args = execution_args(tmp_path, command)
    if memory is not None:
        args += ["--task-memory-mb", str(memory)]
    assert cli.main(args) == 0
    calls = len(offline_cli.requests)
    assert cli.main([*args, "--resume"]) == 0
    assert len(offline_cli.requests) == calls
    output = tmp_path / "result"
    previous = {name: (output / name).read_bytes() for name in ("run_settings.json", "summary.json")}
    with pytest.raises(ValueError, match="settings differ"):
        cli.main([*args, "--resume", "--task-memory-mb", "6144"])
    assert len(offline_cli.requests) == calls
    assert all((output / name).read_bytes() == contents for name, contents in previous.items())


@pytest.mark.parametrize("command", ["run-task", "baseline", "run"])
@pytest.mark.parametrize("environment,argument,expected", [
    ("8192", None, 8192), ("8192", "6144", 6144), ("invalid", "6144", 6144),
])
def test_environment_default_and_explicit_cli_precedence(
    tmp_path, monkeypatch, offline_cli, command, environment, argument, expected,
):
    monkeypatch.setenv("STAGE1_TASK_MEMORY_MB", environment)
    args = execution_args(tmp_path, command)
    if argument is not None:
        args += ["--task-memory-mb", argument]
    assert cli.main(args) == 0
    assert offline_cli.constructors[0]["task_memory_mb"] == expected
    assert all(value == expected for value, _ in offline_cli.requests)
    if command != "run-task":
        runtime = json.loads((tmp_path / "result/run_settings.json").read_text())["runtime"]
        assert runtime["task_memory_mb"] == expected


@pytest.mark.parametrize("command", ["run-task", "baseline", "run"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
@pytest.mark.parametrize("source", ["argument", "environment"])
def test_invalid_cli_or_environment_memory_stops_before_task_execution(
    tmp_path, monkeypatch, offline_cli, command, value, source,
):
    args = execution_args(tmp_path, command)
    if source == "argument":
        args += ["--task-memory-mb", value]
    else:
        monkeypatch.setenv("STAGE1_TASK_MEMORY_MB", value)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert not offline_cli.constructors and not offline_cli.requests
    assert not (tmp_path / "result").exists()
