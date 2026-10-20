"""Runner-level outcomes with real Git delivery and an in-process task backend."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from agent_company.evolution.loop import RunRequest, TaskSpec
from agent_company.runtime.relay import ProviderConfig
from agent_company.roadmap.runner import RoadmapRunner, RoadmapTeamRunner
from agent_company.runtime.workspace import DockerWorkspace
from test_roadmap import make_task


def runtime_result(**overrides):
    return {
        "stop_reason": "native_cli_completed", "execution": {"returncode": 0, "timed_out": False},
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "cost_complete": False, "known_estimated_cost": None, "currency": None,
        **overrides,
    }


def test_prepared_image_is_applied_without_changing_official_metadata(rig):
    task = replace(rig.request.task, payload={**rig.request.task.payload,
                                            "prepared_image": "local/offline:1"})
    result = rig.runner(replace(rig.request, task=task))
    assert rig.workspaces[0].task.image == "local/offline:1"
    assert result.metrics["image_config"]["prepared_image"] == "local/offline:1"
    assert result.metrics["task"]["image"] != "local/offline:1"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    package, node = tmp_path / "qwen", tmp_path / "node"
    package.mkdir()
    (node / "bin").mkdir(parents=True)
    (package / "cli.js").write_text("// not executed")
    (node / "bin/node").write_text("not executed")
    state = SimpleNamespace(events=[], workspaces=[], runtime=runtime_result(),
                            verifier={"status": "completed", "reward": 0.75},
                            team_error=None, verifier_error=None, cleanup_error=None,
                            start_error=None,
                            seen_policies=[], providers=[], modes=[], runtime_outputs=[])

    class Workspace:
        def __init__(self, supplied_task, mounts, platform, container_name):
            assert container_name.startswith("ac-stage1-")
            assert supplied_task.repository_id == "owner/thread-pool"
            assert len(mounts) == 2 and all(readonly for _, _, readonly in mounts)
            assert not any(Path(source) == task.path for source, _, _ in mounts)
            self.task = supplied_task
            self.repo = tmp_path / f"container-{len(state.workspaces)}"
            self.repo_path = str(self.repo)
            self.resources = {"cpus": 2, "memory_mb": supplied_task.memory_mb, "shared_by_all_agents": True}
            self.ended = False
            self.stopped = False
            self.diagnostics = []
            state.workspaces.append(self)

        def start(self):
            if state.start_error:
                self.diagnostics.append({"phase": "docker_run", "outcome": "startup_exception"})
                raise state.start_error
            self.repo.mkdir()
            self.run(["git", "init"])
            self.run(["git", "config", "user.name", "Runner Test"])
            self.run(["git", "config", "user.email", "test@localhost"])
            self.run(["git", "config", "commit.gpgsign", "false"])
            (self.repo / "committed.txt").write_text("initial\n")
            (self.repo / "dirty.txt").write_text("initial\n")
            self.run(["git", "add", "."])
            self.run(["git", "commit", "-qm", "initial task image"])
            state.events.append("start")

        def run(self, command, **kwargs):
            state.events.append(tuple(command))
            return subprocess.run(command, cwd=self.repo, text=True, capture_output=True,
                                  check=True, **kwargs)

        def download(self, source, destination):
            assert self.ended
            destination.mkdir(parents=True)
            (destination / "native-trace.json").write_text('{"retained": true}')
            state.events.append("download-traces")

        def stop(self):
            self.stopped = True
            state.events.append("stop")
            if state.cleanup_error:
                raise state.cleanup_error

    class NativeTeam:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, workspace, instruction, assignment, communication, output, provider, timeout,
                *, execution_mode="team"):
            assert not (workspace.repo / "new-source.py").exists()
            assert (workspace.repo / "committed.txt").read_text() == "initial\n"
            assert not workspace.ended
            assert timeout == 7200
            state.seen_policies.append((assignment, communication))
            state.modes.append(execution_mode)
            state.runtime_outputs.append(output)
            state.providers.append(provider)
            output.mkdir(parents=True)
            (output / "qwen_stdout.jsonl").write_text('{"type":"result","is_error":false}\n')
            if state.team_error:
                raise state.team_error
            # An integrated commit plus uncommitted edits and a new file must all be delivered.
            (workspace.repo / "committed.txt").write_text("integrated worker commit\n")
            workspace.run(["git", "add", "committed.txt"])
            workspace.run(["git", "commit", "-qm", "integrated worker commit"])
            (workspace.repo / "dirty.txt").write_text("pending final edits\n")
            (workspace.repo / "new-source.py").write_text("answer = 42\n")
            workspace.ended = True
            state.events.append("team-ended")
            return copy.deepcopy(state.runtime)

    def verify(supplied_task, workspace, output):
        assert workspace.ended, "Hidden scoring started while the team was still active"
        # Final scoring observes the whole delivery repository, not one worker or just HEAD.
        assert (workspace.repo / "committed.txt").read_text() == "integrated worker commit\n"
        assert (workspace.repo / "dirty.txt").read_text() == "pending final edits\n"
        assert (workspace.repo / "new-source.py").read_text() == "answer = 42\n"
        state.events.append("score-hidden")
        if state.verifier_error:
            raise state.verifier_error
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(state.verifier))
        return copy.deepcopy(state.verifier)

    monkeypatch.setattr("agent_company.roadmap.runner.DockerWorkspace", Workspace)
    monkeypatch.setattr("agent_company.roadmap.runner.QwenTeamRunner", NativeTeam)
    monkeypatch.setattr("agent_company.roadmap.runner.score", verify)
    state.runner = RoadmapTeamRunner(ProviderConfig("model", "https://provider.invalid", api_key="test-secret"),
                                    package, node)
    state.request = RunRequest(TaskSpec(task.task_id, "owner/thread-pool", {"path": str(task.path)}),
                               "test", 0, "notify dependencies", "fixed dynamic assignment",
                               tmp_path / "run-0", seed=3)
    return state


def test_paired_runs_start_fresh_and_deliver_commits_dirty_and_new_files(rig):
    first = rig.runner(rig.request)
    second = rig.runner(replace(rig.request, output_dir=rig.request.output_dir.parent / "run-1",
                                communication_policy="notify only interface changes",
                                assignment_policy="assign interface owners before parallel work", policy_version=1))
    assert first.status == second.status == "completed"
    assert first.score == second.score == 0.75
    assert len(rig.workspaces) == 2
    assert rig.workspaces[0].repo_path != rig.workspaces[1].repo_path
    assert all(workspace.stopped for workspace in rig.workspaces)
    patch = (rig.request.output_dir / "delivery.patch").read_text()
    assert "+integrated worker commit" in patch
    assert "+pending final edits" in patch
    assert "+answer = 42" in patch
    assert rig.events.index("team-ended") < rig.events.index("score-hidden") < rig.events.index("stop")
    assert rig.seen_policies == [("fixed dynamic assignment", "notify dependencies"),
                                ("assign interface owners before parallel work", "notify only interface changes")]
    assert "delivery.patch" in first.trace_files
    assert first.metrics["initial_revision"] != first.metrics["delivery_revision"]


def test_single_uses_same_delivery_verifier_resources_and_accounting_without_policies(rig):
    assert RoadmapTeamRunner is RoadmapRunner
    rig.runtime.update(cost_complete=True, currency="USD", known_estimated_cost="0.4")
    # Even direct callers supplying A/C cannot inject team policies into solo.
    result = rig.runner(replace(rig.request, execution_mode="single"))
    assert result.status == "completed" and result.score == 0.75
    assert result.cost_usd == result.known_cost_usd == 0.4
    assert result.usage == rig.runtime["usage"]
    assert result.metrics["execution_mode"] == "single"
    assert result.metrics["policy_applied"] is False
    assert "team" not in result.metrics
    assert result.metrics["single"]["stop_reason"] == "native_cli_completed"
    assert result.metrics["resources"] == rig.workspaces[0].resources
    assert rig.seen_policies == [("", "")]
    assert rig.modes == ["single"]
    assert rig.runtime_outputs == [rig.request.output_dir / "single"]
    assert rig.events.index("team-ended") < rig.events.index("score-hidden") < rig.events.index("stop")
    assert rig.workspaces[0].stopped
    assert "single/qwen_stdout.jsonl" in result.trace_files
    assert not (rig.request.output_dir / "team").exists()
    assert "+answer = 42" in (rig.request.output_dir / "delivery.patch").read_text()
    workspace_record = json.loads((rig.request.output_dir / "workspace.json").read_text())
    assert workspace_record["execution_mode"] == "single"
    assert workspace_record["network_policy_requested"] == "relay-only-v1"
    assert result.metrics["network_policy_requested"] == "relay-only-v1"
    assert "network_policy" not in workspace_record and "network_policy" not in result.metrics


@pytest.mark.parametrize("enforced", [False, True])
def test_runner_records_network_fact_before_cleanup_without_assuming_startup_succeeded(rig, monkeypatch, enforced):
    import agent_company.roadmap.runner as module
    workspace_type = module.DockerWorkspace
    observed = {"policy": "relay-only-v1", "enforced": enforced,
                "allowed_relay": {"ip": "192.0.2.1", "port": 12345} if enforced else None,
                "dependency_access": "preinstalled-or-offline",
                "verifier_network": "same-restricted-namespace"}
    monkeypatch.setattr(workspace_type, "network_policy_status", observed, raising=False)
    if not enforced:
        rig.start_error = RuntimeError("network boundary setup failed")
    original_stop = workspace_type.stop

    def stop(workspace):
        observed["enforced"] = False
        original_stop(workspace)

    monkeypatch.setattr(workspace_type, "stop", stop)
    result = rig.runner(rig.request)
    assert result.status == ("completed" if enforced else "failed")
    workspace_record = json.loads((rig.request.output_dir / "workspace.json").read_text())
    saved = result.metrics["network_policy"]
    assert saved["enforced"] is enforced
    assert workspace_record["network_policy"] == saved
    assert not observed["enforced"]


@pytest.mark.parametrize(("reason", "status", "score"), [
    ("task_timeout", "completed", 0.75), ("cli_error", "failed", 0.75),
    ("missing_cli_result", "failed", 0.75),
])
def test_single_preserves_deadline_and_infrastructure_failure_semantics(rig, reason, status, score):
    rig.runtime.update(stop_reason=reason, cost_complete=False, currency="USD",
                       known_estimated_cost="0.2")
    result = rig.runner(replace(rig.request, execution_mode="single"))
    assert result.status == status and result.score == score
    assert result.metrics["verifier"]["reward"] == 0.75
    assert result.cost_usd is None and result.known_cost_usd == 0.2
    assert rig.workspaces[0].stopped


def test_single_restart_without_container_identity_is_not_assumed_safe(rig, tmp_path):
    directory = tmp_path / "old-single"
    (directory / "single").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="container identity"):
        rig.runner.assert_can_restart(directory)


@pytest.mark.parametrize(("complete", "currency", "known", "expected"), [
    (False, None, None, None), (False, "USD", "0.25", None),
    (True, "CNY", "0.25", None), (True, "USD", "0.25", 0.25), (True, "USD", "0", 0.0),
])
def test_unknown_or_incomplete_cost_is_not_zero(rig, complete, currency, known, expected):
    rig.runtime.update(cost_complete=complete, currency=currency, known_estimated_cost=known)
    result = rig.runner(rig.request)
    assert result.cost_usd == expected
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}
    assert result.metrics["team"]["known_estimated_cost"] == known


def test_benchmark_deadline_keeps_valid_partial_score(rig):
    rig.runtime = runtime_result(stop_reason="task_timeout", execution={"returncode": 124, "timed_out": True})
    result = rig.runner(rig.request)
    assert result.status == "completed"
    assert result.score == 0.75
    assert result.metrics["team"]["execution"]["timed_out"]


@pytest.mark.parametrize("reason", ["cli_error", "launcher_error", "native_result_error"])
@pytest.mark.parametrize("reward", [0, 0.75])
def test_native_infrastructure_failure_retains_completed_verifier_score(rig, reason, reward):
    rig.runtime = runtime_result(stop_reason=reason, cost_complete=True, currency="USD",
                                known_estimated_cost="0.5")
    rig.verifier = {"status": "completed", "reward": reward}
    result = rig.runner(rig.request)
    assert result.status == "failed"
    assert result.score == reward
    assert reason in result.error
    assert result.metrics["verifier"]["reward"] == reward
    assert result.cost_usd == 0.5
    assert rig.workspaces[0].stopped
    saved = json.loads((rig.request.output_dir / "runner_result.json").read_text())
    assert saved["status"] == "failed" and saved["score"] == reward
    assert saved["error"] == result.error


@pytest.mark.parametrize("status", ["verifier_timeout", "verifier_error", "running"])
def test_incomplete_verifier_is_execution_failure_even_if_it_wrote_partial_reward(rig, status):
    rig.verifier = {"status": status, "reward": 0.5}
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None
    assert result.metrics["verifier"]["reward"] == 0.5
    assert status in result.error


@pytest.mark.parametrize("verifier", [
    {"status": "invalid_reward", "reward": None},
    {"status": "verifier_error", "reward": None},
    {"status": "verifier_timeout"},
])
def test_invalid_or_unavailable_verifier_reward_does_not_invent_zero(rig, verifier):
    rig.verifier = verifier
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None
    assert verifier["status"] in result.error


@pytest.mark.parametrize("mode", ["team", "single"])
@pytest.mark.parametrize("reward", [0, 0.75])
def test_cleanup_failure_preserves_completed_verifier_score_and_diagnostics(rig, mode, reward):
    rig.verifier = {"status": "completed", "reward": reward}
    rig.cleanup_error = RuntimeError("container cleanup failed")
    result = rig.runner(replace(rig.request, execution_mode=mode))
    assert result.status == "failed" and result.score == reward
    assert "cleanup failed" in result.error
    assert result.metrics["cleanup_error"] == "RuntimeError"
    assert result.metrics["cleanup_diagnostic"]["message"] == "container cleanup failed"
    assert result.metrics["verifier"] == rig.verifier
    assert result.metrics[mode]["stop_reason"] == "native_cli_completed"
    assert rig.events.index("score-hidden") < rig.events.index("stop")
    saved = json.loads((rig.request.output_dir / "runner_result.json").read_text())
    assert saved["status"] == "failed" and saved["score"] == reward
    assert saved["metrics"]["cleanup_diagnostic"] == result.metrics["cleanup_diagnostic"]


def test_native_raise_cleans_up_without_hidden_scoring_and_redacts_credentials(rig):
    rig.team_error = RuntimeError("provider refused test-secret")
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None
    assert "score-hidden" not in rig.events
    assert rig.workspaces[0].stopped
    assert "test-secret" not in result.error
    assert "[REDACTED]" in result.error


def test_verifier_raise_still_preserves_paid_usage_and_cleans_up(rig):
    rig.runtime.update(cost_complete=True, currency="USD", known_estimated_cost="0.4")
    rig.verifier_error = OSError("verifier copy failed")
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None
    assert result.cost_usd == 0.4
    assert result.usage["completion_tokens"] == 20
    assert rig.workspaces[0].stopped


def test_fresh_start_false_is_rejected_before_creating_a_container(rig):
    with pytest.raises(ValueError, match="fresh"):
        rig.runner(replace(rig.request, fresh_start=False))
    assert not rig.workspaces


def test_seed_support_is_explicit_and_original_provider_is_not_mutated(rig):
    no_seed = rig.runner(rig.request)
    assert not no_seed.metrics["provider_seed_applied"]
    rig.runner.provider_supports_seed = True
    with_seed = rig.runner(replace(rig.request, output_dir=rig.request.output_dir.parent / "run-1"))
    assert with_seed.metrics["provider_seed_applied"]
    assert rig.providers[-1].extra_body["seed"] == 3
    assert "seed" not in rig.runner.provider.extra_body
    assert with_seed.metrics["provider_seed_determinism_claimed"] is False


@pytest.mark.parametrize(("elapsed", "oom", "expected_timeout"), [(0.25, False, False),
                                                                  (12.0, False, True),
                                                                  (12.0, True, False)])
def test_sigkill_is_not_automatically_a_benchmark_timeout(tmp_path, monkeypatch, elapsed, oom,
                                                       expected_timeout):
    workspace = DockerWorkspace(make_task(tmp_path))
    workspace.container_id = "owned-test-container"
    times = iter([100.0, 100.0 + elapsed, 100.0 + elapsed])
    monkeypatch.setattr("agent_company.runtime.workspace.time.monotonic", lambda: next(times))

    class Process:
        def wait(self, timeout):
            return 137

    monkeypatch.setattr("agent_company.runtime.workspace.subprocess.Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(DockerWorkspace, "_docker", lambda self, *a, **kw:
                        subprocess.CompletedProcess([], 0, json.dumps({"OOMKilled": oom}), ""))
    result = workspace._long_command(["qwen"], env={}, timeout_seconds=10,
                                     stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr")
    assert result["timed_out"] is expected_timeout
    assert result["returncode"] == 137
    assert result["oom_killed"] is oom


def test_startup_timeout_persists_phase_and_outputs_without_sensitive_command(rig):
    rig.start_error = subprocess.TimeoutExpired(
        ["docker", "run", "--env", "API_KEY=test-secret", "private-prompt"], 600,
        output=b"pulling image\n", stderr=b"Authorization: Bearer test-secret\n")
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None
    assert result.metrics["failure"]["phase"] == "workspace_start"
    assert result.metrics["failure"]["timeout"] == 600
    assert result.metrics["failure"]["stdout"] == "pulling image\n"
    assert "test-secret" not in json.dumps(result.metrics)
    assert "private-prompt" not in json.dumps(result.metrics)
    assert "private-prompt" not in result.error
    assert "score-hidden" not in rig.events
    assert rig.workspaces[0].stopped
    diagnostics = rig.request.output_dir / "workspace_diagnostics.json"
    assert json.loads(diagnostics.read_text())[0]["phase"] == "docker_run"
    assert "workspace_diagnostics.json" in result.trace_files
