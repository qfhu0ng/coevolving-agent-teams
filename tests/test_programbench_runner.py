"""Mock execution/evaluation lifecycle for the independent ProgramBench runner."""
from dataclasses import replace
import json
import tarfile
from types import SimpleNamespace

import pytest

from agent_company.evolution.loop import RunRequest, TaskSpec
from agent_company.programbench import runner as module
from agent_company.runtime.relay import ProviderConfig

INFERENCE_IMAGE_ID = "sha256:" + "c" * 64


@pytest.fixture
def rig(tmp_path, monkeypatch):
    task_dir = tmp_path / "tomnomnom__gron.88a6234"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text(
        "repository: tomnomnom/gron\ncommit: 88a6234ea2d0c487090988182ad9a7cdf6def924\nlanguage: go\n")
    instruction = tmp_path / "official_instructions.md"
    instruction.write_text("Public synthetic black-box task")
    package, node = tmp_path / "qwen", tmp_path / "node"
    package.mkdir()
    (package / "cli.js").write_text("unused")
    (node / "bin").mkdir(parents=True)
    (node / "bin/node").write_text("unused")
    rig = SimpleNamespace(events=[], workspaces=[], calls=[], eval_calls=[], failures={},
                          runtime={
        "model": "test-model", "stop_reason": "native_cli_completed", "execution_mode": "single",
        "execution": {"returncode": 0, "timed_out": False, "elapsed_seconds": 12,
                      "stdout": "PRIVATE_DIAGNOSTIC", "environment": "PRIVATE_SECRET"},
        "native_result": {"explicit_error": False, "missing_result": False,
                          "result": {"type": "result", "subtype": "success", "result": "PRIVATE_REASONING"},
                          "lifecycle": {"events": ["PRIVATE_TRACE"]}},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                  "cached_prompt_tokens": 2, "private": "PRIVATE_USAGE_BODY"},
        "usage_complete": False, "cost_complete": False, "currency": "USD", "known_estimated_cost": "0.01",
        "workers": {"builder": "/workers/builder"}, "runtime_dir": "/private/runtime",
        "errors": [{"error": "PRIVATE_ERROR_BODY"}], "diagnostics": ["PRIVATE_DIAGNOSTIC"],
        "max_model_calls": 1000, "logical_model_calls": 4, "provider_attempts": 5,
    }, verification={"status": "completed", "reward": 0.75, "resolved": False, "almost": False,
                     "passed_tests": 3, "valid_tests": 4, "official_test_fraction": 0.75,
                     "reward_details": {"log": "PRIVATE_HIDDEN_TEST"},
                     "error_details": "PRIVATE_HIDDEN_ERROR"})

    class FakeWorkspace:
        def __init__(self, task, **kwargs):
            self.task, self.kwargs = task, kwargs
            self.agent_finished = False
            self.stopped = False
            self.diagnostics = ["PRIVATE_DIAGNOSTIC"]
            self.network_policy_status = {"policy": "relay-only", "enforced": True,
                                          "namespace_container_id": "synthetic-network",
                                          "diagnostics": "PRIVATE_NETWORK_DIAGNOSTIC"}
            self.resources = {"cpus": task.cpus, "memory_mb": task.memory_mb,
                              "memory_swap_mb": task.memory_mb, "shared_by_all_agents": True,
                              "secret": "PRIVATE_RESOURCE_SECRET"}
            rig.workspaces.append(self)

        def check(self, phase):
            rig.events.append(phase)
            if phase in rig.failures:
                raise rig.failures[phase]

        def start(self):
            # Like the real workspace: the identity is observed right after docker run,
            # so it exists even when later startup setup fails.
            self.image_identity = {"reference": self.task.image, "image_id": INFERENCE_IMAGE_ID, "error": None,
                                   "source": "synthetic", "diagnostic": "PRIVATE_IMAGE_DIAGNOSTIC"}
            self.check("start")

        def finish_agent(self):
            self.check("finish_agent")
            self.agent_finished = True

        def export_source(self, destination):
            assert self.agent_finished
            self.check("export_source")
            destination.mkdir()
            (destination / "compile.sh").write_text("#!/bin/sh\ncp main.sh executable\n")
            (destination / "main.sh").write_text("#!/bin/sh\nprintf synthetic\n")
            (destination / "untracked.txt").write_text("new file included in full submission")

        def export_git_bundle(self, destination):
            assert self.agent_finished and (destination.parent / "submission.tar.gz").is_file()
            self.check("export_git_bundle")
            destination.write_bytes(b"# v2 git bundle\n")
            return {"path": str(destination), "bytes": 16, "sha256": "synthetic"}

        def stop(self):
            self.check("stop")
            self.stopped = True
            self.network_policy_status["enforced"] = False

    class FakeQwen:
        def __init__(self, *args, **kwargs):
            rig.qwen_options = kwargs

        def run(self, workspace, instruction, assignment, communication, output, provider, timeout_seconds, **options):
            rig.events.append("agent")
            rig.workspace_json_at_agent = json.loads((output.parent / "workspace.json").read_text())
            rig.calls.append({"instruction": instruction, "assignment": assignment, "communication": communication,
                              "provider": provider, "timeout_seconds": timeout_seconds, "options": options})
            output.mkdir()
            (output / "qwen_stdout.jsonl").write_text("PRIVATE_RAW_TRACE")
            if "agent" in rig.failures:
                raise rig.failures["agent"]
            return rig.runtime

    def evaluate(task, submission, output, **kwargs):
        assert rig.workspaces[-1].agent_finished
        assert rig.workspaces[-1].stopped
        assert submission.is_file()
        rig.events.append("evaluate")
        rig.eval_calls.append({"task": task, "submission": submission, "options": kwargs})
        if "evaluate" in rig.failures:
            raise rig.failures["evaluate"]
        output.mkdir()
        (output / "result.json").write_text(json.dumps(rig.verification))
        return rig.verification

    monkeypatch.setattr(module, "ProgramBenchWorkspace", FakeWorkspace)
    monkeypatch.setattr(module, "QwenTeamRunner", FakeQwen)
    monkeypatch.setattr(module, "evaluate_submission", evaluate)
    monkeypatch.setattr(module, "programbench_source_boundary", lambda: "PROGRAMBENCH_ORACLE_BOUNDARY")
    provider = ProviderConfig("test-model", "http://unused.invalid", api_key="PRIVATE_SECRET")
    rig.runner = module.ProgramBenchRunner(provider, package, node, upstream=object())
    rig.request = RunRequest(
        TaskSpec(task_dir.name, "tomnomnom/gron", {"path": str(task_dir), "instruction_path": str(instruction)}),
        "test", 3, "COMMUNICATION_POLICY", "ASSIGNMENT_POLICY", tmp_path / "result", execution_mode="single")
    return rig


def test_full_submission_is_scored_only_after_agent_stop_and_inference_cleanup(rig):
    result = rig.runner(rig.request)
    assert rig.events == ["start", "agent", "finish_agent", "export_source", "export_git_bundle", "stop", "evaluate"]
    assert result.status == "completed" and result.score == 0.75
    assert result.metrics["agent_termination_confirmed"] is True
    assert result.metrics["inference_cleanup_confirmed"] is True
    assert result.metrics["network_policy"]["enforced"] is True
    assert result.metrics["official_verifier_environment"] == "separate"
    with tarfile.open(rig.request.output_dir / "submission.tar.gz") as archive:
        assert set(archive.getnames()) == {"compile.sh", "main.sh", "untracked.txt"}
    assert not (rig.request.output_dir / "delivery.patch").exists()


def test_inference_image_identity_is_recorded_before_the_agent_and_kept_after_cleanup(rig):
    rig.verification["evaluator_image"] = {"reference": "programbench/x:tag",
                                           "before": {"image_id": INFERENCE_IMAGE_ID, "error": None},
                                           "after": {"image_id": INFERENCE_IMAGE_ID, "error": None}}
    result = rig.runner(rig.request)
    expected = {"reference": rig.workspaces[0].task.image, "image_id": INFERENCE_IMAGE_ID,
                "error": None, "source": "synthetic"}
    assert result.metrics["inference_image"] == expected
    assert rig.workspace_json_at_agent["inference_image"] == expected
    record = json.loads((rig.request.output_dir / "workspace.json").read_text())
    assert record["inference_image"] == expected and record["inference_cleanup_confirmed"] is True
    assert json.loads((rig.request.output_dir / "execution.json").read_text())["inference_image"] == expected
    assert result.metrics["verifier"]["evaluator_image"] == rig.verification["evaluator_image"]


def test_startup_failure_after_docker_run_still_records_inference_image(rig):
    rig.failures["start"] = RuntimeError("PRIVATE_SETUP")
    result = rig.runner(rig.request)
    assert result.status == "failed" and rig.eval_calls == []
    assert result.metrics["inference_image"]["image_id"] == INFERENCE_IMAGE_ID
    record = json.loads((rig.request.output_dir / "workspace.json").read_text())
    assert record["inference_image"]["image_id"] == INFERENCE_IMAGE_ID
    assert "PRIVATE_" not in json.dumps(record)


def test_default_resources_and_single_prompt_preserve_baseline_and_shared_limit(rig):
    result = rig.runner(rig.request)
    task = rig.workspaces[0].task
    assert (task.agent_timeout_sec, task.cpus, task.memory_mb) == (21600, 20, 61440)
    call = rig.calls[0]
    assert call["assignment"] == call["communication"] == ""
    assert call["timeout_seconds"] == 21600
    assert call["options"] == {"execution_mode": "single", "max_model_calls": 1000,
                               "source_boundary": "PROGRAMBENCH_ORACLE_BOUNDARY"}
    assert rig.eval_calls[0]["options"] == {"docker_cpus": 10, "docker_executable": "docker"}
    assert result.metrics["policy_applied"] is False


def test_team_consumes_explicit_policies_and_each_arm_gets_a_fresh_workspace(rig):
    rig.runner(rig.request)
    team = replace(rig.request, execution_mode="team", output_dir=rig.request.output_dir.parent / "team-result")
    result = rig.runner(team)
    assert len(rig.workspaces) == 2
    assert rig.workspaces[0].kwargs["container_name"] != rig.workspaces[1].kwargs["container_name"]
    assert rig.calls[1]["assignment"] == "ASSIGNMENT_POLICY"
    assert rig.calls[1]["communication"] == "COMMUNICATION_POLICY"
    assert result.metrics["policy_applied"] is True


def test_summaries_exclude_raw_native_text_errors_diagnostics_and_hidden_test_bodies(rig):
    result = rig.runner(rig.request)
    serialized = json.dumps(result.metrics)
    assert "PRIVATE_" not in serialized
    assert "PRIVATE_" not in (rig.request.output_dir / "execution.json").read_text()
    assert "PRIVATE_" not in (rig.request.output_dir / "runner_result.json").read_text()
    assert all("qwen_stdout" not in name and "relay/" not in name for name in result.trace_files)
    assert result.metrics["single"]["worker_count"] == 1
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                            "cached_prompt_tokens": 2}
    assert result.cost_usd is None and result.known_cost_usd == 0.01


@pytest.mark.parametrize("reason,expected_status", [("native_cli_completed", "completed"),
    ("task_timeout", "failed"), ("model_call_limit_exceeded", "completed"), ("cli_error", "failed"),
    ("provider_error", "failed"), ("unhandled_worker_failure", "failed")])
def test_runtime_outcome_does_not_discard_valid_official_reward(rig, reason, expected_status):
    rig.runtime["stop_reason"] = reason
    result = rig.runner(rig.request)
    assert result.status == expected_status and result.score == 0.75
    assert rig.events[-1] == "evaluate"
    if reason == "model_call_limit_exceeded":
        assert result.metrics["runtime_outcome"] == "budget_limited"


@pytest.mark.parametrize("status,reward", [("verifier_error", 0.25), ("missing_evaluation", None),
    ("empty_evaluation", None), ("completed", None), ("completed", float("nan")), ("completed", True)])
def test_missing_incomplete_or_invalid_official_result_is_unknown_not_zero(rig, status, reward):
    rig.verification.update(status=status, reward=reward)
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None


def test_genuine_official_zero_is_completed(rig):
    rig.verification.update(reward=0.0, official_test_fraction=0.0)
    result = rig.runner(rig.request)
    assert result.status == "completed" and result.score == 0.0


@pytest.mark.parametrize("complete,known,currency,expected_cost,expected_known", [
    (False, None, None, None, None), (False, "0.02", "USD", None, 0.02),
    (True, "0.02", "USD", 0.02, 0.02), (True, "0.02", "EUR", None, None),
])
def test_real_relay_cost_fields_preserve_unknown_and_partial_accounting(rig, complete, known, currency,
                                                                     expected_cost, expected_known):
    rig.runtime.update(cost_complete=complete, known_estimated_cost=known, currency=currency)
    result = rig.runner(rig.request)
    assert result.cost_usd == expected_cost
    assert result.known_cost_usd == expected_known


def test_delivery_history_is_saved_after_the_archive_and_its_failure_is_not_a_run_failure(rig):
    result = rig.runner(rig.request)
    assert result.metrics["delivery_history"]["bytes"] == 16
    assert (rig.request.output_dir / "delivery_history.bundle").is_file()
    rig.failures["export_git_bundle"] = RuntimeError("PRIVATE_SECRET")
    rig.events.clear()
    result = rig.runner(replace(rig.request, output_dir=rig.request.output_dir.parent / "second"))
    assert result.status == "completed" and result.score == 0.75
    assert result.metrics["delivery_history"] == {"error_type": "RuntimeError"}
    assert rig.events == ["start", "agent", "finish_agent", "export_source", "export_git_bundle", "stop", "evaluate"]
    assert "PRIVATE_" not in json.dumps(result.metrics)


@pytest.mark.parametrize("phase", ["start", "finish_agent", "export_source", "stop"])
def test_inference_boundary_failure_prevents_official_evaluation_and_redacts_exception(rig, phase):
    rig.failures[phase] = RuntimeError("PRIVATE_SECRET PRIVATE_REASONING")
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None
    assert rig.eval_calls == []
    assert "PRIVATE_" not in result.error
    assert "PRIVATE_" not in json.dumps(result.metrics)
    assert "stop" in rig.events
    if phase == "stop":
        assert result.metrics["inference_cleanup_confirmed"] is False


def test_agent_launcher_exception_still_scores_only_after_confirmed_termination(rig):
    rig.failures["agent"] = RuntimeError("PRIVATE_SECRET")
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score == 0.75
    assert rig.events == ["start", "agent", "finish_agent", "export_source", "export_git_bundle", "stop", "evaluate"]
    assert result.usage == {} and result.cost_usd is None
    assert result.metrics["single"]["usage_complete"] is False
    assert "PRIVATE_" not in result.error


def test_evaluation_exception_preserves_agent_usage_and_never_restarts_inference(rig):
    rig.failures["evaluate"] = RuntimeError("PRIVATE_HIDDEN_TEST")
    result = rig.runner(rig.request)
    assert result.status == "failed" and result.score is None
    assert result.usage["total_tokens"] == 15
    assert len(rig.calls) == len(rig.eval_calls) == 1
    assert "PRIVATE_" not in result.error
    assert result.metrics["inference_cleanup_confirmed"] is True


def test_explicit_resource_overrides_are_custom_and_seed_requires_support(rig):
    runner = replace(rig.runner, task_cpus=4, task_memory_mb=16384, agent_timeout_seconds=1200,
                     evaluator_cpus=3, max_model_calls=15, provider_supports_seed=True)
    result = runner(replace(rig.request, seed=7))
    assert result.metrics["resource_config"]["resource_limits_source"] == "custom"
    assert (rig.workspaces[0].task.cpus, rig.workspaces[0].task.memory_mb) == (4, 16384)
    assert rig.calls[0]["options"]["max_model_calls"] == 15
    assert rig.calls[0]["provider"].extra_body["seed"] == 7
    assert rig.runner.provider.extra_body == {}
    assert rig.eval_calls[0]["options"]["docker_cpus"] == 3


def test_seed_is_not_sent_without_provider_support(rig):
    result = rig.runner(replace(rig.request, seed=7))
    assert "seed" not in rig.calls[0]["provider"].extra_body
    assert result.metrics["provider_seed_applied"] is False


def test_custom_docker_wrapper_is_shared_by_inference_and_official_evaluator(rig):
    wrapper = "/synthetic path/docker-wrapper"
    rig.verification.update(docker_executable=wrapper, evaluator_invocation="official_api_subprocess")
    result = replace(rig.runner, docker=wrapper)(rig.request)
    assert rig.workspaces[0].kwargs["docker"] == wrapper
    assert rig.eval_calls[0]["options"]["docker_executable"] == wrapper
    assert result.metrics["verifier"]["docker_executable"] == wrapper
    assert result.metrics["verifier"]["evaluator_invocation"] == "official_api_subprocess"


def test_duplicate_output_and_nonfresh_request_fail_before_any_new_execution(rig):
    with pytest.raises(ValueError, match="fresh"):
        rig.runner(replace(rig.request, fresh_start=False))
    assert not rig.workspaces
    rig.runner(rig.request)
    with pytest.raises(FileExistsError, match="fresh"):
        rig.runner(rig.request)
    assert len(rig.workspaces) == len(rig.calls) == 1


def test_task_identity_and_team_policy_errors_fail_before_execution(rig):
    wrong = replace(rig.request.task, repo_id="different/repository")
    with pytest.raises(ValueError, match="identity"):
        rig.runner(replace(rig.request, task=wrong))
    with pytest.raises(ValueError, match="both assignment"):
        rig.runner(replace(rig.request, execution_mode="team", assignment_policy=""))
    assert not rig.workspaces


@pytest.mark.parametrize("field,value", [("task_memory_mb", True), ("task_cpus", 0),
    ("agent_timeout_seconds", float("inf")), ("max_model_calls", 0), ("evaluator_cpus", 1.5),
    ("team_stall_timeout_seconds", -1)])
def test_invalid_runner_limits_fail_early(rig, field, value):
    with pytest.raises(ValueError):
        replace(rig.runner, **{field: value})
    assert not rig.workspaces


def test_recovery_checks_exact_old_container_and_refuses_active_agents(rig, monkeypatch):
    rig.runner(rig.request)
    calls = []

    def docker_state(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="running\n")

    monkeypatch.setattr(module.subprocess, "run", docker_state)
    with pytest.raises(RuntimeError, match="still active"):
        rig.runner.assert_can_restart(rig.request.output_dir)
    name = rig.workspaces[0].kwargs["container_name"]
    assert calls[0] == ["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.State}}"]


def _blocked_by_cleanup(rig):
    rig.failures["stop"] = module.subprocess.TimeoutExpired(["docker", "rm"], 30)
    result = rig.runner(rig.request)
    assert result.status == "failed" and rig.eval_calls == []
    (rig.request.output_dir / "task_result.json").write_text(json.dumps(module.asdict(result)))
    return rig.request.output_dir


def _containers(monkeypatch, *, exists):
    calls = []

    def inspect(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if exists else 1, stdout="running\n" if exists else "",
                               stderr="" if exists else "Error: No such container: x")

    monkeypatch.setattr(module.subprocess, "run", inspect)
    return calls


def test_a_trial_blocked_only_by_cleanup_is_verified_once_its_containers_are_gone(rig, monkeypatch):
    output = _blocked_by_cleanup(rig)
    before = (output / "task_result.json").read_text()
    calls = _containers(monkeypatch, exists=False)
    rig.workspaces[-1].stopped = True  # the containers were removed after the run
    result = module.complete_verification(output, rig.workspaces[-1].task, evaluator_cpus=3, docker="tools/docker")
    name = rig.workspaces[-1].kwargs["container_name"]
    assert [command[-1] for command in calls] == [name, f"{name}-network"] and calls[0][0] == "tools/docker"
    assert len(rig.eval_calls) == 1
    assert rig.eval_calls[0]["options"] == {"docker_cpus": 3, "docker_executable": "tools/docker"}
    assert result["status"] == "completed" and result["score"] == 0.75 and result["error"] is None
    metrics = result["metrics"]
    assert metrics["failures"] == [{"phase": "inference_cleanup", "error_type": "TimeoutExpired"}]
    assert metrics["inference_cleanup_confirmed"] is False
    assert metrics["verification_after_cleanup"]["cleanup_confirmed"] is True
    assert metrics["verifier"]["passed_tests"] == 3 and "PRIVATE_" not in json.dumps(result)
    assert (output / "task_result.before_verification.json").read_text() == before
    assert json.loads((output / "task_result.json").read_text()) == result
    assert json.loads((output / "execution.json").read_text()) == metrics
    with pytest.raises(ValueError, match="already"):
        module.complete_verification(output, rig.workspaces[-1].task)
    assert len(rig.eval_calls) == 1


def test_verification_after_cleanup_refuses_live_containers_and_other_failures(rig, monkeypatch):
    output = _blocked_by_cleanup(rig)
    _containers(monkeypatch, exists=True)
    with pytest.raises(RuntimeError, match="still exists"):
        module.complete_verification(output, rig.workspaces[-1].task)
    assert rig.eval_calls == [] and not list(output.glob("*.before_verification.json"))
    del rig.failures["stop"]
    completed = rig.request.output_dir.parent / "completed"
    (completed / "task_result.json").parent.mkdir()
    result = rig.runner(replace(rig.request, output_dir=completed))
    (completed / "task_result.json").write_text(json.dumps(module.asdict(result)))
    _containers(monkeypatch, exists=False)
    with pytest.raises(ValueError, match="single failure"):
        module.complete_verification(completed, rig.workspaces[-1].task)
    assert len(rig.eval_calls) == 1  # only the completed run's own evaluation
