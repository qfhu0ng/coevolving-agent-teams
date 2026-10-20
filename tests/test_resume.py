"""Crash/restart behavior without a model or Docker daemon."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_company.runtime_io import write_json_atomic
from agent_company.evolution.loop import PolicyProposal, TaskResult, TaskSpec, run_baseline, run_stage1
from agent_company.evolution.recovery import experiment_session, interrupted_usage
from agent_company.roadmap.runner import RoadmapTeamRunner
from agent_company.runtime.relay import ProviderConfig


def baseline_options(tmp_path):
    return dict(tasks=[TaskSpec("one", "repo"), TaskSpec("two", "repo")],
                communication_policy="C0", assignment_policy="A0", output_dir=tmp_path / "run")


def test_baseline_resume_reuses_success_and_terminal_failure(tmp_path):
    calls = []

    def run(request):
        calls.append(request.task.task_id)
        return TaskResult("failed", error="transport", cost_usd=.1) if len(calls) == 1 else TaskResult(
            "completed", score=.5, cost_usd=.2)

    options = baseline_options(tmp_path)
    first = run_baseline(**options, run_task=run)
    assert calls == ["one", "two"]
    second = run_baseline(**options, run_task=run, resume=True, retry_interrupted=True)
    assert second == first
    assert calls == ["one", "two"]
    assert first["results"]["known_cost_usd"] == pytest.approx(.3)
    assert first["optimizer_invoked"] is False


@pytest.mark.parametrize("runner_checkpoint_only", [False, True])
def test_resume_reuses_scored_execution_failure_without_rerun(tmp_path, runner_checkpoint_only):
    calls = []

    def run(request):
        calls.append(request.task.task_id)
        return TaskResult("failed", score=0.0, error="cli_error",
                          metrics={"verifier": {"status": "completed", "reward": 0.0}})

    options = baseline_options(tmp_path)
    first = run_baseline(**options, run_task=run)
    original_results = {}
    for record in first["task_runs"]:
        directory = Path(record["output_dir"])
        result_file = directory / "task_result.json"
        original_results[result_file] = result_file.read_bytes()
        if runner_checkpoint_only:
            # Simulate a crash after runner_result, before controller task_result.
            write_json_atomic(directory / "runner_result.json",
                              {key: record[key] for key in TaskResult.__dataclass_fields__})
            result_file.unlink()
    second = run_baseline(**options, run_task=run, resume=True)
    assert calls == ["one", "two"]
    assert second["results"]["scored_runs"] == second["results"]["failed"] == 2
    assert second["results"]["mean_observed_score"] == 0.0
    assert all(row["score"] == 0.0 and row["status"] == "failed" for row in second["task_runs"])
    if not runner_checkpoint_only:
        assert all(path.read_bytes() == contents for path, contents in original_results.items())


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_interrupted_task_is_archived_and_prior_spend_retained(tmp_path, execution_mode):
    calls = []

    def run(request):
        calls.append(request.task.task_id)
        if request.task.task_id == "two" and len(calls) == 2:
            relay = request.output_dir / execution_mode / "relay"
            relay.mkdir(parents=True)
            write_json_atomic(request.output_dir / execution_mode / "launch.json", {"rates": {"currency": "USD"}})
            rows = [{"event": "provider_attempt_started", "attempt": 1},
                    {"event": "provider_attempt_finished", "attempt": 1,
                     "usage": {"total_tokens": 123}, "estimated_cost": ".7"},
                    {"event": "provider_attempt_started", "attempt": 2}]
            (relay / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + '\n{"partial":')
            raise KeyboardInterrupt
        return TaskResult("completed", score=.5, cost_usd=.1)

    options = baseline_options(tmp_path)
    options["execution_mode"] = execution_mode
    with pytest.raises(KeyboardInterrupt):
        run_baseline(**options, run_task=run)
    with pytest.raises(RuntimeError, match="retry-interrupted"):
        run_baseline(**options, run_task=run, resume=True)
    assert calls == ["one", "two"]
    result = run_baseline(**options, run_task=run, resume=True, retry_interrupted=True)
    assert calls == ["one", "two", "two"]
    assert result["results"]["known_cost_usd"] == pytest.approx(.9)
    assert result["results"]["runs_with_unknown_cost"] == 1
    assert result["results"]["interrupted_attempts"] == 1
    assert result["results"]["runs_with_retries"] == 1
    prior = result["task_runs"][1]["interrupted_attempts"][0]
    assert prior["usage"]["total_tokens"] == 123
    assert prior["unsettled_attempts"] == 1
    assert prior["execution_mode"] == execution_mode
    assert prior["policy_applied"] == (execution_mode == "team")
    assert (Path(prior["output_dir"]) / execution_mode / "relay/events.jsonl").exists()
    again = run_baseline(**options, run_task=run, resume=True)
    assert again == result


@pytest.mark.parametrize(("initial_runtime", "changed_runtime"), [
    ({"model": "a"}, {"model": "b"}),
    ({"node_directory": "/runtime/node-v22.23.2-linux-x64"},
     {"node_directory": "/runtime/node-v22.23.2-linux-arm64"}),
])
def test_changed_settings_rejected_before_callback_and_checkpoint_unchanged(
    tmp_path, initial_runtime, changed_runtime,
):
    options = baseline_options(tmp_path)
    run_baseline(**options, run_task=lambda _: TaskResult("completed", score=0),
                 runtime_config=initial_runtime)
    path = options["output_dir"] / "summary.json"
    original = path.read_bytes()
    settings = options["output_dir"] / "run_settings.json"
    original_settings = settings.read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        run_baseline(**options, run_task=lambda _: pytest.fail("No new call"), resume=True,
                     runtime_config=changed_runtime)
    assert path.read_bytes() == original
    assert settings.read_bytes() == original_settings


def test_second_controller_cannot_enter_same_output(tmp_path):
    with experiment_session(tmp_path, {"model": "test"}):
        code = ("from pathlib import Path; from agent_company.evolution.recovery import experiment_session; "
                "\nwith experiment_session(Path(__import__('sys').argv[1]), {'model':'test'}, resume=True): pass")
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
        assert result.returncode != 0
        assert "Another controller" in result.stderr


def evolution_options(tmp_path):
    return dict(train_tasks=[TaskSpec("one", "train"), TaskSpec("two", "train")],
                test_tasks=[TaskSpec("heldout", "test")], initial_communication="C0",
                assignment_policy="A0", output_dir=tmp_path / "evolution", batch_size=1,
                baseline_mode="team", selection_mode="legacy_no_replay")


@pytest.mark.parametrize("failed_file", ["assignment_v001.md", "communication_v001.md"])
@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
def test_evolution_resumes_after_committed_pair_before_policy_write(tmp_path, monkeypatch, failed_file, failure):
    import agent_company.evolution.loop as module
    original = module.write_text_atomic
    calls, proposals = [], []

    def run(request):
        calls.append((request.task.task_id, request.communication_policy, request.assignment_policy))
        return TaskResult("completed", score=.5, cost_usd=.1)

    def propose(request):
        proposals.append(request.batch_index)
        return PolicyProposal(f"C{request.batch_index}", f"A{request.batch_index}", cost_usd=.2)

    def interrupt(path, text):
        if path.name == failed_file:
            raise failure
        original(path, text)

    options = evolution_options(tmp_path)
    monkeypatch.setattr(module, "write_text_atomic", interrupt)
    with pytest.raises(failure):
        run_stage1(**options, run_task=run, propose_policy=propose)
    monkeypatch.setattr(module, "write_text_atomic", original)
    result = run_stage1(**options, run_task=run, propose_policy=propose, resume=True)
    assert proposals == [1, 2]
    assert calls == [("one", "C0", "A0"), ("two", "C1", "A1"),
                     ("heldout", "C0", "A0"), ("heldout", "C2", "A2")]
    for version in range(3):
        assert (options["output_dir"] / "policies" / f"assignment_v{version:03d}.md").read_text() == f"A{version}"
        assert (options["output_dir"] / "policies" / f"communication_v{version:03d}.md").read_text() == f"C{version}"
    assert result["optimizer"]["known_cost_usd"] == pytest.approx(.4)
    assert result["final_policy_version"] == 2
    run_stage1(**options, run_task=lambda _: pytest.fail("No replay"),
               propose_policy=lambda _: pytest.fail("No replay"), resume=True)


def test_interrupted_generic_proposal_is_not_reissued(tmp_path):
    options = evolution_options(tmp_path)

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_stage1(**options, run_task=lambda _: TaskResult("completed", score=.2),
                   propose_policy=interrupted)
    versions = []

    def next_proposal(request):
        versions.append((request.batch_index, request.current_version))
        return PolicyProposal("new", "new assignment")

    result = run_stage1(**options, run_task=lambda _: TaskResult("completed", score=.2),
                        propose_policy=next_proposal, resume=True)
    assert versions == [(2, 0)]
    assert result["optimizer"]["failed"] == 1
    assert result["optimizer"]["runs_with_unknown_cost"] >= 1


def test_real_runner_refuses_restart_when_old_container_is_active(tmp_path, monkeypatch):
    write_json_atomic(tmp_path / "workspace.json", {"container_name": "ac-stage1-test"})
    runner = RoadmapTeamRunner(ProviderConfig("test", "https://example.invalid"), tmp_path, tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="running\n"))
    with pytest.raises(RuntimeError, match="still active"):
        runner.assert_can_restart(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="exited\n"))
    runner.assert_can_restart(tmp_path)


def test_hard_process_exit_then_resume_skips_completed_work(tmp_path):
    output = tmp_path / "process"
    code = '''
import os, sys
from pathlib import Path
from agent_company.evolution.loop import run_baseline, TaskSpec, TaskResult
root = Path(sys.argv[1])
def run(request):
    if request.task.task_id == "two":
        os._exit(77)
    return TaskResult("completed", score=.4, cost_usd=.1)
run_baseline(tasks=[TaskSpec("one","repo"),TaskSpec("two","repo")],
             communication_policy="C0", assignment_policy="A0", output_dir=root, run_task=run)
'''
    result = subprocess.run([sys.executable, "-c", code, str(output)], capture_output=True, text=True)
    assert result.returncode == 77, result.stderr
    calls = []

    def run(request):
        calls.append(request.task.task_id)
        return TaskResult("completed", score=.7)

    options = baseline_options(tmp_path)
    options["output_dir"] = output
    result = run_baseline(**options, run_task=run, resume=True, retry_interrupted=True)
    assert calls == ["two"]
    assert result["results"]["completed"] == 2


def test_runner_result_survives_interruption_before_controller_result(tmp_path):
    from dataclasses import asdict
    options = baseline_options(tmp_path)
    calls = []

    def run(request):
        calls.append(request.task.task_id)
        result = TaskResult("completed", score=.8, cost_usd=.3)
        write_json_atomic(request.output_dir / "runner_result.json", asdict(result))
        if request.task.task_id == "one":
            raise KeyboardInterrupt
        return result

    with pytest.raises(KeyboardInterrupt):
        run_baseline(**options, run_task=run)
    result = run_baseline(**options, run_task=run, resume=True)
    assert calls == ["one", "two"]
    assert result["results"]["known_cost_usd"] == pytest.approx(.6)


@pytest.mark.parametrize("first_mode,next_mode", [("single", "team"), ("team", "single")])
def test_baseline_resume_cannot_change_execution_mode(tmp_path, first_mode, next_mode):
    options = baseline_options(tmp_path)
    run_baseline(**options, execution_mode=first_mode, run_task=lambda _: TaskResult("completed", score=1))
    original = (options["output_dir"] / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        run_baseline(**options, execution_mode=next_mode, resume=True,
                     run_task=lambda _: pytest.fail("Must not reuse or run another execution mode"))
    assert (options["output_dir"] / "summary.json").read_bytes() == original


@pytest.mark.parametrize("first_mode,next_mode", [("single", "team"), ("team", "single")])
def test_evolution_resume_cannot_change_baseline_mode(tmp_path, first_mode, next_mode):
    options = evolution_options(tmp_path)
    options["baseline_mode"] = first_mode
    run_stage1(**options, run_task=lambda _: TaskResult("completed", score=1),
               propose_policy=lambda _: PolicyProposal("C1", "A1"))
    original = (options["output_dir"] / "summary.json").read_bytes()
    options["baseline_mode"] = next_mode
    with pytest.raises(ValueError, match="settings differ"):
        run_stage1(**options, resume=True, run_task=lambda _: pytest.fail("No task"),
                   propose_policy=lambda _: pytest.fail("No optimizer"))
    assert (options["output_dir"] / "summary.json").read_bytes() == original


def test_single_baseline_resume_ignores_unused_policy_arguments(tmp_path):
    options = baseline_options(tmp_path)
    first = run_baseline(**options, run_task=lambda _: TaskResult("completed", score=1))
    options.update(communication_policy="different unused C", assignment_policy="different unused A")
    second = run_baseline(**options, resume=True, run_task=lambda _: pytest.fail("No new call"))
    assert second == first


def test_legacy_settings_without_execution_mode_are_not_silently_reused(tmp_path):
    options = baseline_options(tmp_path)
    run_baseline(**options, execution_mode="team", run_task=lambda _: TaskResult("completed", score=1))
    path = options["output_dir"] / "run_settings.json"
    saved = json.loads(path.read_text())
    del saved["execution_mode"]
    write_json_atomic(path, saved)
    with pytest.raises(ValueError, match="Legacy checkpoint has no execution-mode"):
        run_baseline(**options, execution_mode="team", resume=True,
                     run_task=lambda _: pytest.fail("No new call"))


def test_task_checkpoint_execution_mode_must_match_request(tmp_path):
    options = baseline_options(tmp_path)
    result = run_baseline(**options, run_task=lambda _: TaskResult("completed", score=1))
    task_result = Path(result["task_runs"][0]["output_dir"]) / "task_result.json"
    record = json.loads(task_result.read_text())
    record["execution_mode"] = "team"
    write_json_atomic(task_result, record)
    with pytest.raises(ValueError, match="result identity differs"):
        run_baseline(**options, resume=True, run_task=lambda _: pytest.fail("No new call"))


def test_interrupted_usage_reads_legacy_team_paths_without_mode(tmp_path):
    relay = tmp_path / "team/relay"
    relay.mkdir(parents=True)
    write_json_atomic(tmp_path / "run_request.json", {"task": {"task_id": "legacy"}})
    write_json_atomic(tmp_path / "team/launch.json", {"rates": {"currency": "USD"}})
    write_json_atomic(relay / "attempt_1_provider_request.json", {})
    rows = [{"event": "provider_attempt_finished", "attempt": 1,
             "usage": {"total_tokens": 50}, "estimated_cost": "0.25"}]
    (relay / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = interrupted_usage(tmp_path)
    assert result["execution_mode"] == "team"
    assert result["policy_applied"] is True
    assert result["known_cost_usd"] == 0.25
    assert result["cost_usd"] is None
    assert result["usage"] == {"total_tokens": 50}
    assert result["provider_attempts"] == 1
    assert result["unsettled_attempts"] == 0
