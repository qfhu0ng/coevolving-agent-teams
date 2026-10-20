"""Synthetic, local evidence only: no provider or Docker execution."""
import json

import pytest

from agent_company.evolution.efficiency import compact_efficiency, efficiency_report
from agent_company.evolution.loop import PolicyProposal, PolicyReview, TaskResult, TaskSpec, run_stage1
from agent_company.evolution.optimizer import SYSTEM as OPTIMIZER_SYSTEM
from agent_company.evolution.reviewer import SYSTEM as REVIEWER_SYSTEM
from agent_company.evolution.selection import SelectionConfig, select_candidate


def usage(n=1):
    return {"prompt_tokens": 10 * n, "completion_tokens": 2 * n,
            "cached_prompt_tokens": 8 * n, "total_tokens": 12 * n}


def result_row(*, score=1, cost=1, elapsed=100, budget=7200, **extra):
    return {"task_id": "task", "repo_id": "repo", "status": "completed",
            "score": score, "cost_usd": cost, "execution_mode": "team", "usage": usage(),
            "metrics": {"task": {"agent_timeout_sec": budget}, "team": {
                "execution": {"elapsed_seconds": elapsed}, "usage_complete": True}}, **extra}


@pytest.mark.parametrize("score,cost,elapsed,decision,reason", [
    (1, 1, 50, "accept", "time_reduced_at_equal_cost"),
    (1, 1, 100, "reject", "cost_and_time_not_improved"),
    (1, 1, 200, "reject", "cost_and_time_not_improved"),
    (1, 2, 1, "reject", "cost_not_reduced"),
    (1, 0.5, 200, "accept", "cost_reduced_without_clear_score_gain"),
    (0.99, 0, 1, "reject", "score_regressed_within_tolerance"),
    (1, None, 1, "inconclusive", "incomplete_cost_evidence"),
    (1, 1, None, "inconclusive", "incomplete_time_evidence_at_equal_cost"),
])
def test_quality_cost_time_order(score, cost, elapsed, decision, reason):
    comparison = select_candidate([result_row()], [result_row(score=score, cost=cost, elapsed=elapsed)],
                                  config=SelectionConfig(quality_tolerance=0.02, efficiency_metric="cost_usd"))
    assert (comparison["decision"], comparison["reason"]) == (decision, reason)
    assert comparison["wall_time_caps_verified"]
    assert comparison["pairs"][0]["candidate_measurements"]["total_tokens"] == 12
    json.dumps(comparison, allow_nan=False)


def test_score_gain_beats_cost_and_time_but_not_unequal_budget():
    new = result_row(score=1, cost=100, elapsed=200)
    assert select_candidate([result_row(score=0.5)], [new])["decision"] == "accept"
    new["metrics"]["task"]["agent_timeout_sec"] = 9000
    assert select_candidate([result_row(score=0.5)], [new])["reason"] == "incomparable_wall_time_caps"


def test_time_must_not_fall_back_to_whole_task_time_or_missing_budget():
    new = result_row(elapsed=None, elapsed_seconds=1)
    assert select_candidate([result_row()], [new])["reason"] == "incomplete_time_evidence_at_equal_tokens"
    assert select_candidate([result_row(budget=None)], [result_row(elapsed=1, budget=None)])["decision"] == "inconclusive"
    assert select_candidate([result_row()], [result_row(budget=None)])["reason"] == "incomparable_wall_time_caps"


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "1"])
def test_invalid_times_never_win(value):
    assert select_candidate([result_row()], [result_row(elapsed=value)])["decision"] == "inconclusive"


def test_interrupted_time_never_wins_and_failed_score_is_not_zero():
    new = result_row(elapsed=1, interrupted_attempts=[{"elapsed_seconds": 300}])
    assert select_candidate([result_row()], [new])["reason"] == "interrupted_token_evidence"
    new.update(status="failed", score=None)
    assert select_candidate([result_row()], [new])["reason"] == "incomplete_score_evidence"


def write_trace(directory, *, missing_usage=False):
    relay = directory / "team/relay"
    relay.mkdir(parents=True)
    events = []
    for index, name in enumerate(("master", "worker", "worker"), 1):
        events.extend([{"event": "provider_attempt_started", "attempt": index, "role": name},
                       {"event": "provider_attempt_finished", "attempt": index, "role": name,
                        "usage": None if missing_usage and index == 3 else usage(), "elapsed_seconds": 5}])
        arguments = {"command": "pytest tests/test_feature.py", "directory": "/app"}
        message = {"role": "assistant", "reasoning_content": "DO_NOT_EXPOSE_REASONING",
                   "content": "Do not treat narrative as measurement", "tool_calls": [{"id": str(index),
                   "function": {"name": "send_message" if index == 1 else "run_shell_command",
                                "arguments": json.dumps(arguments)}}]}
        response = {"choices": [{"message": message}], "usage": usage()}
        (relay / f"attempt_{index:05d}_{name}_qwen_response.json").write_text(json.dumps(response))
        # Neither the provider copy nor arbitrary numbers of history copies count.
        (relay / f"attempt_{index:05d}_{name}_provider_response.json").write_text(json.dumps(response))
        (relay / f"attempt_{index:05d}_{name}_qwen_request.json").write_text(json.dumps([message] * 15))
    (relay / "events.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
    row = result_row()
    row["usage"] = usage(2 if missing_usage else 3)
    row["metrics"]["team"].update(provider_attempts=3, usage_complete=not missing_usage)
    return row


def test_efficiency_counts_delivered_calls_once_with_provenance_not_history(tmp_path):
    row = write_trace(tmp_path)
    report = efficiency_report(row, tmp_path)
    assert report["coverage"]["attempts_complete"] and report["coverage"]["role_usage_complete"]
    assert report["by_role"][0]["known_tokens"] == usage()
    assert report["by_role"][1]["known_tokens"] == usage(2)
    assert report["by_role"][1]["known_provider_seconds"] == 10
    assert report["by_role"][1]["tool_calls"] == {"run_shell_command": 2}
    assert report["repeated_shell_group_count"] == 1
    assert report["repeated_shell_examples"][0]["count"] == 2
    assert report["repeated_shell_examples"][0]["references"][0]["path"].endswith("00002_worker_qwen_response.json")
    assert "DO_NOT_EXPOSE_REASONING" not in json.dumps(report)
    assert "pytest tests" not in json.dumps(report)  # references, not raw shell content
    assert compact_efficiency(report)["by_role"][0]["shell_calls"] == 2


def test_missing_usage_events_and_response_are_incomplete_not_zero_work(tmp_path):
    row = write_trace(tmp_path, missing_usage=True)
    report = efficiency_report(row, tmp_path)
    assert not report["coverage"]["role_usage_complete"]
    assert report["by_role"][1]["attempts_with_missing_usage"] == 1
    (tmp_path / "team/relay/attempt_00003_worker_qwen_response.json").unlink()
    assert not efficiency_report(row, tmp_path)["coverage"]["attempts_complete"]
    absent = efficiency_report(row, tmp_path / "missing")
    assert absent["by_role"] == [] and not absent["coverage"]["events_present"]
    assert not absent["coverage"]["attempts_complete"]


def test_symlink_and_interrupted_evidence_not_included(tmp_path):
    task = tmp_path / "task"
    row = write_trace(task)
    outside = tmp_path / "heldout.json"
    outside.write_text('{"choices": [{"message": {"tool_calls": []}}]}')
    (task / "team/relay/attempt_00004_spy_qwen_response.json").symlink_to(outside)
    write_trace(task / ".interrupted/old")
    report = efficiency_report(row, task)
    assert report["coverage"]["observed_delivered_responses"] == 3
    assert report["coverage"]["attempts_complete"]


def test_duplicate_and_malformed_events_do_not_inflate_usage(tmp_path):
    row = write_trace(tmp_path)
    path = tmp_path / "team/relay/events.jsonl"
    lines = path.read_text().splitlines()
    with path.open("a") as stream:
        stream.write(lines[-1] + "\nnot json\n")
    report = efficiency_report(row, tmp_path)
    assert report["by_role"][1]["known_tokens"] == usage(2)
    assert report["coverage"]["parse_issues"] == {"duplicate_finished_events": 1, "invalid_event_lines": 1}
    assert not report["coverage"]["attempts_complete"]


def test_controller_passes_objective_and_current_evidence_to_both_roles_and_resumes(tmp_path):
    captures = []

    def run(request):
        row = write_trace(request.output_dir)
        row["metrics"]["team"]["execution"]["elapsed_seconds"] = 50 if request.split == "train_replay" else 100
        return TaskResult("completed", score=1, metrics=row["metrics"], usage=row["usage"], cost_usd=1)

    def propose(request):
        captures.append(request.as_prompt_data())
        report_path = request.batch_results[0]["efficiency"]["report_file"]
        report = json.loads(request.evidence.read_file(report_path)["text"])
        assert report["coverage"]["role_usage_complete"]
        return PolicyProposal("C1", "A1", rationale="Test removing redundant handoffs")

    def review(request):
        captures.append(request.as_prompt_data()["incumbent"])
        return PolicyReview("uncertain", "Replay must test the hypothesis")

    opts = dict(train_tasks=[TaskSpec("train", "train-repo")], test_tasks=[TaskSpec("heldout", "test-repo")],
                initial_communication="C0", assignment_policy="A0", output_dir=tmp_path / "experiment",
                run_task=run, propose_policy=propose, review_policy=review,
                selection_config=SelectionConfig(quality_tolerance=0.02))
    result = run_stage1(**opts)
    assert result["final_policy_version"] == 1
    decision = result["selection_results"][0]["candidate_results"][0]
    assert decision["reason"] == "time_reduced_at_equal_tokens"
    assert captures[0]["selection_objective"] == captures[1]["selection_objective"]
    assert captures[0]["selection_objective"]["thresholds"]["quality_tolerance"] == 0.02
    assert all("heldout" not in f["directory"] for f in captures[0]["evidence_files"]["tasks"])
    assert run_stage1(**opts, resume=True)["final_policy_version"] == 1
    assert len(captures) == 2  # durable submissions/results recovered, no fresh calls
    assert "DELETE" in OPTIMIZER_SYSTEM and "causal hypotheses" in OPTIMIZER_SYSTEM
    assert "not\nproven wasted work" in REVIEWER_SYSTEM and "regression risk" in REVIEWER_SYSTEM


def test_diagnostic_failure_does_not_change_official_outcome(tmp_path, monkeypatch):
    from agent_company.evolution import loop as evolution

    def broken(*_):
        raise OSError("Unrelated diagnostic read failure")

    monkeypatch.setattr(evolution, "efficiency_report", broken)
    result = evolution.run_baseline(
        tasks=[TaskSpec("task", "repo")], output_dir=tmp_path / "run",
        run_task=lambda _: TaskResult("completed", score=1, cost_usd=1))
    assert result["status"] == "completed"
    saved = next((tmp_path / "run").rglob("task_result.json"))
    row = json.loads(saved.read_text())
    assert row["status"] == "completed" and row["score"] == 1
    assert row["efficiency"] == {"available": False, "error_type": "OSError"}


def test_previous_protocol_settings_cannot_be_silently_resumed(tmp_path):
    opts = dict(train_tasks=[TaskSpec("train", "repo")], test_tasks=[TaskSpec("test", "other")],
                initial_communication="C0", assignment_policy="A0", output_dir=tmp_path / "experiment",
                run_task=lambda _: TaskResult("completed", score=1, cost_usd=1),
                propose_policy=lambda _: PolicyProposal("C1", "A1"),
                review_policy=lambda _: PolicyReview("uncertain", "Unmeasured"))
    run_stage1(**opts)
    path = tmp_path / "experiment/run_settings.json"
    settings = json.loads(path.read_text())
    assert settings.pop("evolution_protocol_version") == 3
    path.write_text(json.dumps(settings))  # synthetic pre-change checkpoint, never real experiment data
    with pytest.raises(ValueError, match="Resume settings differ"):
        run_stage1(**opts, resume=True)
