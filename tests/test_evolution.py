import json
from pathlib import Path

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.loop import (
    PolicyProposal, PolicyReview, RunRequest, TaskResult, TaskSpec, _aggregate,
    _paired_measurements, run_baseline, run_stage1,
)


def tasks(count, repo="train-repo"):
    return [TaskSpec(f"task-{index}", repo, {"source": f"manifest/{index}"}) for index in range(count)]


def token_usage(prompt=10, completion=20, cached=3, **extra):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "cached_prompt_tokens": cached, **extra}


def test_heldout_report_separates_agent_wall_time_task_time_and_stage_tokens(tmp_path):
    def run(request):
        scale = {"train": 1, "train_replay": 2}.get(request.split,
                  3 if request.execution_mode == "single" else 4)
        runtime = {"execution": {"elapsed_seconds": 100 + scale},
                   "elapsed_seconds": 1000 + scale, "usage_complete": scale != 4}
        return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5,
                          metrics={"task": {"agent_timeout_sec": 7200}, request.execution_mode: runtime},
                          usage=token_usage(10 * scale, 20 * scale, scale),
                          cost_usd=0.1, elapsed_seconds=9000 + scale)

    result = run_stage1(
        train_tasks=tasks(1), test_tasks=tasks(1, "heldout"), initial_communication="C0",
        assignment_policy="A0", run_task=run, output_dir=tmp_path / "run",
        propose_policy=lambda _: PolicyProposal("C1", "A1", usage=token_usage(100, 200), cost_usd=0.2),
        review_policy=lambda _: PolicyReview("prefer_candidate", "Advisory review",
                                              usage=token_usage(7, 8, 0, unknown_usage_attempts=1)))
    pair = result["paired_test_results"][0]
    assert pair["baseline_score"] == pair["final_score"] == 0.5
    assert pair["score_difference"] == 0 and pair["comparable"]
    for arm, scale in (("baseline", 3), ("final", 4)):
        assert pair[f"{arm}_agent_elapsed_seconds"] == 100 + scale
        assert pair[f"{arm}_task_elapsed_seconds"] == 9000 + scale
        assert pair[f"{arm}_wall_time_budget_seconds"] == 7200
        for name, value in token_usage(10 * scale, 20 * scale, scale).items():
            assert pair[f"{arm}_{name}"] == value
    assert pair["baseline_usage_complete"] and not pair["final_usage_complete"]
    assert pair["token_usage_scope"] == "current_attempt_only"
    for stage, total in {"train": 30, "replay": 60, "optimizer": 300, "reviewer": 15,
                         "test_baseline": 90, "test_final": 120}.items():
        assert result[stage]["known_tokens"]["total_tokens"] == total
        assert result[stage]["records_with_unknown_usage"] == (1 if stage in {"reviewer", "test_final"} else 0)
    # Completeness is independent of whether a pricing table/cost is available.
    assert result["reviewer"]["runs_with_unknown_cost"] == 1
    assert result["test_runs"][1]["usage"] == token_usage(40, 80, 4)


@pytest.mark.parametrize("usage,metrics,complete", [
    (token_usage(), {}, True),
    (token_usage(0, 0, 0), {}, True),
    (token_usage(unknown_usage_attempts=1), {}, False),
    (token_usage(calls_missing_usage=1), {}, False),
    (token_usage(), {"team": {"usage_complete": False}}, False),
    ({"total_tokens": 30}, {}, False),
    ({}, {}, False),
])
def test_report_usage_completeness_does_not_infer_missing_tokens_or_use_cost(usage, metrics, complete):
    row = {"status": "completed", "execution_mode": "team", "usage": usage,
           "metrics": metrics, "cost_usd": None, "score": 0.5}
    paired = _paired_measurements(row)
    assert paired["usage_complete"] is complete
    assert paired["agent_elapsed_seconds"] is None and paired["wall_time_budget_seconds"] is None
    assert paired["task_elapsed_seconds"] is None
    if "prompt_tokens" not in usage:
        assert paired["prompt_tokens"] is None
    aggregate = _aggregate([row])
    assert aggregate["usage_complete"] is complete
    assert aggregate["records_with_unknown_usage"] == (0 if complete else 1)
    assert aggregate["known_tokens"]["total_tokens"] == usage.get("total_tokens", 0)


def test_report_interrupted_tokens_are_separate_from_current_attempt():
    historical = {"status": "interrupted", "usage": token_usage(100, 200), "unsettled_attempts": 1}
    current = {"status": "completed", "score": 0.5, "cost_usd": None, "usage": token_usage(),
               "interrupted_attempts": [historical]}
    aggregate = _aggregate([current])
    assert aggregate["known_tokens"]["total_tokens"] == 30 and aggregate["usage_complete"]
    assert aggregate["token_usage_scope"] == "current_attempt_only"
    old = aggregate["interrupted_token_usage"]
    assert old["attempts"] == 1 and old["known_tokens"]["total_tokens"] == 300
    assert old["records_with_unknown_usage"] == 1 and not old["usage_complete"]
    assert current["usage"] == token_usage()  # Reporting never rewrites original attempt usage.


def test_sequential_batches_pairing_and_optimizer_sees_only_training(tmp_path):
    requests, proposals = [], []
    test_root = tmp_path / "experiment" / "test"

    def run(request):
        requests.append(request)
        assert request.fresh_start
        assert list(request.output_dir.iterdir()) == [request.output_dir / "run_request.json"]
        (request.output_dir / "trace.txt").write_text(f"Evidence for {request.task.task_id}")
        assert request.assignment_policy == f"assignment-v{request.policy_version}"
        assert request.communication_policy == f"communication-v{request.policy_version}"
        return TaskResult("completed", score=0.5, usage={"total_tokens": 20}, cost_usd=0.01)

    def propose(request):
        proposals.append(request)
        assert not test_root.exists()
        assert all(row["split"] == "train" for row in request.batch_results)
        assert "test_tasks" not in request.as_prompt_data()
        assert request.assignment_policy == f"assignment-v{request.current_version}"
        assert request.communication_policy == f"communication-v{request.current_version}"
        with pytest.raises(PermissionError):
            request.evidence.read_file(str(test_root / "secret.txt"))
        paths = [item["path"] for item in request.evidence.list_files()["files"]]
        assert any(path.endswith("trace.txt") for path in paths)
        return PolicyProposal(f"communication-v{request.current_version + 1}",
                              f"assignment-v{request.current_version + 1}",
                              rationale="Use evidence", cost_usd=0.02)

    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=tasks(5), test_tasks=tasks(2, "heldout-repo"),
                        initial_communication="communication-v0",
                        assignment_policy="assignment-v0",
                        run_task=run, propose_policy=propose,
                        output_dir=tmp_path / "experiment", batch_size=2,
                        test_repetitions=2, seed=123, baseline_mode="team")
    train = [request for request in requests if request.split == "train"]
    assert [request.policy_version for request in train] == [0, 0, 1, 1, 2]
    assert [len(request.batch_results) for request in proposals] == [2, 2, 1]
    assert result["final_policy_version"] == 3
    assert result["evolved_strategies"] == ["assignment", "communication"]
    assert result["baseline_mode"] == "team"
    assert all(request.execution_mode == "team" for request in requests)
    for version in range(4):
        for strategy in ("assignment", "communication"):
            path = tmp_path / "experiment/policies" / f"{strategy}_v{version:03d}.md"
            assert path.read_text() == f"{strategy}-v{version}"
    assert result["final_policy_had_training_rollout"] is False
    assert len(requests) == 5 + 2 * 2 * 2
    assert len({request.output_dir for request in requests}) == len(requests)
    assert len(result["paired_test_results"]) == 4
    for pair in result["paired_test_results"]:
        rows = [request for request in requests if request.pair_id == pair["pair_id"]]
        assert len(rows) == 2
        assert {request.policy_version for request in rows} == {0, 3}
        assert len({request.seed for request in rows}) == 1
        assert pair["score_difference"] == 0
        assert pair["baseline_execution_mode"] == "team"
        assert pair["baseline_policy_applied"] is True
        assert pair["baseline_score"] == pair["initial_score"]
    assert result["optimizer"]["known_cost_usd"] == pytest.approx(0.06)
    assert result["train"]["known_cost_usd"] == pytest.approx(0.05)
    saved = json.loads((tmp_path / "experiment/proposals/batch_001/result.json").read_text())
    assert saved["assignment_policy"] == "assignment-v1"
    assert saved["communication_policy"] == "communication-v1"


def test_repository_split_rejected_before_any_callback(tmp_path):
    def unexpected(*_):
        pytest.fail("No callback should execute for an overlapping split")

    with pytest.raises(ValueError, match="Repository split"):
        run_stage1(train_tasks=[TaskSpec("a", "same")], test_tasks=[TaskSpec("b", "same")],
                   initial_communication="initial", assignment_policy="fixed",
                   run_task=unexpected, propose_policy=unexpected, output_dir=tmp_path)


def test_explicit_task_split_allows_same_repository_but_not_same_task(tmp_path):
    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=[TaskSpec("a", "same")], test_tasks=[TaskSpec("b", "same")],
                        split_mode="task", initial_communication="initial", assignment_policy="fixed",
                        run_task=lambda _: TaskResult("completed", score=0),
                        propose_policy=lambda _: PolicyProposal("final", "new assignment"), output_dir=tmp_path)
    assert result["split_mode"] == "task"
    assert result["test_baseline"]["scored_runs"] == 1
    assert result["test_baseline"]["mean_observed_score"] == 0
    with pytest.raises(ValueError, match="both train and test"):
        run_stage1(train_tasks=[TaskSpec("a", "same")], test_tasks=[TaskSpec("a", "same")],
                   split_mode="task", initial_communication="initial", assignment_policy="fixed",
                   run_task=lambda _: None, propose_policy=lambda _: None, output_dir=tmp_path / "other")


def test_failures_keep_cost_and_do_not_become_zero_or_new_policy(tmp_path):
    class KnownFailure(RuntimeError):
        usage = {"total_tokens": 100}
        cost_usd = 0.04
        elapsed_seconds = 0.7
        output_dir = tmp_path / "optimizer_raw"

    versions, proposals = [], []

    def run(request):
        if request.split == "train":
            versions.append(request.policy_version)
            if request.task.task_id == "task-0":
                raise KnownFailure("runner unavailable")
        if request.split == "test" and request.policy_version == 1:
            return TaskResult("failed", error="evaluation unavailable")
        return TaskResult("completed", score=0.25, cost_usd=0.01)

    def propose(request):
        proposals.append(request)
        if request.batch_index == 1:
            raise KnownFailure("optimizer context limit")
        assert request.change_history[0]["status"] == "proposal_failed"
        return PolicyProposal("final", "new assignment", cost_usd=0.02)

    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=tasks(2), test_tasks=tasks(1, "test-repo"), batch_size=1,
                        initial_communication="initial", assignment_policy="fixed",
                        run_task=run, propose_policy=propose, output_dir=tmp_path / "run")
    assert versions == [0, 0]
    assert result["status"] == "completed_with_failures"
    assert result["proposal_failures"] == 1
    assert result["train"]["mean_observed_score"] == 0.25
    assert result["train"]["failed"] == 1
    assert result["training_runs"][0]["score"] is None
    assert result["train"]["known_cost_usd"] == pytest.approx(0.05)
    assert result["optimizer"]["known_cost_usd"] == pytest.approx(0.06)
    assert result["optimizer_runs"][0]["usage"] == {"total_tokens": 100}
    assert result["optimizer_runs"][0]["output_dir"] == str(KnownFailure.output_dir)
    assert result["paired_test_results"][0]["score_difference"] is None
    assert result["paired_test_results"][0]["final_score"] is None


def test_execution_failure_with_official_score_reaches_outer_aggregate_and_pairs(tmp_path):
    def run(request):
        reward = 0.0 if request.execution_mode == "single" else 0.75
        return TaskResult("failed", score=reward, error="cli_error",
                          metrics={"verifier": {"status": "completed", "reward": reward}},
                          usage=token_usage(), cost_usd=0.1)

    def propose(request):
        assert request.batch_results[0]["status"] == "failed"
        assert request.batch_results[0]["score"] == 0.75
        assert request.batch_results[0]["official_result"]["raw_score"] == 0.75
        return PolicyProposal("new communication", "new assignment")

    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=tasks(1),
                        test_tasks=tasks(1, "test-repo"), baseline_mode="single",
                        initial_communication="initial", assignment_policy="fixed",
                        run_task=run, propose_policy=propose, output_dir=tmp_path / "run")
    assert result["status"] == "completed_with_failures"
    assert result["proposal_failures"] == 0
    for key, reward in (("train", 0.75), ("test_baseline", 0.0), ("test_final", 0.75)):
        aggregate = result[key]
        assert aggregate["failed"] == aggregate["scored_runs"] == aggregate["scored_failed_runs"] == 1
        assert aggregate["unscored_runs"] == 0
        assert aggregate["mean_observed_score"] == reward
    pair = result["paired_test_results"][0]
    assert pair["baseline_status"] == pair["final_status"] == "failed"
    assert pair["baseline_score"] == 0.0 and pair["final_score"] == 0.75
    assert pair["score_difference"] == 0.75 and pair["comparable"]


def test_score_mean_includes_scored_execution_failures_and_reports_missing_coverage():
    rows = [{"status": "completed", "score": 1.0},
            {"status": "failed", "score": 0.0},
            {"status": "failed", "score": None}]
    result = _aggregate(rows)
    assert result["mean_observed_score"] == 0.5
    assert result["scored_runs"] == 2 and result["scored_failed_runs"] == 1
    assert result["unscored_runs"] == 1 and result["failed"] == 2


def test_evidence_restricts_paths_and_paginates(tmp_path):
    train = tmp_path / "train"
    current, future = train / "batch_001", train / "batch_002"
    current.mkdir(parents=True)
    future.mkdir()
    test = tmp_path / "test"
    test.mkdir()
    (test / "answer.txt").write_text("HIDDEN ANSWER")
    (future / "later.txt").write_text("FUTURE TRAIN DATA")
    (current / "trace.txt").write_text("needle one\nneedle two\nneedle three\n" + "x" * 20000)
    (current / "escape.txt").symlink_to(test / "answer.txt")
    evidence = TrainingEvidence(train, [current])
    for path in ("../test/answer.txt", str(test / "answer.txt"), "batch_001/escape.txt", "batch_002/later.txt"):
        with pytest.raises(PermissionError):
            evidence.read_file(path)
    files = evidence.list_files()
    assert [item["path"] for item in files["files"]] == ["batch_001/trace.txt"]
    page = evidence.read_file("batch_001/trace.txt", limit=100000)
    assert len(page["text"]) == 12000
    assert page["next_offset"] == 12000
    rest = evidence.read_file("batch_001/trace.txt", offset=page["next_offset"])
    assert rest["next_offset"] is None
    search = evidence.search("needle", limit=2)
    assert len(search["matches"]) == 2
    assert search["next_offset"] == 2
    last = evidence.search("needle", offset=search["next_offset"], limit=2)
    assert len(last["matches"]) == 1
    assert last["matches"][0]["line"] == 3
    assert last["next_offset"] is None


def test_task_summary_bounds_large_metrics_but_keeps_raw_evidence(tmp_path):
    def run(request):
        return TaskResult("completed", score=1, metrics={"large": "y" * 50000})

    def propose(request):
        assert len(request.batch_results[0]["metrics"]["large"]) < 1300
        first = request.evidence.read_file(request.batch_results[0]["result_file"])
        assert first["next_offset"] is not None
        return PolicyProposal("final", "new assignment")

    run_stage1(selection_mode="legacy_no_replay", train_tasks=tasks(1), test_tasks=tasks(1, "test-repo"),
               initial_communication="initial", assignment_policy="fixed",
               run_task=run, propose_policy=propose, output_dir=tmp_path)


@pytest.mark.parametrize("policies", [
    {"communication_policy": " ", "assignment_policy": "A1"},
    {"communication_policy": "C1", "assignment_policy": " "},
    {"communication_policy": "C1", "assignment_policy": None},
    {"communication_policy": "C1"},
    {"assignment_policy": "A1"},
])
def test_invalid_proposal_preserves_spent_usage_and_both_policies(tmp_path, policies):
    seen = []

    def run(request):
        seen.append((request.communication_policy, request.assignment_policy))
        return TaskResult("completed", score=0.5)

    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=tasks(1), test_tasks=tasks(1, "test-repo"),
                        initial_communication="initial", assignment_policy="fixed",
                        run_task=run,
                        propose_policy=lambda _: {**policies, "usage": {"total_tokens": 12}, "cost_usd": 0.02},
                        output_dir=tmp_path, baseline_mode="team")
    assert result["final_policy_version"] == 0
    assert result["optimizer"]["failed"] == 1
    assert result["optimizer"]["known_cost_usd"] == 0.02
    assert result["optimizer_runs"][0]["usage"] == {"total_tokens": 12}
    assert seen == [("initial", "fixed")] * 3
    assert not (tmp_path / "policies/assignment_v001.md").exists()
    assert not (tmp_path / "policies/communication_v001.md").exists()


def test_nested_metrics_do_not_expand_optimizer_overview(tmp_path):
    metrics = {f"group-{index}": {f"value-{inner}": "x" * 1000 for inner in range(24)}
               for index in range(24)}

    def propose(request):
        assert len(json.dumps(request.batch_results[0])) < 6000
        assert "result_file" in request.batch_results[0]
        return PolicyProposal("final", "new assignment")

    run_stage1(selection_mode="legacy_no_replay", train_tasks=tasks(1), test_tasks=tasks(1, "test-repo"),
               initial_communication="initial", assignment_policy="fixed",
               run_task=lambda _: TaskResult("completed", score=1, metrics=metrics),
               propose_policy=propose, output_dir=tmp_path)


def test_evolution_defaults_to_single_baseline_against_final_team(tmp_path):
    requests, proposals = [], []

    def run(request):
        requests.append(request)
        if request.execution_mode == "single":
            assert request.split == "test"
            assert request.communication_policy == request.assignment_policy == ""
            assert request.policy_version == 0
            return TaskResult("completed", score=0.25)
        assert request.communication_policy == f"C{request.policy_version}"
        assert request.assignment_policy == f"A{request.policy_version}"
        return TaskResult("completed", score=0.75)

    def propose(request):
        proposals.append(request)
        assert all(row["execution_mode"] == "team" for row in request.batch_results)
        assert all(row["policy_applied"] for row in request.batch_results)
        return PolicyProposal(f"C{request.current_version + 1}", f"A{request.current_version + 1}")

    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=tasks(3), test_tasks=tasks(2, "heldout"),
                        initial_communication="C0", assignment_policy="A0",
                        run_task=run, propose_policy=propose, output_dir=tmp_path,
                        batch_size=2, test_repetitions=2, seed=123)
    assert [r.execution_mode for r in requests[:3]] == ["team"] * 3
    assert [r.policy_version for r in requests[:3]] == [0, 0, 1]
    assert len(proposals) == 2
    assert result["baseline_mode"] == "single"
    assert result["baseline_arm"] == "baseline"
    assert result["baseline_label"] == "single_agent_without_team_policy"
    assert result["training_execution_mode"] == result["final_execution_mode"] == "team"
    assert "test_initial" not in result
    assert result["test_baseline"]["mean_observed_score"] == 0.25
    assert result["test_final"]["mean_observed_score"] == 0.75
    assert [r.execution_mode for r in requests[3:]] == [
        "single", "team", "team", "single", "team", "single", "single", "team"]
    for pair in result["paired_test_results"]:
        assert pair["baseline_execution_mode"] == "single"
        assert pair["baseline_policy_applied"] is False
        assert pair["final_execution_mode"] == "team"
        assert pair["final_policy_applied"] is True
        assert pair["score_difference"] == 0.5
        assert "initial_score" not in pair
        paired_requests = [r for r in requests if r.pair_id == pair["pair_id"]]
        assert len(paired_requests) == 2
        assert len({r.seed for r in paired_requests}) == 1
    for row in result["test_runs"]:
        assert row["policy_applied"] == (row["execution_mode"] == "team")
        saved = json.loads((Path(row["output_dir"]) / "run_request.json").read_text())
        assert saved["execution_mode"] == row["execution_mode"]


@pytest.mark.parametrize("supplied", [{}, {"communication_policy": "unused C", "assignment_policy": "unused A"}])
def test_single_baseline_needs_no_team_policy_or_optimizer(tmp_path, supplied):
    requests = []

    def run(request):
        requests.append(request)
        assert request.execution_mode == "single"
        assert request.assignment_policy == request.communication_policy == ""
        return TaskResult("completed", score=1)

    result = run_baseline(tasks=tasks(1), run_task=run, output_dir=tmp_path,
                          repetitions=2, seed=9, **supplied)
    assert len(requests) == 2
    assert [r.seed for r in requests] == [9, 10]
    assert result["execution_mode"] == "single"
    assert result["policy_applied"] is False
    assert result["optimizer_invoked"] is False
    assert all(row["execution_mode"] == "single" and not row["policy_applied"] for row in result["task_runs"])
    settings = json.loads((tmp_path / "run_settings.json").read_text())
    assert settings["assignment"] == settings["communication"] == ""
    assert settings["execution_mode"] == "single"
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["execution_mode"] == "single"
    assert progress["policy_applied"] is False
    assert not (tmp_path / "proposals").exists()


def test_explicit_team_baseline_requires_and_applies_both_policies(tmp_path):
    with pytest.raises(ValueError, match="Both policies"):
        run_baseline(tasks=tasks(1), execution_mode="team", run_task=lambda _: pytest.fail("No run"),
                     output_dir=tmp_path)

    requests = []

    def run(request):
        requests.append(request)
        return TaskResult("completed", score=1)

    result = run_baseline(tasks=tasks(1), execution_mode="team", assignment_policy="A0",
                          communication_policy="C0", run_task=run, output_dir=tmp_path)
    assert requests[0].execution_mode == "team"
    assert (requests[0].assignment_policy, requests[0].communication_policy) == ("A0", "C0")
    assert result["execution_mode"] == "team"
    assert result["policy_applied"] is True


@pytest.mark.parametrize("mode", ["", "other", None, True])
def test_execution_modes_are_validated_before_work(tmp_path, mode):
    with pytest.raises(ValueError, match="execution_mode"):
        RunRequest(tasks(1)[0], "baseline", 0, "", "", tmp_path, execution_mode=mode)
    with pytest.raises(ValueError, match="execution_mode"):
        run_baseline(tasks=tasks(1), execution_mode=mode, output_dir=tmp_path,
                     run_task=lambda _: pytest.fail("No run"))
    with pytest.raises(ValueError, match="baseline_mode"):
        run_stage1(train_tasks=tasks(1), test_tasks=tasks(1, "heldout"), baseline_mode=mode,
                   initial_communication="C0", assignment_policy="A0", output_dir=tmp_path,
                   run_task=lambda _: pytest.fail("No run"), propose_policy=lambda _: pytest.fail("No proposal"))
