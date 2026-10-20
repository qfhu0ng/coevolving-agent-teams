"""Offline controller tests for advisory review, fresh replay, and selection."""

from collections import Counter
import json
import math
from pathlib import Path

import pytest

from agent_company.runtime_io import write_json_atomic
from agent_company.evolution.loop import (
    PolicyProposal, PolicyReview, TaskResult, TaskSpec, run_stage1,
)
from agent_company.evolution.selection import SelectionConfig


def options(tmp_path, count=2, **overrides):
    return {
        "train_tasks": [TaskSpec(f"train-{index}", "train-repo") for index in range(count)],
        "test_tasks": [TaskSpec("heldout", "test-repo")],
        "initial_communication": "C0", "assignment_policy": "A0",
        "output_dir": tmp_path / "experiment", **overrides,
    }


def propose(request):
    return PolicyProposal(f"C{request.batch_index}", f"A{request.batch_index}",
                          rationale="Use current training evidence", cost_usd=0.2)


def review(_):
    return PolicyReview("prefer_candidate", "The candidate plausibly addresses the evidence.", cost_usd=0.3)


class SpentFailure(RuntimeError):
    def __init__(self, message, cost):
        super().__init__(message)
        self.cost_usd = cost
        self.usage = {"total_tokens": 17}
        self.elapsed_seconds = 0.01


@pytest.mark.parametrize("score,cost,decision,version", [
    (0.75, 0.1, "accept", 1),
    (0.25, 0.01, "reject", 0),
    (0.5, 0.05, "inconclusive", 0),  # Cheaper USD alone cannot establish a token win.
    (0.5, 0.1, "inconclusive", 0),
    (0.5, None, "inconclusive", 0),
])
def test_default_protocol_selects_official_replay_not_review(tmp_path, score, cost, decision, version):
    runs, reviews = [], []

    def run(request):
        runs.append(request)
        return TaskResult("completed", score=score if request.split == "train_replay" else 0.5,
                          cost_usd=cost if request.split == "train_replay" else 0.1)

    def negative_review(request):
        reviews.append(request)
        return PolicyReview("prefer_incumbent", "Advisory concern, not an official score.", cost_usd=0.3)

    opts = options(tmp_path)
    result = run_stage1(**opts, run_task=run, propose_policy=propose, review_policy=negative_review)
    assert result["selection_mode"] == "review_replay" and result["batch_size"] == 5
    assert result["final_policy_version"] == version
    assert len(runs) == 6 and len(reviews) == 1
    assert [request.split for request in runs] == ["train", "train", "train_replay", "train_replay", "test", "test"]
    selection = result["selection_results"][0]
    assert selection["decision"] == decision and selection["heldout_used"] is False
    candidate = selection["candidate_results"][0]
    assert candidate["decision"] == decision and candidate["comparable_count"] == 2
    assert candidate["reviewer_recommendation"] == "prefer_incumbent"
    assert candidate["reviewer_advisory"] is True
    assert reviews[0].proposal_request.current_version == 0
    assert reviews[0].candidate.assignment_policy == "A1"
    assert reviews[0].candidate.communication_policy == "C1"
    assert reviews[0].candidate_index == 1
    assert reviews[0].storage_key == reviews[0].proposal_request.storage_key
    root = opts["output_dir"]
    for prefix in ("proposals", "reviews"):
        saved = json.loads((root / prefix / "batch_001/candidate_001/result.json").read_text())
        assert saved["status"] == "completed"
    assert json.loads((root / "selections/batch_001.json").read_text()) == selection
    assert result["train"]["known_cost_usd"] == pytest.approx(0.2)
    assert result["optimizer"]["known_cost_usd"] == pytest.approx(0.2)
    assert result["reviewer"]["known_cost_usd"] == pytest.approx(0.3)
    assert result["replay"]["runs"] == 2
    if cost is None:
        assert result["all_costs"]["cost_usd"] is None
        assert result["all_costs"]["records_with_unknown_cost"] == 2
    else:
        assert result["all_costs"]["cost_usd"] == pytest.approx(0.9 + 2 * cost)
    assert (root / "policies/assignment_v001.md").exists() == (decision == "accept")
    baseline, final = [request for request in runs if request.split == "test"]
    assert baseline.execution_mode == "single" and baseline.assignment_policy == baseline.communication_policy == ""
    assert final.execution_mode == "team" and final.policy_version == version


@pytest.mark.parametrize("bad_split", ["train", "train_replay"])
@pytest.mark.parametrize("bad_status", ["failed", "completed"])
def test_missing_official_score_keeps_entire_pairing_inconclusive(tmp_path, bad_split, bad_status):
    def run(request):
        if request.split == bad_split and request.task.task_id == "train-1":
            return TaskResult(bad_status, score=None, cost_usd=0.1)
        return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5, cost_usd=0.1)

    result = run_stage1(**options(tmp_path), run_task=run, propose_policy=propose, review_policy=review)
    candidate = result["selection_results"][0]["candidate_results"][0]
    assert candidate["decision"] == "inconclusive" and candidate["reason"] == "incomplete_score_evidence"
    assert candidate["pair_count"] == 2 and candidate["comparable_count"] == 1
    assert candidate["mean_score_difference"] is None
    assert result["final_policy_version"] == 0
    assert result["replay"]["runs"] == 2


def test_review_failure_retains_spend_but_does_not_skip_official_replay(tmp_path):
    def failed_review(_):
        raise SpentFailure("review transport failed", 0.3)

    result = run_stage1(**options(tmp_path),
                        run_task=lambda request: TaskResult("completed", cost_usd=0.1,
                            score=0.75 if request.split == "train_replay" else 0.5),
                        propose_policy=propose, review_policy=failed_review)
    assert result["final_policy_version"] == 1 and result["replay"]["completed"] == 2
    assert result["status"] == "completed_with_failures"
    assert result["reviewer"]["failed"] == 1
    assert result["reviewer"]["known_cost_usd"] == pytest.approx(0.3)
    assert result["reviewer_runs"][0]["usage"] == {"total_tokens": 17}
    assert result["all_costs"]["cost_usd"] == pytest.approx(1.1)
    candidate = result["selection_results"][0]["candidate_results"][0]
    assert candidate["decision"] == "accept" and candidate["reviewer_status"] == "failed"
    assert candidate["reviewer_recommendation"] is None


def test_missing_review_callback_stops_before_any_runner_or_optimizer(tmp_path):
    calls = []

    def unexpected(request):
        calls.append(request)
        raise AssertionError("No callback should execute without review_policy")

    opts = options(tmp_path)
    with pytest.raises(ValueError, match="review_policy"):
        run_stage1(**opts, run_task=unexpected, propose_policy=unexpected)
    assert calls == [] and not opts["output_dir"].exists()


def test_k_candidates_share_unchanged_incumbent_and_fresh_identical_replay_subset(tmp_path):
    runs, proposals, reviews, initial_contents = [], [], [], []

    def run(request):
        runs.append(request)
        initial_contents.append(sorted(path.name for path in request.output_dir.iterdir()))
        (request.output_dir / "trace.txt").write_text(f"{request.split}: {request.task.task_id}")
        scores = {1: 0.75, 2: 1.0, 3: 1.0, 4: 0.625, 5: 0.5, 6: 0.75}
        return TaskResult("completed", cost_usd=0.1,
                          score=scores[request.policy_version] if request.split == "train_replay" else 0.5)

    def multiple_proposals(request):
        proposals.append(request)
        version = (request.batch_index - 1) * 3 + request.candidate_index
        return PolicyProposal(f"C{version}", f"A{version}", cost_usd=0.2)

    def independent_review(request):
        reviews.append(request)
        return review(request)

    opts = options(tmp_path, count=6)
    result = run_stage1(**opts, run_task=run, propose_policy=multiple_proposals,
                        review_policy=independent_review, candidates_per_batch=3, replay_task_count=2, seed=43)
    assert result["status"] == "completed"
    assert [(request.batch_index, request.candidate_index, request.current_version) for request in proposals] == [
        (1, 1, 0), (1, 2, 0), (1, 3, 0), (2, 1, 2), (2, 2, 2), (2, 3, 2)]
    assert [(request.assignment_policy, request.communication_policy) for request in proposals] == [
        ("A0", "C0")] * 3 + [("A2", "C2")] * 3
    assert [len(request.batch_results) for request in proposals] == [5, 5, 5, 1, 1, 1]
    assert len({request.storage_key for request in proposals}) == 6
    assert [request.proposal_request for request in reviews] == proposals
    for batch_index in (1, 2):
        batch_proposals = [request for request in proposals if request.batch_index == batch_index]
        assert all(request.batch_results == batch_proposals[0].batch_results for request in batch_proposals)
        assert all(request.evidence is batch_proposals[0].evidence for request in batch_proposals)
        assert all(request.change_history == () for request in batch_proposals)
        # Even after every candidate has finished, the shared capability cannot
        # discover earlier candidates' replay or feedback from this same batch.
        assert all(item["path"].startswith(f"batch_{batch_index:03d}/task_")
                   for item in batch_proposals[0].evidence.list_files()["files"])
    assert len(result["training_runs"]) == 6 and len(result["replay_runs"]) == 9
    assert result["final_policy_version"] == 6
    assert [row["selected_candidate_index"] for row in result["selection_results"]] == [2, 3]
    assert [len(row["candidate_results"]) for row in result["selection_results"]] == [3, 3]
    assert [[candidate["pair_count"] for candidate in selection["candidate_results"]]
            for selection in result["selection_results"]] == [[2, 2, 2], [1, 1, 1]]
    third = result["selection_results"][0]["candidate_results"][2]
    assert third["decision"] == "accept"
    assert third["comparison_with_previous_candidate"]["decision"] == "inconclusive"
    assert all(contents == ["run_request.json"] for contents in initial_contents)
    assert len({request.output_dir for request in runs}) == len(runs)
    train_by_pair = {request.pair_id: request for request in runs if request.split == "train"}
    replay = [request for request in runs if request.split == "train_replay"]
    for request in replay:
        incumbent = train_by_pair[request.pair_id]
        assert request.task == incumbent.task and request.seed == incumbent.seed == 43
        assert request.fresh_start and request.execution_mode == "team"
        assert request.output_dir != incumbent.output_dir
        assert request.output_dir.relative_to(opts["output_dir"]).parts[2] == "replay"
    for version in (1, 2, 3):
        assert [r.task.task_id for r in replay if r.policy_version == version] == ["train-0", "train-1"]
    for version in (4, 5, 6):
        assert [r.task.task_id for r in replay if r.policy_version == version] == ["train-5"]


def test_failures_in_every_phase_remain_in_total_cost_and_are_not_zero_scores(tmp_path):
    def run(request):
        if request.split == "train" and request.task.task_id == "train-0":
            raise SpentFailure("train failed", 0.11)
        if request.split == "train":
            return TaskResult("completed", score=0.5, cost_usd=0.12)
        if request.split == "train_replay":
            raise SpentFailure("replay failed", 0.41)
        if request.execution_mode == "single":
            return TaskResult("completed", score=0.5, cost_usd=0.51)
        raise SpentFailure("final evaluation failed", 0.52)

    def fail_second_proposal(request):
        if request.batch_index == 2:
            raise SpentFailure("optimizer failed", 0.22)
        return PolicyProposal("C1", "A1", cost_usd=0.21)

    def fail_review(_):
        raise SpentFailure("review failed", 0.31)

    result = run_stage1(**options(tmp_path, batch_size=1), run_task=run,
                        propose_policy=fail_second_proposal, review_policy=fail_review)
    assert result["status"] == "completed_with_failures" and result["final_policy_version"] == 0
    for stage, cost in {"train": 0.23, "optimizer": 0.43, "reviewer": 0.31, "replay": 0.41,
                        "test_baseline": 0.51, "test_final": 0.52}.items():
        assert result[stage]["known_cost_usd"] == pytest.approx(cost)
    assert result["all_costs"]["cost_usd"] == pytest.approx(2.41)
    assert result["all_costs"]["records"] == 8
    assert result["all_costs"]["records_with_unknown_cost"] == 0
    for key in ("training_runs", "replay_runs", "test_runs"):
        for record in result[key]:
            if record["status"] == "failed":
                assert record["score"] is None and record["usage"] == {"total_tokens": 17}
    assert len(result["reviewer_runs"]) == len(result["replay_runs"]) == 1
    assert len(result["optimizer_runs"]) == 2
    assert result["paired_test_results"][0]["score_difference"] is None


@pytest.mark.parametrize("scope", [None, "cumulative"])
def test_evidence_scope_and_history_do_not_leak_to_proposal_or_review(tmp_path, scope):
    proposals, reviews = [], []
    paths = {}

    def run(request):
        trace = request.output_dir / "trace.txt"
        trace.write_text(f"{request.split} LOG {request.task.task_id}")
        paths[(request.split, request.task.task_id)] = trace
        return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5, cost_usd=0.1)

    def capture_proposal(request):
        proposals.append(request)
        return PolicyProposal(f"C{request.batch_index}", f"A{request.batch_index}",
                              rationale="OLD_PROPOSAL_REASON_SENTINEL" if request.batch_index == 1 else "new", cost_usd=0.2)

    def capture_review(request):
        reviews.append(request)
        return review(request)

    opts = options(tmp_path, batch_size=1)
    if scope is not None:
        opts["evidence_scope"] = scope
    result = run_stage1(**opts, run_task=run, propose_policy=capture_proposal, review_policy=capture_review)
    assert result["status"] == "completed" and len(proposals) == len(reviews) == 2
    old_train, old_replay = paths[("train", "train-0")], paths[("train_replay", "train-0")]
    current_train = paths[("train", "train-1")]
    current_replay = paths[("train_replay", "train-1")]
    old_feedback = opts["output_dir"] / "train/batch_001/feedback/candidate_001/review.json"
    for request in (proposals[1], reviews[1].proposal_request):
        assert "train-1" in request.evidence.read_file(str(current_train))["text"]
        prompt = request.as_prompt_data()
        for path in (paths[("test", "heldout")], current_replay):
            with pytest.raises(PermissionError):
                request.evidence.read_file(str(path))
        if scope is None:
            assert request.change_history == ()
            assert "OLD_PROPOSAL_REASON_SENTINEL" not in json.dumps(prompt)
            for path in (old_train, old_replay, old_feedback):
                with pytest.raises(PermissionError):
                    request.evidence.read_file(str(path))
            assert all(item["path"].startswith("batch_002/task_")
                       for item in request.evidence.list_files()["files"])
        else:
            assert len(request.change_history) == 1 and request.history_total == 1
            assert "OLD_PROPOSAL_REASON_SENTINEL" in json.dumps(prompt)
            for path in (old_train, old_replay, old_feedback):
                assert request.evidence.read_file(str(path))["text"]


def test_final_heldout_failure_cannot_change_selected_policy_or_create_outer_feedback(tmp_path):
    outer_events, run_events = [], []

    def run(request):
        run_events.append(request.split)
        if request.split == "test":
            assert len(outer_events) == 2
            return TaskResult("completed", score=1 if request.execution_mode == "single" else 0, cost_usd=0.1)
        return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5, cost_usd=0.1)

    def capture_proposal(request):
        outer_events.append("propose")
        assert "test" not in run_events
        return propose(request)

    def capture_review(request):
        outer_events.append("review")
        assert "test" not in run_events
        return review(request)

    result = run_stage1(**options(tmp_path), run_task=run, propose_policy=capture_proposal,
                        review_policy=capture_review)
    assert result["status"] == "completed"
    assert outer_events == ["propose", "review"]
    assert result["final_policy_version"] == 1
    assert result["selection_results"][0]["decision"] == "accept"
    assert result["paired_test_results"][0]["score_difference"] == -1
    assert len(result["change_history"]) == 1


def test_resume_interrupted_replay_reuses_proposal_review_and_completed_tasks(tmp_path):
    calls, proposals, reviews = [], [], []

    def run(request):
        key = request.split, request.task.task_id
        calls.append(key)
        if key == ("train_replay", "train-1") and calls.count(key) == 1:
            runtime = request.output_dir / "team"
            (runtime / "relay").mkdir(parents=True)
            write_json_atomic(runtime / "launch.json", {"rates": {"currency": "USD"}})
            events = [{"event": "provider_attempt_started", "attempt": 1},
                      {"event": "provider_attempt_finished", "attempt": 1,
                       "usage": {"total_tokens": 123}, "estimated_cost": ".7"},
                      {"event": "provider_attempt_started", "attempt": 2}]
            (runtime / "relay/events.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
            raise KeyboardInterrupt
        return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5, cost_usd=0.1)

    def capture_proposal(request):
        proposals.append(request)
        return propose(request)

    def capture_review(request):
        reviews.append(request)
        return review(request)

    opts = options(tmp_path)
    callbacks = {"run_task": run, "propose_policy": capture_proposal, "review_policy": capture_review}
    with pytest.raises(KeyboardInterrupt):
        run_stage1(**opts, **callbacks)
    with pytest.raises(RuntimeError, match="retry-interrupted"):
        run_stage1(**opts, **callbacks, resume=True)
    result = run_stage1(**opts, **callbacks, resume=True, retry_interrupted=True)
    assert len(proposals) == len(reviews) == 1
    counts = Counter(calls)
    assert counts == {("train", "train-0"): 1, ("train", "train-1"): 1,
                      ("train_replay", "train-0"): 1, ("train_replay", "train-1"): 2,
                      ("test", "heldout"): 2}
    assert result["final_policy_version"] == 1
    retried = result["replay_runs"][1]
    assert retried["cost_usd"] is None and retried["known_cost_usd"] == pytest.approx(0.8)
    assert retried["interrupted_attempts"][0]["usage"] == {"total_tokens": 123}
    assert retried["interrupted_attempts"][0]["unsettled_attempts"] == 1
    assert (Path(retried["interrupted_attempts"][0]["output_dir"]) / "team/relay/events.jsonl").is_file()
    assert result["replay"]["known_cost_usd"] == pytest.approx(0.9)
    assert result["all_costs"]["known_cost_usd"] == pytest.approx(1.8)
    assert result["all_costs"]["cost_usd"] is None
    again = run_stage1(**opts, run_task=lambda _: pytest.fail("No repeated task"),
                       propose_policy=lambda _: pytest.fail("No repeated proposal"),
                       review_policy=lambda _: pytest.fail("No repeated review"), resume=True)
    assert again == result


@pytest.mark.parametrize("changed", [
    {"candidates_per_batch": 2}, {"replay_task_count": 1}, {"evidence_scope": "cumulative"},
    {"selection_config": SelectionConfig(min_score_gain=0.125)},
])
def test_resume_cannot_silently_change_selection_settings(tmp_path, changed):
    opts = options(tmp_path)
    result = run_stage1(**opts, run_task=lambda _: TaskResult("completed", score=0.5, cost_usd=0.1),
                        propose_policy=propose, review_policy=review)
    original = (opts["output_dir"] / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        run_stage1(**opts, **changed, run_task=lambda _: pytest.fail("No repeated task"),
                   propose_policy=lambda _: pytest.fail("No repeated proposal"),
                   review_policy=review, resume=True)
    assert (opts["output_dir"] / "summary.json").read_bytes() == original
    assert result["final_policy_version"] == 0


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_resume_after_durable_proposal_before_review_does_not_reissue_proposer(tmp_path, monkeypatch, failure):
    import agent_company.evolution.loop as module

    opts = options(tmp_path)
    root = opts["output_dir"]
    review_request = root / "reviews/batch_001/candidate_001/request.json"
    original_write = module.write_json_atomic
    tasks, proposals, reviews = [], [], []

    def run(request):
        tasks.append((request.split, request.task.task_id, request.execution_mode))
        return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5, cost_usd=0.1)

    def capture_proposal(request):
        proposals.append(request)
        return propose(request)

    def capture_review(request):
        reviews.append(request)
        return review(request)

    def interrupt_before_review_request(path, value):
        if path == review_request:
            raise failure("Interrupted writing review request")
        original_write(path, value)

    callbacks = {"run_task": run, "propose_policy": capture_proposal, "review_policy": capture_review}
    monkeypatch.setattr(module, "write_json_atomic", interrupt_before_review_request)
    with pytest.raises(failure, match="Interrupted writing review request"):
        run_stage1(**opts, **callbacks)
    proposal_file = root / "proposals/batch_001/candidate_001/result.json"
    proposal_bytes = proposal_file.read_bytes()
    assert json.loads(proposal_bytes)["status"] == "completed"
    assert len(proposals) == 1 and reviews == []
    assert len(tasks) == 2 and not review_request.exists()
    monkeypatch.setattr(module, "write_json_atomic", original_write)
    result = run_stage1(**opts, **callbacks, resume=True)
    assert result["status"] == "completed" and result["final_policy_version"] == 1
    assert len(proposals) == len(reviews) == 1
    assert len(tasks) == len(set(tasks)) == 6
    assert proposal_file.read_bytes() == proposal_bytes
    assert result["optimizer"]["known_cost_usd"] == pytest.approx(0.2)
    assert result["reviewer"]["known_cost_usd"] == pytest.approx(0.3)
    assert result["all_costs"]["cost_usd"] == pytest.approx(1.1)


@pytest.mark.parametrize("failed_file", ["assignment_v001.md", "communication_v001.md"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_resume_after_durable_selection_completes_policy_pair_without_replaying_work(
    tmp_path, monkeypatch, failed_file, failure,
):
    import agent_company.evolution.loop as module

    opts = options(tmp_path)
    root = opts["output_dir"]
    original_write = module.write_text_atomic
    tasks, proposals, reviews = [], [], []

    def run(request):
        tasks.append((request.split, request.task.task_id, request.execution_mode))
        return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5, cost_usd=0.1)

    def capture_proposal(request):
        proposals.append(request)
        return propose(request)

    def capture_review(request):
        reviews.append(request)
        return review(request)

    def interrupt_adoption(path, value):
        if path == root / "policies" / failed_file:
            raise failure("Interrupted writing accepted policy")
        original_write(path, value)

    callbacks = {"run_task": run, "propose_policy": capture_proposal, "review_policy": capture_review}
    monkeypatch.setattr(module, "write_text_atomic", interrupt_adoption)
    with pytest.raises(failure, match="Interrupted writing accepted policy"):
        run_stage1(**opts, **callbacks)
    selection_file = root / "selections/batch_001.json"
    selection_bytes = selection_file.read_bytes()
    selection = json.loads(selection_bytes)
    assert selection["decision"] == "accept" and selection["selected_version"] == 1
    assert len(tasks) == 4 and len(proposals) == len(reviews) == 1
    assert not (root / "policies" / failed_file).exists()
    monkeypatch.setattr(module, "write_text_atomic", original_write)
    result = run_stage1(**opts, **callbacks, resume=True)
    assert result["status"] == "completed" and result["final_policy_version"] == 1
    assert len(tasks) == len(set(tasks)) == 6
    assert len(proposals) == len(reviews) == 1
    assert result["selection_results"] == [selection]
    assert selection_file.read_bytes() == selection_bytes
    assert (root / "policies/assignment_v001.md").read_text() == "A1"
    assert (root / "policies/communication_v001.md").read_text() == "C1"
    assert result["all_costs"]["cost_usd"] == pytest.approx(1.1)
    again = run_stage1(**opts, run_task=lambda _: pytest.fail("No task should repeat"),
                       propose_policy=lambda _: pytest.fail("No proposal should repeat"),
                       review_policy=lambda _: pytest.fail("No review should repeat"), resume=True)
    assert again == result


@pytest.mark.parametrize("malformed,expected_cost,expected_known,expected_usage", [
    ({"cost_usd": "not money", "known_cost_usd": 0.07}, None, 0.07, {"total_tokens": 17}),
    ({"cost_usd": float("nan"), "known_cost_usd": 0.07}, None, 0.07, {"total_tokens": 17}),
    ({"cost_usd": -1, "known_cost_usd": 0.07}, None, 0.07, {"total_tokens": 17}),
    ({"cost_usd": True, "known_cost_usd": 0.07}, None, 0.07, {"total_tokens": 17}),
    ({"known_cost_usd": "not money"}, 0.3, None, {"total_tokens": 17}),
    ({"elapsed_seconds": float("inf")}, 0.3, None, {"total_tokens": 17}),
    ({"elapsed_seconds": -1}, 0.3, None, {"total_tokens": 17}),
    ({"usage": "not usage"}, 0.3, None, {}),
    ({"usage": [17]}, 0.3, None, {}),
    ({"cost_usd": float("inf"), "known_cost_usd": float("nan"),
      "elapsed_seconds": "not time", "usage": None}, None, None, {}),
])
def test_malformed_review_accounting_is_normalized_without_losing_valid_spend(
    tmp_path, malformed, expected_cost, expected_known, expected_usage,
):
    def malformed_review(_):
        return {"recommendation": "prefer_candidate", "rationale": "Advisory review with malformed accounting.",
                "cost_usd": 0.3, "usage": {"total_tokens": 17}, **malformed}

    opts = options(tmp_path)
    result = run_stage1(**opts,
                        run_task=lambda request: TaskResult("completed", cost_usd=0.1,
                            score=0.75 if request.split == "train_replay" else 0.5),
                        propose_policy=propose, review_policy=malformed_review)
    assert result["status"] == "completed_with_failures" and result["final_policy_version"] == 1
    assert result["replay"]["completed"] == 2
    saved = result["reviewer_runs"][0]
    assert saved["status"] == "failed" and "ValueError" in saved["error"]
    assert saved["cost_usd"] == expected_cost and saved["known_cost_usd"] == expected_known
    assert saved["usage"] == expected_usage
    assert math.isfinite(saved["elapsed_seconds"]) and saved["elapsed_seconds"] >= 0
    assert json.loads((opts["output_dir"] / "reviews/batch_001/candidate_001/result.json").read_text()) == saved
    known_review_cost = expected_known if expected_known is not None else expected_cost
    assert result["reviewer"]["known_cost_usd"] == pytest.approx(known_review_cost or 0)
    assert result["all_costs"]["known_cost_usd"] == pytest.approx(0.8 + (known_review_cost or 0))
    if expected_cost is None:
        assert result["all_costs"]["cost_usd"] is None
        assert result["all_costs"]["records_with_unknown_cost"] == 1
    else:
        assert result["all_costs"]["cost_usd"] == pytest.approx(0.8 + expected_cost)
        assert result["all_costs"]["records_with_unknown_cost"] == 0
    json.dumps(result, allow_nan=False)
