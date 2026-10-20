"""Offline coverage for reviewer-selected, shared current-batch replay subsets."""

from collections import Counter
import json

import pytest

from agent_company.runtime_io import write_json_atomic
from agent_company.evolution.loop import (
    PolicyProposal, PolicyReview, TaskResult, TaskSpec, run_stage1,
)


def options(tmp_path, count=5, **overrides):
    return {
        "train_tasks": [TaskSpec(f"train-{index}", "train-repo") for index in range(count)],
        "test_tasks": [TaskSpec("heldout", "test-repo")],
        "initial_communication": "C0", "assignment_policy": "A0",
        "output_dir": tmp_path / "experiment", "replay_task_count": 2,
        "replay_selection": "reviewer", **overrides,
    }


def identity(index):
    return {"task_id": f"train-{index}", "repo_id": "train-repo"}


def propose(request):
    return PolicyProposal(f"C{request.candidate_index}", f"A{request.candidate_index}",
                          rationale="Use this training batch.", cost_usd=0.2)


def selected_review(indices=(4, 2)):
    return PolicyReview("prefer_incumbent", "Advisory concern, not a replay veto.",
                        replay_tasks=[identity(index) for index in indices],
                        replay_rationale="Exercise the changed coordination rule and its regression risk.",
                        cost_usd=0.3)


def improving_run(request):
    return TaskResult("completed", score=0.75 if request.split == "train_replay" else 0.5,
                      cost_usd=0.1)


def test_selected_subset_pairs_same_original_tasks_and_keeps_all_training_evidence(tmp_path):
    runs, reviews, proposals = [], [], []
    old_scores = {"train-0": 0.95, "train-1": 0.95, "train-2": 0.1,
                  "train-3": 0.8, "train-4": 0.4}

    def run(request):
        runs.append(request)
        (request.output_dir / "trace.txt").write_text(f"TRACE {request.task.task_id}")
        score = old_scores.get(request.task.task_id, 0.5)
        if request.split == "train_replay":
            score += 0.2
        return TaskResult("completed", score=score, cost_usd=0.1)

    def capture_proposal(request):
        proposals.append(request)
        return propose(request)

    def capture_review(request):
        reviews.append(request)
        assert len(request.proposal_request.batch_results) == 5
        assert request.replay_task_count == 2 and request.replay_tasks == ()
        assert not any(row.split == "train_replay" for row in runs)
        return selected_review()

    opts = options(tmp_path)
    result = run_stage1(**opts, run_task=run, propose_policy=capture_proposal,
                        review_policy=capture_review, seed=43)
    replay = [request for request in runs if request.split == "train_replay"]
    assert [request.task.task_id for request in replay] == ["train-2", "train-4"]
    assert [request.pair_id for request in replay] == ["batch_001_task_003", "batch_001_task_005"]
    train_by_id = {request.task.task_id: request for request in runs if request.split == "train"}
    for request in replay:
        original = train_by_id[request.task.task_id]
        assert request.task == original.task and request.seed == original.seed == 43
        assert request.fresh_start and request.output_dir != original.output_dir
    selection = result["selection_results"][0]
    candidate = selection["candidate_results"][0]
    assert selection["replay_task_ids"] == [identity(2), identity(4)]
    assert candidate["decision"] == "accept" and result["final_policy_version"] == 1
    assert candidate["pair_count"] == candidate["comparable_count"] == 2
    assert candidate["incumbent_mean_score"] == pytest.approx(0.25)
    assert candidate["mean_score_difference"] == pytest.approx(0.2)
    assert [pair["incumbent_score"] for pair in candidate["pairs"]] == [0.1, 0.4]
    assert candidate["reviewer_recommendation"] == "prefer_incumbent"
    assert candidate["reviewer_advisory"] is True
    assert len(proposals[0].batch_results) == len(result["training_runs"]) == 5
    assert result["replay"]["runs"] == 2
    assert result["all_costs"]["cost_usd"] == pytest.approx(1.4)
    assert reviews[0].evidence is proposals[0].evidence
    for original in train_by_id.values():
        assert original.task.task_id in reviews[0].evidence.read_file(
            str(original.output_dir / "trace.txt"))["text"]
    for request in runs:
        if request.split in ("train_replay", "test"):
            with pytest.raises(PermissionError):
                reviews[0].evidence.read_file(str(request.output_dir / "trace.txt"))


def test_all_candidates_share_first_selected_subset_even_if_later_review_requests_change(tmp_path):
    runs, reviews = [], []

    def run(request):
        runs.append(request)
        return improving_run(request)

    def review(request):
        reviews.append(request)
        assert len(request.proposal_request.batch_results) == 5
        if request.candidate_index == 1:
            assert request.replay_tasks == ()
            return selected_review((3, 1))
        assert request.replay_tasks == (identity(1), identity(3))
        return selected_review((0, 4))

    result = run_stage1(**options(tmp_path), run_task=run, propose_policy=propose,
                        review_policy=review, candidates_per_batch=3)
    assert len(reviews) == 3 and result["replay"]["runs"] == 6
    assert result["selection_results"][0]["replay_task_ids"] == [identity(1), identity(3)]
    assert result["final_policy_version"] == 1  # Equal later candidates do not replace the winner.
    replay = [request for request in runs if request.split == "train_replay"]
    assert len({request.output_dir for request in replay}) == 6
    for version in (1, 2, 3):
        selected = [request for request in replay if request.policy_version == version]
        assert [request.task.task_id for request in selected] == ["train-1", "train-3"]
        assert [request.pair_id for request in selected] == ["batch_001_task_002", "batch_001_task_004"]
    candidates = result["selection_results"][0]["candidate_results"]
    assert [candidate["pair_count"] for candidate in candidates] == [2, 2, 2]
    assert candidates[1]["comparison_with_previous_candidate"]["pair_count"] == 2


@pytest.mark.parametrize("tasks", [
    [], [identity(4)], [identity(0), identity(2), identity(4)],
    [identity(4), identity(4)],
    [identity(4), {"task_id": "heldout", "repo_id": "test-repo"}],
    [identity(4), {"task_id": "train-2", "repo_id": "wrong-repo"}],
    [identity(4), identity(9)],
    [identity(4), {"task_id": "train-2"}],
    [identity(4), "train-2"],
], ids=["missing", "too-few", "too-many", "duplicate", "heldout", "wrong-repo",
        "outside-batch", "incomplete-identity", "malformed-identity"])
def test_invalid_selection_is_inconclusive_without_unrequested_replay(tmp_path, tasks):
    def review(_):
        return PolicyReview("prefer_incumbent", "The review remains advisory.",
                            replay_tasks=tasks, replay_rationale="Invalid test selection.", cost_usd=0.3)

    result = run_stage1(**options(tmp_path), run_task=improving_run, propose_policy=propose,
                        review_policy=review)
    assert result["selection_results"][0]["replay_task_ids"] == []
    assert result["replay_runs"] == []
    assert result["final_policy_version"] == 0
    assert result["selection_results"][0]["candidate_results"][0]["reason"] == "reviewer_replay_selection_unavailable"


def test_failed_first_review_skips_replay_until_a_later_authorized_candidate_has_valid_selection(tmp_path):
    reviews = []

    def review(request):
        reviews.append(request)
        if request.candidate_index == 1:
            raise RuntimeError("Review service unavailable")
        assert request.replay_tasks == ()
        return selected_review()

    result = run_stage1(**options(tmp_path), run_task=improving_run, propose_policy=propose,
                        review_policy=review, candidates_per_batch=2)
    assert len(reviews) == 2 and result["reviewer"]["failed"] == 1
    assert result["selection_results"][0]["replay_task_ids"] == [identity(2), identity(4)]
    assert [record["task_id"] for record in result["replay_runs"]] == [
        "train-2", "train-4"]
    assert result["final_policy_version"] == 2
    assert result["selection_results"][0]["candidate_results"][0]["decision"] == "inconclusive"


def test_first_successful_proposal_chooses_subset_after_prior_proposal_failure(tmp_path):
    reviews = []

    def fail_first_proposal(request):
        if request.candidate_index == 1:
            raise RuntimeError("First proposal failed")
        return propose(request)

    def review(request):
        reviews.append(request)
        assert request.candidate_index == 2 and request.replay_tasks == ()
        return selected_review()

    result = run_stage1(**options(tmp_path), run_task=improving_run,
                        propose_policy=fail_first_proposal, review_policy=review, candidates_per_batch=2)
    assert len(reviews) == 1 and result["optimizer"]["failed"] == 1
    assert result["selection_results"][0]["replay_task_ids"] == [identity(2), identity(4)]
    assert [record["task_id"] for record in result["replay_runs"]] == ["train-2", "train-4"]
    assert result["final_policy_version"] == 2


def test_all_failed_proposals_do_not_review_or_replay(tmp_path):
    def fail_proposal(_):
        raise RuntimeError("No valid proposal")

    result = run_stage1(**options(tmp_path), run_task=improving_run,
                        propose_policy=fail_proposal,
                        review_policy=lambda _: pytest.fail("Nothing to review"), candidates_per_batch=2)
    assert result["reviewer"]["runs"] == result["replay"]["runs"] == 0
    assert result["replay_runs"] == [] and result["final_policy_version"] == 0


def test_reviewer_selection_is_current_batch_only_and_short_last_batch_uses_available_count(tmp_path):
    reviews = []

    def review(request):
        reviews.append(request)
        if request.batch_index == 1:
            assert request.replay_task_count == 2
            return selected_review()
        assert request.replay_task_count == 1 and request.replay_tasks == ()
        assert len(request.proposal_request.batch_results) == 1
        # Valid training tasks from an earlier batch still cannot be selected.
        return selected_review((4,))

    result = run_stage1(**options(tmp_path, count=6), run_task=improving_run,
                        propose_policy=propose, review_policy=review)
    assert len(reviews) == 2
    assert [selection["replay_task_ids"] for selection in result["selection_results"]] == [
        [identity(2), identity(4)], []]
    assert result["replay"]["runs"] == 2


def test_resume_after_durable_review_before_any_replay_reuses_selected_tasks(tmp_path, monkeypatch):
    import agent_company.evolution.loop as module

    calls, proposals, reviews = [], [], []
    original_run = module._run

    def interrupt_before_replay(request, callback, **kwargs):
        if request.split == "train_replay":
            raise KeyboardInterrupt("After durable review, before first replay")
        return original_run(request, callback, **kwargs)

    def run(request):
        calls.append((request.split, request.task.task_id, request.execution_mode))
        return improving_run(request)

    def capture_proposal(request):
        proposals.append(request)
        return propose(request)

    def capture_review(request):
        reviews.append(request)
        return selected_review()

    opts = options(tmp_path)
    callbacks = {"run_task": run, "propose_policy": capture_proposal, "review_policy": capture_review}
    monkeypatch.setattr(module, "_run", interrupt_before_replay)
    with pytest.raises(KeyboardInterrupt):
        run_stage1(**opts, **callbacks)
    saved_review = opts["output_dir"] / "reviews/batch_001/candidate_001/result.json"
    saved_bytes = saved_review.read_bytes()
    assert json.loads(saved_bytes)["replay_tasks"] == [identity(4), identity(2)]
    assert len(proposals) == len(reviews) == 1 and len(calls) == 5
    monkeypatch.setattr(module, "_run", original_run)
    result = run_stage1(**opts, **callbacks, resume=True)
    assert len(proposals) == len(reviews) == 1
    assert saved_review.read_bytes() == saved_bytes
    assert len(calls) == len(set(calls)) == 9
    assert [record["task_id"] for record in result["replay_runs"]] == ["train-2", "train-4"]
    again = run_stage1(**opts, run_task=lambda _: pytest.fail("No repeated task"),
                       propose_policy=lambda _: pytest.fail("No repeated proposal"),
                       review_policy=lambda _: pytest.fail("No repeated review"), resume=True)
    assert again == result


def test_resume_mid_replay_preserves_choice_and_does_not_repeat_completed_subset_task(tmp_path):
    calls, reviews = [], []

    def run(request):
        key = request.split, request.task.task_id, request.execution_mode
        calls.append(key)
        if key == ("train_replay", "train-4", "team") and calls.count(key) == 1:
            raise KeyboardInterrupt
        return improving_run(request)

    def review(request):
        reviews.append(request)
        return selected_review()

    opts = options(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run_stage1(**opts, run_task=run, propose_policy=propose, review_policy=review)
    result = run_stage1(**opts, run_task=run,
                        propose_policy=lambda _: pytest.fail("Do not propose again"),
                        review_policy=lambda _: pytest.fail("Do not review again"),
                        resume=True, retry_interrupted=True)
    assert len(reviews) == 1
    counts = Counter(calls)
    assert counts[("train_replay", "train-2", "team")] == 1
    assert counts[("train_replay", "train-4", "team")] == 2
    assert sum(counts.values()) == 10
    assert result["selection_results"][0]["replay_task_ids"] == [identity(2), identity(4)]


@pytest.mark.parametrize("original_mode,changed_mode", [("first_n", "reviewer"), ("reviewer", "first_n")])
def test_resume_cannot_switch_replay_selection_mode(tmp_path, original_mode, changed_mode):
    opts = options(tmp_path, replay_selection=original_mode)
    result = run_stage1(**opts, run_task=improving_run, propose_policy=propose,
                        review_policy=lambda _: selected_review())
    before = (opts["output_dir"] / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        run_stage1(**{**opts, "replay_selection": changed_mode},
                   run_task=lambda _: pytest.fail("No task before settings check"),
                   propose_policy=lambda _: pytest.fail("No proposal before settings check"),
                   review_policy=lambda _: pytest.fail("No review before settings check"), resume=True)
    assert result["replay"]["runs"] == 2
    assert (opts["output_dir"] / "summary.json").read_bytes() == before


def test_old_first_n_settings_without_new_mode_field_remain_resumable(tmp_path):
    opts = options(tmp_path, replay_selection="first_n")
    result = run_stage1(**opts, run_task=improving_run, propose_policy=propose,
                        review_policy=lambda _: PolicyReview("uncertain", "Keep official replay decisive."))
    settings_path = opts["output_dir"] / "run_settings.json"
    settings = json.loads(settings_path.read_text())
    settings.pop("replay_selection", None)
    write_json_atomic(settings_path, settings)
    again = run_stage1(**opts, run_task=lambda _: pytest.fail("No repeated task"),
                       propose_policy=lambda _: pytest.fail("No repeated proposal"),
                       review_policy=lambda _: pytest.fail("No repeated review"), resume=True)
    assert again == result
    assert result["selection_results"][0]["replay_task_ids"] == [identity(0), identity(1)]
