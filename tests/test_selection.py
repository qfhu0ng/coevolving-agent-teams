from dataclasses import FrozenInstanceError, asdict
from functools import partial
import json

import pytest

from agent_company.evolution.selection import SelectionConfig as Config, select_candidate as select

# Preserve explicit regression coverage for the supported legacy cost mode.
SelectionConfig = partial(Config, efficiency_metric="cost_usd")
select_candidate = partial(select, config=SelectionConfig())


def row(task="one", *, repo="repo", score=0.5, status="completed", cost=1.0, **extra):
    return {"repo_id": repo, "task_id": task, "status": status,
            "score": score, "cost_usd": cost, **extra}


def test_pairs_by_repo_and_task_and_compares_official_mean_not_binary_success():
    incumbent = [row("same", repo="a", score=0.0), row("same", repo="b", score=1.0)]
    candidate = [row("same", repo="b", score=0.5), row("same", repo="a", score=0.75)]
    result = select_candidate(incumbent, candidate)
    assert result["decision"] == "accept"
    assert result["reason"] == "score_improved"
    assert result["incumbent_mean_score"] == 0.5
    assert result["candidate_mean_score"] == 0.625
    assert result["mean_score_difference"] == 0.125
    assert result["pair_count"] == result["comparable_count"] == 2
    assert [pair["score_difference"] for pair in result["pairs"]] == [0.75, -0.5]


@pytest.mark.parametrize("cost", [None, -1, float("nan"), float("inf"), True, "1"])
def test_quality_improvement_can_accept_without_claiming_known_cost(cost):
    result = select_candidate([row()], [row(score=0.75, cost=cost, known_cost_usd=0.1)])
    assert result["decision"] == "accept"
    assert result["reason"] == "score_improved"
    assert result["cost_complete"] is False
    assert result["candidate_total_cost_usd"] is None
    assert result["cost_reduction_fraction"] is None
    json.dumps(result, allow_nan=False)


def test_quality_improvement_has_priority_even_when_cost_increases():
    result = select_candidate([row(cost=1)], [row(score=0.75, cost=10)])
    assert result["decision"] == "accept"
    assert result["reason"] == "score_improved"
    assert result["cost_reduction_fraction"] == -9


@pytest.mark.parametrize("candidate_score", [0.0, 0.499999999])
def test_default_rejects_any_quality_regression_even_if_free(candidate_score):
    result = select_candidate([row()], [row(score=candidate_score, cost=0)])
    assert result["decision"] == "reject"
    assert result["reason"] == "score_regressed_beyond_tolerance"


@pytest.mark.parametrize("candidate_cost,decision,reason", [
    (0.5, "accept", "cost_reduced_without_clear_score_gain"),
    (1.0, "inconclusive", "incomplete_time_evidence_at_equal_cost"),
    (2.0, "reject", "cost_not_reduced"),
    (None, "inconclusive", "incomplete_cost_evidence"),
])
def test_equal_quality_compares_cost_before_time(candidate_cost, decision, reason):
    result = select_candidate([row()], [row(cost=candidate_cost)])
    assert (result["decision"], result["reason"]) == (decision, reason)


def test_zero_incumbent_cost_cannot_be_reduced_and_does_not_divide_by_zero():
    result = select_candidate([row(cost=0)], [row(cost=0)])
    assert result["decision"] == "inconclusive"
    assert result["cost_reduction_fraction"] is None


@pytest.mark.parametrize("old_cost,new_cost", [(None, 0.1), (1, None), (-1, 0.1), (1, -1)])
def test_known_cost_subtotal_never_substitutes_for_complete_cost(old_cost, new_cost):
    result = select_candidate([row(cost=old_cost, known_cost_usd=10)],
                              [row(cost=new_cost, known_cost_usd=0.01)])
    assert result["decision"] == "inconclusive"
    assert result["reason"] == "incomplete_cost_evidence"


@pytest.mark.parametrize("delta,cost,decision,reason", [
    (-0.25, 0.5, "reject", "score_regressed_beyond_tolerance"),
    (-0.125, 0.5, "reject", "score_regressed_within_tolerance"),
    (0.125, None, "inconclusive", "incomplete_cost_evidence"),
    (0.25, None, "accept", "score_improved"),
])
def test_tolerance_never_allows_score_loss_for_savings(delta, cost, decision, reason):
    result = select_candidate([row()], [row(score=0.5 + delta, cost=cost)],
                              config=SelectionConfig(quality_tolerance=0.125))
    assert (result["decision"], result["reason"]) == (decision, reason)


@pytest.mark.parametrize("gain,decision", [(0.125, "inconclusive"), (0.25, "accept")])
def test_minimum_score_gain_is_inclusive_but_must_also_exceed_tolerance(gain, decision):
    result = select_candidate([row()], [row(score=0.5 + gain, cost=None)],
                              config=SelectionConfig(min_score_gain=0.25))
    assert result["decision"] == decision


def test_gain_below_minimum_can_still_win_on_cost():
    result = select_candidate([row()], [row(score=0.625, cost=0.75)],
                              config=SelectionConfig(min_score_gain=0.25))
    assert result["reason"] == "cost_reduced_without_clear_score_gain"


@pytest.mark.parametrize("cost,decision", [(0.75, "accept"), (0.8, "reject")])
def test_cost_reduction_threshold_is_fractional_and_inclusive(cost, decision):
    result = select_candidate([row()], [row(cost=cost)],
                              config=SelectionConfig(min_cost_reduction_fraction=0.25))
    assert result["decision"] == decision
    assert result["cost_reduction_fraction"] == pytest.approx(1 - cost)


@pytest.mark.parametrize("status,score", [
    ("failed", None), ("failed", float("nan")), ("failed", float("inf")),
    ("failed", True), ("failed", "1"), ("running", 1.0), ("interrupted", 1.0),
    ("completed", None),
    ("completed", float("nan")), ("completed", float("inf")),
    ("completed", True), ("completed", "1"),
])
def test_incomplete_score_pairs_are_not_dropped_before_selection(status, score):
    result = select_candidate([row("one"), row("two")],
                              [row("one", score=1), row("two", status=status, score=score)])
    assert result["decision"] == "inconclusive"
    assert result["reason"] == "incomplete_score_evidence"
    assert result["pair_count"] == 2 and result["comparable_count"] == 1
    assert result["candidate_mean_score"] is None
    assert result["mean_score_difference"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failed_arm", ["incumbent", "candidate"])
@pytest.mark.parametrize("reward", [0, 0.75])
def test_failed_terminal_numeric_score_is_compared_without_changing_status(failed_arm, reward):
    incumbent = row(score=reward, cost=1, status="failed" if failed_arm == "incumbent" else "completed")
    candidate = row(score=reward, cost=0.5, status="failed" if failed_arm == "candidate" else "completed")
    failed_row = incumbent if failed_arm == "incumbent" else candidate
    failed_row["error"] = "CLI exited with code 137"
    result = select_candidate([incumbent], [candidate])
    assert result["decision"] == "accept"
    assert result["reason"] == "cost_reduced_without_clear_score_gain"
    assert result["pair_count"] == result["comparable_count"] == 1
    assert result["incumbent_mean_score"] == result["candidate_mean_score"] == reward
    assert result["mean_score_difference"] == 0
    pair = result["pairs"][0]
    assert pair[f"{failed_arm}_status"] == "failed"
    assert pair[f"{failed_arm}_score"] == reward
    assert failed_row["status"] == "failed" and failed_row["error"] == "CLI exited with code 137"


@pytest.mark.parametrize("old,new,reason", [
    ([], [], "empty_batch"),
    ([row()], [], "empty_batch"),
    ([row()], [row("other")], "mismatched_task_identities"),
    ([row(), row()], [row()], "duplicate_task_identity"),
    ([row()], [row(), row()], "duplicate_task_identity"),
    ([row()], [row(repo="")], "invalid_task_identity"),
    ([row()], [None], "invalid_task_identity"),
])
def test_invalid_pair_sets_are_inconclusive(old, new, reason):
    result = select_candidate(old, new)
    assert (result["decision"], result["reason"]) == ("inconclusive", reason)


def test_cost_comparison_uses_total_complete_batch_not_per_task_wins():
    result = select_candidate([row("one", cost=10), row("two", cost=1)],
                              [row("two", cost=2), row("one", cost=5)])
    assert result["decision"] == "accept"
    assert result["incumbent_total_cost_usd"] == 11
    assert result["candidate_total_cost_usd"] == 7
    assert result["cost_reduction_fraction"] == pytest.approx(4 / 11)


@pytest.mark.parametrize("name", ["quality_tolerance", "min_score_gain", "min_cost_reduction_fraction"])
@pytest.mark.parametrize("value", [True, False, -0.1, float("nan"), float("inf"), "0", None])
def test_config_rejects_invalid_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        SelectionConfig(**{name: value})


@pytest.mark.parametrize("value", [1.0, 2.0])
def test_fraction_must_be_less_than_one(value):
    with pytest.raises(ValueError, match="less than 1"):
        SelectionConfig(min_cost_reduction_fraction=value)


def test_config_is_frozen_and_reported_and_inputs_are_not_mutated():
    config = SelectionConfig(quality_tolerance=0.125, min_score_gain=0.25,
                             min_cost_reduction_fraction=0.5)
    with pytest.raises(FrozenInstanceError):
        config.min_score_gain = 0
    old, new = [row()], [row(cost=0.25)]
    before = json.dumps([old, new])
    result = select_candidate(old, new, config=config)
    assert result["config"] == asdict(config)
    assert json.dumps([old, new]) == before
