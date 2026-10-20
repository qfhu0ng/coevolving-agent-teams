"""Token selection and bounded research sessions; synthetic evidence only."""
import json

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.efficiency import efficiency_report
from agent_company.evolution.outer_budget import OuterBudget, size
from agent_company.evolution.selection import SelectionConfig, select_candidate
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.evolution.reviewer import StrategyReviewer
from test_optimizer import call, config, proposal_request, response, submit
from test_reviewer import review_request, submit as submit_review


def row(tokens=100, score=1, seconds=10, **extra):
    return {"task_id": "task", "repo_id": "repo", "status": "completed", "score": score,
            "execution_mode": "team", "cost_usd": None,
            "usage": {"prompt_tokens": tokens - 1, "completion_tokens": 1,
                      "cached_prompt_tokens": 0, "total_tokens": tokens},
            "metrics": {"task": {"agent_timeout_sec": 7200}, "team": {
                "model": "same-model", "usage_complete": True, "execution": {"elapsed_seconds": seconds}}},
            **extra}


@pytest.mark.parametrize("new,decision,reason", [
    (row(80), "accept", "tokens_reduced_without_clear_score_gain"),
    (row(120, seconds=1), "reject", "tokens_not_reduced"),
    (row(100, seconds=5), "accept", "time_reduced_at_equal_tokens"),
    (row(100), "reject", "tokens_and_time_not_improved"),
    (row(80, score=.9), "reject", "score_regressed_beyond_tolerance"),
    (row(80, usage={"total_tokens": 80}), "inconclusive", "incomplete_token_evidence"),
    (row(80, interrupted_attempts=[{}]), "inconclusive", "interrupted_token_evidence"),
])
def test_default_token_order_without_any_prices(new, decision, reason):
    result = select_candidate([row()], [new])
    assert (result["decision"], result["reason"]) == (decision, reason)
    assert result["selection_objective"]["priority"][1] == "complete_execution_total_tokens"
    assert result["cost_complete"] is False


@pytest.mark.parametrize("change", [
    {"unknown_usage_attempts": 1}, {"calls_missing_usage": 1}, {"total_tokens": 1},
    {"cached_prompt_tokens": 1000}, {"completion_tokens": True},
])
def test_incomplete_or_inconsistent_usage_never_wins(change):
    new = row(80)
    new["usage"].update(change)
    assert select_candidate([row()], [new])["reason"] == "incomplete_token_evidence"


def test_score_gain_and_model_budget_boundaries():
    assert select_candidate([row(score=.5)], [row(1000)])["reason"] == "score_improved"
    new = row()
    new["metrics"]["team"]["model"] = "other-model"
    assert select_candidate([row(score=.5)], [new])["reason"] == "incomparable_models"
    new = row()
    new["metrics"]["task"]["agent_timeout_sec"] = 3600
    assert select_candidate([row()], [new])["reason"] == "incomparable_wall_time_caps"
    assert select_candidate([row()], [row(80)], config=SelectionConfig(min_token_reduction_fraction=.25))["reason"] == "token_reduction_below_threshold"


def test_notebook_scope_rotation_preserves_initial_input_and_no_extra_rounds(tmp_path):
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    train.mkdir()
    heldout.mkdir()
    (train / "trace.txt").write_text("A handoff was proposed; receipt unknown")
    (heldout / "trace.txt").write_text("Not training")
    evidence = TrainingEvidence(tmp_path, [train])
    budget = OuterBudget("system", 12000, 4, "submit_policy", 65536)
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "A0 C0 batch"}]
    assert not budget.prepare(messages, 0)
    assert budget.output_limit() == 4096
    budget.record_note("Potential redundant handoff; inspect receipt", ["train/trace.txt"], evidence)
    with pytest.raises(PermissionError):
        budget.record_note("forbidden", ["heldout/trace.txt"], evidence)
    messages.append({"role": "assistant", "content": "x" * 30000})
    assert budget.prepare(messages, 1)
    assert size(messages) <= 12000 and budget.rollovers == 1
    assert messages[1]["content"] == "A0 C0 batch"
    assert "train/trace.txt" in messages[2]["content"]
    assert "model_observation_or_hypothesis_not_verified_fact" in messages[2]["content"]
    assert budget.output_limit() == 65536
    assert "remaining_model_calls_including_this\": 3" in messages[0]["content"]
    messages.append({"role": "assistant", "content": "x" * 30000})
    with pytest.raises(ValueError, match="context limit"):
        budget.prepare(messages, 2)  # Never unlimited rollovers.


@pytest.mark.parametrize("kind", ["optimizer", "reviewer"])
def test_real_outer_loop_recovers_oversized_response_and_archives_original(tmp_path, kind):
    req = proposal_request(tmp_path) if kind == "optimizer" else review_request(tmp_path)
    cls = StrategyOptimizer if kind == "optimizer" else StrategyReviewer
    final = submit if kind == "optimizer" else submit_review
    calls = []

    def complete(payload):
        calls.append(json.loads(json.dumps(payload)))
        if len(calls) == 1:
            return 200, response(call("list_training_files", {}), content="large response " * 2000)
        assert size(payload["messages"]) < 18000
        assert len(payload["tools"]) == 1
        assert "session_transition" in payload["messages"][2]["content"]
        return 200, final()

    output = tmp_path / "outer"
    result = cls(config(), output, completion=complete, max_context_chars=18000, max_tool_rounds=2)(req)
    assert result.usage["provider_attempts"] == 2
    directory = output / req.storage_key
    original = json.loads((directory / "conversation_000.json").read_text())
    assert len(original[2]["content"]) > 18000
    assert original[-1]["role"] == "tool"
    assert json.loads((directory / "research_notes.json").read_text())["rollovers"] == 1


def test_collaboration_index_is_not_execution_or_receipt_evidence(tmp_path):
    relay = tmp_path / "team" / "relay"
    relay.mkdir(parents=True)
    calls = [call("send_message", {"recipient": "worker-a", "content": "private body"}),
             call("agent", {"name": "bad", "__agent_company_tool_error": "invalid directory"})]
    (relay / "attempt_1_master_qwen_response.json").write_text(json.dumps({
        "choices": [{"message": {"tool_calls": calls, "reasoning_content": "private reasoning"}}]}))
    events = efficiency_report(row(), tmp_path)["collaboration_events"]
    assert len(events) == 2
    assert events[0]["execution_status"] == "delivered_call_execution_unconfirmed"
    assert events[1]["execution_status"] == "rejected_before_execution"
    assert all(e["receipt_confirmed"] is None and e["causal_effect"] == "unknown" for e in events)
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize("kind", ["optimizer", "reviewer"])
def test_configurable_inspection_allowance_reaches_actual_provider(tmp_path, kind):
    provider = config()
    provider.max_output_tokens = 65536
    provider.inspection_max_output_tokens = 32768
    provider.stream_response = True
    req = proposal_request(tmp_path) if kind == "optimizer" else review_request(tmp_path)
    cls = StrategyOptimizer if kind == "optimizer" else StrategyReviewer
    final = submit if kind == "optimizer" else submit_review
    def complete(payload):
        assert payload["max_tokens"] == 32768
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return 200, final()
    result = cls(provider, tmp_path / "outer", completion=complete)(req)
    assert result.usage["provider_attempts"] == 1


def test_inspection_allowance_does_not_exceed_provider_maximum():
    budget = OuterBudget("system", 180000, 8, "submit_review", 8192, 32768)
    assert budget.output_limit() == 8192
    budget.submission_only = True
    assert budget.output_limit() == 8192
