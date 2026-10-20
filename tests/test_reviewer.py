"""Scripted completions exercise reviewer tools/accounting without paid calls."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.loop import PolicyProposal, ProposalRequest, ReviewRequest
from agent_company.runtime.relay import ProviderConfig, TokenRates
from agent_company.evolution.reviewer import PolicyReviewError, StrategyReviewer, _review_fields


USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}
ONE_CALL_COST = 0.00022


def provider(**overrides):
    return ProviderConfig(**({"model": "review-scripted-model", "base_url": "https://provider.invalid/v1",
                            "api_key": "review-fake-key-never-on-network", "max_output_tokens": 256,
                            "rates": TokenRates(2, 0.5, 4)} | overrides))


def call(name, args, identifier="call-1"):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def response(*calls, usage=USAGE, content=None):
    result = {"choices": [{"message": {"role": "assistant", "content": content,
                                       "tool_calls": list(calls)}}]}
    if usage is not None:
        result["usage"] = copy.deepcopy(usage)
    return result


def submit(recommendation="prefer_candidate", rationale="The candidate addresses observed delayed handoffs.",
           *, evidence_refs=None, risks=None, usage=USAGE, replay_tasks=None, replay_rationale=None):
    fields = {"recommendation": recommendation, "rationale": rationale,
              "evidence_refs": evidence_refs or [], "risks": risks or []}
    if replay_tasks is not None:
        fields["replay_tasks"] = replay_tasks
    if replay_rationale is not None:
        fields["replay_rationale"] = replay_rationale
    return response(call("submit_review", fields), usage=usage)


def scripted(replies):
    captured, iterator = [], iter(replies)

    def complete(payload):
        captured.append(copy.deepcopy(payload))
        reply = next(iterator)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(payload)
        return reply if isinstance(reply, tuple) else (200, reply)

    return captured, complete


def review_request(tmp_path, *, trace="RECEIPT_AND_ACTION evidence", batch_index=1):
    root = tmp_path / "train"
    run = root / f"batch_{batch_index:03d}" / "task_001"
    run.mkdir(parents=True)
    (run / "trace.txt").write_text(trace)
    proposal_request = ProposalRequest(current_version=0, communication_policy="C0 complete incumbent",
                                      assignment_policy="A0 complete incumbent", batch_index=batch_index,
                                      batch_results=({"task_id": "train-task", "repo_id": "train-repo", "score": 0.5},),
                                      change_history=(), history_total=0,
                                      evidence=TrainingEvidence(root, [run]))
    candidate = PolicyProposal(communication_policy="C1 complete candidate",
                               assignment_policy="A1 complete candidate", rationale="Earlier targeted handoffs")
    return ReviewRequest(proposal_request=proposal_request, candidate=candidate)


def selection_request(tmp_path, *, count=2, fixed=()):
    request = review_request(tmp_path)
    batch = tuple({"task_id": f"train-{index}", "repo_id": "training-repo", "score": 0.5}
                  for index in range(1, 6))
    return replace(request, proposal_request=replace(request.proposal_request, batch_results=batch),
                   replay_task_count=count, replay_tasks=fixed)


CHOSEN_TASKS = [{"task_id": "train-3", "repo_id": "training-repo"},
                {"task_id": "train-5", "repo_id": "training-repo"}]
REPLAY_REASON = "Train-3 tests the changed handoff mechanism; train-5 covers integration regression risk."


def test_reviewer_selects_relevant_nonprefix_tasks_and_recovers_the_selection(tmp_path):
    request = selection_request(tmp_path)
    captured, complete = scripted([submit("prefer_incumbent", replay_tasks=CHOSEN_TASKS,
                                           replay_rationale=REPLAY_REASON)])
    reviewer = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)
    result = reviewer(request)
    assert result.replay_tasks == CHOSEN_TASKS and result.replay_rationale == REPLAY_REASON
    assert result.recommendation == "prefer_incumbent"  # Selection does not force a positive advisory review.
    assert result.cost_usd == pytest.approx(ONE_CALL_COST)
    assert reviewer.recover(request) == result and len(captured) == 1
    schema = next(tool["function"]["parameters"] for tool in captured[0]["tools"]
                  if tool["function"]["name"] == "submit_review")
    item_schema = schema["properties"]["replay_tasks"]["items"]
    assert set(item_schema["properties"]) == {"task_id", "repo_id"}
    assert item_schema["additionalProperties"] is False
    assert "replay_tasks" not in schema["required"]  # Old full-batch mode remains compatible.
    system = captured[0]["messages"][0]["content"]
    assert "regression risks" in system and "make the candidate win" in system
    assert "never skips official replay" in system
    prompt = json.loads(captured[0]["messages"][1]["content"])
    assert prompt["replay_selection"]["task_count"] == 2 and prompt["replay_selection"]["fixed_tasks"] == []
    assert len(prompt["incumbent"]["batch_results"]) == 5
    saved = json.loads((tmp_path / "reviewer/batch_0001/review.json").read_text())
    assert saved["replay_tasks"] == CHOSEN_TASKS and saved["replay_rationale"] == REPLAY_REASON


@pytest.mark.parametrize("chosen,reason", [
    ([], REPLAY_REASON),
    (CHOSEN_TASKS[:1], REPLAY_REASON),
    (CHOSEN_TASKS + [{"task_id": "train-1", "repo_id": "training-repo"}], REPLAY_REASON),
    ([CHOSEN_TASKS[0], CHOSEN_TASKS[0]], REPLAY_REASON),
    ([CHOSEN_TASKS[0], {"task_id": "heldout", "repo_id": "test-repo"}], REPLAY_REASON),
    ([CHOSEN_TASKS[0], {"task_id": "train-5", "repo_id": "wrong-repo"}], REPLAY_REASON),
    ([CHOSEN_TASKS[0], {"task_id": "../test/answer", "repo_id": "training-repo"}], REPLAY_REASON),
    ([CHOSEN_TASKS[0], CHOSEN_TASKS[1] | {"path": "/private/answer"}], REPLAY_REASON),
    ([CHOSEN_TASKS[0], {"task_id": "train-5"}], REPLAY_REASON),
    ([CHOSEN_TASKS[0], {"task_id": "", "repo_id": "training-repo"}], REPLAY_REASON),
    ({"task_id": "train-3", "repo_id": "training-repo"}, REPLAY_REASON),
    (CHOSEN_TASKS, " "),
    (CHOSEN_TASKS, None),
    (CHOSEN_TASKS, 123),
])
def test_invalid_replay_selection_gets_bounded_tool_correction_and_keeps_spend(tmp_path, chosen, reason):
    def corrected(payload):
        assert "error" in json.loads(payload["messages"][-1]["content"])
        return submit(replay_tasks=CHOSEN_TASKS, replay_rationale=REPLAY_REASON)

    captured, complete = scripted([submit(replay_tasks=chosen, replay_rationale=reason), corrected])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(selection_request(tmp_path))
    assert len(captured) == 2 and result.replay_tasks == CHOSEN_TASKS
    assert result.usage["total_tokens"] == 240 and result.cost_usd == pytest.approx(2 * ONE_CALL_COST)


@pytest.mark.parametrize("submitted", [None, [], CHOSEN_TASKS, list(reversed(CHOSEN_TASKS))])
def test_fixed_batch_subset_is_preserved_without_reselection_or_a_new_reason(tmp_path, submitted):
    request = selection_request(tmp_path, fixed=tuple(CHOSEN_TASKS))
    captured, complete = scripted([submit("uncertain", replay_tasks=submitted)])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(request)
    assert result.replay_tasks == CHOSEN_TASKS and result.replay_rationale == ""
    assert len(captured) == 1


def test_fixed_subset_cannot_be_replaced_for_a_later_candidate(tmp_path):
    request = selection_request(tmp_path, fixed=tuple(CHOSEN_TASKS))

    def corrected(payload):
        assert "already fixed" in json.loads(payload["messages"][-1]["content"])["error"]
        return submit()

    different = [CHOSEN_TASKS[0], {"task_id": "train-1", "repo_id": "training-repo"}]
    captured, complete = scripted([submit(replay_tasks=different, replay_rationale=REPLAY_REASON), corrected])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(request)
    assert len(captured) == 2 and result.replay_tasks == CHOSEN_TASKS


def test_invalid_selection_at_interaction_limit_keeps_usage_and_recovers_without_retry(tmp_path):
    request = selection_request(tmp_path)
    captured, complete = scripted([submit(replay_tasks=[])])
    reviewer = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete, max_tool_rounds=1)
    with pytest.raises(PolicyReviewError, match="interaction limit") as failed:
        reviewer(request)
    assert failed.value.usage["total_tokens"] == 120 and failed.value.cost_usd == pytest.approx(ONE_CALL_COST)
    assert not (tmp_path / "reviewer/batch_0001/review.json").exists()
    with pytest.raises(PolicyReviewError, match="no additional model calls") as recovered:
        reviewer.recover(request)
    assert recovered.value.usage["total_tokens"] == 120
    assert recovered.value.known_cost_usd == pytest.approx(ONE_CALL_COST) and len(captured) == 1


def test_review_field_compatibility_without_a_selection_request():
    result = _review_fields({"recommendation": "uncertain", "rationale": "Limited evidence"})
    assert result["replay_tasks"] == [] and result["replay_rationale"] == ""
    with pytest.raises(ValueError, match="current-batch review request"):
        _review_fields({"recommendation": "uncertain", "rationale": "Limited evidence",
                        "replay_tasks": CHOSEN_TASKS, "replay_rationale": REPLAY_REASON})


def test_reviewer_reads_lists_searches_and_paginates_before_advisory_submission(tmp_path):
    request = review_request(tmp_path, trace="x" * 12000 + "RECEIPT_AND_ACTION")
    path = "batch_001/task_001/trace.txt"

    def read_first(payload):
        initial = json.loads(payload["messages"][1]["content"])
        assert initial["incumbent"]["assignment_policy"] == "A0 complete incumbent"
        assert initial["incumbent"]["communication_policy"] == "C0 complete incumbent"
        assert initial["candidate"]["assignment_policy"] == "A1 complete candidate"
        assert initial["candidate"]["communication_policy"] == "C1 complete candidate"
        assert initial["candidate"]["rationale"] == "Earlier targeted handoffs"
        listing = json.loads(payload["messages"][-1]["content"])
        assert listing["files"][0]["path"] == path
        return response(call("read_training_file", {"path": path, "limit": 999999}))

    def read_second(payload):
        page = json.loads(payload["messages"][-1]["content"])
        assert len(page["text"]) == 12000 and page["next_offset"] == 12000
        return response(call("read_training_file", {"path": path, "offset": page["next_offset"]}))

    def search(payload):
        page = json.loads(payload["messages"][-1]["content"])
        assert page["text"] == "RECEIPT_AND_ACTION" and page["next_offset"] is None
        return response(call("search_training_files", {"query": "RECEIPT_AND_ACTION"}))

    def finish(payload):
        result = json.loads(payload["messages"][-1]["content"])
        assert result["matches"][0]["path"] == path
        return submit(evidence_refs=[path], risks=["Additional messages may increase cost."])

    captured, complete = scripted([response(call("list_training_files", {})), read_first, read_second, search, finish])
    reviewer = StrategyReviewer(provider(extra_body={"temperature": 0}), tmp_path / "reviewer", completion=complete)
    result = reviewer(request)
    assert result.recommendation == "prefer_candidate" and result.evidence_refs == [path]
    assert result.risks == ["Additional messages may increase cost."]
    assert result.usage["provider_attempts"] == 5 and result.usage["total_tokens"] == 600
    assert result.cost_usd == pytest.approx(5 * ONE_CALL_COST) and result.known_cost_usd == result.cost_usd
    assert captured[0]["model"] == "review-scripted-model" and captured[0]["max_tokens"] == 256
    assert captured[0]["temperature"] == 0 and captured[0]["stream"] is False
    assert {tool["function"]["name"] for tool in captured[0]["tools"]} == {
        "list_training_files", "read_training_file", "search_training_files", "record_research_note", "submit_review"}
    assert "advisory, not a veto" in captured[0]["messages"][0]["content"]
    saved = json.loads((tmp_path / "reviewer/batch_0001/review.json").read_text())
    assert saved == asdict(result)
    for step in range(5):
        for kind in ("request", "response", "conversation"):
            assert (tmp_path / f"reviewer/batch_0001/{kind}_{step:03d}.json").is_file()


def test_reviewer_cannot_read_or_search_heldout_and_has_no_shell(tmp_path):
    request = review_request(tmp_path)
    heldout = tmp_path / "test"
    heldout.mkdir()
    hidden = heldout / "answer.txt"
    hidden.write_text("HELDOUT_SECRET_SENTINEL")

    def denied(payload):
        assert "error" in json.loads(payload["messages"][-1]["content"])
        assert "HELDOUT_SECRET_SENTINEL" not in json.dumps(payload)
        return response(call("search_training_files", {"path": str(heldout), "query": "SECRET"}))

    def no_shell(payload):
        assert "error" in json.loads(payload["messages"][-1]["content"])
        return response(call("run_shell_command", {"command": "do not execute this"}))

    def uncertain(payload):
        assert "Unknown review tool" in payload["messages"][-1]["content"]
        return submit("uncertain", "Available training evidence does not resolve the tradeoff.")

    captured, complete = scripted([response(call("read_training_file", {"path": str(hidden)})),
                                   denied, no_shell, uncertain])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(request)
    assert result.recommendation == "uncertain" and len(captured) == 4
    for path in (tmp_path / "reviewer").rglob("*.json"):
        assert "HELDOUT_SECRET_SENTINEL" not in path.read_text()


@pytest.mark.parametrize("args", [
    {"recommendation": "approve", "rationale": "invalid enum"},
    {"recommendation": "uncertain"},
    {"recommendation": "uncertain", "rationale": " "},
    {"recommendation": [], "rationale": "wrong type"},
    {"recommendation": "uncertain", "rationale": "ok", "evidence_refs": "not a list"},
    {"recommendation": "uncertain", "rationale": "ok", "risks": [1]},
    {"recommendation": "uncertain", "rationale": "ok", "risks": [""]},
    {"recommendation": "uncertain", "rationale": "ok", "score": 1},
])
def test_invalid_review_is_tool_feedback_then_can_be_corrected(tmp_path, args):
    def corrected(payload):
        assert "error" in json.loads(payload["messages"][-1]["content"])
        return response(call("submit_review", {"recommendation": "prefer_incumbent", "rationale": "Regression risk"}))

    captured, complete = scripted([response(call("submit_review", args)), corrected])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(review_request(tmp_path))
    assert result.recommendation == "prefer_incumbent" and result.evidence_refs == [] and result.risks == []
    assert len(captured) == 2 and result.cost_usd == pytest.approx(2 * ONE_CALL_COST)


def test_plain_text_gets_submission_reminder(tmp_path):
    captured, complete = scripted([response(content="Tentative review without submission"), submit("uncertain")])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(review_request(tmp_path))
    assert result.recommendation == "uncertain"
    assert "Use submit_review" in captured[1]["messages"][-1]["content"]


@pytest.mark.parametrize("usage", [None, {"prompt_tokens": "invalid", "completion_tokens": 20},
                                     {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 11}])
def test_unknown_usage_is_not_zero_cost(tmp_path, usage):
    _, complete = scripted([submit(usage=usage)])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(review_request(tmp_path))
    assert result.cost_usd is None and result.known_cost_usd == 0
    assert result.usage["unknown_usage_attempts"] == 1
    assert json.loads((tmp_path / "reviewer/batch_0001/summary.json").read_text())["cost_usd"] is None


@pytest.mark.parametrize("rates", [None, TokenRates(2, 0.5, 4, currency="EUR")])
def test_missing_or_non_usd_rates_never_claim_usd_cost(tmp_path, rates):
    _, complete = scripted([submit()])
    result = StrategyReviewer(provider(rates=rates), tmp_path / "reviewer", completion=complete)(review_request(tmp_path))
    assert result.cost_usd is None and result.known_cost_usd is None
    assert result.usage["total_tokens"] == 120 and result.usage["unknown_usage_attempts"] == 0


def test_http_error_preserves_known_cost(tmp_path):
    _, complete = scripted([(429, {"usage": USAGE, "error": "rate limited"})])
    with pytest.raises(PolicyReviewError, match="HTTP 429") as raised:
        StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(review_request(tmp_path))
    assert raised.value.cost_usd == pytest.approx(ONE_CALL_COST)
    assert raised.value.known_cost_usd == pytest.approx(ONE_CALL_COST)
    assert raised.value.usage["total_tokens"] == 120
    assert not (Path(raised.value.output_dir) / "review.json").exists()


def test_mixed_usage_transport_failure_preserves_known_subtotal_and_redacts_error(tmp_path):
    config = provider()
    _, complete = scripted([response(call("list_training_files", {})), OSError("interrupted " + config.api_key)])
    with pytest.raises(PolicyReviewError) as raised:
        StrategyReviewer(config, tmp_path / "reviewer", completion=complete)(review_request(tmp_path))
    error = raised.value
    assert error.cost_usd is None and error.known_cost_usd == pytest.approx(ONE_CALL_COST)
    assert error.usage["provider_attempts"] == 2 and error.usage["unknown_usage_attempts"] == 1
    assert config.api_key not in str(error) and error.elapsed_seconds >= 0
    for path in Path(error.output_dir).glob("*.json"):
        assert config.api_key not in path.read_text()


@pytest.mark.parametrize("failed_trace", ["response_000.json", "conversation_000.json", "summary.json"])
def test_trace_and_failure_summary_write_errors_preserve_observed_usage(tmp_path, monkeypatch, failed_trace):
    config = provider()
    captured, complete = scripted([submit()])
    reviewer = StrategyReviewer(config, tmp_path / "reviewer", completion=complete)
    original_write = reviewer._write

    def fail_writes(directory, name, value):
        if name in (failed_trace, "summary.json"):
            raise OSError("trace volume unavailable " + config.api_key)
        original_write(directory, name, value)

    monkeypatch.setattr(reviewer, "_write", fail_writes)
    with pytest.raises(PolicyReviewError, match="failed to persist reviewer summary") as raised:
        reviewer(review_request(tmp_path))
    error = raised.value
    assert len(captured) == 1 and error.usage["total_tokens"] == 120
    assert error.usage["unknown_usage_attempts"] == 0
    assert error.cost_usd == pytest.approx(ONE_CALL_COST)
    assert error.known_cost_usd == pytest.approx(ONE_CALL_COST)
    assert error.elapsed_seconds >= 0 and config.api_key not in str(error)
    assert not (Path(error.output_dir) / "review.json").exists()


def test_transport_and_summary_write_failures_preserve_unknown_usage(tmp_path, monkeypatch):
    config = provider()
    _, complete = scripted([response(call("list_training_files", {})), OSError("provider disconnected")])
    reviewer = StrategyReviewer(config, tmp_path / "reviewer", completion=complete)
    original_write = reviewer._write

    def fail_summary(directory, name, value, **options):
        if name == "summary.json":
            raise OSError("summary destination unavailable")
        original_write(directory, name, value, **options)

    monkeypatch.setattr(reviewer, "_write", fail_summary)
    with pytest.raises(PolicyReviewError, match="provider disconnected") as raised:
        reviewer(review_request(tmp_path))
    assert raised.value.cost_usd is None
    assert raised.value.known_cost_usd == pytest.approx(ONE_CALL_COST)
    assert raised.value.usage["provider_attempts"] == 2
    assert raised.value.usage["unknown_usage_attempts"] == 1


def test_context_cap_after_evidence_read_preserves_submission_and_spend(tmp_path):
    request = review_request(tmp_path, trace="x" * 20000)
    def finish(payload):
        page = json.loads(payload["messages"][-1]["content"])
        assert page["context_limited"] and 0 < len(page["text"]) < 12000
        assert len(json.dumps(payload["messages"], ensure_ascii=False)) <= 16000
        assert payload["tool_choice"]["function"]["name"] == "submit_review"
        return submit("uncertain")
    captured, complete = scripted([response(call("read_training_file", {"path": "batch_001/task_001/trace.txt"})), finish])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete, max_context_chars=16000)(request)
    assert len(captured) == 2 and result.cost_usd == pytest.approx(2 * ONE_CALL_COST)
    assert (tmp_path / "reviewer/batch_0001/review.json").exists()


def test_review_output_truncation_is_not_committed(tmp_path):
    reply = submit("uncertain")
    reply["choices"][0]["finish_reason"] = "length"
    captured, complete = scripted([reply])
    with pytest.raises(PolicyReviewError, match="output token limit") as raised:
        StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)(review_request(tmp_path))
    assert raised.value.usage["total_tokens"] == 120 and len(captured) == 1
    assert not (tmp_path / "reviewer/batch_0001/review.json").exists()


def test_review_last_round_allows_complete_review_and_pairs_all_previous_tools(tmp_path):
    request = review_request(tmp_path, trace="X" * 30000)
    reply = response(call("read_training_file", {"path": "batch_001/task_001/trace.txt"}, "one"),
                     call("read_training_file", {"path": "batch_001/task_001/trace.txt", "offset": 12000}, "two"))
    def finish(payload):
        assert len(json.dumps(payload["messages"], ensure_ascii=False)) <= 16000
        assert [m["tool_call_id"] for m in payload["messages"] if m["role"] == "tool"] == ["one", "two"]
        assert [t["function"]["name"] for t in payload["tools"]] == ["submit_review"]
        return submit("uncertain")
    captured, complete = scripted([reply, finish])
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete,
                              max_context_chars=16000, max_tool_rounds=2)(request)
    assert len(captured) == 2 and result.recommendation == "uncertain"


def test_interaction_limit_is_model_requests_not_tool_calls(tmp_path):
    captured, complete = scripted([response(call("list_training_files", {}, "one"),
                                           call("search_training_files", {"query": "evidence"}, "two"))])
    with pytest.raises(PolicyReviewError, match="interaction limit") as raised:
        StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete, max_tool_rounds=1)(review_request(tmp_path))
    assert len(captured) == 1 and raised.value.usage["provider_attempts"] == 1
    assert raised.value.cost_usd == pytest.approx(ONE_CALL_COST)


def test_committed_review_recovers_without_new_paid_call_and_context_is_new_each_batch(tmp_path):
    first = review_request(tmp_path)
    second = review_request(tmp_path, batch_index=2)
    captured, complete = scripted([submit(rationale="FIRST_REVIEW_ONLY_SENTINEL"), submit("uncertain")])
    reviewer = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)
    result = reviewer(first)
    assert reviewer.recover(first) == result and len(captured) == 1
    reviewer(second)
    assert len(captured[1]["messages"]) == 2
    assert "FIRST_REVIEW_ONLY_SENTINEL" not in json.dumps(captured[1])
    assert (tmp_path / "reviewer/batch_0002/review.json").exists()


def test_same_batch_candidates_have_distinct_durable_reviews_and_fresh_context(tmp_path):
    first = review_request(tmp_path)
    second = replace(first, proposal_request=replace(first.proposal_request, candidate_index=2),
                     candidate=PolicyProposal(assignment_policy="A2", communication_policy="C2", rationale="Second idea"))
    captured, complete = scripted([submit(rationale="FIRST_CANDIDATE_REVIEW_ONLY"), submit("prefer_incumbent")])
    reviewer = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete)
    first_result, second_result = reviewer(first), reviewer(second)
    assert reviewer.recover(first) == first_result
    assert reviewer.recover(second) == second_result
    assert len(captured) == 2 and len(captured[1]["messages"]) == 2
    assert "FIRST_CANDIDATE_REVIEW_ONLY" not in json.dumps(captured[1])
    assert json.loads(captured[0]["messages"][1]["content"])["incumbent"]["assignment_policy"] == "A0 complete incumbent"
    assert json.loads(captured[1]["messages"][1]["content"])["incumbent"]["assignment_policy"] == "A0 complete incumbent"
    assert (tmp_path / "reviewer/batch_0001/review.json").is_file()
    assert (tmp_path / "reviewer/batch_0001_candidate_0002/review.json").is_file()


@pytest.mark.parametrize("last_response", [None, "not valid json"])
def test_interrupted_review_recovers_known_and_unknown_spend_without_retry(tmp_path, last_response):
    request = review_request(tmp_path)
    directory = tmp_path / "reviewer/batch_0001"
    directory.mkdir(parents=True)
    (directory / "request_000.json").write_text("{}")
    (directory / "response_000.json").write_text(json.dumps(response()))
    (directory / "request_001.json").write_text("{}")
    if last_response is not None:
        (directory / "response_001.json").write_text(last_response)
    reviewer = StrategyReviewer(provider(), tmp_path / "reviewer",
                                completion=lambda _: pytest.fail("Recovery must not call the model"))
    with pytest.raises(PolicyReviewError, match="no additional model calls") as raised:
        reviewer.recover(request)
    assert raised.value.usage["provider_attempts"] == 2 and raised.value.usage["unknown_usage_attempts"] == 1
    assert raised.value.cost_usd is None and raised.value.known_cost_usd == pytest.approx(ONE_CALL_COST)
    assert not (directory / "review.json").exists()


def test_all_recorded_usage_remains_known_when_review_never_submitted(tmp_path):
    request = review_request(tmp_path)
    captured, complete = scripted([response(content="incomplete")])
    reviewer = StrategyReviewer(provider(), tmp_path / "reviewer", completion=complete, max_tool_rounds=1)
    with pytest.raises(PolicyReviewError):
        reviewer(request)
    with pytest.raises(PolicyReviewError) as raised:
        reviewer.recover(request)
    assert len(captured) == 1 and raised.value.cost_usd == pytest.approx(ONE_CALL_COST)
    assert raised.value.known_cost_usd == pytest.approx(ONE_CALL_COST)


@pytest.mark.parametrize("secret", ["review-fake-key", 'fake-"quoted"-key\nwith-换行'])
def test_secrets_are_redacted_in_nested_review_fields_and_all_artifacts(tmp_path, secret):
    request = review_request(tmp_path, trace="observed " + secret)
    request.candidate.rationale = "candidate accidentally contains " + secret
    _, complete = scripted([response(call("read_training_file", {"path": "batch_001/task_001/trace.txt"})),
                             submit(rationale="rationale " + secret, evidence_refs=["ref " + secret],
                                    risks=["risk " + secret])])
    result = StrategyReviewer(provider(api_key=secret), tmp_path / "reviewer", completion=complete)(request)
    assert secret not in json.dumps(asdict(result), ensure_ascii=False)
    assert result.rationale == "rationale [REDACTED]"
    assert result.evidence_refs == ["ref [REDACTED]"] and result.risks == ["risk [REDACTED]"]
    for path in (tmp_path / "reviewer").rglob("*.json"):
        text = path.read_text()
        assert secret not in text
        assert json.dumps(secret, ensure_ascii=False)[1:-1] not in text
        assert json.dumps(secret, ensure_ascii=True)[1:-1] not in text
        parsed = json.loads(text)
        if path.name.startswith("response_"):
            for tool_call in parsed["choices"][0]["message"]["tool_calls"]:
                assert secret not in str(json.loads(tool_call["function"]["arguments"]))


@pytest.mark.parametrize("limits", [{"max_tool_rounds": 0}, {"max_tool_rounds": True},
                                      {"max_context_chars": 0}, {"max_context_chars": 1.5}])
def test_invalid_limits_are_rejected_without_model_call(tmp_path, limits):
    with pytest.raises(ValueError, match="positive integers"):
        StrategyReviewer(provider(), tmp_path / "reviewer", completion=lambda _: pytest.fail("No call expected"), **limits)
