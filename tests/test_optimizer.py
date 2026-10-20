"""Exercise the real optimizer/HTTP transport with local scripted HTTP replies."""
import copy
import io
import json
from pathlib import Path
import urllib.error
import urllib.request

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.loop import ProposalRequest, TaskResult, TaskSpec, run_stage1
from agent_company.evolution.optimizer import StrategyOptimizer, PolicyOptimizationError
from agent_company.runtime.relay import ProviderConfig, TokenRates


USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}
ONE_CALL_COST = 0.00022


def config(**overrides):
    return ProviderConfig(model="fake-only", base_url="https://provider.invalid/v1",
                          api_key="fake-test-key-never-used-on-network", max_output_tokens=256,
                          rates=TokenRates(2, 0.5, 4), **overrides)


def call(name, args, identifier="call-1"):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def response(*calls, usage=USAGE, content=None):
    result = {"choices": [{"message": {"role": "assistant", "content": content,
                                        "tool_calls": list(calls)}}]}
    if usage is not None:
        result["usage"] = copy.deepcopy(usage)
    return result


def submit(text="Notify the affected teammate when an interface changes.",
           assignment="Assign interface owners and a final integration worker.", **kwargs):
    return response(call("submit_policy", {"communication_policy": text,
                                            "assignment_policy": assignment,
                                            "rationale": "Interface ownership supports targeted handoffs."}), **kwargs)


def fake_http(monkeypatch, replies):
    """Replace only HTTP I/O; real serialization, deadline and body parsing run."""
    import aiohttp
    from agent_company.runtime import transport

    requests = []
    scripted = iter(replies)
    real_timeout = transport.asyncio.timeout
    current_timeout = None

    def observe_timeout(delay):
        nonlocal current_timeout
        # The optional inner progress scope is disabled for these providers;
        # observing timeout(None) must not erase the real outer total deadline.
        if delay is not None:
            current_timeout = delay
        return real_timeout(delay)  # Observe, never bypass the production deadline.

    class Reply:
        content_type = "application/json"
        def __init__(self, status, raw):
            self.status = status
            self.buffer = io.BytesIO(raw)
            self.length = len(raw)
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            self.buffer.close()

        async def read(self, size=-1):
            return self.buffer.read(size)

        async def readany(self):
            return self.buffer.read()

        def at_eof(self):
            return self.buffer.tell() == self.length

    def post(session, url, **kwargs):
        # Use the actual ClientSession JSON serializer. urllib.Request is only
        # a metadata envelope, retaining the helper's former header casing; it
        # never opens a URL or resolves a hostname.
        request = urllib.request.Request(str(url),
            data=session._json_serialize(kwargs["json"]).encode(), headers=kwargs["headers"])
        payload = json.loads(request.data)
        requests.append({"payload": payload, "url": request.full_url,
                         "timeout": current_timeout, "headers": dict(request.headers)})
        reply = next(scripted)
        if callable(reply):
            reply = reply(payload)
        if isinstance(reply, urllib.error.HTTPError):
            return Reply(reply.code, reply.read(32768))
        if isinstance(reply, Exception):
            raise reply
        status, body = reply if isinstance(reply, tuple) else (200, reply)
        return Reply(status, json.dumps(body).encode())

    monkeypatch.setattr(transport.asyncio, "timeout", observe_timeout)
    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    return requests


def proposal_request(tmp_path, *, trace="training evidence", batch_index=1):
    root = tmp_path / "train"
    run = root / f"batch_{batch_index:03d}" / "task_001"
    run.mkdir(parents=True)
    (run / "trace.txt").write_text(trace)
    return ProposalRequest(current_version=0, communication_policy="Initial policy.",
                           assignment_policy="Initial assignment strategy.", batch_index=batch_index,
                           batch_results=({"task_id": "train-a", "score": 0.5},),
                           change_history=(), history_total=0,
                           evidence=TrainingEvidence(root, [run]))


def test_real_http_optimizer_reads_pages_then_submits_complete_policy(tmp_path, monkeypatch):
    request = proposal_request(tmp_path, trace="x" * 12000 + "RECIPIENT_ACTED_AFTER_MESSAGE")

    def inspect_first_page(payload):
        page = json.loads(payload["messages"][-1]["content"])
        assert len(page["text"]) == 12000
        assert page["next_offset"] == 12000
        return response(call("read_training_file", {"path": page["path"], "offset": page["next_offset"]}))

    def inspect_second_page(payload):
        page = json.loads(payload["messages"][-1]["content"])
        assert page["text"] == "RECIPIENT_ACTED_AFTER_MESSAGE"
        assert page["next_offset"] is None
        return submit("Full replacement: send interface changes immediately to dependents.")

    requests = fake_http(monkeypatch, [response(call("read_training_file", {
        "path": "batch_001/task_001/trace.txt", "limit": 999999})),
        inspect_first_page, inspect_second_page])
    optimizer = StrategyOptimizer(config(), tmp_path / "optimizer")
    result = optimizer(request)
    assert result.communication_policy.startswith("Full replacement:")
    assert result.cost_usd == pytest.approx(3 * ONE_CALL_COST)
    assert result.usage["provider_attempts"] == 3
    assert result.usage["total_tokens"] == 360
    assert requests[0]["url"] == "https://provider.invalid/v1/chat/completions"
    assert requests[0]["payload"]["stream"] is False
    assert requests[0]["payload"]["max_tokens"] == 256
    assert {tool["function"]["name"] for tool in requests[0]["payload"]["tools"]} == {
        "list_training_files", "read_training_file", "search_training_files", "record_research_note", "submit_policy"}
    submission_tool = next(t["function"] for t in requests[0]["payload"]["tools"]
                           if t["function"]["name"] == "submit_policy")
    assert set(submission_tool["parameters"]["required"]) == {
        "assignment_policy", "communication_policy", "rationale"}
    saved = json.loads((tmp_path / "optimizer/batch_0001/proposal.json").read_text())
    assert saved["communication_policy"] == result.communication_policy
    assert saved["assignment_policy"] == result.assignment_policy
    assert result.assignment_policy == "Assign interface owners and a final integration worker."


def test_optimizer_tools_reject_test_artifacts_without_exposing_contents(tmp_path, monkeypatch):
    request = proposal_request(tmp_path)
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    hidden = test_dir / "answer.txt"
    hidden.write_text("HIDDEN_TEST_SENTINEL")

    def inspect_denied(payload):
        result = json.loads(payload["messages"][-1]["content"])
        assert "error" in result
        assert "HIDDEN_TEST_SENTINEL" not in json.dumps(payload)
        return response(call("search_training_files", {"query": "evidence"}))

    def inspect_search(payload):
        result = json.loads(payload["messages"][-1]["content"])
        assert result["matches"][0]["path"] == "batch_001/task_001/trace.txt"
        return submit()

    fake_http(monkeypatch, [response(call("read_training_file", {"path": str(hidden)})),
                            inspect_denied, inspect_search])
    StrategyOptimizer(config(), tmp_path / "optimizer")(request)
    for path in (tmp_path / "optimizer").rglob("*.json"):
        assert "HIDDEN_TEST_SENTINEL" not in path.read_text()


@pytest.mark.parametrize("usage", [None, {"prompt_tokens": "invalid", "completion_tokens": 0}])
def test_unknown_usage_is_recorded_and_never_reported_as_free(tmp_path, monkeypatch, usage):
    fake_http(monkeypatch, [submit(usage=usage)])
    result = StrategyOptimizer(config(), tmp_path / "optimizer")(proposal_request(tmp_path))
    assert result.cost_usd is None
    assert result.usage["unknown_usage_attempts"] == 1
    summary = json.loads((tmp_path / "optimizer/batch_0001/summary.json").read_text())
    assert summary["cost_usd"] is None


def test_http_error_preserves_known_usage_cost_and_error_artifact(tmp_path, monkeypatch):
    fake_http(monkeypatch, [(429, {"usage": USAGE, "error": {"message": "rate limited"}})])
    with pytest.raises(PolicyOptimizationError, match="HTTP 429") as raised:
        StrategyOptimizer(config(), tmp_path / "optimizer")(proposal_request(tmp_path))
    assert raised.value.cost_usd == pytest.approx(ONE_CALL_COST)
    assert raised.value.usage["total_tokens"] == 120
    assert (Path(raised.value.output_dir) / "response_000.json").exists()
    assert not (Path(raised.value.output_dir) / "proposal.json").exists()


def test_transport_failure_is_unknown_usage_not_zero_cost(tmp_path, monkeypatch):
    fake_http(monkeypatch, [urllib.error.URLError("connection interrupted")])
    with pytest.raises(PolicyOptimizationError) as raised:
        StrategyOptimizer(config(), tmp_path / "optimizer")(proposal_request(tmp_path))
    assert raised.value.cost_usd is None
    assert raised.value.usage["provider_attempts"] == 1
    assert raised.value.usage["unknown_usage_attempts"] == 1


def test_context_cap_pages_evidence_and_preserves_a_full_submission_opportunity(tmp_path, monkeypatch):
    request = proposal_request(tmp_path, trace="x" * 20000)
    def finish(payload):
        assert len(json.dumps(payload["messages"], ensure_ascii=False)) <= 14000
        page = json.loads(payload["messages"][-1]["content"])
        assert page["context_limited"] and 0 < len(page["text"]) < 12000
        assert page["next_offset"] == len(page["text"])
        assert payload["tool_choice"]["function"]["name"] == "submit_policy"
        return submit("C complete", assignment="A complete")

    requests = fake_http(monkeypatch, [response(call("read_training_file", {
        "path": "batch_001/task_001/trace.txt", "limit": 12000})), finish])
    optimizer = StrategyOptimizer(config(), tmp_path / "optimizer", max_context_chars=14000)
    result = optimizer(request)
    assert len(requests) == 2
    assert result.cost_usd == pytest.approx(2 * ONE_CALL_COST)
    assert result.usage["total_tokens"] == 240
    full = json.loads((tmp_path / "optimizer/batch_0001/tool_results_000.json").read_text())
    assert len(full[0]["result"]["text"]) == 12000


def test_near_cap_multiple_tools_keep_pairs_reasoning_and_final_submission(tmp_path, monkeypatch):
    request = proposal_request(tmp_path, trace='"\\\n' * 10000)
    path = "batch_001/task_001/trace.txt"
    reading = response(*(call("read_training_file", {"path": path}, f"read-{i}") for i in range(3)))
    reading["choices"][0]["message"]["reasoning_content"] = "Preserve this provider-required reasoning"

    def finish(payload):
        assert len(json.dumps(payload["messages"], ensure_ascii=False)) <= 18000
        assert payload["messages"][2]["reasoning_content"] == reading["choices"][0]["message"]["reasoning_content"]
        assert [m["tool_call_id"] for m in payload["messages"] if m["role"] == "tool"] == ["read-0", "read-1", "read-2"]
        assert all(json.loads(m["content"])["context_limited"] for m in payload["messages"] if m["role"] == "tool")
        assert [t["function"]["name"] for t in payload["tools"]] == ["submit_policy"]
        assert "remaining_model_calls_including_this\": 1" in payload["messages"][0]["content"]
        return submit("完整 C" * 100, assignment="完整 A" * 100)

    fake_http(monkeypatch, [reading, finish])
    result = StrategyOptimizer(config(), tmp_path / "optimizer", max_context_chars=18000,
                               max_tool_rounds=2)(request)
    assert result.assignment_policy == "完整 A" * 100 and result.communication_policy == "完整 C" * 100
    originals = json.loads((tmp_path / "optimizer/batch_0001/tool_results_000.json").read_text())
    assert len(originals) == 3 and all(len(row["result"]["text"]) == 12000 for row in originals)


def test_oversized_assistant_archives_all_pairs_and_rotates_before_next_request(tmp_path, monkeypatch):
    request = proposal_request(tmp_path)
    reply = response(*(call("list_training_files", {}, f"read-{index}") for index in range(40)),
                     content="Oversized model prose" * 2000)
    requests = fake_http(monkeypatch, [reply, submit()])
    result = StrategyOptimizer(config(), tmp_path / "optimizer", max_context_chars=18000)(request)
    assert len(requests) == result.usage["provider_attempts"] == 2
    assert len(json.dumps(requests[1]["payload"]["messages"], ensure_ascii=False)) <= 18000
    assert [t["function"]["name"] for t in requests[1]["payload"]["tools"]] == ["submit_policy"]
    directory = tmp_path / "optimizer/batch_0001"
    conversation = json.loads((directory / "conversation_000.json").read_text())
    assert len([row for row in conversation if row["role"] == "tool"]) == 40
    assert conversation[2]["content"] == reply["choices"][0]["message"]["content"]
    assert len(json.loads((directory / "tool_results_000.json").read_text())) == 40
    assert (directory / "proposal.json").exists()
    assert json.loads((directory / "research_notes.json").read_text())["rollovers"] == 1


def test_output_limit_is_explicit_failure_even_with_parseable_partial_submission(tmp_path, monkeypatch):
    reply = submit("Partial C", assignment="Partial A")
    reply["choices"][0]["finish_reason"] = "length"
    fake_http(monkeypatch, [reply])
    with pytest.raises(PolicyOptimizationError, match="output token limit") as raised:
        StrategyOptimizer(config(), tmp_path / "optimizer")(proposal_request(tmp_path))
    assert raised.value.usage["total_tokens"] == 120
    directory = tmp_path / "optimizer/batch_0001"
    assert not (directory / "proposal.json").exists()
    conversation = json.loads((directory / "conversation_000.json").read_text())
    assert conversation[-1]["tool_call_id"] == "call-1"


def test_single_model_call_budget_offers_submission_not_evidence(tmp_path, monkeypatch):
    def finish(payload):
        assert [t["function"]["name"] for t in payload["tools"]] == ["submit_policy"]
        assert payload["tool_choice"]["function"]["name"] == "submit_policy"
        return submit()
    fake_http(monkeypatch, [finish])
    result = StrategyOptimizer(config(), tmp_path / "optimizer", max_tool_rounds=1)(proposal_request(tmp_path))
    assert result.usage["provider_attempts"] == 1


def test_final_complete_pair_can_exceed_history_cap_without_another_request(tmp_path, monkeypatch):
    requests = fake_http(monkeypatch, [submit("C" * 8000, assignment="A" * 8000)])
    result = StrategyOptimizer(config(), tmp_path / "optimizer", max_context_chars=10000)(proposal_request(tmp_path))
    assert len(requests) == 1 and len(result.communication_policy) == len(result.assignment_policy) == 8000
    saved = json.loads((tmp_path / "optimizer/batch_0001/conversation_000.json").read_text())
    assert len(json.dumps(saved)) > 10000
    assert json.loads(saved[-1]["content"]) == {"submitted": True}


@pytest.mark.parametrize("invalid", [
    {"communication_policy": "C1", "rationale": "missing assignment"},
    {"assignment_policy": "A1", "rationale": "missing communication"},
    {"assignment_policy": " ", "communication_policy": "C1", "rationale": "empty"},
    {"assignment_policy": "A" * 101, "communication_policy": "C1", "rationale": "oversized"},
    {"assignment_policy": "A1", "communication_policy": "C" * 101, "rationale": "oversized"},
])
def test_optimizer_rejects_partial_or_oversized_pair_and_can_correct_it(tmp_path, monkeypatch, invalid):
    def corrected(payload):
        assert "error" in json.loads(payload["messages"][-1]["content"])
        return submit("C1", assignment="A1")

    requests = fake_http(monkeypatch, [response(call("submit_policy", invalid)), corrected])
    result = StrategyOptimizer(config(), tmp_path / "optimizer", max_policy_chars=100)(proposal_request(tmp_path))
    assert (result.assignment_policy, result.communication_policy) == ("A1", "C1")
    assert len(requests) == 2
    assert result.cost_usd == pytest.approx(2 * ONE_CALL_COST)


def test_plain_text_and_invalid_tools_can_recover_to_valid_submission(tmp_path, monkeypatch):
    request = proposal_request(tmp_path)

    def after_plain_text(payload):
        assert "submit_policy" in payload["messages"][-1]["content"]
        return response(call("submit_policy", {"communication_policy": "", "rationale": "empty"}))

    def after_invalid(payload):
        assert "error" in json.loads(payload["messages"][-1]["content"])
        return submit()

    fake_http(monkeypatch, [response(content="An incomplete suggestion"), after_plain_text, after_invalid])
    result = StrategyOptimizer(config(), tmp_path / "optimizer")(request)
    assert result.communication_policy
    assert result.usage["provider_attempts"] == 3


def test_controller_and_real_optimizer_complete_multiple_batches_without_test_feedback(tmp_path, monkeypatch):
    captured_versions, task_policies = [], []

    def reply(payload):
        initial = json.loads(payload["messages"][1]["content"])
        captured_versions.append(initial["current_version"])
        assert "heldout-repo" not in json.dumps(payload)
        version = initial["current_version"]
        assert initial["assignment_policy"] == f"A{version}"
        assert initial["communication_policy"] == f"C{version}"
        return submit(f"C{version + 1}", assignment=f"A{version + 1}")

    requests = fake_http(monkeypatch, [reply, reply, reply])
    optimizer = StrategyOptimizer(config(), tmp_path / "optimizer")

    def run_task(request):
        task_policies.append((request.split, request.assignment_policy, request.communication_policy))
        (request.output_dir / "trace.txt").write_text("message received, action recorded")
        return TaskResult("completed", score=0.5, usage={"total_tokens": 100}, cost_usd=0.1)

    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=[TaskSpec(f"train-{index}", "train-repo") for index in range(5)],
                        test_tasks=[TaskSpec("heldout", "heldout-repo")], batch_size=2,
                        initial_communication="C0", assignment_policy="A0",
                        run_task=run_task, propose_policy=optimizer, output_dir=tmp_path / "experiment")
    assert captured_versions == [0, 1, 2]
    assert len(requests) == 3
    assert result["status"] == "completed"
    assert result["final_policy_version"] == 3
    assert result["optimizer"]["known_cost_usd"] == pytest.approx(3 * ONE_CALL_COST)
    assert result["optimizer"]["completed"] == 3
    assert result["paired_test_results"][0]["score_difference"] == 0
    assert task_policies == [("train", "A0", "C0"), ("train", "A0", "C0"),
                             ("train", "A1", "C1"), ("train", "A1", "C1"),
                             ("train", "A2", "C2"), ("test", "", ""),
                             ("test", "A3", "C3")]


def test_unknown_usage_keeps_known_cost_subtotal_in_controller(tmp_path, monkeypatch):
    fake_http(monkeypatch, [response(call("list_training_files", {})), submit(usage=None)])
    optimizer = StrategyOptimizer(config(), tmp_path / "optimizer")
    result = run_stage1(selection_mode="legacy_no_replay", train_tasks=[TaskSpec("train-a", "train-repo")],
                        test_tasks=[TaskSpec("test-a", "heldout-repo")],
                        initial_communication="Initial", assignment_policy="Fixed",
                        run_task=lambda _: TaskResult("completed", score=0.5, cost_usd=None,
                                                      known_cost_usd=0.03),
                        propose_policy=optimizer, output_dir=tmp_path / "experiment")
    assert result["optimizer_runs"][0]["cost_usd"] is None
    assert result["optimizer_runs"][0]["known_cost_usd"] == pytest.approx(ONE_CALL_COST)
    assert result["optimizer"]["known_cost_usd"] == pytest.approx(ONE_CALL_COST)
    assert result["optimizer"]["runs_with_unknown_cost"] == 1
    assert result["train"]["known_cost_usd"] == 0.03
    assert result["train"]["runs_with_unknown_cost"] == 1


def test_mixed_known_unknown_failure_preserves_subtotal(tmp_path, monkeypatch):
    fake_http(monkeypatch, [response(call("list_training_files", {})),
                            urllib.error.URLError("interrupted")])
    with pytest.raises(PolicyOptimizationError) as raised:
        StrategyOptimizer(config(), tmp_path / "optimizer")(proposal_request(tmp_path))
    assert raised.value.cost_usd is None
    assert raised.value.known_cost_usd == pytest.approx(ONE_CALL_COST)
    assert raised.value.usage["unknown_usage_attempts"] == 1
    saved = json.loads((Path(raised.value.output_dir) / "summary.json").read_text())
    assert saved["known_cost_usd"] == pytest.approx(ONE_CALL_COST)


@pytest.mark.parametrize("failed_name", ["response_000.json", "conversation_000.json", "summary.json"])
def test_optimizer_trace_write_failure_retains_observed_usage(tmp_path, monkeypatch, failed_name):
    fake_http(monkeypatch, [submit()])
    request = proposal_request(tmp_path)
    original = Path.write_text

    def failing_write(path, *args, **kwargs):
        if path.name in (failed_name, "summary.json"):
            raise OSError("scripted trace storage failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)
    with pytest.raises(PolicyOptimizationError, match="storage failure") as raised:
        StrategyOptimizer(config(), tmp_path / "optimizer")(request)
    assert raised.value.usage["provider_attempts"] == 1
    assert raised.value.usage["total_tokens"] == 120
    assert raised.value.cost_usd == pytest.approx(ONE_CALL_COST)
    assert raised.value.known_cost_usd == pytest.approx(ONE_CALL_COST)


def test_policy_output_and_logs_redact_test_key(tmp_path, monkeypatch):
    provider = config()
    fake_http(monkeypatch, [submit(f"Never retain this accidental credential: {provider.api_key}",
                                  assignment=f"Accidental credential in assignment: {provider.api_key}")])
    result = StrategyOptimizer(provider, tmp_path / "optimizer")(proposal_request(tmp_path))
    assert provider.api_key not in result.communication_policy
    assert provider.api_key not in result.assignment_policy
    for path in (tmp_path / "optimizer").rglob("*"):
        if path.is_file():
            assert provider.api_key not in path.read_text()


def test_recover_committed_optimizer_submission_without_another_call(tmp_path, monkeypatch):
    request = proposal_request(tmp_path)
    calls = fake_http(monkeypatch, [submit("Recovered communication policy")])
    optimizer = StrategyOptimizer(config(), tmp_path / "optimizer")
    result = optimizer(request)
    recovered = optimizer.recover(request)
    assert recovered == result
    assert len(calls) == 1


def test_interrupted_optimizer_retains_known_and_unknown_usage_without_replay(tmp_path, monkeypatch):
    request = proposal_request(tmp_path)
    directory = tmp_path / "optimizer/batch_0001"
    directory.mkdir(parents=True)
    (directory / "request_000.json").write_text("{}")
    (directory / "response_000.json").write_text(json.dumps(response(usage=USAGE)))
    (directory / "request_001.json").write_text("{}")
    calls = fake_http(monkeypatch, [])
    optimizer = StrategyOptimizer(config(), tmp_path / "optimizer")
    with pytest.raises(PolicyOptimizationError) as caught:
        optimizer.recover(request)
    assert caught.value.usage["provider_attempts"] == 2
    assert caught.value.usage["unknown_usage_attempts"] == 1
    assert caught.value.usage["total_tokens"] == 120
    assert caught.value.known_cost_usd == pytest.approx(ONE_CALL_COST)
    assert caught.value.cost_usd is None
    assert not calls
