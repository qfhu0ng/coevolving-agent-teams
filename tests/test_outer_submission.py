"""DeepSeek/GLM submission compatibility, with no live provider calls."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.optimizer import PolicyOptimizationError, StrategyOptimizer
from agent_company.evolution.outer_submission import submission_tool_choice
from agent_company.evolution.reviewer import StrategyReviewer
from test_optimizer import call, config, fake_http, proposal_request, response, submit
from test_reviewer import review_request, submit as submit_review


THINKING = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}


def deepseek():
    return replace(config(), model="deepseek-flash", base_url="https://api.deepseek.com/v1",
                   extra_body=copy.deepcopy(THINKING))


@pytest.fixture(params=["optimizer", "reviewer"])
def outer(request, tmp_path):
    if request.param == "optimizer":
        return SimpleNamespace(kind="optimizer", cls=StrategyOptimizer,
                               request=proposal_request(tmp_path), submit=submit,
                               tool="submit_policy", record="proposal.json", context_limit=14000)
    return SimpleNamespace(kind="reviewer", cls=StrategyReviewer,
                           request=review_request(tmp_path), submit=submit_review,
                           tool="submit_review", record="review.json", context_limit=16000)


@pytest.mark.parametrize("base_url,model,extra,auto", [
    ("https://api.deepseek.com/v1", "deepseek-flash", THINKING, True),
    ("https://api.deepseek.com/beta", "deepseek-v4-pro", {}, True),
    ("https://api.deepseek.com/v1", "deepseek-flash", {"thinking": {"type": "disabled"}}, False),
    ("https://api.deepseek.com/v1", "deepseek-flash", {"reasoning_effort": "none"}, False),
    ("https://api.deepseek.com/v1", "deepseek-flash", {"reasoning_effort": "max"}, True),
    ("https://api.deepseek.com/v1", "deepseek-flash", {"thinking": {"type": "disabled"},
                                                       "reasoning_effort": "high"}, True),
    ("https://api.deepseek.com/v1", "deepseek-flash", {"thinking": {"type": "disabled"},
                                                       "reasoning_effort": "ultra"}, True),
    ("https://api.deepseek.com/v1", "deepseek-flash", {"thinking": {"type": "enabled"},
                                                       "reasoning_effort": "none"}, True),
    ("https://proxy.invalid/v1", "deepseek/deepseek-v4-pro", THINKING, True),
    ("https://proxy.invalid/v1", "deepseek-reasoner", {}, False),
    ("https://proxy.invalid/v1", "deepseek-chat", {}, False),
    ("https://proxy.invalid/v1", "deepseek-v4-pro", {}, False),
    ("https://provider.invalid/v1", "ordinary-model", THINKING, False),
    ("https://api.deepseek.com.not-deepseek.invalid/v1", "ordinary-model", THINKING, False),
    ("https://provider.invalid/api.deepseek.com/v1", "ordinary-model", THINKING, False),
    ("https://open.bigmodel.cn/api/paas/v4", "glm-5.3", {"reasoning_effort": "max"}, True),
    ("https://api.z.ai/api/paas/v4", "glm-5.3", {}, True),
    ("https://api.z.ai/api/coding/paas/v4", "glm-5.3", THINKING, True),
    ("https://proxy.invalid/v1", "glm-5.3", THINKING, False),
    ("https://open.bigmodel.cn.not-official.invalid/v4", "glm-5.3", THINKING, False),
    ("https://open.bigmodel.cn/api/paas/v4", "ordinary-model", THINKING, False),
])
def test_submission_choice_is_provider_specific_and_does_not_change_thinking(base_url, model, extra, auto):
    provider = replace(config(), base_url=base_url, model=model, extra_body=copy.deepcopy(extra))
    payload = copy.deepcopy(extra) | {"tool_choice": "required"}
    submission_tool_choice(payload, provider, "submit_policy")
    if auto:
        assert "tool_choice" not in payload  # Tools present => documented default auto.
    else:
        assert payload["tool_choice"] == {"type": "function", "function": {"name": "submit_policy"}}
    assert {key: value for key, value in payload.items() if key != "tool_choice"} == extra
    assert provider.extra_body == extra


@pytest.mark.parametrize("thinking", [True, False])
def test_single_call_http_submission_preserves_thinking_and_generic_forcing(tmp_path, monkeypatch, outer, thinking):
    provider = deepseek() if thinking else config()

    def finish(payload):
        assert [tool["function"]["name"] for tool in payload["tools"]] == [outer.tool]
        if thinking:
            assert "tool_choice" not in payload
            assert payload["thinking"] == THINKING["thinking"]
            assert payload["reasoning_effort"] == "high"
        else:
            assert payload["tool_choice"]["function"]["name"] == outer.tool
        assert payload["max_tokens"] == provider.max_output_tokens
        notice = json.loads(payload["messages"][0]["content"].split("Research-loop budget: ", 1)[1])
        assert notice["phase"] == "submit_now" and notice["remaining_model_calls_including_this"] == 1
        return outer.submit()

    requests = fake_http(monkeypatch, [finish])
    result = outer.cls(provider, tmp_path / outer.kind, max_tool_rounds=1)(outer.request)
    assert result.usage["provider_attempts"] == len(requests) == 1
    assert 0 < requests[0]["timeout"] <= provider.request_timeout_seconds
    assert (tmp_path / outer.kind / "batch_0001" / outer.record).is_file()


def test_glm53_final_submission_keeps_budget_and_validates_tool(tmp_path, monkeypatch, outer):
    provider = replace(config(), model="glm-5.3", base_url="https://open.bigmodel.cn/api/paas/v4",
                       extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "max"})

    def finish(payload):
        assert "tool_choice" not in payload
        assert [tool["function"]["name"] for tool in payload["tools"]] == [outer.tool]
        assert payload["reasoning_effort"] == "max"
        assert payload["thinking"] == {"type": "enabled"}
        return outer.submit()

    requests = fake_http(monkeypatch, [finish])
    result = outer.cls(provider, tmp_path / outer.kind, max_tool_rounds=1)(outer.request)
    assert result.usage["provider_attempts"] == len(requests) == 1
    assert (tmp_path / outer.kind / "batch_0001" / outer.record).is_file()


def test_http_replays_actual_reasoning_after_tools_and_plain_text_before_submission(tmp_path, monkeypatch, outer):
    read = response(call("read_training_file", {"path": "batch_001/task_001/trace.txt"}, "read-1"))
    read["choices"][0]["message"]["reasoning_content"] = "Provider reasoning from the evidence read."
    plain = response(content="A hypothesis, not a submission.")
    plain["choices"][0]["message"]["reasoning_content"] = "Provider reasoning from the plain-text turn."

    def after_read(payload):
        assert payload["messages"][2] == read["choices"][0]["message"]
        assert payload["messages"][3]["tool_call_id"] == "read-1"
        assert "training evidence" in payload["messages"][3]["content"] or outer.kind == "reviewer"
        return plain

    def finish(payload):
        assistants = [message for message in payload["messages"] if message["role"] == "assistant"]
        assert assistants == [read["choices"][0]["message"], plain["choices"][0]["message"]]
        assert outer.tool in payload["messages"][-1]["content"]
        assert "tool_choice" not in payload
        assert [tool["function"]["name"] for tool in payload["tools"]] == [outer.tool]
        assert payload["thinking"] == THINKING["thinking"] and payload["reasoning_effort"] == "high"
        return outer.submit()

    requests = fake_http(monkeypatch, [read, after_read, finish])
    result = outer.cls(deepseek(), tmp_path / outer.kind, max_tool_rounds=3)(outer.request)
    assert len(requests) == result.usage["provider_attempts"] == 3
    assert result.usage["total_tokens"] == 360
    assert requests[2]["payload"]["messages"][2]["reasoning_content"] == read["choices"][0]["message"]["reasoning_content"]


def test_context_pressure_keeps_reasoning_and_only_submit_tool_with_auto(tmp_path, monkeypatch, outer):
    (outer.request.evidence.root / "batch_001/task_001/trace.txt").write_text("x" * 20000)
    read = response(call("read_training_file", {"path": "batch_001/task_001/trace.txt"}, "read-1"))
    read["choices"][0]["message"]["reasoning_content"] = "Provider reasoning retained near the context cap."

    def finish(payload):
        assert len(json.dumps(payload["messages"], ensure_ascii=False)) <= outer.context_limit
        assert payload["messages"][2] == read["choices"][0]["message"]
        assert json.loads(payload["messages"][-1]["content"])["context_limited"] is True
        assert [tool["function"]["name"] for tool in payload["tools"]] == [outer.tool]
        assert "tool_choice" not in payload
        notice = json.loads(payload["messages"][0]["content"].split("Research-loop budget: ", 1)[1])
        assert notice["phase"] == "submit_now" and notice["remaining_model_calls_including_this"] == 3
        return outer.submit()

    requests = fake_http(monkeypatch, [read, finish])
    result = outer.cls(deepseek(), tmp_path / outer.kind, max_tool_rounds=4,
                       max_context_chars=outer.context_limit)(outer.request)
    assert len(requests) == result.usage["provider_attempts"] == 2


@pytest.mark.parametrize("reply_kind", ["plain", "invalid", "closed_evidence", "length", "http_400"])
def test_auto_submission_failure_is_bounded_accounted_and_never_committed(tmp_path, monkeypatch, outer, reply_kind):
    if reply_kind == "plain":
        reply = response(content="I have not submitted.")
    elif reply_kind == "invalid":
        reply = response(call(outer.tool, {}))
    elif reply_kind == "closed_evidence":
        reply = response(call("list_training_files", {}))
    else:
        reply = outer.submit()
    if reply_kind == "length":
        reply["choices"][0]["finish_reason"] = "length"
    elif reply_kind == "http_400":
        reply = (400, {"error": {"message": "Scripted rejection"}, "usage": reply["usage"]})

    def forbidden_read(*args, **kwargs):
        pytest.fail("Submission-only must not execute evidence tools, even if auto emits one")

    monkeypatch.setattr(TrainingEvidence, "list_files", forbidden_read)
    requests = fake_http(monkeypatch, [reply])
    loop = outer.cls(deepseek(), tmp_path / outer.kind, max_tool_rounds=1)
    expected = "output token limit" if reply_kind == "length" else "HTTP 400" if reply_kind == "http_400" else "interaction limit"
    with pytest.raises(PolicyOptimizationError, match=expected) as failed:
        loop(outer.request)
    assert len(requests) == failed.value.usage["provider_attempts"] == 1
    assert failed.value.usage["total_tokens"] == 120 and failed.value.known_cost_usd > 0
    directory = tmp_path / outer.kind / "batch_0001"
    assert not (directory / outer.record).exists()
    if reply_kind == "closed_evidence":
        results = json.loads((directory / "tool_results_000.json").read_text())
        assert "Evidence reading is closed" in results[0]["result"]["error"]
    with pytest.raises(PolicyOptimizationError):
        loop.recover(outer.request)
    assert len(requests) == 1  # Recovery never buys a new submission attempt.
