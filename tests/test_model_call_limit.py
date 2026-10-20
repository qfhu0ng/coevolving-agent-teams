"""Shared logical-call budget: synthetic upstream and mocked native runtime only."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from agent_company.runtime import qwen
from agent_company.runtime.relay import MODEL_CALL_LIMIT_MESSAGE, ProviderConfig, TeamRelay
from agent_company.runtime.transport import RequestCancelled, RequestControl


def body():
    return {"messages": [{"role": "user", "content": "Synthetic task"}], "tools": []}


def completion():
    return {"choices": [{"message": {"role": "assistant", "content": "synthetic"},
                          "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


def relay_at(path, *, limit=2, mode="single", upstream=None, names=(), attempts=3):
    registry = SimpleNamespace(contains=lambda name: name in names,
                               snapshot=lambda: {name: f"/workers/{name}" for name in names},
                               metadata=lambda: {name: {"working_dir": f"/workers/{name}",
                                                         "creation_head": None} for name in names})
    return TeamRelay(ProviderConfig("fake", "http://unused.invalid", max_attempts=attempts,
                                    retry_backoff_seconds=0), registry, "test policy", path, 60,
                     upstream or (lambda _: (200, completion())), execution_mode=mode,
                     require_caller_metadata=True, max_model_calls=limit)


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "2", float("nan")])
def test_invalid_limits_fail_before_output_or_upstream(tmp_path, limit):
    output = tmp_path / "relay"
    with pytest.raises(ValueError, match="max_model_calls"):
        relay_at(output, limit=limit)
    assert not output.exists()


def test_single_budget_counts_admitted_calls_and_rejects_before_upstream(tmp_path):
    seen = []
    relay = relay_at(tmp_path, upstream=lambda payload: (seen.append(payload) or (200, completion())))
    for _ in range(2):
        assert relay.handle(body(), owner="single", purpose="coding")[0] == 200
    status, rejected = relay.handle(body(), owner="single", purpose="coding")
    assert status == 424 and rejected["error"]["code"] == "model_call_limit_exceeded"
    assert rejected["error"]["message"] == MODEL_CALL_LIMIT_MESSAGE
    summary = relay.summary()
    assert len(seen) == summary["logical_model_calls"] == summary["provider_attempts"] == 2
    assert summary["max_model_calls"] == 2
    assert summary["logical_calls_by_role"] == {"single": 2}
    assert summary["logical_calls_by_purpose"] == {"coding": 2}
    assert summary["model_call_budget_rejections"] == 1
    assert summary["model_call_budget_rejections_by_role"] == {"single": 1}
    assert summary["usage_complete"] is True
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event"] == "logical_model_call_admitted"]) == 2
    assert events[-1]["event"] == "model_call_limit_rejected"
    assert events[-1]["upstream_started"] is False
    assert json.loads((tmp_path / "usage.json").read_text())["model_call_budget_rejections"] == 1


def test_team_shares_budget_with_every_worker_and_auxiliary_summary(tmp_path):
    relay = relay_at(tmp_path, mode="team", limit=3, names=("builder",))
    for role, purpose in (("master", "coding"), ("builder", "coding"), ("builder", "chat-compression")):
        assert relay.handle(body(), owner=role, purpose=purpose)[0] == 200
    assert relay.handle(body(), owner="master", purpose="coding")[0] == 424
    summary = relay.summary()
    assert summary["logical_calls_by_role"] == {"master": 1, "builder": 2}
    assert summary["logical_calls_by_purpose"] == {"coding": 2, "chat-compression": 1}
    assert summary["provider_attempts"] == summary["logical_model_calls"] == 3


def test_transport_retries_use_one_reservation_and_identical_warned_payload(tmp_path):
    seen = []
    original = body()

    def upstream(payload):
        seen.append(copy.deepcopy(payload))
        return (524, {"error": {"message": "synthetic"}}) if len(seen) < 3 else (200, completion())

    relay = relay_at(tmp_path, limit=1, upstream=upstream)
    assert relay.handle(original, owner="single", purpose="coding")[0] == 200
    assert seen[0] == seen[1] == seen[2]
    assert "0 further calls remain" in json.dumps(seen[0])
    assert original == body()
    summary = relay.summary()
    assert summary["logical_model_calls"] == 1
    assert summary["calls_by_role"] == {"single": 3}  # Existing field retains attempt semantics.
    assert summary["logical_calls_by_role"] == {"single": 1}
    assert summary["provider_attempts"] == 3 and summary["usage_complete"] is False
    assert relay.handle(body(), owner="single", purpose="coding")[0] == 424
    assert len(seen) == 3


def test_unlimited_default_has_no_limit_or_budget_prompt(tmp_path):
    seen = []
    relay = relay_at(tmp_path, limit=None, upstream=lambda p: (seen.append(p) or (200, completion())))
    for _ in range(3):
        assert relay.handle(body(), owner="single", purpose="coding")[0] == 200
    assert relay.summary()["max_model_calls"] is None
    assert relay.summary()["logical_model_calls"] == 3
    assert relay.summary()["model_call_budget_rejections"] == 0
    assert "Shared logical model-call budget" not in json.dumps(seen)


def test_warning_starts_when_twenty_further_calls_remain(tmp_path):
    seen = []
    relay = relay_at(tmp_path, limit=22, upstream=lambda p: (seen.append(p) or (200, completion())))
    relay.handle(body(), owner="single", purpose="coding")
    relay.handle(body(), owner="single", purpose="coding")
    assert "Shared logical model-call budget" not in json.dumps(seen[0])
    assert "20 further calls remain" in json.dumps(seen[1])


def test_concurrent_roles_cannot_exceed_shared_admission_limit(tmp_path):
    names = tuple(f"worker{i}" for i in range(8))
    start = threading.Barrier(len(names))
    admitted = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    seen = []

    def upstream(payload):
        with lock:
            seen.append(payload)
            if len(seen) >= 3:
                admitted.set()
        assert release.wait(3)
        return 200, completion()

    relay = relay_at(tmp_path, mode="team", names=names, limit=3, upstream=upstream)

    def call(name):
        start.wait(timeout=3)
        return relay.handle(body(), owner=name, purpose="coding")[0]

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = [pool.submit(call, name) for name in names]
        try:
            assert admitted.wait(3)
        finally:
            release.set()
        statuses = [future.result(timeout=3) for future in futures]
    assert statuses.count(200) == 3 and statuses.count(424) == 5
    assert len(seen) == relay.summary()["logical_model_calls"] == 3
    assert relay.summary()["model_call_budget_rejections"] == 5
    assert relay.summary()["active_requests"] == 0


def test_cancelled_unknown_and_outstanding_requests_do_not_spend_steps(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def upstream(_):
        entered.set()
        assert release.wait(3)
        return 200, completion()

    relay = relay_at(tmp_path, mode="team", names=("builder",), upstream=upstream)
    with pytest.raises(ValueError, match="Unregistered"):
        relay.handle(body(), owner="unknown", purpose="coding")
    control = RequestControl(30)
    control.cancel()
    with pytest.raises(RequestCancelled):
        relay.handle(body(), owner="builder", purpose="coding", control=control)
    assert relay.summary()["logical_model_calls"] == 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(relay.handle, body(), owner="master", purpose="coding")
        try:
            assert entered.wait(3)
            with pytest.raises(ValueError, match="still in flight"):
                relay.handle(body(), owner="master", purpose="coding")
            assert relay.summary()["logical_model_calls"] == 1
        finally:
            release.set()
        assert future.result(timeout=3)[0] == 200


@pytest.mark.parametrize("event_name", ["logical_model_call_admitted", "model_call_limit_rejected"])
def test_budget_evidence_failure_never_calls_provider_or_claims_normal_budget_end(tmp_path, event_name):
    calls = []
    relay = relay_at(tmp_path, limit=1, upstream=lambda p: (calls.append(p) or (200, completion())))
    if event_name == "model_call_limit_rejected":
        relay.handle(body(), owner="single", purpose="coding")
    before = len(calls)
    original = relay.event

    def broken(event, **fields):
        if event == event_name:
            raise OSError("synthetic audit storage failure")
        return original(event, **fields)

    relay.event = broken
    status, result = relay.handle(body(), owner="single", purpose="coding")
    assert status == 424 and result["error"]["code"] == "evidence_write_failed"
    assert len(calls) == before
    assert relay.summary()["errors"][-1]["error_kind"] == "evidence_write_failed"


def native_error(text=MODEL_CALL_LIMIT_MESSAGE, *, explicit=True):
    return {"explicit_error": explicit, "missing_result": False,
            "result": {"type": "result", "result": f"[API Error: {text}]"}}


def test_budget_stop_requires_local_rejection_and_explicit_native_marker():
    execution = {"returncode": 0, "timed_out": False}
    summary = {"model_call_budget_rejections": 1, "errors": []}
    assert qwen.runtime_stop_reason(execution, native_error(), None, summary) == "model_call_limit_exceeded"
    assert qwen.runtime_stop_reason(execution, native_error(), None, {}) == "cli_error"
    assert qwen.runtime_stop_reason(execution, native_error("unrelated failure"), None, summary) == "cli_error"
    assert qwen.runtime_stop_reason({**execution, "returncode": 1}, native_error(), None, summary) == "cli_error"
    assert qwen.runtime_stop_reason(execution, native_error(explicit=False), None, summary) == "native_cli_completed"
    assert qwen.runtime_stop_reason(execution, native_error(), "launch failure", summary) == "launcher_error"


def test_budget_stop_does_not_hide_evidence_lifecycle_or_timeout():
    execution = {"returncode": 0, "timed_out": False}
    summary = {"model_call_budget_rejections": 1, "errors": []}
    fatal = {**summary, "errors": [{"error_kind": "evidence_write_failed", "error": "write failed"}]}
    assert qwen.runtime_stop_reason(execution, native_error(), None, fatal) == "provider_error"
    native = native_error()
    native["lifecycle"] = {"unobserved_failure_notices": ["unrelated-worker-failure"]}
    assert qwen.runtime_stop_reason(execution, native, None, summary) == "unhandled_worker_failure"
    assert qwen.runtime_stop_reason({**execution, "timed_out": True}, native_error(), None, summary) == "task_timeout"


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("limit", [None, 1000])
def test_qwen_forwards_limit_and_records_honest_prompt_without_native_execution(tmp_path, monkeypatch, mode, limit):
    captured = {}
    ws = SimpleNamespace(repo_path="/workspace/solution", worker_root="/workers", runtime_root="/runtime",
                         relay_bind_host="127.0.0.1", relay_connect_host="127.0.0.1")
    ws.write_file = lambda path, content: None

    def start_server(relay, host):
        captured["relay"] = relay
        return SimpleNamespace(server_port=12345, shutdown=lambda: None, server_close=lambda: None)

    def run_agent(command, **kwargs):
        captured["command"] = command
        Path(kwargs["stdout_path"]).write_text(json.dumps({"type": "result", "subtype": "success", "result": "done"}))
        Path(kwargs["stderr_path"]).write_text("")
        return {"returncode": 0, "timed_out": False}

    ws.run_agent = run_agent
    monkeypatch.setattr(qwen, "start_server", start_server)
    output = tmp_path / "native"
    result = qwen.QwenTeamRunner("unused", upstream=lambda _: pytest.fail("No model call expected")).run(
        ws, "Public synthetic task", "assignment", "communication", output,
        ProviderConfig("fake", "http://unused.invalid"), 21600, execution_mode=mode, max_model_calls=limit)
    launch = json.loads((output / "launch.json").read_text())
    assert captured["relay"].max_model_calls == result["max_model_calls"] == launch["max_model_calls"] == limit
    prompt = captured["command"][-1]
    if limit is None:
        assert "No fixed" in prompt
        assert launch["model_call_limit_scope"] is None
    else:
        assert "shared limit of 1000 logical model calls" in prompt
        assert "summaries" in prompt and "Transport retries" in prompt
        assert "No fixed" not in prompt
        assert launch["model_call_limit_scope"] == "task-shared"
        assert launch["official_multi_agent_call_limit_claimed"] is False


def test_qwen_rejects_invalid_limit_before_workspace_mutation(tmp_path):
    ws = SimpleNamespace(write_file=lambda *_: pytest.fail("Must validate first"))
    with pytest.raises(ValueError, match="max_model_calls"):
        qwen.QwenTeamRunner("unused").run(ws, "task", "", "", tmp_path / "native",
            ProviderConfig("fake", "http://unused.invalid"), 30, execution_mode="single", max_model_calls=0)
    assert not (tmp_path / "native").exists()
