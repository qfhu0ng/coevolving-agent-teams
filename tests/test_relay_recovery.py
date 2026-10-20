"""Bounded retries and native caller metadata, without external model calls."""

import copy
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from agent_company.runtime.relay import ProviderConfig, TeamRelay, TokenRates, start_server
from agent_company.runtime.team import WorkerRegistry
from test_team import call, request, response, workspace
from test_transport import peer, reply, wait_for_client_close


def make_relay(tmp_path, upstream=None, *, strict=False, mode="team", **config):
    registry = WorkerRegistry(workspace(tmp_path))
    return TeamRelay(ProviderConfig("fake", config.pop("base_url", "http://localhost"),
                                    rates=TokenRates("1", "0.1", "2"), **config),
                     registry, "COMMUNICATION_SENTINEL", tmp_path / "logs", 30, upstream,
                     execution_mode=mode, require_caller_metadata=strict)


def test_retry_attempts_are_sequential_and_individually_metered(tmp_path):
    seen = []

    def upstream(payload):
        seen.append(copy.deepcopy(payload))
        if len(seen) == 1:
            return 503, {"error": {"message": "busy"},
                         "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        if len(seen) == 2:
            raise ConnectionError("disconnected")
        return 200, response(usage={"prompt_tokens": 11, "completion_tokens": 4})

    relay = make_relay(tmp_path, upstream, retry_backoff_seconds=0)
    assert relay.handle(request())[0] == 200
    assert len(seen) == 3 and seen[0] == seen[1] == seen[2]
    summary = relay.summary()
    assert summary["provider_attempts"] == summary["completed_attempts"] == 3
    assert summary["inflight_attempts"] == 0 and not relay._active_requests
    assert summary["usage"]["total_tokens"] == 25
    assert not summary["usage_complete"] and not summary["cost_complete"]
    assert summary["known_estimated_cost"] == "0.000032"
    assert [r["retry_index"] for r in relay._records] == [0, 1, 2]
    assert [r["error_kind"] for r in summary["errors"]] == [
        "provider_error", "provider_transport_error"]
    assert len(list((tmp_path / "logs").glob("*_provider_request.json"))) == 3


def test_expired_http_request_is_explicit_budget_end_without_upstream(tmp_path):
    calls = []
    relay = make_relay(tmp_path, lambda p: calls.append(p))
    relay.deadline = time.monotonic() - 1
    server = start_server(relay)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps(request()).encode(),
            headers={"Authorization": "Bearer " + relay.token, "X-Agent-Company-Owner": "master"})
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        assert caught.value.code == 408
        assert json.loads(caught.value.read())["error"]["code"] == "task_deadline_exceeded"
        assert relay.summary()["deadline_rejections"] == ["master"]
        assert calls == []
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("status,raw", [
    (401, {"error": {"message": "503 is mentioned but this is authentication"}}),
    (403, {"error": {"message": "forbidden"}}),
    (400, {"error": {"message": "bad request"}}),
    (200, None),
    (200, {"error": {"message": "unknown error without retryable status"}}),
])
def test_nontransient_or_invalid_response_not_blindly_retried(tmp_path, status, raw):
    seen = []
    relay = make_relay(tmp_path, lambda p: (seen.append(p) or (status, raw)),
                       retry_backoff_seconds=0)
    assert relay.handle(request())[0] == 424
    assert len(seen) == 1 and relay.summary()["provider_attempts"] == 1


def test_tool_rejection_is_feedback_not_retry(tmp_path):
    relay = make_relay(tmp_path, lambda _: (200, response([
        call("run_shell_command", {"command": "must not run as master"})])),
        retry_backoff_seconds=0)
    status, result = relay.handle(request())
    assert status == 200
    assert "__agent_company_tool_error" in json.dumps(result)
    assert relay.summary()["provider_attempts"] == 1


def test_retry_limit_stops_after_three_actual_calls(tmp_path):
    relay = make_relay(tmp_path, lambda _: (503, {"error": "busy"}), retry_backoff_seconds=0)
    assert relay.handle(request())[0] == 424
    assert relay.summary()["provider_attempts"] == 3
    assert not relay._active_requests and not relay._controls


def test_retry_backoff_keeps_owner_reserved_and_can_be_cancelled(tmp_path):
    scheduled = threading.Event()
    calls = []

    def upstream(payload):
        calls.append(payload)
        if "COMMUNICATION_SENTINEL" in json.dumps(payload):
            return 200, response(usage={"prompt_tokens": 1, "completion_tokens": 1})
        return 503, {"error": "busy", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    relay = make_relay(tmp_path, upstream, retry_backoff_seconds=2)
    relay.registry.register("builder")
    original = relay.event

    def event(name, **fields):
        original(name, **fields)
        if name == "provider_retry_scheduled":
            scheduled.set()

    relay.event = event
    result = []
    thread = threading.Thread(target=lambda: result.append(relay.handle(request())))
    thread.start()
    try:
        assert scheduled.wait(2)
        snapshot = relay.summary()
        assert snapshot["active_requests"] == 1 and snapshot["inflight_attempts"] == 0
        assert not snapshot["cost_complete"] and not snapshot["usage_complete"]
        with pytest.raises(ValueError, match="still in flight"):
            relay.handle(request())
        assert relay.handle(request("builder"))[0] == 200
        relay.close()
    finally:
        relay.close()
        thread.join(2)
    assert not thread.is_alive()
    assert result[0][1]["error"]["code"] == "request_cancelled"
    assert len(calls) == 2  # One master attempt, one independent worker; no retry after cancel.
    assert not relay._active_requests
    saved = json.loads((tmp_path / "logs/usage.json").read_text())
    assert saved["active_requests"] == 0 and saved["cost_complete"] and saved["usage_complete"]


def test_real_transport_retries_share_original_deadline(tmp_path):
    def upstream(handler, state):
        if len(state.requests) < 3:
            time.sleep(0.04)
            reply(handler, {"error": "temporarily busy"}, status=503)
        else:
            wait_for_client_close(handler, state)

    with peer(upstream) as (url, state):
        relay = make_relay(tmp_path, base_url=url, request_timeout_seconds=0.25,
                           retry_backoff_seconds=0.01)
        start = time.monotonic()
        status, body = relay.handle(request())
        elapsed = time.monotonic() - start
        assert status == 424 and body["error"]["code"] == "provider_timeout"
        assert 0.20 <= elapsed < 0.7
        assert len(state.requests) == 3 and state.disconnected.wait(1)
        assert relay.summary()["provider_attempts"] == 3


def test_retry_evidence_failure_stops_without_another_provider_call(tmp_path, monkeypatch):
    relay = make_relay(tmp_path, lambda _: (503, {"error": "busy"}), retry_backoff_seconds=0)
    original = relay.event

    def event(name, **fields):
        if name == "provider_retry_scheduled":
            raise OSError("disk full")
        return original(name, **fields)

    monkeypatch.setattr(relay, "event", event)
    status, body = relay.handle(request())
    assert status == 424 and body["error"]["code"] == "evidence_write_failed"
    assert relay.summary()["provider_attempts"] == 1
    assert relay.summary()["errors"][0]["error_kind"] == "evidence_write_failed"


@pytest.mark.parametrize("phase,kind", [("worktree", "relay_execution_error"),
                                       ("audit", "evidence_write_failed")])
def test_local_spawn_failures_never_retry_paid_model_request(tmp_path, monkeypatch, phase, kind):
    calls = []

    def upstream(payload):
        calls.append(payload)
        return 200, response([call("agent", {"name": "builder", "prompt": "implement"})],
                             usage={"prompt_tokens": 7, "completion_tokens": 3})

    relay = make_relay(tmp_path, upstream, retry_backoff_seconds=0)
    if phase == "worktree":
        def register(_):
            raise OSError("worktree creation failed")
        monkeypatch.setattr(relay.registry, "register", register)
    else:
        original = relay.event

        def event(event_name, **fields):
            if event_name == "worker_registered":
                raise OSError("audit disk full")
            return original(event_name, **fields)
        monkeypatch.setattr(relay, "event", event)
    status, body = relay.handle(request())
    assert status == 424 and body["error"]["code"] == kind
    assert len(calls) == relay.summary()["provider_attempts"] == 1
    assert relay.summary()["usage"]["total_tokens"] == 10


def test_programming_error_in_provider_client_is_not_treated_as_a_network_retry(tmp_path):
    def upstream(_):
        raise TypeError("wrong client argument")
    relay = make_relay(tmp_path, upstream, retry_backoff_seconds=0)
    status, body = relay.handle(request())
    assert status == 424 and body["error"]["code"] == "provider_client_error"
    assert relay.summary()["provider_attempts"] == 1


def test_auxiliary_owner_and_purpose_are_independent_of_system_text(tmp_path):
    relay = make_relay(tmp_path, strict=True, max_output_tokens=9000,
                       extra_body={"temperature": 0.9})
    relay.registry.register("builder")
    body = {"messages": [{"role": "system", "content": "Summarize the conversation."}],
            "max_tokens": 321, "temperature": 0.1, "tools": []}
    role, payload = relay.payload(body, owner="builder", purpose="chat-compression")
    assert role == "builder" and payload["max_tokens"] == 321
    assert payload["temperature"] == 0.1 and payload["tools"] == []
    assert "COMMUNICATION_SENTINEL" not in json.dumps(payload)
    from agent_company.policy.source_boundary import SOURCE_BOUNDARY
    assert payload["messages"] == [{"role": "system", "content":
                                     "Summarize the conversation.\n\n" + SOURCE_BOUNDARY}]
    assert body["messages"] == [{"role": "system", "content": "Summarize the conversation."}]
    with pytest.raises(ValueError, match="owner and purpose"):
        relay.handle(body)
    with pytest.raises(ValueError, match="disagrees"):
        relay.payload(request("builder"), owner="master", purpose="coding")


def test_auxiliary_response_cannot_register_or_execute_workers(tmp_path):
    relay = make_relay(tmp_path, lambda _: (200, response([
        call("agent", {"name": "accidental", "prompt": "do not run"})])))
    status, body = relay.handle(request(), owner="master", purpose="chat-compression")
    assert status == 200 and "Auxiliary model requests" in json.dumps(body)
    assert relay.registry.snapshot() == {} and relay.summary()["provider_attempts"] == 1


@pytest.mark.parametrize("mode,owner", [("team", "master"), ("single", "single")])
def test_native_headers_are_consumed_locally_and_missing_metadata_is_logged(tmp_path, mode, owner):
    with peer(lambda h, _: reply(h, response(usage={"prompt_tokens": 1, "completion_tokens": 1}))) as (url, state):
        relay = make_relay(tmp_path, base_url=url, strict=True, mode=mode)
        server = start_server(relay)
        try:
            headers = {"Authorization": "Bearer " + relay.token, "Content-Type": "application/json"}
            destination = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(destination, data=json.dumps(request()).encode(),
                                                             headers=headers), timeout=3)
            assert error.value.code == 400 and relay.summary()["provider_attempts"] == 0
            headers.update({"X-Agent-Company-Owner": owner, "X-Agent-Company-Purpose": "coding"})
            with urllib.request.urlopen(urllib.request.Request(destination, data=json.dumps(request()).encode(),
                                                               headers=headers), timeout=3) as result:
                assert result.status == 200
            assert len(state.requests) == 1
            assert not any(k.lower().startswith("x-agent-company") for k in state.requests[0]["headers"])
            events = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()]
            rejected = next(e for e in events if e["event"] == "request_rejected")
            assert rejected["owner"] is None and rejected["conflicting_attempts"] == []
        finally:
            relay.close()
            server.shutdown()
            server.server_close()


@pytest.mark.parametrize("config", [
    {"max_attempts": 0}, {"max_attempts": True}, {"max_attempts": 2.5},
    {"retry_backoff_seconds": -1}, {"retry_backoff_seconds": float("inf")},
    {"retry_backoff_seconds": True},
])
def test_retry_configuration_is_validated(config):
    with pytest.raises(ValueError):
        ProviderConfig("fake", "http://localhost", **config)
