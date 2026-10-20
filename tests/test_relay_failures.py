"""No-provider-call regressions for the 2026-09-14 relay incident."""

from contextlib import contextmanager
import copy
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from agent_company.runtime.qwen import provider_incident_summary, runtime_stop_reason
from agent_company.runtime.relay import (
    OpenAIChatTransport, ProviderConfig, TeamRelay, TokenRates, start_server, stream_response,
)
from agent_company.runtime.team import WorkerRegistry
from agent_company.runtime.transport import RequestDeadlineExceeded
from test_team import assert_tool_rejection, call, request, response, workspace
from test_transport import peer, reply as http_reply


@contextmanager
def serving(relay):
    server = start_server(relay)
    try:
        yield server
    finally:
        relay.close()
        relay.drain(2)
        server.shutdown()
        server.server_close()


def post(relay, server, body, timeout=2):
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + relay.token},
    )
    return urllib.request.urlopen(req, timeout=timeout)


@pytest.mark.parametrize("raw", [
    {"error": {"message": "Unable to start processing within the 900-second timeout"}},
    {"error": {}}, {}, {"choices": []}, [], None,
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"role": "assistant"}}]},
    {"choices": [{"message": {"role": "user", "content": "wrong role"}}]},
    response([call("", {})]), response([call(" \t", {})]),
])
def test_http_200_error_or_empty_completion_never_becomes_success(tmp_path, raw):
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", max_attempts=1),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 30,
                      lambda _: (200, raw))
    status, delivered = relay.handle(request())
    assert status == 424 and delivered["error"]["provider_http_status"] == 200
    assert delivered["error"]["code"] in {"provider_error", "invalid_provider_response"}
    summary = relay.summary()
    assert len(summary["errors"]) == 1 and summary["completed_attempts"] == 1
    assert not summary["usage_complete"] and not summary["cost_complete"]
    assert summary["workers"] == {}
    assert json.loads((tmp_path / "logs/attempt_00001_master_provider_response.json").read_text()) == raw
    with pytest.raises(ValueError):
        list(stream_response(raw))


@pytest.mark.parametrize("provider_status,raw,kind", [
    (200, b"not-json", "invalid_provider_response"),
    (200, b"\xff", "invalid_provider_response"),
    (201, b"", "invalid_provider_response"),
    (503, b"temporary provider outage", "provider_error"),
])
def test_rejected_protocol_reply_remains_recorded_after_master_success(
    tmp_path, provider_status, raw, kind,
):
    def upstream(handler, state):
        if len(state.requests) == 1:
            http_reply(handler, raw, status=provider_status)
        else:
            http_reply(handler, response(usage={"prompt_tokens": 7, "completion_tokens": 3}))

    with peer(upstream) as (url, state):
        registry = WorkerRegistry(workspace(tmp_path))
        registry.register("worker")
        relay = TeamRelay(ProviderConfig("fake", url, max_attempts=1), registry, "policy", tmp_path / "logs", 30)
        with serving(relay) as server:
            with pytest.raises(urllib.error.HTTPError) as caught:
                post(relay, server, request("worker") | {"stream": True})
            with caught.value as failure:
                assert failure.code == 424 and failure.headers["x-should-retry"] == "false"
                encoded = failure.read().decode()
            assert "[DONE]" not in encoded and json.loads(encoded)["error"]["code"] == kind
            with post(relay, server, request()) as completed:
                assert completed.status == 200 and json.loads(completed.read())["choices"]
        assert len(state.requests) == 2  # One failed worker call, one master call; no retries.

    summary = relay.summary()
    assert summary["provider_attempts"] == summary["completed_attempts"] == 2
    assert summary["inflight_attempts"] == 0
    assert len(summary["errors"]) == 1 and summary["errors"][0]["error_kind"] == kind
    assert not summary["usage_complete"] and not summary["cost_complete"]
    assert summary["usage"]["total_tokens"] == 10
    assert provider_incident_summary(summary)["count"] == (1 if kind == "provider_error" else 0)
    native = {"explicit_error": False, "missing_result": False}
    for timed_out in (False, True):
        expected = "task_timeout" if timed_out else "native_cli_completed"
        execution = {"returncode": 124 if timed_out else 0, "timed_out": timed_out}
        assert runtime_stop_reason(execution, native, None, summary) == expected


@pytest.mark.parametrize("provider_status", [200, 401, 429, 503])
def test_stream_client_gets_structured_terminal_error_not_empty_sse(tmp_path, provider_status):
    usage = {"prompt_tokens": 7, "completion_tokens": 3}
    raw = {"error": {"message": "timeout / HTTP 429 / HTTP 503 fake-secret", "code": 503},
           "usage": usage}
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", api_key="fake-secret", max_attempts=1,
                                    rates=TokenRates("1", "0.1", "2")),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 30,
                      lambda _: (provider_status, raw))
    with serving(relay) as server:
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(relay, server, request() | {"stream": True})
        with caught.value as failure:
            assert failure.code == 424
            assert failure.headers["Content-Type"] == "application/json"
            assert failure.headers["x-should-retry"] == "false"
            encoded = failure.read().decode()
        assert "[DONE]" not in encoded and "fake-secret" not in encoded
        error = json.loads(encoded)["error"]
        assert error["code"] == "provider_error"  # Never reuse a numeric provider retry code.
        assert error["provider_http_status"] == provider_status
    summary = relay.summary()
    assert summary["cost_complete"] and summary["known_estimated_cost"] == "0.000013"
    assert summary["usage"] == {"prompt_tokens": 7, "completion_tokens": 3,
                                "cached_prompt_tokens": 0, "total_tokens": 10}
    assert summary["errors"][0]["raw_usage"] == usage
    assert "fake-secret" not in "".join(p.read_text() for p in (tmp_path / "logs").iterdir())


def test_valid_completion_still_streams_content_and_usage(tmp_path):
    raw = response([call("send_message", {"recipient": "worker", "content": "Ready"})],
                   {"prompt_tokens": 3, "completion_tokens": 2})
    original = copy.deepcopy(raw)
    relay = TeamRelay(ProviderConfig("fake", "http://localhost"),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 30,
                      lambda _: (200, raw))
    with serving(relay) as server, post(relay, server, request() | {"stream": True}) as result:
        assert result.status == 200 and result.headers["Content-Type"] == "text/event-stream"
        data = result.read().decode()
        assert '"send_message"' in data and '"prompt_tokens": 3' in data and "data: [DONE]" in data
    assert raw == original and relay.summary()["errors"] == []


@pytest.mark.parametrize("stream", [False, True])
def test_tool_rejection_is_delivered_as_native_feedback_not_an_http_retry(tmp_path, stream):
    raw = response([call("send_message", {"recipient": "worker", "content": "Ready"}),
                    call("run_shell_command", {"command": "echo forbidden for master"})],
                   {"prompt_tokens": 7, "completion_tokens": 3})
    original, requests = copy.deepcopy(raw), []

    def upstream(payload):
        requests.append(payload)
        return 200, raw

    registry = WorkerRegistry(workspace(tmp_path))
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", rates=TokenRates("1", "0.1", "2")),
                      registry, "policy", tmp_path / "logs", 30, upstream)
    with serving(relay) as server, post(relay, server, request() | {"stream": stream}) as result:
        assert result.status == 200
        encoded = result.read().decode()
        if stream:
            assert result.headers["Content-Type"] == "text/event-stream"
            assert encoded.endswith("data: [DONE]\n\n")
            chunks = [json.loads(line.removeprefix("data: ")) for line in encoded.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            delivered = copy.deepcopy(original)
            delta = chunks[0]["choices"][0]["delta"]
            for tool_call in delta["tool_calls"]:
                tool_call.pop("index")
            delivered["choices"][0]["message"] = delta
            delivered["usage"] = chunks[-1]["usage"]
        else:
            assert result.headers["Content-Type"] == "application/json"
            delivered = json.loads(encoded)
        assert len(assert_tool_rejection(original, delivered)) == 2
    assert len(requests) == 1 and raw == original and registry.snapshot() == {}
    summary = relay.summary()
    assert summary["completed_attempts"] == summary["provider_attempts"] == 1
    assert summary["usage"]["total_tokens"] == 10
    assert summary["cost_complete"] and summary["known_estimated_cost"] == "0.000013"
    assert summary["errors"][0]["error_kind"] == "tool_response_rejected"
    assert summary["errors"][0]["provider_http_status"] == summary["errors"][0]["relay_http_status"] == 200


def test_rejected_tool_feedback_and_saved_evidence_redact_secrets_without_mutating_raw_response(tmp_path):
    secret = "fake-secret"
    raw = response([call("agent", {"name": secret, "prompt": "Do work " + secret,
                                    "working_dir": "/wrong/" + secret})],
                   {"prompt_tokens": 7, "completion_tokens": 3})
    original = copy.deepcopy(raw)
    registry = WorkerRegistry(workspace(tmp_path))
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", api_key=secret,
                                    rates=TokenRates("1", "0.1", "2")),
                      registry, "policy", tmp_path / "logs", 30, lambda _: (200, raw))
    status, delivered = relay.handle(request())
    assert status == 200 and raw == original and registry.snapshot() == {}
    redacted_original = json.loads(json.dumps(original).replace(secret, "[REDACTED]"))
    feedback = assert_tool_rejection(redacted_original, delivered)[0]
    assert "Omit working_dir" in feedback and "[REDACTED]" in feedback
    assert secret not in json.dumps(delivered) and secret not in json.dumps(relay.summary())
    assert secret not in "".join(path.read_text() for path in (tmp_path / "logs").iterdir())
    assert json.loads((tmp_path / "logs/attempt_00001_master_provider_response.json").read_text()) == redacted_original
    assert json.loads((tmp_path / "logs/attempt_00001_master_qwen_response.json").read_text()) == delivered
    assert relay.summary()["usage"]["total_tokens"] == 10


def test_inflight_owner_rejected_without_blocking_other_workers_or_hiding_usage(tmp_path):
    registry = WorkerRegistry(workspace(tmp_path))
    registry.register("builder")
    first_started, release = threading.Event(), threading.Event()
    calls, result = [], []

    def upstream(payload):
        calls.append(payload)
        if len(calls) == 1:
            first_started.set()
            assert release.wait(3)
        return 200, response(usage={"prompt_tokens": 7, "completion_tokens": 3})

    relay = TeamRelay(ProviderConfig("fake", "http://localhost", rates=TokenRates("1", "0.1", "2")),
                      registry, "policy", tmp_path / "logs", 30, upstream)
    worker = threading.Thread(target=lambda: result.append(relay.handle(request())))
    worker.start()
    try:
        assert first_started.wait(2)
        saved = json.loads((tmp_path / "logs/usage.json").read_text())
        assert saved["provider_attempts"] == 1 and saved["inflight_attempts"] == 1
        with pytest.raises(ValueError, match="still in flight"):
            relay.handle(request())
        assert len(calls) == 1
        assert relay.handle(request("builder"))[0] == 200  # No global worker barrier.
        relay.close()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert result[0][0] == 424 and result[0][1]["error"]["code"] == "request_cancelled"
    summary = relay.summary()
    assert summary["provider_attempts"] == summary["completed_attempts"] == 2
    assert summary["inflight_attempts"] == 0 and summary["cost_complete"]
    assert summary["usage"]["total_tokens"] == 20  # Retain a late response's known usage.


def test_client_disconnect_cancels_upstream_before_it_finishes(tmp_path):
    started, finished = threading.Event(), threading.Event()

    class CancellableFake(OpenAIChatTransport):
        def __call__(self, payload, *, control=None, observation=None):
            started.set()
            try:
                while True:
                    control.check()
                    time.sleep(0.01)
            finally:
                finished.set()

    config = ProviderConfig("fake", "http://localhost", request_timeout_seconds=10)
    relay = TeamRelay(config, WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 30,
                      CancellableFake(config))
    with serving(relay) as server:
        # A read timeout alone need not close urllib's socket (tracebacks can
        # retain its file wrapper). Explicitly close the client like fetch abort.
        body = json.dumps(request() | {"stream": True}).encode()
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=0.2) as client:
            client.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                            f"Authorization: Bearer {relay.token}\r\nContent-Length: {len(body)}\r\n"
                            "Content-Type: application/json\r\n\r\n").encode() + body)
            with pytest.raises(TimeoutError):
                client.recv(1)
        assert started.is_set() and finished.wait(1)
        relay.drain(1)
        summary = relay.summary()
        assert summary["provider_attempts"] == summary["completed_attempts"] == 1
        assert summary["inflight_attempts"] == 0
        assert summary["errors"][0]["error_kind"] == "request_cancelled"
        assert not summary["usage_complete"] and not summary["cost_complete"]
        assert "upstream_cancel_requested" in (tmp_path / "logs/events.jsonl").read_text()


@pytest.mark.parametrize("phase", [
    "qwen_request", "provider_request", "started_event", "started_summary",
    "provider_response", "qwen_response", "finished_event", "finished_summary",
])
@pytest.mark.parametrize("provider_status", [200, 503])
def test_evidence_write_failure_releases_attempt_and_preserves_known_usage(
        tmp_path, monkeypatch, phase, provider_status):
    usage = {"prompt_tokens": 7, "completion_tokens": 3}
    raw = (response(usage=usage) if provider_status == 200
           else {"error": {"message": "Provider failed"}, "usage": usage})
    calls = []

    def upstream(payload):
        calls.append(payload)
        return provider_status, raw

    relay = TeamRelay(ProviderConfig("fake", "http://localhost", api_key="fake-secret", max_attempts=1,
                                    rates=TokenRates("1", "0.1", "2")),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 30, upstream)
    original_save, original_event = relay.save, relay.event
    original_notify = relay._changed.notify_all
    summary_writes, injected, notifications = 0, False, []

    def maybe_fail(point):
        nonlocal injected
        if point == phase and not injected:
            injected = True
            raise OSError("Injected disk full fake-secret")

    def save(path, data):
        nonlocal summary_writes
        if path.name == "usage.json":
            summary_writes += 1
            maybe_fail("started_summary" if summary_writes == 1 else "finished_summary")
        else:
            for point in ("qwen_request", "provider_request", "provider_response", "qwen_response"):
                if path.name.endswith(f"_{point}.json"):
                    maybe_fail(point)
        original_save(path, data)

    def event(name, **fields):
        maybe_fail({"provider_attempt_started": "started_event",
                    "provider_attempt_finished": "finished_event"}.get(name))
        original_event(name, **fields)

    def notify():
        assert not relay._inflight and not relay._controls
        notifications.append(True)
        original_notify()

    with monkeypatch.context() as patch:
        patch.setattr(relay, "save", save)
        patch.setattr(relay, "event", event)
        patch.setattr(relay._changed, "notify_all", notify)
        status, delivered = relay.handle(request())

    assert injected and notifications == [True]
    assert status == 424 and delivered["error"]["code"] == "evidence_write_failed"
    assert "fake-secret" not in json.dumps(delivered)
    before_upstream = phase in {"qwen_request", "provider_request", "started_event", "started_summary"}
    expected_status = None if before_upstream else provider_status
    assert len(calls) == (0 if before_upstream else 1)
    assert delivered["error"]["provider_http_status"] == expected_status
    summary = relay.summary()
    assert summary["provider_attempts"] == summary["completed_attempts"] == 1
    assert summary["inflight_attempts"] == 0 and not relay._controls
    record = summary["errors"][0]
    assert record["error_kind"] == "evidence_write_failed" and record["relay_http_status"] == 424
    assert record["provider_http_status"] == expected_status
    assert record["upstream_started"] is (not before_upstream)
    assert record["raw_usage"] == (None if before_upstream else usage)
    assert summary["cost_complete"] is (not before_upstream)
    assert summary["known_estimated_cost"] == ("0" if before_upstream else "0.000013")
    assert summary["usage"] == ({} if before_upstream else {
        "prompt_tokens": 7, "completion_tokens": 3, "cached_prompt_tokens": 0, "total_tokens": 10})
    # The same role can make a new request after the evidence failure is removed.
    assert relay.handle(request())[0] == (200 if provider_status == 200 else 424)
    assert len(calls) == (1 if before_upstream else 2)
    assert relay.summary()["completed_attempts"] == 2


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0, -1, True, "600"])
def test_nonfinite_provider_deadline_rejected(value):
    with pytest.raises(ValueError, match="finite positive"):
        ProviderConfig("fake", "http://localhost", request_timeout_seconds=value)


@pytest.mark.parametrize("task_expired", [False, True])
def test_request_timeout_is_distinct_from_the_shared_task_deadline(tmp_path, task_expired):
    def upstream(payload):
        if task_expired:
            relay.deadline = time.monotonic() - 1
        raise RequestDeadlineExceeded("Scripted deadline")

    relay = TeamRelay(ProviderConfig("fake", "http://localhost"),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 30,
                      upstream)
    status, delivered = relay.handle(request())
    expected = "task_deadline_exceeded" if task_expired else "provider_timeout"
    assert status == 424 and delivered["error"]["code"] == expected
    record = relay.summary()["errors"][0]
    assert record["error_kind"] == expected and record["upstream_started"]
    assert record["usage"] is None
