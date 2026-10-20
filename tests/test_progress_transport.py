"""Progress timers against localhost peers; no provider calls or deployment."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import aiohttp
import pytest

from agent_company.runtime.transport import (
    InvalidProviderResponse, OpenAIChatTransport, RequestCancelled, RequestControl,
    RequestDeadlineExceeded, RequestProgressTimeout,
)
from test_transport import config, peer, reply, wait_for_client_close
from test_stream_transport import chunk, event, headers


def progress_config(url, *, total=2, first=.18, idle=.12):
    value = config(url, total)
    value.first_progress_timeout_seconds = first
    value.progress_idle_timeout_seconds = idle
    return value


def send(handler, value):
    handler.wfile.write(event(value))
    handler.wfile.flush()


def finish(handler):
    send(handler, chunk({}, "stop"))
    send(handler, "[DONE]")


def assert_safe(observation):
    enums = {"unknown", "sse", "json", "in_flight", "completed", "http_error", "cancelled",
             "total_deadline", "first_progress", "progress_idle", "invalid_response", "transport_error"}
    assert all(isinstance(value, (int, float, bool)) or value in enums for value in observation.values())
    assert "SECRET-PROGRESS" not in json.dumps(observation)
    assert not any(name in observation for name in ("usage", "content", "reasoning_content", "arguments"))


def test_continuous_semantic_progress_can_outlast_first_timer_and_records_all_channels():
    deltas = [{"content": "SECRET-PROGRESS"}, {"reasoning_content": "r"}, {"reasoning": "r"},
              {"refusal": "r"}, {"tool_calls": [{"index": 0, "id": "c", "type": "function",
                  "function": {"name": "submit", "arguments": "{"}}]},
              {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}]

    def serve(h, _):
        headers(h)
        for delta in deltas:
            send(h, chunk(delta))
            time.sleep(.045)
        finish(h)

    observation = {}
    with peer(serve) as (url, state):
        status, body = OpenAIChatTransport(progress_config(url))({"stream": True}, observation=observation)
        assert len(state.requests) == 1
    assert status == 200 and body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert observation["elapsed_seconds"] > .18
    assert observation["response_complete"] and observation["progress_observable"]
    assert observation["response_format"] == "sse" and observation["stop_reason"] == "completed"
    assert observation["sse_events"] == 8 and observation["bytes_received"] > 0
    assert observation["tool_arguments_progress_events"] == 2
    for channel in ("content", "reasoning_content", "reasoning", "refusal", "tool_name"):
        assert observation[channel + "_progress_events"] == 1
    for name in ("headers_received_at", "first_byte_at", "first_progress_at", "last_progress_at"):
        assert observation[name] >= observation["request_started_at"]
    assert_safe(observation)


@pytest.mark.parametrize("phase", ["no_headers", "headers", "partial_line"])
def test_first_progress_timeout_closes_blocked_connection_even_with_partial_bytes(phase):
    def serve(h, state):
        if phase != "no_headers":
            headers(h)
        if phase == "partial_line":
            h.wfile.write(b'data: {"choices": [{"delta": {"content": "SECRET-PROGRESS')
            h.wfile.flush()
        wait_for_client_close(h, state)

    observation = {}
    with peer(serve) as (url, state):
        with pytest.raises(RequestProgressTimeout) as caught:
            OpenAIChatTransport(progress_config(url, first=.12))({}, observation=observation)
        assert caught.value.phase == "first_progress"
        assert state.disconnected.wait(.5)
    assert observation["stop_reason"] == "first_progress" and not observation["response_complete"]
    assert "first_progress_at" not in observation
    if phase == "no_headers":
        assert "provider_http_status" not in observation and "headers_received_at" not in observation
    else:
        assert observation["provider_http_status"] == 200
    if phase == "partial_line":
        assert observation["bytes_received"] > 0 and "first_byte_at" in observation
    assert_safe(observation)


def test_first_progress_timeout_covers_connection_and_cleans_up_coroutine(monkeypatch):
    finished = threading.Event()

    async def block(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    monkeypatch.setattr(aiohttp.TCPConnector, "_create_connection", block)
    observation = {}
    with pytest.raises(RequestProgressTimeout) as caught:
        OpenAIChatTransport(progress_config("https://example.invalid", first=.06))({}, observation=observation)
    assert caught.value.phase == "first_progress" and finished.is_set()
    assert observation["bytes_received"] == 0 and "provider_http_status" not in observation


@pytest.mark.parametrize("progress", [False, True])
@pytest.mark.parametrize("nonprogress", [
    b": heartbeat\n\n", event(chunk({"role": "assistant"})), event(chunk({"content": " \t\n"})),
    event(chunk({"reasoning_content": " "})), event({"choices": [], "usage": {"total_tokens": 5}}),
    event(chunk({"tool_calls": [{"index": 0, "id": "new-id", "type": "function",
                                 "function": {"name": "", "arguments": " "}}]})),
])
def test_heartbeats_metadata_and_whitespace_do_not_start_or_renew_progress(progress, nonprogress):
    def serve(h, state):
        headers(h)
        if progress:
            send(h, chunk({"content": "SECRET-PROGRESS"}))
        while not state.stop.wait(.02):
            h.wfile.write(nonprogress)
            h.wfile.flush()

    observation = {}
    with peer(serve) as (url, state):
        with pytest.raises(RequestProgressTimeout) as caught:
            OpenAIChatTransport(progress_config(url, first=.12, idle=.12))({}, observation=observation)
        assert state.disconnected.wait(.5)
    assert caught.value.phase == ("progress_idle" if progress else "first_progress")
    assert observation["content_progress_events"] == int(progress)
    assert observation["response_complete"] is False
    assert_safe(observation)


def test_idle_timeout_discards_partial_tool_arguments_and_usage():
    def serve(h, state):
        headers(h)
        send(h, chunk({"tool_calls": [{"index": 0, "id": "c", "type": "function",
                "function": {"name": "run_shell_command", "arguments": '{"command":"SECRET-PROGRESS'}}]}))
        send(h, {"choices": [], "usage": {"total_tokens": 17}})
        wait_for_client_close(h, state)

    observation = {}
    with peer(serve) as (url, state):
        with pytest.raises(RequestProgressTimeout) as caught:
            OpenAIChatTransport(progress_config(url))({}, observation=observation)
        assert state.disconnected.wait(.5)
    assert caught.value.phase == "progress_idle"
    assert not hasattr(caught.value, "response") and not hasattr(caught.value, "usage")
    assert observation["tool_name_progress_events"] == observation["tool_arguments_progress_events"] == 1
    assert observation["max_progress_gap_seconds"] >= .1
    assert_safe(observation)


def test_fragmented_utf8_and_sse_event_only_count_after_complete_parse():
    def serve(h, _):
        headers(h)
        payload = ('data: ' + json.dumps(chunk({"content": "界"}), ensure_ascii=False) + '\n\n').encode()
        # Split one UTF-8 codepoint as well as the JSON/SSE delimiters.
        position = payload.index("界".encode()) + 1
        for piece in (payload[:position], payload[position:position + 1], payload[position + 1:-1], payload[-1:]):
            h.wfile.write(piece)
            h.wfile.flush()
            time.sleep(.02)
        finish(h)

    observation = {}
    with peer(serve) as (url, _):
        status, body = OpenAIChatTransport(progress_config(url))({}, observation=observation)
    assert status == 200 and body["choices"][0]["message"]["content"] == "界"
    assert observation["content_progress_events"] == 1 and observation["sse_events"] == 3
    assert observation["first_progress_elapsed_seconds"] > observation["first_byte_elapsed_seconds"] + .03


def test_finish_reason_without_done_still_needs_complete_stream_before_idle_deadline():
    def serve(h, state):
        headers(h)
        send(h, chunk({"content": "done"}, "stop"))
        wait_for_client_close(h, state)

    observation = {}
    with peer(serve) as (url, _):
        with pytest.raises(RequestProgressTimeout) as caught:
            OpenAIChatTransport(progress_config(url))({}, observation=observation)
    assert caught.value.phase == "progress_idle" and not observation["response_complete"]


@pytest.mark.parametrize("shorter_control", [False, True])
def test_total_deadline_still_stops_continuous_progress(shorter_control):
    def serve(h, state):
        headers(h)
        while not state.stop.wait(.025):
            send(h, chunk({"reasoning_content": "SECRET-PROGRESS"}))

    observation = {}
    with peer(serve) as (url, state):
        control = RequestControl(.17 if shorter_control else 2)
        with pytest.raises(RequestDeadlineExceeded):
            OpenAIChatTransport(progress_config(url, total=2 if shorter_control else .17))(
                {}, control=control, observation=observation)
        assert state.disconnected.wait(.5)
    assert observation["stop_reason"] == "total_deadline"
    assert observation["reasoning_content_progress_events"] > 2
    assert observation["response_complete"] is False


def test_equal_first_and_total_deadlines_report_total():
    observation = {}
    with peer(wait_for_client_close) as (url, _):
        with pytest.raises(RequestDeadlineExceeded):
            OpenAIChatTransport(progress_config(url, total=.1, first=.1))({}, observation=observation)
    assert observation["stop_reason"] == "total_deadline"


def test_cancellation_wins_and_closes_socket_with_progress_timers_enabled():
    ready = threading.Event()

    def serve(h, state):
        headers(h)
        send(h, chunk({"content": "SECRET-PROGRESS"}))
        ready.set()
        wait_for_client_close(h, state)

    observation = {}
    with peer(serve) as (url, state), ThreadPoolExecutor(max_workers=1) as pool:
        control = RequestControl(2)
        future = pool.submit(OpenAIChatTransport(progress_config(url, first=1, idle=1)), {},
                             control=control, observation=observation)
        assert ready.wait(1)
        control.cancel("user cancelled")
        with pytest.raises(RequestCancelled, match="user cancelled"):
            future.result(timeout=.8)
        assert state.disconnected.wait(.5)
    assert observation["stop_reason"] == "cancelled"


def test_json_fallback_is_unobservable_and_cannot_renew_first_timer():
    def serve(h, state):
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.end_headers()
        h.wfile.write(b'{"choices":[{"message":{"content":"')
        h.wfile.flush()
        while not state.stop.wait(.025):
            h.wfile.write(b"SECRET-PROGRESS")
            h.wfile.flush()

    observation = {}
    with peer(serve) as (url, _):
        with pytest.raises(RequestProgressTimeout) as caught:
            OpenAIChatTransport(progress_config(url, first=.12))({}, observation=observation)
    assert caught.value.phase == "first_progress"
    assert observation["response_format"] == "json" and not observation["progress_observable"]
    assert observation["bytes_received"] > 0 and observation["sse_events"] == 0
    assert "first_progress_at" not in observation


@pytest.mark.parametrize("status", [200, 401, 403, 503])
def test_complete_json_or_http_error_preserves_status_and_body(status):
    body = {"choices": [{"message": {"content": "OK"}}]} if status == 200 else {"error": {"message": "bad"}}
    observation = {}
    with peer(lambda h, _: reply(h, body, status=status)) as (url, _):
        assert OpenAIChatTransport(progress_config(url))({}, observation=observation) == (status, body)
    assert observation["provider_http_status"] == status and observation["response_complete"]
    assert observation["stop_reason"] == ("completed" if status == 200 else "http_error")
    assert not observation["progress_observable"] and "first_progress_at" not in observation


def test_http_error_headers_then_blocked_body_retains_status_on_progress_timeout():
    def serve(h, state):
        h.send_response(503)
        h.send_header("Content-Type", "application/json")
        h.end_headers()
        wait_for_client_close(h, state)

    observation = {}
    with peer(serve) as (url, state):
        with pytest.raises(RequestProgressTimeout):
            OpenAIChatTransport(progress_config(url, first=.1))({}, observation=observation)
        assert state.disconnected.wait(.5)
    assert observation["provider_http_status"] == 503 and observation["stop_reason"] == "first_progress"


@pytest.mark.parametrize("first,idle", [(None, None), (.1, None), (None, .1)])
def test_independent_optional_timers_and_legacy_behavior(first, idle):
    def serve(h, _):
        headers(h)
        if first is None:
            time.sleep(.15)  # No first timer: quiet before the first delta is allowed.
        send(h, chunk({"content": "one"}))
        if idle is None:
            time.sleep(.15)  # No idle timer: quiet after progress is allowed.
        finish(h)

    observation = {}
    with peer(serve) as (url, _):
        status, _ = OpenAIChatTransport(progress_config(url, first=first, idle=idle))({}, observation=observation)
    assert status == 200 and observation["stop_reason"] == "completed"


@pytest.mark.parametrize("rows", [[chunk({"content": "partial"})],
                                   [chunk({"content": "partial"}), "[DONE]"], ["invalid JSON"]])
def test_invalid_or_incomplete_sse_stays_invalid_response_not_progress_timeout(rows):
    def serve(h, _):
        headers(h)
        for row in rows:
            send(h, row)

    observation = {}
    with peer(serve) as (url, _):
        with pytest.raises(InvalidProviderResponse):
            OpenAIChatTransport(progress_config(url))({}, observation=observation)
    assert observation["stop_reason"] == "invalid_response" and not observation["response_complete"]


def test_concurrent_calls_do_not_share_progress_or_observation_state():
    def serve(h, state):
        headers(h)
        if state.requests[-1]["body"].get("stalled"):
            wait_for_client_close(h, state)
        else:
            send(h, chunk({"content": "SECRET-PROGRESS"}))
            finish(h)

    stalled, complete = {}, {}
    with peer(serve) as (url, _), ThreadPoolExecutor(max_workers=2) as pool:
        transport = OpenAIChatTransport(progress_config(url))
        bad = pool.submit(transport, {"stalled": True}, observation=stalled)
        time.sleep(.03)
        good = pool.submit(transport, {"stalled": False}, observation=complete)
        assert good.result(timeout=1)[0] == 200
        with pytest.raises(RequestProgressTimeout):
            bad.result(timeout=1)
    assert stalled["content_progress_events"] == 0 and complete["content_progress_events"] == 1
    assert stalled["stop_reason"] == "first_progress" and complete["stop_reason"] == "completed"


@pytest.mark.parametrize("field", ["first_progress_timeout_seconds", "progress_idle_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_progress_timeout_rejected_without_network(field, value):
    provider = progress_config("https://example.invalid")
    setattr(provider, field, value)
    with pytest.raises(ValueError):
        OpenAIChatTransport(provider)({})
