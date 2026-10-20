"""Progress recovery through real local HTTP/SSE sockets, never a paid provider.

The native tests also use the installed Qwen CLI: only the HTTP peer is scripted.
Partial generations must be discarded before retrying, not executed or spliced.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, TokenRates, role_of
from agent_company.runtime.transport import RequestControl
from test_rejection_integration import native_qwen, reply as native_reply
from test_relay_recovery import make_relay
from test_stream_transport import chunk, event, headers
from test_team import call, request, response, workspace
from test_transport import peer, wait_for_client_close


def emit(handler, value):
    handler.wfile.write(event(value))
    handler.wfile.flush()


def complete(handler, result, *, headers_sent=False):
    """Send one complete deterministic OpenAI SSE response including usage."""
    if not headers_sent:
        headers(handler)
    for choice in result["choices"]:
        delta = copy.deepcopy(choice["message"])
        delta.pop("role", None)
        if delta.get("tool_calls"):
            delta["tool_calls"] = [{"index": index, **tool}
                                   for index, tool in enumerate(delta["tool_calls"])]
        emit(handler, chunk(delta, choice.get("finish_reason") or (
            "tool_calls" if delta.get("tool_calls") else "stop")))
    if result.get("usage") is not None:
        emit(handler, {"choices": [], "usage": result["usage"]})
    emit(handler, "[DONE]")


def block_generation(handler, state, phase):
    headers(handler)
    if phase == "progress_idle":
        # Even a syntactically complete argument object is NOT a complete
        # response: there is no finish_reason or [DONE]. It must never execute.
        emit(handler, chunk({"tool_calls": [{"index": 0, "id": "discarded-call",
            "type": "function", "function": {"name": "write_file", "arguments":
                '{"file_path":"discarded.txt","content":"PARTIAL_MUST_NOT_SURVIVE"}'}}]}))
    else:
        # A role delta and SSE heartbeat are not meaningful model progress.
        emit(handler, chunk({"role": "assistant"}))
        handler.wfile.write(b": heartbeat\n\n")
        handler.wfile.flush()
    wait_for_client_close(handler, state)


@pytest.mark.parametrize("phase", ["first_progress", "progress_idle"])
def test_progress_timeout_retries_after_socket_cleanup_without_merging_partials(tmp_path, phase):
    first_closed_before_retry = []
    good = call("write_file", {"file_path": "accepted.txt", "content": "complete"})

    def serve(handler, state):
        if len(state.requests) == 1:
            block_generation(handler, state, phase)
            return
        first_closed_before_retry.append(state.disconnected.wait(0.5))
        complete(handler, response([good], usage={"prompt_tokens": 7, "completion_tokens": 3}))

    with peer(serve) as (url, state):
        relay = make_relay(tmp_path, mode="single", base_url=url, stream_response=True,
                           request_timeout_seconds=2, first_progress_timeout_seconds=0.15,
                           progress_idle_timeout_seconds=0.15, retry_backoff_seconds=0)
        status, delivered = relay.handle(request("master", ("write_file",)))
        assert status == 200 and len(state.requests) == 2
        assert state.requests[0]["body"] == state.requests[1]["body"]
        assert first_closed_before_retry == [True]
        assert delivered["choices"][0]["message"]["tool_calls"] == [good]
        assert "PARTIAL_MUST_NOT_SURVIVE" not in json.dumps(delivered)
        summary = relay.summary()
        assert summary["logical_model_calls"] == 1
        assert summary["provider_attempts"] == summary["completed_attempts"] == 2
        assert summary["active_requests"] == summary["inflight_attempts"] == 0
        assert [row["retry_index"] for row in relay._records] == [0, 1]
        assert [row["error_kind"] for row in summary["errors"]] == [f"provider_{phase}_timeout"]
        # A later known usage record cannot erase the failed attempt's unknown usage.
        assert not summary["usage_complete"] and not summary["cost_complete"]
        assert summary["usage"]["total_tokens"] == 10
        assert summary["errors"][0]["usage"] is None
        transport = summary["errors"][0]["transport"]
        assert isinstance(transport, dict) and transport
        assert transport["stop_reason"] == phase and transport["provider_http_status"] == 200
        assert not transport["response_complete"] and transport["response_format"] == "sse"
        assert ("first_progress_at" in transport) == (phase == "progress_idle")
        assert transport["tool_arguments_progress_events"] == int(phase == "progress_idle")
        assert all(isinstance(value, (int, float, bool)) if key not in {"stop_reason", "response_format"}
                   else isinstance(value, str) for key, value in transport.items())
        assert "PARTIAL_MUST_NOT_SURVIVE" not in json.dumps(transport)
        assert relay._records[1]["transport"]["response_complete"]
        assert not relay._active_requests and not relay._controls


def test_repeated_progress_timeouts_stop_at_max_attempts(tmp_path):
    with peer(lambda h, s: block_generation(h, s, "first_progress")) as (url, state):
        relay = make_relay(tmp_path, base_url=url, stream_response=True,
                           request_timeout_seconds=3, first_progress_timeout_seconds=0.12,
                           progress_idle_timeout_seconds=0.12, max_attempts=3, retry_backoff_seconds=0)
        status, delivered = relay.handle(request())
        assert status == 424 and delivered["error"]["code"] == "provider_first_progress_timeout"
        assert len(state.requests) == 3
        summary = relay.summary()
        assert summary["logical_model_calls"] == 1 and summary["provider_attempts"] == 3
        assert len(summary["errors"]) == 3 and not summary["usage_complete"]
        assert summary["active_requests"] == summary["inflight_attempts"] == 0


def test_total_deadline_is_not_renewed_or_retried_as_progress_timeout(tmp_path):
    with peer(lambda h, s: block_generation(h, s, "first_progress")) as (url, state):
        relay = make_relay(tmp_path, base_url=url, stream_response=True,
                           request_timeout_seconds=0.18, first_progress_timeout_seconds=1,
                           progress_idle_timeout_seconds=1, max_attempts=4, retry_backoff_seconds=0)
        started = time.monotonic()
        status, delivered = relay.handle(request())
        assert status == 424 and delivered["error"]["code"] == "provider_timeout"
        assert time.monotonic() - started < 0.9
        assert len(state.requests) == 1 and state.disconnected.wait(0.5)
        assert relay.summary()["errors"][0]["error_kind"] == "provider_timeout"


@pytest.mark.parametrize("http_status", [400, 401, 403])
def test_nonretryable_http_status_with_stalled_body_is_not_retried(tmp_path, http_status):
    def serve(handler, state):
        handler.send_response(http_status)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        wait_for_client_close(handler, state)

    with peer(serve) as (url, state):
        relay = make_relay(tmp_path, base_url=url, stream_response=True,
                           request_timeout_seconds=2, first_progress_timeout_seconds=0.15,
                           progress_idle_timeout_seconds=0.15, max_attempts=4, retry_backoff_seconds=0)
        status, delivered = relay.handle(request())
        assert status == 424
        assert delivered["error"]["code"] == "provider_first_progress_timeout"
        assert len(state.requests) == 1
        record = relay.summary()["errors"][0]
        assert record["transport"]["provider_http_status"] == http_status
        assert record["provider_http_status"] == http_status


def truncated_error_body(handler, http_status):
    """Known HTTP headers followed by an incomplete body and a real disconnect."""
    handler.send_response(http_status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", "100")
    handler.end_headers()
    handler.wfile.write(b'{"error":')
    handler.wfile.flush()
    handler.close_connection = True


@pytest.mark.parametrize("http_status", [400, 401, 403])
def test_nonretryable_http_status_with_truncated_body_is_not_retried(tmp_path, http_status):
    with peer(lambda h, _: truncated_error_body(h, http_status)) as (url, state):
        relay = make_relay(tmp_path, base_url=url, stream_response=True,
                           request_timeout_seconds=2, first_progress_timeout_seconds=1,
                           progress_idle_timeout_seconds=1, max_attempts=3, retry_backoff_seconds=0)
        status, delivered = relay.handle(request())
        assert status == 424 and delivered["error"]["code"] == "provider_transport_error"
        assert len(state.requests) == 1
        summary = relay.summary()
        assert summary["provider_attempts"] == summary["logical_model_calls"] == 1
        record = summary["errors"][0]
        assert record["provider_http_status"] == http_status
        assert record["transport"]["provider_http_status"] == http_status
        assert not record["transport"]["response_complete"]


def test_retryable_http_status_with_truncated_body_can_recover(tmp_path):
    def serve(handler, state):
        if len(state.requests) == 1:
            truncated_error_body(handler, 503)
        else:
            complete(handler, response(usage={"prompt_tokens": 7, "completion_tokens": 3}))

    with peer(serve) as (url, state):
        relay = make_relay(tmp_path, base_url=url, stream_response=True,
                           request_timeout_seconds=2, first_progress_timeout_seconds=1,
                           progress_idle_timeout_seconds=1, max_attempts=3, retry_backoff_seconds=0)
        status, _ = relay.handle(request())
        assert status == 200 and len(state.requests) == 2
        summary = relay.summary()
        assert summary["logical_model_calls"] == 1 and summary["provider_attempts"] == 2
        assert len(summary["errors"]) == 1
        record = summary["errors"][0]
        assert record["error_kind"] == "provider_transport_error" and record["provider_http_status"] == 503
        assert not summary["usage_complete"] and summary["usage"]["total_tokens"] == 10


def test_progress_retries_share_one_total_deadline(tmp_path):
    with peer(lambda h, s: block_generation(h, s, "first_progress")) as (url, state):
        relay = make_relay(tmp_path, base_url=url, stream_response=True,
                           request_timeout_seconds=0.42, first_progress_timeout_seconds=0.16,
                           progress_idle_timeout_seconds=0.16, max_attempts=10, retry_backoff_seconds=0)
        started = time.monotonic()
        status, delivered = relay.handle(request())
        elapsed = time.monotonic() - started
        assert status == 424 and delivered["error"]["code"] == "provider_timeout"
        assert 0.35 <= elapsed < 1
        assert 2 <= len(state.requests) <= 3  # Final attempt gets only the remaining deadline.
        assert relay.summary()["logical_model_calls"] == 1
        assert relay._records[-1]["error_kind"] == "provider_timeout"


def test_cancellation_during_stream_never_retries(tmp_path):
    with peer(lambda h, s: block_generation(h, s, "progress_idle")) as (url, state):
        relay = make_relay(tmp_path, mode="single", base_url=url, stream_response=True,
                           request_timeout_seconds=4, first_progress_timeout_seconds=2,
                           progress_idle_timeout_seconds=2, retry_backoff_seconds=0)
        control = RequestControl(4)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(relay.handle, request("master", ("write_file",)), control=control)
            assert state.started.wait(1)
            control.cancel("user requested pause")
            status, delivered = future.result(timeout=1)
        assert status == 424 and delivered["error"]["code"] == "request_cancelled"
        assert len(state.requests) == 1 and state.disconnected.wait(0.5)
        assert relay.summary()["provider_attempts"] == 1
        assert not relay._active_requests and not relay._controls


def test_progress_retry_backoff_reserves_role_until_cancelled(tmp_path, monkeypatch):
    scheduled = threading.Event()
    with peer(lambda h, s: block_generation(h, s, "first_progress")) as (url, state):
        relay = make_relay(tmp_path, base_url=url, stream_response=True,
                           request_timeout_seconds=4, first_progress_timeout_seconds=0.12,
                           progress_idle_timeout_seconds=0.12, retry_backoff_seconds=2)
        original_event = relay.event

        def record(name, **fields):
            original_event(name, **fields)
            if name == "provider_retry_scheduled":
                scheduled.set()

        monkeypatch.setattr(relay, "event", record)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(relay.handle, request())
            try:
                assert scheduled.wait(2)
                assert relay.summary()["active_requests"] == 1
                assert relay.summary()["inflight_attempts"] == 0
                with pytest.raises(ValueError, match="still in flight"):
                    relay.handle(request())
            finally:
                relay.close()
            status, delivered = future.result(timeout=1)
        assert status == 424 and delivered["error"]["code"] == "request_cancelled"
        assert len(state.requests) == 1 and not relay._active_requests


class NativeProgressScenario:
    """One stalled response, then the same session succeeds without replacing its worker.

    The optional artifact reader lets the same fixture check files inside an
    official ProgramBench container, without treating container paths as host paths.
    """

    def __init__(self, ws, mode, phase="progress_idle", artifact_reader=None):
        assert phase in {"first_progress", "progress_idle"}
        self.mode = mode
        self.phase = phase
        self.path = Path(ws.repo_path) / "executed-once.txt"
        self.artifact_reader = artifact_reader or (
            lambda path: path.read_text() if path.exists() else None)
        self.steps = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.shutdown_seen = threading.Event()
        self.first_closed = threading.Event()
        self.failed_payload = None
        self.errors = []
        self.team_name = "progress-recovery-" + uuid.uuid4().hex[:8]

    def artifact_contents(self):
        return self.artifact_reader(self.path)

    @staticmethod
    def await_native_event(handler, event, timeout):
        """Keep unrelated master generation progressing while its worker recovers."""
        headers(handler)
        deadline = time.monotonic() + timeout
        while not event.is_set():
            assert time.monotonic() < deadline, "Native fixture event did not arrive"
            emit(handler, chunk({"content": "."}))
            event.wait(0.05)

    def serve(self, handler, state):
        try:
            self._serve(handler, state)
        except Exception as exc:
            self.errors.append(repr(exc))
            raise

    def _serve(self, handler, state):
        # The real upstream receives no local routing headers; role is taken
        # from the scripted task prompt, just like the existing native fixtures.
        payload = handler.provider_payload
        role = role_of(payload)
        with self.lock:
            step = self.steps.get(role, 0)
            self.steps[role] = step + 1
            assert sum(self.steps.values()) <= 15, "Native recovery did not converge"
        messages = payload["messages"]
        users = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
        if self.mode == "team" and role == "builder" and "shutdown" in (users[-1] if users else "").lower():
            self.shutdown_seen.set()
            complete(handler, native_reply(payload, [call("send_message", {
                "to": "leader", "message": "shutdown_approved"})])[1])
            return
        if self.mode == "team" and role == "master":
            headers_sent = False
            if step == 0:
                result = native_reply(payload, [call("team_create", {"team_name": self.team_name})])[1]
            elif step == 1:
                result = native_reply(payload, [call("agent", {"name": "builder", "description": "Write once",
                    "prompt": "Write the delivery file once and report completion."})])[1]
            elif step == 2:
                self.await_native_event(handler, self.done, 10)
                headers_sent = True
                result = native_reply(payload, [call("request_shutdown", {"to": "builder"})])[1]
            else:
                self.await_native_event(handler, self.shutdown_seen, 10)
                headers_sent = True
                result = native_reply(payload, content="Delivery complete after transparent request retry.")[1]
            complete(handler, result, headers_sent=headers_sent)
            return
        assert role == ("builder" if self.mode == "team" else "master")
        if step == 0:
            self.failed_payload = copy.deepcopy(payload)
            headers(handler)
            if self.phase == "progress_idle":
                emit(handler, chunk({"tool_calls": [{"index": 0, "id": "discarded-tool", "type": "function",
                    "function": {"name": "run_shell_command", "arguments": json.dumps({
                        "command": f"printf 'BAD\\n' >> {shlex.quote(str(self.path))}",
                        "description": "This partial generation must not run"})}}]}))
            else:
                emit(handler, chunk({"role": "assistant"}))
                handler.wfile.write(b": heartbeat\n\n")
                handler.wfile.flush()
            wait_for_client_close(handler, state)
            self.first_closed.set()
            return
        if step == 1:
            assert self.first_closed.wait(1), "Retry began before the old socket closed"
            assert payload == self.failed_payload, "Retry changed the original native request"
            assert self.artifact_contents() is None, "Discarded partial tool call executed"
            complete(handler, native_reply(payload, [call("run_shell_command", {
                "command": f"printf 'ONCE\\n' >> {shlex.quote(str(self.path))} && printf RECOVERED_NATIVE",
                "description": "Record exactly one successful execution"})])[1])
            return
        assert "RECOVERED_NATIVE" in json.dumps(messages)
        assert "discarded-tool" not in json.dumps(messages)
        self.done.set()
        complete(handler, native_reply(payload, content="WORKER_DELIVERY_COMPLETE")[1])


@contextmanager
def native_peer(scenario):
    """Keep each concurrent worker's request attached to its own HTTP handler."""
    state = SimpleNamespace(requests=[], disconnected=threading.Event())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.provider_payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(self.provider_payload)
            try:
                scenario.serve(self, state)
            except (BrokenPipeError, ConnectionResetError):
                state.disconnected.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("phase", ["first_progress", "progress_idle"])
def test_native_same_session_progress_recovery_executes_only_complete_retry(tmp_path, mode, phase):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    scenario = NativeProgressScenario(ws, mode, phase)
    with native_peer(scenario) as (url, state):
        provider = ProviderConfig("native-local-sse", url, stream_response=True, request_timeout_seconds=15,
            first_progress_timeout_seconds=0.2, progress_idle_timeout_seconds=0.2,
            max_attempts=3, retry_backoff_seconds=0, rates=TokenRates("1", "0.1", "2"))
        result = QwenTeamRunner(str(cli), node=node, team_stall_timeout_seconds=1).run(ws,
            "Write executed-once.txt once and complete the task.",
            "Delegate to one writable teammate." if mode == "team" else "",
            "Report concrete completion and approve shutdown." if mode == "team" else "",
            tmp_path / "native", provider, timeout_seconds=60, execution_mode=mode)
        assert scenario.errors == []
        assert result["execution"]["returncode"] == 0 and not result["execution"]["timed_out"], result
        assert result["stop_reason"] == "native_cli_completed"
        assert scenario.artifact_contents() == "ONCE\n"
        assert not result["scripted_provider"]  # Actual transport, not Python upstream injection.
        assert len(result["errors"]) == 1
        assert result["errors"][0]["error_kind"] == f"provider_{phase}_timeout"
        observation = result["errors"][0]["transport"]
        assert observation["stop_reason"] == phase and observation["provider_http_status"] == 200
        assert not observation["response_complete"]
        assert ("first_progress_at" in observation) == (phase == "progress_idle")
        assert result["provider_attempts"] == result["logical_model_calls"] + 1
        assert not result["usage_complete"] and not result["cost_complete"]
        launch = json.loads((tmp_path / "native/launch.json").read_text())
        assert launch["native_request_timeout_seconds"] == 45
        assert launch["native_team_inactivity_timeout_seconds"] == (105 if mode == "team" else None)
        if mode == "team":
            assert set(result["workers"]) == {"builder"}
            assert scenario.shutdown_seen.is_set()
            lifecycle = result["native_result"]["lifecycle"]["events"]
            assert not any(row["event"] == "failure_notice_queued" for row in lifecycle)
