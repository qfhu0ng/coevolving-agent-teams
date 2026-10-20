"""Real local HTTP sockets and scripted peers; no provider or Docker calls."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import ssl
import threading
import time
from types import SimpleNamespace

import aiohttp
import pytest

from agent_company.runtime.transport import (
    InvalidProviderResponse, OpenAIChatTransport, RequestCancelled, RequestControl,
    RequestDeadlineExceeded, TransportError,
)


@contextmanager
def peer(behavior):
    state = SimpleNamespace(requests=[], started=threading.Event(), disconnected=threading.Event(),
                            stop=threading.Event())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(raw)})
            state.started.set()
            try:
                behavior(self, state)
            except (BrokenPipeError, ConnectionResetError):
                state.disconnected.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        state.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def config(url, seconds=2, key="scripted-secret"):
    return SimpleNamespace(base_url=url, request_timeout_seconds=seconds, api_key=key)


def reply(handler, body, status=200, headers=None):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    for name, value in (headers or {}).items():
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(data)
    handler.wfile.flush()


def wait_for_client_close(handler, state):
    handler.connection.settimeout(1.5)
    if handler.connection.recv(1) == b"":
        state.disconnected.set()


def test_normal_json_request_and_response_keep_status_usage_and_secure_tls_defaults():
    body = {"choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    with peer(lambda h, _: reply(h, body)) as (url, state):
        transport = OpenAIChatTransport(config(url + "/v1"))
        assert transport({"model": "scripted", "messages": []}) == (200, body)
        assert state.requests[0]["path"] == "/v1/chat/completions"
        assert state.requests[0]["headers"]["Authorization"] == "Bearer scripted-secret"
        assert state.requests[0]["headers"]["User-Agent"] == "AgentCompany/Stage1"
        assert state.requests[0]["headers"]["Accept"] == "application/json"
        assert state.requests[0]["body"] == {"model": "scripted", "messages": []}
        assert transport._ssl_context.check_hostname is True
        assert transport._ssl_context.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("provider_seconds,control_seconds", [(0.12, 3), (3, 0.12)])
def test_deadline_covers_slow_headers_and_uses_shorter_control_or_provider_timeout(
    provider_seconds, control_seconds,
):
    with peer(wait_for_client_close) as (url, state):
        started = time.monotonic()
        with pytest.raises(RequestDeadlineExceeded):
            OpenAIChatTransport(config(url, provider_seconds))({}, control=RequestControl(control_seconds))
        assert time.monotonic() - started < 0.8
        assert state.disconnected.wait(0.5)
        assert len(state.requests) == 1  # The transport never retries.


@pytest.mark.parametrize("status", [200, 503])
def test_continuous_whitespace_in_normal_or_error_body_cannot_extend_total_deadline(status):
    def trickle(handler, state):
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        while not state.stop.wait(0.02):
            handler.wfile.write(b" \n")
            handler.wfile.flush()

    with peer(trickle) as (url, state):
        started = time.monotonic()
        with pytest.raises(RequestDeadlineExceeded):
            OpenAIChatTransport(config(url, 0.14))({})
        assert time.monotonic() - started < 0.8
        assert state.disconnected.wait(0.5)
        assert len(state.requests) == 1


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_explicit_cancel_closes_connection_and_is_not_reported_as_deadline(phase):
    ready = threading.Event()

    def blocked(handler, state):
        if phase == "body":
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(b" ")
            handler.wfile.flush()
        ready.set()
        wait_for_client_close(handler, state)

    with peer(blocked) as (url, state), ThreadPoolExecutor(max_workers=1) as executor:
        control = RequestControl(10)
        future = executor.submit(OpenAIChatTransport(config(url, 10)), {}, control=control)
        assert ready.wait(1)
        started = time.monotonic()
        control.cancel("Caller disconnected")
        control.cancel("A later cancellation must not replace the first reason")
        with pytest.raises(RequestCancelled, match="Caller disconnected"):
            future.result(timeout=0.8)
        assert time.monotonic() - started < 0.8
        assert state.disconnected.wait(0.5)
        assert len(state.requests) == 1


def test_cancellation_during_connect_finishes_http_coroutine(monkeypatch):
    entered, finished = threading.Event(), threading.Event()

    async def blocked_connect(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    monkeypatch.setattr(aiohttp.TCPConnector, "_create_connection", blocked_connect)
    with ThreadPoolExecutor(max_workers=1) as executor:
        control = RequestControl(10)
        future = executor.submit(OpenAIChatTransport(config("https://example.invalid", 10)), {}, control=control)
        assert entered.wait(1)
        control.cancel("Cancelled while connecting")
        with pytest.raises(RequestCancelled, match="connecting"):
            future.result(timeout=0.8)
        assert finished.is_set()


def test_total_deadline_during_connect_finishes_http_coroutine(monkeypatch):
    finished = threading.Event()

    async def blocked_connect(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    monkeypatch.setattr(aiohttp.TCPConnector, "_create_connection", blocked_connect)
    started = time.monotonic()
    with pytest.raises(RequestDeadlineExceeded):
        OpenAIChatTransport(config("https://example.invalid", 0.12))({})
    assert time.monotonic() - started < 0.8
    assert finished.is_set()


def test_slow_system_dns_cannot_delay_return_or_resume_an_http_request(monkeypatch):
    entered, release, dns_finished = threading.Event(), threading.Event(), threading.Event()

    with peer(lambda h, _: reply(h, {"ok": True})) as (url, state):
        port = int(url.rsplit(":", 1)[1])

        def slow_system_dns():
            entered.set()
            try:
                release.wait(2)
                return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]
            finally:
                dns_finished.set()

        async def getaddrinfo(loop, *_args, **_kwargs):
            return await loop.run_in_executor(None, slow_system_dns)

        monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", getaddrinfo)
        with ThreadPoolExecutor(max_workers=1) as executor:
            control = RequestControl(10)
            future = executor.submit(OpenAIChatTransport(config("https://dns-test.invalid", 10)), {}, control=control)
            try:
                assert entered.wait(1)
                control.cancel("Cancelled during DNS")
                with pytest.raises(RequestCancelled, match="DNS"):
                    future.result(timeout=0.8)
                assert not dns_finished.is_set()
                assert state.requests == []
            finally:
                release.set()
                assert dns_finished.wait(1)
        assert state.requests == []


def test_precancelled_or_expired_control_does_not_send_a_request():
    with peer(lambda h, _: reply(h, {"ok": True})) as (url, state):
        cancelled = RequestControl(2)
        cancelled.cancel("Already stopped")
        with pytest.raises(RequestCancelled, match="Already stopped"):
            OpenAIChatTransport(config(url))({}, control=cancelled)
        expired = RequestControl(0.001)
        threading.Event().wait(0.01)
        with pytest.raises(RequestDeadlineExceeded):
            OpenAIChatTransport(config(url))({}, control=expired)
        assert state.requests == []


def test_precancelled_reason_cannot_expose_provider_key():
    control = RequestControl(2)
    control.cancel("Stop scripted-secret")
    with pytest.raises(RequestCancelled, match=r"Stop \[REDACTED\]"):
        OpenAIChatTransport(config("https://example.invalid"))({}, control=control)


def test_redirect_is_not_followed_and_cannot_forward_bearer_credentials():
    with peer(lambda h, _: reply(h, {"unexpected": True})) as (target_url, target):
        with peer(lambda h, _: reply(h, {"redirect": True}, status=307,
                                    headers={"Location": target_url + "/capture"})) as (url, source):
            assert OpenAIChatTransport(config(url))({}) == (307, {"redirect": True})
            assert len(source.requests) == 1 and target.requests == []


def test_proxy_environment_is_not_inherited(monkeypatch):
    with peer(lambda h, _: reply(h, {"proxy": True})) as (proxy_url, proxy):
        with peer(lambda h, _: reply(h, {"direct": True})) as (url, direct):
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                monkeypatch.setenv(name, proxy_url)
            monkeypatch.setenv("NO_PROXY", "")
            monkeypatch.setenv("no_proxy", "")
            assert OpenAIChatTransport(config(url))({}) == (200, {"direct": True})
            assert len(direct.requests) == 1 and proxy.requests == []


@pytest.mark.parametrize("status", [200, 429, 503])
def test_json_application_errors_keep_original_http_status_without_retry_and_redact_key(status):
    with peer(lambda h, _: reply(h, {"error": {"message": "echo scripted-secret"}}, status=status)) as (url, state):
        assert OpenAIChatTransport(config(url))({}) == (status, {"error": {"message": "echo [REDACTED]"}})
        assert len(state.requests) == 1


def test_non_json_error_is_bounded_and_successful_invalid_json_is_explicit_failure():
    with peer(lambda h, _: reply(h, b"scripted-secret " + b"x" * 40000, status=503)) as (url, _):
        status, body = OpenAIChatTransport(config(url))({})
        assert status == 503 and "scripted-secret" not in json.dumps(body)
        assert len(body["error"]["message"]) <= 32768
    with peer(lambda h, _: reply(h, b"not-json scripted-secret")) as (url, _):
        with pytest.raises(InvalidProviderResponse, match="invalid JSON") as error:
            OpenAIChatTransport(config(url))({})
        assert isinstance(error.value, TransportError)
        assert "scripted-secret" not in str(error.value)


@pytest.mark.parametrize("url", ["http://example.com", "ftp://localhost", "https://user:password@example.com",
                                "https://@example.com", "https://example.com/?secret=x", "https://example.com/#x"])
def test_unsafe_provider_urls_are_rejected_before_connect(url):
    with pytest.raises(ValueError):
        OpenAIChatTransport(config(url))


@pytest.mark.parametrize("seconds", [0, -1, True, False, float("nan"), float("inf"), "1", 10 ** 1000])
def test_invalid_deadlines_are_rejected(seconds):
    with pytest.raises(ValueError):
        RequestControl(seconds)
    with pytest.raises(ValueError):
        OpenAIChatTransport(config("https://example.invalid", seconds))
