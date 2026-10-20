"""Cancellable HTTP transport with one wall-clock deadline for the whole request.

Each synchronous call owns an async HTTP task and event loop in its calling
thread. Cancellation closes that task's response/session; it does not leave a
background HTTP request running. A system DNS lookup already dispatched to the
default resolver's executor cannot be forcibly interrupted. It may finish later,
but its cancelled coroutine cannot proceed to connect or send a request.
"""
from __future__ import annotations

import asyncio
import json
import math
import ssl
import threading
import time
from urllib.parse import urlparse

import aiohttp


class RequestCancelled(RuntimeError):
    """The caller explicitly cancelled this request."""


class RequestDeadlineExceeded(TimeoutError):
    """The total request deadline elapsed, regardless of incoming keepalives."""


class RequestProgressTimeout(TimeoutError):
    """An attempt had no semantic progress; the logical request may still have time.

    ``phase`` is ``first_progress`` before the first meaningful parsed SSE delta,
    or ``progress_idle`` after it. No partial completion is carried by this error.
    """

    def __init__(self, phase):
        if phase not in {"first_progress", "progress_idle"}:
            raise ValueError("Unknown progress timeout phase")
        self.phase = phase
        super().__init__(f"Provider {phase} timeout")


class TransportError(RuntimeError):
    """A transport or JSON-protocol failure, with credentials removed."""


class InvalidProviderResponse(TransportError):
    """A complete successful HTTP response could not be decoded as JSON."""


class ChatStream:
    """Reassemble OpenAI SSE deltas; never expose a partial tool call to runtime."""

    def __init__(self):
        self.result = {"object": "chat.completion"}
        self.choices = {}
        self.done = False

    def feed(self, data):
        if data == "[DONE]":
            self.done = True
            return {}
        channels = {}

        def progress(channel, value):
            if isinstance(value, str) and value.strip():
                channels[channel] = 1

        try:
            chunk = json.loads(data)
            if not isinstance(chunk, dict):
                raise ValueError("Expected object")
            if chunk.get("error"):
                raise ValueError("Provider reported an SSE error")
            for key in ("id", "model", "created", "system_fingerprint"):
                if key in chunk:
                    self.result[key] = chunk[key]
            if chunk.get("usage") is not None:
                self.result["usage"] = chunk["usage"]
            for item in chunk.get("choices", []):
                index = item["index"]
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError("Invalid choice index")
                choice = self.choices.setdefault(index, {"index": index,
                    "message": {"role": "assistant", "content": ""}, "finish_reason": None, "tools": {}})
                delta = item.get("delta") or {}
                if choice["finish_reason"] is not None and delta:
                    raise ValueError("Delta after finished choice")
                message = choice["message"]
                for key in ("content", "reasoning_content", "reasoning", "refusal"):
                    if delta.get(key) is not None:
                        if not isinstance(delta[key], str):
                            raise ValueError("Non-string text delta")
                        message[key] = message.get(key, "") + delta[key]
                        progress(key, delta[key])
                for call in delta.get("tool_calls") or []:
                    slot = call["index"]
                    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
                        raise ValueError("Invalid tool index")
                    tool = choice["tools"].setdefault(slot, {"id": "", "type": "function",
                                                           "function": {"name": "", "arguments": ""}})
                    if call.get("id"):
                        tool["id"] += call["id"]
                    if call.get("type") not in (None, "function"):
                        raise ValueError("Unsupported tool type")
                    for key in ("name", "arguments"):
                        value = (call.get("function") or {}).get(key) or ""
                        tool["function"][key] += value
                        progress("tool_" + key, value)
                if item.get("finish_reason") is not None:
                    choice["finish_reason"] = item["finish_reason"]
        except (ValueError, KeyError, TypeError, AttributeError):
            raise InvalidProviderResponse("Invalid provider SSE event") from None
        return channels

    def complete(self):
        if not self.done or not self.choices or any(c["finish_reason"] is None for c in self.choices.values()):
            raise InvalidProviderResponse("Incomplete provider SSE stream")
        choices = []
        for _, choice in sorted(self.choices.items()):
            tools = choice.pop("tools")
            if tools:
                choice["message"]["tool_calls"] = [t for _, t in sorted(tools.items())]
            choices.append(choice)
        return {**self.result, "choices": choices}


async def read_chat_stream(response, control, watch=None):
    stream, parts, pending = ChatStream(), [], b""
    while True:
        raw = await response.content.readany()
        control.check()
        if not raw:
            break
        if watch is not None:
            watch.received(raw)
        pending += raw
        while b"\n" in pending:
            raw_line, pending = pending.split(b"\n", 1)
            try:
                line = raw_line.decode("utf-8").rstrip("\r")
            except UnicodeDecodeError:
                raise InvalidProviderResponse("Invalid UTF-8 in SSE stream") from None
            if not line:
                if parts:
                    channels = stream.feed("\n".join(parts))
                    parts = []
                    if watch is not None:
                        watch.event(channels)
                    if stream.done:
                        return stream.complete()
            elif line.startswith("data:"):
                parts.append(line[5:].lstrip(" "))
    raise InvalidProviderResponse("Provider SSE ended before complete DONE event")


def _positive_seconds(value):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Request timeout must be a finite positive number")
    return value


class _ProgressWatch:
    """Per-attempt timers and content-free observations; never shared by callers."""

    CHANNELS = ("content", "reasoning_content", "reasoning", "refusal", "tool_name", "tool_arguments")

    def __init__(self, config, observation, started):
        self.started = started
        self.wall_started = time.time()
        self.first = getattr(config, "first_progress_timeout_seconds", None)
        self.idle = getattr(config, "progress_idle_timeout_seconds", None)
        for seconds in (self.first, self.idle):
            if seconds is not None:
                _positive_seconds(seconds)
        self.last = None
        self.timer = None
        self.out = {} if observation is None else observation
        self.out.clear()
        self.out.update(request_started_at=self.wall_started, bytes_received=0, sse_events=0,
                        max_progress_gap_seconds=0.0, response_format="unknown",
                        progress_observable=False, response_complete=False, stop_reason="in_flight")
        self.out.update({channel + "_progress_events": 0 for channel in self.CHANNELS})

    def stamp(self, name, elapsed_name=None):
        elapsed = max(0.0, time.monotonic() - self.started)
        self.out[name + "_at"] = self.wall_started + elapsed
        self.out[(elapsed_name or name) + "_elapsed_seconds"] = elapsed

    def received(self, raw):
        self.check()
        if "first_byte_at" not in self.out:
            self.stamp("first_byte")
        self.out["bytes_received"] += len(raw)

    def event(self, channels):
        self.check()
        self.out["sse_events"] += 1
        if not channels:
            return
        now = time.monotonic()
        if self.last is None:
            self.stamp("first_progress")
        else:
            self.out["max_progress_gap_seconds"] = max(
                self.out["max_progress_gap_seconds"], now - self.last)
        self.last = now
        self.stamp("last_progress")
        for channel in channels:
            self.out[channel + "_progress_events"] += 1
        # First-progress and inter-progress limits are independent optional knobs.
        self.timer.reschedule(None if self.idle is None else asyncio.get_running_loop().time() + self.idle)

    def check(self):
        self.control.check()
        now = time.monotonic()
        if now >= self.deadline:
            raise RequestDeadlineExceeded("Total request deadline exceeded")
        seconds, start = (self.first, self.started) if self.last is None else (self.idle, self.last)
        if seconds is not None and now >= start + seconds:
            raise RequestProgressTimeout("first_progress" if self.last is None else "progress_idle")

    def finish(self, reason):
        now = time.monotonic()
        self.out["elapsed_seconds"] = max(0.0, now - self.started)
        if self.last is not None:
            self.out["max_progress_gap_seconds"] = max(
                self.out["max_progress_gap_seconds"], now - self.last)
        self.out["stop_reason"] = reason


class RequestControl:
    """One monotonic deadline plus thread-safe, first-reason-wins cancellation."""

    def __init__(self, timeout_seconds):
        self._deadline = time.monotonic() + _positive_seconds(timeout_seconds)
        self._lock = threading.Lock()
        self._reason = None
        self._callbacks = set()

    @property
    def deadline(self):
        return self._deadline

    def cancel(self, reason="Request cancelled"):
        with self._lock:
            if self._reason is not None:
                return
            self._reason = str(reason)
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            callback()

    def check(self):
        with self._lock:
            reason = self._reason
        if reason is not None:
            raise RequestCancelled(reason)
        if time.monotonic() >= self.deadline:
            raise RequestDeadlineExceeded("Total request deadline exceeded")

    def _subscribe(self, callback):
        with self._lock:
            self._callbacks.add(callback)
            cancelled = self._reason is not None
        if cancelled:
            callback()

        def unsubscribe():
            with self._lock:
                self._callbacks.discard(callback)

        return unsubscribe


class OpenAIChatTransport:
    """POST once, without redirects, inherited proxies, cookies, or HTTP retries.

    ``control`` may impose a shorter task-remaining deadline and can be cancelled
    from another thread. The provider timeout remains an independent upper bound.
    HTTP status and JSON body are returned unchanged except credential redaction;
    the relay remains responsible for recognizing application errors in HTTP 200.
    """

    def __init__(self, config):
        self.config = config
        _positive_seconds(config.request_timeout_seconds)
        parsed = urlparse(config.base_url)
        if (parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Provider base URL must not contain credentials, a query, or a fragment")
        if not parsed.hostname or (parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        )):
            raise ValueError("Provider endpoint must use HTTPS (except local tests)")
        self.url = config.base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self._ssl_context = ssl.create_default_context()

    def _redact(self, value):
        key = self.config.api_key
        if isinstance(value, str):
            return value.replace(key, "[REDACTED]") if key else value
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {self._redact(name): self._redact(item) for name, item in value.items()}
        return value

    def __call__(self, payload, *, control=None, observation=None):
        """Execute one attempt, optionally filling a caller-owned observation dict.

        Observations contain only numeric values, booleans and fixed enums, never
        content, reasoning, tool arguments or usage estimates. Optional first-
        progress timing begins before connecting. Only parsed meaningful SSE
        deltas renew the idle timer; JSON fallback cannot expose progress and must
        complete under the first-progress timer. Neither timer extends ``control``.
        """
        if observation is not None and not isinstance(observation, dict):
            raise TypeError("observation must be a dict or None")
        started = time.monotonic()
        watch = _ProgressWatch(self.config, observation, started)
        seconds = _positive_seconds(self.config.request_timeout_seconds)
        own_deadline = started + seconds
        control = RequestControl(seconds) if control is None else control
        if not isinstance(control, RequestControl):
            raise TypeError("control must be RequestControl or None")
        try:
            control.check()
        except RequestCancelled as exc:
            watch.finish("cancelled")
            raise RequestCancelled(self._redact(str(exc))) from None
        except RequestDeadlineExceeded:
            watch.finish("total_deadline")
            raise
        deadline = min(own_deadline, control.deadline)
        loop = asyncio.new_event_loop()
        task = loop.create_task(self._request(payload, control, deadline, watch))

        def cancel_task():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # Completion and cancellation can race with loop.close().

        unsubscribe = control._subscribe(cancel_task)
        try:
            return loop.run_until_complete(task)
        except asyncio.CancelledError:
            watch.finish("cancelled")
            try:
                control.check()
            except RequestCancelled as exc:
                raise RequestCancelled(self._redact(str(exc))) from None
            raise RequestCancelled("Request cancelled") from None
        except RequestCancelled as exc:
            watch.finish("cancelled")
            raise RequestCancelled(self._redact(str(exc))) from None
        except RequestProgressTimeout as exc:
            watch.finish(exc.phase)
            raise
        except TimeoutError:
            watch.finish("total_deadline")
            raise RequestDeadlineExceeded("Total request deadline exceeded") from None
        except InvalidProviderResponse:
            watch.finish("invalid_response")
            raise
        except (aiohttp.ClientError, OSError, ValueError) as exc:
            watch.finish("transport_error")
            raise TransportError(self._redact(f"{type(exc).__name__}: {exc}")) from None
        finally:
            unsubscribe()
            # Finish cancellation and session/connector cleanup before returning.
            # Do not await shutdown_default_executor(): getaddrinfo itself is an
            # OS call we cannot cancel, and waiting for it defeats the deadline.
            pending = asyncio.all_tasks(loop)
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _request(self, payload, control, deadline, watch):
        control.check()
        watch.control, watch.deadline = control, deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestDeadlineExceeded("Total request deadline exceeded")
        # The outer timeout never moves. The inner scope is rescheduled only by
        # semantic progress, and covers connect/headers plus blocked body reads.
        async with asyncio.timeout(remaining):
            first_remaining = None if watch.first is None else max(
                0.0, watch.started + watch.first - time.monotonic())
            progress_timer = asyncio.timeout(first_remaining)
            watch.timer = progress_timer
            try:
                async with progress_timer:
                    return await self._http_request(payload, control, deadline, watch)
            except TimeoutError:
                # Cancellation and the immutable overall deadline take precedence
                # even when two timers become ready on the same event-loop tick.
                control.check()
                if time.monotonic() >= deadline:
                    raise RequestDeadlineExceeded("Total request deadline exceeded") from None
                if progress_timer.expired():
                    raise RequestProgressTimeout("first_progress" if watch.last is None else "progress_idle") from None
                raise

    async def _http_request(self, payload, control, deadline, watch):
        # Identify this client explicitly: the self-hosted GLM gateway rejected
        # default urllib headers during diagnosis (Cloudflare 1010). No browser
        # impersonation, credential-bearing custom headers or proxy inheritance.
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "AgentCompany/Stage1"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        connector = aiohttp.TCPConnector(ssl=self._ssl_context, force_close=True)
        async with aiohttp.ClientSession(connector=connector, trust_env=False,
                                        cookie_jar=aiohttp.DummyCookieJar(),
                                        timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.post(self.url, json=payload, headers=headers,
                                    allow_redirects=False) as response:
                watch.stamp("headers_received", "headers")
                watch.out["provider_http_status"] = response.status
                watch.check()
                is_sse = response.status < 300 and response.content_type == "text/event-stream"
                watch.out.update(response_format="sse" if is_sse else "json", progress_observable=is_sse)
                if is_sse:
                    body = await read_chat_stream(response, control, watch)
                    watch.check()
                    watch.out["response_complete"] = True
                    watch.finish("completed")
                    return response.status, self._redact(body)
                if response.status >= 300:
                    # Preserve the prior bounded HTTP-error body behavior.
                    chunks, length = [], 0
                    while length < 32768:
                        chunk = await response.content.read(32768 - length)
                        if not chunk:
                            break
                        watch.received(chunk)
                        chunks.append(chunk)
                        length += len(chunk)
                    raw = b"".join(chunks)
                else:
                    chunks = []
                    while chunk := await response.content.readany():
                        watch.received(chunk)
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                control.check()
                if time.monotonic() >= deadline:
                    raise RequestDeadlineExceeded("Total request deadline exceeded")
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    if response.status < 300:
                        raise InvalidProviderResponse(
                            f"Provider returned invalid JSON (HTTP {response.status})") from None
                    body = {"error": {"message": raw.decode("utf-8", "replace")}}
                body = self._redact(body)
                watch.check()
                watch.out["response_complete"] = response.content.at_eof()
                watch.finish("http_error" if response.status >= 300 else "completed")
                return response.status, body
