"""Mock-only checks for the user-authorized branch's request retry policy."""
import copy
import json
import time

import pytest

from agent_company.runtime.relay import ProviderConfig, TeamRelay
from agent_company.runtime.transport import RequestControl
from agent_company.runtime.team import WorkerRegistry
from test_team import workspace, request, response


def make_relay(tmp_path, statuses, *, backoff=0, all_usage=False, retry_after=None, nested_hint=False):
    seen = []

    def upstream(payload):
        seen.append(copy.deepcopy(payload))
        status = statuses[min(len(seen) - 1, len(statuses) - 1)]
        if status == 200:
            return status, response(usage={"prompt_tokens": 11, "completion_tokens": 4})
        raw = {"error": {"message": "test-only failure", "retryable": False}}
        if retry_after is not None:
            (raw["error"] if nested_hint else raw)["retry_after"] = retry_after
        if all_usage:
            raw["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
        return status, raw

    relay = TeamRelay(ProviderConfig("fake", "http://unused.invalid", max_attempts=4,
                                    request_timeout_seconds=600, retry_backoff_seconds=backoff),
                      WorkerRegistry(workspace(tmp_path)), "", tmp_path / "logs", 7200,
                      upstream, execution_mode="single")
    return relay, seen


def events(relay):
    return [json.loads(line) for line in (relay.output / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("statuses", [[520, 200], [522, 200], [525, 200], [520, 522, 525, 200]])
def test_success_with_at_most_three_retries(tmp_path, statuses):
    relay, seen = make_relay(tmp_path, statuses)
    assert relay.handle(request("single"))[0] == 200
    assert len(seen) == len(statuses)
    assert all(payload == seen[0] for payload in seen)
    assert [r["retry_index"] for r in relay._records] == list(range(len(statuses)))
    assert relay.summary()["usage"]["total_tokens"] == 15
    assert relay.summary()["usage_complete"] is False


@pytest.mark.parametrize("statuses", [[520], [522], [525], [520, 522, 525, 520, 200]])
def test_exhaustion_stops_after_four_attempts(tmp_path, statuses):
    relay, seen = make_relay(tmp_path, statuses)
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 4
    assert [r["retry_index"] for r in relay._records] == [0, 1, 2, 3]
    assert len([e for e in events(relay) if e["event"] == "provider_retry_scheduled"]) == 3
    assert not relay._active_requests and not relay._controls


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_errors_not_retried_even_if_body_mentions_520(tmp_path, status):
    relay, seen = make_relay(tmp_path, [status])
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 1


def test_failed_attempt_usage_is_not_discarded(tmp_path):
    relay, seen = make_relay(tmp_path, [525, 520, 503, 200], all_usage=True)
    assert relay.handle(request("single"))[0] == 200
    assert len(seen) == 4
    assert relay.summary()["usage"]["total_tokens"] == 45
    assert relay.summary()["usage_complete"] is True


def test_retry_uses_original_request_deadline(tmp_path):
    relay, seen = make_relay(tmp_path, [525], backoff=1)
    control = RequestControl(0.5)
    deadline = control.deadline
    assert relay.handle(request("single"), control=control)[0] == 424
    assert len(seen) == 1 and control.deadline == deadline
    assert not any(e["event"] == "provider_retry_scheduled" for e in events(relay))


def test_retry_stops_when_task_deadline_cannot_fit_backoff(tmp_path):
    relay, seen = make_relay(tmp_path, [520], backoff=1)
    relay.deadline = time.monotonic() + 0.5
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 1


def test_cancellation_during_retry_prevents_next_attempt(tmp_path):
    relay, seen = make_relay(tmp_path, [525], backoff=1)
    control = RequestControl(600)
    original = relay.event

    def event(name, **fields):
        result = original(name, **fields)
        if name == "provider_retry_scheduled":
            control.cancel()
        return result

    relay.event = event
    assert relay.handle(request("single"), control=control)[0] == 424
    assert len(seen) == 1


def test_backoff_schedule_is_one_three_seven_seconds(tmp_path, monkeypatch):
    relay, seen = make_relay(tmp_path, [525], backoff=1)
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds + 1e-6))
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 4
    assert [e["delay_seconds"] for e in events(relay)
            if e["event"] == "provider_retry_scheduled"] == [1, 3, 7]


@pytest.mark.parametrize("nested_hint", [False, True])
def test_522_retry_after_120_seconds_is_honored(tmp_path, monkeypatch, nested_hint):
    relay, seen = make_relay(tmp_path, [522, 200], backoff=1,
                            retry_after=120, nested_hint=nested_hint)
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds + 1e-6))
    control = RequestControl(600)
    deadline = control.deadline
    assert relay.handle(request("single"), control=control)[0] == 200
    assert len(seen) == 2 and control.deadline == deadline
    assert [e["delay_seconds"] for e in events(relay)
            if e["event"] == "provider_retry_scheduled"] == [120]
    assert relay._records[0]["provider_retry_after_seconds"] == 120
    assert relay.summary()["usage_complete"] is False


@pytest.mark.parametrize("hint", [True, False, -1, "120", float("nan"),
                                  float("inf"), 10 ** 1000, [], {}])
def test_malformed_retry_after_does_not_change_backoff(tmp_path, monkeypatch, hint):
    relay, seen = make_relay(tmp_path, [522, 200], backoff=1, retry_after=hint)
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds + 1e-6))
    assert relay.handle(request("single"))[0] == 200
    assert len(seen) == 2
    assert [e["delay_seconds"] for e in events(relay)
            if e["event"] == "provider_retry_scheduled"] == [1]


def test_retry_after_cannot_extend_shared_deadline(tmp_path):
    relay, seen = make_relay(tmp_path, [522], backoff=1, retry_after=120)
    control = RequestControl(100)
    deadline = control.deadline
    assert relay.handle(request("single"), control=control)[0] == 424
    assert len(seen) == 1 and control.deadline == deadline
    assert not any(e["event"] == "provider_retry_scheduled" for e in events(relay))


def test_short_hint_cannot_shorten_backoff(tmp_path, monkeypatch):
    relay, seen = make_relay(tmp_path, [522], backoff=1, retry_after=0.5)
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds + 1e-6))
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 4
    assert [e["delay_seconds"] for e in events(relay)
            if e["event"] == "provider_retry_scheduled"] == [1, 3, 7]


def test_hint_does_not_make_permanent_http_error_retryable(tmp_path):
    relay, seen = make_relay(tmp_path, [401], retry_after=120)
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 1


def test_retry_after_preserves_task_deadline_and_cancellation(tmp_path):
    relay, seen = make_relay(tmp_path, [522], backoff=1, retry_after=120)
    relay.deadline = time.monotonic() + 100
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 1


def test_522_cancellation_during_long_wait_prevents_second_attempt(tmp_path):
    relay, seen = make_relay(tmp_path, [522], backoff=1, retry_after=120)
    control = RequestControl(600)
    original = relay.event
    def event(name, **fields):
        result = original(name, **fields)
        if name == "provider_retry_scheduled":
            control.cancel()
        return result
    relay.event = event
    assert relay.handle(request("single"), control=control)[0] == 424
    assert len(seen) == 1
