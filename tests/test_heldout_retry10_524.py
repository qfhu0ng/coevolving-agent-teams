"""Mock-only checks for the heldout retry10 + HTTP524 policy, now synced to main.

Run against the main package, or with the preserved variant on PYTHONPATH.
The real retry predicate is exercised without contacting any provider.
"""
import copy
import json
import time

import pytest

from agent_company.runtime.relay import ProviderConfig, TeamRelay
from agent_company.runtime.team import WorkerRegistry
from agent_company.runtime.transport import RequestControl
from test_team import request, response, workspace


def make_relay(tmp_path, statuses, *, role="single", backoff=0,
               retry_after=None, nested_hint=False, failed_usage=False):
    """All calls terminate in the injected function; no provider is contacted."""
    seen = []

    def upstream(payload):
        seen.append(copy.deepcopy(payload))
        status = statuses[min(len(seen) - 1, len(statuses) - 1)]
        if status == 200:
            return status, response(usage={"prompt_tokens": 11, "completion_tokens": 4})
        raw = {"error": {"message": "mock HTTP524 failure"}}
        if retry_after is not None:
            (raw["error"] if nested_hint else raw)["retry_after"] = retry_after
        if failed_usage:
            raw["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
        return status, raw

    registry = WorkerRegistry(workspace(tmp_path))
    if role not in {"single", "master"}:
        registry.register(role)
    relay = TeamRelay(
        ProviderConfig("mock-only", "http://unused.invalid", max_attempts=11,
                       request_timeout_seconds=600, retry_backoff_seconds=backoff),
        registry, "mock communication policy", tmp_path / "logs", 7200, upstream,
        execution_mode="single" if role == "single" else "team")
    return relay, seen


def events(relay):
    return [json.loads(line) for line in
            (relay.output / "events.jsonl").read_text().splitlines()]


def fake_clock(monkeypatch):
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds + 1e-6))
    return now


@pytest.mark.parametrize("role", ["single", "master", "integrator"])
def test_524_recovers_for_single_master_and_worker_without_losing_unknown_usage(tmp_path, role):
    relay, seen = make_relay(tmp_path, [524, 200], role=role)
    control = RequestControl(600)
    deadline = control.deadline
    assert relay.handle(request(role), control=control)[0] == 200
    assert control.deadline == deadline
    assert len(seen) == 2 and seen[0] == seen[1]
    assert [record["retry_index"] for record in relay._records] == [0, 1]
    assert [record["provider_http_status"] for record in relay._records] == [524, 200]
    assert relay.summary()["usage"]["total_tokens"] == 15
    assert relay.summary()["usage_complete"] is False
    assert not relay._active_requests and not relay._controls


@pytest.mark.parametrize("role", ["single", "master", "integrator"])
def test_524_exhaustion_is_initial_plus_ten_not_eleven_retries(tmp_path, role):
    relay, seen = make_relay(tmp_path, [524], role=role)
    assert relay.handle(request(role))[0] == 424
    assert len(seen) == 11
    assert all(payload == seen[0] for payload in seen)
    assert [record["retry_index"] for record in relay._records] == list(range(11))
    assert len([event for event in events(relay)
                if event["event"] == "provider_retry_scheduled"]) == 10
    assert relay.summary()["usage_complete"] is False
    assert not relay._active_requests and not relay._controls


def test_tenth_524_retry_can_succeed_when_original_budget_allows(tmp_path):
    relay, seen = make_relay(tmp_path, [524] * 10 + [200])
    assert relay.handle(request("single"))[0] == 200
    assert len(seen) == 11
    assert relay._records[-1]["retry_index"] == 10
    assert relay.summary()["usage"]["total_tokens"] == 15
    assert relay.summary()["usage_complete"] is False


def test_failed_attempt_usage_is_retained_when_provider_supplies_it(tmp_path):
    relay, seen = make_relay(tmp_path, [524, 200], failed_usage=True)
    assert relay.handle(request("single"))[0] == 200
    assert len(seen) == 2
    assert relay.summary()["usage"]["total_tokens"] == 25
    assert relay.summary()["usage_complete"] is True


@pytest.mark.parametrize("nested_hint", [False, True])
def test_body_retry_after_120_keeps_original_600_second_deadline(tmp_path, monkeypatch, nested_hint):
    relay, seen = make_relay(tmp_path, [524, 200], backoff=1,
                            retry_after=120, nested_hint=nested_hint)
    fake_clock(monkeypatch)
    control = RequestControl(600)
    deadline = control.deadline
    task_deadline = relay.deadline
    assert relay.handle(request("single"), control=control)[0] == 200
    assert len(seen) == 2
    assert control.deadline == deadline and relay.deadline == task_deadline
    retries = [event for event in events(relay)
               if event["event"] == "provider_retry_scheduled"]
    assert [event["delay_seconds"] for event in retries] == [120]
    assert relay._records[0]["provider_retry_after_seconds"] == 120
    assert relay.summary()["usage_complete"] is False


def test_body_hint_cannot_extend_request_deadline(tmp_path):
    relay, seen = make_relay(tmp_path, [524], backoff=1, retry_after=120)
    control = RequestControl(100)
    deadline = control.deadline
    assert relay.handle(request("single"), control=control)[0] == 424
    assert len(seen) == 1 and control.deadline == deadline
    assert not any(event["event"] == "provider_retry_scheduled" for event in events(relay))


def test_body_hint_cannot_extend_task_deadline(tmp_path):
    relay, seen = make_relay(tmp_path, [524], backoff=1, retry_after=120)
    relay.deadline = time.monotonic() + 100
    deadline = relay.deadline
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 1 and relay.deadline == deadline
    assert not any(event["event"] == "provider_retry_scheduled" for event in events(relay))


def test_cancellation_during_hint_wait_does_not_start_another_attempt(tmp_path):
    relay, seen = make_relay(tmp_path, [524], backoff=1, retry_after=120)
    control = RequestControl(600)
    original_event = relay.event

    def event(name, **fields):
        result = original_event(name, **fields)
        if name == "provider_retry_scheduled":
            control.cancel()
        return result

    relay.event = event
    assert relay.handle(request("single"), control=control)[0] == 424
    assert len(seen) == 1
    assert not relay._active_requests and not relay._controls


def test_original_backoff_and_600_second_budget_can_stop_before_tenth_retry(tmp_path, monkeypatch):
    relay, seen = make_relay(tmp_path, [524], backoff=1)
    fake_clock(monkeypatch)
    control = RequestControl(600)
    deadline = control.deadline
    assert relay.handle(request("single"), control=control)[0] == 424
    assert control.deadline == deadline
    assert len(seen) == 9
    assert [event["delay_seconds"] for event in events(relay)
            if event["event"] == "provider_retry_scheduled"] == [
                1, 3, 7, 15, 31, 63, 127, 255]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_524_text_does_not_make_permanent_http_error_retryable(tmp_path, status):
    relay, seen = make_relay(tmp_path, [status], retry_after=120)
    assert relay.handle(request("single"))[0] == 424
    assert len(seen) == 1
