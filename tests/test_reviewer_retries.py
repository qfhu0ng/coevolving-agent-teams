"""Reviewer transport retries and the optimizer reading audit (2026-09-30).

The reviewer used to fail on its first transport error; it now resends the
failed step's request through the optimizer's shared helper. Both recoveries
count retried attempts. The optimizer's reading audit reaches the reviewer.
Scripted completions only: no paid calls.
"""
from dataclasses import replace
import json

import pytest

from agent_company.evolution.loop import ReviewRequest
from agent_company.evolution.optimizer import PolicyOptimizationError, StrategyOptimizer
from agent_company.runtime.relay import OpenAIChatTransport, ProviderConfig, TokenRates
from agent_company.evolution.reviewer import PolicyReviewError, StrategyReviewer
from agent_company.runtime.transport import InvalidProviderResponse, RequestCancelled, TransportError
from test_optimizer_working_memory import (
    Script, request as optimizer_request, retry_clock as shared_retry_clock, submit as submit_policy,
)
from test_reviewer import ONE_CALL_COST, USAGE, call, provider, response, review_request, scripted, submit

# The optimizer tests' fake clock, which also drives the shared retry helper.
retry_clock = shared_retry_clock


def reviewer(tmp_path, replies, *, retries=2, timeout=120):
    captured, complete = scripted(replies)
    delays = []
    review = StrategyReviewer(provider(request_timeout_seconds=timeout), tmp_path / "reviewer",
                              completion=complete, transport_retries=retries, sleep=delays.append)
    return review, captured, delays


def read_trace():
    return response(call("read_training_file", {"path": "batch_001/task_001/trace.txt"}))


def test_failed_step_is_resent_without_restarting_the_session(tmp_path):
    review, captured, delays = reviewer(tmp_path, [read_trace(), OSError("SSLV3_ALERT_BAD_RECORD_MAC"), submit()])
    result = review(review_request(tmp_path))
    assert result.recommendation == "prefer_candidate"
    # The step-1 request is resent unchanged; the step-0 tool round stays in it.
    assert len(captured) == 3 and captured[1] == captured[2]
    assert len(captured[2]["messages"]) > len(captured[0]["messages"])
    assert delays == [1.0]
    assert result.usage["provider_attempts"] == 3 and result.usage["unknown_usage_attempts"] == 1
    assert result.cost_usd is None and result.known_cost_usd == pytest.approx(2 * ONE_CALL_COST)
    directory = tmp_path / "reviewer/batch_0001"
    record = json.loads((directory / "retry_001_01.json").read_text())
    assert "SSLV3_ALERT_BAD_RECORD_MAC" in record["error"] and record["delay_seconds"] == 1.0
    assert sorted(path.name for path in directory.glob("request_*.json")) == ["request_000.json", "request_001.json"]


def test_retryable_status_is_resent_after_retry_after_and_its_usage_kept(tmp_path):
    busy = (524, {"error": {"message": "timeout", "retry_after": 5}, "usage": dict(USAGE)})
    review, captured, delays = reviewer(tmp_path, [busy, submit()])
    result = review(review_request(tmp_path))
    assert delays == [5.0] and len(captured) == 2
    assert result.usage["provider_attempts"] == 2 and result.usage["unknown_usage_attempts"] == 0
    assert result.known_cost_usd == pytest.approx(2 * ONE_CALL_COST)


@pytest.mark.parametrize("reply, retries", [((400, {"error": {"message": "bad request"}}), 2),
                                            (OSError("reset"), 0)])
def test_non_retryable_status_and_the_default_single_attempt_fail_the_review(tmp_path, reply, retries):
    review, captured, delays = reviewer(tmp_path, [reply, submit()], retries=retries)
    with pytest.raises(PolicyReviewError) as caught:
        review(review_request(tmp_path))
    assert len(captured) == 1 and delays == []
    assert caught.value.usage["provider_attempts"] == 1 and caught.value.usage["unknown_usage_attempts"] == 1
    assert not (tmp_path / "reviewer/batch_0001/review.json").exists()


@pytest.mark.parametrize("error", [RequestCancelled("cancelled"), TimeoutError("timed out"),
                                   InvalidProviderResponse("invalid JSON"), ValueError("client bug")])
def test_cancellation_deadlines_invalid_replies_and_bugs_are_not_resent(tmp_path, error):
    review, captured, delays = reviewer(tmp_path, [error, submit()])
    with pytest.raises(PolicyReviewError):
        review(review_request(tmp_path))
    assert len(captured) == 1 and delays == []


def test_retries_share_the_request_deadline_through_the_real_transport(tmp_path, monkeypatch, retry_clock):
    prov = provider(request_timeout_seconds=600)
    transport = OpenAIChatTransport(prov)
    attempts = []

    async def exchange(payload, control, deadline, watch):
        attempts.append((retry_clock.now, control, deadline))
        if len(attempts) == 1:
            retry_clock.advance(598)
            raise TransportError("reset")
        retry_clock.advance(1)
        control.check()
        return 200, submit()  # A renewed deadline would wrongly accept this.

    monkeypatch.setattr(transport, "_request", exchange)
    review = StrategyReviewer(prov, tmp_path / "reviewer", completion=transport,
                              transport_retries=2, sleep=retry_clock.advance)
    with pytest.raises(PolicyReviewError, match="deadline") as caught:
        review(review_request(tmp_path))
    assert [start for start, _, _ in attempts] == [0, 599]
    assert [deadline for _, _, deadline in attempts] == [600, 600]
    assert attempts[0][1] is attempts[1][1]
    assert caught.value.usage["provider_attempts"] == 2


def write_interrupted_session(directory, *, last_resent=True):
    # Step 0 answered; step 1 failed twice. Its third attempt was in flight
    # (last_resent) or never sent because the session stopped during backoff.
    directory.mkdir(parents=True)
    (directory / "request_000.json").write_text("{}")
    (directory / "response_000.json").write_text(json.dumps(response()))
    (directory / "request_001.json").write_text("{}")
    (directory / "retry_001_01.json").write_text(json.dumps(
        {"error": "OSError: reset", "delay_seconds": 1.0, "resent": True}))
    (directory / "retry_001_02.json").write_text(json.dumps(
        {"http_status": 524, "response": {"usage": USAGE}, "delay_seconds": 3.0, "resent": last_resent}))


def test_interrupted_review_recovery_counts_retried_attempts(tmp_path):
    request = review_request(tmp_path)
    write_interrupted_session(tmp_path / "reviewer/batch_0001")
    review = StrategyReviewer(provider(), tmp_path / "reviewer",
                              completion=lambda _: pytest.fail("Recovery must not call the model"))
    with pytest.raises(PolicyReviewError, match="no additional model calls") as caught:
        review.recover(request)
    assert caught.value.usage["provider_attempts"] == 4 and caught.value.usage["unknown_usage_attempts"] == 2
    assert caught.value.cost_usd is None and caught.value.known_cost_usd == pytest.approx(2 * ONE_CALL_COST)


def test_interrupted_optimizer_recovery_counts_retried_attempts(tmp_path):
    # Before 2026-09-30 only request_*.json were counted, so retried attempts
    # (and their unknown usage) disappeared from an interrupted session's total.
    request = review_request(tmp_path).proposal_request
    write_interrupted_session(tmp_path / "optimizer/batch_0001")
    optimizer = StrategyOptimizer(provider(), tmp_path / "optimizer",
                                  completion=lambda _: pytest.fail("Recovery must not call the model"))
    with pytest.raises(PolicyOptimizationError) as caught:
        optimizer.recover(request)
    assert caught.value.usage["provider_attempts"] == 4 and caught.value.usage["unknown_usage_attempts"] == 2
    assert caught.value.known_cost_usd == pytest.approx(2 * ONE_CALL_COST)


def test_reviewer_sees_the_optimizer_reading_audit_or_that_none_was_recorded(tmp_path):
    request = review_request(tmp_path)
    secret = provider().api_key
    audit = {"mode": "working_memory", "searches": 3,
             "files_shown": [{"path": "batch_001/task_001/trace.txt", "page_chars": 27}],
             "citations": [{"path": "batch_001/task_001/trace.txt", "quote": "RECEIPT " + secret}],
             "caveat": "being shown text is not the same as using it correctly"}
    audited = replace(request, candidate=replace(request.candidate, reading_audit=audit))
    captured, complete = scripted([submit(), submit()])
    StrategyReviewer(provider(), tmp_path / "audited", completion=complete)(audited)
    StrategyReviewer(provider(), tmp_path / "unaudited", completion=complete)(request)
    seen = [json.loads(payload["messages"][1]["content"])["optimizer_reading_audit"] for payload in captured]
    assert seen[0]["recorded"] is True and "check the evidence yourself" in seen[0]["note"]
    assert seen[0]["audit"]["files_shown"] == audit["files_shown"]
    assert seen[0]["audit"]["citations"][0]["quote"] == "RECEIPT [REDACTED]"
    assert secret not in json.dumps(captured)
    # No audit is reported as unrecorded, never as an empty reading.
    assert seen[1]["recorded"] is False and seen[1]["note"].startswith("Not recorded")
    assert "audit" not in seen[1]


def test_recovery_does_not_count_a_resend_that_was_scheduled_but_never_sent(tmp_path):
    request = review_request(tmp_path)
    write_interrupted_session(tmp_path / "reviewer/batch_0001", last_resent=False)
    review = StrategyReviewer(provider(), tmp_path / "reviewer",
                              completion=lambda _: pytest.fail("Recovery must not call the model"))
    with pytest.raises(PolicyReviewError) as caught:
        review.recover(request)
    # Step 0, then step 1's two failed attempts; no third attempt was sent.
    assert caught.value.usage["provider_attempts"] == 3 and caught.value.usage["unknown_usage_attempts"] == 1


BUSY = (524, {"error": {"message": "timeout"}, "usage": dict(USAGE)})


def stop_during_backoff(clock):
    return lambda delay: clock.advance(delay + 700)  # Suspended past the request deadline.


def test_reviewer_runtime_and_recovery_agree_when_the_deadline_ends_the_backoff(tmp_path, retry_clock):
    # Review 2026-09-30: the retry record was written before backoff, so recovery
    # counted a second, never-sent attempt and marked complete spend incomplete.
    request = review_request(tmp_path)
    captured, complete = scripted([BUSY, submit()])
    review = StrategyReviewer(provider(request_timeout_seconds=600), tmp_path / "reviewer", completion=complete,
                              transport_retries=2, sleep=stop_during_backoff(retry_clock))
    with pytest.raises(PolicyReviewError, match="deadline") as live:
        review(request)
    with pytest.raises(PolicyReviewError) as recovered:
        review.recover(request)
    assert len(captured) == 1
    for caught in (live, recovered):
        assert caught.value.usage["provider_attempts"] == 1 and caught.value.usage["unknown_usage_attempts"] == 0
        assert caught.value.cost_usd == pytest.approx(ONE_CALL_COST)
    record = json.loads((tmp_path / "reviewer/batch_0001/retry_000_01.json").read_text())
    assert record["resent"] is False and record["http_status"] == 524


def test_optimizer_runtime_and_recovery_agree_when_the_deadline_ends_the_backoff(tmp_path, retry_clock):
    request = review_request(tmp_path).proposal_request
    captured, complete = scripted([BUSY])
    optimizer = StrategyOptimizer(provider(request_timeout_seconds=600), tmp_path / "optimizer",
                                  completion=complete, transport_retries=2, sleep=stop_during_backoff(retry_clock))
    with pytest.raises(PolicyOptimizationError, match="deadline") as live:
        optimizer(request)
    with pytest.raises(PolicyOptimizationError) as recovered:
        optimizer.recover(request)
    assert len(captured) == 1
    for caught in (live, recovered):
        assert caught.value.usage["provider_attempts"] == 1 and caught.value.usage["unknown_usage_attempts"] == 0
        assert caught.value.known_cost_usd == pytest.approx(ONE_CALL_COST)


def test_a_sent_resend_is_marked_before_it_is_attempted(tmp_path):
    review, captured, _ = reviewer(tmp_path, [OSError("reset"), submit()])
    review(review_request(tmp_path))
    assert json.loads((tmp_path / "reviewer/batch_0001/retry_000_01.json").read_text())["resent"] is True


def test_optimizer_redacts_its_own_key_from_the_audit_before_the_reviewer_sees_it(tmp_path):
    # Review 2026-09-30: a verified quote can carry the optimizer's key out of the
    # evidence; a reviewer with a different key could not redact it. Fictional keys.
    optimizer_key, reviewer_key = "optimizer-key-K1-fictional", "reviewer-key-K2-fictional"
    quote = "deploy note: token=" + optimizer_key
    request = optimizer_request(tmp_path, {"task/notes.txt": quote + "\nother text"})
    optimizer = StrategyOptimizer(
        ProviderConfig(model="fake", base_url="https://provider.invalid/v1", api_key=optimizer_key,
                       max_output_tokens=256, rates=TokenRates(2, 0.5, 4)),
        tmp_path / "optimizer", require_citations=True,
        completion=Script(submit_policy([{"path": "task/notes.txt", "quote": quote, "supports": "a note"}])))
    proposal = optimizer(request)
    assert proposal.reading_audit["citations"][0]["quote"] == "deploy note: token=[REDACTED]"
    for path in (tmp_path / "optimizer").rglob("*.json"):
        assert optimizer_key not in path.read_text(), path.name
    captured, complete = scripted([submit()])
    StrategyReviewer(provider(api_key=reviewer_key), tmp_path / "reviewer", completion=complete)(
        ReviewRequest(proposal_request=request, candidate=proposal))
    assert optimizer_key not in json.dumps(captured)
    sent = json.loads(captured[0]["messages"][1]["content"])["optimizer_reading_audit"]
    assert sent["audit"]["citations"][0]["quote"] == "deploy note: token=[REDACTED]"
