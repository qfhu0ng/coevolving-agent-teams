"""Zero-API fixtures mirror admission, retry, native and verifier records."""
import json
from pathlib import Path

import pytest

from agent_company.deploy.run_health import runtime_call_outcomes, s0_readiness


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def runtime(root, *, calls=None, stop="native_cli_completed", workers=(), terminals=(), extra_events=()):
    """Each call lists its retries as error kinds, followed by None for success."""
    calls = calls or [("master", [None])]
    events, attempt, errors = [], 0, []
    for source, (role, retries) in enumerate(calls, 1):
        events.append({"event": "logical_model_call_admitted", "role": role, "purpose": "coding",
                       "logical_model_call": source})
        for retry, kind in enumerate(retries):
            attempt += 1
            identity = {"attempt": attempt, "source_request": source, "role": role,
                        "purpose": "coding", "retry_index": retry}
            events.append({"event": "provider_attempt_started", **identity})
            finished = {"event": "provider_attempt_finished", **identity, "time": attempt,
                        "provider_http_status": None if kind else 200, "relay_http_status": 424 if kind else 200,
                        "error_kind": kind, "error": f"{kind}: provider failed" if kind else None,
                        "usage": None if kind else {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
            events.append(finished)
            if kind:
                errors.append(finished)
    lifecycle = [{"event": "compatibility_loaded"}, *terminals]
    summary = {"execution_mode": "team", "stop_reason": stop,
               "execution": {"returncode": 124 if stop == "task_timeout" else 0,
                             "timed_out": stop == "task_timeout"}, "workers": {worker: "/work/" + worker for worker in workers},
               "logical_model_calls": len(calls), "provider_attempts": attempt, "completed_attempts": attempt,
               "active_requests": 0, "inflight_attempts": 0, "errors": errors,
               "native_result": {"lifecycle": {"events": lifecycle}}}
    write(root / "summary.json", summary)
    write(root / "relay/usage.json", summary)
    (root / "relay/events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in [*events, *extra_events]))
    (root / "qwen_stderr.txt").write_text("".join("Agent Company lifecycle: " + json.dumps(event) + "\n" for event in lifecycle))
    return root


def s0(tmp_path, step, **kwargs):
    directory = tmp_path / step
    trial = directory / "run/pandoc/team_p0"
    runtime(trial / "team", **kwargs)
    stop = kwargs.get("stop", "native_cli_completed")
    write(trial / "task_result.json", {"status": "failed" if stop == "task_timeout" else "completed", "score": 0,
        "metrics": {"inference_cleanup_confirmed": True, "failures": [],
                    "verifier": {"status": "completed", "passed_tests": 0, "valid_tests": 200, "reward": 0}}})
    return {"directory": str(directory), "launch": {"step": step}, "terminal": True, "pid_alive": False,
            "refused_before_output": False, "progress": {"state": "completed", "running": [], "pending": [],
                "cleanup_attention": [], "rows": [{"task_id": "pandoc", "arm_id": "team_p0", "inference_cleanup_confirmed": True}]}}


def pair(tmp_path, **kwargs):
    return [s0(tmp_path, "s0_propose", **kwargs), s0(tmp_path, "s0_global", **kwargs)]


def test_transient_retry_recovers_same_logical_call_with_usage(tmp_path):
    result = runtime_call_outcomes(runtime(tmp_path, calls=[("master", ["provider_transport_error", None])]))
    assert result["complete"] and result["passed"]
    assert result["total_calls"] == 1 and result["failed_calls"] == 0
    assert result["all_final_usage"] is True
    assert len(result["transient_failures"]) == 1 and result["hard_failures"] == []
    assert result["calls"][0]["attempts"] == 2


def test_replacement_worker_does_not_recover_failed_logical_call(tmp_path):
    result = runtime_call_outcomes(runtime(tmp_path, calls=[("worker1", ["provider_transport_error"]), ("worker2", [None])]))
    assert result["complete"] and not result["passed"]
    assert result["total_calls"] == 2 and result["failed_calls"] == 1
    assert result["all_final_usage"] is False and result["hard_failures"] == []


def test_late_delivery_cancellation_overrides_successful_attempt(tmp_path):
    result = runtime_call_outcomes(runtime(tmp_path, extra_events=[{
        "event": "native_request_ended", "role": "master", "source_request": 1,
        "error_kind": "task_deadline_exceeded", "role_policy_batch_committed": False}]))
    assert result["complete"] and not result["passed"] and result["failed_calls"] == 1


def test_successful_response_without_usage_is_hard_failure(tmp_path):
    runtime(tmp_path)
    path = tmp_path / "relay/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1]["usage"] = None
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = runtime_call_outcomes(tmp_path)
    assert result["complete"] and not result["passed"]
    assert "successful_call_usage_unknown" in result["hard_failures"]


@pytest.mark.parametrize("corruption", ["bad_line", "missing_finish", "duplicate_finish", "counter", "missing_source"])
def test_unknown_evidence_never_passes(tmp_path, corruption):
    runtime(tmp_path)
    path = tmp_path / "relay/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "missing_finish":
        events.pop()
    elif corruption == "duplicate_finish":
        events.append(events[-1])
    elif corruption == "missing_source":
        del events[-1]["source_request"]
    elif corruption == "counter":
        summary = json.loads((tmp_path / "relay/usage.json").read_text())
        summary["logical_model_calls"] = 2
        write(tmp_path / "relay/usage.json", summary)
    path.write_text("".join(json.dumps(event) + "\n" for event in events) + ("bad\n" if corruption == "bad_line" else ""))
    result = runtime_call_outcomes(tmp_path)
    assert not result["complete"] and not result["passed"] and result["evidence_issues"]


def test_evidence_write_failure_outside_events_remains_fatal(tmp_path):
    runtime(tmp_path)
    summary = json.loads((tmp_path / "relay/usage.json").read_text())
    summary["errors"] = [{"error_kind": "evidence_write_failed", "error": "late save failed", "role_policy": True}]
    write(tmp_path / "relay/usage.json", summary)
    result = runtime_call_outcomes(tmp_path)
    assert result["complete"] and not result["passed"]
    assert result["hard_failures"] == ["evidence_write_failed"]


def test_zero_official_scores_are_valid_and_diagnostic_unknown_does_not_gate(tmp_path):
    rows = pair(tmp_path)
    for row in rows:
        write(Path(row["directory"]) / "run/s0_report.json", {"usage_classification": None, "six_hour_extrapolation": None})
    result = s0_readiness(rows)
    assert result["satisfied"] and not result["reasons"]


def test_normal_wall_clock_budget_expiry_is_not_abnormal_exit(tmp_path):
    result = s0_readiness(pair(tmp_path, stop="task_timeout"))
    assert result["satisfied"]
    assert result["details"]["s0_propose"]["details"]["runtime_stop_reason"] == "task_timeout"


@pytest.mark.parametrize("state", ["missing", "duplicate", "live", "refused"])
def test_two_unique_terminal_s0_runs_are_required(tmp_path, state):
    rows = pair(tmp_path)
    if state == "missing":
        rows.pop()
    elif state == "duplicate":
        rows.append(rows[0])
    elif state == "live":
        rows[0].update(terminal=False, pid_alive=True)
    else:
        rows[0]["refused_before_output"] = True
    assert not s0_readiness(rows)["satisfied"]


@pytest.mark.parametrize("mutation", ["score", "cleanup", "exception", "oom"])
def test_s0_hard_failures_block(tmp_path, mutation):
    rows = pair(tmp_path)
    trial = Path(rows[0]["directory"]) / "run/pandoc/team_p0"
    result = json.loads((trial / "task_result.json").read_text())
    if mutation == "score":
        result["score"] = None
    elif mutation == "cleanup":
        result["metrics"]["inference_cleanup_confirmed"] = False
    elif mutation == "exception":
        result["metrics"]["failures"] = [{"phase": "submission_archive", "error_type": "ValueError"}]
    else:
        summary = json.loads((trial / "team/summary.json").read_text())
        summary["execution"]["oom_killed"] = True
        write(trial / "team/summary.json", summary)
    write(trial / "task_result.json", result)
    assert not s0_readiness(rows)["satisfied"]


@pytest.mark.parametrize(("failures", "expected"), [(1, True), (2, False)])
def test_final_failure_twenty_percent_boundary(tmp_path, failures, expected):
    calls = [("master", ["provider_transport_error"] if i < failures else [None]) for i in range(5)]
    result = s0_readiness(pair(tmp_path, calls=calls))
    assert result["satisfied"] is expected


def failed_worker(worker, error="provider_transport_error: provider failed"):
    return [{"event": "worker_terminal", "worker": worker, "status": "failed", "user_cancelled": False},
            {"event": "failure_notice_queued", "worker": worker, "error": "[API Error] " + error}]


@pytest.mark.parametrize(("deaths", "expected"), [(0, True), (1, False), (2, False)])
def test_at_least_half_provider_worker_deaths_boundary(tmp_path, deaths, expected):
    # Keep failed logical calls below 20%, isolating the worker condition.
    calls = [("master", [None])] * 10 + [("w1", ["provider_transport_error"]), ("w2", ["provider_transport_error"])]
    terminals = (failed_worker("w1") if deaths else []) + (failed_worker("w2") if deaths == 2 else [])
    result = s0_readiness(pair(tmp_path, calls=calls, workers=("w1", "w2"), terminals=terminals))
    assert result["satisfied"] is expected
    workers = result["details"]["s0_propose"]["details"]["workers"]
    assert len(workers["provider_failed_workers"]) == deaths


def test_worker_failure_not_inferred_from_unrelated_provider_incident(tmp_path):
    calls = [("master", [None])] * 10 + [("w1", ["provider_transport_error"])]
    rows = pair(tmp_path, calls=calls, workers=("w1",), terminals=failed_worker("w1", "unrelated native error"))
    result = s0_readiness(rows)
    assert not result["satisfied"]
    workers = result["details"]["s0_propose"]["details"]["workers"]
    assert workers["provider_failed_workers"] == [] and workers["unknown_failed_workers"] == ["w1"]


def test_missing_native_lifecycle_does_not_claim_zero_worker_deaths(tmp_path):
    rows = pair(tmp_path)
    (Path(rows[0]["directory"]) / "run/pandoc/team_p0/team/qwen_stderr.txt").unlink()
    assert not s0_readiness(rows)["satisfied"]


def test_actual_uncapped_relay_retry_is_one_successful_logical_call(tmp_path):
    from agent_company.runtime.relay import ProviderConfig, TeamRelay
    from agent_company.runtime.team import WorkerRegistry
    from test_team import request, workspace

    attempts = []

    def upstream(_payload):
        attempts.append(None)
        if len(attempts) == 1:
            return 503, {"error": {"message": "temporary unavailable"}}
        return 200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"},
                                   "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

    output = tmp_path / "runtime"
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", retry_backoff_seconds=0),
                      WorkerRegistry(workspace(tmp_path)), "", output / "relay", 30, upstream,
                      max_model_calls=None)
    assert relay.handle(request())[0] == 200
    assert "logical_model_call_admitted" not in (output / "relay/events.jsonl").read_text()
    write(output / "summary.json", {**relay.summary(), "stop_reason": "native_cli_completed",
                                    "execution": {"returncode": 0, "timed_out": False}})
    result = runtime_call_outcomes(output)
    assert result["complete"] and result["passed"] and result["total_calls"] == 1
    assert result["calls"][0]["attempts"] == 2
    assert result["transient_failures"][0]["provider_http_status"] == 503


@pytest.mark.parametrize("code", [-9, -15, 1, 0])
def test_timeout_label_does_not_hide_abnormal_process_exit(tmp_path, code):
    rows = pair(tmp_path, stop="task_timeout")
    path = Path(rows[0]["directory"]) / "run/pandoc/team_p0/team/summary.json"
    summary = json.loads(path.read_text())
    summary["execution"]["returncode"] = code
    write(path, summary)
    assert not s0_readiness(rows)["satisfied"]


def test_inferred_root_task_deadline_is_normal_budget_end(tmp_path):
    rows = pair(tmp_path, stop="task_timeout")
    for row in rows:
        path = Path(row["directory"]) / "run/pandoc/team_p0/team/summary.json"
        summary = json.loads(path.read_text())
        summary.update(execution={"returncode": 0, "timed_out": False}, deadline_rejections=["master"],
                       native_result={"explicit_error": True, "result": {
                           "result": "[API Error] task_deadline_exceeded: task budget exhausted"}})
        write(path, summary)
    assert s0_readiness(rows)["satisfied"]


def test_native_timeout_notice_matches_latest_request_before_worker_death(tmp_path):
    calls = [("master", [None])] * 10 + [("w1", ["provider_timeout"]), ("w2", [None])]
    terminals = failed_worker("w1", "Request timeout")
    terminals[0]["at"] = 11500
    rows = pair(tmp_path, calls=calls, workers=("w1", "w2"), terminals=terminals)
    result = s0_readiness(rows)
    assert not result["satisfied"]  # One of two workers is exactly the blocking boundary.
    assert result["details"]["s0_propose"]["details"]["workers"]["provider_failed_workers"] == ["w1"]


@pytest.mark.parametrize(("kind", "http", "transient"), [
    ("provider_error", 429, True), ("provider_error", 503, True), ("provider_error", 529, True),
    ("provider_error", 408, False), ("provider_error", 401, False), ("provider_error", 403, False),
    ("provider_error", 200, False), ("provider_timeout", None, False),
    ("provider_first_progress_timeout", 200, True), ("provider_progress_idle_timeout", 200, True),
    ("provider_transport_error", None, True), ("provider_transport_error", 401, False),
    ("provider_first_progress_timeout", 403, False), ("invalid_provider_response", 200, False),
    ("unrecognized_failure", None, False),
])
def test_only_user_class_a_failures_allow_preparation_retry(tmp_path, kind, http, transient):
    runtime(tmp_path, calls=[("master", [kind])])
    path = tmp_path / "relay/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1]["provider_http_status"] = http
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = runtime_call_outcomes(tmp_path)
    assert result["complete"] and not result["passed"]
    assert bool(result["transient_failures"]) is transient
    assert bool(result["hard_failures"]) is not transient


@pytest.mark.parametrize("fault", ["provider_error", "provider_transport_error", "provider_timeout"])
def test_health_reads_actual_native_replacement_lifecycle(tmp_path, fault):
    from agent_company.deploy.run_health import _worker_failures
    from test_network_recovery_integration import test_native_replacement_completion_retains_historical_network_failure

    # This existing integration drives the installed native CLI with an entirely
    # scripted provider. It writes the same uncapped records S0 will produce.
    test_native_replacement_completion_retains_historical_network_failure(tmp_path, fault)
    root = tmp_path / "run"
    result = runtime_call_outcomes(root)
    assert result["complete"] and not result["passed"]
    assert result["failed_calls"] == 1
    summary = json.loads((root / "summary.json").read_text())
    workers = _worker_failures(root, summary, result["calls"])
    assert workers["worker_count"] == 2
    assert workers["provider_failed_workers"] == ["flaky"]
    assert workers["unknown_failed_workers"] == []
    assert workers["at_least_half"] is True


def policy_run(**overrides):
    return {"mode": "propose", "voided_at_run_end": 0, "lifecycle_tail": {
        "terminal_events": 0, "invalid_lines": 0, "lines_read": 4,
        "partial_line_pending": False, "callback_errors": [], "max_consume_delay_seconds": None}, **overrides}


@pytest.mark.parametrize("fault", ["finalization_error", "callback_errors", "invalid_lines", "partial_line_pending", "missing_tail"])
def test_policy_finalization_and_tail_failure_blocks_smoke_and_s0_without_usage_error(tmp_path, fault):
    rows = pair(tmp_path)
    root = Path(rows[0]["directory"]) / "run/pandoc/team_p0/team"
    summary = json.loads((root / "summary.json").read_text())
    record = policy_run()
    if fault == "finalization_error":
        record["finalization_error"] = "PermissionError: cannot persist final state"
    elif fault == "missing_tail":
        record.pop("lifecycle_tail")
    else:
        record["lifecycle_tail"][fault] = {"callback_errors": ["OSError: cannot persist terminal"],
            "invalid_lines": 1, "partial_line_pending": True}[fault]
    summary["role_policy_run"] = record
    write(root / "summary.json", summary)
    # These finalization diagnostics are not emitted in relay usage.errors.
    assert json.loads((root / "relay/usage.json").read_text())["errors"] == []
    result = runtime_call_outcomes(root)
    assert result["complete"] and not result["passed"]
    if fault in {"finalization_error", "callback_errors"}:
        assert result["hard_failures"] and result["local_failures"]
    else:
        assert result["evidence_issues"]
    assert not s0_readiness(rows)["satisfied"]


def test_actual_lifecycle_callback_error_is_hard_even_with_successful_calls(tmp_path):
    from agent_company.policy.role_policy_runtime import LifecycleTail

    runtime(tmp_path)
    stderr = tmp_path / "qwen_stderr.txt"
    stderr.write_text('Agent Company lifecycle: {"event":"worker_terminal","worker":"w1","status":"failed"}\n')

    def cannot_persist(_worker, _source):
        raise OSError("terminal persistence failed")

    tail = LifecycleTail(stderr, cannot_persist)
    tail.stop()
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["role_policy_run"] = policy_run(lifecycle_tail=tail.summary())
    write(tmp_path / "summary.json", summary)
    result = runtime_call_outcomes(tmp_path)
    assert not result["passed"]
    assert "role_policy_lifecycle_callback_error" in result["hard_failures"]


@pytest.mark.parametrize("mode", ["team", "single"])
def test_off_or_single_does_not_require_policy_tail_or_invented_stopped_field(tmp_path, mode):
    runtime(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["execution_mode"] = mode
    assert "role_policy_run" not in summary
    write(tmp_path / "summary.json", summary)
    assert runtime_call_outcomes(tmp_path)["passed"]


def test_healthy_policy_tail_has_no_stopped_field_and_passes(tmp_path):
    runtime(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["role_policy_run"] = policy_run()
    assert "stopped" not in summary["role_policy_run"]["lifecycle_tail"]
    write(tmp_path / "summary.json", summary)
    assert runtime_call_outcomes(tmp_path)["passed"]


def test_active_policy_without_final_policy_record_is_unknown(tmp_path):
    runtime(tmp_path)
    usage = json.loads((tmp_path / "relay/usage.json").read_text())
    usage["role_policy"] = {"mode": "global", "counts": {}}
    write(tmp_path / "relay/usage.json", usage)
    result = runtime_call_outcomes(tmp_path)
    assert not result["passed"] and "role_policy_run_missing" in result["evidence_issues"]
