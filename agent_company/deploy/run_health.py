"""Read existing runtime evidence for the formal launcher's user-approved gates.

Retries belong to one logical call (role, source_request). A replaced worker's
new request is a different call. These readers never retry, cancel or mutate a
run, and do not use policy U labels or S0 diagnostic/extrapolation thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_company.programbench.total_selection import official_counts

PROVIDER_FAILURES = {
    "provider_error", "provider_transport_error", "provider_timeout",
    "provider_first_progress_timeout", "provider_progress_idle_timeout",
}
LOCAL_FAILURES = {"relay_execution_error", "evidence_write_failed", "provider_client_error"}
NORMAL_STOPS = {"native_cli_completed", "model_call_limit_exceeded", "task_timeout"}


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {Path(path).name}")
    return value


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _usage_known(value):
    return (isinstance(value, dict)
            and all(_integer(value.get(key)) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            and value["total_tokens"] == value["prompt_tokens"] + value["completion_tokens"])


def _policy_completion(summary, usage):
    """Read final policy persistence/tail diagnostics kept outside usage.errors."""
    run = summary.get("role_policy_run")
    active = (usage.get("role_policy") or {}).get("mode") in {"global", "master", "propose"}
    if run is None:
        return [], ["role_policy_run_missing"] if active else []
    if not isinstance(run, dict):
        return [], ["role_policy_run_unknown"]
    failures = [name for name, key in (("role_policy_finalization_error", "finalization_error"),
                                       ("rule_finalization_error", "rule_finalization_error")) if run.get(key)]
    tail = run.get("lifecycle_tail")
    if not isinstance(tail, dict):
        return failures, ["role_policy_lifecycle_tail_unknown"]
    issues = []
    callbacks = tail.get("callback_errors")
    if not isinstance(callbacks, list):
        issues.append("role_policy_callback_errors_unknown")
    elif callbacks:
        failures.append("role_policy_lifecycle_callback_error")
    if not _integer(tail.get("invalid_lines")) or tail["invalid_lines"] != 0:
        issues.append("role_policy_lifecycle_invalid_lines")
    if tail.get("partial_line_pending") is not False:
        issues.append("role_policy_lifecycle_partial_line_unknown")
    # LifecycleTail.summary has no `stopped` field. Do not require an invented
    # default or mistake off/Single's absent policy-run object for a failed tail.
    return failures, issues


def runtime_call_outcomes(runtime_dir):
    """Final logical-call outcomes, including retries and late delivery failures.

    ``complete`` means the existing attempt records and final counters
    agree. ``passed`` additionally requires normal CLI completion, every logical
    call successful and usage on each final attempt. An earlier transient attempt
    may have unknown usage without erasing its later success (cost accounting
    must still account for that unknown attempt separately).
    """
    root = Path(runtime_dir)
    result = {"complete": False, "passed": False, "reasons": [], "total_calls": None,
              "failed_calls": None, "failed_fraction": None, "all_final_usage": None,
              "calls": [], "local_failures": [], "evidence_issues": [],
              "hard_failures": [], "transient_failures": []}
    reasons = result["reasons"]
    try:
        summary = _object(root / "summary.json")
        usage = _object(root / "relay/usage.json")
        policy_failures, policy_issues = _policy_completion(summary, usage)
        result["evidence_issues"].extend(policy_issues)
        reasons.extend(policy_issues)
        events = []
        for line in (root / "relay/events.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Non-object event")
                events.append(event)
        admitted, starts, finishes, ended = {}, {}, {}, {}
        for event in events:
            kind = event.get("event")
            if kind not in {"logical_model_call_admitted", "provider_attempt_started",
                            "provider_attempt_finished", "native_request_ended"}:
                continue
            role = event.get("role")
            source = event.get("logical_model_call" if kind == "logical_model_call_admitted" else "source_request")
            if not isinstance(role, str) or not role or not _integer(source, 1):
                raise ValueError("Missing logical-call identity")
            key = (role, source)
            if kind == "logical_model_call_admitted":
                if key in admitted:
                    raise ValueError("Duplicate logical-call admission")
                admitted[key] = event
            elif kind == "native_request_ended":
                ended[key] = event
            else:
                identifier = event.get("attempt")
                if not _integer(identifier, 1) or not _integer(event.get("retry_index")):
                    raise ValueError("Missing attempt identity")
                target = starts if kind == "provider_attempt_started" else finishes
                if identifier in target:
                    raise ValueError("Duplicate attempt event")
                target[identifier] = event
        if set(starts) != set(finishes):
            raise ValueError("Unfinished provider attempts")
        # Uncapped formal runs do not emit logical_model_call_admitted. The
        # source_request carried by every attempt/terminal event is authoritative;
        # admission events, when enabled by a cap, are additional cross-checks.
        identities = {(event["role"], event["source_request"]) for event in starts.values()} | set(ended)
        if not identities or len({source for _, source in identities}) != len(identities):
            raise ValueError("Missing or nonunique logical-call IDs")
        if admitted and set(admitted) != identities:
            raise ValueError("Admissions disagree with logical-call identities")
        groups = {key: [] for key in identities}
        for identifier, start in starts.items():
            finish = finishes[identifier]
            if any(start.get(key) != finish.get(key) for key in ("role", "source_request", "retry_index", "purpose")):
                raise ValueError("Started and finished attempt identities disagree")
            key = (start["role"], start["source_request"])
            groups[key].append(finish)
        for saved in (summary, usage):
            if (saved.get("logical_model_calls") != len(identities)
                    or saved.get("provider_attempts") != len(starts)
                    or saved.get("completed_attempts") != len(finishes)
                    or saved.get("active_requests") != 0 or saved.get("inflight_attempts") != 0):
                raise ValueError("Final relay counters disagree or are not quiescent")
        local = [record for record in [*finishes.values(), *summary.get("errors", []), *usage.get("errors", [])]
                 if record.get("error_kind") in LOCAL_FAILURES]
        role_policy = usage.get("role_policy") or {}
        if (role_policy.get("counts") or {}).get("evidence_failures", 0):
            local.append({"error_kind": "evidence_write_failed"})
        local.extend({"error_kind": kind} for kind in policy_failures)
        result["local_failures"] = sorted({record["error_kind"] for record in local})
        result["hard_failures"].extend(result["local_failures"])
        for record in finishes.values():
            kind = record.get("error_kind")
            http = record.get("provider_http_status")
            temporary_http = (http == 429 or (_integer(http) and 500 <= http < 600))
            transient = (kind in (PROVIDER_FAILURES - {"provider_timeout"})
                         and (temporary_http if kind == "provider_error" else http in {None, 200} or temporary_http))
            if transient:
                result["transient_failures"].append({key: record.get(key) for key in
                    ("role", "source_request", "attempt", "error_kind", "provider_http_status")})
            elif kind and kind not in LOCAL_FAILURES | {"tool_response_rejected", "task_deadline_exceeded"}:
                result["hard_failures"].append(kind)
        for (role, source), attempts in sorted(groups.items(), key=lambda item: item[0][1]):
            ordered = sorted(attempts, key=lambda attempt: attempt["retry_index"])
            if [attempt["retry_index"] for attempt in ordered] != list(range(len(ordered))):
                raise ValueError("Missing or duplicate retry index")
            terminal = ended.get((role, source))
            if not ordered and not terminal:
                raise ValueError("Logical call has neither attempt nor terminal record")
            final = ordered[-1] if ordered else {"attempt": None}
            # Native tool feedback is a delivered model response, not a failed
            # transport. A late policy commit cancellation is not delivered.
            succeeded = (final.get("provider_http_status") == 200 and final.get("relay_http_status") == 200
                         and final.get("error_kind") in {None, "tool_response_rejected"}
                         and (not final.get("error") or final.get("error_kind") == "tool_response_rejected")
                         and not terminal)
            result["calls"].append({"role": role, "source_request": source, "attempt": final["attempt"],
                "succeeded": succeeded, "usage_known": _usage_known(final.get("usage")),
                "error_kind": terminal.get("error_kind") if terminal else final.get("error_kind"),
                "error": terminal.get("error") if terminal else final.get("error"),
                "finished_at": terminal.get("time") if terminal else final.get("time"),
                "attempts": len(ordered)})
        failed = sum(not call["succeeded"] for call in result["calls"])
        known = all(call["usage_known"] for call in result["calls"])
        result.update(complete=True, total_calls=len(identities), failed_calls=failed,
                      failed_fraction=failed / len(identities), all_final_usage=known)
        if local:
            reasons.append("local_runtime_or_evidence_failure")
        execution = summary.get("execution") or {}
        if (summary.get("stop_reason") != "native_cli_completed" or execution.get("returncode") != 0
                or execution.get("timed_out") is not False):
            reasons.append("runtime_not_normally_completed")
        if failed:
            reasons.append("logical_calls_finally_failed")
        if not known:
            reasons.append("final_usage_unknown")
        if any(call["succeeded"] and not call["usage_known"] for call in result["calls"]):
            result["hard_failures"].append("successful_call_usage_unknown")
        if (execution.get("returncode") != 0 or execution.get("timed_out") is not False
                or summary.get("stop_reason") not in {"native_cli_completed", "cli_error"}):
            result["hard_failures"].append("runtime_not_normally_completed")
        elif summary.get("stop_reason") == "cli_error":
            root_role = "single" if summary.get("execution_mode") == "single" else "master"
            if not any(call["role"] == root_role and call["error_kind"] in PROVIDER_FAILURES
                       for call in result["calls"] if not call["succeeded"]):
                result["hard_failures"].append("unexplained_cli_error")
        result["hard_failures"] = sorted(set(result["hard_failures"]))
        result["passed"] = not reasons and not result["hard_failures"]
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        issue = f"logical_call_evidence_unknown:{type(exc).__name__}:{exc}"
        reasons.append(issue)
        result["evidence_issues"].append(issue)
    return result


def _worker_failures(runtime_dir, summary, calls):
    """Bind native failed-worker notices to that worker's final provider error."""
    prefix = "Agent Company lifecycle: "
    events = []
    for line in (Path(runtime_dir) / "qwen_stderr.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            event = json.loads(line[len(prefix):])
            if not isinstance(event, dict):
                raise ValueError("Invalid lifecycle record")
            events.append(event)
    if not any(event.get("event") == "compatibility_loaded" for event in events):
        raise ValueError("Native lifecycle unavailable")
    registry = summary.get("workers")
    if not isinstance(registry, dict):
        raise ValueError("Worker inventory unavailable")
    observed = {call["role"] for call in calls if call["role"] not in {"master", "single"}}
    observed.update(event["worker"] for event in events if event.get("event") == "worker_terminal")
    if observed != set(registry):
        raise ValueError("Registered and observed worker identities differ")
    provider_deaths, unknown_deaths = set(), set()
    for position, event in enumerate(events):
        if event.get("event") != "worker_terminal" or event.get("status") != "failed" or event.get("user_cancelled"):
            continue
        worker = event["worker"]
        notices = [entry for entry in events[position + 1:]
                   if entry.get("event") == "failure_notice_queued" and entry.get("worker") == worker]
        notice = notices[0] if notices else None
        failed = [call for call in calls if call["role"] == worker and not call["succeeded"]]
        # Binding the actual error text avoids attributing an unrelated later
        # native failure to any historical provider incident in this role.
        matches = [call for call in failed if notice and isinstance(call["error"], str)
                   and call["error"] and call["error"] in str(notice.get("error", ""))]
        if any(call["error_kind"] in PROVIDER_FAILURES for call in matches):
            provider_deaths.add(worker)
        elif (notice and "Request timeout" in str(notice.get("error", "")) and failed):
            # Qwen formats provider_timeout as its fixed "Request timeout"
            # diagnostic (native network-recovery integration pins this). Bind
            # to the last request before this terminal timestamp, not an old
            # provider incident or a later retry/replacement.
            terminal_at = event.get("at")
            before = [call for call in calls if call["role"] == worker
                      and isinstance(terminal_at, (int, float))
                      and isinstance(call["finished_at"], (int, float))
                      and call["finished_at"] * 1000 <= terminal_at]
            latest = max(before, key=lambda call: call["source_request"]) if before else None
            if latest and not latest["succeeded"] and latest["error_kind"] == "provider_timeout":
                provider_deaths.add(worker)
            else:
                unknown_deaths.add(worker)
        elif not matches:
            unknown_deaths.add(worker)
    return {"worker_count": len(observed), "provider_failed_workers": sorted(provider_deaths),
            "unknown_failed_workers": sorted(unknown_deaths),
            "at_least_half": bool(observed) and len(provider_deaths) * 2 >= len(observed)}


def _s0_row_health(row):
    reasons, details = [], {}
    if (row.get("terminal") is not True or row.get("pid_alive") is not False
            or row.get("refused_before_output")):
        return {"satisfied": False, "reasons": ["s0_not_completed"], "details": details}
    try:
        progress = row["progress"]
        completed = progress.get("rows")
        if (progress.get("state") != "completed" or progress.get("running") != []
                or progress.get("pending") != [] or progress.get("cleanup_attention") != []
                or not isinstance(completed, list) or len(completed) != 1):
            raise ValueError("S0 progress is not a completed single arm")
        entry = completed[0]
        root = Path(row["directory"]) / "run"
        trial = root / entry["task_id"] / entry["arm_id"]
        if not trial.resolve().is_relative_to(root.resolve()):
            raise ValueError("Trial outside run")
        result = _object(trial / "task_result.json")
        metrics = result["metrics"]
        runtime = trial / "team"
        summary = _object(runtime / "summary.json")
        stop = summary.get("stop_reason")
        execution = summary.get("execution") or {}
        details.update(trial=str(trial), runtime_stop_reason=stop, official_score=result.get("score"))
        # The runner currently labels a wall-clock budget expiry status=failed,
        # even when the official verifier scored its delivery. That normal task
        # deadline is not an abnormal process exit under the user's S0 rule.
        code, timed_out = execution.get("returncode"), execution.get("timed_out")
        native = summary.get("native_result") or {}
        native_text = str((native.get("result") or {}).get("result", ""))
        deadline_exit = ((timed_out is True and code in {124, 137})
                         or (code == 0 and timed_out is False and native.get("explicit_error") is True
                             and "task_deadline_exceeded:" in native_text
                             and ("master" in summary.get("deadline_rejections", [])
                                  or any(record.get("role") == "master"
                                         and record.get("error_kind") == "task_deadline_exceeded"
                                         for record in summary.get("errors", [])))))
        normal = (stop in NORMAL_STOPS and type(code) is int
                  and (deadline_exit if stop == "task_timeout" else code == 0 and timed_out is False)
                  and not execution.get("oom_killed") and not summary.get("launcher_error"))
        if not normal or metrics.get("failures") != []:
            reasons.append("s0_abnormal_execution")
        if result.get("status") not in {"completed", "failed"} or (result.get("status") == "failed" and stop != "task_timeout"):
            reasons.append("s0_whole_run_failed")
        if official_counts(result) is None:
            reasons.append("s0_no_official_score")
        if (metrics.get("inference_cleanup_confirmed") is not True
                or entry.get("inference_cleanup_confirmed") is not True):
            reasons.append("s0_cleanup_unconfirmed")
        outcomes = runtime_call_outcomes(runtime)
        details["logical_calls"] = outcomes
        if outcomes["evidence_issues"]:
            reasons.append("s0_runtime_evidence_unknown")
        if not outcomes["complete"]:
            reasons.append("s0_logical_calls_unknown")
        elif outcomes["failed_calls"] * 5 > outcomes["total_calls"]:
            reasons.append("s0_final_call_failures_over_20_percent")
        if outcomes["local_failures"] or "successful_call_usage_unknown" in outcomes["hard_failures"]:
            reasons.append("s0_local_runtime_or_evidence_failure")
        workers = _worker_failures(runtime, summary, outcomes["calls"])
        details["workers"] = workers
        if workers["unknown_failed_workers"]:
            reasons.append("s0_worker_failure_cause_unknown")
        if workers["at_least_half"]:
            reasons.append("s0_provider_worker_deaths_at_least_half")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        reasons.append(f"s0_evidence_unknown:{type(exc).__name__}:{exc}")
    return {"satisfied": not reasons, "reasons": reasons, "details": details}


def s0_readiness(rows):
    """Both named S0 runs must be terminal with no user-defined hard failure."""
    details, reasons = {}, []
    for step in ("s0_propose", "s0_global"):
        matching = [row for row in rows if row.get("launch", {}).get("step") == step]
        if len(matching) != 1:
            details[step] = {"satisfied": False, "reasons": ["missing_or_duplicate_s0"], "details": {}}
        else:
            details[step] = _s0_row_health(matching[0])
        reasons.extend(f"{step}:{reason}" for reason in details[step]["reasons"])
    return {"satisfied": not reasons, "reasons": reasons, "details": details}
