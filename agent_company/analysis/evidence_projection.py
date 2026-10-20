"""Project existing ProgramBench Team evidence without exposing model reasoning.

Shared by scripts/project_programbench_ac_evidence.py (CLI) and the study runners.

This module uses only the Python standard library. It never imports a runner,
launches a model, writes source data, reads request history, or reads test bodies.
run_dir is one trial directory containing task_result.json and team/relay/; from
team/qwen_stderr.txt only "Agent Company lifecycle:" metadata lines are read, and
only allowlisted counts, states, identifiers and times from them are kept.
Collaboration text is an outward tool argument, not an assistant reasoning block.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re

from agent_company.analysis.score_layers import load_score_layers


TEXT_LIMIT = 8000
TOKENS = ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "total_tokens")
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
RESPONSE_NAME = re.compile(r"attempt_(\d+)_([A-Za-z][A-Za-z0-9_-]{0,63})_qwen_response\.json\Z")
COLLABORATION = {"agent", "task_create", "task_update", "send_message", "request_shutdown"}
ARGUMENT_TEXT = {
    "name", "recipient", "to", "task_id", "taskId", "owner", "status", "type",
    "subagent_type", "prompt", "description", "content", "message", "summary",
    "subject", "activeForm", "reason", "request_id", "requestId",
}
SCHEMA_VERSION = 5
LIFECYCLE_PREFIX = "Agent Company lifecycle: "
DELIVERY_MODES = {"native_idle", "between_tool_calls"}
MESSAGE_ID = re.compile(r"msg-[1-9][0-9]{0,8}\Z")
SENDER = re.compile(r"[A-Za-z][A-Za-z0-9_:-]{0,63}\Z")
WORKER_STATUS = {"running", "idle", "completed", "failed", "cancelled", "paused", "unknown"}
TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")
ISO_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z\Z")
EDIT_TOOLS = {"edit", "write_file"}
SLEEP_COMMAND = re.compile(r"(?:^|[;&|(\s])sleep\s+[0-9]")
MESSAGE_ROW_LIMIT = 400

# Mirrors the export in programbench_workspace.py (_EXPORT) and the runner's
# phases; tests pin it to that code. Mechanics only, not when to stop working.
SUBMISSION_CONTRACT = {
    "deliverable_root": "/workspace/solution",
    "facts": [
        "When the agent phase ends (normal finish or the task deadline), every agent is stopped before export.",
        "The deliverable is the regular files and directories under /workspace/solution at that moment.",
        "Any .git directory and the top-level executable and reference paths are not delivered; "
        "Git history is not part of the deliverable.",
        "A symbolic link or special file under /workspace/solution makes the export fail.",
        "Worker worktrees are separate; only what is under /workspace/solution is delivered.",
        "The official evaluator rebuilds ./executable with compile.sh in a fresh environment.",
    ],
    "scope": "Delivery mechanics only. Build duration and when to stop changing the deliverable "
             "are task- and implementation-specific and are not stated here.",
}

# Qwen 0.23.1 runtime mechanics (docs/stage1_master_loop_guard_fix_plan_20261002.md
# section 2); tests pin the numbers to the installed CLI. Whether a run told its
# agents is in runtime_rules.effective.runtime_disclosure, never assumed.
RUNTIME_RULES = {
    "loop_guards": [
        "Always on for the master, workers and Single; the CLI's default skipLoopDetection does not disable them.",
        "The fifth consecutive call of one tool with identical arguments is not executed and stops the turn. "
        "read_file counts even when the file changed between reads; only task_list is exempt while its result "
        "keeps changing.",
        "Only with effective.tool_call_cap = native_default: a turn is also stopped once it has more than 100 "
        "tool calls and one identical call occurs at least 6 times in it (not necessarily in a row), or once it "
        "has more than 1000 tool calls. With disabled there is no per-turn tool-call cap for any role.",
        "Eight consecutive tool calls that only inspect Git through run_shell_command (git status, git ls-files, "
        "git diff without a path) stop the turn.",
    ],
    "turn_boundaries": "Master: counts restart whenever a teammate message or check-in is delivered to it, "
                       "including with tool results mid-turn. Worker: one work turn, from a message or task "
                       "until it answers without tool calls, which can last hours; messages delivered between "
                       "tool calls do not restart it. Single: normally the whole task.",
    "consequences": {
        "master": "The whole CLI run ends and exit cleanup stops every teammate, except the case covered by "
                  "effective.leader_loop_recovery.",
        "worker": "That work turn ends with an error. If a message is already queued for the worker, it "
                  "continues with it and a turn that then ends normally clears the error; otherwise the worker "
                  "is marked failed and the master receives a runtime failure notice.",
        "single": "The run ends.",
    },
    "master_waiting": "A master turn without tool calls waits, without model calls, while a teammate is working; "
                      "teammate messages, automatic end-of-turn reports and check-ins wake it. With no working "
                      "teammate and no queued message, it ends the run.",
    "scope": "Runtime mechanics only, not a recommendation for checkpoints or polling.",
}

DEFINITIONS = {
    "attempt_started_at": "Relay wall-clock seconds when that provider request started. It is not when a "
                          "tool call was issued or executed.",
    "lifecycle_at_ms": "Container wall-clock milliseconds when the runtime recorded the event.",
    "checkin_snapshot_times": "sample_started_at is when sampling began, and every *_ago_s value is measured "
                              "from it (last_commit_ago_s uses Git committer time); at is when sampling finished. "
                              "Sources are read at different moments in between.",
    "checkin_snapshot_coverage": "Snapshots exist only for check-ins that were sent. Status refreshes are not "
                                 "kept, sending pauses while workers negotiate shutdown, and there is no final "
                                 "snapshot, so the closing phase may have no record.",
    "message_observed": "The delivery's unique marker appeared in the recipient's coding request. It is "
                        "not provider acceptance and not evidence the recipient acted on it.",
    "receipt_rate": "Observed message_ids / queued message_ids. A message drained twice (requeued) "
                    "counts once. Runs without delivery records are unknown, never zero.",
    "commits_not_in_delivery_history": "Git ancestry only. Cherry-picks and squashes make delivered work "
                                       "look absent; the deliverable is judged by its files, not commit SHAs.",
    "delivery_commits": "Commits in the delivery repository since launch. Uncommitted changes under "
                        "/workspace/solution are delivered too and are not counted here.",
    "final_export": "Regular files and source bytes in the exported deliverable when the agent phase ended, "
                    "uncommitted files included. Final state only: not when integration happened, not "
                    "whether the code works, and not a history of file changes. null when not recorded.",
    "sleep_commands_issued": "Proposed shell commands containing sleep. Not executed time, and not by "
                             "itself wasted time.",
    "leader_loop": "Master loop-guard stops and scope-A recoveries (the fifth identical read_file of the "
                   "runtime status file as the only call in its response, while a teammate was running). "
                   "Recorded only in runs with "
                   "leader_loop_recovery enabled; otherwise null, not zero.",
    "teammate_limit": "The teammate-count setting explicitly recorded by compatibility_loaded events. "
                      "unlimited removes only the count cap, not shared CPU, memory or the task deadline. "
                      "Missing records are unknown; conflicting or invalid records are conflicting. "
                      "Neither the current harness nor a native version establishes an older run's setting.",
    "last_edit_proposal_attempt_started_at": "attempt_started_at of the latest response proposing an edit "
                                             "or write_file call; the edit may not have run or been kept.",
    "source_request": "Run-wide logical request number assigned when the relay admitted a request. Retries "
                      "of one request share it, and so do its injected policy snapshot and policy batch.",
    "role_policies_status": "not_instrumented: the run predates role-policy records. off: recorded as off. "
                            "on: role-policy state was recorded; zero activity then means it was not used.",
    "role_policy_committed": "The relay validated the whole response and persisted the change. It does not "
                             "mean the native tool ran, the message arrived, or any model read it.",
    "role_policy_injected": "The request snapshot included this state. Injection is not reading, "
                            "understanding or compliance; an unadmitted request was never sent upstream.",
    "first_carry_latency_s": "Relay seconds from commit to the start of the first provider attempt that "
                             "carried the version or proposal (upstream_started). Not a reading time.",
    "role_policy_version_of_call": "Versions carried by the request that produced the call: the calling "
                                   "worker's own slot version and the global C version. A version committed "
                                   "during that request is not used. messages_by_version groups by the C "
                                   "version in global mode and by the slot version otherwise, and leaves out "
                                   "calls the relay rejected before execution.",
    "policy_control": "send_message whose first line is a role-policy control header. These are counted apart "
                      "from task communication.",
    "notification_unknown": "A committed control message with no native tool result in a later request. "
                            "Not a failure and not a delivery.",
    "prompt_cache_reported": "Attempts whose raw provider usage carried a cache field. Missing fields are "
                             "unknown and excluded from the cached share, never counted as zero hits.",
}

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", re.I),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{8,})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"(?i)\b(?:[A-Z0-9_]*API[_-]?KEY|[A-Z0-9_]*(?:ACCESS|AUTH|REFRESH|SESSION)[_-]?TOKEN|TOKEN|PASSWORD|PASSWD|SECRET|AUTHORIZATION)\b[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"),
    re.compile(r"(?i)(?<=://)[^/\s:@]+:[^/\s@]+@"),
    re.compile(r"(?i)(?:[?&](?:api[_-]?key|token|access_token|secret|password)=)[^&#\s]+"),
)


def redact(text: str) -> str:
    """Redact credential-shaped strings; this is not a claim of semantic DLP."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def text_leaf(value: object, limit: int = TEXT_LIMIT) -> dict | None:
    if not isinstance(value, str):
        return None
    cleaned = redact(value)
    return {"text": cleaned[:limit], "truncated": len(cleaned) > limit,
            "original_characters": len(value), "redacted": cleaned != value}


def number(value: object, *, integer: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if integer and not isinstance(value, int):
        return None
    try:
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def scalar_fields(source: object, *, strings=(), numbers=(), booleans=()) -> dict:
    """Copy only explicitly named scalar leaves, never a source parent object."""
    source = mapping(source)
    out = {}
    for key in strings:
        if isinstance(source.get(key), str):
            out[key] = text_leaf(source[key])
    for key in numbers:
        if key in source:
            out[key] = number(source[key])
    for key in booleans:
        if isinstance(source.get(key), bool):
            out[key] = source[key]
    return out


def token_projection(value: object) -> dict:
    source = mapping(value)
    return {key: number(source.get(key), integer=True) for key in TOKENS}


def valid_usage(tokens: dict) -> bool:
    return (all(value is not None for value in tokens.values())
            and tokens["total_tokens"] == tokens["prompt_tokens"] + tokens["completion_tokens"]
            and tokens["cached_prompt_tokens"] <= tokens["prompt_tokens"])


def safe_file(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
        return (path.is_file() and path.resolve().is_relative_to(root)
                and not any(root.joinpath(*relative.parts[:i]).is_symlink()
                            for i in range(1, len(relative.parts) + 1)))
    except (OSError, ValueError):
        return False


def task_projection(raw: dict) -> dict:
    metrics = mapping(raw.get("metrics"))
    runtime = mapping(metrics.get("team"))
    task = mapping(metrics.get("task"))
    out = scalar_fields(raw, strings=("task_id", "repo_id", "status", "execution_mode"),
                        numbers=("score", "elapsed_seconds"))
    out["task"] = scalar_fields(task, strings=("task_id", "repository_id"),
                                numbers=("agent_timeout_sec", "cpus", "memory_mb"))
    out["usage"] = token_projection(raw.get("usage"))
    out["runtime"] = scalar_fields(runtime, strings=("stop_reason", "execution_mode"),
        numbers=("provider_attempts", "completed_attempts", "logical_model_calls", "max_model_calls",
                 "model_call_budget_rejections", "worker_count", "elapsed_seconds"),
        booleans=("usage_complete", "cost_complete", "policy_applied"))
    usage_source = mapping(raw.get("usage"))
    out["runtime"]["usage_complete"] = (runtime.get("usage_complete") is True
        and valid_usage(out["usage"]) and not usage_source.get("unknown_usage_attempts")
        and not usage_source.get("calls_missing_usage"))
    out["runtime"]["execution"] = scalar_fields(runtime.get("execution"),
        numbers=("elapsed_seconds",),
        booleans=("timed_out", "oom_killed", "oom_killed_observed"))
    # POSIX signal exits use negative return codes; they are not missing data.
    returncode = mapping(runtime.get("execution")).get("returncode")
    if isinstance(returncode, int) and not isinstance(returncode, bool):
        out["runtime"]["execution"]["returncode"] = returncode
    native = mapping(runtime.get("native_result"))
    out["runtime"]["native"] = scalar_fields(native, booleans=("explicit_error", "missing_result"))
    out["runtime"]["native"].update(scalar_fields(native.get("result"),
        strings=("type", "subtype"), booleans=("is_error",)))
    out["runtime"]["logical_calls_by_role"] = {
        redact(role): count for role, value in mapping(runtime.get("logical_calls_by_role")).items()
        if isinstance(role, str) and IDENTIFIER.fullmatch(role)
        and (count := number(value, integer=True)) is not None}
    out["official_result"] = scalar_fields(metrics.get("verifier"),
        strings=("status", "reward_source", "error_code", "evaluator_exception_type"),
        numbers=("reward", "official_test_fraction", "passed_tests", "valid_tests", "warnings_count",
                 "verifier_elapsed_seconds"), booleans=("resolved", "almost"))
    out["resources"] = scalar_fields(metrics.get("resource_config"),
        strings=("memory_limit_scope", "model_call_limit_scope"),
        numbers=("agent_timeout_seconds", "cpus", "memory_mb", "max_model_calls", "evaluator_cpus"))
    out["cleanup"] = scalar_fields(metrics, booleans=("agent_termination_confirmed", "inference_cleanup_confirmed"))
    # The runner's record of the exported deliverable (uncommitted files included).
    # Final state only: it does not say when, or whether, integration happened.
    final_export = mapping(metrics.get("submission"))
    out["final_export"] = ({**scalar_fields(final_export, numbers=("files", "source_bytes")),
                            "scope": "final_export_only"} if final_export else None)
    # Failure text and diagnostics are not part of the evidence interface.
    out["failures"] = [scalar_fields(item, strings=("phase", "error_type"))
                       for item in raw_failures if isinstance(item, dict)] if isinstance(
                           raw_failures := metrics.get("failures"), list) else []
    return out


def shell_category(arguments: dict) -> str:
    command = arguments.get("command")
    if not isinstance(command, str):
        return "unknown_shell"
    if re.search(r"\b(?:pytest|vitest|jest|tsc|eslint)\b|\b(?:cargo|go|npm|pnpm|yarn)\s+(?:test|check|build)\b|(?:^|[/\s])compile\.sh\b", command):
        return "test_build_lint"
    if re.search(r"\bgit\b", command):
        return "git"
    if "/opt/ac-programbench-reference/executable" in command:
        return "reference_query"
    return "other_shell"


def read_lifecycle(path: Path, root: Path) -> tuple[list[dict] | None, Counter]:
    """Agent Company lifecycle events; every other stderr line is skipped unread."""
    stats = Counter()
    if not safe_file(path, root):
        return None, stats
    events = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith(LIFECYCLE_PREFIX):
                continue
            try:
                event = json.loads(line[len(LIFECYCLE_PREFIX):])
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise ValueError("not an event")
            except (ValueError, TypeError):
                stats["invalid_lifecycle_lines"] += 1
                continue
            stats["lifecycle_events"] += 1
            events.append(event)
    return events, stats


def _name(value, pattern=IDENTIFIER) -> str | None:
    return redact(value) if isinstance(value, str) and pattern.fullmatch(value) else None


def _summary(values: list) -> dict:
    known = sorted(value for value in values if value is not None)
    middle = len(known) // 2
    median = (None if not known else known[middle] if len(known) % 2
              else (known[middle - 1] + known[middle]) / 2)
    return {"count": len(known), "median": median, "max": known[-1] if known else None}


def delivery_projection(events: list[dict] | None, issues: Counter) -> dict:
    """Per-message queue, drain and observation records (between_tool_calls runs only)."""
    loaded = [event.get("team_message_delivery") for event in events or ()
              if event.get("event") == "compatibility_loaded"]
    modes = {mode for mode in loaded if isinstance(mode, str)}
    mode = (next(iter(modes)) if len(modes) == 1 and modes <= DELIVERY_MODES
            else "conflicting" if len(modes) > 1 else "unknown")
    out = {"mode": mode, "status": "not_recorded", "receipt_rate": None}
    if mode != "between_tool_calls":
        out["reason"] = ("native_idle runs do not record per-message queue and receipt events"
                         if mode == "native_idle" else
                         "no delivery-mode record in this run (recorded since 2026-09-30)"
                         if mode == "unknown" else "conflicting delivery-mode records")
        return out
    rows = {}
    for event in events:
        kind = event.get("event")
        if kind == "team_message_queued":
            identifier = event.get("message_id")
            if not isinstance(identifier, str) or not MESSAGE_ID.fullmatch(identifier) or identifier in rows:
                issues["invalid_or_duplicate_queued_message"] += 1
                continue
            rows[identifier] = {"message_id": identifier, "to": _name(event.get("to")),
                                "from": _name(event.get("from"), SENDER),
                                "queued_at_ms": number(event.get("at")), "drained_modes": [],
                                "requeued": 0, "observed": None}
        elif kind in {"team_message_drained", "team_message_observed"}:
            items = event.get("items")
            for item in items if isinstance(items, list) else ():
                item = mapping(item)
                row = rows.get(item.get("message_id"))
                if row is None:
                    issues["delivery_event_without_queued_message"] += 1
                elif kind == "team_message_drained":
                    mode_name = event.get("mode")
                    row["drained_modes"].append(mode_name if mode_name in {"between_tool_calls", "idle_flush"}
                                                else "unknown")
                elif row["observed"] is None:
                    row["observed"] = {"at_ms": number(event.get("at")),
                                       "mode": item.get("mode") if item.get("mode") in {"between_tool_calls", "idle_flush"}
                                       else "unknown",
                                       "queue_to_observed_ms": number(item.get("waited_ms"))}
        elif kind == "teammate_inputs_requeued":
            identifiers = event.get("message_ids")
            for identifier in identifiers if isinstance(identifiers, list) else ():
                if identifier in rows:
                    rows[identifier]["requeued"] += 1
        elif kind == "team_message_unobservable":
            if event.get("message_id") in rows:
                rows[event["message_id"]]["unobservable"] = True
    observed = [row for row in rows.values() if row["observed"] is not None]
    recipients = {}
    for row in rows.values():
        counts = recipients.setdefault(row["to"] or "unknown", {"queued": 0, "observed": 0})
        counts["queued"] += 1
        counts["observed"] += row["observed"] is not None
    ordered = sorted(rows.values(), key=lambda row: int(row["message_id"][4:]))
    count = lambda name: sum(event.get("event") == name for event in events)  # noqa: E731
    out.update(
        status="recorded", queued_messages=len(rows), observed_messages=len(observed),
        never_observed_messages=len(rows) - len(observed),
        receipt_rate=len(observed) / len(rows) if rows else None,
        queue_to_observed_ms=_summary([row["observed"]["queue_to_observed_ms"] for row in observed]),
        by_recipient=dict(sorted(recipients.items())),
        task_notices={"queued": count("task_notice_queued"), "observed_events": count("task_notice_observed")},
        shutdown_notices={"drained": count("shutdown_notice_drained"), "observed": count("shutdown_notice_observed")},
        messages=ordered[:MESSAGE_ROW_LIMIT], messages_omitted=max(0, len(ordered) - MESSAGE_ROW_LIMIT))
    return out


def _snapshot(raw: dict) -> dict:
    """Allowlisted check-in sample: counts, states, names and tool names."""
    counts = lambda source, keys: {key: number(mapping(source).get(key)) for key in keys}  # noqa: E731
    delivery = raw.get("delivery")
    workers = []
    for worker in raw.get("workers") if isinstance(raw.get("workers"), list) else ():
        worker = mapping(worker)
        tools = worker.get("pending_tools")
        git = worker.get("git")
        workers.append({
            "name": _name(worker.get("name")),
            "status": worker.get("status") if worker.get("status") in WORKER_STATUS else "unknown",
            "shutdown_pending": worker.get("shutdown_pending") is True,
            **counts(worker, ("last_activity_ago_s", "queued_messages", "open_tasks")),
            "pending_tools": [name for name in tools if isinstance(name, str) and TOOL_NAME.fullmatch(name)]
                             if isinstance(tools, list) else None,
            "git": counts(git, ("commits_since_worker_creation", "commits_not_in_delivery_history",
                                "uncommitted_paths", "last_commit_ago_s")) if isinstance(git, dict) else None})
    unknown = raw.get("unknown")
    iso = lambda key: raw[key] if isinstance(raw.get(key), str) and ISO_TIME.fullmatch(raw[key]) else None  # noqa: E731
    return {"sample_started_at": iso("sample_started_at"), "at": iso("at"),
            **counts(raw, ("elapsed_s", "remaining_s", "unassigned_open_tasks")),
            "delivery": counts(delivery, ("commits_since_launch", "last_commit_ago_s"))
                        if isinstance(delivery, dict) else None,
            "workers": workers,
            "unknown_sources": len(unknown) if isinstance(unknown, list) else None}


def checkin_projection(events: list[dict] | None) -> dict:
    if events is None or not any(event.get("event") == "checkin_enabled" for event in events):
        return {"status": "not_enabled_or_not_recorded", "snapshots": []}
    sent = [event for event in events if event.get("event") == "checkin_sent"]
    snapshots = [_snapshot(event["snapshot"]) for event in sent if isinstance(event.get("snapshot"), dict)]
    skipped = Counter(event.get("reason") if isinstance(event.get("reason"), str)
                      and re.fullmatch(r"[a-z_]{1,40}", event["reason"]) else "unknown"
                      for event in events if event.get("event") == "checkin_skipped")
    return {"status": "recorded", "sent": len(sent),
            "observed": sum(event.get("event") == "checkin_observed" for event in events),
            "superseded": sum("supersedes_id" in event for event in sent),
            "errors": sum(event.get("event") == "checkin_error" for event in events),
            "skipped_reasons": dict(sorted(skipped.items())),
            # Coverage of sent check-ins only; see definitions.checkin_snapshot_coverage.
            "snapshots_status": ("recorded_for_sent_checkins" if snapshots and len(snapshots) == len(sent)
                                 else "none_sent" if not sent
                                 else "not_recorded_before_2026_09_30" if not snapshots else "partial"),
            "snapshots": snapshots}


def _setting(loaded: list[dict], name: str):
    values = {event.get(name) for event in loaded}
    if not loaded or values == {None}:
        return "unknown"  # Recorded since 2026-10-02; older runs are not assumed off.
    return values.pop() if len(values) == 1 and values <= {True, False} else "conflicting"


def runtime_rules_projection(events: list[dict] | None) -> dict:
    """Static runtime mechanics plus the settings and loop events this run recorded."""
    loaded = [event for event in events or () if event.get("event") == "compatibility_loaded"]
    effective = {name: _setting(loaded, name) for name in ("runtime_disclosure", "leader_loop_recovery")}
    caps = {event.get("tool_call_cap") for event in loaded}
    # Requested mode as recorded (since 2026-10-02); older runs stay unknown.
    effective["tool_call_cap"] = ("unknown" if not loaded or caps == {None}
                                  else caps.pop() if len(caps) == 1 and caps <= {"native_default", "disabled"}
                                  else "conflicting")
    # Do not infer an older run's cap from today's implementation or its native version.
    limits = [event.get("teammate_limit") for event in loaded]
    effective["teammate_limit"] = ("unknown" if not limits or all(value is None for value in limits)
                                   else "unlimited" if all(value == "unlimited" for value in limits)
                                   else limits[0] if all(type(value) is int and value > 0
                                                        and value == limits[0] for value in limits)
                                   else "conflicting")
    leader = None
    if effective["leader_loop_recovery"] is True:
        kind = lambda name: [event for event in events if event.get("event") == name]  # noqa: E731
        label = lambda value: (value if isinstance(value, str)  # noqa: E731
                               and re.fullmatch(r"[a-z_]{1,40}", value) else "unknown")
        resumed = kind("leader_loop_resumed")
        leader = {
            "guard_stops": dict(sorted(Counter(label(event.get("loop_type"))
                                               for event in kind("leader_loop_guard")).items())),
            "recovered": len(kind("leader_loop_recovered")),
            "not_recovered": dict(sorted(Counter(label(event.get("reason"))
                                                 for event in kind("leader_loop_not_recovered")).items())),
            "resumed": len(resumed), "resumed_by_checkin": sum(event.get("checkin") is True for event in resumed),
            "resume_wait_s": _summary([number(event.get("waited_s")) for event in resumed]),
            "notice_observed": len(kind("leader_loop_notice_observed")),
            "unresolved": len(kind("leader_loop_unresolved")),
        }
    return {**RUNTIME_RULES, "effective": effective, "leader_loop": leader}


def project_evidence(run_dir: str | Path, *, layer_rules: dict | None = None) -> dict:
    """Return JSON-safe training evidence from one existing Team trial."""
    root = Path(run_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("run-dir must be a trial directory")
    issues = Counter()
    report = {"schema_version": SCHEMA_VERSION, "scope": "one_current_team_attempt",
              "source_trial_name": text_leaf(root.name), "task_result": None,
              "submission_contract": SUBMISSION_CONTRACT, "runtime_rules": None, "definitions": DEFINITIONS,
              "collaboration_timeline": [], "event_timeline": [], "by_role": [],
              "delivery": None, "checkins": None, "prompt_cache": None, "role_policies": None,
              "coverage": {}, "notes": [
                  "Tool calls are delivered proposals, not proof of execution or message receipt.",
                  "Collaboration text is outward tool-argument evidence, not internal model reasoning.",
                  "Known tokens are lower bounds when usage is missing; missing usage is not zero.",
                  "Provider durations may overlap and are not worker idle time or task latency.",
                  "Non-numeric timestamps remain null; no timestamp or missing measurement is replaced with zero.",
                  "Counts and repeated actions alone do not establish waste, collaboration benefit, or causality.",
                  "Credential-shaped text is redacted heuristically; no credentials are intentionally read.",
                  "No requests, assistant prose, reasoning, stdout, shell output, source, or hidden tests are exposed.",
                  "Only allowlisted metadata from Agent Company lifecycle lines is read from stderr.",
                  "Field meanings and clocks are in definitions; read them before comparing times.",
                  "Role-policy injection, notification and approval are not evidence that a model read, "
                  "understood or followed a policy; message counts per version are descriptive only."]}
    result_path = root / "task_result.json"
    raw_result = {}
    if safe_file(result_path, root):
        try:
            raw_result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(raw_result, dict):
                raise ValueError("not an object")
            report["task_result"] = task_projection(raw_result)
        except (OSError, UnicodeError, ValueError, TypeError):
            issues["invalid_task_result"] += 1
    else:
        issues["missing_or_unsafe_task_result"] += 1
    task_result = report["task_result"] or {}
    raw_task = mapping(mapping(mapping(raw_result).get("metrics")).get("task"))
    layer_projection = load_score_layers(
        root, layer_rules, task_id=mapping(raw_result).get("task_id") or raw_task.get("task_id") or "",
        score=task_result.get("score"))
    report["score_layers"] = layer_projection["score_layers"]

    roles = {}
    def role_row(role):
        return roles.setdefault(role, {"role": redact(role), "started_attempts": 0, "finished_attempts": 0,
            "finished_attempts_missing_or_invalid_usage": 0, "known_tokens": dict.fromkeys(TOKENS, 0),
            "known_provider_seconds": 0, "finished_attempts_missing_time": 0,
            "delivered_response_count": 0, "expected_error_only_response_count": 0,
            "tool_calls": Counter(), "shell_categories": Counter(), "sleep_commands_issued": 0,
            "last_edit_proposal_attempt_started_at": None,
            "retry_scheduled_count": 0, "known_scheduled_retry_delay_seconds": 0,
            "provider_http_status_counts": Counter(), "error_kind_counts": Counter()})

    relay = root / "team" / "relay"
    events_path = relay / "events.jsonl"
    started, finished, event_seen, attempt_roles, retry_seen = set(), set(), set(), {}, set()
    failed_attempts = set()
    event_times = {}
    raw_events: list[dict] = []
    if safe_file(events_path, root):
        with events_path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("not an object")
                except (ValueError, TypeError):
                    issues["invalid_event_lines"] += 1
                    continue
                raw_events.append(event)
                kind, role = event.get("event"), event.get("role")
                if kind not in {"provider_attempt_started", "provider_attempt_finished", "provider_retry_scheduled"}:
                    continue
                if not isinstance(role, str) or not IDENTIFIER.fullmatch(role):
                    issues["invalid_event_role"] += 1
                    continue
                identifier = number(event.get("after_attempt" if kind == "provider_retry_scheduled" else "attempt"), integer=True)
                if identifier is None or identifier < 1:
                    issues["invalid_event_attempt"] += 1
                    continue
                if identifier in attempt_roles and attempt_roles[identifier] != role:
                    issues["conflicting_attempt_role"] += 1
                    continue
                attempt_roles[identifier] = role
                retry_index = number(event.get("retry_index"), integer=True)
                identity = (role, identifier, retry_index) if kind == "provider_retry_scheduled" else (kind, identifier)
                seen = retry_seen if kind == "provider_retry_scheduled" else event_seen
                if identity in seen:
                    issues["duplicate_retry_events" if kind == "provider_retry_scheduled" else "duplicate_attempt_events"] += 1
                    continue
                seen.add(identity)
                row = role_row(role)
                projected = {"event": kind, "role": redact(role), "path": "team/relay/events.jsonl", "line": line_number,
                             "attempt" if kind != "provider_retry_scheduled" else "after_attempt": identifier}
                for key in ("time", "started_at", "elapsed_seconds", "retry_index", "delay_seconds", "provider_retry_after_seconds",
                            "provider_http_status", "relay_http_status", "source_request"):
                    if key in event:
                        projected[key] = number(event[key])
                for key in ("purpose", "error_kind"):
                    if isinstance(event.get(key), str):
                        projected[key] = text_leaf(event[key], 200)
                if kind == "provider_attempt_started":
                    started.add(identifier)
                    row["started_attempts"] += 1
                    event_times[identifier] = number(event.get("time"))
                elif kind == "provider_attempt_finished":
                    finished.add(identifier)
                    row["finished_attempts"] += 1
                    tokens = token_projection(event.get("usage"))
                    projected["usage"] = tokens
                    projected["usage_valid"] = valid_usage(tokens)
                    if projected["usage_valid"]:
                        for key in TOKENS:
                            row["known_tokens"][key] += tokens[key]
                    else:
                        row["finished_attempts_missing_or_invalid_usage"] += 1
                    elapsed = number(event.get("elapsed_seconds"))
                    if elapsed is None:
                        row["finished_attempts_missing_time"] += 1
                    else:
                        row["known_provider_seconds"] += elapsed
                    status = number(event.get("provider_http_status"), integer=True)
                    row["provider_http_status_counts"][str(status) if status is not None else "unknown"] += 1
                    error = event.get("error_kind")
                    if isinstance(error, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", error):
                        failed_attempts.add(identifier)
                        row["error_kind_counts"][redact(error)] += 1
                    elif status is not None and status >= 300:
                        failed_attempts.add(identifier)
                else:
                    row["retry_scheduled_count"] += 1
                    delay = number(event.get("delay_seconds"))
                    if delay is None:
                        issues["retry_event_missing_delay"] += 1
                    else:
                        row["known_scheduled_retry_delay_seconds"] += delay
                report["event_timeline"].append(projected)
    else:
        issues["missing_or_unsafe_events"] += 1

    response_seen = set()
    candidates = []
    # Exact delivered-response suffix only; never requests or provider copies.
    for path in relay.glob("attempt_*_qwen_response.json"):
        match = RESPONSE_NAME.fullmatch(path.name)
        if match:
            candidates.append((int(match[1]), path.name, match[2], path))
    for identifier, _, role, path in sorted(candidates):
        if identifier < 1 or not safe_file(path, root):
            issues["unsafe_response_file"] += 1
            continue
        if identifier in response_seen:
            issues["duplicate_response_attempts"] += 1
            continue
        response_seen.add(identifier)
        if identifier in attempt_roles and attempt_roles[identifier] != role:
            issues["response_role_conflict"] += 1
            continue
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
            choices = mapping(response).get("choices")
            if (isinstance(response, dict) and not choices
                    and ("error" in response or identifier in failed_attempts)):
                # The relay deliberately delivers error objects after a failed
                # provider request. Absence of a completion here is not corruption.
                row = role_row(role)
                row["delivered_response_count"] += 1
                row["expected_error_only_response_count"] += 1
                continue
            if not isinstance(choices, list) or not choices:
                raise ValueError("missing first choice")
            # Do not copy/read assistant content or alternate completions.
            calls = mapping(mapping(choices[0]).get("message")).get("tool_calls", [])
            if not isinstance(calls, list):
                raise ValueError("invalid tool calls")
        except (OSError, UnicodeError, ValueError, TypeError):
            issues["invalid_delivered_response"] += 1
            continue
        row = role_row(role)
        row["delivered_response_count"] += 1
        tool_ids = set()
        for tool_index, call in enumerate(calls):
            if not isinstance(call, dict):
                issues["invalid_tool_call"] += 1
                continue
            tool_id = call.get("id")
            if isinstance(tool_id, str):
                if tool_id in tool_ids:
                    issues["duplicate_tool_call_id"] += 1
                    continue
                tool_ids.add(tool_id)
            function = mapping(call.get("function"))
            tool = function.get("name")
            if not isinstance(tool, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", tool):
                issues["invalid_tool_name"] += 1
                continue
            row["tool_calls"][redact(tool)] += 1
            started_at = event_times.get(identifier)
            if tool in EDIT_TOOLS and started_at is not None:
                latest = row["last_edit_proposal_attempt_started_at"]
                row["last_edit_proposal_attempt_started_at"] = started_at if latest is None else max(latest, started_at)
            if tool not in COLLABORATION and tool != "run_shell_command":
                continue
            arguments = function.get("arguments")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(arguments, dict):
                    raise ValueError("arguments not an object")
            except (TypeError, ValueError):
                issues["invalid_tool_arguments"] += 1
                continue
            if tool == "run_shell_command":
                row["shell_categories"][shell_category(arguments)] += 1
                if isinstance(arguments.get("command"), str) and SLEEP_COMMAND.search(arguments["command"]):
                    row["sleep_commands_issued"] += 1
                continue
            fields = {key: value for key in sorted(ARGUMENT_TEXT)
                      if (value := text_leaf(arguments.get(key))) is not None}
            report["collaboration_timeline"].append({
                "attempt": identifier, "role": redact(role), "tool": tool, "tool_index": tool_index,
                "path": redact(str(path.relative_to(root)).replace(role, redact(role))),
                "attempt_started_at": event_times.get(identifier),
                "fields": fields, "evidence_kind": "delivered_tool_proposal",
                "execution_status": "rejected_before_execution" if "__agent_company_tool_error" in arguments
                                    else "execution_unconfirmed",
                "receipt_confirmed": None, "causal_effect": "unknown"})

    report["by_role"] = [roles[key] for key in sorted(roles)]
    lifecycle, lifecycle_stats = read_lifecycle(root / "team" / "qwen_stderr.txt", root)
    report["delivery"] = delivery_projection(lifecycle, issues)
    report["checkins"] = checkin_projection(lifecycle)
    report["runtime_rules"] = runtime_rules_projection(lifecycle)
    report["prompt_cache"] = prompt_cache_projection(raw_events)
    report["role_policies"] = role_policy_projection(root, raw_events, lifecycle, report["collaboration_timeline"],
                                                     issues)
    report["coverage"] = {
        "observed_started_attempts": len(started), "observed_finished_attempts": len(finished),
        "observed_response_files_unique_attempts": len(response_seen),
        "parsed_delivered_responses": sum(row["delivered_response_count"] for row in roles.values()),
        "expected_error_only_responses": sum(row["expected_error_only_response_count"] for row in roles.values()),
        "collaboration_proposal_count": len(report["collaboration_timeline"]),
        "retry_scheduled_count": len(retry_seen), "issues": dict(issues),
        "message_receipt_or_tool_execution_verified": False,
        "lifecycle": {"stderr_present": lifecycle is not None, **lifecycle_stats},
        "score_layers": layer_projection["coverage"],
    }
    # Fail rather than emitting non-standard JSON or unexpected objects.
    json.dumps(report, allow_nan=False)
    return report


def model_evidence_view(report: dict) -> dict:
    """Keep aggregate score layers for A/C research without test/module diagnostics.

    Other evidence fields retain the existing projection. Unknown passed counts
    stay null even when the diagnostic view has observed-only partial counts;
    a known denominator is preserved without filling absent values with zero.
    """
    raw_layers = mapping(report.get("score_layers"))
    layers = mapping(raw_layers.get("layers"))
    minimal_layers = None
    if raw_layers:
        minimal_layers = {"layers": {
            name: {"passed": (number(mapping(layers.get(name)).get("passed"), integer=True)
                              if raw_layers.get("status") == "known" else None),
                   "total": number(mapping(layers.get(name)).get("total"), integer=True)}
            for name in ("core", "information")}}
    coverage = mapping(report.get("coverage"))
    layer_coverage = mapping(coverage.get("score_layers"))
    return {**report, "score_layers": minimal_layers,
            "coverage": {**coverage, "score_layers": {"reasons": layer_coverage.get("reasons")}}}


def project_model_evidence(run_dir: str | Path, *, layer_rules: dict | None = None) -> dict:
    """The projection written into optimizer/reviewer evidence directories."""
    return model_evidence_view(project_evidence(run_dir, layer_rules=layer_rules))


# ----- schema 4: prompt cache and role policies (2026-10-10) ------------------------

POLICY_HEADER = re.compile(r"(ROLE-POLICY|POLICY-PROPOSAL|POLICY-DECISION|GLOBAL-POLICY)(?=[ \t]|$)")


def _cache_tokens(raw_usage: object) -> int | None:
    """Cached prompt tokens if the provider reported a cache field; None when absent."""
    usage = mapping(raw_usage)
    if "prompt_cache_hit_tokens" in usage:
        return number(usage["prompt_cache_hit_tokens"], integer=True)
    details = mapping(usage.get("prompt_tokens_details"))
    if "cached_tokens" in details:
        return number(details["cached_tokens"], integer=True)
    return None


def prompt_cache_projection(events: list[dict]) -> dict:
    """Per-role cache reporting from raw provider usage, before relay normalization."""
    rows: dict[str, dict] = {}
    for event in events:
        if event.get("event") != "provider_attempt_finished" or not isinstance(event.get("role"), str):
            continue
        role = _name(event["role"])
        if role is None:
            continue
        row = rows.setdefault(role, {"attempts": 0, "attempts_without_usage": 0, "attempts_reporting_cache": 0,
                                     "prompt_tokens_reporting_cache": 0, "cached_prompt_tokens": 0,
                                     "prompt_tokens_without_cache_field": 0})
        row["attempts"] += 1
        raw = event.get("raw_usage")
        prompt = number(mapping(raw).get("prompt_tokens"), integer=True)
        if prompt is None:
            row["attempts_without_usage"] += 1
            continue
        cached = _cache_tokens(raw)
        if cached is None or cached > prompt:
            row["prompt_tokens_without_cache_field"] += prompt
            continue
        row["attempts_reporting_cache"] += 1
        row["prompt_tokens_reporting_cache"] += prompt
        row["cached_prompt_tokens"] += cached
    total = {key: sum(row[key] for row in rows.values()) for key in (
        "attempts", "attempts_without_usage", "attempts_reporting_cache", "prompt_tokens_reporting_cache",
        "cached_prompt_tokens", "prompt_tokens_without_cache_field")}
    for row in [*rows.values(), total]:
        row["cached_share_of_reporting_prompt"] = (round(row["cached_prompt_tokens"] / row["prompt_tokens_reporting_cache"], 4)
                                                   if row["prompt_tokens_reporting_cache"] else None)
    return {"by_role": dict(sorted(rows.items())), "total": total}


def _policy_kind(fields: dict) -> str | None:
    message = mapping(fields.get("message")).get("text")
    if not isinstance(message, str):
        return None
    match = POLICY_HEADER.match(message.split("\n", 1)[0].strip())
    return match.group(1) if match else None


def _first(values):
    known = [value for value in values if value is not None]
    return min(known) if known else None


def role_policy_projection(root: Path, events: list[dict], lifecycle: list[dict] | None,
                           collaboration: list[dict], issues: Counter) -> dict:
    """Role-policy ledger, injection, notification and per-version behavior (plan section 3.6)."""
    loaded = {event.get("role_policy_mode") for event in lifecycle or () if event.get("event") == "compatibility_loaded"}
    batches = [event for event in events if event.get("event") == "role_policy_batch"]
    for entry in collaboration:
        if entry.get("tool") == "send_message":
            entry["policy_control"] = _policy_kind(mapping(entry.get("fields")))
    if not batches:
        status = "off" if loaded == {"off"} else "not_instrumented"
        return {"status": status, "mode": "off" if status == "off" else None}
    from agent_company.policy.role_policy import ReplayError, RolePolicyStore  # standard library only
    try:
        state = RolePolicyStore.replay(events, lambda record: None).export()
        replay = "ok"
    except (ReplayError, KeyError, TypeError, ValueError) as exc:
        issues["role_policy_replay_failed"] += 1
        state, replay = None, redact(f"{type(exc).__name__}: {exc}")[:300]
    first_changes = mapping(batches[0]).get("changes")
    mode = mapping(first_changes[0]).get("mode") if isinstance(first_changes, list) and first_changes else None
    commits: list[tuple[float | None, dict]] = []
    counts = Counter()
    renotified = Counter()
    for batch in batches:
        at = number(batch.get("time"))
        for change in batch.get("changes") or []:
            if not isinstance(change, dict):
                continue
            kind = change.get("event")
            counts[kind] += 1
            if kind in {"role_policy_committed", "role_policy_proposed", "role_policy_decided",
                        "global_policy_committed"}:
                commits.append((at, change))
            elif kind == "role_policy_proposal_voided":
                counts[f"voided_{change.get('reason')}"] += 1
            elif kind == "role_policy_proposal_renotified":
                renotified[change.get("proposal_id")] += 1
            if kind == "role_policy_decided":
                counts[f"decided_{change.get('decision')}"] += 1

    injected = [event for event in events if event.get("event") == "role_policy_injected"]
    attempts: dict[int, list[tuple[float | None, bool]]] = {}
    attempt_request: dict[int, int] = {}
    for event in events:
        if event.get("event") != "provider_attempt_finished":
            continue
        request, attempt = number(event.get("source_request"), integer=True), number(event.get("attempt"), integer=True)
        if request is None:
            continue
        attempts.setdefault(request, []).append((number(event.get("started_at")), event.get("upstream_started") is True))
        if attempt is not None:
            attempt_request[attempt] = request

    def carried_start(request):
        return _first(start for start, upstream in attempts.get(request, []) if upstream)

    by_request = {(event.get("role"), number(event.get("source_request"), integer=True)): event
                  for event in injected if event.get("admitted") is True}
    injection_rows: dict[str, dict] = {}
    for event in injected:
        role = _name(event.get("role")) or "unknown"
        row = injection_rows.setdefault(role, {"admitted": 0, "unadmitted": 0, "chars": []})
        row["admitted" if event.get("admitted") is True else "unadmitted"] += 1
        row["chars"].append(number(event.get("injected_chars"), integer=True))
    injection = {role: {"admitted": row["admitted"], "unadmitted": row["unadmitted"],
                        "injected_chars": _summary(row["chars"])} for role, row in sorted(injection_rows.items())}

    def first_carry(predicate, commit_time):
        starts = [carried_start(number(event.get("source_request"), integer=True))
                  for event in injected if event.get("admitted") is True and predicate(event)]
        start = _first(starts)
        return None if start is None or commit_time is None else round(start - commit_time, 3)

    slots = mapping(state).get("slots", {}) if state else {}
    bindings = mapping(state).get("bindings", {}) if state else {}
    version_rows, proposal_rows = [], []
    for at, change in commits:
        if change.get("event") == "role_policy_committed":
            slot, version = change.get("slot"), change.get("version")
            version_rows.append({
                "slot": _name(slot), "scope": "role", "version": number(version, integer=True),
                "provenance": change.get("provenance"),
                "actor": _name(change.get("actor")), "proposal_id": number(change.get("proposal_id"), integer=True),
                "committed_at": change.get("committed_at") if ISO_TIME.fullmatch(str(change.get("committed_at"))) else None,
                "source_request": number(change.get("source_request"), integer=True),
                "text": text_leaf(change.get("text")),
                "first_carry_latency_s": first_carry(
                    lambda event, slot=slot, version=version: event.get("role") != "master"
                    and mapping(event.get("slot_versions")).get(slot) == version, at)})
        elif change.get("event") == "global_policy_committed":
            version = change.get("version")
            version_rows.append({
                "slot": None, "scope": "global", "version": number(version, integer=True), "provenance": "master_set",
                "actor": _name(change.get("actor")), "proposal_id": None,
                "committed_at": change.get("committed_at") if ISO_TIME.fullmatch(str(change.get("committed_at"))) else None,
                "source_request": number(change.get("source_request"), integer=True),
                "text": text_leaf(change.get("text")),
                "first_carry_latency_s": first_carry(
                    lambda event, version=version: event.get("role") != "master"
                    and event.get("global_version") == version, at)})
        elif change.get("event") == "role_policy_proposed":
            identifier = change.get("proposal_id")
            final = mapping(mapping(state).get("proposals", {}).get(str(identifier))) if state else {}
            decided = next((t for t, c in commits if c.get("event") == "role_policy_decided"
                            and c.get("proposal_id") == identifier), None)
            proposal_rows.append({
                "proposal_id": number(identifier, integer=True), "slot": _name(change.get("slot")),
                "proposer": _name(change.get("proposer")), "base_version": number(change.get("base_version"), integer=True),
                "source_request": number(change.get("source_request"), integer=True),
                "status": final.get("status"), "void_reason": final.get("void_reason"),
                "replaced_by": final.get("replaced_by"), "decided_version": final.get("decided_version"),
                "renotified": renotified.get(identifier, 0),
                "change": text_leaf(change.get("change")), "reason": text_leaf(change.get("reason")),
                "evidence": text_leaf(change.get("evidence")),
                "first_carry_to_master_latency_s": first_carry(
                    lambda event, identifier=identifier: event.get("role") == "master"
                    and identifier in (event.get("proposal_ids") or []), at),
                "decision_latency_s": (round(decided - at, 3) if decided is not None and at is not None else None)})

    committed_controls = {(batch.get("actor"), change.get("tool_call_id"), batch.get("source_request"))
                          for batch in batches for change in batch.get("changes") or []
                          if isinstance(change, dict) and change.get("tool_call_id")
                          and change.get("event") in {"role_policy_committed", "global_policy_committed",
                                                      "role_policy_proposed", "role_policy_proposal_renotified",
                                                      "role_policy_decided"}
                          and change.get("provenance") != "spawn_initial"}
    results = Counter()
    resolved = set()
    for event in events:
        kind = event.get("event")
        if kind in {"role_policy_message_sent", "role_policy_message_send_partial", "role_policy_message_send_failed"}:
            results[kind.removeprefix("role_policy_message_")] += 1
            resolved.add((event.get("role"), event.get("tool_call_id"), event.get("source_request")))
    results["unknown"] = len(committed_controls - resolved)

    rejected = [event for event in events if event.get("event") == "role_policy_message_rejected"]
    spawn_results = Counter(str(event.get("outcome")) for event in events if event.get("event") == "role_policy_spawn_result")
    broadcasts = [event for event in lifecycle or () if event.get("event") == "team_broadcast"]
    outcomes = Counter(str(mapping(recipient).get("outcome")) for event in broadcasts
                       for recipient in event.get("recipients") or [])

    # Behavior per version: the version carried by the request that produced each call.
    # In global mode the C version is the policy a worker acts under; in role modes its slot version.
    behavior: dict[tuple, dict] = {}
    for entry in collaboration:
        role = entry.get("role")
        request = attempt_request.get(entry.get("attempt"))
        snapshot = by_request.get((role, request))
        binding = mapping(bindings.get(role)) if isinstance(role, str) else {}
        slot = binding.get("slot_id")
        version = mapping(mapping(snapshot).get("slot_versions")).get(slot) if snapshot and slot else None
        global_version = number(mapping(snapshot).get("global_version"), integer=True) if snapshot else None
        entry["role_policy_version"] = ({"slot": slot, "version": version, "global_version": global_version,
                                         "source_request": request} if slot is not None else None)
        if (entry.get("tool") != "send_message" or entry.get("policy_control") or slot is None
                or entry.get("execution_status") == "rejected_before_execution"):
            continue
        to = mapping(mapping(entry.get("fields")).get("to")).get("text")
        key = ("C", global_version) if mode == "global" else (slot, version)
        row = behavior.setdefault(key, {"scope": "global" if mode == "global" else "role", "slot": key[0],
                                        "version": key[1], "messages": 0, "to_leader": 0, "to_teammate": 0,
                                        "broadcast": 0})
        row["messages"] += 1
        row["to_leader" if to == "leader" else "broadcast" if to == "*" else "to_teammate"] += 1

    tail = {}
    summary_path = root / "team" / "summary.json"
    if safe_file(summary_path, root):
        try:
            run = mapping(json.loads(summary_path.read_text(encoding="utf-8")).get("role_policy_run"))
            lifecycle_tail = mapping(run.get("lifecycle_tail"))
            tail = {"terminal_events": number(lifecycle_tail.get("terminal_events"), integer=True),
                    "invalid_lines": number(lifecycle_tail.get("invalid_lines"), integer=True),
                    "callback_errors": len(lifecycle_tail.get("callback_errors") or []),
                    "max_consume_delay_seconds": number(lifecycle_tail.get("max_consume_delay_seconds")),
                    "voided_at_run_end": number(run.get("voided_at_run_end"), integer=True),
                    "finalization_error": "finalization_error" in run,
                    "rule_finalization_error": "rule_finalization_error" in run}
        except (OSError, UnicodeError, ValueError, TypeError):
            issues["invalid_team_summary"] += 1

    proposed = counts["role_policy_proposed"]
    decided = sum(counts[f"decided_{kind}"] for kind in ("approve", "reject", "amend"))
    return {
        "status": "on", "mode": mode if mode in {"global", "master", "propose"} else None,
        "ledger": {"batches": len(batches), "replay": replay,
                   "slots": {name: {"current_instance": mapping(slot).get("current_instance"),
                                    "versions": len(mapping(slot).get("versions") or [])}
                             for name, slot in sorted(slots.items())},
                   "instances": {name: {key: mapping(binding).get(key)
                                        for key in ("slot_id", "lifecycle", "predecessor", "end_reason")}
                                 for name, binding in sorted(bindings.items())},
                   "global_versions": len(mapping(state).get("global_versions") or []) if state else None},
        "versions": version_rows,
        "proposals": {"rows": proposal_rows, "proposed": proposed, "renotified": sum(renotified.values()),
                      "replaced": counts["voided_replaced"],
                      "voided": {reason: counts[f"voided_{reason}"] for reason in
                                 ("replaced", "policy_superseded", "proposer_terminal", "run_ended")},
                      "decided": {kind: counts[f"decided_{kind}"] for kind in ("approve", "reject", "amend")},
                      "decision_rate": round(decided / proposed, 4) if proposed else None,
                      "approval_rate": (round((counts["decided_approve"] + counts["decided_amend"]) / decided, 4)
                                        if decided else None)},
        "injection": injection,
        "notifications": dict(sorted(results.items())),
        "rejections": {"by_stage": dict(sorted(Counter(str(e.get("stage")) for e in rejected).items())),
                       "by_code": dict(sorted(Counter(str(e.get("code")) for e in rejected).items())),
                       "stale_decisions": sum(e.get("code") == "stale_proposal" for e in rejected),
                       "discarded_calls": sum(number(e.get("discarded_calls"), integer=True) or 0 for e in rejected)},
        "spawn_results": dict(sorted(spawn_results.items())),
        "broadcasts": {"count": len(broadcasts), "global_policy": sum(e.get("global_policy") is True for e in broadcasts),
                       "zero_recipients": sum(not e.get("recipients") for e in broadcasts),
                       "recipient_outcomes": dict(sorted(outcomes.items()))},
        "messages_by_version": sorted(behavior.values(), key=lambda row: (row["slot"], row["version"] is None,
                                                                           row["version"] or 0)),
        "lifecycle_tail": tail or None,
    }

