"""Offline action/accounting projections; no model calls or semantic judgments.

Only the current attempt's relay events and delivered responses are read. Request
history, provider response copies, reasoning text and interrupted attempts are not
counted as actions. Tool calls are proposals delivered to Qwen, not proof of execution.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from agent_company.evolution.selection import measurements

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "total_tokens")
ATTEMPT = re.compile(r"attempt_(\d+)_([A-Za-z][A-Za-z0-9_-]{0,63})_qwen_response\.json$")
TEST_COMMAND = re.compile(r"\b(?:pytest|vitest|jest|tsc|eslint)\b|\b(?:cargo|go|npm|pnpm|yarn)\s+(?:test|check|build)\b")


def _safe_file(path: Path, root: Path) -> bool:
    return path.is_file() and path.resolve().is_relative_to(root.resolve())


def efficiency_report(record: dict, directory: Path) -> dict:
    directory = Path(directory)
    mode = record.get("execution_mode", "team")
    if mode not in ("single", "team"):
        raise ValueError("Unknown execution mode")
    relay = directory / mode / "relay"
    runtime = (record.get("metrics") or {}).get(mode) or {}
    roles, started, finished = {}, set(), set()
    errors = Counter()

    def role(name):
        if name not in roles:
            roles[name] = {"role": name, "observed_finished_attempts": 0,
                           "attempts_with_missing_usage": 0,
                           "known_tokens": dict.fromkeys(TOKEN_FIELDS, 0),
                           "known_provider_seconds": 0.0, "attempts_with_missing_time": 0,
                           "delivered_responses": 0, "tool_calls": Counter()}
        return roles[name]

    events = relay / "events.jsonl"
    events_present = _safe_file(events, directory)
    if events_present:
        with events.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("Not an event")
                except (ValueError, TypeError):
                    errors["invalid_event_lines"] += 1
                    continue
                kind, identifier, name = event.get("event"), event.get("attempt"), event.get("role")
                if kind not in {"provider_attempt_started", "provider_attempt_finished"}:
                    continue
                if (not isinstance(identifier, int) or isinstance(identifier, bool) or identifier < 1
                        or not isinstance(name, str)
                        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name)):
                    errors["invalid_attempt_events"] += 1
                    continue
                if kind == "provider_attempt_started":
                    started.add(identifier)
                    role(name)
                    continue
                if identifier in finished:
                    errors["duplicate_finished_events"] += 1
                    continue
                finished.add(identifier)
                item = role(name)
                item["observed_finished_attempts"] += 1
                usage = event.get("usage") or {}
                valid = isinstance(usage, dict) and all(
                    isinstance(usage.get(k), int) and not isinstance(usage[k], bool) and usage[k] >= 0
                    for k in TOKEN_FIELDS)
                if valid:
                    for key in TOKEN_FIELDS:
                        item["known_tokens"][key] += usage[key]
                else:
                    item["attempts_with_missing_usage"] += 1
                elapsed = event.get("elapsed_seconds")
                if (isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                        and math.isfinite(elapsed) and elapsed >= 0):
                    item["known_provider_seconds"] += elapsed
                else:
                    item["attempts_with_missing_time"] += 1

    repeated = defaultdict(list)
    collaboration = []
    classifications = Counter()
    observed_responses = set()
    response_ids = set()
    for path in sorted(relay.glob("attempt_*_qwen_response.json")):
        match = ATTEMPT.fullmatch(path.name)
        if not match or not _safe_file(path, directory):
            continue
        identifier, name = int(match[1]), match[2]
        if identifier in response_ids:
            errors["duplicate_response_attempts"] += 1
            continue
        response_ids.add(identifier)
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
            choices = response.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError("Invalid choices")
            calls = []
            # Qwen consumes the first completion, not hypothetical alternatives.
            if choices:
                calls = choices[0]["message"].get("tool_calls") or []
                if not isinstance(calls, list):
                    raise ValueError("Invalid tool calls")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            errors["unreadable_responses"] += 1
            continue
        observed_responses.add(identifier)
        item = role(name)
        item["delivered_responses"] += 1
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                errors["invalid_tool_calls"] += 1
                continue
            function = call["function"]
            tool = function.get("name")
            if not isinstance(tool, str) or not tool:
                errors["invalid_tool_calls"] += 1
                continue
            item["tool_calls"][tool] += 1
            if tool in {"agent", "send_message", "task_create", "task_update", "request_shutdown"}:
                try:
                    args = function.get("arguments")
                    args = json.loads(args) if isinstance(args, str) else args
                    if not isinstance(args, dict):
                        raise ValueError("Invalid collaboration arguments")
                    # No message bodies, shell text or inferred receipt/causality.
                    fields = {k: args[k][:120] for k in ("recipient", "name", "task_id", "owner", "status", "type")
                              if isinstance(args.get(k), str)}
                    collaboration.append({"attempt": identifier, "role": name, "tool": tool,
                        "fields": fields, "path": str(path.relative_to(directory)), "tool_index": index,
                        "execution_status": "rejected_before_execution" if "__agent_company_tool_error" in args
                                            else "delivered_call_execution_unconfirmed",
                        "receipt_confirmed": None, "causal_effect": "unknown"})
                except (ValueError, TypeError):
                    errors["invalid_collaboration_arguments"] += 1
            if tool != "run_shell_command":
                continue
            args = function.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else args
                if not isinstance(args, dict):
                    raise ValueError("Invalid arguments")
                command = args.get("command")
                if not isinstance(command, str):
                    raise ValueError("Missing command")
            except (ValueError, TypeError):
                errors["invalid_shell_arguments"] += 1
                continue
            # Include cwd/options via the entire argument object. Do not output
            # shell text (possibly sensitive); evidence references support inspection.
            category = ("test_build_lint" if TEST_COMMAND.search(command) else
                        "git" if re.search(r"\bgit\b", command) else "other_shell")
            classifications[category] += 1
            key = json.dumps(args, sort_keys=True, ensure_ascii=False)
            repeated[key].append({"role": name, "path": str(path.relative_to(directory)),
                                  "tool_index": index, "category": category})

    expected = runtime.get("provider_attempts")
    expected = expected if isinstance(expected, int) and not isinstance(expected, bool) and expected >= 0 else None
    coverage = {"events_present": events_present, "expected_attempts": expected,
                "observed_started_attempts": len(started), "observed_finished_attempts": len(finished),
                "observed_delivered_responses": len(observed_responses), "parse_issues": dict(errors)}
    coverage["attempts_complete"] = (events_present and not errors and expected is not None
                                     and len(started) == expected and started == finished == observed_responses)
    grouped = [values for values in repeated.values() if len(values) > 1]
    grouped.sort(key=lambda values: -len(values))
    totals = measurements(record)
    role_tokens_match = all(totals[key] is not None and totals[key] == sum(
        item["known_tokens"][key] for item in roles.values()) for key in TOKEN_FIELDS)
    coverage["role_usage_complete"] = (coverage["attempts_complete"] and totals["usage_complete"]
                                        and role_tokens_match and all(
                                            item["attempts_with_missing_usage"] == 0 for item in roles.values()))
    return {"scope": "current_attempt_only", "measurements": totals,
            "collaboration_events": collaboration,
            "sources": {"events": str(events.relative_to(directory)) if events_present else None,
                        "responses": f"{mode}/relay/attempt_*_qwen_response.json"},
            "coverage": coverage, "by_role": list(roles.values()),
            "shell_classification_heuristic": dict(classifications),
            "repeated_shell_group_count": len(grouped),
            "repeated_shell_examples": [{"count": len(values),
                                         "roles": sorted({v["role"] for v in values}),
                                         "references": values[:4]} for values in grouped[:8]],
            "notes": ["Delivered tool calls are not confirmed executions or message receipts.",
                      "Repeated exact shell arguments are possible rework, not proven waste; revisions may differ.",
                      "Shell categories are text heuristics, not precise test or handoff counts.",
                      "Provider seconds sum across overlapping calls; not agent wall time or worker idle time.",
                      "Missing traces/usage stay incomplete; zero observed events is not zero work.",
                      "No request-history copies, reasoning text or interrupted attempts are counted."]}


def compact_efficiency(report: dict) -> dict:
    """Keep detailed counters in efficiency.json, a small projection in prompts."""
    roles = sorted(report["by_role"], key=lambda item: -item["known_tokens"]["total_tokens"])
    return {"report_file": "efficiency.json", "coverage": report["coverage"],
            "by_role": [{key: item[key] for key in ("role", "observed_finished_attempts", "known_tokens",
                         "known_provider_seconds", "attempts_with_missing_usage", "attempts_with_missing_time")}
                        | {"message_calls": item["tool_calls"].get("send_message", 0),
                           "shell_calls": item["tool_calls"].get("run_shell_command", 0)}
                        for item in roles[:8]],
            "roles_omitted": max(0, len(roles) - 8),
            "repeated_shell_group_count": report["repeated_shell_group_count"],
            "collaboration_event_count": len(report.get("collaboration_events", [])),
            "collaboration_examples": report.get("collaboration_events", [])[:8],
            "shell_classification_heuristic": report["shell_classification_heuristic"],
            "note": "Observed repetitions are not proven waste; provider times overlap. Read report_file for references and coverage."}
