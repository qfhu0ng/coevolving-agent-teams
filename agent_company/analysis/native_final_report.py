"""Read-only retrospective check of native final-report delivery.

Unlike the allowlist optimizer projection, this explicitly reads delivered text
and subsequent master user messages or task_list tool results. It never inspects reasoning fields
or emits message bodies. A suspicion is separate from official score validity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from agent_company.analysis.evidence_projection import IDENTIFIER, number, safe_file


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content if isinstance(p, dict)
                         and p.get("type") == "text" and isinstance(p.get("text"), str))
    return ""


def inspect_final_report(directory: Path) -> dict:
    """A run's saved summary/launch/events and delivered requests/responses only."""
    root = Path(directory).absolute()
    issues = []
    result = {"final_report_dropped_suspected": None, "status": "unknown",
              "run": str(root), "issues": issues, "workers": {},
              "last_master_attempt": None, "seen_in_later_master_request": None,
              "exit_timing": "unknown_no_absolute_cli_end_timestamp",
              "official_score_validity_changed": False, "reads_reasoning": False}

    def read(relative, lines=False):
        path = root / relative
        if not safe_file(path, root):
            raise ValueError(f"missing_or_unsafe:{relative}")
        try:
            if lines:
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if not all(isinstance(row, dict) for row in rows):
                    raise ValueError
                return rows
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError
            return value
        except (OSError, UnicodeError, ValueError):
            raise ValueError(f"invalid:{relative}") from None

    def artifact(event, kind):
        return read(f"relay/attempt_{event['attempt']:05d}_{event['role']}_{kind}.json")

    def message(event):
        body = artifact(event, "qwen_response")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or not isinstance(choices[0].get("message"), dict):
            raise ValueError("invalid_delivered_message")
        msg = choices[0]["message"]
        content, calls = msg.get("content"), msg.get("tool_calls")
        if ("content" not in msg and not calls or content is not None and not isinstance(content, (str, list))
                or calls is not None and (not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls))
                or isinstance(content, list) and not all(isinstance(c, dict) and c.get("type") == "text"
                                                        and isinstance(c.get("text"), str) for c in content)):
            raise ValueError("invalid_delivered_message_fields")
        return msg

    try:
        launch, summary = read("launch.json"), read("summary.json")
        mode = launch.get("execution_mode", summary.get("execution_mode"))
        if mode == "single":
            return {**result, "status": "not_applicable_single", "final_report_dropped_suspected": False}
        if mode != "team":
            raise ValueError("unknown_execution_mode")
        stop = summary.get("stop_reason")
        if not isinstance(stop, str) or not stop:
            raise ValueError("missing_or_unknown_stop_reason")
        if stop != "native_cli_completed":
            return {**result, "status": "not_native_completion", "final_report_dropped_suspected": None}
        events = read("relay/events.jsonl", lines=True)
        started, finished = {}, {}
        last_time = None
        for position, event in enumerate(events):
            kind = event.get("event")
            if kind not in {"provider_attempt_started", "provider_attempt_finished"} or event.get("purpose", "coding") != "coding":
                continue
            role, attempt, at = event.get("role"), event.get("attempt"), event.get("time")
            if (not isinstance(role, str) or not IDENTIFIER.fullmatch(role)
                    or number(at) is None or number(attempt, integer=True) is None):
                raise ValueError("invalid_attempt_metadata")
            if last_time is not None and at < last_time:
                raise ValueError("event_clock_reversed")
            last_time = at
            event = {**event, "position": position}
            target = started if kind.endswith("started") else finished
            if attempt in target:
                raise ValueError("duplicate_attempt_metadata")
            target[attempt] = event
        if not started or set(started) != set(finished):
            raise ValueError("incomplete_attempts")
        for attempt, end in finished.items():
            start = started[attempt]
            if start["role"] != end["role"] or start["time"] > end["time"]:
                raise ValueError("conflicting_attempt_metadata")
        masters = [e for e in finished.values() if e["role"] == "master"]
        workers = [e for e in finished.values() if e["role"] != "master"]
        if not masters or not workers:
            raise ValueError("missing_master_or_worker_response")
        def order(event):
            return event["time"], event["position"]
        last_master = max(masters, key=order)
        result["last_master_attempt"] = last_master["attempt"]
        if last_master.get("relay_http_status") != 200:
            raise ValueError("last_master_response_not_delivered")
        final = message(last_master)
        if not _text(final.get("content")).strip() or final.get("tool_calls"):
            return {**result, "status": "master_final_turn_unknown"}
        # Inspect every observed worker, not just whichever worker finished last.
        # A tool response does not prove that its final turn/handshake completed.
        for role in sorted({e["role"] for e in workers}):
            worker = max((e for e in workers if e["role"] == role), key=order)
            row = {"attempt": worker["attempt"], "status": "unknown", "issues": [],
                   "final_report_dropped_suspected": None, "seen_in_later_master_request": None,
                   "report_kind": None, "delivery_sources": []}
            result["workers"][role] = row
            try:
                if worker.get("relay_http_status") != 200:
                    raise ValueError("last_worker_response_not_delivered")
                last = message(worker)
                if last.get("tool_calls"):
                    raise ValueError("last_worker_turn_ends_with_tool_calls")
                text = _text(last.get("content")).strip()
                row["report_kind"] = "text" if text else "empty_turn_notice"
                text = text or (f"{role} completed a turn without a model-visible final answer. "
                                "Check the shared task list or send a follow-up if more detail is needed.")
                escaped = re.sub(r"<(/?\s*teammate_message)(?=[\s>/]|$)", r"&lt;\1", text, flags=re.IGNORECASE)
                envelope = f'<teammate_message from="{role}">\n{escaped}\n</teammate_message>'
                request_errors, earlier_errors = [], []
                earlier_count = later_count = 0
                for event in started.values():
                    if event["role"] != "master":
                        continue
                    later = order(event) > order(worker)
                    count = 0
                    try:
                        messages = artifact(event, "provider_request").get("messages")
                        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                            raise ValueError("invalid_master_request")
                        call_names = {}
                        for m in messages:
                            if m.get("role") == "assistant":
                                calls = m.get("tool_calls") or []
                                if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
                                    raise ValueError("invalid_master_tool_calls")
                                call_names = {call.get("id"): call.get("function", {}).get("name")
                                              for call in calls if isinstance(call.get("id"), str)
                                              and isinstance(call.get("function"), dict)}
                                for call_id in call_names:
                                    if sum(call.get("id") == call_id for call in calls) != 1:
                                        call_names[call_id] = None
                            source = "user" if m.get("role") == "user" else "task_list" if (
                                m.get("role") == "tool" and call_names.get(m.get("tool_call_id")) == "task_list") else None
                            if source is None:
                                if (m.get("role") == "tool" and not call_names.get(m.get("tool_call_id"))
                                        and envelope in _text(m.get("content"))):
                                    raise ValueError("unlinked_tool_report_receipt_unknown")
                                continue
                            content = m.get("content")
                            if not isinstance(content, (str, list)) or isinstance(content, list) and not all(
                                    isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                                    for part in content):
                                raise ValueError("invalid_master_request_content")
                            if envelope in _text(content) and (source == "user" or "--- Teammate messages ---" in _text(content)):
                                count += _text(content).count(envelope)
                                if later:
                                    row["delivery_sources"].append({"attempt": event["attempt"], "source": source})
                        if later:
                            later_count = max(later_count, count)
                        else:
                            earlier_count = max(earlier_count, count)
                    except (ValueError, OSError, TypeError, KeyError) as exc:
                        (request_errors if later else earlier_errors).append(str(exc))
                if later_count and (earlier_errors or later_count <= earlier_count):
                    row["issues"].extend(earlier_errors or ["receipt_identity_ambiguous_repeated_report"])
                    continue
                seen = later_count > earlier_count
                # One verified receipt suffices; incomplete evidence cannot prove absence.
                if not seen and (request_errors or row["report_kind"] == "empty_turn_notice"):
                    row["issues"].extend(request_errors or ["empty_turn_notice_not_observed; native_idle_or_explicit_report_unknown"])
                    continue
                row.update(seen_in_later_master_request=seen, final_report_dropped_suspected=not seen,
                           status="final_report_observed" if seen else "final_report_dropped_suspected")
            except (ValueError, OSError, TypeError, KeyError) as exc:
                row["issues"].append(str(exc))
        issues.extend(f"{role}:{issue}" for role, row in result["workers"].items() for issue in row["issues"])
        values = [row["final_report_dropped_suspected"] for row in result["workers"].values()]
        suspicion = True if True in values else None if None in values else False
        result.update(final_report_dropped_suspected=suspicion,
                      seen_in_later_master_request=all(v is False for v in values) if None not in values else None,
                      status="final_report_dropped_suspected" if suspicion is True else "unknown" if suspicion is None else "final_reports_observed",
                      completion_evidence="last_master_text_then_native_cli_completed_no_later_master_response",
                      scope="last_observed_turn_per_worker; no_shutdown_verification")
        return result
    except (ValueError, OSError, TypeError, KeyError) as exc:
        issues.append(str(exc))
        return result
