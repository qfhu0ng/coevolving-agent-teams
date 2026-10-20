"""Offline S0 mechanism readings from relay events and an optional saved projection.

No provider, runner, response/request bodies, or reasoning text is read. The sibling
relay usage summary is checked when present for failures missing from events. JSON null
means unknown, including an incomplete peak (the observed peak is kept separately).
Run-wide policy totals require a replayed role_policy_run_ended marker; observed
counts remain available for an unfinished ledger. Neither establishes task quality.

``committed_controls`` follows role_policy_projection's notification denominator:
distinct (actor, source_request, tool_call_id) for proposal/decision/policy controls,
excluding spawn_initial. A decision and its resulting version count once. Actual
new versions are a separate count. A rejected response can discard multiple calls,
including spawn markers, so it cannot establish the total control-attempt count.
The relay also discards control calls without a role-policy rejection event: a
response rejected before staging (e.g. a bad spawn beside a control), or a staged
batch whose late commit stopped. Either makes the attempt count unknown, except that
a complete saved projection shows the former's discarded control calls (never the
latter's, whose delivered file is not rewritten).
Existing events have no rejected-to-corrected operation link: after a rejection,
corrections remain unknown. Never infer correction from an unrelated later success.

Format errors here are malformed_header/body, too_long, conflicting_markers and
invalid_name; authorization/staleness/lifecycle errors are counted separately.
The U classifier implements the plan's ordered, mutually exclusive D5 rules; it
does not score policy quality. R labels require human directed-smoke assessment.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from agent_company.analysis.evidence_projection import IDENTIFIER, _cache_tokens, number, prompt_cache_projection, safe_file
from agent_company.policy.role_policy import ERROR_CODES, PolicyRejection, ReplayError, RolePolicyStore, parse_send_message

CONTROL_CHANGES = frozenset({
    "role_policy_committed", "global_policy_committed", "role_policy_proposed",
    "role_policy_proposal_renotified", "role_policy_decided",
})
FORMAT_CODES = frozenset({"malformed_header", "malformed_body", "too_long", "conflicting_markers", "invalid_name"})
REVIEW_CONCLUSIONS = ("improved", "no change", "side effects", "cannot judge yet")
REVIEW = re.compile(r"^POLICY-REVIEW (?P<target>[A-Za-z][A-Za-z0-9_-]{0,63}) v[1-9][0-9]*: "
                    r"(?P<conclusion>improved|no change|side effects|cannot judge yet)"
                    r"(?=$|[\s.,;:])(?P<facts>[\s\S]*)$")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def classify_usage(attempts: int | None, commits: int | None,
                   uncorrected_format_rejections: int | None) -> str | None:
    """D5 U rules, using exact counts only; return None when evidence is insufficient."""
    if attempts is None or commits is None:
        return None
    for value in (attempts, commits, uncorrected_format_rejections):
        if value is not None and number(value, integer=True) is None:
            raise ValueError("counts must be nonnegative integers or None")
    if commits > attempts:
        raise ValueError("commits cannot exceed attempts")
    if attempts == 0:
        return "U0"
    if attempts >= 3 and commits == 0:
        return "U3"
    if commits == 0:
        return "U2"
    if uncorrected_format_rejections is None:
        return None
    return "U2" if uncorrected_format_rejections >= commits else "U1"


def _read_json(path: Path, root: Path, issues: Counter, *, lines: bool = False, kind="projection") -> Any:
    if not safe_file(path, root):
        issues["missing_or_unsafe_events" if lines else "missing_or_unsafe_" + kind] += 1
        return [] if lines else None
    try:
        if not lines:
            value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            if not isinstance(value, dict):
                raise ValueError
            return value
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, object_pairs_hook=_unique_object)
                    if not isinstance(record, dict):
                        raise ValueError
                    records.append(record)
                except ValueError:
                    issues["invalid_event_lines"] += 1
        return records
    except (OSError, UnicodeError, ValueError):
        issues["unreadable_events" if lines else "invalid_" + kind] += 1
        return [] if lines else None


def _prompt_peaks(events: list[dict], reliable: bool) -> tuple[dict, list[dict]]:
    """Max of individual provider requests (retries are not summed)."""
    roles: dict[str, dict] = {}
    finished: list[dict] = []
    seen: set[tuple] = set()
    for event in events:
        if event.get("event") not in {"provider_attempt_started", "provider_attempt_finished"}:
            continue
        role = event.get("role")
        if not isinstance(role, str) or not IDENTIFIER.fullmatch(role):
            continue
        row = roles.setdefault(role, {"started": set(), "finished": set(), "prompts": [],
                                      "missing": 0, "ambiguous": False})
        attempt = number(event.get("attempt"), integer=True)
        key = (role, attempt, event["event"])
        if attempt is None or key in seen:
            row["ambiguous"] = True
            continue
        seen.add(key)
        row["started" if event["event"].endswith("started") else "finished"].add(attempt)
        if event["event"] != "provider_attempt_finished":
            continue
        finished.append(event)
        raw = event.get("raw_usage")
        prompt = number(raw.get("prompt_tokens"), integer=True) if isinstance(raw, dict) else None
        if prompt is None:
            row["missing"] += 1
        else:
            row["prompts"].append(prompt)
    result = {}
    for role, row in sorted(roles.items()):
        observed = max(row["prompts"], default=None)
        unfinished = len(row["started"] - row["finished"])
        complete = reliable and not row["ambiguous"] and not row["missing"] and not unfinished
        result[role] = {
            "finished_attempts": len(row["finished"]), "attempts_without_prompt_usage": row["missing"],
            "unfinished_attempts": unfinished, "duplicate_or_invalid_attempt": row["ambiguous"],
            "observed_peak_prompt_tokens": observed,
            "peak_prompt_tokens": observed if complete else None,
            "peak_plus_reserved_output_65536": observed + 65536 if complete and observed is not None else None,
        }
    return result, finished


def _complete_projection(projection: dict | None) -> bool:
    return (isinstance(projection, dict) and isinstance(projection.get("collaboration_timeline"), list)
            and isinstance(projection.get("coverage"), dict) and projection["coverage"].get("issues") == {})


def _projection_matches_events(projection: dict | None, events: list[dict]) -> bool:
    """Bind projected attempts to the supplied log; counts alone cannot do that."""
    if not isinstance(projection, dict) or not isinstance(projection.get("event_timeline"), list):
        return False
    kinds = {"provider_attempt_started", "provider_attempt_finished"}

    def attempts(rows, projected=False):
        found = {}
        for row in rows:
            if not isinstance(row, dict):
                return None
            if row.get("event") not in kinds:
                continue
            role, attempt = row.get("role"), number(row.get("attempt"), integer=True)
            if not isinstance(role, str) or not IDENTIFIER.fullmatch(role) or attempt is None:
                return None
            key = (row["event"], role, attempt)
            if key in found:
                return None
            values = []
            for field in ("source_request", "purpose", "error_kind", "provider_http_status", "relay_http_status"):
                value = row.get(field)
                if projected and field in {"purpose", "error_kind"} and isinstance(value, dict):
                    value = value.get("text")
                values.append(value)
            found[key] = values
        return found

    raw = attempts(events)
    if raw is None or attempts(projection["event_timeline"], projected=True) != raw:
        return False
    timeline = projection.get("collaboration_timeline")
    return isinstance(timeline, list) and all(
        isinstance(row, dict) and isinstance(row.get("role"), str)
        and ("provider_attempt_finished", row.get("role"), number(row.get("attempt"), integer=True)) in raw
        for row in timeline)


def _relay_usage(events_path: Path, events: list[dict], issues: Counter) -> dict:
    """Read only the sibling metadata summary; never claim it proves no lost write."""
    path, root = _anchored(events_path)
    path = path.with_name("usage.json")
    scope = "saved_usage_only; absent_or_stale_summary_cannot_rule_out_unrecorded_control_discards"
    if not path.exists() and not path.is_symlink():
        return {"status": "not_recorded", "control_evidence_failed": None, "scope": scope}
    summary = _read_json(path, root, issues, kind="relay_usage")
    if (summary is None or not isinstance(summary.get("errors"), list)
            or any(not isinstance(row, dict) for row in summary["errors"])):
        issues["relay_usage_unknown"] += 1
        return {"status": "unknown", "control_evidence_failed": None, "scope": scope}
    errors = summary["errors"]
    policy = summary.get("role_policy")
    counts = policy.get("counts", {}) if isinstance(policy, dict) else {}
    failures = number(counts.get("evidence_failures"), integer=True) if isinstance(counts, dict) else None

    def request_identity(row):
        role, request = row.get("role"), number(row.get("source_request"), integer=True)
        return (role, request) if isinstance(role, str) and IDENTIFIER.fullmatch(role) and request is not None else None

    rejected_requests = {identity for row in events if row.get("event") == "role_policy_message_rejected"
                         if (identity := request_identity(row)) is not None}
    # Initial policy rejections are normal recorded tool failures. They signal lost
    # evidence only if the corresponding rejection event is absent from this log.
    failed = bool(failures) or any(
        row.get("role_policy") is True or row.get("error_kind") == "evidence_write_failed"
        or (isinstance(row.get("tool_rejection"), dict) and row["tool_rejection"].get("reason") == "role_policy"
            and request_identity(row) not in rejected_requests)
        for row in errors)
    if failed:
        issues["relay_control_evidence_failed"] += 1
    return {"status": "read", "control_evidence_failed": failed, "scope": scope}


def _discarded_controls(projection: dict | None) -> tuple[int | None, int | None]:
    """Projected control send_message calls the relay returned unexecuted, and rejected
    calls the relay's parser treats as controls but the projection did not classify."""
    if not _complete_projection(projection):
        return None, None
    discarded = unclassified = 0
    for entry in projection["collaboration_timeline"]:
        if (not isinstance(entry, dict) or entry.get("tool") != "send_message"
                or entry.get("execution_status") != "rejected_before_execution"):
            continue
        if entry.get("policy_control"):
            discarded += 1
            continue
        fields = entry.get("fields")
        leaf = fields.get("message") if isinstance(fields, dict) else None
        text = leaf.get("text") if isinstance(leaf, dict) else None
        try:
            candidate = parse_send_message("projected", None, text) is not None
        except (PolicyRejection, ValueError):
            candidate = True  # a malformed header is still a control attempt
        unclassified += candidate
    return discarded, unclassified


def _policy_counts(events: list[dict], reliable: bool, issues: Counter, projection: dict | None = None,
                   *, unfinished_attempts: int = 0, projection_matches: bool = False,
                   usage_evidence_failed: bool = False) -> dict:
    batches = [e for e in events if e.get("event") == "role_policy_batch"]
    replay_ok = False
    run_ended = None
    if batches:
        try:
            replayed = RolePolicyStore.replay(events, lambda record: None)
            replay_ok = True
            run_ended = replayed.export()["closed"]
            if not run_ended:
                issues["role_policy_run_not_ended"] += 1
        except (ReplayError, KeyError, TypeError, ValueError, IndexError):
            issues["role_policy_replay_failed"] += 1
    else:
        issues["role_policy_not_instrumented"] += 1
    changes = [(batch, change) for batch in batches for change in batch.get("changes", [])
               if isinstance(change, dict)] if replay_ok else []
    controls = set()
    versions = []
    counts: Counter = Counter()
    for batch, change in changes:
        kind = change.get("event")
        counts[kind] += 1
        if kind == "role_policy_decided":
            counts["decision_" + str(change.get("decision"))] += 1
            counts["decision_with_note"] += bool(isinstance(change.get("note"), str) and change["note"].strip())
        if kind == "role_policy_proposed":
            counts["proposal_with_fields"] += all(isinstance(change.get(k), str) and change[k].strip()
                                                    for k in ("change", "reason", "evidence"))
        if kind in {"role_policy_committed", "global_policy_committed"}:
            if change.get("provenance") == "spawn_initial":
                counts["initial_role_versions"] += 1
            else:
                versions.append(change)
        if kind in CONTROL_CHANGES and change.get("provenance") != "spawn_initial":
            actor, request, call = batch.get("actor"), batch.get("source_request"), change.get("tool_call_id")
            if isinstance(actor, str) and isinstance(call, str) and number(request, integer=True) is not None:
                controls.add((actor, request, call))
            else:
                issues["control_missing_identity"] += 1
    rejections = [e for e in events if e.get("event") == "role_policy_message_rejected"]
    by_code = Counter(e.get("code") if isinstance(e.get("code"), str) and e["code"] in ERROR_CODES
                      else "unknown" for e in rejections)
    # Discards without a rejection event: prepare_response raised before staging, or a
    # staged batch's late commit was stopped (cancelled or past the deadline).
    nonpolicy = sum(1 for e in events if e.get("event") == "provider_attempt_finished"
                    and e.get("purpose") == "coding" and e.get("error_kind") == "tool_response_rejected"
                    and (e["tool_rejection"].get("reason") if isinstance(e.get("tool_rejection"), dict)
                         else None) != "role_policy")
    uncommitted = sum(1 for e in events if e.get("event") == "native_request_ended"
                      and e.get("role_policy_batch_committed") is False)
    discarded, unclassified = _discarded_controls(projection if projection_matches else None)
    if discarded is None:
        hidden = None if nonpolicy else 0
    elif unclassified or (discarded and not (nonpolicy or rejections)):
        hidden = None  # the projection and the events disagree about what was discarded
    else:
        hidden = discarded
    known = reliable and run_ended is True and replay_ok and not issues["control_missing_identity"]
    # Closing the ledger prevents new commits, but a still-running provider can
    # finish with a rejection or a discarded late batch after that marker.
    complete = reliable and run_ended is True and not unfinished_attempts
    if unfinished_attempts:
        issues["unfinished_provider_attempts"] += unfinished_attempts
    committed = len(controls) if known else None
    if projection is not None and not projection_matches:
        hidden = None
    attempts = (committed + hidden if known and complete and hidden is not None and not rejections and not uncommitted
                and not usage_evidence_failed
                else None)
    formatted = sum(by_code[code] for code in FORMAT_CODES)
    uncorrected = 0 if complete and not formatted and not by_code["unknown"] else None
    label = classify_usage(attempts, committed, uncorrected)
    # Any correction needs operation-level evidence absent from the current event schema.
    return {
        "ledger_replay": "ok" if replay_ok else "unknown",
        "ledger_run_ended": run_ended if reliable else None,
        "control_attempts": attempts,
        "observed_control_attempts_lower_bound": len(controls) + (hidden or 0) if replay_ok else None,
        "observed_committed_controls": len(controls) if replay_ok else None,
        "committed_controls": committed,
        "rejected_responses": len(rejections) if complete else None,
        "observed_rejected_responses": len(rejections) if events else None,
        "nonpolicy_rejected_responses": nonpolicy if complete else None,
        "observed_nonpolicy_rejected_responses": nonpolicy if events else None,
        "uncommitted_late_batches": uncommitted if complete else None,
        "observed_uncommitted_late_batches": uncommitted if events else None,
        "discarded_control_calls": discarded if complete else None,
        "observed_discarded_control_calls": discarded,
        "rejections_by_code": dict(sorted(by_code.items())) if complete else None,
        "observed_rejections_by_code": dict(sorted(by_code.items())) if events else None,
        "format_rejected_responses": formatted if complete and not by_code["unknown"] else None,
        "corrected_rejections": 0 if complete and not rejections else None,
        "uncorrected_format_rejections": uncorrected,
        "policy_version_updates": len(versions) if known else None,
        "initial_role_versions": counts["initial_role_versions"] if known else None,
        "proposals": counts["role_policy_proposed"] if known else None,
        "proposals_with_change_reason_evidence": counts["proposal_with_fields"] if known else None,
        "renotifications": counts["role_policy_proposal_renotified"] if known else None,
        "decisions": {kind: counts["decision_" + kind] if known else None for kind in ("approve", "reject", "amend")},
        "decisions_with_note": counts["decision_with_note"] if known else None,
        "usage_classification": label,
        "classification_scope": "saved_evidence_only",
        "correction_evidence": "no_operation_link_in_current_events" if rejections else None,
        "version_carry": _version_carry(events, versions) if known else None,
    }


def _version_carry(events: list[dict], versions: list[dict]) -> dict:
    upstream = {(e.get("role"), e.get("source_request")) for e in events
                if e.get("event") == "provider_attempt_finished" and e.get("upstream_started") is True
                and isinstance(e.get("role"), str) and IDENTIFIER.fullmatch(e["role"])
                and number(e.get("source_request"), integer=True) is not None}
    injections = [e for e in events if e.get("event") == "role_policy_injected"
                  and e.get("admitted") is True and e.get("role") != "master"
                  and isinstance(e.get("role"), str) and number(e.get("source_request"), integer=True) is not None
                  and (e.get("role"), e.get("source_request")) in upstream]
    carried = 0
    for version in versions:
        if version["event"] == "global_policy_committed":
            found = any(e.get("global_version") == version.get("version") for e in injections)
        else:
            found = any(isinstance(e.get("slot_versions"), dict)
                        and e["slot_versions"].get(version.get("slot")) == version.get("version") for e in injections)
        carried += found
    return {"updated_versions": len(versions), "observed_carried_to_worker_upstream": carried,
            "without_observed_worker_carry": len(versions) - carried,
            "meaning": "request_carriage_only_not_model_reading_or_effectiveness"}


def _cache_after_commit(events: list[dict], reliable: bool) -> dict:
    """Compare adjacent old/new-version admissions, never completion-time windows."""
    scope = ("per role: last older-version admission before the commit and first exact-new-version "
             "admission after it, in saved event order; attempts/retries share their source request")
    if not reliable:
        return {"versions": None, "status": "unknown", "scope": scope, "causal_effect": "unknown"}
    injected, attempts, identities = {}, {}, {}
    unlinked = 0
    for index, event in enumerate(events):
        kind = event.get("event")
        if kind not in {"role_policy_injected", "provider_attempt_started", "provider_attempt_finished"}:
            continue
        role, request = event.get("role"), number(event.get("source_request"), integer=True)
        if not isinstance(role, str) or not IDENTIFIER.fullmatch(role) or request is None:
            unlinked += 1
            continue
        key = (role, request)
        if kind == "role_policy_injected":
            if event.get("admitted") is True:
                injected.setdefault(key, []).append((index, event))
        else:
            attempts.setdefault(key, []).append(event)
            attempt = number(event.get("attempt"), integer=True)
            if attempt is not None:
                identities.setdefault(attempt, set()).add(key)

    def request_cache(key):
        if key is None:
            return None
        rows = attempts.get(key, [])
        started, finished = {}, {}
        ambiguous = len(injected[key]) != 1
        for event in rows:
            attempt = number(event.get("attempt"), integer=True)
            if attempt is None:
                ambiguous = True
                continue
            target = finished if event["event"] == "provider_attempt_finished" else started
            if attempt in target or len(identities[attempt]) != 1:
                ambiguous = True
            target.setdefault(attempt, event)
        prompts, cached, sent = [], [], 0
        for event in finished.values():
            if event.get("upstream_started") is not True:
                continue
            sent += 1
            raw = event.get("raw_usage")
            prompt = number(raw.get("prompt_tokens"), integer=True) if isinstance(raw, dict) else None
            hit = _cache_tokens(raw)
            if prompt is not None and hit is not None and hit <= prompt:
                prompts.append(prompt)
                cached.append(hit)
        unfinished = len(set(started) - set(finished))
        complete = bool(finished) and not ambiguous and not unfinished and len(prompts) == len(finished)
        prompt_total, hit_total = sum(prompts), sum(cached)
        return {"source_request": key[1], "finished_attempts": len(finished),
                "upstream_confirmed_attempts": sent, "attempts_reporting_cache": len(prompts),
                "unknown_usage_attempts": len(finished) - len(prompts), "unfinished_attempts": unfinished,
                "duplicate_or_conflicting_metadata": ambiguous, "usage_complete": complete,
                "prompt_tokens": prompt_total if complete else None,
                "cached_prompt_tokens": hit_total if complete else None,
                "uncached_prompt_tokens": prompt_total - hit_total if complete else None,
                "uncached_share": (prompt_total - hit_total) / prompt_total if complete and prompt_total else None,
                "observed_reporting_prompt_tokens": prompt_total,
                "observed_cached_prompt_tokens": hit_total}

    versions = []
    for index, batch in enumerate(events):
        if batch.get("event") != "role_policy_batch":
            continue
        for change in batch.get("changes", []):
            if (change.get("event") not in {"role_policy_committed", "global_policy_committed"}
                    or change.get("provenance") == "spawn_initial"):
                continue
            global_update = change["event"] == "global_policy_committed"
            version = change["version"]
            slot = None if global_update else change["slot"]
            def carried(event):
                value = event.get("global_version") if global_update else (
                    event.get("slot_versions", {}).get(slot) if isinstance(event.get("slot_versions"), dict) else None)
                return number(value, integer=True)
            by_role = {}
            roles = sorted({key[0] for key in injected})
            for role in roles:
                candidates = [(at, key, carried(event)) for key, entries in injected.items() if key[0] == role
                              for at, event in entries]
                # A role-specific policy is absent from unrelated workers' snapshots.
                if not global_update and not any(value is not None for _, _, value in candidates):
                    continue
                before = [(at, key) for at, key, value in candidates if at < index
                          and value is not None and value < version]
                after = [(at, key) for at, key, value in candidates if at > index and value == version]
                pre = request_cache(max(before)[1] if before else None)
                post = request_cache(min(after)[1] if after else None)
                delta = (post["uncached_share"] - pre["uncached_share"]
                         if pre and post and pre["uncached_share"] is not None
                         and post["uncached_share"] is not None else None)
                by_role[role] = {"pre_last_old_version": pre, "post_first_new_version": post,
                                 "uncached_share_change": delta}
            versions.append({"scope": "global" if global_update else "role", "slot": slot,
                             "version": version, "control_source_request": batch.get("source_request"),
                             "by_role": by_role})
    return {"versions": versions, "status": "descriptive", "scope": scope,
            "unlinked_events": unlinked, "causal_effect": "unknown"}


def _review_counts(projection: dict | None, events: list[dict]) -> dict | None:
    if not isinstance(projection, dict) or not isinstance(projection.get("collaboration_timeline"), list):
        return None
    def request_identity(event):
        role, request = event.get("role"), number(event.get("source_request"), integer=True)
        return (role, request) if isinstance(role, str) and request is not None else None
    cancelled_requests = {request_identity(event) for event in events
                          if event.get("event") == "native_request_ended"
                          and event.get("role_policy_batch_committed") is False}
    linked = set()
    cancelled_attempts = set()
    for event in events:
        if event.get("event") != "provider_attempt_finished":
            continue
        request = request_identity(event)
        if request is not None and request in cancelled_requests:
            linked.add(request)
        if (request is not None and request in cancelled_requests) or event.get("relay_http_status") == 424:
            role, attempt = event.get("role"), number(event.get("attempt"), integer=True)
            if isinstance(role, str) and attempt is not None:
                cancelled_attempts.add((role, attempt))
    counts = {"projected_messages": 0, "rejected_before_execution": 0, "cancelled_before_delivery": 0, "format_unknown": 0,
              "conclusions": dict.fromkeys(REVIEW_CONCLUSIONS, 0)}
    for entry in projection["collaboration_timeline"]:
        if not isinstance(entry, dict) or entry.get("tool") != "send_message":
            continue
        fields = entry.get("fields", {})
        leaf = fields.get("message", {}) if isinstance(fields, dict) else {}
        message = leaf.get("text") if isinstance(leaf, dict) else None
        if not isinstance(message, str) or not re.match(r"POLICY-REVIEW(?:\s|$)", message.lstrip()):
            continue
        counts["projected_messages"] += 1
        if (isinstance(entry.get("role"), str)
                and (entry["role"], number(entry.get("attempt"), integer=True)) in cancelled_attempts):
            counts["cancelled_before_delivery"] += 1
            continue
        if entry.get("execution_status") == "rejected_before_execution":
            counts["rejected_before_execution"] += 1
            continue
        match = REVIEW.fullmatch(message)
        if not match or leaf.get("truncated") is True or not match["facts"].strip(" \t\r\n.,;:"):
            counts["format_unknown"] += 1
        else:
            counts["conclusions"][match["conclusion"]] += 1
    complete = _complete_projection(projection) and not (cancelled_requests - linked)
    return {"counts": counts if complete else None, "observed_counts": counts,
            "meaning": "projected_self_reports_not_verified_execution_or_actual_improvement"}


def _control_times(events: list[dict], launch: dict | None, reliable: bool) -> dict:
    """The saved native launch wall clock is not the relay's monotonic deadline."""
    env = launch.get("environment") if isinstance(launch, dict) else None
    def epoch(key):
        value = env.get(key) if isinstance(env, dict) else None
        try:
            return number(float(value)) if isinstance(value, str) else number(value)
        except ValueError:
            return None
    start = epoch("AGENT_COMPANY_TASK_STARTED_EPOCH_MS")
    deadline = epoch("AGENT_COMPANY_TASK_DEADLINE_EPOCH_MS")
    timeout = number(launch.get("timeout_seconds")) if isinstance(launch, dict) else None
    valid = (start is not None and deadline is not None and deadline > start and timeout is not None
             and abs(deadline - start - timeout * 1000) <= 2)
    # Relay writes wall-clock timestamps under its event lock. A decrease in
    # saved order makes these clock-derived readings ambiguous, even in range.
    times = [at for event in events if (at := number(event.get("time"))) is not None]
    clock_conflict = any(later < earlier for earlier, later in zip(times, times[1:]))
    valid = valid and not clock_conflict
    rows = []
    for event in events:
        kind = event.get("event")
        controls = {}
        if kind == "role_policy_batch" and isinstance(event.get("changes"), list):
            for change in event["changes"]:
                if (isinstance(change, dict) and change.get("event") in CONTROL_CHANGES
                        and change.get("provenance") != "spawn_initial"):
                    call = change.get("tool_call_id")
                    controls.setdefault(call if isinstance(call, str) else None, set()).add(change["event"])
            actor = event.get("actor")
        elif kind == "role_policy_message_rejected":
            call = event.get("tool_call_id")
            controls[call if isinstance(call, str) else None] = {kind}
            actor = event.get("role")
        else:
            continue
        at = number(event.get("time"))
        request = number(event.get("source_request"), integer=True)
        for call, kinds in controls.items():
            known = (reliable and valid and at is not None and start <= at * 1000 <= deadline
                     and request is not None and isinstance(actor, str) and IDENTIFIER.fullmatch(actor)
                     and isinstance(call, str))
            rows.append({"role": actor if isinstance(actor, str) and IDENTIFIER.fullmatch(actor) else None,
                         "source_request": request, "events": sorted(kinds),
                         "elapsed_seconds": round(at - start / 1000, 3) if known else None,
                         "remaining_seconds": round(deadline / 1000 - at, 3) if known else None})
    return {"scope": "nominal_checkin_schedule",
            "clock": "saved_launch_checkin_schedule_epoch_not_relay_monotonic",
            "clock_status": "known" if valid else "unknown", "clock_conflict": clock_conflict,
            "start_epoch_ms": start if valid else None, "deadline_epoch_ms": deadline if valid else None,
            "controls": rows, "unknown_controls": sum(row["remaining_seconds"] is None for row in rows),
            "meaning": "nominal_schedule_remainder_not_exact_runtime_budget; missing_conflicting_or_outside_schedule_is_unknown"}


def _anchored(path: Path) -> tuple[Path, Path]:
    """Resolve only the parent (e.g. macOS /tmp); safe_file still rejects a symlinked file."""
    path = Path(path).absolute()
    root = path.parent.resolve()
    return root / path.name, root


def summarize_s0(events_path: Path, *, projection_path: Path | None = None,
                 launch_path: Path | None = None) -> dict:
    """Read events and sibling usage metadata; optionally reuse a saved projection."""
    issues: Counter = Counter()
    events = _read_json(*_anchored(events_path), issues, lines=True)
    if not events and not issues:
        issues["empty_events"] += 1
    reliable = not issues
    peaks, finished = _prompt_peaks(events, reliable)
    unfinished = sum(row["unfinished_attempts"] for row in peaks.values())
    projection = _read_json(*_anchored(projection_path), issues) if projection_path is not None else None
    projection_matches = _projection_matches_events(projection, events)
    if projection is not None and not projection_matches:
        issues["projection_events_mismatch"] += 1
    relay_usage = _relay_usage(events_path, events, issues)
    usage_failed = relay_usage["control_evidence_failed"] is True or relay_usage["status"] == "unknown"
    policy = _policy_counts(events, reliable, issues, projection, unfinished_attempts=unfinished,
                            projection_matches=projection_matches, usage_evidence_failed=usage_failed)
    launch = _read_json(*_anchored(launch_path), issues, kind="launch") if launch_path is not None else None
    mode = launch.get("role_policy_mode") if isinstance(launch, dict) else None
    ledger_expected = any(e.get("event") == "role_policy_batch" for e in events) or mode in {
        "global", "master", "propose"}
    complete = reliable and (not ledger_expected or policy["ledger_run_ended"] is True)
    if not complete:
        for row in peaks.values():
            row["peak_prompt_tokens"] = row["peak_plus_reserved_output_65536"] = None
    reviews = _review_counts(projection, events)
    if reviews is not None and (not complete or unfinished or not projection_matches or usage_failed):
        reviews["counts"] = None
    if reviews is not None and not projection_matches:
        reviews["observed_counts"] = None
    return {"schema_version": 1, "scope": "offline_saved_evidence_only", "policy": policy,
            "prompt_peak_completeness_scope": "closed_policy_ledger_and_per_role_attempts" if ledger_expected else "observed_per_role_attempts_only_no_run_end_claim",
            "prompt_peaks_by_role": peaks, "prompt_peak_unit": "one_provider_attempt_including_retries_not_sum",
            "relay_usage_evidence": relay_usage,
            "prompt_cache": prompt_cache_projection(finished) if finished else None,
            "policy_review_projected_messages": reviews["counts"]["projected_messages"]
                if reviews and reviews["counts"] is not None else None,
            "observed_policy_review_projected_messages": reviews["observed_counts"]["projected_messages"]
                if reviews and reviews["observed_counts"] is not None else None,
            "policy_review_conclusions": reviews,
            "prompt_cache_scope": "deduplicated_observed_finished_attempts_only",
            "proposal_four_questions_assessment": None,
            "cache_change_after_commit": _cache_after_commit(events, reliable and policy["ledger_replay"] == "ok"),
            "remaining_time_at_control": _control_times(events, launch, reliable and policy["ledger_replay"] == "ok"),
            "single_extrapolation": None, "six_hour_extrapolation": None, "directed_smoke_R": None,
            "issues": {key: count for key, count in sorted(issues.items()) if count}}
