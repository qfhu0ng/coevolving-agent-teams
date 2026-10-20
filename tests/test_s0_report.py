"""Zero-API S0 report tests using real policy ledger transactions, never model bodies."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from agent_company.analysis.evidence_projection import project_evidence
from agent_company.policy.role_policy import PolicyPersistError, PolicyRejection, RolePolicyStore, parse_send_message, parse_spawn
from agent_company.analysis.s0_report import classify_usage, summarize_s0
from test_role_policy_relay import relay_for, spawn, turn
from test_team import call


def ledger(mode="propose"):
    events = []
    store = RolePolicyStore(mode, "Initial policy", events.append)
    return store, events


def commit(store, actor, request, calls):
    store.commit(store.stage(actor, store.snapshot(actor), calls, source_request=request))


def save(tmp_path, events):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def usage(role, attempt, prompt=None, **extra):
    return {"event": "provider_attempt_finished", "role": role, "attempt": attempt,
            "source_request": attempt, "upstream_started": True,
            "raw_usage": {"prompt_tokens": prompt, "prompt_cache_hit_tokens": 0}, **extra}


def projection_data(events, timeline):
    """Synthetic projection with the production attempt metadata and linked calls."""
    keys = ("event", "role", "attempt", "source_request", "purpose", "error_kind",
            "provider_http_status", "relay_http_status")
    return {"collaboration_timeline": [{"role": "master", "attempt": 1, **row} for row in timeline],
            "event_timeline": [{key: row[key] for key in keys if key in row} for row in events
                               if row.get("event") in {"provider_attempt_started", "provider_attempt_finished"}],
            "coverage": {"issues": {}}}


@pytest.mark.parametrize("attempts,commits,rejects,expected", [
    (0, 0, 0, "U0"), (1, 0, None, "U2"), (2, 0, 9, "U2"), (3, 0, 0, "U3"),
    (4, 0, None, "U3"), (3, 1, 1, "U2"), (3, 2, 1, "U1"),
    (3, 1, 0, "U1"), (3, 1, None, None), (None, 0, 0, None), (1, None, 0, None),
])
def test_d5_order_and_unknown(attempts, commits, rejects, expected):
    assert classify_usage(attempts, commits, rejects) == expected


def test_bad_counts_are_not_coerced():
    for args in ((-1, 0, 0), (True, 0, 0), (0, 1, 0), (1.0, 1, 0)):
        with pytest.raises(ValueError):
            classify_usage(*args)


@pytest.mark.parametrize("committed", [0, 1])
def test_unfinished_ledger_keeps_observations_but_not_run_totals_or_u(tmp_path, committed):
    store, events = ledger("global")
    if committed:
        commit(store, "master", 1, [parse_send_message("set", "*", "GLOBAL-POLICY\nNew")])
    events.append(usage("master", 1, 100))
    projection = tmp_path / "projection.json"
    projection.write_text(json.dumps(projection_data(events, [{"tool": "send_message",
        "fields": {"message": {"text": "POLICY-REVIEW C v1: no change. Observed fact."}}}])))
    report = summarize_s0(save(tmp_path, events), projection_path=projection)
    policy = report["policy"]
    assert policy["ledger_replay"] == "ok" and policy["ledger_run_ended"] is False
    assert report["issues"]["role_policy_run_not_ended"] == 1
    for field in ("control_attempts", "committed_controls", "usage_classification", "rejected_responses",
                  "nonpolicy_rejected_responses", "uncommitted_late_batches", "discarded_control_calls",
                  "format_rejected_responses", "rejections_by_code", "policy_version_updates", "initial_role_versions",
                  "proposals", "proposals_with_change_reason_evidence", "renotifications", "decisions_with_note",
                  "corrected_rejections", "uncorrected_format_rejections", "version_carry"):
        assert policy[field] is None, field
    assert set(policy["decisions"].values()) == {None}
    assert policy["observed_committed_controls"] == policy["observed_control_attempts_lower_bound"] == committed
    assert policy["observed_rejections_by_code"] == {}
    assert report["prompt_peaks_by_role"]["master"]["observed_peak_prompt_tokens"] == 100
    assert report["prompt_peaks_by_role"]["master"]["peak_prompt_tokens"] is None
    assert report["prompt_peaks_by_role"]["master"]["peak_plus_reserved_output_65536"] is None
    assert report["policy_review_conclusions"]["counts"] is None
    assert report["observed_policy_review_projected_messages"] == 1
    # The exact same prefix acquires run-wide counts only after the real ledger ends.
    store.end_run()
    ended = summarize_s0(save(tmp_path, events), projection_path=projection)
    assert ended["issues"] == {}
    assert ended["policy"]["ledger_run_ended"] is True
    assert ended["policy"]["control_attempts"] == ended["policy"]["committed_controls"] == committed
    assert ended["policy"]["usage_classification"] == ("U1" if committed else "U0")
    assert ended["prompt_peaks_by_role"]["master"]["peak_prompt_tokens"] == 100
    assert ended["policy_review_conclusions"]["counts"]["projected_messages"] == 1


def test_end_marker_must_be_replayed_not_a_top_level_event(tmp_path):
    _, events = ledger()
    events.append({"event": "role_policy_run_ended"})
    report = summarize_s0(save(tmp_path, events))
    assert report["policy"]["ledger_run_ended"] is False
    assert report["policy"]["usage_classification"] is None
    assert report["issues"]["role_policy_run_not_ended"] == 1


def test_closed_ledger_does_not_mean_an_inflight_provider_attempt_has_finished(tmp_path):
    store, events = ledger("global")
    events.append({"event": "provider_attempt_started", "purpose": "coding", "role": "master",
                   "attempt": 1, "source_request": 1})
    store.end_run()
    report = summarize_s0(save(tmp_path, events))
    policy = report["policy"]
    assert policy["ledger_run_ended"] is True and policy["committed_controls"] == 0
    assert policy["control_attempts"] is policy["usage_classification"] is None
    assert policy["rejected_responses"] is policy["uncommitted_late_batches"] is None
    assert policy["observed_committed_controls"] == 0
    assert policy["uncorrected_format_rejections"] is policy["corrected_rejections"] is None
    assert report["issues"]["unfinished_provider_attempts"] == 1
    events.append(usage("master", 1, 100))
    finished = summarize_s0(save(tmp_path, events))
    assert finished["issues"] == {}
    assert finished["policy"]["usage_classification"] == "U0"


def test_broken_ended_ledger_or_unreliable_events_keep_rejections_unknown(tmp_path):
    store, events = ledger("global")
    events.append({"event": "role_policy_message_rejected", "code": "malformed_header"})
    store.end_run()
    path = save(tmp_path, events)
    path.write_text(path.read_text() + "{truncated\n")
    report = summarize_s0(path)
    assert report["policy"]["ledger_run_ended"] is None
    assert report["policy"]["rejections_by_code"] is None
    assert report["policy"]["observed_rejections_by_code"] == {"malformed_header": 1}
    events[-1]["batch"] += 1
    report = summarize_s0(save(tmp_path, events))
    assert report["issues"]["role_policy_replay_failed"] == 1
    assert report["policy"]["ledger_run_ended"] is None
    assert report["policy"]["usage_classification"] is None


def test_proposal_decision_counts_and_actual_worker_carriage(tmp_path):
    store, events = ledger()
    commit(store, "master", 1, [parse_spawn("spawn", "worker", "Task\nROLE-POLICY:\nInitial\nEND-ROLE-POLICY")])
    store.record_first_request("worker")
    commit(store, "worker", 2, [parse_send_message("proposal", "leader",
        "POLICY-PROPOSAL\nCHANGE: New\nREASON: Reason\nEVIDENCE: Evidence")])
    commit(store, "master", 3, [parse_send_message("decision", "worker", "POLICY-DECISION #1 approve\nNOTE: Accepted")])
    events.extend([
        {"event": "role_policy_injected", "role": "worker", "source_request": 4, "admitted": True,
         "slot_versions": {"worker": 2}},
        usage("worker", 4, 500),
    ])
    store.end_run()
    result = summarize_s0(save(tmp_path, events))
    policy = result["policy"]
    assert policy["ledger_replay"] == "ok"
    assert policy["committed_controls"] == policy["control_attempts"] == 2
    assert policy["policy_version_updates"] == 1
    assert policy["initial_role_versions"] == policy["proposals"] == policy["decisions_with_note"] == 1
    assert policy["decisions"] == {"approve": 1, "reject": 0, "amend": 0}
    assert policy["usage_classification"] == "U1"
    assert policy["version_carry"]["observed_carried_to_worker_upstream"] == 1
    assert result["issues"] == {}


def test_injection_without_upstream_is_not_version_carriage(tmp_path):
    store, events = ledger("global")
    commit(store, "master", 1, [parse_send_message("update", "*", "GLOBAL-POLICY\nNew policy")])
    events.extend([{"event": "role_policy_injected", "role": "worker", "source_request": 2,
                    "admitted": True, "global_version": 2},
                   usage("worker", 2, upstream_started=False)])
    store.end_run()
    result = summarize_s0(save(tmp_path, events))
    assert result["policy"]["version_carry"]["observed_carried_to_worker_upstream"] == 0


def test_later_success_does_not_prove_correction_or_total_attempts(tmp_path):
    store, events = ledger("global")
    events.append({"event": "role_policy_message_rejected", "stage": "initial", "role": "master",
                   "source_request": 1, "tool_call_id": "bad", "code": "malformed_header", "discarded_calls": 4})
    commit(store, "master", 2, [parse_send_message("other", "*", "GLOBAL-POLICY\nUnrelated new policy")])
    store.end_run()
    result = summarize_s0(save(tmp_path, events))["policy"]
    assert result["committed_controls"] == 1
    assert result["rejected_responses"] == result["format_rejected_responses"] == 1
    assert result["control_attempts"] is result["corrected_rejections"] is result["usage_classification"] is None
    assert result["observed_control_attempts_lower_bound"] == 1


def test_prompt_peaks_are_per_request_not_sum_and_keep_missing_unknown(tmp_path):
    store, events = ledger()
    events.extend([usage("master", 1, 100), usage("master", 2, 150, source_request=1),
                   usage("worker", 3, 200), usage("worker", 4),
                   {"event": "provider_attempt_started", "role": "inflight", "attempt": 5}])
    store.end_run()
    result = summarize_s0(save(tmp_path, events))
    peaks = result["prompt_peaks_by_role"]
    assert peaks["master"]["peak_prompt_tokens"] == 150
    assert peaks["master"]["peak_plus_reserved_output_65536"] == 65686
    assert peaks["worker"]["observed_peak_prompt_tokens"] == 200
    assert peaks["worker"]["peak_prompt_tokens"] is None
    assert peaks["worker"]["attempts_without_prompt_usage"] == 1
    assert peaks["inflight"]["peak_prompt_tokens"] is None
    assert peaks["inflight"]["unfinished_attempts"] == 1
    assert result["single_extrapolation"] is result["six_hour_extrapolation"] is None


def test_duplicate_attempt_does_not_fake_a_peak_or_cache_denominator(tmp_path):
    store, events = ledger()
    events.extend([usage("master", 1, 100), usage("master", 1, 100)])
    store.end_run()
    result = summarize_s0(save(tmp_path, events))
    assert result["prompt_peaks_by_role"]["master"]["peak_prompt_tokens"] is None
    assert result["policy"]["ledger_run_ended"] is True
    assert result["prompt_cache"]["total"]["attempts"] == 1


@pytest.mark.parametrize("bad_prompt", [None, True, -1, float("nan"), 1.5])
def test_invalid_prompt_usage_is_unknown(tmp_path, bad_prompt):
    store, events = ledger()
    events.append(usage("master", 1, bad_prompt))
    store.end_run()
    result = summarize_s0(save(tmp_path, events))
    assert result["prompt_peaks_by_role"]["master"]["peak_prompt_tokens"] is None
    assert result["policy"]["ledger_run_ended"] is True
    json.dumps(result, allow_nan=False)


def test_broken_log_or_replay_never_claims_zero_or_valid_classification(tmp_path):
    store, events = ledger()
    store.end_run()
    clean = summarize_s0(save(tmp_path, events))
    assert clean["policy"]["ledger_run_ended"] is True
    assert clean["policy"]["usage_classification"] == "U0"
    path = save(tmp_path, events)
    path.write_text(path.read_text() + "{truncated\n")
    result = summarize_s0(path)
    assert result["policy"]["committed_controls"] is result["policy"]["usage_classification"] is None
    assert result["issues"]["invalid_event_lines"] == 1
    result = summarize_s0(save(tmp_path, events + events))
    assert result["issues"]["role_policy_replay_failed"] == 1
    assert result["policy"]["committed_controls"] is None


def test_missing_log_and_symlink_are_unknown(tmp_path):
    for path in (tmp_path / "missing.jsonl", tmp_path / "link.jsonl"):
        if path.name == "link.jsonl":
            path.symlink_to(save(tmp_path, []))
        result = summarize_s0(path)
        assert result["policy"]["committed_controls"] is None
        assert result["prompt_cache"] is None
        for key in ("observed_rejected_responses", "observed_nonpolicy_rejected_responses",
                    "observed_uncommitted_late_batches", "observed_rejections_by_code"):
            assert result["policy"][key] is None
        assert result["issues"]["missing_or_unsafe_events"] == 1


def test_only_explicit_metadata_inputs_are_read_and_no_policy_text_is_emitted(tmp_path, monkeypatch):
    store, events = ledger("global")
    commit(store, "master", 1, [parse_send_message("set", "*", "GLOBAL-POLICY\nPRIVATE_POLICY_CONTENT")])
    events.append(usage("master", 1, 100))
    store.end_run()
    path = save(tmp_path, events)
    projection = tmp_path / "projection.json"
    projection.write_text(json.dumps(projection_data(events, [
        {"tool": "send_message", "fields": {"message": {"text": "POLICY-REVIEW C v2: cannot judge yet\nPRIVATE_TOOL_BODY"}}},
        {"tool": "send_message", "fields": {"message": {"text": "POLICY-REVIEWER is not the review prefix"}}}])))
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self in {path, projection}, f"unexpected read: {self}"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = summarize_s0(path, projection_path=projection)
    assert report["policy_review_projected_messages"] == 1
    assert "PRIVATE" not in json.dumps(report)


def test_cli_outputs_json_without_requiring_provider(tmp_path):
    store, events = ledger()
    store.end_run()
    path = save(tmp_path, events)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/report_programbench_s0.py"), str(path)],
                            check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["scope"] == "offline_saved_evidence_only"
    assert report["policy"]["usage_classification"] == "U0"


def injection(role, request, version, **extra):
    return {"event": "role_policy_injected", "role": role, "source_request": request,
            "admitted": True, "global_version": version, **extra}


def cached(role, attempt, prompt, hit, **extra):
    return usage(role, attempt, raw_usage={"prompt_tokens": prompt, "prompt_cache_hit_tokens": hit}, **extra)


def cache_evidence(*, ended=False):
    store, events = ledger("global")
    events.append(injection("worker", 1, 1))
    events.append(cached("worker", 1, 100, 80))
    commit(store, "master", 2, [parse_send_message("update", "*", "GLOBAL-POLICY\nNew policy")])
    events.append(injection("worker", 3, 2))
    events.append(cached("worker", 3, 200, 100))
    if ended:
        store.end_run()
    return events


def test_cache_uses_injected_version_for_late_finish_retries_and_role_lag(tmp_path):
    events = cache_evidence()
    old_finish = next(e for e in events if e.get("attempt") == 1)
    events.remove(old_finish)
    events.extend([old_finish, cached("worker", 4, 300, 240, source_request=1),
                   injection("lagging", 5, 1), cached("lagging", 5, 100, 90)])
    cache = summarize_s0(save(tmp_path, events))["cache_change_after_commit"]
    row = cache["versions"][0]["by_role"]["worker"]
    assert row["pre_last_old_version"]["source_request"] == 1
    assert row["pre_last_old_version"]["finished_attempts"] == 2
    assert row["pre_last_old_version"]["prompt_tokens"] == 400
    assert row["pre_last_old_version"]["uncached_share"] == pytest.approx(0.2)
    assert row["post_first_new_version"]["source_request"] == 3
    assert row["uncached_share_change"] == pytest.approx(0.3)
    assert cache["versions"][0]["by_role"]["lagging"]["post_first_new_version"] is None
    assert cache["causal_effect"] == "unknown"


@pytest.mark.parametrize("fault", ["missing_cache", "zero_prompt", "duplicate_injection",
                                   "duplicate_finish", "attempt_conflict", "unfinished", "not_sent"])
def test_cache_missing_or_ambiguous_metadata_cannot_form_a_change(tmp_path, fault):
    events = cache_evidence()
    post = next(e for e in events if e.get("attempt") == 3)
    if fault == "missing_cache":
        del post["raw_usage"]["prompt_cache_hit_tokens"]
    elif fault == "zero_prompt":
        post["raw_usage"] = {"prompt_tokens": 0, "prompt_cache_hit_tokens": 0}
    elif fault == "duplicate_injection":
        events.append(injection("worker", 3, 2))
    elif fault == "duplicate_finish":
        events.append(dict(post))
    elif fault == "attempt_conflict":
        events.append(cached("other", 3, 200, 100, source_request=7))
    elif fault == "unfinished":
        events.append({"event": "provider_attempt_started", "role": "worker", "source_request": 3, "attempt": 4})
    else:
        post["upstream_started"] = False
    row = summarize_s0(save(tmp_path, events))["cache_change_after_commit"]["versions"][0]["by_role"]["worker"]
    assert row["uncached_share_change"] is None
    assert row["post_first_new_version"]["uncached_share"] is None


def test_cache_role_policy_does_not_treat_unrelated_workers_as_updated(tmp_path):
    store, events = ledger()
    commit(store, "master", 1, [parse_spawn("spawn", "worker", "Task\nROLE-POLICY:\nInitial\nEND-ROLE-POLICY")])
    store.record_first_request("worker")
    events.extend([injection("worker", 2, 1, slot_versions={"worker": 1}), cached("worker", 2, 100, 90)])
    commit(store, "worker", 3, [parse_send_message("proposal", "leader",
        "POLICY-PROPOSAL\nCHANGE: New\nREASON: Reason\nEVIDENCE: Evidence")])
    commit(store, "master", 4, [parse_send_message("decision", "worker", "POLICY-DECISION #1 approve")])
    events.extend([injection("worker", 5, 1, slot_versions={"worker": 2}), cached("worker", 5, 100, 20),
                   injection("unrelated", 6, 1, slot_versions={"unrelated": 1}), cached("unrelated", 6, 100, 50)])
    version = summarize_s0(save(tmp_path, events))["cache_change_after_commit"]["versions"][0]
    assert version["scope"] == "role"
    assert set(version["by_role"]) == {"worker"}
    assert version["by_role"]["worker"]["uncached_share_change"] == pytest.approx(0.7)


def test_review_distribution_keeps_rejections_format_and_coverage_separate(tmp_path):
    store, events = ledger()
    events.append(usage("master", 1, 100))
    store.end_run()
    conclusions = ("improved", "no change", "side effects", "cannot judge yet")
    def entry(message, **extra):
        return {"tool": "send_message", "fields": {"message": {"text": message}}, **extra}
    rows = [entry(f"POLICY-REVIEW C v2: {conclusion}. PRIVATE_FACT") for conclusion in conclusions]
    rows += [entry("POLICY-REVIEW C v2: improved. PRIVATE", execution_status="rejected_before_execution"),
             entry(" POLICY-REVIEW C v2: improved. PRIVATE"), entry("POLICY-REVIEW C v2: improved"),
             entry("POLICY-REVIEW C v0: improved. PRIVATE"), entry("POLICY-REVIEW C v2: improvedness. PRIVATE"),
             entry("POLICY-REVIEWER C v2: improved. PRIVATE")]
    truncated = entry("POLICY-REVIEW C v2: improved. PRIVATE")
    truncated["fields"]["message"]["truncated"] = True
    rows.append(truncated)
    projection = tmp_path / "projection.json"
    evidence = projection_data(events, rows)
    projection.write_text(json.dumps(evidence))
    report = summarize_s0(save(tmp_path, events), projection_path=projection)
    counts = report["policy_review_conclusions"]["counts"]
    assert counts == {"projected_messages": 10, "rejected_before_execution": 1, "cancelled_before_delivery": 0, "format_unknown": 5,
                      "conclusions": dict.fromkeys(conclusions, 1)}
    assert "PRIVATE" not in json.dumps(report)
    evidence["coverage"]["issues"] = {"missing_files": 1}
    projection.write_text(json.dumps(evidence))
    partial = summarize_s0(save(tmp_path, events), projection_path=projection)["policy_review_conclusions"]
    assert partial["counts"] is None
    assert partial["observed_counts"] == counts


def schedule(tmp_path, **extra):
    path = tmp_path / "launch.json"
    path.write_text(json.dumps({"execution_mode": "native_team", "timeout_seconds": 100,
        "environment": {"AGENT_COMPANY_TASK_STARTED_EPOCH_MS": "1000000",
                        "AGENT_COMPANY_TASK_DEADLINE_EPOCH_MS": "1100000", "PRIVATE_KEY": "PRIVATE_SECRET"},
        "command": ["PRIVATE_COMMAND"], **extra}))
    return path


def timed_controls():
    store, events = ledger("global")
    commit(store, "master", 1, [parse_send_message("update", "*", "GLOBAL-POLICY\nNew policy")])
    for event in events:
        event["time"] = 1050
    events.append({"event": "role_policy_message_rejected", "time": 1060, "role": "master",
                   "source_request": 2, "tool_call_id": "bad", "code": "malformed_header"})
    return events


def test_control_times_use_only_saved_nominal_schedule_and_no_environment_output(tmp_path):
    result = summarize_s0(save(tmp_path, timed_controls()), launch_path=schedule(tmp_path))
    clock = result["remaining_time_at_control"]
    assert clock["scope"] == "nominal_checkin_schedule"
    assert clock["clock_status"] == "known"
    assert [row["remaining_seconds"] for row in clock["controls"]] == [50, 40]
    assert "PRIVATE" not in json.dumps(result)
    assert result["proposal_four_questions_assessment"] is result["six_hour_extrapolation"] is None


@pytest.mark.parametrize("fault", ["no_launch", "duration", "backwards", "outside", "missing_time", "duplicate_key"])
def test_ambiguous_schedule_is_unknown_not_clamped(tmp_path, fault):
    events = timed_controls()
    launch = schedule(tmp_path)
    if fault == "no_launch":
        launch = None
    elif fault == "duration":
        launch = schedule(tmp_path, timeout_seconds=90)
    elif fault == "backwards":
        events[-1]["time"] = 1040
    elif fault == "outside":
        events[-1]["time"] = 1200
    elif fault == "missing_time":
        del events[-1]["time"]
    else:
        launch.write_text('{"timeout_seconds":100,"timeout_seconds":90}')
    report = summarize_s0(save(tmp_path, events), launch_path=launch)
    clock = report["remaining_time_at_control"]
    assert clock["controls"][-1]["remaining_seconds"] is None
    if fault == "backwards":
        assert clock["clock_conflict"] is True
        assert all(row["remaining_seconds"] is None for row in clock["controls"])


def test_duplicate_event_keys_make_metadata_unknown(tmp_path):
    events = cache_evidence(ended=True)
    clean = summarize_s0(save(tmp_path, events))
    assert clean["policy"]["ledger_run_ended"] is True
    assert clean["policy"]["usage_classification"] is not None
    path = save(tmp_path, events)
    path.write_text(path.read_text() + '{"event":"role_policy_injected","source_request":3,"source_request":4}\n')
    report = summarize_s0(path)
    assert report["issues"]["invalid_event_lines"] == 1
    assert report["cache_change_after_commit"]["status"] == "unknown"
    assert report["policy"]["usage_classification"] is None


def test_cli_accepts_explicit_launch_and_emits_no_extra_fields(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path, launch = save(tmp_path, timed_controls()), schedule(tmp_path)
    result = subprocess.run([sys.executable, str(root / "scripts/report_programbench_s0.py"), str(path),
                             "--launch", str(launch)], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["remaining_time_at_control"]["controls"][0]["remaining_seconds"] == 50
    assert "PRIVATE" not in result.stdout


# ----- discards without a role-policy rejection event (2026-10-14) -----------------------

def bad_spawn(relay):
    """Rejected before staging: the delivery repository is not the assigned worktree."""
    return call("agent", {"name": "decoder", "description": "d", "prompt": "Work.",
                          "working_dir": relay.registry.workspace.repo_path})


def project(tmp_path, relay):
    """The allowlist projection of this relay's own saved evidence."""
    trial = tmp_path / "trial"
    (trial / "team").mkdir(parents=True)
    shutil.copytree(relay.output, trial / "team/relay")
    (trial / "task_result.json").write_text("{}")
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(project_evidence(trial)))
    return path


def test_controls_discarded_before_staging_are_attempts_not_u0(tmp_path):
    relay, upstream = relay_for(tmp_path, "global")
    for _ in range(3):
        status, _, _ = turn(relay, upstream, "master", [
            call("send_message", {"to": "*", "message": "GLOBAL-POLICY\nRule."}), bad_spawn(relay)])
        assert status == 200
    relay.role_policy.end_run()
    events = tmp_path / "logs/events.jsonl"
    policy = summarize_s0(events)["policy"]
    assert (policy["committed_controls"], policy["rejected_responses"], policy["nonpolicy_rejected_responses"],
            policy["uncommitted_late_batches"]) == (0, 0, 3, 0)
    assert policy["control_attempts"] is policy["usage_classification"] is policy["discarded_control_calls"] is None
    assert policy["observed_control_attempts_lower_bound"] == 0
    policy = summarize_s0(events, projection_path=project(tmp_path, relay))["policy"]
    assert policy["discarded_control_calls"] == policy["control_attempts"] == 3
    assert policy["observed_control_attempts_lower_bound"] == 3
    assert policy["usage_classification"] == "U3"


def test_projection_adds_only_discarded_controls_to_committed_ones(tmp_path):
    relay, upstream = relay_for(tmp_path, "global")
    turn(relay, upstream, "master", [call("send_message", {"to": "*", "message": "GLOBAL-POLICY\nRule one."})])
    turn(relay, upstream, "master", [call("send_message", {"to": "*", "message": "GLOBAL-POLICY\nRule two."}),
                                     call("send_message", {"to": "*", "message": "Ordinary note."}), bad_spawn(relay)])
    relay.role_policy.end_run()
    report = summarize_s0(tmp_path / "logs/events.jsonl", projection_path=project(tmp_path, relay))
    policy = report["policy"]
    assert (policy["committed_controls"], policy["discarded_control_calls"], policy["control_attempts"]) == (1, 1, 2)
    assert policy["usage_classification"] == "U1"
    assert report["issues"] == {}


@pytest.mark.parametrize("fault", ["partial_coverage", "unclassified_header", "no_rejection_event"])
def test_discards_the_projection_cannot_account_for_leave_attempts_unknown(tmp_path, fault):
    relay, upstream = relay_for(tmp_path, "global")
    # The bad spawn comes first, so the relay never parses the malformed header after it.
    message = "**GLOBAL-POLICY**\nRule." if fault == "unclassified_header" else "GLOBAL-POLICY\nRule."
    turn(relay, upstream, "master", [bad_spawn(relay), call("send_message", {"to": "*", "message": message})])
    relay.role_policy.end_run()
    events, projection = tmp_path / "logs/events.jsonl", project(tmp_path, relay)
    if fault == "partial_coverage":
        evidence = json.loads(projection.read_text())
        evidence["coverage"]["issues"] = {"missing_files": 1}
        projection.write_text(json.dumps(evidence))
    elif fault == "no_rejection_event":  # e.g. a projection of another run
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        for row in rows:
            if row.get("error_kind") == "tool_response_rejected":
                row["error_kind"] = None
        events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    policy = summarize_s0(events, projection_path=projection)["policy"]
    assert policy["discarded_control_calls"] == (0 if fault == "unclassified_header" else None)
    assert policy["control_attempts"] is policy["usage_classification"] is None
    assert policy["observed_control_attempts_lower_bound"] == 0


def test_a_stopped_late_commit_keeps_attempts_unknown_even_with_a_projection(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    store, original_stage = relay.role_policy, relay.role_policy.stage

    def stage_then_close(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        relay.close()
        return staged
    store.stage = stage_then_close
    status, _, _ = turn(relay, upstream, "master",
                        [call("send_message", {"to": "decoder", "message": "ROLE-POLICY decoder\nV2"})])
    assert status == 424
    relay.role_policy.end_run()
    events = tmp_path / "logs/events.jsonl"
    # The delivered file is not rewritten, so the projection shows no discarded control.
    for projection, discarded in ((None, None), (project(tmp_path, relay), 0)):
        policy = summarize_s0(events, projection_path=projection)["policy"]
        assert (policy["committed_controls"], policy["uncommitted_late_batches"]) == (0, 1)
        assert policy["discarded_control_calls"] == discarded
        assert policy["control_attempts"] is policy["usage_classification"] is None
        assert policy["observed_control_attempts_lower_bound"] == 0


def test_paths_through_a_symlinked_parent_are_read_but_a_symlinked_file_is_not(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    store, events = ledger("global")
    commit(store, "master", 1, [parse_send_message("update", "*", "GLOBAL-POLICY\nNew policy")])
    store.end_run()
    save(real, events)
    (real / "projection.json").write_text(json.dumps(projection_data(events, [])))
    schedule(real)
    linked = tmp_path / "link"
    linked.symlink_to(real, target_is_directory=True)  # like macOS /tmp -> /private/tmp
    report = summarize_s0(linked / "events.jsonl", projection_path=linked / "projection.json",
                          launch_path=linked / "launch.json")
    assert report["issues"] == {}
    assert report["policy"]["control_attempts"] == 1 and report["policy_review_projected_messages"] == 0
    assert report["remaining_time_at_control"]["clock_status"] == "known"
    for name in ("events.jsonl", "projection.json", "launch.json"):
        (tmp_path / name).symlink_to(real / name)
    report = summarize_s0(tmp_path / "events.jsonl", projection_path=tmp_path / "projection.json",
                          launch_path=tmp_path / "launch.json")
    assert {key: report["issues"].get(key) for key in (
        "missing_or_unsafe_events", "missing_or_unsafe_projection", "missing_or_unsafe_launch")} == dict.fromkeys(
        ("missing_or_unsafe_events", "missing_or_unsafe_projection", "missing_or_unsafe_launch"), 1)
    assert report["policy"]["committed_controls"] is None


@pytest.mark.parametrize("execution_mode,role,mode", [("single", "single", "off"), ("team", "master", "off"), ("team", "master", None)])
def test_no_policy_ledger_preserves_observed_per_role_peaks(tmp_path, execution_mode, role, mode):
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"execution_mode": execution_mode, "role_policy_mode": mode}))
    result = summarize_s0(save(tmp_path, [usage(role, 1, 100)]), launch_path=launch)
    assert result["policy"]["ledger_run_ended"] is None
    assert result["prompt_peaks_by_role"][role]["peak_prompt_tokens"] == 100
    assert result["prompt_peak_completeness_scope"] == "observed_per_role_attempts_only_no_run_end_claim"
    launch.write_text(json.dumps({"execution_mode": "team", "role_policy_mode": "propose"}))
    assert summarize_s0(save(tmp_path, [usage(role, 1, 100)]), launch_path=launch)["prompt_peaks_by_role"][role]["peak_prompt_tokens"] is None


@pytest.mark.parametrize("fault", ["missing_attempt", "extra_attempt", "role", "source_request", "error_kind",
                                 "relay_http_status", "call_attempt", "call_role", "missing_event_timeline"])
def test_projection_attempt_mismatch_never_supplies_discarded_control_counts(tmp_path, fault):
    relay, upstream = relay_for(tmp_path, "global")
    turn(relay, upstream, "master", [
        call("send_message", {"to": "*", "message": "GLOBAL-POLICY\nRule."}), bad_spawn(relay),
        call("send_message", {"to": "*", "message": "POLICY-REVIEW C v1: improved. Synthetic fact."})])
    relay.role_policy.end_run()
    path = project(tmp_path, relay)
    bound = summarize_s0(tmp_path / "logs/events.jsonl", projection_path=path)
    assert bound["observed_policy_review_projected_messages"] == 1
    assert bound["policy_review_conclusions"]["observed_counts"]["projected_messages"] == 1
    data = json.loads(path.read_text())
    if fault == "missing_event_timeline":
        data.pop("event_timeline")
    elif fault == "missing_attempt":
        data["event_timeline"].pop()
    elif fault == "extra_attempt":
        data["event_timeline"].append({"event": "provider_attempt_finished", "role": "master", "attempt": 99})
    elif fault in {"call_attempt", "call_role"}:
        data["collaboration_timeline"][0][fault.removeprefix("call_")] = 99 if fault == "call_attempt" else "other"
    else:
        row = next(row for row in data["event_timeline"] if row["event"] == "provider_attempt_finished")
        row[fault] = {"role": "other", "source_request": 99, "error_kind": {"text": "other"}, "relay_http_status": 424}[fault]
    path.write_text(json.dumps(data))
    report = summarize_s0(tmp_path / "logs/events.jsonl", projection_path=path)
    policy = report["policy"]
    assert report["issues"]["projection_events_mismatch"] == 1
    assert policy["discarded_control_calls"] is policy["observed_discarded_control_calls"] is None
    assert policy["control_attempts"] is policy["usage_classification"] is None
    assert policy["observed_control_attempts_lower_bound"] == 0
    assert report["observed_policy_review_projected_messages"] is None
    assert report["policy_review_conclusions"]["counts"] is None
    assert report["policy_review_conclusions"]["observed_counts"] is None


@pytest.mark.parametrize("mode,code", [
    ("global", "malformed_header"), ("master", "malformed_header"), ("propose", "malformed_header"),
    ("master", "not_authorized"), ("propose", "not_authorized"),
])
def test_real_recorded_policy_rejection_is_not_a_usage_evidence_failure(tmp_path, mode, code):
    relay, upstream = relay_for(tmp_path, mode)
    role, recipient, message = "master", "*", "**GLOBAL-POLICY**\nRule."
    if code == "not_authorized":
        spawn(relay, upstream, "decoder")
        role, recipient, message = "decoder", "decoder", "ROLE-POLICY decoder\nRule."
    status, _, _ = turn(relay, upstream, role, [call("send_message", {"to": recipient, "message": message})])
    assert status == 200
    relay.role_policy.end_run()
    relay.save(relay.output / "usage.json", relay.summary())
    summary = json.loads((relay.output / "usage.json").read_text())
    (error,) = [row for row in summary["errors"] if row.get("tool_rejection", {}).get("reason") == "role_policy"]
    assert error["tool_rejection"]["code"] == code
    events = [json.loads(line) for line in (relay.output / "events.jsonl").read_text().splitlines()]
    (rejection,) = [row for row in events if row.get("event") == "role_policy_message_rejected"]
    assert (rejection["role"], rejection["source_request"]) == (error["role"], error["source_request"])
    report = summarize_s0(relay.output / "events.jsonl", projection_path=project(tmp_path, relay))
    assert report["relay_usage_evidence"]["control_evidence_failed"] is False
    assert "relay_control_evidence_failed" not in report["issues"]
    assert report["policy"]["rejections_by_code"] == {code: 1}
    assert report["policy"]["committed_controls"] == 0


def test_real_initial_rejection_event_write_failure_is_unknown_from_saved_usage(tmp_path, monkeypatch):
    relay, upstream = relay_for(tmp_path, "global")
    original_event = relay.event
    failed_events = []

    def fail_initial_rejection(event, **fields):
        if event == "role_policy_message_rejected" and fields.get("stage") == "initial":
            failed_events.append(fields)
            raise OSError("synthetic initial rejection event write failure")
        return original_event(event, **fields)

    monkeypatch.setattr(relay, "event", fail_initial_rejection)
    status, delivered, _ = turn(relay, upstream, "master", [
        call("send_message", {"to": "*", "message": "**GLOBAL-POLICY**\nRule."})])
    assert status == 424 and delivered["error"]["code"] == "tool_response_rejected"
    assert len(failed_events) == 1 and failed_events[0]["code"] == "malformed_header"
    relay.role_policy.end_run()
    relay.save(relay.output / "usage.json", relay.summary())
    saved_usage = json.loads((relay.output / "usage.json").read_text())
    (error,) = saved_usage["errors"]
    assert error["tool_rejection"]["reason"] == "role_policy"
    assert (error["role"], error["source_request"]) == ("master", failed_events[0]["source_request"])
    events = [json.loads(line) for line in (relay.output / "events.jsonl").read_text().splitlines()]
    assert not any(row.get("event") == "role_policy_message_rejected" for row in events)
    report = summarize_s0(relay.output / "events.jsonl", projection_path=project(tmp_path, relay))
    assert report["policy"]["ledger_run_ended"] is True
    assert report["relay_usage_evidence"]["control_evidence_failed"] is True
    assert report["issues"]["relay_control_evidence_failed"] == 1
    assert report["policy"]["control_attempts"] is report["policy"]["usage_classification"] is None
    assert report["policy"]["observed_committed_controls"] == 0


@pytest.mark.parametrize("fault", ["missing", "role", "source_request", "missing_identity", "invalid_request"])
def test_usage_policy_rejection_needs_matching_role_and_source_request(tmp_path, fault):
    store, events = ledger("global")
    store.end_run()
    error = {"role": "master", "source_request": 1, "tool_rejection": {"reason": "role_policy"}}
    rejection = {"event": "role_policy_message_rejected", "code": "malformed_header", **error}
    if fault == "role":
        rejection["role"] = "other"
    elif fault == "source_request":
        rejection["source_request"] = 2
    elif fault == "missing_identity":
        error.pop("source_request")
        rejection.pop("source_request")
    elif fault == "invalid_request":
        error["source_request"] = rejection["source_request"] = True
    if fault != "missing":
        events.append(rejection)
    (tmp_path / "usage.json").write_text(json.dumps({"errors": [error]}))
    report = summarize_s0(save(tmp_path, events))
    assert report["relay_usage_evidence"]["control_evidence_failed"] is True
    assert report["issues"]["relay_control_evidence_failed"] == 1
    assert report["policy"]["control_attempts"] is report["policy"]["usage_classification"] is None


@pytest.mark.parametrize("error", [
    {"role_policy": True, "error": "PRIVATE_ERROR"},
    {"error_kind": "evidence_write_failed", "error": "PRIVATE_ERROR"},
    {"tool_rejection": {"reason": "role_policy"}, "error": "PRIVATE_ERROR"},
])
def test_relay_usage_errors_invalidate_control_totals_without_leaking_errors(tmp_path, error):
    store, events = ledger("global")
    store.end_run()
    if error.get("role_policy") is True or error.get("error_kind") == "evidence_write_failed":
        # A matching ordinary rejection never cancels an explicit write failure.
        error = {**error, "role": "master", "source_request": 1}
        events.append({"event": "role_policy_message_rejected", "code": "malformed_header",
                       "role": "master", "source_request": 1})
    (tmp_path / "usage.json").write_text(json.dumps({"errors": [error]}))
    report = summarize_s0(save(tmp_path, events))
    assert report["policy"]["control_attempts"] is report["policy"]["usage_classification"] is None
    assert report["policy"]["observed_committed_controls"] == 0
    assert report["issues"]["relay_control_evidence_failed"] == 1
    assert report["relay_usage_evidence"]["control_evidence_failed"] is True
    assert "PRIVATE_ERROR" not in json.dumps(report)


def test_relay_usage_policy_failure_counter_also_invalidates_control_totals(tmp_path):
    store, events = ledger("global")
    store.end_run()
    (tmp_path / "usage.json").write_text(json.dumps({"errors": [], "role_policy": {"counts": {"evidence_failures": 1}}}))
    report = summarize_s0(save(tmp_path, events))
    assert report["policy"]["usage_classification"] is None
    assert report["issues"]["relay_control_evidence_failed"] == 1


@pytest.mark.parametrize("kind", ["missing", "unrelated", "invalid", "symlink"])
def test_relay_usage_missing_unsafe_and_unrelated_evidence_are_distinct(tmp_path, kind):
    store, events = ledger("global")
    store.end_run()
    path = tmp_path / "usage.json"
    if kind == "invalid":
        path.write_text("{")
    elif kind == "symlink":
        target = tmp_path / "other.json"
        target.write_text(json.dumps({"errors": []}))
        path.symlink_to(target)
    elif kind == "unrelated":
        path.write_text(json.dumps({"errors": [{"error_kind": "provider_error"}]}))
    report = summarize_s0(save(tmp_path, events))
    if kind in {"invalid", "symlink"}:
        assert report["policy"]["usage_classification"] is None
        assert report["relay_usage_evidence"]["status"] == "unknown"
    else:
        assert report["policy"]["usage_classification"] == "U0"
        assert report["relay_usage_evidence"]["status"] == ("not_recorded" if kind == "missing" else "read")
    assert "cannot_rule_out_unrecorded_control_discards" in report["relay_usage_evidence"]["scope"]


@pytest.mark.parametrize("failure", ["commit_persist", "late_rejection_write"])
def test_real_late_policy_failure_without_events_is_detected_from_saved_usage(tmp_path, monkeypatch, failure):
    relay, upstream = relay_for(tmp_path, "global")
    original = relay.role_policy.commit
    def fail_commit(*args, **kwargs):
        if failure == "commit_persist":
            raise PolicyPersistError("synthetic write failure")
        raise PolicyRejection("stale_version", "synthetic late failure")
    monkeypatch.setattr(relay.role_policy, "commit", fail_commit)
    if failure == "late_rejection_write":
        def fail_rejection(*args, **kwargs):
            raise OSError("synthetic event write failure")
        monkeypatch.setattr(relay, "_policy_rejected", fail_rejection)
    status, delivered, _ = turn(relay, upstream, "master", [
        call("send_message", {"to": "*", "message": "GLOBAL-POLICY\nRule."})])
    assert status == 424 and delivered["error"]["code"] == "evidence_write_failed"
    monkeypatch.setattr(relay.role_policy, "commit", original)
    relay.role_policy.end_run()
    # A later persisted relay summary contains the otherwise unlogged failure.
    relay.save(relay.output / "usage.json", relay.summary())
    events = [json.loads(line) for line in (relay.output / "events.jsonl").read_text().splitlines()]
    assert not any(row.get("event") in {"role_policy_message_rejected", "native_request_ended"} for row in events)
    report = summarize_s0(relay.output / "events.jsonl", projection_path=project(tmp_path, relay))
    assert report["policy"]["usage_classification"] is report["policy"]["control_attempts"] is None
    assert report["issues"]["relay_control_evidence_failed"] == 1


def test_real_cancelled_late_batch_review_is_excluded_from_conclusions(tmp_path, monkeypatch):
    relay, upstream = relay_for(tmp_path, "global")
    original_stage = relay.role_policy.stage
    def stage_then_close(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        relay.close()
        return staged
    monkeypatch.setattr(relay.role_policy, "stage", stage_then_close)
    status, _, _ = turn(relay, upstream, "master", [
        call("send_message", {"to": "*", "message": "GLOBAL-POLICY\nRule."}),
        call("send_message", {"to": "*", "message": "POLICY-REVIEW C v1: improved. Synthetic fact."})])
    assert status == 424
    relay.role_policy.end_run()
    report = summarize_s0(relay.output / "events.jsonl", projection_path=project(tmp_path, relay))
    counts = report["policy_review_conclusions"]["counts"]
    assert counts["projected_messages"] == counts["cancelled_before_delivery"] == 1
    assert counts["conclusions"] == dict.fromkeys(("improved", "no change", "side effects", "cannot judge yet"), 0)


@pytest.mark.parametrize("fault", ["status_424", "unlinked_late_batch"])
def test_cancelled_review_status_and_ambiguous_link_do_not_claim_delivery(tmp_path, fault):
    store, events = ledger("global")
    events.append(usage("master", 1, 100, relay_http_status=424 if fault == "status_424" else 200))
    if fault == "unlinked_late_batch":
        events.append({"event": "native_request_ended", "role": "master", "source_request": 99,
                       "role_policy_batch_committed": False})
    store.end_run()
    data = projection_data(events, [{"tool": "send_message", "fields": {"message": {
        "text": "POLICY-REVIEW C v1: improved. Synthetic fact."}}}])
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(data))
    review = summarize_s0(save(tmp_path, events), projection_path=path)["policy_review_conclusions"]
    if fault == "status_424":
        assert review["counts"]["cancelled_before_delivery"] == 1
        assert review["counts"]["conclusions"]["improved"] == 0
    else:
        assert review["counts"] is None
