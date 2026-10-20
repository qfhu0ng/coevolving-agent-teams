"""Schema 4 evidence projection: role policies and prompt cache (S3).

Specification: docs/stage1_role_policy_harness_plan_20261009.md v3.3 sections 3.3.1 and 3.6.
"""
import json
import shutil

from agent_company.analysis.evidence_projection import (
    SCHEMA_VERSION, project_evidence, prompt_cache_projection)
from test_role_policy_integration import run


def trial_from(tmp_path, output):
    trial = tmp_path / "trial"
    (trial / "team").mkdir(parents=True)
    shutil.copytree(output / "relay", trial / "team/relay")
    for name in ("qwen_stderr.txt", "summary.json"):
        shutil.copy(output / name, trial / "team" / name)
    return trial


def test_projection_of_a_real_proposal_run(tmp_path):
    scenario, result, lifecycle, events, output = run(tmp_path, "proposal", "propose")
    report = project_evidence(trial_from(tmp_path, output))
    assert report["schema_version"] == SCHEMA_VERSION == 5
    policies = report["role_policies"]
    assert (policies["status"], policies["mode"], policies["ledger"]["replay"]) == ("on", "propose", "ok")
    assert policies["ledger"]["instances"]["alpha"]["lifecycle"] == "terminal"
    versions = [(row["slot"], row["version"], row["provenance"]) for row in policies["versions"]]
    assert versions == [("alpha", 1, "spawn_initial"), ("alpha", 2, "proposal_approved")]
    assert all(row["first_carry_latency_s"] is not None and row["first_carry_latency_s"] >= 0
               for row in policies["versions"])
    (proposal,) = policies["proposals"]["rows"]
    assert (proposal["status"], proposal["decided_version"], proposal["proposer"]) == ("approved", 2, "alpha")
    assert proposal["change"]["text"].startswith("Message the leader only for blockers")
    assert proposal["first_carry_to_master_latency_s"] >= 0 and proposal["decision_latency_s"] >= 0
    assert policies["proposals"]["decision_rate"] == policies["proposals"]["approval_rate"] == 1.0
    assert policies["notifications"]["sent"] >= 2 and policies["notifications"]["unknown"] == 0
    assert policies["injection"]["alpha"]["admitted"] >= 3 and policies["injection"]["master"]["admitted"] >= 3
    assert policies["lifecycle_tail"]["terminal_events"] >= 1 and policies["lifecycle_tail"]["voided_at_run_end"] == 0
    timeline = report["collaboration_timeline"]
    controls = sorted(entry["policy_control"] for entry in timeline if entry.get("policy_control"))
    assert controls == ["POLICY-DECISION", "POLICY-PROPOSAL"]
    proposal_entry = next(entry for entry in timeline if entry.get("policy_control") == "POLICY-PROPOSAL")
    assert proposal_entry["role_policy_version"]["version"] == 1
    assert all("source_request" in row for row in report["event_timeline"] if row["event"] == "provider_attempt_finished")
    cache = report["prompt_cache"]["total"]
    assert cache["attempts"] >= 1 and cache["attempts_reporting_cache"] == 0  # the scripted provider reports none
    assert cache["cached_share_of_reporting_prompt"] is None
    json.dumps(report, allow_nan=False)


def test_off_and_old_runs_are_distinguished(tmp_path):
    scenario, result, lifecycle, events, output = run(tmp_path, "ordinary_silent", "off")
    report = project_evidence(trial_from(tmp_path, output))
    assert report["role_policies"] == {"status": "off", "mode": "off"}
    legacy = tmp_path / "legacy"
    (legacy / "team/relay").mkdir(parents=True)
    (legacy / "team/relay/events.jsonl").write_text(json.dumps({"event": "logical_model_call_admitted"}) + "\n")
    assert project_evidence(legacy)["role_policies"] == {"status": "not_instrumented", "mode": None}


def test_prompt_cache_reporting_keeps_missing_fields_unknown():
    def finished(role, raw):
        return {"event": "provider_attempt_finished", "role": role, "raw_usage": raw}
    events = [finished("master", {"prompt_tokens": 100, "prompt_cache_hit_tokens": 80}),
              finished("master", {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 10}}),
              finished("master", {"prompt_tokens": 70}),
              finished("w1", None),
              finished("w1", {"prompt_tokens": 10, "prompt_cache_hit_tokens": 20})]  # inconsistent: not counted
    cache = prompt_cache_projection(events)
    master = cache["by_role"]["master"]
    assert (master["attempts_reporting_cache"], master["prompt_tokens_reporting_cache"], master["cached_prompt_tokens"]) \
        == (2, 150, 90)
    assert master["prompt_tokens_without_cache_field"] == 70 and master["cached_share_of_reporting_prompt"] == 0.6
    worker = cache["by_role"]["w1"]
    assert (worker["attempts_without_usage"], worker["attempts_reporting_cache"]) == (1, 0)
    assert worker["cached_share_of_reporting_prompt"] is None
    assert cache["total"]["cached_share_of_reporting_prompt"] == 0.6


# ----- second review (2026-10-11) -------------------------------------------------------

from agent_company.analysis.evidence_projection import project_evidence as _project  # noqa: E402
from test_role_policy_relay import (PROPOSAL, Conversation, fixed_call, relay_for, spawn,  # noqa: E402
                                           turn)
from test_team import call  # noqa: E402


def relay_trial(tmp_path):
    trial = tmp_path / "trial"
    (trial / "team").mkdir(parents=True)
    shutil.copytree(tmp_path / "logs", trial / "team/relay")
    return trial


def test_calls_of_a_late_rejected_response_are_not_counted_as_behavior(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder", "frontend")
    turn(relay, upstream, "decoder")
    store, original_stage = relay.role_policy, relay.role_policy.stage

    def stage_then_master_update(actor, snapshot, calls, **kwargs):
        staged = original_stage(actor, snapshot, calls, **kwargs)
        if actor == "decoder":
            master_call = parse("m1", "decoder", "ROLE-POLICY decoder\nMaster update")
            store.commit(original_stage("master", store.snapshot("master"), [master_call]))
        return staged
    store.stage = stage_then_master_update
    turn(relay, upstream, "decoder", [call("send_message", {"to": "leader", "message": PROPOSAL}),
                                      call("send_message", {"to": "frontend", "message": "Flag parser changed."})])
    store.stage = original_stage
    turn(relay, upstream, "decoder", [call("send_message", {"to": "frontend", "message": "Counted message."})])
    report = _project(relay_trial(tmp_path))
    decoder = [entry for entry in report["collaboration_timeline"] if entry["role"] == "decoder"]
    assert [entry["execution_status"] for entry in decoder] == [
        "rejected_before_execution", "rejected_before_execution", "execution_unconfirmed"]
    assert report["role_policies"]["messages_by_version"] == [
        {"scope": "role", "slot": "decoder", "version": 2, "messages": 1, "to_leader": 0, "to_teammate": 1,
         "broadcast": 0}]
    assert report["role_policies"]["rejections"]["by_stage"] == {"late": 1}


def parse(identifier, to, message):
    from agent_company.policy.role_policy import parse_send_message
    return parse_send_message(identifier, to, message)


def test_a_reused_call_id_without_a_result_stays_unknown(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    worker = Conversation(relay, upstream, "decoder")
    worker.turn()
    proposal = lambda: fixed_call("call_0", "send_message", {"to": "leader", "message": PROPOSAL})  # noqa: E731
    worker.turn([proposal()], ['Message sent to "leader".'])
    worker.turn([proposal()], ['Teammate "leader" is no longer active.'])
    worker.turn([proposal()])  # no later request carries this call's result
    notifications = _project(relay_trial(tmp_path))["role_policies"]["notifications"]
    assert notifications == {"send_failed": 1, "sent": 1, "unknown": 1}


def test_global_mode_projects_c_versions_and_behavior_per_version(tmp_path):
    scenario, result, lifecycle, events, output = run(tmp_path, "global", "global")
    policies = project_evidence(trial_from(tmp_path, output))["role_policies"]
    (version,) = [row for row in policies["versions"] if row["scope"] == "global"]
    assert version["version"] == 2 and version["text"]["text"] == "Report only blockers and results."
    assert version["first_carry_latency_s"] is not None and version["first_carry_latency_s"] >= 0
    # v1: "Starting the build."; v2: the acknowledgement and the shutdown confirmation.
    by_version = {row["version"]: row["messages"] for row in policies["messages_by_version"]}
    assert by_version == {1: 1, 2: 2} and {row["scope"] for row in policies["messages_by_version"]} == {"global"}


def test_a_corrupt_ledger_is_reported_not_raised(tmp_path):
    trial = tmp_path / "trial"
    (trial / "team/relay").mkdir(parents=True)
    (trial / "team/relay/events.jsonl").write_text(
        json.dumps({"event": "role_policy_batch", "schema": 1, "batch": 1, "kind": "init", "changes": []}) + "\n")
    report = project_evidence(trial)
    assert report["role_policies"]["status"] == "on" and report["role_policies"]["mode"] is None
    assert report["role_policies"]["ledger"]["replay"].startswith("ReplayError")
    assert report["coverage"]["issues"]["role_policy_replay_failed"] == 1
