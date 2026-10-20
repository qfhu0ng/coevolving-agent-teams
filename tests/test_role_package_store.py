"""Transactions, shared roles, proposals and replay: no native CLI or model calls."""
import copy
import json
import threading

import pytest

from agent_company.policy.role_package import create_p0_package
from agent_company.policy.role_package_store import (
    PackageRejection, RolePackageStore, parse_send_message, parse_spawn,
)
from agent_company.policy.role_package_runtime import render_injection, render_tail
from agent_company.policy.role_policy import PolicyPersistError, ReplayError


class Log:
    def __init__(self):
        self.records = []
        self.fail = False

    def __call__(self, record):
        if self.fail:
            raise OSError("disk full")
        self.records.append(json.loads(json.dumps(record)))


def setup(mode="propose", package=None):
    log = Log()
    return RolePackageStore(mode, package or create_p0_package("A", "C"), log,
                            clock=lambda: 1760000000), log


def msg(store, actor, tid, to, text):
    snap = store.snapshot(actor)
    return parse_send_message(tid, to, text, mode=store.mode, actor=actor, role_id=snap.role_id)


def act(store, actor, *calls):
    return store.commit(store.stage(actor, store.snapshot(actor), calls, source_request=1))


def spawn(store, name, role="verifier", succeeds=None, active=True):
    prompt = f"ROLE-TYPE: {role}" + (f"\nSUCCEEDS: {succeeds}" if succeeds else "")
    result = act(store, "master", parse_spawn(f"spawn-{name}", name, prompt))
    assert not result.rejections
    if active:
        store.record_first_request(name)
    return result


def propose(store, name, target="role:verifier", text="Report tested revision.", op="append", tid="p"):
    return msg(store, name, tid, "leader", f"POLICY-PROPOSAL {target} {op}\nCHANGE: {text}\nREASON: stale report\nEVIDENCE: task 2")


def decide(store, pid, name, decision="approve", body=""):
    return msg(store, "master", f"d{pid}", name, f"POLICY-DECISION #{pid} {decision}" + ("\n" + body if body else ""))


def test_two_instances_share_updates_and_new_worker_inherits_without_succession():
    store, log = setup()
    spawn(store, "one")
    spawn(store, "two")
    old = store.snapshot("two")
    act(store, "one", propose(store, "one"))
    result = act(store, "master", decide(store, 1, "one"))
    assert not result.rejections
    assert store.snapshot("one").target("role:verifier").text == "Report tested revision."
    assert store.snapshot("two").target("role:verifier").version == 1
    assert old.target("role:verifier").version == 0  # transport retry snapshot is stable
    spawn(store, "three")
    assert store.snapshot("three").target("role:verifier").version == 1
    assert RolePackageStore.replay(log.records, lambda r: None).export() == store.export()


def test_each_proposer_can_have_c_and_own_role_pending_and_update_supersedes_target_only():
    store, _ = setup()
    spawn(store, "one")
    spawn(store, "two")
    act(store, "one", propose(store, "one", tid="a"), propose(store, "one", "C", tid="b"))
    act(store, "two", propose(store, "two", tid="c"))
    assert len(store.snapshot("master").proposals) == 3
    act(store, "master", decide(store, 1, "one"))
    assert [(p.proposal_id, p.target) for p in store.snapshot("master").proposals] == [(2, "C")]
    assert "superseded" in render_tail(store.snapshot("two"))
    assert store.snapshot("one").proposals[0].status == "approved"


def test_replacement_preserves_exact_id_binding_and_identical_renotify():
    store, _ = setup()
    spawn(store, "one")
    act(store, "one", propose(store, "one"))
    master = store.snapshot("master")
    act(store, "one", propose(store, "one", tid="same"))
    assert store.snapshot("master").pending_ids == {1}
    act(store, "one", propose(store, "one", text="Different rule", tid="different"))
    result = store.commit(store.stage("master", master, [decide(store, 1, "one")]))
    assert result.rejections[0].code == "stale_proposal"
    assert store.snapshot("master").pending_ids == {2}
    assert store.snapshot("one").target("role:verifier").version == 0


def test_late_cas_failure_only_rejects_that_call_and_keeps_spawn():
    store, _ = setup()
    stale = store.snapshot("master")
    update = msg(store, "master", "stale", "*", "GLOBAL-POLICY append\nLater")
    staged = store.stage("master", stale, [update, parse_spawn("s", "one", "ROLE-TYPE: verifier")])
    act(store, "master", msg(store, "master", "first", "*", "GLOBAL-POLICY append\nFirst"))
    result = store.commit(staged)
    assert [e.code for e in result.rejections] == ["stale_version"]
    assert result.accepted_call_ids == ("s",)
    assert store.snapshot("master").global_policy.text == "C\nFirst"
    store.record_first_request("one")
    assert store.snapshot("one").role_id == "verifier"


def test_concurrent_commits_on_same_base_have_one_winner():
    store, _ = setup()
    staged = [store.stage("master", store.snapshot("master"), [msg(store, "master", str(n), "*", f"GLOBAL-POLICY append\n{n}")]) for n in range(2)]
    barrier, results = threading.Barrier(2), []
    def commit(batch):
        barrier.wait()
        results.append(store.commit(batch))
    threads = [threading.Thread(target=commit, args=(b,)) for b in staged]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(bool(r.accepted_call_ids) for r in results) == 1
    assert sum(bool(r.rejections) for r in results) == 1
    assert store.snapshot("master").global_policy.version == 1


def test_persist_failure_and_cancellation_publish_nothing():
    store, log = setup()
    batch = store.stage("master", store.snapshot("master"), [msg(store, "master", "a", "*", "GLOBAL-POLICY replace\nNew")])
    before = store.export()
    log.fail = True
    with pytest.raises(PolicyPersistError):
        store.commit(batch)
    assert store.export() == before
    log.fail = False
    def cancelled():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        store.commit(batch, check=cancelled)
    assert store.export() == before
    assert len(log.records) == 1


def test_empty_replace_amend_and_append_limits_are_post_combination():
    store, _ = setup()
    spawn(store, "one")
    act(store, "master", msg(store, "master", "set", "*", "ROLE-POLICY verifier replace\n" + "x" * 2000))
    bad = act(store, "master", msg(store, "master", "over", "*", "ROLE-POLICY verifier append\ny"))
    assert bad.rejections[0].code == "too_long"
    act(store, "one", propose(store, "one", text="", op="replace"))
    act(store, "master", decide(store, 1, "one", "amend"))
    assert store.snapshot("one").target("role:verifier").text == ""
    assert store.snapshot("one").target("role:verifier").version == 2
    bad = act(store, "master", msg(store, "master", "empty", "*", "GLOBAL-POLICY replace\n"))
    assert bad.rejections[0].code == "malformed_body"


def test_amend_replaces_entire_target_and_preserves_proposal_source():
    store, _ = setup()
    spawn(store, "one")
    act(store, "one", propose(store, "one", "C"))
    result = act(store, "master", decide(store, 1, "one", "amend", "Entire replacement"))
    assert store.snapshot("master").global_policy.text == "Entire replacement"
    commit = next(c for c in result.changes if c["event"] == "role_policy_committed")
    assert commit["value"]["provenance"] == "master_amend"
    assert commit["value"]["proposal_id"] == 1
    assert commit["proposer_role_id"] == "verifier"
    assert commit["before_text"] == "C"


def test_pending_budget_rejection_restores_old_proposal_and_does_not_hide_tail():
    store, _ = setup()
    for n in range(5):
        spawn(store, f"worker{n}")
    # Four long but valid role proposals; the next exceeds the rendered 8000-char block.
    for n in range(4):
        result = act(store, f"worker{n}", propose(store, f"worker{n}", text="x" * 1800))
        assert not result.rejections
    before = store.snapshot("master")
    bad = act(store, "worker4", propose(store, "worker4", text="y" * 1700))
    assert bad.rejections[0].code == "pending_display_full"
    assert store.snapshot("master") == before
    assert len(render_tail(before)) <= 8000
    assert "worker3" in render_tail(before)
    # Replacement rejection is atomic too: enlarging proposal 1 over the total budget.
    bad = act(store, "worker0", propose(store, "worker0", text="z" * 1940))
    assert bad.rejections[0].code in {"pending_display_full", "too_long"}
    assert store.snapshot("master") == before


def test_aggregate_budget_counts_retired_text_and_rejects_only_offender():
    p = create_p0_package("A", "C")
    for n in range(7):
        p["roles"][f"old{n}"] = {"policy": "x" * 2000, "retired": True}
    p["roles"]["prober"]["policy"] = "x" * 2000
    store, _ = setup(package=p)
    result = act(store, "master", msg(store, "master", "over", "*", "ROLE-POLICY verifier append\nx"),
                 parse_spawn("s", "one", "ROLE-TYPE: verifier"))
    assert result.rejections[0].code == "too_long"
    assert result.accepted_call_ids == ("s",)
    assert not any(r.role_id.startswith("old") for r in store.snapshot("master").roles)


def test_terminal_voids_proposals_and_succession_keeps_role_and_slot_only():
    store, log = setup()
    spawn(store, "one")
    act(store, "one", propose(store, "one"))
    result = act(store, "master", parse_spawn("early", "two", "ROLE-TYPE: verifier\nSUCCEEDS: one"))
    assert result.rejections[0].code == "predecessor_not_terminal"
    store.record_terminal("one", source="native-event")
    assert not store.snapshot("master").pending_ids
    bad = act(store, "master", parse_spawn("mismatch", "two", "SUCCEEDS: one"))
    assert bad.rejections[0].code == "role_type_mismatch"
    spawn(store, "two", succeeds="one")
    assert store.snapshot("two").bindings[-1].slot_id == "one"
    assert not store.snapshot("two").proposals
    assert RolePackageStore.replay(log.records, lambda r: None).export() == store.export()


def test_failed_spawn_not_in_stable_mapping_and_no_unbound_fallback():
    store, _ = setup()
    spawn(store, "one", active=False)
    assert "one=" not in render_injection(store.snapshot("master"))
    assert store.record_spawn_result("one", ok=False) == "voided"
    with pytest.raises(RuntimeError):
        store.record_first_request("one")
    with pytest.raises(RuntimeError):
        store.snapshot("missing")
    spawn(store, "one")  # failed dispatch may reuse its never-created name
    system = render_injection(store.snapshot("master"))
    assert "one=verifier" in system
    store.record_terminal("one")
    assert render_injection(store.snapshot("master")) == system


def test_off_has_bindings_but_never_writes_policy_or_accepts_proposals():
    store, _ = setup("off")
    spawn(store, "one")
    result = act(store, "one", propose(store, "one"))
    assert result.rejections[0].code == "not_authorized"
    result = act(store, "master", msg(store, "master", "u", "*", "GLOBAL-POLICY replace\nNew"))
    assert result.rejections[0].code == "not_authorized"
    assert render_tail(store.snapshot("master")) == ""
    assert store.snapshot("master").global_policy.text == "C"


def test_worker_can_only_propose_own_role_or_c_to_leader():
    store, _ = setup()
    spawn(store, "one")
    wrong = act(store, "one", propose(store, "one", "role:prober"))
    assert wrong.rejections[0].code == "not_authorized"
    wrong = act(store, "one", msg(store, "one", "u", "*", "ROLE-POLICY verifier replace\nDirect"))
    assert wrong.rejections[0].code == "not_authorized"
    assert not store.snapshot("master").pending_ids


def test_end_run_and_replay_keep_full_history_and_detect_corrupt_version():
    store, log = setup()
    spawn(store, "one")
    act(store, "one", propose(store, "one"))
    act(store, "master", decide(store, 1, "one"))
    act(store, "one", propose(store, "one", "C", tid="c"))
    assert store.end_run() == 1
    assert store.end_run() == 0
    assert store.export()["closed"]
    assert RolePackageStore.replay(log.records, lambda r: None).export() == store.export()
    bad = copy.deepcopy(log.records)
    for batch in bad:
        for change in batch["changes"]:
            if change["event"] == "role_policy_committed":
                change["value"]["version"] += 1
    with pytest.raises(ReplayError):
        RolePackageStore.replay(bad, lambda r: None)


@pytest.mark.parametrize("prompt,code", [
    ("ROLE-TYPE: master", "reserved_role"),
    ("ROLE-TYPE: verifier\nROLE-TYPE: verifier", "malformed_body"),
    ("ROLE-TYPE:", "invalid_role_id"),
    ("role-type: verifier", "malformed_header"),
    ("**ROLE-TYPE: verifier**", "malformed_header"),
    ("ROLE-TYPE: Verifier", "invalid_role_id"),
    ("ROLE-POLICY:\nRule", "inline_policy_not_supported"),
    ("SUCCEEDS: one\nSUCCEEDS: two", "malformed_body"),
])
def test_invalid_markers_are_not_silent_defaults(prompt, code):
    with pytest.raises(PackageRejection) as info:
        parse_spawn("s", "new", prompt)
    assert info.value.code == code


def test_default_and_unknown_role_do_not_change_one_another():
    store, _ = setup()
    default = parse_spawn("s", "default", "Implement the task.")
    assert default.role_id == "implementer" and default.binding_source == "default"
    result = act(store, "master", default, parse_spawn("bad", "unknown", "ROLE-TYPE: nonexistent"))
    assert result.accepted_call_ids == ("s",)
    assert result.rejections[0].code == "invalid_role_id"


def test_confirmed_spawn_is_not_undone_by_late_failed_spawn_result():
    store, _ = setup()
    spawn(store, "one", active=False)
    assert store.record_spawn_result("one", ok=True) == "ok"
    before = render_injection(store.snapshot("master"))
    assert "one=verifier" in before
    assert store.record_spawn_result("one", ok=False) == "conflict"
    assert render_injection(store.snapshot("master")) == before
    store.record_first_request("one")
    assert store.snapshot("one").role_id == "verifier"


def test_different_target_updates_can_both_commit_from_one_snapshot():
    store, _ = setup()
    result = act(store, "master", msg(store, "master", "c", "*", "GLOBAL-POLICY append\nTeam rule"),
                 msg(store, "master", "role", "*", "ROLE-POLICY master append\nMaster rule"))
    assert result.accepted_call_ids == ("c", "role")
    assert store.snapshot("master").global_policy.version == 1
    assert store.snapshot("master").target("role:master").version == 1


def test_optional_base_never_overrides_the_snapshot_and_wrong_decision_recipient_fails():
    store, _ = setup()
    spawn(store, "one")
    bad = act(store, "one", msg(store, "one", "p", "leader", "POLICY-PROPOSAL C append base=v1\nCHANGE: rule\nREASON: why\nEVIDENCE: task 1"))
    assert bad.rejections[0].code == "stale_version"
    act(store, "one", propose(store, "one"))
    bad = act(store, "master", msg(store, "master", "d", "*", "POLICY-DECISION #1 approve"))
    assert bad.rejections[0].code == "not_authorized"
    assert store.snapshot("master").pending_ids == {1}


@pytest.mark.parametrize("header,mode,actor,role_id,fragment", [
    ("**POLICY-PROPOSAL**", "propose", "one", "verifier", "role:verifier append|replace"),
    ("GLOBAL-POLICY", "propose", "master", "master", "Use GLOBAL-POLICY append|replace"),
    ("GLOBAL-POLICY", "off", "master", "master", "not available in mode off"),
])
def test_malformed_header_uses_the_approved_mode_specific_variant(header, mode, actor, role_id, fragment):
    with pytest.raises(PackageRejection) as info:
        parse_send_message("bad", "leader", header, mode=mode, actor=actor, role_id=role_id)
    assert info.value.code == "malformed_header"
    assert fragment in info.value.detail


def test_closed_run_rejects_new_controls_and_replay_trailing_changes():
    store, log = setup()
    snapshot = store.snapshot("master")
    call = msg(store, "master", "c", "*", "GLOBAL-POLICY append\nrule")
    batch = store.stage("master", snapshot, [call])
    store.end_run()
    result = store.commit(batch)
    assert result.rejections[0].code == "run_ended"
    assert not result.accepted_call_ids
    records = copy.deepcopy(log.records)
    records.append({"event": "role_package_batch", "schema": 1, "batch": len(records) + 1,
                    "kind": "lifecycle", "changes": [{"event": "role_policy_run_ended", "at": "late"}]})
    with pytest.raises(ReplayError):
        RolePackageStore.replay(records, lambda r: None)


def test_amend_note_is_recorded_but_not_written_to_policy():
    store, log = setup()
    spawn(store, "one")
    act(store, "one", propose(store, "one"))
    result = act(store, "master", decide(store, 1, "one", "amend",
                 "Report the tested revision.\nNOTE: This explanation contains task-specific details."))
    assert not result.rejections
    assert store.snapshot("one").target("role:verifier").text == "Report the tested revision."
    row = next(c for c in result.changes if c["event"] == "role_policy_decided")
    assert row["note"] == "This explanation contains task-specific details."
    assert RolePackageStore.replay(log.records, lambda r: None).export() == store.export()


@pytest.mark.parametrize("body", [
    "NOTE: Narrow scope.\nNew policy.",
    "Policy A.\nNOTE: reason\nPolicy B.",
    "NOTE: explanation only",
    "\nNOTE: explanation only",
    "Policy A.\nNOTE: reason\nNOTE: second reason",
])
def test_misplaced_amend_note_rejects_without_clearing_or_truncating_policy(body):
    store, log = setup()
    spawn(store, "one")
    act(store, "master", msg(store, "master", "initial", "*", "ROLE-POLICY verifier replace\nKeep this rule."))
    act(store, "one", propose(store, "one"))
    before, records = store.export(), copy.deepcopy(log.records)
    result = act(store, "master", decide(store, 1, "one", "amend", body))
    assert [r.code for r in result.rejections] == ["malformed_body"]
    assert result.rejections[0].detail == (
        "Amend supplies the complete replacement policy for role:verifier below the header, "
        "without another control header. A role amendment may be empty to clear its policy; C must remain nonempty.")
    assert store.export() == before and log.records == records
    assert store.snapshot("master").pending_ids == {1}


def test_empty_amend_still_clears_role_after_misplaced_note_is_rejected():
    store, _ = setup()
    spawn(store, "one")
    act(store, "master", msg(store, "master", "initial", "*", "ROLE-POLICY verifier replace\nKeep this rule."))
    act(store, "one", propose(store, "one"))
    bad = act(store, "master", decide(store, 1, "one", "amend", "NOTE: not a policy"))
    assert bad.rejections[0].code == "malformed_body"
    good = act(store, "master", decide(store, 1, "one", "amend"))
    assert not good.rejections
    assert store.snapshot("one").target("role:verifier").text == ""


def test_replayed_store_supports_nested_lifecycle_lock():
    store, log = setup()
    spawn(store, "one", active=False)
    replayed = RolePackageStore.replay(log.records, log)
    with replayed._lock:
        acquired = replayed._lock.acquire(blocking=False)
        assert acquired, "delivery cleanup holds the store lock before recording lifecycle evidence"
        try:
            assert replayed.record_spawn_result("one", ok=False, tool_call_id="undelivered") == "voided"
        finally:
            replayed._lock.release()
    assert replayed.export()["bindings"]["one"]["lifecycle"] == "voided"


@pytest.mark.parametrize("mode,actor,text", [
    ("off", "master", "POLICY-PROPOSAL C append\nmalformed"),
    ("propose", "master", "POLICY-PROPOSAL C append\nmalformed"),
    ("off", "master", "POLICY-DECISION #1 approve\nmalformed"),
    ("propose", "one", "POLICY-DECISION #1 approve\nmalformed"),
])
def test_denied_controls_do_not_teach_body_format_first(mode, actor, text):
    store, _ = setup(mode)
    if actor != "master":
        spawn(store, actor)
    result = act(store, actor, msg(store, actor, "denied", "leader", text))
    assert [r.code for r in result.rejections] == ["not_authorized"]


def test_authorized_proposal_still_validates_body_after_recipient_and_target():
    store, _ = setup()
    spawn(store, "one")
    malformed = "POLICY-PROPOSAL role:verifier append\nmalformed"
    assert act(store, "one", msg(store, "one", "bad", "leader", malformed)).rejections[0].code == "malformed_body"
    assert act(store, "one", msg(store, "one", "wrong", "*", malformed)).rejections[0].code == "not_authorized"


@pytest.mark.parametrize("sentence", ["Succeeds when tests pass.", "Role-policy questions go to the master.",
                                       "ROLE-TYPE verifier", "END-ROLE-POLICY questions"])
def test_plain_prose_is_not_a_spawn_marker(sentence):
    parsed = parse_spawn("s", "new", sentence)
    assert parsed.role_id == "implementer" and parsed.succeeds is None


def test_invalid_role_syntax_uses_real_catalog():
    with pytest.raises(PackageRejection) as info:
        parse_spawn("s", "new", "ROLE-TYPE: Bad!", role_ids=("master", "implementer", "custom"))
    assert info.value.code == "invalid_role_id"
    assert info.value.detail == "That role type is not in the role directory. Use one of: custom, implementer, master."


def test_duplicate_predecessor_name_reports_current_slot_holder():
    store, _ = setup()
    spawn(store, "one")
    store.record_terminal("one")
    spawn(store, "two", succeeds="one")
    result = act(store, "master", parse_spawn("again", "one", "ROLE-TYPE: verifier"))
    assert result.rejections[0].code == "duplicate_name"
    assert result.rejections[0].detail == "one is already in use by two; choose a new name."
