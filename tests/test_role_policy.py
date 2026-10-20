"""Role-policy state (S1): parsing, staging/late commit, lifecycle, snapshots, replay.

Specification: docs/stage1_role_policy_harness_plan_20261009.md v3.3 sections 3.1, 3.2, 3.6.
No relay, native CLI or model is involved.
"""
import dataclasses
import json
import random
import threading

import pytest

from agent_company.policy.role_policy import (
    BATCH_EVENT, GLOBAL_POLICY_MAX_CHARS, PROPOSAL_MAX_CHARS, ROLE_POLICY_MAX_CHARS, DecisionCall,
    GlobalPolicyCall, PolicyPersistError, PolicyRejection, ProposalCall, ReplayError, RolePolicyCall,
    RolePolicyStore, SpawnCall, parse_send_message, parse_spawn)


class Log:
    """Event writer that keeps JSON round-tripped records and can be made to fail."""

    def __init__(self):
        self.records, self.fail = [], False

    def __call__(self, record):
        if self.fail:
            raise OSError("disk full")
        self.records.append(json.loads(json.dumps(record)))


def ticking(start=1_760_000_000.0):
    now = [start]

    def clock():
        now[0] += 60
        return now[0]
    return clock


def new_store(mode="propose"):
    log = Log()
    return RolePolicyStore(mode, "Starting communication policy C.", log, clock=ticking()), log


def respond(store, actor, calls, source_request=None):
    staged = store.stage(actor, store.snapshot(actor), calls, source_request=source_request)
    return store.commit(staged)


def msg(tid, to, text):
    return parse_send_message(tid, to, text)


def proposal(change="Message frontend directly on CLI flag changes.", reason="Two handoffs were redone.",
             evidence="Task 4 and task 7 were both reopened.", base=""):
    return f"POLICY-PROPOSAL{base}\nCHANGE: {change}\nREASON: {reason}\nEVIDENCE: {evidence}"


def team(store, *names):
    """Master spawns the named workers (decoder gets an initial policy); each makes a first request."""
    # In global mode the relay does not parse spawn markers (plan 3.2), so none are sent.
    markers = store.mode != "global"
    calls = [parse_spawn(f"spawn-{name}", name,
                         "Implement the decoder.\nROLE-POLICY:\nReport interface changes to frontend.\nEND-ROLE-POLICY"
                         if name == "decoder" and markers else f"Work on {name}.")
             for name in names]
    respond(store, "master", calls)
    for name in names:
        assert store.record_first_request(name) == "bound"


def rejected(code, func, *args, **kwargs):
    with pytest.raises(PolicyRejection) as info:
        func(*args, **kwargs)
    assert info.value.code == code, info.value
    return info.value


# -- parsing -------------------------------------------------------------------------------

def test_ordinary_messages_are_not_control_messages():
    for text in ("hello", "Role policy question: should I message frontend?", "", "Policy proposal later",
                 "ROLE POLICY decoder"):
        assert msg("t", "leader", text) is None
    assert msg("t", "leader", None) is None


def test_control_headers_parse_with_optional_base():
    call = msg("t1", "decoder", "ROLE-POLICY decoder\nSend interface changes to frontend.")
    assert call == RolePolicyCall("t1", "decoder", "decoder", None, "Send interface changes to frontend.")
    assert msg("t2", "decoder", "ROLE-POLICY decoder base=v2\nX").base == 2
    assert msg("t3", "*", "GLOBAL-POLICY base=v1\r\nNew C\r\n") == GlobalPolicyCall("t3", "*", 1, "New C")
    assert msg("t4", "leader", proposal(base=" base=v0")) == ProposalCall(
        "t4", "leader", 0, "Message frontend directly on CLI flag changes.", "Two handoffs were redone.",
        "Task 4 and task 7 were both reopened.")
    assert msg("t5", "decoder", "POLICY-DECISION #3 Approve\nNOTE: good evidence") == DecisionCall(
        "t5", "decoder", 3, "approve", "good evidence", None)
    assert msg("t6", "decoder", "POLICY-DECISION #3 reject").note is None
    assert msg("t7", "decoder", "POLICY-DECISION #4 amend\nOnly message frontend for flag changes.").amended_text == \
        "Only message frontend for flag changes."


@pytest.mark.parametrize("header", [
    "role-policy decoder", "**POLICY-PROPOSAL**", "POLICY_PROPOSAL", "POLICY-PROPOSAL base=3",
    "POLICY-PROPOSAL #3", "POLICY-DECISION 3 approve", "POLICY-DECISION #3 accept", "GLOBAL-POLICY base=v1 now",
    "ROLE-POLICY", "## Global-Policy",
])
def test_near_miss_headers_are_rejected_not_delivered_as_ordinary_text(header):
    error = rejected("malformed_header", msg, "t9", "leader", header + "\nbody")
    assert error.tool_call_id == "t9"


def test_control_bodies_are_validated():
    rejected("malformed_body", msg, "t", "decoder", "ROLE-POLICY decoder\n  \n")
    rejected("too_long", msg, "t", "decoder", "ROLE-POLICY decoder\n" + "x" * (ROLE_POLICY_MAX_CHARS + 1))
    assert msg("t", "decoder", "ROLE-POLICY decoder\n" + "x" * ROLE_POLICY_MAX_CHARS) is not None
    rejected("too_long", msg, "t", "*", "GLOBAL-POLICY\n" + "x" * (GLOBAL_POLICY_MAX_CHARS + 1))
    for body in ("REASON: r\nCHANGE: c\nEVIDENCE: e", "CHANGE: c\nEVIDENCE: e", "Preamble\nCHANGE: c\nREASON: r\nEVIDENCE: e",
                 "CHANGE: c\nREASON:\nEVIDENCE: e", "CHANGE: c\nEVIDENCE: e\nREASON: r"):
        rejected("malformed_body", msg, "t", "leader", "POLICY-PROPOSAL\n" + body)
    rejected("too_long", msg, "t", "leader", proposal(change="x" * PROPOSAL_MAX_CHARS))
    multiline = msg("t", "leader", "POLICY-PROPOSAL\nCHANGE: line one\nline two\nREASON: r\nEVIDENCE: e\nmore")
    assert (multiline.change, multiline.evidence) == ("line one\nline two", "e\nmore")
    rejected("malformed_body", msg, "t", "decoder", "POLICY-DECISION #1 approve\nlooks good")
    rejected("malformed_body", msg, "t", "decoder", "POLICY-DECISION #1 amend")
    rejected("malformed_body", msg, "t", "decoder", "POLICY-DECISION #1 amend\nROLE-POLICY decoder\ntext")


def test_spawn_markers():
    block = parse_spawn("s", "decoder", "Task.\nROLE-POLICY:\nRule one.\nRule two.\nEND-ROLE-POLICY\nMore task.")
    assert block == SpawnCall("s", "decoder", "Rule one.\nRule two.", None)
    assert parse_spawn("s", "decoder", "Task.\nROLE-POLICY:\nRule to the end.").initial_policy == "Rule to the end."
    assert parse_spawn("s", "decoder2", "Continue.\nSUCCEEDS: decoder\n").succeeds == "decoder"
    assert parse_spawn("s", "frontend", "Plain prompt; the role policy: none.") == SpawnCall("s", "frontend", None, None)
    rejected("conflicting_markers", parse_spawn, "s", "x", "ROLE-POLICY:\nRule\nEND-ROLE-POLICY\nSUCCEEDS: decoder")
    rejected("conflicting_markers", parse_spawn, "s", "x", "ROLE-POLICY:\nRule\nSUCCEEDS: decoder")
    rejected("malformed_body", parse_spawn, "s", "x", "SUCCEEDS: a\nSUCCEEDS: b")
    rejected("malformed_body", parse_spawn, "s", "x", "ROLE-POLICY:\nA\nEND-ROLE-POLICY\nROLE-POLICY:\nB")
    rejected("malformed_body", parse_spawn, "s", "x", "ROLE-POLICY:\n\nEND-ROLE-POLICY")
    rejected("malformed_body", parse_spawn, "s", "x", "END-ROLE-POLICY")
    rejected("malformed_header", parse_spawn, "s", "x", "Role-Policy:\nRule")
    rejected("malformed_header", parse_spawn, "s", "x", "succeeds: decoder")
    rejected("too_long", parse_spawn, "s", "x", "ROLE-POLICY:\n" + "x" * (ROLE_POLICY_MAX_CHARS + 1))


# -- authorization and modes ---------------------------------------------------------------

def test_authorization_matrix_in_propose_mode():
    store, _ = new_store("propose")
    team(store, "decoder", "frontend")
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    rejected("not_authorized", respond, store, "decoder", [msg("w1", "decoder", "ROLE-POLICY decoder\nX")])
    rejected("not_authorized", respond, store, "decoder", [msg("w2", "leader", "POLICY-DECISION #1 approve")])
    rejected("not_authorized", respond, store, "master", [msg("m1", "leader", proposal())])
    rejected("not_authorized", respond, store, "master", [msg("m2", "frontend", "ROLE-POLICY decoder\nX")])
    rejected("not_authorized", respond, store, "master", [msg("m3", "*", "ROLE-POLICY decoder\nX")])
    rejected("not_authorized", respond, store, "decoder", [msg("w3", "master", proposal())])
    rejected("not_authorized", respond, store, "master", [msg("m4", "frontend", "POLICY-DECISION #1 approve")])
    rejected("not_authorized", respond, store, "master", [msg("m5", "*", "POLICY-DECISION #1 approve")])
    rejected("not_authorized", respond, store, "decoder", [parse_spawn("w4", "helper", "")])
    rejected("not_authorized", respond, store, "master", [msg("m6", "*", "GLOBAL-POLICY\nNew C")])


def test_modes_gate_their_prefixes():
    store, _ = new_store("global")
    team(store, "decoder")
    rejected("not_authorized", respond, store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nX")])
    rejected("not_authorized", respond, store, "master", [msg("m2", "decoder", "GLOBAL-POLICY\nNew C")])
    rejected("not_authorized", respond, store, "master", [parse_spawn("m3", "helper", "ROLE-POLICY:\nX")])
    rejected("not_authorized", respond, store, "decoder", [msg("w1", "leader", proposal())])
    respond(store, "master", [msg("m4", "*", "GLOBAL-POLICY\nNew C")])
    assert store.snapshot("decoder").global_policy.text == "New C"

    store, _ = new_store("master")
    team(store, "decoder")
    rejected("not_authorized", respond, store, "decoder", [msg("w2", "leader", proposal())])
    rejected("not_authorized", respond, store, "master", [msg("m5", "decoder", "POLICY-DECISION #1 approve")])
    rejected("not_authorized", respond, store, "master", [msg("m6", "*", "GLOBAL-POLICY\nNew C")])
    respond(store, "master", [msg("m7", "decoder", "ROLE-POLICY decoder\nNew role rule")])
    assert store.snapshot("decoder").slots[0].policy.version == 2


def test_off_mode_has_no_store():
    with pytest.raises(ValueError):
        RolePolicyStore("off", "C", Log())


# -- versions ------------------------------------------------------------------------------

def test_role_policy_versions_come_from_the_request_snapshot():
    store, _ = new_store("master")
    team(store, "decoder", "frontend")
    master = store.snapshot("master")
    assert [(s.slot_id, s.policy.version, s.policy.provenance) for s in master.slots] == [
        ("decoder", 1, "spawn_initial"), ("frontend", 0, None)]
    respond(store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nV2")])
    respond(store, "master", [msg("m2", "decoder", "ROLE-POLICY decoder base=v2\nV3")])
    error = rejected("stale_version", respond, store, "master", [msg("m3", "decoder", "ROLE-POLICY decoder base=v2\nX")])
    assert error.details["snapshot_version"] == 3
    old = store.snapshot("master")
    respond(store, "master", [msg("m4", "decoder", "ROLE-POLICY decoder\nV4")])
    error = rejected("stale_version", store.stage, "master", old, [msg("m5", "decoder", "ROLE-POLICY decoder\nX")])
    assert (error.details["snapshot_version"], error.details["current_version"]) == (3, 4)
    rejected("unknown_slot", respond, store, "master", [msg("m6", "ghost", "ROLE-POLICY ghost\nX")])
    respond(store, "master", [msg("m7", "frontend", "ROLE-POLICY frontend base=v0\nFirst frontend rule")])
    assert store.snapshot("frontend").slots[0].policy.text == "First frontend rule"


def test_role_policy_may_target_a_planned_instance_but_not_a_new_slot_in_the_same_response():
    store, _ = new_store("master")
    respond(store, "master", [parse_spawn("s1", "decoder", "Task")])
    respond(store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nBefore first request")])
    assert store.snapshot("decoder").slots[0].lifecycle == "planned"
    rejected("unknown_slot", respond, store, "master",
             [parse_spawn("s2", "frontend", "Task"), msg("m2", "frontend", "ROLE-POLICY frontend\nX")])


def test_global_policy_versions():
    store, _ = new_store("global")
    assert store.snapshot("master").global_policy.version == 1
    old = store.snapshot("master")
    respond(store, "master", [msg("m1", "*", "GLOBAL-POLICY base=v1\nC2")])
    rejected("stale_version", store.stage, "master", old, [msg("m2", "*", "GLOBAL-POLICY\nC3")])
    assert store.snapshot("master").global_policy.text == "C2"


# -- proposals -----------------------------------------------------------------------------

def test_identical_resend_renotifies_and_any_difference_replaces():
    store, _ = new_store()
    team(store, "decoder")
    first = respond(store, "decoder", [msg("p1", "leader", proposal())])
    created = first.changes[0]["created_at"]
    again = respond(store, "decoder", [msg("p2", "leader", proposal())])
    assert [c["event"] for c in again.changes] == ["role_policy_proposal_renotified"]
    (pending,) = store.snapshot("master").proposals
    assert (pending.proposal_id, pending.created_at) == (1, created)
    replaced = respond(store, "decoder", [msg("p3", "leader", proposal(reason="A different reason."))])
    assert [(c["event"], c.get("proposal_id"), c.get("replaced_by")) for c in replaced.changes] == [
        ("role_policy_proposal_voided", 1, 2), ("role_policy_proposed", 2, None)]
    assert replaced.changes[0]["reason"] == "replaced"
    master = store.snapshot("master")
    assert [(p.proposal_id, p.change, p.reason, p.evidence) for p in master.proposals] == [
        (2, "Message frontend directly on CLI flag changes.", "A different reason.",
         "Task 4 and task 7 were both reopened.")]
    worker = store.snapshot("decoder")
    assert [(p.proposal_id, p.status, p.replaced_by) for p in worker.proposals] == [
        (1, "voided", 2), (2, "pending", None)]


def test_two_proposals_in_one_response_follow_the_same_rule_in_one_batch():
    store, log = new_store()
    team(store, "decoder")
    before = len(log.records)
    result = respond(store, "decoder", [msg("p1", "leader", proposal()), msg("p2", "leader", proposal())])
    assert [c["event"] for c in result.changes] == ["role_policy_proposed", "role_policy_proposal_renotified"]
    result = respond(store, "decoder", [msg("p3", "leader", proposal(change="A")), msg("p4", "leader", proposal(change="B"))])
    assert [c["event"] for c in result.changes] == [
        "role_policy_proposal_voided", "role_policy_proposed", "role_policy_proposal_voided", "role_policy_proposed"]
    assert len(log.records) == before + 2
    assert [p.change for p in store.snapshot("master").proposals] == ["B"]


def test_proposal_base_is_checked_against_snapshot_and_current_version():
    store, _ = new_store()
    team(store, "decoder")
    rejected("stale_version", respond, store, "decoder", [msg("p1", "leader", proposal(base=" base=v0"))])
    worker = store.snapshot("decoder")
    staged = store.stage("decoder", worker, [msg("p2", "leader", proposal())])
    respond(store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nV2")])
    rejected("stale_version", store.commit, staged)
    rejected("stale_version", store.stage, "decoder", worker, [msg("p3", "leader", proposal())])
    assert store.snapshot("master").proposals == ()


# -- decisions -----------------------------------------------------------------------------

def test_approve_reject_and_amend():
    store, _ = new_store()
    team(store, "decoder")
    respond(store, "decoder", [msg("p1", "leader", proposal(change="Approved text"))])
    respond(store, "master", [msg("d1", "decoder", "POLICY-DECISION #1 approve\nNOTE: agreed")])
    slot = store.snapshot("decoder").slots[0].policy
    assert (slot.version, slot.text, slot.provenance, slot.proposal_id) == (2, "Approved text", "proposal_approved", 1)
    assert store.snapshot("decoder").proposals[0].status == "approved"

    respond(store, "decoder", [msg("p2", "leader", proposal(change="Rejected text"))])
    respond(store, "master", [msg("d2", "decoder", "POLICY-DECISION #2 reject")])
    assert store.snapshot("decoder").slots[0].policy.version == 2
    assert store.snapshot("decoder").proposals[-1].status == "rejected"

    respond(store, "decoder", [msg("p3", "leader", proposal(change="Proposed text"))])
    respond(store, "master", [msg("d3", "decoder", "POLICY-DECISION #3 amend\nAmended text")])
    slot = store.snapshot("decoder").slots[0].policy
    assert (slot.version, slot.text, slot.provenance) == (3, "Amended text", "proposal_amended")


def test_a_decision_never_follows_a_replacement():
    store, _ = new_store()
    team(store, "decoder")
    respond(store, "decoder", [msg("p1", "leader", proposal(change="First"))])
    master = store.snapshot("master")
    staged = store.stage("master", master, [msg("d1", "decoder", "POLICY-DECISION #1 approve")])
    respond(store, "decoder", [msg("p2", "leader", proposal(change="Second"))])
    error = rejected("stale_proposal", store.commit, staged)
    assert error.details["current_pending"] == 2
    error = rejected("stale_proposal", respond, store, "master", [msg("d2", "decoder", "POLICY-DECISION #1 approve")])
    assert "#2" in str(error)
    assert store.snapshot("decoder").slots[0].policy.version == 1
    respond(store, "master", [msg("d3", "decoder", "POLICY-DECISION #2 approve")])
    assert store.snapshot("decoder").slots[0].policy.text == "Second"


def test_a_decision_requires_the_proposal_in_the_masters_snapshot():
    store, _ = new_store()
    team(store, "decoder")
    master = store.snapshot("master")
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    error = rejected("stale_proposal", store.stage, "master", master,
                     [msg("d1", "decoder", "POLICY-DECISION #1 approve")])
    assert "not in your supplied snapshot" in str(error)
    rejected("unknown_proposal", respond, store, "master", [msg("d2", "decoder", "POLICY-DECISION #9 approve")])


def test_master_set_supersedes_pending_and_a_later_decision_is_stale():
    store, _ = new_store()
    team(store, "decoder")
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    result = respond(store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nMaster rule")])
    assert [(c["event"], c.get("reason")) for c in result.changes] == [
        ("role_policy_committed", None), ("role_policy_proposal_voided", "policy_superseded")]
    rejected("stale_proposal", respond, store, "master", [msg("d1", "decoder", "POLICY-DECISION #1 approve")])


# -- whole-response batches and late commit -------------------------------------------------

def test_any_invalid_call_rejects_the_whole_batch_without_state_or_records():
    store, log = new_store()
    team(store, "decoder")
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    before, records = store.export(), len(log.records)
    for calls in ([msg("m1", "decoder", "ROLE-POLICY decoder\nOK"), msg("m2", "ghost", "ROLE-POLICY ghost\nX")],
                  [msg("m3", "decoder", "ROLE-POLICY decoder\nOK"), msg("m4", "decoder", "POLICY-DECISION #1 approve")],
                  [msg("m5", "decoder", "ROLE-POLICY decoder\nA"), msg("m5", "decoder", "ROLE-POLICY decoder\nB")]):
        with pytest.raises(PolicyRejection):
            respond(store, "master", calls)
        assert store.export() == before and len(log.records) == records
    rejected("duplicate_tool_call", respond, store, "master",
             [msg("m6", "decoder", "ROLE-POLICY decoder\nA"), msg("m6", "decoder", "ROLE-POLICY decoder\nB")])


def test_late_commit_resimulates_on_the_latest_state():
    store, _ = new_store()
    team(store, "decoder", "frontend")
    respond(store, "frontend", [msg("f1", "leader", proposal(change="Frontend rule"))])
    staged = store.stage("decoder", store.snapshot("decoder"), [msg("p1", "leader", proposal())])
    assert staged.preview[0]["proposal_id"] == 2
    respond(store, "frontend", [msg("f2", "leader", proposal(change="Frontend rule v2"))])
    # frontend's replacement took id 2 after staging; the late commit allocates the next id.
    result = store.commit(staged)
    assert result.changes[0]["proposal_id"] == 3
    assert sorted(p.proposal_id for p in store.snapshot("master").proposals) == [2, 3]


def test_failed_write_or_check_publishes_nothing():
    store, log = new_store()
    team(store, "decoder")
    staged = store.stage("master", store.snapshot("master"), [msg("m1", "decoder", "ROLE-POLICY decoder\nV2")])
    before, records = store.export(), len(log.records)
    log.fail = True
    with pytest.raises(PolicyPersistError):
        store.commit(staged)
    assert store.export() == before
    log.fail = False

    def cancelled():
        raise RuntimeError("request cancelled")
    with pytest.raises(RuntimeError):
        store.commit(staged, check=cancelled)
    assert store.export() == before and len(log.records) == records
    result = store.commit(staged)
    assert result.batch == log.records[-1]["batch"] == records + 1


def test_a_response_without_policy_calls_writes_nothing():
    store, log = new_store()
    records = len(log.records)
    assert store.commit(store.stage("master", store.snapshot("master"), [])).batch is None
    assert len(log.records) == records


def test_snapshot_must_belong_to_the_actor():
    store, _ = new_store()
    team(store, "decoder")
    with pytest.raises(ValueError):
        store.stage("master", store.snapshot("decoder"), [msg("m1", "decoder", "ROLE-POLICY decoder\nX")])


# -- lifecycle and succession --------------------------------------------------------------

def test_terminal_voids_pending_and_is_idempotent():
    store, _ = new_store()
    team(store, "decoder")
    assert store.record_first_request("decoder") == "active"
    assert store.record_first_request("ghost") == "unknown"
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    assert store.record_terminal("decoder", source="stderr:120") == "terminal"
    assert store.record_terminal("decoder") == "terminal"
    assert store.record_terminal("ghost") == "unknown"
    assert store.snapshot("master").proposals == ()
    assert store.snapshot("decoder").proposals[0].void_reason == "proposer_terminal"
    rejected("not_authorized", respond, store, "decoder", [msg("p2", "leader", proposal())])
    rejected("not_authorized", respond, store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nX")])


def test_succession_keeps_slot_and_version_without_the_predecessors_proposals():
    store, _ = new_store()
    team(store, "decoder")
    respond(store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nV2")])
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    store.record_terminal("decoder")
    result = respond(store, "master", [parse_spawn("s2", "decoder2", "Continue.\nSUCCEEDS: decoder")])
    assert [c["event"] for c in result.changes] == ["role_policy_planned"]
    store.record_first_request("decoder2")
    view = store.snapshot("decoder2")
    assert (view.slots[0].slot_id, view.slots[0].instance, view.slots[0].policy.version) == ("decoder", "decoder2", 2)
    assert view.proposals == ()
    respond(store, "master", [msg("m2", "decoder2", "ROLE-POLICY decoder\nV3")])
    respond(store, "decoder2", [msg("p2", "leader", proposal())])
    assert [(p.proposal_id, p.proposer) for p in store.snapshot("master").proposals] == [(2, "decoder2")]


def test_succession_requires_a_confirmed_terminal_predecessor():
    store, _ = new_store()
    team(store, "decoder")
    rejected("predecessor_not_terminal", respond, store, "master", [parse_spawn("s", "decoder2", "SUCCEEDS: decoder")])
    rejected("unknown_slot", respond, store, "master", [parse_spawn("s", "decoder2", "SUCCEEDS: ghost")])
    rejected("duplicate_name", respond, store, "master", [parse_spawn("s", "decoder", "Again")])


def test_only_one_of_two_concurrent_successions_wins():
    store, _ = new_store()
    team(store, "decoder")
    store.record_terminal("decoder")
    first = store.stage("master", store.snapshot("master"), [parse_spawn("s1", "decoder2", "SUCCEEDS: decoder")])
    second = store.stage("master", store.snapshot("master"), [parse_spawn("s2", "decoder3", "SUCCEEDS: decoder")])
    store.commit(first)
    error = rejected("stale_succession", store.commit, second)
    assert error.details["current_instance"] == "decoder2"
    rejected("duplicate_name", respond, store, "master",
             [parse_spawn("s3", "helper", "Task"), parse_spawn("s4", "helper", "Task")])


def test_failed_spawn_frees_the_name_and_restores_the_slot():
    store, _ = new_store()
    team(store, "decoder")
    store.record_terminal("decoder")
    respond(store, "master", [parse_spawn("s1", "decoder2", "SUCCEEDS: decoder")])
    assert store.record_spawn_result("decoder2", ok=True) == "ok"
    assert store.record_spawn_result("decoder2", ok=False, tool_call_id="s1") == "voided"
    assert store.snapshot("master").slot("decoder").instance == "decoder"
    respond(store, "master", [parse_spawn("s2", "decoder2", "SUCCEEDS: decoder")])
    store.record_first_request("decoder2")
    assert store.record_spawn_result("decoder2", ok=False) == "conflict"

    respond(store, "master", [parse_spawn("s3", "frontend", "ROLE-POLICY:\nFirst try")])
    store.record_spawn_result("frontend", ok=False)
    respond(store, "master", [parse_spawn("s4", "frontend", "ROLE-POLICY:\nSecond try")])
    policy = store.snapshot("master").slot("frontend").policy
    assert (policy.version, policy.text) == (2, "Second try")


def test_end_run_voids_pending_and_closes_writes():
    store, _ = new_store()
    team(store, "decoder")
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    respond(store, "master", [parse_spawn("s9", "late", "Never starts.")])
    assert store.end_run() == 1
    assert store.end_run() == 0
    assert store.snapshot("decoder").proposals[0].void_reason == "run_ended"
    state = store.export()
    assert (state["bindings"]["late"]["lifecycle"], state["bindings"]["late"]["end_reason"]) == ("voided", "never_started")
    assert state["bindings"]["decoder"]["lifecycle"] == "active" and state["slots"]["late"]["current_instance"] is None
    rejected("run_ended", respond, store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nX")])


# -- snapshots -----------------------------------------------------------------------------

def test_snapshots_are_pure_immutable_and_scoped():
    store, _ = new_store()
    team(store, "frontend", "decoder")
    respond(store, "decoder", [msg("p1", "leader", proposal())])
    first, second = store.snapshot("master"), store.snapshot("master")
    assert first == second
    assert [s.slot_id for s in first.slots] == ["decoder", "frontend"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.slots[0].policy.text = "changed"  # type: ignore[misc]
    respond(store, "master", [msg("m1", "decoder", "ROLE-POLICY decoder\nV2")])
    assert first.slot("decoder").policy.version == 1 and first.pending_ids == {1}
    worker = store.snapshot("frontend")
    assert [s.slot_id for s in worker.slots] == ["frontend"] and worker.proposals == ()
    assert store.snapshot("stranger").slots == ()


# -- replay --------------------------------------------------------------------------------

def scenario(store):
    team(store, "decoder", "frontend")
    respond(store, "decoder", [msg("p1", "leader", proposal())], source_request=10)
    respond(store, "decoder", [msg("p2", "leader", proposal(change="Changed"))], source_request=11)
    respond(store, "master", [msg("d1", "decoder", "POLICY-DECISION #2 amend\nAmended")], source_request=12)
    respond(store, "frontend", [msg("f1", "leader", proposal(change="Frontend"))], source_request=13)
    store.record_terminal("frontend", source="stderr:7")
    respond(store, "master", [parse_spawn("s1", "frontend2", "SUCCEEDS: frontend")], source_request=14)
    store.record_first_request("frontend2")
    respond(store, "master", [msg("m1", "frontend2", "ROLE-POLICY frontend\nNew frontend rule")], source_request=15)


def test_replay_rebuilds_the_live_state_exactly():
    store, log = new_store()
    scenario(store)
    replayed = RolePolicyStore.replay(log.records, Log(), clock=ticking())
    assert replayed.export() == store.export()
    for role in ("master", "decoder", "frontend", "frontend2"):
        assert replayed.snapshot(role) == store.snapshot(role)
    assert all(r["event"] == BATCH_EVENT for r in log.records)
    assert [r["batch"] for r in log.records] == list(range(1, len(log.records) + 1))


def test_replay_from_an_events_file_with_other_events(tmp_path):
    store, log = new_store()
    scenario(store)
    path = tmp_path / "events.jsonl"
    with path.open("w") as stream:
        for index, record in enumerate(log.records):
            stream.write(json.dumps({"time": 1.0 + index, "event": "logical_model_call_admitted"}) + "\n")
            stream.write(json.dumps({"time": 1.5 + index, **record}) + "\n")
    replayed = RolePolicyStore.replay(RolePolicyStore.read_log(path), Log())
    assert replayed.export() == store.export()
    with path.open("a") as stream:
        stream.write('{"event": "role_policy_batch", "batch"')
    with pytest.raises(ReplayError):
        RolePolicyStore.read_log(path)


def test_replay_refuses_incomplete_or_inconsistent_logs():
    store, log = new_store()
    scenario(store)
    records = log.records
    with pytest.raises(ReplayError):
        RolePolicyStore.replay(records[:3] + records[4:], Log())
    with pytest.raises(ReplayError):
        RolePolicyStore.replay(records[1:], Log())
    with pytest.raises(ReplayError):
        RolePolicyStore.replay([], Log())
    tampered = json.loads(json.dumps(records))
    for record in tampered:
        for change in record["changes"]:
            if change["event"] == "role_policy_committed" and change["slot"] == "decoder":
                change["version"] += 5
    with pytest.raises(ReplayError):
        RolePolicyStore.replay(tampered, Log())
    truncated = json.loads(json.dumps(records))
    truncated[-1]["changes"][0].pop("text", None)
    truncated[-1]["changes"][0].pop("slot", None)
    with pytest.raises(ReplayError):
        RolePolicyStore.replay(truncated, Log())


def test_replayed_store_continues_the_batch_sequence():
    store, log = new_store()
    scenario(store)
    tail = Log()
    replayed = RolePolicyStore.replay(log.records, tail, clock=ticking())
    result = respond(replayed, "decoder", [msg("p9", "leader", proposal(change="After replay"))])
    assert result.batch == len(log.records) + 1 == tail.records[0]["batch"]


# -- concurrency ---------------------------------------------------------------------------

def test_concurrent_proposals_and_decisions_keep_invariants():
    store, log = new_store()
    workers = [f"w{i}" for i in range(6)]
    team(store, *workers)
    errors = []

    def worker(name, seed):
        rng = random.Random(seed)
        for step in range(40):
            try:
                respond(store, name, [msg(f"{name}-{step}", "leader", proposal(change=f"{name} rule {rng.randint(0, 3)}"))])
            except PolicyRejection as exc:
                if exc.code != "stale_version":
                    errors.append(exc)

    def master(seed):
        rng = random.Random(seed)
        for step in range(120):
            snapshot = store.snapshot("master")
            if not snapshot.proposals:
                continue
            target = rng.choice(snapshot.proposals)
            decision = rng.choice(["approve", "reject", "amend\nAmended by master"])
            try:
                store.commit(store.stage("master", snapshot, [msg(
                    f"d-{step}", target.proposer, f"POLICY-DECISION #{target.proposal_id} {decision}")]))
            except PolicyRejection as exc:
                if exc.code not in {"stale_proposal", "stale_version"}:
                    errors.append(exc)

    threads = [threading.Thread(target=worker, args=(name, i)) for i, name in enumerate(workers)]
    threads.append(threading.Thread(target=master, args=(99,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    state = store.export()
    assert [r["batch"] for r in log.records] == list(range(1, len(log.records) + 1))
    for slot, data in state["slots"].items():
        assert [v["version"] for v in data["versions"]] == list(range(1, len(data["versions"]) + 1))
    pending_by_slot = {}
    for proposal_id, item in state["proposals"].items():
        if item["status"] == "pending":
            assert item["slot_id"] not in pending_by_slot
            # export() keys proposals by str(id); pending holds the int ids.
            pending_by_slot[item["slot_id"]] = int(proposal_id)
    assert pending_by_slot == state["pending"]
    assert RolePolicyStore.replay(log.records, Log()).export() == state
