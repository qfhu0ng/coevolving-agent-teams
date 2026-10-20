"""Role policies wired into the relay with a scripted upstream (S2b; no CLI, no model).

Specification: docs/stage1_role_policy_harness_plan_20261009.md v3.3 sections 3.2, 3.3, 3.6.
"""
import json

import pytest

from agent_company.programbench.comparison import runner_inference_options
from agent_company.runtime.qwen import RUNTIME_COMMUNICATION_POLICY, QwenTeamRunner, master_system
from agent_company.runtime.relay import TOOL_ERROR_FIELD, ProviderConfig, TeamRelay
from agent_company.policy.role_policy import parse_send_message
from agent_company.policy.role_policy_runtime import DISCLOSURE_P_W, DISCLOSURE_R_M
from agent_company.runtime.team import WorkerRegistry
from test_team import call, request, response, workspace

MASTER_TOOLS = ("agent", "send_message", "task_list")
WORKER_TOOLS = ("send_message", "run_shell_command")
PROPOSAL = "POLICY-PROPOSAL\nCHANGE: Message frontend on flag changes.\nREASON: Rework.\nEVIDENCE: Task 4."


class Upstream:
    """Scripted provider: returns queued responses and keeps every payload it saw."""

    def __init__(self):
        self.queue, self.payloads = [], []

    def __call__(self, payload):
        self.payloads.append(payload)
        return 200, self.queue.pop(0)


def relay_for(tmp_path, mode="propose"):
    upstream = Upstream()
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      "Starting C.", tmp_path / "logs", 60, upstream, role_policy_mode=mode)
    return relay, upstream


def system_text(payload):
    return payload["messages"][0]["content"]


def turn(relay, upstream, role, calls=(), history=()):
    upstream.queue.append(response(list(calls)))
    body = request(role, MASTER_TOOLS if role == "master" else WORKER_TOOLS)
    body["messages"].extend(history)
    status, delivered = relay.handle(body)
    return status, delivered, upstream.payloads[-1]


def events(tmp_path, name=None):
    rows = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()]
    return [row for row in rows if name is None or row["event"] == name]


def spawn(relay, upstream, *names):
    calls = [call("agent", {"name": name, "description": name,
                            "prompt": "Implement it.\nROLE-POLICY:\nTell frontend about flag changes.\nEND-ROLE-POLICY"
                            if name == "decoder" else f"Work on {name}."}) for name in names]
    status, delivered, _ = turn(relay, upstream, "master", calls)
    assert status == 200
    return calls


def rejection_text(delivered):
    args = json.loads(delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    return args[TOOL_ERROR_FIELD]


def test_off_mode_keeps_every_request_unchanged(tmp_path):
    relay, upstream = relay_for(tmp_path, "off")
    spawn(relay, upstream, "decoder")
    _, _, payload = turn(relay, upstream, "decoder")
    assert "Communication policy for this task:\nStarting C." in system_text(payload)
    assert "Communication policy C (version" not in system_text(payload)
    assert relay.role_policy is None and "role_policy" not in relay.summary()
    assert not [row for row in events(tmp_path) if row["event"].startswith("role_policy")]
    (tmp_path / "s").mkdir()
    with pytest.raises(ValueError):
        TeamRelay(ProviderConfig("fake", "http://localhost"), WorkerRegistry(workspace(tmp_path / "s")), "",
                  tmp_path / "s-logs", 60, Upstream(), execution_mode="single", role_policy_mode="master")


def test_spawn_marker_binding_and_injection(tmp_path):
    relay, upstream = relay_for(tmp_path)
    _, _, master_payload = turn(relay, upstream, "master")
    assert "Communication policy C (version 1):\nStarting C." in system_text(master_payload)
    assert DISCLOSURE_R_M in system_text(master_payload) and "No slots yet." in system_text(master_payload)
    calls = spawn(relay, upstream, "decoder", "frontend")
    prompt = json.loads(relay._model_views[("master", calls[0]["id"])][0]["visible"])["prompt"] \
        if ("master", calls[0]["id"]) in relay._model_views else None
    assert prompt is not None and prompt.startswith("Implement it.")
    _, _, worker_payload = turn(relay, upstream, "decoder")
    text = system_text(worker_payload)
    assert "Communication policy for this task" not in text
    assert DISCLOSURE_P_W in text
    assert "Your role slot: decoder; you are decoder. Your role policy version 1" in text
    assert "Tell frontend about flag changes." in text
    _, _, master_payload = turn(relay, upstream, "master")
    assert "Slot decoder: instance decoder (running)" in system_text(master_payload)
    assert "Slot frontend: instance frontend (running); no role policy; C applies." in system_text(master_payload)
    injected = events(tmp_path, "role_policy_injected")
    assert [row["source_request"] for row in injected] == list(range(1, len(injected) + 1))
    assert injected[-1]["slot_versions"] == {"decoder": 1, "frontend": 0}


def test_proposal_decision_round_trip_through_the_relay(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    proposal_call = call("send_message", {"to": "leader", "message": PROPOSAL})
    status, delivered, _ = turn(relay, upstream, "decoder", [proposal_call])
    assert status == 200 and TOOL_ERROR_FIELD not in delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    _, _, master_payload = turn(relay, upstream, "master")
    assert ("Proposal #1 for slot decoder from decoder, base version 1" in system_text(master_payload)
            and "CHANGE: Message frontend on flag changes." in system_text(master_payload))
    turn(relay, upstream, "master", [call("send_message", {"to": "decoder", "message": "POLICY-DECISION #1 approve"})])
    _, _, worker_payload = turn(relay, upstream, "decoder", history=answered(proposal_call, 'Message sent to "leader".'))
    text = system_text(worker_payload)
    assert "Your role policy version 2" in text and "Message frontend on flag changes." in text
    assert "Your proposal #1 was approved as version 2." in text
    assert [row["kind"] for row in events(tmp_path, "role_policy_message_sent")] == ["ProposalCall"]
    counts = relay.summary()["role_policy"]["counts"]
    assert counts["role_policy_proposed"] == 1 and counts["role_policy_decided"] == 1
    assert counts["notification_sent"] == 1 and relay.summary()["role_policy"]["unresolved_notifications"] == 1
    batches = events(tmp_path, "role_policy_batch")
    assert [b["batch"] for b in batches] == list(range(1, len(batches) + 1))


def test_initial_rejection_returns_the_whole_response_unexecuted(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    turn(relay, upstream, "decoder")  # first request binds the instance
    before = relay.role_policy.export()
    status, delivered, _ = turn(relay, upstream, "decoder", [
        call("send_message", {"to": "decoder", "message": "ROLE-POLICY decoder\nX"}),
        call("run_shell_command", {"command": "ls"})])
    assert status == 200 and "not_authorized" in rejection_text(delivered)
    assert all(TOOL_ERROR_FIELD in json.loads(c["function"]["arguments"])
               for c in delivered["choices"][0]["message"]["tool_calls"])
    assert relay.role_policy.export() == before
    (rejected,) = events(tmp_path, "role_policy_message_rejected")
    assert (rejected["stage"], rejected["code"], rejected["discarded_calls"]) == ("initial", "not_authorized", 2)
    finished = events(tmp_path, "provider_attempt_finished")[-1]
    assert finished["tool_rejection"]["reason"] == "role_policy"


def test_malformed_header_is_rejected_not_sent(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    _, delivered, _ = turn(relay, upstream, "decoder", [call("send_message", {"to": "leader", "message": "**POLICY-PROPOSAL**\nx"})])
    assert "malformed_header" in rejection_text(delivered)


def test_late_commit_rejection_after_a_concurrent_replacement(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    turn(relay, upstream, "decoder", [call("send_message", {"to": "leader", "message": PROPOSAL})])
    turn(relay, upstream, "master")  # the master's snapshot now carries #1
    store, original_stage = relay.role_policy, relay.role_policy.stage

    def stage_then_replace(actor, snapshot, calls, **kwargs):
        staged = original_stage(actor, snapshot, calls, **kwargs)
        replacement = parse_send_message("p2", "leader", PROPOSAL.replace("Task 4", "Task 5"))
        store.commit(original_stage("decoder", store.snapshot("decoder"), [replacement]))
        return staged
    store.stage = stage_then_replace
    decision = call("send_message", {"to": "decoder", "message": "POLICY-DECISION #1 approve"})
    status, delivered, _ = turn(relay, upstream, "master", [decision])
    store.stage = original_stage
    assert status == 200 and "stale_proposal" in rejection_text(delivered)
    assert [p.proposal_id for p in store.snapshot("master").proposals] == [2]
    assert store.snapshot("decoder").slots[0].policy.version == 1
    late = events(tmp_path, "role_policy_message_rejected")[-1]
    assert (late["stage"], late["code"]) == ("late", "stale_proposal")
    # The delivered file is what the CLI received; the staged version is kept beside it.
    (staged,) = (tmp_path / "logs").glob("attempt_*_master_qwen_response_policy_staged.json")
    returned = json.loads(staged.with_name(staged.name.replace("_policy_staged", "")).read_text())
    assert TOOL_ERROR_FIELD in json.loads(returned["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert TOOL_ERROR_FIELD not in staged.read_text()
    assert len(relay._model_views[("master", decision["id"])]) == 1  # accepted view unchanged; rejection added


def test_late_commit_is_refused_after_the_relay_closes(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    store, original_stage = relay.role_policy, relay.role_policy.stage

    def stage_then_close(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        relay.close()
        return staged
    store.stage = stage_then_close
    status, delivered, _ = turn(relay, upstream, "master",
                                [call("send_message", {"to": "decoder", "message": "ROLE-POLICY decoder\nV2"})])
    assert status == 424 and delivered["error"]["code"] in {"request_cancelled", "task_deadline_exceeded"}
    assert store.snapshot("master").slot("decoder").policy.version == 1


def test_spawn_failure_result_frees_the_name(tmp_path):
    relay, upstream = relay_for(tmp_path)
    calls = spawn(relay, upstream, "decoder")
    failure = [{"role": "assistant", "content": "", "tool_calls": [calls[0]]},
               {"role": "tool", "tool_call_id": calls[0]["id"],
                "content": 'Failed to spawn teammate "decoder": Maximum number of teammates (1) reached.'}]
    turn(relay, upstream, "master", history=failure)
    assert relay.role_policy.snapshot("master").slots[0].instance is None
    assert events(tmp_path, "role_policy_spawn_result")[-1]["outcome"] == "voided"
    status, delivered, _ = turn(relay, upstream, "master", [call("agent", {"name": "decoder", "description": "d",
                                                                           "prompt": "Again."})])
    assert status == 200 and relay.role_policy.snapshot("master").slots[0].instance == "decoder"


def test_duplicate_spawn_name_is_rejected_before_native(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    turn(relay, upstream, "decoder")
    _, delivered, _ = turn(relay, upstream, "master", [call("agent", {"name": "decoder", "description": "d",
                                                                      "prompt": "Again."})])
    assert "duplicate_name" in rejection_text(delivered)


def test_global_mode_broadcast_and_versioned_c(tmp_path):
    relay, upstream = relay_for(tmp_path, "global")
    _, delivered, _ = turn(relay, upstream, "master", [call("agent", {
        "name": "decoder", "description": "d", "prompt": "Task\nROLE-POLICY:\nIgnored in global mode."})])
    assert TOOL_ERROR_FIELD not in delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    _, delivered, _ = turn(relay, upstream, "master", [call("send_message", {"to": "decoder", "message": "GLOBAL-POLICY\nC2"})])
    assert "must be broadcast" in rejection_text(delivered)
    turn(relay, upstream, "master", [call("send_message", {"to": "*", "message": "GLOBAL-POLICY base=v1\nC2"})])
    _, _, worker_payload = turn(relay, upstream, "decoder")
    text = system_text(worker_payload)
    assert "Communication policy C (version 2):\nC2" in text and "Role-policy snapshot" not in text
    assert "Your role slot" not in text


def test_master_prompt_and_plan_options():
    static = master_system("A", "Static C text")
    runtime = master_system("A", "Static C text", communication_in_runtime=True)
    assert "Static C text" in static and "Static C text" not in runtime and RUNTIME_COMMUNICATION_POLICY in runtime
    inference = {"memory_mb": 1, "cpus": 1, "agent_timeout_sec": 10}
    assert runner_inference_options(inference)["role_policy_mode"] == "off"
    assert runner_inference_options({**inference, "role_policy_mode": "propose"})["role_policy_mode"] == "propose"
    with pytest.raises(ValueError):
        runner_inference_options({**inference, "role_policy_mode": "on"})
    with pytest.raises(ValueError):
        QwenTeamRunner("cli.js", role_policy_mode="everyone")


# ----- review fixes (2026-10-10) -----------------------------------------------------

def fixed_call(identifier, name, args):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def answered(call_obj, text):
    return [{"role": "assistant", "content": "", "tool_calls": [call_obj]},
            {"role": "tool", "tool_call_id": call_obj["id"], "content": [{"type": "text", "text": text}]}]


@pytest.mark.parametrize("spawn_error", ["Maximum number of teammates (10) reached.",
                                         "Synthetic backend spawn failure."])
def test_native_bare_error_text_counts_as_spawn_and_send_failure(tmp_path, spawn_error):
    relay, upstream = relay_for(tmp_path)
    calls = spawn(relay, upstream, "decoder")
    # Native hands the model only error.message for a failed tool (no "Failed to ..." prefix).
    turn(relay, upstream, "master", history=answered(calls[0], spawn_error))
    assert relay.role_policy.snapshot("master").slots[0].instance is None
    status, delivered, _ = turn(relay, upstream, "master", [call("agent", {"name": "decoder", "description": "d",
                                                                           "prompt": "Retry."})])
    assert status == 200 and TOOL_ERROR_FIELD not in delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    turn(relay, upstream, "decoder")
    proposal_call = call("send_message", {"to": "leader", "message": PROPOSAL})
    turn(relay, upstream, "decoder", [proposal_call])
    turn(relay, upstream, "decoder", history=answered(proposal_call, 'Teammate "leader" is no longer active.'))
    (failed,) = events(tmp_path, "role_policy_message_send_failed")
    assert failed["detail"].startswith('Teammate "leader"') and relay.summary()["role_policy"]["counts"]["notification_send_failed"] == 1
    assert [p.proposal_id for p in relay.role_policy.snapshot("master").proposals] == [1]  # state is not rolled back


def test_results_are_matched_per_role_and_per_call_arguments(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder", "frontend")
    turn(relay, upstream, "decoder")
    turn(relay, upstream, "frontend")
    first = fixed_call("call_0", "send_message", {"to": "leader", "message": PROPOSAL})
    second = fixed_call("call_0", "send_message", {"to": "leader", "message": PROPOSAL.replace("Task 4", "Task 9")})
    turn(relay, upstream, "decoder", [first])
    turn(relay, upstream, "frontend", [second])
    assert relay.summary()["role_policy"]["unresolved_notifications"] == 2
    # A later, different call reusing the id is not taken as the proposal's result.
    reused = fixed_call("call_0", "send_message", {"to": "leader", "message": "ordinary note"})
    turn(relay, upstream, "decoder", history=answered(reused, 'Message sent to "leader".'))
    assert relay.summary()["role_policy"]["unresolved_notifications"] == 2
    turn(relay, upstream, "decoder", history=answered(first, 'Message sent to "leader".'))
    turn(relay, upstream, "frontend", history=answered(second, 'Message sent to "leader".'))
    assert relay.summary()["role_policy"]["counts"]["notification_sent"] == 2
    assert relay.summary()["role_policy"]["unresolved_notifications"] == 0


def test_injection_evidence_failure_does_not_leave_the_role_in_flight(tmp_path):
    relay, upstream = relay_for(tmp_path)
    original = relay._policy_injected
    failures = iter([OSError("disk full")])

    def failing(*args, **kwargs):
        error = next(failures, None)
        if error is not None:
            raise error
        return original(*args, **kwargs)
    relay._policy_injected = failing
    status, delivered = relay.handle(request("master", MASTER_TOOLS))
    assert status == 424 and delivered["error"]["code"] == "evidence_write_failed"
    assert "master" not in relay._active_requests and upstream.payloads == []
    status, _, _ = turn(relay, upstream, "master")
    assert status == 200


def test_every_attempt_carries_its_logical_request_number(tmp_path):
    relay, upstream = relay_for(tmp_path)
    spawn(relay, upstream, "decoder")
    turn(relay, upstream, "decoder")
    attempts = events(tmp_path, "provider_attempt_finished")
    injected = events(tmp_path, "role_policy_injected")
    assert [a["source_request"] for a in attempts] == [i["source_request"] for i in injected] == [1, 2]



# ----- second review (2026-10-11) -------------------------------------------------------

class Conversation:
    """One role's native history, grown turn by turn like the CLI does."""

    def __init__(self, relay, upstream, role):
        self.relay, self.upstream, self.role, self.messages = relay, upstream, role, []

    def turn(self, calls=(), results=()):
        """Send the history, return the delivered response; results answer the calls in order."""
        status, delivered, payload = turn(self.relay, self.upstream, self.role, list(calls), history=self.messages)
        self.messages.append({"role": "assistant", "content": "", "tool_calls": list(calls)})
        for call_obj, text in zip(calls, results):
            self.messages.append({"role": "tool", "tool_call_id": call_obj["id"],
                                  "content": [{"type": "text", "text": text}]})
        return status, delivered


def test_a_reused_call_id_in_one_role_never_answers_the_new_call(tmp_path):
    relay, upstream = relay_for(tmp_path)
    master = Conversation(relay, upstream, "master")
    first = fixed_call("call_0", "agent", {"name": "decoder", "description": "d", "prompt": "First try."})
    second = fixed_call("call_0", "agent", {"name": "decoder", "description": "d", "prompt": "Second try."})
    master.turn([first], ["Maximum number of teammates (10) reached."])
    master.turn([second], ['Teammate "decoder" is now running concurrently. Task: d'])
    master.turn()  # history holds the old failure and the new success under the same id
    assert relay.role_policy.snapshot("master").slot("decoder").instance == "decoder"
    assert relay.role_policy.record_first_request("decoder") == "bound"
    outcomes = [row["outcome"] for row in events(tmp_path, "role_policy_spawn_result")]
    assert outcomes == ["voided", "ok"]

    worker = Conversation(relay, upstream, "decoder")
    sent = fixed_call("call_0", "send_message", {"to": "leader", "message": PROPOSAL})
    again = fixed_call("call_0", "send_message", {"to": "leader", "message": PROPOSAL})
    worker.turn([sent], ['Message sent to "leader".'])
    worker.turn([again], ['Teammate "leader" is no longer active.'])
    worker.turn()
    results = [(row["event"], row["source_request"]) for row in events(tmp_path)
               if row["event"] in {"role_policy_message_sent", "role_policy_message_send_failed"}]
    assert [kind for kind, _ in results] == ["role_policy_message_sent", "role_policy_message_send_failed"]
    assert results[0][1] != results[1][1]


def test_injection_failure_is_recorded_as_never_sent_upstream(tmp_path):
    relay, upstream = relay_for(tmp_path)
    relay._policy_injected = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full"))
    status, _ = relay.handle(request("master", MASTER_TOOLS))
    assert status == 424
    (error,) = [row for row in relay.summary()["errors"] if row.get("role_policy")]
    assert error["upstream_started"] is False


def test_started_attempts_carry_the_logical_request_number(tmp_path):
    relay, upstream = relay_for(tmp_path)
    turn(relay, upstream, "master")
    turn(relay, upstream, "master")
    assert [row["source_request"] for row in events(tmp_path, "provider_attempt_started")] == [1, 2]
