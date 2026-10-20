"""Opt-in executability protocol for the optimizer and reviewer, with scripted completions.

Plan: docs/stage1_optimizer_harness_feasibility_plan_20261012.md (v2), steps B-D. Without
target_harness the requests, tools and records stay as before; with it, the optimizer must
submit execution_notes and the reviewer an executability list. Checks are structural only
(fields, verbatim quotes, resolvable pointers); dispatch is unchanged.
"""
import json
from pathlib import Path

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.loop import (
    EXECUTABILITY_INSTRUCTION, EXECUTABILITY_REVIEW_INSTRUCTION, PolicyProposal, PolicyReview, ProposalRequest,
    ReviewRequest)
from agent_company.evolution.harness_capabilities import target_harness
from agent_company.evolution.optimizer import EXECUTABILITY_GUIDE, MAX_EXECUTION_NOTES, StrategyOptimizer
from agent_company.runtime.relay import ProviderConfig
from agent_company.evolution.reviewer import (
    EXECUTABILITY_GUIDE as REVIEW_GUIDE, MAX_EXECUTABILITY_ITEMS, StrategyReviewer, _review_fields)


ROOT = Path(__file__).resolve().parents[1]
HARNESS = target_harness(json.loads((ROOT / "configs/stage1/programbench_ffmpeg_s0_propose_20261011.json")
                                    .read_text())["inference"], execution_mode="team")
KEY = "fake-protocol-key-never-on-network"
ASSIGNMENT = "Assign one worker per decoder family. The integration worker merges every hour."
COMMUNICATION = ("Report a changed interface to the master within one tool call.\n"
                 "If the master does not answer, continue and report again at the next check-in.")


def provider():
    return ProviderConfig(model="fake-only", base_url="https://provider.invalid/v1", api_key=KEY,
                          max_output_tokens=256)


def call(name, args, identifier="call-1"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*calls, content=None):
    return 200, {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": list(calls)}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


class Script:
    def __init__(self, *steps):
        self.steps, self.payloads = list(steps), []

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        step = self.steps.pop(0)
        return step(payload) if callable(step) else step


def proposal_request(tmp_path, *, harness=HARNESS):
    root = tmp_path / "train"
    run = root / "batch_001" / "task_001"
    run.mkdir(parents=True, exist_ok=True)
    (run / "trace.txt").write_text("training evidence")
    return ProposalRequest(current_version=0, communication_policy="C0 incumbent.",
                           assignment_policy="A0 incumbent.", batch_index=1,
                           batch_results=({"task_id": "train-a", "score": 0.5},), change_history=(),
                           history_total=0, evidence=TrainingEvidence(root, [run]), target_harness=harness)


def note(**overrides):
    return {"policy_part": "communication", "rule_quote": "Report a changed interface to the master",
            "actor": "worker", "observable_trigger": "Its own edit changes a public header.",
            "existing_action": "send_message", "capability_refs": ["/messages/delivery_mode", "/wakeups"],
            "timing_and_fallback": "Delivered at the master's next model step; repeat at the next check-in.",
            "cost_and_limitations": "One message per interface change."} | overrides


def submit_policy(notes, *, assignment=ASSIGNMENT, communication=COMMUNICATION):
    args = {"assignment_policy": assignment, "communication_policy": communication,
            "rationale": "Late interface reports delayed integration."}
    if notes is not ...:
        args["execution_notes"] = notes
    return reply(call("submit_policy", args, "submit"))


def submit_tool(payload, name):
    return next(tool["function"]["parameters"] for tool in payload["tools"] if tool["function"]["name"] == name)


# ---- requests: absent target_harness keeps the earlier prompt data ----

def test_prompt_data_changes_only_when_target_harness_is_supplied(tmp_path):
    plain = proposal_request(tmp_path, harness=None)
    with_harness = proposal_request(tmp_path)
    assert not plain.executability_protocol and with_harness.executability_protocol
    base = plain.as_prompt_data()
    assert "target_harness" not in base and "executability_protocol" not in base
    data = with_harness.as_prompt_data()
    assert data == {**base, "target_harness": HARNESS, "executability_protocol": EXECUTABILITY_INSTRUCTION}

    candidate = PolicyProposal(communication_policy=COMMUNICATION, assignment_policy=ASSIGNMENT, rationale="r")
    old = ReviewRequest(proposal_request=plain, candidate=candidate).as_prompt_data()
    assert set(old["candidate"]) == {"assignment_policy", "communication_policy", "rationale"}
    assert "executability_review" not in old
    new = ReviewRequest(proposal_request=with_harness, candidate=candidate).as_prompt_data()
    assert new["candidate"]["execution_notes"] == "not_recorded"  # Absent is not "no notes".
    assert new["executability_review"] == EXECUTABILITY_REVIEW_INSTRUCTION
    assert new["incumbent"]["target_harness"] == HARNESS
    noted = ReviewRequest(proposal_request=with_harness, candidate=PolicyProposal(
        communication_policy=COMMUNICATION, assignment_policy=ASSIGNMENT, rationale="r", execution_notes=[]))
    assert noted.as_prompt_data()["candidate"]["execution_notes"] == []


# ---- optimizer ----

def test_optimizer_without_the_protocol_keeps_tools_and_record(tmp_path):
    script = Script(submit_policy([note()]))  # Unrequested notes are ignored, as before.
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)(
        proposal_request(tmp_path, harness=None))
    assert result.execution_notes is None
    schema = submit_tool(script.payloads[0], "submit_policy")
    assert "execution_notes" not in schema["properties"] and "execution_notes" not in schema["required"]
    assert EXECUTABILITY_GUIDE not in script.payloads[0]["messages"][0]["content"]
    assert json.loads((tmp_path / "optimizer/batch_0001/proposal.json").read_text())["execution_notes"] is None


def test_optimizer_protocol_requires_and_records_execution_notes(tmp_path):
    notes = [note(), note(policy_part="assignment", rule_quote="The integration worker   merges\nevery hour",
                      actor="integration worker", capability_refs=["/timing/agent_wall_time_seconds"])]
    script = Script(submit_policy(notes))
    optimizer = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)
    request = proposal_request(tmp_path)
    result = optimizer(request)
    assert result.execution_notes == notes  # Whitespace differences in a quote are accepted, text kept.
    payload = script.payloads[0]
    assert EXECUTABILITY_GUIDE in payload["messages"][0]["content"]
    assert json.loads(payload["messages"][1]["content"])["target_harness"] == HARNESS
    schema = submit_tool(payload, "submit_policy")
    assert "execution_notes" in schema["required"]
    assert schema["properties"]["execution_notes"]["items"]["additionalProperties"] is False
    # The shared tool list is not mutated for later, protocol-free requests.
    later = Script(submit_policy(...))
    StrategyOptimizer(provider(), tmp_path / "plain", completion=later)(proposal_request(tmp_path, harness=None))
    assert "execution_notes" not in submit_tool(later.payloads[0], "submit_policy")["properties"]
    saved = json.loads((tmp_path / "optimizer/batch_0001/proposal.json").read_text())
    assert saved["execution_notes"] == notes
    assert optimizer.recover(request) == result and len(script.payloads) == 1


def test_optimizer_accepts_an_empty_note_list_for_an_unchanged_policy(tmp_path):
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=Script(submit_policy([])))(
        proposal_request(tmp_path))
    assert result.execution_notes == []


@pytest.mark.parametrize("notes,message", [
    (..., "list"),
    ("not a list", "list"),
    ([note()] * (MAX_EXECUTION_NOTES + 1), "at most"),
    ([{k: v for k, v in note().items() if k != "actor"}], "exactly"),
    ([note(extra="x")], "exactly"),
    ([note(policy_part="both")], "policy_part"),
    ([note(policy_part=["communication"])], "policy_part"),
    ([note(actor=" ")], "actor"),
    ([note(existing_action=7)], "existing_action"),
    ([note(rule_quote="Report")], "verbatim"),
    ([note(rule_quote="Report a changed API to the master")], "verbatim"),
    ([note(policy_part="assignment")], "verbatim"),  # Quote is from the other policy.
    ([note(rule_quote="x" * 601)], "verbatim"),
    ([note(capability_refs=[])], "pointers"),
    ([note(capability_refs="/wakeups")], "pointers"),
    ([note(capability_refs=["/wakeups/no_such_field"])], "not a target_harness field"),
    ([note(capability_refs=["wakeups"])], "not a target_harness field"),
])
def test_invalid_notes_are_tool_feedback_then_a_corrected_submission_commits(tmp_path, notes, message):
    def corrected(payload):
        assert message in json.loads(payload["messages"][-1]["content"])["error"]
        assert not (tmp_path / "optimizer/batch_0001/proposal.json").exists()
        return submit_policy([note()])

    script = Script(submit_policy(notes), corrected)
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)(proposal_request(tmp_path))
    assert result.execution_notes == [note()] and len(script.payloads) == 2


def test_optimizer_notes_are_redacted(tmp_path):
    communication = f"Report a changed interface to the master. Never paste {KEY}."
    script = Script(submit_policy([note(actor=f"worker {KEY}")], communication=communication))
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)(proposal_request(tmp_path))
    assert KEY not in json.dumps(result.execution_notes)
    for path in (tmp_path / "optimizer").rglob("proposal.json"):
        assert KEY not in path.read_text()


# ---- reviewer ----

def review_request(tmp_path, *, harness=HARNESS, notes=None):
    return ReviewRequest(proposal_request=proposal_request(tmp_path, harness=harness),
                         candidate=PolicyProposal(communication_policy=COMMUNICATION, assignment_policy=ASSIGNMENT,
                                                  rationale="Late interface reports.", execution_notes=notes))


def item(**overrides):
    return {"policy_part": "communication", "rule_quote": "report again at the next check-in",
            "capability_refs": ["/wakeups/checkin_interval_seconds"], "verdict": "supported_with_conditions",
            "reason": "Check-ins exist every 30 minutes, not at arbitrary times.",
            "conditions": "The policy must accept a delay of up to one check-in interval."} | overrides


def submit_review(items, recommendation="uncertain"):
    args = {"recommendation": recommendation, "rationale": "Mixed evidence."}
    if items is not ...:
        args["executability"] = items
    return reply(call("submit_review", args, "review"))


def test_reviewer_without_the_protocol_rejects_the_field_and_keeps_tools(tmp_path):
    with pytest.raises(ValueError, match="accepts only"):
        _review_fields({"recommendation": "uncertain", "rationale": "x", "executability": [item()]})
    script = Script(submit_review(...))
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=script)(
        review_request(tmp_path, harness=None))
    assert result.executability is None
    schema = submit_tool(script.payloads[0], "submit_review")
    assert "executability" not in schema["properties"] and "executability" not in schema["required"]
    assert REVIEW_GUIDE not in script.payloads[0]["messages"][0]["content"]


def test_reviewer_protocol_requires_and_records_its_own_assessment(tmp_path):
    items = [item(), item(policy_part="assignment", rule_quote="merges every hour", verdict="unknown",
                          capability_refs=[], reason="No timer for arbitrary time points is listed.")]
    for entry in items[1:]:
        entry.pop("conditions")
    script = Script(submit_review(items, "prefer_candidate"))
    reviewer = StrategyReviewer(provider(), tmp_path / "reviewer", completion=script)
    request = review_request(tmp_path, notes=[note()])
    result = reviewer(request)
    assert result.recommendation == "prefer_candidate"  # Verdicts do not change the recommendation.
    assert result.executability == [items[0], {**items[1], "conditions": ""}]
    payload = script.payloads[0]
    assert REVIEW_GUIDE in payload["messages"][0]["content"]
    prompt = json.loads(payload["messages"][1]["content"])
    assert prompt["incumbent"]["target_harness"] == HARNESS
    assert prompt["candidate"]["execution_notes"] == [note()]
    schema = submit_tool(payload, "submit_review")
    assert "executability" in schema["required"] and schema["properties"]["executability"]["minItems"] == 1
    saved = json.loads((tmp_path / "reviewer/batch_0001/review.json").read_text())
    assert saved["executability"] == result.executability
    assert reviewer.recover(request) == result and len(script.payloads) == 1


@pytest.mark.parametrize("items,message", [
    (..., "list 1-"),
    ([], "list 1-"),
    ([item()] * (MAX_EXECUTABILITY_ITEMS + 1), "list 1-"),
    ([{k: v for k, v in item().items() if k != "reason"}], "needs"),
    ([item(score=1)], "needs"),
    ([item(verdict="executable")], "unknown policy_part or verdict"),
    ([item(policy_part="both")], "unknown policy_part or verdict"),
    ([item(verdict=["supported"])], "unknown policy_part or verdict"),
    ([item(reason="")], "reason"),
    ([item(conditions=" ")], "conditions"),
    ([item(conditions=3, verdict="supported")], "conditions"),
    ([item(rule_quote="C0 incumbent.")], "verbatim"),  # Quotes must come from the candidate.
    ([item(rule_quote=None)], "verbatim"),
    ([item(capability_refs="/wakeups")], "pointers"),
    ([item(capability_refs=["/online_policy/no_such_field"])], "not a target_harness field"),
])
def test_invalid_assessment_is_tool_feedback_then_can_be_corrected(tmp_path, items, message):
    def corrected(payload):
        assert message in json.loads(payload["messages"][-1]["content"])["error"]
        return submit_review([item()])

    script = Script(submit_review(items), corrected)
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=script)(review_request(tmp_path))
    assert result.executability == [item()] and len(script.payloads) == 2


# ---- records written before the protocol ----

def test_records_without_the_new_fields_recover_as_not_recorded(tmp_path):
    request = review_request(tmp_path)
    proposal = PolicyProposal(communication_policy=COMMUNICATION, assignment_policy=ASSIGNMENT, rationale="r")
    old_proposal = {k: v for k, v in vars(proposal).items() if k != "execution_notes"}
    path = tmp_path / "optimizer/batch_0001/proposal.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old_proposal))
    recovered = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=Script()).recover(
        request.proposal_request)
    assert recovered.execution_notes is None and recovered.communication_policy == COMMUNICATION

    review = PolicyReview("uncertain", "Old review")
    old_review = {k: v for k, v in vars(review).items() if k != "executability"}
    path = tmp_path / "reviewer/batch_0001/review.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old_review))
    recovered = StrategyReviewer(provider(), tmp_path / "reviewer", completion=Script()).recover(request)
    assert recovered.executability is None and recovered.recommendation == "uncertain"


# ---- context retention ----

def test_harness_and_note_requirement_survive_page_rotation_and_the_submission_only_step(tmp_path):
    from agent_company.evolution.outer_budget import WorkingMemory
    root = tmp_path / "train"
    (root / "task").mkdir(parents=True)
    (root / "task/a.txt").write_text("PAGE_A " * 100)
    (root / "task/b.txt").write_text("PAGE_B " * 100)
    request = ProposalRequest(current_version=0, communication_policy="C0 incumbent.",
                              assignment_policy="A0 incumbent.", batch_index=1,
                              batch_results=({"task_id": "t", "score": 0.5},), change_history=(), history_total=0,
                              evidence=TrainingEvidence(root, [root / "task"]), target_harness=HARNESS)

    def kept(payload):
        assert EXECUTABILITY_GUIDE in payload["messages"][0]["content"]
        assert json.loads(payload["messages"][1]["content"])["target_harness"] == HARNESS
        assert "execution_notes" in submit_tool(payload, "submit_policy")["required"]

    def second(payload):
        kept(payload)
        return reply(call("read_training_file", {"path": "task/b.txt"}, "r2"))

    def last(payload):
        kept(payload)
        pages = [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]
        assert any(page.get("elided_from_working_context") for page in pages)  # Page A left the context.
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["submit_policy"]
        return submit_policy([note()])

    def first(payload):
        kept(payload)
        return reply(call("read_training_file", {"path": "task/a.txt"}, "r1"))

    script = Script(first, second, last)
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script, max_tool_rounds=3,
                               working_memory=WorkingMemory(page_window=1))(request)
    assert result.execution_notes == [note()] and len(script.payloads) == 3


def test_reviewer_submission_only_step_still_requires_the_assessment(tmp_path):
    def only_submit(payload):
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["submit_review"]
        assert "executability" in submit_tool(payload, "submit_review")["required"]
        assert json.loads(payload["messages"][1]["content"])["incumbent"]["target_harness"] == HARNESS
        return submit_review([item()])

    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=Script(only_submit),
                              max_tool_rounds=1)(review_request(tmp_path))
    assert result.executability == [item()]
