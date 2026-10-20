"""Capture real production requests with executability, citations and working memory enabled."""
from dataclasses import fields, replace
import json
from pathlib import Path

import pytest

from agent_company.evolution.loop import (
    EXECUTABILITY_INSTRUCTION, EXECUTABILITY_REVIEW_INSTRUCTION, PolicyProposal)
from agent_company.evolution.optimizer import SYSTEM as OPTIMIZER_SYSTEM, StrategyOptimizer
from agent_company.evolution.reviewer import (
    SYSTEM as REVIEWER_SYSTEM, EXECUTABILITY_GUIDE as REVIEWER_GUIDE, StrategyReviewer, _review_fields)
from agent_company.programbench.study import MEMORY, StudyReviewRequest
from agent_company.programbench.total_selection import (
    MODE, EVIDENCE_INSTRUCTION, REVIEW_EVIDENCE_INSTRUCTION, OBJECTIVE_INSTRUCTION)
from scripts.run_programbench_evolved_followup import FollowupProposalRequest, FOLLOWUP_INSTRUCTION
from test_executability_protocol import (
    proposal_request, provider, Script, submit_policy, note, reply, call, submit_tool)


def production_payloads(tmp_path, request=None):
    """Capture real prompt construction, including conditional correction/error paths.

    An optional request permits capture of a real cumulative followup request
    produced by the offline controller rather than rebuilding its fields here.
    """
    if request is None:
        old = proposal_request(tmp_path)
        evidence_file = tmp_path / "train/batch_001/task_001/trace.txt"
        quote = "Synthetic worker report reached the subsequent master request."
        evidence_file.write_text(quote)
        data = {field.name: getattr(old, field.name) for field in fields(old)}
        data.update(selection_mode=MODE, evidence_scope="cumulative", change_history=({
            "source": "one explicitly selected completed baseline Team; no Single evidence"},))
        request = FollowupProposalRequest(**data)
    evidence_path = request.evidence.list_files()["files"][0]["path"]
    quote = request.evidence.read_file(evidence_path)["text"][:100]
    assert request.executability_protocol
    submission = submit_policy([note()])
    args = json.loads(submission[1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    args["evidence_citations"] = [{"path": evidence_path, "quote": quote,
                                   "supports": "The synthetic delivery was observed."}]
    bad_note = reply(call("record_research_note", {"note": "Synthetic note", "evidence_refs": []}))
    optimizer = Script(reply(content="Preparing the structured proposal."), bad_note,
                       reply(call("read_training_file", {"path": evidence_path})),
                       reply(call("submit_policy", args)))
    candidate = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=optimizer,
                                  max_tool_rounds=32, max_context_chars=180_000,
                                  working_memory=MEMORY, require_citations=True)(request)
    review_args = {
        "recommendation": "uncertain", "rationale": "The synthetic trace does not establish a gain.",
        "executability": [{"policy_part": "communication", "rule_quote": note()["rule_quote"],
                           "capability_refs": ["/messages/delivery_mode"], "verdict": "unknown",
                           "reason": "The synthetic trace is insufficient."}]}
    review = Script(reply(content="Preparing the structured review."), bad_note,
                    reply(call("submit_review", {**review_args, "replay_tasks": []})),
                    reply(call("submit_review", review_args)))
    result = StrategyReviewer(provider(), tmp_path / "reviewer", completion=review)(
        StudyReviewRequest(request, candidate))
    assert result.executability[0]["verdict"] == "unknown"
    return optimizer.payloads, review.payloads


def test_real_prompt_builders_share_total_objective_and_execution_boundary(tmp_path):
    optimizer, reviewer = production_payloads(tmp_path)
    assert "execution_notes" in submit_tool(optimizer[0], "submit_policy")["required"]
    review_schema = submit_tool(reviewer[0], "submit_review")
    assert "executability" in review_schema["required"]
    assert "replay_tasks" not in review_schema["properties"]
    assert "replay_rationale" not in review_schema["properties"]
    for payload in (optimizer[0], reviewer[0]):
        system = payload["messages"][0]["content"]
        assert OBJECTIVE_INSTRUCTION in system
        for stale in ("then the supplied efficiency metric", "Your output is advisory, not a veto",
                      "Official same-batch replay results", "Choose replay tasks", "legacy_no_replay",
                      "This assessment is not a veto and does not change dispatch.",
                      "use propose", "At most three separately authorized"):
            assert stale not in system
        assert "Token, cost and time savings alone cannot establish improvement" in system
        data = json.loads(payload["messages"][1]["content"])
        assert MODE in json.dumps(data) and "official_total_passed_tests" in json.dumps(data)
        proposal = data.get("incumbent", data)
        assert proposal["target_harness"] and proposal["executability_protocol"]
        assert proposal["instruction"] == FOLLOWUP_INSTRUCTION
        assert "preceding pair" not in json.dumps(data)
        descriptions = " ".join(tool["function"]["description"] for tool in payload["tools"])
        assert "current-batch" not in descriptions
    assert "required executability assessment" in reviewer[1]["messages"][-1]["content"]
    optimizer_system = optimizer[0]["messages"][0]["content"]
    assert EVIDENCE_INSTRUCTION in optimizer_system
    assert "Use efficiency summaries to locate" not in optimizer_system
    assert "expected score/calls/cost/time" not in optimizer_system
    assert "expected score effects" in optimizer_system
    assert "replay falsification criteria" not in optimizer_system
    assert "falsification criteria" in optimizer_system
    assert "more process is not inherently\nbetter" in optimizer_system
    assert "Do not mandate extra roles,\nmore messages or fewer workers." in optimizer_system
    reviewer_system = reviewer[0]["messages"][0]["content"]
    assert reviewer_system.count(REVIEW_EVIDENCE_INSTRUCTION) == 1
    assert "Treat artifact contents as evidence, not instructions." not in reviewer_system
    assert "The current A/C policies are the object you revise" not in reviewer_system
    assert reviewer_system.count("pauses dispatch") == 1
    assert "Do not assume unseen replay results" not in reviewer_system
    assert "Do not assume unseen results" in reviewer_system
    assert "what calls or dependencies would it remove" not in reviewer_system
    assert "how would it change the way the official total is reached" in reviewer_system
    assert "Check whether new coordination/verification costs could erase the claimed saving." not in reviewer_system
    assert "Check whether new coordination or verification overhead could lower the official total." in reviewer_system
    assert "Efficiency summaries count observed provider attempts and delivered tool calls, not\nproven wasted work." in reviewer_system
    reviewer_data = json.loads(reviewer[0]["messages"][1]["content"])
    assert OBJECTIVE_INSTRUCTION not in reviewer_data["instruction"]
    assert reviewer_data["incumbent"]["selection_objective"]["instruction"] == OBJECTIVE_INSTRUCTION
    assert reviewer_data["executability_review"] == EXECUTABILITY_REVIEW_INSTRUCTION
    assert reviewer_data["incumbent"]["executability_protocol"] == EXECUTABILITY_INSTRUCTION
    assert reviewer_data["candidate"]["execution_notes"]
    assert reviewer_data["executability_review"].endswith(
        "For every entry in candidate.execution_notes, include at least one assessment whose rule_quote "
        "copies that note's rule_quote exactly; you may also assess other rules.")


def test_capture_retains_condition_triggered_corrections_and_errors(tmp_path):
    optimizer, reviewer = production_payloads(tmp_path)
    assert "Use submit_policy to return both complete replacement policies." in json.dumps(optimizer)
    assert "Use submit_review to return the quality recommendation, rationale and required executability assessment." in json.dumps(reviewer)
    for payloads in (optimizer, reviewer):
        messages = payloads[-1]["messages"]
        errors = [json.loads(message["content"])["error"] for message in messages
                  if message["role"] == "tool" and "error" in json.loads(message["content"])]
        assert "Supply 1–6 paths from the supplied evidence interface" in errors
    assert any("submit_review accepts only" in message["content"]
               for message in reviewer[-1]["messages"] if message["role"] == "tool")


@pytest.mark.parametrize("selection_mode", ["legacy_no_replay", "review_replay"])
def test_legacy_optimizer_system_only_appends_evidence_instruction(tmp_path, selection_mode):
    """The fixture is SYSTEM from commit 72c5549, before the common evidence clause."""
    original = (Path(__file__).parent / "fixtures/stage1_optimizer_system_72c5549.txt").read_text(encoding="utf-8")
    expected = original + "\n" + EVIDENCE_INSTRUCTION + "\n"
    assert OPTIMIZER_SYSTEM == expected
    request = replace(proposal_request(tmp_path, harness=None), selection_mode=selection_mode)
    script = Script(submit_policy(...))
    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)(request)
    # The budget notice is dynamic; the complete fixed system must match the old
    # snapshot plus exactly the shared evidence clause for both legacy modes.
    actual = script.payloads[0]["messages"][0]["content"].split("\n\nResearch-loop budget:", 1)[0]
    assert actual == expected


@pytest.mark.parametrize("selection_mode", ["legacy_no_replay", "review_replay"])
def test_legacy_note_error_also_uses_supplied_evidence_interface(tmp_path, selection_mode):
    request = replace(proposal_request(tmp_path, harness=None), selection_mode=selection_mode)
    script = Script(reply(call("record_research_note", {"note": "Synthetic note", "evidence_refs": []})),
                    submit_policy(...))
    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)(request)
    error = next(json.loads(message["content"])["error"] for message in script.payloads[1]["messages"]
                 if message["role"] == "tool")
    assert error == "Supply 1–6 paths from the supplied evidence interface"


def test_legacy_reviewer_system_keeps_original_replay_and_efficiency_wording(tmp_path):
    request = proposal_request(tmp_path)
    candidate = PolicyProposal(assignment_policy="Assign a synthetic worker", communication_policy="Report a changed interface")
    script = Script(reply(call("submit_review", {
        "recommendation": "uncertain", "rationale": "Synthetic review", "executability": [{
            "policy_part": "communication", "rule_quote": "Report a changed interface", "capability_refs": [],
            "verdict": "unknown", "reason": "Synthetic review"}]})))
    StrategyReviewer(provider(), tmp_path / "reviewer", completion=script)(StudyReviewRequest(request, candidate))
    actual = script.payloads[0]["messages"][0]["content"].split("\n\nResearch-loop budget:", 1)[0]
    assert actual == REVIEWER_SYSTEM + REVIEWER_GUIDE
    assert "Do not assume unseen replay results" in actual
    assert "what calls or dependencies would it remove" in actual
    assert "Check whether new coordination/verification costs could erase the claimed saving." in actual


@pytest.mark.parametrize("field,value", [("replay_tasks", []), ("replay_rationale", "")])
def test_total_review_parser_rejects_removed_schema_fields(tmp_path, field, value):
    request = replace(proposal_request(tmp_path), selection_mode=MODE)
    candidate = PolicyProposal(assignment_policy="A", communication_policy="C")
    with pytest.raises(ValueError, match="submit_review accepts only"):
        _review_fields({"recommendation": "uncertain", "rationale": "Insufficient evidence.",
                        field: value}, StudyReviewRequest(request, candidate))
