"""Real outer-agent loops and controller, with only in-process scripted completions."""

import copy
import json
from collections import Counter

import pytest

from agent_company.evolution.loop import TaskResult, TaskSpec, run_stage1
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.runtime.relay import OpenAIChatTransport, ProviderConfig, TokenRates
from agent_company.evolution.reviewer import StrategyReviewer


USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}
ONE_CALL_COST = 0.00022


def _call(name, arguments, identifier):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def _response(*calls):
    return 200, {"choices": [{"message": {"role": "assistant", "content": None,
                                           "tool_calls": list(calls)}}], "usage": dict(USAGE)}


class EvidenceScript:
    """Perform actual bounded evidence tools before each candidate submission/review."""

    def __init__(self, role):
        self.role = role
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        messages = payload["messages"]
        initial = json.loads(messages[1]["content"])
        request = initial["incumbent"] if self.role == "reviewer" else initial
        batch, candidate = request["batch_index"], request["candidate_index"]
        step = sum(message["role"] == "assistant" for message in messages)
        task_id = "train-one" if batch == 1 else "train-two"
        prefix = f"batch_{batch:03d}/"
        serialized = json.dumps(payload)
        assert "heldout-private-repo" not in serialized and "heldout-private-task" not in serialized
        assert "HELDOUT_ANSWER_SENTINEL" not in serialized
        assert request["evidence_scope"] == "current_batch" and request["change_history"] == []
        assert len(request["batch_results"]) == 1 and request["batch_results"][0]["task_id"] == task_id
        assert request["current_version"] == (0 if batch == 1 else 2)
        assert request["assignment_policy"] == f"assignment-policy-{request['current_version']}"
        assert request["communication_policy"] == f"communication-policy-{request['current_version']}"
        assert all(row["directory"].startswith(prefix) for row in request["evidence_files"]["tasks"])
        if batch == 2:
            # The incumbent policy persists, but old task evidence and outer-agent
            # rationales/summary content must not be copied into a new context.
            assert "BATCH_1_PRIVATE_EVIDENCE" not in serialized
            assert "PROPOSAL_RATIONALE_BATCH_1" not in serialized
            assert "REVIEW_RATIONALE_BATCH_1" not in serialized
        if step == 0:
            assert len(messages) == 2
            return _response(_call("list_training_files", {}, "list"))

        replies = {message["tool_call_id"]: json.loads(message["content"])
                   for message in messages if message["role"] == "tool"}
        if step == 1:
            paths = [row["path"] for row in replies["list"]["files"]]
            assert all(path.startswith(prefix) and "/replay/" not in path and "/feedback/" not in path
                       for path in paths)
            trace = next(path for path in paths if path.endswith("trace.txt"))
            return _response(_call("read_training_file", {"path": trace}, "read"))
        if step == 2:
            assert replies["read"]["text"] == f"BATCH_{batch}_PRIVATE_EVIDENCE: RECIPIENT_ACTED_AFTER_MESSAGE"
            calls = [_call("search_training_files", {"query": "RECIPIENT_ACTED_AFTER_MESSAGE"}, "search")]
            if batch == 2:
                calls.append(_call("read_training_file", {"path": "batch_001/task_001_train-one/trace.txt"},
                                   "old_batch_denied"))
            if candidate == 2:
                calls.append(_call("read_training_file", {
                    "path": prefix + "feedback/candidate_001/proposal.json"}, "other_candidate_denied"))
            return _response(*calls)
        assert step == 3
        matches = replies["search"]["matches"]
        assert len(matches) == 1 and matches[0]["path"] == replies["read"]["path"]
        if batch == 2:
            assert "Only available training-run artifacts" in replies["old_batch_denied"]["error"]
        if candidate == 2:
            assert "Only available training-run artifacts" in replies["other_candidate_denied"]["error"]
        version = (batch - 1) * 2 + candidate
        if self.role == "optimizer":
            return _response(_call("submit_policy", {
                "assignment_policy": f"assignment-policy-{version}",
                "communication_policy": f"communication-policy-{version}",
                "rationale": f"PROPOSAL_RATIONALE_BATCH_{batch}_CANDIDATE_{candidate}"}, "submit"))
        assert initial["candidate"]["assignment_policy"] == f"assignment-policy-{version}"
        assert initial["candidate"]["communication_policy"] == f"communication-policy-{version}"
        return _response(_call("submit_review", {
            # A negative recommendation must not prevent a candidate with a
            # better official replay score from being selected.
            "recommendation": "prefer_incumbent",
            "rationale": f"REVIEW_RATIONALE_BATCH_{batch}_CANDIDATE_{candidate}",
            "evidence_refs": [matches[0]["path"]], "risks": ["Replay must verify the hypothesis."]}, "submit"))


def test_real_optimizer_reviewer_multi_candidate_current_batch_cost_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(OpenAIChatTransport, "__call__",
                        lambda *_: pytest.fail("No provider transport may run in this scripted test"))
    root = tmp_path / "experiment"
    provider = ProviderConfig(model="outer-scripted-only", base_url="https://provider.invalid/v1",
                              api_key="fake-no-network-key", max_output_tokens=256,
                              rates=TokenRates(2, 0.5, 4))
    optimizer_script, reviewer_script = EvidenceScript("optimizer"), EvidenceScript("reviewer")
    optimizer = StrategyOptimizer(provider, root / "optimizer_traces", completion=optimizer_script)
    reviewer = StrategyReviewer(provider, root / "reviewer_traces", completion=reviewer_script)
    task_calls = []

    def run_task(request):
        task_calls.append(request)
        if request.split == "train":
            batch = 1 if request.task.task_id == "train-one" else 2
            trace = f"BATCH_{batch}_PRIVATE_EVIDENCE: RECIPIENT_ACTED_AFTER_MESSAGE"
        elif request.split == "train_replay":
            trace = "REPLAY_PRIVATE_EVIDENCE_NOT_FOR_OTHER_CANDIDATES"
        else:
            trace = "HELDOUT_ANSWER_SENTINEL"
        (request.output_dir / "trace.txt").write_text(trace, encoding="utf-8")
        return TaskResult("completed", score=(request.policy_version + 1) / 5,
                          usage={"total_tokens": 12}, cost_usd=0.1, known_cost_usd=0.1)

    arguments = {
        "train_tasks": [TaskSpec("train-one", "training-repo"), TaskSpec("train-two", "training-repo")],
        "test_tasks": [TaskSpec("heldout-private-task", "heldout-private-repo")],
        "initial_communication": "communication-policy-0", "assignment_policy": "assignment-policy-0",
        "run_task": run_task, "propose_policy": optimizer, "review_policy": reviewer,
        "output_dir": root, "batch_size": 1, "candidates_per_batch": 2,
    }
    summary = run_stage1(**arguments)
    assert summary["status"] == "completed" and summary["final_policy_version"] == 4
    assert summary["evidence_scope"] == "current_batch"
    assert summary["selection_counts"] == {"accept": 2, "reject": 0, "inconclusive": 0}
    assert [row["selected_candidate_index"] for row in summary["selection_results"]] == [2, 2]
    assert all(row["recommendation"] == "prefer_incumbent" for row in summary["reviewer_runs"])
    assert Counter(request.split for request in task_calls) == {"train": 2, "train_replay": 4, "test": 2}
    assert len(optimizer_script.payloads) == len(reviewer_script.payloads) == 16
    assert summary["optimizer"]["known_cost_usd"] == pytest.approx(16 * ONE_CALL_COST)
    assert summary["reviewer"]["known_cost_usd"] == pytest.approx(16 * ONE_CALL_COST)
    expected_cost = 8 * 0.1 + 32 * ONE_CALL_COST
    assert summary["all_costs"]["cost_usd"] == pytest.approx(expected_cost)
    assert summary["all_costs"]["known_cost_usd"] == pytest.approx(expected_cost)
    assert summary["all_costs"]["records"] == 16
    assert summary["all_costs"]["records_with_unknown_cost"] == 0

    expected_keys = {"batch_0001", "batch_0001_candidate_0002", "batch_0002", "batch_0002_candidate_0002"}
    for role, result_file in (("optimizer", "proposal.json"), ("reviewer", "review.json")):
        traces = root / f"{role}_traces"
        assert {path.name for path in traces.iterdir()} == expected_keys
        for directory in traces.iterdir():
            result = json.loads((directory / result_file).read_text())
            assert result["usage"]["provider_attempts"] == 4
            assert result["usage"]["total_tokens"] == 480
            assert result["cost_usd"] == pytest.approx(4 * ONE_CALL_COST)
            assert len(list(directory.glob("request_*.json"))) == 4
            assert len(list(directory.glob("response_*.json"))) == 4

    # Completed outer and task checkpoints must recover all K candidates without
    # repeating model calls, task execution, or charging the same usage twice.
    resumed = run_stage1(**arguments, resume=True)
    assert resumed["all_costs"] == summary["all_costs"]
    assert resumed["selection_results"] == summary["selection_results"]
    assert len(task_calls) == 8
    assert len(optimizer_script.payloads) == len(reviewer_script.payloads) == 16
