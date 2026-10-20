"""Plan v4 section 12, B1: reasoning goes back to the provider verbatim and never reaches disk."""
import json
from pathlib import Path

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.loop import PolicyProposal, ProposalRequest, ReviewRequest
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.evolution.outer_budget import WorkingMemory
from agent_company.evolution.reviewer import StrategyReviewer
from agent_company.evolution.trace import without_reasoning
from agent_company.runtime.relay import ProviderConfig

SENTINEL = "SENTINEL-REASONING-7f3a"


def provider():
    return ProviderConfig(model="fake-only", base_url="https://provider.invalid/v1",
                          api_key="fake-test-key-never-used-on-network", max_output_tokens=256, max_attempts=2)


def call(name, args, identifier):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def request(tmp_path):
    root = tmp_path / "train"
    (root / "task").mkdir(parents=True)
    (root / "task/a.txt").write_text("alpha reported the interface change late", encoding="utf-8")
    return ProposalRequest(current_version=0, communication_policy="C", assignment_policy="A", batch_index=1,
                           batch_results=({"task_id": "t", "score": 0.5},), change_history=(), history_total=0,
                           evidence=TrainingEvidence(root, [root / "task"]))


class Provider:
    """Scripted turns; every assistant turn carries its own reasoning sentinel."""

    def __init__(self, turns, retry_first=False):
        self.turns, self.payloads, self.retry_first = list(turns), [], retry_first

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        if self.retry_first:
            self.retry_first = False
            # A retryable reply that itself carries reasoning: the retry record must not keep it.
            return 503, {"choices": [{"message": {"role": "assistant", "content": "",
                                                  "reasoning_content": f"{SENTINEL}-retry"}}]}
        index = len(self.payloads)
        message = {"role": "assistant", "content": None, "reasoning_content": f"{SENTINEL}-{index}",
                   "tool_calls": [self.turns.pop(0)]}
        return 200, {"choices": [{"message": message, "finish_reason": "tool_calls"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    def assistant_reasoning(self, index):
        return [m.get("reasoning_content") for m in self.payloads[index]["messages"] if m["role"] == "assistant"]


def assert_never_on_disk(directory: Path):
    files = [path for path in directory.rglob("*") if path.is_file()]
    assert files
    leaked = [str(path) for path in files if SENTINEL in path.read_text(encoding="utf-8")]
    assert leaked == []


def test_optimizer_passes_reasoning_back_and_never_writes_it(tmp_path):
    turns = [call("list_training_files", {}, "c1"),
             call("read_training_file", {"path": "task/a.txt"}, "c2"),
             call("record_research_note", {"note": "Late interface notice.", "evidence_refs": ["task/a.txt"]}, "c3"),
             call("submit_policy", {"assignment_policy": "A2", "communication_policy": "C2", "rationale": "r"}, "c4")]
    model = Provider(turns, retry_first=True)
    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=model, working_memory=WorkingMemory(),
                      transport_retries=1, sleep=lambda seconds: None)(request(tmp_path))
    # payloads[0] is the retried attempt, payloads[1] its resend; later rounds follow.
    assert model.assistant_reasoning(1) == []
    assert model.assistant_reasoning(2) == [f"{SENTINEL}-2"]
    assert model.assistant_reasoning(3) == [f"{SENTINEL}-2", f"{SENTINEL}-3"]
    assert model.assistant_reasoning(4) == [f"{SENTINEL}-2", f"{SENTINEL}-3", f"{SENTINEL}-4"]
    assert_never_on_disk(tmp_path / "optimizer")
    saved = json.loads((tmp_path / "optimizer/batch_0001/request_003.json").read_text())
    assert [m["role"] for m in saved["messages"]].count("assistant") == 3  # the turns are kept, only reasoning removed


def test_reviewer_passes_reasoning_back_and_never_writes_it(tmp_path):
    candidate = PolicyProposal(communication_policy="C2", assignment_policy="A2", rationale="r")
    turns = [call("list_training_files", {}, "v1"),
             call("read_training_file", {"path": "task/a.txt"}, "v2"),
             call("record_research_note", {"note": "Checked.", "evidence_refs": ["task/a.txt"]}, "v3"),
             call("submit_review", {"recommendation": "uncertain", "rationale": "Not enough evidence."}, "v4")]
    model = Provider(turns)
    StrategyReviewer(provider(), tmp_path / "reviewer", completion=model)(ReviewRequest(request(tmp_path), candidate))
    assert model.assistant_reasoning(3) == [f"{SENTINEL}-1", f"{SENTINEL}-2", f"{SENTINEL}-3"]
    assert_never_on_disk(tmp_path / "reviewer")


def test_failed_sessions_do_not_write_reasoning_either(tmp_path):
    model = Provider([call("read_training_file", {"path": "task/a.txt"}, f"r{i}") for i in range(3)])
    try:
        StrategyOptimizer(provider(), tmp_path / "optimizer", completion=model, max_tool_rounds=3,
                          working_memory=WorkingMemory())(request(tmp_path))
    except Exception:
        pass
    assert_never_on_disk(tmp_path / "optimizer")


def test_without_reasoning_strips_model_messages_only():
    payload = {"reasoning": {"effort": "high"}, "thinking": {"type": "enabled"},
               "messages": [{"role": "assistant", "content": "a", "reasoning_content": "x", "reasoning_details": [1]},
                            {"role": "tool", "content": json_text({"reasoning": "evidence key"})}]}
    assert without_reasoning(payload) == {
        "reasoning": {"effort": "high"}, "thinking": {"type": "enabled"},
        "messages": [{"role": "assistant", "content": "a"},
                     {"role": "tool", "content": json_text({"reasoning": "evidence key"})}]}
    response = {"choices": [{"message": {"content": "", "reasoning": "y"}, "delta": {"reasoning_content": "z"}}]}
    assert without_reasoning(response) == {"choices": [{"message": {"content": ""}, "delta": {}}]}
    assert payload["messages"][0]["reasoning_content"] == "x"  # the original is untouched
    # Evidence results are kept as they were, even with a key named reasoning.
    assert without_reasoning([{"tool_call_id": "c", "result": {"reasoning": "kept"}}]) == [
        {"tool_call_id": "c", "result": {"reasoning": "kept"}}]


def json_text(value):
    return json.dumps(value)


def test_a_reply_without_a_role_is_stripped_in_requests_and_conversations_too(tmp_path):
    class RolelessReply(Provider):
        def __call__(self, payload):
            status, body = super().__call__(payload)
            body["choices"][0]["message"].pop("role")
            return status, body
    turns = [call("list_training_files", {}, "c1"),
             call("submit_policy", {"assignment_policy": "A2", "communication_policy": "C2", "rationale": "r"}, "c2")]
    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=RolelessReply(turns),
                      working_memory=WorkingMemory())(request(tmp_path))
    assert_never_on_disk(tmp_path / "optimizer")
    conversation = [{"content": "a", "reasoning_content": "x"}, {"role": "user", "content": "u"}]
    assert without_reasoning(conversation, message=True) == [{"content": "a"}, {"role": "user", "content": "u"}]
    assert without_reasoning({"messages": conversation}) == {"messages": [{"content": "a"}, {"role": "user", "content": "u"}]}
