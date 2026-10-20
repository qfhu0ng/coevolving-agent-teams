"""Opt-in optimizer research protocol: working memory, structured reads, citations.

The default protocol is covered by test_optimizer.py and must stay unchanged.
These scripted completions replace only the provider call.
"""
import json
from types import SimpleNamespace

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.loop import ProposalRequest
from agent_company.evolution.optimizer import (
    SYSTEM, TOOLS, PolicyOptimizationError, StrategyOptimizer, unmatched_identifiers,
)
from agent_company.evolution.outer_budget import WorkingMemory
from agent_company.runtime.relay import ProviderConfig
from agent_company.runtime.transport import (
    InvalidProviderResponse, OpenAIChatTransport, RequestCancelled, RequestDeadlineExceeded, TransportError,
)

REAL_QUOTE = "You are the SPEC SCOUT for a cleanroom reimplementation task."


def provider():
    return ProviderConfig(model="fake-only", base_url="https://provider.invalid/v1",
                          api_key="fake-test-key-never-used-on-network", max_output_tokens=256)


def call(name, args, identifier="call-1"):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*calls, content=None):
    return 200, {"choices": [{"message": {"role": "assistant", "content": content,
                                           "tool_calls": list(calls)}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def submit(citations=None, rationale="Interface ownership reduced stale handoffs."):
    args = {"assignment_policy": "A2", "communication_policy": "C2", "rationale": rationale}
    if citations is not None:
        args["evidence_citations"] = citations
    return reply(call("submit_policy", args, "submit"))


class Script:
    """Each step receives the provider payload; records every payload."""

    def __init__(self, *steps):
        self.steps, self.payloads = list(steps), []

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        step = self.steps.pop(0)
        return step(payload) if callable(step) else step


def request(tmp_path, files):
    root = tmp_path / "train"
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    directories = sorted({root / relative.split("/")[0] for relative in files})
    return ProposalRequest(current_version=3, communication_policy="C", assignment_policy="A",
                           batch_index=1, batch_results=({"task_id": "t", "score": 0.5},),
                           change_history=(), history_total=0,
                           evidence=TrainingEvidence(root, directories))


COLLABORATION = json.dumps({"scope": "outward_tool_proposals_not_confirmed_execution", "entries": [
    {"attempt": 2, "role": "master", "tool": "agent",
     "fields": {"prompt": {"text": REAL_QUOTE + " Probe the reference."}}},
    {"attempt": 9, "role": "scout", "tool": "send_message",
     "fields": {"content": {"text": "Spec ready.\nSee spec/behavior.md"}}},
]}, indent=2)


def tool_messages(payload):
    return [json.loads(message["content"]) for message in payload["messages"] if message["role"] == "tool"]


def test_default_protocol_keeps_earlier_tools_prompt_and_outputs(tmp_path):
    script = Script(submit())
    optimizer = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)
    result = optimizer(request(tmp_path, {"task/a.json": COLLABORATION}))
    payload = script.payloads[0]
    assert payload["tools"] == TOOLS
    assert payload["messages"][0]["content"].startswith(SYSTEM)
    assert "evidence_index" not in json.loads(payload["messages"][1]["content"])
    assert result.reading_audit == {}
    assert not (tmp_path / "optimizer/batch_0001/reading_ledger.json").exists()


def test_pages_leave_working_context_while_notes_persist(tmp_path):
    files = {"task/a.txt": "PAGE_A " * 100, "task/b.txt": "PAGE_B " * 100}

    def after_first_read(payload):
        assert tool_messages(payload)[-1]["text"].startswith("PAGE_A")
        return reply(call("record_research_note", {"note": "A shows X", "evidence_refs": ["task/a.txt"]}, "n1"),
                     call("read_training_file", {"path": "task/b.txt"}, "r2"))

    def after_second_read(payload):
        results = tool_messages(payload)
        page_a = next(r for r in results if r.get("path") == "task/a.txt")
        assert page_a["elided_from_working_context"] is True and "text" not in page_a
        assert page_a["chars"] == len("PAGE_A " * 100)
        assert results[-1]["text"].startswith("PAGE_B")
        assert "A shows X" in json.dumps(payload["messages"])  # The note call stays in history.
        return submit()

    script = Script(reply(call("read_training_file", {"path": "task/a.txt"}, "r1")),
                    after_first_read, after_second_read)
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script,
                               working_memory=WorkingMemory(page_window=1))(request(tmp_path, files))
    audit = result.reading_audit
    assert audit["mode"] == "working_memory" and audit["elided_pages"] == 1 and audit["notes"] == 1
    assert audit["coverage"]["task"] == {"files_opened": 2, "files_touched": 2, "search_excerpt_chars": 0,
                                         "files_total": 2, "chars_shown": 1400,
                                         "bytes_total": 1400, "approx_fraction": 1.0}
    # The full page remains in the on-disk trace even after elision from context.
    trace = json.loads((tmp_path / "optimizer/batch_0001/tool_results_000.json").read_text())
    assert trace[0]["result"]["text"] == "PAGE_A " * 100


def test_fully_read_non_ascii_file_counts_as_complete(tmp_path):
    files = {"task/zh.txt": "中文证据" * 500, "task/unopened.txt": "x" * 2000}
    script = Script(reply(call("read_training_file", {"path": "task/zh.txt"})), submit())
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script,
                               working_memory=WorkingMemory())(request(tmp_path, files))
    row = result.reading_audit["coverage"]["task"]
    # Opened files count in characters; the unopened one is estimated by bytes.
    assert row["chars_shown"] == 2000 and row["approx_fraction"] == 0.5


def _reader(paths):
    queue, reads = list(paths), []

    def step(payload):
        names = [tool["function"]["name"] for tool in payload["tools"]]
        if queue and "read_training_file" in names:
            reads.append(queue.pop(0))
            return reply(call("read_training_file", {"path": reads[-1]}, f"r{len(reads)}"))
        if queue and names == ["record_research_note", "submit_policy"]:
            return reply(call("record_research_note", {"note": f"read {len(reads)} pages",
                                                       "evidence_refs": [reads[-1]]}))
        return submit()
    return step, reads


def test_working_memory_covers_both_tasks_where_paged_history_stops_early(tmp_path):
    files = {f"{task}/page_{index:02d}.txt": f"{task}{index:02d}" + "x" * 11_990
             for task in ("gron", "fd") for index in range(10)}
    paths = sorted(files, key=lambda path: (path.startswith("fd"), path))  # gron first, as in PB1

    legacy_step, legacy_reads = _reader(paths)
    StrategyOptimizer(provider(), tmp_path / "legacy", completion=lambda p: legacy_step(p),
                      max_context_chars=60_000, max_tool_rounds=30)(request(tmp_path / "l", files))
    assert len(legacy_reads) < 6 and not any(path.startswith("fd") for path in legacy_reads)

    memory_step, memory_reads = _reader(paths)
    result = StrategyOptimizer(provider(), tmp_path / "memory", completion=lambda p: memory_step(p),
                               max_context_chars=60_000, max_tool_rounds=30,
                               working_memory=WorkingMemory())(request(tmp_path / "m", files))
    assert len(memory_reads) == 20
    coverage = result.reading_audit["coverage"]
    assert coverage["gron"]["approx_fraction"] == coverage["fd"]["approx_fraction"] == 1.0
    assert coverage["fd"]["files_opened"] == 10


def test_rotation_carries_notes_and_keeps_inspection_open_while_calls_remain(tmp_path):
    first = reply(call("record_research_note", {"note": "Keep this finding", "evidence_refs": ["task/a.txt"]}),
                  content="Long assistant prose. " * 2500)

    def after_rotation(payload):
        assert len(payload["messages"]) == 3
        notebook = json.loads(payload["messages"][2]["content"])
        assert notebook["notes"][0]["note"] == "Keep this finding" and notebook["rotations"] == 1
        assert "read_training_file" in [tool["function"]["name"] for tool in payload["tools"]]
        return submit()

    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=Script(first, after_rotation),
                               max_context_chars=60_000, working_memory=WorkingMemory())(
        request(tmp_path, {"task/a.txt": "evidence"}))
    assert result.reading_audit["rotations"] == 1


def test_fabricated_citation_is_rejected_and_can_be_corrected(tmp_path):
    fabricated = [{"path": "task/collab.json", "quote": "PRB123 onepasser reported Sending 5 facts",
                   "supports": "Workers sent redundant facts"}]
    genuine = [{"path": "task/collab.json", "quote": REAL_QUOTE, "supports": "Master briefed the scout"}]

    def after_rejection(payload):
        error = tool_messages(payload)[-1]["error"]
        assert "Citation check failed" in error and "not found verbatim" in error
        return submit(genuine)

    result = StrategyOptimizer(provider(), tmp_path / "optimizer",
                               completion=Script(submit(fabricated), after_rejection),
                               require_citations=True)(request(tmp_path, {"task/collab.json": COLLABORATION}))
    assert result.reading_audit["citations"] == [{"path": "task/collab.json", "quote": REAL_QUOTE,
                                                  "supports": "Master briefed the scout"}]
    saved = json.loads((tmp_path / "optimizer/batch_0001/proposal.json").read_text())
    assert saved["reading_audit"]["citations"][0]["quote"] == REAL_QUOTE


@pytest.mark.parametrize("citations", [None, [], [{"path": "task/collab.json", "quote": "too short",
                                                     "supports": "x"}]])
def test_required_citations_must_be_present_and_substantial(tmp_path, citations):
    genuine = [{"path": "task/collab.json", "quote": REAL_QUOTE, "supports": "briefing"}]

    def corrected(payload):
        assert "error" in tool_messages(payload)[-1]
        return submit(genuine)

    script = Script(submit(citations), corrected)
    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script,
                      require_citations=True)(request(tmp_path, {"task/collab.json": COLLABORATION}))
    submit_tool = script.payloads[0]["tools"][-1]["function"]["parameters"]
    assert "evidence_citations" in submit_tool["required"]


def test_two_final_calls_leave_room_for_one_citation_correction(tmp_path):
    genuine = [{"path": "task/collab.json", "quote": REAL_QUOTE, "supports": "briefing"}]
    script = Script(reply(call("read_training_json", {"path": "task/collab.json", "pointer": "/entries"})),
                    submit([{"path": "task/collab.json", "quote": "an invented but long enough quote",
                             "supports": "x"}]),
                    submit(genuine))
    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script, max_tool_rounds=3,
                      require_citations=True, working_memory=WorkingMemory(final_calls=2))(
        request(tmp_path, {"task/collab.json": COLLABORATION}))
    assert [[t["function"]["name"] for t in p["tools"]] for p in script.payloads[1:]] == [
        ["submit_policy"], ["submit_policy"]]


def test_rationale_identifiers_absent_from_shown_evidence_are_disclosed(tmp_path):
    rationale = "attempt_00002 briefed the scout; PRB123 and onepasser_2 wasted turns; P3 kept."
    assert unmatched_identifiers(rationale, "attempt_00002 ... P3") == ["PRB123", "onepasser_2"]
    genuine = [{"path": "task/collab.json", "quote": REAL_QUOTE, "supports": "briefing"}]
    script = Script(reply(call("read_training_file", {"path": "task/collab.json"})),
                    submit(genuine, rationale="Scout briefing at attempt 2; PRB123 was wasteful."))
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script,
                               working_memory=WorkingMemory(), require_citations=True)(
        request(tmp_path, {"task/collab.json": COLLABORATION}))
    assert result.reading_audit["unmatched_rationale_identifiers"] == ["PRB123"]


def test_index_and_structured_reads_navigate_without_cutting_items(tmp_path):
    evidence = request(tmp_path, {"task/collab.json": COLLABORATION, "task/log.jsonl": "{}\n{}\n",
                                  "task/notes.txt": "plain"}).evidence
    index = evidence.index()
    digests = {row["path"]: row for row in index["tasks"][0]["file_digests"]}
    assert digests["task/collab.json"]["entries"] == {
        "count": 2, "role_counts": {"master": 1, "scout": 1},
        "tool_counts": {"agent": 1, "send_message": 1}, "attempt_range": [2, 9]}
    assert digests["task/log.jsonl"]["lines"] == 2 and digests["task/notes.txt"]["type"] == "text"
    assert "file_digests_omitted" in evidence.index(max_files_per_directory=2)["tasks"][0]
    page = evidence.read_json("task/collab.json", pointer="/entries", start=1, count=5)
    assert page["total_items"] == 2 and page["next_start"] is None and page["items"][0]["role"] == "scout"
    assert evidence.read_json("task/collab.json", pointer="/entries/0/role")["value"] == "master"
    with pytest.raises(ValueError, match="segment not found"):
        evidence.read_json("task/collab.json", pointer="/missing")
    with pytest.raises(PermissionError):
        evidence.read_json("../outside.json")


def test_large_objects_list_keys_and_oversized_items_are_not_split(tmp_path):
    large = json.dumps({"small": 1, "big": "y" * 20_000, "rows": [{"blob": "z" * 13_000}, {"ok": 1}]})
    evidence = request(tmp_path, {"task/big.json": large}).evidence
    root = evidence.read_json("task/big.json")
    assert root["too_large_for_one_page"] and [key["key"] for key in root["keys"]] == ["small", "big", "rows"]
    rows = evidence.read_json("task/big.json", pointer="/rows")
    assert rows["items"] == [] and "exceeds one page" in rows["hint"]
    assert evidence.read_json("task/big.json", pointer="/big")["truncated"] is True


def test_quote_verification_tolerates_whitespace_and_one_json_escape_level(tmp_path):
    evidence = request(tmp_path, {"task/collab.json": COLLABORATION}).evidence
    assert evidence.verify_quote("task/collab.json", "Spec ready.\\nSee spec/behavior.md")["verified"]
    assert evidence.verify_quote("task/collab.json", "Spec ready.\nSee   spec/behavior.md")["verified"]
    assert not evidence.verify_quote("task/collab.json", "Spec ready and fully verified by tests")["verified"]
    with pytest.raises(ValueError):
        evidence.verify_quote("task/collab.json", "short")


def test_structured_json_tool_is_not_offered_by_default(tmp_path):
    def after_unknown(payload):
        assert tool_messages(payload)[-1] == {"error": "Unknown evidence tool"}
        return submit()

    script = Script(reply(call("read_training_json", {"path": "task/collab.json"})), after_unknown)
    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script)(
        request(tmp_path, {"task/collab.json": COLLABORATION}))


def _names(payload):
    return [tool["function"]["name"] for tool in payload["tools"]]


def test_heavy_reasoning_triggers_a_notes_only_call_before_rotation(tmp_path):
    """PB2 attempt 1: GLM reasoning (20–37k characters per turn) forced rotations
    that carried only four notes. Rotation is now preceded by a consolidation call."""
    phases = []

    def model(payload):
        names = _names(payload)
        notice = json.loads(payload["messages"][0]["content"].split("Research-loop budget: ")[1])
        phases.append(notice["phase"])
        if names == ["record_research_note", "submit_policy"]:
            return reply(call("record_research_note", {"note": "Consolidated: X then Y",
                                                       "evidence_refs": ["task/a.txt"]}))
        if len(payload["messages"]) == 3:
            notebook = json.loads(payload["messages"][2]["content"])
            assert notebook["notes"][-1]["note"] == "Consolidated: X then Y"
            assert "read_training_file" in names
            return submit()
        status, body = reply(call("read_training_file", {"path": "task/a.txt"}))
        body["choices"][0]["message"]["reasoning_content"] = "r" * 15_000
        return status, body

    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=model, max_context_chars=60_000,
                               working_memory=WorkingMemory(min_page_chars=2000))(
        request(tmp_path, {"task/a.txt": "evidence " * 50}))
    assert "consolidate_notes" in phases and phases[phases.index("consolidate_notes") + 1] == "inspect_or_submit"
    audit = result.reading_audit
    assert audit["consolidations"] == 1 and audit["rotations"] == 1 and audit["largest_turn_chars"] > 15_000


def test_evidence_tools_are_refused_during_consolidation(tmp_path):
    sneaked = []

    def model(payload):
        if _names(payload) == ["record_research_note", "submit_policy"]:
            sneaked.append(1)
            return reply(call("read_training_file", {"path": "task/a.txt"}, "sneak"))
        if sneaked:
            return submit()
        status, body = reply(call("read_training_file", {"path": "task/a.txt"}))
        body["choices"][0]["message"]["reasoning_content"] = "r" * 15_000
        return status, body

    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=model, max_context_chars=60_000,
                      working_memory=WorkingMemory(min_page_chars=2000))(request(tmp_path, {"task/a.txt": "e"}))
    results = [row["result"] for path in sorted((tmp_path / "optimizer/batch_0001").glob("tool_results_*.json"))
               for row in json.loads(path.read_text())]
    assert {"error": "Evidence reading is paused for this consolidation call; record notes now"} in results


def _retrying(tmp_path, first, *, retries=2, timeout=120):
    delays, attempts = [], []

    def model(payload):
        attempts.append(1)
        if len(attempts) == 1:
            if isinstance(first, Exception):
                raise first
            return first
        return submit()

    prov = ProviderConfig(model="fake-only", base_url="https://provider.invalid/v1", api_key=None,
                          max_output_tokens=256, request_timeout_seconds=timeout)
    optimizer = StrategyOptimizer(prov, tmp_path / "optimizer", completion=model,
                                  transport_retries=retries, sleep=delays.append)
    return optimizer, delays, attempts


def test_transport_error_is_resent_with_backoff_and_unknown_usage(tmp_path):
    optimizer, delays, attempts = _retrying(tmp_path, OSError("SSLV3_ALERT_BAD_RECORD_MAC"))
    result = optimizer(request(tmp_path, {"task/a.txt": "e"}))
    assert delays == [1.0] and len(attempts) == 2
    assert result.usage["provider_attempts"] == 2 and result.usage["unknown_usage_attempts"] == 1
    record = json.loads((tmp_path / "optimizer/batch_0001/retry_000_01.json").read_text())
    assert "SSLV3_ALERT_BAD_RECORD_MAC" in record["error"] and record["delay_seconds"] == 1.0


def test_retryable_status_honours_structured_retry_after(tmp_path):
    optimizer, delays, _ = _retrying(tmp_path, (524, {"error": {"message": "timeout", "retry_after": 5}}))
    optimizer(request(tmp_path, {"task/a.txt": "e"}))
    assert delays == [5.0]


def test_non_retryable_status_and_spent_deadline_are_not_resent(tmp_path):
    from agent_company.evolution.optimizer import PolicyOptimizationError

    optimizer, delays, _ = _retrying(tmp_path / "a", (400, {"error": {"message": "bad request"}}))
    with pytest.raises(PolicyOptimizationError, match="HTTP 400"):
        optimizer(request(tmp_path / "a", {"task/a.txt": "e"}))
    optimizer, delays, _ = _retrying(tmp_path / "b", OSError("reset"), timeout=0.5)
    with pytest.raises(PolicyOptimizationError, match="reset"):
        optimizer(request(tmp_path / "b", {"task/a.txt": "e"}))
    assert delays == []


@pytest.fixture
def retry_clock(monkeypatch):
    """Advance request time without sleeping or changing the event loop's clock."""
    from agent_company.evolution import optimizer
    from agent_company.runtime import transport

    clock = SimpleNamespace(now=0.0)

    def advance(seconds):
        clock.now += seconds

    clock.advance = advance
    clock.monotonic = lambda: clock.now
    for module in (optimizer, transport):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock.monotonic, time=clock.monotonic))
    return clock


@pytest.mark.parametrize("first_failure", [TransportError("reset"), 503])
def test_optimizer_real_transport_retries_share_original_deadline(tmp_path, monkeypatch, retry_clock,
                                                                first_failure):
    """Exercise real transport call/control plumbing; only HTTP work is scripted."""
    prov = ProviderConfig(model="fake", base_url="https://provider.invalid/v1",
                          request_timeout_seconds=600)
    transport = OpenAIChatTransport(prov)
    attempts = []

    async def exchange(payload, control, deadline, watch):
        attempts.append((retry_clock.now, control, deadline))
        if len(attempts) == 1:
            retry_clock.advance(598)
            if isinstance(first_failure, Exception):
                raise first_failure
            return first_failure, {"error": {"message": "busy"}}
        retry_clock.advance(1)
        control.check()
        return submit()  # A wrongly renewed deadline would accept this result.

    monkeypatch.setattr(transport, "_request", exchange)
    optimizer = StrategyOptimizer(prov, tmp_path / "optimizer", completion=transport,
                                  transport_retries=2, sleep=retry_clock.advance)
    with pytest.raises(PolicyOptimizationError, match="deadline") as caught:
        optimizer(request(tmp_path, {"task/a.txt": "e"}))
    assert retry_clock.now == 600
    assert [start for start, _, _ in attempts] == [0, 599]
    assert [deadline for _, _, deadline in attempts] == [600, 600]
    assert attempts[0][1] is attempts[1][1]
    assert caught.value.usage["provider_attempts"] == 2


@pytest.mark.parametrize("end_backoff", ["overslept", "cancelled"])
def test_optimizer_does_not_start_retry_after_backoff_ends_request(tmp_path, monkeypatch, retry_clock,
                                                                 end_backoff):
    prov = ProviderConfig(model="fake", base_url="https://provider.invalid/v1",
                          request_timeout_seconds=600)
    transport = OpenAIChatTransport(prov)
    controls = []

    async def exchange(payload, control, deadline, watch):
        controls.append(control)
        retry_clock.advance(598)
        raise TransportError("reset")

    def backoff(delay):
        if end_backoff == "cancelled":
            controls[0].cancel("cancelled during backoff")
        else:
            retry_clock.advance(delay + 700)  # Suspended or delayed beyond the deadline.

    monkeypatch.setattr(transport, "_request", exchange)
    optimizer = StrategyOptimizer(prov, tmp_path / "optimizer", completion=transport,
                                  transport_retries=2, sleep=backoff)
    with pytest.raises(PolicyOptimizationError, match="deadline|cancelled") as caught:
        optimizer(request(tmp_path, {"task/a.txt": "e"}))
    assert len(controls) == 1
    assert caught.value.usage["provider_attempts"] == 1
    assert caught.value.usage["unknown_usage_attempts"] == 1


@pytest.mark.parametrize("error", [
    RequestCancelled("cancelled"), RequestDeadlineExceeded("deadline"), TimeoutError("timed out"),
    InvalidProviderResponse("invalid JSON"), ValueError("bad argument"),
    TypeError("client bug"), KeyError("missing field"), AssertionError("client invariant"),
])
def test_optimizer_does_not_retry_terminal_or_programming_errors(tmp_path, error):
    optimizer, delays, attempts = _retrying(tmp_path, error)
    with pytest.raises(PolicyOptimizationError) as caught:
        optimizer(request(tmp_path, {"task/a.txt": "e"}))
    assert caught.value.__cause__ is error
    assert len(attempts) == 1 and delays == []
    assert caught.value.usage["unknown_usage_attempts"] == 1


def test_late_injected_completion_is_rejected_but_its_usage_is_retained(tmp_path, retry_clock):
    calls = []

    def model(payload):
        calls.append(1)
        if len(calls) == 1:
            retry_clock.advance(598)
            raise OSError("reset")
        retry_clock.advance(2)
        return submit()

    prov = ProviderConfig(model="fake", base_url="https://provider.invalid/v1",
                          request_timeout_seconds=600)
    optimizer = StrategyOptimizer(prov, tmp_path / "optimizer", completion=model,
                                  transport_retries=2, sleep=retry_clock.advance)
    with pytest.raises(PolicyOptimizationError, match="deadline") as caught:
        optimizer(request(tmp_path, {"task/a.txt": "e"}))
    assert retry_clock.now == 601 and len(calls) == 2
    assert caught.value.usage["total_tokens"] == 12
    assert caught.value.usage["unknown_usage_attempts"] == 1
    directory = tmp_path / "optimizer/batch_0001"
    assert json.loads((directory / "response_000.json").read_text())["usage"]["prompt_tokens"] == 10
    assert not (directory / "proposal.json").exists()


def test_optimizer_deadline_is_per_logical_call_not_entire_session(tmp_path, monkeypatch, retry_clock):
    prov = ProviderConfig(model="fake", base_url="https://provider.invalid/v1",
                          request_timeout_seconds=600)
    transport = OpenAIChatTransport(prov)
    controls = []

    async def exchange(payload, control, deadline, watch):
        controls.append(control)
        retry_clock.advance(599)
        control.check()
        return (reply(call("read_training_file", {"path": "task/a.txt"}))
                if len(controls) == 1 else submit())

    monkeypatch.setattr(transport, "_request", exchange)
    result = StrategyOptimizer(prov, tmp_path / "optimizer", completion=transport)(
        request(tmp_path, {"task/a.txt": "e"}))
    assert retry_clock.now == 1198
    assert controls[0] is not controls[1]
    assert result.usage["provider_attempts"] == 2


def test_reasoning_window_keeps_latest_reasoning_and_stubs_older_in_place(tmp_path):
    from agent_company.evolution.outer_budget import REASONING_STUB
    seen = []

    def model(payload):
        seen.append([m.get("reasoning_content") for m in payload["messages"] if m["role"] == "assistant"])
        if len(seen) == 4:
            return submit()
        status, body = reply(call("read_training_file", {"path": "task/a.txt"}, f"r{len(seen)}"))
        body["choices"][0]["message"]["reasoning_content"] = f"thinking-{len(seen)}"
        return status, body

    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=model,
                               working_memory=WorkingMemory(reasoning_window=1))(
        request(tmp_path, {"task/a.txt": "e"}))
    assert seen[3] == [REASONING_STUB, REASONING_STUB, "thinking-3"]  # Key kept, text omitted.
    assert result.reading_audit["reasoning_elided"] == 2
    # Reasoning is never persisted (plan v4 12, B1), not even in the raw response copy.
    response = json.loads((tmp_path / "optimizer/batch_0001/response_000.json").read_text())
    assert "reasoning_content" not in response["choices"][0]["message"]


def test_default_working_memory_keeps_every_reasoning_text(tmp_path):
    seen = []

    def model(payload):
        seen.append([m.get("reasoning_content") for m in payload["messages"] if m["role"] == "assistant"])
        if len(seen) == 3:
            return submit()
        status, body = reply(call("read_training_file", {"path": "task/a.txt"}, f"r{len(seen)}"))
        body["choices"][0]["message"]["reasoning_content"] = f"thinking-{len(seen)}"
        return status, body

    StrategyOptimizer(provider(), tmp_path / "optimizer", completion=model,
                      working_memory=WorkingMemory())(request(tmp_path, {"task/a.txt": "e"}))
    assert seen[2] == ["thinking-1", "thinking-2"]
    with pytest.raises(ValueError):
        WorkingMemory(reasoning_window=0)


def test_search_excerpts_count_as_evidence_shown(tmp_path):
    script = Script(reply(call("search_training_files", {"query": "SPEC SCOUT"})), submit())
    result = StrategyOptimizer(provider(), tmp_path / "optimizer", completion=script,
                               working_memory=WorkingMemory())(
        request(tmp_path, {"task/collab.json": COLLABORATION, "task/other.txt": "x"}))
    row = result.reading_audit["coverage"]["task"]
    assert row["files_opened"] == 0 and row["files_touched"] == 1
    assert row["search_excerpt_chars"] == row["chars_shown"] > 0
