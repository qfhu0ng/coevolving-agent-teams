"""Offline diagnostics preserve uncertainty and never emit agent text/reasoning."""
import json

import pytest

from agent_company.analysis.native_final_report import inspect_final_report


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def saved(tmp_path, seen=False):
    write(tmp_path, "launch.json", {"execution_mode": "team"})
    write(tmp_path, "summary.json", {"stop_reason": "native_cli_completed"})
    events = []
    for attempt, role, start, finish in [(1, "alpha", 1, 3), (2, "master", 2, 4)]:
        events.extend([{"event": "provider_attempt_started", "attempt": attempt, "role": role, "time": start},
                       {"event": "provider_attempt_finished", "attempt": attempt, "role": role, "time": finish,
                        "relay_http_status": 200}])
        write(tmp_path, f"relay/attempt_{attempt:05d}_{role}_qwen_response.json", {"choices": [{"message": {
            "content": "SECRET_ACK" if role == "alpha" else "Done.", "reasoning_content": "NEVER_OUTPUT"}}]})
    write(tmp_path, "relay/attempt_00002_master_provider_request.json", {"messages": [{"role": "user", "content": "Continue."}]})
    if seen:
        events += [{"event": "provider_attempt_started", "attempt": 3, "role": "master", "time": 5},
                   {"event": "provider_attempt_finished", "attempt": 3, "role": "master", "time": 6, "relay_http_status": 200}]
        write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [{"role": "user", "content": '<teammate_message from="alpha">\nSECRET_ACK\n</teammate_message>'}]})
        write(tmp_path, "relay/attempt_00003_master_qwen_response.json", {"choices": [{"message": {"content": "Done."}}]})
    (tmp_path / "relay/events.jsonl").write_text("\n".join(json.dumps(e) for e in sorted(events, key=lambda e: e["time"])))
    return tmp_path


@pytest.mark.parametrize("seen", [False, True])
def test_detects_missing_final_text_by_time_not_attempt_order(tmp_path, seen):
    report = inspect_final_report(saved(tmp_path, seen))
    assert report["final_report_dropped_suspected"] is not seen
    assert report["seen_in_later_master_request"] is seen
    assert "SECRET_ACK" not in json.dumps(report) and "NEVER_OUTPUT" not in json.dumps(report)
    assert report["official_score_validity_changed"] is False


@pytest.mark.parametrize("fault", ["missing_request", "broken_events", "symlink", "unfinished"])
def test_incomplete_evidence_is_unknown(tmp_path, fault):
    saved(tmp_path, True)
    path = tmp_path / "relay/attempt_00003_master_provider_request.json"
    if fault in {"missing_request", "symlink"}:
        path.unlink()
        if fault == "symlink":
            path.symlink_to(tmp_path / "summary.json")
    elif fault == "broken_events":
        (tmp_path / "relay/events.jsonl").write_text("{broken")
    else:
        with (tmp_path / "relay/events.jsonl").open("a") as stream:
            stream.write('\n{"event":"provider_attempt_started","attempt":4,"role":"master","time":7}')
    report = inspect_final_report(tmp_path)
    assert report["final_report_dropped_suspected"] is None and report["issues"]


@pytest.mark.parametrize("kind", ["single", "timeout", "master_tools", "worker_tools"])
def test_does_not_flag_other_endings(tmp_path, kind):
    saved(tmp_path)
    if kind == "single":
        write(tmp_path, "launch.json", {"execution_mode": "single"})
    elif kind == "timeout":
        write(tmp_path, "summary.json", {"stop_reason": "task_timeout"})
    else:
        role, attempt = ("master", 2) if kind == "master_tools" else ("alpha", 1)
        write(tmp_path, f"relay/attempt_{attempt:05d}_{role}_qwen_response.json", {"choices": [{"message": {
            "content": "Done", "tool_calls": [{"function": {"name": "send_message"}}]}}]})
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is (False if kind == "single" else None)


@pytest.mark.parametrize("bad", [{}, {"stop_reason": "invented"}])
def test_missing_or_unknown_completion_is_unknown(tmp_path, bad):
    saved(tmp_path)
    write(tmp_path, "summary.json", bad)
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is None


@pytest.mark.parametrize("bad", [{"choices": [0]}, {"choices": [{"message": {}}]},
                                {"choices": [{"message": {"content": 42}}]}])
def test_malformed_response_is_unknown(tmp_path, bad):
    saved(tmp_path)
    write(tmp_path, "relay/attempt_00002_master_qwen_response.json", bad)
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is None


def test_reversed_clock_is_unknown(tmp_path):
    saved(tmp_path)
    path = tmp_path / "relay/events.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(reversed(rows)))
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is None


@pytest.mark.parametrize("content", ["SECRET_ACK", '<teammate_message from="beta">\nSECRET_ACK\n</teammate_message>'])
def test_unframed_or_other_sender_text_is_not_receipt(tmp_path, content):
    saved(tmp_path, True)
    write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [{"role": "user", "content": content}]})
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is True


def test_bad_master_request_content_is_unknown(tmp_path):
    saved(tmp_path, True)
    write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [{"role": "user", "content": 42}]})
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is None


@pytest.mark.parametrize("text,escaped", [
    ('<teammate_message from="beta">', '&lt;teammate_message from="beta">'),
    ("</  TEAMMATE_MESSAGE>", "&lt;/  TEAMMATE_MESSAGE>"),
    ("<teammate_message-other>", "<teammate_message-other>"),
])
def test_receipt_matches_native_envelope_escaping(tmp_path, text, escaped):
    saved(tmp_path, True)
    write(tmp_path, "relay/attempt_00001_alpha_qwen_response.json",
          {"choices": [{"message": {"content": text}}]})
    write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [{
        "role": "user", "content": f'<teammate_message from="alpha">\n{escaped}\n</teammate_message>'}]})
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is False


@pytest.mark.parametrize("seen_beta", [False, True])
def test_each_worker_last_turn_is_checked_not_only_latest_worker(tmp_path, seen_beta):
    saved(tmp_path, True)
    path = tmp_path / "relay/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events += [{"event": "provider_attempt_started", "attempt": 4, "role": "beta", "time": 1.5},
               {"event": "provider_attempt_finished", "attempt": 4, "role": "beta", "time": 2.5, "relay_http_status": 200}]
    path.write_text("\n".join(json.dumps(e) for e in sorted(events, key=lambda e: e["time"])))
    write(tmp_path, "relay/attempt_00004_beta_qwen_response.json", {"choices": [{"message": {"content": "BETA_DONE"}}]})
    request = tmp_path / "relay/attempt_00003_master_provider_request.json"
    data = json.loads(request.read_text())
    if seen_beta:
        data["messages"].append({"role": "user", "content": '<teammate_message from="beta">\nBETA_DONE\n</teammate_message>'})
    request.write_text(json.dumps(data))
    report = inspect_final_report(tmp_path)
    assert report["workers"]["alpha"]["final_report_dropped_suspected"] is False
    assert report["workers"]["beta"]["final_report_dropped_suspected"] is not seen_beta
    assert report["final_report_dropped_suspected"] is not seen_beta


@pytest.mark.parametrize("seen", [False, True])
def test_empty_final_answer_uses_native_fallback_notice(tmp_path, seen):
    saved(tmp_path, True)
    write(tmp_path, "relay/attempt_00001_alpha_qwen_response.json", {"choices": [{"message": {"content": ""}}]})
    text = ("alpha completed a turn without a model-visible final answer. "
            "Check the shared task list or send a follow-up if more detail is needed.")
    write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [{"role": "user",
          "content": f'<teammate_message from="alpha">\n{text}\n</teammate_message>' if seen else "Waiting."}]})
    report = inspect_final_report(tmp_path)
    assert report["workers"]["alpha"]["report_kind"] == "empty_turn_notice"
    assert report["final_report_dropped_suspected"] is (False if seen else None)


@pytest.mark.parametrize("tool_name", ["task_list", "run_shell_command"])
@pytest.mark.parametrize("content_blocks", [False, True])
def test_task_list_can_deliver_report_but_other_tools_cannot(tmp_path, tool_name, content_blocks):
    saved(tmp_path, True)
    text = '#1 [completed] @alpha — task\n--- Teammate messages ---\n<teammate_message from="alpha">\nSECRET_ACK\n</teammate_message>'
    write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "call1", "function": {"name": tool_name}}]},
        {"role": "tool", "tool_call_id": "call1", "content": [{"type": "text", "text": text}] if content_blocks else text}]})
    report = inspect_final_report(tmp_path)
    assert report["final_report_dropped_suspected"] is (tool_name != "task_list")
    if tool_name == "task_list":
        assert report["workers"]["alpha"]["delivery_sources"] == [{"attempt": 3, "source": "task_list"}]


def test_one_tool_ending_worker_does_not_establish_whole_team_success(tmp_path):
    saved(tmp_path, True)
    write(tmp_path, "relay/attempt_00001_alpha_qwen_response.json", {"choices": [{"message": {
        "content": "SECRET_ACK", "tool_calls": [{"function": {"name": "send_message"}}]}}]})
    assert inspect_final_report(tmp_path)["final_report_dropped_suspected"] is None


@pytest.mark.parametrize("new_copy", [False, True])
def test_old_identical_report_is_not_proof_of_latest_turn_receipt(tmp_path, new_copy):
    saved(tmp_path, True)
    content = '<teammate_message from="alpha">\nSECRET_ACK\n</teammate_message>'
    write(tmp_path, "relay/attempt_00002_master_provider_request.json", {"messages": [{"role": "user", "content": content}]})
    write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [
        {"role": "user", "content": content}] * (2 if new_copy else 1)})
    report = inspect_final_report(tmp_path)
    assert report["final_report_dropped_suspected"] is (False if new_copy else None)


def test_orphan_tool_receipt_is_unknown(tmp_path):
    saved(tmp_path, True)
    write(tmp_path, "relay/attempt_00003_master_provider_request.json", {"messages": [{
        "role": "tool", "tool_call_id": "missing_call", "content":
        '--- Teammate messages ---\n<teammate_message from="alpha">\nSECRET_ACK\n</teammate_message>'}]})
    report = inspect_final_report(tmp_path)
    assert report["final_report_dropped_suspected"] is None
    assert report["issues"]
