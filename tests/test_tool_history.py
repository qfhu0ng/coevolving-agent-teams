"""Duplicate-key tool arguments and model-visible history (2026-09-29).

A master spawn once carried repeated JSON keys; last-value-wins parsing started
worker ``dec`` with the prompt "placeholder". Native Qwen also kept the relay's
rewritten spawn arguments in the master's history (five keys, working_dir last),
which differs from what the model wrote and from the advertised schema.
"""
import copy
import json
import uuid

from agent_company.runtime.relay import (TOOL_ERROR_FIELD, ProviderConfig, TeamRelay,
                                       model_visible_arguments, parse_tool_arguments)
from agent_company.runtime.team import WorkerRegistry
from test_team import call, request, response, workspace

SPAWN_NOTE = "The run's communication policy is supplied in your system instructions"


def raw_call(name, arguments):
    return {"id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def relay_for(tmp_path, upstream, mode="team"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = WorkerRegistry(workspace(tmp_path))
    relay = TeamRelay(ProviderConfig("model", "http://localhost"), registry, "policy", tmp_path / "logs", 60,
                      upstream, execution_mode=mode)
    return relay, registry


def native_history(role, delivered, results=None):
    """Mimic native Qwen: re-serialized delivered arguments plus one tool result per call."""
    body = request(role, ("agent", "send_message", "run_shell_command", "write_file"))
    message = copy.deepcopy(delivered["choices"][0]["message"])
    for item in message.get("tool_calls") or []:
        item["function"]["arguments"] = json.dumps(json.loads(item["function"]["arguments"]),
                                                   separators=(",", ":"))
    body["messages"].append(message)
    for item in message.get("tool_calls") or []:
        body["messages"].append({"role": "tool", "tool_call_id": item["id"],
                                 "content": (results or {}).get(item["id"], "ok")})
    return body


def history_calls(payload):
    return {item["id"]: item for message in payload["messages"] if message.get("role") == "assistant"
            for item in message.get("tool_calls") or []}


def test_parse_reports_repeated_keys_at_any_depth_without_collapsing():
    assert parse_tool_arguments('{"a": 1, "b": {"c": 2}}') == ({"a": 1, "b": {"c": 2}}, {})
    _, repeated = parse_tool_arguments('{"a": 1, "a": 2, "a": 3, "b": {"c": 1, "c": 2}}')
    assert repeated == {"a": 3, "c": 2}
    assert model_visible_arguments('{"a": 1}') == '{"a": 1}'
    for text in ('{"a": 1, "a": 2}', '{"a": ', "[1, 2]"):
        assert model_visible_arguments(text) == "{}"


def test_duplicate_spawn_prompt_is_rejected_before_any_worker_exists(tmp_path):
    # The observed failure: the real prompt first, then repeated keys ending in "placeholder".
    bad = raw_call("agent", '{"name": "dec", "description": "Decoder", "prompt": "Implement the decoder", '
                            '"subagent_type": "general-purpose", "prompt": "placeholder"}')
    sibling = call("send_message", {"to": "scout", "message": "status?"})
    raw = response([bad, sibling], {"prompt_tokens": 7, "completion_tokens": 3})
    original = copy.deepcopy(raw)
    relay, registry = relay_for(tmp_path, lambda _: (200, raw))
    status, delivered = relay.handle(request())
    assert status == 200 and raw == original
    assert registry.snapshot() == {}
    first, second = delivered["choices"][0]["message"]["tool_calls"]
    assert [first["id"], second["id"]] == [bad["id"], sibling["id"]]
    # The ambiguous call carries only the adapter error; no value is chosen for it.
    first_args = json.loads(first["function"]["arguments"])
    assert set(first_args) == {TOOL_ERROR_FIELD}
    assert "placeholder" not in first["function"]["arguments"]
    error = first_args[TOOL_ERROR_FIELD]
    assert '"prompt" x2' in error and "never executed" in error
    # A valid sibling is rejected with its own arguments intact (all-or-none).
    second_args = json.loads(second["function"]["arguments"])
    assert second_args.pop(TOOL_ERROR_FIELD) == error
    assert second_args == {"to": "scout", "message": "status?"}
    record = relay.summary()["errors"][0]
    assert record["error_kind"] == "tool_response_rejected" and record["relay_http_status"] == 200
    assert record["tool_rejection"] == {"reason": "duplicate_keys", "tool": "agent",
                                        "repeated_keys": {"prompt": 2}}


def test_nested_duplicate_and_single_mode_write_are_rejected(tmp_path):
    bad = raw_call("write_file", '{"file_path": "/tmp/x", "content": "a", "content": "b"}')
    raw = response([bad], {"prompt_tokens": 1, "completion_tokens": 1})
    relay, _ = relay_for(tmp_path, lambda _: (200, raw), mode="single")
    _, delivered = relay.handle(request("master", ("write_file",)))
    args = json.loads(delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert set(args) == {TOOL_ERROR_FIELD} and '"content" x2' in args[TOOL_ERROR_FIELD]

    nested = raw_call("run_shell_command", '{"command": "true", "env": {"A": "1", "A": "2"}}')
    raw2 = response([nested])
    relay2, _ = relay_for(tmp_path / "nested", lambda _: (200, raw2), mode="single")
    _, delivered2 = relay2.handle(request("master", ("run_shell_command",)))
    args2 = json.loads(delivered2["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert set(args2) == {TOOL_ERROR_FIELD} and '"A" x2' in args2[TOOL_ERROR_FIELD]


def test_spawn_history_shows_model_text_while_execution_keeps_assigned_worktree(tmp_path):
    written = '{"name": "builder", "description": "Build", "prompt": "Implement the parser"}'
    spawn = raw_call("agent", written)
    replies = [response([spawn]), response()]
    sent = []

    def upstream(payload):
        sent.append(copy.deepcopy(payload))
        return 200, replies[len(sent) - 1]

    relay, registry = relay_for(tmp_path, upstream)
    _, delivered = relay.handle(request())
    delivered_args = json.loads(delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    # Execution version unchanged: assigned worktree, fixed type and appended note.
    assert delivered_args["working_dir"] == registry.snapshot()["builder"]
    assert delivered_args["subagent_type"] == "general-purpose" and SPAWN_NOTE in delivered_args["prompt"]

    body = native_history("master", delivered)
    before = copy.deepcopy(body)
    relay.handle(body)
    assert body == before  # The native request object is never modified.
    shown = history_calls(sent[1])[spawn["id"]]["function"]["arguments"]
    assert shown == written
    assert relay.summary()["model_history"] == {"rewritten_calls": 1, "restored": 1}
    saved = json.loads((tmp_path / "logs/attempt_00002_master_qwen_request.json").read_text())
    assert "working_dir" in json.dumps(saved)  # Evidence keeps what native actually sent.


def test_rejected_history_drops_marker_and_omits_only_ambiguous_calls(tmp_path):
    bad = raw_call("agent", '{"name": "a", "prompt": "x", "prompt": "y"}')
    sibling_text = '{"to": "scout", "message": "status?"}'
    sibling = raw_call("send_message", sibling_text)
    relay, _ = relay_for(tmp_path, lambda _: (200, response([bad, sibling])))
    _, delivered = relay.handle(request())
    role, payload = relay.payload(native_history("master", delivered))
    calls = history_calls(payload)
    assert calls[bad["id"]]["function"]["arguments"] == "{}"
    assert calls[sibling["id"]]["function"]["arguments"] == sibling_text
    assert TOOL_ERROR_FIELD not in json.dumps(
        [m for m in payload["messages"] if m.get("role") == "assistant"])


def test_restoration_is_scoped_to_role_name_and_delivered_version(tmp_path):
    spawn = raw_call("agent", '{"name": "builder", "prompt": "Implement"}')
    relay, registry = relay_for(tmp_path, lambda _: (200, response([spawn])))
    _, delivered = relay.handle(request())
    body = native_history("master", delivered)

    # Another role's history with the same call ID and arguments is left alone.
    worker_body = native_history("builder", delivered)
    worker_body["tools"] = [{"type": "function", "function": {"name": "run_shell_command"}}]
    counts = {}
    _, worker_payload = relay.payload(worker_body, history=counts)
    assert history_calls(worker_payload)[spawn["id"]] == history_calls(worker_body)[spawn["id"]]
    assert counts == {"unrecorded": 1}

    # A different tool name or changed arguments under the same ID is not replaced.
    renamed = copy.deepcopy(body)
    history_calls(renamed)[spawn["id"]]["function"]["name"] = "send_message"
    edited = copy.deepcopy(body)
    item = history_calls(edited)[spawn["id"]]
    args = json.loads(item["function"]["arguments"])
    args["prompt"] = "something else"
    item["function"]["arguments"] = json.dumps(args)
    for variant in (renamed, edited):
        counts = {}
        _, payload = relay.payload(variant, history=counts)
        assert history_calls(payload)[spawn["id"]] == history_calls(variant)[spawn["id"]]
        assert counts == {"mismatched": 1}

    # Unrelated calls are never touched.
    other = call("send_message", {"to": "builder", "message": "hi"})
    body["messages"].append({"role": "assistant", "content": "", "tool_calls": [other]})
    _, payload = relay.payload(body)
    assert history_calls(payload)[other["id"]] == other


def test_reused_call_id_restores_only_the_matching_delivery(tmp_path):
    shared_id = "call_0"  # Some providers restart IDs in every response.
    first = {"id": shared_id, "type": "function",
             "function": {"name": "agent", "arguments": '{"name": "one", "prompt": "First"}'}}
    second = {"id": shared_id, "type": "function",
              "function": {"name": "agent", "arguments": '{"name": "two", "prompt": "Second"}'}}
    replies = iter([response([first]), response([second])])
    relay, _ = relay_for(tmp_path, lambda _: (200, next(replies)))
    _, one = relay.handle(request())
    _, two = relay.handle(request())
    body = native_history("master", one)
    body["messages"] += native_history("master", two)["messages"][1:]
    _, payload = relay.payload(body)
    shown = [item["function"]["arguments"] for message in payload["messages"]
             if message.get("role") == "assistant" for item in message["tool_calls"]]
    assert shown == [first["function"]["arguments"], second["function"]["arguments"]]
