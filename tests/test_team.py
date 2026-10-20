import copy
import json
from pathlib import Path
import subprocess
import uuid

import pytest

from agent_company.runtime.relay import (COORD_TOOLS, MASTER_TOOLS, READ_ONLY_TOOLS, SINGLE_EXCLUDED_TOOLS,
                                       SINGLE_TOOLS, TOOL_ERROR_FIELD, WORKER_TOOLS,
                                       ProviderConfig, TeamRelay, TokenRates, parse_tool_arguments,
                                       usage_and_cost)
from agent_company.runtime.team import LocalWorkspace, WorkerRegistry
from agent_company.policy.source_boundary import SOURCE_BOUNDARY


def workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "original.txt").write_text("initial\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-m", "initial"], check=True, capture_output=True)
    return LocalWorkspace(str(repo), str(tmp_path / "workers"), str(tmp_path / "runtime"))


def request(role="master", tools=("agent", "send_message", "run_shell_command")):
    system = "You are the master" if role == "master" else f'You are agent "{role}" in team "test".'
    return {"model": "untrusted-override", "messages": [{"role": "system", "content": system}],
            "tools": [{"type": "function", "function": {"name": name}} for name in tools]}


def call(name, args):
    return {"id": "test-" + uuid.uuid4().hex, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def response(calls=(), usage=None):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "",
            "tool_calls": list(calls)}, "finish_reason": "tool_calls"}], "usage": usage}


def test_worker_creation_head_is_recorded_before_work_and_survives_later_commits(tmp_path):
    ws = workspace(tmp_path)
    registry = WorkerRegistry(ws)
    task_head = ws.read_head(ws.repo_path)
    commit = ["git", "-c", "user.name=Test", "-c", "user.email=test@localhost",
              "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm"]
    subprocess.run([*commit, "earlier integration"], cwd=ws.repo_path, check=True)
    created_head = ws.read_head(ws.repo_path)
    assert created_head != task_head
    worker = registry.register("late")
    assert registry.metadata()["late"] == {"working_dir": worker, "creation_head": created_head}
    subprocess.run([*commit, "worker's own commit"], cwd=worker, check=True)
    assert ws.read_head(worker) != created_head
    assert registry.register("late") == worker
    assert registry.metadata()["late"]["creation_head"] == created_head
    counts = [subprocess.check_output(["git", "-C", worker, "rev-list", "--count", f"{base}..HEAD"],
                                     text=True).strip() for base in (task_head, created_head)]
    assert counts == ["2", "1"]


def test_worker_creation_head_failure_stays_unknown_instead_of_sampling_later(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    registry = WorkerRegistry(ws)
    read_head = ws.read_head
    monkeypatch.setattr(ws, "read_head", lambda _: (_ for _ in ()).throw(OSError("unavailable")))
    worker = registry.register("alpha")
    assert registry.metadata()["alpha"]["creation_head"] is None
    monkeypatch.setattr(ws, "read_head", read_head)
    assert registry.register("alpha") == worker
    assert registry.metadata()["alpha"]["creation_head"] is None


def assert_tool_rejection(original, delivered):
    """The native adapter receives every original call plus explicit failure feedback."""
    restored = copy.deepcopy(delivered)
    feedback = []
    for before_choice, after_choice in zip(original["choices"], restored["choices"], strict=True):
        before_calls = before_choice["message"].get("tool_calls") or []
        after_calls = after_choice["message"].get("tool_calls") or []
        for before, after in zip(before_calls, after_calls, strict=True):
            arguments = json.loads(after["function"]["arguments"])
            error = arguments.pop(TOOL_ERROR_FIELD)
            assert "Tool response rejected before execution:" in error
            assert "No tools from this response were executed" in error
            assert "resubmit" in error
            feedback.append(error)
            try:
                expected, repeated = parse_tool_arguments(before["function"]["arguments"])
            except ValueError:
                expected, repeated = None, {}
            if repeated:
                expected = {}  # Repeated keys are never collapsed into one value.
            elif not isinstance(expected, dict):
                expected = {"__agent_company_original_arguments": before["function"]["arguments"]}
            assert arguments == expected
            after["function"]["arguments"] = before["function"]["arguments"]
    assert feedback and restored == original
    return feedback


def test_spawn_declaration_matches_existing_execution_boundary_without_mutating_input(tmp_path):
    body = request()
    body["tools"][0]["function"].update(
        description="Use Explore or fork; set read_only=true for investigation.",
        parameters={"type": "object", "properties": {
            key: {"type": "string"} for key in
            ("name", "prompt", "description", "model", "read_only", "isolation",
             "working_dir", "fork_turns", "fork_tools", "fork_profile", "plan_mode_required")},
            "required": ["description", "prompt"]})
    original = copy.deepcopy(body)
    relay = TeamRelay(ProviderConfig("same-model", "http://localhost"),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 60)
    role, payload = relay.payload(body)
    assert role == "master" and body == original
    function = next(t["function"] for t in payload["tools"] if t["function"]["name"] == "agent")
    schema = function["parameters"]
    assert set(schema["properties"]) == {"name", "description", "prompt", "subagent_type"}
    assert schema["properties"]["subagent_type"]["enum"] == ["general-purpose"]
    assert set(schema["required"]) == {"name", "description", "prompt"}
    assert schema["additionalProperties"] is False
    assert "read_only=true" not in function["description"] and "Explore" not in function["description"]
    assert "investigation" in function["description"]
    assert payload["tools"][1] == body["tools"][1]  # Other native tool declarations are unchanged.


@pytest.mark.parametrize("native_limit", ["10", "Infinity"])
def test_team_create_description_discloses_shared_resources_not_a_count_cap(tmp_path, native_limit):
    from agent_company.runtime.qwen import master_system, single_system
    from agent_company.runtime.relay import TEAM_SIZE_NOTE

    body = request(tools=("team_create", "send_message"))
    body["tools"][0]["function"].update(
        description=f"Create team. Spawn teammates (max {native_limit} teammates per team). Keep team records.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}})
    original = copy.deepcopy(body)
    relay = TeamRelay(ProviderConfig("same-model", "http://localhost"),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 60)
    _, payload = relay.payload(body)
    function = payload["tools"][0]["function"]
    assert body == original
    assert function["parameters"] == body["tools"][0]["function"]["parameters"]
    assert f"max {native_limit}" not in function["description"]
    assert "Keep team records." in function["description"] and TEAM_SIZE_NOTE in function["description"]
    assert payload["tools"][1] == body["tools"][1]
    # Always disclosed, even with check-ins / optional runtime disclosure off.
    assert TEAM_SIZE_NOTE in master_system("A", "C")
    assert TEAM_SIZE_NOTE not in single_system()


def test_dynamic_worktrees_and_policy_delivered_to_five_workers(tmp_path):
    ws = workspace(tmp_path)
    registry = WorkerRegistry(ws)
    policy = "Notify a dependency owner when the interface changes."
    config = ProviderConfig("same-model", "http://localhost")
    workers = ["parser", "backend", "docs", "test_runner", "integrator5"]
    raw = response([call("agent", {"name": worker, "prompt": "Implement your part"}) for worker in workers],
                   {"prompt_tokens": 10, "completion_tokens": 5})
    relay = TeamRelay(config, registry, policy, tmp_path / "logs", 60, lambda _: (200, raw))
    status, delivered = relay.handle(request())
    assert status == 200
    assert len(registry.snapshot()) == 5
    for worker in workers:
        assert (Path(registry.snapshot()[worker]) / "original.txt").read_text() == "initial\n"
        role, payload = relay.payload(request(worker))
        assert role == worker and payload["model"] == "same-model"
        assert json.dumps(payload["messages"]).count(policy) == 1
        assert {t["function"]["name"] for t in payload["tools"]} == {"send_message", "run_shell_command"}
    for tool in delivered["choices"][0]["message"]["tool_calls"]:
        assert "system instructions" in json.loads(tool["function"]["arguments"])["prompt"]
    assert policy not in json.dumps(raw)
    assert policy not in (tmp_path / "logs/attempt_00001_master_provider_response.json").read_text()
    assert "system instructions" in (tmp_path / "logs/attempt_00001_master_qwen_response.json").read_text()
    assert registry.register(workers[0]) == registry.snapshot()[workers[0]]


@pytest.mark.parametrize("changes", [{"name": "../escape"}, {"working_dir": "/tmp/elsewhere"},
                                      {"model": "another"}, {"read_only": True},
                                      {"subagent_type": "Explore"}, {"subagent_type": "fork"},
                                      {"isolation": "worktree"}, {"run_in_background": False},
                                      {"name": "leader"}, {"prompt": ""}])
def test_reject_bad_spawns_before_creating_worktrees(tmp_path, changes):
    ws = workspace(tmp_path)
    registry = WorkerRegistry(ws)
    args = {"name": "builder", "prompt": "Implement"} | changes
    raw = response([call("agent", args)], {"prompt_tokens": 7, "completion_tokens": 3})
    original = copy.deepcopy(raw)
    relay = TeamRelay(ProviderConfig("model", "http://localhost"), registry, "policy", tmp_path / "logs", 60,
                      lambda _: (200, raw))
    status, delivered = relay.handle(request())
    assert status == 200 and registry.snapshot() == {}
    assert raw == original
    assert_tool_rejection(original, delivered)
    # Rejected tool calls were still paid provider calls, so retain their usage.
    assert relay.summary()["usage"]["total_tokens"] == 10
    assert relay.summary()["errors"][0]["error_kind"] == "tool_response_rejected"
    assert relay.summary()["errors"][0]["relay_http_status"] == 200
    assert not Path(registry.workspace.worker_root).exists()


def test_unknown_identity_and_master_shell_rejected(tmp_path):
    registry = WorkerRegistry(workspace(tmp_path))
    raw = response([call("run_shell_command", {"command": "false"})])
    relay = TeamRelay(ProviderConfig("model", "http://localhost"), registry, "policy", tmp_path / "logs", 60,
                      lambda _: (200, raw))
    with pytest.raises(ValueError, match="Unregistered"):
        relay.payload(request("unregistered"))
    status, delivered = relay.handle(request())
    assert status == 200
    assert "Disallowed master tool: run_shell_command" in assert_tool_rejection(raw, delivered)[0]
    assert registry.snapshot() == {}


def test_master_has_read_only_inspection_but_no_edit_or_shell(tmp_path):
    """2026-09-26: the master may read/list/search; edit, write and shell stay worker-only."""
    registry = WorkerRegistry(workspace(tmp_path))
    everything = tuple(sorted(COORD_TOOLS | SINGLE_TOOLS))
    relay = TeamRelay(ProviderConfig("model", "http://localhost"), registry, "policy", tmp_path / "logs", 60,
                      lambda _: (200, response()))
    role, payload = relay.payload(request("master", everything))
    assert role == "master"
    assert {tool["function"]["name"] for tool in payload["tools"]} == MASTER_TOOLS
    assert READ_ONLY_TOOLS <= MASTER_TOOLS and not ({"edit", "write_file", "run_shell_command"} & MASTER_TOOLS)
    # A worker request is unchanged by the master's new tools.
    registry.register("worker")
    _, worker_payload = relay.payload(request("worker", everything))
    assert {tool["function"]["name"] for tool in worker_payload["tools"]} == WORKER_TOOLS

    # Delivered read-only calls pass validation unchanged.
    read = response([call("read_file", {"absolute_path": str(registry.workspace.repo_path) + "/original.txt"}),
                     call("grep_search", {"pattern": "initial", "path": str(registry.workspace.repo_path)})])
    relay.upstream = lambda _: (200, read)
    status, delivered = relay.handle(request("master", everything))
    assert status == 200
    for tool_call in delivered["choices"][0]["message"]["tool_calls"]:
        assert TOOL_ERROR_FIELD not in json.loads(tool_call["function"]["arguments"])

    # Master edit/write/shell are rejected before execution, as before.
    for name, args in (("edit", {"file_path": "x", "old_string": "a", "new_string": "b"}),
                       ("write_file", {"file_path": "x", "content": "y"}),
                       ("run_shell_command", {"command": "false"})):
        raw = response([call(name, args)])
        relay.upstream = lambda _, raw=raw: (200, raw)
        status, delivered = relay.handle(request("master", everything))
        assert status == 200
        assert f"Disallowed master tool: {name}" in assert_tool_rejection(raw, delivered)[0]
    assert set(registry.snapshot()) == {"worker"}


@pytest.mark.parametrize("invalid_first", [False, True])
def test_invalid_spawn_rejects_every_call_without_registering_workers(tmp_path, monkeypatch, invalid_first):
    registry = WorkerRegistry(workspace(tmp_path))
    raw = response([call("agent", {"name": "valid", "prompt": "Implement"}),
                    call("agent", {"name": "invalid", "prompt": "Inspect", "read_only": True})],
                   {"prompt_tokens": 7, "completion_tokens": 3})
    if invalid_first:
        raw["choices"][0]["message"]["tool_calls"].reverse()
    original = copy.deepcopy(raw)
    registered = []
    monkeypatch.setattr(registry, "register", lambda name: registered.append(name))
    relay = TeamRelay(ProviderConfig("model", "http://localhost"), registry, "policy", tmp_path / "logs", 60,
                      lambda _: (200, raw))
    status, delivered = relay.handle(request())
    assert status == 200 and raw == original
    feedback = assert_tool_rejection(original, delivered)
    assert len(feedback) == 2 and len(set(feedback)) == 1
    assert registry.snapshot() == {} and registered == []
    assert not Path(registry.workspace.worker_root).exists()
    assert relay.summary()["usage"]["total_tokens"] == 10
    assert relay.summary()["errors"][0]["error_kind"] == "tool_response_rejected"
    assert json.loads((tmp_path / "logs/attempt_00001_master_provider_response.json").read_text()) == original
    assert json.loads((tmp_path / "logs/attempt_00001_master_qwen_response.json").read_text()) == delivered


@pytest.mark.parametrize("arguments", ['{"name":', "not json", "[]", "null", '"text"', "42", "true"])
def test_invalid_json_or_nonobject_arguments_preserved_for_native_tool_feedback(tmp_path, arguments):
    registry = WorkerRegistry(workspace(tmp_path))
    malformed = call("agent", {})
    malformed["function"]["arguments"] = arguments
    raw = response([call("agent", {"name": "valid", "prompt": "Implement"}), malformed],
                   {"prompt_tokens": 7, "completion_tokens": 3})
    original = copy.deepcopy(raw)
    relay = TeamRelay(ProviderConfig("model", "http://localhost"), registry, "policy", tmp_path / "logs", 60,
                      lambda _: (200, raw))
    status, delivered = relay.handle(request())
    assert status == 200 and raw == original
    assert len(assert_tool_rejection(original, delivered)) == 2
    assert registry.snapshot() == {} and not Path(registry.workspace.worker_root).exists()
    assert relay.summary()["usage"]["total_tokens"] == 10
    assert relay.summary()["errors"][0]["error_kind"] == "tool_response_rejected"


def test_metering_distinguishes_missing_usage_and_no_price(tmp_path):
    rates = TokenRates("2", "0.2", "5")
    usage, cost = usage_and_cost({"prompt_tokens": 100, "completion_tokens": 20,
                                  "prompt_tokens_details": {"cached_tokens": 40}}, rates)
    assert usage["total_tokens"] == 120 and str(cost) == "0.000228"
    relay = TeamRelay(ProviderConfig("model", "http://localhost", api_key="fake-secret", rates=rates,
                                    max_attempts=1),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "logs", 60,
                      lambda _: (503, {"error": "fake-secret unavailable"}))
    assert relay.handle(request())[0] == 424
    summary = relay.summary()
    assert not summary["usage_complete"] and not summary["cost_complete"]
    assert summary["provider_attempts"] == 1
    assert summary["errors"][0]["provider_http_status"] == 503
    assert "fake-secret" not in "".join(p.read_text() for p in (tmp_path / "logs").iterdir())


def test_local_timeout_stops_subprocess_group(tmp_path):
    ws = workspace(tmp_path)
    result = ws.run_agent(["/bin/sleep", "30"], env={}, timeout_seconds=0.1,
                          stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")
    assert result["timed_out"] and result["returncode"] != 0


def test_single_relay_has_only_coding_tools_no_policy_and_records_single_usage(tmp_path):
    registry = WorkerRegistry(workspace(tmp_path))
    body = request(tools=tuple(sorted(SINGLE_TOOLS | SINGLE_EXCLUDED_TOOLS)))
    original = copy.deepcopy(body)
    raw = response([call("read_file", {"file_path": "/repo/original.txt"})],
                   {"prompt_tokens": 100, "completion_tokens": 20})
    relay = TeamRelay(ProviderConfig("model", "http://localhost", rates=TokenRates("2", "0.2", "5")),
                      registry, "DO_NOT_INJECT_THIS_POLICY", tmp_path / "logs", 60,
                      lambda _: (200, raw), execution_mode="single")
    role, payload = relay.payload(body)
    assert role == "single" and body == original
    assert payload["messages"] == [{"role": "system", "content":
                                     body["messages"][0]["content"] + "\n\n" + SOURCE_BOUNDARY}]
    assert {tool["function"]["name"] for tool in payload["tools"]} == SINGLE_TOOLS
    assert not ({tool["function"]["name"] for tool in payload["tools"]} & COORD_TOOLS)
    assert relay.handle(body)[0] == 200
    summary = relay.summary()
    assert summary["execution_mode"] == "single" and summary["policy_applied"] is False
    assert summary["calls_by_role"] == {"single": 1} and summary["workers"] == {}
    assert summary["usage"]["total_tokens"] == 120
    assert summary["cost_complete"] and summary["known_estimated_cost"] == "0.0003"
    for suffix in ("qwen_request", "provider_request", "provider_response", "qwen_response"):
        assert (tmp_path / f"logs/attempt_00001_single_{suffix}.json").is_file()
    assert "DO_NOT_INJECT_THIS_POLICY" not in "".join(p.read_text() for p in (tmp_path / "logs").iterdir())
    started = json.loads((tmp_path / "logs/events.jsonl").read_text().splitlines()[0])
    assert started["worker_system_policy_injected"] is False
    assert not Path(registry.workspace.worker_root).exists()
    with pytest.raises(ValueError, match="disabled in single mode"):
        relay.payload(request("unexpected-worker"))


@pytest.mark.parametrize("tool", sorted(SINGLE_EXCLUDED_TOOLS | {"web_fetch"}))
def test_single_rejects_non_coding_tool_response_and_retains_cost(tmp_path, tool):
    registry = WorkerRegistry(workspace(tmp_path))
    raw = response([call(tool, {"name": "worker", "prompt": "Do work"})],
                   {"prompt_tokens": 7, "completion_tokens": 3})
    original = copy.deepcopy(raw)
    relay = TeamRelay(ProviderConfig("model", "http://localhost", rates=TokenRates("2", "0.2", "5")),
                      registry, "", tmp_path / "logs", 60, lambda _: (200, raw), execution_mode="single")
    status, delivered = relay.handle(request(tools=tuple(SINGLE_TOOLS | {tool})))
    assert status == 200 and raw == original
    assert f"Disallowed single tool: {tool}" in assert_tool_rejection(original, delivered)[0]
    assert relay.summary()["usage"]["total_tokens"] == 10
    assert relay.summary()["cost_complete"] and relay.summary()["known_estimated_cost"] == "0.000029"
    assert relay.summary()["errors"][0]["error_kind"] == "tool_response_rejected"
    assert registry.snapshot() == {} and not Path(registry.workspace.worker_root).exists()


def test_relay_rejects_unknown_execution_mode_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="execution_mode"):
        TeamRelay(ProviderConfig("model", "http://localhost"), WorkerRegistry(workspace(tmp_path)),
                  "", tmp_path / "logs", 60, execution_mode="unknown")
    assert not (tmp_path / "logs").exists()
