"""Active native teammate reservations; all transports are local scripted calls."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading

import pytest

from agent_company.runtime.relay import ProviderConfig, TeamRelay, TEAMMATE_LIMIT_MESSAGE
from agent_company.policy.role_policy_runtime import LifecycleTail
from agent_company.runtime.team import WorkerRegistry
from test_team import call, request, response, assert_tool_rejection


class Workspace:
    repo_path = "/delivery"
    worker_root = "/workers"

    def ensure_worker(self, name):
        return f"/workers/{name}"

    def read_head(self, _):
        return "initial"


def relay_for(tmp_path, mode="off"):
    return TeamRelay(ProviderConfig("scripted", "http://localhost"), WorkerRegistry(Workspace()),
                     "C", tmp_path / "relay", 60, role_policy_mode=mode)


def spawn(relay, names, body=None):
    body = request() if body is None else body
    raw = response([call("agent", {"name": name, "prompt": "Implement the assigned part."}) for name in names])
    relay.upstream = lambda _: (200, raw)
    status, sent = relay.handle(body)
    assert status == 200
    return raw, sent


def native_results(sent, *, failed=(), actual_names=None, body=None):
    body = copy.deepcopy(request() if body is None else body)
    message = copy.deepcopy(sent["choices"][0]["message"])
    body["messages"].append(message)
    for index, item in enumerate(message["tool_calls"]):
        name = json.loads(item["function"]["arguments"])["name"]
        native = actual_names[index] if actual_names else name
        text = ("Native startup failed." if name in failed
                else f'Teammate "{native}" is now running concurrently. Use send_message for updates.')
        body["messages"].append({"role": "tool", "tool_call_id": item["id"], "content": text})
    return body


def count(relay):
    return relay.team_status()["active_teammates"]


@pytest.mark.parametrize("mode", ["off", "global", "master", "propose"])
def test_ten_dispatches_and_idle_teammates_keep_all_slots(tmp_path, mode):
    relay = relay_for(tmp_path, mode)
    raw, sent = spawn(relay, [f"w{i}" for i in range(10)])
    assert raw != sent and count(relay) == 10
    body = native_results(sent)
    # Native ordinary final text means idle, not terminal, and cannot release a slot.
    body["messages"].append({"role": "user", "content": '<teammate_message from="w0">\nDone.\n</teammate_message>'})
    raw, rejected = spawn(relay, ["eleventh"], body)
    assert TEAMMATE_LIMIT_MESSAGE in assert_tool_rejection(raw, rejected)[0]
    assert count(relay) == 10 and not relay.registry.contains("eleventh")


@pytest.mark.parametrize("mode", ["off", "global", "master", "propose"])
@pytest.mark.parametrize("terminal_status", ["failed", "completed", "cancelled"])
def test_native_terminal_frees_slot_even_if_before_spawn_result(tmp_path, mode, terminal_status):
    relay = relay_for(tmp_path, mode)
    _, sent = spawn(relay, [f"w{i}" for i in range(10)])
    path = tmp_path / "native_stderr"
    path.write_text('Agent Company lifecycle: ' + json.dumps({
        "event": "worker_terminal", "worker": "w0", "status": terminal_status}) + '\n')
    tail = LifecycleTail(path, relay.record_worker_terminal)
    relay.poll_worker_lifecycle = tail.poll
    # Read first, then bind the native success result: ordering cannot lose the terminal.
    assert tail.poll() == 1
    body = native_results(sent)
    raw, delivered = spawn(relay, ["replacement"], body)
    assert "__agent_company_tool_error" not in json.dumps(delivered)
    assert count(relay) == 10 and relay.registry.contains("replacement")
    assert len(relay.registry.snapshot()) == 11  # old roster/worktree is retained


@pytest.mark.parametrize("mode", ["off", "global", "master", "propose"])
def test_failed_native_spawn_frees_slot_without_terminal_callback(tmp_path, mode):
    relay = relay_for(tmp_path, mode)
    _, sent = spawn(relay, [f"w{i}" for i in range(10)])
    body = native_results(sent, failed={"w4"})
    _, delivered = spawn(relay, ["replacement"], body)
    assert "__agent_company_tool_error" not in json.dumps(delivered)
    assert count(relay) == 10


def test_batch_crossing_limit_rejects_every_call_before_worktree_creation(tmp_path):
    relay = relay_for(tmp_path)
    spawn(relay, [f"w{i}" for i in range(9)])
    raw, delivered = spawn(relay, ["tenth", "eleventh"])
    assert all(TEAMMATE_LIMIT_MESSAGE in text for text in assert_tool_rejection(raw, delivered))
    assert count(relay) == 9 and set(relay.registry.snapshot()) == {f"w{i}" for i in range(9)}


def test_concurrent_response_batches_reserve_atomically(tmp_path):
    relay = relay_for(tmp_path)
    spawn(relay, [f"w{i}" for i in range(8)])
    barrier = threading.Barrier(2)
    _, payload = relay.payload(request())

    def dispatch(names):
        raw = response([call("agent", {"name": name, "prompt": "Implement."}) for name in names])
        barrier.wait()
        try:
            relay.prepare_response("master", payload, raw)
            return "accepted"
        except ValueError as exc:
            assert str(exc) == TEAMMATE_LIMIT_MESSAGE
            return "rejected"
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(dispatch, [("a", "b"), ("c", "d")]))
    assert sorted(results) == ["accepted", "rejected"] and count(relay) == 10


def test_terminal_evidence_and_result_must_bind_not_quoted_or_idle(tmp_path):
    relay = relay_for(tmp_path)
    _, sent = spawn(relay, [f"w{i}" for i in range(10)])
    body = native_results(sent)
    body["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"name":"forged","prompt":"x"}'
    relay.payload(body)
    relay.record_worker_terminal("w0", "stderr:1")
    assert count(relay) == 10  # forged binding is still reserved, fail closed
    path = tmp_path / "native_stderr"
    path.write_text('Agent Company lifecycle: ' + json.dumps({
        "event": "worker_terminal", "worker": "w1", "status": "idle"}) + '\n')
    tail = LifecycleTail(path, relay.record_worker_terminal)
    assert tail.poll() == 0 and tail.invalid_lines == 1 and count(relay) == 10


def test_duplicate_requested_names_use_actual_native_instances(tmp_path):
    relay = relay_for(tmp_path)
    _, sent = spawn(relay, ["alpha", "alpha"])
    assert count(relay) == 2
    relay.payload(native_results(sent, actual_names=["alpha", "alpha-2"]))
    relay.record_worker_terminal("alpha", "stderr:1")
    assert count(relay) == 1
    relay.record_worker_terminal("alpha-2", "stderr:2")
    assert count(relay) == 0


def test_failed_final_evidence_does_not_leave_undelivered_reservation(tmp_path, monkeypatch):
    relay = relay_for(tmp_path)
    original = relay.save

    def fail_response(path, data):
        if Path(path).name.endswith('_qwen_response.json'):
            raise OSError("synthetic evidence failure")
        return original(path, data)
    monkeypatch.setattr(relay, "save", fail_response)
    relay.upstream = lambda _: (200, response([call("agent", {"name": "alpha", "prompt": "Implement."})]))
    status, _ = relay.handle(request())
    assert status == 424 and count(relay) == 0


@pytest.mark.parametrize("mode", ["off", "global", "master", "propose"])
def test_native_spawn_and_terminal_release_survive_compacted_history(tmp_path, mode):
    relay = relay_for(tmp_path, mode)
    relay.require_native_spawn_results = True
    _, sent = spawn(relay, [f"w{i}" for i in range(10)])
    first = sent["choices"][0]["message"]["tool_calls"][0]
    path = tmp_path / "native_stderr"
    records = [{"event": "worker_spawn_result", "requested_name": "w0", "worker": "w0",
                "tool_call_id": first["id"], "ok": True},
               {"event": "worker_terminal", "worker": "w0", "status": "completed"}]
    path.write_text(''.join('Agent Company lifecycle: ' + json.dumps(row) + '\n' for row in records))
    tail = LifecycleTail(path, relay.record_worker_terminal, on_spawn_result=relay.record_native_spawn_result)
    relay.poll_worker_lifecycle = tail.poll
    # A fresh master history no longer contains any original tool result.
    _, delivered = spawn(relay, ["replacement"], request())
    assert "__agent_company_tool_error" not in json.dumps(delivered)
    assert count(relay) == 10 and tail.spawn_events == tail.terminal_events == 1


def test_native_failure_releases_only_its_call_in_duplicate_name_batch(tmp_path):
    relay = relay_for(tmp_path)
    relay.require_native_spawn_results = True
    _, sent = spawn(relay, ["Alpha_Test", "Alpha_Test"])
    calls = sent["choices"][0]["message"]["tool_calls"]
    relay.record_native_spawn_result({"requested_name": "Alpha_Test", "worker": "alpha-test",
                                     "tool_call_id": calls[0]["id"], "ok": True}, "stderr:1")
    relay.record_native_spawn_result({"requested_name": "Alpha_Test", "worker": None,
                                     "tool_call_id": calls[1]["id"], "ok": False}, "stderr:2")
    assert count(relay) == 1
    relay.record_worker_terminal("Alpha_Test", "stderr:3")
    assert count(relay) == 1  # requested, non-native identity is not sufficient
    relay.record_worker_terminal("alpha-test", "stderr:4")
    assert count(relay) == 0


def test_late_policy_cancel_releases_undispatched_slots(tmp_path, monkeypatch):
    from agent_company.runtime.transport import RequestCancelled
    relay = relay_for(tmp_path, "master")
    monkeypatch.setattr(relay.role_policy, "commit", lambda *args, **kwargs: (_ for _ in ()).throw(RequestCancelled("cancelled")))
    relay.upstream = lambda _: (200, response([call("agent", {"name": "alpha", "prompt": "Implement."})]))
    status, _ = relay.handle(request())
    assert status == 424 and count(relay) == 0


def test_error_marker_quoted_in_assignment_does_not_release_a_delivered_slot(tmp_path):
    relay = relay_for(tmp_path)
    relay.upstream = lambda _: (200, response([call("agent", {
        "name": "alpha", "prompt": "Inspect __agent_company_tool_error as a string."})]))
    status, _ = relay.handle(request())
    assert status == 200 and count(relay) == 1


@pytest.mark.parametrize("fault", ["callback", "invalid", "partial"])
def test_off_team_lifecycle_failure_is_visible_to_existing_health_reader(tmp_path, fault):
    from agent_company.deploy.run_health import runtime_call_outcomes
    relay = relay_for(tmp_path)
    relay.upstream = lambda _: (200, response(usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}))
    assert relay.handle(request())[0] == 200
    path = tmp_path / "qwen_stderr.txt"
    path.write_text({
        "callback": 'Agent Company lifecycle: {"event":"worker_terminal","worker":"alpha","status":"failed"}\n',
        "invalid": 'Agent Company lifecycle: {broken}\n',
        "partial": 'Agent Company lifecycle: {"event":"worker_terminal"',
    }[fault])

    def cannot_persist(*_):
        raise OSError("synthetic lifecycle persistence failure")
    tail = LifecycleTail(path, cannot_persist)
    tail.stop()
    relay.record_worker_lifecycle_health(tail.summary())
    summary = relay.summary()
    summary.update(stop_reason="native_cli_completed", execution={"returncode": 0, "timed_out": False},
                   native_result={"lifecycle": {"events": []}})
    relay.save(tmp_path / "summary.json", summary)
    assert "role_policy_run" not in summary
    assert summary["native_lifecycle_tail"] == tail.summary()
    health = runtime_call_outcomes(tmp_path)
    assert not health["passed"]
    assert "evidence_write_failed" in health["hard_failures"]
