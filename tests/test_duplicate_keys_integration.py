"""Actual Qwen: a duplicate-key call is rejected, rewritten in the same session, run once.

Only model responses are scripted; the installed CLI executes tools and keeps history.
"""
from collections import defaultdict
import json
from pathlib import Path
import shlex
import threading
import uuid

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, TokenRates, role_of
from test_rejection_integration import native_qwen, reply
from test_team import workspace


def raw_call(name, arguments):
    return {"id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def carried(payload, call_id):
    """The call as the next model request shows it, and its native tool result."""
    calls = {item["id"]: item for message in payload["messages"] if message.get("role") == "assistant"
             for item in message.get("tool_calls") or []}
    results = {message["tool_call_id"]: str(message.get("content", "")) for message in payload["messages"]
               if message.get("role") == "tool" and message.get("tool_call_id")}
    assert call_id in calls and call_id in results, "Native history lost the rejected call"
    return calls[call_id]["function"]["arguments"], results[call_id]


class DuplicateSpawnScenario:
    def __init__(self, ws):
        self.ws = ws
        self.steps = defaultdict(int)
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.shutdown_seen = threading.Event()
        self.real = "DECODER_ASSIGNMENT_" + uuid.uuid4().hex
        self.team_name = "dup-keys-" + uuid.uuid4().hex[:10]
        self.bad = raw_call("agent", json.dumps({"name": "dec", "description": "Decoder",
                                                 "prompt": self.real})[:-1]
                            + ', "subagent_type": "general-purpose", "prompt": "placeholder"}')
        self.written = json.dumps({"name": "dec", "description": "Decoder", "prompt": self.real})
        self.good = raw_call("agent", self.written)
        self.checks = []

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            assert sum(self.steps.values()) <= 20, "Scripted recovery did not converge"
        messages = payload["messages"]
        users = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
        if role != "master" and "shutdown" in (users[-1] if users else "").lower():
            self.shutdown_seen.set()
            return reply(payload, [raw_call("send_message", json.dumps(
                {"to": "leader", "message": "shutdown_approved"}))])
        if role == "master":
            if step == 0:
                return reply(payload, [raw_call("team_create", json.dumps({"team_name": self.team_name}))])
            if step == 1:
                return reply(payload, [self.bad])
            if step == 2:
                shown, result = carried(payload, self.bad["id"])
                assert shown == "{}", "An ambiguous call must not be replayed or collapsed"
                assert '"prompt" x2' in result and "never executed" in result
                assert not Path(self.ws.worker_root).exists() or not any(Path(self.ws.worker_root).iterdir())
                self.checks.append("rejected_in_history")
                return reply(payload, [self.good])
            if step == 3:
                shown, _ = carried(payload, self.good["id"])
                assert shown == self.written, "Master history must show the spawn as written"
                self.checks.append("spawn_as_written")
                assert self.done.wait(25), "Worker did not run its tool"
                return reply(payload, [raw_call("request_shutdown", json.dumps({"to": "dec"}))])
            assert self.shutdown_seen.wait(25), "Worker did not receive native shutdown"
            return reply(payload, content="Decoder delivered after correcting the duplicate-key spawn.")
        assert role == "dec"
        text = json.dumps(messages)
        if step == 0:
            assert self.real in text and "placeholder" not in text
            self.checks.append("worker_prompt_intact")
            target = shlex.quote(str(Path(self.ws.worker_root) / "dec" / "runs.txt"))
            return reply(payload, [raw_call("run_shell_command", json.dumps(
                {"command": f"printf 'run\\n' >> {target} && printf DEC_TOOL_OK",
                 "description": "Record one execution"}))])
        assert "DEC_TOOL_OK" in text
        self.done.set()
        return reply(payload, content="DEC_COMPLETE")


def test_native_team_duplicate_spawn_rejected_then_rewritten_and_run_once(tmp_path):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    scenario = DuplicateSpawnScenario(ws)
    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario).run(
        ws, "Implement the decoder with one teammate.", "Delegate the decoder to one teammate.",
        "Report completion to the coordinator.", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35,
                       rates=TokenRates("1", "0.1", "2")), timeout_seconds=90)
    assert result["execution"]["returncode"] == 0, (result, (output / "qwen_stderr.txt").read_text())
    assert result["stop_reason"] == "native_cli_completed" and not result["execution"]["timed_out"], result
    # Master and worker turns interleave, so only the set of checks is fixed.
    assert sorted(scenario.checks) == ["rejected_in_history", "spawn_as_written", "worker_prompt_intact"]
    assert result["workers"] == {"dec": str(Path(ws.worker_root) / "dec")}
    assert (Path(ws.worker_root) / "dec" / "runs.txt").read_text() == "run\n"
    rejected = [error for error in result["errors"] if error["error_kind"] == "tool_response_rejected"]
    assert len(rejected) == 1 and rejected[0]["tool_rejection"]["reason"] == "duplicate_keys"
    assert result["model_history"]["restored"] >= 2 and not result["model_history"].get("mismatched")
    assert not result["model_history"].get("unrecorded")
    # Native itself still executed the assigned-worktree version.
    delivered = json.loads((output / "relay/attempt_00003_master_qwen_response.json").read_text())
    args = json.loads(delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["working_dir"] == str(Path(ws.worker_root) / "dec")


class DuplicateWriteScenario:
    def __init__(self, ws):
        self.path = Path(ws.repo_path) / "single.txt"
        self.bad = raw_call("write_file", json.dumps({"file_path": str(self.path), "content": "first\n"})[:-1]
                            + ', "content": "second\\n"}')
        self.steps = 0
        self.checks = []

    def __call__(self, payload):
        step = self.steps
        self.steps += 1
        if step == 0:
            return reply(payload, [self.bad])
        if step == 1:
            shown, result = carried(payload, self.bad["id"])
            assert shown == "{}" and '"content" x2' in result
            assert not self.path.exists()
            self.checks.append("rejected_in_history")
            return reply(payload, [raw_call("write_file", json.dumps(
                {"file_path": str(self.path), "content": "chosen\n"}))])
        assert step == 2, "Single-agent fixture did not converge"
        return reply(payload, content="Wrote single.txt once.")


def test_native_single_duplicate_write_rejected_then_written_once(tmp_path):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    scenario = DuplicateWriteScenario(ws)
    output = tmp_path / "single"
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario).run(
        ws, "Create single.txt.", "", "", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35,
                       rates=TokenRates("1", "0.1", "2")), timeout_seconds=60, execution_mode="single")
    assert result["execution"]["returncode"] == 0, (result, (output / "qwen_stderr.txt").read_text())
    assert scenario.steps == 3 and scenario.checks == ["rejected_in_history"]
    assert scenario.path.read_text() == "chosen\n"
    assert [error["tool_rejection"]["reason"] for error in result["errors"]] == ["duplicate_keys"]
