"""Actual Qwen turns recover from relay validation via native tool-result history."""

from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import threading
import time
import uuid

import pytest

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, TokenRates, role_of
from test_team import call, response, workspace


def native_qwen():
    default = Path(__file__).parents[1] / "external/qwen-study/package/cli.js"
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(default)))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Set QWEN_CODE_CLI and QWEN_NODE to execute the native no-provider integration test")
    return cli.resolve(), node


def reply(payload, tools=(), content="", usage=None):
    return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()),
                 "model": payload["model"], "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": content, "tool_calls": list(tools)},
                 "finish_reason": "tool_calls" if tools else "stop"}],
                 "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3}}


def tool_results(payload, calls):
    """Check the real next model request, not just the relay's saved response."""
    by_id = {message["tool_call_id"]: message for message in payload["messages"]
             if message.get("role") == "tool" and message.get("tool_call_id")}
    missing = {item["id"] for item in calls} - by_id.keys()
    assert not missing, f"Native history did not carry failed call IDs: {missing}"
    assistant_calls = {item["id"]: item for message in payload["messages"]
                       if message.get("role") == "assistant"
                       for item in message.get("tool_calls", [])}
    for original in calls:
        carried = assistant_calls[original["id"]]
        assert carried["function"]["name"] == original["function"]["name"]
        original_args = json.loads(original["function"]["arguments"])
        carried_args = json.loads(carried["function"]["arguments"])
        assert all(carried_args.get(name) == value for name, value in original_args.items())
    return [str(by_id[item["id"]].get("content", "")) for item in calls]


def test_native_read_only_rejection_reaches_model_and_preserves_usage(tmp_path):
    cli, node = native_qwen()
    calls = []
    raw_usage = {"prompt_tokens": 7, "completion_tokens": 3}
    rejected_call = call("agent", {"name": "scout", "description": "Inspect repository",
                                 "prompt": "Inspect the repository without making changes.",
                                 "subagent_type": "general-purpose", "read_only": True})
    raw = response([rejected_call], raw_usage)

    def upstream(payload):
        calls.append(copy.deepcopy(payload))
        if len(calls) == 1:
            # Do not silently convert requested read-only into a writable agent.
            return 200, raw
        assert len(calls) == 2, "The model should see one failed tool result, not an HTTP retry"
        failures = tool_results(payload, [rejected_call])
        assert "read_only" in failures[0] or "writable" in failures[0]
        return reply(payload, content="The requested read-only teammate mode is unsupported; no teammate was created.")

    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=upstream).run(
        workspace(tmp_path), "Inspect the repository with a teammate.",
        "Delegate repository inspection to one suitable teammate.",
        "Return concrete findings to the coordinator.", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=5,
                       rates=TokenRates("1", "0.1", "2")),
        timeout_seconds=30,
    )

    assert len(calls) == 2
    assert result["provider_attempts"] == result["completed_attempts"] == 2
    assert result["inflight_attempts"] == 0 and result["workers"] == {}
    assert result["scripted_provider"] and not result["execution"]["timed_out"]
    assert result["stop_reason"] == "native_cli_completed"
    assert not result["native_result"]["explicit_error"]
    assert result["elapsed_seconds"] < 30
    assert result["usage_complete"] and result["cost_complete"]
    assert result["usage"]["total_tokens"] == 20
    assert result["known_estimated_cost"] == "0.000026"

    assert len(result["errors"]) == 1
    error = result["errors"][0]
    assert error["provider_http_status"] == error["relay_http_status"] == 200
    assert error["raw_usage"] == raw_usage
    assert error["error_kind"] == "tool_response_rejected"

    relay_dir = output / "relay"
    saved_raw = json.loads((relay_dir / "attempt_00001_master_provider_response.json").read_text())
    assert saved_raw == raw
    delivered = json.loads((relay_dir / "attempt_00001_master_qwen_response.json").read_text())
    delivered_call = delivered["choices"][0]["message"]["tool_calls"][0]
    assert delivered_call["id"] == rejected_call["id"]
    delivered_args = json.loads(delivered_call["function"]["arguments"])
    assert delivered_args["read_only"] is True
    assert isinstance(delivered_args["__agent_company_tool_error"], str)


class RejectedBatchScenario:
    """The installed CLI executes all tools; only the model responses are scripted."""

    def __init__(self, ws):
        self.ws = ws
        self.steps = defaultdict(int)
        self.lock = threading.Lock()
        self.integrated = threading.Event()
        self.shutdown_seen = threading.Event()
        self.failed_results_seen = False
        self.corrected_spawn_seen = False
        self.worker_failure_seen = False
        self.worker_cwd_seen = False
        self.team_name = "reject-recover-" + uuid.uuid4().hex[:10]
        self.sentinel = "BATCH_MUST_NOT_EXECUTE_" + uuid.uuid4().hex
        self.bad_calls = [
            call("agent", {"name": "integrator", "description": "Integrate delivery",
                           "prompt": "Edit and test the delivery repository.",
                           "subagent_type": "general-purpose", "working_dir": ws.repo_path}),
            call("task_create", {"subject": self.sentinel, "description": "Must not enter the task board."}),
            call("send_message", {"to": "leader", "message": self.sentinel}),
        ]
        self.bad_raw = None
        self.next_payload = None
        self.worker_bad_call = call("agent", {"name": "nested", "description": "Forbidden delegation",
                                               "prompt": "This nested teammate must not be created."})

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            assert sum(self.steps.values()) <= 20, "Scripted recovery did not converge"
        messages = payload["messages"]
        text = json.dumps(messages)
        users = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
        if role != "master" and "shutdown" in (users[-1] if users else "").lower():
            self.shutdown_seen.set()
            return reply(payload, [call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        if role == "master":
            if step == 0:
                return reply(payload, [call("team_create", {"team_name": self.team_name})])
            if step == 1:
                assert self.team_name in text
                result = reply(payload, self.bad_calls)
                self.bad_raw = copy.deepcopy(result[1])
                return result
            if step == 2:
                failures = tool_results(payload, self.bad_calls)
                for failure in failures:
                    assert "working_dir" in failure
                    assert "omit working_dir" in failure.lower()
                    assert "git -C" in failure and self.ws.repo_path in failure
                # The batch is all-or-none: no valid sibling may leak effects.
                root = Path(self.ws.worker_root)
                assert not root.exists() or list(root.iterdir()) == []
                for pattern in ("**/tasks/**/*.json", "**/inboxes/**/*.json"):
                    for path in Path(self.ws.runtime_root).glob(pattern):
                        assert self.sentinel not in path.read_text(), path
                self.failed_results_seen = True
                self.next_payload = copy.deepcopy(payload)
                self.corrected_spawn_seen = True
                return reply(payload, [call("agent", {
                    "name": "integrator", "description": "Integrate in assigned worktree",
                    "prompt": "Use your assigned worktree, then an absolute path and git -C to modify the delivery repository.",
                    "subagent_type": "general-purpose"})])
            if step == 3:
                assert self.integrated.wait(25), "Worker did not finish delivery through its actual native shell"
                return reply(payload, [call("request_shutdown", {"to": "integrator"})])
            if step >= 4:
                assert self.shutdown_seen.wait(25), "Worker did not receive native shutdown"
                return reply(payload, content="Delivery integrated and verified after correcting the rejected tool batch.")

        assert role == "integrator" and self.corrected_spawn_seen
        if step == 0:
            return reply(payload, [self.worker_bad_call])
        if step == 1:
            failures = tool_results(payload, [self.worker_bad_call])
            assert "agent" in failures[0]
            assert not (Path(self.ws.worker_root) / "nested").exists()
            self.worker_failure_seen = True
            repo = shlex.quote(self.ws.repo_path)
            own = shlex.quote(str(Path(self.ws.worker_root) / "integrator"))
            deliver = shlex.quote(str(Path(self.ws.repo_path) / "recovered.txt"))
            command = (
                f"pwd && test \"$PWD\" = {own} && "
                f"printf '%s\\n' recovered-through-native-tool > {deliver} && "
                f"git -C {repo} add recovered.txt && "
                f"git -C {repo} -c user.name=RuntimeTest -c user.email=test@localhost "
                "-c commit.gpgsign=false commit -m recovered-delivery && "
                f"git -C {repo} show HEAD:recovered.txt && printf '%s\\n' NATIVE_RECOVERY_DELIVERY_OK")
            return reply(payload, [call("run_shell_command", {
                "command": command, "description": "Verify assigned cwd and update delivery by explicit path"})])
        assert "NATIVE_RECOVERY_DELIVERY_OK" in text
        assert str(Path(self.ws.worker_root) / "integrator") in text
        self.worker_cwd_seen = True
        self.integrated.set()
        return reply(payload, content="RECOVERY_DELIVERY_COMPLETE")


def test_native_rejected_batch_has_no_side_effects_and_model_corrects_delivery_spawn(tmp_path):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    scenario = RejectedBatchScenario(ws)
    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario).run(
        ws, "Create recovered.txt with a teammate and integrate it into the delivery repository.",
        "Delegate implementation and final integration to one writable teammate.",
        "Report the delivery path and concrete verification to the coordinator.", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35,
                       rates=TokenRates("1", "0.1", "2")), timeout_seconds=90)
    assert result["execution"]["returncode"] == 0, (result, (output / "qwen_stderr.txt").read_text())
    assert result["stop_reason"] == "native_cli_completed" and not result["execution"]["timed_out"], result
    assert scenario.failed_results_seen and scenario.corrected_spawn_seen and scenario.worker_failure_seen
    assert scenario.worker_cwd_seen
    assert scenario.integrated.is_set() and scenario.shutdown_seen.is_set()
    assert (Path(ws.repo_path) / "recovered.txt").read_text() == "recovered-through-native-tool\n"
    assert result["workers"] == {"integrator": str(Path(ws.worker_root) / "integrator")}
    assert result["usage_complete"] and result["cost_complete"] and result["scripted_provider"]
    assert result["completed_attempts"] == result["provider_attempts"] == sum(scenario.steps.values())
    assert result["usage"]["total_tokens"] == result["provider_attempts"] * 10
    assert len(result["errors"]) == 2
    error = result["errors"][0]
    assert error["error_kind"] == "tool_response_rejected" and error["relay_http_status"] == 200
    assert result["errors"][1]["role"] == "integrator"
    assert result["errors"][1]["error_kind"] == "tool_response_rejected"
    raw = json.loads((output / "relay/attempt_00002_master_provider_response.json").read_text())
    assert raw == scenario.bad_raw
    delivered = json.loads((output / "relay/attempt_00002_master_qwen_response.json").read_text())
    annotated = delivered["choices"][0]["message"]["tool_calls"]
    assert [item["id"] for item in annotated] == [item["id"] for item in scenario.bad_calls]
    for original, item in zip(scenario.bad_calls, annotated):
        assert item["function"]["name"] == original["function"]["name"]
        args = json.loads(item["function"]["arguments"])
        marker = args.pop("__agent_company_tool_error")
        assert "working_dir" in marker and args == json.loads(original["function"]["arguments"])
    # Real native stream traces retain the failed tool-result correlation as well.
    stdout = (output / "qwen_stdout.jsonl").read_text()
    for item in scenario.bad_calls:
        assert item["id"] in stdout
    worker_requests = sorted((output / "relay").glob("attempt_*_integrator_provider_request.json"))
    assert any("NATIVE_RECOVERY_DELIVERY_OK" in path.read_text() for path in worker_requests)
    assert any(scenario.worker_bad_call["id"] in path.read_text() for path in worker_requests)
