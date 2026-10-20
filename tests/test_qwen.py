"""No network providers: scripted replies drive the actual installed Qwen CLI."""
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import threading
import time
import uuid

import pytest

import agent_company.runtime.qwen as qwen_module
from agent_company.runtime.qwen import QwenTeamRunner, native_result
from agent_company.runtime.relay import (SINGLE_EXCLUDED_TOOLS, SINGLE_TOOLS,
                                       ProviderConfig, TokenRates, role_of)
from test_team import call, workspace


@pytest.mark.parametrize("fields", [{"is_error": True}, {"subtype": "error_max_turns"},
                                    {"result": "[API Error] unavailable"},
                                    {"result": "[API Error: 400 rejected tool call]"},
                                    {"result": "  [api error: 400 rejected tool call]"},
                                    {"result": "API Error: unavailable"}])
def test_explicit_api_failures_even_if_cli_exits_zero(tmp_path, fields):
    path = tmp_path / "stdout"
    path.write_text(json.dumps({"type": "result", "subtype": "success", "is_error": False, **fields}) + "\n")
    assert native_result(path)["explicit_error"]


class FourWorkerScenario:
    names = ("alpha", "beta", "gamma", "delta")

    def __init__(self, ws):
        self.ws = ws
        self.steps = defaultdict(int)
        self.lock = threading.Lock()
        self.peer_sent = threading.Event()
        self.beta_busy = threading.Event()
        self.committed = {name: threading.Event() for name in self.names}
        self.integrated = threading.Event()
        self.nonce = "handoff-" + uuid.uuid4().hex
        self.observations = []
        self.shutdown = False

    def shell(self, command):
        return call("run_shell_command", {"command": command, "description": "Run scripted native-runtime check"})

    def respond(self, payload):
        role = role_of(payload)
        text = json.dumps(payload["messages"])
        if role != "master":
            assert text.count("INTERFACE_CHANGE_POLICY") == 1, "Worker policy must persist exactly once"
        users = [str(m.get("content", "")) for m in payload["messages"] if m.get("role") == "user"]
        if role != "master" and "shutdown" in (users[-1] if users else "").lower():
            return "", [call("send_message", {"to": "leader", "message": "shutdown_approved"})]
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
        if sum(self.steps.values()) > 100:
            raise RuntimeError("Scripted fixture did not converge")
        if role == "master":
            if step == 0:
                return "", [call("team_create", {"team_name": "stage1-test-" + uuid.uuid4().hex[:6]})]
            if step == 1:
                return "", [call("agent", {"name": name, "subagent_type": "general-purpose",
                    "description": f"Implement {name}", "prompt": f"Write {name}.txt in your worktree and commit. "
                    "Retain this session for follow-up. This is a deterministic local runtime check."})
                    for name in self.names]
            if self.integrated.is_set():
                if not self.shutdown:
                    self.shutdown = True
                    return "", [call("request_shutdown", {"to": name}) for name in self.names]
                return "Four workers integrated and tested; native same-session continuation verified.", []
            return "Waiting for native teammate reports.", []
        if step == 0:
            if role == "beta":
                self.beta_busy.set()
                if not self.peer_sent.wait(30):
                    raise RuntimeError("Peer message never sent while beta was working")
            command = (f"printf '%s\\n' {shlex.quote(role)} > {role}.txt && git add {role}.txt && "
                       f"git -c user.name=RuntimeTest -c user.email=test@localhost -c commit.gpgsign=false commit -m {shlex.quote(role)} && "
                       f"printf '%s\\n' {shlex.quote('WRITTEN_' + role)}")
            return "", [self.shell(command)]
        if f"WRITTEN_{role}" not in text:
            raise RuntimeError(f"Real worker shell failed: {role}")
        self.committed[role].set()
        if role == "alpha":
            if step == 1:
                if not self.beta_busy.wait(30):
                    raise RuntimeError("Beta was not active")
                return "", [call("send_message", {"to": "beta", "message": self.nonce + ": integrate after finishing this turn."})]
            self.peer_sent.set()
            return "ALPHA_DONE", []
        if role != "beta":
            return role.upper() + "_DONE", []
        if step == 1:
            self.observations.append({"phase": "before_idle", "nonce_visible": self.nonce in text})
            assert self.nonce not in text, "Mailbox changed: message arrived before the full work turn ended"
            return "BETA_FIRST_TURN_DONE", []
        if step == 2:
            self.observations.append({"phase": "continuation", "nonce_visible": self.nonce in text,
                                      "history_visible": "BETA_FIRST_TURN_DONE" in text and "WRITTEN_beta" in text})
            assert self.nonce in text and "BETA_FIRST_TURN_DONE" in text and "WRITTEN_beta" in text
            for event in self.committed.values():
                if not event.wait(30):
                    raise RuntimeError("A worker did not commit")
            commands = []
            for other in self.names:
                if other != "beta":
                    otherpath = shlex.quote(str(Path(self.ws.worker_root) / other))
                    commands.append(f"git -c user.name=RuntimeTest -c user.email=test@localhost -c commit.gpgsign=false cherry-pick $(git -C {otherpath} rev-parse HEAD)")
            commands += [f"git -C {shlex.quote(self.ws.repo_path)} merge --ff-only HEAD".replace("HEAD", "$(git rev-parse HEAD)"),
                         "test -f alpha.txt -a -f beta.txt -a -f gamma.txt -a -f delta.txt",
                         "printf '%s\\n' INTEGRATION_OK"]
            return "", [self.shell(" && ".join(commands))]
        assert "INTEGRATION_OK" in text, "Integration command did not finish"
        self.integrated.set()
        return "BETA_INTEGRATED", []

    def __call__(self, payload):
        content, tools = self.respond(payload)
        return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()),
                     "model": payload["model"], "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content, "tool_calls": tools},
                     "finish_reason": "tool_calls" if tools else "stop"}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def test_native_four_workers_multiturn_mailbox_and_delivery(tmp_path):
    default = Path(__file__).parents[1] / "external/qwen-study/package/cli.js"
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(default)))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Set QWEN_CODE_CLI and QWEN_NODE to execute the native no-provider integration test")
    ws = workspace(tmp_path)
    scenario = FourWorkerScenario(ws)
    result = QwenTeamRunner(str(cli.resolve()), node=node, upstream=scenario).run(
        ws, "Write four independent files and integrate them into the delivery repository.",
        "Choose suitable teammates; assign integration to one of them.",
        "INTERFACE_CHANGE_POLICY: share actionable interface changes with their dependency owners.",
        tmp_path / "run", ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=40),
        timeout_seconds=100)
    assert result["execution"]["returncode"] == 0, (result, (tmp_path / "run/qwen_stderr.txt").read_text())
    assert not result["errors"]
    assert scenario.integrated.is_set(), (result, (tmp_path / "run/qwen_stdout.jsonl").read_text())
    assert len(result["workers"]) == 4
    assert scenario.observations == [{"phase": "before_idle", "nonce_visible": False},
        {"phase": "continuation", "nonce_visible": True, "history_visible": True}]
    for name in scenario.names:
        assert (Path(ws.repo_path) / f"{name}.txt").read_text().strip() == name
    assert result["usage_complete"] and result["scripted_provider"]
    launch = json.loads((tmp_path / "run/launch.json").read_text())
    assert launch["native_team_scheduler_and_mailbox_modified"] is True
    assert launch["native_team_inactivity_timeout_seconds"] == 1800
    assert launch["teammate_limit"] == 10
    assert "native Team inactivity timeout 1800s" in (tmp_path / "run/qwen_stderr.txt").read_text()
    assert result["execution_mode"] == "team" and result["policy_applied"] is True


class SingleAgentScenario:
    """Only scripted provider replies; the installed CLI performs the actual I/O."""

    def __init__(self, ws):
        self.ws = ws
        self.steps = 0
        self.observations = []

    def __call__(self, payload):
        tools = {tool["function"]["name"] for tool in payload["tools"]}
        assert tools <= SINGLE_TOOLS
        assert {"read_file", "write_file", "run_shell_command"} <= tools
        text = json.dumps(payload["messages"])
        assert "Communication policy for this task:" not in text
        assert "Assignment policy (fixed for this run):" not in text
        assert 'You are agent "' not in text
        tool_results = [m for m in payload["messages"] if m.get("role") == "tool"]
        last_result = str(tool_results[-1].get("content", "")) if tool_results else ""
        repo = Path(self.ws.repo_path)
        step = self.steps
        self.steps += 1
        if step == 0:
            calls = [call("read_file", {"file_path": str(repo / "original.txt")})]
        elif step == 1:
            assert "initial" in last_result, "Native read_file must return the existing file"
            self.observations.append("read_original")
            calls = [call("write_file", {"file_path": str(repo / "single.txt"),
                                         "content": "native-single-delivery\n"})]
        elif step == 2:
            calls = [call("read_file", {"file_path": str(repo / "single.txt")})]
        elif step == 3:
            assert "native-single-delivery" in last_result, "Native write_file/read_file roundtrip failed"
            self.observations.append("write_readback")
            calls = [call("run_shell_command", {
                "command": "pwd && test -f single.txt && printf '%s\\n' NATIVE_SINGLE_SHELL_OK",
                "description": "Verify direct single-agent delivery and shell execution"})]
        elif step == 4:
            assert self.ws.repo_path in last_result, "Native shell must execute in the delivery repository"
            assert "NATIVE_SINGLE_SHELL_OK" in last_result, "Native shell result must return to the model"
            self.observations.append("shell_verified")
            calls = []
        else:
            raise AssertionError("Single-agent fixture did not converge")
        return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()),
                     "model": payload["model"], "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "Single-agent delivery verified." if not calls else "",
                                 "tool_calls": calls},
                     "finish_reason": "tool_calls" if calls else "stop"}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def test_native_single_agent_direct_read_write_shell_without_team(tmp_path):
    default = Path(__file__).parents[1] / "external/qwen-study/package/cli.js"
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(default)))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Set QWEN_CODE_CLI and QWEN_NODE to execute the native no-provider integration test")
    ws = workspace(tmp_path)
    scenario = SingleAgentScenario(ws)
    output = tmp_path / "single"
    result = QwenTeamRunner(str(cli.resolve()), node=node, upstream=scenario).run(
        ws, "Read original.txt, create single.txt, and verify the direct repository delivery.",
        "", "", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=40,
                       rates=TokenRates("2", "0.2", "5")),
        timeout_seconds=100, execution_mode="single")
    assert result["execution"]["returncode"] == 0, (result, (output / "qwen_stderr.txt").read_text())
    assert result["stop_reason"] == "native_cli_completed" and not result["errors"], result
    assert scenario.steps == 5 and scenario.observations == ["read_original", "write_readback", "shell_verified"]
    assert (Path(ws.repo_path) / "single.txt").read_text() == "native-single-delivery\n"
    assert not Path(ws.worker_root).exists() and result["workers"] == {}
    assert result["execution_mode"] == "single" and result["policy_applied"] is False
    assert result["calls_by_role"] == {"single": 5}
    assert result["scripted_provider"] and result["usage_complete"] and result["cost_complete"]
    assert result["usage"]["total_tokens"] == 550 and result["known_estimated_cost"] == "0.00125"
    launch = json.loads((output / "launch.json").read_text())
    assert launch["execution_mode"] == "single" and launch["policy_applied"] is False
    assert launch["assignment_policy"] is None and launch["communication_policy"] is None
    assert launch["environment"]["QWEN_CODE_ENABLE_AGENT_TEAM"] == "0"
    assert launch["native_team_inactivity_timeout_seconds"] is None
    assert launch["teammate_limit"] is None
    assert "--max-subagent-depth" not in launch["command"]
    imports = [launch["command"][i + 1] for i, arg in enumerate(launch["command"]) if arg == "--import"]
    assert imports == [str(Path(result["runtime_dir"]) / name)
                       for name in ("qwen_native_compat.mjs", "qwen_tool_errors.mjs")]
    assert launch["tool_error_feedback"] == "native_tool_result"
    assert SINGLE_EXCLUDED_TOOLS <= set(launch["command"])
    settings = json.loads((Path(result["runtime_dir"]) / "profile/settings.json").read_text())
    assert settings["experimental"]["agentTeam"] is False
    assert settings["model"]["generationConfig"]["maxRetries"] == 0
    assert launch["native_request_timeout_seconds"] > launch["upstream_request_deadline_seconds"]
    assert launch["environment"]["QWEN_CODE_UNATTENDED_RETRY"] == "false"
    for suffix in ("qwen_request", "provider_request", "provider_response", "qwen_response"):
        files = sorted((output / "relay").glob(f"attempt_*_single_{suffix}.json"))
        assert len(files) == 5
        for file in files:
            record = json.loads(file.read_text())
            if suffix.endswith("request"):
                assert not ({t["function"]["name"] for t in record["tools"]} & SINGLE_EXCLUDED_TOOLS)
    stdout = [json.loads(line) for line in (output / "qwen_stdout.jsonl").read_text().splitlines() if line]
    native_calls = []
    for event in stdout:
        if event.get("type") == "system" and event.get("subtype") == "init":
            assert not (set(event["tools"]) & SINGLE_EXCLUDED_TOOLS)
        content = event.get("message", {}).get("content", [])
        if isinstance(content, list):
            native_calls.extend(part["name"] for part in content if part.get("type") == "tool_use")
    assert native_calls == ["read_file", "write_file", "read_file", "run_shell_command"]


@pytest.mark.parametrize("policies", [("", ""), ("IGNORED_ASSIGNMENT_SENTINEL", "IGNORED_COMMUNICATION_SENTINEL")])
def test_single_launch_never_injects_policy_or_worker_root(tmp_path, monkeypatch, policies):
    ws = workspace(tmp_path)

    def fake_run(command, *, env, timeout_seconds, stdout_path, stderr_path):
        Path(stdout_path).write_text(json.dumps({"type": "result", "subtype": "success", "result": "done"}) + "\n")
        Path(stderr_path).write_text("")
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(ws, "run_agent", fake_run)
    result = QwenTeamRunner("unused-cli", upstream=lambda _: pytest.fail("No model call expected")).run(
        ws, "Implement directly", *policies, tmp_path / "single",
        ProviderConfig("model", "http://localhost"), 10, execution_mode="single")
    launch = json.loads((tmp_path / "single/launch.json").read_text())
    serialized = json.dumps(launch)
    assert "IGNORED_ASSIGNMENT_SENTINEL" not in serialized and "IGNORED_COMMUNICATION_SENTINEL" not in serialized
    assert ws.worker_root not in launch["command"][-1]
    assert "coordination-only master" not in serialized
    assert result["execution_mode"] == "single" and result["workers"] == {}
    assert not Path(ws.worker_root).exists()


@pytest.mark.parametrize("execution_mode", ["single", "team"])
@pytest.mark.parametrize("deadline, expected_timeout_ms", [
    (1.0001, 31001), (300, 330000), (600.0001, 660001),
])
def test_registered_provider_timeout_matches_native_environment(
        tmp_path, monkeypatch, execution_mode, deadline, expected_timeout_ms):
    """Auth refresh must retain the relay deadline even for registered models."""
    ws = workspace(tmp_path)

    def fake_run(command, *, env, timeout_seconds, stdout_path, stderr_path):
        Path(stdout_path).write_text(json.dumps({"type": "result", "subtype": "success", "result": "done"}) + "\n")
        Path(stderr_path).write_text("")
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(ws, "run_agent", fake_run)
    output = tmp_path / "run"
    result = QwenTeamRunner("unused-cli", upstream=lambda _: pytest.fail("No model call expected")).run(
        ws, "Implement the task", "Assign work", "Report progress", output,
        ProviderConfig("deepseek-flash", "https://api.deepseek.com",
                       request_timeout_seconds=deadline), 10, execution_mode=execution_mode)
    launch = json.loads((output / "launch.json").read_text())
    settings = json.loads((Path(result["runtime_dir"]) / "profile/settings.json").read_text())
    generation = settings["modelProviders"]["openai"][0]["generationConfig"]
    assert generation == {"maxRetries": 0, "timeout": expected_timeout_ms,
                          "contextWindowSize": 1_000_000}
    assert generation["timeout"] == int(launch["environment"]["QWEN_CODE_API_TIMEOUT_MS"])


@pytest.mark.parametrize("status,message", [
    (409, "Intentional request conflict."),
    (424, "Provider timeout after an upstream HTTP 429 rate limit or HTTP 503 overload."),
    (424, 'Provider request failed: timeout {"error":{"code":503}}'),
])
def test_native_client_does_not_retry_terminal_http_failures(tmp_path, monkeypatch, status, message):
    """A local HTTP fixture exercises SDK configuration and native outer retry decisions."""
    default = Path(__file__).parents[1] / "external/qwen-study/package/cli.js"
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(default)))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Set QWEN_CODE_CLI and QWEN_NODE to execute the native no-provider integration test")

    requests = []

    def start_failure_server(_relay, host):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                requests.append(self.headers.get("x-stainless-retry-count"))
                data = json.dumps({"error": {"message": message, "type": "provider_error",
                                             "code": "provider_error", "provider_http_status": 503}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if status == 424:
                    self.send_header("x-should-retry", "false")
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer((host, 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    monkeypatch.setattr(qwen_module, "start_server", start_failure_server)
    output = tmp_path / "single"
    result = QwenTeamRunner(str(cli.resolve()), node=node,
                            upstream=lambda _: pytest.fail("No upstream provider call expected")).run(
        workspace(tmp_path), "Complete the repository task.", "", "", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=1),
        timeout_seconds=15, execution_mode="single")

    assert len(requests) == 1, (requests, result, (output / "qwen_stderr.txt").read_text())
    assert result["stop_reason"] == "cli_error" and result["native_result"]["explicit_error"], result
    assert not result["execution"]["timed_out"]
    launch = json.loads((output / "launch.json").read_text())
    assert int(launch["environment"]["QWEN_CODE_API_TIMEOUT_MS"]) >= 31000
    settings = json.loads((Path(result["runtime_dir"]) / "profile/settings.json").read_text())
    assert settings["model"]["generationConfig"]["maxRetries"] == 0


@pytest.mark.parametrize("mode, policies, error", [
    ("unknown", ("A", "C"), "execution_mode"),
    ("team", ("", "C"), "Both policies"),
    ("team", ("A", ""), "Both policies"),
])
def test_invalid_mode_and_missing_team_policies_fail_before_launch(tmp_path, mode, policies, error):
    ws = workspace(tmp_path)
    with pytest.raises(ValueError, match=error):
        QwenTeamRunner("unused-cli").run(ws, "Task", *policies, tmp_path / "run",
                                        ProviderConfig("model", "http://localhost"), 10, execution_mode=mode)
    assert not (tmp_path / "run").exists()
