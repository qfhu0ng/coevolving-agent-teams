"""Actual native teammates can replace a worker after a scripted provider fault."""

from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import shlex
import threading
import uuid

import pytest

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, TokenRates, role_of
from agent_company.runtime.transport import RequestDeadlineExceeded
from test_rejection_integration import native_qwen, reply
from test_team import call, workspace


class ReplacementScenario:
    """Only model responses are scripted; spawning, shell and shutdown are native."""

    def __init__(self, ws, fault):
        self.ws, self.fault = ws, fault
        self.steps = defaultdict(int)
        self.lock = threading.Lock()
        self.master_waiting = threading.Event()
        self.failure_notice_seen = False
        self.integrated = threading.Event()
        self.shutdown_seen = threading.Event()
        self.team_name = "network-recovery-" + uuid.uuid4().hex[:10]

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            assert sum(self.steps.values()) <= 16, "Worker replacement did not converge"
        text = json.dumps(payload["messages"])
        users = [str(message.get("content", "")) for message in payload["messages"]
                 if message.get("role") == "user"]

        if role == "flaky":
            assert step == 0, "A failed worker request must not be retried automatically"
            assert self.master_waiting.wait(20), "Master did not enter the last-worker wait"
            if self.fault == "provider_error":
                return 200, {"error": {"message": "Scripted temporary provider outage"}}
            if self.fault == "provider_timeout":
                raise RequestDeadlineExceeded("Scripted temporary request timeout")
            raise ConnectionError("Scripted temporary server disconnect")

        if role == "master":
            if step == 0:
                return reply(payload, [call("team_create", {"team_name": self.team_name})])
            if step == 2:
                self.master_waiting.set()
                return reply(payload, content="Waiting for the only worker to report its delivery.")
            if step in {1, 3}:
                name = "flaky" if step == 1 else "replacement"
                if step == 3:
                    assert 'from=\\"runtime:system\\"' in text
                    assert "native-worker-failure-" in text and "worker flaky FAILED" in text
                    self.failure_notice_seen = True
                return reply(payload, [call("agent", {
                    "name": name, "description": "Write and verify the delivery",
                    "prompt": "Use your assigned worktree and an absolute delivery path to write recovered.txt.",
                    "subagent_type": "general-purpose"})])
            if "REPLACEMENT_DELIVERY_COMPLETE" in text and not self.shutdown_seen.is_set():
                return reply(payload, [call("request_shutdown", {"to": "replacement"})])
            if "shutdown_approved" in text:
                return reply(payload, content="Delivery verified by the replacement after a temporary provider failure.")
            return reply(payload, content="Waiting for replacement delivery and shutdown reports.")

        assert role == "replacement" and self.failure_notice_seen
        if "shutdown" in (users[-1] if users else "").lower():
            self.shutdown_seen.set()
            return reply(payload, [call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        if step == 0:
            own = shlex.quote(str(Path(self.ws.worker_root) / "replacement"))
            delivery = shlex.quote(str(Path(self.ws.repo_path) / "recovered.txt"))
            command = (
                f"pwd && test \"$PWD\" = {own} && "
                f"printf '%s\\n' recovered-through-native-replacement > {delivery} && "
                f"cat {delivery} && printf '%s\\n' NATIVE_NETWORK_RECOVERY_OK")
            return reply(payload, [call("run_shell_command", {
                "command": command, "description": "Write and verify replacement delivery"})])
        assert "NATIVE_NETWORK_RECOVERY_OK" in text
        assert str(Path(self.ws.worker_root) / "replacement") in text
        self.integrated.set()
        return reply(payload, content="REPLACEMENT_DELIVERY_COMPLETE")


@pytest.mark.parametrize("fault", ["provider_error", "provider_transport_error", "provider_timeout"])
def test_native_replacement_completion_retains_historical_network_failure(tmp_path, fault):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    scenario = ReplacementScenario(ws, fault)
    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario).run(
        ws, "Create recovered.txt using a teammate, replacing a failed worker when needed.",
        "Delegate implementation and delivery verification to a writable teammate.",
        "Report concrete verification and approve requested shutdown.", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35,
                       max_attempts=1,
                       rates=TokenRates("1", "0.1", "2")), timeout_seconds=90)

    assert result["execution"]["returncode"] == 0, (result, (output / "qwen_stderr.txt").read_text())
    assert result["stop_reason"] == "native_cli_completed", result
    assert not result["execution"]["timed_out"] and not result["native_result"]["explicit_error"]
    assert result["scripted_provider"] and scenario.integrated.is_set() and scenario.shutdown_seen.is_set()
    assert scenario.failure_notice_seen
    lifecycle = result["native_result"]["lifecycle"]
    assert lifecycle["unobserved_failure_notices"] == [] and lifecycle["notification_errors"] == []
    assert sum(event["event"] == "failure_notice_observed" for event in lifecycle["events"]) == 1
    terminal_notice = next(event for event in lifecycle["events"] if event["event"] == "failure_notice_queued")
    # Native formats timeout errors into its own diagnostic instead of keeping
    # the provider's exact message. Both accessors supply a real terminal cause.
    expected_cause = "Request timeout" if fault == "provider_timeout" else "Scripted temporary"
    assert expected_cause in terminal_notice["error"], terminal_notice
    assert "Unknown terminal failure" not in terminal_notice["error"]
    assert scenario.steps["flaky"] == 1 and scenario.steps["replacement"] >= 3
    assert (Path(ws.repo_path) / "recovered.txt").read_text() == "recovered-through-native-replacement\n"
    assert result["workers"] == {name: str(Path(ws.worker_root) / name)
                                 for name in ("flaky", "replacement")}
    assert result["provider_attempts"] == result["completed_attempts"] == sum(scenario.steps.values())
    assert result["inflight_attempts"] == 0
    assert len(result["errors"]) == 1
    error = result["errors"][0]
    assert error["role"] == "flaky" and error["error_kind"] == fault
    assert error["relay_http_status"] == 424 and error["raw_usage"] is None
    incidents = {"count": 1, "by_kind": {fault: 1}, "by_role": {"flaky": 1}}
    assert result["provider_incidents"] == incidents
    assert not result["usage_complete"] and not result["cost_complete"]
    successful_attempts = result["provider_attempts"] - 1
    assert result["usage"]["total_tokens"] == successful_attempts * 10
    assert Decimal(result["known_estimated_cost"]) == successful_attempts * Decimal("0.000013")
    saved = json.loads((output / "summary.json").read_text())
    assert saved["stop_reason"] == "native_cli_completed" and saved["errors"] == result["errors"]
    assert saved["provider_incidents"] == incidents
    worker_requests = (output / "relay").glob("attempt_*_replacement_provider_request.json")
    assert any("NATIVE_NETWORK_RECOVERY_OK" in path.read_text() for path in worker_requests)
