"""Real native CLI against scripted responses only; never a paid model call."""
from pathlib import Path

import pytest

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig
from test_rejection_integration import native_qwen, reply
from test_team import call, workspace


@pytest.mark.parametrize("transient_failure", [False, True])
def test_native_single_stops_at_logical_limit_after_executing_last_tool(tmp_path, transient_failure):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    delivered = Path(ws.repo_path) / "last-admitted-call.txt"
    attempts = []

    def upstream(payload):
        attempts.append(payload)
        if transient_failure and len(attempts) == 1:
            return 524, {"error": {"message": "synthetic temporary failure"}}
        assert len(attempts) == 1 + int(transient_failure)
        return reply(payload, tools=[call("write_file", {
            "file_path": str(delivered), "content": "synthetic delivery\n"})])

    result = QwenTeamRunner(str(cli), node=node, upstream=upstream).run(
        ws, "Write the requested file and report completion.", "", "", tmp_path / "run",
        ProviderConfig("scripted-only", "http://unused.invalid", request_timeout_seconds=15,
                       max_attempts=11, retry_backoff_seconds=0),
        timeout_seconds=30, execution_mode="single", max_model_calls=1)
    assert delivered.read_text() == "synthetic delivery\n"
    assert result["stop_reason"] == "model_call_limit_exceeded"
    assert result["execution"]["returncode"] == 0
    assert not result["execution"]["timed_out"]
    assert result["logical_model_calls"] == 1
    assert result["model_call_budget_rejections"] >= 1
    assert result["provider_attempts"] == len(attempts) == 1 + int(transient_failure)
    assert result["usage_complete"] is (not transient_failure)
