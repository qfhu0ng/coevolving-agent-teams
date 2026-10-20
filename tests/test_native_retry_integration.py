"""One real native request survives a transient relay-owned provider retry."""
import copy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, TokenRates
from test_rejection_integration import native_qwen, reply
from test_team import workspace


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("fault", ["http_503", "disconnect"])
def test_native_request_sees_only_recovered_reply_and_retains_attempt_accounting(tmp_path, mode, fault):
    cli, node = native_qwen()
    requests = []
    failed_usage = {"prompt_tokens": 11, "completion_tokens": 5}
    marker = "TRANSIENT_RETRY_ERROR_NOT_MODEL_VISIBLE"
    final = "Recovered reply from the original native request."

    def upstream(payload):
        requests.append(copy.deepcopy(payload))
        assert len(requests) <= 2, "Recovery must not trigger another native model turn"
        if len(requests) == 1:
            if fault == "disconnect":
                raise ConnectionError(marker)
            return 503, {"error": {"message": marker}, "usage": failed_usage}
        assert requests[0] == payload, "Relay retry must reuse the same input, not inject an API error turn"
        return reply(payload, content=final)

    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=upstream).run(
        workspace(tmp_path), "Return one final acknowledgement without using tools.",
        "Delegate repository work if required." if mode == "team" else "",
        "Communicate concrete handoff evidence." if mode == "team" else "",
        output, ProviderConfig("native-scripted-model", "http://localhost",
                               request_timeout_seconds=15, max_attempts=3, retry_backoff_seconds=0,
                               rates=TokenRates("1", "0.1", "2")),
        timeout_seconds=30, execution_mode=mode)

    assert result["stop_reason"] == "native_cli_completed", result
    assert result["execution"]["returncode"] == 0 and not result["execution"]["timed_out"]
    assert result["provider_attempts"] == result["completed_attempts"] == len(requests) == 2
    assert result["inflight_attempts"] == 0 and not result["native_result"]["explicit_error"]
    assert result["native_result"]["result"]["result"] == final
    assert result["native_result"]["lifecycle"]["unobserved_failure_notices"] == []
    assert result["workers"] == {}
    errors = result["errors"]
    assert len(errors) == 1 and errors[0]["retry_index"] == 0
    assert errors[0]["error_kind"] == ("provider_error" if fault == "http_503" else "provider_transport_error")
    assert errors[0]["raw_usage"] == (failed_usage if fault == "http_503" else None)
    assert result["usage"]["total_tokens"] == (26 if fault == "http_503" else 10)
    assert Decimal(result["known_estimated_cost"]) == Decimal("0.000034" if fault == "http_503" else "0.000013")
    assert result["usage_complete"] is (fault == "http_503")
    assert result["cost_complete"] is (fault == "http_503")
    stdout = (output / "qwen_stdout.jsonl").read_text()
    assert marker not in stdout and "[API Error" not in stdout
    # Check the actual native recorder, not only the relay's retry inputs.
    chats = list((Path(result["runtime_dir"]) / "runtime").glob("**/chats/*.jsonl"))
    assert chats
    native_history = "\n".join(path.read_text() for path in chats)
    assert marker not in native_history and final in native_history
    assert not any(json.loads(line).get("type") == "api_error"
                   for line in native_history.splitlines() if line.strip())
