"""Offline classification of worker failures versus normal runtime shutdown."""

from dataclasses import replace
import json
import time

import pytest

from agent_company.runtime.qwen import provider_incident_summary, runtime_stop_reason
from agent_company.runtime.relay import ProviderConfig, TeamRelay
from agent_company.runtime.team import WorkerRegistry
from test_runner import rig as rig, runtime_result
from test_team import assert_tool_rejection, call, request, response, workspace


def cli_result(*, explicit_error=False, missing_result=False):
    return {"explicit_error": explicit_error, "missing_result": missing_result,
            "result": None if missing_result else {"type": "result", "subtype": "success"}}


def relay_error(kind, *, error="Worker request failed", role="worker", usage_error=None):
    return {"attempt": 1, "role": role, "error_kind": kind,
            "error": error, "usage_error": usage_error}


FATAL_KINDS = ["evidence_write_failed"]


def test_http_budget_expiry_before_process_timeout_is_task_timeout():
    native = cli_result(explicit_error=True)
    native["result"]["result"] = "[API Error: task_deadline_exceeded: task budget exhausted]"
    execution = {"returncode": 0, "timed_out": False}
    summary = {"deadline_rejections": ["master"], "errors": []}
    assert runtime_stop_reason(execution, native, None, summary) == "task_timeout"
    assert runtime_stop_reason(execution, native, None, {"errors": []}) == "cli_error"
    assert runtime_stop_reason({**execution, "returncode": 137}, native, None, summary) == "cli_error"
    summary["errors"] = [relay_error("evidence_write_failed")]
    assert runtime_stop_reason(execution, native, None, summary) == "provider_error"


PROVIDER_INCIDENTS = ["provider_error", "provider_transport_error", "provider_timeout"]


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("ending", ["task_deadline", "provider_deadline", "http_524"])
def test_native_inflight_deadline_requires_task_evidence(tmp_path, monkeypatch, mode, ending):
    """Replay the 10-02 deadline path and the distinct 10-01 P0 final 524 path."""
    import agent_company.runtime.qwen as qwen_module
    from agent_company.runtime.transport import RequestDeadlineExceeded
    from test_rejection_integration import native_qwen

    cli, node = native_qwen()
    relays = []

    def capture_relay(*args, **kwargs):
        relay = TeamRelay(*args, **kwargs)
        relays.append(relay)
        return relay

    monkeypatch.setattr(qwen_module, "TeamRelay", capture_relay)

    def upstream(payload):
        if ending == "task_deadline":
            # The native request was already admitted. Expire only the task
            # deadline while it is in flight, before the process timeout fires.
            relays[0].deadline = time.monotonic() - 1
        if ending == "http_524":
            # A backoff larger than the remaining task budget prevents retry;
            # this is still a provider error, not a task deadline rejection.
            return 524, {"error": {"message": "Upstream unavailable", "retry_after": 120}}
        raise RequestDeadlineExceeded("Total request deadline exceeded")

    result = qwen_module.QwenTeamRunner(str(cli), node=node, upstream=upstream).run(
        workspace(tmp_path), "Return one final acknowledgement without using tools.",
        "Delegate repository work if required." if mode == "team" else "",
        "Communicate concrete handoff evidence." if mode == "team" else "",
        tmp_path / "run", ProviderConfig("native-scripted-model", "http://localhost",
            request_timeout_seconds=1800, max_attempts=11, retry_backoff_seconds=120),
        timeout_seconds=30, execution_mode=mode)

    assert result["execution"]["returncode"] == 0 and not result["execution"]["timed_out"]
    assert result["native_result"]["explicit_error"]
    assert result["deadline_rejections"] == []
    assert result["provider_attempts"] == result["completed_attempts"] == 1
    assert result["errors"][0]["error_kind"] == {
        "task_deadline": "task_deadline_exceeded", "provider_deadline": "provider_timeout",
        "http_524": "provider_error"}[ending]
    assert result["stop_reason"] == ("task_timeout" if ending == "task_deadline" else "cli_error")


@pytest.mark.parametrize("case,expected", [
    ("matched", "task_timeout"),
    ("worker_only", "cli_error"),
    ("provider_timeout", "cli_error"),
    ("different_native_error", "cli_error"),
    ("different_error_message", "cli_error"),
    ("no_relay_error", "cli_error"),
    ("usage_error_only", "cli_error"),
    ("nonzero_exit", "cli_error"),
    ("normal_completion", "native_cli_completed"),
    ("evidence_failure", "provider_error"),
])
def test_inflight_deadline_classification_needs_matching_root_error(case, expected):
    message = "RequestDeadlineExceeded: Total request deadline exceeded"
    native = cli_result(explicit_error=True)
    native["result"]["result"] = f"[API Error: task_deadline_exceeded: 424 {message}]"
    execution = {"returncode": 0, "timed_out": False, "elapsed_seconds": 21600}
    error = relay_error("task_deadline_exceeded", role="single", error=message)
    summary = {"execution_mode": "single", "deadline_rejections": [], "errors": [error]}
    if case == "worker_only":
        error["role"] = "worker"
    elif case == "provider_timeout":
        error["error_kind"] = "provider_timeout"
    elif case == "different_native_error":
        native["result"]["result"] = f"[API Error: provider_timeout: 424 {message}]"
    elif case == "different_error_message":
        native["result"]["result"] = "[API Error: task_deadline_exceeded: another error]"
    elif case == "no_relay_error":
        summary["errors"] = []
    elif case == "usage_error_only":
        error.update(error=None, usage_error="Missing usage")
    elif case == "nonzero_exit":
        execution["returncode"] = 137
    elif case == "normal_completion":
        native["explicit_error"] = False
        native["result"]["result"] = f"Completed despite the earlier task_deadline_exceeded: {message}"
    elif case == "evidence_failure":
        summary["errors"].append(relay_error("evidence_write_failed"))
    assert runtime_stop_reason(execution, native, None, summary) == expected


@pytest.mark.parametrize("kind", FATAL_KINDS)
@pytest.mark.parametrize("ending", ["master_success", "task_timeout", "missing_result"])
def test_evidence_failure_is_not_hidden_by_completion_or_deadline(kind, ending):
    execution = {"returncode": 124 if ending == "task_timeout" else 0,
                 "timed_out": ending == "task_timeout"}
    native = cli_result(missing_result=ending in {"task_timeout", "missing_result"})
    # Later normal shutdown records must not erase an earlier worker failure.
    summary = {"errors": [relay_error("tool_response_rejected"), relay_error(kind),
                           relay_error("request_cancelled", role="master")],
               "usage": {"prompt_tokens": 150, "completion_tokens": 25}}
    assert runtime_stop_reason(execution, native, None, summary) == "provider_error"


@pytest.mark.parametrize("kind", PROVIDER_INCIDENTS + ["invalid_provider_response"])
@pytest.mark.parametrize("role", ["master", "worker", "replaced-worker"])
@pytest.mark.parametrize("timed_out", [False, True])
def test_historical_provider_incident_does_not_veto_final_completion(kind, role, timed_out):
    error = relay_error(kind, role=role)
    summary = {"errors": [error], "usage_complete": False, "cost_complete": False}
    execution = {"returncode": 124 if timed_out else 0, "timed_out": timed_out}
    expected = "task_timeout" if timed_out else "native_cli_completed"
    assert runtime_stop_reason(execution, cli_result(missing_result=timed_out), None, summary) == expected
    assert summary == {"errors": [error], "usage_complete": False, "cost_complete": False}


@pytest.mark.parametrize("kind", PROVIDER_INCIDENTS + ["invalid_provider_response"])
@pytest.mark.parametrize("native,returncode,expected", [
    (cli_result(explicit_error=True), 0, "cli_error"),
    (cli_result(), 1, "cli_error"),
    (cli_result(missing_result=True), 0, "missing_cli_result"),
])
def test_provider_incident_does_not_hide_actual_cli_failure(kind, native, returncode, expected):
    summary = {"errors": [relay_error(kind)]}
    assert runtime_stop_reason({"returncode": returncode, "timed_out": False}, native,
                               None, summary) == expected


def test_all_workers_provider_failures_remain_visible_without_claiming_recovery():
    errors = [relay_error("provider_transport_error", role="worker1"),
              relay_error("provider_timeout", role="worker2"),
              relay_error("provider_error", role="worker1"),
              relay_error("tool_response_rejected", role="master"),
              relay_error("provider_error", error=None, usage_error="Missing usage")]
    summary = {"errors": errors, "workers": {"worker1": "/w/1", "worker2": "/w/2"}}
    assert provider_incident_summary(summary) == {
        "count": 3, "by_kind": {"provider_transport_error": 1, "provider_timeout": 1, "provider_error": 1},
        "by_role": {"worker1": 2, "worker2": 1},
    }
    # Neither a normal master exit nor registered workers prove successful work.
    # The final repo still has to be scored; a zero score must stay zero.
    assert runtime_stop_reason({"returncode": 0, "timed_out": False}, cli_result(),
                               None, summary) == "native_cli_completed"


@pytest.mark.parametrize("kind", ["task_deadline_exceeded", "request_cancelled", "tool_response_rejected"])
@pytest.mark.parametrize("timed_out", [False, True])
def test_normal_shutdown_and_recovered_tool_rejection_are_not_provider_failures(kind, timed_out):
    execution = {"returncode": 124 if timed_out else 0, "timed_out": timed_out}
    summary = {"errors": [relay_error(kind)]}
    expected = "task_timeout" if timed_out else "native_cli_completed"
    assert runtime_stop_reason(execution, cli_result(), None, summary) == expected


@pytest.mark.parametrize("kind", [None, "provider_error", "provider_timeout"])
def test_usage_only_record_does_not_invalidate_execution_even_with_error_kind(kind):
    summary = {"usage_complete": False, "cost_complete": False,
               "errors": [relay_error(kind, error=None, usage_error="Missing prompt_tokens")]}
    assert runtime_stop_reason({"returncode": 0, "timed_out": False}, cli_result(),
                               None, summary) == "native_cli_completed"


@pytest.mark.parametrize("execution,native,launcher,expected", [
    ({"returncode": 0, "timed_out": False}, cli_result(), None, "native_cli_completed"),
    ({"returncode": 124, "timed_out": True}, cli_result(missing_result=True), None, "task_timeout"),
    ({"returncode": 0, "timed_out": False}, cli_result(missing_result=True), None, "missing_cli_result"),
    ({"returncode": 1, "timed_out": False}, cli_result(), None, "cli_error"),
    ({"returncode": 0, "timed_out": False}, cli_result(explicit_error=True), None, "cli_error"),
    ({}, cli_result(missing_result=True), "Container launch failed", "launcher_error"),
])
def test_existing_runtime_outcomes_remain_unchanged_without_worker_failure(execution, native, launcher, expected):
    assert runtime_stop_reason(execution, native, launcher, {"errors": []}) == expected


@pytest.mark.parametrize("execution,native,launcher,expected", [
    ({"returncode": 0, "timed_out": False}, cli_result(), "Launcher failed", "launcher_error"),
    ({"returncode": 124, "timed_out": True}, cli_result(), "Launcher failed", "launcher_error"),
    ({"returncode": 1, "timed_out": False}, cli_result(), None, "cli_error"),
    ({"returncode": 0, "timed_out": False}, cli_result(explicit_error=True), None, "cli_error"),
])
def test_explicit_launcher_or_nontimeout_master_error_takes_priority(execution, native, launcher, expected):
    summary = {"errors": [relay_error("provider_transport_error")]}
    assert runtime_stop_reason(execution, native, launcher, summary) == expected


@pytest.mark.parametrize("mode", ["team", "single"])
@pytest.mark.parametrize("cost_complete", [False, True])
def test_protocol_failure_is_failed_but_retains_verifier_reward_usage_and_known_cost(rig, mode, cost_complete):
    usage = {"prompt_tokens": 201, "completion_tokens": 37, "total_tokens": 238}
    errors = [relay_error("invalid_provider_response", error="Provider returned invalid JSON")]
    rig.runtime = runtime_result(stop_reason="provider_error", usage=usage, errors=errors,
                                native_result=cli_result(), cost_complete=cost_complete,
                                currency="USD", known_estimated_cost="0.5")
    result = rig.runner(replace(rig.request, execution_mode=mode))
    assert result.status == "failed" and result.score == 0.75
    assert "provider_error" in result.error
    assert result.metrics["verifier"] == {"status": "completed", "reward": 0.75}
    assert result.metrics[mode]["native_result"]["explicit_error"] is False
    assert result.metrics[mode]["errors"] == errors and result.usage == usage
    assert result.known_cost_usd == 0.5
    assert result.cost_usd == (0.5 if cost_complete else None)
    assert rig.events.index("team-ended") < rig.events.index("score-hidden") < rig.events.index("stop")
    assert rig.workspaces[0].stopped
    assert "+answer = 42" in (rig.request.output_dir / "delivery.patch").read_text()
    saved = json.loads((rig.request.output_dir / "runner_result.json").read_text())
    assert saved["status"] == "failed" and saved["score"] == 0.75
    assert saved["metrics"]["verifier"]["reward"] == 0.75 and saved["usage"] == usage


@pytest.mark.parametrize("mode", ["team", "single"])
@pytest.mark.parametrize("reward", [0, 0.75])
def test_oom_cli_failure_keeps_official_score_and_runtime_diagnostics(rig, mode, reward):
    from agent_company.evolution.loop import _record_result, _task_summary
    from agent_company.evolution.selection import select_candidate

    execution = {"returncode": 137, "timed_out": False, "oom_killed": True}
    native = cli_result(missing_result=True)
    reason = runtime_stop_reason(execution, native, None, {"errors": []})
    assert reason == "cli_error"
    rig.runtime = runtime_result(stop_reason=reason, execution=execution, native_result=native)
    rig.verifier = {"status": "completed", "reward": reward}
    request = replace(rig.request, execution_mode=mode)
    result = rig.runner(request)
    assert result.status == "failed" and result.score == reward
    assert "cli_error" in result.error
    assert result.metrics[mode]["execution"] == execution
    assert result.metrics[mode]["native_result"]["missing_result"] is True
    assert result.metrics["verifier"] == rig.verifier
    record = _record_result(request, result, None)
    saved = json.loads((request.output_dir / "task_result.json").read_text())
    assert saved["status"] == "failed" and saved["score"] == reward
    compact = _task_summary(record, request.output_dir.parent)
    assert compact["status"] == "failed" and compact["score"] == reward
    assert compact["official_result"]["raw_score"] == reward
    assert compact["runtime_outcome"]["terminal_reason"] == "cli_error"
    assert compact["runtime_outcome"]["execution"] == execution
    assert compact["error"] == result.error
    candidate = {**record, "status": "completed", "score": reward + 0.125}
    decision = select_candidate([record], [candidate])
    assert decision["decision"] == "accept" and decision["comparable_count"] == 1
    assert decision["pairs"][0]["incumbent_status"] == "failed"
    assert decision["pairs"][0]["incumbent_score"] == reward


def test_actual_worker_transport_error_followed_by_master_success_can_be_scored(tmp_path):
    registry = WorkerRegistry(workspace(tmp_path))
    registry.register("worker")

    def upstream(payload):
        if "agent \"worker\"" in payload["messages"][0]["content"]:
            raise ConnectionError("Upstream server disconnected")
        return 200, response(usage={"prompt_tokens": 11, "completion_tokens": 7})

    relay = TeamRelay(ProviderConfig("test", "https://provider.invalid", max_attempts=1),
                      registry, "policy", tmp_path / "relay", 60, upstream)
    worker_status, _ = relay.handle(request("worker"))
    master_status, _ = relay.handle(request())
    assert worker_status == 424 and master_status == 200
    summary = relay.summary()
    assert summary["errors"][0]["role"] == "worker"
    assert summary["errors"][0]["error_kind"] == "provider_transport_error"
    assert runtime_stop_reason({"returncode": 0, "timed_out": False}, cli_result(),
                               None, summary) == "native_cli_completed"
    assert not summary["usage_complete"] and not summary["cost_complete"]


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.parametrize("reward", [0, 0.75])
def test_scored_delivery_with_provider_incidents_preserves_unknown_cost(rig, mode, timed_out, reward):
    from agent_company.evolution.loop import _record_result, _task_summary
    from agent_company.evolution.selection import select_candidate

    errors = [relay_error(kind) for kind in PROVIDER_INCIDENTS]
    usage = {"prompt_tokens": 201, "completion_tokens": 37, "total_tokens": 238}
    execution = {"returncode": 124 if timed_out else 0, "timed_out": timed_out}
    native = cli_result(missing_result=timed_out)
    reason = runtime_stop_reason(execution, native, None, {"errors": errors})
    rig.runtime = runtime_result(stop_reason=reason, execution=execution, native_result=native,
                                 errors=errors, usage=usage, usage_complete=False,
                                 cost_complete=False, currency="USD", known_estimated_cost="0.5",
                                 provider_incidents=provider_incident_summary({"errors": errors}))
    rig.verifier = {"status": "completed", "reward": reward}
    result = rig.runner(replace(rig.request, execution_mode=mode))
    assert result.status == "completed" and result.score == reward and result.error is None
    assert result.metrics[mode]["errors"] == errors
    assert result.metrics[mode]["usage_complete"] is False
    assert result.usage == usage and result.cost_usd is None and result.known_cost_usd == 0.5
    saved = json.loads((rig.request.output_dir / "runner_result.json").read_text())
    assert saved["score"] == reward and saved["metrics"][mode]["errors"] == errors
    assert saved["cost_usd"] is None
    record = _record_result(replace(rig.request, execution_mode=mode), result, None)
    compact = _task_summary(record, rig.request.output_dir.parent)
    assert compact["provider_incidents"] == rig.runtime["provider_incidents"]
    record["metrics"].update({f"large_metric_{index}": "x" * 1300 for index in range(8)})
    compact = _task_summary(record, rig.request.output_dir.parent)
    assert len(json.dumps(compact)) < 9000
    assert compact["official_result"]["raw_score"] == reward
    assert compact["runtime_outcome"]["terminal_reason"] == reason
    assert compact["provider_incidents"] == rig.runtime["provider_incidents"]
    # Missing usage cannot create a spurious efficiency win at equal quality.
    row = {"task_id": "one", "repo_id": "repo", "status": "completed", "score": reward}
    decision = select_candidate([{**row, "cost_usd": 1}], [{**row, "cost_usd": result.cost_usd}])
    assert decision["decision"] == "inconclusive" and decision["reason"] == "incomplete_token_evidence"


@pytest.mark.parametrize("mode", ["single", "team"])
def test_provider_incident_does_not_hide_verifier_failure(rig, mode):
    rig.runtime = runtime_result(errors=[relay_error("provider_transport_error")],
                                 usage_complete=False)
    rig.verifier = {"status": "verifier_error", "reward": None}
    result = rig.runner(replace(rig.request, execution_mode=mode))
    assert result.status == "failed" and result.score is None
    assert "verifier_error" in result.error


def test_actual_tool_schema_rejection_can_recover_without_provider_failure(tmp_path):
    invalid = response([call("run_shell_command", {"command": "echo forbidden for master"})],
                       {"prompt_tokens": 4, "completion_tokens": 2})
    replies = iter([invalid,
                    response(usage={"prompt_tokens": 5, "completion_tokens": 3})])
    relay = TeamRelay(ProviderConfig("test", "https://provider.invalid"),
                      WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "relay", 60,
                      lambda _: (200, next(replies)))
    rejected, delivered = relay.handle(request())
    recovered, _ = relay.handle(request())
    assert rejected == recovered == 200
    assert "Disallowed master tool" in assert_tool_rejection(invalid, delivered)[0]
    summary = relay.summary()
    assert summary["errors"][0]["error_kind"] == "tool_response_rejected"
    assert summary["usage"]["prompt_tokens"] == 9
    assert runtime_stop_reason({"returncode": 0, "timed_out": False}, cli_result(),
                               None, summary) == "native_cli_completed"


@pytest.mark.parametrize("native,returncode,expected", [
    (cli_result(explicit_error=True), 0, "cli_error"),
    (cli_result(), 1, "cli_error"),
    (cli_result(missing_result=True), 0, "missing_cli_result"),
])
def test_tool_feedback_does_not_turn_an_unrecovered_native_failure_into_success(native, returncode, expected):
    summary = {"errors": [relay_error("tool_response_rejected")]}
    assert runtime_stop_reason({"returncode": returncode, "timed_out": False}, native,
                               None, summary) == expected


@pytest.mark.parametrize("mode", ["team", "single"])
@pytest.mark.parametrize("stop_reason", ["native_cli_completed", "task_timeout"])
def test_recovered_tool_failure_keeps_score_and_cost_without_hiding_rejection(rig, mode, stop_reason):
    errors = [relay_error("tool_response_rejected", error="Disallowed tool", role=mode)]
    usage = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    rig.runtime = runtime_result(stop_reason=stop_reason, errors=errors, usage=usage,
                                cost_complete=True, currency="USD", known_estimated_cost="0.2")
    result = rig.runner(replace(rig.request, execution_mode=mode))
    assert result.status == "completed" and result.score == 0.75
    assert result.error is None
    assert result.metrics[mode]["errors"] == errors
    assert result.usage == usage and result.cost_usd == result.known_cost_usd == 0.2
