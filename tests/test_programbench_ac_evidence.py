"""Synthetic projection checks: no models, SSH, Docker, or actual run data."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent_company.analysis import evidence_projection as MODULE


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/project_programbench_ac_evidence.py"


def usage():
    return {"prompt_tokens": 10, "completion_tokens": 2, "cached_prompt_tokens": 1, "total_tokens": 12}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def call(tool, args, identifier="call-1"):
    return {"id": identifier, "function": {"name": tool, "arguments": json.dumps(args)}}


def response(path, calls, alternatives=()):
    write_json(path, {"reasoning_content": "TOP_REASONING_POISON", "choices": [
        {"message": {"reasoning_content": "REASONING_POISON", "thinking": "THINKING_POISON",
                     "content": "ASSISTANT_CONTENT_POISON", "tool_calls": calls}}, *alternatives]})


def fixture_run(tmp_path):
    root = tmp_path / "trial"
    relay = root / "team/relay"
    relay.mkdir(parents=True)
    write_json(root / "task_result.json", {"status": "failed", "score": .8, "usage": usage(),
        "reasoning_content": "TASK_REASONING_POISON", "error": "RAW_ERROR_POISON",
        "metrics": {"task": {"task_id": "synthetic-task", "repository_id": "repo", "instruction": "INSTRUCTION_POISON"},
                    "team": {"logical_model_calls": 2, "usage_complete": True, "stop_reason": "task_timeout",
                             "logical_calls_by_role": {"master": 1, "worker": 1},
                             "execution": {"elapsed_seconds": 7, "timed_out": True, "diagnostics": "DIAGNOSTICS_POISON"},
                             "native_result": {"explicit_error": False, "result": {"subtype": "success", "result": "NATIVE_POISON"}}},
                    "verifier": {"status": "completed", "reward": .8, "passed_tests": 8, "valid_tests": 10,
                                 "hidden_tests": "HIDDEN_TEST_POISON", "system_errors": ["ERROR_BODY_POISON"]},
                    "resource_config": {"max_model_calls": None, "cpus": 20, "memory_mb": 61440, "credentials": "CREDENTIAL_POISON"}}})
    return root, relay


def events(relay, rows):
    (relay / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_projection_excludes_prohibited_parents_and_redacts_tool_text(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [
        {"event": "provider_attempt_started", "attempt": 1, "role": "master", "time": 1, "reasoning_content": "EVENT_POISON"},
        {"event": "provider_attempt_finished", "attempt": 1, "role": "master", "usage": usage(), "elapsed_seconds": 2,
         "provider_http_status": 200, "error": "ERROR_POISON", "raw_usage": {"body": "RAW_USAGE_POISON"}},
    ])
    response(relay / "attempt_00001_master_qwen_response.json", [call("send_message", {
        "recipient": "worker", "content": "Please test it. API_KEY=abc-secret-value Authorization: Bearer opaque-secret",
        "reasoning_content": "ARG_REASONING_POISON", "thinking": "ARG_THINKING_POISON",
        "api_key": "CREDENTIAL_ARG_POISON", "metadata": {"content": "NESTED_POISON"}})])
    (root / "qwen_stdout.jsonl").write_text("STDOUT_POISON")
    (relay / "attempt_00001_master_qwen_request.json").write_text("REQUEST_POISON")
    (relay / "attempt_00001_master_provider_response.json").write_text("PROVIDER_POISON")
    report = MODULE.project_evidence(root)
    serialized = json.dumps(report, allow_nan=False)
    assert "POISON" not in serialized
    assert "abc-secret-value" not in serialized and "opaque-secret" not in serialized
    item = report["collaboration_timeline"][0]
    assert item["fields"]["content"]["redacted"]
    assert item["fields"]["content"]["text"].startswith("Please test it.")
    assert item["execution_status"] == "execution_unconfirmed" and item["receipt_confirmed"] is None
    assert report["task_result"]["official_result"]["reward"] == .8
    assert report["task_result"]["resources"]["max_model_calls"] is None
    assert report["task_result"]["runtime"]["logical_model_calls"] == 2


def test_first_choice_only_and_duplicate_attempt_event_tool_ids(tmp_path):
    root, relay = fixture_run(tmp_path)
    row = {"event": "provider_attempt_finished", "attempt": 1, "role": "master", "usage": usage(), "elapsed_seconds": 3}
    events(relay, [row, row, {"event": "provider_retry_scheduled", "after_attempt": 1, "role": "master", "retry_index": 1, "delay_seconds": 120},
                   {"event": "provider_retry_scheduled", "after_attempt": 1, "role": "master", "retry_index": 1, "delay_seconds": 120}])
    first = call("agent", {"name": "worker", "prompt": "Implement a behavior"})
    response(relay / "attempt_00001_master_qwen_response.json", [first, first], [
        {"message": {"tool_calls": [call("send_message", {"content": "ALTERNATE_POISON"})]}}])
    response(relay / "attempt_1_master_qwen_response.json", [call("send_message", {"content": "DUPLICATE_POISON"})])
    report = MODULE.project_evidence(root)
    assert "POISON" not in json.dumps(report)
    assert len(report["collaboration_timeline"]) == 1
    row = report["by_role"][0]
    assert row["known_tokens"] == usage()
    assert row["known_scheduled_retry_delay_seconds"] == 120
    assert row["tool_calls"] == {"agent": 1}
    assert report["coverage"]["issues"] == {"duplicate_attempt_events": 1, "duplicate_retry_events": 1,
                                              "duplicate_tool_call_id": 1, "duplicate_response_attempts": 1}


def test_invalid_arguments_and_nonfinite_usage_stay_unknown(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [
        {"event": "provider_attempt_finished", "attempt": 1, "role": "master", "usage": {**usage(), "total_tokens": float("nan")}, "elapsed_seconds": float("inf")},
        {"event": "provider_attempt_finished", "attempt": 2, "role": "master", "usage": None, "elapsed_seconds": 4},
    ])
    malformed = {"id": "bad", "function": {"name": "send_message", "arguments": "NOT_JSON_POISON"}}
    response(relay / "attempt_00001_master_qwen_response.json", [malformed, call("task_update", ["ARRAY_POISON"], "array")])
    response(relay / "attempt_00002_master_qwen_response.json", [call("task_update", {"task_id": "t1", "status": "completed", "__agent_company_tool_error": "ERROR_POISON"})])
    report = MODULE.project_evidence(root)
    assert "POISON" not in json.dumps(report, allow_nan=False)
    row = report["by_role"][0]
    assert row["finished_attempts_missing_or_invalid_usage"] == 2
    assert row["known_tokens"]["total_tokens"] == 0
    assert row["finished_attempts_missing_time"] == 1
    assert report["coverage"]["issues"]["invalid_tool_arguments"] == 2
    assert report["collaboration_timeline"][0]["execution_status"] == "rejected_before_execution"


def test_truncation_shell_classification_and_no_code_exposure(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [])
    response(relay / "attempt_00001_master_qwen_response.json", [
        call("task_create", {"description": "a" * 8100}, "task"),
        call("run_shell_command", {"command": "pytest SECRET_SHELL_POISON", "description": "SHELL_DESCRIPTION_POISON"}, "shell"),
        call("write_file", {"content": "CODE_POISON"}, "write"),
    ])
    report = MODULE.project_evidence(root)
    assert "POISON" not in json.dumps(report)
    field = report["collaboration_timeline"][0]["fields"]["description"]
    assert field == {"text": "a" * 8000, "truncated": True, "original_characters": 8100, "redacted": False}
    assert report["by_role"][0]["shell_categories"] == {"test_build_lint": 1}
    assert report["by_role"][0]["tool_calls"] == {"task_create": 1, "run_shell_command": 1, "write_file": 1}


@pytest.mark.parametrize("secret", [
    "sk-secret123456789", "ghp_abcdefghijk", "github_pat_abcdefghijk", "AKIA1234567890ABCDEF",
    "xoxb-123456789-secret", "Bearer not-a-real-key", "password='my password'",
    "STAGE1_GLM_API_KEY=secretvalue", "https://alice:privatevalue@example.com/api",
    "token=privatevalue",
    "https://example.com/?access_token=privatevalue", "-----BEGIN OPENSSH PRIVATE KEY-----\nprivatevalue\n-----END OPENSSH PRIVATE KEY-----",
])
def test_credential_shapes_redacted(secret):
    projected = MODULE.text_leaf(secret)
    assert projected["redacted"] and "[REDACTED]" in projected["text"]
    assert "privatevalue" not in projected["text"]


def test_symlink_source_is_not_followed_and_cli_json_only(tmp_path):
    root, relay = fixture_run(tmp_path)
    outside = tmp_path / "outside.json"
    response(outside, [call("send_message", {"content": "LINK_POISON"})])
    (relay / "attempt_00001_master_qwen_response.json").symlink_to(outside)
    events(relay, [])
    result = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(root)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert result.stderr == "" and "POISON" not in result.stdout
    assert report["coverage"]["issues"]["unsafe_response_file"] == 1
    assert report["collaboration_timeline"] == []


def test_task_projection_never_preserves_untrusted_nested_values(tmp_path):
    root, relay = fixture_run(tmp_path)
    write_json(root / "task_result.json", {"score": True, "usage": usage(), "metrics": {
        "team": {"usage_complete": True, "logical_model_calls": {"content": "PARENT_POISON"},
                 "stop_reason": {"reasoning_content": "NESTED_REASONING_POISON"}, "execution": {"elapsed_seconds": -1}},
        "verifier": {"reward": float("nan"), "status": {"content": "VERIFIER_POISON"}}}})
    events(relay, [])
    report = MODULE.project_evidence(root)
    assert "POISON" not in json.dumps(report, allow_nan=False)
    assert report["task_result"]["score"] is None
    assert report["task_result"]["official_result"]["reward"] is None
    assert report["task_result"]["runtime"]["logical_model_calls"] is None


def test_negative_exit_and_missing_usage_are_not_discarded(tmp_path):
    root, relay = fixture_run(tmp_path)
    write_json(root / "task_result.json", {"usage": {**usage(), "unknown_usage_attempts": 1},
        "metrics": {"team": {"usage_complete": True, "execution": {"returncode": -9}}}})
    events(relay, [])
    report = MODULE.project_evidence(root)
    assert report["task_result"]["runtime"]["execution"]["returncode"] == -9
    assert report["task_result"]["runtime"]["usage_complete"] is False


def test_credential_shaped_role_and_path_are_redacted(tmp_path):
    root, relay = fixture_run(tmp_path)
    role = "sk-secret123456789"
    events(relay, [{"event": "provider_attempt_started", "attempt": 1, "role": role}])
    response(relay / f"attempt_00001_{role}_qwen_response.json", [call("send_message", {"recipient": role, "content": "hello"})])
    report = MODULE.project_evidence(root)
    assert role not in json.dumps(report)
    assert report["by_role"][0]["role"] == "[REDACTED]"


def test_expected_error_response_is_not_reported_as_corrupt_and_time_not_zero(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [{"event": "provider_attempt_started", "attempt": 1, "role": "master", "time": "2026-09-24T00:00:00Z"},
                   {"event": "provider_attempt_finished", "attempt": 1, "role": "master", "provider_http_status": 524,
                    "error_kind": "provider_error", "usage": None}])
    write_json(relay / "attempt_00001_master_qwen_response.json", {"error": {"message": "PROVIDER_ERROR_POISON"}})
    write_json(relay / "attempt_00002_master_qwen_response.json", {"error": {"reasoning_content": "ERROR_REASONING_POISON"}})
    write_json(relay / "attempt_00003_master_qwen_response.json", {"choices": "MALFORMED_POISON"})
    report = MODULE.project_evidence(root)
    assert "POISON" not in json.dumps(report)
    assert report["coverage"]["expected_error_only_responses"] == 2
    assert report["coverage"]["issues"]["invalid_delivered_response"] == 1
    assert report["event_timeline"][0]["time"] is None
    assert report["by_role"][0]["expected_error_only_response_count"] == 2


# ----- 2026-09-30: delivery, check-in snapshots, contract, definitions (schema 2) -----

def lifecycle(root, rows, noise=()):
    lines = [*noise, *("Agent Company lifecycle: " + json.dumps(row) for row in rows)]
    (root / "team/qwen_stderr.txt").write_text("\n".join(lines) + "\n")


def queued(identifier, sender, recipient, at):
    return {"event": "team_message_queued", "message_id": identifier, "from": sender, "to": recipient,
            "at": at, "chars": 12}


def items(*pairs):
    return [{"message_id": identifier, "waited_ms": waited, **({"mode": mode} if mode else {})}
            for identifier, waited, mode in pairs]


DELIVERY_RUN = [
    {"event": "compatibility_loaded", "team_message_delivery": "between_tool_calls", "at": 1},
    {"event": "compatibility_loaded", "team_message_delivery": "between_tool_calls", "at": 2},  # CLI relaunch
    queued("msg-1", "leader", "alpha", 1000), queued("msg-2", "beta", "alpha", 1100),
    queued("msg-3", "leader", "beta", 1200),
    {"event": "team_message_drained", "mode": "between_tool_calls", "at": 1500,
     "items": items(("msg-1", 500, None), ("msg-2", 400, None))},
    {"event": "team_message_observed", "at": 1600, "items": items(("msg-1", 600, "between_tool_calls"))},
    {"event": "teammate_inputs_requeued", "message_ids": ["msg-2"], "task_ids": [], "at": 1700},
    {"event": "team_message_drained", "mode": "idle_flush", "at": 1800, "items": items(("msg-2", 700, None))},
    {"event": "team_message_observed", "at": 1900, "items": items(("msg-2", 800, "idle_flush"))},
    {"event": "task_notice_queued", "task_id": "4", "at": 1950},
    {"event": "shutdown_notice_drained", "at": 2000},
    {"event": "checkin_enabled", "interval_s": 1800},
    {"event": "checkin_sent", "checkin_id": "checkin-1", "reason": "periodic", "snapshot": {
        "sample_started_at": "2026-09-30T05:58:30.000Z", "at": "2026-09-30T06:00:00.000Z",
        "elapsed_s": 1800, "remaining_s": 19800, "unassigned_open_tasks": 1,
        "delivery": {"commits_since_launch": 3, "last_commit_ago_s": 120, "last_commit_subject": "SUBJECT_POISON"},
        "workers": [{"name": "alpha", "status": "running", "shutdown_pending": False, "last_activity_ago_s": 5,
                     "queued_messages": 0, "open_tasks": 1, "pending_tools": ["run_shell_command", "bad tool!"],
                     "tasks": [{"subject": "TASK_SUBJECT_POISON"}],
                     "git": {"commits_since_worker_creation": 4, "commits_not_in_delivery_history": 2,
                             "uncommitted_paths": 1, "last_commit_ago_s": 60, "head": "HEAD_SHA_POISON"}}],
        "unknown": ["relay"]}},
    {"event": "checkin_observed", "checkin_id": "checkin-1"},
    {"event": "checkin_error", "error": "ERROR_TEXT_POISON"},
]


def test_delivery_receipt_is_counted_once_per_message_and_only_metadata_is_read(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [])
    lifecycle(root, DELIVERY_RUN, noise=["native trace STDERR_POISON sk-secret123456789",
                                         "Agent Company lifecycle: {not json"])
    report = MODULE.project_evidence(root)
    assert "POISON" not in json.dumps(report) and "sk-secret" not in json.dumps(report)
    assert report["schema_version"] == 5
    delivery = report["delivery"]
    assert delivery["mode"] == "between_tool_calls" and delivery["status"] == "recorded"
    # msg-2 was drained twice (requeued) and still counts once.
    assert (delivery["queued_messages"], delivery["observed_messages"], delivery["never_observed_messages"]) == (3, 2, 1)
    assert delivery["receipt_rate"] == pytest.approx(2 / 3)
    assert delivery["by_recipient"] == {"alpha": {"queued": 2, "observed": 2}, "beta": {"queued": 1, "observed": 0}}
    second = delivery["messages"][1]
    assert second["drained_modes"] == ["between_tool_calls", "idle_flush"] and second["requeued"] == 1
    assert second["observed"] == {"at_ms": 1900, "mode": "idle_flush", "queue_to_observed_ms": 800}
    assert delivery["messages"][2]["observed"] is None
    assert delivery["queue_to_observed_ms"] == {"count": 2, "median": 700, "max": 800}
    assert delivery["task_notices"] == {"queued": 1, "observed_events": 0}
    assert delivery["shutdown_notices"] == {"drained": 1, "observed": 0}
    assert report["coverage"]["lifecycle"] == {"stderr_present": True, "lifecycle_events": len(DELIVERY_RUN),
                                               "invalid_lifecycle_lines": 1}


def test_checkin_snapshot_keeps_counts_states_and_tool_names_only(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [])
    lifecycle(root, DELIVERY_RUN)
    checkins = MODULE.project_evidence(root)["checkins"]
    assert (checkins["sent"], checkins["observed"], checkins["errors"]) == (1, 1, 1)
    assert checkins["snapshots_status"] == "recorded_for_sent_checkins"
    snapshot = checkins["snapshots"][0]
    assert snapshot["delivery"] == {"commits_since_launch": 3, "last_commit_ago_s": 120}
    assert snapshot["workers"] == [{"name": "alpha", "status": "running", "shutdown_pending": False,
                                    "last_activity_ago_s": 5, "queued_messages": 0, "open_tasks": 1,
                                    "pending_tools": ["run_shell_command"],
                                    "git": {"commits_since_worker_creation": 4, "commits_not_in_delivery_history": 2,
                                            "uncommitted_paths": 1, "last_commit_ago_s": 60}}]
    # Ages are measured from the sample's start, which is kept beside its finish.
    assert snapshot["sample_started_at"] == "2026-09-30T05:58:30.000Z" and snapshot["at"] == "2026-09-30T06:00:00.000Z"
    assert snapshot["unknown_sources"] == 1
    definitions = MODULE.project_evidence(root)["definitions"]
    assert "measured from it" in definitions["checkin_snapshot_times"]
    assert "closing phase may have no record" in definitions["checkin_snapshot_coverage"]


@pytest.mark.parametrize("rows, mode, reason", [
    (None, "unknown", "no delivery-mode record"),
    ([{"event": "compatibility_loaded", "team_message_delivery": "native_idle"}], "native_idle", "native_idle runs"),
    ([{"event": "compatibility_loaded", "root_owner": "master"}], "unknown", "no delivery-mode record"),
])
def test_runs_without_delivery_records_are_unknown_not_zero(tmp_path, rows, mode, reason):
    root, relay = fixture_run(tmp_path)
    events(relay, [])
    if rows is not None:
        lifecycle(root, rows)
    report = MODULE.project_evidence(root)
    assert report["delivery"]["mode"] == mode and report["delivery"]["status"] == "not_recorded"
    assert report["delivery"]["receipt_rate"] is None and reason in report["delivery"]["reason"]
    assert report["checkins"]["status"] == "not_enabled_or_not_recorded"
    assert report["coverage"]["lifecycle"]["stderr_present"] is (rows is not None)


@pytest.mark.parametrize("rows, expected", [
    (None, "unknown"),
    ([], "unknown"),
    ([{"event": "compatibility_loaded", "native_version": "0.23.1"}], "unknown"),
    ([{"event": "compatibility_loaded", "teammate_limit": None}], "unknown"),
    ([{"event": "unrelated", "teammate_limit": "unlimited"}], "unknown"),
    ([{"event": "compatibility_loaded", "teammate_limit": "unlimited"}] * 2, "unlimited"),
    ([{"event": "compatibility_loaded", "teammate_limit": 10}] * 2, 10),
    ([{"event": "compatibility_loaded", "teammate_limit": True}], "conflicting"),
    ([{"event": "compatibility_loaded", "teammate_limit": 10},
      {"event": "compatibility_loaded", "teammate_limit": "unlimited"}], "conflicting"),
    ([{"event": "compatibility_loaded", "teammate_limit": "unlimited"},
      {"event": "compatibility_loaded"}], "conflicting"),
    ([{"event": "compatibility_loaded", "teammate_limit": 10}], 10),
    ([{"event": "compatibility_loaded", "teammate_limit": {"content": "LIMIT_POISON"}}], "conflicting"),
])
def test_teammate_limit_comes_only_from_explicit_compatibility_records(tmp_path, rows, expected):
    root, relay = fixture_run(tmp_path)
    events(relay, [])
    # Neither a launch file nor a current target setting proves an older run's actual setting.
    write_json(root / "team/launch.json", {"teammate_limit": "unlimited"})
    if rows is not None:
        lifecycle(root, rows)
    report = MODULE.project_evidence(root)
    assert report["runtime_rules"]["effective"]["teammate_limit"] == expected
    assert "POISON" not in json.dumps(report, allow_nan=False)
    assert "not shared CPU, memory or the task deadline" in report["definitions"]["teammate_limit"]


def test_checkins_recorded_before_snapshots_existed_are_marked_so(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [])
    lifecycle(root, [{"event": "checkin_enabled"}, {"event": "checkin_sent", "checkin_id": "checkin-1"}])
    checkins = MODULE.project_evidence(root)["checkins"]
    assert checkins["sent"] == 1 and checkins["snapshots"] == []
    assert checkins["snapshots_status"] == "not_recorded_before_2026_09_30"


def test_sleep_commands_and_last_edit_proposal_time_are_proposals_not_execution(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [{"event": "provider_attempt_started", "attempt": 1, "role": "worker", "time": 100},
                   {"event": "provider_attempt_started", "attempt": 2, "role": "worker", "time": 250}])
    response(relay / "attempt_00001_worker_qwen_response.json", [
        call("run_shell_command", {"command": "sleep 60 && git status"}, "a"),
        call("write_file", {"content": "CODE_POISON"}, "b")])
    response(relay / "attempt_00002_worker_qwen_response.json", [
        call("run_shell_command", {"command": "echo asleep; ./sleepy --sleep 3"}, "c"),
        call("edit", {"content": "CODE_POISON"}, "d")])
    row = MODULE.project_evidence(root)["by_role"][0]
    assert row["sleep_commands_issued"] == 1
    assert row["last_edit_proposal_attempt_started_at"] == 250


def test_submission_contract_and_definitions_match_the_export_code(tmp_path):
    from agent_company.programbench import runner as programbench_runner
    from agent_company.programbench import workspace as programbench_workspace

    root, relay = fixture_run(tmp_path)
    events(relay, [])
    report = MODULE.project_evidence(root)
    contract = report["submission_contract"]
    export = programbench_workspace._EXPORT
    assert contract["deliverable_root"] == "/workspace/solution" and "pathlib.Path('/workspace/solution')" in export
    facts = " ".join(contract["facts"])
    assert "p.name == '.git'" in export and ".git" in facts
    assert "rel.parts[0] in ('executable', 'reference')" in export and "executable and reference" in facts
    assert "Submission links and special files are unsupported" in export and "symbolic link" in facts
    runner = Path(programbench_runner.__file__).read_text()
    assert runner.index("workspace.finish_agent()") < runner.index("workspace.export_source(")
    assert "stopped before export" in facts
    assert "second" not in facts.lower()  # No build-time claim: it is an observation, not a rule.
    assert {"attempt_started_at", "message_observed", "receipt_rate", "commits_not_in_delivery_history",
            "sleep_commands_issued"} <= set(report["definitions"])


@pytest.mark.parametrize("values, median", [([600, 800], 700), ([3, None, 1, 2], 2), ([5], 5), ([], None)])
def test_wait_summary_uses_the_standard_median(values, median):
    # Review 2026-10-01: an even sample returned the upper middle value.
    assert MODULE._summary(values)["median"] == median


def test_enabled_checkins_with_none_sent_are_not_called_recorded(tmp_path):
    root, relay = fixture_run(tmp_path)
    events(relay, [])
    lifecycle(root, [{"event": "checkin_enabled"}])
    checkins = MODULE.project_evidence(root)["checkins"]
    assert checkins["sent"] == 0 and checkins["snapshots_status"] == "none_sent"



def test_final_export_counts_are_projected_with_their_scope(tmp_path):
    root = tmp_path / "trial"
    (root / "team/relay").mkdir(parents=True)
    write_json(root / "task_result.json", {"usage": usage(), "metrics": {"submission": {
        "path": "/secret/run/submission.tar.gz", "files": 412, "source_bytes": 1048576,
        "excluded_paths": [".git"]}}})
    final = MODULE.project_evidence(root)["task_result"]["final_export"]
    assert final == {"files": 412, "source_bytes": 1048576, "scope": "final_export_only"}
    assert "final_export" in MODULE.project_evidence(root)["definitions"]
    write_json(root / "task_result.json", {"usage": usage(), "metrics": {}})
    assert MODULE.project_evidence(root)["task_result"]["final_export"] is None  # not recorded, not zero
