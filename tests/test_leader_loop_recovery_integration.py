"""Actual Qwen: the master's repeated status-file read (2026-10-02).

Only model responses are scripted; the installed CLI runs the team, the loop
guard, the native wait and the wake-up. The control run shows native behavior:
the guard on the master ends the whole run (the P1 FFmpeg incident).
"""
import json
from pathlib import Path

import pytest

from agent_company.analysis.evidence_projection import project_evidence
from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig
from leader_loop_scenario import StatusLoopScenario, text, unpaired_tool_calls
from tool_call_cap_scenario import CallVolumeScenario
from test_rejection_integration import native_qwen
from test_team import workspace


def run_scenario(tmp_path, variant="status", *, recovery=True, disclosure=False, worker_seconds=8.0,
                 checkin_seconds=3600, timeout_seconds=150):
    cli, node = native_qwen()
    scenario = StatusLoopScenario(variant, worker_seconds=worker_seconds)
    output = tmp_path / "run"
    # By default a long check-in interval keeps check-ins out of the measured wait.
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario, checkin_interval_seconds=checkin_seconds,
                            runtime_disclosure=disclosure, leader_loop_recovery=recovery).run(
        workspace(tmp_path), "Run the build with one teammate.", "Assign one worker.", "Message only on decisions.",
        output, ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35),
        timeout_seconds=timeout_seconds)
    stderr = (output / "qwen_stderr.txt").read_text()
    events = [json.loads(line.split("Agent Company lifecycle: ", 1)[1]) for line in stderr.splitlines()
              if line.startswith("Agent Company lifecycle: ")]
    return scenario, result, events, output


def kinds(events):
    # The CLI's bootstrap parent loads the same imports, so patch records repeat.
    return [event["event"] for event in events
            if event["event"].startswith("leader_loop") and event["event"] != "leader_loop_patch_applied"]


def patched(events):
    return {event["part"] for event in events if event["event"] == "leader_loop_patch_applied"}


def projection(tmp_path, output):
    trial = tmp_path / "trial"
    (trial / "team/relay").mkdir(parents=True)
    (trial / "team/qwen_stderr.txt").write_text((output / "qwen_stderr.txt").read_text())
    return project_evidence(trial)["runtime_rules"]


def test_status_read_loop_ends_only_the_master_turn_and_waits_for_the_worker(tmp_path):
    scenario, result, events, output = run_scenario(tmp_path)
    stderr = (output / "qwen_stderr.txt").read_text()
    assert result["stop_reason"] == "native_cli_completed", (result["stop_reason"], stderr[-3000:])
    assert scenario.worker_done.is_set() and scenario.shutdown_sent
    # Five reads were proposed; the fifth was stopped before execution.
    assert len(scenario.read_ids) == 5
    resumed = scenario.resumed
    assert resumed is not None, "the master was not resumed with the recovery notice"
    history = json.dumps(resumed["messages"])
    assert all(identifier in history for identifier in scenario.read_ids[:4])
    assert scenario.read_ids[4] not in history and unpaired_tool_calls(resumed) == []
    # No model call during the wait: the next master request is the resumed one,
    # sent after the worker finished (the build sleeps 8 s).
    times = [at for at, _ in scenario.master_requests]
    blocked = scenario.read_times[4]
    after = [at for at in times if at > blocked]
    assert scenario.master_requests[times.index(after[0])][1] is resumed
    assert after[0] - blocked > 4
    latest = text(resumed["messages"][-1])
    assert latest.index("Runtime notice [leader-loop-1]:") < latest.index("ALPHA_DONE")
    assert patched(events) == {"detect", "runner"}
    assert kinds(events) == ["leader_loop_guard", "leader_loop_recovered", "leader_loop_resumed",
                             "leader_loop_notice_observed"]
    guard = next(event for event in events if event["event"] == "leader_loop_guard")
    assert guard["loop_type"] == "consecutive_identical_tool_calls" and guard["recoverable"] is True
    assert guard["status_file_target"] is True and guard["response_calls"] == 1
    assert "file_path" not in json.dumps(guard)  # Arguments are never recorded.
    recovered = next(event for event in events if event["event"] == "leader_loop_recovered")
    assert recovered["running"] == 1
    resumed_event = next(event for event in events if event["event"] == "leader_loop_resumed")
    assert resumed_event["checkin"] is False and resumed_event["waited_s"] >= 4
    loaded = next(event for event in events if event["event"] == "compatibility_loaded")
    assert loaded["leader_loop_recovery"] is True and loaded["runtime_disclosure"] is False
    launch = json.loads((output / "launch.json").read_text())
    assert launch["leader_loop_recovery"] is True and launch["runtime_disclosure"] is False
    assert launch["environment"]["AGENT_COMPANY_LEADER_LOOP_RECOVERY"] == "1"
    assert "AGENT_COMPANY_RUNTIME_DISCLOSURE" not in launch["environment"]
    # Recovery alone injects no disclosure text.
    master_system = "\n".join(text(m) for m in scenario.master_requests[0][1]["messages"] if m.get("role") == "system")
    assert "Runtime rules" not in master_system
    rules = projection(tmp_path, output)
    assert rules["effective"] == {"runtime_disclosure": False, "leader_loop_recovery": True,
                                  "tool_call_cap": "native_default", "teammate_limit": 10}
    assert rules["leader_loop"]["recovered"] == 1 and rules["leader_loop"]["resumed"] == 1
    assert rules["leader_loop"]["unresolved"] == 0 and rules["leader_loop"]["notice_observed"] == 1


def test_control_without_recovery_keeps_the_native_whole_run_exit(tmp_path):
    scenario, result, events, output = run_scenario(tmp_path, recovery=False)
    assert result["stop_reason"] == "cli_error"
    assert len(scenario.read_ids) == 5 and scenario.resumed is None and not scenario.shutdown_sent
    assert kinds(events) == [] and patched(events) == set()
    loaded = next(event for event in events if event["event"] == "compatibility_loaded")
    assert loaded["leader_loop_recovery"] is False
    launch = json.loads((output / "launch.json").read_text())
    assert launch["leader_loop_recovery"] is False and "AGENT_COMPANY_LEADER_LOOP_RECOVERY" not in launch["environment"]
    assert projection(tmp_path, output)["leader_loop"] is None


@pytest.mark.parametrize("variant, reason, running", [
    ("other", "not_status_file_read", 1),
    ("mixed", "other_calls_in_response", 1),
])
def test_other_reads_and_mixed_responses_keep_native_failure(tmp_path, variant, reason, running):
    scenario, result, events, output = run_scenario(tmp_path, variant)
    assert result["stop_reason"] == "cli_error"
    assert len(scenario.read_ids) == 5 and scenario.resumed is None
    assert kinds(events) == ["leader_loop_guard"]
    guard = next(event for event in events if event["event"] == "leader_loop_guard")
    assert guard["recoverable"] is False and guard["reason"] == reason
    assert guard["response_calls"] == (2 if variant == "mixed" else 1)


def test_no_running_teammate_keeps_native_failure(tmp_path):
    scenario, result, events, output = run_scenario(tmp_path, "idle")
    assert result["stop_reason"] == "cli_error"
    # The worker's report arrives with the first read's result and restarts the counts.
    assert scenario.resumed is None and len(scenario.read_ids) == 6
    assert kinds(events) == ["leader_loop_guard", "leader_loop_not_recovered"]
    refused = next(event for event in events if event["event"] == "leader_loop_not_recovered")
    assert refused["reason"] == "no_running_teammate" and refused["running"] == 0 and refused["idle"] == 1


def test_disclosure_only_states_rules_without_changing_the_guard(tmp_path):
    scenario, result, events, output = run_scenario(tmp_path, recovery=False, disclosure=True)
    assert result["stop_reason"] == "cli_error" and kinds(events) == [] and patched(events) == set()
    worker_system = "\n".join(text(m) for m in scenario.worker_payloads[0]["messages"] if m.get("role") == "system")
    assert "Eight consecutive tool calls that only inspect Git" in worker_system
    assert "clears the error; otherwise you are marked failed and the leader is notified" in worker_system
    master_system = "\n".join(text(m) for m in scenario.master_requests[0][1]["messages"] if m.get("role") == "system")
    assert "Runtime rules (Qwen CLI loop guards, always on)" in master_system
    assert "ends the whole run and stops every teammate" in master_system and "Exception in this run" not in master_system
    assert "about every 60 minutes" in master_system and "refreshed about every 60 seconds" in master_system
    # The status file the master read carried the runtime note.
    reads = [m for _, payload in scenario.master_requests for m in payload["messages"] if m.get("role") == "tool"]
    assert any("runtime_note" in text(message) for message in reads)
    loaded = next(event for event in events if event["event"] == "compatibility_loaded")
    assert loaded["runtime_disclosure"] is True and loaded["leader_loop_recovery"] is False
    assert projection(tmp_path, output)["effective"] == {"runtime_disclosure": True, "leader_loop_recovery": False,
                                                         "tool_call_cap": "native_default", "teammate_limit": 10}


def test_check_in_wakes_a_recovered_master_and_a_second_loop_recovers_again(tmp_path):
    # The worker outlasts several 6-second check-ins. After the first wake-up the
    # master polls again: the guard state of the first stop must not carry over.
    scenario, result, events, output = run_scenario(tmp_path, "again", worker_seconds=25, checkin_seconds=6)
    stderr = (output / "qwen_stderr.txt").read_text()
    assert result["stop_reason"] == "native_cli_completed", (result["stop_reason"], stderr[-3000:])
    assert scenario.notices == ["leader-loop-1", "leader-loop-2"] and scenario.shutdown_sent
    assert kinds(events).count("leader_loop_recovered") == 2 and "leader_loop_unresolved" not in kinds(events)
    resumed = [event for event in events if event["event"] == "leader_loop_resumed"]
    assert [event["recovery_id"] for event in resumed] == ["leader-loop-1", "leader-loop-2"]
    assert resumed[0]["checkin"] is True  # The worker was still building.
    for _, payload in scenario.master_requests:
        assert unpaired_tool_calls(payload) == []
    rules = projection(tmp_path, output)["leader_loop"]
    assert rules["recovered"] == 2 and rules["resumed"] == 2 and rules["resumed_by_checkin"] >= 1


def test_task_deadline_still_ends_a_recovered_wait(tmp_path):
    scenario, result, events, output = run_scenario(tmp_path, worker_seconds=120, timeout_seconds=30)
    # Recovered, woken by the half-time check-in, waiting again when the deadline ends the run.
    assert kinds(events)[:2] == ["leader_loop_guard", "leader_loop_recovered"] and not scenario.worker_done.is_set()
    assert result["stop_reason"] == "task_timeout" and result["elapsed_seconds"] < 45
    assert "leader_loop_unresolved" not in kinds(events)


# ----- per-turn tool-call cap: native default vs disabled (all roles) ---------

def run_volume(tmp_path, mode, cap):
    cli, node = native_qwen()
    scenario = CallVolumeScenario(mode)
    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario, tool_call_cap=cap).run(
        workspace(tmp_path), "Run the checks.", "Assign one worker.", "Message only on decisions.", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35),
        timeout_seconds=150, execution_mode="team" if mode == "worker" else "single")
    stderr = (output / "qwen_stderr.txt").read_text()
    events = [json.loads(line.split("Agent Company lifecycle: ", 1)[1]) for line in stderr.splitlines()
              if line.startswith("Agent Company lifecycle: ")]
    launch = json.loads((output / "launch.json").read_text())
    settings = json.loads((Path(launch["runtime_dir"]) / "profile/settings.json").read_text())
    return scenario, result, events, output, launch, settings


@pytest.mark.parametrize("mode", ["single", "worker"])
def test_native_default_cap_stops_a_long_turn_with_a_recurring_call(tmp_path, mode):
    scenario, result, events, output, launch, settings = run_volume(tmp_path, mode, "native_default")
    # Adaptive default: the key is absent (never null, which native reads as a hard cap of 100).
    assert "maxToolCallsPerTurn" not in settings["model"] and launch["native_max_tool_calls_per_turn"] is None
    assert launch["tool_call_cap"] == "native_default" and "AGENT_COMPANY_TOOL_CALL_CAP" not in launch["environment"]
    assert not scenario.finished.is_set() and scenario.batches == 26  # Call 101 is in the 26th response.
    if mode == "single":
        assert result["stop_reason"] == "cli_error"
        assert "turn_tool_call_cap" in (output / "qwen_stdout.jsonl").read_text()
    else:
        failures = [event for event in events if event["event"] == "failure_notice_queued"]
        assert [event["worker"] for event in failures] == ["alpha"]
        assert "turn_tool_call_cap" in failures[0]["error"]


@pytest.mark.parametrize("mode", ["single", "worker"])
def test_disabled_cap_lets_the_same_turn_finish(tmp_path, mode):
    scenario, result, events, output, launch, settings = run_volume(tmp_path, mode, "disabled")
    assert settings["model"]["maxToolCallsPerTurn"] == 0 and launch["native_max_tool_calls_per_turn"] == 0
    assert launch["environment"]["AGENT_COMPANY_TOOL_CALL_CAP"] == "disabled"
    assert "no per-turn tool-call cap" in launch["adaptations"]
    assert scenario.finished.is_set() and scenario.batches == CallVolumeScenario.RESPONSES
    assert result["stop_reason"] == "native_cli_completed", (output / "qwen_stderr.txt").read_text()[-3000:]
    assert not any(event["event"] == "failure_notice_queued" for event in events)
    loaded = [event for event in events if event["event"] == "compatibility_loaded"]
    assert loaded and all(event["tool_call_cap"] == "disabled" for event in loaded)
    trial = tmp_path / "trial"
    (trial / "team/relay").mkdir(parents=True)
    (trial / "team/qwen_stderr.txt").write_text((output / "qwen_stderr.txt").read_text())
    assert project_evidence(trial)["runtime_rules"]["effective"]["tool_call_cap"] == "disabled"


def test_consecutive_identical_guard_still_applies_without_the_cap(tmp_path):
    # The user kept this guard (2026-10-02): only the per-turn count cap is removed.
    cli, node = native_qwen()
    scenario = StatusLoopScenario("status")
    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario, checkin_interval_seconds=3600,
                            tool_call_cap="disabled").run(
        workspace(tmp_path), "Run the build with one teammate.", "Assign one worker.", "Message only on decisions.",
        output, ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35),
        timeout_seconds=150)
    assert result["stop_reason"] == "cli_error" and len(scenario.read_ids) == 5 and scenario.resumed is None
