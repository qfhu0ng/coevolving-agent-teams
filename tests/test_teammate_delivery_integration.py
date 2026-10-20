"""Actual Qwen: a busy teammate receives messages at its next model step (2026-09-30).

Only model responses are scripted; the installed CLI runs the team, tools and
queues. The control run shows native behavior: the same message waits until the
teammate's round ends.
"""
import json

from agent_company.analysis.evidence_projection import project_evidence
from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, TokenRates
from delivery_scenario import BusyWorkerScenario, text
from test_rejection_integration import native_qwen
from test_team import workspace


def run_scenario(tmp_path, between):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    scenario = BusyWorkerScenario(between)
    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario,
                            team_message_delivery="between_tool_calls" if between else "native_idle").run(
        ws, "Run the delivery scenario with one teammate.", "Assign one worker.", "Message only on decisions.",
        output, ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35,
                               rates=TokenRates("1", "0.1", "2")), timeout_seconds=120)
    stderr = (output / "qwen_stderr.txt").read_text()
    events = [json.loads(line.split("Agent Company lifecycle: ", 1)[1]) for line in stderr.splitlines()
              if line.startswith("Agent Company lifecycle: ")]
    return scenario, result, events, output


def test_busy_teammate_receives_message_task_and_shutdown_notice_mid_round(tmp_path):
    scenario, result, events, output = run_scenario(tmp_path, between=True)
    assert result["execution"]["returncode"] == 0, (result, (output / "qwen_stderr.txt").read_text()[-3000:])
    assert scenario.checks == ["mid_round_delivery", "shutdown_notice", "shutdown_prompt"], scenario.checks
    # Exactly once: the message appears in one delivery, not again at idle.
    final_history = json.dumps(scenario.worker_payloads[-1]["messages"])
    assert final_history.count(scenario.nonce) == 1
    drained = [event for event in events if event["event"] == "team_message_drained"]
    assert [event["mode"] for event in drained] == ["between_tool_calls"]
    assert drained[0]["items"][0]["from"] == "leader" and drained[0]["items"][0]["waited_ms"] is not None
    # Received, not just drained: the worker's next model request carried it.
    observed = [event for event in events if event["event"] == "team_message_observed"]
    assert [(event["worker"], event["count"]) for event in observed] == [("busy", 1)]
    kinds = [event["event"] for event in events]
    assert "teammate_delivery_patch_applied" in kinds and "task_notice_drained" in kinds
    assert kinds.count("task_notice_observed") == 1 and "teammate_inputs_requeued" not in kinds
    assert kinds.count("shutdown_notice_drained") == 1
    launch = json.loads((output / "launch.json").read_text())
    assert launch["team_message_delivery"] == "between_tool_calls"
    # The training-evidence projection reads these real lifecycle records.
    trial = tmp_path / "trial"
    (trial / "team/relay").mkdir(parents=True)
    (trial / "team/qwen_stderr.txt").write_text((output / "qwen_stderr.txt").read_text())
    delivery = project_evidence(trial)["delivery"]
    assert delivery["mode"] == "between_tool_calls" and delivery["status"] == "recorded"
    assert (delivery["queued_messages"], delivery["observed_messages"], delivery["receipt_rate"]) == (1, 1, 1.0)
    assert delivery["by_recipient"] == {"busy": {"queued": 1, "observed": 1}}
    assert delivery["messages"][0]["from"] == "leader" and delivery["messages"][0]["observed"]["mode"] == "between_tool_calls"
    assert delivery["task_notices"]["observed_events"] == 1 and delivery["shutdown_notices"] == {"drained": 1, "observed": 1}
    assert launch["environment"]["AGENT_COMPANY_TEAM_MESSAGE_DELIVERY"] == "between_tool_calls"
    worker_system = "\n".join(text(m) for m in scenario.worker_payloads[0]["messages"] if m.get("role") == "system")
    assert "Runtime note: messages from the leader and teammates are delivered at your next model step" in worker_system


def test_native_mode_control_waits_until_the_round_ends(tmp_path):
    scenario, result, events, output = run_scenario(tmp_path, between=False)
    assert result["execution"]["returncode"] == 0, (result, (output / "qwen_stderr.txt").read_text()[-3000:])
    # Native: nothing mid-round; at round end the queued task prompt takes a whole
    # round, then shutdown (highest priority) wins, and the leader's message is
    # never delivered: the starvation observed in the 2026-09-30 FFmpeg run.
    assert scenario.checks == ["native_waited", "native_task_round", "shutdown_prompt"], scenario.checks
    assert all(scenario.nonce not in json.dumps(p["messages"]) for p in scenario.worker_payloads)
    kinds = [event["event"] for event in events]
    assert "teammate_delivery_patch_applied" not in kinds and "team_message_drained" not in kinds
    launch = json.loads((output / "launch.json").read_text())
    assert launch["team_message_delivery"] == "native_idle"
    assert "AGENT_COMPANY_TEAM_MESSAGE_DELIVERY" not in launch["environment"]
    worker_system = "\n".join(text(m) for m in scenario.worker_payloads[0]["messages"] if m.get("role") == "system")
    assert "Runtime note: messages from the leader" not in worker_system
