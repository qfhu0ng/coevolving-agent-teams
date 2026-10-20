"""Actual Qwen CLI with role policies (S2c). Only model responses are scripted.

Specification: docs/stage1_role_policy_harness_plan_20261009.md v3.3; D10 option C.
"""
import json
from pathlib import Path
import subprocess
import threading

import pytest

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, role_of
from agent_company.policy.role_policy import RolePolicyStore
from agent_company.policy import role_policy_runtime as runtime
from role_policy_scenario import LIMIT_WORKERS, TURN_NOTICE, RolePolicyScenario, system, text
from test_leader_loop_recovery import native_source, node
from test_rejection_integration import native_qwen
from test_team import workspace

ROOT = Path(__file__).parents[1]
PATCH = ROOT / "agent_company/runtime/qwen_native_patch.mjs"
COMPAT = ROOT / "agent_company/runtime/qwen_native_compat.mjs"


def run(tmp_path, variant, mode):
    """Run one scenario; every scenario must end by itself, not by the wall-time limit."""
    cli, node_path = native_qwen()
    scenario = RolePolicyScenario(variant)
    output = tmp_path / "run"
    scenario.native_stderr = output / "qwen_stderr.txt"
    instruction = ("Run scripted directory checks with twelve teammates." if variant == "spawn_active_limit"
                   else "Run the scripted build with one teammate.")
    assignment = ("Assign ten active workers and replace two only after shutdown." if variant == "spawn_active_limit"
                  else "Assign one worker.")
    result = QwenTeamRunner(str(cli), node=node_path, upstream=scenario, role_policy_mode=mode).run(
        workspace(tmp_path), instruction, assignment,
        "Starting C: message the leader on blockers.", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=35), timeout_seconds=120)
    stderr = (output / "qwen_stderr.txt").read_text()
    lifecycle = [json.loads(line.split("Agent Company lifecycle: ", 1)[1]) for line in stderr.splitlines()
                 if line.startswith("Agent Company lifecycle: ")]
    events = [json.loads(line) for line in (output / "relay/events.jsonl").read_text().splitlines()]
    assert result["stop_reason"] == "native_cli_completed", (result["stop_reason"], stderr[-3000:])
    assert result["execution"].get("timed_out") is not True and result["elapsed_seconds"] < 100
    return scenario, result, lifecycle, events, output


def changes(events):
    return [change["event"] for row in events if row["event"] == "role_policy_batch" for change in row["changes"]]


def test_proposal_approval_reaches_the_worker_and_the_text_report_is_forwarded(tmp_path):
    scenario, result, lifecycle, events, output = run(tmp_path, "proposal", "propose")
    stderr = (output / "qwen_stderr.txt").read_text()
    assert result["stop_reason"] == "native_cli_completed", (result["stop_reason"], stderr[-3000:])
    # Native forwards nonempty end-of-turn text after any leader message; not D10 evidence.
    assert scenario.master_saw("ALPHA_REPORT")
    assert [e["worker"] for e in lifecycle if e["event"] == "policy_proposal_not_report"] == ["alpha"]
    assert any(e["event"] == "policy_report_patch_applied" for e in lifecycle)
    assert scenario.decision_sent and scenario.worker_after_update is not None
    worker_system = system(scenario.worker_after_update)
    assert "Your role policy version 2" in worker_system and "Your proposal #1 was approved as version 2." in worker_system
    assert "Message the leader only for blockers" in worker_system
    first_worker = system(scenario.worker_payloads[0])
    assert "Your role slot: alpha; you are alpha. Your role policy version 1" in first_worker
    assert "Starting C: message the leader on blockers." not in system(scenario.master_payloads[0]).split(
        "Communication policy C (version 1):")[0], "the master's static prompt still carries C"
    assert changes(events)[:5] == ["role_policy_initialized", "role_policy_planned", "role_policy_committed",
                                   "role_policy_bound", "role_policy_proposed"]
    assert "role_policy_decided" in changes(events) and "role_policy_binding_ended" in changes(events)
    sent = [row for row in events if row["event"] == "role_policy_message_sent"]
    assert {row["kind"] for row in sent} >= {"ProposalCall", "DecisionCall"}
    replayed = RolePolicyStore.replay(RolePolicyStore.read_log(output / "relay/events.jsonl"), [].append)
    assert replayed.export() == json.loads((output / "relay/role_policy_state.json").read_text())
    run_summary = result["role_policy_run"]
    assert run_summary["lifecycle_tail"]["terminal_events"] >= 1 and run_summary["voided_at_run_end"] == 0
    launch = json.loads((output / "launch.json").read_text())
    assert launch["role_policy_mode"] == "propose" and launch["environment"]["AGENT_COMPANY_ROLE_POLICY_MODE"] == "propose"
    assert launch["role_policy_disclosure_version"] == runtime.DISCLOSURE_VERSION


@pytest.mark.parametrize("variant", ["proposal_silent", "ordinary_silent"])
def test_native_forwards_the_last_visible_text_after_any_leader_message(tmp_path, variant):
    """In these scenarios the later nonempty text is forwarded either way (plan section 10.1).

    This is not a patch on/off comparison and does not show the patch is never needed.

    alpha writes its report and messages the leader in one response, then gives a
    reasoning-only response. Native re-requests that text-free response; the next
    nonempty text clears the explicit-report mark and is forwarded, whether the
    earlier leader message was a proposal or an ordinary note.
    """
    scenario, result, lifecycle, events, output = run(tmp_path, variant, "propose")
    assert scenario.shutdown_sent
    assert scenario.steps["alpha"] >= 3, "native accepted a text-free final response"
    assert scenario.master_saw("Idle.")
    assert not scenario.master_saw("ALPHA_REPORT") and not scenario.master_saw(TURN_NOTICE)
    exempted = [e["worker"] for e in lifecycle if e["event"] == "policy_proposal_not_report"]
    assert exempted == (["alpha"] if variant == "proposal_silent" else [])


def test_global_policy_broadcast_reaches_the_worker_with_recipient_outcomes(tmp_path):
    scenario, result, lifecycle, events, output = run(tmp_path, "global", "global")
    stderr = (output / "qwen_stderr.txt").read_text()
    assert result["stop_reason"] == "native_cli_completed", (result["stop_reason"], stderr[-3000:])
    broadcasts = [e for e in lifecycle if e["event"] == "team_broadcast" and e["global_policy"]]
    assert broadcasts and broadcasts[0]["recipients"] == [{"name": "alpha", "outcome": "accepted"}]
    assert "global_policy_committed" in changes(events)
    assert "Communication policy C (version 2):\nReport only blockers and results." in system(scenario.worker_after_update)
    assert "Role-policy snapshot" not in system(scenario.master_payloads[-1])
    assert not any(e["event"] == "policy_report_patch_applied" for e in lifecycle)


def test_off_mode_launch_is_unchanged(tmp_path):
    scenario, result, lifecycle, events, output = run(tmp_path, "ordinary_silent", "off")
    launch = json.loads((output / "launch.json").read_text())
    assert "AGENT_COMPANY_ROLE_POLICY_MODE" not in launch["environment"] and launch["role_policy_mode"] == "off"
    assert launch["role_policy_disclosure_version"] is None
    assert "role_policy_run" not in result and not [row for row in events if row["event"].startswith("role_policy")]
    assert "Communication policy for this task:\nStarting C" in system(scenario.worker_payloads[0])
    assert not [e for e in lifecycle if e["event"] in {"team_broadcast", "policy_report_patch_applied"}]


# ----- patch facts (no CLI run) --------------------------------------------------

def patch(source, **options):
    script = (f"const m = await import({json.dumps(PATCH.as_uri())});"
              "let s = ''; for await (const c of process.stdin) s += c;"
              f"process.stdout.write(m.patchNativeSource(s, {json.dumps(options)}));")
    return subprocess.run([node(), "--input-type=module", "-e", script], input=source,
                          capture_output=True, text=True)


def test_policy_report_anchor_is_unique_and_the_patch_is_opt_in():
    script = f"const m = await import({json.dumps(PATCH.as_uri())}); console.log(JSON.stringify(m.POLICY_REPORT_ANCHOR));"
    anchor = json.loads(subprocess.run([node(), "--input-type=module", "-e", script], check=True,
                                       capture_output=True, text=True).stdout)
    owners = [name for name, source in native_source().items() if source.count(anchor) == 1]
    assert len(owners) == 1 and sum(source.count(anchor) for source in native_source().values()) == 1
    source = "class X {\n  m(sender, automatic, message) {\n" + anchor + "\n  }\n}\n"
    assert patch(source).stdout == source
    patched = patch(source, policyProposalReports=True)
    assert patched.returncode == 0 and ".policyProposalMessage(message, sender.name)" in patched.stdout
    drifted = patch(source.replace("explicitLeaderReports", "explicitReports"), policyProposalReports=True)
    assert drifted.returncode != 0 and "anchor changed" in drifted.stderr


@pytest.mark.parametrize("value", ["on", "Propose", "1"])
def test_invalid_role_policy_mode_fails_before_the_cli(value):
    cli, node_path = native_qwen()
    completed = subprocess.run([node_path, "--import", str(COMPAT), "--input-type=module", "-e", "0"],
                               env={"PATH": "/usr/bin:/bin", "AGENT_COMPANY_QWEN_CLI": str(cli),
                                    "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:9/v1",
                                    "AGENT_COMPANY_ROOT_OWNER": "master", "AGENT_COMPANY_ROLE_POLICY_MODE": value},
                               capture_output=True, text=True)
    assert completed.returncode != 0 and "Unsupported role policy mode" in completed.stderr


def test_native_team_enforces_ten_active_then_reuses_two_terminal_slots(tmp_path):
    scenario, result, lifecycle, events, output = run(tmp_path, "spawn_active_limit", "master")
    expected = set(LIMIT_WORKERS)
    assert scenario.spawn_batches == [LIMIT_WORKERS[:10], LIMIT_WORKERS[10:11], LIMIT_WORKERS[10:]]
    assert scenario.shutdown_sent and scenario.shutdown_seen == expected
    spawn_results = [row for row in events if row["event"] == "role_policy_spawn_result"]
    assert len(spawn_results) == len(expected)
    assert {row["instance"] for row in spawn_results} == expected
    assert all(row["outcome"] == "ok" for row in spawn_results), spawn_results
    assert not [row for row in events if row["event"] == "role_policy_message_rejected"]
    # Count actual worker requests, not only planned worktrees or spawn replies.
    assert {role_of(payload) for payload in scenario.worker_payloads} == expected
    directories = {name: str((tmp_path / "workers" / name).resolve()) for name in LIMIT_WORKERS}
    assert {name: str(Path(path).resolve()) for name, path in result["workers"].items()} == directories
    assert len(set(directories.values())) == len(expected)
    for name, directory in directories.items():
        tool_results = [text(message) for payload in scenario.worker_payloads if role_of(payload) == name
                        for message in payload["messages"] if message.get("role") == "tool"]
        assert any(directory in body for body in tool_results), (name, tool_results)
        assert scenario.master_saw(f"WORKER_CWD_READY:{name}:")
    descriptions = [tool["function"].get("description", "") for payload in scenario.master_payloads
                    for tool in payload.get("tools", []) if tool["function"]["name"] == "team_create"]
    assert descriptions and all("at most 10 active teammates at a time" in description for description in descriptions)
    assert all("max 10" not in description and "Infinity" not in description for description in descriptions)
    launch = json.loads((output / "launch.json").read_text())
    assert launch["teammate_limit"] == 10
    loaded = [event for event in lifecycle if event["event"] == "compatibility_loaded"]
    assert loaded and all(event["teammate_limit"] == 10 for event in loaded)
    state = json.loads((output / "relay/role_policy_state.json").read_text())
    assert all(state["bindings"][name]["lifecycle"] == "terminal" for name in LIMIT_WORKERS)
    assert result["scripted_provider"] and result["role_policy_run"]["voided_at_run_end"] == 0
    registrations = [e for e in events if e["event"] == "worker_registered"]
    assert max(e["active_teammates"] for e in registrations) == 10
    assert any("Teammate limit reached" in str(e) for e in result["errors"])


def test_active_limit_fixture_waits_for_saved_native_terminal_records(tmp_path):
    scenario = RolePolicyScenario("spawn_active_limit")
    scenario.native_stderr = tmp_path / "native_stderr.txt"
    scenario.native_stderr.write_text("", encoding="utf-8")
    records = "".join("Agent Company lifecycle: " + json.dumps({
        "event": "worker_terminal", "worker": name, "status": "cancelled"
    }) + "\n" for name in LIMIT_WORKERS[:2])
    writer = threading.Timer(0.02, scenario.native_stderr.write_text, args=(records,), kwargs={"encoding": "utf-8"})
    writer.start()
    try:
        scenario.wait_for_native_terminals(LIMIT_WORKERS[:2], timeout=1)
    finally:
        writer.join()


def test_active_limit_fixture_does_not_accept_idle_as_terminal(tmp_path):
    scenario = RolePolicyScenario("spawn_active_limit")
    scenario.native_stderr = tmp_path / "native_stderr.txt"
    scenario.native_stderr.write_text("Agent Company lifecycle: " + json.dumps({
        "event": "worker_terminal", "worker": "w0", "status": "idle"
    }) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not observe native terminal records"):
        scenario.wait_for_native_terminals(["w0"], timeout=0)


def test_native_success_texts_the_relay_relies_on_are_unchanged():
    sources = native_source()
    bundle = "\n".join(sources.values())
    send = (ROOT / "external/qwen-study/package/chunks").glob("send-message-*.js")
    send_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in send)
    assert 'const msg = `Teammate "${name}" is now running concurrently.' in bundle
    assert 'const msg = `Message sent to "${to}".`;' in send_text
    assert '"Message broadcast to all teammates."' in send_text
    assert "`Message broadcast delivered to ${reached} of ${total} recipient(s)" in send_text
    # Failed tools reach the model as the bare error message.
    assert "response: { error: error3.message }" in bundle



def test_master_text_ack_reaches_later_master_request_and_shutdown_completes(tmp_path):
    # Default switches regression. Formal first-round smoke.main coverage is
    # a separate batch and must use the profile's actual runtime switches.
    scenario, result, lifecycle, events, output = run(tmp_path, "master_set_text_ack", "master")
    assert scenario.worker_after_update is not None and scenario.master_saw("ALPHA_ACK")
    assert scenario.shutdown_sent and scenario.shutdown_seen == {"alpha"}
    assert scenario.master_saw("shutdown_approved")
    assert any(e["event"] == "compatibility_loaded" and "automatic_report_final_drain" in e["repairs"]
               for e in lifecycle)
