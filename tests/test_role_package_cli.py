"""Real native CLI, local tools and scripted replies for the new package path.

Every request and the native run evidence are retained in the pytest base dir.
These fixtures prove delivery and state transitions, not model compliance.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path
import re
import threading

import pytest

from agent_company.policy.role_package import create_p0_package
from agent_company.policy.role_package_runtime import RP_BINDING_MASTER
from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.collaboration_text import DELIVERY_PROTOCOL
from agent_company.runtime.relay import ProviderConfig, role_of
from basic_smoke_scenario import BasicSmokeScenario
from role_policy_scenario import call, reply, system, text, user_texts
from test_rejection_integration import native_qwen
from test_team import workspace


RULE = "Include the checked revision when handing work to another role."
STOP_HOOK = json.loads((Path(__file__).parents[1] / 'configs/stage1/role_series_pandoc_batch1.template.json').read_text())['inference']['stop_hook']
EARLY_FINAL = 'PACKAGE_EARLY_FINAL_BEFORE_WORK'


class PackageScenario:
    def __init__(self, mode, early_stop=False):
        self.mode = mode
        self.early_stop = early_stop
        self.steps = defaultdict(int)
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
        self.approved = self.beta_spawned = self.shutdown = False
        self.shutdown_seen = set()
        self.receipt_id = None
        self.resolved = self.acknowledged = False

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            self.requests[role].append(copy.deepcopy(payload))
            assert sum(self.steps.values()) < 70, "Bounded native fixture failed to converge"
        return self.master(payload, step) if role == "master" else self.worker(payload, step)

    def master(self, payload, step):
        if self.early_stop:
            if step == 0:
                return reply(payload, content=EARLY_FINAL)
            step -= 1
        if step == 0:
            return reply(payload, [call("team_create", {"team_name": "package-smoke"})])
        if step == 1:
            return reply(payload, [call("task_create", {"subject": "Fixture handoff",
                "description": "Writing scope: original.txt", "activeForm": "Checking fixture handoff"})])
        if step == 2:
            return self.spawn(payload, "alpha")
        if step == 3:
            return reply(payload, [call("task_update", {"taskId": "1", "owner": "alpha", "status": "in_progress"})])
        serialized = json.dumps(payload["messages"])
        if self.mode == "propose" and not self.approved and "Proposal #1 for role:implementer" in serialized:
            self.approved = True
            return reply(payload, [call("send_message", {"to": "alpha", "message":
                "POLICY-DECISION #1 approve\nNOTE: The scripted handoff identifies a reusable missing check."})])
        if self.mode == "propose" and self.approved and not self.beta_spawned:
            self.beta_spawned = True
            return self.spawn(payload, "beta")
        reports = "\n".join(user_texts(payload))
        expected = ["PACKAGE_DONE_alpha"] + (["PACKAGE_DONE_beta"] if self.mode == "propose" else [])
        if not self.shutdown and all(marker in reports for marker in expected):
            self.shutdown = True
            names = ["alpha", "beta"] if self.mode == "propose" else ["alpha"]
            return reply(payload, [call("request_shutdown", {"to": name}) for name in names])
        return reply(payload, content="Team complete." if self.shutdown else "Waiting for the fixture reports.")

    @staticmethod
    def spawn(payload, name):
        return reply(payload, [call("agent", {"name": name, "subagent_type": "general-purpose",
            "description": "Fixture handoff", "prompt": "ROLE-TYPE: implementer\nComplete the scripted handoff."})])

    def worker(self, payload, step):
        role = role_of(payload)
        if any("The team leader has requested that you shut down." in body for body in user_texts(payload)):
            self.shutdown_seen.add(role)
            return reply(payload, [call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        if role == "beta":
            assert RULE in system(payload), "Later same-role instance must inherit the committed policy"
            return reply(payload, content="PACKAGE_DONE_beta")
        if step == 0:
            record = {"op": "report", "summary": "Fixture handoff has not been rechecked.",
                      "paths": ["original.txt"], "tested_commit": None, "build_id": None,
                      "checks": "unknown", "recipient": "leader"}
            calls = [call("send_message", {"to": "leader", "message": "COORD: " + json.dumps(record)})]
            if self.mode == "propose":
                calls.append(call("send_message", {"to": "leader", "message":
                    "POLICY-PROPOSAL role:implementer append\nCHANGE: " + RULE +
                    "\nREASON: A handoff needs an identifiable checked revision.\nEVIDENCE: Reported fixture handoff."}))
            return reply(payload, calls)
        # The record ID must come from an actual native send_message result,
        # not from the summary or from peeking at controller state.
        receipts = [text(message) for message in payload["messages"] if message.get("role") == "tool"
                    and '"coord_status":"accepted"' in text(message)]
        if not self.receipt_id:
            assert receipts, "Native tool result did not carry the persisted COORD receipt"
            self.receipt_id = re.search(r'"record_id":"([^\"]+)"', receipts[0]).group(1)
        if not self.resolved:
            self.resolved = True
            record = {"op": "resolve", "issue_id": self.receipt_id, "fix_commit": "abcdef1", "checks": "unknown"}
            return reply(payload, [call("send_message", {"to": "leader", "message": "COORD: " + json.dumps(record)})])
        if not self.acknowledged:
            self.acknowledged = True
            record = {"op": "ack", "issue_id": self.receipt_id, "tested_commit": None, "checks": "unknown"}
            return reply(payload, [call("send_message", {"to": "leader", "message": "COORD: " + json.dumps(record)})])
        if self.mode == "propose" and RULE not in system(payload):
            return reply(payload, [call("run_shell_command", {"command": "pwd", "description": "Fixture directory check"})])
        return reply(payload, content="PACKAGE_DONE_alpha")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mode,early_stop", [("off", False), ("propose", False), ("propose", True)])
def test_package_native_receipts_roles_and_shutdown(tmp_path, mode, early_stop):
    cli, node = native_qwen()
    scenario = PackageScenario(mode, early_stop=early_stop)
    package = create_p0_package("Delegate the fixture handoff.", "Report concrete handoff evidence.")
    result = None
    try:
        result = QwenTeamRunner(str(cli), node=node, upstream=scenario,
                               role_policy_mode=mode, policy_package=package,
                               stop_hook=STOP_HOOK if early_stop else None,
                               checkin_interval_seconds=1800, team_message_delivery="between_tool_calls",
                               runtime_disclosure=True, leader_loop_recovery=True, tool_call_cap="disabled").run(
            workspace(tmp_path), "Complete the synthetic handoff.", package["team"]["assignment"],
            package["team"]["communication"], tmp_path / "run",
            ProviderConfig("native-scripted-model", "http://unused.invalid", request_timeout_seconds=10),
            timeout_seconds=1200 if early_stop else 90)
    finally:
        save_json(tmp_path / "requests.json", dict(scenario.requests))
        save_json(tmp_path / "result.json", result)
    assert result["stop_reason"] == "native_cli_completed", result
    assert result["execution"]["returncode"] == 0 and not result["execution"]["timed_out"]
    assert result["usage_complete"] and not result["errors"]
    assert scenario.receipt_id and scenario.resolved and scenario.acknowledged and scenario.shutdown
    expected_workers = {"alpha", "beta"} if mode == "propose" else {"alpha"}
    assert scenario.shutdown_seen == expected_workers
    master_messages = "\n".join(body for payload in scenario.requests["master"] for body in user_texts(payload))
    for worker in expected_workers:
        assert re.search(rf'<teammate_message from="{worker}">\s*shutdown_approved\s*</teammate_message>',
                         master_messages), "Shutdown approval must reach a subsequent master request"
    assert RP_BINDING_MASTER in system(scenario.requests["master"][0])
    assert all(RP_BINDING_MASTER not in system(p) for p in scenario.requests["alpha"])
    assert all(system(p).count("Fixed collaboration protocol;") == 1 for ps in scenario.requests.values() for p in ps)
    assert all(system(p).count(DELIVERY_PROTOCOL) == 1 for ps in scenario.requests.values() for p in ps)
    if early_stop:
        assert_early_stop_continued(result, scenario.requests['master'])
        assert all('Time remaining:' not in json.dumps(scenario.requests[worker]) for worker in expected_workers)
    assert not any(e.get("status") == "failed" for e in result["native_result"]["lifecycle"]["events"])
    events = [json.loads(line) for line in (tmp_path / "run/relay/events.jsonl").read_text().splitlines()]
    tasks = [event for event in events if event.get("event") == "collaboration_task_observed"]
    assert any(event.get("task_id") == "1" and event.get("paths") == ["original.txt"] for event in tasks)
    if mode == "propose":
        assert scenario.approved and scenario.beta_spawned
        assert any(RULE in system(p) for p in scenario.requests["alpha"])
        assert RULE in system(scenario.requests["beta"][0])
    else:
        assert not scenario.approved and not scenario.beta_spawned
        assert all(RULE not in system(p) for ps in scenario.requests.values() for p in ps)


@pytest.mark.parametrize('early_stop', [False, True])
def test_package_single_gets_delivery_and_closing_but_no_team_protocol(tmp_path, early_stop):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    package = create_p0_package("TEAM_ONLY_ASSIGNMENT", "TEAM_ONLY_COMMUNICATION")
    scenario = BasicSmokeScenario("single", str(tmp_path / "single-marker.txt"))
    requests = []

    def upstream(payload):
        requests.append(copy.deepcopy(payload))
        if early_stop and len(requests) == 1:
            return reply(payload, content=EARLY_FINAL)
        return scenario(payload)

    result = None
    try:
        result = QwenTeamRunner(str(cli), node=node, upstream=upstream, policy_package=package,
                               stop_hook=STOP_HOOK if early_stop else None).run(
            ws, "Complete the synthetic file roundtrip.", "", "", tmp_path / "run",
            ProviderConfig("native-scripted-model", "http://unused.invalid", request_timeout_seconds=10),
            timeout_seconds=1200 if early_stop else 90, execution_mode="single")
    finally:
        save_json(tmp_path / "requests.json", requests)
        save_json(tmp_path / "result.json", result)
    assert result["stop_reason"] == "native_cli_completed" and result["usage_complete"], result
    assert scenario.file_verified
    serialized = json.dumps(requests)
    assert "Fixed delivery protocol; policies cannot override it:" in serialized
    assert all(system(p).count(DELIVERY_PROTOCOL) == 1 for p in requests)
    if early_stop:
        assert_early_stop_continued(result, requests)
    else:
        assert "Closing window:" in serialized
    for absent in ("TEAM_ONLY_ASSIGNMENT", "TEAM_ONLY_COMMUNICATION", "Fixed collaboration protocol;",
                   "runtime_collaboration_state", "POLICY-PROPOSAL", "ROLE-TYPE:"):
        assert absent not in serialized


def assert_early_stop_continued(result, requests):
    """Real root Stop hook must deliver the approved reminder before tools run."""
    assert any(m.get('role') == 'assistant' and text(m) == EARLY_FINAL
               and not m.get('tool_calls') for m in requests[1]['messages'])
    bodies = '\n'.join(user_texts(requests[1]))
    remaining = re.findall(r'Time remaining: (\d+) seconds\.', bodies)
    assert len(remaining) == 1 and 900 < int(remaining[0]) <= 1200
    assert STOP_HOOK['reason'].format(remaining_seconds=remaining[0]) in bodies
    decisions = [e for e in result['native_result']['lifecycle']['events'] if e.get('event') == 'stop_hook_decision']
    assert decisions and decisions[0]['decision'] == 'block'
    assert any(m.get('role') == 'assistant' and m.get('tool_calls')
               for p in requests[2:] for m in p['messages'])
    assert result['elapsed_seconds'] < 120, 'Fixture must not wait for the 1200-second deadline'
