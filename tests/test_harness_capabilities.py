"""target_harness is derived from the runner's own parameters and texts, without model calls."""
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

import agent_company.runtime.qwen as qwen_module
from agent_company.evolution.harness_capabilities import (
    PROVIDER_FIELDS, STATUS_FILE, executability_summary, native_version, resolve_pointer, target_harness)
from agent_company.programbench.runner import ProgramBenchRunner
from agent_company.runtime.qwen import (
    CHECKIN_MILESTONES, WORKER_DELIVERY_NOTE, QwenTeamRunner, checkin_refresh_seconds, checkin_system,
    delivery_text, runtime_rules)
from agent_company.runtime.relay import MASTER_TOOLS, SINGLE_TOOLS, TEAM_SIZE_NOTE, WORKER_TOOLS, ProviderConfig
from agent_company.policy.role_policy_runtime import disclosure
from test_team import workspace


ROOT = Path(__file__).resolve().parents[1]
S0 = {mode: json.loads((ROOT / f"configs/stage1/programbench_ffmpeg_s0_{mode}_20261011.json").read_text())
      for mode in ("propose", "global")}
MINIMAL = {"agent_timeout_sec": 3600, "cpus": 4, "memory_mb": 8192}


def provider(**overrides):
    return ProviderConfig(**({"model": "public-model", "base_url": "http://ENDPOINT_SENTINEL.invalid/v1",
                              "api_key": "KEY_SENTINEL", "extra_body": {"private": "EXTRA_SENTINEL"},
                              "request_timeout_seconds": 1800, "first_progress_timeout_seconds": 600,
                              "progress_idle_timeout_seconds": 600, "max_attempts": 11} | overrides))


def test_s0_propose_harness_uses_the_agents_own_texts_and_no_private_values():
    inference = S0["propose"]["inference"]
    harness = target_harness(inference, execution_mode="team", provider=provider(),
                             code_version="abc123", native_version="0.23.1")
    checkin = inference["team_checkin_seconds"]
    assert harness["scope"] == {"execution_mode": "team", "role_policy_mode": "propose",
                                "code_version": "abc123", "native_version": "0.23.1"}
    assert harness["guards"]["master_rules"] == {"text": runtime_rules(
        "master", checkin_interval_seconds=checkin, leader_loop_recovery=True, tool_call_cap="disabled"),
        "shown_to_agents": True}
    assert harness["guards"]["worker_rules"]["text"] == runtime_rules("worker", tool_call_cap="disabled")
    assert harness["wakeups"]["master_text"]["text"] == checkin_system(checkin, STATUS_FILE)
    assert harness["wakeups"]["checkin_milestone_fractions"] == list(CHECKIN_MILESTONES)
    assert harness["wakeups"]["status_refresh_seconds"] == checkin_refresh_seconds(checkin)
    assert harness["messages"]["master_text"]["text"] == delivery_text("between_tool_calls")
    assert harness["messages"]["worker_text"]["text"] == WORKER_DELIVERY_NOTE
    assert harness["online_policy"]["master_disclosure"] == disclosure("propose", "master")
    assert harness["online_policy"]["worker_disclosure"] == disclosure("propose", "worker")
    assert harness["online_policy"]["assignment_policy_changes_in_run"] is False
    assert harness["roles_and_tools"]["master"]["tools"] == sorted(MASTER_TOOLS)
    assert harness["roles_and_tools"]["worker"]["tools"] == sorted(WORKER_TOOLS)
    assert set(harness["timing"]["provider"]) == set(PROVIDER_FIELDS)
    assert harness["unknown"] == []
    assert harness["lifecycle"]["teammate_limit"] == 10
    assert harness["lifecycle"]["teammate_limit_meaning"] == TEAM_SIZE_NOTE
    text = json.dumps(harness)
    for private in ("KEY_SENTINEL", "ENDPOINT_SENTINEL", "EXTRA_SENTINEL", "api_key", "base_url"):
        assert private not in text


@pytest.mark.parametrize("mode", ["off", "global", "master", "propose"])
def test_each_role_policy_mode_discloses_exactly_what_its_agents_receive(mode):
    harness = target_harness({**MINIMAL, "team_checkin_seconds": 600, "role_policy_mode": mode},
                             execution_mode="team")
    online = harness["online_policy"]
    assert online["mode"] == harness["scope"]["role_policy_mode"] == mode
    if mode == "off":
        assert "master_disclosure" not in online and "worker_disclosure" not in online
    else:
        assert online["master_disclosure"] == disclosure(mode, "master")
        assert online["worker_disclosure"] == disclosure(mode, "worker")
    succession = harness["lifecycle"]["succession"]
    assert ("SUCCEEDS" in succession) == (mode in {"master", "propose"})
    assert harness["lifecycle"]["teammate_limit"] == 10
    assert "/lifecycle/teammate_limit" not in harness["unknown"]


def test_defaults_come_from_the_runner_resolution_not_from_the_harness():
    harness = target_harness(MINIMAL, execution_mode="team")
    assert harness["messages"]["delivery_mode"] == "native_idle"
    assert harness["messages"]["master_text"]["text"] == delivery_text("native_idle")
    assert "worker_text" not in harness["messages"]
    assert harness["wakeups"]["checkin_interval_seconds"] is None
    assert harness["wakeups"]["checkin_milestone_fractions"] == []
    assert "master_text" not in harness["wakeups"] and "status_refresh_seconds" not in harness["wakeups"]
    assert harness["guards"]["tool_call_cap"] == "native_default" and harness["guards"]["leader_loop_recovery"] is False
    # Rules are listed even when a run does not show them, and say so.
    assert harness["guards"]["master_rules"] == {"text": runtime_rules("master"), "shown_to_agents": False}
    assert harness["online_policy"]["mode"] == "off"
    assert set(harness["unknown"]) == {"/scope/code_version", "/scope/native_version", "/timing/provider",
                                       "/timing/native_request_timeout_seconds"}
    assert harness["timing"]["max_model_calls"] is None
    with pytest.raises(ValueError, match="role_policy_mode"):
        target_harness({**MINIMAL, "role_policy_mode": "worker"}, execution_mode="team")
    with pytest.raises(ValueError, match="execution_mode"):
        target_harness(MINIMAL, execution_mode="pair")


def test_single_mode_has_no_team_sections_or_role_policy():
    harness = target_harness(S0["propose"]["inference"], execution_mode="single", provider=provider())
    assert set(harness) == {"schema", "nature", "scope", "roles_and_tools", "timing", "guards",
                            "delivery_and_evaluation", "unknown"}
    assert harness["scope"]["role_policy_mode"] == "off"
    assert harness["roles_and_tools"] == {"single": {"tools": sorted(SINGLE_TOOLS), "can_spawn": False}}
    assert harness["guards"]["single_rules"]["text"] == runtime_rules("single", tool_call_cap="disabled")
    assert "native_team_inactivity_timeout_seconds" not in harness["timing"]


@pytest.mark.parametrize("limits,expected", [({}, 1800), ({"first_progress_timeout_seconds": 600}, 2040),
                                             ({"progress_idle_timeout_seconds": 600}, 2040)])
def test_reported_timeouts_equal_what_the_runner_launches(tmp_path, monkeypatch, limits, expected):
    """The harness and a launched Team must agree, from the runner's default inactivity setting."""
    config = ProviderConfig("test-model", "http://localhost", request_timeout_seconds=1800, **limits)
    monkeypatch.setattr(qwen_module, "start_server", lambda *args: SimpleNamespace(
        server_port=12345, shutdown=lambda: None, server_close=lambda: None))

    def run_agent(command, *, env, timeout_seconds, stdout_path, stderr_path):
        stdout_path.write_text(json.dumps({"type": "result", "result": "ok", "is_error": False}) + "\n")
        stderr_path.write_text("")
        return {"returncode": 0, "timed_out": False}

    ws = workspace(tmp_path)
    monkeypatch.setattr(ws, "run_agent", run_agent)
    stall = next(f.default for f in ProgramBenchRunner.__dataclass_fields__.values()
                 if f.name == "team_stall_timeout_seconds")
    QwenTeamRunner("unused-cli", upstream=lambda _: pytest.fail("No model call expected"),
                   team_stall_timeout_seconds=stall).run(
        ws, "A local test", "Assignment", "Communication", tmp_path / "run", config,
        timeout_seconds=3600, execution_mode="team")
    launch = json.loads((tmp_path / "run/launch.json").read_text())
    harness = target_harness(MINIMAL, execution_mode="team", provider=config)
    timing = harness["timing"]
    assert timing["native_request_timeout_seconds"] == launch["native_request_timeout_seconds"] == 1980
    assert timing["native_team_inactivity_timeout_seconds"] == launch["native_team_inactivity_timeout_seconds"] == expected
    assert harness["lifecycle"]["teammate_limit"] == launch["teammate_limit"] == 10


def test_checkin_milestones_match_the_scheduler_default():
    source = (ROOT / "agent_company/runtime/qwen_team_checkin.mjs").read_text()
    match = re.search(r"milestones = \[([^\]]*)\]", source)
    assert match is not None
    assert tuple(float(value) for value in match.group(1).split(",")) == CHECKIN_MILESTONES


def test_pointer_resolution_follows_rfc_6901():
    document = {"a/b": {"m~n": [10, {"x": None}]}, "": 1}
    assert resolve_pointer(document, "") is document
    assert resolve_pointer(document, "/a~1b/m~0n/0") == 10
    assert resolve_pointer(document, "/a~1b/m~0n/1/x") is None
    assert resolve_pointer(document, "/") == 1
    for missing in ("/a~1b/m~0n/2", "/a/b", "/a~1b/m~0n/-1", "a~1b", "/a~1b/m~0n/0/x", None):
        with pytest.raises(KeyError):
            resolve_pointer(document, missing)
    harness = target_harness(S0["global"]["inference"], execution_mode="team")
    assert resolve_pointer(harness, "/wakeups/checkin_interval_seconds") == 1800
    assert resolve_pointer(harness, "/online_policy/mode") == "global"


def test_native_version_reads_only_a_nonempty_version(tmp_path):
    assert native_version(tmp_path) is None
    (tmp_path / "package.json").write_text(json.dumps({"version": "0.23.1"}))
    assert native_version(tmp_path) == "0.23.1"
    for bad in ("not json", json.dumps({"version": ""}), json.dumps({"version": 1}), json.dumps([1])):
        (tmp_path / "package.json").write_text(bad)
        assert native_version(tmp_path) is None


def test_executability_summary_counts_and_flags_without_changing_dispatch():
    assert executability_summary(None) == {"recorded": False}
    items = [{"verdict": "supported", "policy_part": "assignment", "rule_quote": "q" * 10, "reason": "r"},
             {"verdict": "unknown", "policy_part": "communication", "rule_quote": "u" * 300, "reason": "x" * 500},
             {"verdict": "unsupported", "policy_part": "assignment", "rule_quote": "v" * 10, "reason": "y"}]
    summary = executability_summary(items)
    assert summary["counts"] == {"supported": 1, "supported_with_conditions": 0, "unsupported": 1, "unknown": 1}
    assert [item["verdict"] for item in summary["unsupported_or_unknown"]] == ["unknown", "unsupported"]
    assert len(summary["unsupported_or_unknown"][0]["rule_quote"]) == 200
    assert len(summary["unsupported_or_unknown"][0]["reason"]) == 400
    assert summary["dispatch"].startswith("unchanged")
