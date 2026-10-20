"""agent_rules_v1 protocol identity from profile to launch record; zero API, no native CLI.

Real Qwen CLI scenarios wait until the draft model-visible texts are approved
(plan v4 section 14 and M4). Here run_agent writes empty native output only.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from agent_company.policy import agent_rules_text as text
from agent_company.policy.role_package import create_m0_package, create_p0_package
from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig
from agent_company.runtime.team import LocalWorkspace
from scripts import run_programbench_policy_comparison as comparison
from test_programbench_policy_comparison import setup as _comparison_setup
from test_programbench_formal_launch import write
from test_programbench_runner import rig as _runner_rig

comparison_setup = _comparison_setup
runner_rig = _runner_rig


class StubWorkspace(LocalWorkspace):
    """Local files, but the native CLI is never started."""

    def run_agent(self, command, env, timeout_seconds, stdout_path, stderr_path):
        Path(stdout_path).write_text("")
        Path(stderr_path).write_text("")
        return {"returncode": 0, "timed_out": False, "elapsed_seconds": 0.0}


def stub_workspace(tmp_path):
    from test_team import workspace
    base = workspace(tmp_path)
    return StubWorkspace(base.repo_path, base.worker_root, base.runtime_root)


def launch(tmp_path, package, mode="propose", **options):
    runner = QwenTeamRunner("/unused/cli.js", node="node", upstream=lambda payload: pytest.fail("no model call"),
                            role_policy_mode=mode, policy_package=package, **options)
    runner.run(stub_workspace(tmp_path), "Synthetic task.", package["team"]["assignment"],
               package["team"]["communication"], tmp_path / "run",
               ProviderConfig("scripted", "http://unused.invalid", request_timeout_seconds=10), timeout_seconds=60)
    return json.loads((tmp_path / "run/launch.json").read_text())


def test_agent_rules_launch_records_protocol_files_env_and_fixed_master_system(tmp_path):
    record = launch(tmp_path, create_m0_package("ASSIGN-PRINCIPLES", "COMM-PRINCIPLES"), checkin_interval_seconds=1800,
                    team_message_delivery="between_tool_calls")
    env, command = record["environment"], record["command"]
    assert record["role_policy_protocol"] == "agent_rules_v1" and record["role_policy_mode"] == "propose"
    assert record["model_visible_text_status"] == "approved_20261018"
    assert record["role_policy_disclosure_version"] == "agent_rules_v1"
    assert env["AGENT_COMPANY_POLICY_PROTOCOL"] == "agent_rules_v1" and "AGENT_COMPANY_ROLE_PACKAGE" not in env
    assert env["AGENT_COMPANY_ROLE_POLICY_MODE"] == "propose"
    status = str(Path(record["runtime_dir"]) / "status")
    assert command[command.index("--include-directories") + 1] == status
    assert record["policy_files"] == {"policies": status + "/agent_policies.md", "lessons": status + "/team_lessons.md",
                                      "board": status + "/team_board.md"}
    assert "No teammates yet." in Path(record["policy_files"]["policies"]).read_text()
    system = command[command.index("--system-prompt") + 1]
    assert text.MASTER_COMMUNICATION_SECTION.format(communication="COMM-PRINCIPLES") in system
    assert "ASSIGN-PRINCIPLES" in system and "also supplied in each teammate" not in system
    assert "policy-review reminder" not in system
    state = json.loads((tmp_path / "run/relay/role_policy_state.json").read_text())
    assert state["closed"] and state["package"]["schema_version"] == 2


def test_agent_rules_team_requires_checkins_and_between_tool_calls_delivery(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="require runtime check-ins"):
        launch(tmp_path / "a", create_m0_package("A", "C"), team_message_delivery="between_tool_calls")
    with pytest.raises(ValueError, match="between_tool_calls"):
        launch(tmp_path / "b", create_m0_package("A", "C"), checkin_interval_seconds=1800)


def test_single_arm_carries_the_package_protocol_identity(tmp_path):
    runner = QwenTeamRunner("/unused/cli.js", node="node", upstream=lambda payload: pytest.fail("no model call"),
                            role_policy_mode="propose", policy_package=create_m0_package("A", "C"))
    runner.run(stub_workspace(tmp_path), "Synthetic task.", "", "", tmp_path / "run",
               ProviderConfig("scripted", "http://unused.invalid", request_timeout_seconds=10), timeout_seconds=60,
               execution_mode="single")
    record = json.loads((tmp_path / "run/launch.json").read_text())
    env = record["environment"]
    assert env["AGENT_COMPANY_POLICY_PROTOCOL"] == "agent_rules_v1" and "AGENT_COMPANY_ROLE_PACKAGE" not in env
    assert record["role_policy_protocol"] == "agent_rules_v1" and "policy_files" not in record


def test_agent_rules_rejects_other_modes_and_role_package_launch_is_unchanged(tmp_path):
    with pytest.raises(ValueError, match="propose"):
        QwenTeamRunner("/unused/cli.js", role_policy_mode="off", policy_package=create_m0_package("A", "C"))
    record = launch(tmp_path, create_p0_package("A", "C"))
    env = record["environment"]
    assert env["AGENT_COMPANY_ROLE_PACKAGE"] == "1" and "AGENT_COMPANY_POLICY_PROTOCOL" not in env
    assert record["role_policy_protocol"] == "role_package_v1"
    assert "policy_files" not in record and "model_visible_text_status" not in record


def agent_rules_profile(setup, package):
    root, path, old, _ = setup
    write(root / "m0.json", package)
    profile = deepcopy(old)
    profile["inference"].update(role_policy_protocol="agent_rules_v1", role_policy_mode="propose",
                                team_checkin_seconds=1800, team_message_delivery="between_tool_calls")
    profile["arms"] = [{"name": "single", "execution_mode": "single", "policy_version": 0, "policy_package_path": "m0.json"},
                       {"name": "team_m0", "execution_mode": "team", "policy_version": 0, "policy_package_path": "m0.json"}]
    write(path, profile)
    return root, path


def test_comparison_plans_agent_rules_arms_and_checks_package_protocol(comparison_setup):
    m0 = create_m0_package("Assign work.", "Write policies.")
    root, path = agent_rules_profile(comparison_setup, m0)
    plan = comparison.prepare_comparison(path, root).plan
    team = next(arm for arm in plan["arms"] if arm["name"] == "team_m0")
    assert team["policy_package"] == m0 and team["role_policy_mode"] == "propose"
    assert plan["inference"]["role_policy_protocol"] == "agent_rules_v1"
    root, path = agent_rules_profile(comparison_setup, create_p0_package("A", "C"))
    with pytest.raises(ValueError, match="protocol disagrees"):
        comparison.prepare_comparison(path, root)


@pytest.mark.parametrize("change,message", [({"role_policy_mode": "off"}, "propose only"),
                                            ({"team_checkin_seconds": None}, "team_checkin_seconds"),
                                            ({"team_message_delivery": "native_idle"}, "between_tool_calls")])
def test_comparison_rejects_off_and_missing_checkins_for_agent_rules(comparison_setup, change, message):
    root, path = agent_rules_profile(comparison_setup, create_m0_package("A", "C"))
    profile = json.loads(path.read_text())
    profile["inference"].update(change)
    write(path, profile)
    with pytest.raises(ValueError, match=message):
        comparison.prepare_comparison(path, root)


def agent_rules_trial(tmp_path, *, launch_protocol="agent_rules_v1"):
    from test_programbench_total_evidence import trial
    package = create_m0_package("A0", "C0")
    path = trial(tmp_path, "run", "team_m0", 40)
    request = json.loads((path / "run_request.json").read_text())
    request.update(assignment_policy="A0", communication_policy="C0", policy_package=package, role_policy_mode="propose")
    write(path / "run_request.json", request)
    settings_path = path.parent.parent / "run_settings.json"
    settings = json.loads(settings_path.read_text())
    settings["inference"]["role_policy_protocol"] = "agent_rules_v1"
    settings["arms"] = [{"arm_id": "team_m0", "execution_mode": "team", "policy_version": 0,
                         "policy_package": package, "role_policy_mode": "propose"}]
    write(settings_path, settings)
    write(path / "team/launch.json", {"native_compatibility_repairs": ["synthetic"], "role_policy_protocol": launch_protocol,
                                      "policy_package": package, "role_policy_mode": "propose"})
    return path


def test_trial_identity_accepts_agent_rules_and_checks_the_native_protocol(tmp_path):
    from agent_company.programbench.total_evidence import trial_record
    record = trial_record(agent_rules_trial(tmp_path / "ok"))
    assert record["condition"]["policy_package"]["protocol"] == {"id": "agent_rules_v1"}
    assert "roles" not in record["condition"]["policy_package"]
    with pytest.raises(ValueError, match="native launch"):
        trial_record(agent_rules_trial(tmp_path / "bad", launch_protocol="role_package_v1"))


def test_runner_passes_agent_rules_package_and_rejects_protocol_mismatch(runner_rig):
    from dataclasses import replace
    rig = runner_rig
    m0 = create_m0_package("Assignment principles", "Communication principles")
    request = replace(rig.request, execution_mode="team", policy_version=0, role_policy_mode="propose",
                      policy_package=m0, assignment_policy=m0["team"]["assignment"],
                      communication_policy=m0["team"]["communication"])
    runner = replace(rig.runner, role_policy_protocol="agent_rules_v1", role_policy_mode="propose",
                     team_checkin_seconds=1800, team_message_delivery="between_tool_calls")
    assert runner(request).status == "completed"
    assert rig.qwen_options["policy_package"] == m0 and rig.qwen_options["role_policy_mode"] == "propose"
    with pytest.raises(ValueError, match="protocol disagrees"):
        runner(replace(request, policy_package=create_p0_package("A", "C"), assignment_policy="A", communication_policy="C"))
    with pytest.raises(ValueError, match="team_checkin_seconds"):
        replace(rig.runner, role_policy_protocol="agent_rules_v1", role_policy_mode="propose", team_checkin_seconds=None,
                team_message_delivery="between_tool_calls")
    with pytest.raises(ValueError, match="between_tool_calls"):
        replace(rig.runner, role_policy_protocol="agent_rules_v1", role_policy_mode="propose", team_checkin_seconds=1800)


class CompletedWorkspace(StubWorkspace):
    """The native CLI never starts; stdout carries one successful native result record."""

    def run_agent(self, command, env, timeout_seconds, stdout_path, stderr_path):
        Path(stdout_path).write_text(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                                                 "result": "done"}) + "\n")
        Path(stderr_path).write_text("")
        return {"returncode": 0, "timed_out": False, "elapsed_seconds": 0.0}


@pytest.mark.parametrize("fault", ["finish_rule_engine", "empty_message", "rule_obligations.json", "role_policy_state.json"])
def test_a_failed_run_end_settlement_is_an_evidence_failure_not_a_completion(tmp_path, monkeypatch, fault):
    # Review 10-18 F1: the rule-obligation record could vanish while the run still read native_cli_completed.
    from agent_company.deploy import run_health
    from agent_company.runtime import relay as relay_module
    from test_team import workspace as base
    if fault in ("finish_rule_engine", "empty_message"):
        def boom(self, path):
            raise OSError() if fault == "empty_message" else OSError("No space left on device (injected)")
        monkeypatch.setattr(relay_module.TeamRelay, "finish_rule_engine", boom)
    else:
        real_save = relay_module.TeamRelay.save

        def save(self, path, data):
            if Path(path).name == fault:
                raise OSError("No space left on device (injected)")
            return real_save(self, path, data)
        monkeypatch.setattr(relay_module.TeamRelay, "save", save)
    package = create_m0_package("ASSIGN", "COMM")
    runner = QwenTeamRunner("/unused/cli.js", node="node", upstream=lambda payload: pytest.fail("no model call"),
                            role_policy_mode="propose", policy_package=package, checkin_interval_seconds=1800,
                            team_message_delivery="between_tool_calls")
    ws = base(tmp_path)
    result = runner.run(CompletedWorkspace(ws.repo_path, ws.worker_root, ws.runtime_root), "Synthetic task.",
                        package["team"]["assignment"], package["team"]["communication"], tmp_path / "run",
                        ProviderConfig("scripted", "http://unused.invalid", request_timeout_seconds=10), timeout_seconds=60)
    assert result["stop_reason"] == "provider_error"
    assert any(e.get("error_kind") == "evidence_write_failed" for e in result["errors"])
    key = "finalization_error" if fault == "role_policy_state.json" else "rule_finalization_error"
    assert result["role_policy_run"][key]
    failures, _ = run_health._policy_completion({"role_policy_run": result["role_policy_run"]},
                                                {"role_policy": {"mode": "propose"}})
    assert failures and failures[0].endswith("finalization_error")


def test_both_arms_of_an_agent_rules_batch_learn_the_shared_resources_and_earlier_runs_do_not():
    resources = {"cpus": 20.0, "memory_mb": 81920}  # ProgramBenchTask stores CPUs as a float
    package = create_m0_package("ASSIGN", "COMM")
    rules = QwenTeamRunner("/unused/cli.js", role_policy_mode="propose", policy_package=package,
                           checkin_interval_seconds=1800, team_message_delivery="between_tool_calls")
    master, worker = rules.system_texts("ASSIGN", "COMM", is_team=True, status_file="/s.json", resources=resources)
    team_note = text.resources_note(resources, team=True)
    assert team_note.startswith("Resources: this run's container has 20 CPUs and 80 GiB of memory.")
    assert master.endswith(team_note) and worker.endswith(team_note)
    single, _ = rules.system_texts("", "", is_team=False, status_file=None, resources=resources)
    assert single.endswith(text.resources_note(resources, team=False))
    legacy = QwenTeamRunner("/unused/cli.js", role_policy_mode="propose", policy_package=create_p0_package("A", "C"))
    legacy_master, legacy_worker = legacy.system_texts("A", "C", is_team=True, status_file=None, resources=resources)
    plain = QwenTeamRunner("/unused/cli.js")
    plain_single, _ = plain.system_texts("", "", is_team=False, status_file=None, resources=resources)
    assert all("Resources:" not in t for t in (legacy_master, legacy_worker, plain_single))
