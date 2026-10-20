"""Single/off/master Team smoke wiring with real-native, scripted-provider tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import s0_native_smoke as smoke
from basic_smoke_scenario import BasicSmokeScenario
from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig
from test_rejection_integration import native_qwen
from test_team import workspace


@pytest.fixture
def profile(tmp_path, monkeypatch):
    task = tmp_path / "example__test.abcdef0"
    task.mkdir()
    (task / "task.yaml").write_text("repository: example/test\ncommit: 'abcdef0" + "0" * 33 + "'\nlanguage: c\n")
    (tmp_path / "instruction.md").write_text("Synthetic public instruction.")
    (tmp_path / "a.md").write_text("Assign one worker.")
    (tmp_path / "c.md").write_text("Report completed work.")
    value = {"tasks": [{"task_id": task.name, "path": task.name, "instructions": "instruction.md"}],
             "arms": [{"name": "single", "execution_mode": "single", "policy_version": 0},
                      {"name": "team_p0", "execution_mode": "team", "policy_version": 0,
                       "assignment_policy_path": "a.md", "communication_policy_path": "c.md"}],
             "provider_config": "missing-provider-never-read.json", "parallelism": 2,
             "inference": {"agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 61440,
                           "memory_swap_mb": 61440, "max_model_calls": None, "role_policy_mode": "off"},
             "evaluation": {"docker_cpus": 10}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(smoke, "CODE", tmp_path)
    monkeypatch.setattr(smoke, "load_provider", lambda *args: pytest.fail("scripted smoke loaded provider"))
    return path


@pytest.mark.parametrize("arm,execution_mode", [("single", "single"), ("team_p0", "team")])
def test_scripted_main_selects_one_arm_without_provider_and_skips_online_policy_ledger(profile, tmp_path, monkeypatch, capsys, arm, execution_mode):
    calls = []

    class Workspace:
        container_id = "synthetic"
        image_identity = {"image_id": "sha256:synthetic"}
        network_policy_status = {"enforced": True}

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            calls.append("start")

        def run(self, *args):
            return SimpleNamespace(stdout="v22.23.2")

        def finish_agent(self):
            calls.append("finish")

        def stop(self):
            calls.append("stop")

    class Runner:
        def __init__(self, *args, upstream, **kwargs):
            self.scenario = upstream

        def run(self, ws, instruction, assignment, communication, runtime, provider, **kwargs):
            assert kwargs["execution_mode"] == execution_mode
            assert provider.model == "scripted-s0-smoke"
            assert isinstance(self.scenario, BasicSmokeScenario)
            assert (assignment == communication == "") == (execution_mode == "single")
            assert ("Perform the file roundtrip directly." in instruction) == (execution_mode == "single")
            self.scenario.file_verified = True
            self.scenario.shutdown_sent = execution_mode == "team"
            runtime.mkdir()
            relay = runtime / "relay"
            relay.mkdir()
            launch = {"execution_mode": execution_mode, "role_policy_mode": None if execution_mode == "single" else "off",
                      "environment": {"QWEN_CODE_ENABLE_AGENT_TEAM": "0" if execution_mode == "single" else "1"},
                      "command": ["cli", "--system-prompt", assignment + "\n" + communication],
                      "policy_applied": execution_mode == "team", "assignment_policy": assignment or None,
                      "communication_policy": communication or None, "runtime_disclosure": False,
                      "tool_call_cap": "native_default", "teammate_limit": 10 if execution_mode == "team" else None}
            (runtime / "launch.json").write_text(json.dumps(launch))
            (runtime / "qwen_stderr.txt").write_text('Agent Company lifecycle: {"event": "compatibility_loaded"}\n')
            (relay / "events.jsonl").write_text('{"event": "provider_attempt_started", "purpose": "coding", "attempt": 1}\n')
            (relay / "attempt_000001_alpha_provider_request.json").write_text(json.dumps(
                {"messages": [{"role": "system", "content": communication}]}))
            return {"execution": {"returncode": 0, "timed_out": False}, "stop_reason": "native_cli_completed",
                    "logical_model_calls": 3, "provider_attempts": 3, "usage_complete": True}

    monkeypatch.setattr(smoke, "ProgramBenchWorkspace", Workspace)
    monkeypatch.setattr(smoke, "QwenTeamRunner", Runner)
    monkeypatch.setattr(smoke, "project_evidence", lambda *args: {"role_policies": {"status": "off"}})
    output = tmp_path / "smoke"
    assert smoke.main(["--profile", str(profile), "--arm", arm, "--assets-root", str(tmp_path),
                       "--output", str(output), "--scripted"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["execution_mode"] == execution_mode and report["external_model_calls"] == 0
    assert report["checks"]["scripted_file_roundtrip"] and all(report["checks"].values())
    assert not report["official_evaluation_run"] and not report["benchmark_solution_attempted"]
    assert calls == ["start", "finish", "stop"]


def test_multi_arm_profile_requires_explicit_arm_before_workspace_creation(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "ProgramBenchWorkspace", lambda *args, **kwargs: pytest.fail("workspace constructed"))
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--profile", str(profile), "--assets-root", str(tmp_path),
                    "--output", str(tmp_path / "output"), "--scripted"])
    assert exc.value.code == 2 and not (tmp_path / "output").exists()


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_native_basic_smoke_file_roundtrip_with_scripted_provider(tmp_path, execution_mode):
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    marker = str(Path(ws.repo_path) / "native-smoke.txt")
    scenario = BasicSmokeScenario(execution_mode, marker)
    output = tmp_path / "native"
    options = {"runtime_disclosure": True, "tool_call_cap": "disabled"}
    assignment, communication = (("", "") if execution_mode == "single" else
                                 ("Assign one worker.", "Report completed work."))
    result = QwenTeamRunner(str(cli), node=node, upstream=scenario, role_policy_mode="off", **options).run(
        ws, "Write and read back the synthetic marker; finish normally.", assignment, communication,
        output, ProviderConfig("scripted-baseline-smoke", "http://unused.invalid", request_timeout_seconds=40),
        timeout_seconds=120, execution_mode=execution_mode, max_model_calls=40)
    assert result["stop_reason"] == "native_cli_completed"
    assert scenario.file_verified and Path(marker).read_text() == "NATIVE_OK\n"
    assert result["scripted_provider"] is True
    launch = json.loads((output / "launch.json").read_text())
    checks = smoke.basic_runtime_checks(launch, smoke.lifecycle(output / "qwen_stderr.txt"), output,
                                        options, {"execution_mode": execution_mode}, assignment, communication)
    assert all(checks.values()), checks
    assert scenario.shutdown_sent == (execution_mode == "team")


def test_native_master_smoke_commits_direct_update_and_injects_it_into_worker(tmp_path):
    from test_role_policy_integration import run
    from agent_company.policy.role_policy import RolePolicyStore
    scenario, result, lifecycle, events, output = run(tmp_path, "master_set", "master")
    assert result["scripted_provider"] and scenario.shutdown_sent and scenario.master_set_sent
    assert not scenario.decision_sent
    assert smoke.scripted_controls_committed("master", events)
    assert smoke.update_reached_worker("master", scenario)
    changes = [change["event"] for row in events if row.get("event") == "role_policy_batch"
               for change in row["changes"]]
    assert "role_policy_proposed" not in changes and "role_policy_decided" not in changes
    launch = json.loads((output / "launch.json").read_text())
    assert launch["role_policy_mode"] == "master"
    assert not any(event["event"] == "policy_report_patch_applied" for event in lifecycle)
    replayed = RolePolicyStore.replay(RolePolicyStore.read_log(output / "relay/events.jsonl"), [].append)
    assert replayed.export() == json.loads((output / "relay/role_policy_state.json").read_text())


@pytest.mark.parametrize("change", [
    {"event": "role_policy_committed", "provenance": "spawn_initial", "slot": "alpha", "version": 1},
    {"event": "role_policy_committed", "provenance": "master_set", "slot": "beta", "version": 2},
    {"event": "role_policy_decided", "slot": "alpha", "version": 2},
])
def test_master_smoke_rejects_unrelated_or_initial_policy_changes(change):
    assert not smoke.scripted_controls_committed("master", [{"event": "role_policy_batch", "changes": [change]}])


@pytest.mark.parametrize("system_text", ["Your role policy version 1", "Your role policy version 2", ""])
def test_master_smoke_needs_both_new_version_and_body_in_worker_request(system_text):
    payload = {"messages": [{"role": "system", "content": system_text}]}
    assert not smoke.update_reached_worker("master", SimpleNamespace(worker_after_update=payload))
    assert not smoke.update_reached_worker("master", SimpleNamespace(worker_after_update=None))


def test_master_smoke_main_uses_first_round_profile_and_formal_switches(tmp_path, monkeypatch, capsys):
    """Real CLI/main/profile; only Docker workspace and model replies are local substitutes."""
    cli, node = native_qwen()
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/stage1/first_round_deepseek_pandoc_master.json"
    formal = json.loads(path.read_text())["inference"]
    local = workspace(tmp_path)
    lifecycle = []
    local.container_id = "synthetic-local-workspace-not-docker"
    local.image_identity = {"source": "local test adapter; no container image measured"}
    local.network_policy_status = {"enforced": True, "source": "scripted provider; no network calls"}
    local.start = lambda: lifecycle.append("start")
    local.finish_agent = lambda: lifecycle.append("finish")
    local.stop = lambda: lifecycle.append("stop")
    local.run = lambda command: SimpleNamespace(stdout="local node test adapter")
    monkeypatch.setattr(smoke, "ProgramBenchWorkspace", lambda *a, **k: local)
    monkeypatch.setattr(smoke, "load_provider", lambda *a: pytest.fail("provider loaded"))
    actual_options = []
    def runner(_cli, *, node, **options):
        actual_options.append(options.copy())
        return QwenTeamRunner(str(cli), node=native_qwen()[1], **options)
    monkeypatch.setattr(smoke, "QwenTeamRunner", runner)
    output = tmp_path / "master_main"
    assert smoke.main(["--profile", str(path), "--arm", "team_p0_master", "--scripted",
                       "--assets-root", str(root), "--output", str(output), "--timeout", "120"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] and all(report["checks"].values())
    assert report["checks"]["scripted_controls_committed"]
    assert report["checks"]["updated_report_reached_master_request"]
    assert report["checks"]["scripted_shutdown_handshake_completed"]
    assert report["external_model_calls"] == 0 and lifecycle == ["start", "finish", "stop"]
    assert report["formal_switches"] == {key: formal[key] for key in report["formal_switches"]}
    assert actual_options[0]["checkin_interval_seconds"] == 1800
