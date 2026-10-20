"""Independent zero-API checks of formal allocation and credential boundaries.

Public planning uses the real comparison planner. Process creation, server
identity, capacity and process inventory are replaced by local test adapters.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_company.deploy import launch
from agent_company.deploy.deployment import Deployment
from agent_company.programbench.comparison import prepare_comparison


ROOT = Path(__file__).resolve().parents[1]
SECRET = "PRIVATE_FORMAL_LAUNCH_SENTINEL_74e518"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path / "final", tmp_path / "assets")
    deployment.code.mkdir(parents=True)
    configs = deployment.code / "configs/stage1"
    configs.mkdir(parents=True)
    names = [*(f"first_round_deepseek_pandoc_{mode}.json" for mode in launch.MODES),
             "s0_deepseek_propose.json", "s0_deepseek_global.json", "deepseek_flash.json", "programbench_instruction.md"]
    for name in names:
        shutil.copyfile(ROOT / "configs/stage1" / name, configs / name)
    shutil.copytree(ROOT / "configs/stage1/policies/fresh_p0_20261001", configs / "policies/fresh_p0_20261001")
    task = deployment.code / "external/programbench-study/src/programbench/data/tasks" / deployment.task_id
    task.mkdir(parents=True)
    (task / "task.yaml").write_text("repository: jgm/pandoc\ncommit: '5caad90" + "0" * 33 + "'\nlanguage: haskell\n")
    (task / "tests.json").write_text("NEVER READ HIDDEN TESTS")
    observed = {"planning": [], "spawned": [], "credentials": []}

    def readonly(command, **options):
        # Exercise runner_plan and the actual public planner, with no process.
        assert Path(command[2]).name == "run_programbench_policy_comparison.py"
        assert "--run" not in command and "--output" not in command
        assert "DEEPSEEK_API_KEY" not in options["env"] and "STAGE1_GLM_API_KEY" not in options["env"]
        assert options["env"]["PYTHONPATH"] == f"{deployment.evaluator}:{deployment.code}"
        assert options["stdin"] == subprocess.DEVNULL
        observed["planning"].append((command.copy(), options))
        def value(flag):
            return command[command.index(flag) + 1]
        plan = prepare_comparison(Path(value("--profile")), Path(value("--project-root")),
                                  repeat_index=int(value("--repeat-index")),
                                  **({"parallelism": int(value("--parallelism"))} if "--parallelism" in command else {}),
                                  **({"pair_id": value("--pair-id")} if "--pair-id" in command else {})).plan
        return SimpleNamespace(returncode=0, stdout=json.dumps(plan), stderr="")

    monkeypatch.setattr(subprocess, "run", readonly)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Unplanned process start"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network attempted"))
    monkeypatch.setattr(launch, "verify_current", lambda _: {"manifest_sha256": "a" * 64})
    monkeypatch.setattr(launch, "capacity", lambda _: {
        "nproc": 64, "mem_total_kb": 256 * 1024 * 1024, "mem_available_kb": 200 * 1024 * 1024,
        "loadavg": "0.1 0.2 0.3 1/100 1", "running_containers": [],
        "container_ownership_scope": "all_running_containers_without_ownership_attribution"})
    monkeypatch.setattr(launch, "active_runner_processes", lambda _: [])
    monkeypatch.setattr(launch, "pid_alive", lambda _: False)
    monkeypatch.setattr(launch, "runtime_call_outcomes", lambda *a: {"passed": True})
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    monkeypatch.setenv("STAGE1_GLM_API_KEY", SECRET)

    def payload():
        observed["credentials"].append("read")
        return {"api_key": SECRET}

    return SimpleNamespace(deployment=deployment, configs=configs, observed=observed, payload=payload)


def spec(rig, mode="off", *, step="first_round", index=None):
    if index is None:
        index = 0 if step.startswith("s0_") else launch.FIRST_ROUND[mode]
    name = f"s0_deepseek_{mode}.json" if step.startswith("s0_") else f"first_round_deepseek_pandoc_{mode}.json"
    return {"runner": "comparison", "step": step, "profile": str(rig.configs / name),
            "arguments": ["--repeat-index", str(index)]}


def paid(rig, name, index, *, step="first_round", state="completed", cleanup=None, alive=False):
    root = rig.deployment.final / "paid_runs" / name
    write(root / "launch.json", {"repeat_index": index, "step": step, "pid": 98765 if alive else None})
    write(root / "run/run_settings.json", {"repeat_index": index})
    write(root / "run/progress.json", {"state": state, "running": [],
          "cleanup_attention": [] if cleanup is None else cleanup, "requires_user_action": state.startswith("blocked_")})
    return root


def forbid_mutation_and_credentials(rig, monkeypatch):
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("Refusal created a directory"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Refusal started a process"))

    def payload():
        pytest.fail("Refusal read credentials")
    return payload


def test_prepare_uses_real_public_planner_without_writes_or_credentials(rig, monkeypatch):
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    result = launch.prepare(rig.deployment, spec(rig))
    assert result["status"] == "prepared_not_launched"
    assert result["name"] == "wave1_off_r1" and result["plan"]["new_rollouts"] == 1
    assert result["runtime"]["satisfied"]
    assert result["directory"].endswith("/paid_runs/wave1_off_r1")
    assert result["command"][-2:] == ["--output", result["directory"] + "/run"]
    assert not Path(result["directory"]).exists() and rig.observed["planning"]
    assert not rig.observed["credentials"] and callable(payload)


@pytest.mark.parametrize("logical_passed", [True, False])
def test_real_smoke_requires_final_logical_success_beyond_script_exit_zero(rig, monkeypatch, logical_passed):
    observed = []
    assessment = {"passed": logical_passed, "complete": True,
                  "failed_calls": 0 if logical_passed else 1}

    def inspect(runtime):
        observed.append(runtime)
        return assessment

    monkeypatch.setattr(launch, "runtime_call_outcomes", inspect)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout='{"passed":true}', stderr=""))
    report = launch.preparation(rig.deployment, kind="smoke_real", arm="team_p0_propose",
                                sequence=1, payload=rig.payload)
    assert report["returncode"] == (0 if logical_passed else 1)
    assert report["logical_call_assessment"] == assessment
    assert observed == [rig.deployment.final / "preparations/smoke_real_team_p0_propose_1/runtime"]
    assert len(rig.observed["credentials"]) == 1


def test_real_smoke_redacts_nested_health_diagnostics(rig, monkeypatch):
    assessment = {"passed": False, "calls": [{"error": f"provider detail {SECRET}"}],
                  "reasons": [f"evidence file {SECRET}"]}
    monkeypatch.setattr(launch, "runtime_call_outcomes", lambda _: assessment)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout=SECRET, stderr=SECRET))
    report = launch.preparation(rig.deployment, kind="smoke_real", arm="single",
                                sequence=1, payload=rig.payload)
    assert SECRET not in json.dumps(report)
    assert report["logical_call_assessment"]["calls"][0]["error"] == "provider detail [REDACTED]"
    assert report["returncode"] == 1


@pytest.mark.parametrize("mode", launch.MODES)
def test_first_round_prepare_does_not_require_s0(rig, monkeypatch, mode):
    # No paid_runs or S0 evidence exists in this fresh deployment.
    forbid_mutation_and_credentials(rig, monkeypatch)
    result = launch.prepare(rig.deployment, spec(rig, mode))
    assert "previous_waves_terminated" not in result["runtime"]["conditions"]
    assert "s0_completed_without_hard_failure" not in result["runtime"]["conditions"]
    assert result["runtime"]["satisfied"] is True


@pytest.mark.parametrize("available", [None, 2 * 61440 * 1024 - 1])
def test_prepare_reports_insufficient_two_arm_memory_then_launch_refuses(rig, monkeypatch, available):
    for index in (1, 2, 3):
        paid(rig, f"wave{index}", index)
    host = launch.capacity(rig.deployment)
    monkeypatch.setattr(launch, "capacity", lambda _: {**host, "mem_available_kb": available})
    result = launch.prepare(rig.deployment, spec(rig, "propose"))
    assert "previous_waves_terminated" not in result["runtime"]["conditions"]
    assert result["runtime"]["conditions"]["concurrent_memory_available"] is False
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Timing or capacity"):
        launch.launch(rig.deployment, spec(rig, "propose"), payload)


@pytest.mark.parametrize("reason", ["existing_directory", "used_index", "wrong_wave", "missing_index", "zero_index",
                                    "different_profile_bytes", "changed_identity", "active_runner"])
def test_launch_refusals_precede_credential_mkdir_and_process(rig, monkeypatch, reason):
    selected = spec(rig)
    if reason == "existing_directory":
        (rig.deployment.final / "paid_runs/wave1_off_r1").mkdir(parents=True)
    elif reason == "used_index":
        paid(rig, "earlier_other_step", 1, step="approved")
    elif reason == "wrong_wave":
        selected = spec(rig, index=2)
    elif reason == "missing_index":
        selected["arguments"] = []
    elif reason == "zero_index":
        selected = spec(rig, index=0)
    elif reason == "different_profile_bytes":
        path = rig.deployment.final / "profiles/copied.json"
        write(path, json.loads(Path(selected["profile"]).read_text()))
        selected["profile"] = str(path)  # semantically equal, different bytes
    elif reason == "changed_identity":
        monkeypatch.setattr(launch, "verify_current", lambda _: (_ for _ in ()).throw(ValueError("Identity changed")))
    elif reason == "active_runner":
        monkeypatch.setattr(launch, "active_runner_processes", lambda _: [98765])
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises((ValueError, FileExistsError)):
        launch.launch(rig.deployment, selected, payload)


@pytest.mark.parametrize("mode", ["global", "propose"])
@pytest.mark.parametrize("mutation", ["policy", "index", "arm_version"])
def test_s0_requires_exact_fresh_p0_and_zero_number(rig, monkeypatch, mode, mutation):
    selected = spec(rig, mode, step=f"s0_{mode}")
    if mutation == "policy":
        policy = rig.configs / "policies/fresh_p0_20261001/assignment_v000.md"
        policy.write_text(policy.read_text() + "changed")
    elif mutation == "index":
        selected["arguments"] = ["--repeat-index", "1"]
    else:
        profile = json.loads(Path(selected["profile"]).read_text())
        profile["arms"][0]["policy_version"] = 1
        write(Path(selected["profile"]), profile)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError):
        launch.launch(rig.deployment, selected, payload)


@pytest.mark.parametrize("state,cleanup", [("blocked_cleanup", []), ("completed", [{"arm_id": "unknown"}]),
                                         ("completed", "missing")])
def test_uncertain_cleanup_prevents_next_launch_before_credentials(rig, monkeypatch, state, cleanup):
    path = paid(rig, "old", 9, step="followup", state=state, cleanup=cleanup)
    if cleanup == "missing":
        progress = json.loads((path / "run/progress.json").read_text())
        progress.pop("cleanup_attention")
        write(path / "run/progress.json", progress)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="cleanup"):
        launch.launch(rig.deployment, spec(rig), payload)


def test_blocked_executability_with_confirmed_cleanup_does_not_block_new_step(rig):
    paid(rig, "paused_followup", 9, step="followup", state="blocked_executability")
    assert launch.prepare(rig.deployment, spec(rig))["runtime"]["satisfied"]


def test_dead_process_with_nonterminal_progress_is_reported_then_refused(rig, monkeypatch):
    paid(rig, "interrupted_operation", 9, step="followup", state="running_evolved")
    result = launch.prepare(rig.deployment, spec(rig))
    assert result["runtime"]["satisfied"] is False
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Timing or capacity"):
        launch.launch(rig.deployment, spec(rig), payload)


def test_memory_equal_to_two_arm_request_passes_and_low_cpu_is_recorded(rig, monkeypatch):
    for index in (1, 2, 3):
        paid(rig, f"wave{index}", index)
    host = launch.capacity(rig.deployment)
    monkeypatch.setattr(launch, "capacity", lambda _: {**host, "mem_available_kb": 2 * 61440 * 1024, "nproc": 1})
    result = launch.prepare(rig.deployment, spec(rig, "propose"))
    assert result["runtime"]["satisfied"]
    assert result["runtime"]["capacity"]["nproc"] == 1


def test_launch_reads_credentials_last_and_starts_with_uncreated_runner_output(rig, monkeypatch):
    selected = spec(rig)
    selected["arguments"] += ["--parallelism", "1", "--pair-id", "explicit pair with spaces"]

    def start(command, **options):
        rig.observed["spawned"].append((command, options))
        output = Path(command[command.index("--output") + 1])
        assert output.name == "run" and not output.exists()
        assert (output.parent / "launch.json").is_file()
        assert options["cwd"] == str(rig.deployment.code)
        assert options["env"]["PYTHONPATH"] == f"{rig.deployment.evaluator}:{rig.deployment.code}"
        assert options["env"]["DEEPSEEK_API_KEY"] == SECRET
        assert "STAGE1_GLM_API_KEY" not in options["env"]
        assert options["stdin"] == subprocess.DEVNULL and options["start_new_session"]
        assert rig.observed["credentials"] == ["read"]
        options["stdout"].write(b"synthetic controller output\n")
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(subprocess, "Popen", start)
    result = launch.launch(rig.deployment, selected, rig.payload)
    assert result["pid"] == 12345 and result["repeat_index"] == 1
    command = result["command"]
    position = command.index("--repeat-index")
    assert command[position:position + len(selected["arguments"])] == selected["arguments"]
    assert SECRET not in json.dumps(result)
    root = rig.deployment.final / "paid_runs/wave1_off_r1"
    assert SECRET not in (root / "launch.json").read_text()
    assert SECRET not in (root / "controller.log").read_text()
    assert not (root / "run").exists()


def test_spawn_error_is_sanitized_and_allocated_directory_is_not_reused(rig, monkeypatch):
    (rig.deployment.final / "paid_runs").mkdir()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET)))
    with pytest.raises(RuntimeError, match="outcome unclear") as error:
        launch.launch(rig.deployment, spec(rig), rig.payload)
    assert SECRET not in str(error.value)
    assert SECRET not in (rig.deployment.final / "paid_runs/wave1_off_r1/launch.json").read_text()
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(FileExistsError):
        launch.launch(rig.deployment, spec(rig), payload)


@pytest.mark.parametrize("flag", ["--source-run", "--previous-round", "--history-trial", "--approved-trial"])
def test_followup_input_outside_final_refuses_before_planning_or_credentials(rig, monkeypatch, flag):
    value = str(rig.deployment.assets / "other/run")
    if flag == "--approved-trial":
        value = "1=" + value
    selected = {"runner": "followup", "step": "followup", "profile": "unread-profile",
                "arguments": ["--repeat-index", "5", flag, value]}
    monkeypatch.setattr(launch, "runner_plan", lambda *a: pytest.fail("Invalid scope reached planning"))
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="within this deployment"):
        launch.launch(rig.deployment, selected, payload)


def test_followup_prepare_preserves_explicit_arguments_and_defers_runner_chain_checks(rig, monkeypatch):
    profile = rig.deployment.final / "profiles/followup.json"
    data = json.loads((rig.configs / "first_round_deepseek_pandoc_off.json").read_text())
    data.pop("first_round_wave")
    write(profile, data)
    plan = prepare_comparison(profile, rig.deployment.code, repeat_index=5).plan
    raw = ["--repeat-index=5", "--source-run", str(rig.deployment.final / "paid_runs/source/run"),
           "--evolution-round", "2", "--previous-round", str(rig.deployment.final / "paid_runs/old/run"),
           "--history-trial", str(rig.deployment.final / "paid_runs/history/run/task/team"),
           "--approved-trial", "1=" + str(rig.deployment.final / "paid_runs/approved/run/task/team")]
    selected = {"runner": "followup", "step": "followup", "profile": str(profile), "arguments": raw}
    seen = []

    def planned(_deployment, _spec, original):
        seen.append(original)
        return ["python", "-u", "followup.py", *original], plan

    monkeypatch.setattr(launch, "runner_plan", planned)
    result = launch.prepare(rig.deployment, selected)
    assert seen == [raw] and result["command"][3:3 + len(raw)] == raw
    assert "not established by prepare" in result["followup_runtime_validation"]
    assert not (rig.deployment.final / "paid_runs").exists()


def confirmation(rig):
    profile = json.loads((rig.configs / "first_round_deepseek_pandoc_propose.json").read_text())
    profile.pop("first_round_wave")
    profile["inference"]["role_policy_mode"] = "off"
    profile["arms"][1]["name"] = "team_p0_off"
    destination = rig.deployment.final / "profiles/confirm_off.json"
    write(destination, profile)
    plan = prepare_comparison(destination, rig.deployment.code, repeat_index=5).plan
    records = []
    for arm, name, index in zip(plan["arms"], ("wave4_propose_r4", "wave1_off_r1"), (4, 1)):
        root = paid(rig, name, index) / "run"
        directory = root / rig.deployment.task_id / arm["arm_id"]
        def policy(key):
            return Path(arm[key]).read_bytes().decode("utf-8") if arm[key] else ""
        metadata = {"repeat_index": index, "pair_id": f"{rig.deployment.task_id}#r{index}"}
        old = deepcopy(plan)
        old.update(arms=[arm], parallelism=1, provider=launch.public_provider(plan["provider_config"]),
                   repeat_index=index, qwen_version="0.23.1", code_version={"kind": "git", "commit": "a" * 40})
        old["tasks"][0].update(metadata)
        write(root / "run_settings.json", old)
        write(directory / "run_request.json", {
            "task": {"task_id": rig.deployment.task_id, "repo_id": "jgm/pandoc"},
            "execution_mode": arm["execution_mode"], "policy_version": arm["policy_version"],
            "assignment_policy": policy("assignment_policy_path"),
            "communication_policy": policy("communication_policy_path"), **metadata})
        write(directory / "task_result.json", {"status": "completed", "score": .1,
              "metrics": {"verifier": {"status": "completed", "reward": .1, "passed_tests": 10, "valid_tests": 100}}})
        write(directory / "verifier/filtered_test_results.json", {"schema_version": 1,
              "task_id": rig.deployment.task_id, "official_status": "completed", "reward": .1, "valid_tests": 100,
              "tests": [{"test_id": str(i), "status": "passed"} for i in range(100)]})
        records.append(directory)
    trigger = rig.deployment.final / "trigger_off.json"
    write(trigger, {"decision": "needs_confirmation", "incumbent_trial": str(records[0]),
                    "candidate_trial": str(records[1])})
    return {"runner": "comparison", "step": "confirm", "profile": str(destination),
            "arguments": ["--repeat-index", "5"], "trigger_report": str(trigger)}, records


def test_confirmation_plans_against_real_trial_records_without_postrun_comparison(rig, monkeypatch):
    from agent_company.programbench import total_evidence as evidence
    selected, records = confirmation(rig)
    monkeypatch.setattr(evidence, "compare_trials", lambda *a, **k: pytest.fail("Postrun comparison called"))
    monkeypatch.setattr(evidence, "confirm_trials", lambda *a, **k: pytest.fail("Postrun confirmation called"))
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    result = launch.prepare(rig.deployment, selected)
    assert result["name"] == "confirm_off_r5" and result["plan"]["parallelism"] == 2
    assert result["input_runs"] == [str(path) for path in records]
    assert result["runtime"]["satisfied"] and callable(payload)


@pytest.mark.parametrize("field", ["policy", "execution_mode", "policy_version", "role_mode", "inference", "evaluation", "provider", "parallelism"])
def test_confirmation_changed_prelaunch_conditions_refuse_before_credentials(rig, monkeypatch, field):
    selected, _ = confirmation(rig)
    path = Path(selected["profile"])
    profile = json.loads(path.read_text())
    if field == "policy":
        alternative = rig.deployment.final / "profiles/changed_policy.md"
        alternative.write_text("different assignment\r\n", encoding="utf-8", newline="")
        profile["arms"][1]["assignment_policy_path"] = str(alternative)
    elif field == "execution_mode":
        profile["arms"][0] = {**profile["arms"][1], "name": "incumbent_team"}
    elif field == "policy_version":
        profile["arms"][1]["policy_version"] = 1
    elif field == "role_mode":
        profile["inference"]["role_policy_mode"] = "global"
    elif field == "inference":
        profile["inference"]["cpus"] = 19
    elif field == "evaluation":
        profile["evaluation"]["docker_cpus"] = 9
    elif field == "provider":
        config = rig.deployment.final / "profiles/provider.json"
        data = json.loads((rig.configs / "deepseek_flash.json").read_text())
        data["model"] = "different-model"
        write(config, data)
        profile["provider_config"] = str(config)
    else:
        profile["parallelism"] = 1
    write(path, profile)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Confirmation"):
        launch.launch(rig.deployment, selected, payload)


@pytest.mark.parametrize("kind,arm", [("smoke_scripted", "single"), ("smoke_scripted", "team_p0_master"),
                                     ("smoke_real", "team_p0_propose"), ("cache_probe", None)])
def test_preparation_fixed_commands_and_credential_redaction(rig, monkeypatch, kind, arm):
    observed = []

    def run(command, **options):
        observed.append(command)
        assert not Path(command[command.index("--output") + 1]).exists()
        assert options["env"]["PYTHONPATH"].startswith(str(rig.deployment.evaluator) + ":")
        assert "STAGE1_GLM_API_KEY" not in options["env"]
        if kind == "smoke_scripted":
            assert not rig.observed["credentials"] and "DEEPSEEK_API_KEY" not in options["env"]
            assert "--scripted" in command and "--real" not in command
        else:
            assert options["env"]["DEEPSEEK_API_KEY"] == SECRET
            assert rig.observed["credentials"] == ["read"]
        if kind != "cache_probe":
            assert command[-4:] == ["--max-calls", "40", "--timeout", "420"]
        else:
            assert "s0_deepseek_propose.json" in command[command.index("--profile") + 1]
        return SimpleNamespace(returncode=1, stdout="provider: " + SECRET if kind != "smoke_scripted" else "scripted",
                               stderr="failure: " + SECRET if kind != "smoke_scripted" else "scripted failure")

    monkeypatch.setattr(subprocess, "run", run)
    result = launch.preparation(rig.deployment, kind=kind, arm=arm, sequence=1, payload=rig.payload)
    assert len(observed) == 1 and result["returncode"] == 1
    assert SECRET not in json.dumps(result) and SECRET not in json.dumps(observed)
    if kind != "smoke_scripted":
        assert "[REDACTED]" in result["stdout"] and "[REDACTED]" in result["stderr"]


@pytest.mark.parametrize("reason", ["existing", "active", "missing_profile", "identity"])
def test_paid_preparation_refusals_precede_stdin_and_subprocess(rig, monkeypatch, reason):
    if reason == "existing":
        (rig.deployment.final / "preparations/smoke_real_single_1").mkdir(parents=True)
    elif reason == "active":
        monkeypatch.setattr(launch, "active_runner_processes", lambda _: [12345])
    elif reason == "missing_profile":
        (rig.configs / "first_round_deepseek_pandoc_propose.json").unlink()
    else:
        monkeypatch.setattr(launch, "verify_current", lambda _: (_ for _ in ()).throw(ValueError("invalid identity")))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Invalid preparation ran a process"))
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises((ValueError, FileExistsError, FileNotFoundError)):
        launch.preparation(rig.deployment, kind="smoke_real", arm="single", sequence=1, payload=payload)


def test_preparation_exception_does_not_expose_stderr_or_credentials(rig, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        subprocess.TimeoutExpired(["python", SECRET], 1, output=SECRET, stderr=SECRET)))
    with pytest.raises(RuntimeError, match="outcome unclear") as error:
        launch.preparation(rig.deployment, kind="cache_probe", sequence=1, payload=rig.payload)
    assert SECRET not in str(error.value)


def test_paid_preparation_refuses_nonterminal_dead_controller(rig, monkeypatch):
    paid(rig, "interrupted_operation", 9, step="followup", state="running_evolved")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Incomplete previous operation dispatched preparation"))
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError):
        launch.preparation(rig.deployment, kind="cache_probe", sequence=1, payload=payload)


def test_status_reports_dead_process_cleanup_and_official_result_without_execution(rig):
    root = paid(rig, "wave1_off_r1", 1, state="blocked_cleanup", cleanup=[{"arm_id": "team"}])
    write(root / "run/task/team/task_result.json", {"score": None,
          "metrics": {"verifier": {"status": "warning", "reward": None}}})
    report = launch.status(rig.deployment)
    row = report["runs"][0]
    assert row["pid_alive"] is False and row["progress"]["state"] == "blocked_cleanup"
    assert row["progress"]["cleanup_attention"] == [{"arm_id": "team"}]
    assert row["official_results"][0]["status"] == "warning"
    assert row["official_results"][0]["score"] is None
    assert not rig.observed["planning"] and not rig.observed["credentials"]


def test_remote_prepare_cli_keeps_stdin_unread_and_calls_production_planning(rig, monkeypatch, capsys):
    from scripts import launch_programbench_formal as cli
    monkeypatch.setattr(cli, "ssh", lambda *a, **k: pytest.fail("Remote CLI attempted SSH"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(read=lambda *a, **k: pytest.fail("Prepare read stdin")))
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    selected = spec(rig)
    args = ["--final", str(rig.deployment.final), "--assets-root", str(rig.deployment.assets), "--remote",
            "--prepare", "--runner", "comparison", "--step", "first_round", "--profile", selected["profile"],
            *selected["arguments"]]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "prepared_not_launched" and rig.observed["planning"]
    assert callable(payload)


def test_smoke_cli_requires_explicit_scripted_or_real_before_transport(rig, monkeypatch, capsys):
    from scripts import launch_programbench_formal as cli
    monkeypatch.setattr(cli, "ssh", lambda *a, **k: pytest.fail("Invalid CLI attempted SSH"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--final", str(rig.deployment.final), "--smoke", "--arm", "single", "--sequence", "1"])
    assert error.value.code == 2
    assert "explicit --scripted or --real" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["none", "nonzero_json", "invalid_json", "timeout"])
def test_workstation_remote_call_passes_key_only_on_stdin_sanitizes_and_never_retries(rig, failure):
    from scripts import launch_programbench_formal as cli
    calls = []
    args = ["--final", str(rig.deployment.final), "--probe", "--sequence", "1"]

    def transport(command, *, stdin, timeout):
        calls.append((command, stdin, timeout))
        assert SECRET not in command
        assert json.loads(stdin) == {"api_key": SECRET}
        assert "--remote" in command
        assert "env -u DEEPSEEK_API_KEY -u STAGE1_GLM_API_KEY" in command
        if failure == "timeout":
            raise subprocess.TimeoutExpired([command], timeout, output=SECRET, stderr=SECRET)
        if failure == "invalid_json":
            return SimpleNamespace(returncode=1, stdout=SECRET, stderr=SECRET)
        return SimpleNamespace(returncode=1 if failure == "nonzero_json" else 0,
                               stdout=json.dumps({"status": "done", "text": SECRET}), stderr=SECRET)

    if failure in {"invalid_json", "timeout"}:
        with pytest.raises(RuntimeError, match="never resend") as error:
            cli._remote_call(rig.deployment, args, paid=True, transport=transport)
        assert SECRET not in str(error.value)
    else:
        result = cli._remote_call(rig.deployment, args, paid=True, transport=transport)
        assert SECRET not in json.dumps(result)
        if failure == "nonzero_json":
            assert result["status"] == "failed" and result["next_action"] == "stop_and_inspect_no_retry"
    assert len(calls) == 1


def test_remote_launch_cli_sanitizes_arbitrary_credential_exception(rig, monkeypatch, capsys):
    from scripts import launch_programbench_formal as cli
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(read=lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("Invalid credential created output"))
    selected = spec(rig)
    args = ["--final", str(rig.deployment.final), "--assets-root", str(rig.deployment.assets), "--remote",
            "--launch", "--runner", "comparison", "--step", "first_round", "--profile", selected["profile"],
            *selected["arguments"]]
    assert cli.main(args) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert json.loads(captured.out)["next_action"] == "stop_and_inspect_no_retry"


@pytest.mark.parametrize("mode,parallelism", [("propose", 1), ("off", 2), ("global", 2), ("master", 2)])
def test_first_round_rejects_effective_parallelism_override_before_side_effects(rig, monkeypatch, mode, parallelism):
    selected = spec(rig, mode)
    selected["arguments"] += ["--parallelism", str(parallelism)]
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="First-round"):
        launch.launch(rig.deployment, selected, payload)
    assert not rig.observed["credentials"]


@pytest.mark.parametrize("case", ["other_uid_unreadable", "own_uid_unrelated_unreadable",
                                 "own_uid_runner_unreadable", "own_uid_runner"])
def test_process_inventory_reads_cmdline_before_relevant_runner_cwd(tmp_path, monkeypatch, case):
    deployment = Deployment(tmp_path / "final", tmp_path / "assets")
    owner, pid = launch.os.getuid(), launch.os.getpid() + 100000
    seen = []

    class Process:
        name = str(pid)

        def stat(self):
            return SimpleNamespace(st_uid=owner + 1 if case == "other_uid_unreadable" else owner)

        def __truediv__(self, name):
            seen.append(name)
            if name == "cwd":
                def resolve(**kwargs):
                    if case.endswith("unreadable"):
                        raise PermissionError("Synthetic proc cwd is inaccessible")
                    return deployment.code
                return SimpleNamespace(resolve=resolve)
            assert name == "cmdline"
            return SimpleNamespace(read_bytes=lambda:
                ("python\0" + ("unrelated.py" if case == "own_uid_unrelated_unreadable" else
                 str(deployment.code / "scripts/run_programbench_policy_comparison.py")) + "\0").encode())

    proc = SimpleNamespace(is_dir=lambda: True, iterdir=lambda: [Process()])
    real_path = Path
    monkeypatch.setattr(launch, "Path", lambda value: proc if value == "/proc" else real_path(value))
    monkeypatch.setattr(launch, "pid_alive", lambda value: value == pid)
    if case == "own_uid_runner_unreadable":
        with pytest.raises(RuntimeError, match="Cannot determine"):
            launch.active_runner_processes(deployment)
        assert seen == ["cmdline", "cwd"]
    elif case == "own_uid_runner":
        assert launch.active_runner_processes(deployment) == [pid]
        assert seen == ["cmdline", "cwd"]
    elif case == "own_uid_unrelated_unreadable":
        assert launch.active_runner_processes(deployment) == []
        assert seen == ["cmdline"]
    else:
        assert launch.active_runner_processes(deployment) == []
        assert seen == []


@pytest.mark.parametrize("mode", ["propose", "global"])
@pytest.mark.parametrize("field", ["provider", "evaluation", "instructions"])
def test_s0_external_profile_cannot_change_committed_measurement_conditions(rig, monkeypatch, mode, field):
    selected = spec(rig, mode, step=f"s0_{mode}")
    profile = json.loads(Path(selected["profile"]).read_text())
    directory = rig.deployment.final / "profiles"
    directory.mkdir()
    if field == "provider":
        provider = json.loads((rig.configs / "deepseek_flash.json").read_text())
        provider["model"] = "synthetic-other-model"
        path = directory / "other_provider.json"
        write(path, provider)
        profile["provider_config"] = str(path)
    elif field == "evaluation":
        profile["evaluation"]["docker_cpus"] = 9
    else:
        path = directory / "other_instruction.md"
        path.write_text("Synthetic alternate public instruction.")
        profile["tasks"][0]["instructions"] = str(path)
    path = directory / "changed_s0.json"
    path.write_text(json.dumps(profile, indent=2) + "\n")
    selected["profile"] = str(path)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="S0.*committed"):
        launch.launch(rig.deployment, selected, payload)
    assert not rig.observed["credentials"]


def test_existing_allocation_lock_blocks_other_launch_before_side_effects(rig, monkeypatch):
    with launch.allocation_lock(rig.deployment):
        payload = forbid_mutation_and_credentials(rig, monkeypatch)
        monkeypatch.setattr(launch, "prepare", lambda *a: pytest.fail("Contending launch performed preflight"))
        with pytest.raises(BlockingIOError):
            launch.launch(rig.deployment, spec(rig), payload)
    assert not rig.observed["credentials"] and not rig.observed["planning"]


def test_readonly_runner_stdin_is_devnull_and_failure_retains_stderr_tail(rig, monkeypatch):
    def reject(command, **options):
        assert options["stdin"] == subprocess.DEVNULL
        assert "DEEPSEEK_API_KEY" not in options["env"]
        assert "STAGE1_GLM_API_KEY" not in options["env"]
        return SimpleNamespace(returncode=2, stdout="", stderr="x" * 5000 + "missing synthetic task.yaml")
    monkeypatch.setattr(subprocess, "run", reject)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="missing synthetic task.yaml") as error:
        launch.launch(rig.deployment, spec(rig), payload)
    assert len(str(error.value)) < 1000


def test_remote_prepare_cli_keeps_the_last_line_of_long_planning_stderr(rig, monkeypatch, capsys):
    from scripts import launch_programbench_formal as cli
    def reject(command, **options):
        assert options["stdin"] == subprocess.DEVNULL
        return SimpleNamespace(returncode=2, stdout="", stderr="x" * 5000 + "final diagnostic: missing task.yaml")
    monkeypatch.setattr(subprocess, "run", reject)
    monkeypatch.setattr(cli, "ssh", lambda *a, **k: pytest.fail("Remote prepare attempted SSH"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(read=lambda *a, **k: pytest.fail("Prepare read credentials")))
    forbid_mutation_and_credentials(rig, monkeypatch)
    selected = spec(rig)
    assert cli.main(["--remote", "--prepare", "--final", str(rig.deployment.final),
                     "--assets-root", str(rig.deployment.assets), "--runner", "comparison",
                     "--step", "first_round", "--profile", selected["profile"], *selected["arguments"]]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed" and len(result["error"]) < 1000
    assert result["error"].endswith("final diagnostic: missing task.yaml")


def refused_record(rig, name="refused_r9", index=9, *, step="followup"):
    root = rig.deployment.final / "paid_runs" / name
    write(root / "launch.json", {"status": "started", "pid": 98765, "step": step, "repeat_index": index})
    (root / "controller.log").write_text("private diagnostic must not appear in status " + SECRET)
    return root


def test_preoutput_refusal_is_terminal_without_returning_log_and_keeps_allocation(rig, monkeypatch):
    root = refused_record(rig)
    rows = launch.paid_records(rig.deployment)
    assert rows[0]["terminal"] and rows[0]["refused_before_output"]
    assert rows[0]["progress"] is None and root.is_dir() and not (root / "run").exists()
    assert launch.series_repeat_indices(rig.deployment.final / "paid_runs") == {9: [str(root)]}
    report = launch.status(rig.deployment)
    assert report["runs"][0]["state"] == "refused_before_output"
    assert SECRET not in json.dumps(report) and "private diagnostic" not in json.dumps(report)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    prepared = launch.prepare(rig.deployment, spec(rig))
    assert prepared["runtime"]["satisfied"] and callable(payload)


def test_refused_wave_keeps_number_but_wave_order_no_longer_blocks_other_steps(rig, monkeypatch):
    refused_record(rig, "wave1_off_r1", 1, step="first_round")
    prepared = launch.prepare(rig.deployment, spec(rig, "global"))
    assert prepared["runtime"]["satisfied"]
    assert "previous_waves_terminated" not in prepared["runtime"]["conditions"]
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(FileExistsError, match="already allocated"):
        launch.launch(rig.deployment, spec(rig), payload)


@pytest.mark.parametrize("case", ["launching", "missing_pid", "noninteger_pid", "run_without_progress"])
def test_uncertain_start_is_never_called_a_zero_fee_preoutput_refusal(rig, monkeypatch, case):
    root = refused_record(rig)
    record = launch.read_json(root / "launch.json")
    if case == "launching":
        record["status"] = "launching"
    elif case == "missing_pid":
        record["pid"] = None
    elif case == "noninteger_pid":
        record["pid"] = "98765"
    else:
        (root / "run").mkdir()
    write(root / "launch.json", record)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError):
        launch.launch(rig.deployment, spec(rig), payload)
    assert not launch.status(rig.deployment)["runs"][0]["refused_before_output"]


def approved(rig):
    source = paid(rig, "followup_propose_r5", 5, step="followup", state="blocked_executability") / "run"
    profile = json.loads((rig.configs / "first_round_deepseek_pandoc_propose.json").read_text())
    profile.pop("first_round_wave")
    profile["parallelism"] = 1
    profile["arms"] = [profile["arms"][1]]
    profile["arms"][0].update(name="candidate_team", policy_version=1)
    policy_texts = {"assignment_policy": "Synthetic approved assignment.\r\n",
                    "communication_policy": "Synthetic approved communication.\r\n"}
    for key, text in policy_texts.items():
        path = rig.deployment.final / "profiles" / (key + ".md")
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        profile["arms"][0][key + "_path"] = str(path)
    path = rig.deployment.final / "profiles/approved.json"
    write(path, profile)
    plan = prepare_comparison(path, rig.deployment.code, repeat_index=6).plan
    saved = {**plan, "repeat_index": 5, "provider": launch.public_provider(plan["provider_config"])}
    write(source / "run_settings.json", saved)
    write(source / "proposal_summary.json", policy_texts)
    write(source / "review_summary.json", {"status": "completed"})
    return {"runner": "comparison", "step": "approved", "profile": str(path),
            "arguments": ["--repeat-index", "6"], "paused_round": str(source)}, source


def test_approved_dispatch_reads_saved_round_and_crlf_policies_before_credentials(rig, monkeypatch):
    selected, source = approved(rig)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    result = launch.prepare(rig.deployment, selected)
    assert result["name"] == "approved_propose_r6" and result["input_runs"] == [str(source)]
    assert result["runtime"]["satisfied"]
    assert "--paused-round" not in result["command"] and callable(payload)


@pytest.mark.parametrize("case", ["missing_paused_round", "outside_paid_runs", "state", "review",
                                 "assignment", "communication", "policy_version", "inference",
                                 "evaluation", "provider", "multiple_arms", "parallelism", "not_team"])
def test_approved_dispatch_binding_refusals_precede_all_paid_side_effects(rig, monkeypatch, case):
    selected, source = approved(rig)
    profile = Path(selected["profile"])
    data = launch.read_json(profile)
    if case == "missing_paused_round":
        selected.pop("paused_round")
    elif case == "outside_paid_runs":
        destination = rig.deployment.assets / "other/run"
        shutil.copytree(source, destination)
        selected["paused_round"] = str(destination)
    elif case == "state":
        progress = launch.read_json(source / "progress.json")
        write(source / "progress.json", {**progress, "state": "blocked_review"})
    elif case == "review":
        write(source / "review_summary.json", {"status": "failed"})
    elif case in {"assignment", "communication"}:
        Path(data["arms"][0][case + "_policy_path"]).write_text("Different policy.", encoding="utf-8")
    elif case == "policy_version":
        data["arms"][0]["policy_version"] = 2
    elif case == "inference":
        data["inference"]["cpus"] = 19
    elif case == "evaluation":
        data["evaluation"]["docker_cpus"] = 9
    elif case == "provider":
        provider = launch.read_json(rig.configs / "deepseek_flash.json")
        provider["model"] = "synthetic-other-model"
        path = rig.deployment.final / "profiles/provider.json"
        write(path, provider)
        data["provider_config"] = str(path)
    elif case == "multiple_arms":
        data["arms"].append({**data["arms"][0], "name": "second_team"})
    elif case == "not_team":
        data["arms"][0].update(execution_mode="single", name="single")
    else:
        data["parallelism"] = 2
    write(profile, data)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError):
        launch.launch(rig.deployment, selected, payload)


def test_confirmation_launch_records_saved_trigger_path_and_bytes_digest(rig, monkeypatch):
    selected, _ = confirmation(rig)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=12345))
    result = launch.launch(rig.deployment, selected, rig.payload)
    trigger = Path(selected["trigger_report"])
    assert result["trigger_report"] == str(trigger.resolve())
    assert result["trigger_report_sha256"] == hashlib.sha256(trigger.read_bytes()).hexdigest()
    saved = launch.read_json(rig.deployment.final / "paid_runs/confirm_off_r5/launch.json")
    assert saved["trigger_report"] == result["trigger_report"]
    assert saved["trigger_report_sha256"] == result["trigger_report_sha256"]
    assert "--trigger-report" not in result["command"] and SECRET not in json.dumps(saved)


@pytest.mark.parametrize("case", ["outside_profiles", "missing_total", "first_round_wave", "trigger_decision",
                                 "trigger_outside_paid_runs", "wrong_task", "nonconfirm_trigger", "nonapproved_paused"])
def test_remaining_preflight_refusals_precede_credentials_mkdir_and_popen(rig, monkeypatch, case):
    selected, records = confirmation(rig)
    profile = Path(selected["profile"])
    data = launch.read_json(profile)
    if case == "outside_profiles":
        path = rig.deployment.final / "outside.json"
        write(path, data)
        selected["profile"] = str(path)
    elif case == "missing_total":
        data.pop("selection_protocol")
        write(profile, data)
    elif case == "first_round_wave":
        data["first_round_wave"] = 5
        write(profile, data)
    elif case == "trigger_decision":
        trigger = launch.read_json(selected["trigger_report"])
        write(Path(selected["trigger_report"]), {**trigger, "decision": "inconclusive"})
    elif case == "trigger_outside_paid_runs":
        destination = rig.deployment.assets / "other/run"
        shutil.copytree(records[0].parent.parent, destination)
        trigger = launch.read_json(selected["trigger_report"])
        trigger["incumbent_trial"] = str(destination / records[0].parent.name / records[0].name)
        write(Path(selected["trigger_report"]), trigger)
    elif case == "wrong_task":
        planner = launch.runner_plan
        def changed(deployment, request, raw):
            command, plan = planner(deployment, request, raw)
            plan["tasks"][0]["task_id"] = "different/task"
            return command, plan
        monkeypatch.setattr(launch, "runner_plan", changed)
    elif case == "nonconfirm_trigger":
        selected = {**spec(rig), "trigger_report": selected["trigger_report"]}
    else:
        selected = {**spec(rig), "paused_round": str(records[0].parent.parent)}
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError):
        launch.launch(rig.deployment, selected, payload)


@pytest.mark.parametrize("kind,arm", [("smoke_real", "single"), ("cache_probe", None)])
def test_preparation_readonly_plan_does_not_read_credentials_write_or_execute(rig, monkeypatch, kind, arm):
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Prepare executed its planned command"))
    result = launch.preparation(rig.deployment, kind=kind, arm=arm, sequence=2,
                                payload=payload, prepare_only=True)
    assert result["status"] == "prepared_not_launched"
    assert result["sequence"] == 2 and result["runtime"]["satisfied"]
    assert not Path(result["output"]).exists()
    assert result["command"][result["command"].index("--output") + 1] == result["output"]
    assert not rig.observed["credentials"]


def test_preparation_readonly_plan_reports_occupied_deployment_without_starting(rig, monkeypatch):
    monkeypatch.setattr(launch, "active_runner_processes", lambda _: [99999])
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Prepare dispatched occupied deployment"))
    result = launch.preparation(rig.deployment, kind="cache_probe", sequence=2,
                                payload=payload, prepare_only=True)
    assert not result["runtime"]["satisfied"]
    assert not result["runtime"]["conditions"]["preparation_exclusive"]
    assert not rig.observed["credentials"]
