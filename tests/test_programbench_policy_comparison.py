"""Synthetic scheduler tests: no Docker, models, network or benchmark test data."""
from copy import deepcopy
import hashlib
import json
import shutil
import socket
import subprocess
import threading

import pytest

from scripts import run_programbench_policy_comparison as comparison
from agent_company.evolution.loop import TaskResult
from agent_company.runtime.relay import ProviderConfig
from agent_company.deploy.code_version import code_version
from agent_company.programbench.total_evidence import compare_trials
from agent_company.programbench.total_selection import MODE as TOTAL_MODE


@pytest.fixture(autouse=True)
def forbid_external_execution(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("external process attempted"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network attempted"))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tasks = []
    for index in (1, 2):
        task_id = f"synthetic__task{index}.{str(index) * 7}"
        directory = tmp_path / task_id
        directory.mkdir()
        (directory / "task.yaml").write_text(
            f"repository: synthetic/task{index}\ncommit: '{str(index) * 40}'\nlanguage: go\n")
        (directory / "tests.json").write_text("DO NOT READ HIDDEN TESTS")
        instruction = tmp_path / f"instruction{index}.md"
        instruction.write_text("Synthetic public implementation instruction.")
        tasks.append({"task_id": task_id, "path": task_id, "instructions": instruction.name})
    for arm in ("incumbent", "candidate"):
        for policy in ("assignment", "communication"):
            (tmp_path / f"{arm}_{policy}.md").write_text(f"{arm} {policy} policy text")
    profile = {"benchmark": "ProgramBench", "provider_config": "provider-never-read-in-plan.json",
               "tasks": tasks, "parallelism": 2,
               "arms": [{"name": arm, "execution_mode": "team", "policy_version": version,
                         "assignment_policy_path": f"{arm}_assignment.md",
                         "communication_policy_path": f"{arm}_communication.md"}
                        for arm, version in (("incumbent", "P3"), ("candidate", "candidate-v4"))],
               "inference": {"agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 61440,
                             "memory_swap_mb": 61440, "max_model_calls": None},
               "evaluation": {"docker_cpus": 10}, "evolution_enabled": False}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    loaders = []

    def load_provider(path):
        loaders.append(path)
        return ProviderConfig("synthetic", "http://unused", api_key="PRIVATE_SYNTHETIC_KEY")

    monkeypatch.setattr(comparison, "load_provider", load_provider)
    return tmp_path, path, profile, loaders


def run(setup, **kwargs):
    root, path, _, _ = setup
    kwargs.setdefault("repeat_index", 0)
    return comparison.run_comparison(path, root, root / "output", qwen_package=root / "qwen",
                                     node_directory=root / "node", docker="mock-docker", **kwargs)


def update_profile(setup, mutate):
    _, path, profile, _ = setup
    changed = deepcopy(profile)
    mutate(changed)
    path.write_text(json.dumps(changed))


def test_plan_and_default_cli_do_not_read_provider_or_create_output(setup, monkeypatch, capsys):
    root, path, _, loaders = setup
    monkeypatch.setattr(comparison, "ProgramBenchRunner", lambda **kw: pytest.fail("runner constructed"))
    plan = comparison.prepare_comparison(path, root).plan
    assert plan["new_rollouts"] == 4 and plan["reused_rollouts"] == 0
    assert plan["waves"] == [
        [{"task_id": task["task_id"], "arm_id": arm} for arm in ("incumbent", "candidate")]
        for task in plan["tasks"]]
    assert not (root / "provider-never-read-in-plan.json").exists()
    assert comparison.main(["--profile", str(path), "--project-root", str(root),
                            "--output", str(root / "output")]) == 0
    assert json.loads(capsys.readouterr().out)["inference"]["max_model_calls"] is None
    assert loaders == [] and not (root / "output").exists()


def test_explicit_run_required_arguments_do_not_load_provider(setup):
    root, path, _, loaders = setup
    with pytest.raises(SystemExit) as error:
        comparison.main(["--profile", str(path), "--project-root", str(root), "--run"])
    assert error.value.code == 2 and loaders == []


def test_two_team_arms_overlap_have_independent_requests_and_exactly_once_outputs(setup, monkeypatch):
    root, _, profile, loaders = setup
    barrier, lock = threading.Barrier(2), threading.Lock()
    captured, constructed, writes = [], [], []
    main_thread = threading.get_ident()
    original_write = comparison.write_json_atomic

    def write(path, value):
        writes.append(threading.get_ident())
        return original_write(path, value)

    class Runner:
        def __init__(self, **options):
            assert options["max_model_calls"] is None
            assert options["task_cpus"] == 20 and options["task_memory_mb"] == 61440
            assert options["agent_timeout_seconds"] == 21600 and options["evaluator_cpus"] == 10
            assert options["docker"] == "mock-docker"
            with lock:
                constructed.append(self)

        def __call__(self, request):
            with lock:
                captured.append(request)
            barrier.wait(timeout=3)
            score = .75 if request.output_dir.name == "candidate" else .5
            return TaskResult(status="completed", score=score, usage={"total_tokens": 17},
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    monkeypatch.setattr(comparison, "write_json_atomic", write)
    result = run(setup)
    assert result["state"] == "completed" and not result["requires_user_action"]
    assert result["terminal"] == result["launched"] == result["planned"] == 4
    assert result["pending"] == result["running"] == []
    assert len({id(runner) for runner in constructed}) == 4
    assert set(writes) == {main_thread}
    assert len(captured) == 4 and len(loaders) == 1
    for request in captured:
        arm = request.output_dir.name
        assert request.fresh_start and request.repeat_index == 0 and request.execution_mode == "team"
        assert request.split == "heldout" and request.pair_id == f"{request.task.task_id}#r0"
        assert request.assignment_policy == f"{arm} assignment policy text"
        assert request.communication_policy == f"{arm} communication policy text"
        assert request.policy_version == ("P3" if arm == "incumbent" else "candidate-v4")
        row = json.loads((request.output_dir / "run_request.json").read_text())
        assert row["arm_id"] == arm and row["fresh_start"] is True
        assert json.loads((request.output_dir / "launch.json").read_text())["dispatched_at"]
        value = json.loads((request.output_dir / "task_result.json").read_text())
        assert value["usage"] == {"total_tokens": 17}
    settings = json.loads((root / "output/run_settings.json").read_text())
    assert settings["arms"][1]["policy_version"] == "candidate-v4"
    assert settings["runtime"] == "custom_Qwen_not_official_mini_SWE"
    assert settings["automatic_restarts"] is settings["evolution_enabled"] is False
    assert all("PRIVATE_SYNTHETIC_KEY" not in file.read_text() for file in (root / "output").rglob("*.json"))
    with pytest.raises(FileExistsError):
        run(setup)
    assert len(captured) == 4 and len(loaders) == 1


def test_three_arms_use_task_local_two_plus_one_waves_and_single_empty_policies(setup, monkeypatch):
    update_profile(setup, lambda p: p["arms"].append({"name": "single", "execution_mode": "single",
                                                    "policy_version": 0}))
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    completed, captured, active = [], [], set()
    peak = 0

    class Runner:
        def __init__(self, **kw):
            pass

        def __call__(self, request):
            nonlocal peak
            task, arm = request.task.task_id, request.output_dir.name
            with lock:
                if arm == "single":
                    assert {(task, "incumbent"), (task, "candidate")} <= set(completed)
                if "task2" in task:
                    assert sum("task1" in item[0] for item in completed) == 3
                active.add((task, arm))
                peak = max(peak, len(active))
                captured.append(request)
            if arm != "single":
                barrier.wait(timeout=3)
            else:
                assert request.assignment_policy == request.communication_policy == ""
            with lock:
                active.remove((task, arm))
                completed.append((task, arm))
            return TaskResult(status="completed", score=.5, metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    result = run(setup)
    assert result["terminal"] == 6 and len(captured) == 6 and peak == 2


@pytest.mark.parametrize("raise_error", [False, True])
def test_unknown_cleanup_completes_current_wave_but_blocks_every_later_wave(setup, monkeypatch, raise_error):
    update_profile(setup, lambda p: p["arms"].append({"name": "single", "execution_mode": "single", "policy_version": 0}))
    barrier = threading.Barrier(2)
    captured = []

    class Runner:
        def __init__(self, **kw):
            pass

        def __call__(self, request):
            captured.append(request)
            barrier.wait(timeout=3)
            if request.output_dir.name == "incumbent":
                if raise_error:
                    raise RuntimeError("PRIVATE_TRACE_AND_CREDENTIAL")
                return TaskResult(status="failed", score=None, metrics={"inference_cleanup_confirmed": False})
            return TaskResult(status="completed", score=.9, metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    result = run(setup)
    assert result["state"] == "blocked_cleanup" and result["requires_user_action"]
    assert result["terminal"] == result["launched"] == len(captured) == 2
    assert result["planned"] == 6 and len(result["pending"]) == 4 and result["running"] == []
    assert [row["arm_id"] for row in result["cleanup_attention"]] == ["incumbent"]
    assert result["rows"][0]["score"] is None and result["rows"][1]["score"] == .9
    for file in (setup[0] / "output").rglob("*.json"):
        assert "PRIVATE_TRACE_AND_CREDENTIAL" not in file.read_text()
    assert not (setup[0] / "output" / setup[2]["tasks"][0]["task_id"] / "single/launch.json").exists()


def test_ordinary_failure_with_confirmed_cleanup_does_not_cancel_other_tasks(setup, monkeypatch):
    captured = []

    class Runner:
        def __init__(self, **kw):
            pass

        def __call__(self, request):
            captured.append(request)
            return TaskResult(status="failed", score=.25,
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    result = run(setup, parallelism=1)
    assert result["terminal"] == 4 and result["state"] == "completed"
    assert not result["requires_user_action"] and all(row["score"] == .25 for row in result["rows"])
    assert [(r.task.task_id, r.output_dir.name) for r in captured] == [
        (task["task_id"], arm) for task in setup[2]["tasks"] for arm in ("incumbent", "candidate")]


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["inference"].update(max_model_calls=1000), "max_model_calls=null"),
    (lambda p: p["inference"].pop("max_model_calls"), "max_model_calls=null"),
    (lambda p: p["inference"].update(memory_swap_mb=122880), "extra swap"),
    (lambda p: p.update(parallelism=3), "parallelism"),
    (lambda p: p.update(parallelism=True), "parallelism"),
    (lambda p: p.update(evolution_enabled=True), "does not evolve"),
    (lambda p: p["arms"][1].update(name="incumbent"), "Arm names"),
    (lambda p: p["arms"][1].update(name="../other"), "Arm names"),
    (lambda p: p["arms"][1].update(policy_version=False), "policy_version"),
    (lambda p: p["arms"][1].update(execution_mode="single"), "empty policy paths"),
    (lambda p: p["arms"][1].update(assignment_policy_path=None), "nonempty path"),
    (lambda p: p["tasks"].append(deepcopy(p["tasks"][0])), "Task IDs"),
    (lambda p: p["tasks"][0].update(task_id="other"), "Task IDs"),
    (lambda p: p.update(arms=[]), "nonempty lists"),
    (lambda p: p["evaluation"].update(docker_cpus=0), "positive integer"),
    (lambda p: p["inference"].update(team_message_delivery="instant"), "team_message_delivery"),
    (lambda p: p["inference"].update(runtime_disclosure="yes"), "runtime_disclosure"),
    (lambda p: p["inference"].update(leader_loop_recovery=1), "leader_loop_recovery"),
    # The recovery's only target is the check-in status file.
    (lambda p: p["inference"].update(leader_loop_recovery=True), "requires team_checkin_seconds"),
    # null/0 must never reach native settings: null there is a hard cap of 100.
    (lambda p: p["inference"].update(tool_call_cap=None), "tool_call_cap"),
    (lambda p: p["inference"].update(tool_call_cap=0), "tool_call_cap"),
])
def test_invalid_plan_never_constructs_runner_loads_provider_or_creates_output(setup, monkeypatch, mutation, match):
    update_profile(setup, mutation)
    monkeypatch.setattr(comparison, "ProgramBenchRunner", lambda **kw: pytest.fail("runner constructed"))
    with pytest.raises(ValueError, match=match):
        run(setup)
    assert setup[3] == [] and not (setup[0] / "output").exists()


@pytest.mark.parametrize("delivery", [None, "native_idle", "between_tool_calls"])
def test_recorded_team_message_delivery_is_the_one_every_runner_uses(setup, monkeypatch, delivery):
    if delivery:
        update_profile(setup, lambda p: p["inference"].update(team_message_delivery=delivery))
    used = []

    class Runner:
        def __init__(self, **options):
            used.append(options["team_message_delivery"])

        def __call__(self, request):
            return TaskResult(status="completed", score=.5, usage={"total_tokens": 1},
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    assert run(setup)["state"] == "completed"
    assert used == [delivery or "native_idle"] * 4
    settings = json.loads((setup[0] / "output/run_settings.json").read_text())
    assert settings["inference"].get("team_message_delivery") == delivery


@pytest.mark.parametrize("switches", [{}, {"runtime_disclosure": True},
                                      {"leader_loop_recovery": True, "team_checkin_seconds": 1800},
                                      {"runtime_disclosure": False, "leader_loop_recovery": False},
                                      {"tool_call_cap": "disabled"}, {"tool_call_cap": "native_default"}])
def test_recorded_runtime_switches_are_the_ones_every_runner_uses(setup, monkeypatch, switches):
    # 2026-10-02: disclosure and master loop recovery are independent opt-ins.
    if switches:
        update_profile(setup, lambda p: p["inference"].update(switches))
    used = []

    class Runner:
        def __init__(self, **options):
            used.append((options["runtime_disclosure"], options["leader_loop_recovery"], options["tool_call_cap"]))

        def __call__(self, request):
            return TaskResult(status="completed", score=.5, usage={"total_tokens": 1},
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    assert run(setup)["state"] == "completed"
    expected = (switches.get("runtime_disclosure", False), switches.get("leader_loop_recovery", False),
                switches.get("tool_call_cap", "native_default"))
    assert used == [expected] * 4
    settings = json.loads((setup[0] / "output/run_settings.json").read_text())
    for name in ("runtime_disclosure", "leader_loop_recovery", "tool_call_cap"):
        assert settings["inference"].get(name) == switches.get(name)  # Absent stays absent, never invented.


@pytest.mark.parametrize("inference", [{}, {"observation_interval_seconds": None},
                                       {"observation_interval_seconds": 60},
                                       {"observation_interval_seconds": 0.5}])
def test_observation_interval_is_opt_in_and_reaches_every_runner(setup, monkeypatch, inference):
    update_profile(setup, lambda p: p["inference"].update(inference))
    used = []

    class Runner:
        def __init__(self, **options):
            used.append({key: options[key] for key in inference})
            if not inference:
                assert "observation_interval_seconds" not in options

        def __call__(self, request):
            return TaskResult(status="completed", score=0,
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    assert run(setup)["state"] == "completed"
    assert used == [inference] * 4
    settings = json.loads((setup[0] / "output/run_settings.json").read_text())
    assert {key: value for key, value in settings["inference"].items()
            if key == "observation_interval_seconds"} == inference


@pytest.mark.parametrize("interval", [0, -1, True, "60", float("nan"), float("inf"), -float("inf")])
def test_invalid_observation_interval_does_not_execute(setup, monkeypatch, interval):
    update_profile(setup, lambda p: p["inference"].update(observation_interval_seconds=interval))
    monkeypatch.setattr(comparison, "ProgramBenchRunner", lambda **kw: pytest.fail("runner constructed"))
    with pytest.raises(ValueError, match="observation_interval_seconds"):
        run(setup)
    assert setup[3] == [] and not (setup[0] / "output").exists()


def test_explicit_profile_round_reaches_request_launch_result_and_progress(setup, monkeypatch):
    update_profile(setup, lambda p: p.update(repeat_index=2))
    captured = []

    class Runner:
        def __init__(self, **kwargs):
            pass

        def __call__(self, request):
            captured.append(request)
            return TaskResult(status="failed", score=0, metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    result = run(setup, repeat_index=2)
    assert len(captured) == 4  # The round label does not schedule extra runs.
    settings = json.loads((setup[0] / "output/run_settings.json").read_text())
    assert settings["repeat_index"] == 2
    for request in captured:
        expected = {"repeat_index": 2, "pair_id": f"{request.task.task_id}#r2"}
        assert request.repeat_index == 2 and request.pair_id == expected["pair_id"]
        for name in ("run_request.json", "launch.json", "task_result.json"):
            saved = json.loads((request.output_dir / name).read_text())
            assert {key: saved[key] for key in expected} == expected
        task = next(task for task in settings["tasks"] if task["task_id"] == request.task.task_id)
        assert {key: task[key] for key in expected} == expected
    assert all(row["repeat_index"] == 2 and row["score"] == 0 and row["status"] == "failed"
               for row in result["rows"])


def test_round_and_single_task_pair_cli_overrides_are_read_only(setup, monkeypatch, capsys):
    update_profile(setup, lambda p: p.update(tasks=p["tasks"][:1], repeat_index=1, pair_id="profile-pair"))
    monkeypatch.setattr(comparison, "ProgramBenchRunner", lambda **kwargs: pytest.fail("runner constructed"))
    assert comparison.main(["--profile", str(setup[1]), "--project-root", str(setup[0]),
                            "--repeat-index", "3", "--pair-id", "explicit-pair"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["repeat_index"] == plan["tasks"][0]["repeat_index"] == 3
    assert plan["tasks"][0]["pair_id"] == "explicit-pair"
    assert setup[3] == []


def test_per_task_pair_ids_and_explicit_round_zero(setup):
    update_profile(setup, lambda p: p["tasks"][0].update(pair_id="chosen-pair"))
    plan = comparison.prepare_comparison(setup[1], setup[0], repeat_index=0).plan
    assert plan["tasks"][0]["pair_id"] == "chosen-pair"
    assert plan["tasks"][1]["pair_id"] == f"{plan['tasks'][1]['task_id']}#r0"


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(repeat_index=-1), "repeat_index"),
    (lambda p: p.update(repeat_index=True), "repeat_index"),
    (lambda p: p.update(repeat_index=1.5), "repeat_index"),
    (lambda p: p.update(repeat_index=None), "repeat_index"),
    (lambda p: p.update(pair_id="one-pair-for-two-tasks"), "exactly one task"),
    (lambda p: p["tasks"][0].update(pair_id=""), "pair_id"),
    (lambda p: p["tasks"][0].update(pair_id=None), "pair_id"),
])
def test_invalid_repeat_identity_does_not_execute(setup, monkeypatch, mutation, match):
    update_profile(setup, mutation)
    monkeypatch.setattr(comparison, "ProgramBenchRunner", lambda **kwargs: pytest.fail("runner constructed"))
    with pytest.raises(ValueError, match=match):
        comparison.prepare_comparison(setup[1], setup[0])
    assert setup[3] == [] and not (setup[0] / "output").exists()


def test_paid_cli_requires_explicit_index_even_with_profile_default(setup, monkeypatch, capsys):
    update_profile(setup, lambda p: p.update(repeat_index=1))
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **k: pytest.fail("execution attempted"))
    with pytest.raises(SystemExit) as exc:
        comparison.main(["--profile", str(setup[1]), "--run", "--output", "out",
                         "--qwen-package", "qwen", "--node-directory", "node"])
    assert exc.value.code == 2 and "explicit --repeat-index" in capsys.readouterr().err


@pytest.mark.parametrize("index", [None, True, -1, 1.5, "1"])
def test_callable_execution_requires_explicit_index_before_any_side_effect(setup, index):
    root, path, _, loaders = setup
    update_profile(setup, lambda p: p.update(repeat_index=1))
    with pytest.raises(ValueError, match="Explicit nonnegative repeat_index"):
        comparison.run_comparison(path, root, root / "output", qwen_package=root / "qwen",
                                  node_directory=root / "node", repeat_index=index)
    assert not loaders and not (root / "output").exists()


def test_total_protocol_rejects_zero_before_provider_output_or_launch_record(setup, monkeypatch):
    root, path, _, loaders = setup
    update_profile(setup, lambda p: p.update(selection_protocol=TOTAL_MODE))
    monkeypatch.setattr(comparison, "ProgramBenchRunner", lambda **kw: pytest.fail("runner constructed"))
    with pytest.raises(ValueError, match="positive repeat_index"):
        comparison.run_comparison(path, root, root / "output", qwen_package=root / "qwen",
                                  node_directory=root / "node", repeat_index=0)
    assert not loaders and not (root / "output").exists()
    assert not list(root.rglob("launch.json"))


@pytest.mark.parametrize("wave", [1, 2, 3, 4])
@pytest.mark.parametrize("index", [0, 5])
def test_first_round_wave_index_refuses_before_provider_output_or_launch_record(setup, monkeypatch, wave, index):
    root, path, _, loaders = setup
    update_profile(setup, lambda p: p.update(selection_protocol=TOTAL_MODE, first_round_wave=wave))
    monkeypatch.setattr(comparison, "ProgramBenchRunner", lambda **kw: pytest.fail("runner constructed"))
    with pytest.raises(ValueError, match="positive repeat_index|must equal first_round_wave"):
        comparison.run_comparison(path, root, root / "output", qwen_package=root / "qwen",
                                  node_directory=root / "node", repeat_index=index)
    assert not loaders and not (root / "output").exists()
    assert not list(root.rglob("launch.json"))


def test_comparison_preserves_utf8_crlf_policies_through_saved_request(setup, monkeypatch):
    root, path, _, _ = setup
    texts = ("分工 \r\nnext\t\r\n\r\n", "communication \r\nlast line\t")
    for name, text in zip(("assignment", "communication"), texts):
        (root / f"incumbent_{name}.md").write_bytes(text.encode("utf-8"))
    prepared = comparison.prepare_comparison(path, root)
    assert prepared.policies["incumbent"] == texts
    captured = []

    class Runner:
        def __init__(self, **kwargs):
            pass

        def __call__(self, request):
            captured.append(request)
            return TaskResult(status="completed", metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    run(setup, repeat_index=1)
    for request in captured:
        if request.output_dir.name == "incumbent":
            saved = json.loads((request.output_dir / "run_request.json").read_text())
            assert (request.assignment_policy, request.communication_policy) == texts
            assert (saved["assignment_policy"], saved["communication_policy"]) == texts


def test_external_single_team_profile_is_saved_without_changing_deployment_manifest(setup, monkeypatch):
    root, _, profile, _ = setup
    (root / "qwen").mkdir()
    (root / "qwen/package.json").write_text(json.dumps({"version": "0.23.1"}))
    code = root / "deployment/code"
    code.mkdir(parents=True)
    for task in profile["tasks"]:
        shutil.copytree(root / task["path"], code / task["path"])
        shutil.copyfile(root / task["instructions"], code / task["instructions"])
    for file in root.glob("*_*.md"):
        shutil.copyfile(file, code / file.name)
    source = code / "agent_company/synthetic.py"
    source.parent.mkdir()
    source.write_bytes(b"synthetic source\n")
    script = code / "scripts/run_programbench_policy_comparison.py"
    script.parent.mkdir()
    script.write_bytes(b"# synthetic deployment script location\n")
    monkeypatch.setattr(comparison, "__file__", str(script))
    files = sorted(p for p in code.rglob("*") if p.is_file())
    manifest = code.parent / "code_sha256.txt"
    manifest.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(code)}\n" for p in files))
    before = code_version(code)
    assert before is not None
    external = root / "run_profiles/confirmation.json"
    external.parent.mkdir()
    changed = deepcopy(profile)
    changed.update(tasks=profile["tasks"][:1], arms=profile["arms"][:1], parallelism=1,
                   selection_protocol=TOTAL_MODE)
    external.write_bytes((json.dumps(changed, indent=2) + "\r\n").encode("utf-8"))
    prepared = comparison.prepare_comparison(external, code, repeat_index=5)
    assert prepared.plan["new_rollouts"] == 1
    assert prepared.plan["selection_protocol"] == TOTAL_MODE
    assert prepared.plan["profile_path"] == str(external)

    class Runner:
        def __init__(self, **kwargs):
            pass

        def __call__(self, request):
            passed = 10 if request.repeat_index == 5 else 20
            filtered = request.output_dir / "verifier/filtered_test_results.json"
            filtered.parent.mkdir()
            filtered.write_text(json.dumps({
                "schema_version": 1, "task_id": request.task.task_id, "official_status": "completed",
                "reward": passed / 100, "valid_tests": 100,
                "tests": [{"test_id": str(index)} for index in range(100)],
            }))
            return TaskResult(status="completed", score=passed / 100, metrics={
                "inference_cleanup_confirmed": True,
                "verifier": {"status": "completed", "reward": passed / 100,
                             "passed_tests": passed, "valid_tests": 100},
            })

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    output = root / "confirmation_evidence"
    summary = comparison.run_comparison(external, code, output, qwen_package=root / "qwen",
                                         node_directory=root / "node", repeat_index=5)
    assert summary["state"] == "completed" and summary["terminal"] == 1
    assert (output / "profile.json").read_bytes() == external.read_bytes()
    assert json.loads((output / "run_settings.json").read_text())["code_version"] == before
    second_output = root / "another_wave_evidence"
    second_summary = comparison.run_comparison(external, code, second_output, qwen_package=root / "qwen",
                                                node_directory=root / "node", repeat_index=6)
    assert second_summary["state"] == "completed" and second_summary["terminal"] == 1
    assert json.loads((second_output / "run_settings.json").read_text())["code_version"] == before
    task_id = profile["tasks"][0]["task_id"]
    report = compare_trials(output / task_id / "incumbent", second_output / task_id / "incumbent")
    assert report["decision"] == "needs_confirmation"
    assert report["test_universe_matched"] is True
    assert code_version(code) == before


@pytest.mark.parametrize("limit", [None, 32])
def test_the_configured_teammate_limit_reaches_every_runner(setup, monkeypatch, limit):
    # Review 10-18 F8: the plan dropped inference.teammate_limit, so every runner used the default 10.
    from agent_company.programbench import runner as runner_module
    if limit is not None:
        update_profile(setup, lambda p: p["inference"].update(teammate_limit=limit))
    used = []

    class Runner:
        def __init__(self, **options):
            used.append(runner_module.ProgramBenchRunner(**options).teammate_limit)

        def __call__(self, request):
            return TaskResult(status="completed", score=.5, usage={"total_tokens": 1},
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    assert run(setup)["state"] == "completed"
    assert used == [limit or 10] * 4
    settings = json.loads((setup[0] / "output/run_settings.json").read_text())
    assert settings["inference"].get("teammate_limit") == limit  # absent stays absent
