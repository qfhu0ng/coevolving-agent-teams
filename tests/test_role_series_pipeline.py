"""Role-series opt-in through the real planner/request/runner boundaries; zero API."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

from agent_company.deploy import launch
from agent_company.evolution.loop import TaskResult
from agent_company.policy.role_package import ROLE_POLICY_PROTOCOL, create_p0_package
from agent_company.programbench.total_evidence import trial_record
from scripts import prepare_programbench_role_series, run_programbench_policy_comparison as comparison
from test_programbench_policy_comparison import setup as _comparison_setup
from test_programbench_runner import rig as _runner_rig
from test_programbench_formal_launch import rig as _formal_rig, write, forbid_mutation_and_credentials
from test_programbench_total_evidence import trial

comparison_setup = _comparison_setup
runner_rig = _runner_rig
formal_rig = _formal_rig


def packages(root):
    p0 = create_p0_package("Assign work.\r\n", "Report changes.\r\n")
    v1 = deepcopy(p0)
    v1.update(package_id="v1", revision=1, parent={"package_id": p0["package_id"], "revision": 0})
    v1["roles"]["verifier"]["policy"] = "After a failed check, report the failure to its owner."
    for name, package in (("p0", p0), ("v1", v1)):
        write(root / f"role_package_{name}.json", package)
    return p0, v1


def series_profile(setup, batch=2):
    root, path, old, _ = setup
    p0, v1 = packages(root)
    profile = deepcopy(old)
    profile["tasks"] = profile["tasks"][:1]
    profile.update(batch_index=batch, selection_protocol="programbench_total_passed_confirmation")
    profile["inference"].update(role_policy_protocol=ROLE_POLICY_PROTOCOL, role_policy_mode="propose")
    profile["arms"] = ([{"name": "single", "execution_mode": "single", "policy_version": 0,
                         "policy_package_path": "role_package_p0.json"},
                        {"name": "team_p0_propose", "execution_mode": "team", "policy_version": 0,
                         "role_policy_mode": "propose", "policy_package_path": "role_package_p0.json"}]
                       if batch == 1 else [
                           {"name": "team_v1_propose", "execution_mode": "team", "policy_version": 1,
                            "role_policy_mode": "propose", "policy_package_path": "role_package_v1.json"},
                           {"name": "team_p0_off", "execution_mode": "team", "policy_version": 0,
                            "role_policy_mode": "off", "policy_package_path": "role_package_p0.json"}])
    write(path, profile)
    return profile, (p0, v1)


@pytest.mark.parametrize("batch", [1, 2])
def test_real_comparison_builds_both_concurrent_requests_and_saved_package_conditions(comparison_setup, monkeypatch, batch):
    root, path, _, loaders = comparison_setup
    profile, (p0, v1) = series_profile(comparison_setup, batch)
    barrier = threading.Barrier(2)
    calls = []

    class Runner:
        def __init__(self, **options):
            assert options["role_policy_protocol"] == ROLE_POLICY_PROTOCOL

        def __call__(self, request):
            calls.append(request)
            barrier.wait(timeout=5)
            write(request.output_dir / "verifier/filtered_test_results.json", {
                "schema_version": 1, "task_id": request.task.task_id, "official_status": "completed",
                "reward": 0.5, "valid_tests": 2, "tests": [{"test_id": "a"}, {"test_id": "b"}]})
            write(request.output_dir / request.execution_mode / "launch.json", {"stop_hook": None,
                "role_policy_protocol": ROLE_POLICY_PROTOCOL, "policy_package": request.policy_package,
                "role_policy_mode": request.role_policy_mode})
            return TaskResult("completed", 0.5, metrics={"inference_cleanup_confirmed": True,
                "verifier": {"status": "completed", "reward": 0.5, "passed_tests": 1, "valid_tests": 2}})

    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    output = root / "output"
    result = comparison.run_comparison(path, root, output, qwen_package=root / "qwen",
        node_directory=root / "node", repeat_index=batch, batch_index=batch)
    assert result["state"] == "completed" and len(calls) == 2 and len(loaders) == 1
    by_mode = {request.execution_mode: request for request in calls}
    assert len({request.pair_id for request in calls}) == 1
    assert {request.repeat_index for request in calls} == {batch}
    settings = json.loads((output / "run_settings.json").read_text())
    assert settings["batch_index"] == batch and len(settings["waves"]) == 1
    for request in calls:
        assert json.loads((request.output_dir / "policy_package.json").read_text()) == request.policy_package
        assert json.loads((request.output_dir / "run_request.json").read_text())["policy_package"] == request.policy_package
        record = trial_record(request.output_dir)
        if request.execution_mode == "single":
            assert request.assignment_policy == request.communication_policy == ""
            assert request.role_policy_mode is None and request.policy_package == p0
            assert record["condition"]["policy_package"] == {"protocol": p0["protocol"]}
        else:
            assert request.assignment_policy.endswith("\r\n")
            assert record["condition"]["role_policy_mode"] == request.role_policy_mode
            assert record["condition"]["policy_package"]["roles"] == request.policy_package["roles"]
    if batch == 1:
        assert by_mode["single"].policy_package == by_mode["team"].policy_package
    else:
        assert {r.role_policy_mode for r in calls} == {"off", "propose"}
        assert next(r for r in calls if r.role_policy_mode == "propose").policy_package == v1
    # The comparison completes exactly this batch; no follow-up dispatch exists.
    assert len(calls) == len(profile["arms"])


@pytest.mark.parametrize("change", ["global", "single_override", "legacy_override", "policy_paths", "resources", "batch_shape"])
def test_invalid_role_arm_plan_refuses_before_provider_or_output(comparison_setup, change):
    root, path, _, loaders = comparison_setup
    profile, _ = series_profile(comparison_setup, 1)
    if change == "global":
        profile["arms"][1]["role_policy_mode"] = "global"
    if change == "single_override":
        profile["arms"][0]["role_policy_mode"] = "off"
    if change == "legacy_override":
        profile["inference"].pop("role_policy_protocol")
    if change == "policy_paths":
        profile["arms"][1]["assignment_policy_path"] = "extra.md"
    if change == "resources":
        profile["arms"][1]["inference"] = {"cpus": 100}
    if change == "batch_shape":
        profile["parallelism"] = 1
    write(path, profile)
    with pytest.raises(ValueError):
        comparison.run_comparison(path, root, root / "output", qwen_package=root / "qwen",
                                  node_directory=root / "node", repeat_index=1, batch_index=1)
    assert not loaders and not (root / "output").exists()


def test_role_series_function_requires_explicit_batch_and_matching_repeat(comparison_setup):
    root, path, _, loaders = comparison_setup
    series_profile(comparison_setup, 2)
    for args in ({"repeat_index": 2}, {"repeat_index": 1, "batch_index": 2}, {"repeat_index": 2, "batch_index": 1}):
        with pytest.raises(ValueError):
            comparison.run_comparison(path, root, root / "output", qwen_package=root / "qwen",
                                      node_directory=root / "node", **args)
    assert not loaders and not (root / "output").exists()


@pytest.mark.parametrize("mode", ["single", "off", "propose"])
def test_real_programbench_runner_passes_package_mode_and_only_team_policy_text(runner_rig, mode):
    rig = runner_rig
    p0 = create_p0_package("Assignment policy", "Communication policy")
    single = mode == "single"
    request = replace(rig.request, execution_mode="single" if single else "team",
        policy_version=0, role_policy_mode=None if single else mode, policy_package=p0,
        assignment_policy="" if single else p0["team"]["assignment"],
        communication_policy="" if single else p0["team"]["communication"])
    runner = replace(rig.runner, role_policy_protocol=ROLE_POLICY_PROTOCOL, role_policy_mode="propose")
    result = runner(request)
    assert result.status == "completed"
    assert rig.qwen_options["policy_package"] == p0
    assert rig.qwen_options["role_policy_mode"] == ("propose" if single else mode)
    assert rig.calls[0]["assignment"] == request.assignment_policy
    assert rig.calls[0]["communication"] == request.communication_policy
    assert result.metrics["policy_applied"] is (not single)


def formal_series(rig, batch):
    inputs = rig.deployment.final / "profiles"
    prepared = prepare_programbench_role_series.prepare_inputs(rig.deployment.code, inputs)
    p0 = json.loads(Path(prepared["p0_package"]).read_text())
    v1 = deepcopy(p0)
    v1.update(package_id="v1", revision=1, parent={"package_id": p0["package_id"], "revision": 0})
    write(inputs / "role_package_v1.json", v1)
    return {"runner": "comparison", "step": "role_series", "profile": prepared["profiles"][batch - 1],
            "arguments": ["--repeat-index", str(batch), "--batch-index", str(batch)]}


def record_first_batch(rig):
    """Persist the actual first-batch request/native launch shapes, without a rollout."""
    profile = rig.deployment.final / "profiles/role_series_pandoc_batch1.json"
    plan = comparison.prepare_comparison(profile, rig.deployment.code,
                                        repeat_index=1, batch_index=1).plan
    root = rig.deployment.final / "paid_runs/role_series_batch1_p0"
    write(root / "launch.json", {"repeat_index": 1, "status": "started", "pid": 98765})
    write(root / "run/run_settings.json", plan)
    write(root / "run/progress.json", {"state": "completed", "running": [], "cleanup_attention": []})
    for arm in plan["arms"]:
        trial = root / "run" / rig.deployment.task_id / arm["arm_id"]
        request = {key: arm.get(key) for key in ("execution_mode", "policy_version", "role_policy_mode", "policy_package")}
        request["repeat_index"] = 1
        write(trial / "run_request.json", request)
        write(trial / arm["execution_mode"] / "launch.json", {
            "role_policy_protocol": ROLE_POLICY_PROTOCOL,
            "role_policy_mode": arm.get("role_policy_mode"), "policy_package": arm["policy_package"]})
    return root


@pytest.fixture
def series_rig(formal_rig):
    root = Path(__file__).resolve().parents[1]
    for batch in (1, 2):
        name = f"role_series_pandoc_batch{batch}.template.json"
        (formal_rig.configs / name).write_bytes((root / "configs/stage1" / name).read_bytes())
    return formal_rig


@pytest.mark.parametrize("batch", [1, 2])
def test_formal_series_prepare_is_readonly_and_names_are_distinct_from_old_waves(series_rig, monkeypatch, batch):
    rig = series_rig
    spec = formal_series(rig, batch)
    if batch == 2:
        record_first_batch(rig)
    forbid_mutation_and_credentials(rig, monkeypatch)
    result = launch.prepare(rig.deployment, spec)
    assert result["name"] == ("role_series_batch1_p0" if batch == 1 else "role_series_batch2_v1_p0")
    assert result["batch_index"] == result["repeat_index"] == batch
    assert result["runtime"]["new_arm_count"] == 2
    assert result["runtime"]["satisfied"]
    assert "--batch-index" in result["command"]
    assert not Path(result["directory"]).exists()


@pytest.mark.parametrize("case", ["missing_batch", "wrong_batch", "wrong_index", "outside_profile", "outside_package"])
def test_formal_series_refusals_precede_allocation_and_credentials(series_rig, monkeypatch, case):
    rig = series_rig
    spec = formal_series(rig, 1)
    if case == "missing_batch":
        spec["arguments"] = ["--repeat-index", "1"]
    if case == "wrong_batch":
        spec["arguments"][-1] = "2"
    if case == "wrong_index":
        spec["arguments"][1] = "2"
    if case == "outside_profile":
        path = rig.configs / "external_role_profile.json"
        path.write_bytes(Path(spec["profile"]).read_bytes())
        spec["profile"] = str(path)
    if case == "outside_package":
        profile = json.loads(Path(spec["profile"]).read_text())
        path = rig.configs / "external_package.json"
        path.write_bytes(Path(profile["arms"][0]["policy_package_path"]).read_bytes())
        for arm in profile["arms"]:
            arm["policy_package_path"] = str(path)
        write(Path(spec["profile"]), profile)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError):
        launch.launch(rig.deployment, spec, payload)


def test_formal_batch_launch_stays_inside_existing_lock_and_does_not_chain(series_rig, monkeypatch):
    rig = series_rig
    spec = formal_series(rig, 1)
    held = []
    original = launch.prepare

    def checked(*args, **kwargs):
        # A second descriptor cannot allocate while the original prepare runs.
        with pytest.raises(BlockingIOError):
            with launch.allocation_lock(rig.deployment):
                pass
        held.append(True)
        return original(*args, **kwargs)

    spawned = []

    def start(command, **options):
        assert not Path(command[command.index("--output") + 1]).exists()
        spawned.append((command, options))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(launch, "prepare", checked)
    monkeypatch.setattr(subprocess, "Popen", start)
    result = launch.launch(rig.deployment, spec, rig.payload)
    assert result["status"] == "started" and held == [True] and len(spawned) == 1
    saved = json.loads((rig.deployment.final / "paid_runs/role_series_batch1_p0/launch.json").read_text())
    assert saved["batch_index"] == 1 and saved["reserved_arm_count"] == 2
    profile = json.loads(Path(spec["profile"]).read_text())
    assert saved["policy_package_sha256_by_arm"] == {
        arm["name"]: hashlib.sha256(Path(arm["policy_package_path"]).read_bytes()).hexdigest()
        for arm in profile["arms"]}
    assert not (rig.deployment.final / "paid_runs/role_series_batch2_v1_p0").exists()


def test_preparation_preserves_p0_and_never_invents_v1(series_rig):
    root = series_rig.deployment.code
    result = prepare_programbench_role_series.prepare_inputs(root, series_rig.deployment.final / "profiles")
    p0 = json.loads(Path(result["p0_package"]).read_text())
    for part in ("assignment", "communication"):
        assert p0["team"][part] == (root / f"configs/stage1/policies/fresh_p0_20261001/{part}_v000.md").read_bytes().decode("utf-8")
    assert all(role["policy"] == "" for role in p0["roles"].values())
    assert result["v1_created"] is False and not Path(result["v1_package"]).exists()


def package_trial(root, name, package, *, single=False):
    path = trial(root, name, "single" if single else "team", 50, single=single)
    request = json.loads((path / "run_request.json").read_text())
    request.update(policy_package=package, role_policy_mode=None if single else "off",
                   assignment_policy="" if single else package["team"]["assignment"],
                   communication_policy="" if single else package["team"]["communication"])
    write(path / "run_request.json", request)
    settings_path = path.parent.parent / "run_settings.json"
    settings = json.loads(settings_path.read_text())
    settings["inference"]["role_policy_protocol"] = ROLE_POLICY_PROTOCOL
    settings["arms"] = [{"arm_id": path.name, "execution_mode": request["execution_mode"],
                         "policy_version": 0, "policy_package": package,
                         **({"role_policy_mode": "off"} if not single else {})}]
    write(settings_path, settings)
    write(path / request["execution_mode"] / "launch.json", {
        "role_policy_protocol": ROLE_POLICY_PROTOCOL, "policy_package": package,
        "role_policy_mode": None if single else "off"})
    return path


def test_conditions_use_effective_mode_and_role_text_but_not_provenance(tmp_path):
    p0 = create_p0_package("A", "C")
    a = package_trial(tmp_path, "a", p0)
    copied = deepcopy(p0)
    copied["package_id"] = "copied-record"
    b = package_trial(tmp_path, "b", copied)
    assert trial_record(a)["condition"] == trial_record(b)["condition"]
    changed = deepcopy(p0)
    changed["roles"]["verifier"]["policy"] = "After a check, report its result."
    c = package_trial(tmp_path, "c", changed)
    assert trial_record(c)["condition"] != trial_record(a)["condition"]
    assert trial_record(c)["condition"]["role_policy_mode"] == "off"  # shared default was propose
    # Single never sees the role/A/C texts; those cannot invent a condition difference.
    sa = package_trial(tmp_path, "sa", p0, single=True)
    sb = package_trial(tmp_path, "sb", changed, single=True)
    assert trial_record(sa)["condition"] == trial_record(sb)["condition"]


@pytest.mark.parametrize("field", ["role_policy_mode", "policy_package", "assignment_policy"])
def test_trial_record_refuses_inconsistent_arm_inputs(tmp_path, field):
    p0 = create_p0_package("A", "C")
    path = package_trial(tmp_path, "a", p0)
    request = json.loads((path / "run_request.json").read_text())
    if field == "role_policy_mode":
        request[field] = "propose"
    elif field == "policy_package":
        request[field]["package_id"] = "not-the-saved-package"
    else:
        request[field] = "Different A"
    write(path / "run_request.json", request)
    with pytest.raises(ValueError):
        trial_record(path)


def test_formal_cli_forwards_explicit_batch_index_only_to_comparison(tmp_path, monkeypatch, capsys):
    from scripts import launch_programbench_formal as cli
    captured = []
    monkeypatch.setattr(cli, "prepare", lambda deployment, spec: captured.append(spec) or {"status": "prepared"})
    args = ["--final", str(tmp_path / "final"), "--remote", "--prepare", "--runner", "comparison",
            "--step", "role_series", "--profile", "profile.json", "--repeat-index", "2", "--batch-index", "2"]
    assert cli.main(args) == 0
    assert captured[0]["arguments"] == ["--repeat-index", "2", "--batch-index", "2"]
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        cli.main([*args[:args.index("--runner")], "--runner", "followup", "--source-run", "old",
                  "--step", "followup", "--profile", "profile.json", "--repeat-index", "2", "--batch-index", "2"])
    assert error.value.code == 2 and len(captured) == 1


@pytest.mark.parametrize("parent", [None, {"package_id": "other-p0", "revision": 0},
                                     {"package_id": "p0", "revision": 1}])
def test_batch_two_parent_must_be_its_off_p0_before_provider_or_output(comparison_setup, parent):
    root, path, _, loaders = comparison_setup
    _, (_, v1) = series_profile(comparison_setup, 2)
    v1["parent"] = parent
    write(root / "role_package_v1.json", v1)
    with pytest.raises(ValueError, match="v1 parent"):
        comparison.run_comparison(path, root, root / "output", qwen_package=root / "qwen",
                                  node_directory=root / "node", repeat_index=2, batch_index=2)
    assert not loaders and not (root / "output").exists()


@pytest.mark.parametrize("where", ["missing", "settings", "request", "native", "native_mode", "native_protocol"])
def test_batch_two_checks_first_batch_actual_p0_before_credentials(series_rig, monkeypatch, where):
    rig = series_rig
    spec = formal_series(rig, 2)
    if where != "missing":
        root = record_first_batch(rig) / "run"
        team = root / rig.deployment.task_id / "team_p0_propose"
        path = {"settings": root / "run_settings.json", "request": team / "run_request.json",
                "native": team / "team/launch.json", "native_mode": team / "team/launch.json",
                "native_protocol": team / "team/launch.json"}[where]
        data = json.loads(path.read_text())
        if where == "settings":
            data["arms"][1]["policy_package"]["package_id"] = "other-p0"
        elif where == "native_mode":
            data["role_policy_mode"] = "off"
        elif where == "native_protocol":
            data["role_policy_protocol"] = None
        else:
            data["policy_package"]["package_id"] = "other-p0"
        write(path, data)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        launch.launch(rig.deployment, spec, payload)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("field", ["role_policy_mode", "policy_package", "role_policy_protocol", "missing"])
def test_result_checks_actual_native_package_launch(tmp_path, single, field):
    path = package_trial(tmp_path, "sample", create_p0_package("A", "C"), single=single)
    native_path = path / ("single" if single else "team") / "launch.json"
    native = json.loads(native_path.read_text())
    if field == "missing":
        native.clear()
    elif field == "role_policy_mode":
        native[field] = "propose"
    elif field == "policy_package":
        native[field]["package_id"] = "unrelated"
    else:
        native[field] = "legacy"
    write(native_path, native)
    with pytest.raises(ValueError, match="Actual native launch"):
        trial_record(path)


def test_legacy_checkpoint_missing_new_default_fields_resumes(tmp_path):
    from agent_company.evolution.loop import RunRequest, TaskSpec, _run
    request = RunRequest(TaskSpec("task", "repo"), "train", 0, "C", "A", tmp_path / "trial")
    original = _run(request, lambda _: TaskResult("completed", 0.5))
    record_path = request.output_dir / "run_request.json"
    record = json.loads(record_path.read_text())
    for field in ("role_policy_mode", "policy_package"):
        record.pop(field)
    write(record_path, record)
    assert _run(request, lambda _: pytest.fail("resumed result ran again"), resume=True) == original
    with pytest.raises(ValueError, match="Task request differs"):
        _run(replace(request, role_policy_mode="propose"), lambda _: pytest.fail("changed request ran"), resume=True)
    with pytest.raises(ValueError, match="Task request differs"):
        _run(replace(request, policy_package=create_p0_package("A", "C")),
             lambda _: pytest.fail("changed package ran"), resume=True)


@pytest.mark.parametrize("kind", ["smoke", "probe"])
def test_legacy_smoke_and_probe_reject_role_protocol_before_side_effects(tmp_path, monkeypatch, capsys, kind):
    from scripts import s0_cache_probe, s0_native_smoke
    script = s0_native_smoke if kind == "smoke" else s0_cache_probe
    plan = {"inference": {"role_policy_protocol": ROLE_POLICY_PROTOCOL}}
    monkeypatch.setattr(script, "prepare_comparison", lambda *a, **k: SimpleNamespace(plan=plan))
    monkeypatch.setattr(script, "load_provider", lambda *a, **k: pytest.fail("provider loaded"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("output created"))
    argv = ["--profile", str(tmp_path / "profile.json"), "--output", str(tmp_path / "out")]
    if kind == "smoke":
        argv += ["--assets-root", str(tmp_path / "assets"), "--scripted"]
    with pytest.raises(SystemExit) as error:
        script.main(argv)
    assert error.value.code == 2
    assert "does not support role_policy_protocol" in capsys.readouterr().err
