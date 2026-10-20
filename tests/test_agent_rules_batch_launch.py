"""agent_rules_v1 batch 1 (Single and the M0 Team) and batch 2 (M1 and M0 Teams) through the real planner,
runner and launcher; zero API."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from agent_company.deploy import launch
from agent_company.policy.role_package import AGENT_RULES_PROTOCOL, create_m0_package
from agent_company.programbench.comparison import prepare_comparison
from scripts import prepare_programbench_agent_rules
from test_programbench_formal_launch import rig as _formal_rig, write, forbid_mutation_and_credentials
from test_programbench_runner import rig as _runner_rig

formal_rig = _formal_rig
runner_rig = _runner_rig
ROOT = Path(__file__).resolve().parents[1]
M0 = ROOT / "configs/stage1/policies/agent_rules_m0_20261017"


@pytest.fixture
def rules_rig(formal_rig):
    name = "agent_rules_pandoc_batch1.template.json"
    (formal_rig.configs / name).write_bytes((ROOT / "configs/stage1" / name).read_bytes())
    shutil.copytree(M0, formal_rig.configs / "policies/agent_rules_m0_20261017")
    return formal_rig


def formal_spec(rig, batch="1", index="1"):
    prepared = prepare_programbench_agent_rules.prepare_inputs(rig.deployment.code, rig.deployment.final / "profiles")
    return {"runner": "comparison", "step": "agent_rules", "profile": prepared["profile"],
            "arguments": ["--repeat-index", index, "--batch-index", batch]}


def test_preparation_uses_the_approved_m0_and_creates_no_candidate(rules_rig):
    result = prepare_programbench_agent_rules.prepare_inputs(rules_rig.deployment.code, rules_rig.deployment.final / "profiles")
    package = json.loads(Path(result["m0_package"]).read_text())
    assert package == create_m0_package((M0 / "assignment.md").read_text(), (M0 / "communication.md").read_text())
    profile = json.loads(Path(result["profile"]).read_text())
    assert {arm["policy_package_path"] for arm in profile["arms"]} == {result["m0_package"]}
    assert result["candidate_created"] is False
    with pytest.raises(ValueError, match="outside"):
        prepare_programbench_agent_rules.prepare_inputs(rules_rig.deployment.code, rules_rig.deployment.code / "profiles")


def test_the_planner_builds_single_and_the_m0_team_and_refuses_them_as_batch_two(rules_rig):
    spec = formal_spec(rules_rig)
    plan = prepare_comparison(Path(spec["profile"]), rules_rig.deployment.code, repeat_index=1, batch_index=1).plan
    assert [(a["arm_id"], a["execution_mode"], a.get("role_policy_mode"), a["policy_version"]) for a in plan["arms"]] == [
        ("single", "single", None, 0), ("team_m0", "team", "propose", 0)]
    assert plan["inference"]["role_policy_protocol"] == AGENT_RULES_PROTOCOL and plan["batch_index"] == 1
    assert plan["arms"][0]["policy_package"] == plan["arms"][1]["policy_package"]
    profile = json.loads(Path(spec["profile"]).read_text())
    profile["batch_index"] = 2
    write(Path(spec["profile"]), profile)
    with pytest.raises(ValueError, match="prescribed arms"):
        prepare_comparison(Path(spec["profile"]), rules_rig.deployment.code, repeat_index=2)


def test_formal_prepare_is_readonly_and_names_its_own_batch(rules_rig, monkeypatch):
    spec = formal_spec(rules_rig)
    forbid_mutation_and_credentials(rules_rig, monkeypatch)
    result = launch.prepare(rules_rig.deployment, spec)
    assert result["name"] == "agent_rules_batch1_m0" and result["batch_index"] == result["repeat_index"] == 1
    assert result["runtime"]["new_arm_count"] == 2 and result["runtime"]["satisfied"]
    assert "--batch-index" in result["command"] and not Path(result["directory"]).exists()


@pytest.mark.parametrize("case", ["missing_batch", "batch_two", "wrong_index", "outside_profile", "outside_package",
                                  "changed_m0"])
def test_formal_refusals_precede_allocation_and_credentials(rules_rig, monkeypatch, case):
    spec = formal_spec(rules_rig)
    profile_path = Path(spec["profile"])
    profile = json.loads(profile_path.read_text())
    if case == "missing_batch":
        spec["arguments"] = ["--repeat-index", "1"]
    if case == "batch_two":
        spec["arguments"] = ["--repeat-index", "2", "--batch-index", "2"]
    if case == "wrong_index":
        spec["arguments"][1] = "2"
    if case == "outside_profile":
        path = rules_rig.configs / "external_agent_rules_profile.json"
        path.write_bytes(profile_path.read_bytes())
        spec["profile"] = str(path)
    if case in ("outside_package", "changed_m0"):
        package = json.loads(Path(profile["arms"][0]["policy_package_path"]).read_text())
        if case == "changed_m0":
            package["team"]["communication"] += "\nAn edited line."
        path = (rules_rig.configs if case == "outside_package" else profile_path.parent) / "other_package.json"
        write(path, package)
        for arm in profile["arms"]:
            arm["policy_package_path"] = str(path)
        write(profile_path, profile)
    payload = forbid_mutation_and_credentials(rules_rig, monkeypatch)
    # batch_two and wrong_index are already refused by the runner's read-only planning.
    reasons = {"missing_batch": "batch 1 or 2, with repeat-index equal to the batch", "batch_two": "repeat_index must equal batch_index",
               "wrong_index": "repeat_index must equal batch_index", "outside_profile": "belongs in profiles/",
               "outside_package": "belong in this deployment's profiles/", "changed_m0": "approved M0 texts"}
    with pytest.raises(ValueError, match=reasons[case]):
        launch.launch(rules_rig.deployment, spec, payload)
    assert not (rules_rig.deployment.final / "paid_runs/agent_rules_batch1_m0").exists()


def test_formal_launch_records_the_package_hashes_and_starts_one_controller(rules_rig, monkeypatch):
    spec = formal_spec(rules_rig)
    spawned = []

    def start(command, **options):
        assert not Path(command[command.index("--output") + 1]).exists()
        spawned.append(command)
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(subprocess, "Popen", start)
    result = launch.launch(rules_rig.deployment, spec, rules_rig.payload)
    assert result["status"] == "started" and len(spawned) == 1
    saved = json.loads((rules_rig.deployment.final / "paid_runs/agent_rules_batch1_m0/launch.json").read_text())
    package = Path(json.loads(Path(spec["profile"]).read_text())["arms"][0]["policy_package_path"])
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    assert saved["batch_index"] == 1 and saved["reserved_arm_count"] == 2
    assert saved["policy_package_sha256_by_arm"] == {"single": digest, "team_m0": digest}


@pytest.mark.parametrize("single", [True, False])
def test_the_real_runner_gives_the_m0_package_to_both_arms_and_policy_text_only_to_the_team(runner_rig, single):
    rig = runner_rig
    m0 = create_m0_package((M0 / "assignment.md").read_text(), (M0 / "communication.md").read_text())
    request = replace(rig.request, execution_mode="single" if single else "team", policy_version=0,
                      role_policy_mode=None if single else "propose", policy_package=m0,
                      assignment_policy="" if single else m0["team"]["assignment"],
                      communication_policy="" if single else m0["team"]["communication"])
    runner = replace(rig.runner, role_policy_protocol=AGENT_RULES_PROTOCOL, role_policy_mode="propose",
                     team_checkin_seconds=1800, team_message_delivery="between_tool_calls")
    result = runner(request)
    assert result.status == "completed"
    assert rig.qwen_options["policy_package"] == m0
    assert rig.calls[0]["assignment"] == request.assignment_policy
    assert rig.calls[0]["communication"] == request.communication_policy
    assert result.metrics["policy_applied"] is (not single)



@pytest.fixture
def batch2_rig(rules_rig):
    name = "agent_rules_pandoc_batch2.template.json"
    (rules_rig.configs / name).write_bytes((ROOT / "configs/stage1" / name).read_bytes())
    # The recorded batch 1, as its runner writes it, in an earlier deployment beside this one.
    profile = Path(prepare_programbench_agent_rules.prepare_inputs(
        rules_rig.deployment.code, rules_rig.deployment.final.parent / "batch1_profiles")["profile"])
    plan = prepare_comparison(profile, rules_rig.deployment.code, repeat_index=1, batch_index=1).plan
    from agent_company.evolution.harness_capabilities import native_version
    settings = {**plan, "provider": launch.public_provider(Path(plan["provider_config"])),
                "qwen_version": native_version(rules_rig.deployment.qwen)}
    rules_rig.batch1 = rules_rig.deployment.final.parent / "earlier_deployment/paid_runs/agent_rules_batch1_m0/run"
    write(rules_rig.batch1 / "run_settings.json", settings)
    write(rules_rig.batch1 / "progress.json", {"state": "completed", "cleanup_attention": [], "terminal": 2})
    write(rules_rig.batch1 / plan["tasks"][0]["task_id"] / "team_m0/task_result.json",
          {"status": "completed", "score": 0.5425776754890679})
    return rules_rig


def reviewed_m1(rig, **changes):
    m0 = create_m0_package((M0 / "assignment.md").read_text(), (M0 / "communication.md").read_text())
    m1 = {**create_m0_package("# Assignment principles\n\nA1", "# Communication principles\n\nC1"),
          "package_id": "M1", "revision": 1, "parent": {"package_id": m0["package_id"], "revision": 0},
          "review_ref": "review_3.json", **changes}
    generation = rig.configs / "generation"
    generation.mkdir(exist_ok=True)
    write(generation / "package.json", m1)
    write(generation / "review_3.json", {"recommendation": "prefer_candidate", "status": "completed"})
    return generation / "package.json"


def batch2_spec(rig, **changes):
    prepared = prepare_programbench_agent_rules.prepare_inputs(
        rig.deployment.code, rig.deployment.final / "profiles", batch=2, candidate=reviewed_m1(rig, **changes))
    return {"runner": "comparison", "step": "agent_rules", "profile": prepared["profile"],
            "arguments": ["--repeat-index", "2", "--batch-index", "2"]}, prepared


def test_batch_two_inputs_copy_only_a_reviewed_m1_of_m0_and_its_review(batch2_rig):
    spec, prepared = batch2_spec(batch2_rig)
    source = batch2_rig.configs / "generation"
    assert Path(prepared["m1_package"]).read_bytes() == (source / "package.json").read_bytes()
    assert Path(prepared["m1_review"]).read_bytes() == (source / "review_3.json").read_bytes()
    assert "m0_package" not in prepared and prepared["batch"] == 2
    profile = json.loads(Path(spec["profile"]).read_text())
    assert [(a["name"], a["policy_package_path"]) for a in profile["arms"]] == [("team_m1", prepared["m1_package"])]
    assert profile["parallelism"] == 1
    with open(ROOT / "configs/stage1/agent_rules_pandoc_batch1.template.json") as one:
        assert profile["inference"] == json.load(one)["inference"]
    m0_team = create_m0_package((M0 / "assignment.md").read_text(), (M0 / "communication.md").read_text())["team"]
    write(source / "review_failed.json", {"status": "failed"})
    for changes, case in (({"review_ref": None}, "unreviewed"), ({"review_ref": "missing.json"}, "no_review_file"),
                          ({"review_ref": "review_failed.json"}, "failed_review"),
                          ({"review_ref": "../review_3.json"}, "review_path"),
                          ({"parent": {"package_id": "M9", "revision": 0}}, "parent"), ({"team": m0_team}, "same")):
        target = batch2_rig.deployment.final / f"profiles_{case}"
        with pytest.raises(ValueError, match="reviewed revision 1 of M0"):
            prepare_programbench_agent_rules.prepare_inputs(batch2_rig.deployment.code, target, batch=2,
                                                            candidate=reviewed_m1(batch2_rig, **changes))
        assert not target.exists()
    with pytest.raises(ValueError, match="only batch 2"):
        prepare_programbench_agent_rules.prepare_inputs(batch2_rig.deployment.code, batch2_rig.deployment.final / "x",
                                                        candidate=reviewed_m1(batch2_rig))


def test_batch_two_plans_the_m1_team_alone_bound_to_batch_one(batch2_rig, monkeypatch):
    spec, _ = batch2_spec(batch2_rig)
    plan = prepare_comparison(Path(spec["profile"]), batch2_rig.deployment.code, repeat_index=2, batch_index=2).plan
    assert [(a["arm_id"], a["execution_mode"], a.get("role_policy_mode"), a["policy_version"]) for a in plan["arms"]] == [
        ("team_m1", "team", "propose", 1)]
    assert plan["batch_index"] == 2 and plan["parallelism"] == 1 and plan["inference"]["teammate_limit"] == 16
    forbid_mutation_and_credentials(batch2_rig, monkeypatch)
    result = launch.prepare(batch2_rig.deployment, spec)
    assert result["name"] == "agent_rules_batch2_m1" and result["batch_index"] == result["repeat_index"] == 2
    assert result["runtime"]["new_arm_count"] == 1 and not Path(result["directory"]).exists()
    assert result["input_runs"] == [str(batch2_rig.batch1.resolve())]


CTAGS = "universal-ctags__ctags.243595e"


def test_batch_one_inputs_for_another_task_change_only_the_task_entry(rules_rig):
    task = rules_rig.deployment.code / prepare_programbench_agent_rules.TASKS / CTAGS
    task.mkdir(parents=True)
    (task / "task.yaml").write_text("repository: universal-ctags/ctags\ncommit: '243595e" + "0" * 33 + "'\n")
    pandoc = prepare_programbench_agent_rules.prepare_inputs(rules_rig.deployment.code, rules_rig.deployment.final / "a")
    ctags = prepare_programbench_agent_rules.prepare_inputs(rules_rig.deployment.code, rules_rig.deployment.final / "b",
                                                            task_id=CTAGS)
    assert Path(ctags["profile"]).name == f"agent_rules_{CTAGS}_batch1.json" and ctags["task_id"] == CTAGS
    first, second = (json.loads(Path(r["profile"]).read_text()) for r in (pandoc, ctags))
    assert second["tasks"] == [{**first["tasks"][0], "task_id": CTAGS,
                                "path": f"{prepare_programbench_agent_rules.TASKS}/{CTAGS}"}]
    assert {k: v for k, v in second.items() if k not in ("tasks", "arms")} == {
        k: v for k, v in first.items() if k not in ("tasks", "arms")}
    plan = prepare_comparison(Path(ctags["profile"]), rules_rig.deployment.code, repeat_index=1, batch_index=1).plan
    assert [t["task_id"] for t in plan["tasks"]] == [CTAGS]
    for bad in ("../x", "unknown__task.1"):
        with pytest.raises(ValueError, match="Unknown ProgramBench task"):
            prepare_programbench_agent_rules.prepare_inputs(rules_rig.deployment.code, rules_rig.deployment.final / bad,
                                                            task_id=bad)


def test_batch_two_ignores_a_recorded_batch_one_of_another_task(batch2_rig, monkeypatch):
    other = json.loads((batch2_rig.batch1 / "run_settings.json").read_text())
    other["tasks"][0]["task_id"] = CTAGS
    write(batch2_rig.deployment.final.parent / "ctags_deployment/paid_runs/agent_rules_batch1_m0/run/run_settings.json",
          other)
    spec, _ = batch2_spec(batch2_rig)
    forbid_mutation_and_credentials(batch2_rig, monkeypatch)
    result = launch.prepare(batch2_rig.deployment, spec)
    assert result["input_runs"] == [str(batch2_rig.batch1.resolve())]


@pytest.mark.parametrize("case", ["changed_inference", "no_batch_one", "two_batch_ones", "other_m0",
                                  "review_missing", "review_failed", "two_arms", "batch_one_unfinished",
                                  "batch_one_unscored", "other_qwen", "other_instructions"])
def test_batch_two_refusals_precede_allocation_and_credentials(batch2_rig, monkeypatch, case):
    spec, prepared = batch2_spec(batch2_rig)
    profile_path = Path(spec["profile"])
    profile = json.loads(profile_path.read_text())
    settings_path = batch2_rig.batch1 / "run_settings.json"
    if case == "changed_inference":
        profile["inference"]["team_checkin_seconds"] = 600
        write(profile_path, profile)
    if case == "no_batch_one":
        settings_path.unlink()
    if case == "two_batch_ones":
        write(batch2_rig.deployment.final.parent / "another/paid_runs/agent_rules_batch1_m0/run/run_settings.json",
              json.loads(settings_path.read_text()))
    if case == "other_m0":
        settings = json.loads(settings_path.read_text())
        settings["arms"][1]["policy_package"]["package_id"] = "M0b"
        write(settings_path, settings)
    if case == "review_missing":
        Path(prepared["m1_review"]).unlink()
    if case == "review_failed":
        write(Path(prepared["m1_review"]), {"status": "failed"})
    if case in ("batch_one_unfinished", "batch_one_unscored", "other_qwen", "other_instructions"):
        settings = json.loads(settings_path.read_text())
        if case == "batch_one_unfinished":
            write(batch2_rig.batch1 / "progress.json", {"state": "running", "cleanup_attention": []})
        if case == "batch_one_unscored":
            write(batch2_rig.batch1 / settings["tasks"][0]["task_id"] / "team_m0/task_result.json",
                  {"status": "verifier_error", "score": None})
        if case == "other_qwen":
            settings["qwen_version"] = "0.0.1"
        if case == "other_instructions":
            other = batch2_rig.deployment.final.parent / "earlier_instructions.md"
            other.write_text("Other instructions.\n")
            settings["tasks"][0]["instructions"] = str(other)
        write(settings_path, settings)
    if case == "two_arms":
        profile["arms"].append({**profile["arms"][0], "name": "team_m1_again"})
        profile["parallelism"] = 2
        write(profile_path, profile)
    payload = forbid_mutation_and_credentials(batch2_rig, monkeypatch)
    reasons = {"changed_inference": "batch 1's task, inference", "no_batch_one": "exactly one recorded",
               "two_batch_ones": "exactly one recorded", "other_m0": "reviewed revision of batch 1's M0",
               "review_missing": "completed review", "review_failed": "completed review",
               "two_arms": "prescribed arms", "batch_one_unfinished": "not finished cleanly",
               "batch_one_unscored": "no completed official result", "other_qwen": "Qwen version",
               "other_instructions": "instruction text"}
    with pytest.raises(ValueError, match=reasons[case]):
        launch.launch(batch2_rig.deployment, spec, payload)
    assert not (batch2_rig.deployment.final / "paid_runs/agent_rules_batch2_m1").exists()


def test_batch_two_launch_records_the_m1_hash_and_batch_one(batch2_rig, monkeypatch):
    spec, prepared = batch2_spec(batch2_rig)
    monkeypatch.setattr(subprocess, "Popen", lambda command, **options: SimpleNamespace(pid=12345))
    result = launch.launch(batch2_rig.deployment, spec, batch2_rig.payload)
    assert result["status"] == "started"
    saved = json.loads((batch2_rig.deployment.final / "paid_runs/agent_rules_batch2_m1/launch.json").read_text())
    assert saved["batch_index"] == 2 and saved["reserved_arm_count"] == 1
    assert saved["policy_package_sha256_by_arm"] == {"team_m1": prepared["sha256"]["agent_rules_package_m1.json"]}
    assert saved["policy_review_sha256_by_arm"] == {"team_m1": prepared["sha256"]["review_3.json"]}
    assert saved["input_runs"] == [str(batch2_rig.batch1.resolve())]
