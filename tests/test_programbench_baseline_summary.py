"""Descriptive reporting over synthetic files; never invoke model/runtime code."""
import json
from pathlib import Path
import socket
import subprocess

import pytest

from agent_company.analysis.baseline_summary import summarize_comparisons
from scripts import summarize_programbench_baseline as cli


@pytest.fixture(autouse=True)
def forbid_external_execution(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("external process attempted"))
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: pytest.fail("network attempted"))


def saved_round(tmp_path, repeat, scores, *, legacy=False, name=None):
    directory = tmp_path / (name or f"round-{repeat}")
    directory.mkdir()
    tasks = [{"task_id": task_id} for task_id in scores]
    if not legacy:
        for task in tasks:
            task.update(repeat_index=repeat, pair_id=f"{task['task_id']}#r{repeat}")
    settings = {"tasks": tasks,
                "arms": [{"arm_id": "single", "execution_mode": "single", "policy_version": 0},
                         {"arm_id": "team_p0", "execution_mode": "team", "policy_version": 0}],
                "inference": {"agent_timeout_sec": 21600}, "evaluation": {"docker_cpus": 10}}
    (directory / "run_settings.json").write_text(json.dumps(settings))
    for task in tasks:
        for arm, value in zip(("single", "team_p0"), scores[task["task_id"]]):
            if value == "missing_result":
                continue
            path = directory / task["task_id"] / arm
            path.mkdir(parents=True)
            result = {"status": "failed" if value == 0 else "completed", "score": value,
                      "trace_files": ["DO_NOT_READ"]}
            if not legacy:
                result.update(**task, arm_id=arm)
            (path / "task_result.json").write_text(json.dumps(result))
    return directory


def summarize(paths, **kwargs):
    return summarize_comparisons(paths, left_arm="single", right_arm="team_p0", **kwargs)


def record_conditions(directory, mutate=None):
    """Complete the settings fields the comparison runner records."""
    path = directory / "run_settings.json"
    settings = json.loads(path.read_text())
    settings.update(provider={"model": "synthetic-model", "base_url": "https://example.invalid"},
                    runtime="custom_Qwen_not_official_mini_SWE", platform="linux/amd64", parallelism=2,
                    project_root="/code", qwen_package="/code/qwen", node_directory="/node", docker="docker",
                    provider_config="/code/provider.json", profile_path=f"/profiles/{directory.name}.json",
                    waves=[[{"task_id": task["task_id"], "arm_id": arm["arm_id"]} for arm in settings["arms"]]
                           for task in settings["tasks"]])
    for arm in settings["arms"]:
        team = arm["execution_mode"] == "team"
        arm.update(assignment_policy_path="/code/a.md" if team else None,
                   communication_policy_path="/code/c.md" if team else None)
    if mutate is not None:
        mutate(settings)
    path.write_text(json.dumps(settings))
    return directory


def write_sidecar(directory, arm, statuses, *, reward=None, task_id="task"):
    """Filtered evaluator metadata whose reward matches the arm's total score."""
    passed = sum(status == "passed" for status in statuses.values())
    path = directory / task_id / arm / "verifier/filtered_test_results.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "task_id": task_id, "official_status": "completed",
        "reward": passed / len(statuses) if reward is None else reward, "valid_tests": len(statuses),
        "passed_tests": passed, "submission_error": False,
        "tests": [{"test_id": test_id, "status": status} for test_id, status in statuses.items()]}))


def set_pair_id(directory, task_id, pair_id):
    path = directory / "run_settings.json"
    settings = json.loads(path.read_text())
    next(task for task in settings["tasks"] if task["task_id"] == task_id)["pair_id"] = pair_id
    path.write_text(json.dumps(settings))
    for result_path in (directory / task_id).glob("*/task_result.json"):
        result = json.loads(result_path.read_text())
        result["pair_id"] = pair_id
        result_path.write_text(json.dumps(result))


def test_three_independent_profiles_pair_by_identity_not_order_and_keep_failed_zero(tmp_path):
    first = saved_round(tmp_path, 1, {"task_b": (.5, .4), "task_a": (0, .1)})
    second = saved_round(tmp_path, 2, {"task_a": (.2, .25), "task_b": (.3, .3)})
    third = saved_round(tmp_path, 3, {"task_a": (.4, .45), "task_b": (.1, .2)})
    report = summarize([third, first, second], expected_repeat_indices=[1, 2, 3])
    a, b = report["tasks"]
    assert a["task_id"] == "task_a" and b["task_id"] == "task_b"
    assert [pair["repeat_index"] for pair in a["pairs"]] == [1, 2, 3]
    assert a["arms"]["single"]["mean"] == pytest.approx(.2)
    assert a["arms"]["single"]["sample_variance"] == pytest.approx(.04)
    assert a["difference"]["mean"] == pytest.approx(1 / 15)
    assert a["pairs"][0]["arms"]["single"]["score"] == 0
    assert a["pairs"][0]["arms"]["single"]["status"] == "failed"
    assert b["difference"]["positive_count"] == b["difference"]["negative_count"] == b["difference"]["zero_count"] == 1
    assert "accept" not in report and "delta" not in report and "alpha" not in report


def test_missing_result_missing_score_and_wholly_missing_round_stay_unknown(tmp_path):
    first = saved_round(tmp_path, 1, {"task": (0, "missing_result")})
    second = saved_round(tmp_path, 2, {"task": (None, .5)})
    report = summarize([first, second], expected_repeat_indices=[1, 2, 3])
    task = report["tasks"][0]
    assert task["arms"]["single"]["known_count"] == task["arms"]["team_p0"]["known_count"] == 1
    assert task["arms"]["single"]["observed_scores_only"]["mean"] == 0
    assert task["arms"]["single"]["mean"] is None
    assert task["difference"]["mean"] is None and task["difference"]["known_count"] == 0
    assert task["difference"]["unknown_count"] == 3
    assert task["pairs"][0]["arms"]["team_p0"]["score_state"] == "missing_result"
    assert task["pairs"][1]["arms"]["single"]["score_state"] == "missing_score"
    assert task["pairs"][2]["arms"]["single"]["score_state"] == "missing_planned_round"


def test_legacy_defaults_are_readable_but_independent_legacy_rounds_are_not_renumbered(tmp_path):
    first = saved_round(tmp_path, 0, {"task": (.2, .3)}, legacy=True)
    report = summarize([first])
    pair = report["tasks"][0]["pairs"][0]
    assert pair["repeat_index"] == 0 and pair["pair_id"] == "task"
    assert report["coverage_scope"] == "supplied_profiles_only"
    second = saved_round(tmp_path, 0, {"task": (.9, .8)}, legacy=True, name="retry")
    with pytest.raises(ValueError, match="Duplicate task/round/arm"):
        summarize([first, second])


def test_conflicting_pair_ids_cannot_be_matched_just_by_round(tmp_path):
    directory = saved_round(tmp_path, 1, {"task": (.1, .2)})
    path = directory / "task/team_p0/task_result.json"
    result = json.loads(path.read_text())
    result["pair_id"] = "unrelated-pair"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="pair_id disagrees"):
        summarize([directory])


@pytest.mark.parametrize("field,value", [("execution_mode", "single"), ("policy_version", 99)])
def test_result_arm_metadata_must_match_the_recorded_plan_when_present(tmp_path, field, value):
    directory = saved_round(tmp_path, 1, {"task": (.1, .2)})
    path = directory / "task/team_p0/task_result.json"
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=f"{field} disagrees"):
        summarize([directory])


def test_public_provider_runtime_and_platform_are_preserved_without_credential_values(tmp_path):
    paths = [saved_round(tmp_path, index, {"task": (.1, .2)}) for index in (1, 2)]
    for index, directory in enumerate(paths):
        path = directory / "run_settings.json"
        settings = json.loads(path.read_text())
        settings.update(provider={"model": f"model-{index}", "base_url": "https://example.invalid",
                                  "api_key": "DO_NOT_COPY", "max_attempts": index + 1},
                        runtime="custom_Qwen_not_official_mini_SWE", platform="linux/amd64")
        path.write_text(json.dumps(settings))
    report = summarize(paths)
    assert [source["provider"]["model"] for source in report["sources"]] == ["model-0", "model-1"]
    assert all(source["platform"] == "linux/amd64" and source["runtime"] for source in report["sources"])
    assert "DO_NOT_COPY" not in json.dumps(report)
    assert any("Different conditions" in text for text in report["limitations"])
    assert any("do not prove identical code" in text for text in report["limitations"])
    conditions = report["tasks"][0]["conditions"]
    assert conditions["status"] == "different_recorded_conditions"
    assert conditions["differing_fields"] == ["provider"]


@pytest.mark.parametrize("score", [True, "0", -1, 2, float("nan"), float("inf")])
def test_invalid_score_is_unknown_not_zero(tmp_path, score):
    directory = saved_round(tmp_path, 0, {"task": (score, .5)})
    pair = summarize([directory])["tasks"][0]["pairs"][0]
    assert pair["arms"]["single"]["score"] is None and pair["difference"] is None
    assert pair["arms"]["single"]["score_state"] == "invalid_score"


def test_zero_sample_variance_does_not_emit_a_threshold_or_error_rate(tmp_path):
    paths = [saved_round(tmp_path, index, {"task": (0, 0)}) for index in (1, 2, 3)]
    report = summarize(paths)
    task = report["tasks"][0]
    assert task["arms"]["team_p0"]["sample_variance"] == 0
    assert task["difference"]["zero_count"] == 3
    assert any("not zero population noise" in text for text in report["limitations"])


def test_cli_reads_only_settings_and_results_and_does_not_modify_them(tmp_path, monkeypatch, capsys):
    directory = saved_round(tmp_path, 1, {"task": (0, .5)})
    before = {str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    original = Path.read_text

    def read_metadata_only(path, *args, **kwargs):
        assert path.name in ("run_settings.json", "task_result.json")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_metadata_only)
    assert cli.main([str(directory), "--left-arm", "single", "--right-arm", "team_p0",
                     "--expected-repeat-indices", "1", "2", "3"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["tasks"][0]["difference"]["unknown_count"] == 2
    after = {str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert before == after


def test_unexpected_round_is_not_silently_dropped(tmp_path):
    directory = saved_round(tmp_path, 4, {"task": (.1, .2)})
    with pytest.raises(ValueError, match="outside expected rounds"):
        summarize([directory], expected_repeat_indices=[1, 2, 3])


def test_layer_repeats_keep_zero_and_propagate_missing_denominators(tmp_path):
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(json.dumps({"task_id": "task", "rule_version": "v1", "tests": {
        "branch/core": {"module": "core", "layer": "core"},
        "branch/info": {"module": "info", "layer": "information"}}}))
    paths = [saved_round(tmp_path, repeat, {"task": (0, .5)}) for repeat in (1, 2, 3)]
    for directory in paths:
        for arm, passed in (("single", 0), ("team_p0", 1)):
            sidecar = directory / "task" / arm / "verifier/filtered_test_results.json"
            sidecar.parent.mkdir()
            sidecar.write_text(json.dumps({"schema_version": 1, "task_id": "task",
                "official_status": "completed", "reward": passed / 2, "valid_tests": 2,
                "passed_tests": passed, "submission_error": not passed, "tests": [
                    {"test_id": "branch/core", "status": "passed" if passed else "not_run"},
                    {"test_id": "branch/info", "status": "failure" if passed else "not_run"}]}))
    report = summarize(paths, layer_rule_paths=[rule_path])
    core = report["tasks"][0]["score_layers"]["core"]
    assert core["denominator"] == 1 and core["rule_version"] == "v1"
    assert core["arms"]["single"]["mean"] == 0 and core["difference"]["mean"] == 1
    assert report["tasks"][0]["comparability"]["status"] == "comparable"
    # This round has a smaller *official* test universe, not a zero missing test.
    changed = paths[-1] / "task/team_p0/verifier/filtered_test_results.json"
    value = json.loads(changed.read_text())
    value.update(valid_tests=1, reward=1)
    value["tests"].pop()
    changed.write_text(json.dumps(value))
    result_path = paths[-1] / "task/team_p0/task_result.json"
    value = json.loads(result_path.read_text())
    value["score"] = 1
    result_path.write_text(json.dumps(value))
    report = summarize(paths, layer_rule_paths=[rule_path])
    task = report["tasks"][0]
    assert task["arms"]["team_p0"]["complete"] is True  # Every raw score is present.
    assert task["comparability"]["status"] == "not_comparable"
    assert task["arms"]["team_p0"]["mean"] is None
    assert task["arms"]["team_p0"]["sample_variance"] is None
    assert task["arms"]["team_p0"]["observed_scores_only"]["mean"] == pytest.approx(2 / 3)
    assert task["arms"]["team_p0"]["observed_scores_only"]["sample_variance"] == pytest.approx(1 / 12)
    assert task["arms"]["team_p0"]["observed_scores_only"]["comparability"] == "not_comparable"
    assert task["arms"]["single"]["comparability"]["status"] == "comparable"
    assert task["arms"]["single"]["mean"] == task["arms"]["single"]["sample_variance"] == 0
    assert task["difference"]["mean"] is None and task["difference"]["positive_count"] is None
    assert task["difference"]["observed_scores_only"]["positive_count"] == 3
    assert task["pairs"][-1]["difference"] is None
    assert task["pairs"][-1]["observed_difference"] == 1
    core = task["score_layers"]["core"]
    assert core["arms"]["team_p0"]["mean"] is None and core["difference"]["mean"] is None
    assert core["arms"]["team_p0"]["known_count"] == 2
    assert "missing_test_ids" in task["pairs"][-1]["arms"]["team_p0"]["layer_coverage"]["reasons"]

    # Put Team back, then change only Single's universe. Team B variance is
    # still a description of the same official test set and must be retained.
    value = json.loads(changed.read_text())
    value.update(valid_tests=2, reward=.5)
    value["tests"].append({"test_id": "branch/info", "status": "failure"})
    changed.write_text(json.dumps(value))
    value = json.loads(result_path.read_text())
    value["score"] = .5
    result_path.write_text(json.dumps(value))
    single_path = paths[-1] / "task/single/verifier/filtered_test_results.json"
    value = json.loads(single_path.read_text())
    value["tests"].pop()
    value["valid_tests"] = 1  # Its valid zero stays zero, despite the universe change.
    single_path.write_text(json.dumps(value))
    task = summarize(paths, layer_rule_paths=[rule_path])["tasks"][0]
    assert task["arms"]["team_p0"]["comparability"]["status"] == "comparable"
    assert task["arms"]["team_p0"]["mean"] == .5 and task["arms"]["team_p0"]["sample_variance"] == 0
    assert task["arms"]["single"]["mean"] is None and task["difference"]["mean"] is None
    assert task["pairs"][-1]["arms"]["single"]["score"] == 0

    # Bad/incomplete classification cannot itself establish different official
    # test sets: return to equal sidecars and remove an assignment from rules.
    value["tests"].append({"test_id": "branch/info", "status": "not_run"})
    value["valid_tests"] = 2
    single_path.write_text(json.dumps(value))
    rule = json.loads(rule_path.read_text())
    rule["tests"].pop("branch/info")
    rule_path.write_text(json.dumps(rule))
    task = summarize(paths, layer_rule_paths=[rule_path])["tasks"][0]
    assert task["comparability"]["status"] == "comparable"
    assert task["arms"]["team_p0"]["mean"] == .5
    assert task["score_layers"]["core"]["arms"]["team_p0"]["mean"] is None
    assert "branch/core" not in json.dumps(task) and "branch/info" not in json.dumps(task)


def test_layer_cli_does_not_guess_rules_for_other_tasks(tmp_path, capsys):
    directory = saved_round(tmp_path, 1, {"task": (0, .5)})
    path = tmp_path / "other-rules.json"
    path.write_text(json.dumps({"task_id": "other", "rule_version": "v1", "tests": {
        "branch/core": {"module": "core", "layer": "core"}}}))
    assert cli.main([str(directory), "--left-arm", "single", "--right-arm", "team_p0",
                     "--layer-rules", str(path), "--expected-repeat-indices", "1", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    task = report["tasks"][0]
    assert task["score_layers"]["core"]["arms"]["single"]["unknown_count"] == 2
    assert task["pairs"][0]["arms"]["single"]["layer_coverage"]["reasons"] == ["missing_rules"]
    assert task["comparability"]["status"] == "unknown"
    assert task["pairs"][0]["difference"] == .5  # Descriptive, explicitly unverified.


def test_old_total_score_summary_is_explicitly_unverified_not_invented_comparable(tmp_path):
    directory = saved_round(tmp_path, 1, {"task": (0, .5)})
    task = summarize([directory])["tasks"][0]
    assert task["comparability"]["status"] == "unknown"
    assert task["difference"]["mean"] == .5
    assert task["difference"]["observed_scores_only"]["comparability"] == "unknown"


@pytest.mark.parametrize("team_score", [.5, None])
def test_unrelated_sidecar_score_cannot_verify_the_official_test_set(tmp_path, team_score):
    directory = saved_round(tmp_path, 1, {"task": (0, team_score)})
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(json.dumps({"task_id": "task", "rule_version": "v1", "tests": {
        "branch/original": {"module": "core", "layer": "core"}}}))
    for arm, test_id, reward in (("single", "branch/original", 0), ("team_p0", "branch/unrelated", 1)):
        path = directory / "task" / arm / "verifier/filtered_test_results.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"schema_version": 1, "task_id": "task",
            "official_status": "completed", "reward": reward, "passed_tests": reward, "valid_tests": 1,
            "submission_error": False, "tests": [{"test_id": test_id,
                                                   "status": "passed" if reward else "failure"}]}))
    task = summarize([directory], layer_rule_paths=[rule_path])["tasks"][0]
    assert task["comparability"]["status"] == "unknown"
    assert task["comparability"]["known_universes"] == 1
    assert task["arms"]["team_p0"]["comparability"]["status"] == "unknown"
    row = task["pairs"][0]["arms"]["team_p0"]
    assert row["score"] == team_score
    assert row["layer_coverage"]["reasons"] == ["task_score_unknown_or_mismatch"]


def test_stale_or_reused_pair_id_cannot_label_two_rounds_or_tasks(tmp_path):
    first = saved_round(tmp_path, 1, {"task": (.1, .2)})
    stale = saved_round(tmp_path, 2, {"task": (.3, .4)})
    # E.g. a profile carrying r1's pair_id launched with --repeat-index 2.
    set_pair_id(stale, "task", "task#r1")
    with pytest.raises(ValueError, match="labels more than one task/round"):
        summarize([first, stale])
    shared = saved_round(tmp_path, 3, {"task_a": (.1, .2), "task_b": (.3, .4)})
    for task_id in ("task_a", "task_b"):
        set_pair_id(shared, task_id, "same-pair")
    with pytest.raises(ValueError, match="labels more than one task/round"):
        summarize([shared])


def test_provenance_copies_recorded_settings_and_same_conditions_keep_statistics(tmp_path):
    paths = [record_conditions(saved_round(tmp_path, index, {"task": scores}))
             for index, scores in ((1, (.1, .2)), (2, (.2, .4)))]
    report = summarize(paths)
    settings = json.loads((paths[0] / "run_settings.json").read_text())
    for key in ("parallelism", "waves", "project_root", "qwen_package", "node_directory", "docker",
                "provider_config"):
        assert report["sources"][0][key] == settings[key]
    # Run directory, profile path and round identity differ by design.
    assert report["sources"][0]["profile_path"] != report["sources"][1]["profile_path"]
    task = report["tasks"][0]
    assert task["conditions"] == {"status": "same_recorded_conditions", "differing_fields": [],
                                  "unrecorded_fields": [], "contributing_sources": 2}
    assert task["arms"]["single"]["mean"] == pytest.approx(.15)
    assert task["difference"]["positive_count"] == 2
    assert task["difference"]["observed_scores_only"]["conditions"] == "same_recorded_conditions"


def separate_waves(settings):
    settings.update(parallelism=1, waves=[[{"task_id": "task", "arm_id": arm["arm_id"]}]
                                          for arm in settings["arms"]])


@pytest.mark.parametrize("mutate,fields", [
    (lambda s: s["inference"].update(agent_timeout_sec=3600), ["inference"]),
    (lambda s: s["evaluation"].update(docker_cpus=4), ["evaluation"]),
    (lambda s: s["provider"].update(model="other-model"), ["provider"]),
    (lambda s: s.update(runtime="other-runtime"), ["runtime"]),
    (lambda s: s.update(platform="linux/arm64"), ["platform"]),
    (lambda s: s.update(project_root="/other-code"), ["project_root"]),
    (lambda s: s.update(qwen_package="/other-qwen"), ["qwen_package"]),
    (lambda s: s.update(node_directory="/other-node"), ["node_directory"]),
    (separate_waves, ["parallelism", "selected_arms_shared_wave"]),
    (lambda s: s["arms"][1].update(policy_version=1), ["arms.team_p0.policy_version"]),
    (lambda s: s["arms"][1].update(communication_policy_path="/code/c2.md"),
     ["arms.team_p0.communication_policy_path"]),
])
def test_different_recorded_conditions_withhold_series_statistics(tmp_path, mutate, fields):
    first = record_conditions(saved_round(tmp_path, 1, {"task": (.1, .2)}))
    second = record_conditions(saved_round(tmp_path, 2, {"task": (.2, .4)}), mutate)
    task = summarize([first, second])["tasks"][0]
    assert task["conditions"]["status"] == "different_recorded_conditions"
    assert task["conditions"]["differing_fields"] == fields
    for stats in (task["arms"]["single"], task["arms"]["team_p0"], task["difference"]):
        assert stats["complete"] and stats["mean"] is None and stats["sample_variance"] is None
        assert stats["observed_scores_only"]["mean"] is not None
        assert stats["observed_scores_only"]["sample_variance"] is not None
        assert stats["observed_scores_only"]["conditions"] == "different_recorded_conditions"
    assert task["difference"]["positive_count"] is None and task["difference"]["zero_count"] is None
    assert task["difference"]["observed_scores_only"]["positive_count"] == 2
    # Each round's own pair shares one source, so its difference survives.
    assert [pair["difference"] for pair in task["pairs"]] == pytest.approx([.1, .2])


def test_ignored_and_unrecorded_fields_never_claim_same_or_different_conditions(tmp_path):
    first = record_conditions(saved_round(tmp_path, 1, {"task": (.1, .2)}))
    second = record_conditions(saved_round(tmp_path, 2, {"task": (.2, .4)}),
                               lambda s: s.update(docker="other-docker", provider_config="/other-provider.json"))
    assert summarize([first, second])["tasks"][0]["conditions"]["status"] == "same_recorded_conditions"
    record_conditions(second, lambda s: (s.pop("platform"), s.pop("waves")))
    task = summarize([first, second])["tasks"][0]
    assert task["conditions"]["status"] == "unknown"
    assert task["conditions"]["unrecorded_fields"] == ["platform", "selected_arms_shared_wave"]
    assert task["arms"]["single"]["mean"] == pytest.approx(.15)
    assert task["difference"]["positive_count"] == 2


def test_different_conditions_withhold_layer_statistics_too(tmp_path):
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(json.dumps({"task_id": "task", "rule_version": "v1", "tests": {
        "branch/core": {"module": "core", "layer": "core"},
        "branch/info": {"module": "info", "layer": "information"}}}))
    paths = [record_conditions(saved_round(tmp_path, 1, {"task": (0, .5)})),
             record_conditions(saved_round(tmp_path, 2, {"task": (0, .5)}),
                               lambda s: s["inference"].update(cpus=4))]
    for directory in paths:
        write_sidecar(directory, "single", {"branch/core": "failure", "branch/info": "failure"})
        write_sidecar(directory, "team_p0", {"branch/core": "passed", "branch/info": "failure"})
    task = summarize(paths, layer_rule_paths=[rule_path])["tasks"][0]
    assert task["comparability"]["status"] == "comparable"
    assert task["conditions"]["differing_fields"] == ["inference"]
    core = task["score_layers"]["core"]
    for stats in (core["arms"]["team_p0"], core["difference"]):
        assert stats["complete"] and stats["mean"] is None and stats["sample_variance"] is None
        assert stats["observed_scores_only"]["mean"] == 1
        assert stats["observed_scores_only"]["conditions"] == "different_recorded_conditions"


def test_incomplete_series_withholds_sign_counts_and_emits_no_top_level_range(tmp_path):
    first = saved_round(tmp_path, 1, {"task": (.1, .2)})
    second = saved_round(tmp_path, 2, {"task": (.3, None)})
    task = summarize([first, second])["tasks"][0]
    difference = task["difference"]
    assert not difference["complete"] and difference["known_count"] == 1
    assert difference["positive_count"] is None and difference["negative_count"] is None
    assert difference["zero_count"] is None
    assert difference["observed_scores_only"]["positive_count"] == 1
    assert difference["observed_scores_only"]["zero_count"] == 0
    for stats in (difference, task["arms"]["single"], task["arms"]["team_p0"]):
        # Available-case extremes exist only under observed_scores_only.
        assert not {"minimum", "maximum", "range"} & set(stats)
        assert "range" in stats["observed_scores_only"]


def test_arm_absent_from_a_supplied_plan_differs_from_a_missing_round(tmp_path):
    first = saved_round(tmp_path, 1, {"task": (.1, .2)})
    second = saved_round(tmp_path, 2, {"task": (.3, .4)})
    third = saved_round(tmp_path, 3, {"task": (.5, .6)})
    for directory, arms in ((second, lambda arms: arms[:1]),
                            (third, lambda arms: [{"arm_id": "team_p1", "execution_mode": "team",
                                                   "policy_version": 1}])):
        path = directory / "run_settings.json"
        settings = json.loads(path.read_text())
        settings["arms"] = arms(settings["arms"])
        path.write_text(json.dumps(settings))
    task = summarize([first, second, third], expected_repeat_indices=[1, 2, 3, 4])["tasks"][0]
    pair = task["pairs"][1]
    assert pair["arms"]["single"]["score"] == .3 and pair["pair_id"] == "task#r2"
    assert pair["arms"]["team_p0"]["score_state"] == "arm_not_in_round_plan"
    assert {arm["score_state"] for arm in task["pairs"][2]["arms"].values()} == {"arm_not_in_round_plan"}
    assert {arm["score_state"] for arm in task["pairs"][3]["arms"].values()} == {"missing_planned_round"}


def test_official_test_sets_are_compared_without_layer_rules_from_identifiers_only(tmp_path, monkeypatch):
    paths = [saved_round(tmp_path, repeat, {"task": (0, .5)}) for repeat in (1, 2)]
    for directory in paths:
        write_sidecar(directory, "single", {"branch/core": "failure", "branch/info": "failure"})
        write_sidecar(directory, "team_p0", {"branch/core": "passed", "branch/info": "failure"})
    original, reads = Path.read_text, set()

    def record(path, *args, **kwargs):
        reads.add(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record)
    task = summarize(paths)["tasks"][0]
    assert task["comparability"]["status"] == "comparable" and "score_layers" not in task
    assert "layer_coverage" not in task["pairs"][0]["arms"]["team_p0"]
    assert reads == {"run_settings.json", "task_result.json", "filtered_test_results.json"}
    # Round 2 Team ran on another official set; its per-test statuses are not
    # inspected (these would fail layer validation) and never reported.
    write_sidecar(paths[1], "team_p0", {"branch/core": "STATUS_POISON", "branch/new": "STATUS_POISON"},
                  reward=.5)
    report = summarize(paths)
    task = report["tasks"][0]
    assert task["arms"]["team_p0"]["comparability"]["status"] == "not_comparable"
    assert task["arms"]["team_p0"]["mean"] is None
    assert task["arms"]["single"]["comparability"]["status"] == "comparable"
    assert "STATUS_POISON" not in json.dumps(report) and "branch/new" not in json.dumps(report)
    # An absent sidecar stays unknown rather than being read or guessed.
    (paths[0] / "task/single/verifier/filtered_test_results.json").unlink()
    task = summarize(paths)["tasks"][0]
    assert task["arms"]["single"]["comparability"]["status"] == "unknown"
