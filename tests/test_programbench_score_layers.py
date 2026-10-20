"""Synthetic metadata only: no official tasks, models, containers or test bodies."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess

import pytest

from agent_company.analysis.score_layers import (
    load_rules, load_score_layers, names_only, official_test_universe)
from agent_company.analysis.evidence_projection import project_evidence
from scripts import summarize_programbench_score_layers as cli


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("process attempted"))
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: pytest.fail("network attempted"))


def rules():
    return {"task_id": "synthetic", "rule_version": "v1", "tests": {
        "branch/help": {"module": "help", "layer": "information"},
        "branch/core": {"module": "codec", "layer": "core"},
        "branch/fate": {"module": "fate", "layer": "core", "tags": ["fate"]}}}


def metadata():
    return {"schema_version": 1, "task_id": "synthetic", "official_status": "completed",
            "reward": 1 / 3, "passed_tests": 1, "valid_tests": 3, "submission_error": False,
            "tests": [{"test_id": "branch/help", "status": "passed"},
                      {"test_id": "branch/core", "status": "failure"},
                      {"test_id": "branch/fate", "status": "skipped"}]}


def write(root, data):
    path = root / "verifier/filtered_test_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def test_complete_layers_have_official_denominators_and_nonadditive_fate_subset(tmp_path):
    write(tmp_path, metadata())
    report = load_score_layers(tmp_path, rules())
    layers = report["score_layers"]
    assert report["coverage"]["status"] == "known"
    assert layers["layers"]["core"]["passed"] == 0
    assert layers["layers"]["core"]["total"] == 2
    assert layers["layers"]["information"]["fraction"] == 1
    assert layers["fate"]["total"] == 1 and layers["fate"]["fraction"] == 0
    assert layers["zero_pass_modules"] == ["codec", "fate"]
    assert sum(row["total"] for row in layers["layers"].values()) == 3


@pytest.mark.parametrize("status,submission_error,official_status,reward,known", [
    ("failure", False, "completed", 0, True),
    ("not_run", True, "completed", 0, True),
    ("not_run", False, "incomplete_evaluation", None, False),
    ("system_error", False, "verifier_error", None, False),
    ("not_run", False, "completed", 0, False),
])
def test_official_zero_and_incomplete_results_are_distinct(
        tmp_path, status, submission_error, official_status, reward, known):
    data = metadata()
    data.update(official_status=official_status, reward=reward, passed_tests=0,
                submission_error=submission_error)
    for row in data["tests"]:
        row["status"] = status
    write(tmp_path, data)
    report = load_score_layers(tmp_path, rules())
    assert report["score_layers"]["layers"]["core"]["fraction"] == (0 if known else None)
    assert report["score_layers"]["zero_pass_modules"] == (["codec", "fate", "help"] if known else None)
    assert report["coverage"]["status"] == ("known" if known else "unknown")


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d["tests"].pop(), "missing_test_ids"),
    (lambda d: d["tests"].append(deepcopy(d["tests"][0])), "duplicate_test_ids"),
    (lambda d: d["tests"][0].update(test_id="branch/new"), "unclassified_test_ids"),
    (lambda d: d.update(valid_tests=4), "official_count_mismatch"),
    (lambda d: d.update(passed_tests=2), "official_count_mismatch"),
    (lambda d: d.update(reward=.9), "official_reward_mismatch"),
    (lambda d: d.update(task_id="other"), "task_mismatch"),
])
def test_coverage_problems_leave_diagnostics_but_never_partial_primary_scores(tmp_path, mutation, reason):
    data = metadata()
    mutation(data)
    write(tmp_path, data)
    report = load_score_layers(tmp_path, rules())
    assert reason in report["coverage"]["reasons"]
    assert report["score_layers"]["layers"]["core"]["fraction"] is None
    assert report["score_layers"]["layers"]["information"]["passed"] is None
    assert report["score_layers"]["zero_pass_modules"] is None
    assert report["score_layers"]["layers"]["core"]["total"] == 2  # Never shrink denominator.


def test_missing_rules_read_nothing_and_missing_legacy_sidecar_is_unknown(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("outcomes read without rules"))
        report = load_score_layers(tmp_path)
    assert report == {"score_layers": None, "coverage": {"status": "unknown", "reasons": ["missing_rules"]}}
    assert load_score_layers(tmp_path, rules())["coverage"]["reasons"] == ["missing_filtered_results"]


def test_zero_information_denominator_does_not_choose_a_core_equals_total_threshold(tmp_path):
    data, rule = metadata(), rules()
    rule["tests"]["branch/help"]["layer"] = "core"
    write(tmp_path, data)
    report = load_score_layers(tmp_path, rule)["score_layers"]
    assert report["layers"]["information"]["total"] == 0
    assert report["layers"]["information"]["fraction"] is None
    assert report["layers"]["core"]["fraction"] == 1 / 3
    assert "threshold" not in json.dumps(report)


def test_names_only_cli_exposes_no_outcomes_or_bodies_and_changes_no_files(tmp_path, monkeypatch, capsys):
    data = metadata()
    data.update(error_details="BODY_POISON", log=["LOG_POISON"])
    data["tests"][0]["extra"] = {"text": "TEST_POISON"}
    path = write(tmp_path, data)
    before = path.read_bytes()
    original = Path.read_text
    def read_filtered_only(path, *args, **kwargs):
        assert path.name == "filtered_test_results.json"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_filtered_only)
    assert cli.main(["--run-dir", str(tmp_path), "--names-only"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == {"schema_version": 1, "task_id": "synthetic",
                                  "official_status": "completed", "record_count": 3, "duplicate_count": 0,
                                  "test_ids": ["branch/core", "branch/fate", "branch/help"]}
    assert "POISON" not in output and path.read_bytes() == before
    assert not any(status in output for status in ("passed", "failure", "skipped"))


def test_names_only_counts_duplicate_records_without_per_test_statuses(tmp_path):
    data = metadata()
    data.update(official_status="incomplete_evaluation", reward=None)
    data["tests"].append({"test_id": "branch/core", "status": "system_error"})
    write(tmp_path, data)
    report = names_only(tmp_path)
    assert report["official_status"] == "incomplete_evaluation"
    assert report["record_count"] == 4 and report["duplicate_count"] == 1
    assert report["test_ids"] == ["branch/core", "branch/fate", "branch/help"]
    assert not any(status in json.dumps(report) for status in ("system_error", "passed", "failure"))


def test_missing_sidecar_names_only_is_a_clean_error_not_a_traceback(tmp_path, capsys):
    with pytest.raises(ValueError, match="Missing verifier/filtered_test_results.json"):
        names_only(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--run-dir", str(tmp_path), "--names-only"])
    error = capsys.readouterr().err
    assert exc.value.code == 2 and "filtered_test_results.json" in error
    assert "Traceback" not in error and "FileNotFoundError" not in error


def test_official_universe_reads_identifiers_not_statuses_and_never_opens_a_missing_sidecar(
        tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("absent sidecar opened"))
        assert official_test_universe(tmp_path, "synthetic", score=1 / 3) is None
    data = metadata()
    for row in data["tests"]:
        row["status"] = {"unexpected": "STATUS_POISON"}  # Not validated or used.
    write(tmp_path, data)
    assert official_test_universe(tmp_path, "synthetic", score=1 / 3) == {
        "branch/help", "branch/core", "branch/fate"}
    assert official_test_universe(tmp_path, "other", score=1 / 3) is None
    assert official_test_universe(tmp_path, "synthetic", score=.5) is None


def test_duplicate_rule_keys_and_fate_information_assignment_are_rejected(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text('{"task_id":"synthetic","task_id":"other"}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_rules(path)
    rule = rules()
    rule["tests"]["branch/help"]["tags"] = ["fate"]
    path.write_text(json.dumps(rule))
    with pytest.raises(ValueError, match="core subset"):
        load_rules(path)


def test_symlink_and_malformed_filtered_artifacts_remain_unknown(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text(json.dumps(metadata()))
    (tmp_path / "verifier").mkdir()
    path = tmp_path / "verifier/filtered_test_results.json"
    path.symlink_to(target)
    assert load_score_layers(tmp_path, rules())["coverage"]["reasons"] == ["invalid_filtered_results"]
    path.unlink()
    path.write_text('{"schema_version":1,"schema_version":1}')
    assert load_score_layers(tmp_path, rules())["coverage"]["reasons"] == ["invalid_filtered_results"]


def test_schema5_projection_preserves_failed_execution_score_and_rejects_stale_sidecar(tmp_path):
    write(tmp_path, metadata())
    (tmp_path / "task_result.json").write_text(json.dumps({
        "status": "failed", "score": 1 / 3, "metrics": {"task": {"task_id": "synthetic"}}}))
    report = project_evidence(tmp_path, layer_rules=rules())
    assert report["schema_version"] == 5 and report["task_result"]["status"]["text"] == "failed"
    assert report["score_layers"]["layers"]["core"]["fraction"] == 0
    assert project_evidence(tmp_path)["score_layers"] is None
    data = metadata()
    data["reward"] = .5
    write(tmp_path, data)
    report = project_evidence(tmp_path, layer_rules=rules())
    assert report["score_layers"] is None
    assert report["coverage"]["score_layers"]["reasons"] == ["task_score_unknown_or_mismatch"]
