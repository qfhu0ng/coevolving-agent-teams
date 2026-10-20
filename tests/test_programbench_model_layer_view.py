"""Offline model evidence minimization; full diagnostic reports remain available."""
from copy import deepcopy
import json
import socket
import subprocess

import pytest

from agent_company.analysis import evidence_projection as projection
from agent_company.evolution.loop import PolicyProposal
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.programbench.study import StudyReviewRequest
from agent_company.evolution.reviewer import StrategyReviewer
from test_programbench_score_layers import metadata, rules, write
from test_executability_protocol import Script, call, note, provider, reply, submit_policy


FORBIDDEN = {"by_module", "fate", "FATE", "zero_pass_modules", "observed_only", "status_counts"}


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("external process attempted"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network attempted"))


def trial(root, data=None):
    data = metadata() if data is None else data
    write(root, data)
    (root / "task_result.json").write_text(json.dumps({
        "status": "failed", "score": data["reward"], "task_id": data["task_id"]}))
    return root


def nested_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    return set()


def test_schema5_model_view_only_keeps_known_layer_counts_and_reasons(tmp_path):
    trial(tmp_path)
    diagnostic = projection.project_evidence(tmp_path, layer_rules=rules())
    before = deepcopy(diagnostic)
    model = projection.project_model_evidence(tmp_path, layer_rules=rules())
    assert model["schema_version"] == 5
    assert model["score_layers"] == {"layers": {
        "core": {"passed": 0, "total": 2}, "information": {"passed": 1, "total": 1}}}
    assert model["coverage"]["score_layers"] == {"reasons": []}
    assert not nested_keys(model["score_layers"]) & FORBIDDEN
    assert not nested_keys(model["coverage"]["score_layers"]) & FORBIDDEN
    assert diagnostic == before
    assert diagnostic["score_layers"]["by_module"]
    assert diagnostic["score_layers"]["fate"]
    assert diagnostic["score_layers"]["zero_pass_modules"]
    assert {k: v for k, v in model.items() if k not in {"score_layers", "coverage"}} == {
        k: v for k, v in diagnostic.items() if k not in {"score_layers", "coverage"}}
    assert {k: v for k, v in model["coverage"].items() if k != "score_layers"} == {
        k: v for k, v in diagnostic["coverage"].items() if k != "score_layers"}


def test_unknown_layer_scores_do_not_use_observed_counts_or_fill_zero(tmp_path):
    data = metadata()
    data["tests"].pop()
    trial(tmp_path, data)
    model = projection.project_model_evidence(tmp_path, layer_rules=rules())
    assert model["score_layers"] == {"layers": {
        "core": {"passed": None, "total": 2}, "information": {"passed": None, "total": 1}}}
    assert "missing_test_ids" in model["coverage"]["score_layers"]["reasons"]
    assert projection.project_model_evidence(tmp_path)["score_layers"] is None
    assert projection.project_model_evidence(tmp_path)["coverage"]["score_layers"] == {
        "reasons": ["missing_rules"]}
    partial = {"score_layers": {"status": "known", "layers": {"core": {"passed": 0}}}, "coverage": {}}
    minimal = projection.model_evidence_view(partial)
    assert minimal["score_layers"]["layers"] == {
        "core": {"passed": 0, "total": None}, "information": {"passed": None, "total": None}}


def test_official_zero_is_preserved_without_module_diagnostics(tmp_path):
    data = metadata()
    data.update(reward=0, passed_tests=0)
    for row in data["tests"]:
        row["status"] = "failure"
    trial(tmp_path, data)
    model = projection.project_model_evidence(tmp_path, layer_rules=rules())
    assert model["score_layers"]["layers"] == {
        "core": {"passed": 0, "total": 2}, "information": {"passed": 0, "total": 1}}


def test_followup_cumulative_tools_expose_only_minimal_layer_view(tmp_path, monkeypatch):
    """Use real controllers, then real optimizer/reviewer reads of both projections."""
    import test_programbench_evolved_followup as fixtures
    setup = fixtures.setup.__wrapped__(tmp_path, monkeypatch)
    actual_project = projection.project_evidence

    def with_explicit_rules(path, **kwargs):
        # Classification is explicit synthetic input; production configuration
        # does not infer rules or unlock hidden evaluator artifacts.
        result_path = path / "task_result.json"
        result = json.loads(result_path.read_text())
        task_id = path.parent.name
        passed = round(result["score"] * 100)
        data = {"schema_version": 1, "task_id": task_id, "official_status": "completed",
                "reward": result["score"], "passed_tests": passed, "valid_tests": 100,
                "tests": [{"test_id": str(i), "status": "passed" if i < passed else "failure"}
                          for i in range(100)]}
        result.update(task_id=task_id)
        fixtures.write(result_path, result)
        write(path, data)
        rule = {"task_id": task_id, "rule_version": "synthetic-v1", "tests": {
            str(i): {"module": f"module-{i}", "layer": "information" if i == 0 else "core",
                     "tags": ["fate"] if i == 99 else []} for i in range(100)}}
        return actual_project(path, layer_rules=rule)

    monkeypatch.setattr(projection, "project_evidence", with_explicit_rules)
    # The same real baseline/cumulative producers used for execution must call
    # project_model_evidence; no direct replacement of their imported wrappers.
    fixtures.test_actual_comparison_trial_can_bind_paused_followup_and_enter_round_two(setup, monkeypatch)
    request = setup[-1]["optimizer"][-1][0]
    files = request.evidence.list_files()["files"]
    paths = [item["path"] for item in files if item["path"].endswith("projection.json")]
    assert len(paths) == 2
    pages = []
    for path in paths:
        text = (request.evidence.root / path).read_text(encoding="utf-8")
        value = json.loads(text)
        assert value["score_layers"]["layers"]["information"] == {"passed": 1, "total": 1}
        assert not nested_keys(value["score_layers"]) & FORBIDDEN
        assert value["coverage"]["score_layers"] == {"reasons": []}
        pages.extend({"path": path, "offset": offset, "limit": 12000} for offset in range(0, len(text), 12000))
    reads = reply(*(call("read_training_file", args, f"read-{i}") for i, args in enumerate(pages)))
    optimizer = Script(reads, submit_policy([note()]))
    candidate = StrategyOptimizer(provider(), tmp_path / "model_optimizer", completion=optimizer)(request)
    assert isinstance(candidate, PolicyProposal)
    review = Script(reads, reply(call("submit_review", {
        "recommendation": "uncertain", "rationale": "Synthetic evidence does not establish a gain.",
        "executability": [{"policy_part": "communication", "rule_quote": note()["rule_quote"],
                           "capability_refs": [], "verdict": "unknown", "reason": "Synthetic input"}]})))
    StrategyReviewer(provider(), tmp_path / "model_reviewer", completion=review)(StudyReviewRequest(request, candidate))
    for payload in (optimizer.payloads[-1], review.payloads[-1]):
        results = [json.loads(message["content"]) for message in payload["messages"] if message["role"] == "tool"]
        assert len(results) == len(pages)
        for path in paths:
            fragments = sorted((result for result in results if result["path"] == path), key=lambda r: r["offset"])
            model_report = json.loads("".join(result["text"] for result in fragments))
            assert set(model_report["score_layers"]) == {"layers"}
            assert not nested_keys(model_report["score_layers"]) & FORBIDDEN
            assert model_report["coverage"]["score_layers"] == {"reasons": []}
