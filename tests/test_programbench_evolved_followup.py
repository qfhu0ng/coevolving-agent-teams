"""Offline follow-up tests: no real model, container, or remote operations."""
from dataclasses import asdict
import json
from pathlib import Path
import socket
import subprocess
from threading import Barrier

import pytest

from agent_company.evolution.loop import PolicyProposal, PolicyReview, TaskResult
from agent_company.programbench.study import prepare_study
from agent_company.runtime.relay import ProviderConfig
from agent_company.evolution.reviewer import PolicyReviewError
from scripts import run_programbench_evolved_followup as followup


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(autouse=True)
def forbid_external(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("external process"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network"))


def outcome(score=.05, *, status="failed", cleanup=True):
    return TaskResult(status=status, score=score, elapsed_seconds=21600,
                      usage={"prompt_tokens": 10, "completion_tokens": 2,
                             "cached_prompt_tokens": 0, "total_tokens": 12},
                      metrics={"inference_cleanup_confirmed": cleanup,
                               "team": {"stop_reason": "task_timeout", "usage_complete": True},
                               "verifier": {"status": "completed", "reward": score,
                                            "official_test_fraction": score}})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path
    task_id = "synthetic__task.1111111"
    task = root / task_id
    task.mkdir()
    (task / "task.yaml").write_text("repository: synthetic/task\ncommit: '" + "1" * 40 + "'\nlanguage: c\n")
    (task / "tests.json").write_text("HIDDEN_TEST_SENTINEL")
    (root / "instructions.md").write_text("Implement the public API.")
    (root / "assignment.md").write_text("P3 assignment")
    (root / "communication.md").write_text("P3 communication")
    provider_config = {"model": "synthetic", "base_url": "http://unused", "api_key_env": "PRIVATE_ENV",
                       "request_timeout_seconds": 1800, "max_attempts": 11, "rates": None}
    write(root / "provider.json", provider_config)
    profile = {"benchmark": "ProgramBench", "provider_config": "provider.json", "parallelism": 2,
               "evolution_enabled": True,
               "tasks": [{"task_id": task_id, "path": task_id, "instructions": "instructions.md"}],
               "arms": [{"name": "single", "execution_mode": "single", "policy_version": 0},
                        {"name": "team_p3", "execution_mode": "team", "policy_version": 3,
                         "assignment_policy_path": "assignment.md", "communication_policy_path": "communication.md"},
                        {"name": "team_evolved", "execution_mode": "team", "policy_version": 4}],
               "inference": {"agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 61440,
                             "memory_swap_mb": 61440, "max_model_calls": None, "team_checkin_seconds": 1800},
               "evaluation": {"docker_cpus": 10}}
    path = root / "profile.json"
    write(path, profile)
    source = root / "old/run"
    study = prepare_study(path, root)
    source_settings = {**study.plan, "arms": study.plan["arms"][:2],
                       "provider": followup.public_provider(ProviderConfig(**{
                           k: v for k, v in provider_config.items() if k != "api_key_env"}))}
    write(source / "run_settings.json", source_settings)
    write(source / "progress.json", {"state": "completed", "terminal": 2, "planned": 2,
                                      "pending": [], "running": [], "cleanup_attention": [],
                                      "rows": [{"arm_id": a["arm_id"], "task_id": task_id,
                                                "inference_cleanup_confirmed": True}
                                               for a in study.plan["arms"][:2]]})
    trial = source / task_id / "team_p3"
    write(trial / "run_request.json", {"execution_mode": "team", "policy_version": 3, "fresh_start": True,
                                       "task": {"task_id": task_id, "repo_id": "synthetic/task"},
                                       "assignment_policy": "P3 assignment", "communication_policy": "P3 communication"})
    write(trial / "task_result.json", asdict(outcome()))
    write(source / task_id / "single/private.json", {"secret": "SINGLE_PRIVATE_SENTINEL"})
    (trial / "team/relay").mkdir(parents=True)
    (trial / "team/relay/events.jsonl").write_text("")
    write(trial / "team/relay/attempt_1_master_qwen_response.json", {
        "choices": [{"message": {"content": "PROSE_SENTINEL", "reasoning_content": "REASONING_SENTINEL",
                                  "tool_calls": [{"id": "call_1", "function": {"name": "send_message",
                                    "arguments": json.dumps({"recipient": "worker", "content": "CHECK_DELIVERY"})}}]}}]})
    write(trial / "team/relay/attempt_1_master_request.json", {"secret": "REQUEST_SENTINEL"})
    calls = {"provider": [], "optimizer": [], "reviewer": [], "runner": []}
    def load_provider(path):
        calls["provider"].append(path)
        return ProviderConfig("synthetic", "http://unused", api_key="PRIVATE_KEY", request_timeout_seconds=1800,
                              max_attempts=11)
    monkeypatch.setattr(followup, "load_provider", load_provider)
    return root, path, source, trial, profile, calls


def mocks(setup, monkeypatch, *, optimizer_fail=False, reviewer_fail=False, run_result=None,
          review_callback=None, run_callback=None):
    calls = setup[-1]
    class Optimizer:
        def __init__(self, provider, directory, **options):
            self.options = options
        def __call__(self, request):
            calls["optimizer"].append((request, self.options))
            if optimizer_fail:
                raise RuntimeError("PRIVATE_FAILURE")
            return PolicyProposal(assignment_policy="new assignment", communication_policy="new communication",
                                  rationale="Evidence-backed candidate")
    class Reviewer:
        def __init__(self, provider, directory, **options):
            # Retries match the optimizer's (provider.max_attempts - 1).
            assert options == {"transport_retries": provider.max_attempts - 1}
        def __call__(self, request):
            calls["reviewer"].append(request)
            if reviewer_fail:
                raise PolicyReviewError("PRIVATE_FAILURE", usage={"provider_attempts": 4,
                                        "unknown_usage_attempts": 1, "total_tokens": 17},
                                        cost_usd=None, known_cost_usd=.02, elapsed_seconds=9,
                                        output_dir=setup[0] / "new/run/reviewer")
            if review_callback is not None:
                return review_callback(request)
            return PolicyReview("prefer_incumbent", "Negative advisory", risks=["possible regression"])
    class Runner:
        def __init__(self, **options):
            self.options = options
        def __call__(self, request):
            calls["runner"].append((request, self.options))
            assert request.fresh_start is True and request.split == "qualitative"
            profile = json.loads(setup[1].read_text())
            expected = next(arm for arm in profile["arms"] if arm["name"] == request.output_dir.name)
            assert request.execution_mode == expected["execution_mode"]
            assert request.policy_version == expected["policy_version"]
            if request.execution_mode == "team":
                assert request.assignment_policy == "new assignment"
                assert request.communication_policy == "new communication"
            else:
                assert request.assignment_policy == request.communication_policy == ""
            review = json.loads((request.output_dir.parent.parent / "review_summary.json").read_text())
            assert review["status"] == "completed"
            assert not (request.output_dir / request.execution_mode).exists()
            assert set(request.task.payload) == {"path", "instruction_path"}
            if run_callback is not None:
                return run_callback(request)
            return run_result if run_result is not None else outcome(.08, status="completed")
    monkeypatch.setattr(followup, "StrategyOptimizer", Optimizer)
    monkeypatch.setattr(followup, "StrategyReviewer", Reviewer)
    monkeypatch.setattr(followup, "ProgramBenchRunner", Runner)


def run(setup):
    root, path, source, *_ = setup
    return followup.run_followup(path, root, root / "new/run", source_run=source,
                                qwen_package=root / "qwen", node_directory=root / "node", docker="mock-docker", repeat_index=0)


def test_prepare_readonly_no_credentials_or_private_evidence(setup, monkeypatch, capsys):
    root, path, source, _, _, calls = setup
    original = Path.read_text
    def checked_read(path, *a, **k):
        assert path.name != "tests.json" and "private" not in path.name and "relay" not in path.parts
        return original(path, *a, **k)
    monkeypatch.setattr(Path, "read_text", checked_read)
    assert followup.main(["--profile", str(path), "--project-root", str(root), "--source-run", str(source)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["new_rollouts"] == 1 and plan["reused_rollouts"] == 0
    assert plan["training_rollouts_reused"] == 1 and plan["comparison_baseline_references"] == 2
    assert len(plan["arms"]) == 1 and plan["arms"][0]["arm_id"] == "team_evolved"
    assert plan["heldout"] is False and plan["automatic_adoption"] is False
    assert plan["review_completion_required"] is True and plan["review_advisory_only"] is True
    assert not any(calls.values()) and not (root / "new").exists()


def test_one_new_team_with_source_timeout_score_and_no_single_evidence(setup, monkeypatch):
    root, _, source, _, _, calls = setup
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    mocks(setup, monkeypatch)
    report = run(setup)
    assert report["state"] == "completed" and report["terminal"] == report["planned"] == 1
    assert {k: len(v) for k, v in calls.items()} == dict(provider=1, optimizer=1, reviewer=1, runner=1)
    assert report["evolution"]["review_recommendation"] == "prefer_incumbent"
    assert report["selection_advisory"]["adopted"] is False
    assert report["running"] == report["pending"] == []
    req, options = calls["runner"][0]
    assert options["max_model_calls"] is None and options["team_checkin_seconds"] == 1800
    assert options["task_cpus"] == 20 and options["task_memory_mb"] == 61440
    assert options["agent_timeout_seconds"] == 21600 and options["evaluator_cpus"] == 10
    proposal_request, optimizer_options = calls["optimizer"][0]
    assert optimizer_options["require_citations"] and optimizer_options["transport_retries"] == 10
    files = proposal_request.evidence.list_files()["files"]
    assert sorted(f["path"] for f in files) == ["baseline_team/projection.json", "baseline_team/task_summary.json"]
    payload = " ".join(p.read_text() for p in (root / "new/run/training_evidence").rglob("*.json"))
    assert "CHECK_DELIVERY" in payload
    for sentinel in ("SINGLE_PRIVATE_SENTINEL", "PROSE_SENTINEL", "REASONING_SENTINEL", "REQUEST_SENTINEL", "HIDDEN_TEST_SENTINEL"):
        assert sentinel not in payload
    with pytest.raises(PermissionError):
        proposal_request.evidence.read_file(str(source / "run_settings.json"))
    assert all(p.read_bytes() == content for p, content in before.items())
    assert "PRIVATE_KEY" not in (root / "new/run/run_settings.json").read_text()


@pytest.mark.parametrize("kind", ["progress", "cleanup", "task", "policy", "score", "official", "provider", "inference"])
def test_wrong_or_incomplete_source_rejected_before_calls(setup, kind):
    root, path, source, trial, _, calls = setup
    if kind in {"progress", "cleanup"}:
        file = source / "progress.json"
        value = json.loads(file.read_text())
        if kind == "progress":
            value["state"] = "running"
        else:
            value["rows"][0]["inference_cleanup_confirmed"] = False
    elif kind in {"task", "policy"}:
        file = trial / "run_request.json"
        value = json.loads(file.read_text())
        if kind == "task":
            value["task"]["task_id"] = "wrong"
        else:
            value["assignment_policy"] = "wrong P3"
    elif kind in {"score", "official"}:
        file = trial / "task_result.json"
        value = json.loads(file.read_text())
        if kind == "score":
            value["score"] = None
        else:
            value["metrics"]["verifier"]["status"] = "pending"
    else:
        file = source / "run_settings.json"
        value = json.loads(file.read_text())
        if kind == "provider":
            value["provider"]["request_timeout_seconds"] = 600
        else:
            value["inference"]["cpus"] = 10
    write(file, value)
    with pytest.raises(ValueError):
        run(setup)
    assert not any(calls.values()) and not (root / "new").exists()


def test_proposal_failure_blocks_without_new_rollout(setup, monkeypatch):
    mocks(setup, monkeypatch, optimizer_fail=True)
    report = run(setup)
    assert report["state"] == "blocked_proposal" and report["terminal"] == report["launched"] == 0
    assert len(setup[-1]["optimizer"]) == 1 and not setup[-1]["reviewer"] and not setup[-1]["runner"]
    assert "PRIVATE_FAILURE" not in json.dumps(report)


def test_reviewer_failure_blocks_candidate_and_preserves_accounting(setup, monkeypatch):
    mocks(setup, monkeypatch, reviewer_fail=True)
    report = run(setup)
    assert report["state"] == "blocked_review" and report["evolution"]["reviewer"] == "failed"
    assert report["terminal"] == report["launched"] == 0 and report["requires_user_action"] is True
    assert [row["arm_id"] for row in report["pending"]] == ["team_evolved"]
    assert report["running"] == [] and report["selection_advisory"] is None
    assert len(setup[-1]["reviewer"]) == 1 and not setup[-1]["runner"]
    output = setup[0] / "new/run"
    assert not (output / setup[4]["tasks"][0]["task_id"] / "team_evolved").exists()
    summary = json.loads((output / "review_summary.json").read_text())
    assert summary["status"] == "failed" and summary["usage"]["unknown_usage_attempts"] == 1
    assert summary["usage"]["provider_attempts"] == 4 and summary["known_cost_usd"] == .02
    assert "PRIVATE_FAILURE" not in json.dumps(summary)
    assert json.loads((output / "progress.json").read_text())["state"] == "blocked_review"


@pytest.mark.parametrize("review", [
    None, PolicyProposal("wrong result type", "wrong result type"),
    PolicyReview("invalid", "Invalid recommendation"), PolicyReview("uncertain", " "),
    PolicyReview("prefer_candidate", "Invalid accounting", elapsed_seconds=float("nan")),
])
def test_missing_or_invalid_review_never_dispatches_candidate(setup, monkeypatch, review):
    mocks(setup, monkeypatch, review_callback=lambda request: review)
    report = run(setup)
    assert report["state"] == "blocked_review" and report["requires_user_action"] is True
    assert report["launched"] == report["terminal"] == 0 and not setup[-1]["runner"]
    assert report["evolution"]["reviewer"] == "failed"
    assert not (setup[0] / "new/run" / setup[4]["tasks"][0]["task_id"] / "team_evolved").exists()


@pytest.mark.parametrize("recommendation", ["prefer_candidate", "prefer_incumbent", "uncertain"])
def test_completed_review_recommendation_stays_advisory(setup, monkeypatch, recommendation):
    def complete(request):
        assert not setup[-1]["runner"]
        progress = json.loads((setup[0] / "new/run/progress.json").read_text())
        assert progress["state"] == "reviewing" and progress["launched"] == 0
        return PolicyReview(recommendation, "Completed synthetic review")

    mocks(setup, monkeypatch, review_callback=complete)
    report = run(setup)
    assert report["state"] == "completed" and len(setup[-1]["runner"]) == 1
    assert report["evolution"]["reviewer"] == "completed"
    assert report["evolution"]["review_recommendation"] == recommendation


SUPPORTED = {"policy_part": "communication", "rule_quote": "new communication", "capability_refs": ["/wakeups"],
             "verdict": "supported", "reason": "Uses the existing event.", "conditions": ""}


UNSUPPORTED = {"policy_part": "communication", "rule_quote": "new communication", "capability_refs": ["/wakeups"],
               "verdict": "unsupported", "reason": "Needs a timer the harness lacks.", "conditions": ""}


@pytest.mark.parametrize("executability", [None, [UNSUPPORTED]])
def test_target_harness_reaches_both_roles_and_executability_is_shown_before_dispatch(
        setup, monkeypatch, executability):
    root, *_ = setup

    def reviewed(request):
        return PolicyReview("prefer_incumbent", "Synthetic", executability=executability)

    def execute(request):
        progress = json.loads((root / "new/run/progress.json").read_text())
        assert progress["evolution"]["executability_protocol"] is True
        shown = progress["evolution"]["review_executability"]
        if executability is None:
            assert shown == {"recorded": False}
        else:
            assert shown["counts"]["unsupported"] == 1 and shown["dispatch"].startswith("unchanged")
        return outcome(.08, status="completed")

    mocks(setup, monkeypatch, review_callback=reviewed, run_callback=execute)
    report = run(setup)
    # Advisory: an unsupported verdict does not block the dispatch.
    assert report["state"] == "completed" and len(setup[-1]["runner"]) == 1
    harness = setup[-1]["optimizer"][0][0].target_harness
    assert setup[-1]["reviewer"][0].proposal_request.target_harness == harness
    assert harness["scope"]["execution_mode"] == "team" and harness["scope"]["native_version"] == "unknown"
    assert harness["wakeups"]["checkin_interval_seconds"] == 1800
    assert harness["timing"]["agent_wall_time_seconds"] == 21600
    assert harness["timing"]["provider"]["max_attempts"] == 11
    assert "PRIVATE_KEY" not in json.dumps(harness) and "http://unused" not in json.dumps(harness)


@pytest.mark.parametrize("score,cleanup,state", [(None, True, "blocked_official_score"), (.1, False, "blocked_cleanup")])
def test_candidate_failure_no_automatic_retry(setup, monkeypatch, score, cleanup, state):
    mocks(setup, monkeypatch, run_result=outcome(score, cleanup=cleanup))
    report = run(setup)
    assert report["state"] == state and report["terminal"] == 1
    assert len(setup[-1]["runner"]) == 1


def test_existing_output_never_relaunches(setup, monkeypatch):
    mocks(setup, monkeypatch)
    run(setup)
    calls_before = {key: len(value) for key, value in setup[-1].items()}
    with pytest.raises(FileExistsError):
        run(setup)
    assert {key: len(value) for key, value in setup[-1].items()} == calls_before


def test_cli_requires_source_and_run_paths(setup):
    root, path, source, *_ = setup
    with pytest.raises(SystemExit):
        followup.main(["--profile", str(path)])
    with pytest.raises(SystemExit):
        followup.main(["--profile", str(path), "--source-run", str(source), "--run"])


def test_output_cannot_modify_source_run(setup, monkeypatch):
    root, path, source, *_ = setup
    mocks(setup, monkeypatch)
    with pytest.raises(ValueError):
        followup.run_followup(path, root, source / "nested", source_run=source,
                              qwen_package=root / "qwen", node_directory=root / "node", repeat_index=0)
    assert not any(setup[-1].values())


def test_omitting_provider_setting_cannot_change_runtime_default(setup):
    root, _, _, _, _, calls = setup
    provider_path = root / "provider.json"
    provider = json.loads(provider_path.read_text())
    del provider["request_timeout_seconds"]
    write(provider_path, provider)
    with pytest.raises(ValueError, match="provider settings"):
        run(setup)
    assert not any(calls.values())


@pytest.mark.parametrize("delivery", [None, "native_idle", "between_tool_calls"])
def test_recorded_team_message_delivery_is_the_one_the_runner_uses(setup, monkeypatch, delivery):
    # 2026-09-30 review: the plan recorded between_tool_calls while the runner
    # silently used the native_idle default.
    root, path, source, _, profile, calls = setup
    if delivery:
        changed = json.loads(json.dumps(profile))
        changed["inference"]["team_message_delivery"] = delivery
        write(path, changed)
        settings = json.loads((source / "run_settings.json").read_text())
        write(source / "run_settings.json", {**settings, "inference": prepare_study(path, root).plan["inference"]})
    mocks(setup, monkeypatch)
    assert run(setup)["state"] == "completed"
    assert [options["team_message_delivery"] for _, options in calls["runner"]] == [delivery or "native_idle"]
    # The optimizer is told the delivery mode the runner uses, from the same resolution.
    harness = calls["optimizer"][0][0].target_harness
    assert harness["messages"]["delivery_mode"] == (delivery or "native_idle")
    recorded = json.loads((root / "new/run/run_settings.json").read_text())
    assert recorded["inference"].get("team_message_delivery") == delivery


def p0_pair(setup, *, repeat_single=True):
    """Move only synthetic source metadata to P0 and opt into a fresh pair."""
    root, path, source, trial, profile, _ = setup
    profile["repeat_single"] = repeat_single
    profile["arms"][1].update(name="team_p0", policy_version=0)
    profile["arms"][2]["policy_version"] = 1
    profile["inference"]["team_message_delivery"] = "between_tool_calls"
    (root / "assignment.md").write_text("P0 assignment")
    (root / "communication.md").write_text("P0 communication")
    write(path, profile)
    trial = trial.rename(trial.with_name("team_p0"))
    request = json.loads((trial / "run_request.json").read_text())
    request.update(policy_version=0, assignment_policy="P0 assignment", communication_policy="P0 communication")
    write(trial / "run_request.json", request)
    settings = json.loads((source / "run_settings.json").read_text())
    study = prepare_study(path, root)
    settings.update(arms=study.plan["arms"][:2], inference=study.plan["inference"])
    write(source / "run_settings.json", settings)
    progress = json.loads((source / "progress.json").read_text())
    progress["rows"][1]["arm_id"] = "team_p0"
    write(source / "progress.json", progress)
    return trial


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_repeat_single_requires_explicit_boolean(setup, value):
    _, path, _, _, profile, calls = setup
    write(path, {**profile, "repeat_single": value})
    with pytest.raises(ValueError, match="repeat_single must be a boolean"):
        run(setup)
    assert not any(calls.values())


def test_explicit_repeat_single_false_keeps_one_rollout(setup, monkeypatch):
    root, path, source, _, profile, calls = setup
    write(path, {**profile, "repeat_single": False})
    plan = followup.prepare_followup(path, root, source).plan
    assert plan["repeat_single"] is False and plan["parallelism"] == plan["new_rollouts"] == 1
    mocks(setup, monkeypatch)
    assert run(setup)["planned"] == 1 and len(calls["runner"]) == 1


def test_p0_repeat_single_plan_is_two_fresh_arms_in_one_wave(setup):
    p0_pair(setup)
    root, path, source, *_ = setup
    plan = followup.prepare_followup(path, root, source).plan
    assert plan["repeat_single"] is True
    assert plan["new_rollouts"] == plan["parallelism"] == 2
    assert plan["reused_rollouts"] == 0 and plan["training_rollouts_reused"] == 1
    assert [a["arm_id"] for a in plan["arms"]] == ["single", "team_evolved"]
    assert [a["policy_version"] for a in plan["arms"]] == [0, 1]
    assert plan["waves"] == [[{"task_id": plan["tasks"][0]["task_id"], "arm_id": name}
                              for name in ("single", "team_evolved")]]
    assert [(a["arm_id"], a["used_for_evolution"]) for a in plan["source_baselines"]] == [
        ("single", False), ("team_p0", True)]


def test_p0_new_single_and_team_overlap_only_after_review_without_source_changes(setup, monkeypatch):
    p0_pair(setup)
    root, _, source, _, _, calls = setup
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    rendezvous = Barrier(2)

    def reviewed(request):
        assert not calls["runner"]
        assert not (root / "new/run" / setup[4]["tasks"][0]["task_id"]).exists()
        progress = json.loads((root / "new/run/progress.json").read_text())
        assert progress["state"] == "reviewing" and progress["launched"] == 0
        assert [a["arm_id"] for a in progress["pending"]] == ["single", "team_evolved"]
        return PolicyReview("prefer_incumbent", "Completed negative advisory is not a veto")

    def execute(request):
        # A sequential implementation fails this bounded barrier.
        rendezvous.wait(timeout=3)
        return outcome(.02 if request.execution_mode == "single" else .08, status="completed")

    mocks(setup, monkeypatch, review_callback=reviewed, run_callback=execute)
    report = run(setup)
    assert report["state"] == "completed" and report["planned"] == report["terminal"] == report["launched"] == 2
    assert report["parallelism"] == 2 and report["pending"] == report["running"] == []
    assert len(calls["optimizer"]) == len(calls["reviewer"]) == 1 and len(calls["runner"]) == 2
    assert {request.execution_mode for request, _ in calls["runner"]} == {"single", "team"}
    assert len({id(options) for _, options in calls["runner"]}) == 2
    assert all(options["team_message_delivery"] == "between_tool_calls" for _, options in calls["runner"])
    assert all(options["agent_timeout_seconds"] == 21600 and options["max_model_calls"] is None
               for _, options in calls["runner"])
    proposal_request = calls["optimizer"][0][0]
    assert proposal_request.current_version == 0
    assert proposal_request.change_history[0]["parent"] == "P0"
    assert proposal_request.change_history[0]["new_candidate"] == 1
    assert proposal_request.assignment_policy == "P0 assignment"
    payload = " ".join(p.read_text() for p in (root / "new/run/training_evidence").rglob("*.json"))
    assert "CHECK_DELIVERY" in payload
    assert all(sentinel not in payload for sentinel in
               ("SINGLE_PRIVATE_SENTINEL", "PROSE_SENTINEL", "REASONING_SENTINEL", "REQUEST_SENTINEL", "HIDDEN_TEST_SENTINEL"))
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    assert [row["arm_id"] for row in report["rows"]] == ["single", "team_evolved"]
    assert report["selection_advisory"]["adopted"] is False


@pytest.mark.parametrize("failure", ["proposal", "review", "invalid_review"])
def test_paired_followup_never_starts_either_arm_before_valid_review(setup, monkeypatch, failure):
    p0_pair(setup)
    mocks(setup, monkeypatch, optimizer_fail=failure == "proposal", reviewer_fail=failure == "review",
          review_callback=(lambda request: None) if failure == "invalid_review" else None)
    report = run(setup)
    assert report["state"] == ("blocked_proposal" if failure == "proposal" else "blocked_review")
    assert report["launched"] == report["terminal"] == 0 and report["planned"] == 2
    assert [a["arm_id"] for a in report["pending"]] == ["single", "team_evolved"]
    assert not setup[-1]["runner"]
    assert not (setup[0] / "new/run" / setup[4]["tasks"][0]["task_id"]).exists()


@pytest.mark.parametrize("failed_mode", ["single", "team"])
@pytest.mark.parametrize("failure", ["missing_score", "cleanup", "exception"])
def test_paired_followup_failure_does_not_cancel_or_retry_sibling(setup, monkeypatch, failed_mode, failure):
    p0_pair(setup)
    def execute(request):
        if request.execution_mode != failed_mode:
            return outcome(.08, status="completed")
        if failure == "exception":
            raise RuntimeError("PRIVATE_FAILURE")
        return outcome(None if failure == "missing_score" else .04, cleanup=failure != "cleanup")
    mocks(setup, monkeypatch, run_callback=execute)
    report = run(setup)
    assert report["state"] == ("blocked_official_score" if failure == "missing_score" else "blocked_cleanup")
    assert report["terminal"] == report["launched"] == 2 and not report["running"]
    assert len(setup[-1]["runner"]) == 2
    other = next(row for row in report["rows"] if row["execution_mode"] != failed_mode)
    assert other["status"] == "completed" and other["score"] == .08
    assert "PRIVATE_FAILURE" not in json.dumps(report)


def test_p0_source_request_version_still_checked_against_selected_baseline(setup, monkeypatch):
    trial = p0_pair(setup)
    request = json.loads((trial / "run_request.json").read_text())
    write(trial / "run_request.json", {**request, "policy_version": 3})
    mocks(setup, monkeypatch)
    with pytest.raises(ValueError, match="baseline Team request identity"):
        run(setup)
    assert not any(setup[-1].values())


def total_profile(setup):
    root, path, source, trial, profile, calls = setup
    from agent_company.programbench.total_selection import MODE
    profile["selection_protocol"] = MODE
    write(path, profile)
    return root, path, source, trial


def test_total_protocol_accepts_solo_team_source_and_requires_run_identity(setup, monkeypatch):
    root, path, source, trial = total_profile(setup)
    settings = json.loads((source / "run_settings.json").read_text())
    settings["arms"] = settings["arms"][1:]
    write(source / "run_settings.json", settings)
    progress = json.loads((source / "progress.json").read_text())
    progress.update(terminal=1, planned=1, rows=progress["rows"][1:])
    write(source / "progress.json", progress)
    prepared = followup.prepare_followup(path, root, source)
    assert prepared.plan["comparison_baseline_references"] == 1
    for index, round_number in [(None, 1), (0, 1), (5, None), (5, 4)]:
        with pytest.raises(ValueError):
            followup.run_followup(path, root, root / "new/run", source_run=source, qwen_package=root / "qwen",
                                  node_directory=root / "node", repeat_index=index, evolution_round=round_number)
    assert not (root / "new/run").exists() and not any(setup[-1].values())


@pytest.mark.parametrize("assessment", [None, [], [UNSUPPORTED], [{**UNSUPPORTED, "verdict": "unknown"}]])
def test_total_protocol_pauses_unsupported_or_unknown_before_runner(setup, monkeypatch, assessment):
    root,path,source,_ = total_profile(setup)
    mocks(setup, monkeypatch, review_callback=lambda request: PolicyReview("prefer_candidate", "Synthetic", executability=assessment))
    report = followup.run_followup(path, root, root / "new/run", source_run=source, qwen_package=root / "qwen",
                                  node_directory=root / "node", repeat_index=5, evolution_round=1)
    assert report["state"] == "blocked_executability" and not setup[-1]["runner"]
    assert report["evolution"]["review_executability"]["dispatch"] == "paused_for_user"
    assert report["evolution"]["review_executability"]["dispatch_reason"] == (
        "reviewer_unsupported_or_unknown" if assessment else "executability_not_recorded")
    request = setup[-1]["optimizer"][0][0]
    assert request.as_prompt_data()["selection_objective"]["efficiency_acceptance"] is False


def test_total_protocol_uses_total_selector_and_transmits_round_identity(setup, monkeypatch):
    root,path,source,_=total_profile(setup)
    mocks(setup, monkeypatch, review_callback=lambda request: PolicyReview("prefer_candidate", "Synthetic", executability=[SUPPORTED]))
    monkeypatch.setattr(followup,"select_candidate",lambda *a,**k: pytest.fail("legacy token selection"))
    seen=[]
    monkeypatch.setattr(followup,"compare_trials",lambda old,new,**kw: seen.append((old,new)) or {"decision":"needs_confirmation"})
    report=followup.run_followup(path,root,root/"new/run",source_run=source,qwen_package=root/"qwen",node_directory=root/"node",
                                 repeat_index=5,evolution_round=1)
    assert report["state"] == "completed" and seen
    assert report["selection_advisory"]["decision"] == "needs_confirmation"
    req=setup[-1]["runner"][0][0]
    assert req.repeat_index == 5 and req.pair_id.endswith("#r5")
    assert not report["selection_advisory"]["adopted"]


def test_total_protocol_round_chain_prevents_skipped_or_fourth_round(setup):
    root,path,source,_=total_profile(setup)
    for round_number, previous in [(2, []), (4, [])]:
        with pytest.raises(ValueError):
            followup.run_followup(path,root,root/"new/run",source_run=source,qwen_package=root/"qwen",node_directory=root/"node",
                                 repeat_index=6,evolution_round=round_number,previous_rounds=previous)
    assert not any(setup[-1].values())


def test_total_protocol_carries_previous_projection_and_policies_into_next_optimizer(setup, monkeypatch):
    root,path,source,_=total_profile(setup)
    mocks(setup, monkeypatch, review_callback=lambda request: PolicyReview("prefer_candidate", "Synthetic", executability=[SUPPORTED]))
    first=root/"new/round1"
    followup.run_followup(path,root,first,source_run=source,qwen_package=root/"qwen",node_directory=root/"node",
                          repeat_index=5,evolution_round=1)
    # Relocating deployment/configuration files does not change task/provider conditions.
    saved = json.loads((first / "run_settings.json").read_text())
    saved["project_root"] = "/old/deployment/code"
    saved["provider_config"] = "/old/deployment/code/provider.json"
    for task in saved["tasks"]:
        task.update(path="/old/deployment/task", instructions="/old/deployment/instruction.md")
    write(first / "run_settings.json", saved)
    second=root/"new/round2"
    report=followup.run_followup(path,root,second,source_run=source,qwen_package=root/"qwen",node_directory=root/"node",
                                 repeat_index=6,evolution_round=2,previous_rounds=[first])
    assert report["state"] == "completed"
    request=setup[-1]["optimizer"][-1][0]
    assert request.history_total == 2 and len(request.batch_results) == 2
    assert request.as_prompt_data()["evidence_scope"] == "cumulative"
    assert (second/"training_evidence/previous_round_1/projection.json").is_file()
    policy=json.loads((second/"training_evidence/previous_round_1/proposal_summary.json").read_text())
    assert "assignment_policy" in policy and "communication_policy" in policy
    assert "historical-run or held-out evidence is available" not in request.as_prompt_data()["instruction"]
    with pytest.raises(ValueError, match="chain"):
        followup.run_followup(path,root,root/"new/wrong",source_run=source,qwen_package=root/"qwen",node_directory=root/"node",
                              repeat_index=7,evolution_round=2,previous_rounds=[second])


@pytest.mark.parametrize("index", [None, True, -1, 1.5, "1"])
def test_callable_followup_requires_explicit_index_before_any_side_effect(setup, index):
    root, path, source, *_ = setup
    with pytest.raises(ValueError, match="Explicit nonnegative repeat_index"):
        followup.run_followup(path, root, root / "new/run", source_run=source,
                              qwen_package=root / "qwen", node_directory=root / "node", repeat_index=index)
    assert not any(setup[-1].values()) and not (root / "new/run").exists()


def test_saved_policy_bytes_are_exactly_the_dispatched_candidate(setup, monkeypatch):
    root, _, _, _, _, calls = setup
    mocks(setup, monkeypatch)
    assignment, communication = "assignment trailing spaces  \n\n", "communication\r\nlast line\t"
    class ExactOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, request):
            return PolicyProposal(assignment_policy=assignment, communication_policy=communication, rationale="test")
    class CaptureRunner:
        def __init__(self, **kwargs):
            pass
        def __call__(self, request):
            calls["runner"].append(request)
            return outcome(.08, status="completed")
    monkeypatch.setattr(followup, "StrategyOptimizer", ExactOptimizer)
    monkeypatch.setattr(followup, "ProgramBenchRunner", CaptureRunner)
    assert run(setup)["state"] == "completed"
    dispatched = calls["runner"][0]
    for name, text in (("assignment", dispatched.assignment_policy), ("communication", dispatched.communication_policy)):
        assert (root / f"new/run/policies/{name}_evolved.md").read_bytes() == text.encode()
        saved = json.loads((dispatched.output_dir / "run_request.json").read_text())
        assert saved[name + "_policy"] == text
    assert dispatched.assignment_policy == assignment and dispatched.communication_policy == communication


@pytest.mark.parametrize("state,running", [("optimizing", []), ("running_evolved", []),
                                           ("completed", [{"arm_id": "team_evolved"}]),
                                           ("blocked_review", None)])
def test_previous_round_must_be_terminal_before_next_output_or_credentials(setup, monkeypatch, state, running):
    root, path, source, _ = total_profile(setup)
    mocks(setup, monkeypatch, review_callback=lambda _: PolicyReview("uncertain", "Synthetic", executability=[]))
    first = root / "new/round1"
    followup.run_followup(path, root, first, source_run=source, qwen_package=root / "qwen",
                          node_directory=root / "node", repeat_index=5, evolution_round=1)
    progress = json.loads((first / "progress.json").read_text())
    write(first / "progress.json", {**progress, "state": state, "running": running})
    before = {key: len(value) for key, value in setup[-1].items()}
    with pytest.raises(ValueError, match="terminal"):
        followup.run_followup(path, root, root / "new/round2", source_run=source, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=6, evolution_round=2, previous_rounds=[first])
    assert not (root / "new/round2").exists()
    assert {key: len(value) for key, value in setup[-1].items()} == before


def known_runtime(setup, monkeypatch):
    root, _, source, trial, *_ = setup
    version = {"kind": "git", "commit": "a" * 40}
    monkeypatch.setattr(followup, "__file__", str(root / "scripts/run_programbench_evolved_followup.py"))
    monkeypatch.setattr(followup, "code_version", lambda _: version)
    monkeypatch.setattr(followup, "native_version", lambda _: "0.23.1")
    settings = json.loads((source / "run_settings.json").read_text())
    write(source / "run_settings.json", {**settings, "code_version": version, "qwen_version": "0.23.1",
                                         "platform": "linux/amd64"})
    return version


def official_artifacts(trial, passed):
    score = passed / 100
    result = outcome(score, status="completed")
    result.metrics["verifier"].update(passed_tests=passed, valid_tests=100)
    task_id = trial.parent.name
    write(trial / "verifier/filtered_test_results.json", {
        "schema_version": 1, "task_id": task_id, "official_status": "completed", "reward": score,
        "valid_tests": 100, "tests": [{"test_id": str(i)} for i in range(100)]})
    write(trial / "team/launch.json", {"native_compatibility_repairs": ["synthetic"]})
    return result


def approved_dispatch(first, destination):
    settings = json.loads((first / "run_settings.json").read_text())
    task = settings["tasks"][0]
    arm = next(arm for arm in settings["arms"] if arm["execution_mode"] == "team")
    proposal = json.loads((first / "proposal_summary.json").read_text())
    trial = destination / task["task_id"] / "approved_candidate"
    write(destination / "run_settings.json", {**settings, "repeat_index": 50,
          "arms": [{**arm, "arm_id": trial.name, "name": trial.name}]})
    write(trial / "run_request.json", {
        "task": {"task_id": task["task_id"], "repo_id": task["repository_id"]},
        "execution_mode": "team", "policy_version": arm["policy_version"], "fresh_start": True,
        "repeat_index": 50, "pair_id": task["task_id"] + "#r50",
        "assignment_policy": proposal["assignment_policy"], "communication_policy": proposal["communication_policy"]})
    write(trial / "task_result.json", asdict(official_artifacts(trial, 20)))
    write(destination / "progress.json", {"state": "completed", "running": [],
          "rows": [{"task_id": task["task_id"], "arm_id": trial.name, "inference_cleanup_confirmed": True}]})
    return trial


def paused_first_round(setup, monkeypatch):
    root, path, source, _ = total_profile(setup)
    known_runtime(setup, monkeypatch)
    mocks(setup, monkeypatch, review_callback=lambda _: PolicyReview("uncertain", "Synthetic", executability=[]))
    first = root / "new/round1"
    report = followup.run_followup(path, root, first, source_run=source, qwen_package=root / "qwen",
                                   node_directory=root / "node", repeat_index=5, evolution_round=1)
    assert report["state"] == "blocked_executability"
    assert not list(first.rglob("launch.json"))
    assert not list(first.rglob("run_request.json"))
    return first


def test_explicit_approved_dispatch_is_bound_and_included_in_cumulative_evidence(setup, monkeypatch):
    first = paused_first_round(setup, monkeypatch)
    root, path, source, *_ = setup
    trial = approved_dispatch(first, root / "approved")
    second = root / "new/round2"
    report = followup.run_followup(path, root, second, source_run=source, qwen_package=root / "qwen",
                                   node_directory=root / "node", repeat_index=6, evolution_round=2,
                                   previous_rounds=[first], approved_trials=[f"1={trial}"])
    assert report["training_rollouts_reused"] == 2
    assert report["training_evidence_rounds"] == 2
    assert "Evolution round 2" in report["interpretation"]
    item = json.loads((second / "training_evidence/previous_round_1/round.json").read_text())
    assert item["state"] == "blocked_executability" and item["approved_trial"] is True
    assert (second / "training_evidence/previous_round_1/projection.json").is_file()
    request = setup[-1]["optimizer"][-1][0]
    assert len(request.batch_results) == request.history_total == 2
    assert request.target_harness["scope"]["code_version"] == "git:aaaaaaaaaaaa"
    summary = json.loads((second / "proposal_summary.json").read_text())
    assert summary["training_rollouts"] == 2
    assert summary["training_evidence"] == ["baseline_team", "previous_round_1"]
    assert (second / "profile.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("unknown", [None, "", " ", "unknown", "Unknown", " UNKNOWN ", 231])
@pytest.mark.parametrize("side", ["round", "approved_trial", "both"])
def test_approved_dispatch_unknown_qwen_is_rejected_before_credentials_or_output(setup, monkeypatch, unknown, side):
    first = paused_first_round(setup, monkeypatch)
    root, path, source, *_ = setup
    trial = approved_dispatch(first, root / "approved")
    for name, directory in (("round", first), ("approved_trial", trial.parent.parent)):
        if side in {name, "both"}:
            settings_path = directory / "run_settings.json"
            settings = json.loads(settings_path.read_text())
            write(settings_path, {**settings, "qwen_version": unknown})
    before = {key: len(value) for key, value in setup[-1].items()}
    output = root / "new/round2"
    with pytest.raises(ValueError, match="Approved trial runtime conditions"):
        followup.run_followup(path, root, output, source_run=source, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=6, evolution_round=2,
                              previous_rounds=[first], approved_trials=[f"1={trial}"])
    assert not output.exists()
    assert {key: len(value) for key, value in setup[-1].items()} == before


@pytest.mark.parametrize("change", [
    "policy", "task", "conditions", "running", "missing_proposal", "result_identity",
    "policy_version", "execution_mode", "code_version_unknown", "code_version_different", "qwen_version_different",
    "round_code_version_unknown", "cleanup_row", "cleanup_result", "completed_round", "local_trial",
    "blocked_review", "blocked_provider", "blocked_proposal", "review_incomplete",
])
def test_approved_dispatch_mismatch_rejected_before_any_next_run_side_effect(setup, monkeypatch, change):
    first = paused_first_round(setup, monkeypatch)
    root, path, source, *_ = setup
    trial = approved_dispatch(first, root / "approved")
    if change in {"policy", "task", "policy_version", "execution_mode"}:
        file = trial / "run_request.json"
        data = json.loads(file.read_text())
        if change == "policy":
            data["assignment_policy"] += "\n"
        elif change == "task":
            data["task"]["task_id"] = "other"
        elif change == "policy_version":
            data["policy_version"] += 1
        else:
            data["execution_mode"] = "single"
    elif change == "result_identity":
        file = trial / "task_result.json"
        data = json.loads(file.read_text())
        data["arm_id"] = "another_trial"
    elif change == "conditions":
        file = trial.parent.parent / "run_settings.json"
        data = json.loads(file.read_text())
        data["inference"]["cpus"] += 1
    elif change in {"code_version_unknown", "code_version_different", "round_code_version_unknown"}:
        file = (first if change == "round_code_version_unknown" else trial.parent.parent) / "run_settings.json"
        data = json.loads(file.read_text())
        data["code_version"] = {"kind": "git", "commit": "b" * 40} if change == "code_version_different" else None
    elif change == "qwen_version_different":
        file = trial.parent.parent / "run_settings.json"
        data = json.loads(file.read_text())
        data["qwen_version"] = "0.23.2"
    elif change == "cleanup_row":
        file = trial.parent.parent / "progress.json"
        data = json.loads(file.read_text())
        data["rows"][0]["inference_cleanup_confirmed"] = False
    elif change == "cleanup_result":
        file = trial / "task_result.json"
        data = json.loads(file.read_text())
        data["metrics"]["inference_cleanup_confirmed"] = None
    elif change in {"completed_round", "blocked_review", "blocked_provider", "blocked_proposal"}:
        file = first / "progress.json"
        data = json.loads(file.read_text())
        data["state"] = "completed" if change == "completed_round" else change
    elif change == "review_incomplete":
        file = first / "review_summary.json"
        data = json.loads(file.read_text())
        data["status"] = "failed"
    elif change == "local_trial":
        file = first / trial.parent.name / "team_evolved/task_result.json"
        data = asdict(outcome())
    elif change == "running":
        file = trial.parent.parent / "progress.json"
        data = json.loads(file.read_text())
        data["running"] = [{"arm_id": trial.name}]
    else:
        file = first / "proposal_summary.json"
        data = {"status": "failed"}
    write(file, data)
    before = {key: len(value) for key, value in setup[-1].items()}
    with pytest.raises(ValueError):
        followup.run_followup(path, root, root / "new/round2", source_run=source, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=6, evolution_round=2,
                              previous_rounds=[first], approved_trials=[f"1={trial}"])
    assert not (root / "new/round2").exists()
    assert {key: len(value) for key, value in setup[-1].items()} == before


@pytest.mark.parametrize("existing", [False, True])
def test_history_trial_invalid_before_output_directory_or_credentials(setup, existing):
    root, path, source, _ = total_profile(setup)
    history = root / "bad_history"
    if existing:
        history.mkdir()
        write(history / "run_request.json", {})
    with pytest.raises((OSError, ValueError, KeyError)):
        followup.run_followup(path, root, root / "new/run", source_run=source, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=5, evolution_round=1,
                              history_trials=[history])
    assert not (root / "new/run").exists() and not any(setup[-1].values())


def test_unassessed_changed_rule_is_unknown_even_with_supported_other_rule(setup, monkeypatch):
    root, path, source, _ = total_profile(setup)
    mocks(setup, monkeypatch, review_callback=lambda _: PolicyReview("prefer_candidate", "Synthetic", executability=[SUPPORTED]))
    class WithChangedRule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, request):
            return PolicyProposal(assignment_policy="new assignment", communication_policy="new communication", rationale="Synthetic", execution_notes=[{
                "policy_part": "assignment", "rule_quote": "new assignment", "actor": "master",
                "observable_trigger": "task start", "existing_action": "agent",
                "capability_refs": ["/roles_and_tools/master"], "timing_and_fallback": "next turn",
                "cost_and_limitations": "one delegation"}])
    monkeypatch.setattr(followup, "StrategyOptimizer", WithChangedRule)
    report = followup.run_followup(path, root, root / "new/run", source_run=source, qwen_package=root / "qwen",
                                   node_directory=root / "node", repeat_index=5, evolution_round=1)
    assert report["state"] == "blocked_executability" and not setup[-1]["runner"]
    assessment = report["evolution"]["review_executability"]
    assert assessment["counts"] == {"supported": 1, "supported_with_conditions": 0, "unsupported": 0, "unknown": 0}
    assert assessment["unsupported_or_unknown"] == []
    assert assessment["unassessed_notes"][0]["rule_quote"] == "new assignment"
    assert assessment["dispatch_reason"] == "execution_notes_unassessed"
    assert not list((root / "new/run").rglob("launch.json"))


def test_real_followup_artifacts_compare_and_crlf_policy_roundtrips_through_comparison(setup, monkeypatch):
    from agent_company.programbench.comparison import prepare_comparison
    from agent_company.programbench.total_evidence import compare_trials, trial_record
    from scripts import run_programbench_policy_comparison as comparison
    root, path, source, baseline = total_profile(setup)
    known_runtime(setup, monkeypatch)
    write(baseline / "task_result.json", asdict(official_artifacts(baseline, 10)))
    mocks(setup, monkeypatch, review_callback=lambda _: PolicyReview("prefer_candidate", "Synthetic", executability=[SUPPORTED]))
    assignment, communication = "new assignment\r\nline two  \r\n", "new communication\r\nend\t"
    class Optimizer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, request):
            return PolicyProposal(assignment_policy=assignment, communication_policy=communication, rationale="Synthetic")
    class Runner:
        def __init__(self, **kwargs):
            pass
        def __call__(self, request):
            return official_artifacts(request.output_dir, 20)
    monkeypatch.setattr(followup, "StrategyOptimizer", Optimizer)
    monkeypatch.setattr(followup, "ProgramBenchRunner", Runner)
    out = root / "new/followup"
    report = followup.run_followup(path, root, out, source_run=source, qwen_package=root / "qwen",
                                   node_directory=root / "node", repeat_index=5, evolution_round=1)
    candidate = out / baseline.parent.name / "team_evolved"
    assert report["state"] == "completed"
    assert report["selection_advisory"]["decision"] == "needs_confirmation"
    assert compare_trials(baseline, candidate)["decision"] == "needs_confirmation"
    profile = json.loads(path.read_text())
    arm = {"name": "candidate", "execution_mode": "team", "policy_version": 4,
           "assignment_policy_path": str(out / "policies/assignment_evolved.md"),
           "communication_policy_path": str(out / "policies/communication_evolved.md")}
    profile.update(evolution_enabled=False, arms=[arm], parallelism=1)
    comparison_profile = root / "outside_code/confirmation_profile.json"
    write(comparison_profile, profile)
    prepared = prepare_comparison(comparison_profile, root, repeat_index=6)
    assert prepared.policies["candidate"] == (assignment, communication)
    monkeypatch.setattr(comparison, "__file__", str(root / "scripts/run_programbench_policy_comparison.py"))
    monkeypatch.setattr(comparison, "code_version", followup.code_version)
    monkeypatch.setattr(comparison, "native_version", followup.native_version)
    monkeypatch.setattr(comparison, "load_provider", followup.load_provider)
    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    repeated = root / "new/comparison"
    comparison.run_comparison(comparison_profile, root, repeated, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=6)
    replay = repeated / baseline.parent.name / "candidate"
    assert trial_record(candidate)["condition"] == trial_record(replay)["condition"]
    assert compare_trials(candidate, replay)["decision"] == "not_distinguished"
    assert (repeated / "profile.json").read_bytes() == comparison_profile.read_bytes()


def test_approved_dispatch_missing_score_stays_unknown_in_evidence(setup, monkeypatch):
    first = paused_first_round(setup, monkeypatch)
    root, path, source, *_ = setup
    trial = approved_dispatch(first, root / "approved")
    write(trial / "task_result.json", asdict(outcome(None, status="failed")))
    second = root / "new/round2"
    followup.run_followup(path, root, second, source_run=source, qwen_package=root / "qwen",
                          node_directory=root / "node", repeat_index=6, evolution_round=2,
                          previous_rounds=[first], approved_trials=[f"1={trial}"])
    request = setup[-1]["optimizer"][-1][0]
    assert request.batch_index == 2
    projection = json.loads((second / "training_evidence/previous_round_1/projection.json").read_text())
    assert projection["task_result"]["score"] is None
    assert request.batch_results[-1]["score"] is None


def test_cli_forwards_explicit_approved_trial_binding_without_dispatch_in_test(setup, monkeypatch, capsys):
    root, path, source, *_ = setup
    received = {}
    def capture(*args, **kwargs):
        received.update(kwargs)
        return {"state": "synthetic_no_execution"}
    monkeypatch.setattr(followup, "run_followup", capture)
    assert followup.main(["--profile", str(path), "--project-root", str(root), "--source-run", str(source),
                          "--run", "--output", str(root / "new/run"), "--qwen-package", str(root / "qwen"),
                          "--node-directory", str(root / "node"), "--repeat-index", "6", "--evolution-round", "2",
                          "--previous-round", str(root / "round1"), "--approved-trial", "1=/recorded/trial"]) == 0
    assert received["approved_trials"] == ["1=/recorded/trial"]
    assert received["previous_rounds"] == [root / "round1"]
    assert not (root / "new/run").exists()
    assert json.loads(capsys.readouterr().out)["state"] == "synthetic_no_execution"


def test_changed_rule_coverage_uses_validators_whitespace_normalization():
    assessed = [{**SUPPORTED, "rule_quote": "new\r\ncommunication"}]
    notes = [{"policy_part": "communication", "rule_quote": "new  communication"}]
    result = followup._review_coverage(assessed, notes)
    assert result["recorded"] is True and result["unsupported_or_unknown"] == []
    assert result["counts"]["supported"] == 1 and result["counts"]["unknown"] == 0


def test_explicit_approved_trial_binding_survives_into_third_round(setup, monkeypatch):
    first = paused_first_round(setup, monkeypatch)
    root, path, source, *_ = setup
    trial = approved_dispatch(first, root / "approved")
    second, third = root / "new/round2", root / "new/round3"
    followup.run_followup(path, root, second, source_run=source, qwen_package=root / "qwen",
                          node_directory=root / "node", repeat_index=6, evolution_round=2,
                          previous_rounds=[first], approved_trials=[f"1={trial}"])
    report = followup.run_followup(path, root, third, source_run=source, qwen_package=root / "qwen",
                                   node_directory=root / "node", repeat_index=7, evolution_round=3,
                                   previous_rounds=[first, second])
    assert report["training_rollouts_reused"] == 2
    assert report["training_evidence_rounds"] == 3
    assert "Finished trial evidence (scores may be unknown)" in report["interpretation"]
    summary = json.loads((third / "proposal_summary.json").read_text())
    assert summary["training_rollouts"] == 2 and summary["training_evidence_rounds"] == 3
    assert "Evolution round 3" in report["interpretation"]
    assert (third / "training_evidence/previous_round_1/projection.json").is_file()
    item = json.loads((third / "training_evidence/previous_round_1/round.json").read_text())
    assert item["approved_trial"] is True
    second_evidence = json.loads((third / "training_evidence/previous_round_2/round.json").read_text())
    assert second_evidence["state"] == "blocked_executability"
    assert not (third / "training_evidence/previous_round_2/projection.json").exists()
    assert setup[-1]["optimizer"][-1][0].batch_index == 3


@pytest.mark.parametrize("copy_quote", [False, True])
def test_review_quote_span_requires_note_copy_without_changing_reviewer_counts(setup, monkeypatch, copy_quote):
    root, path, source, _ = total_profile(setup)
    note_quote = "communication"
    assessed = {**SUPPORTED, "rule_quote": note_quote if copy_quote else "new communication"}
    mocks(setup, monkeypatch, review_callback=lambda _: PolicyReview("prefer_candidate", "Synthetic", executability=[assessed]))
    class Optimizer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, request):
            return PolicyProposal(assignment_policy="new assignment", communication_policy="new communication", rationale="Synthetic", execution_notes=[{
                "policy_part": "communication", "rule_quote": note_quote, "actor": "worker",
                "observable_trigger": "next turn", "existing_action": "send_message",
                "capability_refs": ["/wakeups"], "timing_and_fallback": "next turn",
                "cost_and_limitations": "one message"}])
    monkeypatch.setattr(followup, "StrategyOptimizer", Optimizer)
    report = followup.run_followup(path, root, root / "new/run", source_run=source,
                                  qwen_package=root / "qwen", node_directory=root / "node",
                                  repeat_index=5, evolution_round=1)
    assessment = report["evolution"]["review_executability"]
    assert assessment["counts"] == {"supported": 1, "supported_with_conditions": 0, "unsupported": 0, "unknown": 0}
    assert assessment["unsupported_or_unknown"] == []
    if copy_quote:
        assert report["state"] == "completed" and setup[-1]["runner"]
        assert assessment["unassessed_notes"] == []
    else:
        assert report["state"] == "blocked_executability" and not setup[-1]["runner"]
        assert assessment["dispatch_reason"] == "execution_notes_unassessed"
        assert "not assessed" in report["reason"]
        assert assessment["unassessed_notes"][0]["rule_quote"] == note_quote
        assert not list((root / "new/run").rglob("launch.json"))


def test_one_approved_trial_cannot_bind_two_rounds(setup, monkeypatch):
    first = paused_first_round(setup, monkeypatch)
    root, path, source, *_ = setup
    second, third = root / "new/round2", root / "new/round3"
    followup.run_followup(path, root, second, source_run=source, qwen_package=root / "qwen",
                          node_directory=root / "node", repeat_index=6, evolution_round=2, previous_rounds=[first])
    trial = approved_dispatch(first, root / "approved")
    before = {key: len(value) for key, value in setup[-1].items()}
    with pytest.raises(ValueError, match="only one round"):
        followup.run_followup(path, root, third, source_run=source, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=7, evolution_round=3,
                              previous_rounds=[first, second], approved_trials=[f"1={trial}", f"2={trial}"])
    assert not third.exists()
    assert {key: len(value) for key, value in setup[-1].items()} == before


@pytest.mark.parametrize("conflict", ["explicit", "inherited"])
def test_inherited_approved_binding_conflict_is_rejected(setup, monkeypatch, conflict):
    first = paused_first_round(setup, monkeypatch)
    root, path, source, *_ = setup
    trial = approved_dispatch(first, root / "approved")
    other = approved_dispatch(first, root / "other_approved")
    second, third = root / "new/round2", root / "new/round3"
    followup.run_followup(path, root, second, source_run=source, qwen_package=root / "qwen",
                          node_directory=root / "node", repeat_index=6, evolution_round=2,
                          previous_rounds=[first], approved_trials=[f"1={trial}"])
    if conflict == "inherited":
        settings = json.loads((first / "run_settings.json").read_text())
        write(first / "run_settings.json", {**settings, "approved_trials": {"1": str(other)}})
    before = {key: len(value) for key, value in setup[-1].items()}
    with pytest.raises(ValueError, match="disagree|cannot bind different"):
        followup.run_followup(path, root, third, source_run=source, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=7, evolution_round=3,
                              previous_rounds=[first, second],
                              approved_trials=[f"1={other}"] if conflict == "explicit" else [])
    assert not third.exists()
    assert {key: len(value) for key, value in setup[-1].items()} == before


def test_actual_comparison_trial_can_bind_paused_followup_and_enter_round_two(setup, monkeypatch):
    from scripts import run_programbench_policy_comparison as comparison
    first = paused_first_round(setup, monkeypatch)
    root, path, source, baseline, *_ = setup
    profile = json.loads(path.read_text())
    profile.update(evolution_enabled=False, parallelism=1, arms=[{
        "name": "approved_candidate", "execution_mode": "team", "policy_version": 4,
        "assignment_policy_path": str(first / "policies/assignment_evolved.md"),
        "communication_policy_path": str(first / "policies/communication_evolved.md")}])
    comparison_profile = root / "outside_code/approved_profile.json"
    write(comparison_profile, profile)
    class Runner:
        def __init__(self, **kwargs):
            pass
        def __call__(self, request):
            assert request.assignment_policy == "new assignment"
            assert request.communication_policy == "new communication"
            return official_artifacts(request.output_dir, 20)
    monkeypatch.setattr(comparison, "__file__", str(root / "scripts/run_programbench_policy_comparison.py"))
    monkeypatch.setattr(comparison, "code_version", followup.code_version)
    monkeypatch.setattr(comparison, "native_version", followup.native_version)
    monkeypatch.setattr(comparison, "load_provider", followup.load_provider)
    monkeypatch.setattr(comparison, "ProgramBenchRunner", Runner)
    approved = root / "actual_comparison"
    assert comparison.run_comparison(comparison_profile, root, approved, qwen_package=root / "qwen",
                                     node_directory=root / "node", repeat_index=50)["state"] == "completed"
    trial = approved / baseline.parent.name / "approved_candidate"
    second = root / "new/round2"
    report = followup.run_followup(path, root, second, source_run=source, qwen_package=root / "qwen",
                                   node_directory=root / "node", repeat_index=6, evolution_round=2,
                                   previous_rounds=[first], approved_trials=[f"1={trial}"])
    assert report["training_rollouts_reused"] == report["training_evidence_rounds"] == 2
    item = json.loads((second / "training_evidence/previous_round_1/round.json").read_text())
    assert item["approved_trial"] is True and item["state"] == "blocked_executability"
    request = setup[-1]["optimizer"][-1][0]
    assert request.batch_results[-1]["score"] == .2
    assert request.batch_results[-1]["evidence_path"] == "previous_round_1/projection.json"
    assert (approved / "profile.json").read_bytes() == comparison_profile.read_bytes()


@pytest.mark.parametrize("approved", [False, True])
def test_round_two_model_payloads_and_evidence_exclude_host_paths(setup, monkeypatch, approved):
    from test_programbench_total_prompts import production_payloads
    root, path, source, *_ = setup
    if approved:
        first = paused_first_round(setup, monkeypatch)
        trial = approved_dispatch(first, root / "separately_approved")
        bindings = [f"1={trial}"]
    else:
        total_profile(setup)
        known_runtime(setup, monkeypatch)
        mocks(setup, monkeypatch, review_callback=lambda _: PolicyReview(
            "prefer_candidate", "Synthetic", executability=[SUPPORTED]))
        first = root / "new/round1"
        followup.run_followup(path, root, first, source_run=source, qwen_package=root / "qwen",
                              node_directory=root / "node", repeat_index=5, evolution_round=1)
        bindings = []
    second = root / "new/round2"
    followup.run_followup(path, root, second, source_run=source, qwen_package=root / "qwen",
                          node_directory=root / "node", repeat_index=6, evolution_round=2,
                          previous_rounds=[first], approved_trials=bindings)
    request = setup[-1]["optimizer"][-1][0]
    # These are the real production request constructors, with scripted local
    # completions; include all later tool-result and correction payloads too.
    optimizer, reviewer = production_payloads(root / "capture", request=request)
    assert str(root) not in json.dumps([optimizer, reviewer])
    listing = request.evidence.list_files(limit=100)
    assert listing["next_offset"] is None
    entries = listing["files"]
    assert entries
    for entry in entries:
        assert not Path(entry["path"]).is_absolute()
        offset, pages = 0, []
        while offset is not None:
            contents = request.evidence.read_file(entry["path"], offset=offset)
            pages.append(contents["text"])
            offset = contents["next_offset"]
        assert str(root) not in "".join(pages)
    item = json.loads(request.evidence.read_file("previous_round_1/round.json")["text"])
    assert set(item) == {"state", "evidence_directory", "score"} | ({"approved_trial"} if approved else set())
    assert item["evidence_directory"] == "previous_round_1"
    assert item in request.change_history
    if approved:
        assert item["approved_trial"] is True
    # Host provenance remains available to the controller for chain validation.
    settings = json.loads((second / "run_settings.json").read_text())
    assert settings["source_run"] == str(source)
    assert settings["previous_rounds"] == [str(first)]
    assert settings["approved_trials"] == ({"1": str(trial)} if approved else {})
