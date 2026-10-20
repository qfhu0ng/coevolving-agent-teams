import json
from pathlib import Path

import pytest

from agent_company.roadmap.cli import load_manifest, main

from agent_company.runtime.provider import load_provider
from agent_company.roadmap.data import download_task


def task_fixture(root: Path, name: str):
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text("Implement the task.")
    (task / "task.toml").write_text(
        f'[metadata]\nname="{name}"\n[environment]\ndocker_image="example/task"\n'
    )
    (task / "tests/test.sh").write_text("exit 0\n")
    return task


def test_manifest_prepared_image_is_preserved(tmp_path):
    task_fixture(tmp_path, "training")
    manifest = tmp_path / "manifest.json"
    document = {"train": [{"path": "training", "repo_id": "demo",
                           "prepared_image": "local/offline:1"}], "test": []}
    manifest.write_text(json.dumps(document))
    _, train, _ = load_manifest(manifest)
    assert train[0].payload["prepared_image"] == "local/offline:1"
    document["train"][0]["prepared_image"] = " "
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="prepared_image"):
        load_manifest(manifest)


def test_download_has_complete_verifier_but_no_oracle_or_source(tmp_path):
    requested = []

    def fetch(url):
        requested.append(url)
        if "/api/datasets/" in url:
            return json.dumps([
                {"type": "file", "path": "tpl-5.0.0-roadmap/tests/test.sh"},
                {"type": "directory", "path": "tpl-5.0.0-roadmap/tests/nested"},
                {"type": "file", "path": "tpl-5.0.0-roadmap/tests/nested/check.cpp"},
            ]).encode()
        return b"fixture contents\n"

    path = download_task("tpl-5.0.0-roadmap", tmp_path, fetch=fetch)
    assert (path / "tests/nested/check.cpp").read_text() == "fixture contents\n"
    assert len(requested) == 5
    assert not any("solution" in url or "/environment/repo" in url for url in requested)
    assert json.loads((path / "data-source.json").read_text())["oracle_downloaded"] is False


def test_download_rejects_paths_outside_task(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        download_task("task", tmp_path, fetch=lambda _: json.dumps(
            [{"type": "file", "path": "task/tests/../../../outside"}]).encode())
    assert not (tmp_path / "outside").exists()


def test_manifest_paths_are_relative_to_manifest_and_repo_is_explicit(tmp_path):
    task_fixture(tmp_path / "data", "training")
    task_fixture(tmp_path / "data", "testing")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"train": [{"path": "data/training", "repo_id": "one"}],
                                "test": [{"path": "data/testing", "repo_id": "two"}]}))
    _, train, test = load_manifest(path)
    assert train[0].repo_id == "one" and test[0].repo_id == "two"
    assert train[0].payload["path"] == str(tmp_path / "data/training")
    path.write_text(json.dumps({"train": [{"path": "data/training"}], "test": []}))
    with pytest.raises(ValueError, match="repo_id"):
        load_manifest(path)


def test_provider_secret_stays_in_environment(tmp_path, monkeypatch):
    path = tmp_path / "provider.json"
    path.write_text(json.dumps({"model": "test-model", "base_url": "http://localhost:1234/v1",
                               "api_key_env": "STAGE1_TEST_KEY"}))
    monkeypatch.setenv("STAGE1_TEST_KEY", "synthetic-test-secret")
    config = load_provider(path)
    assert config.api_key == "synthetic-test-secret"
    assert "synthetic-test-secret" not in repr(config)
    assert "synthetic-test-secret" not in path.read_text()
    path.write_text(json.dumps({"api_key": "do-not-use-this-field"}))
    with pytest.raises(ValueError, match="api_key_env"):
        load_provider(path)


def test_inspect_executes_no_container_or_model(tmp_path, capsys):
    task = task_fixture(tmp_path, "inspect-task")
    assert main(["inspect", str(task), "--repo-id", "example-repository"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["repository_id"] == "example-repository"
    assert "instruction" not in info


def test_baseline_cli_resume_never_constructs_optimizer_or_reviewer(tmp_path, monkeypatch):
    from agent_company.evolution.loop import TaskResult
    import agent_company.roadmap.cli as module
    task_fixture(tmp_path, "one")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"path": "one", "repo_id": "example"}]}))
    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps({"model": "fake-only", "base_url": "http://localhost"}))
    calls = []

    def run(request):
        assert request.execution_mode == "single"
        assert request.assignment_policy == request.communication_policy == ""
        calls.append(request.task.task_id)
        return TaskResult("completed", score=1)

    monkeypatch.setattr(module, "RoadmapRunner", lambda *a, **k: run)
    monkeypatch.setattr(module, "StrategyOptimizer", lambda *a, **k: pytest.fail("No optimizer"))
    monkeypatch.setattr(module, "StrategyReviewer", lambda *a, **k: pytest.fail("No reviewer"))
    args = ["baseline", "--manifest", str(manifest), "--provider-config", str(provider),
            "--qwen-package", str(tmp_path), "--node-directory", str(tmp_path),
            "--output", str(tmp_path / "result")]
    assert main(args) == 0
    assert main([*args, "--resume"]) == 0
    assert calls == ["one"]
    (tmp_path / "one/instruction.md").write_text("Changed task")
    with pytest.raises(ValueError, match="settings differ"):
        main([*args, "--resume"])


def execution_args(tmp_path, command):
    task = task_fixture(tmp_path, "training")
    task_fixture(tmp_path, "testing")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": [{"path": "training", "repo_id": "train-repo"}],
                                    "test": [{"path": "testing", "repo_id": "test-repo"}]}))
    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps({"model": "fake-only", "base_url": "http://localhost"}))
    args = [command, "--provider-config", str(provider), "--qwen-package", str(tmp_path),
            "--node-directory", str(tmp_path), "--output", str(tmp_path / "result")]
    if command == "run-task":
        args += ["--task", str(task), "--repo-id", "train-repo"]
    else:
        args += ["--manifest", str(manifest)]
    return args


@pytest.mark.parametrize("command", ["run-task", "baseline"])
@pytest.mark.parametrize("mode", [None, "single", "team"])
def test_cli_selects_real_solo_or_team_and_only_team_reads_policies(tmp_path, monkeypatch,
                                                                capsys, command, mode):
    from agent_company.evolution.loop import TaskResult
    import agent_company.roadmap.cli as module
    requests = []

    def run(request):
        requests.append(request)
        return TaskResult("completed", score=1)

    monkeypatch.setattr(module, "RoadmapRunner", lambda *a, **k: run)
    monkeypatch.setattr(module, "StrategyOptimizer", lambda *a, **k: pytest.fail("No optimizer"))
    monkeypatch.setattr(module, "StrategyReviewer", lambda *a, **k: pytest.fail("No reviewer"))
    args = execution_args(tmp_path, command)
    if mode:
        args += ["--mode", mode]
    assignment, communication = tmp_path / "assignment.md", tmp_path / "communication.md"
    if mode == "team":
        assignment.write_text("choose scopes and dependencies")
        communication.write_text("notify affected owners")
    # Intentionally nonexistent in solo: neither a default nor explicit policy is consumed.
    args += ["--assignment-policy", str(assignment), "--communication-policy", str(communication)]
    assert main(args) == 0
    expected = mode or "single"
    assert requests and all(r.execution_mode == expected for r in requests)
    assert all(r.assignment_policy == (assignment.read_text() if mode == "team" else "")
               for r in requests)
    assert all(r.communication_policy == (communication.read_text() if mode == "team" else "")
               for r in requests)
    printed = json.loads(capsys.readouterr().out)
    assert printed["execution_mode"] == expected
    if command == "run-task":
        saved = json.loads((tmp_path / "result/run_request.json").read_text())
        assert saved["execution_mode"] == expected
    else:
        with pytest.raises(ValueError, match="settings differ"):
            main([*args, "--mode", "single" if expected == "team" else "team", "--resume",
                  "--assignment-policy", str(module.POLICIES / "assignment.md"),
                  "--communication-policy", str(module.POLICIES / "communication.md")])


@pytest.mark.parametrize("reference_mode", [None, "single", "team"])
def test_legacy_evolution_cli_trains_team_and_uses_requested_heldout_reference(tmp_path, monkeypatch,
                                                                            capsys, reference_mode):
    from agent_company.evolution.loop import PolicyProposal, TaskResult
    import agent_company.roadmap.cli as module
    requests, proposals = [], []

    def run(request):
        requests.append(request)
        return TaskResult("completed", score=1)

    def propose(request):
        proposals.append(request)
        return PolicyProposal("evolved handoffs", "evolved assignment")

    monkeypatch.setattr(module, "RoadmapRunner", lambda *a, **k: run)
    monkeypatch.setattr(module, "StrategyOptimizer", lambda *a, **k: propose)
    monkeypatch.setattr(module, "StrategyReviewer", lambda *a, **k: pytest.fail("No legacy reviewer"))
    args = [*execution_args(tmp_path, "run"), "--selection-mode", "legacy_no_replay"]
    if reference_mode:
        args += ["--baseline-mode", reference_mode]
    assert main(args) == 0
    assert len(proposals) == 1
    train = [r for r in requests if r.split == "train"]
    heldout = [r for r in requests if r.split == "test"]
    assert len(train) == 1 and train[0].execution_mode == "team"
    expected = reference_mode or "single"
    assert len(heldout) == 2
    assert [r.execution_mode for r in heldout] == [expected, "team"]
    assert heldout[-1].assignment_policy == "evolved assignment"
    assert heldout[-1].communication_policy == "evolved handoffs"
    if expected == "single":
        assert heldout[0].assignment_policy == heldout[0].communication_policy == ""
    printed = json.loads(capsys.readouterr().out)
    assert printed["baseline_mode"] == expected
    assert ("test_baseline" if expected == "single" else "test_initial") in printed
    assert printed["test_final"]["runs"] == 1


@pytest.fixture
def wired_evolution(monkeypatch):
    """Inspect CLI wiring without invoking models, containers or the controller."""
    import agent_company.roadmap.cli as module
    seen = {}
    optimizer, reviewer = object(), object()

    def make_optimizer(provider, output):
        seen["optimizer"] = (provider, output)
        return optimizer

    def make_reviewer(provider, output, **limits):
        seen["reviewer"] = (provider, output)
        seen["reviewer_limits"] = limits
        return reviewer

    def run(**kwargs):
        seen["run"] = kwargs
        assert kwargs["propose_policy"] is optimizer
        assert kwargs["review_policy"] is (reviewer if kwargs["selection_mode"] == "review_replay" else None)
        return {"status": "completed", "baseline_mode": "single", "final_policy_version": 1,
                "reviewer": {"runs": 1}, "replay": {"runs": 5},
                "selection_mode": kwargs["selection_mode"], "selection": {"accepted": 1},
                "evidence_scope": kwargs["evidence_scope"] or (
                    "current_batch" if kwargs["selection_mode"] == "review_replay" else "cumulative"),
                "selection_results": [{"accepted": True}],
                "all_costs": {"cost_usd": None, "known_cost_usd": 0.25}}

    monkeypatch.setattr(module, "RoadmapRunner", lambda *a, **k: object())
    monkeypatch.setattr(module, "StrategyOptimizer", make_optimizer)
    monkeypatch.setattr(module, "StrategyReviewer", make_reviewer)
    monkeypatch.setattr(module, "run_stage1", run)
    return seen


@pytest.mark.parametrize("separate_reviewer", [False, True])
def test_review_replay_default_wires_independent_reviewer_and_public_config(
        tmp_path, monkeypatch, capsys, wired_evolution, separate_reviewer):
    args = execution_args(tmp_path, "run")
    monkeypatch.setenv("STAGE1_OPTIMIZER_TEST_KEY", "synthetic-optimizer-secret")
    optimizer_path = tmp_path / "optimizer.json"
    optimizer_path.write_text(json.dumps({"model": "optimizer-fixture", "base_url": "http://localhost",
                                          "api_key_env": "STAGE1_OPTIMIZER_TEST_KEY"}))
    args += ["--optimizer-provider-config", str(optimizer_path)]
    if separate_reviewer:
        monkeypatch.setenv("STAGE1_REVIEWER_TEST_KEY", "synthetic-reviewer-secret")
        reviewer_path = tmp_path / "reviewer.json"
        reviewer_path.write_text(json.dumps({"model": "reviewer-fixture", "base_url": "http://localhost",
                                             "api_key_env": "STAGE1_REVIEWER_TEST_KEY",
                                             "rates": {"input_per_million": "1", "cached_input_per_million": "0.1",
                                                       "output_per_million": "2"}}))
        args += ["--reviewer-provider-config", str(reviewer_path)]
    assert main(args) == 0
    wired = wired_evolution["run"]
    assert wired["batch_size"] == 5
    assert wired["candidates_per_batch"] == 1
    assert wired["replay_task_count"] == 2
    assert wired["replay_selection"] == "reviewer"
    assert wired["selection_mode"] == "review_replay"
    assert wired["evidence_scope"] is None  # The controller resolves the mode-specific default.
    assert wired_evolution["reviewer_limits"] == {"max_tool_rounds": 8, "max_context_chars": 180000}
    optimizer_provider, optimizer_output = wired_evolution["optimizer"]
    reviewer_provider, reviewer_output = wired_evolution["reviewer"]
    assert optimizer_provider.model == "optimizer-fixture"
    assert reviewer_provider.model == ("reviewer-fixture" if separate_reviewer else "optimizer-fixture")
    if not separate_reviewer:
        assert reviewer_provider is optimizer_provider
    assert optimizer_output == tmp_path / "result/optimizer_traces"
    assert reviewer_output == tmp_path / "result/reviewer_traces"
    public = wired["runtime_config"]
    assert public["reviewer_provider"]["model"] == reviewer_provider.model
    assert public["reviewer_limits"] == wired_evolution["reviewer_limits"]
    assert "api_key" not in public["reviewer_provider"]
    assert "api_key" not in public["optimizer_provider"]
    assert "synthetic-reviewer-secret" not in json.dumps(public)
    assert "synthetic-optimizer-secret" not in json.dumps(public)
    printed = json.loads(capsys.readouterr().out)
    for key in ("reviewer", "replay", "selection", "selection_results", "all_costs"):
        assert key in printed
    assert printed["selection_mode"] == "review_replay"
    assert printed["evidence_scope"] == "current_batch"


@pytest.mark.parametrize(("manifest_config", "overrides", "expected"), [
    ({}, [], {"quality_tolerance": 0.0, "min_score_gain": 0.0, "min_cost_reduction_fraction": 0.0}),
    ({"quality_tolerance": 0.01, "min_score_gain": 0.02, "min_cost_reduction_fraction": 0.1}, [],
     {"quality_tolerance": 0.01, "min_score_gain": 0.02, "min_cost_reduction_fraction": 0.1}),
    ({"quality_tolerance": 0.01, "min_score_gain": 0.02, "min_cost_reduction_fraction": 0.1},
     ["--quality-tolerance", "0", "--min-score-gain", "0.03"],
     {"quality_tolerance": 0.0, "min_score_gain": 0.03, "min_cost_reduction_fraction": 0.1}),
    ({}, ["--min-cost-reduction-fraction", "0.2"],
     {"quality_tolerance": 0.0, "min_score_gain": 0.0, "min_cost_reduction_fraction": 0.2}),
])
def test_selection_thresholds_use_cli_over_manifest_over_defaults(tmp_path, wired_evolution,
                                                                 manifest_config, overrides, expected):
    from dataclasses import asdict
    args = execution_args(tmp_path, "run")
    manifest = tmp_path / "manifest.json"
    document = json.loads(manifest.read_text())
    document["selection_config"] = {**manifest_config, "efficiency_metric": "cost_usd"}
    document["batch_size"] = 7
    manifest.write_text(json.dumps(document))
    assert main([*args, *overrides]) == 0
    assert asdict(wired_evolution["run"]["selection_config"]) == {
        **expected, "efficiency_metric": "cost_usd", "min_token_reduction_fraction": 0.0}
    assert wired_evolution["run"]["batch_size"] == 7


def test_legacy_requires_explicit_flag_and_keeps_network_policy(tmp_path, wired_evolution):
    args = execution_args(tmp_path, "run")
    assert main([*args, "--selection-mode", "legacy_no_replay", "--batch-size", "4",
                 "--reviewer-provider-config", str(tmp_path / "unused-reviewer.json")]) == 0
    wired = wired_evolution["run"]
    assert wired["selection_mode"] == "legacy_no_replay" and wired["review_policy"] is None
    assert wired["selection_config"] is None
    assert wired["evidence_scope"] is None
    assert wired["batch_size"] == 4
    assert "reviewer" not in wired_evolution
    assert set(wired["runtime_config"]) == {
        "provider", "qwen_package", "node_directory", "platform", "team_stall_timeout",
        "provider_supports_seed", "task_inputs", "optimizer_provider", "network_policy",
    }


@pytest.mark.parametrize("mode", ["single", "team"])
def test_network_policy_is_saved_and_open_network_checkpoint_cannot_resume(tmp_path, monkeypatch, mode):
    from agent_company.evolution.loop import TaskResult
    from agent_company.runtime.network import NETWORK_POLICY
    import agent_company.roadmap.cli as module
    calls = []

    def run(request):
        calls.append(request)
        return TaskResult("completed", score=1)

    monkeypatch.setattr(module, "RoadmapRunner", lambda *a, **k: run)
    args = [*execution_args(tmp_path, "baseline"), "--mode", mode]
    assert main(args) == 0
    settings_path = tmp_path / "result/run_settings.json"
    settings = json.loads(settings_path.read_text())
    assert settings["runtime"]["network_policy"] == NETWORK_POLICY
    completed_calls = len(calls)
    assert main([*args, "--resume"]) == 0
    assert len(calls) == completed_calls
    settings["runtime"].pop("network_policy")
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="settings differ"):
        main([*args, "--resume"])
    assert len(calls) == completed_calls


def test_manifest_cannot_silently_enable_legacy_no_replay(tmp_path, wired_evolution):
    args = execution_args(tmp_path, "run")
    manifest = tmp_path / "manifest.json"
    document = json.loads(manifest.read_text())
    document["selection_mode"] = "legacy_no_replay"
    manifest.write_text(json.dumps(document))
    assert main(args) == 0
    assert wired_evolution["run"]["selection_mode"] == "review_replay"


@pytest.mark.parametrize(("selection_mode", "scope"), [
    ("review_replay", None), ("review_replay", "current_batch"),
    ("review_replay", "cumulative"), ("legacy_no_replay", None),
    ("legacy_no_replay", "cumulative"),
])
def test_evidence_scope_is_explicit_and_kept_out_of_runtime_config(tmp_path, wired_evolution,
                                                                 capsys, selection_mode, scope):
    args = [*execution_args(tmp_path, "run"), "--selection-mode", selection_mode]
    if scope is not None:
        args += ["--evidence-scope", scope]
    assert main(args) == 0
    wired = wired_evolution["run"]
    assert wired["evidence_scope"] == scope
    assert "evidence_scope" not in wired["runtime_config"]
    expected = scope or ("current_batch" if selection_mode == "review_replay" else "cumulative")
    assert json.loads(capsys.readouterr().out)["evidence_scope"] == expected


@pytest.mark.parametrize("cli_override", [False, True])
def test_candidate_replay_and_review_limits_are_configurable(tmp_path, wired_evolution, cli_override):
    args = execution_args(tmp_path, "run")
    manifest = tmp_path / "manifest.json"
    document = json.loads(manifest.read_text())
    document.update(batch_size=9, candidates_per_batch=3, replay_task_count=4, replay_selection="reviewer",
                    reviewer_max_tool_rounds=12, reviewer_max_context_chars=150000)
    manifest.write_text(json.dumps(document))
    if cli_override:
        args += ["--batch-size", "6", "--candidates-per-batch", "2", "--replay-task-count", "3",
                 "--reviewer-max-tool-rounds", "10", "--reviewer-max-context-chars", "120000",
                 "--replay-selection", "first_n"]
    assert main(args) == 0
    wired = wired_evolution["run"]
    assert wired["batch_size"] == (6 if cli_override else 9)
    assert wired["candidates_per_batch"] == (2 if cli_override else 3)
    assert wired["replay_task_count"] == (3 if cli_override else 4)
    assert wired["replay_selection"] == ("first_n" if cli_override else "reviewer")
    assert wired_evolution["reviewer_limits"] == {
        "max_tool_rounds": 10 if cli_override else 12,
        "max_context_chars": 120000 if cli_override else 150000,
    }
    assert wired["runtime_config"]["reviewer_limits"] == wired_evolution["reviewer_limits"]


@pytest.mark.parametrize("overrides", [["--candidates-per-batch", "2"], ["--replay-task-count", "1"],
                                      ["--replay-selection", "reviewer"]])
def test_legacy_rejects_candidate_or_replay_configuration(tmp_path, wired_evolution, overrides):
    args = execution_args(tmp_path, "run")
    with pytest.raises(ValueError, match="legacy_no_replay requires"):
        main([*args, "--selection-mode", "legacy_no_replay", *overrides])
    assert "optimizer" not in wired_evolution
    assert "reviewer" not in wired_evolution


def test_legacy_rejects_selection_thresholds_instead_of_ignoring_them(tmp_path, wired_evolution):
    args = execution_args(tmp_path, "run")
    with pytest.raises(ValueError, match="legacy_no_replay does not accept selection thresholds"):
        main([*args, "--selection-mode", "legacy_no_replay", "--quality-tolerance", "0.1"])
    assert "optimizer" not in wired_evolution


def test_reviewer_initialization_failure_does_not_fall_back_to_legacy(tmp_path, monkeypatch,
                                                                   wired_evolution):
    import agent_company.roadmap.cli as module

    def fail(*args, **kwargs):
        raise ValueError("Reviewer setup failed")

    monkeypatch.setattr(module, "StrategyReviewer", fail)
    with pytest.raises(ValueError, match="Reviewer setup failed"):
        main(execution_args(tmp_path, "run"))
    assert "run" not in wired_evolution


def test_selection_config_must_be_manifest_object(tmp_path, wired_evolution):
    args = execution_args(tmp_path, "run")
    manifest = tmp_path / "manifest.json"
    document = json.loads(manifest.read_text())
    document["selection_config"] = [0, 0, 0]
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="selection_config must be an object"):
        main(args)
    assert "optimizer" not in wired_evolution
    assert "reviewer" not in wired_evolution
