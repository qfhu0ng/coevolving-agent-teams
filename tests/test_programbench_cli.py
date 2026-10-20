import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from agent_company.programbench import cli
from agent_company.evolution.loop import TaskResult
from test_programbench import task as task_fixture

task = task_fixture


def test_prepare_public_official_prompt_drops_only_runtime_conventions(tmp_path):
    official = Path("external/mini-swe-agent-study/src/minisweagent/config/benchmarks/programbench.yaml")
    if not official.exists():
        pytest.skip("Optional public official checkout not installed")
    output = tmp_path / "instruction.md"
    text = cli.prepare_instructions(official, output)
    assert "MUST NOT decompile" in text and "compile.sh" in text
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in text
    assert "/opt/ac-programbench-reference/executable" in text
    with pytest.raises(FileExistsError):
        cli.prepare_instructions(official, output)


def test_describe_does_not_load_provider(task, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_provider", lambda *a: pytest.fail("credentials read"))
    cli.main(["describe", "--task", str(task.path), "--instructions", str(task.instruction_path)])
    assert json.loads(capsys.readouterr().out)["task_id"] == task.task_id


@pytest.mark.parametrize("cleanup_confirmed", [True, False])
def test_pair_exactly_one_per_mode_and_no_overwrite(task, tmp_path, monkeypatch, cleanup_confirmed):
    from agent_company.programbench import runner as programbench_runner
    from agent_company.runtime.relay import ProviderConfig
    captured = []

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs["max_model_calls"] == 1000
            assert kwargs["task_cpus"] == 20
            assert kwargs["agent_timeout_seconds"] == 21600

        def __call__(self, request):
            captured.append(request)
            return TaskResult(status="completed", score=0.0,
                              metrics={"inference_cleanup_confirmed": cleanup_confirmed})

    monkeypatch.setattr(programbench_runner, "ProgramBenchRunner", Runner)
    monkeypatch.setattr(cli, "load_provider", lambda *a: ProviderConfig("synthetic", "http://unused"))
    args = SimpleNamespace(profile=Path("configs/stage1/programbench_glm_p3.json"), task=task.path,
                           instructions=task.instruction_path, project_root=Path.cwd(),
                           qwen_package=tmp_path, node_directory=tmp_path, platform="linux/amd64",
                           docker="docker", output=tmp_path / "pair")
    if not cleanup_confirmed:
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
            cli.run_pair(args)
        assert len(captured) == 1
        assert json.loads((args.output / "progress.json").read_text())["terminal"] == 1
        return
    assert cli.run_pair(args)["terminal"] == 2
    assert [r.execution_mode for r in captured] == ["single", "team"]
    assert captured[0].assignment_policy == captured[0].communication_policy == ""
    assert captured[1].assignment_policy and captured[1].communication_policy
    assert json.loads((args.output / "single/run_request.json").read_text())["fresh_start"]
    with pytest.raises(FileExistsError):
        cli.run_pair(args)
    assert len(captured) == 2


@pytest.fixture
def pair_args(task, tmp_path, monkeypatch):
    from agent_company.runtime.relay import ProviderConfig
    monkeypatch.setattr(cli, "load_provider", lambda *a: ProviderConfig("synthetic", "http://unused"))
    return SimpleNamespace(profile=Path("configs/stage1/programbench_glm_p3.json"), task=task.path,
                           instructions=task.instruction_path, project_root=Path.cwd(),
                           qwen_package=tmp_path, node_directory=tmp_path, platform="linux/amd64",
                           docker="synthetic-docker-wrapper", output=tmp_path / "pair", parallelism=2)


def test_parallel_pair_overlaps_and_keeps_policies_outputs_and_usage_independent(pair_args, monkeypatch):
    from agent_company.programbench import runner as programbench_runner
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    requests, runners, writer_threads = [], [], []
    scheduling_thread = threading.get_ident()
    profile = json.loads(pair_args.profile.read_text())
    paths = {key: pair_args.project_root / profile[key] for key in
             ("assignment_policy_path", "communication_policy_path")}
    original = {key: path.read_bytes() for key, path in paths.items()}
    actual_write = cli.write_json_atomic

    def record_write(path, value):
        if path.name in {"progress.json", "task_result.json"}:
            writer_threads.append(threading.get_ident())
        return actual_write(path, value)

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs["docker"] == "synthetic-docker-wrapper"
            assert kwargs["task_cpus"] == 20 and kwargs["task_memory_mb"] == 61440
            assert kwargs["agent_timeout_seconds"] == 21600 and kwargs["max_model_calls"] == 1000
            with lock:
                runners.append(self)

        def __call__(self, request):
            with lock:
                requests.append(request)
            barrier.wait(timeout=3)  # Both calls must run before either can finish.
            assert len(requests) == 2
            tokens = 11 if request.execution_mode == "single" else 22
            return TaskResult(status="completed", score=tokens / 100,
                              usage={"total_tokens": tokens}, cost_usd=tokens / 1000,
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(programbench_runner, "ProgramBenchRunner", Runner)
    monkeypatch.setattr(cli, "write_json_atomic", record_write)
    result = cli.run_pair(pair_args)
    assert result["terminal"] == result["launched"] == 2 and result["parallelism"] == 2
    assert result["requires_user_action"] is False
    assert len(runners) == 2 and runners[0] is not runners[1]
    assert set(writer_threads) == {scheduling_thread}
    modes = {request.execution_mode: request for request in requests}
    assert set(modes) == {"single", "team"}
    assert modes["single"].assignment_policy == modes["single"].communication_policy == ""
    assert modes["team"].assignment_policy == original["assignment_policy_path"].decode()
    assert modes["team"].communication_policy == original["communication_policy_path"].decode()
    assert modes["team"].policy_version == 3
    assert all(request.fresh_start and request.repeat_index == 0 for request in requests)
    assert modes["single"].output_dir != modes["team"].output_dir
    for mode, tokens in (("single", 11), ("team", 22)):
        row = json.loads((pair_args.output / mode / "task_result.json").read_text())
        assert row["usage"] == {"total_tokens": tokens} and row["score"] == tokens / 100
        assert row["cost_usd"] == tokens / 1000
        request = json.loads((pair_args.output / mode / "run_request.json").read_text())
        assert request["execution_mode"] == mode and request["task"]["task_id"] == pair_args.task.name
    settings = json.loads((pair_args.output / "run_settings.json").read_text())
    assert settings["new_rollouts"] == 2 and settings["reused_rollouts"] == 0
    assert settings["evolution_enabled"] is False and settings["automatic_restarts"] is False
    assert settings["parallelism"] == 2
    assert all(path.read_bytes() == original[key] for key, path in paths.items())
    with pytest.raises(FileExistsError):
        cli.run_pair(pair_args)
    assert len(requests) == len(runners) == 2


@pytest.mark.parametrize("raised", [True, False])
def test_parallel_failure_or_unknown_cleanup_does_not_cancel_or_repeat_other_arm(pair_args, monkeypatch, raised):
    from agent_company.programbench import runner as programbench_runner
    barrier = threading.Barrier(2)
    calls = []

    class Runner:
        def __init__(self, **kwargs):
            pass

        def __call__(self, request):
            calls.append(request.execution_mode)
            barrier.wait(timeout=3)
            if request.execution_mode == "single":
                if raised:
                    raise RuntimeError("PRIVATE_PROVIDER_CREDENTIAL_AND_TRACE")
                return TaskResult(status="failed", metrics={"inference_cleanup_confirmed": False})
            return TaskResult(status="completed", score=0.75, usage={"total_tokens": 17},
                              metrics={"inference_cleanup_confirmed": True})

    monkeypatch.setattr(programbench_runner, "ProgramBenchRunner", Runner)
    result = cli.run_pair(pair_args)
    assert sorted(calls) == ["single", "team"]
    assert result["terminal"] == result["launched"] == 2
    assert result["requires_user_action"] is True and result["cleanup_attention_modes"] == ["single"]
    rows = {row["mode"]: row for row in result["rows"]}
    assert rows["single"]["status"] == "failed" and rows["single"]["score"] is None
    assert rows["team"]["status"] == "completed" and rows["team"]["score"] == 0.75
    team = json.loads((pair_args.output / "team/task_result.json").read_text())
    assert team["usage"] == {"total_tokens": 17}
    if raised:
        single = json.loads((pair_args.output / "single/task_result.json").read_text())
        assert single["error"] == "RuntimeError" and single["metrics"]["runner_exception_type"] == "RuntimeError"
        assert single["metrics"]["inference_cleanup_confirmed"] is None
    assert all("PRIVATE_" not in path.read_text() for path in pair_args.output.rglob("*.json"))


def test_sequential_exception_records_type_and_does_not_launch_next_arm(pair_args, monkeypatch):
    from agent_company.programbench import runner as programbench_runner
    pair_args.parallelism = 1
    calls = []

    class Runner:
        def __init__(self, **kwargs):
            pass

        def __call__(self, request):
            calls.append(request.execution_mode)
            raise OSError("PRIVATE_FAILURE_BODY")

    monkeypatch.setattr(programbench_runner, "ProgramBenchRunner", Runner)
    with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
        cli.run_pair(pair_args)
    assert calls == ["single"]
    progress = json.loads((pair_args.output / "progress.json").read_text())
    assert progress["terminal"] == progress["launched"] == 1 and progress["requires_user_action"]
    assert progress["rows"][0]["exception_type"] == "OSError"
    assert not (pair_args.output / "team/task_result.json").exists()


@pytest.mark.parametrize("parallelism", [0, 3, -1, True, "2", 1.0, None])
def test_illegal_parallelism_is_rejected_before_output_or_provider(pair_args, monkeypatch, parallelism):
    pair_args.parallelism = parallelism
    monkeypatch.setattr(cli, "load_provider", lambda *a: pytest.fail("Provider should not load"))
    with pytest.raises(ValueError, match="parallelism"):
        cli.run_pair(pair_args)
    assert not pair_args.output.exists()


@pytest.mark.parametrize("arguments,expected", [([], 1), (["--parallelism", "2"], 2)])
def test_cli_parallelism_default_and_option(task, monkeypatch, arguments, expected, capsys):
    captured = []
    monkeypatch.setattr(cli, "run_pair", lambda args: captured.append(args.parallelism) or {"terminal": 0})
    cli.main(["run-pair", "--task", str(task.path), "--instructions", str(task.instruction_path),
              "--profile", "unused.json", "--qwen-package", "unused", "--node-directory", "unused",
              "--output", "unused", *arguments])
    assert captured == [expected]
    assert json.loads(capsys.readouterr().out) == {"terminal": 0}
