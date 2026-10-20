"""Opt-in real Docker observation/export smoke; no model, Qwen or evaluator.

Run this file alone after reviewing the checkout, with a locally cached public
ProgramBench task directory (only task.yaml is read). Example, on the execution
host with its existing Docker wrapper and cached cleanroom/network helper images:

  STAGE1_PROGRAMBENCH_OBSERVER_TASK=/path/to/jgm__pandoc.5caad90 \
  STAGE1_PROGRAMBENCH_OBSERVER_DOCKER=/path/to/tools/docker \
  STAGE1_PROGRAMBENCH_OBSERVER_OUTPUT=/path/to/new-smoke-output \
  python -m pytest -q tests/test_programbench_observer_docker.py

Absent TASK means skip, following the existing opt-in Docker test convention.
Each parameter creates its own fresh container. Only this test's owned containers
are stopped/removed. The synthetic scenario uses 1 CPU/1 GiB and period 60 seconds;
it manually samples state changes, not a six-hour periodic-load experiment.
Team registration events are explicitly synthetic fixtures, not native agent
lifecycle evidence. A pass requires readable cgroup-v2 counters on this host;
unavailable counters remain null in saved evidence and fail this smoke assertion.
observer_smoke.json records whole-smoke elapsed time, per-sample wall time and
the last inference memory peak. Each bounded sample must complete within 10 seconds
(the existing 3-second command timeout plus host scheduling margin).
Network-boundary setup uses the existing ProgramBenchWorkspace implementation.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest

from agent_company.programbench.task import ProgramBenchTask, create_submission
from agent_company.programbench.observation import ProgramBenchObserver
from agent_company.programbench.workspace import ProgramBenchWorkspace


SYNTHETIC_DELIVERY = {
    "observer-delivery.txt": "OBSERVER_SYNTHETIC_DELIVERY\n",
    "compile.sh": "#!/bin/sh\nset -eu\ncp observer-delivery.txt executable\n",
}


def _rows(path):
    return {row["phase"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def test_synthetic_delivery_satisfies_submission_contract(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name, content in SYNTHETIC_DELIVERY.items():
        (source / name).write_text(content, encoding="utf-8")
    submission = create_submission(source, tmp_path / "submission.tar.gz")
    assert submission["files"] == len(SYNTHETIC_DELIVERY)
    assert submission["source_bytes"] == sum(len(content.encode()) for content in SYNTHETIC_DELIVERY.values())


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_real_docker_observation_and_export(tmp_path, monkeypatch, execution_mode):
    task_path = os.environ.get("STAGE1_PROGRAMBENCH_OBSERVER_TASK")
    if not task_path:
        pytest.skip("Set STAGE1_PROGRAMBENCH_OBSERVER_TASK to run the real Docker observer smoke")
    docker = os.environ.get("STAGE1_PROGRAMBENCH_OBSERVER_DOCKER", "docker")
    root = Path(os.environ.get("STAGE1_PROGRAMBENCH_OBSERVER_OUTPUT", str(tmp_path / "smoke")))
    output = root / execution_mode
    output.mkdir(parents=True, exist_ok=False)
    instruction = output / "synthetic_instruction.md"
    instruction.write_text("Infrastructure observation test only. Do not solve the benchmark task.\n")
    task = ProgramBenchTask.load(task_path, instruction_path=instruction,
                                agent_timeout_sec=120, cpus=1, memory_mb=1024)
    # Reuse the existing Docker-integration convention: do not implicitly pull a
    # missing benchmark image. This reads one scalar, never container Config/env.
    subprocess.run([docker, "image", "inspect", "--format", "{{.Id}}", task.image],
                   check=True, capture_output=True, text=True, timeout=30)
    name = "ac-pb-observer-smoke-" + uuid.uuid4().hex
    workspace = ProgramBenchWorkspace(task, docker=docker, platform="linux/amd64", container_name=name)
    runtime = output / "runtime"
    (runtime / "relay").mkdir(parents=True)
    registration = ({"event": "worker_registered", "name": "synthetic_registered"}
                    if execution_mode == "team" else None)
    (runtime / "relay/events.jsonl").write_text(json.dumps(registration) + "\n" if registration else "")
    observer = None
    facts = {"kind": "synthetic_observer_docker_smoke", "execution_mode": execution_mode,
             "task_id": task.task_id, "container_name": name, "external_model_calls": 0,
             "native_agent_run": False, "official_evaluation_run": False,
             "worker_registration_source": "synthetic_fixture",
             "diagnostic_resources": {"cpus": 1, "memory_mb": 1024},
             "passed": False, "cleanup_confirmed": False}
    started = time.monotonic()
    # Fail immediately if a future edit accidentally broadens this test.
    monkeypatch.setattr(workspace, "run_agent", lambda *a, **k: pytest.fail("No agent execution in observer smoke"))
    monkeypatch.setattr(workspace, "run_verifier", lambda *a, **k: pytest.fail("No evaluator in observer smoke"))
    try:
        workspace.start()
        facts["image_identity"] = workspace.image_identity
        assert workspace.network_policy_status["enforced"]
        assert workspace.network_policy_status["allowed_relay"] is None
        observer = ProgramBenchObserver(workspace, output / "observations.jsonl", runtime,
                                         interval_seconds=60, execution_mode=execution_mode)
        workspace.observation_before_quiesce = observer.request_stop
        observer.start()
        initial = _rows(observer.output)["initial"]
        assert isinstance(initial["file_count"], int) and initial["commits_since_start"] == 0
        assert initial["uncommitted_paths"] == 0

        for name, content in SYNTHETIC_DELIVERY.items():
            workspace.write_file(workspace.repo_path + "/" + name, content)
        # All four locations are outside export scope; none may inflate readings.
        workspace.write_file(workspace.repo_path + "/executable", "excluded synthetic binary\n")
        workspace.write_file(workspace.repo_path + "/reference/observer-hidden", "excluded\n")
        workspace.write_file(workspace.repo_path + "/observer-excluded/.git/note", "excluded\n")
        worker = workspace.ensure_worker("unregistered_worktree")
        workspace.write_file(worker + "/not-delivered.txt", "excluded worker file\n")
        observer.sample("uncommitted")
        dirty = _rows(observer.output)["uncommitted"]
        assert dirty["file_count"] == initial["file_count"] + len(SYNTHETIC_DELIVERY)
        assert dirty["source_bytes"] == initial["source_bytes"] + sum(
            len(content.encode()) for content in SYNTHETIC_DELIVERY.values())
        assert dirty["uncommitted_paths"] == len(SYNTHETIC_DELIVERY) and dirty["commits_since_start"] == 0
        assert dirty["staging_without_commits"] is True
        assert dirty["registered_worker_count"] == (1 if execution_mode == "team" else 0)

        workspace.run(["git", "add", "--", *SYNTHETIC_DELIVERY])
        workspace.run(["git", "commit", "-qm", "Synthetic observer delivery"])
        observer.sample("committed")
        committed = _rows(observer.output)["committed"]
        assert committed["uncommitted_paths"] == 0 and committed["commits_since_start"] == 1
        assert committed["staging_without_commits"] is False
        assert committed["latest_write_epoch_seconds"] is not None

        # Stop must not wait for observation. Archive the delivered bytes before
        # final sampling can invoke any agent-configured Git filters.
        workspace.finish_agent()
        exported = output / "source"
        workspace.export_source(exported)
        submission = create_submission(exported, output / "submission.tar.gz")
        facts["submission"] = submission
        observer.finish()
        rows = _rows(observer.output)
        before = observer.summary()["last_inference_memory_observation"]
        final = rows["final"]
        assert before["completed_before_shutdown_seconds"] >= 0
        assert "pre_quiesce" not in rows
        assert final["memory_scope"] == "post_quiesce_container"
        assert isinstance(before["memory_peak_bytes"], int) and before["memory_peak_bytes"] > 0
        assert isinstance(before["memory_events"], dict) and "oom" in before["memory_events"]
        assert final["file_count"] == committed["file_count"]
        assert final["source_bytes"] == committed["source_bytes"]
        files = [path for path in exported.rglob("*") if path.is_file()]
        assert len(files) == final["file_count"]
        assert sum(path.stat().st_size for path in files) == final["source_bytes"]
        for name, content in SYNTHETIC_DELIVERY.items():
            assert (exported / name).read_text() == content
        assert all(".git" not in path.relative_to(exported).parts for path in files)
        assert not (exported / "executable").exists() and not (exported / "reference").exists()
        assert not any(path.name == "not-delivered.txt" for path in files)
        # Observer writes stay on the host: the container's runtime tree is untouched.
        unchanged = workspace.run(["python3", "-I", "-c",
                                   "import pathlib,sys; print(len(list(pathlib.Path(sys.argv[1]).rglob('*'))))",
                                   workspace.runtime_root])
        assert int(unchanged.stdout) == 0
        assert observer.summary()["injected_into_agent"] is False
        durations = {phase: row["observation_wall_seconds"] for phase, row in rows.items()}
        facts.update(sample_wall_seconds=durations, memory_peak_bytes=before["memory_peak_bytes"])
        assert all(0 <= duration <= 10 for duration in durations.values())
        facts.update(passed=True, exported_files=len(files), observer=observer.summary())
    finally:
        if observer is not None:
            observer.request_stop()
        try:
            workspace.stop()
            facts["cleanup_confirmed"] = True
        finally:
            if observer is not None:
                observer.stop_periodic()
                facts["observer"] = observer.summary()
            facts.setdefault("image_identity", workspace.image_identity)
            facts["elapsed_seconds"] = time.monotonic() - started
            if observer is not None:
                facts["sample_wall_seconds"] = {
                    phase: row["observation_wall_seconds"] for phase, row in _rows(observer.output).items()
                } if observer.output.is_file() else {}
                memory = observer.summary().get("last_inference_memory_observation")
                facts["memory_peak_bytes"] = memory.get("memory_peak_bytes") if memory else None
            (output / "observer_smoke.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")
