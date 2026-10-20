"""Local metadata/Git and mocked runner coverage; no Docker/provider/evaluator API."""
from dataclasses import replace
import json
import subprocess
import sys
import tarfile
import threading
from types import SimpleNamespace

import pytest

from agent_company.programbench.observation import ProgramBenchObserver, SAMPLE_SCRIPT
from agent_company.programbench.workspace import ProgramBenchWorkspace
from agent_company.programbench import runner as runner_module
from test_programbench_runner import rig as rig


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "solution"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "README").write_text("initial documentation")
    (root / ".gitignore").write_text("/executable\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "initial")
    return root


def script_read(root, tmp_path, initial="", *, membership="0::/\n", missing_memory=False):
    cg = tmp_path / "cgroup"
    cg.mkdir(exist_ok=True)
    (cg / "memory.peak").write_text("12345\n")
    (cg / "memory.events").write_text("low 0\nhigh 2\noom 1\noom_kill 0\n")
    membership_path = tmp_path / "membership"
    membership_path.write_text(membership)
    code = SAMPLE_SCRIPT.replace("'/proc/self/cgroup'", repr(str(membership_path)))
    code = code.replace("'/sys/fs/cgroup'", repr(str(cg)))
    if missing_memory:
        (cg / "memory.peak").unlink()
        (cg / "memory.events").unlink()
    proc = subprocess.run([sys.executable, "-I", "-c", code, str(root), initial, "all"],
                          capture_output=True, text=True, check=True, timeout=4)
    return json.loads(proc.stdout)


def test_real_metadata_scope_matches_export_and_uncommitted_files_survive_zero_commits(source, tmp_path):
    initial = git(source, "rev-parse", "HEAD")
    (source / "main.py").write_text("PRIVATE_BODY")
    (source / "executable").write_bytes(b"x" * 1000)
    (source / "reference").mkdir()
    (source / "reference/hidden").write_text("excluded")
    (source / "nested/.git").mkdir(parents=True)
    (source / "nested/.git/hidden").write_text("excluded")
    (tmp_path / "workers").mkdir()
    (tmp_path / "workers/not-delivered").write_text("excluded")
    out = script_read(source, tmp_path, initial)
    assert out["file_count"] == 3
    assert out["source_bytes"] == sum((source / name).stat().st_size for name in ("README", ".gitignore", "main.py"))
    assert out["uncommitted_paths"] == 1 and out["commits_since_start"] == 0
    assert out["latest_write_epoch_seconds"] > 0
    assert out["memory_peak_bytes"] == 12345 and out["memory_events"]["oom"] == 1
    assert "PRIVATE_BODY" not in json.dumps(out)
    git(source, "add", "main.py")
    git(source, "commit", "-qm", "delivery")
    out = script_read(source, tmp_path, initial)
    assert out["uncommitted_paths"] == 0 and out["commits_since_start"] == 1


def test_git_rename_and_newline_filename_count_once(source, tmp_path):
    initial = git(source, "rev-parse", "HEAD")
    git(source, "mv", "README", "new\nname")
    out = script_read(source, tmp_path, initial)
    assert out["uncommitted_paths"] == 1


@pytest.mark.parametrize("membership,missing", [("0::/\n", True), ("1:memory:/\n", False), ("0::/not-mounted\n", False)])
def test_missing_cgroup_v2_is_unknown_never_host_root_or_zero(source, tmp_path, membership, missing):
    out = script_read(source, tmp_path, membership=membership, missing_memory=missing)
    assert out["memory_peak_bytes"] is out["memory_events"] is None
    assert "memory_peak_unavailable" in out["issues"]
    assert out["file_count"] == 2


def values(**overrides):
    return {"file_count": 2, "source_bytes": 12, "latest_write_epoch_seconds": 100,
            "uncommitted_paths": 1, "commits_since_start": 0,
            "memory_peak_bytes": 456, "memory_events": {"oom": 0}, "issues": [],
            "_initial_head": "a" * 40, **overrides}


def observer(tmp_path, run, mode="team", interval=60, relay=True):
    workspace = SimpleNamespace(repo_path="/workspace/solution", run=run)
    runtime = tmp_path / "team"
    runtime.mkdir(exist_ok=True)
    if relay:
        (runtime / "relay").mkdir()
    return ProgramBenchObserver(workspace, tmp_path / "observations.jsonl", runtime,
                                interval_seconds=interval, execution_mode=mode)


@pytest.mark.parametrize("relay", [False, True])
def test_team_sample_before_first_relay_event_is_zero_registered_not_a_failure(tmp_path, relay):
    obs = observer(tmp_path, lambda *a, **k: SimpleNamespace(stdout=json.dumps(values())), relay=relay)
    obs.sample("initial")
    (obs.runtime / "relay").mkdir(exist_ok=True)
    (obs.runtime / "relay/events.jsonl").write_text(json.dumps({"event": "worker_registered", "name": "one"}) + "\n")
    obs.sample("periodic")
    rows = [json.loads(line) for line in obs.output.read_text().splitlines()]
    assert rows[0]["registered_worker_count"] == 0 and rows[0]["issues"] == []
    assert rows[1]["registered_worker_count"] == 1 and rows[1]["issues"] == []
    assert obs.summary()["sample_failures"] == 0


@pytest.mark.parametrize("content", ["", json.dumps({"event": "worker_registered", "name": "one"}) + "\n"])
def test_team_events_file_vanishing_after_being_seen_stays_unknown(tmp_path, content):
    obs = observer(tmp_path, lambda *a, **k: SimpleNamespace(stdout=json.dumps(values())))
    path = obs.runtime / "relay/events.jsonl"
    path.write_text(content)
    obs.sample("initial")
    path.unlink()
    obs.sample("periodic")
    rows = [json.loads(line) for line in obs.output.read_text().splitlines()]
    assert rows[0]["registered_worker_count"] == (1 if content else 0) and rows[0]["issues"] == []
    assert rows[1]["registered_worker_count"] is None
    assert rows[1]["issues"] == ["registered_worker_count_unavailable"]
    assert obs.summary()["sample_failures"] == 1


def test_registered_count_uses_events_not_worker_dirs_or_liveness(tmp_path):
    obs = observer(tmp_path, lambda *a, **k: SimpleNamespace(stdout=json.dumps(values())))
    path = obs.runtime / "relay/events.jsonl"
    path.write_text('\n'.join(json.dumps(e) for e in [
        {"event": "worker_registered", "name": "one"}, {"event": "worker_registered", "name": "one"},
        {"event": "worker_registered", "name": "two"}, {"event": "worker_ended", "name": "one"}]) + '\n')
    obs.sample("initial")
    with path.open("a") as stream:
        stream.write('{"event":"worker_reg')
    obs.sample("periodic")
    with path.open("a") as stream:
        stream.write('istered","name":"three"}\n')
    obs.finish()
    rows = [json.loads(line) for line in obs.output.read_text().splitlines()]
    assert [row["registered_worker_count"] for row in rows] == [2, None, 3]
    assert rows[-1]["staging_without_commits"] is True
    assert "_initial_head" not in json.dumps(rows)


def test_periodic_sampling_and_final_snapshot_are_bounded_and_not_injected(tmp_path):
    periodic = threading.Event()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {"cwd": "/", "timeout": 3}
        assert command[:3] == ["timeout", "--signal=KILL", "2s"]
        if len(calls) >= 2:
            periodic.set()
        return SimpleNamespace(stdout=json.dumps(values()))

    obs = observer(tmp_path, run, mode="single", interval=0.01)
    obs.start()
    assert periodic.wait(1)
    obs.request_stop()
    obs.finish()
    rows = [json.loads(line) for line in obs.output.read_text().splitlines()]
    assert rows[0]["phase"] == "initial" and any(row["phase"] == "periodic" for row in rows)
    assert rows[-1]["phase"] == "final"
    assert not any(row["phase"] == "pre_quiesce" for row in rows)
    assert obs.summary()["last_inference_memory_observation"]["completed_before_shutdown_seconds"] >= 0
    assert rows[-1]["memory_scope"] == "post_quiesce_container"
    assert obs.summary()["observed_wall_seconds"] >= 0
    assert obs.summary()["injected_into_agent"] is False
    assert all(row["registered_worker_count"] == 0 for row in rows)


def test_failed_reads_and_writes_keep_unknown_without_leaking_errors(tmp_path):
    def run(*args, **kwargs):
        raise RuntimeError("PRIVATE_PROVIDER_OR_SOURCE_TEXT")

    obs = observer(tmp_path, run)
    obs.output.mkdir()
    obs.sample("initial")
    obs.finish()
    summary = obs.summary()
    assert summary["write_failures"] == 2 and summary["sample_failures"] == 2
    assert summary["final_snapshot"]["file_count"] is None
    assert summary["final_snapshot"]["staging_without_commits"] is None
    assert "PRIVATE" not in json.dumps(summary)


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("fail", [False, True])
def test_runner_final_sampling_follows_archive_and_cannot_change_submission(rig, monkeypatch, mode, fail):
    rig.runner.observation_interval_seconds = 60
    cls = runner_module.ProgramBenchWorkspace
    cls.repo_path = "/workspace/solution"
    original_finish = cls.finish_agent
    original_export = cls.export_source
    original_archive = runner_module.create_submission

    def export(self, destination):
        original_export(self, destination)
        (destination / "main.sh").write_text(getattr(self, "delivery", "original agent delivery"))

    def archive(*args, **kwargs):
        result = original_archive(*args, **kwargs)
        rig.events.append("create_submission")
        return result

    def finish(self):
        callback = getattr(self, "observation_before_quiesce", None)
        if callback is not None:
            self.observation_before_quiesce = None
            callback()
        original_finish(self)

    def run(self, command, **kwargs):
        rig.events.append("sample_final" if self.agent_finished else "sample_before")
        if fail:
            raise RuntimeError("PRIVATE_OBSERVER_FAILURE")
        if self.agent_finished:
            # Simulate a user Git filter rewriting the container's delivered code.
            self.delivery = "changed by final observer Git filter"
        return SimpleNamespace(stdout=json.dumps(values()))

    monkeypatch.setattr(cls, "finish_agent", finish)
    monkeypatch.setattr(cls, "run", run, raising=False)
    monkeypatch.setattr(cls, "export_source", export)
    monkeypatch.setattr(runner_module, "create_submission", archive)
    result = rig.runner(replace(rig.request, execution_mode=mode))
    assert result.status == "completed" and result.score == 0.75
    assert result.metrics[mode]["stop_reason"] == "native_cli_completed"
    assert (rig.events.index("finish_agent") < rig.events.index("export_source")
            < rig.events.index("create_submission") < rig.events.index("sample_final") < rig.events.index("stop"))
    with tarfile.open(rig.request.output_dir / "submission.tar.gz") as submitted:
        assert submitted.extractfile("main.sh").read() == b"original agent delivery"
    assert (rig.request.output_dir / "source/main.sh").read_text() == "original agent delivery"
    if not fail:
        assert rig.workspaces[-1].delivery == "changed by final observer Git filter"
    assert rig.calls[0]["options"]["max_model_calls"] == 1000
    assert "observation_interval_seconds" not in rig.qwen_options
    summary = result.metrics["observation"]
    assert summary["interval_seconds"] == 60 and summary["final_snapshot"]["file_count"] == (None if fail else 2)
    assert summary["final_snapshot"]["staging_without_commits"] is (None if fail else True)
    assert "PRIVATE_OBSERVER" not in json.dumps(result.metrics)


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("failure_phase", ["source_export", "submission_archive"])
def test_failed_delivery_still_samples_final_before_cleanup(rig, monkeypatch, mode, failure_phase):
    rig.runner.observation_interval_seconds = 60
    cls = runner_module.ProgramBenchWorkspace
    monkeypatch.setattr(cls, "repo_path", "/workspace/solution", raising=False)
    original_finish = cls.finish_agent

    def finish(self):
        self.observation_before_quiesce()
        original_finish(self)

    def sample(self, *args, **kwargs):
        assert not self.stopped
        rig.events.append("sample_final" if self.agent_finished else "sample_before")
        return SimpleNamespace(stdout=json.dumps(values()))

    def archive(*args, **kwargs):
        rig.events.append("create_submission")
        raise ValueError("PRIVATE_ARCHIVE_FAILURE")

    monkeypatch.setattr(cls, "finish_agent", finish)
    monkeypatch.setattr(cls, "run", sample, raising=False)
    monkeypatch.setattr(runner_module, "create_submission", archive)
    if failure_phase == "source_export":
        rig.failures["export_source"] = RuntimeError("PRIVATE_EXPORT_FAILURE")
    result = rig.runner(replace(rig.request, execution_mode=mode))
    assert result.status == "failed" and result.score is None
    assert result.metrics["failures"] == [{"phase": failure_phase,
        "error_type": "RuntimeError" if failure_phase == "source_export" else "ValueError"}]
    assert rig.eval_calls == [] and result.metrics["inference_cleanup_confirmed"] is True
    assert (rig.events.index("finish_agent") < rig.events.index("export_source")
            < rig.events.index("sample_final") < rig.events.index("stop"))
    if failure_phase == "submission_archive":
        assert rig.events.index("create_submission") < rig.events.index("sample_final")
    else:
        assert "create_submission" not in rig.events
    rows = [json.loads(line) for line in (rig.request.output_dir / "observations.jsonl").read_text().splitlines()]
    assert rows[-1]["phase"] == "final"
    assert rows[-1]["file_count"] == 2
    assert result.metrics["observation"]["final_snapshot"] == rows[-1]
    assert "PRIVATE_" not in json.dumps(result.metrics)


def test_workspace_observer_runs_before_stop_and_cannot_prevent_quiesce(monkeypatch):
    from agent_company.runtime.workspace import DockerWorkspace
    calls = []
    ws = object.__new__(ProgramBenchWorkspace)
    ws.agent_finished = False

    def fail():
        calls.append("observe")
        raise RuntimeError("ignore observation failure")

    ws.observation_before_quiesce = fail
    monkeypatch.setattr(DockerWorkspace, "finish_agent", lambda self: calls.append("quiesce"))
    ws.finish_agent()
    assert calls == ["observe", "quiesce"] and ws.observation_before_quiesce is None


def test_thread_start_failure_cannot_escape_finish(tmp_path, monkeypatch):
    obs = observer(tmp_path, lambda *a, **k: SimpleNamespace(stdout=json.dumps(values())), mode="single")
    monkeypatch.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("no thread")))
    obs.start()
    obs.finish()
    assert obs.summary()["sample_failures"] >= 1
    assert obs.summary()["final_snapshot"]["file_count"] == 2


def test_source_symlink_is_unknown_rather_than_counting_outside_delivery(source, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("not delivered")
    (source / "link").symlink_to(outside)
    result = script_read(source, tmp_path)
    assert result["file_count"] is result["source_bytes"] is None
    assert "delivery_metadata_unavailable" in result["issues"]


def test_failed_agent_quiesce_never_claims_a_final_delivery_snapshot(rig, monkeypatch):
    rig.runner.observation_interval_seconds = 60
    cls = runner_module.ProgramBenchWorkspace
    monkeypatch.setattr(cls, "repo_path", "/workspace/solution", raising=False)
    monkeypatch.setattr(cls, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(values())), raising=False)
    rig.failures["finish_agent"] = RuntimeError("quiesce failed")
    result = rig.runner(rig.request)
    assert result.score is None and result.status == "failed"
    assert result.metrics["observation"]["final_snapshot"] is None
    rows = [json.loads(line) for line in (rig.request.output_dir / "observations.jsonl").read_text().splitlines()]
    assert all(row["phase"] not in {"cleanup_attempt", "final"} for row in rows)


def test_shutdown_signal_does_not_sample_join_or_wait_for_observer_lock(tmp_path, monkeypatch):
    from agent_company.runtime.workspace import DockerWorkspace
    obs = observer(tmp_path, lambda *a, **k: pytest.fail("No pre-stop sampling"), mode="single")
    obs._thread = SimpleNamespace(join=lambda **k: pytest.fail("No pre-stop join"))
    ws = object.__new__(ProgramBenchWorkspace)
    ws.agent_finished = False
    ws.observation_before_quiesce = obs.request_stop
    stopped = []
    monkeypatch.setattr(DockerWorkspace, "finish_agent", lambda self: stopped.append(obs._stop.is_set()))
    obs._lock.acquire()
    try:
        ws.finish_agent()
    finally:
        obs._lock.release()
    assert stopped == [True]
    assert obs.summary()["last_inference_memory_observation"] is None


def test_sample_overlapping_shutdown_is_not_an_inference_peak(tmp_path):
    holder = {}
    def run(*a, **k):
        if holder.get("overlap"):
            holder["obs"].request_stop()
        return SimpleNamespace(stdout=json.dumps(values(memory_peak_bytes=999 if holder.get("overlap") else 100)))
    obs = holder["obs"] = observer(tmp_path, run, mode="single")
    obs.sample("initial")
    holder["overlap"] = True
    obs.sample("periodic")
    rows = [json.loads(line) for line in obs.output.read_text().splitlines()]
    assert rows[-1]["memory_scope"] == "unknown_shutdown_overlap"
    assert obs.summary()["last_inference_memory_observation"]["memory_peak_bytes"] == 100
    assert obs.summary()["last_inference_memory_observation"]["completed_before_shutdown_seconds"] >= 0


def test_abnormal_cleanup_stops_workspace_before_joining_observer(rig, monkeypatch):
    rig.runner.observation_interval_seconds = 60
    cls = runner_module.ProgramBenchWorkspace
    monkeypatch.setattr(cls, "repo_path", "/workspace/solution", raising=False)
    monkeypatch.setattr(cls, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(values())), raising=False)
    rig.failures["finish_agent"] = RuntimeError("quiesce failed")
    original = ProgramBenchObserver.stop_periodic
    def join(self):
        rig.events.append("observer_join")
        original(self)
    monkeypatch.setattr(ProgramBenchObserver, "stop_periodic", join)
    result = rig.runner(rig.request)
    assert result.status == "failed"
    assert rig.events.index("stop") < rig.events.index("observer_join")


@pytest.mark.parametrize("cpu,expected", [
    ({"usage_usec": 5_000_000, "nr_throttled": 3, "throttled_usec": 70}, {"usage_usec": 5_000_000, "nr_throttled": 3, "throttled_usec": 70}),
    ({"usage_usec": -1}, None), ({"unknown_key": 1}, None), ("12", None)])
def test_cgroup_cpu_counters_are_kept_only_when_valid(tmp_path, cpu, expected):
    rows = tmp_path / "observations.jsonl"
    payload = json.dumps(values(cpu_stat=cpu))
    sampled = observer(tmp_path, lambda *a, **k: SimpleNamespace(stdout=payload))
    sampled.sample("periodic")
    row = json.loads(rows.read_text().splitlines()[-1])
    assert row["cpu_stat"] == expected
