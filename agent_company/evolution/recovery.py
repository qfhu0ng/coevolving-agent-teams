"""Task-boundary checkpoints; never pretend to restore an in-memory Qwen team."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from agent_company.runtime_io import write_json_atomic


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def experiment_session(output, settings, *, resume=False):
    """An OS lock prevents two controllers from spending against one checkpoint."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another controller is using this output directory") from exc
        try:
            path = root / "run_settings.json"
            normalized = json.loads(json.dumps(settings))
            if path.exists():
                if not resume:
                    raise FileExistsError("Output already initialized; use --resume")
                saved = read(path)
                if any(key in normalized and key not in saved for key in ("execution_mode", "baseline_mode")):
                    raise ValueError("Legacy checkpoint has no execution-mode setting; cannot safely resume "
                                     "it as a single-agent or team experiment")
                if saved != normalized:
                    raise ValueError("Resume settings differ from the saved experiment")
            elif resume:
                raise ValueError("No resumable run_settings.json; legacy outputs are not overwritten")
            else:
                if any(p.name != ".controller.lock" for p in root.iterdir()):
                    raise FileExistsError("Output must be empty for a new experiment")
                write_json_atomic(path, normalized)
            yield root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def interrupted_usage(directory):
    """Read settled events, not the possibly stale last usage summary."""
    directory = Path(directory)
    request_path = directory / "run_request.json"
    execution_mode = read(request_path).get("execution_mode", "team") if request_path.exists() else (
        "single" if (directory / "single").exists() and not (directory / "team").exists() else "team")
    if execution_mode not in ("single", "team"):
        raise ValueError("Interrupted task has an invalid execution_mode")
    runtime = directory / execution_mode
    events = runtime / "relay/events.jsonl"
    started = {int(p.name.split("_")[1]) for p in
               (runtime / "relay").glob("attempt_*_provider_request.json")}
    finished, usage, known = {}, {}, 0.0
    if events.exists():
        lines = events.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break  # A process can die partway through the final append.
                raise
            if row.get("event") == "provider_attempt_started":
                started.add(row["attempt"])
            if row.get("event") == "provider_attempt_finished":
                finished[row["attempt"]] = row
        for row in finished.values():
            for key, value in (row.get("usage") or {}).items():
                usage[key] = usage.get(key, 0) + value
            if row.get("estimated_cost") is not None:
                known += float(row["estimated_cost"])
    launch = runtime / "launch.json"
    rates = read(launch).get("rates") if launch.exists() else None
    usd = rates and rates.get("currency") == "USD"
    return {"output_dir": str(directory), "status": "interrupted", "score": None,
            "execution_mode": execution_mode, "policy_applied": execution_mode == "team",
            "provider_attempts": len(started | finished.keys()),
            "unsettled_attempts": len(started - finished.keys()), "usage": usage,
            "cost_usd": None, "known_cost_usd": known if usd else None,
            "note": "Task restarted from a fresh environment; old usage may be incomplete."}


def prepare_task(request, callback, *, resume, retry_interrupted):
    """Return a terminal checkpoint, or archive an explicitly retried attempt."""
    from dataclasses import asdict
    from agent_company.evolution.loop import TaskResult, _record_result, _validate_result

    directory = request.output_dir
    if not directory.exists():
        return None
    if not resume:
        raise FileExistsError(directory)
    saved = directory / "run_request.json"
    if not saved.exists() and not any(directory.iterdir()):
        directory.rmdir()  # Crash between mkdir and the first checkpoint; no callback ran.
        return None
    expected = asdict(request)
    expected["output_dir"] = str(directory)
    expected = json.loads(json.dumps(expected))
    recorded = read(saved) if saved.exists() else None
    # These opt-in fields did not exist in older checkpoints. An omitted
    # legacy field has the same meaning as the current None default, while
    # non-default package/mode differences must still reject the resume.
    if isinstance(recorded, dict):
        for key in ("role_policy_mode", "policy_package"):
            recorded.setdefault(key, None)
    if recorded != expected:
        raise ValueError(f"Task request differs or is incomplete: {directory}")
    result = directory / "task_result.json"
    if result.exists():
        record = read(result)
        for key, expected_value in (("task_id", request.task.task_id), ("repo_id", request.task.repo_id),
                                    ("split", request.split), ("policy_version", request.policy_version),
                                    ("repeat_index", request.repeat_index), ("pair_id", request.pair_id),
                                    ("execution_mode", request.execution_mode),
                                    ("policy_applied", request.execution_mode == "team")):
            if record.get(key) != expected_value:
                raise ValueError(f"Task result identity differs: {directory}")
        fields = TaskResult.__dataclass_fields__
        _validate_result(TaskResult(**{key: record[key] for key in fields if key in record}))
        return record
    runner_result = directory / "runner_result.json"
    if runner_result.exists():
        completed = TaskResult(**read(runner_result))
        _validate_result(completed)
        return _record_result(request, completed, None)
    if not retry_interrupted:
        raise RuntimeError(f"Interrupted task at {directory}; stop old agents, then use "
                           "--resume --retry-interrupted to restart only this unfinished task")
    check = getattr(callback, "assert_can_restart", None)
    if check:
        check(directory)
    archive = directory.with_name(directory.name + ".interrupted")
    archive.mkdir(exist_ok=True)
    destination = archive / f"attempt_{len(list(archive.iterdir())) + 1:04d}"
    record = interrupted_usage(directory)
    record["output_dir"] = str(destination)
    write_json_atomic(directory / "interruption.json", record)
    directory.rename(destination)
    return None


def previous_attempts(directory):
    archive = directory.with_name(directory.name + ".interrupted")
    return [read(path) for path in sorted(archive.glob("*/interruption.json"))]
