"""RoadmapBench's Harbor task format and final verifier results.

Task code and public instructions are exposed during execution. The ``tests`` and
``solution`` directories remain on the host until the final verification phase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any


@dataclass(frozen=True)
class Task:
    path: Path
    task_id: str
    instruction: str
    image: str | None
    agent_timeout_sec: float
    verifier_timeout_sec: float
    build_timeout_sec: float
    cpus: float
    memory_mb: int
    storage_mb: int
    repository_id: str
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path, *, repository_id: str | None = None) -> "Task":
        path = Path(path).resolve()
        raw = tomllib.loads((path / "task.toml").read_text())
        metadata = raw.get("metadata", {})
        environment = raw.get("environment", {})
        task_id = metadata.get("name", path.name)
        repository = repository_id or metadata.get("repository", metadata.get("repo"))
        if not repository:
            repository = re.split(r"-\d", task_id, maxsplit=1)[0]
        task = cls(
            path=path, task_id=task_id,
            instruction=(path / "instruction.md").read_text(),
            image=environment.get("docker_image"),
            agent_timeout_sec=float(raw.get("agent", {}).get("timeout_sec", 600)),
            verifier_timeout_sec=float(raw.get("verifier", {}).get("timeout_sec", 600)),
            build_timeout_sec=float(environment.get("build_timeout_sec", 600)),
            cpus=float(environment.get("cpus", 1)),
            memory_mb=int(environment.get("memory_mb", 2048)),
            storage_mb=int(environment.get("storage_mb", 10240)),
            repository_id=str(repository), metadata=metadata,
        )
        for key in ("agent_timeout_sec", "verifier_timeout_sec", "build_timeout_sec", "cpus",
                    "memory_mb", "storage_mb"):
            value = getattr(task, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}: {value}")
        if not (path / "tests" / "test.sh").is_file():
            raise FileNotFoundError(path / "tests" / "test.sh")
        if not task.image and not (path / "environment" / "Dockerfile").is_file():
            raise ValueError("Task needs environment.docker_image or environment/Dockerfile")
        return task

    def describe(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        value.pop("instruction")
        return value


@dataclass(frozen=True)
class Reward:
    """Keep official detail shapes; do not reconstruct weighted scores from counts."""
    reward: float
    details: Any
    source: str


def parse_reward(directory: str | Path) -> Reward:
    directory = Path(directory)
    text_file, json_file = directory / "reward.txt", directory / "reward.json"
    details = json.loads(json_file.read_text()) if json_file.is_file() else None
    if text_file.is_file():
        value, source = float(text_file.read_text().strip()), "reward.txt"
    elif isinstance(details, dict) and "reward" in details:
        value, source = float(details["reward"]), "reward.json"
    elif isinstance(details, (float, int)) and not isinstance(details, bool):
        value, source = float(details), "reward.json"
    else:
        raise ValueError("No official scalar reward found; target counts are not a weighted score")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Official reward is outside [0, 1]: {value}")
    return Reward(value, details, source)


def score(task: Task, workspace: Any, output_dir: str | Path) -> dict[str, Any]:
    """Run the hidden verifier once, after all agent processes have been terminated.

    A container stop/restart clears every background agent process while preserving
    the complete repository and installed dependencies. The verifier is uploaded
    only after that transition. Results never go back into a live worker session.
    """
    import time
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.finish_agent()
    workspace.run(["mkdir", "-p", "/tests", "/logs/verifier"])
    workspace.run(["sh", "-c", "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json"])
    workspace.upload(task.path / "tests", "/tests", contents=True)
    started = time.monotonic()
    execution = workspace.run_verifier(
        ["bash", "/tests/test.sh"], timeout_seconds=task.verifier_timeout_sec,
        stdout_path=output_dir / "stdout.log", stderr_path=output_dir / "stderr.log",
    )
    # A timed-out exec may have stopped the container; docker cp works while stopped.
    workspace.download("/logs/verifier/.", output_dir)
    result: dict[str, Any] = {
        "task_id": task.task_id, "repository_id": task.repository_id,
        "reward": None, "status": "verifier_timeout" if execution["timed_out"] else "completed",
        "verifier": execution, "verifier_elapsed_seconds": time.monotonic() - started,
    }
    try:
        reward = parse_reward(output_dir)
        result.update(reward=reward.reward, reward_details=reward.details, reward_source=reward.source)
    except (ValueError, OSError, TypeError) as error:
        result["reward_error"] = str(error)
        if result["status"] == "completed":
            result["status"] = "invalid_reward"
    if execution["returncode"] and result["status"] == "completed":
        result["status"] = "verifier_error"
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
