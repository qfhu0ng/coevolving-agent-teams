"""ProgramBench public task inputs, source submissions and official evaluation.

The agent workspace must be stopped before calling ``evaluate_submission``.
Only that host-side function loads evaluation metadata; ``ProgramBenchTask.load``
reads public task.yaml and an explicitly supplied official instruction file.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any

from agent_company.runtime_io import write_json_atomic

from agent_company.roadmap.task import Task


CLEANROOM_TAG = "task_cleanroom_v6"
SMOKE_TASK_ID = "tomnomnom__gron.88a6234"
# Public mini-swe-agent ProgramBench baseline inference configuration. These
# limits apply to the whole task, not separately to each team worker.
OFFICIAL_INFERENCE_TIMEOUT_SEC = 21600
OFFICIAL_INFERENCE_CPUS = 20
OFFICIAL_INFERENCE_MEMORY_MB = 61440  # Docker --memory 60g (60 GiB).
# ProgramBench 1.2.4 evaluator resources are separate from inference resources.
OFFICIAL_EVALUATOR_CPUS = 10
OFFICIAL_EVALUATOR_COMPILE_TIMEOUT_SEC = 900
OFFICIAL_EVALUATOR_BRANCH_TIMEOUT_SEC = 3600
_IMAGE_ID = re.compile(r"sha256:[0-9a-fA-F]{64}")


def is_image_id(value: object) -> bool:
    return isinstance(value, str) and _IMAGE_ID.fullmatch(value) is not None


def _docker_image_id(docker: str, reference: str) -> dict[str, str | None]:
    """Best effort: the image ID the local daemon holds for ``reference`` right now.

    Never raises and never keeps Docker output text; unknown stays None.
    """
    try:
        value = subprocess.run([docker, "inspect", "--type", "image", "--format", "{{.Id}}", reference],
                               text=True, capture_output=True, check=True, timeout=30).stdout.strip()
    except Exception as exc:
        return {"image_id": None, "error": type(exc).__name__}
    if not is_image_id(value):
        return {"image_id": None, "error": "unexpected_format"}
    return {"image_id": value, "error": None}

# Select a custom Docker executable before the official module's import-time
# constants load, without mutating the parent process environment or its cache.
_OFFICIAL_EVAL_SUBPROCESS = """
import sys
from pathlib import Path
from programbench.eval.eval_batch import run_eval_batch
run_eval_batch(sources=[Path(sys.argv[1])], output=Path(sys.argv[2]),
               workers=1, branch_workers=1, docker_cpus=int(sys.argv[3]),
               image_tag=sys.argv[4], filter_spec=sys.argv[5], force=False)
"""


@dataclass(frozen=True)
class ProgramBenchTask(Task):
    instruction_path: Path

    @classmethod
    def load(cls, path: str | Path, *, instruction_path: str | Path,
             agent_timeout_sec: float = OFFICIAL_INFERENCE_TIMEOUT_SEC,
             verifier_timeout_sec: float = OFFICIAL_EVALUATOR_BRANCH_TIMEOUT_SEC,
             cpus: float = OFFICIAL_INFERENCE_CPUS,
             memory_mb: int = OFFICIAL_INFERENCE_MEMORY_MB) -> "ProgramBenchTask":
        import yaml

        task_path = Path(path).resolve()
        instructions = Path(instruction_path).resolve()
        local_parts = instructions.relative_to(task_path).parts if instructions.is_relative_to(task_path) else ()
        if instructions.name in {"tests.json", "task.yaml"} or any(
                name in {"tests", "solution", "solutions"} for name in local_parts):
            raise ValueError("Supply an official public agent instruction file")
        # No tests.json, test blob or solution is consulted by this loader.
        raw = yaml.safe_load((task_path / "task.yaml").read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("task.yaml must contain public task metadata")
        repository, commit = raw.get("repository"), raw.get("commit")
        if not isinstance(repository, str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid public repository identifier")
        if not isinstance(commit, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
            raise ValueError("Invalid public commit identifier")
        # Official task directories and Docker image names are lowercase even when
        # task.yaml keeps the repository's own casing (e.g. FFmpeg/FFmpeg).
        task_id = (repository.replace("/", "__") + "." + commit[:7]).lower()
        if task_path.name != task_id:
            raise ValueError("Task directory disagrees with public repository/commit")
        for name, value in (("agent_timeout_sec", agent_timeout_sec),
                            ("verifier_timeout_sec", verifier_timeout_sec), ("cpus", cpus),
                            ("memory_mb", memory_mb)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        if not isinstance(memory_mb, int):
            raise ValueError("memory_mb must be an integer")
        instruction = instructions.read_text(encoding="utf-8")
        if not instruction.strip():
            raise ValueError("Official agent instructions must not be empty")
        metadata = {key: raw[key] for key in ("repository", "commit", "language", "difficulty")
                    if key in raw}
        reference_limits = {"agent_timeout_sec": OFFICIAL_INFERENCE_TIMEOUT_SEC,
                            "cpus": OFFICIAL_INFERENCE_CPUS,
                            "memory_mb": OFFICIAL_INFERENCE_MEMORY_MB}
        requested_limits = {"agent_timeout_sec": agent_timeout_sec, "cpus": cpus,
                            "memory_mb": memory_mb}
        overrides = {key: value for key, value in requested_limits.items()
                     if value != reference_limits[key]}
        metadata.update(benchmark="ProgramBench", image_tag=CLEANROOM_TAG,
                        resource_limits_source="custom" if overrides else "official_baseline",
                        resource_limits_reference="mini-swe-agent ProgramBench baseline config",
                        official_inference_resource_limits=reference_limits,
                        inference_resource_overrides=overrides,
                        resource_limits_scope="task-shared",
                        official_verifier_timeouts=True,
                        verifier_timeout_scope="official_per_step",
                        verifier_aggregate_timeout_enforced=False)
        return cls(path=task_path, task_id=task_id, instruction=instruction,
                   image=f"programbench/{task_id.replace('__', '_1776_')}:{CLEANROOM_TAG}",
                   agent_timeout_sec=float(agent_timeout_sec),
                   verifier_timeout_sec=float(verifier_timeout_sec), build_timeout_sec=600,
                   cpus=float(cpus), memory_mb=memory_mb, storage_mb=10240,
                   repository_id=repository, metadata=metadata, instruction_path=instructions)

    def describe(self) -> dict[str, Any]:
        value = super().describe()
        value["instruction_path"] = str(self.instruction_path)
        return value


def _member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError("Submission member must remain inside the source directory")
    return path


def validate_submission(archive_path: str | Path) -> dict[str, int]:
    """Inspect member headers only; never extract an untrusted archive on the host."""
    files = 0
    total_bytes = 0
    compile_found = False
    seen: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            path = _member_path(member.name)
            if str(path) in seen:
                raise ValueError("Duplicate submission member")
            seen.add(str(path))
            if not member.isfile() and not member.isdir():
                raise ValueError("Submission links and special files are not supported")
            if ".git" in path.parts or path.parts[:1] in {("reference",), ("executable",)}:
                raise ValueError("Submission contains protected or prebuilt delivery material")
            if member.isfile():
                files += 1
                total_bytes += member.size
                compile_found |= path == PurePosixPath("compile.sh")
    if not compile_found:
        raise ValueError("Submission needs a regular compile.sh at its root")
    return {"files": files, "source_bytes": total_bytes}


def create_submission(source_dir: str | Path, archive_path: str | Path, *,
                      excluded_paths: tuple[str | Path, ...] = ()) -> dict[str, Any]:
    """Archive the entire delivered source tree, including uncommitted/new files.

    Preserve executable script modes. Exclude Git internals, the protected
    reference directory, and ./executable (the official build recreates it).
    Other exact protected paths may be supplied relative to the source root.
    Symlinks/special files fail explicitly; no link is followed into reference
    material or arbitrary host files. Official evaluation also checks reference
    hashes; this helper does not replace those benchmark checks.
    """
    source_input = Path(source_dir)
    if source_input.is_symlink():
        raise ValueError("Source directory must not be a symlink")
    source = source_input.resolve(strict=True)
    target = Path(archive_path).absolute()
    if not source.is_dir() or target.is_relative_to(source):
        raise ValueError("Archive must be outside the source directory")
    exclusions = {PurePosixPath("reference"), PurePosixPath("executable")}
    for item in excluded_paths:
        path = Path(item)
        if path.is_absolute():
            if not path.is_relative_to(source):
                continue  # Already outside the submitted tree, e.g. /workspace/reference.
            path = path.relative_to(source)
        exclusion = _member_path(path.as_posix())
        if exclusion == PurePosixPath("."):
            raise ValueError("Cannot exclude the complete source directory")
        exclusions.add(exclusion)
    entries = []
    omitted = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            relative = PurePosixPath(path.relative_to(source).as_posix())
            if path.name == ".git" or any(relative == x or x in relative.parents for x in exclusions):
                omitted.append(str(relative))
                continue
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
                raise ValueError(f"Submission links and special files are not supported: {relative}")
            entries.append((path, relative))
            if stat.S_ISDIR(mode):
                visit(path)

    visit(source)
    if not any(relative == PurePosixPath("compile.sh") and path.is_file()
               for path, relative in entries):
        raise ValueError("Submission needs compile.sh at its root")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a prior submission. The stopped workspace cannot mutate
    # concurrently; no hashes or additional freeze machinery are needed.
    with target.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", dereference=True) as archive:
        for path, relative in entries:
            archive.add(path, arcname=str(relative), recursive=False)
    details = validate_submission(target)
    return {"path": str(target), **details, "excluded_paths": omitted}


def _official_api():
    """Import optional ProgramBench 1.2.4 only on the host evaluator path."""
    from programbench.eval.eval import EvaluationResult
    from programbench.eval.eval_batch import InstanceEvalSummary, run_eval_batch
    from programbench.submission import NEAR_RESOLVED_THRESHOLD, RESOLVED_THRESHOLD
    from programbench.utils.load_data import get_active_branches, get_ignored_tests, load_all_instances

    return (EvaluationResult, InstanceEvalSummary, run_eval_batch, get_active_branches,
            get_ignored_tests, load_all_instances, RESOLVED_THRESHOLD, NEAR_RESOLVED_THRESHOLD)


def summarize_official_result(task_id: str, eval_path: str | Path, *,
                              filtered_results_path: str | Path | None = None) -> dict[str, Any]:
    """Use official ``info`` filtering and summary, without exposing test bodies.

    The fraction is retained diagnostically even if evaluation is incomplete.
    Missing or infrastructure-failed scoring is never converted into reward 0.
    """
    path = Path(eval_path)
    base: dict[str, Any] = {"task_id": task_id, "reward": None, "resolved": None,
                            "almost": None, "reward_source": "ProgramBench official info filtering",
                            "eval_json": str(path)}
    if not path.is_file():
        return {**base, "status": "missing_evaluation", "error_code": "missing_eval_json"}
    (EvaluationResult, InstanceEvalSummary, _, active_branches, ignored_tests,
     load_instances, resolved_threshold, almost_threshold) = _official_api()
    # Same active-branch and ignored-test behavior as programbench info. This
    # metadata remains exclusively in the evaluator, never in the agent prompt.
    instance = next((item for item in load_instances(include_tests=True)
                     if item["instance_id"] == task_id), None)
    if instance is None:
        return {**base, "status": "invalid_evaluation", "error_code": "unknown_official_instance"}
    result = EvaluationResult.model_validate_json(path.read_text(encoding="utf-8"))
    active = active_branches(instance)
    ignored = set(result.test_branches) - set(active)
    result = result.for_branches(active).without_ignored(ignored_tests(instance))
    if ignored:
        result.warnings = [w for w in result.warnings if not any(f"branch {b}" in w for b in ignored)]
    summary = InstanceEvalSummary.from_eval_result(task_id, result)
    counts = dict(Counter(test.status for test in result.test_results))
    fraction = summary.score if summary.n_tests else None
    diagnostic = {"official_test_fraction": fraction, "passed_tests": summary.n_resolved,
                  "valid_tests": summary.n_tests, "test_status_counts": counts,
                  "error_code": summary.error_code, "test_branch_errors": summary.test_branch_errors,
                  "system_errors": summary.n_system_errors, "warnings_count": summary.n_warnings}
    base.update(diagnostic)
    base["reward_details"] = dict(diagnostic)
    submission_error = summary.error_code in {"compile_failed", "copy_executable_failed"}
    if not summary.n_tests:
        status = "empty_evaluation"
    elif (summary.error_code and not submission_error) or summary.test_branch_errors or summary.n_system_errors:
        status = "verifier_error"
    elif counts.get("not_run", 0) and not submission_error:
        status = "incomplete_evaluation"
    elif summary.n_warnings:
        status = "evaluation_warning"
    else:
        status = "completed"
        base.update(reward=fraction, resolved=fraction >= resolved_threshold,
                    almost=fraction >= almost_threshold, submission_error=submission_error)
    base["status"] = status
    if filtered_results_path is not None:
        # Export only metadata from the *same filtered result* used above. Raw
        # eval JSON includes ignored tests and potentially hidden error bodies.
        # A metadata write failure must not turn a valid official score unknown.
        try:
            write_json_atomic(Path(filtered_results_path), {
                "schema_version": 1, "task_id": task_id, "official_status": status,
                "reward": base["reward"], "passed_tests": summary.n_resolved,
                "valid_tests": summary.n_tests,
                "submission_error": base.get("submission_error", False),
                "tests": [{"test_id": f"{test.branch}/{test.name}" if test.branch else test.name,
                           "status": test.status} for test in result.test_results]})
        except (OSError, ValueError, TypeError) as exc:
            base["filtered_results_error"] = type(exc).__name__
    return base


def evaluate_submission(task: ProgramBenchTask, submission_path: str | Path,
                        output_dir: str | Path, *,
                        docker_cpus: int = OFFICIAL_EVALUATOR_CPUS,
                        docker_executable: str | None = None) -> dict[str, Any]:
    """Evaluate once with official API in fresh, separate source/result directories.

    Must run after all agents stopped. The official evaluator owns new compile
    and test containers, cleanup, test retries, source protections and scoring.
    Its official per-step timeouts apply; ``task.verifier_timeout_sec`` is not an
    additional aggregate deadline that could interrupt container cleanup.
    The official 10-CPU evaluator default is independent of the inference task's
    20-CPU default. Explicit evaluator overrides are recorded as custom settings.
    An explicit Docker executable uses a fresh official-API subprocess because
    ProgramBench reads that setting at import time. No aggregate timeout is added.
    """
    if isinstance(docker_cpus, bool) or not isinstance(docker_cpus, int) or docker_cpus < 1:
        raise ValueError("docker_cpus must be a positive integer")
    if docker_executable is not None and (not isinstance(docker_executable, str)
            or not docker_executable.strip() or "\0" in docker_executable):
        raise ValueError("docker_executable must be a nonempty executable path or None")
    submission = Path(submission_path).resolve(strict=True)
    validate_submission(submission)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / "submission_run"
    instance_dir = run / task.task_id
    instance_dir.mkdir(parents=True)
    shutil.copyfile(submission, instance_dir / "submission.tar.gz")
    evaluation_root = output / "official_eval"
    eval_path = evaluation_root / run.name / task.task_id / f"{task.task_id}.eval.json"
    # Same naming and executable as the official evaluator (programbench.constants).
    image_docker = docker_executable or os.environ.get("PROGRAMBENCH_DOCKER_EXECUTABLE", "docker")
    image_reference = (f"{os.environ.get('PROGRAMBENCH_DOCKER_ORG', 'programbench')}/"
                       f"{task.task_id.replace('__', '_1776_')}:{CLEANROOM_TAG}")
    image_before = _docker_image_id(image_docker, image_reference)
    started = time.monotonic()
    failure: str | None = None
    try:
        with (output / "official_eval.log").open("w", encoding="utf-8") as log:
            filter_spec = "^" + re.escape(task.task_id) + "$"
            if docker_executable is None:
                with redirect_stdout(log), redirect_stderr(log):
                    _, _, run_eval_batch, *_ = _official_api()
                    run_eval_batch(sources=[run], output=evaluation_root, workers=1, branch_workers=1,
                                   docker_cpus=docker_cpus, image_tag=CLEANROOM_TAG,
                                   filter_spec=filter_spec, force=False)
            else:
                environment = {**os.environ, "PROGRAMBENCH_DOCKER_EXECUTABLE": docker_executable}
                subprocess.run([sys.executable, "-c", _OFFICIAL_EVAL_SUBPROCESS,
                                str(run), str(evaluation_root), str(docker_cpus), CLEANROOM_TAG, filter_spec],
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = summarize_official_result(
            task.task_id, eval_path, filtered_results_path=output / "filtered_test_results.json")
    except Exception as exc:
        # Exception text may contain hidden test/log bodies. Keep only its type.
        failure = type(exc).__name__
        result = {"task_id": task.task_id, "status": "verifier_error", "reward": None,
                  "resolved": None, "almost": None, "error_code": failure,
                  "eval_json": str(eval_path)}
    elapsed = time.monotonic() - started  # Excludes both provenance inspects.
    # docker run pulls a missing tag implicitly, so "before" may be unknown while "after" is set.
    image_after = _docker_image_id(image_docker, image_reference)
    result.update(repository_id=task.repository_id, verifier_elapsed_seconds=elapsed,
                  official_evaluator="programbench.eval.eval_batch.run_eval_batch",
                  official_info_filtering=True, inference_environment_reused=False,
                  docker_cpus=docker_cpus, image_tag=CLEANROOM_TAG,
                  docker_executable=docker_executable,
                  evaluator_invocation=("official_api_subprocess" if docker_executable is not None
                                        else "official_api_in_process"),
                  evaluator_resource_limits_source=("official_default" if docker_cpus == OFFICIAL_EVALUATOR_CPUS
                                                    else "custom"),
                  official_evaluator_docker_cpus=OFFICIAL_EVALUATOR_CPUS,
                  official_compile_timeout_seconds=OFFICIAL_EVALUATOR_COMPILE_TIMEOUT_SEC,
                  official_test_branch_timeout_seconds=OFFICIAL_EVALUATOR_BRANCH_TIMEOUT_SEC,
                  verifier_timeout_scope="official_per_step", evaluator_exception_type=failure,
                  evaluator_image={"reference": image_reference, "before": image_before, "after": image_after,
                                   "source": "docker inspect --type image .Id before and after the official evaluator",
                                   "scope": "tagged compile image; per-branch committed images are not observed"})
    write_json_atomic(output / "result.json", result)
    return result
