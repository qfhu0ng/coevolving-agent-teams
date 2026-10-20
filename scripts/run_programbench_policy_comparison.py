#!/usr/bin/env python3
"""Plan, or explicitly run, fresh ProgramBench tasks across named policy arms.

Planning reads only public tasks, instructions and policy files, never provider
credentials. --run executes every configured task/arm once, in task-local waves
of at most two arms. This entry point neither evolves policies nor resumes runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.runtime_io import write_json_atomic
from agent_company.deploy.code_version import code_version
from agent_company.evolution.harness_capabilities import native_version
from agent_company.runtime.provider import public_provider, load_provider
from agent_company.evolution.loop import RunRequest, TaskResult, TaskSpec
# Planning lives in the package; these names stay importable from this script.
from agent_company.programbench.comparison import (  # noqa: F401
    PreparedComparison, _path, prepare_comparison, runner_inference_options)
from agent_company.programbench.runner import ProgramBenchRunner


def run_comparison(profile_path: Path, project_root: Path, output: Path, *, qwen_package: Path,
                   node_directory: Path, platform="linux/amd64", docker="docker", parallelism=None,
                   repeat_index=None, pair_id=None, batch_index=None) -> dict:
    """Explicit execution entry; no retries or reuse of prior task outputs."""
    if type(repeat_index) is not int or repeat_index < 0:
        raise ValueError("Explicit nonnegative repeat_index is required for execution")
    prepared = prepare_comparison(profile_path, project_root, parallelism=parallelism,
                                  repeat_index=repeat_index, pair_id=pair_id, batch_index=batch_index)
    plan = prepared.plan
    if "batch_index" in plan and batch_index is None:
        raise ValueError("Role-series execution requires explicit batch_index")
    output = Path(output).resolve()
    # The existing no-overwrite directory boundary prevents duplicate paid runs.
    output.mkdir(parents=True, exist_ok=False)
    # Confirmation profiles may live beside the run rather than in immutable
    # deployment code/. Keep their exact source with the execution evidence.
    (output / "profile.json").write_bytes(Path(profile_path).read_bytes())
    provider = load_provider(Path(plan["provider_config"]))
    inference = plan["inference"]
    options = dict(provider=provider, qwen_package=Path(qwen_package).resolve(),
                   node_directory=Path(node_directory).resolve(), platform=platform, docker=docker,
                   **runner_inference_options(inference), evaluator_cpus=plan["evaluation"]["docker_cpus"])
    write_json_atomic(output / "run_settings.json", {**plan, "provider": public_provider(provider),
                      "qwen_package": str(options["qwen_package"]), "node_directory": str(options["node_directory"]),
                      "platform": platform, "docker": docker, "code_version": (code_version(project_root)
                                       if Path(project_root).resolve() == Path(__file__).resolve().parents[1] else None),
                      "qwen_version": native_version(options["qwen_package"])})
    requests, metadata, order = {}, {}, []
    task_metadata = {row["task_id"]: row for row in plan["tasks"]}
    for task in prepared.tasks:
        round_metadata = {key: task_metadata[task.task_id][key] for key in ("repeat_index", "pair_id")}
        for arm in plan["arms"]:
            key = (task.task_id, arm["arm_id"])
            directory = output / task.task_id / arm["arm_id"]
            directory.mkdir(parents=True, exist_ok=False)
            assignment, communication = prepared.policies[arm["arm_id"]]
            request = RunRequest(
                task=TaskSpec(task.task_id, task.repository_id,
                              {"path": str(task.path), "instruction_path": str(task.instruction_path)}),
                split="heldout", policy_version=arm["policy_version"],
                communication_policy=communication, assignment_policy=assignment,
                output_dir=directory, fresh_start=True, **round_metadata,
                execution_mode=arm["execution_mode"], role_policy_mode=arm.get("role_policy_mode"),
                policy_package=arm.get("policy_package"))
            if request.policy_package is not None:
                write_json_atomic(directory / "policy_package.json", request.policy_package)
            requests[key] = request
            metadata[key] = {"task_id": task.task_id, "arm_id": arm["arm_id"],
                             "execution_mode": arm["execution_mode"], "policy_version": arm["policy_version"],
                             **round_metadata,
                             **({"role_policy_mode": request.role_policy_mode}
                                if request.execution_mode == "team" and request.policy_package is not None else {})}
            order.append(key)
            write_json_atomic(directory / "run_request.json", {
                **asdict(request), "output_dir": str(directory), "arm_id": arm["arm_id"]})
    launched, finished = {}, {}
    state = "running"

    def progress():
        attention = [metadata[key] for key in order if key in finished
                     and finished[key]["inference_cleanup_confirmed"] is not True]
        rows = [finished[key] for key in order if key in finished]
        return {"state": state, "terminal": len(finished), "planned": len(order), "launched": len(launched),
                "parallelism": plan["parallelism"], "rows": rows, "requires_user_action": bool(attention),
                "cleanup_attention": attention, "pending": [metadata[key] for key in order if key not in launched],
                "running": [metadata[key] for key in order if key in launched and key not in finished],
                "automatic_restarts": False}

    def save_progress():
        write_json_atomic(output / "progress.json", progress())

    def execute(request):
        try:
            return ProgramBenchRunner(**options)(request)
        except Exception as exc:
            # Do not expose exception bodies: commands may contain keys or traces.
            name = type(exc).__name__
            return TaskResult(status="failed", error=name, metrics={
                "runner_exception_type": name, "inference_cleanup_confirmed": None})

    save_progress()
    with ThreadPoolExecutor(max_workers=plan["parallelism"], thread_name_prefix="programbench-policy") as executor:
        for wave in plan["waves"]:
            if progress()["requires_user_action"]:
                state = "blocked_cleanup"
                break
            futures = {}
            for item in wave:
                key = (item["task_id"], item["arm_id"])
                launched[key] = datetime.now(timezone.utc).isoformat()
                write_json_atomic(requests[key].output_dir / "launch.json", {
                    **metadata[key], "dispatched_at": launched[key], "fresh_start": True,
                    "automatic_restarts": False})
                save_progress()
                futures[executor.submit(execute, requests[key])] = key
            # Complete the current wave even if one arm fails or cleanup is unknown.
            for future in as_completed(futures):
                key, result = futures[future], future.result()
                write_json_atomic(requests[key].output_dir / "task_result.json", {**asdict(result), **metadata[key]})
                finished[key] = {**metadata[key], "status": result.status, "score": result.score,
                                 "dispatched_at": launched[key], "finished_at": datetime.now(timezone.utc).isoformat(),
                                 "inference_cleanup_confirmed": result.metrics.get("inference_cleanup_confirmed"),
                                 "exception_type": result.metrics.get("runner_exception_type")}
                save_progress()
    state = "blocked_cleanup" if progress()["requires_user_action"] else "completed"
    save_progress()
    return progress()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--parallelism", type=int, choices=(1, 2))
    parser.add_argument("--repeat-index", type=int, help="Explicit round, overriding the profile; does not repeat execution")
    parser.add_argument("--pair-id", help="Override the pair ID (single-task profiles only)")
    parser.add_argument("--batch-index", type=int, choices=(1, 2), help="Explicit new role-series batch")
    parser.add_argument("--run", action="store_true", help="Explicitly execute real inference and evaluation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qwen-package", type=Path)
    parser.add_argument("--node-directory", type=Path)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args(argv)
    if args.run:
        if args.repeat_index is None:
            parser.error("--run requires explicit --repeat-index; a profile default is not execution authorization")
        if any(getattr(args, name) is None for name in ("output", "qwen_package", "node_directory")):
            parser.error("--run requires --output, --qwen-package and --node-directory")
        result = run_comparison(args.profile, args.project_root, args.output, qwen_package=args.qwen_package,
                                node_directory=args.node_directory, platform=args.platform, docker=args.docker,
                                parallelism=args.parallelism, repeat_index=args.repeat_index,
                                pair_id=args.pair_id, batch_index=args.batch_index)
    else:
        result = prepare_comparison(args.profile, args.project_root, parallelism=args.parallelism,
                                    repeat_index=args.repeat_index, pair_id=args.pair_id,
                                    batch_index=args.batch_index).plan
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("requires_user_action") else 0


if __name__ == "__main__":
    raise SystemExit(main())
