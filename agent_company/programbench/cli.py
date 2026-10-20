"""Explicit ProgramBench preparation, Single/P3 execution and official evaluation.

Nothing runs on import; describing/preparing a task never loads provider credentials.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

from agent_company.runtime_io import write_json_atomic
from agent_company.runtime.provider import load_provider, public_provider
from agent_company.evolution.loop import RunRequest, TaskResult, TaskSpec
from agent_company.programbench.task import ProgramBenchTask, evaluate_submission
from agent_company.programbench.workspace import programbench_source_boundary


def prepare_instructions(config_path: Path, output: Path) -> str:
    """Read the public official mini-SWE prompt; omit its runtime/tool syntax."""
    import yaml
    config = yaml.safe_load(config_path.read_text())
    system = config["agent"]["system_template"].strip()
    instance = config["agent"]["instance_template"]
    marker = "## Command Execution Rules"
    if marker not in instance or "## Recommended Workflow" not in instance:
        raise ValueError("Official prompt structure changed; review before adapting")
    instance = instance.split(marker, 1)[0].strip()
    instructions = (system + "\n\n" + instance + "\n\n## Qwen workspace path mapping\n\n"
                    "In the official text above, the reference ./executable is relocated to "
                    "/opt/ac-programbench-reference/executable. Use that absolute path for all "
                    "reference queries. The clean implementation root is /workspace/solution; "
                    "./compile.sh and its output ./executable refer to that source root (or a "
                    "worker's own worktree), not to the reference. Documentation is in the "
                    "source root. Full delivered source, including untracked files, will be "
                    "submitted after all agents stop. Qwen's own tools and completion mechanism "
                    "replace mini-SWE's command formatting/submission sentinel.\n\n"
                    + programbench_source_boundary() + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(instructions)
    return instructions


def run_pair(args) -> dict:
    from agent_company.programbench.runner import ProgramBenchRunner
    parallelism = getattr(args, "parallelism", 1)
    if isinstance(parallelism, bool) or not isinstance(parallelism, int) or parallelism not in {1, 2}:
        raise ValueError("parallelism must be 1 or 2")
    profile = json.loads(args.profile.read_text())
    if (profile["benchmark"] != "ProgramBench" or profile["execution_modes"] != ["single", "team"]
            or profile["runs_per_mode"] != 1 or profile["evolution_enabled"] is not False
            or profile["policy_version"] != "P3"):
        raise ValueError("This entry point runs exactly one Single/P3 pair, with no evolution")
    inference = profile["inference"]
    if inference["memory_swap_mb"] != inference["memory_mb"]:
        raise ValueError("ProgramBench inference must not add swap beyond the memory ceiling")
    task = ProgramBenchTask.load(args.task, instruction_path=args.instructions,
                                agent_timeout_sec=inference["agent_timeout_sec"],
                                cpus=inference["cpus"], memory_mb=inference["memory_mb"])
    if task.task_id != profile["task_id"]:
        raise ValueError("Task disagrees with the selected run profile")
    root = args.project_root.resolve()
    assignment = (root / profile["assignment_policy_path"]).read_text()
    communication = (root / profile["communication_policy_path"]).read_text()
    provider = load_provider(root / profile["provider_config"])
    runner_options = dict(provider=provider, qwen_package=args.qwen_package,
                          node_directory=args.node_directory, platform=args.platform,
                          task_memory_mb=inference["memory_mb"], task_cpus=inference["cpus"],
                          agent_timeout_seconds=inference["agent_timeout_sec"],
                          max_model_calls=inference["max_model_calls"],
                          evaluator_cpus=profile["evaluation"]["docker_cpus"], docker=args.docker)
    # A new output directory is the ordinary no-overwrite boundary. No implicit resume.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "run_settings.json", {
        "profile": profile, "provider": public_provider(provider), "task": task.describe(),
        "source_instruction": str(args.instructions.resolve()), "evolution_enabled": False,
        "runtime": "custom_Qwen_not_official_mini_SWE", "new_rollouts": 2, "reused_rollouts": 0,
        "parallelism": parallelism, "automatic_restarts": False})
    spec = TaskSpec(task.task_id, task.repository_id,
                    {"path": str(task.path), "instruction_path": str(task.instruction_path)})
    requests = []
    for mode in ("single", "team"):
        output = args.output / mode
        output.mkdir()
        request = RunRequest(spec, "integration", 3, communication if mode == "team" else "",
                             assignment if mode == "team" else "", output,
                             pair_id=task.task_id, execution_mode=mode)
        write_json_atomic(output / "run_request.json", {**asdict(request), "output_dir": str(output)})
        requests.append(request)

    def execute(request):
        try:
            # A separate runner/workspace/relay per arm. Explicit docker selects
            # an official-evaluator subprocess: no global environment or logging
            # redirection is shared by the two inference/evaluation threads.
            return ProgramBenchRunner(**runner_options)(request)
        except Exception as exc:
            # Never expose subprocess commands, credentials or transcript text.
            name = type(exc).__name__
            return TaskResult(status="failed", error=name, metrics={
                "runner_exception_type": name, "inference_cleanup_confirmed": None})

    finished = {}
    launched = 0

    def progress():
        rows = [finished[request.execution_mode] for request in requests if request.execution_mode in finished]
        attention = [row["mode"] for row in rows if row["inference_cleanup_confirmed"] is not True]
        return {"terminal": len(rows), "planned": 2, "launched": launched, "parallelism": parallelism,
                "rows": rows, "requires_user_action": bool(attention), "cleanup_attention_modes": attention,
                "automatic_restarts": False}

    def record(request, result):
        # Called only by the scheduling thread, including for concurrent runs.
        write_json_atomic(request.output_dir / "task_result.json", asdict(result))
        finished[request.execution_mode] = {
            "mode": request.execution_mode, "status": result.status, "score": result.score,
            "inference_cleanup_confirmed": result.metrics.get("inference_cleanup_confirmed"),
            "exception_type": result.metrics.get("runner_exception_type")}
        write_json_atomic(args.output / "progress.json", progress())

    write_json_atomic(args.output / "progress.json", progress())
    if parallelism == 1:
        for request in requests:
            launched += 1
            write_json_atomic(args.output / "progress.json", progress())
            result = execute(request)
            record(request, result)
            if result.metrics.get("inference_cleanup_confirmed") is not True:
                raise RuntimeError("Previous inference cleanup is unconfirmed; do not launch the next arm")
    else:
        # Exactly one submission per arm; a failure never cancels, restarts or
        # terminates the other arm. Unconfirmed cleanup remains visible for the
        # user to resolve after the independently running arm finishes.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="programbench-arm") as executor:
            futures = {executor.submit(execute, request): request for request in requests}
            launched = len(futures)
            write_json_atomic(args.output / "progress.json", progress())
            for future in as_completed(futures):
                record(futures[future], future.result())
    return progress()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-instructions", help="Public prompt only; no models or tests")
    prepare.add_argument("--official-config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    describe = commands.add_parser("describe", help="Public task metadata only")
    evaluate = commands.add_parser("evaluate", help="Evaluate an existing source archive once")
    run = commands.add_parser("run-pair", help="Launch one real Single/P3 pair (requires credentials)")
    for item in (describe, evaluate, run):
        item.add_argument("--task", type=Path, required=True)
        item.add_argument("--instructions", type=Path, required=True)
    evaluate.add_argument("--submission", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--evaluator-cpus", type=int, default=10)
    evaluate.add_argument("--docker", default="docker", help="Official evaluator Docker executable")
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--project-root", type=Path, default=Path.cwd())
    run.add_argument("--qwen-package", type=Path, required=True)
    run.add_argument("--node-directory", type=Path, required=True)
    run.add_argument("--platform", default="linux/amd64")
    run.add_argument("--docker", default="docker", help="Docker executable or sudo wrapper")
    run.add_argument("--parallelism", type=int, choices=(1, 2), default=1,
                     help="Run sequentially (1) or concurrently (2); each arm runs exactly once")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-instructions":
        prepare_instructions(args.official_config, args.output)
        result = {"instruction_path": str(args.output.resolve()), "model_calls": 0}
    elif args.command == "describe":
        result = ProgramBenchTask.load(args.task, instruction_path=args.instructions).describe()
    elif args.command == "evaluate":
        task = ProgramBenchTask.load(args.task, instruction_path=args.instructions)
        result = evaluate_submission(task, args.submission, args.output, docker_cpus=args.evaluator_cpus,
                                     docker_executable=args.docker)
    else:
        result = run_pair(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
