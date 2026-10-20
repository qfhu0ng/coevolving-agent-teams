"""Command line entry points for preparing and running stage-one experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from agent_company.roadmap.data import download_task
from agent_company.evolution.loop import RunRequest, TaskSpec, run_baseline, run_stage1
from agent_company.runtime.network import NETWORK_POLICY
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.runtime.provider import load_provider, public_provider
from agent_company.evolution.reviewer import StrategyReviewer
from agent_company.roadmap.task import Task
from agent_company.roadmap.runner import RoadmapRunner
from agent_company.evolution.selection import SelectionConfig


POLICIES = Path(__file__).with_name("policies")


def _positive_memory_mb(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Task memory must be a positive integer (MiB)") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Task memory must be a positive integer (MiB)")
    return parsed


def load_manifest(path: Path, *, baseline: bool = False) -> tuple[dict, list[TaskSpec], list[TaskSpec]]:
    path = path.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    groups = []
    for split in (("tasks",) if baseline and "tasks" in document else ("train", "test")):
        tasks = []
        for row in document[split]:
            if not row.get("repo_id"):
                raise ValueError("Every task needs an explicit repo_id for the research split")
            task_path = (path.parent / row["path"]).resolve()
            task = Task.load(task_path, repository_id=row["repo_id"])
            if row.get("task_id", task.task_id) != task.task_id:
                raise ValueError("Manifest task_id disagrees with task.toml")
            payload = {"path": str(task_path)}
            if "prepared_image" in row:
                image = row["prepared_image"]
                if not isinstance(image, str) or not image or any(c.isspace() for c in image):
                    raise ValueError("prepared_image must be a nonempty Docker image reference")
                payload["prepared_image"] = image
            tasks.append(TaskSpec(task.task_id, row["repo_id"], payload))
        groups.append(tasks)
    return document, groups[0], groups[1] if len(groups) > 1 else []


def _runtime_arguments(parser):
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--qwen-package", type=Path, required=True,
                        help="Released Qwen package containing direct cli.js")
    parser.add_argument("--node-directory", type=Path, required=True,
                        help="Linux Node >=22.15 directory with bin/node; prepared ARM bundles may "
                             "include compat/aarch64-linux-gnu for amd64 task images")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--task-memory-mb", type=_positive_memory_mb,
                        default=os.environ.get("STAGE1_TASK_MEMORY_MB"),
                        help="Shared task-container RAM limit in MiB for all modes/phases; "
                             "default: STAGE1_TASK_MEMORY_MB or official task.toml. "
                             "8192 = 8 GiB, not 8 GiB per worker; CPU and time remain unchanged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assignment-policy", type=Path, default=POLICIES / "assignment.md")
    parser.add_argument("--communication-policy", type=Path, default=POLICIES / "communication.md")
    parser.add_argument("--team-stall-timeout", type=float, default=1800,
                        help="Team-only Qwen inactivity timeout; task wall-time comes from task.toml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--provider-supports-seed", action="store_true",
                        help="Explicitly allow passing the requested seed to this provider")


def _recovery_arguments(parser):
    parser.add_argument("--resume", action="store_true",
                        help="Reuse terminal task/proposal records and continue pending work")
    parser.add_argument("--retry-interrupted", action="store_true",
                        help="With --resume, archive unfinished tasks and restart from fresh environments")


def _mode_argument(parser):
    parser.add_argument("--mode", choices=("single", "team"), default="single",
                        help="single (default): one writable coding agent; team: master assigns workers")


def _runtime_config(args, provider, tasks):
    return {"provider": public_provider(provider),
            "network_policy": NETWORK_POLICY,
            **({"task_memory_mb": args.task_memory_mb} if args.task_memory_mb is not None else {}),
            "qwen_package": str(args.qwen_package.resolve()),
            "node_directory": str(args.node_directory.resolve()),
            "platform": args.platform, "team_stall_timeout": args.team_stall_timeout,
            "provider_supports_seed": args.provider_supports_seed,
            "task_inputs": [{"path": task.payload["path"],
                             "task_toml": (Path(task.payload["path"]) / "task.toml").read_text(),
                             "instruction": (Path(task.payload["path"]) / "instruction.md").read_text()}
                            for task in tasks]}


def _selection_config(document, args):
    values = document.get("selection_config", {})
    if not isinstance(values, dict):
        raise ValueError("Manifest selection_config must be an object")
    values = dict(values)
    for name in ("quality_tolerance", "min_score_gain", "min_cost_reduction_fraction",
                 "min_token_reduction_fraction", "efficiency_metric"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    return SelectionConfig(**values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download official task interface and verifier")
    download.add_argument("task_ids", nargs="+")
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument("--revision", default="main")
    inspect = commands.add_parser("inspect", help="Inspect a task's metadata without executing it")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--repo-id")
    one = commands.add_parser("run-task", help="Run a single agent or team and the final verifier on one fresh task")
    _runtime_arguments(one)
    _mode_argument(one)
    one.add_argument("--task", type=Path, required=True)
    one.add_argument("--repo-id", required=True)
    run = commands.add_parser("run", help="Jointly evolve team policies, then compare with a held-out baseline")
    _runtime_arguments(run)
    _recovery_arguments(run)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--optimizer-provider-config", type=Path,
                     help="Defaults to the task provider; can use a separate optimizer model")
    run.add_argument("--reviewer-provider-config", type=Path,
                     help="Defaults to the optimizer provider, in an independent review conversation")
    run.add_argument("--reviewer-max-tool-rounds", type=int,
                     help="Maximum reviewer model requests (default 8); overrides the manifest")
    run.add_argument("--reviewer-max-context-chars", type=int,
                     help="Reviewer serialized-context character limit (default 180000); overrides the manifest")
    run.add_argument("--selection-mode", choices=("review_replay", "legacy_no_replay"),
                     default="review_replay",
                     help="review_replay (default): review then execute/select; legacy_no_replay: explicit old path")
    run.add_argument("--evidence-scope", choices=("current_batch", "cumulative"),
                     help="Training evidence visible to optimizer/reviewer: current_batch by default; legacy uses cumulative")
    run.add_argument("--quality-tolerance", type=float,
                     help="Minimum clear-gain tolerance (never allows score regression); overrides manifest")
    run.add_argument("--min-score-gain", type=float,
                     help="Minimum selection score gain; overrides manifest selection_config")
    run.add_argument("--min-cost-reduction-fraction", type=float,
                     help="Minimum relative USD cost reduction at equivalent quality; overrides manifest selection_config")
    run.add_argument("--min-token-reduction-fraction", type=float,
                     help="Minimum relative total-token reduction without score regression")
    run.add_argument("--efficiency-metric", choices=("tokens", "cost_usd"),
                     help="Default tokens; cost_usd is explicit legacy cost selection")
    run.add_argument("--batch-size", type=int,
                     help="Training tasks per batch (default 5); overrides the manifest")
    run.add_argument("--candidates-per-batch", type=int,
                     help="Candidates proposed from the same incumbent per batch (default 1)")
    run.add_argument("--replay-task-count", type=int,
                     help="Replay N batch tasks for every candidate (default: 2, capped by batch size)")
    run.add_argument("--replay-selection", choices=("first_n", "reviewer"),
                     help="Select replay tasks by reviewer (default) or explicit manifest order; overrides manifest")
    run.add_argument("--test-repetitions", type=int)
    run.add_argument("--baseline-mode", choices=("single", "team"), default="single",
                     help="Held-out reference: single (default) or the fixed initial team policies")
    baseline = commands.add_parser("baseline", help="Evaluate manifest tasks without an optimizer or reviewer")
    _runtime_arguments(baseline)
    _mode_argument(baseline)
    _recovery_arguments(baseline)
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "download":
        paths = [str(download_task(task, args.destination, revision=args.revision))
                 for task in args.task_ids]
        print(json.dumps({"downloaded": paths}, indent=2))
        return 0
    if args.command == "inspect":
        print(json.dumps(Task.load(args.path, repository_id=args.repo_id).describe(), indent=2))
        return 0
    provider = load_provider(args.provider_config)
    # Solo tasks do not consume team policies, including their default files.
    uses_policies = args.command == "run" or args.mode == "team"
    assignment = args.assignment_policy.read_text(encoding="utf-8") if uses_policies else ""
    communication = args.communication_policy.read_text(encoding="utf-8") if uses_policies else ""
    runner = RoadmapRunner(provider, args.qwen_package, args.node_directory,
                               platform=args.platform,
                               team_stall_timeout_seconds=args.team_stall_timeout,
                               task_memory_mb=args.task_memory_mb,
                               provider_supports_seed=args.provider_supports_seed)
    if args.command == "run-task":
        task = Task.load(args.task, repository_id=args.repo_id)
        args.output.mkdir(parents=True, exist_ok=False)
        request = RunRequest(TaskSpec(task.task_id, args.repo_id, {"path": str(task.path)}),
                             "development", 0, communication, assignment,
                             args.output, seed=args.seed, execution_mode=args.mode)
        serialized = asdict(request)
        serialized["output_dir"] = str(args.output.resolve())
        (args.output / "run_request.json").write_text(json.dumps(serialized, indent=2) + "\n")
        result = asdict(runner(request))
        result["execution_mode"] = args.mode
        result["policy_applied"] = args.mode == "team"
        (args.output / "task_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in
                          ("status", "execution_mode", "score", "cost_usd", "error")}, indent=2))
        return 0 if result["status"] == "completed" else 1
    document, train, test = load_manifest(args.manifest, baseline=args.command == "baseline")
    runtime_config = _runtime_config(args, provider, train + test)
    if args.command == "baseline":
        summary = run_baseline(tasks=train + test, communication_policy=communication,
                               assignment_policy=assignment, run_task=runner, output_dir=args.output,
                               repetitions=args.repetitions, seed=args.seed,
                               resume=args.resume, retry_interrupted=args.retry_interrupted,
                               runtime_config=runtime_config, execution_mode=args.mode)
        print(json.dumps({key: summary[key] for key in
                          ("status", "execution_mode", "results", "optimizer_invoked")}, indent=2))
        return 0 if summary["status"] == "completed" else 1
    if args.selection_mode == "legacy_no_replay" and (
            document.get("selection_config") or any(getattr(args, name) is not None for name in
                ("quality_tolerance", "min_score_gain", "min_cost_reduction_fraction",
                 "min_token_reduction_fraction", "efficiency_metric"))):
        raise ValueError("legacy_no_replay does not accept selection thresholds")
    selection_config = _selection_config(document, args) if args.selection_mode == "review_replay" else None
    candidates_per_batch = (args.candidates_per_batch if args.candidates_per_batch is not None
                            else document.get("candidates_per_batch", 1))
    replay_task_count = (args.replay_task_count if args.replay_task_count is not None
                         else document.get("replay_task_count", 2 if args.selection_mode == "review_replay" else None))
    replay_selection = (args.replay_selection if args.replay_selection is not None
                        else document.get("replay_selection", "reviewer" if args.selection_mode == "review_replay" else "first_n"))
    if args.selection_mode == "legacy_no_replay" and (
        candidates_per_batch != 1 or replay_task_count is not None or replay_selection != "first_n"
    ):
        raise ValueError("legacy_no_replay requires candidates_per_batch=1 and no replay selection")
    optimizer_provider = (load_provider(args.optimizer_provider_config)
                          if args.optimizer_provider_config else provider)
    optimizer = StrategyOptimizer(optimizer_provider, args.output / "optimizer_traces")
    runtime_config["optimizer_provider"] = public_provider(optimizer_provider)
    reviewer = None
    if args.selection_mode == "review_replay":
        reviewer_provider = (load_provider(args.reviewer_provider_config)
                             if args.reviewer_provider_config else optimizer_provider)
        reviewer_limits = {
            "max_tool_rounds": (args.reviewer_max_tool_rounds if args.reviewer_max_tool_rounds is not None
                                else document.get("reviewer_max_tool_rounds", 8)),
            "max_context_chars": (args.reviewer_max_context_chars if args.reviewer_max_context_chars is not None
                                   else document.get("reviewer_max_context_chars", 180000)),
        }
        reviewer = StrategyReviewer(reviewer_provider, args.output / "reviewer_traces", **reviewer_limits)
        runtime_config["reviewer_provider"] = public_provider(reviewer_provider)
        runtime_config["reviewer_limits"] = reviewer_limits
    summary = run_stage1(
        train_tasks=train, test_tasks=test, initial_communication=communication,
        assignment_policy=assignment, run_task=runner, propose_policy=optimizer,
        output_dir=args.output,
        batch_size=args.batch_size if args.batch_size is not None else document.get("batch_size", 5),
        split_mode=document.get("split_mode", "repo"),
        test_repetitions=(args.test_repetitions if args.test_repetitions is not None
                          else document.get("test_repetitions", 1)), seed=args.seed,
        resume=args.resume, retry_interrupted=args.retry_interrupted, runtime_config=runtime_config,
        baseline_mode=args.baseline_mode, review_policy=reviewer,
        selection_mode=args.selection_mode, selection_config=selection_config,
        candidates_per_batch=candidates_per_batch, replay_task_count=replay_task_count,
        evidence_scope=args.evidence_scope, replay_selection=replay_selection,
    )
    print(json.dumps({key: summary[key] for key in
                      ("status", "baseline_mode", "final_policy_version", "train",
                       "test_baseline", "test_initial", "test_final", "optimizer", "reviewer",
                       "replay", "selection_mode", "replay_selection", "evidence_scope", "selection", "selection_results",
                       "all_costs") if key in summary},
                     indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
