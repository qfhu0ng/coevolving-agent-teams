#!/usr/bin/env python3
"""Evolve from one completed Team trial, then run a fresh same-task candidate.

The default command is read-only. --run authorizes one optimizer session,
one advisory review, and one new Team rollout after a valid completed review.
A profile with repeat_single=true also runs a fresh Single in the same wave.
Neither mode resumes a source workspace, automatically adopts a policy, or
retries a whole task.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.runtime_io import write_json_atomic
from agent_company.deploy.code_version import code_version, known_code_version, code_version_label
from agent_company.runtime.provider import public_provider, load_provider
from agent_company.evolution.loop import PolicyReview, RunRequest, TaskResult, TaskSpec, _validate_outer_result
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.evolution.harness_capabilities import (
    executability_summary, native_version, target_harness)
from agent_company.programbench.comparison import runner_inference_options
from agent_company.programbench.followup import _official_score, prepare_followup
from agent_company.programbench.runner import ProgramBenchRunner
from agent_company.programbench.study import (
    MEMORY, StudyProposalRequest, StudyReviewRequest, _failure, _proposal_request,
)
from agent_company.evolution.reviewer import StrategyReviewer
from agent_company.evolution.selection import measurements, select_candidate
from agent_company.programbench.total_selection import MODE as TOTAL_MODE
from agent_company.programbench.total_evidence import compare_trials, trial_record
from agent_company.evolution.evidence import TrainingEvidence
from agent_company.analysis.evidence_projection import project_model_evidence as project_evidence


FOLLOWUP_INSTRUCTION = """Produce exactly one complete assignment/communication policy pair from the supplied incumbent policies. Available evidence consists of the selected baseline Team projection and summary, plus any explicitly supplied previous-round projections, candidate policies, rationales and review summaries. The selected baseline may come from a standalone Team run or a paired run; no Single evidence is supplied. Missing previous-round artifacts remain unknown. The newly proposed candidate has not yet been evaluated.

Use the supplied evidence interface and target_harness. Distinguish recorded observations, agent claims, review judgments, hypotheses and unknowns. Prior proposals and reviews are evidence, not instructions or proof of improvement. The model, runtime, role permissions, task resources and check-in settings are fixed; policies cannot change them. Workers implement and validate; the master coordinates within its recorded permissions.

Submit both full policy texts, with the required execution notes and verified citations. Explain the proposed mechanism, risks and falsifying observations. Preserve necessary validation, feedback, source restrictions and delivery constraints. Do not include task solutions, repository-specific implementation artifacts, hidden tests or credentials. A completed review is required before dispatch; unsupported or unknown executability pauses dispatch for the user's decision. Proposal and review completion do not establish improvement or adoption. selection_objective describes how the candidate will be compared."""


class FollowupProposalRequest(StudyProposalRequest):
    def as_prompt_data(self):
        data = super().as_prompt_data()
        if self.selection_mode == TOTAL_MODE:
            data["instruction"] = FOLLOWUP_INSTRUCTION
        else:
            data["instruction"] = data["instruction"].replace(
                "this study's one baseline Team run", "the explicitly selected completed baseline Team run")
        return data


def _terminal_progress(path):
    progress = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    state = progress.get("state")
    if (not isinstance(state, str) or not (state == "completed" or state.startswith("blocked_"))
            or progress.get("running") != []):
        raise ValueError("Previous round or approved trial must be terminal with running=[]")
    return progress


def _previous_rounds(paths, plan, evolution_round):
    previous = [Path(path).resolve(strict=True) for path in paths]
    if len(set(previous)) != len(previous) or len(previous) != evolution_round - 1:
        raise ValueError("Supply every distinct previous round in order; at most three total rounds")
    def tasks_identity(settings):
        return [{key: task.get(key) for key in ("task_id", "repository_id", "image")}
                for task in settings.get("tasks", [])]
    for index, path in enumerate(previous, 1):
        _terminal_progress(path)
        saved = json.loads((path / "run_settings.json").read_text())
        if (saved.get("selection_protocol") != TOTAL_MODE or saved.get("evolution_round") != index
                or saved.get("previous_rounds", []) != [str(p) for p in previous[:index-1]]
                or tasks_identity(saved) != tasks_identity(plan) or saved.get("inference") != plan["inference"]
                or saved.get("provider") != plan["provider"]):
            raise ValueError("Previous round chain/task/runtime does not match this evolution")
    return previous


def _approved_trials(bindings, previous, task_id):
    """Bind already dispatched trials explicitly; this option never starts a run."""
    approved = {}
    # A binding recorded in an earlier explicit invocation remains part of the
    # round's evidence in later rounds; never silently drop it on round three.
    inherited = {}
    for source in previous:
        saved = json.loads((source / "run_settings.json").read_text(encoding="utf-8"))
        for number, path in saved.get("approved_trials", {}).items():
            if number in inherited and inherited[number] != path:
                raise ValueError("Previous rounds disagree about an approved trial binding")
            inherited[number] = path
    bindings = [f"{number}={path}" for number, path in inherited.items()] + list(bindings)
    for binding in bindings:
        try:
            number_text, path_text = str(binding).split("=", 1)
            number = int(number_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("approved_trial must be ROUND=PATH") from exc
        if number < 1 or number > len(previous) or not path_text:
            raise ValueError("approved_trial requires one distinct previous round per trial")
        trial = Path(path_text).resolve(strict=True)
        if number in approved:
            if approved[number] != trial:
                raise ValueError("A round cannot bind different approved trials")
            continue
        source = previous[number - 1]
        progress = _terminal_progress(source)
        if progress["state"] != "blocked_executability":
            raise ValueError("approved_trial requires a blocked_executability previous round")
        review = json.loads((source / "review_summary.json").read_text(encoding="utf-8"))
        if review.get("status") != "completed":
            raise ValueError("approved_trial requires a completed review")
        own = json.loads((source / "run_settings.json").read_text(encoding="utf-8"))
        proposal = json.loads((source / "proposal_summary.json").read_text(encoding="utf-8"))
        settings = json.loads((trial.parent.parent / "run_settings.json").read_text(encoding="utf-8"))
        request = json.loads((trial / "run_request.json").read_text(encoding="utf-8"))
        result = json.loads((trial / "task_result.json").read_text(encoding="utf-8"))
        final = _terminal_progress(trial.parent.parent)
        arm = next(a for a in own["arms"] if a["execution_mode"] == "team")
        local_trial = source / task_id / arm["arm_id"]
        if (local_trial / "task_result.json").exists():
            raise ValueError("Previous round already has a trial; cannot replace it with approved_trial")
        task = next(t for t in own["tasks"] if t["task_id"] == task_id)
        if (request.get("execution_mode") != "team" or request.get("fresh_start") is not True
                or request.get("policy_version") != arm["policy_version"]
                or request.get("task", {}).get("task_id") != task_id
                or request.get("task", {}).get("repo_id") != task["repository_id"]
                or trial.parent.name != task_id
                or type(request.get("repeat_index")) is not int or request["repeat_index"] < 1
                or result.get("status") not in {"completed", "failed"}):
            raise ValueError("Approved trial identity does not match its round")
        identity = {"task_id": task_id, "repo_id": task["repository_id"], "arm_id": trial.name,
                    "execution_mode": "team", "policy_version": arm["policy_version"],
                    "repeat_index": request["repeat_index"], "pair_id": request.get("pair_id")}
        if any(key in recorded and recorded[key] != expected
               for recorded in (request, result) for key, expected in identity.items()):
            raise ValueError("Approved trial result identity disagrees with request")
        for key in ("assignment_policy", "communication_policy"):
            if not isinstance(proposal.get(key), str) or request.get(key) != proposal[key]:
                raise ValueError("Approved trial policy texts must exactly match proposal_summary")
        if (not known_code_version(own.get("code_version"))
                or any(not isinstance(saved.get("qwen_version"), str)
                       or not saved["qwen_version"].strip()
                       or saved["qwen_version"].strip().lower() == "unknown" for saved in (own, settings))
                or any(settings.get(key) != own.get(key) for key in
                       ("inference", "provider", "evaluation", "runtime", "platform", "code_version", "qwen_version"))):
            raise ValueError("Approved trial runtime conditions do not match its round")
        selected = [a for a in settings.get("arms", []) if a.get("arm_id") == trial.name]
        task_matches = [t for t in settings.get("tasks", []) if t.get("task_id") == task_id]
        rows = [row for row in final.get("rows", [])
                if row.get("task_id") == task_id and row.get("arm_id") == trial.name]
        if (len(selected) != 1 or any(selected[0].get(k) != request[k] for k in ("execution_mode", "policy_version"))
                or len(task_matches) != 1 or any(task_matches[0].get(k) != task[k] for k in ("repository_id", "image"))
                or len(rows) != 1 or rows[0].get("inference_cleanup_confirmed") is not True
                or result.get("metrics", {}).get("inference_cleanup_confirmed") is not True):
            raise ValueError("Approved trial settings or completion records do not match")
        if trial in approved.values():
            raise ValueError("An approved trial may belong to only one round")
        approved[number] = trial
    return approved


def _review_coverage(items, execution_notes):
    # The reviewer must assess each quoted changed rule; unrelated supported
    # verdicts cannot make a partially reviewed proposal executable.
    assessment = executability_summary(items) if items else {"recorded": False}
    def rule_key(item):
        # Match the existing optimizer/reviewer quote validation semantics.
        return item["policy_part"], " ".join(item["rule_quote"].split())
    checked = {rule_key(item) for item in items or []}
    missing = [{"verdict": "unknown", "policy_part": note["policy_part"], "rule_quote": note["rule_quote"],
                "reason": "Changed rule has no corresponding executability assessment"}
               for note in execution_notes or [] if rule_key(note) not in checked]
    # Coverage is the controller's finding, not a verdict supplied by the reviewer.
    assessment["unassessed_notes"] = missing
    return assessment


def _round_trial(source, task_id, index, approved):
    settings = json.loads((source / "run_settings.json").read_text(encoding="utf-8"))
    arm = next(a for a in settings["arms"] if a["execution_mode"] == "team")
    return approved.get(index, source / task_id / arm["arm_id"])


def _cumulative_request(request, previous, task_id, approved):
    root = request.evidence.root
    directories = [root / "baseline_team"]
    history, results = [], list(request.batch_results)
    for index, source in enumerate(previous, 1):
        directory = root / f"previous_round_{index}"
        directory.mkdir(exist_ok=False)
        trial = _round_trial(source, task_id, index, approved)
        progress = _terminal_progress(source)
        item = {"state": progress["state"], "evidence_directory": directory.name}
        if index in approved:
            item["approved_trial"] = True
        if (trial / "task_result.json").is_file():
            projection = project_evidence(trial)
            write_json_atomic(directory / "projection.json", projection)
            item["score"] = projection["task_result"]["score"]
            results.append({"task_id": task_id, "score": item["score"],
                            "evidence_path": directory.name + "/projection.json", "evolution_round": index})
        for name in ("proposal_summary.json", "review_summary.json"):
            path = source / name
            if path.is_file():
                raw = json.loads(path.read_text())
                safe = {key: raw[key] for key in ("status", "assignment_policy", "communication_policy", "rationale",
                        "recommendation", "executability", "execution_notes") if key in raw}
                write_json_atomic(directory / name, safe)
        write_json_atomic(directory / "round.json", item)
        directories.append(directory)
        history.append(item)
    return replace(request, evidence=TrainingEvidence(root, directories), batch_results=tuple(results),
                   change_history=tuple(history) + request.change_history, history_total=len(history) + 1,
                   evidence_scope="cumulative", batch_index=len(previous) + 1)


def run_followup(profile_path: Path, project_root: Path, output: Path, *, source_run: Path,
                 qwen_package: Path, node_directory: Path, platform="linux/amd64", docker="docker",
                 repeat_index: int | None = None, evolution_round: int | None = None, previous_rounds=(), history_trials=(), approved_trials=()) -> dict:
    if type(repeat_index) is not int or repeat_index < 0:
        raise ValueError("Explicit nonnegative repeat_index is required for execution")
    prepared = prepare_followup(profile_path, project_root, source_run)
    plan, task = prepared.plan, prepared.study.task
    total_protocol = plan.get("selection_protocol") == TOTAL_MODE
    if total_protocol:
        if type(repeat_index) is not int or repeat_index < 1:
            raise ValueError("Explicit positive repeat_index is required for this protocol")
        if type(evolution_round) is not int or not 1 <= evolution_round <= 3:
            raise ValueError("Explicit evolution_round must be 1, 2 or 3")
        previous = _previous_rounds(previous_rounds, plan, evolution_round)
        approved = _approved_trials(approved_trials, previous, task.task_id)
        history_trials = [trial_record(path)["path"] for path in history_trials]
        plan.update(repeat_index=repeat_index, evolution_round=evolution_round, max_evolution_rounds=3,
                    previous_rounds=[str(path) for path in previous], history_trials=[str(path) for path in history_trials],
                    approved_trials={str(number): str(path) for number, path in approved.items()},
                    training_rollouts_reused=1 + sum(
                        (_round_trial(source, task.task_id, index, approved) / "task_result.json").is_file()
                        for index, source in enumerate(previous, 1)),
                    training_evidence_rounds=len(previous) + 1,
                    training_evidence=["baseline_team"] + [f"previous_round_{i}" for i in range(1, len(previous) + 1)],
                    interpretation=(f"Evolution round {evolution_round}: the selected baseline Team and {len(previous)} "
                                    "explicit previous rounds supply cumulative evidence. Finished trial evidence (scores may be unknown) is "
                                    "available; paused rounds retain their proposals and reviews, with separately approved "
                                    "dispatch trials only when explicitly bound. The candidate uses a different API time "
                                    "window from the baseline; not held-out validation, causal proof, or automatic adoption."))
    if not total_protocol and (previous_rounds or approved_trials or history_trials):
        raise ValueError("previous_rounds, approved_trials and history_trials require the total-score protocol")
    round_index = repeat_index
    if type(round_index) is not int or round_index < 0:
        raise ValueError("repeat_index must be a nonnegative integer")
    output = Path(output).resolve()
    if output == Path(source_run).resolve() or output.is_relative_to(Path(source_run).resolve()):
        raise ValueError("Follow-up output must not be inside the source run")
    output.mkdir(parents=True, exist_ok=False)
    (output / "profile.json").write_bytes(Path(profile_path).read_bytes())
    arms = plan["arms"]
    evolved = next(a for a in arms if a["study_role"] == "evolved")
    baseline = prepared.study.plan["arms"][1]
    metadata = {a["arm_id"]: {key: a[key] for key in
                ("arm_id", "study_role", "execution_mode", "policy_version")} for a in arms}
    for value in metadata.values():
        value.update(task_id=task.task_id, repo_id=task.repository_id)
    settings = {**plan, "qwen_package": str(Path(qwen_package).resolve()),
                "node_directory": str(Path(node_directory).resolve()), "platform": platform, "docker": docker,
                "code_version": (code_version(project_root)
                                 if Path(project_root).resolve() == Path(__file__).resolve().parents[1] else None), "qwen_version": native_version(qwen_package)}
    write_json_atomic(output / "run_settings.json", settings)
    state, reason = "preparing", None
    evolution = {"optimizer": "pending", "reviewer": "pending", "adopted": False}
    requests, launched, finished, results = {}, {}, {}, {}

    def snapshot():
        rows = [finished[a["arm_id"]] for a in arms if a["arm_id"] in finished]
        attention = [row for row in rows if row["inference_cleanup_confirmed"] is not True]
        return {"state": state, "reason": reason, "terminal": len(rows), "planned": len(arms),
                "launched": len(launched), "parallelism": plan["parallelism"], "rows": rows,
                "pending": [metadata[a["arm_id"]] for a in arms if a["arm_id"] not in launched],
                "running": [metadata[a["arm_id"]] for a in arms
                            if a["arm_id"] in launched and a["arm_id"] not in finished],
                "requires_user_action": state.startswith("blocked_"), "cleanup_attention": attention,
                "evolution": evolution, "automatic_restarts": False, "automatic_adoption": False,
                "heldout": False, "study_mode": plan["study_mode"]}

    def save():
        write_json_atomic(output / "progress.json", snapshot())

    def finish():
        save()
        report = {**snapshot(), "source_run": plan["source_run"],
                  "training_rollouts_reused": plan["training_rollouts_reused"], "comparison_baseline_references": plan["comparison_baseline_references"],
                  "interpretation": plan["interpretation"], "selection_advisory": None}
        if total_protocol:
            report["training_evidence_rounds"] = plan["training_evidence_rounds"]
        if evolved["arm_id"] in results:
            selected = (compare_trials(prepared.source_trial, output / task.task_id / evolved["arm_id"],
                                       history_paths=history_trials)
                        if total_protocol else select_candidate([prepared.baseline_result], [results[evolved["arm_id"]]]))
            report["selection_advisory"] = {**selected, "advisory_only": True, "adopted": False}
        write_json_atomic(output / "study_report.json", report)
        return report

    save()
    try:
        provider = load_provider(Path(plan["provider_config"]))
    except Exception as exc:
        state, reason = "blocked_provider", type(exc).__name__
        return finish()
    settings["provider"] = public_provider(provider)
    write_json_atomic(output / "run_settings.json", settings)
    state, evolution["optimizer"] = "optimizing", "running"
    save()
    try:
        harness = target_harness(plan["inference"], execution_mode="team", provider=provider,
                                 native_version=settings["qwen_version"], code_version=code_version_label(settings["code_version"]))
        evolution["executability_protocol"] = True
        base_request = _proposal_request(prepared.study, output, prepared.source_trial, target_harness=harness)
        proposal_request = FollowupProposalRequest(**{
            field.name: getattr(base_request, field.name) for field in fields(base_request)})
        proposal_request = replace(proposal_request, change_history=({
            "parent": f"P{baseline['policy_version']}", "new_candidate": evolved["policy_version"],
            "source": "one explicitly selected completed baseline Team; no Single evidence"},))
        if total_protocol:
            proposal_request = _cumulative_request(proposal_request, previous, task.task_id, approved)
        proposal = StrategyOptimizer(provider, output / "optimizer", max_tool_rounds=32,
                                     max_context_chars=180_000, working_memory=MEMORY,
                                     require_citations=True,
                                     transport_retries=provider.max_attempts - 1)(proposal_request)
        _validate_outer_result(proposal)
    except Exception as exc:
        evolution["optimizer"] = "failed"
        write_json_atomic(output / "proposal_summary.json", _failure(exc))
        state, reason = "blocked_proposal", "Candidate generation failed; no rollout or restart"
        return finish()
    policies = output / "policies"
    policies.mkdir(exist_ok=False)
    for name, value in (("assignment_evolved.md", proposal.assignment_policy),
                        ("communication_evolved.md", proposal.communication_policy),
                        ("rationale_evolved.md", proposal.rationale)):
        with (policies / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(value)
    evolution["optimizer"] = "completed"
    write_json_atomic(output / "proposal_summary.json", {
        **asdict(proposal), "status": "proposed_not_adopted", "adopted": False,
        "training_rollouts": plan["training_rollouts_reused"],
        **({"training_evidence_rounds": plan["training_evidence_rounds"]} if total_protocol else {}),
        "training_evidence": plan.get("training_evidence", "baseline_team"), "heldout_feedback_used": False})
    state, evolution["reviewer"] = "reviewing", "running"
    save()
    try:
        review = StrategyReviewer(provider, output / "reviewer", transport_retries=provider.max_attempts - 1)(
            StudyReviewRequest(proposal_request, proposal))
        if not isinstance(review, PolicyReview):
            raise ValueError("Reviewer did not return a valid PolicyReview")
        _validate_outer_result(review)
        review_summary = {**asdict(review), "status": "completed", "advisory_only": not total_protocol,
                          "quality_recommendation_advisory_only": True}
        evolution["reviewer"] = "completed"
    except Exception as exc:
        review_summary = {**_failure(exc), "advisory_only": True}
        evolution["reviewer"] = "failed"
    write_json_atomic(output / "review_summary.json", review_summary)
    evolution["review_recommendation"] = review_summary.get("recommendation")
    # Quality and executability are separate; the TOTAL protocol pauses unsupported/unknown dispatch.
    evolution["review_executability"] = (_review_coverage(review_summary.get("executability"), proposal.execution_notes)
                                       if total_protocol else executability_summary(review_summary.get("executability")))
    if evolution["reviewer"] != "completed":
        state, reason = "blocked_review", "Reviewer did not complete a valid review; no rollout or restart"
        return finish()
    if total_protocol:
        assessment = evolution["review_executability"]
        if assessment.get("unsupported_or_unknown"):
            reason = "Reviewer assessed unsupported/unknown executability; requires the user's dispatch decision"
            assessment["dispatch_reason"] = "reviewer_unsupported_or_unknown"
        elif assessment.get("unassessed_notes"):
            reason = "Execution notes were not assessed; requires the user's dispatch decision"
            assessment["dispatch_reason"] = "execution_notes_unassessed"
        elif not assessment.get("recorded"):
            reason = "No executability assessment was recorded; requires the user's dispatch decision"
            assessment["dispatch_reason"] = "executability_not_recorded"
        if "dispatch_reason" in assessment:
            state = "blocked_executability"
            assessment["dispatch"] = "paused_for_user"
            return finish()
        assessment["dispatch"] = "supported_review_complete"
    # Quality recommendation stays advisory; the current protocol pauses unknown executability.
    for arm in arms:
        key = arm["arm_id"]
        directory = output / task.task_id / key
        directory.mkdir(parents=True, exist_ok=False)
        assignment, communication = ((proposal.assignment_policy, proposal.communication_policy)
                                     if arm["execution_mode"] == "team" else ("", ""))
        request = RunRequest(
            task=TaskSpec(task.task_id, task.repository_id,
                          {"path": str(task.path), "instruction_path": str(task.instruction_path)}),
            split="qualitative", policy_version=arm["policy_version"],
            assignment_policy=assignment, communication_policy=communication,
            output_dir=directory, fresh_start=True, repeat_index=round_index,
            pair_id=f"{task.task_id}#r{round_index}" if repeat_index is not None else task.task_id, execution_mode=arm["execution_mode"])
        requests[key] = request
        write_json_atomic(directory / "run_request.json", {
            **asdict(request), "output_dir": str(directory), **metadata[key]})
    state = "running_evolved"
    save()
    inference = plan["inference"]
    options = dict(provider=provider, qwen_package=Path(settings["qwen_package"]),
                   node_directory=Path(settings["node_directory"]), platform=platform, docker=docker,
                   **runner_inference_options(inference), evaluator_cpus=plan["evaluation"]["docker_cpus"])

    def execute(request):
        try:
            return ProgramBenchRunner(**options)(request)
        except Exception as exc:
            return TaskResult(status="failed", error=type(exc).__name__, metrics={
                "runner_exception_type": type(exc).__name__, "inference_cleanup_confirmed": None})

    def run_wave():
        # The controller alone writes progress; each arm has its own runner and
        # fresh workspace. Never cancel the sibling after an arm fails.
        with ThreadPoolExecutor(max_workers=plan["parallelism"], thread_name_prefix="programbench-followup") as executor:
            futures = {}
            for arm in arms:
                key = arm["arm_id"]
                launched[key] = datetime.now(timezone.utc).isoformat()
                write_json_atomic(requests[key].output_dir / "launch.json", {
                    **metadata[key], "dispatched_at": launched[key], "fresh_start": True,
                    "automatic_restarts": False})
                save()
                futures[executor.submit(execute, requests[key])] = key
            for future in as_completed(futures):
                key, result = futures[future], future.result()
                results[key] = {**asdict(result), **metadata[key]}
                write_json_atomic(requests[key].output_dir / "task_result.json", results[key])
                finished[key] = {**metadata[key], "status": result.status, "score": result.score,
                                 "usage": result.usage, "cost_usd": result.cost_usd,
                                 "known_cost_usd": result.known_cost_usd, "elapsed_seconds": result.elapsed_seconds,
                                 "measurements": measurements(asdict(result)),
                                 "dispatched_at": launched[key], "finished_at": datetime.now(timezone.utc).isoformat(),
                                 "inference_cleanup_confirmed": result.metrics.get("inference_cleanup_confirmed"),
                                 "exception_type": result.metrics.get("runner_exception_type")}
                save()

    run_wave()
    if snapshot()["cleanup_attention"]:
        state, reason = "blocked_cleanup", "Follow-up inference cleanup is not confirmed"
    elif any(not _official_score(result) for result in results.values()):
        state, reason = "blocked_official_score", "Follow-up official score missing; no restart or invented zero"
    else:
        state, reason = "completed", None
    return finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--evolution-round", type=int, choices=(1, 2, 3))
    parser.add_argument("--previous-round", type=Path, action="append", default=[])
    parser.add_argument("--history-trial", type=Path, action="append", default=[])
    parser.add_argument("--approved-trial", action="append", default=[], metavar="ROUND=PATH",
                        help="Bind an already completed, user-approved dispatch trial to a paused previous round; never dispatches")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qwen-package", type=Path)
    parser.add_argument("--node-directory", type=Path)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args(argv)
    if args.run:
        if args.repeat_index is None:
            parser.error("--run requires explicit --repeat-index")
        if any(getattr(args, name) is None for name in ("output", "qwen_package", "node_directory")):
            parser.error("--run requires --output, --qwen-package and --node-directory")
        report = run_followup(args.profile, args.project_root, args.output, source_run=args.source_run,
                              qwen_package=args.qwen_package, node_directory=args.node_directory,
                              platform=args.platform, docker=args.docker,
                              repeat_index=args.repeat_index, evolution_round=args.evolution_round, previous_rounds=args.previous_round, history_trials=args.history_trial,
                              approved_trials=args.approved_trial)
    else:
        report = prepare_followup(args.profile, args.project_root, args.source_run).plan
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report.get("requires_user_action") else 0


if __name__ == "__main__":
    raise SystemExit(main())
