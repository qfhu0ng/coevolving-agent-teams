"""Sequential joint assignment/communication evolution with held-out evaluation.

New experiments review each candidate independently, replay the same training
batch in fresh environments, and select using paired official outcomes. Test
results never select a policy. The explicit legacy mode can resume old runs
whose proposals were adopted without replay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from agent_company.runtime_io import write_json_atomic, write_text_atomic
from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.efficiency import compact_efficiency, efficiency_report
from agent_company.evolution.recovery import experiment_session, prepare_task, previous_attempts, read
from agent_company.evolution.selection import (
    SELECTION_RULE_VERSION, SelectionConfig, select_candidate, selection_objective,
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    repo_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRequest:
    task: TaskSpec
    split: str
    policy_version: int
    communication_policy: str
    assignment_policy: str
    output_dir: Path
    fresh_start: bool = True
    repeat_index: int = 0
    pair_id: str | None = None
    seed: int | None = None
    execution_mode: str = "team"
    # New role-package comparisons may select a Team mode per arm. Legacy
    # requests keep the runner's shared mode and carry no package.
    role_policy_mode: str | None = None
    policy_package: dict | None = None

    def __post_init__(self):
        _validate_execution_mode(self.execution_mode)


@dataclass
class TaskResult:
    # "completed" includes evaluated task failures (e.g. a genuine score of 0).
    # Execution status and official score are independent: "failed" may retain
    # a completed verifier's reward. Missing evaluation must never become a 0.
    status: str
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    elapsed_seconds: float | None = None
    error: str | None = None
    trace_files: list[str] = field(default_factory=list)
    known_cost_usd: float | None = None


EXECUTABILITY_INSTRUCTION = (
    "Treat target_harness as the execution boundary for the candidate: it describes the configuration and "
    "implementation the next run will use, not past behaviour. Historical evidence supports improvement "
    "hypotheses; it does not guarantee a capability. For every new or changed mechanism, and for any known "
    "conflict that remains in the full A/C, give an execution note: who executes it, how that actor observes "
    "the trigger, which existing action it uses, the timing tolerance and fallback, and the cost and "
    "limitations, citing target_harness fields by JSON pointer. Do not present an action the target harness "
    "does not provide as executable; list ideas that need a runtime change separately in the rationale as "
    "unmet dependencies. Tolerances and fallbacks that change behaviour belong in the A/C text itself: the "
    "coding agents never see the notes. Keeping A/C unchanged is allowed; then submit an empty list and say so.")
EXECUTABILITY_REVIEW_INSTRUCTION = (
    "Judge executability separately from your recommendation. Read incumbent.target_harness first, then the "
    "complete candidate; do not limit yourself to the optimizer's execution_notes, which are claims, not proof. "
    "Look for unstated prerequisites (exact wake-ups at arbitrary times, workers creating successors, instant "
    "message reading, interrupting a running tool, zero commits meaning zero delivery). For each rule you "
    "assess, check that the actor has the action, can observe the trigger and when, whether timing is supported "
    "by existing events or merely conditional, whether the fallback is real and avoids busy waiting, endless "
    "resends or ending the run early, and whether required modes are active. Verdicts: supported, "
    "supported_with_conditions (conditions must be stated in the A/C), unsupported (a known missing or disabled "
    "capability) or unknown (the inputs cannot settle it). Executable is not effective, and supported does not "
    "mean the mechanism has run. For every entry in candidate.execution_notes, include at least one "
    "assessment whose rule_quote copies that note's rule_quote exactly; you may also assess other rules.")


@dataclass(frozen=True)
class ProposalRequest:
    current_version: int
    communication_policy: str
    assignment_policy: str
    batch_index: int
    batch_results: tuple[dict[str, Any], ...]
    change_history: tuple[dict[str, Any], ...]
    history_total: int
    evidence: TrainingEvidence
    selection_mode: str = "review_replay"
    candidate_index: int = 1
    evidence_scope: str = "current_batch"
    selection_config: SelectionConfig = field(default_factory=SelectionConfig)
    # Opt-in executability protocol (docs/stage1_optimizer_harness_feasibility_plan_20261012.md):
    # the harness the candidate will run under. None keeps the earlier request unchanged.
    target_harness: dict[str, Any] | None = None

    @property
    def executability_protocol(self) -> bool:
        return self.target_harness is not None

    @property
    def storage_key(self) -> str:
        base = f"batch_{self.batch_index:04d}"
        return base if self.candidate_index == 1 else f"{base}_candidate_{self.candidate_index:04d}"

    def as_prompt_data(self) -> dict[str, Any]:
        return {"current_version": self.current_version,
                "communication_policy": self.communication_policy,
                "assignment_policy": self.assignment_policy,
                "batch_index": self.batch_index,
                "candidate_index": self.candidate_index,
                "batch_results": self.batch_results,
                "change_history": self.change_history,
                "history_total": self.history_total,
                "selection_mode": self.selection_mode,
                "selection_objective": selection_objective(self.selection_config)
                if self.selection_mode == "review_replay" else {"mode": "legacy_no_replay"},
                "evidence_scope": self.evidence_scope,
                "evidence_files": self.evidence.overview(),
                "instruction": "Jointly improve assignment_policy and communication_policy; "
                               "submit the full text of both as one policy version. Read relevant training "
                               "evidence with the bounded evidence tools as needed. "
                               "Different batches are different tasks; do not treat their "
                               "scores as a paired policy comparison. "
                               + ("The candidate will be independently reviewed and replayed on this "
                                  "training batch before selection; submission is not adoption. "
                                  "Replay is training evidence, not held-out validation."
                                  if self.selection_mode == "review_replay" else
                                  "Legacy protocol: valid submissions are adopted without replay."),
                **({"target_harness": self.target_harness, "executability_protocol": EXECUTABILITY_INSTRUCTION}
                   if self.executability_protocol else {})}


@dataclass
class PolicyProposal:
    communication_policy: str
    assignment_policy: str
    rationale: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    elapsed_seconds: float | None = None
    known_cost_usd: float | None = None
    # Opt-in optimizer protocols only: coverage shown, verified citations and
    # unsupported rationale identifiers. Empty for the default protocol.
    reading_audit: dict[str, Any] = field(default_factory=dict)
    # Executability protocol only: how each changed rule executes on the target harness.
    # None means not recorded (another protocol or an older record), not "no notes".
    execution_notes: list[dict[str, Any]] | None = None


def optimizer_reading_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """What the reviewer is told about the optimizer's reading of the evidence."""
    if not audit:
        # An empty audit means none was recorded, not that nothing was read.
        return {"recorded": False,
                "note": "Not recorded: the optimizer ran without working_memory or require_citations, "
                        "so which evidence it was shown is unknown."}
    return {"recorded": True,
            "note": "Which training evidence was shown to the optimizer and which quotes it cited. "
                    "Use it to find claims that rest on evidence it did not read. Being shown text is "
                    "not the same as using it correctly; check the evidence yourself.",
            "audit": audit}


@dataclass(frozen=True)
class ReviewRequest:
    proposal_request: ProposalRequest
    candidate: PolicyProposal
    replay_task_count: int | None = None
    replay_tasks: tuple[dict[str, str], ...] = ()

    @property
    def evidence(self) -> TrainingEvidence:
        return self.proposal_request.evidence

    @property
    def batch_index(self) -> int:
        return self.proposal_request.batch_index

    @property
    def candidate_index(self) -> int:
        return self.proposal_request.candidate_index

    @property
    def storage_key(self) -> str:
        return self.proposal_request.storage_key

    def as_prompt_data(self) -> dict[str, Any]:
        candidate = {name: getattr(self.candidate, name) for name in
                     ("assignment_policy", "communication_policy", "rationale")}
        if self.proposal_request.executability_protocol:
            notes = self.candidate.execution_notes
            candidate["execution_notes"] = notes if notes is not None else "not_recorded"
        data = {"incumbent": self.proposal_request.as_prompt_data(),
                "candidate": candidate,
                "instruction": "Independently compare both full policy pairs using only available "
                               "training evidence. Cite evidence and identify risks. Your recommendation "
                               "is advisory; only paired official replay outcomes select the candidate. "
                               "Do not request held-out data or confuse plausible reasoning with a gain.",
                "optimizer_reading_audit": optimizer_reading_audit(self.candidate.reading_audit)}
        if self.proposal_request.executability_protocol:
            data["executability_review"] = EXECUTABILITY_REVIEW_INSTRUCTION
        if self.replay_task_count is not None:
            data["replay_selection"] = {
                "task_count": self.replay_task_count,
                "fixed_tasks": self.replay_tasks,
                "instruction": (
                    "This batch's replay tasks are already selected; use this same subset. Do not reselect."
                    if self.replay_tasks else
                    "Choose exactly task_count distinct tasks from incumbent.batch_results using task_id "
                    "and repo_id. Explain how they test the proposed change and possible regressions, "
                    "not which tasks make the candidate easiest to approve. Choose before any replay. "
                    "Prefer tasks with valid incumbent official scores for pairing; flag missing scores "
                    "instead of inventing them. "
                    "All candidates in this batch will use this same subset; no held-out or other-batch tasks."
                ),
            }
        return data


@dataclass
class PolicyReview:
    recommendation: str
    rationale: str
    evidence_refs: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    known_cost_usd: float | None = None
    # Executability protocol only; None means not recorded, not "found no problem".
    executability: list[dict[str, Any]] | None = None
    elapsed_seconds: float | None = None
    # agent_rules_v1 M generation only: the parts checked for task-specific content and the
    # findings; parts not checked stay unknown. None means not recorded (plan v4 12, item 5).
    task_specificity: dict[str, Any] | None = None
    replay_tasks: list[dict[str, str]] = field(default_factory=list)
    replay_rationale: str = ""


def _reviewer_replay_subset(review: PolicyReview | None, items, count: int):
    """Resolve the first reviewed candidate's subset once; later candidates reuse it.

    The durable review result is sufficient to reconstruct the same choice on
    resume. Invalid or failed selection leaves the candidate unevaluated; it
    must not silently change reviewer selection into manifest-order selection.
    """
    reason = "Reviewer failed or did not submit a replay selection"
    if review is not None:
        try:
            if not isinstance(review.replay_tasks, list) or len(review.replay_tasks) != count:
                raise ValueError(f"Reviewer must select exactly {count} tasks")
            identities = []
            for task in review.replay_tasks:
                if (not isinstance(task, dict) or set(task) != {"task_id", "repo_id"}
                        or any(not isinstance(value, str) or not value.strip() for value in task.values())):
                    raise ValueError("Replay tasks must contain only nonempty task_id and repo_id")
                identities.append((task["repo_id"], task["task_id"]))
            if len(set(identities)) != count:
                raise ValueError("Reviewer selected duplicate tasks")
            available = {(task.repo_id, task.task_id) for _, task in items}
            if not set(identities) <= available:
                raise ValueError("Reviewer selected a task outside the current batch")
            if not isinstance(review.replay_rationale, str) or not review.replay_rationale.strip():
                raise ValueError("Reviewer must explain its replay selection")
            selected = [(index, task) for index, task in items
                        if (task.repo_id, task.task_id) in set(identities)]
            return selected, {"source": "reviewer", "rationale": review.replay_rationale}
        except ValueError as exc:
            reason = str(exc)
    return [], {"source": "review_unavailable", "rationale": reason}


def _validate_outer_result(result: PolicyProposal | PolicyReview) -> None:
    for name in ("cost_usd", "known_cost_usd", "elapsed_seconds"):
        value = getattr(result, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number or None")
    if not isinstance(result.usage, dict):
        raise ValueError("usage must be a dictionary")
    if isinstance(result, PolicyProposal):
        for name in ("assignment_policy", "communication_policy"):
            policy = getattr(result, name)
            if not isinstance(policy, str) or not policy.strip() or len(policy) > 24_000:
                raise ValueError(f"Optimizer returned an empty or oversized {name}")
        if not isinstance(result.rationale, str):
            raise ValueError("Proposal rationale must be text")
    else:
        if result.recommendation not in ("prefer_candidate", "prefer_incumbent", "uncertain"):
            raise ValueError("Invalid reviewer recommendation")
        if not isinstance(result.rationale, str) or not result.rationale.strip():
            raise ValueError("Reviewer rationale must be nonempty text")
        for name in ("evidence_refs", "risks"):
            items = getattr(result, name)
            if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                raise ValueError(f"Reviewer {name} must be a list of strings")


def _outer_call(request, callback, result_type, directory: Path, *, resume: bool,
                from_version: int, label: str) -> tuple[dict, Any]:
    """Persist submission separately from adoption; interrupted calls are never reissued.

    Existing atomic result files and each model loop's recover() are sufficient:
    no extra identity/hash layer is needed beyond the experiment settings and
    batch index. A result-file write failure propagates instead of discarding a
    successfully submitted model result and quietly continuing.
    """
    result_path = directory / "result.json"
    if resume and result_path.exists():
        saved = read(result_path)
        if saved["status"] == "failed":
            return saved, None
        if saved["status"] != "completed":
            raise ValueError(f"Invalid {label} checkpoint status")
        result = result_type(**{key: saved[key] for key in result_type.__dataclass_fields__ if key in saved})
        _validate_outer_result(result)
        return saved, result
    pending = resume and (directory / "request.json").exists()
    directory.mkdir(parents=True, exist_ok=resume)
    if not pending:
        write_json_atomic(directory / "request.json", request.as_prompt_data())
    started = time.monotonic()
    record = {"batch_index": request.batch_index, "from_version": from_version,
              "candidate_index": request.candidate_index}
    result = None
    try:
        if pending:
            recover = getattr(callback, "recover", None)
            if recover is None:
                raise RuntimeError(f"{label} was interrupted; no new model call is issued")
            returned = recover(request)
        else:
            returned = callback(request)
        data = asdict(returned) if isinstance(returned, result_type) else dict(returned)
        # Preserve spent usage even when the callback submits malformed fields.
        record.update({key: value for key, value in data.items() if key in result_type.__dataclass_fields__})
        result = result_type(**data)
        _validate_outer_result(result)
        record.update(status="completed", **asdict(result))
    except Exception as exc:
        result = None
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      usage=getattr(exc, "usage", record.get("usage", {})),
                      cost_usd=getattr(exc, "cost_usd", record.get("cost_usd")),
                      known_cost_usd=getattr(exc, "known_cost_usd", record.get("known_cost_usd")),
                      elapsed_seconds=getattr(exc, "elapsed_seconds", record.get("elapsed_seconds")))
        if getattr(exc, "output_dir", None) is not None:
            record["output_dir"] = str(exc.output_dir)
        # A malformed callback must not poison aggregate arithmetic or turn an
        # advisory review error into an unrelated controller crash. Keep valid
        # accounting, but never sum strings, negative charges or NaN as money.
        for name in ("cost_usd", "known_cost_usd", "elapsed_seconds"):
            value = record.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                record[name] = None
        if not isinstance(record.get("usage"), dict):
            record["usage"] = {}
    record["controller_elapsed_seconds"] = time.monotonic() - started
    if record.get("elapsed_seconds") is None:
        record["elapsed_seconds"] = record["controller_elapsed_seconds"]
    if "task_specificity" in record and record["task_specificity"] is None:
        # Only agent_rules_v1 reviews record it; earlier records and feedback stay as they were (M5 review minor 13).
        del record["task_specificity"]
    write_json_atomic(result_path, record)
    return record, result


def _all_costs(records: Sequence[dict]) -> dict:
    known = [row.get("known_cost_usd") if row.get("known_cost_usd") is not None
             else row.get("cost_usd") for row in records]
    unknown = sum(row.get("cost_usd") is None for row in records)
    return {"cost_usd": None if unknown else sum(row["cost_usd"] for row in records),
            "known_cost_usd": sum(value for value in known if value is not None)
                              if any(value is not None for value in known) else (None if records else 0.0),
            "records": len(records), "records_with_unknown_cost": unknown}


def _compact(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 1200 else value[:1200] + "… [truncated]"
    if depth >= 3 and isinstance(value, (dict, list, tuple)):
        return "[See raw task_result.json for nested data]"
    if isinstance(value, dict):
        items = list(value.items())
        result = {str(key): _compact(item, depth + 1) for key, item in items[:24]}
        if len(items) > 24:
            result["_omitted_keys"] = len(items) - 24
        return result
    if isinstance(value, (list, tuple)):
        result = [_compact(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            result.append(f"[{len(value) - 12} additional items in raw result]")
        return result
    return value


def _component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip(".")[:80] or "task"


# Metric values left out of optimizer summaries. inference_image is run provenance
# (an image digest), not policy evidence; the oversize key list also omits that name.
_OPTIMIZER_HIDDEN_METRICS = frozenset({"team", "single", "verifier", "task", "resources", "inference_image"})


def _task_summary(record: dict, train_root: Path) -> dict:
    summary = {key: _compact(record.get(key)) for key in
               ("task_id", "repo_id", "split", "policy_version", "execution_mode", "policy_applied", "status", "score",
                "elapsed_seconds", "error", "usage")}
    metrics = record.get("metrics", {})
    runtime = metrics.get(record.get("execution_mode", "team"), {})
    native = runtime.get("native_result") or {}
    final = native.get("result") or {}
    verifier = metrics.get("verifier") or {}
    summary["runtime_outcome"] = {
        "execution": _compact(runtime.get("execution")),
        "terminal_reason": _compact(runtime.get("terminal_reason", runtime.get("stop_reason"))),
        "native_explicit_error": native.get("explicit_error"),
        "native_missing_result": native.get("missing_result"),
        "native_subtype": final.get("subtype"),
        "last_text": _compact(final.get("result")),
    }
    summary["official_result"] = {"status": verifier.get("status"),
                                  "raw_score": verifier.get("reward"),
                                  "details": _compact(verifier.get("reward_details"))}
    summary["agent_elapsed_seconds"] = (runtime.get("execution") or {}).get("elapsed_seconds")
    summary["wall_time_budget_seconds"] = (metrics.get("task") or {}).get("agent_timeout_sec")
    summary["efficiency"] = dict(record.get("efficiency") or {
        "available": False, "note": "No generated efficiency report for this attempt; inspect its traces."})
    if summary["efficiency"].get("report_file") == "efficiency.json":
        summary["efficiency"]["report_file"] = str(
            (Path(record["output_dir"]) / "efficiency.json").relative_to(train_root))
    summary["current_attempt"] = {
        "usage_scope": "current_attempt_only", "usage_complete": _paired_measurements(record)["usage_complete"],
        "cost_usd": record.get("current_attempt_cost_usd", record.get("cost_usd")),
        "known_cost_usd": record.get("current_attempt_known_cost_usd", (
            float(runtime["known_estimated_cost"]) if runtime.get("known_estimated_cost") is not None
            and runtime.get("currency") == "USD"
            else record.get("known_cost_usd") if not record.get("interrupted_attempts") else None)),
    }
    # Retain small non-runtime metrics for generic callers, without dumping the
    # task spec, all worker names, native usage duplicates and resource settings.
    summary["metrics"] = _compact({key: value for key, value in metrics.items()
                                   if key not in _OPTIMIZER_HIDDEN_METRICS})
    if "provider_incidents" in runtime:
        # Keep request failures visible even when nested metrics are compacted.
        # This describes observed incidents, not proof of recovery or solution quality.
        summary["provider_incidents"] = _compact(runtime["provider_incidents"])
    summary["result_file"] = str((Path(record["output_dir"]) / "task_result.json").relative_to(train_root))
    output = Path(record["output_dir"])
    candidates = [output / "runner_result.json", output / "execution.json", output / "delivery.patch"]
    candidates += [output / record.get("execution_mode", "team") / name
                   for name in ("qwen_stdout.jsonl", "qwen_stderr.txt", "relay/events.jsonl")]
    summary["trace_files"] = [str(path.relative_to(train_root)) for path in candidates if path.is_file()]
    previous = record.get("interrupted_attempts", [])
    summary["interrupted_attempt_count"] = len(previous)
    summary["interrupted_attempts"] = [{key: _compact(row.get(key)) for key in
                                        ("output_dir", "status", "usage", "cost_usd", "known_cost_usd",
                                         "unsettled_attempts")} for row in previous[-4:]]
    # Nested metric trees can remain large despite per-field clipping. Keep a
    # strict compact overview and leave the detailed evidence available by page.
    if len(json.dumps(summary, ensure_ascii=False)) > 6000:
        for key in ("metrics", "usage"):
            names = [name for name in record.get(key, {})
                     if not (key == "metrics" and name == "inference_image")]
            summary[key] = {"detail": "Read result_file for full data",
                            "keys": [str(name)[:80] for name in names][:16]}
        # Large role sets should not crowd out task outcomes or the opportunity
        # to read evidence. Keep the report pointer and coverage, not all roles.
        if len(json.dumps(summary, ensure_ascii=False)) > 6000 and summary["efficiency"].get("by_role"):
            projected = summary["efficiency"]
            projected["roles_omitted"] += len(projected["by_role"])
            projected["by_role"] = []
    return summary


def _validate_execution_mode(value: str, name: str = "execution_mode") -> None:
    if value not in ("single", "team"):
        raise ValueError(f"{name} must be 'single' or 'team'")


def _validate_tasks(train: Sequence[TaskSpec], test: Sequence[TaskSpec], mode: str) -> None:
    if mode not in ("repo", "task"):
        raise ValueError("split_mode must be 'repo' or 'task'")
    if not train or not test:
        raise ValueError("Both train_tasks and test_tasks must be nonempty")
    for tasks in (train, test):
        identities = [(task.repo_id, task.task_id) for task in tasks]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate task identities within a split")
        if any(not task.task_id or not task.repo_id for task in tasks):
            raise ValueError("task_id and manifest-sourced repo_id are required")
    train_ids = {(task.repo_id, task.task_id) for task in train}
    test_ids = {(task.repo_id, task.task_id) for task in test}
    if train_ids & test_ids:
        raise ValueError("A task cannot occur in both train and test")
    if mode == "repo" and {task.repo_id for task in train} & {task.repo_id for task in test}:
        raise ValueError("Repository split contains a repository shared by train and test")


def _validate_result(result: TaskResult) -> None:
    if result.status not in ("completed", "failed"):
        raise ValueError("TaskResult.status must be completed or failed")
    for name in ("score", "cost_usd", "known_cost_usd", "elapsed_seconds"):
        value = getattr(result, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError(f"TaskResult.{name} must be a finite number or None")


def _run(request: RunRequest, callback: Callable[[RunRequest], TaskResult | Mapping], *,
         resume=False, retry_interrupted=False) -> dict:
    cached = prepare_task(request, callback, resume=resume, retry_interrupted=retry_interrupted)
    if cached is not None:
        return cached
    request.output_dir.mkdir(parents=True, exist_ok=False)
    serialized = asdict(request)
    serialized["output_dir"] = str(request.output_dir)
    write_json_atomic(request.output_dir / "run_request.json", serialized)
    started = time.monotonic()
    try:
        returned = callback(request)
        result = returned if isinstance(returned, TaskResult) else TaskResult(**returned)
        _validate_result(result)
    except Exception as exc:
        result = TaskResult(status="failed", error=f"{type(exc).__name__}: {exc}",
                            usage=getattr(exc, "usage", {}),
                            cost_usd=getattr(exc, "cost_usd", None),
                            elapsed_seconds=getattr(exc, "elapsed_seconds", None),
                            known_cost_usd=getattr(exc, "known_cost_usd", None))
    measured = time.monotonic() - started
    if result.elapsed_seconds is None:
        result.elapsed_seconds = measured
    return _record_result(request, result, measured)


def _record_result(request: RunRequest, result: TaskResult, measured: float | None) -> dict:
    record = {"task_id": request.task.task_id, "repo_id": request.task.repo_id,
              "split": request.split, "policy_version": request.policy_version,
              "execution_mode": request.execution_mode, "policy_applied": request.execution_mode == "team",
              "repeat_index": request.repeat_index, "pair_id": request.pair_id,
              "requested_seed": request.seed, "fresh_start_requested": True,
              "output_dir": str(request.output_dir), "controller_elapsed_seconds": measured,
              **asdict(result)}
    interrupted = previous_attempts(request.output_dir)
    if interrupted:
        record["interrupted_attempts"] = interrupted
        record["current_attempt_cost_usd"] = result.cost_usd
        known = result.known_cost_usd if result.known_cost_usd is not None else result.cost_usd
        record["current_attempt_known_cost_usd"] = known
        subtotals = [row["known_cost_usd"] for row in interrupted] + [known]
        record["known_cost_usd"] = (sum(v for v in subtotals if v is not None)
                                    if any(v is not None for v in subtotals) else None)
        record["cost_usd"] = None
    # Diagnostics are generated only for new attempts. Cached results and old
    # artifacts are never rewritten or enriched during resume. A diagnostic
    # failure must not relabel a valid official result as a task failure.
    try:
        report = efficiency_report(record, request.output_dir)
        write_json_atomic(request.output_dir / "efficiency.json", report)
        record["efficiency"] = compact_efficiency(report)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        record["efficiency"] = {"available": False, "error_type": type(exc).__name__}
    write_json_atomic(request.output_dir / "task_result.json", record)
    return record


_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens")


def _token_measurements(record: dict) -> dict:
    """Known current-attempt usage; incomplete counters are never a complete zero."""
    usage = record.get("usage") or {}
    metrics = record.get("metrics") or {}
    runtime = metrics.get(record.get("execution_mode")) or {}
    tokens = {key: value if isinstance(value := usage.get(key), int)
              and not isinstance(value, bool) and value >= 0 else None for key in _TOKEN_FIELDS}
    complete = (all(value is not None for value in tokens.values())
                and not usage.get("unknown_usage_attempts")
                and not usage.get("calls_missing_usage")
                and runtime.get("usage_complete") is not False
                and record.get("status") != "interrupted")
    return {**tokens, "usage_complete": complete}


def _token_totals(records: Sequence[dict]) -> dict:
    measurements = [_token_measurements(row) for row in records]
    unknown = sum(not row["usage_complete"] for row in measurements)
    return {"known_tokens": {key: sum(row[key] or 0 for row in measurements) for key in _TOKEN_FIELDS},
            "records_with_unknown_usage": unknown, "usage_complete": unknown == 0}


def _paired_measurements(record: dict) -> dict:
    metrics = record.get("metrics") or {}
    runtime = metrics.get(record.get("execution_mode")) or {}
    return {**_token_measurements(record),
            "agent_elapsed_seconds": (runtime.get("execution") or {}).get("elapsed_seconds"),
            "wall_time_budget_seconds": (metrics.get("task") or {}).get("agent_timeout_sec"),
            "task_elapsed_seconds": record.get("elapsed_seconds")}


def _aggregate(records: Sequence[dict]) -> dict:
    scores = [row["score"] for row in records
              if row["status"] in {"completed", "failed"} and row.get("score") is not None]
    costs = [row["cost_usd"] for row in records if row.get("cost_usd") is not None]
    known_costs = [row.get("known_cost_usd") if row.get("known_cost_usd") is not None
                   else row.get("cost_usd") for row in records]
    historical = [attempt for row in records for attempt in row.get("interrupted_attempts", [])]
    return {"runs": len(records), "completed": sum(row["status"] == "completed" for row in records),
            "failed": sum(row["status"] == "failed" for row in records),
            "interrupted_attempts": sum(len(row.get("interrupted_attempts", [])) for row in records),
            "runs_with_retries": sum(bool(row.get("interrupted_attempts")) for row in records),
            "scored_runs": len(scores), "mean_observed_score": sum(scores) / len(scores) if scores else None,
            "scored_failed_runs": sum(row["status"] == "failed" and row.get("score") is not None
                                      for row in records),
            "unscored_runs": len(records) - len(scores),
            "known_cost_usd": sum(value for value in known_costs if value is not None),
            "runs_with_unknown_cost": len(records) - len(costs),
            "elapsed_seconds": sum(row.get("elapsed_seconds") or 0 for row in records),
            **_token_totals(records), "token_usage_scope": "current_attempt_only",
            "interrupted_token_usage": {"attempts": len(historical), **_token_totals(historical)},
            "note": "Official scores are included independently of execution status. Only unscored runs "
                    "are excluded from observed-score mean; coverage and execution failures are reported separately."}


def run_stage1(*, train_tasks: Sequence[TaskSpec], test_tasks: Sequence[TaskSpec],
               initial_communication: str, assignment_policy: str,
               run_task: Callable[[RunRequest], TaskResult | Mapping],
               propose_policy: Callable[[ProposalRequest], PolicyProposal | Mapping],
               output_dir: Path, batch_size: int = 5, split_mode: str = "repo",
               test_repetitions: int = 1, seed: int | None = None,
               resume: bool = False, retry_interrupted: bool = False,
               runtime_config: dict | None = None, baseline_mode: str = "single",
               review_policy: Callable[[ReviewRequest], PolicyReview | Mapping] | None = None,
               selection_mode: str = "review_replay", selection_config: SelectionConfig | None = None,
               candidates_per_batch: int = 1, replay_task_count: int | None = None,
               evidence_scope: str | None = None, replay_selection: str = "first_n") -> dict:
    _validate_execution_mode(baseline_mode, "baseline_mode")
    _validate_tasks(train_tasks, test_tasks, split_mode)
    if selection_mode not in ("review_replay", "legacy_no_replay"):
        raise ValueError("selection_mode must be review_replay or legacy_no_replay")
    if replay_selection not in ("first_n", "reviewer"):
        raise ValueError("replay_selection must be first_n or reviewer")
    scope = evidence_scope or ("current_batch" if selection_mode == "review_replay" else "cumulative")
    if scope not in ("current_batch", "cumulative"):
        raise ValueError("evidence_scope must be current_batch or cumulative")
    if selection_mode == "review_replay" and not callable(review_policy):
        raise ValueError("review_replay requires an independent review_policy callback")
    for name, value in (("batch_size", batch_size), ("candidates_per_batch", candidates_per_batch),
                        ("test_repetitions", test_repetitions), ("replay_task_count", replay_task_count)):
        if name == "replay_task_count" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if selection_mode == "legacy_no_replay" and (
        candidates_per_batch != 1 or replay_task_count is not None or selection_config is not None
        or review_policy is not None or scope != "cumulative" or replay_selection != "first_n"
    ):
        raise ValueError("Legacy mode cannot use reviewer, candidate/replay counts or selection configuration")
    rule = selection_config if selection_config is not None else SelectionConfig()
    if not isinstance(rule, SelectionConfig):
        raise ValueError("selection_config must be SelectionConfig or None")
    if retry_interrupted and not resume:
        raise ValueError("retry_interrupted requires resume")
    settings = {"mode": "evolution", "evolved_strategies": ["assignment", "communication"],
                "baseline_mode": baseline_mode, "training_execution_mode": "team",
                "train": [asdict(t) for t in train_tasks],
                "test": [asdict(t) for t in test_tasks], "batch_size": batch_size,
                "split_mode": split_mode, "test_repetitions": test_repetitions,
                "seed": seed, "assignment": assignment_policy,
                "communication": initial_communication, "runtime": runtime_config}
    # Explicit legacy mode retains the old settings shape for existing runs.
    # New-mode settings prevent resuming a no-replay run as a replay experiment.
    if selection_mode == "review_replay":
        settings.update(selection_mode=selection_mode, selection_config=asdict(rule),
                        candidates_per_batch=candidates_per_batch, replay_task_count=replay_task_count,
                        evidence_scope=scope, evolution_protocol_version=SELECTION_RULE_VERSION)
        # Omission retains compatibility with existing first-N run settings.
        if replay_selection != "first_n":
            settings["replay_selection"] = replay_selection
    with experiment_session(output_dir, settings, resume=resume) as root:
        return _run_stage1(train_tasks=train_tasks, test_tasks=test_tasks,
                           initial_communication=initial_communication, assignment_policy=assignment_policy,
                           run_task=run_task, propose_policy=propose_policy, output_dir=root,
                           batch_size=batch_size, split_mode=split_mode,
                           test_repetitions=test_repetitions, seed=seed,
                           resume=resume, retry_interrupted=retry_interrupted, baseline_mode=baseline_mode,
                           review_policy=review_policy, selection_mode=selection_mode, selection_config=rule,
                           candidates_per_batch=candidates_per_batch, replay_task_count=replay_task_count,
                           evidence_scope=scope, replay_selection=replay_selection)


def run_baseline(*, tasks: Sequence[TaskSpec], communication_policy: str = "", assignment_policy: str = "",
                 run_task: Callable, output_dir: Path, repetitions: int = 1, seed: int | None = None,
                 resume: bool = False, retry_interrupted: bool = False,
                 runtime_config: dict | None = None, execution_mode: str = "single") -> dict:
    """Evaluate single agents or a fixed-policy team, without constructing an optimizer."""
    _validate_execution_mode(execution_mode)
    if not tasks or len({(t.repo_id, t.task_id) for t in tasks}) != len(tasks):
        raise ValueError("Baseline requires nonempty, unique task identities")
    if any(not t.repo_id or not t.task_id for t in tasks):
        raise ValueError("task_id and repo_id are required")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")
    if execution_mode == "single":
        communication_policy, assignment_policy = "", ""
    elif any(not isinstance(policy, str) or not policy.strip()
             for policy in (communication_policy, assignment_policy)):
        raise ValueError("Both policies are required")
    if retry_interrupted and not resume:
        raise ValueError("retry_interrupted requires resume")
    settings = {"mode": "baseline", "tasks": [asdict(t) for t in tasks],
                "execution_mode": execution_mode,
                "communication": communication_policy, "assignment": assignment_policy,
                "repetitions": repetitions, "seed": seed, "runtime": runtime_config}
    with experiment_session(output_dir, settings, resume=resume) as root:
        records = []
        for index, task in enumerate(tasks, 1):
            for repeat in range(repetitions):
                directory = root / "tasks" / f"task_{index:03d}_{_component(task.task_id)}_repeat_{repeat:03d}"
                request = RunRequest(task, "baseline", 0, communication_policy, assignment_policy,
                                     directory, repeat_index=repeat,
                                     seed=None if seed is None else seed + repeat,
                                     execution_mode=execution_mode)
                records.append(_run(request, run_task, resume=resume,
                                    retry_interrupted=retry_interrupted))
                write_json_atomic(root / "progress.json", {"completed_slots": len(records),
                                  "total_slots": len(tasks) * repetitions,
                                  "execution_mode": execution_mode, "policy_applied": execution_mode == "team",
                                  "results": _aggregate(records)})
        summary = {"status": "completed_with_failures" if any(r["status"] == "failed" for r in records)
                   else "completed", "mode": "baseline", "execution_mode": execution_mode,
                   "policy_applied": execution_mode == "team", "results": _aggregate(records),
                   "task_runs": records, "optimizer_invoked": False}
        write_json_atomic(root / "summary.json", summary)
        return summary


def _run_stage1(*, train_tasks: Sequence[TaskSpec], test_tasks: Sequence[TaskSpec],
               initial_communication: str, assignment_policy: str,
               run_task: Callable[[RunRequest], TaskResult | Mapping],
               propose_policy: Callable[[ProposalRequest], PolicyProposal | Mapping],
               output_dir: Path, batch_size: int = 5, split_mode: str = "repo",
               test_repetitions: int = 1, seed: int | None = None,
               resume: bool = False, retry_interrupted: bool = False,
               baseline_mode: str = "single", review_policy: Callable | None = None,
               selection_mode: str = "review_replay", selection_config: SelectionConfig | None = None,
               candidates_per_batch: int = 1, replay_task_count: int | None = None,
               evidence_scope: str = "current_batch", replay_selection: str = "first_n") -> dict:
    """Train a joint-policy team, then compare it with single or initial-team trials.

    ``repo_id`` must come from task metadata supplied by the caller. A seed is
    passed as a request only; each runner records actual backend seed support.
    Every callback receives a new empty output directory and must provision a
    fresh task workspace. The controller does not itself restore repositories.
    """
    _validate_execution_mode(baseline_mode, "baseline_mode")
    _validate_tasks(train_tasks, test_tasks, split_mode)
    for name, value in (("batch_size", batch_size), ("test_repetitions", test_repetitions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")
    if not initial_communication.strip() or not assignment_policy.strip():
        raise ValueError("Both initial policy texts are required")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental second invocation from replacing
    # the first run's policy/history. Per-task directories are also exclusive.
    (root / "policies").mkdir(exist_ok=resume)
    write_text_atomic(root / "policies" / "assignment_v000.md", assignment_policy)
    write_text_atomic(root / "policies" / "communication_v000.md", initial_communication)
    manifest = {"split_mode": split_mode, "batch_size": batch_size,
                "evolved_strategies": ["assignment", "communication"],
                "baseline_mode": baseline_mode, "training_execution_mode": "team",
                "final_execution_mode": "team",
                "baseline_arm": "baseline" if baseline_mode == "single" else "initial",
                "baseline_label": "single_agent_without_team_policy" if baseline_mode == "single"
                                  else "initial_team_policy",
                "test_repetitions": test_repetitions, "requested_seed": seed,
                "seed_note": "Requested pairing only; provider determinism is not assumed. "
                             "Runner metrics must report whether the seed was supported/applied.",
                "train_tasks": [asdict(task) for task in train_tasks],
                "test_tasks": [asdict(task) for task in test_tasks],
                "selection_mode": selection_mode, "evidence_scope": evidence_scope,
                "change_history_in_prompt": "none" if evidence_scope == "current_batch" else "last 8 compact records",
                "training_mode": ("sequential_batches_with_candidate_review_and_replay"
                                  if selection_mode == "review_replay" else
                                  "sequential_batches_without_candidate_replay"),
                "selection": ("paired official training replay; reviewer advisory; no test feedback"
                              if selection_mode == "review_replay" else
                              "last successful proposal; no validation or test feedback"),
                "status": "running"}
    if selection_mode == "review_replay":
        manifest.update(selection_config=asdict(selection_config), candidates_per_batch=candidates_per_batch,
                        replay_task_count=replay_task_count,
                        replay_subset="first N tasks in manifest batch order; identical for all candidates",
                        candidate_version_note="Candidate versions count proposals across batches; "
                                               "rejected versions are not adopted, so accepted versions may skip.",
                        reviewer_role="advisory only; failures recorded, official replay still decides")
        if replay_selection == "reviewer":
            manifest.update(replay_selection=replay_selection,
                            replay_subset="first reviewed candidate chooses N current-batch tasks before replay; "
                                          "identical for all candidates; invalid advice falls back to first N",
                            reviewer_role="advisory policy recommendation plus shared replay task selection; "
                                          "failures fall back to first N, official replay still decides")
    write_json_atomic(root / "manifest.json", manifest)
    current, current_assignment, version = initial_communication, assignment_policy, 0
    train_records, history, outer_records, allowed = [], [], [], []
    replay_records, review_records, selections = [], [], []
    trained_versions: set[int] = set()
    for start in range(0, len(train_tasks), batch_size):
        batch_index = start // batch_size + 1
        batch_name = f"batch_{batch_index:03d}"
        items = list(enumerate(train_tasks[start:start + batch_size], start=start + 1))
        batch = []
        current_allowed = []
        for index, task in items:
            task_dir = root / "train" / batch_name / f"task_{index:03d}_{_component(task.task_id)}"
            record = _run(RunRequest(task, "train", version, current, current_assignment,
                                     task_dir, seed=seed, execution_mode="team",
                                     pair_id=f"{batch_name}_task_{index:03d}"
                                     if selection_mode == "review_replay" else None), run_task, resume=resume,
                          retry_interrupted=retry_interrupted)
            batch.append(record)
            train_records.append(record)
            current_allowed.append(task_dir)
            current_allowed.extend(p for p in task_dir.with_name(task_dir.name + ".interrupted").glob("*")
                                   if p.is_dir())
        allowed.extend(current_allowed)
        trained_versions.add(version)
        # This object snapshots the allowed directories before any current-batch
        # candidate runs. All K proposals/reviews receive the same evidence scope.
        evidence = TrainingEvidence(root / "train", current_allowed if evidence_scope == "current_batch" else allowed)
        compact_batch = tuple(_task_summary(record, root / "train") for record in batch)
        replay_count = min(replay_task_count or len(batch), len(batch))
        replay_items, reference = items[:replay_count], batch[:replay_count]
        replay_choice = None
        candidate_results, winner, winner_replay = [], None, None
        feedback_dirs = []
        for candidate_index in range(1, candidates_per_batch + 1):
            candidate_name = f"candidate_{candidate_index:03d}"
            candidate_version = (batch_index - 1) * candidates_per_batch + candidate_index
            proposal_request = ProposalRequest(version, current, current_assignment, batch_index,
                                               compact_batch,
                                               () if evidence_scope == "current_batch" else tuple(_compact(history[-8:])),
                                               len(history), evidence, selection_mode, candidate_index, evidence_scope,
                                               selection_config=selection_config or SelectionConfig())
            proposal_dir = root / "proposals" / batch_name
            if selection_mode == "review_replay":
                proposal_dir = proposal_dir / candidate_name
            outer, proposal = _outer_call(proposal_request, propose_policy, PolicyProposal,
                                          proposal_dir, resume=resume, from_version=version, label="Optimizer")
            outer_records.append(outer)
            if selection_mode == "legacy_no_replay":
                if proposal is None:
                    change = {"batch_index": batch_index, "from_version": version,
                              "to_version": version, "status": "proposal_failed",
                              "error": _compact(outer["error"])}
                else:
                    version += 1
                    current, current_assignment = proposal.communication_policy, proposal.assignment_policy
                    write_text_atomic(root / "policies" / f"assignment_v{version:03d}.md", current_assignment)
                    write_text_atomic(root / "policies" / f"communication_v{version:03d}.md", current)
                    change = {"batch_index": batch_index, "from_version": version - 1,
                              "to_version": version, "status": "proposed",
                              "rationale": _compact(proposal.rationale), "accepted_without_replay": True}
                continue

            feedback = root / "train" / batch_name / "feedback" / candidate_name
            feedback_dirs.append(feedback)
            write_json_atomic(feedback / "proposal.json", outer)
            entry = {"batch_index": batch_index, "candidate_index": candidate_index,
                     "candidate_version": candidate_version, "from_version": version,
                     "proposal_file": str((proposal_dir / "result.json").relative_to(root))}
            if proposal is None:
                entry.update(decision="inconclusive", reason="proposal_failed", error=_compact(outer["error"]))
            else:
                review_dir = root / "reviews" / batch_name / candidate_name
                review_request = ReviewRequest(
                    proposal_request, proposal,
                    replay_task_count=replay_count if replay_selection == "reviewer" else None,
                    replay_tasks=tuple({"task_id": task.task_id, "repo_id": task.repo_id}
                                       for _, task in replay_items) if replay_choice is not None else (),
                )
                review_record, review = _outer_call(review_request, review_policy,
                                                    PolicyReview, review_dir, resume=resume,
                                                    from_version=version, label="Reviewer")
                review_records.append(review_record)
                if replay_selection == "reviewer" and (replay_choice is None or
                                                         replay_choice["source"] == "review_unavailable"):
                    replay_items, replay_choice = _reviewer_replay_subset(review, items, replay_count)
                    reference_by_id = {(row["repo_id"], row["task_id"]): row for row in batch}
                    reference = [reference_by_id[(task.repo_id, task.task_id)] for _, task in replay_items]
                    replay_choice.update(
                        candidate_index=candidate_index,
                        tasks=[{"task_id": task.task_id, "repo_id": task.repo_id} for _, task in replay_items],
                    )
                if replay_choice is not None:
                    entry["replay_selection"] = replay_choice
                write_json_atomic(feedback / "review.json", {
                    **review_record, **({"replay_selection": replay_choice} if replay_choice is not None else {})
                })
                replay = []
                # Even a negative or failed advisory review does not skip real
                # execution or substitute a subjective score for official reward.
                for index, task in replay_items:
                    task_dir = (root / "train" / batch_name / "replay" / candidate_name /
                                f"task_{index:03d}_{_component(task.task_id)}")
                    record = _run(RunRequest(task, "train_replay", candidate_version,
                                             proposal.communication_policy, proposal.assignment_policy,
                                             task_dir, seed=seed, execution_mode="team",
                                             pair_id=f"{batch_name}_task_{index:03d}"), run_task,
                                  resume=resume, retry_interrupted=retry_interrupted)
                    replay.append(record)
                    replay_records.append(record)
                    allowed.append(task_dir)
                    allowed.extend(p for p in task_dir.with_name(task_dir.name + ".interrupted").glob("*")
                                   if p.is_dir())
                if replay:
                    trained_versions.add(candidate_version)
                entry.update(select_candidate(reference, replay, config=selection_config),
                             reviewer_status=review_record["status"],
                             reviewer_recommendation=review.recommendation if review is not None else None,
                             reviewer_advisory=True,
                             review_file=str((review_dir / "result.json").relative_to(root)),
                             replay_result_files=[str((Path(r["output_dir"]) / "task_result.json").relative_to(root))
                                                  for r in replay])
                if replay_choice is not None and replay_choice["source"] == "review_unavailable":
                    entry.update(decision="inconclusive", reason="reviewer_replay_selection_unavailable")
                if entry["decision"] == "accept":
                    versus_winner = (select_candidate(winner_replay, replay, config=selection_config)
                                     if winner is not None else None)
                    if versus_winner is not None:
                        entry["comparison_with_previous_candidate"] = versus_winner
                    if winner is None or versus_winner["decision"] == "accept":
                        winner = (candidate_version, proposal, candidate_index)
                        winner_replay = replay
            candidate_results.append(entry)
            write_json_atomic(proposal_dir / "selection.json", entry)
            write_json_atomic(feedback / "selection.json", entry)

        if selection_mode == "review_replay":
            selected_version = winner[0] if winner is not None else version
            decision = ("accept" if winner is not None else "inconclusive"
                        if any(row["decision"] == "inconclusive" for row in candidate_results) else "reject")
            selection = {"batch_index": batch_index, "from_version": version,
                         "selected_version": selected_version,
                         "selected_candidate_index": winner[2] if winner is not None else None,
                         "decision": decision, "config": asdict(selection_config),
                         "replay_task_ids": [{"task_id": task.task_id, "repo_id": task.repo_id}
                                             for _, task in replay_items],
                         "candidate_results": candidate_results,
                         "candidate_order_rule": "All compare to the unchanged incumbent; among accepted "
                                                 "candidates, replace the earlier choice only on an explicit "
                                                 "paired improvement under the same rule. Ties keep earlier.",
                         "training_selection_only": True, "heldout_used": False}
            if replay_selection == "reviewer":
                selection["replay_selection"] = replay_choice or {
                    "source": "not_run", "rationale": "No valid candidate proposal; no replay was executed",
                    "tasks": [],
                }
            selection_path = root / "selections" / f"{batch_name}.json"
            if resume and selection_path.exists():
                if read(selection_path) != selection:
                    raise ValueError("Recomputed training selection differs from its saved result")
            else:
                write_json_atomic(selection_path, selection)
            selections.append(selection)
            change = {"batch_index": batch_index, "from_version": version, "to_version": selected_version,
                      "status": decision, "accepted_without_replay": False,
                      "selected_candidate_index": selection["selected_candidate_index"],
                      "selection_file": str(selection_path.relative_to(root)),
                      "candidate_decisions": [{"candidate_index": r["candidate_index"],
                                               "decision": r["decision"], "reason": r["reason"]}
                                              for r in candidate_results]}
            if winner is not None:
                version, proposal, _ = winner
                current, current_assignment = proposal.communication_policy, proposal.assignment_policy
                change["rationale"] = _compact(proposal.rationale)
                # Adoption files are derived only after the durable batch
                # selection, never merely because a proposal was submitted.
                write_text_atomic(root / "policies" / f"assignment_v{version:03d}.md", current_assignment)
                write_text_atomic(root / "policies" / f"communication_v{version:03d}.md", current)
            batch_feedback = root / "train" / batch_name / "feedback" / "batch_selection"
            write_json_atomic(batch_feedback / "selection.json", selection)
            allowed.extend([*feedback_dirs, batch_feedback])
        history.append(change)
        write_json_atomic(root / "change_history.json", history)
        write_json_atomic(root / "progress.json", {"phase": "training", "completed_batches": batch_index,
                          "current_policy_version": version, "train": _aggregate(train_records),
                          "replay": _aggregate(replay_records), "optimizer": _aggregate(outer_records),
                          "reviewer": _aggregate(review_records),
                          "all_costs": _all_costs(train_records + replay_records + outer_records + review_records)})
    test_records = []
    baseline_arm = manifest["baseline_arm"]
    # Alternating arm order avoids making final consistently later in wall time.
    for index, task in enumerate(test_tasks, start=1):
        for repeat in range(test_repetitions):
            pair_id = f"task_{index:03d}_repeat_{repeat:03d}"
            baseline_policies = (("", "") if baseline_mode == "single"
                                 else (initial_communication, assignment_policy))
            arms = [(baseline_arm, 0, *baseline_policies, baseline_mode),
                    ("final", version, current, current_assignment, "team")]
            if (index - 1 + repeat) % 2:
                arms.reverse()
            for arm, arm_version, policy, assignment, execution_mode in arms:
                task_dir = root / "test" / arm / f"{pair_id}_{_component(task.task_id)}"
                request = RunRequest(task, "test", arm_version, policy, assignment,
                                     task_dir, repeat_index=repeat, pair_id=pair_id,
                                     seed=None if seed is None else seed + repeat,
                                     execution_mode=execution_mode)
                record = _run(request, run_task, resume=resume, retry_interrupted=retry_interrupted)
                record["arm"] = arm
                write_json_atomic(task_dir / "task_result.json", record)
                test_records.append(record)
    paired = []
    for pair_id in dict.fromkeys(row["pair_id"] for row in test_records):
        pair = {row["arm"]: row for row in test_records if row["pair_id"] == pair_id}
        baseline, final = pair[baseline_arm], pair["final"]
        available = all(row["status"] in {"completed", "failed"} and row["score"] is not None
                        for row in (baseline, final))
        comparison = {"pair_id": pair_id, "task_id": baseline["task_id"], "repo_id": baseline["repo_id"],
                      "baseline_arm": baseline_arm, "baseline_execution_mode": baseline_mode,
                      "baseline_policy_applied": baseline_mode == "team",
                      "final_execution_mode": "team", "final_policy_applied": True,
                      "baseline_status": baseline["status"], "final_status": final["status"],
                      "baseline_score": baseline["score"], "final_score": final["score"],
                      "score_difference": final["score"] - baseline["score"] if available else None,
                      "comparable": available, "token_usage_scope": "current_attempt_only"}
        for arm, record in (("baseline", baseline), ("final", final)):
            comparison.update({f"{arm}_{key}": value for key, value in _paired_measurements(record).items()})
        if baseline_mode == "team":
            comparison.update(initial_status=baseline["status"], initial_score=baseline["score"])
        paired.append(comparison)
    proposal_failures = sum(row["status"] == "failed" for row in outer_records)
    review_failures = sum(row["status"] == "failed" for row in review_records)
    execution_failures = sum(row["status"] == "failed" for row in train_records + replay_records + test_records)
    successful_proposals = len(outer_records) - proposal_failures
    accepted_updates = (sum(row["decision"] == "accept" for row in selections)
                        if selection_mode == "review_replay" else successful_proposals)
    if selection_mode == "review_replay":
        final_note = (f"{successful_proposals} candidate proposals completed; {len(replay_records)} replay task runs "
                      f"executed; {accepted_updates} policy updates accepted. ")
        if not replay_records:
            final_note += "No paired replay was executed; retaining P0 is fallback, not replay validation."
        elif not accepted_updates:
            final_note += "P0 retained; consult selection reasons to distinguish rejection from inconclusive evidence."
        else:
            final_note += "The final incumbent includes updates selected on training replay, not a held-out gain guarantee."
    else:
        final_note = (f"{successful_proposals} proposals completed under legacy adoption without replay. "
                      "No monotonic improvement is assumed; held-out outcomes did not select policies.")
    manifest.update(status="completed_with_failures" if proposal_failures or review_failures or execution_failures else "completed",
                    proposal_failures=proposal_failures, reviewer_failures=review_failures,
                    execution_failures=execution_failures,
                    final_policy_version=version,
                    final_policy_had_training_rollout=version in trained_versions,
                    successful_proposals=successful_proposals, accepted_policy_updates=accepted_updates,
                    executed_replay_tasks=len(replay_records), final_policy_note=final_note)
    summary = {**manifest, "train": _aggregate(train_records),
               f"test_{baseline_arm}": _aggregate([row for row in test_records if row["arm"] == baseline_arm]),
               "test_final": _aggregate([row for row in test_records if row["arm"] == "final"]),
               "optimizer": _aggregate(outer_records), "reviewer": _aggregate(review_records),
               "replay": _aggregate(replay_records), "paired_test_results": paired,
               "training_runs": train_records, "test_runs": test_records,
               "optimizer_runs": outer_records, "reviewer_runs": review_records, "replay_runs": replay_records,
               "selection_results": selections,
               "selection_counts": {decision: sum(row["decision"] == decision for row in selections)
                                    for decision in ("accept", "reject", "inconclusive")},
               "all_costs": _all_costs(train_records + replay_records + test_records + outer_records + review_records),
               "change_history": history}
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "summary.json", summary)
    write_json_atomic(root / "progress.json", {"phase": "completed", "status": summary["status"],
                      "current_policy_version": version, "all_costs": summary["all_costs"],
                      "selection_counts": summary["selection_counts"]})
    return summary
