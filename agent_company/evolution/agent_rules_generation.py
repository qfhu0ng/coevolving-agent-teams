"""Generate-only entry for agent_rules_v1: evidence -> optimizer -> reviewer -> candidate M (plan v4 section 12, item 6).

It reuses StrategyOptimizer and StrategyReviewer, writes a schema 2 candidate package, the
candidate, the review and a report, and returns. It never constructs a run request, calls a
runner or deploys anything; whether and how the candidate runs is the user's decision.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

from agent_company.analysis.agent_rules_evidence import agent_rules_record
from agent_company.analysis.evidence_projection import project_model_evidence
from agent_company.evolution import agent_rules_outer_text as outer_text
from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.harness_capabilities import executability_summary, target_harness
from agent_company.evolution.loop import (PolicyProposal, PolicyReview, ProposalRequest, ReviewRequest,
                                          _validate_outer_result)
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.evolution.outer_budget import WorkingMemory
from agent_company.evolution.reviewer import StrategyReviewer
from agent_company.policy.role_package import AGENT_RULES_PROTOCOL, _unique_pairs, validate_agent_rules_package
from agent_company.runtime.provider import public_provider
from agent_company.runtime.relay import ProviderConfig
from agent_company.runtime_io import write_json_atomic

# Every reasoning text stays in the in-memory conversation: no placeholder, no window (B1).
MEMORY = WorkingMemory(page_window=2, notebook_chars=40_000, final_calls=2, min_page_chars=12_000,
                       reasoning_window=None)
# The first real review spent all 8 calls reading and missed its one submit-only call (2026-10-19):
# DeepSeek thinking mode cannot force the submit tool, so keep two submit-only calls, as the optimizer does.
# The second spent the first of them reading and had its submission refused for one invalid harness
# reference, so a refused submission in a submit-only call earns up to two correction calls.
REVIEWER_ROUNDS, REVIEWER_FINAL_CALLS, REVIEWER_REPAIR_CALLS = 16, 2, 2


class AgentRulesProposalRequest(ProposalRequest):
    def as_prompt_data(self):
        data = super().as_prompt_data()
        data.update(selection_mode=outer_text.MODE,
                    selection_objective={"rule_version": outer_text.MODE, "instruction": outer_text.OPTIMIZER_OBJECTIVE},
                    instruction=outer_text.GENERATION_INSTRUCTION)
        return data


class AgentRulesReviewRequest(ReviewRequest):
    def as_prompt_data(self):
        data = super().as_prompt_data()
        data["instruction"] = outer_text.REVIEW_INSTRUCTION
        return data


def _failure(exc: Exception) -> dict:
    # Exception bodies can carry provider text; keep the type and accounting only.
    return {"status": "failed", "error_type": type(exc).__name__,
            **{name: getattr(exc, name, None) for name in ("usage", "cost_usd", "known_cost_usd", "elapsed_seconds")}}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_strategy(trial: Path) -> dict:
    """The master strategy a recorded run used, from its run request (external review #7)."""
    request = json.loads((Path(trial) / "run_request.json").read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(request, dict) or not request.get("assignment_policy") or not request.get("communication_policy"):
        raise ValueError(f"{trial} is not an agent_rules_v1 Team trial: its run request has no master strategy texts")
    package = validate_agent_rules_package(request.get("policy_package"))
    team = package["team"]
    if (request.get("assignment_policy"), request.get("communication_policy")) != (team["assignment"],
                                                                                    team["communication"]):
        raise ValueError(f"{trial}: the run request's master strategy differs from its package")
    return package


def strategy_problems(incumbent: dict, sources: list[dict]) -> list[str]:
    """One package id and revision names one pair of texts, and some run must have used the incumbent."""
    def ref(package):
        return package["package_id"], package["revision"]
    problems = [f"run {index} used {source['package_id']} revision {source['revision']} with texts that differ "
                "from the incumbent of that id and revision"
                for index, source in enumerate(sources, 1)
                if ref(source) == ref(incumbent) and source["team"] != incumbent["team"]]
    if not any(ref(source) == ref(incumbent) and source["team"] == incumbent["team"] for source in sources):
        problems.append("no recorded run used the incumbent master strategy")
    return problems


def canonical_sha256(package: dict) -> str:
    """One hash for a package whether it came from a file or a run request (sorted keys, compact)."""
    return hashlib.sha256(json.dumps(package, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_evidence(trials: list[Path], root: Path, incumbent: dict) -> tuple[list[Path], tuple[dict, ...]]:
    """One directory per recorded run: the existing projection, the agent_rules_v1 record and the master strategy it ran."""
    directories, summaries = [], []
    for index, trial in enumerate(trials, 1):
        record = agent_rules_record(trial)
        if record is None:
            raise ValueError(f"{trial} is not an agent_rules_v1 Team trial")
        source = source_strategy(trial)
        projection = project_model_evidence(trial)
        directory = root / f"run_{index}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json_atomic(directory / "projection.json", projection)
        write_json_atomic(directory / "agent_rules.json", record)
        write_json_atomic(directory / "master_strategy.json", {
            "package_id": source["package_id"], "revision": source["revision"], "parent": source["parent"],
            "assignment_policy": source["team"]["assignment"], "communication_policy": source["team"]["communication"]})
        result = projection.get("task_result") or {}
        official = result.get("official_result") or {}
        summary = {"run": index, "task_id": (result.get("task_id") or {}).get("text"), "execution_mode": "team",
                   "status": (result.get("status") or {}).get("text"), "score": result.get("score"),
                   # The objective is the official total of passed tests; score is that total's fraction.
                   "official_passed_tests": official.get("passed_tests"), "official_valid_tests": official.get("valid_tests"),
                   # Texts only: the same texts under another id read the same to the optimizer (review of #7).
                   "master_strategy": {"package_id": source["package_id"], "revision": source["revision"],
                                       "same_as_current": source["team"] == incumbent["team"]},
                   "evidence_paths": [f"run_{index}/projection.json", f"run_{index}/agent_rules.json",
                                      f"run_{index}/master_strategy.json"]}
        write_json_atomic(directory / "task_summary.json", summary)
        directories.append(directory)
        summaries.append(summary)
    return directories, tuple(summaries)


def _review_assessment(review: PolicyReview, notes) -> dict:
    """The controller's view: executability counts, unassessed notes, unchecked task-specificity parts."""
    assessment = executability_summary(review.executability)
    checked = {(item["policy_part"], " ".join(item["rule_quote"].split())) for item in review.executability or []}
    assessment["unassessed_notes"] = [
        {"verdict": "unknown", "policy_part": note["policy_part"], "rule_quote": note["rule_quote"],
         "reason": "Changed rule has no corresponding executability assessment"}
        for note in notes or [] if (note["policy_part"], " ".join(note["rule_quote"].split())) not in checked]
    task = review.task_specificity
    assessment["task_specificity"] = ({"recorded": False, "unchecked": ["assignment", "communication"]}
                                      if task is None else {"recorded": True, **task})
    assessment["dispatch"] = "not_dispatched: generate-only entry; the user decides"
    return assessment


def _proposal_request(incumbent: dict, root: Path, directories: list[Path], summaries: tuple,
                      harness: dict) -> AgentRulesProposalRequest:
    return AgentRulesProposalRequest(
        current_version=incumbent["revision"], assignment_policy=incumbent["team"]["assignment"],
        communication_policy=incumbent["team"]["communication"], batch_index=1, batch_results=summaries,
        change_history=({"incumbent": incumbent["package_id"], "revision": incumbent["revision"],
                         "parent": incumbent["parent"], "runs": len(directories)},), history_total=1,
        evidence=TrainingEvidence(root, directories), selection_mode=outer_text.MODE,
        evidence_scope="cumulative" if len(directories) > 1 else "current_batch", target_harness=harness)


def generate(*, package: dict, trials: list[Path], inference: dict, provider: ProviderConfig, output: Path,
             candidate_id: str, code_version: str | None = None, native_version: str | None = None,
             package_path: Path | None = None, generator: dict | None = None,
             optimizer_completion: Callable | None = None, reviewer_completion: Callable | None = None) -> dict:
    """Produce and review one candidate M; returns the report. Real provider calls happen unless completions are given."""
    if (optimizer_completion is None or reviewer_completion is None) and not outer_text.TEXT_STATUS.startswith("approved_"):
        # Real optimizer/reviewer calls read these texts; they need the user's approval first (plan v4 M5).
        raise ValueError("The outer-loop texts are not approved; real optimizer/reviewer calls are refused")
    incumbent = validate_agent_rules_package(package)
    trials = [Path(trial).resolve(strict=True) for trial in trials]
    if not trials or len(set(trials)) != len(trials):
        raise ValueError("Supply one or more distinct recorded agent_rules_v1 Team trials")
    if not isinstance(candidate_id, str) or not candidate_id.strip() or candidate_id == incumbent["package_id"]:
        raise ValueError("The candidate needs a new package_id")
    output = Path(output).resolve()
    if any(output == trial or output.is_relative_to(trial) for trial in trials):
        raise ValueError("Generation output must be outside every source trial")
    if output.exists():
        raise ValueError("Generation output already exists")
    sources = [source_strategy(trial) for trial in trials]
    problems = strategy_problems(incumbent, sources)
    if problems:
        raise ValueError("; ".join(problems))
    harness = target_harness(inference, execution_mode="team", provider=provider, code_version=code_version,
                             native_version=native_version, policy_protocol=AGENT_RULES_PROTOCOL)
    # Evidence is built beside the output first, so a failure leaves no half-made output (M5 review minor 16).
    building = output.with_name(output.name + ".building")
    if building.exists():
        # Left by an earlier failed evidence build; never removed here (M5 review R2-8).
        raise ValueError(f"{building.name} is left from an earlier failed build; check and remove it first")
    building.mkdir(parents=True)
    build_evidence(trials, building / "training_evidence", incumbent)
    building.rename(output)
    root = output / "training_evidence"
    directories = [root / f"run_{index}" for index in range(1, len(trials) + 1)]
    summaries = tuple(json.loads((d / "task_summary.json").read_text(encoding="utf-8")) for d in directories)
    request = _proposal_request(incumbent, root, directories, summaries, harness)
    record = {"entry": "agent_rules_generation", "text_status": outer_text.TEXT_STATUS,
              "incumbent": {"package_id": incumbent["package_id"], "revision": incumbent["revision"],
                            "canonical_sha256": canonical_sha256(incumbent),
                            **({"sha256": _sha256(package_path)} if package_path else {})},
              "trials": [str(trial) for trial in trials], "evidence_directories": [d.name for d in directories],
              "trial_strategies": [{"package_id": s["package_id"], "revision": s["revision"],
                                    "canonical_sha256": canonical_sha256(s)} for s in sources],
              "provider": public_provider(provider), "code_version": code_version, "native_version": native_version,
              "generator": generator, "run_dispatched": False, "adopted": False}
    write_json_atomic(output / "generation.json", record)
    report = {"status": "proposal_failed", "candidate": None, "review": None, "run_dispatched": False,
              "adopted": False, "requires_user_decision": True}
    retries = provider.max_attempts - 1
    proposal = None
    try:
        proposal = StrategyOptimizer(provider, output / "optimizer", completion=optimizer_completion,
                                     max_tool_rounds=32, max_context_chars=180_000, working_memory=MEMORY,
                                     require_citations=True, transport_retries=retries)(request)
        _validate_outer_result(proposal)
        candidate = validate_agent_rules_package({
            "schema_version": 2, "package_id": candidate_id, "revision": incumbent["revision"] + 1,
            "parent": {"package_id": incumbent["package_id"], "revision": incumbent["revision"]},
            "protocol": {"id": AGENT_RULES_PROTOCOL},
            "team": {"assignment": proposal.assignment_policy, "communication": proposal.communication_policy},
            "changes": [{"source": "optimizer", "evidence_runs": len(trials), "rationale": proposal.rationale}],
            "review_ref": None})
    except Exception as exc:
        failure = _failure(exc)
        if proposal is not None:
            # The optimizer finished and spent its calls; only its result was rejected (M5 review R2-8).
            failure.update(usage=proposal.usage, known_cost_usd=proposal.known_cost_usd)
        write_json_atomic(output / "candidate.json", failure)
        report.update(usage={"optimizer": failure["usage"]}, known_cost_usd={"optimizer": failure["known_cost_usd"]})
        write_json_atomic(output / "report.json", report)
        return report
    write_json_atomic(output / "candidate.json", {**asdict(proposal), "status": "proposed_not_adopted"})
    report.update(status="review_failed", candidate={"package_id": candidate_id, "revision": candidate["revision"]},
                  usage={"optimizer": proposal.usage}, known_cost_usd={"optimizer": proposal.known_cost_usd})
    review = None
    try:
        review = StrategyReviewer(provider, output / "reviewer", completion=reviewer_completion,
                                  transport_retries=retries, max_tool_rounds=REVIEWER_ROUNDS,
                                  final_calls=REVIEWER_FINAL_CALLS,
                                  repair_calls=REVIEWER_REPAIR_CALLS)(AgentRulesReviewRequest(request, proposal))
        if not isinstance(review, PolicyReview):
            raise ValueError("Reviewer did not return a valid PolicyReview")
        _validate_outer_result(review)
    except Exception as exc:
        failure = _failure(exc)
        if isinstance(review, PolicyReview):
            failure.update(usage=review.usage, known_cost_usd=review.known_cost_usd)
        write_json_atomic(output / "review.json", failure)
        # Not a reviewed candidate: kept under a name that says so.
        write_json_atomic(output / "package.unreviewed.json", candidate)
        report["usage"]["reviewer"], report["known_cost_usd"]["reviewer"] = failure["usage"], failure["known_cost_usd"]
        write_json_atomic(output / "report.json", report)
        return report
    candidate["review_ref"] = "review.json"
    write_json_atomic(output / "review.json", {**asdict(review), "status": "completed"})
    write_json_atomic(output / "package.json", validate_agent_rules_package(candidate))
    report.update(status="reviewed", review={"recommendation": review.recommendation,
                                             "assessment": _review_assessment(review, proposal.execution_notes)},
                  usage={"optimizer": proposal.usage, "reviewer": review.usage},
                  known_cost_usd={"optimizer": proposal.known_cost_usd, "reviewer": review.known_cost_usd})
    write_json_atomic(output / "report.json", report)
    return report


def _rebuilt_harness(record: dict, inference: dict, provider: ProviderConfig) -> dict:
    """The harness description from this profile's inference and the versions the generation recorded."""
    return target_harness(inference, execution_mode="team", provider=provider, code_version=record.get("code_version"),
                          native_version=record.get("native_version"), policy_protocol=AGENT_RULES_PROTOCOL)


def review_problems(generation: Path, incumbent: dict, provider: ProviderConfig, label: str = "2",
                    inference: dict | None = None) -> list[str]:
    """Free checks before reviewing an existing candidate again; an empty list means the review may run."""
    generation = Path(generation)
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", label):
        return ["--label must be 1-20 letters, digits, '_' or '-'"]
    try:
        record = json.loads((generation / "generation.json").read_text(encoding="utf-8"))
        report = json.loads((generation / "report.json").read_text(encoding="utf-8"))
        candidate = json.loads((generation / "candidate.json").read_text(encoding="utf-8"))
        package = validate_agent_rules_package(
            json.loads((generation / "package.unreviewed.json").read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        return [f"not a generation whose review failed: {type(exc).__name__}: {exc}"]
    problems = []
    if (generation / "package.json").exists():
        problems.append("package.json exists: this candidate already has a completed review")
    elif report.get("status") != "review_failed":
        problems.append(f"only a candidate whose review failed is reviewed again (status {report.get('status')})")
    if (record.get("incumbent") or {}).get("canonical_sha256") != canonical_sha256(incumbent):
        problems.append("--package is not the incumbent this candidate was generated from")
    if package["parent"] != {"package_id": incumbent["package_id"], "revision": incumbent["revision"]}:
        problems.append("the unreviewed package's parent is not the incumbent")
    if (package["team"]["assignment"], package["team"]["communication"]) != (candidate.get("assignment_policy"),
                                                                               candidate.get("communication_policy")):
        problems.append("package.unreviewed.json and candidate.json carry different texts")
    if record.get("provider") != public_provider(provider):
        problems.append("the provider differs from the one the candidate was generated with")
    taken = [name for name in (f"reviewer_{label}", f"review_{label}.json", f"report_review_{label}.json")
             if (generation / name).exists()]
    if taken:
        problems.append(f"already present: {', '.join(taken)}")
    if inference is not None:
        # The reviewer must see the harness the optimizer saw: compare with the optimizer's first request.
        try:
            first = json.loads((generation / "optimizer/batch_0001/request_000.json").read_text(encoding="utf-8"))
            shown = json.loads(first["messages"][1]["content"])["target_harness"]
            rebuilt = json.loads(json.dumps(_rebuilt_harness(record, inference, provider)))
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            problems.append(f"cannot compare with the harness the optimizer saw: {type(exc).__name__}: {exc}")
        else:
            if rebuilt != shown:
                problems.append("--profile does not give the harness description the candidate was generated against")
    return problems


def _repairs_granted(reviewer_dir: Path) -> int:
    """How many correction calls the reviewer was granted (0 when it needed none)."""
    granted = 0
    for path in reviewer_dir.glob("*/submission_repairs.json"):
        granted += json.loads(path.read_text(encoding="utf-8")).get("granted", 0)
    return granted


def review_candidate(*, package: dict, generation: Path, inference: dict, provider: ProviderConfig,
                     code_version: str | None = None, reviewer_completion: Callable | None = None,
                     label: str = "2") -> dict:
    """Review an existing candidate again after its review failed; the optimizer is not called (2026-10-19).

    The candidate, its evidence and the harness description are the ones it was generated with. The first
    review's files stay as they are; this one writes reviewer_<label>/, review_<label>.json and
    report_review_<label>.json, plus package.json only when the review completes.
    """
    if reviewer_completion is None and not outer_text.TEXT_STATUS.startswith("approved_"):
        raise ValueError("The outer-loop texts are not approved; real reviewer calls are refused")
    incumbent = validate_agent_rules_package(package)
    generation = Path(generation).resolve()
    problems = review_problems(generation, incumbent, provider, label, inference)
    if problems:
        raise ValueError("; ".join(problems))
    record = json.loads((generation / "generation.json").read_text(encoding="utf-8"))
    candidate = json.loads((generation / "candidate.json").read_text(encoding="utf-8"))
    package_candidate = validate_agent_rules_package(
        json.loads((generation / "package.unreviewed.json").read_text(encoding="utf-8")))
    harness = _rebuilt_harness(record, inference, provider)
    root = generation / "training_evidence"
    directories = [root / name for name in record["evidence_directories"]]
    summaries = tuple(json.loads((d / "task_summary.json").read_text(encoding="utf-8")) for d in directories)
    request = _proposal_request(incumbent, root, directories, summaries, harness)
    # candidate.json is stored with sorted keys, so key order inside execution_notes and reading_audit may differ
    # from the first review's request; the content is the same.
    proposal = PolicyProposal(**{f.name: candidate[f.name] for f in fields(PolicyProposal) if f.name in candidate})
    report = {"entry": "agent_rules_review", "earlier_report": "report.json", "status": "review_failed",
              "candidate": {"package_id": package_candidate["package_id"], "revision": package_candidate["revision"]},
              "review": None, "reviewer_rounds": REVIEWER_ROUNDS, "reviewer_final_calls": REVIEWER_FINAL_CALLS,
              "reviewer_repair_calls": REVIEWER_REPAIR_CALLS,
              "code_version": code_version, "run_dispatched": False, "adopted": False, "requires_user_decision": True}
    review = None
    try:
        review = StrategyReviewer(provider, generation / f"reviewer_{label}", completion=reviewer_completion,
                                  transport_retries=provider.max_attempts - 1, max_tool_rounds=REVIEWER_ROUNDS,
                                  final_calls=REVIEWER_FINAL_CALLS,
                                  repair_calls=REVIEWER_REPAIR_CALLS)(AgentRulesReviewRequest(request, proposal))
        if not isinstance(review, PolicyReview):
            raise ValueError("Reviewer did not return a valid PolicyReview")
        _validate_outer_result(review)
    except Exception as exc:
        failure = _failure(exc)
        if isinstance(review, PolicyReview):
            failure.update(usage=review.usage, known_cost_usd=review.known_cost_usd)
        write_json_atomic(generation / f"review_{label}.json", failure)
        report.update(usage={"reviewer": failure["usage"]}, known_cost_usd={"reviewer": failure["known_cost_usd"]},
                      reviewer_repairs_granted=_repairs_granted(generation / f"reviewer_{label}"))
        write_json_atomic(generation / f"report_review_{label}.json", report)
        return report
    write_json_atomic(generation / f"review_{label}.json", {**asdict(review), "status": "completed"})
    report.update(review={"recommendation": review.recommendation,
                          "assessment": _review_assessment(review, proposal.execution_notes)},
                  usage={"reviewer": review.usage}, known_cost_usd={"reviewer": review.known_cost_usd},
                  reviewer_repairs_granted=_repairs_granted(generation / f"reviewer_{label}"))
    package_candidate["review_ref"] = f"review_{label}.json"
    try:
        write_json_atomic(generation / "package.json", validate_agent_rules_package(package_candidate))
    except Exception as exc:
        # The review is complete and recorded; only the reviewed package could not be written.
        report.update(status="review_completed_package_not_written", error_type=type(exc).__name__)
        write_json_atomic(generation / f"report_review_{label}.json", report)
        return report
    # report.json keeps the first review's outcome; this report is the current one for the candidate.
    report["status"] = "reviewed"
    write_json_atomic(generation / f"report_review_{label}.json", report)
    return report
