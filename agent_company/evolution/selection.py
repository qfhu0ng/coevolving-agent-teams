"""Quality-first selection using complete, paired training-batch evaluations.

This is training-time candidate selection, not held-out validation. Official
scalar scores are compared as reported; no binary success metric is invented.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

SELECTION_RULE_VERSION = 3


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class SelectionConfig:
    """Score margins and a configurable token (default) or legacy USD tiebreak.

    ``quality_tolerance`` defines the no-clear-improvement interval for gains.
    Score gain must be strictly greater than that tolerance and at least
    ``min_score_gain`` to accept on quality alone. Otherwise, a candidate that
    has not regressed at all needs lower complete usage meeting the configured
    reduction fraction, or shorter complete agent time at exactly equal usage.
    Tolerance is not permission to exchange a score loss for efficiency.
    """

    quality_tolerance: float = 0.0
    min_score_gain: float = 0.0
    min_cost_reduction_fraction: float = 0.0
    min_token_reduction_fraction: float = 0.0
    efficiency_metric: str = "tokens"

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == "efficiency_metric":
                continue
            number = _number(getattr(self, name))
            if number is None or number < 0:
                raise ValueError(f"{name} must be a finite nonnegative number, not bool")
        if self.min_cost_reduction_fraction >= 1:
            raise ValueError("min_cost_reduction_fraction must be less than 1")
        if self.min_token_reduction_fraction >= 1:
            raise ValueError("min_token_reduction_fraction must be less than 1")
        if self.efficiency_metric not in {"tokens", "cost_usd"}:
            raise ValueError("efficiency_metric must be tokens or cost_usd")
        if self.efficiency_metric == "tokens" and self.min_cost_reduction_fraction:
            raise ValueError("Use min_token_reduction_fraction for token selection")


def _score(row: Mapping) -> float | None:
    # Failed execution can still yield an official score. The runner only sets
    # score after completed verification; do not discard it for a CLI/OOM error.
    return _number(row.get("score")) if row.get("status") in {"completed", "failed"} else None


def _cost(row: Mapping) -> float | None:
    # Partial known spend is not a complete cost, especially after interrupted retries.
    cost = _number(row.get("cost_usd"))
    return cost if cost is not None and cost >= 0 else None


def _arm_stats(rows: Sequence[dict]) -> tuple[float | None, float | None]:
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        return None, None
    scores, costs = [_score(row) for row in rows], [_cost(row) for row in rows]
    mean, total = None, None
    if all(score is not None for score in scores):
        try:
            mean = _number(math.fsum(score / len(scores) for score in scores))
        except OverflowError:
            pass
    if all(cost is not None for cost in costs):
        try:
            total = _number(math.fsum(costs))
        except OverflowError:
            pass
    return mean, total


def _identity(row) -> tuple[str, str] | None:
    if not isinstance(row, Mapping):
        return None
    repo_id, task_id = row.get("repo_id"), row.get("task_id")
    if any(not isinstance(value, str) or not value.strip() for value in (repo_id, task_id)):
        return None
    return repo_id, task_id


def measurements(row: Mapping) -> dict:
    """Use agent time, never preparation/verifier time or summed worker time."""
    metrics = row.get("metrics") or {}
    runtime = metrics.get(row.get("execution_mode", "team")) or {}
    elapsed = _number((runtime.get("execution") or {}).get("elapsed_seconds"))
    budget = _number((metrics.get("task") or {}).get("agent_timeout_sec"))
    usage = row.get("usage") or {}
    tokens = {key: value if isinstance(value := usage.get(key), int)
              and not isinstance(value, bool) and value >= 0 else None for key in
              ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "total_tokens")}
    return {"agent_elapsed_seconds": elapsed if elapsed is not None and elapsed >= 0 else None,
            "wall_time_budget_seconds": budget if budget is not None and budget > 0 else None,
            **tokens,
            "usage_complete": all(v is not None for v in tokens.values())
            and tokens["total_tokens"] == tokens["prompt_tokens"] + tokens["completion_tokens"]
            and tokens["cached_prompt_tokens"] <= tokens["prompt_tokens"]
            and not usage.get("unknown_usage_attempts") and not usage.get("calls_missing_usage")
            and runtime.get("usage_complete") is not False and row.get("status") != "interrupted",
            "has_interrupted_attempts": bool(row.get("interrupted_attempts"))}


def selection_objective(config: SelectionConfig) -> dict:
    if config.efficiency_metric == "tokens":
        return {"rule_version": SELECTION_RULE_VERSION, "thresholds": asdict(config),
                "priority": ["official_mean_score", "complete_execution_total_tokens", "agent_elapsed_seconds"],
                "instruction": "Compare the same model and per-task wall-clock cap. Reject any mean-score "
                "decrease; clear score improvement takes priority. Otherwise prefer fewer complete total "
                "tokens, then shorter agent wall time at equal tokens. Missing usage is unknown, not zero. "
                "Tokens are not GPU compute or money. Report input/output/cache separately and outer-loop "
                "usage separately from execution. Retain official scores even when execution status is failed; "
                "report execution failures separately. This is empirical selection, not significance evidence."}
    return {"rule_version": SELECTION_RULE_VERSION, "thresholds": asdict(config),
            "priority": ["official_mean_score", "complete_execution_cost_usd", "agent_elapsed_seconds"],
            "instruction": "Use the same per-task wall-clock cap. Reject any observed mean-score "
                           "decrease, even inside quality_tolerance. A clear score gain takes priority "
                           "over cost/time. Otherwise require lower complete cost meeting its threshold, "
                           "or shorter complete agent time at exactly equal complete cost and verified "
                           "equal caps. Unknown costs are not zero; tokens do not substitute for dollars. "
                           "Retain official scores even when execution status is failed; report failures separately. "
                           "These are empirical selection rules, not significance tests."}


def select_candidate(incumbent: Sequence[dict], candidate: Sequence[dict], *,
                     config: SelectionConfig | None = None) -> dict:
    """Return accept/reject/inconclusive without dropping failed or missing tasks.

    Pair by (repo_id, task_id), ignoring input order. Incomplete identity/score
    evidence is always inconclusive. Reject any negative mean delta; accept a
    clear positive gain even when costs are unknown. Otherwise compare complete
    token usage (or explicit legacy USD), then agent times at matched caps.
    Missing caps in both arms are reported as unverified for legacy/generic runners;
    one-sided or differing caps cannot support selection. Never drop failed pairs.
    """
    config = SelectionConfig() if config is None else config
    if not isinstance(config, SelectionConfig):
        raise TypeError("config must be SelectionConfig or None")
    incumbent, candidate = list(incumbent), list(candidate)
    incumbent_mean, incumbent_cost = _arm_stats(incumbent)
    candidate_mean, candidate_cost = _arm_stats(candidate)
    result = {
        "decision": "inconclusive", "reason": None, "config": asdict(config),
        "incumbent_runs": len(incumbent), "candidate_runs": len(candidate),
        "pair_count": 0, "comparable_count": 0, "pairs": [],
        "incumbent_mean_score": incumbent_mean, "candidate_mean_score": candidate_mean,
        "mean_score_difference": None,
        "incumbent_total_cost_usd": incumbent_cost, "candidate_total_cost_usd": candidate_cost,
        "cost_complete": incumbent_cost is not None and candidate_cost is not None,
        "cost_reduction_fraction": None,
        "selection_objective": selection_objective(config),
        "wall_time_caps_verified": False, "time_complete": False,
        "incumbent_agent_seconds": None, "candidate_agent_seconds": None,
        "agent_seconds_difference": None,
    }
    if result["cost_complete"] and incumbent_cost > 0:
        result["cost_reduction_fraction"] = _number((incumbent_cost - candidate_cost) / incumbent_cost)
    if not incumbent or not candidate:
        return {**result, "reason": "empty_batch"}
    incumbent_ids, candidate_ids = list(map(_identity, incumbent)), list(map(_identity, candidate))
    if None in incumbent_ids or None in candidate_ids:
        return {**result, "reason": "invalid_task_identity"}
    if len(set(incumbent_ids)) != len(incumbent_ids) or len(set(candidate_ids)) != len(candidate_ids):
        return {**result, "reason": "duplicate_task_identity"}
    candidate_by_id = dict(zip(candidate_ids, candidate))
    for identity, old in zip(incumbent_ids, incumbent):
        if identity not in candidate_by_id:
            continue
        new = candidate_by_id[identity]
        old_score, new_score = _score(old), _score(new)
        old_measure, new_measure = measurements(old), measurements(new)
        delta = _number(new_score - old_score) if old_score is not None and new_score is not None else None
        result["pairs"].append({
            "repo_id": identity[0], "task_id": identity[1],
            "incumbent_status": old.get("status"), "candidate_status": new.get("status"),
            "incumbent_score": old_score, "candidate_score": new_score,
            "score_difference": delta, "comparable": delta is not None,
            "incumbent_cost_usd": _cost(old), "candidate_cost_usd": _cost(new),
            "incumbent_measurements": old_measure, "candidate_measurements": new_measure,
        })
    result["pair_count"] = len(result["pairs"])
    result["comparable_count"] = sum(pair["comparable"] for pair in result["pairs"])
    if set(incumbent_ids) != set(candidate_ids):
        return {**result, "reason": "mismatched_task_identities"}
    def model(row):
        return ((row.get("metrics") or {}).get(row.get("execution_mode", "team")) or {}).get("model")
    if any(model(old) != model(candidate_by_id[identity])
           for identity, old in zip(incumbent_ids, incumbent)):
        return {**result, "reason": "incomparable_models"}
    caps = [(p["incumbent_measurements"]["wall_time_budget_seconds"],
             p["candidate_measurements"]["wall_time_budget_seconds"]) for p in result["pairs"]]
    result["wall_time_caps_verified"] = all(a is not None and a == b for a, b in caps)
    if any(a != b for a, b in caps):
        return {**result, "reason": "incomparable_wall_time_caps"}
    times = [[p[f"{arm}_measurements"]["agent_elapsed_seconds"] for p in result["pairs"]]
             for arm in ("incumbent", "candidate")]
    if all(t is not None for arm in times for t in arm):
        try:
            old_time, new_time = map(math.fsum, times)
            if all(math.isfinite(t) for t in (old_time, new_time, new_time - old_time)):
                result.update(time_complete=True, incumbent_agent_seconds=old_time,
                              candidate_agent_seconds=new_time,
                              agent_seconds_difference=new_time - old_time)
        except OverflowError:
            pass
    if (result["comparable_count"] != len(incumbent) or incumbent_mean is None
            or candidate_mean is None):
        return {**result, "reason": "incomplete_score_evidence"}
    delta = _number(candidate_mean - incumbent_mean)
    result["mean_score_difference"] = delta
    if delta is None:
        return {**result, "reason": "nonfinite_score_comparison"}
    if delta < 0:
        if delta >= -config.quality_tolerance:
            return {**result, "decision": "reject", "reason": "score_regressed_within_tolerance"}
        return {**result, "decision": "reject", "reason": "score_regressed_beyond_tolerance"}
    if delta > config.quality_tolerance and delta >= config.min_score_gain:
        return {**result, "decision": "accept", "reason": "score_improved"}
    if config.efficiency_metric == "tokens":
        measures = [measurements(r) for r in incumbent + candidate]
        if not all(m["usage_complete"] for m in measures):
            return {**result, "reason": "incomplete_token_evidence"}
        if any(m["has_interrupted_attempts"] for m in measures):
            return {**result, "reason": "interrupted_token_evidence"}
        if not result["wall_time_caps_verified"]:
            return {**result, "reason": "unverified_wall_time_caps"}
        old_tokens = sum(measurements(r)["total_tokens"] for r in incumbent)
        new_tokens = sum(measurements(r)["total_tokens"] for r in candidate)
        reduction = (old_tokens - new_tokens) / old_tokens if old_tokens else None
        result.update(incumbent_total_tokens=old_tokens, candidate_total_tokens=new_tokens,
                      token_reduction_fraction=reduction, token_usage_complete=True)
        if new_tokens > old_tokens:
            return {**result, "decision": "reject", "reason": "tokens_not_reduced"}
        if new_tokens < old_tokens:
            if reduction < config.min_token_reduction_fraction:
                return {**result, "decision": "reject", "reason": "token_reduction_below_threshold"}
            return {**result, "decision": "accept", "reason": "tokens_reduced_without_clear_score_gain"}
        if not result["time_complete"]:
            return {**result, "reason": "incomplete_time_evidence_at_equal_tokens"}
        if result["candidate_agent_seconds"] < result["incumbent_agent_seconds"]:
            return {**result, "decision": "accept", "reason": "time_reduced_at_equal_tokens"}
        return {**result, "decision": "reject", "reason": "tokens_and_time_not_improved"}
    if not result["cost_complete"]:
        return {**result, "reason": "incomplete_cost_evidence"}
    if candidate_cost > incumbent_cost:
        return {**result, "decision": "reject", "reason": "cost_not_reduced"}
    if candidate_cost == incumbent_cost:
        if not result["time_complete"] or not result["wall_time_caps_verified"]:
            return {**result, "reason": "incomplete_time_evidence_at_equal_cost"}
        if any(measurements(r)["has_interrupted_attempts"] for r in incumbent + candidate):
            return {**result, "reason": "interrupted_time_evidence_at_equal_cost"}
        if result["candidate_agent_seconds"] < result["incumbent_agent_seconds"]:
            return {**result, "decision": "accept", "reason": "time_reduced_at_equal_cost"}
        return {**result, "decision": "reject", "reason": "cost_and_time_not_improved"}
    if result["cost_reduction_fraction"] < config.min_cost_reduction_fraction:
        return {**result, "decision": "reject", "reason": "cost_reduction_below_threshold"}
    return {**result, "decision": "accept", "reason": "cost_reduced_without_clear_score_gain"}
