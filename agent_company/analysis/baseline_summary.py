"""Read-only, descriptive summaries of independently launched comparison rounds.

By default only run_settings.json, task_result.json and, when present, the test
identifiers and official totals in verifier/filtered_test_results.json (for the
test-set check) are read. Explicit layer rules opt into reading per-test
statuses. No provider, task assets, trajectories or policy-selection code is
imported. Official scores survive execution failures; absent scores stay
unknown. These statistics do not define an acceptance threshold, estimate an
error rate or authorize a follow-up run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Iterable

from agent_company.analysis.score_layers import LAYERS, load_rules, load_score_layers, official_test_universe


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _round(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("repeat_index must be a nonnegative integer")
    return value


def _component(value, label: str) -> str:
    # IDs select files below the supplied run directory, never arbitrary paths.
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value):
        raise ValueError(f"{label} must be a path component")
    return value


def _pair_id(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pair_id must be a nonempty string")
    return value


def _score(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else None


def load_comparison_results(directory: Path, *, layer_rules: dict | None = None) -> tuple[list[dict], dict]:
    """Load every planned task/arm, including runs with no result file yet.

    Old comparison settings predate explicit round metadata. Their documented
    defaults were round zero and pair_id=task_id; importing two such rounds is
    rejected as a duplicate by the summarizer, never silently renumbered.
    """
    directory = Path(directory).resolve()
    settings = _object(directory / "run_settings.json")
    rows = []
    for task in settings["tasks"]:
        task_id = _component(task["task_id"], "task_id")
        repeat_index = _round(task.get("repeat_index", settings.get("repeat_index", 0)))
        pair_id = _pair_id(task.get("pair_id", task_id))
        for arm in settings["arms"]:
            arm_id = _component(arm["arm_id"], "arm_id")
            identity = {"task_id": task_id, "repeat_index": repeat_index,
                        "pair_id": pair_id, "arm_id": arm_id}
            path = directory / task_id / arm_id / "task_result.json"
            row = {**identity, "execution_mode": arm["execution_mode"],
                   "policy_version": arm["policy_version"], "status": "unknown",
                   "score": None, "score_state": "missing_result", "source": str(path)}
            if path.exists():
                result = _object(path)
                recorded_identity = {**identity, "execution_mode": arm["execution_mode"],
                                     "policy_version": arm["policy_version"]}
                for field, expected in recorded_identity.items():
                    if field in result and result[field] != expected:
                        raise ValueError(f"Result {field} disagrees with run settings: {path}")
                score = _score(result.get("score"))
                row.update(status=result.get("status", "unknown"), score=score,
                           score_state=("known" if score is not None else
                                        "missing_score" if result.get("score") is None else "invalid_score"))
            if layer_rules is not None:
                layers = load_score_layers(path.parent, layer_rules.get(task_id),
                                           task_id=task_id, score=row["score"])
                row.update(score_layers=layers["score_layers"], layer_coverage=layers["coverage"])
            rows.append(row)
    provider = settings.get("provider")
    public_provider = ({key: provider[key] for key in (
        "model", "base_url", "max_output_tokens", "stream_response", "max_attempts",
        "request_timeout_seconds", "first_progress_timeout_seconds", "progress_idle_timeout_seconds",
        "retry_backoff_seconds", "extra_body", "rates") if key in provider}
        if isinstance(provider, dict) else None)
    provenance = {"run_directory": str(directory), "profile_path": settings.get("profile_path"),
                  "inference": settings.get("inference"), "evaluation": settings.get("evaluation"),
                  "provider": public_provider, "runtime": settings.get("runtime"),
                  "platform": settings.get("platform"),
                  **{key: settings.get(key) for key in (
                      "parallelism", "waves", "project_root", "qwen_package", "node_directory",
                      "docker", "provider_config")},
                  "arms": settings["arms"]}
    return rows, provenance


# Compared across the sources contributing to one task. Run directory, profile
# path and round identity differ by design and are excluded. Equal recorded
# paths do not prove identical code or file content behind them.
_SOURCE_CONDITIONS = ("inference", "evaluation", "provider", "runtime", "platform", "parallelism",
                      "project_root", "qwen_package", "node_directory")
_ARM_CONDITIONS = ("execution_mode", "policy_version", "assignment_policy_path", "communication_policy_path")
_SIGN_COUNTS = ("positive_count", "negative_count", "zero_count")
_UNRECORDED = object()


def _shared_wave(waves, task_id: str, selected: tuple[str, str]):
    if not isinstance(waves, list) or not all(
            isinstance(wave, list) and all(isinstance(item, dict) for item in wave) for wave in waves):
        return _UNRECORDED
    return any(set(selected) <= {item.get("arm_id") for item in wave if item.get("task_id") == task_id}
               for wave in waves)


def _recorded_conditions(source: dict, task_id: str, selected: tuple[str, str]) -> dict:
    # A null top-level field was not recorded; a null Single policy path was.
    values = {field: _UNRECORDED if source.get(field) is None else source[field]
              for field in _SOURCE_CONDITIONS}
    values["selected_arms_shared_wave"] = _shared_wave(source.get("waves"), task_id, selected)
    arms = {arm.get("arm_id"): arm for arm in source["arms"] if isinstance(arm, dict)}
    for arm_id in selected:
        # An arm outside this round's plan contributes no arm conditions; its
        # absence already shows as arm_not_in_round_plan and an unshared wave.
        if arm_id in arms:
            values.update({f"arms.{arm_id}.{field}": arms[arm_id].get(field, _UNRECORDED)
                           for field in _ARM_CONDITIONS})
    return values


def _conditions(sources: list[dict], task_id: str, selected: tuple[str, str]) -> dict:
    recorded = [_recorded_conditions(source, task_id, selected) for source in sources]
    fields = [*_SOURCE_CONDITIONS, "selected_arms_shared_wave",
              *(f"arms.{arm_id}.{field}" for arm_id in selected for field in _ARM_CONDITIONS)]
    differing, unrecorded = [], []
    for field in fields:
        values = [item[field] for item in recorded if field in item]
        known = [value for value in values if value is not _UNRECORDED]
        # Canonical JSON keeps e.g. true and 1 distinct.
        if len({json.dumps(value, sort_keys=True) for value in known}) > 1:
            differing.append(field)
        elif not known or len(known) != len(values):
            unrecorded.append(field)
    status = ("different_recorded_conditions" if differing else
              "unknown" if unrecorded else "same_recorded_conditions")
    return {"status": status, "differing_fields": differing, "unrecorded_fields": unrecorded,
            "contributing_sources": len(recorded)}


def _statistics(values: list[float | None], *, signs=False) -> dict:
    observed = [value for value in values if value is not None]
    complete = bool(values) and len(observed) == len(values)
    observed_stats = {
        "mean": statistics.fmean(observed) if observed else None,
        "minimum": min(observed) if observed else None,
        "maximum": max(observed) if observed else None,
        "range": max(observed) - min(observed) if observed else None,
        "sample_variance": statistics.variance(observed) if len(observed) >= 2 else None,
    }
    if signs:
        observed_stats.update(positive_count=sum(value > 0 for value in observed),
                              negative_count=sum(value < 0 for value in observed),
                              zero_count=sum(value == 0 for value in observed))
    stats = {"expected_count": len(values), "known_count": len(observed),
             "unknown_count": len(values) - len(observed), "complete": complete,
             # Unknown scores cannot silently turn an incomplete series into a
             # complete-series mean/variance/sign count. Available-case values
             # are labeled; minimum/maximum/range exist only there.
             "mean": observed_stats["mean"] if complete else None,
             "sample_variance": observed_stats["sample_variance"] if complete else None}
    if signs:
        stats.update({name: observed_stats[name] if complete else None for name in _SIGN_COUNTS})
    stats["observed_scores_only"] = observed_stats
    return stats


def _withhold(stats: dict) -> dict:
    """Null common-series values; observed_scores_only keeps the arithmetic."""
    stats["mean"] = stats["sample_variance"] = None
    stats.update({name: None for name in _SIGN_COUNTS if name in stats})
    return stats


def _comparability(universes: list[frozenset[str] | None]) -> dict:
    known = [value for value in universes if value is not None]
    if len(set(known)) > 1:
        status, reason = "not_comparable", "different_official_test_sets"
    elif not universes or len(known) != len(universes):
        status, reason = "unknown", "official_test_sets_not_fully_verified"
    else:
        status, reason = "comparable", "same_official_test_set"
    return {"status": status, "reason": reason,
            "known_universes": len(known), "expected_universes": len(universes)}


def _comparison_statistics(values: list[float | None], comparability: dict, conditions: dict, *,
                           signs=False) -> dict:
    stats = _statistics(values, signs=signs)
    stats["comparability"] = comparability
    stats["observed_scores_only"].update(comparability=comparability["status"],
                                         conditions=conditions["status"])
    if (comparability["status"] == "not_comparable"
            or conditions["status"] == "different_recorded_conditions"):
        # Scores remain complete observations, but different test sets or
        # recorded conditions cannot form a common B noise/comparison series.
        # Arithmetic is retained only in explicitly labeled observed_scores_only
        # diagnostics.
        _withhold(stats)
    return stats


def summarize_comparisons(directories: Iterable[Path], *, left_arm: str, right_arm: str,
                          expected_repeat_indices: Iterable[int] | None = None,
                          layer_rule_paths: Iterable[Path] | None = None) -> dict:
    """Summarize right-minus-left by task and explicit round, never file order.

    Supply expected_repeat_indices to expose a wholly missing round. Otherwise
    completeness describes only the profiles supplied, not a preplanned K.
    A duplicate task/round/arm is ambiguous (e.g. an original and a retry), so
    callers must report those series separately rather than pick a winner. A
    pair_id must label exactly one task/round; a reused or stale one is rejected.
    """
    if left_arm == right_arm:
        raise ValueError("left_arm and right_arm must differ")
    selected = (_component(left_arm, "left_arm"), _component(right_arm, "right_arm"))
    layer_rules = None
    if layer_rule_paths is not None:
        layer_rules = {}
        for path in layer_rule_paths:
            rule = load_rules(path)
            if rule["task_id"] in layer_rules:
                raise ValueError("Supply one layer rule per task; report different rules separately")
            layer_rules[rule["task_id"]] = rule
    expected = None
    if expected_repeat_indices is not None:
        expected = [_round(value) for value in expected_repeat_indices]
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("Expected repeat indices must be nonempty and unique")
        expected.sort()
    rows, sources, by_identity, pair_ids, pair_owners, universes = [], [], {}, {}, {}, {}
    supplied_rounds, task_sources = set(), {}
    for directory in directories:
        loaded, source = load_comparison_results(directory, layer_rules=layer_rules)
        sources.append(source)
        for row in loaded:
            supplied_rounds.add((row["task_id"], row["repeat_index"]))
            if row["arm_id"] not in selected:
                continue
            key = (row["task_id"], row["repeat_index"], row["arm_id"])
            if key in by_identity:
                raise ValueError(f"Duplicate task/round/arm {key}; report retries or condition changes separately")
            pair_key = key[:2]
            if pair_key in pair_ids and pair_ids[pair_key] != row["pair_id"]:
                raise ValueError(f"Conflicting pair_id for task/round {pair_key}")
            if pair_owners.setdefault(row["pair_id"], pair_key) != pair_key:
                raise ValueError(f"pair_id {row['pair_id']!r} labels more than one task/round; "
                                 "report stale or reused pair IDs separately")
            if expected is not None and row["repeat_index"] not in expected:
                raise ValueError(f"Observed repeat_index {row['repeat_index']} is outside expected rounds")
            pair_ids[pair_key] = row["pair_id"]
            by_identity[key] = row
            contributing = task_sources.setdefault(row["task_id"], [])
            if not any(item is source for item in contributing):
                contributing.append(source)
            # Independent of layer rules: only test identifiers and official
            # totals are read; a run without the sidecar stays unverified.
            universes[key] = official_test_universe(Path(row["source"]).parent, row["task_id"], score=row["score"])
            rows.append(row)
    if not sources or not rows:
        raise ValueError("No comparison records supplied for the selected arms")
    for arm_id in selected:
        if not any(row["arm_id"] == arm_id for row in rows):
            raise ValueError(f"Arm {arm_id!r} is absent from every supplied plan")
    tasks = []
    for task_id in sorted({row["task_id"] for row in rows}):
        rounds = expected if expected is not None else sorted(
            {row["repeat_index"] for row in rows if row["task_id"] == task_id})
        pairs = []
        for repeat_index in rounds:
            pair_id = pair_ids.get((task_id, repeat_index))
            arms = {}
            for arm_id in selected:
                row = by_identity.get((task_id, repeat_index, arm_id))
                if row is None:
                    # A supplied round that never planned this arm differs from
                    # a round with no supplied directory for this task at all.
                    state = ("arm_not_in_round_plan" if (task_id, repeat_index) in supplied_rounds
                             else "missing_planned_round")
                    row = {"task_id": task_id, "repeat_index": repeat_index, "pair_id": pair_id,
                           "arm_id": arm_id, "status": "unknown", "score": None,
                           "score_state": state, "source": None}
                arms[arm_id] = row
            left, right = (arms[arm]["score"] for arm in selected)
            difference = right - left if left is not None and right is not None else None
            comparable = _comparability([universes.get((task_id, repeat_index, arm)) for arm in selected])
            pairs.append({"task_id": task_id, "repeat_index": repeat_index, "pair_id": pair_id,
                          "arms": arms, "observed_difference": difference, "comparability": comparable,
                          "difference": difference if comparable["status"] != "not_comparable" else None})
        differences = [pair["observed_difference"] for pair in pairs]
        comparable = _comparability([universes.get((task_id, repeat_index, arm))
                                     for repeat_index in rounds for arm in selected])
        conditions = _conditions(task_sources[task_id], task_id, selected)
        task_report = {"task_id": task_id, "pairs": pairs, "comparability": comparable,
                      "conditions": conditions,
                      "arms": {arm: _comparison_statistics(
                          [pair["arms"][arm]["score"] for pair in pairs],
                          _comparability([universes.get((task_id, repeat_index, arm)) for repeat_index in rounds]),
                          conditions)
                               for arm in selected},
                      "difference": _comparison_statistics(differences, comparable, conditions, signs=True)}
        if layer_rules is not None:
            task_report["score_layers"] = _layer_statistics(pairs, selected, conditions)
        tasks.append(task_report)
    return {"schema_version": 1, "kind": "descriptive_programbench_repeats",
            "left_arm": left_arm, "right_arm": right_arm, "difference": "right_minus_left",
            "coverage_scope": "expected_rounds" if expected is not None else "supplied_profiles_only",
            "expected_repeat_indices": expected, "sources": sources, "tasks": tasks,
            "limitations": ["No acceptance decision, delta threshold or error-rate guarantee.",
                            "Score completeness and test-set comparability are separate. Known different official test sets are not a common B comparison/noise series; their arithmetic survives only as observed diagnostics. Unknown test sets leave historical descriptive statistics unverified. Per-arm comparability is assessed separately.",
                            "Recorded conditions are compared per task; equal recorded paths (project root, Qwen package, policy files) do not prove identical code or file content, and equal effective runtime is not established by this summary. Different conditions must not be interpreted as same-configuration variance.",
                            "Zero sample variance means no variation observed in this sample, not zero population noise.",
                            "Missing scores stay unknown; valid scores, including zero, survive failed execution status."]}


def _layer_statistics(pairs: list[dict], selected: tuple[str, str], conditions: dict) -> dict:
    """Same explicit per-task rule/universe for all rounds; unknowns propagate.

    Different recorded conditions withhold the same main values as total scores.
    """
    def described(values):
        stats = _statistics(values)
        stats["observed_scores_only"]["conditions"] = conditions["status"]
        return _withhold(stats) if conditions["status"] == "different_recorded_conditions" else stats

    reports = {}
    for layer in LAYERS:
        values = {arm: [] for arm in selected}
        denominators, versions = set(), set()
        for pair in pairs:
            for arm in selected:
                report = pair["arms"][arm].get("score_layers")
                row = report["layers"][layer] if report else None
                if report:
                    versions.add(report["rule_version"])
                    denominators.add(row["total"])
                values[arm].append(row["fraction"] if row else None)
        same_measure = len(versions) == len(denominators) == 1
        if not same_measure:
            values = {arm: [None] * len(pairs) for arm in selected}
        differences = [right - left if left is not None and right is not None else None
                       for left, right in zip(values[selected[0]], values[selected[1]])]
        reports[layer] = {"rule_version": next(iter(versions)) if len(versions) == 1 else None,
                          "denominator": next(iter(denominators)) if len(denominators) == 1 else None,
                          "arms": {arm: described(scores) for arm, scores in values.items()},
                          "difference": described(differences)}
    return reports
