"""Descriptive score layers from explicit rules and filtered evaluator metadata.

Standard library only. Never loads raw eval JSON, tests.json, evaluator code,
trajectories or test bodies. No default classification or selection rule.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path


STATUSES = {"passed", "skipped", "failure", "error", "system_error", "not_run"}
LAYERS = ("core", "information")
_UNSET = object()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _label(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_rules(rules: dict) -> dict:
    """Keep only exact IDs, module/layer assignments and the optional FATE tag."""
    if (not isinstance(rules, dict) or not _label(rules.get("task_id"))
            or not _label(rules.get("rule_version")) or not isinstance(rules.get("tests"), dict)
            or not rules["tests"]):
        raise ValueError("Layer rules require task_id, rule_version and nonempty tests mapping")
    tests = {}
    for test_id, row in rules["tests"].items():
        if (not _label(test_id) or not isinstance(row, dict) or not _label(row.get("module"))
                or row.get("layer") not in LAYERS):
            raise ValueError("Each rule needs an exact test ID, module and core/information layer")
        tags = row.get("tags", [])
        if (not isinstance(tags, list) or any(tag != "fate" for tag in tags)
                or len(set(tags)) != len(tags) or (tags and row["layer"] != "core")):
            raise ValueError("FATE is an optional core subset, not an additional score layer")
        tests[test_id] = {"module": row["module"], "layer": row["layer"], "tags": tags}
    return {"task_id": rules["task_id"], "rule_version": rules["rule_version"], "tests": tests}


def load_rules(path: str | Path) -> dict:
    return validate_rules(_object(Path(path)))


def _sidecar_path(run_dir: str | Path) -> Path:
    root = Path(run_dir).resolve()
    path = root / "verifier/filtered_test_results.json"
    if any(root.joinpath(*path.relative_to(root).parts[:index]).is_symlink()
           for index in (1, 2)):
        raise ValueError("Unsafe filtered result path")
    return path


def _filtered(run_dir: str | Path) -> dict:
    value = _object(_sidecar_path(run_dir))
    if value.get("schema_version") != 1 or not _label(value.get("task_id")):
        raise ValueError("Invalid filtered result metadata")
    rows = value.get("tests")
    if not isinstance(rows, list) or any(not isinstance(row, dict)
            or not _label(row.get("test_id")) or row.get("status") not in STATUSES for row in rows):
        raise ValueError("Invalid filtered test records")
    return value


def names_only(run_dir: str | Path) -> dict:
    """Identifiers and record counts only: no per-test status, reward, error or pass count."""
    try:
        data = _filtered(run_dir)
    except FileNotFoundError:
        raise ValueError("Missing verifier/filtered_test_results.json; names-only export needs the "
                         "filtered evaluator sidecar, which older results do not have") from None
    ids = [row["test_id"] for row in data["tests"]]
    return {"schema_version": 1, "task_id": data["task_id"],
            "official_status": data.get("official_status"),
            "record_count": len(ids), "duplicate_count": len(ids) - len(set(ids)),
            "test_ids": sorted(set(ids))}


def official_test_universe(run_dir: str | Path, task_id: str, *, score) -> frozenset[str] | None:
    """Internal comparison metadata, independent of classification rules.

    Reads test identifiers and official totals only; per-test statuses are
    neither validated nor returned, and an absent sidecar is never opened.
    Callers must not serialize this set into score reports. Incomplete or
    inconsistent artifacts cannot establish a complete official test universe.
    """
    try:
        path = _sidecar_path(run_dir)
        if not path.is_file():
            return None
        data = _object(path)
        rows = data.get("tests")
        if (data.get("schema_version") != 1 or data.get("task_id") != task_id
                or not isinstance(rows, list)
                or any(not isinstance(row, dict) or not _label(row.get("test_id")) for row in rows)):
            return None
        ids = frozenset(row["test_id"] for row in rows)
        if (data.get("official_status") == "completed" and ids
                and _scores_match(score, data.get("reward"))
                and _count(data.get("valid_tests"))
                and data["valid_tests"] == len(rows) == len(ids)):
            return ids
    except (OSError, UnicodeError, ValueError, TypeError):
        pass
    return None


def _count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _reward(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def _scores_match(score, reward) -> bool:
    return (_reward(score) and _reward(reward)
            and math.isclose(score, reward, rel_tol=0, abs_tol=1e-12))


def summarize_layers(data: dict, rules: dict) -> dict:
    """No missing-record insertion or best-effort denominator substitution."""
    rules = validate_rules(rules)
    rows, assignments = data["tests"], rules["tests"]
    ids = [row["test_id"] for row in rows]
    observed, expected = set(ids), set(assignments)
    duplicate_count = len(ids) - len(observed)
    counts = Counter(row["status"] for row in rows)
    reasons = []
    if data["task_id"] != rules["task_id"]:
        reasons.append("task_mismatch")
    if duplicate_count:
        reasons.append("duplicate_test_ids")
    if observed - expected:
        reasons.append("unclassified_test_ids")
    if expected - observed:
        reasons.append("missing_test_ids")
    if (not _count(data.get("valid_tests")) or data["valid_tests"] != len(rows)
            or not _count(data.get("passed_tests")) or data["passed_tests"] != counts["passed"]):
        reasons.append("official_count_mismatch")
    # Incomplete official fractions are diagnostics, not scores. Submission
    # compile/copy failures can legitimately be complete zero with not_run rows.
    if (data.get("official_status") != "completed" or not rows or not _reward(data.get("reward"))):
        reasons.append("official_score_unknown")
    elif not math.isclose(data["reward"], counts["passed"] / len(rows), rel_tol=0, abs_tol=1e-12):
        reasons.append("official_reward_mismatch")
    if counts["system_error"] or (counts["not_run"] and data.get("submission_error") is not True):
        reasons.append("incomplete_test_statuses")
    known = not reasons

    def group(test_ids):
        matched = [row for row in rows if row["test_id"] in test_ids]
        statuses = dict(sorted(Counter(row["status"] for row in matched).items()))
        passed = statuses.get("passed", 0)
        return {"passed": passed if known else None, "total": len(test_ids),
                "fraction": passed / len(test_ids) if known and test_ids else None,
                "observed_only": {"records": len(matched), "passed": passed, "status_counts": statuses}}

    layers = {layer: group({key for key, row in assignments.items() if row["layer"] == layer})
              for layer in LAYERS}
    modules = {module: group({key for key, row in assignments.items() if row["module"] == module})
               for module in sorted({row["module"] for row in assignments.values()})}
    fate = {key for key, row in assignments.items() if "fate" in row["tags"]}
    return {"score_layers": {"task_id": rules["task_id"], "rule_version": rules["rule_version"],
                             "status": "known" if known else "unknown", "layers": layers,
                             "fate": group(fate) if fate else None, "by_module": modules,
                             "zero_pass_modules": [name for name, row in modules.items()
                                                   if row["passed"] == 0] if known else None},
            "coverage": {"status": "known" if known else "unknown", "reasons": reasons,
                         "official_status": data.get("official_status"),
                         "expected_tests": len(expected), "observed_records": len(rows),
                         "unclassified_tests": len(observed - expected),
                         "missing_tests": len(expected - observed), "duplicate_records": duplicate_count}}


def load_score_layers(run_dir: str | Path, rules: dict | None = None, *,
                      task_id: str | None = None, score=_UNSET) -> dict:
    """Read one safe metadata file only when rules were explicitly supplied."""
    def unknown(reason):
        return {"score_layers": None, "coverage": {"status": "unknown", "reasons": [reason]}}
    if rules is None:
        return unknown("missing_rules")
    try:
        data = _filtered(run_dir)
    except FileNotFoundError:
        return unknown("missing_filtered_results")
    except (OSError, UnicodeError, ValueError, TypeError):
        return unknown("invalid_filtered_results")
    if task_id is not None and data["task_id"] != task_id:
        return unknown("task_mismatch")
    if score is not _UNSET and not _scores_match(score, data.get("reward")):
        return unknown("task_score_unknown_or_mismatch")
    return summarize_layers(data, rules)
