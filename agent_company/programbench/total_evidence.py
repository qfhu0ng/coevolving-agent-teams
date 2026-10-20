"""Bind total-score arithmetic to existing run IDs, policies and official test sets.

No new digest or contract: use saved primary identities and compare existing
metadata. Equal counts alone cannot establish a same-task, same-condition pair.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from agent_company.analysis.score_layers import official_test_universe
from agent_company.deploy.code_version import known_code_version
from agent_company.programbench.total_selection import compare_totals, confirm_totals, official_counts


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {Path(path).name}")
    return value


def trial_record(path):
    path = Path(path).resolve(strict=True)
    request, result = _read(path / "run_request.json"), _read(path / "task_result.json")
    settings = _read(path.parent.parent / "run_settings.json")
    task = request["task"]
    task_id, repo = task["task_id"], task["repo_id"]
    if not task_id or not repo or not official_counts(result):
        raise ValueError("Incomplete official score or identity")
    universe = official_test_universe(path, task_id, score=result["score"])
    if universe is None or len(universe) != official_counts(result)[1]:
        raise ValueError("Unknown official test universe")
    expected = {"task_id": task_id, "repo_id": repo, "arm_id": path.name,
                "execution_mode": request["execution_mode"], "policy_version": request["policy_version"]}
    if path.parent.name != task_id:
        raise ValueError("Task directory disagrees with request")
    for name, value in expected.items():
        if name in result and result[name] != value or name in request and request[name] != value:
            raise ValueError(f"Recorded {name} disagrees with the requested run")
    arm = None
    if "arms" in settings:
        selected = [arm for arm in settings["arms"] if arm.get("arm_id") == path.name]
        if len(selected) != 1 or any(selected[0].get(k) != expected[k] for k in ("execution_mode", "policy_version")):
            raise ValueError("Arm settings disagree with request")
        arm = selected[0]
    if "tasks" in settings and not any(t.get("task_id") == task_id and t.get("repository_id") == repo
                                       for t in settings["tasks"]):
        raise ValueError("Task settings disagree with request")
    task_settings = [t for t in settings.get("tasks", []) if t.get("task_id") == task_id]
    for key in ("repeat_index", "pair_id"):
        values = [record[key] for record in (request, result, settings, *task_settings) if key in record]
        if values and any(value != values[0] for value in values):
            raise ValueError(f"Recorded {key} disagrees across run artifacts")
    inference = dict(settings["inference"])
    mode = inference.pop("role_policy_mode", "off")
    execution = request["execution_mode"]
    if execution not in {"single", "team"}:
        raise ValueError("Unknown execution mode")
    package_condition = None
    if inference.get("role_policy_protocol") is not None:
        from agent_company.policy.role_package import (
            PACKAGE_PROTOCOLS, PROTOCOL_MODES, package_policy_identity, policy_protocol, validate_policy_package)
        if inference["role_policy_protocol"] not in PACKAGE_PROTOCOLS:
            raise ValueError("Unknown recorded role-policy protocol")
        package = validate_policy_package(request.get("policy_package"))
        if policy_protocol(package) != inference["role_policy_protocol"]:
            raise ValueError("Recorded package protocol disagrees with the recorded inference protocol")
        if arm is None or arm.get("policy_package") != package:
            raise ValueError("Arm package disagrees with request")
        if execution == "team":
            mode = request.get("role_policy_mode")
            if mode not in PROTOCOL_MODES[inference["role_policy_protocol"]] or arm.get("role_policy_mode") != mode:
                raise ValueError("Arm effective mode disagrees with request")
            if "role_policy_mode" in result and result["role_policy_mode"] != mode:
                raise ValueError("Result effective mode disagrees with request")
            if (request.get("assignment_policy") != package["team"]["assignment"]
                    or request.get("communication_policy") != package["team"]["communication"]):
                raise ValueError("Run policy text disagrees with its package")
            package_condition = package_policy_identity(package)
        else:
            if request.get("role_policy_mode") is not None or request.get("assignment_policy") or request.get("communication_policy"):
                raise ValueError("Single request applies unexpected Team policies")
            package_condition = {"protocol": package["protocol"]}
    elif request.get("role_policy_mode") is not None or request.get("policy_package") is not None:
        raise ValueError("Per-arm policy settings require recorded role_package_v1")
    shared = {"task_id": task_id, "repo_id": repo, "inference": inference,
              "provider": settings["provider"], "evaluation": settings["evaluation"]}
    # Existing launch repairs distinguish the known automatic-report condition
    # change. Equal recorded paths alone do not establish identical source bytes.
    native_path = path / execution / "launch.json"
    native = _read(native_path) if native_path.is_file() else None
    if package_condition is not None:
        expected_native = {"role_policy_protocol": inference["role_policy_protocol"],
                           "policy_package": package,
                           "role_policy_mode": mode if execution == "team" else None}
        if native is None or any(native.get(key) != value for key, value in expected_native.items()):
            raise ValueError("Actual native launch package, mode or protocol disagrees with request")
    native_runtime = ({key: native.get(key) for key in ("native_compatibility_repairs", "team_message_delivery",
        "runtime_disclosure", "leader_loop_recovery", "tool_call_cap", "native_team_inactivity_timeout_seconds",
        "role_policy_disclosure_version", "stop_hook")} if native is not None else None)
    provenance = {key: settings.get(key) for key in ("runtime", "platform", "code_version", "qwen_version")}
    condition = {**shared, "runtime_provenance": provenance, "native_runtime": native_runtime, "execution_mode": execution, "role_policy_mode": mode if execution == "team" else None,
                 "policy_version": request["policy_version"],
                 "assignment_policy": request.get("assignment_policy"),
                 "communication_policy": request.get("communication_policy")}
    if package_condition is not None:
        condition["policy_package"] = package_condition
    return {"path": path, "result": result, "request": request, "settings": settings,
            "shared": shared, "condition": condition, "universe": universe}


def _comparable(a, b):
    if a["shared"] != b["shared"] or a["universe"] != b["universe"]:
        raise ValueError("Different task, runtime/provider/resources or official test universe")
    ca, cb = a["condition"], b["condition"]
    if any(not known_code_version(c["runtime_provenance"].get("code_version")) for c in (ca, cb)):
        raise ValueError("Unknown recorded code version")
    if any(not isinstance(c["runtime_provenance"].get("qwen_version"), str)
           or not c["runtime_provenance"]["qwen_version"].strip()
           or c["runtime_provenance"]["qwen_version"].strip().lower() == "unknown" for c in (ca, cb)):
        raise ValueError("Unknown recorded Qwen version")
    if ca["runtime_provenance"] != cb["runtime_provenance"]:
        raise ValueError("Different recorded runtime provenance")
    if ca["execution_mode"] == cb["execution_mode"] and ca["native_runtime"] != cb["native_runtime"]:
        raise ValueError("Different native runtime repairs/settings")
    if ca["execution_mode"] == cb["execution_mode"] == "team" and ca["role_policy_mode"] != cb["role_policy_mode"]:
        raise ValueError("Different Team modes are descriptive only")


def _concurrent(a, b):
    if a["path"].parent.parent != b["path"].parent.parent or a["path"] == b["path"]:
        return False
    if any(a["request"].get(key) != b["request"].get(key) for key in ("repeat_index", "pair_id")):
        return False
    plan = a["settings"]
    keys = {(a["shared"]["task_id"], a["path"].name), (b["shared"]["task_id"], b["path"].name)}
    if plan.get("parallelism") != 2 or not any(
            keys <= {(r["task_id"], r["arm_id"]) for r in wave} for wave in plan.get("waves", [])):
        return False
    rows = _read(a["path"].parent.parent / "progress.json").get("rows", [])
    rows = [r for r in rows if (r.get("task_id"), r.get("arm_id")) in keys]
    if len(rows) != 2 or {(r.get("task_id"), r.get("arm_id")) for r in rows} != keys:
        return False
    # Existing dispatch/finish timestamps establish overlap; no new timing gate.
    starts = [datetime.fromisoformat(r["dispatched_at"]) for r in rows]
    ends = [datetime.fromisoformat(r["finished_at"]) for r in rows]
    return all(start < end for start, end in zip(starts, ends)) and max(starts) < min(ends)


def compare_trials(incumbent, candidate, *, history_paths=()):
    try:
        a, b = trial_record(incumbent), trial_record(candidate)
        if a["path"] == b["path"]:
            raise ValueError("A run cannot be compared with itself")
        _comparable(a, b)
        history = {r["path"]: r for r in (a, b)}
        for path in history_paths:
            record = trial_record(path)
            history[record["path"]] = record
        groups = []
        for arm in (a, b):
            groups.append([official_counts(r["result"])[0] for r in history.values()
                           if arm["condition"]["execution_mode"] == "team"
                           and arm["condition"]["native_runtime"] is not None
                           and r["condition"] == arm["condition"] and r["universe"] == arm["universe"]])
        report = compare_totals(a["result"], b["result"], same_condition_team_counts=groups[0],
                                other_condition_team_counts=groups[1])
        return {**report, "incumbent_trial": str(a["path"]), "candidate_trial": str(b["path"]),
                "task_id": a["shared"]["task_id"],
                "incumbent_execution_mode": a["condition"]["execution_mode"],
                "candidate_execution_mode": b["condition"]["execution_mode"],
                "candidate_role_policy_mode": b["condition"]["role_policy_mode"], "concurrent_pair": _concurrent(a, b),
                "comparison_time_scope": "concurrent" if _concurrent(a, b) else "different_time_windows",
                "team_history_counts": list(map(len, groups)), "test_universe_matched": True,
                "conditions_scope": "saved_code_version_runtime_and_policy_metadata; deployment_paths_ignored"}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"decision": "inconclusive", "reason": str(exc), "adopted": False}


def series_repeat_indices(series_root, *, exclude_run_roots=()):
    """Read allocated indices from a paid_runs series, including unscored runs.

    The launcher calls this before allocating a new step. Confirmation excludes
    its own runner output, whose two arms intentionally share one index. Explicit
    offline series may also contain runner outputs directly, without step/run/.
    """
    root = Path(series_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Series root must be a directory")
    excluded = {Path(path).resolve(strict=True) for path in exclude_run_roots}
    indices = {}
    for step in sorted(root.iterdir()):
        if not step.is_dir():
            continue
        run = step / "run" if (step / "run").is_dir() else step
        if run.resolve() in excluded:
            continue
        records = [path for path in (step / "launch.json", run / "run_settings.json") if path.is_file()]
        records.extend(sorted(run.glob("*/*/run_request.json")))
        if not records:
            raise ValueError(f"Series run has no readable allocation record: {step.name}")
        values = set()
        for path in records:
            value = _read(path).get("repeat_index")
            if type(value) is not int or value < 0:
                raise ValueError(f"Unknown repeat_index in series record: {path.relative_to(root)}")
            values.add(value)
        if len(values) != 1:
            raise ValueError(f"Conflicting repeat_index records in series run: {step.name}")
        indices.setdefault(values.pop(), []).append(str(run.resolve()))
    return indices


def _confirmation_series(records, series_root):
    inferred = [next((parent for parent in record["path"].parents if parent.name == "paid_runs"), None)
                for record in records]
    if series_root is None:
        if not any(inferred):
            return None
        if None in inferred or len(set(inferred)) != 1:
            raise ValueError("Confirmation and trigger must belong to the same paid_runs series")
        return inferred[0]
    root = Path(series_root).resolve(strict=True)
    if any(not record["path"].is_relative_to(root) for record in records):
        raise ValueError("Confirmation and trigger must be within the selected series root")
    if any(parent is not None and parent != root for parent in inferred):
        raise ValueError("Explicit series root must match the paid_runs series")
    return root


def confirm_trials(trigger, incumbent, candidate, *, series_root=None):
    try:
        a, b = trial_record(incumbent), trial_record(candidate)
        old_a, old_b = trial_record(trigger["incumbent_trial"]), trial_record(trigger["candidate_trial"])
        if len({r["path"] for r in (a, b, old_a, old_b)}) != 4:
            raise ValueError("Confirmation requires two new runs")
        old_rounds = {r["request"]["repeat_index"] for r in (old_a, old_b) if "repeat_index" in r["request"]}
        if any(type(r["request"].get("repeat_index")) is not int or r["request"]["repeat_index"] < 1 for r in (a, b)):
            raise ValueError("Confirmation requires a positive repeat_index")
        if any(r["request"].get("repeat_index") in old_rounds for r in (a, b)):
            raise ValueError("Confirmation must use a new repeat_index")
        root = _confirmation_series((a, b, old_a, old_b), series_root)
        if root is not None:
            indices = series_repeat_indices(root, exclude_run_roots=(a["path"].parent.parent, b["path"].parent.parent))
            for record in (a, b):
                index = record["request"].get("repeat_index")
                if index in indices:
                    raise ValueError("Confirmation repeat_index was already allocated in this series")
        _comparable(a, b)
        if any(new["condition"] != old["condition"] or new["universe"] != old["universe"]
               for new, old in ((a, old_a), (b, old_b))):
            raise ValueError("Confirmation does not match the triggering arm conditions")
        report = confirm_totals(trigger, a["result"], b["result"], concurrent_pair=_concurrent(a, b))
        return {**report, "trigger_trials": [str(old_a["path"]), str(old_b["path"])],
                "confirmation_trials": [str(a["path"]), str(b["path"])],
                "repeat_index_scope": "all_recorded_runs_in_series" if root is not None else "trigger_trials_only_no_series_context",
                "series_root": str(root) if root is not None else None}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"decision": "inconclusive", "reason": str(exc), "adopted": False}
