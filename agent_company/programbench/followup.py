"""Read-only planning for one evolved follow-up from a completed Single/Team pair.

Shared by scripts/run_programbench_evolved_followup.py and its launcher. It reads
public inputs and source identity/outcomes only; it never loads credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from agent_company.runtime.provider import public_provider
from agent_company.evolution.loop import TaskResult
from agent_company.programbench.study import PreparedStudy, _valid_score, prepare_study
from agent_company.runtime.relay import ProviderConfig, TokenRates
from agent_company.programbench.total_selection import MODE as TOTAL_MODE


@dataclass
class PreparedFollowup:
    plan: dict
    study: PreparedStudy
    source_trial: Path
    baseline_result: dict


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _official_score(result: dict) -> bool:
    verifier = result.get("metrics", {}).get("verifier", {})
    reward = verifier.get("reward")
    return (_valid_score(TaskResult(status=result.get("status"), score=result.get("score")))
            and verifier.get("status") == "completed"
            and not isinstance(reward, bool) and isinstance(reward, (int, float))
            and math.isfinite(reward) and reward == result["score"])


def prepare_followup(profile_path: Path, project_root: Path, source_run: Path) -> PreparedFollowup:
    """Read public inputs and source identity/outcomes; never load credentials."""
    repeat_single = _read(Path(profile_path)).get("repeat_single", False)
    if not isinstance(repeat_single, bool):
        raise ValueError("repeat_single must be a boolean")
    study = prepare_study(profile_path, project_root)
    source_run = Path(source_run).resolve(strict=True)
    source_settings = _read(source_run / "run_settings.json")
    total_protocol = study.plan.get("selection_protocol") == TOTAL_MODE
    source_count = len(source_settings.get("arms", []))
    progress = _read(source_run / "progress.json")
    if (progress.get("state") != "completed" or progress.get("terminal") != source_count
            or progress.get("planned") != source_count or progress.get("running") or progress.get("pending")
            or progress.get("cleanup_attention") or progress.get("requires_user_action")):
        raise ValueError("Source pair must be completed with both inference cleanups confirmed")
    task = study.task
    source_tasks = source_settings.get("tasks", [])
    if (len(source_tasks) != 1 or source_tasks[0].get("task_id") != task.task_id
            or source_tasks[0].get("repository_id") != task.repository_id
            or source_tasks[0].get("image") != task.image):
        raise ValueError("Source task does not match the selected public task")
    baseline = study.plan["arms"][1]
    single = study.plan["arms"][0]
    expected = ({baseline["arm_id"]: baseline} if total_protocol and source_count == 1
                else {single["arm_id"]: single, baseline["arm_id"]: baseline})
    source_arms = source_settings.get("arms", [])
    if len(source_arms) != len(expected) or {a.get("arm_id") for a in source_arms} != set(expected):
        raise ValueError("Source must be the selected Single/baseline Team pair")
    for arm in source_arms:
        wanted = expected[arm["arm_id"]]
        if any(arm.get(key) != wanted[key] for key in ("execution_mode", "policy_version")):
            raise ValueError("Source arm mode/version mismatch")
    rows = progress.get("rows", [])
    if (len(rows) != len(expected) or {row.get("arm_id") for row in rows} != set(expected)
            or any(row.get("task_id") != task.task_id
                   or row.get("inference_cleanup_confirmed") is not True for row in rows)):
        raise ValueError("Source pair cleanup records are missing or mismatched")
    if source_settings.get("inference") != study.plan["inference"]:
        raise ValueError("Follow-up inference settings must match the source pair")
    if source_settings.get("evaluation", {}).get("docker_cpus") != study.plan["evaluation"]["docker_cpus"]:
        raise ValueError("Follow-up evaluator resources must match the source pair")
    provider_config = _read(Path(study.plan["provider_config"]))
    if "api_key" in provider_config:
        raise ValueError("Provider config must use a named environment credential")
    provider_values = {key: value for key, value in provider_config.items() if key != "api_key_env"}
    if provider_values.get("rates") is not None:
        provider_values["rates"] = TokenRates(**provider_values["rates"])
    # Normalize defaults without resolving api_key_env. Omitting a setting must
    # not silently switch the follow-up back to a different runtime default.
    expected_provider = public_provider(ProviderConfig(**provider_values))
    if source_settings.get("provider") != expected_provider:
        raise ValueError("Follow-up provider settings must match the source pair")
    source_trial = source_run / task.task_id / baseline["arm_id"]
    request = _read(source_trial / "run_request.json")
    if (request.get("execution_mode") != "team" or request.get("policy_version") != baseline["policy_version"]
            or request.get("fresh_start") is not True
            or request.get("task", {}).get("task_id") != task.task_id
            or request.get("task", {}).get("repo_id") != task.repository_id):
        raise ValueError("Source baseline Team request identity does not match")
    policies = study.policies[baseline["arm_id"]]
    if (request.get("assignment_policy"), request.get("communication_policy")) != policies:
        raise ValueError("Source baseline Team policy texts do not match")
    baseline_result = _read(source_trial / "task_result.json")
    if not _official_score(baseline_result):
        raise ValueError("Source baseline Team needs a valid completed official score; missing is not zero")
    if baseline_result.get("metrics", {}).get("inference_cleanup_confirmed") is not True:
        raise ValueError("Source baseline Team inference cleanup is not confirmed")
    evolved = study.plan["arms"][2]
    arms = [single, evolved] if repeat_single else [evolved]
    plan = {**study.plan, "provider": expected_provider, "arms": arms, "parallelism": len(arms), "repeat_single": repeat_single,
            "waves": [[{"task_id": task.task_id, "arm_id": a["arm_id"]} for a in arms]],
            "new_rollouts": len(arms), "reused_rollouts": 0, "training_rollouts_reused": 1,
            "comparison_baseline_references": len(expected), "source_run": str(source_run),
            "source_trial": str(source_trial),
            "source_baselines": [{"task_id": task.task_id, "arm_id": a["arm_id"],
                                  "execution_mode": a["execution_mode"],
                                  "path": str(source_run / task.task_id / a["arm_id"]),
                                  "used_for_evolution": a["execution_mode"] == "team"}
                                 for a in expected.values()],
            "study_mode": "same_task_qualitative_evolved_followup",
            "source_workspace_reused": False,
            "interpretation": ("A fresh Single and candidate Team run concurrently after review. "
                               if repeat_single else "One fresh candidate Team run after review. ") +
                              "Only the selected source Team informs evolution; source Single is comparison-only. "
                              "The source and follow-up use different API time windows; "
                              "not held-out validation, causal proof, or automatic adoption."}
    return PreparedFollowup(plan, study, source_trial, baseline_result)
