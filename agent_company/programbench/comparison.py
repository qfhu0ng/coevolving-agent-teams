"""Validate a fresh ProgramBench policy comparison plan (shared by its CLI and launchers).

Planning reads only public tasks, instructions and policy files, never provider
credentials, and starts no process.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from agent_company.programbench.task import ProgramBenchTask
from agent_company.programbench.total_selection import MODE as TOTAL_MODE
from agent_company.runtime.qwen import validate_stop_hook
from agent_company.policy.role_package import (AGENT_RULES_PROTOCOL, PACKAGE_PROTOCOLS, PROTOCOL_MODES,
                                               ROLE_POLICY_PROTOCOL, load_policy_package, policy_protocol)


@dataclass
class PreparedComparison:
    plan: dict
    tasks: list[ProgramBenchTask]
    policies: dict[str, tuple[str, str]]


def _path(root: Path, value, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path")
    return (root / value).resolve()


def runner_inference_options(inference: dict) -> dict:
    """ProgramBenchRunner keywords for a plan's inference block.

    Every ProgramBench entry point builds its runner from this, so a setting
    recorded in a plan cannot silently fall back to the runner default (the
    triad and follow-up runners once recorded between_tool_calls and ran
    native_idle).
    """
    delivery = inference.get("team_message_delivery", "native_idle")
    if delivery not in ("native_idle", "between_tool_calls"):
        raise ValueError("inference.team_message_delivery must be native_idle or between_tool_calls")
    switches = {}
    for name in ("runtime_disclosure", "leader_loop_recovery"):
        switches[name] = inference.get(name, False)
        if not isinstance(switches[name], bool):
            raise ValueError(f"inference.{name} must be a boolean")
    if switches["leader_loop_recovery"] and inference.get("team_checkin_seconds") is None:
        raise ValueError("inference.leader_loop_recovery requires team_checkin_seconds")
    # Applies to both arms; native_default keeps Qwen's adaptive per-turn cap.
    switches["tool_call_cap"] = inference.get("tool_call_cap", "native_default")
    if switches["tool_call_cap"] not in ("native_default", "disabled"):
        raise ValueError("inference.tool_call_cap must be native_default or disabled")
    # Team arms only; off keeps earlier runs' behavior.
    switches["role_policy_mode"] = inference.get("role_policy_mode", "off")
    if switches["role_policy_mode"] not in ("off", "global", "master", "propose"):
        raise ValueError("inference.role_policy_mode must be off, global, master or propose")
    if "role_policy_protocol" in inference:
        protocol = inference["role_policy_protocol"]
        if protocol not in PACKAGE_PROTOCOLS:
            raise ValueError("Unknown inference.role_policy_protocol")
        if switches["role_policy_mode"] not in PROTOCOL_MODES[protocol]:
            raise ValueError("role_package_v1 implements off and propose only" if protocol == ROLE_POLICY_PROTOCOL
                             else f"{protocol} implements mode propose only")
        if protocol != ROLE_POLICY_PROTOCOL and inference.get("team_checkin_seconds") is None:
            raise ValueError(f"{protocol} requires inference.team_checkin_seconds")
        if protocol != ROLE_POLICY_PROTOCOL and inference.get("team_message_delivery") != "between_tool_calls":
            raise ValueError(f"{protocol} requires inference.team_message_delivery between_tool_calls")
        switches["role_policy_protocol"] = protocol
    if "stop_hook" in inference:
        switches["stop_hook"] = validate_stop_hook(inference["stop_hook"])
    if "teammate_limit" in inference:
        from agent_company.runtime.relay import validate_teammate_limit
        switches["teammate_limit"] = validate_teammate_limit(inference["teammate_limit"])
    # Observations are opt-in; omitted legacy profiles keep the runner default.
    if "observation_interval_seconds" in inference:
        interval = inference["observation_interval_seconds"]
        if interval is not None and (isinstance(interval, bool)
                                     or not isinstance(interval, (int, float))
                                     or not math.isfinite(interval) or interval <= 0):
            raise ValueError("inference.observation_interval_seconds must be null or a finite positive number")
        switches["observation_interval_seconds"] = interval
    return dict(task_memory_mb=inference["memory_mb"], task_cpus=inference["cpus"],
                agent_timeout_seconds=inference["agent_timeout_sec"], max_model_calls=None,
                team_checkin_seconds=inference.get("team_checkin_seconds"),
                team_message_delivery=delivery, **switches)


def comparison_round(profile: dict, *, repeat_index=None, pair_id=None) -> tuple[int, bool, str | None]:
    """Resolve an optional round override, preserving legacy pair IDs.

    An explicit round (including zero) defaults to ``<task>#r<round>``. Old
    profiles without a round keep ``repeat_index=0, pair_id=<task>``. A top-level
    pair ID is meaningful only for a single-task profile; multi-task profiles
    can set a pair ID on each task instead.
    """
    explicit = repeat_index is not None or "repeat_index" in profile
    repeat_index = profile.get("repeat_index", 0) if repeat_index is None else repeat_index
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 0:
        raise ValueError("repeat_index must be a nonnegative integer")
    pair_id = profile.get("pair_id") if pair_id is None else pair_id
    if pair_id is not None:
        _validate_pair_id(pair_id)
        if len(profile["tasks"]) != 1:
            raise ValueError("A top-level pair_id requires exactly one task; use tasks[].pair_id instead")
    return repeat_index, explicit, pair_id


def _validate_pair_id(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pair_id must be a nonempty string")
    return value


def prepare_comparison(profile_path: Path, project_root: Path, *, parallelism=None,
                       repeat_index=None, pair_id=None, batch_index=None) -> PreparedComparison:
    """Validate a public plan without loading its provider configuration."""
    root = Path(project_root).resolve()
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    if profile.get("benchmark", "ProgramBench") != "ProgramBench":
        raise ValueError("benchmark must be ProgramBench")
    if profile.get("evolution_enabled", False) is not False:
        raise ValueError("This comparison does not evolve policies")
    workers = profile.get("parallelism", 2) if parallelism is None else parallelism
    if isinstance(workers, bool) or not isinstance(workers, int) or workers not in (1, 2):
        raise ValueError("parallelism must be 1 or 2")
    inference = profile["inference"]
    if "max_model_calls" not in inference or inference["max_model_calls"] is not None:
        raise ValueError("This comparison requires explicit max_model_calls=null")
    if inference["memory_swap_mb"] != inference["memory_mb"]:
        raise ValueError("ProgramBench inference must not add extra swap")
    checkin = inference.get("team_checkin_seconds")
    options = runner_inference_options(inference)
    delivery = options["team_message_delivery"]
    if checkin is not None and (isinstance(checkin, bool) or not isinstance(checkin, (int, float)) or checkin <= 0):
        raise ValueError("inference.team_checkin_seconds must be null or a positive number of seconds")
    evaluator_cpus = profile["evaluation"]["docker_cpus"]
    if isinstance(evaluator_cpus, bool) or not isinstance(evaluator_cpus, int) or evaluator_cpus < 1:
        raise ValueError("evaluation.docker_cpus must be a positive integer")
    raw_tasks, raw_arms = profile["tasks"], profile["arms"]
    if not isinstance(raw_tasks, list) or not raw_tasks or not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("tasks and arms must be nonempty lists")
    repeat_index, explicit_round, common_pair_id = comparison_round(
        profile, repeat_index=repeat_index, pair_id=pair_id)
    if profile.get("selection_protocol") == TOTAL_MODE and explicit_round and repeat_index < 1:
        raise ValueError("Explicit positive repeat_index is required for this protocol")
    first_round_wave = profile.get("first_round_wave")
    if first_round_wave is not None:
        if type(first_round_wave) is not int or first_round_wave not in (1, 2, 3, 4):
            raise ValueError("first_round_wave must be an integer from 1 to 4")
        if profile.get("selection_protocol") != TOTAL_MODE:
            raise ValueError("first_round_wave requires the official-total selection protocol")
        if explicit_round and repeat_index != first_round_wave:
            raise ValueError("First-round repeat_index must equal first_round_wave")
    package_protocol = inference.get("role_policy_protocol")
    package_mode = package_protocol in PACKAGE_PROTOCOLS
    configured_batch = profile.get("batch_index")
    if batch_index is not None and configured_batch != batch_index:
        raise ValueError("batch_index must match the selected role-series profile")
    if configured_batch is not None:
        # agent_rules_v1 batch 1: Single and the M0 Team; batch 2: the reviewed M1 Team alone (user, 2026-10-19).
        batches = {ROLE_POLICY_PROTOCOL: (1, 2), AGENT_RULES_PROTOCOL: (1, 2)}.get(package_protocol, ())
        if (type(configured_batch) is not int or configured_batch not in batches
                or first_round_wave is not None or profile.get("selection_protocol") != TOTAL_MODE):
            raise ValueError("Series batch_index requires role_package_v1 or agent_rules_v1 (batch 1 or 2), and TOTAL")
        if explicit_round and repeat_index != configured_batch:
            raise ValueError("Series repeat_index must equal batch_index")
    tasks, task_rows, ids = [], [], set()
    for item in raw_tasks:
        task = ProgramBenchTask.load(
            _path(root, item["path"], "task path"),
            instruction_path=_path(root, item["instructions"], "task instructions"),
            agent_timeout_sec=inference["agent_timeout_sec"], cpus=inference["cpus"],
            memory_mb=inference["memory_mb"])
        if item["task_id"] != task.task_id or task.task_id in ids:
            raise ValueError("Task IDs must be unique and match public task metadata")
        ids.add(task.task_id)
        tasks.append(task)
        default_pair_id = f"{task.task_id}#r{repeat_index}" if explicit_round else task.task_id
        task_pair_id = _validate_pair_id(
            common_pair_id if common_pair_id is not None else item.get("pair_id", default_pair_id))
        task_rows.append({"task_id": task.task_id, "repository_id": task.repository_id,
                          "path": str(task.path), "instructions": str(task.instruction_path),
                          "image": task.image, "repeat_index": repeat_index, "pair_id": task_pair_id})
    arms, policies, names = [], {}, set()
    for arm in raw_arms:
        name, mode, version = arm["name"], arm["execution_mode"], arm["policy_version"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name)
                or name in names):
            raise ValueError("Arm names must be unique safe path components")
        names.add(name)
        if mode not in ("single", "team"):
            raise ValueError("execution_mode must be single or team")
        if not ((isinstance(version, int) and not isinstance(version, bool) and version >= 0)
                or (isinstance(version, str) and version.strip())):
            raise ValueError("policy_version must be a nonnegative integer or nonempty label")
        paths, texts = {}, []
        if "role_policy_mode" in arm and (not package_mode or mode != "team"):
            raise ValueError("Per-arm role_policy_mode is only available to role_package_v1 Team arms")
        if package_mode:
            allowed = {"name", "execution_mode", "policy_version", "policy_package_path", "role_policy_mode",
                       "assignment_policy_path", "communication_policy_path"}
            if set(arm) - allowed:
                raise ValueError("Role-package arms may override only role_policy_mode; inference/resources remain shared")
            if any(arm.get(field) not in (None, "") for field in
                   ("assignment_policy_path", "communication_policy_path")):
                raise ValueError("policy_package_path and assignment/communication paths are mutually exclusive")
            path = _path(root, arm.get("policy_package_path"), "policy_package_path")
            policy_package = load_policy_package(path)
            if policy_protocol(policy_package) != package_protocol:
                raise ValueError("The policy package's protocol disagrees with inference.role_policy_protocol")
            if version != policy_package["revision"]:
                raise ValueError("policy_version must equal the policy package revision")
            effective_mode = arm.get("role_policy_mode", options["role_policy_mode"])
            if mode == "team" and effective_mode not in PROTOCOL_MODES[package_protocol]:
                raise ValueError("role_package_v1 Team arms allow off or propose only"
                                 if package_protocol == ROLE_POLICY_PROTOCOL else f"{package_protocol} Team arms allow propose only")
            paths = {"policy_package_path": str(path), "policy_package": policy_package,
                     "assignment_policy_path": None, "communication_policy_path": None}
            if mode == "team":
                paths["role_policy_mode"] = effective_mode
            texts = ([policy_package["team"][key] for key in ("assignment", "communication")]
                     if mode == "team" else ["", ""])
        elif "policy_package_path" in arm:
            raise ValueError("policy_package_path requires role_package_v1")
        for field in (() if package_mode else ("assignment_policy_path", "communication_policy_path")):
            value = arm.get(field)
            if mode == "single":
                if value not in (None, ""):
                    raise ValueError("Single arms must have empty policy paths")
                paths[field], text = None, ""
            else:
                path = _path(root, value, field)
                # A policy's exact text is part of the recorded run condition.
                # Universal-newline decoding would silently turn CRLF into LF.
                text = path.read_bytes().decode("utf-8")
                if not text.strip():
                    raise ValueError("Team arms require nonempty assignment and communication policies")
                paths[field] = str(path)
            texts.append(text)
        policies[name] = tuple(texts)
        arms.append({"name": name, "arm_id": name, "execution_mode": mode,
                     "policy_version": version, **paths})
    if configured_batch is not None:
        # role_package_v1 batch 2 runs v1 beside its P0 parent; agent_rules_v1 batch 2 runs the reviewed M1 alone.
        alone = configured_batch == 2 and package_protocol == AGENT_RULES_PROTOCOL
        expected = ([('single', None, 0), ('team', 'propose', 0)] if configured_batch == 1
                    else [('team', 'propose', 1)] if alone else [('team', 'propose', 1), ('team', 'off', 0)])
        actual = [(a["execution_mode"], a.get("role_policy_mode"), a["policy_version"]) for a in arms]
        if len(tasks) != 1 or workers != len(expected) or actual != expected:
            raise ValueError("Series batches require their prescribed arms concurrently on one task")
        if configured_batch == 1 and arms[0]["policy_package"] != arms[1]["policy_package"]:
            raise ValueError("Batch 1 Single and propose must use the same P0 input package")
        if configured_batch == 2 and alone:
            # The launcher binds the parent to the approved M0; here only its shape is checked.
            if (arms[0]["policy_package"]["parent"] or {}).get("revision") != 0:
                raise ValueError("Batch 2 M1 must be a revision of a revision-0 package")
        elif configured_batch == 2:
            parent = arms[1]["policy_package"]
            if arms[0]["policy_package"]["parent"] != {
                    "package_id": parent["package_id"], "revision": 0}:
                raise ValueError("Batch 2 v1 parent must be the off arm's P0 package identity")
    waves = [[{"task_id": task.task_id, "arm_id": arm["arm_id"]} for arm in arms[offset:offset + workers]]
             for task in tasks for offset in range(0, len(arms), workers)]
    plan = {"benchmark": "ProgramBench", "profile_path": str(profile_path), "project_root": str(root),
            "provider_config": str(_path(root, profile["provider_config"], "provider_config")),
            "tasks": task_rows, "arms": arms, "waves": waves, "parallelism": workers,
            "inference": {key: inference[key] for key in
                          ("agent_timeout_sec", "cpus", "memory_mb", "memory_swap_mb", "max_model_calls")},
            "evaluation": {"docker_cpus": evaluator_cpus}, "new_rollouts": len(tasks) * len(arms),
            "reused_rollouts": 0, "runs_per_task_arm": 1, "fresh_start": True,
            "repeat_index": repeat_index,
            "evolution_enabled": False, "automatic_restarts": False,
            "runtime": "custom_Qwen_not_official_mini_SWE"}
    if "selection_protocol" in profile:
        plan["selection_protocol"] = profile["selection_protocol"]
    if first_round_wave is not None:
        plan["first_round_wave"] = first_round_wave
    if configured_batch is not None:
        plan["batch_index"] = configured_batch
    if "team_checkin_seconds" in inference:
        plan["inference"]["team_checkin_seconds"] = checkin
    if "team_message_delivery" in inference:
        plan["inference"]["team_message_delivery"] = delivery
    for name in ("runtime_disclosure", "leader_loop_recovery", "tool_call_cap", "role_policy_mode",
                 "observation_interval_seconds", "stop_hook", "role_policy_protocol", "teammate_limit"):
        if name in inference:
            plan["inference"][name] = options[name]
    return PreparedComparison(plan, tasks, policies)
