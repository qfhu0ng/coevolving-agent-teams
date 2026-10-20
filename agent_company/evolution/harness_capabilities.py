"""The target harness a candidate A/C will run under, for the optimizer and reviewer.

Plan: docs/stage1_optimizer_harness_feasibility_plan_20261012.md (v2), step A.

Values come from the same parameter chain the runner uses (runner_inference_options and
the timeout functions in qwen.py). Runtime semantics are the texts the coding agents
receive, produced by the same functions (runtime_rules, delivery_text, checkin_system,
role_policy_runtime.disclosure, the spawn tool description), so the optimizer reads the
same wording the agents read. A text an agent would not be shown is marked so.

This is a description of the target configuration and current implementation, not an
observation of any run. Anything this module cannot establish is listed under unknown.
"""
from __future__ import annotations

from dataclasses import fields
import json
from typing import Any

from agent_company.programbench.comparison import runner_inference_options
from agent_company.analysis.evidence_projection import SUBMISSION_CONTRACT
from agent_company.programbench.runner import ProgramBenchRunner
from agent_company.runtime.qwen import (CHECKIN_MILESTONES, NATIVE_TEAM_INACTIVITY_DEFAULT_SECONDS, WORKER_DELIVERY_NOTE,
                   checkin_refresh_seconds, checkin_system, delivery_text, native_request_timeout_seconds,
                   runtime_rules, team_inactivity_timeout_seconds)
from agent_company.runtime.relay import (MASTER_TOOLS, SINGLE_TOOLS, TEAM_SIZE_NOTE, WORKER_TOOLS, ProviderConfig,
                                         describe_team_spawn, team_size_note)
from agent_company.policy.role_policy_runtime import disclosure
from agent_company.policy import agent_rules, agent_rules_runtime, agent_rules_text
from agent_company.policy.role_package import AGENT_RULES_PROTOCOL
from agent_company.evolution import agent_rules_outer_text as outer_text

SCHEMA = 1
STATUS_FILE = "<the run's runtime status file>"
# Public technical provider parameters only: never the endpoint, credential variables or extra_body.
PROVIDER_FIELDS = ("model", "max_output_tokens", "request_timeout_seconds", "first_progress_timeout_seconds",
                   "progress_idle_timeout_seconds", "max_attempts", "retry_backoff_seconds", "stream_response")
ONLINE_POLICY = {
    "off": "No online policy changes: A and C stay as supplied for the whole run.",
    "global": "The master may replace the team-wide communication policy C during the run (GLOBAL-POLICY "
              "broadcast); every role sees the new version from its next request. There are no role policies.",
    "master": "C is fixed for the run. The master may set and replace each worker role's communication policy "
              "(a spawn-prompt marker for the first version, ROLE-POLICY messages later).",
    "propose": "C is fixed for the run. As in master mode, plus each worker may propose a change to its own "
               "role's communication policy, which takes effect only when the master approves or amends it.",
}


def _runner_default(name: str):
    return next(field.default for field in fields(ProgramBenchRunner) if field.name == name)


def _spawn_description() -> str:
    tool = {"function": {"name": "agent"}}
    describe_team_spawn(tool)
    return tool["function"]["description"]


def agent_rules_online_policy() -> dict[str, Any]:
    """The agent_rules_v1 mechanisms, with the approved texts the agents themselves read."""
    files = {"policies_file": "<policies file>", "lessons_file": "<lessons file>", "board_file": "<board file>"}
    return {"mode": "propose", "protocol": AGENT_RULES_PROTOCOL, "summary": outer_text.CAPABILITY_SUMMARY,
            # The plain-string descriptions below are not shown to the agents; {"text", "shown_to_agents"} entries are.
            "descriptions_shown_to_agents": False,
            "assignment_policy_changes_in_run": False, "master_strategy_visible_to": "master only",
            "master_strategy_in_master_system": {
                "assignment": {"text": "Assignment policy (fixed for this run):\n<assignment_policy>",
                               "shown_to_agents": True},
                "communication": {"text": agent_rules_text.MASTER_COMMUNICATION_SECTION.format(
                    communication="<communication_policy>"), "shown_to_agents": True}},
            **outer_text.CAPABILITIES,
            "rule_triggers": dict(outer_text.TRIGGER_MEANINGS),
            "wall_time_percentages": list(agent_rules.WALL_TIME_FRACTIONS),
            "parties": dict(outer_text.PARTY_MEANINGS),
            "fields": {"names": list(agent_rules.FIELDS), "meaning": outer_text.FIELD_MEANING},
            "runtime_diagnostics": False,
            "master_disclosure": {"text": agent_rules_runtime.master_card(**files), "shown_to_agents": True},
            "worker_disclosure": {"text": agent_rules_runtime.worker_card(files["board_file"]), "shown_to_agents": True}}


def target_harness(inference: dict, *, execution_mode: str, provider: ProviderConfig | None = None,
                   code_version: str | None = None, native_version: str | None = None,
                   policy_protocol: str | None = None) -> dict[str, Any]:
    """Describe the harness one arm will run under, from the runner's own parameter resolution.

    ``policy_protocol=agent_rules_v1`` describes the per-teammate policy runtime (plan v4
    section 12, item 3); without it the earlier description is unchanged.
    """
    if execution_mode not in {"single", "team"}:
        raise ValueError("execution_mode must be single or team")
    if policy_protocol not in (None, AGENT_RULES_PROTOCOL):
        raise ValueError("Unknown policy protocol for the target harness")
    rules_protocol = policy_protocol == AGENT_RULES_PROTOCOL and execution_mode == "team"
    if rules_protocol and inference.get("role_policy_protocol") != AGENT_RULES_PROTOCOL:
        # runner_inference_options then enforces the check-ins, delivery and mode the runtime requires.
        raise ValueError("An agent_rules_v1 harness needs inference.role_policy_protocol agent_rules_v1")
    options = runner_inference_options(inference)
    team = execution_mode == "team"
    shown = options["runtime_disclosure"]
    cap = options["tool_call_cap"]
    checkin = options["team_checkin_seconds"] if team else None
    delivery = options["team_message_delivery"] if team else None
    recovery = bool(options["leader_loop_recovery"]) and team
    mode = options["role_policy_mode"] if team else "off"
    unknown = []
    if code_version is None:
        unknown.append("/scope/code_version")
    if native_version is None:
        unknown.append("/scope/native_version")

    def rules(role):
        text = runtime_rules(role, checkin_interval_seconds=checkin if role == "master" else None,
                             leader_loop_recovery=recovery if role == "master" else False, tool_call_cap=cap)
        return {"text": text, "shown_to_agents": shown}

    timing: dict[str, Any] = {
        "agent_wall_time_seconds": options["agent_timeout_seconds"],
        "max_model_calls": options["max_model_calls"],
        "max_model_calls_meaning": "null means no task-wide model-call limit",
        "provider_retries_share_one_request_deadline": True,
        "task_deadline_is_not_extended_by_retries": True,
    }
    if provider is None:
        unknown += ["/timing/provider", "/timing/native_request_timeout_seconds"]
    else:
        timing["provider"] = {name: getattr(provider, name) for name in PROVIDER_FIELDS}
        timing["native_request_timeout_seconds"] = native_request_timeout_seconds(provider)
        if team:
            # As the launch record: None leaves native's own default in force.
            timing["native_team_inactivity_timeout_seconds"] = team_inactivity_timeout_seconds(
                _runner_default("team_stall_timeout_seconds"), provider,
                is_team=True) or NATIVE_TEAM_INACTIVITY_DEFAULT_SECONDS

    harness: dict[str, Any] = {
        "schema": SCHEMA,
        "nature": "Target configuration and current implementation for the candidate's next run; "
                  "not an observation of any run.",
        "scope": {"execution_mode": execution_mode, "role_policy_mode": mode,
                  "code_version": code_version or "unknown", "native_version": native_version or "unknown",
                  **({"policy_protocol": AGENT_RULES_PROTOCOL} if rules_protocol else {})},
        "roles_and_tools": ({
            "master": {"tools": sorted(MASTER_TOOLS), "can_edit_or_run_commands": False},
            "worker": {"tools": sorted(WORKER_TOOLS), "can_spawn": False},
            "spawn_tool_description": (_spawn_description() + " " + agent_rules_text.SPAWN_TOOL_NOTE
                                       if rules_protocol else _spawn_description()),
        } if team else {"single": {"tools": sorted(SINGLE_TOOLS), "can_spawn": False}}),
        "timing": timing,
        "guards": {"tool_call_cap": cap, "leader_loop_recovery": recovery,
                   **({"master_rules": rules("master"), "worker_rules": rules("worker")} if team
                      else {"single_rules": rules("single")})},
        "delivery_and_evaluation": {**SUBMISSION_CONTRACT,
                                    "hidden_tests": "Not readable during the agent phase; the official score "
                                                    "is computed in a separate environment afterwards."},
        "unknown": unknown,
    }
    if team:
        wake = {"checkin_interval_seconds": checkin,
                "checkin_milestone_fractions": list(CHECKIN_MILESTONES) if checkin is not None else [],
                "other_wake_sources": "A teammate message, or a teammate's end-of-turn report, wakes a "
                                      "waiting master. There is no timer for arbitrary time points."}
        if checkin is not None:
            wake["status_refresh_seconds"] = checkin_refresh_seconds(checkin)
            text = (agent_rules_text.CHECKIN_SYSTEM.format(interval_minutes=f"{checkin / 60:g}", status_file=STATUS_FILE)
                    if rules_protocol else checkin_system(checkin, STATUS_FILE))
            limit = options.get("teammate_limit", 10)
            if rules_protocol and limit != 10:
                # As the runtime does for the master's system (qwen.py).
                text = text.replace("At most 10 teammates can be active", f"At most {limit} teammates can be active")
            wake["master_text"] = {"text": text, "shown_to_agents": True}
        harness["wakeups"] = wake
        harness["messages"] = {
            "delivery_mode": delivery,
            "master_text": {"text": delivery_text(delivery), "shown_to_agents": True},
            **({"worker_text": {"text": WORKER_DELIVERY_NOTE, "shown_to_agents": True}}
               if delivery == "between_tool_calls" else {})}
        harness["online_policy"] = (agent_rules_online_policy() if rules_protocol else {
            "mode": mode, "summary": ONLINE_POLICY[mode], "assignment_policy_changes_in_run": False,
            **({"master_disclosure": disclosure(mode, "master"), "worker_disclosure": disclosure(mode, "worker")}
               if mode != "off" else {})})
        if rules_protocol:
            # The exact sentence the master and every worker read about the shared container (2026-10-18).
            harness["online_policy"]["resources_disclosure"] = {"text": agent_rules_text.resources_note(
                {"cpus": options["task_cpus"], "memory_mb": options["task_memory_mb"]}, team=True), "shown_to_agents": True}
        harness["lifecycle"] = {
            "failed_workers_revived_automatically": False,
            "succession": (outer_text.SUCCESSION_TEXT if rules_protocol else
                           "SUCCEEDS keeps the role slot and its current policy version only; not the "
                           "conversation, worktree or task ownership." if mode in {"master", "propose"}
                           else "No succession mechanism; a new worker starts from its spawn prompt."),
            "teammate_limit": options.get("teammate_limit", 10),
            "teammate_limit_meaning": team_size_note(options.get("teammate_limit", 10))}
    json.dumps(harness, allow_nan=False)
    return harness


def resolve_pointer(document: Any, pointer: str) -> Any:
    """RFC 6901 lookup; raises KeyError for a path that does not exist."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise KeyError(pointer)
    value = document
    for token in pointer.split("/")[1:] if pointer else []:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            raise KeyError(pointer)
    return value


def native_version(qwen_package) -> str | None:
    """The installed Qwen CLI version from its package.json, or None when it cannot be read."""
    from pathlib import Path
    try:
        version = json.loads((Path(qwen_package) / "package.json").read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError, AttributeError):
        return None
    return version if isinstance(version, str) and version else None


def executability_summary(items) -> dict[str, Any]:
    """What a progress record shows before dispatch: verdict counts and the rules not plainly supported."""
    if items is None:
        return {"recorded": False}
    counts = {verdict: 0 for verdict in ("supported", "supported_with_conditions", "unsupported", "unknown")}
    flagged = []
    for item in items:
        counts[item["verdict"]] += 1
        if item["verdict"] in {"unsupported", "unknown"}:
            flagged.append({"verdict": item["verdict"], "policy_part": item["policy_part"],
                            "rule_quote": item["rule_quote"][:200], "reason": item["reason"][:400]})
    return {"recorded": True, "counts": counts, "unsupported_or_unknown": flagged,
            "dispatch": "unchanged: a completed review dispatches the candidate; this is advisory"}
