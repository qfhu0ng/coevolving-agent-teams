"""Run a task with the released Qwen CLI, directly or with its native Team.

Paid calls happen only when a real transport is supplied through ProviderConfig.
Tests inject a scripted upstream while exercising the actual installed CLI.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlsplit
import uuid

from agent_company.runtime.relay import (COORD_TOOLS, MASTER_TOOLS, MODEL_CALL_LIMIT_MESSAGE, SINGLE_EXCLUDED_TOOLS, SINGLE_TOOLS, TEAM_SIZE_NOTE, TEAMMATE_LIMIT, team_size_note, validate_teammate_limit, WORKER_TOOLS,
                    ProviderConfig, TeamRelay, start_server, validate_model_call_limit)
from agent_company.policy.role_policy_runtime import DISCLOSURE_VERSION as ROLE_POLICY_DISCLOSURE_VERSION, LifecycleTail
from agent_company.policy.role_policy_runtime import disclosure as role_policy_disclosure
from agent_company.runtime.team import WorkerRegistry, Workspace
from agent_company.runtime.workspace import exception_diagnostic
from agent_company.runtime.collaboration import CollaborationTail
from agent_company.runtime.collaboration_text import CHECKIN_SYSTEM
from agent_company.policy import role_package_runtime
from agent_company.policy import agent_rules_runtime, agent_rules_text
from agent_company.policy.role_package import (AGENT_RULES_PROTOCOL, PROTOCOL_MODES, ROLE_POLICY_PROTOCOL,
                                               policy_protocol, validate_policy_package)


def native_model_providers(provider: ProviderConfig, relay_url: str) -> dict:
    """Keep DeepSeek reasoning controls identifiable through the local relay.

    Qwen otherwise detects these controls from the upstream hostname, which it
    cannot see here. Do not select a reasoning effort: shared summaries and
    coding retain the provider's existing defaults.
    """
    if urlsplit(provider.base_url).hostname != "api.deepseek.com":
        return {}
    return {"openai": [{
        "id": provider.model,
        "baseUrl": relay_url,
        "capabilities": {"reasoning": {
            "thinking": True, "disableField": "thinking", "efforts": ["low", "high", "max"]}},
        # Auth refresh replaces generation settings from the registered provider.
        # Keep the native deadline and disabled SDK retries in that same scope.
        "generationConfig": {"maxRetries": 0,
                             "timeout": math.ceil(native_request_timeout_seconds(provider) * 1000),
                             # deepseek-flash: 1M context, per the official model table.
                             # https://api-docs.deepseek.com/quick_start/pricing/
                             "contextWindowSize": 1000000},
    }]}


def provider_incident_summary(relay_summary: dict) -> dict:
    """Expose observed request failures without claiming that they recovered."""
    kinds = {"provider_error", "provider_transport_error", "provider_timeout",
             "provider_first_progress_timeout", "provider_progress_idle_timeout"}
    incidents = [record for record in relay_summary.get("errors", [])
                 if record.get("error") and record.get("error_kind") in kinds]
    return {"count": len(incidents),
            "by_kind": dict(Counter(record["error_kind"] for record in incidents)),
            "by_role": dict(Counter(record.get("role") or "unknown" for record in incidents))}


def native_result(stdout_path: Path) -> dict:
    """Recognize explicit CLI error records, including exit-zero API failures."""
    final = None
    if stdout_path.exists():
        for line in stdout_path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "result":
                final = event
    if final is None:
        return {"result": None, "explicit_error": False, "missing_result": True}
    text = str(final.get("result", "")).lstrip()
    explicit_error = bool(final.get("is_error")) or str(final.get("subtype", "")).startswith("error")
    explicit_error |= text.lower().startswith(("[api error]", "[api error:", "api error:"))
    return {"result": final, "explicit_error": explicit_error, "missing_result": False}


def native_lifecycle(stderr_path: Path) -> dict:
    """Retain native terminal/notification facts independently of verifier score."""
    events = []
    prefix = "Agent Company lifecycle: "
    if stderr_path.exists():
        for line in stderr_path.read_text(errors="replace").splitlines():
            if line.startswith(prefix):
                try:
                    events.append(json.loads(line[len(prefix):]))
                except ValueError:
                    events.append({"event": "invalid_lifecycle_record"})
    queued = {event["notice_id"] for event in events if event["event"] == "failure_notice_queued"}
    observed = {event["notice_id"] for event in events if event["event"] == "failure_notice_observed"}
    return {"events": events, "unobserved_failure_notices": sorted(queued - observed),
            "notification_errors": [event for event in events if event["event"] in {
                "failure_notice_delivery_error", "invalid_lifecycle_record"}]}


def runtime_stop_reason(execution: dict, cli_result: dict, launcher_error: str | None,
                        relay_summary: dict) -> str:
    """Classify the final execution, not whether every provider call succeeded.

    Provider incidents stay in the relay's errors and usage accounting. They do
    not prove that the final task failed: a later turn or a replacement worker
    can still deliver within the original budget. Conversely, a normal CLI exit
    is not proof of a correct solution; the official verifier scores delivery.
    Evidence corruption remains fatal independently of that score; rejected
    provider replies may be followed by a valid continuation or replacement.
    """
    if launcher_error:
        return "launcher_error"
    timed_out = execution.get("timed_out", False)
    # The HTTP deadline may end the CLI before the process timeout fires.
    # Admission rejections and already-admitted requests record task expiry in
    # different places. Match the final root-role API error to relay evidence;
    # neither elapsed time nor a worker's deadline can establish the CLI outcome.
    native_text = str((cli_result.get("result") or {}).get("result", ""))
    root_role = "single" if relay_summary.get("execution_mode") == "single" else "master"
    rejected_at_deadline = (
        root_role in relay_summary.get("deadline_rejections", [])
        and "task_deadline_exceeded: task budget exhausted" in native_text)
    inflight_task_deadline = "task_deadline_exceeded:" in native_text and any(
        record.get("role") == root_role and record.get("error_kind") == "task_deadline_exceeded"
        and isinstance(record.get("error"), str) and record["error"] and record["error"] in native_text
        for record in relay_summary.get("errors", []))
    if (execution.get("returncode") == 0
            and cli_result["explicit_error"]
            and (rejected_at_deadline or inflight_task_deadline)):
        timed_out = True
    model_call_limited = (execution.get("returncode") == 0
                          and cli_result["explicit_error"]
                          and bool(relay_summary.get("model_call_budget_rejections"))
                          and MODEL_CALL_LIMIT_MESSAGE in native_text)
    if not timed_out and not model_call_limited and (execution.get("returncode") != 0 or cli_result["explicit_error"]):
        return "cli_error"
    fatal_kinds = {"evidence_write_failed"}
    # A rejected malformed provider reply is a per-attempt failure, not evidence
    # corruption. Evidence-write failures remain fatal; no bad reply is executed.
    # Do not require the failed worker itself to recover: it may be replaced.
    if any(record.get("error") and record.get("error_kind") in fatal_kinds
           for record in relay_summary.get("errors", [])):
        return "provider_error"
    if timed_out:
        return "task_timeout"
    lifecycle = cli_result.get("lifecycle", {})
    if lifecycle.get("notification_errors") or lifecycle.get("unobserved_failure_notices"):
        return "unhandled_worker_failure"
    if model_call_limited:
        return "model_call_limit_exceeded"
    return "missing_cli_result" if cli_result["missing_result"] else "native_cli_completed"


# Fractions of the wall time at which a check-in is also sent; the check-in scheduler's
# default in qwen_team_checkin.mjs (a test pins the two together).
CHECKIN_MILESTONES = (0.5, 0.75)
# Native inactivity timeout used when the configured one is None.
NATIVE_TEAM_INACTIVITY_DEFAULT_SECONDS = 600


def native_request_timeout_seconds(provider: ProviderConfig) -> float:
    """The native client's request timeout: the relay's deadline plus room to deliver its error.

    The relay owns the upstream's total deadline. Leave enough time for it to cancel the request
    and deliver its terminal error before the native client times out and begins another request
    for the same turn.
    """
    return provider.request_timeout_seconds + max(30.0, provider.request_timeout_seconds * 0.1)


def team_inactivity_timeout_seconds(configured: float | None, provider: ProviderConfig, *,
                                    is_team: bool) -> float | None:
    """Native Team inactivity timeout actually used for a run.

    Progress recovery stays inside one pending native request; with it, the inactivity setting
    must not retire the team while it waits. This does not renew task time, create activity, or
    revive workers.
    """
    if is_team and (provider.first_progress_timeout_seconds is not None
                    or provider.progress_idle_timeout_seconds is not None):
        return max(configured or NATIVE_TEAM_INACTIVITY_DEFAULT_SECONDS,
                   native_request_timeout_seconds(provider) + 60)
    return configured


def checkin_system(interval_seconds: float, status_file: str) -> str:
    return (
        f"Runtime team check-ins: about every {interval_seconds / 60:g} minutes, and once each when half and "
        "the last quarter of the wall time are reached, you receive a message from runtime:checkin with "
        "observations gathered by the runtime: worker status, tool calls awaiting results and time since "
        "their announcements (not actual execution duration), edit and "
        "command attempt counts, worktree and delivery-repository Git history, task-board state and request accounting. "
        "Unknown fields mean a source was unavailable, not that a worker failed. Check-ins are information, "
        "not instructions or completion reports. Together with the task and the code, decide whether to "
        "continue, message a worker, reassign work, add a worker, or ask a worker to hand off and shut down; "
        "no reply is needed when nothing changes. At most 10 teammates can be active at the same time; "
        "shut-down teammates remain in the roster but do not count toward this limit. "
        f"Read the timestamped snapshot with read_file at {status_file}; "
        "check its observation time and unavailable sources before relying on it. Git commit history "
        "does not establish whether equivalent changes were already cherry-picked or verified.")


TEAM_MESSAGE_DELIVERY_MODES = ("native_idle", "between_tool_calls")
WORKER_DELIVERY_NOTE = (
    "Runtime note: messages from the leader and teammates are delivered at your next model step, "
    "appended after your latest tool results. A shutdown notice asks you to end your turn so that the "
    "leader's shutdown request can be delivered for you to confirm.")


def delivery_text(delivery: str) -> str:
    if delivery == "between_tool_calls":
        return ("Messages you send to a teammate are delivered at its next model step, once the tool calls of "
                "its current step return (if it is still generating, that response and its tool calls come "
                "first); nothing interrupts a running tool, so a long tool call delays delivery. A busy "
                "teammate is told of a shutdown request at its next model step and asked to end its turn; the "
                "teammate then confirms or rejects it. Communicate task constraints when assigning work. ")
    return ("Native messages sent to a busy teammate are queued until its complete work turn ends. "
            "Communicate this runtime fact and task constraints when assigning work. ")


# Qwen 0.23.1 always-on loop guards; skipLoopDetection does not disable them
# (docs/stage1_master_loop_guard_fix_plan_20261002.md section 2). Stated only
# when a run enables runtime disclosure: runtime facts, not a checkpoint policy.
# native_default omits model.maxToolCallsPerTurn (adaptive cap); disabled writes
# 0, which native reads as no per-turn cap. Never write null: native treats any
# present value as explicit, and null resolves to a hard cap of 100.
TOOL_CALL_CAP_MODES = ("native_default", "disabled")
CONSECUTIVE_GUARD_RULE = (
    "Runtime rules (Qwen CLI loop guards, always on): the fifth consecutive call of one tool with identical "
    "arguments is not executed and stops the turn. This includes read_file even when the file changed between "
    "reads; only task_list is exempt while its result keeps changing.")
TURN_CAP_RULE = (
    " A turn is also stopped once it has more than 100 tool calls and one identical call occurs at least 6 times "
    "in it (not necessarily in a row), or once it has more than 1000 tool calls.")
NO_TURN_CAP_RULE = " There is no limit on the number of tool calls in a turn."
LOOP_GUARD_RULES = CONSECUTIVE_GUARD_RULE + TURN_CAP_RULE
SHELL_GUARD_RULE = (
    " Eight consecutive tool calls that only inspect Git through run_shell_command (git status, git ls-files, "
    "or git diff without a path) also stop the turn.")


def checkin_refresh_seconds(interval_seconds: float) -> float:
    """Status-file refresh period; mirrors refreshMs in qwen_native_compat.mjs."""
    return max(0.25, min(60.0, interval_seconds / 4))


def runtime_rules(role: str, *, checkin_interval_seconds: float | None = None,
                  leader_loop_recovery: bool = False, tool_call_cap: str = "native_default") -> str:
    """Role-specific waiting semantics and loop-guard consequences."""
    if tool_call_cap not in TOOL_CALL_CAP_MODES:
        raise ValueError("tool_call_cap must be native_default or disabled")
    capped = tool_call_cap == "native_default"
    guards = CONSECUTIVE_GUARD_RULE + (TURN_CAP_RULE if capped else NO_TURN_CAP_RULE)
    if role == "master":
        rules = (guards + " Counts restart whenever a teammate message or check-in is delivered to "
                 "you. For you, a stopped turn ends the whole run and stops every teammate.")
        if leader_loop_recovery:
            rules += (" Exception in this run: when the stopped call is the fifth identical read_file of the "
                      "runtime status file, it is the only tool call in that response, and a teammate is running, "
                      "only your turn ends and you wait as described below; every other stop, including a response "
                      "with any other tool call, still ends the run.")
        wait = ("Waiting: when you have no coordination action to take, end your turn without calling tools. "
                "While a teammate is working, the runtime then waits without model calls and wakes you when a "
                "teammate message arrives or a teammate ends its work turn (it reports to you automatically)")
        if checkin_interval_seconds is not None:
            wait += (f", or when a check-in is sent: about every {checkin_interval_seconds / 60:g} minutes and "
                     "once each at half and three quarters of the wall time while a teammate is working; an unread "
                     "check-in is merged into the next one, and none is sent while all teammates are shutting down "
                     "or near the deadline. The status file is refreshed about every "
                     f"{checkin_refresh_seconds(checkin_interval_seconds):g} seconds; reading it again within that "
                     "interval returns the same snapshot")
        return rules + "\n" + wait + (
            f". Messages{' and check-ins' if checkin_interval_seconds is not None else ''} that arrive while you "
            "are working are delivered with your next tool results. If no teammate is working and no message is "
            "queued, ending your turn ends the run.")
    if role == "worker":
        return guards + SHELL_GUARD_RULE + (
            " A turn runs from the message or task that starts your work until you answer without tool calls, "
            "however long that takes; messages that reach you during the turn do not restart the counts, the "
            "next turn does." if capped else "") + (
            " For you, a stopped turn ends that turn with an error. If a message is already queued "
            "for you, you continue with it, and a turn that then ends normally clears the error; otherwise you "
            "are marked failed and the leader is notified.")
    if role == "single":
        return guards + SHELL_GUARD_RULE + (
            " The whole task normally runs as one turn, so these counts accumulate across the task."
            if capped else "") + " A stopped turn ends the run."
    raise ValueError("role must be master, worker or single")


ROLE_POLICY_MODES = ("off", "global", "master", "propose")
# Replaces the static C section of the master prompt when a role-policy mode supplies
# a versioned C on every request (docs/stage1_role_policy_harness_plan_20261009.md 3.3).
RUNTIME_COMMUNICATION_POLICY = (
    "Communication policy: the runtime supplies the current versioned communication policy C, the "
    "control-message rules for this run and the current policy state in your system instructions on every "
    "request. Teammates receive C and their own policy state the same way.")


def master_system(assignment_policy: str, communication_policy: str, checkin: str | None = None,
                  delivery: str = "native_idle", rules: str | None = None,
                  communication_in_runtime: bool = False, communication_section: str | None = None,
                  team_size: str = TEAM_SIZE_NOTE) -> str:
    return "\n\n".join([
        "You are the coordination-only master of a coding team. Use native Team tools to assign, "
        "coordinate, inspect team status and request shutdown. You may inspect the repository "
        "read-only with read_file, glob and grep_search; inspect the relevant code, interfaces and "
        "shared files before deciding the decomposition and file ownership. Delegate all "
        "implementation, shell execution, testing and final integration to writable general-purpose "
        "teammates. You cannot edit files or run commands yourself. Choose the number of teammates and "
        "their concrete assignments according to the assignment policy and task. Create the team "
        "with team_create before spawning named teammates. Every teammate inherits your model; "
        "omit model overrides, isolation and read_only. Use short unique names starting with a letter "
        "and containing letters, digits, underscores or hyphens. The runtime creates each named "
        "teammate's Git worktree automatically; omit working_dir or use the documented worker root. "
        "The worker's startup directory is not the delivery repository. For integration, spawn "
        "without working_dir and tell that worker to use git -C in the delivery repository; "
        "do not set the delivery repository as its spawn working_dir. "
        "Keep existing teammates for follow-up: send_message continues their persistent sessions. "
        + delivery_text(delivery) +
        "Delegate integration into the delivery repository and final checks to a worker; base claims "
        "on the reported checks. Before final completion request_shutdown for all teammates and "
        "wait for their shutdown. Workers acknowledge with send_message to leader containing "
        "shutdown_approved. Preserve team records and worktrees for the experiment.",
        team_size,
        *([checkin] if checkin else []),
        *([rules] if rules else []),
        "Assignment policy (fixed for this run):\n" + assignment_policy,
        (communication_section if communication_section is not None else
         RUNTIME_COMMUNICATION_POLICY if communication_in_runtime else
         "Communication policy (also supplied in each teammate's system instructions):\n" + communication_policy),
    ])


def single_system(rules: str | None = None) -> str:
    text = (
        "You are a coding agent completing a repository task directly. Inspect the repository, "
        "implement the requested changes, run relevant checks and report the result and remaining "
        "issues. Work in the delivery repository using the available file and shell tools. "
        "Do not create or delegate to other agents; team, messaging and task-board tools are disabled. "
        "Preserve unrelated changes and base completion claims on checks you actually performed."
    )
    return text + "\n\n" + rules if rules else text


def validate_stop_hook(value: dict | None) -> dict | None:
    """Validate explicit model text; absent settings preserve earlier runs.

    The reminder and corresponding budget notice are supplied together so the
    run cannot both promise early completion and block it. Text approval is a
    human review step; this function does not infer or manufacture approval.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"reason", "budget_notice"}:
        raise ValueError("stop_hook must contain reason and budget_notice, or be null")
    if any(not isinstance(text, str) or not text.strip() for text in value.values()):
        raise ValueError("stop_hook reason and budget_notice must be nonempty strings")
    if value["reason"].count("{remaining_seconds}") != 1:
        raise ValueError("stop_hook reason must contain one {remaining_seconds} placeholder")
    return dict(value)


class QwenTeamRunner:
    def __init__(self, qwen_cli: str, node: str = "node", upstream=None,
                 team_stall_timeout_seconds: float | None = 1800,
                 checkin_interval_seconds: float | None = None,
                 team_message_delivery: str = "native_idle",
                 runtime_disclosure: bool = False, leader_loop_recovery: bool = False,
                 tool_call_cap: str = "native_default", role_policy_mode: str = "off",
                 stop_hook: dict | None = None, policy_package: dict | None = None,
                 teammate_limit: int = TEAMMATE_LIMIT):
        # Explicit paths make execution independent of ignored study directories
        # and let Docker use its own architecture-correct Qwen installation.
        self.qwen_cli, self.node, self.upstream = str(qwen_cli), str(node), upstream
        if team_stall_timeout_seconds is not None and (not math.isfinite(team_stall_timeout_seconds)
                                                       or team_stall_timeout_seconds <= 0):
            raise ValueError("Team inactivity timeout must be positive or None for native behavior")
        self.team_stall_timeout_seconds = team_stall_timeout_seconds
        # Opt-in runtime team check-ins (Team only). None keeps earlier runs' behavior.
        if checkin_interval_seconds is not None and (isinstance(checkin_interval_seconds, bool)
                                                     or not math.isfinite(checkin_interval_seconds)
                                                     or checkin_interval_seconds <= 0):
            raise ValueError("Check-in interval must be positive or None to disable check-ins")
        self.checkin_interval_seconds = checkin_interval_seconds
        # Opt-in: native_idle keeps Qwen's idle-only teammate message delivery.
        if team_message_delivery not in TEAM_MESSAGE_DELIVERY_MODES:
            raise ValueError("team_message_delivery must be native_idle or between_tool_calls")
        self.team_message_delivery = team_message_delivery
        # Independent opt-ins (2026-10-02), both off by default. Disclosure states
        # waiting and loop-guard rules per role; recovery lets the master's repeated
        # status-file read end only its turn (Team runs with check-ins only).
        for name, value in (("runtime_disclosure", runtime_disclosure),
                            ("leader_loop_recovery", leader_loop_recovery)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if leader_loop_recovery and checkin_interval_seconds is None:
            raise ValueError("leader_loop_recovery requires runtime check-ins (its target is the status file)")
        self.runtime_disclosure = runtime_disclosure
        self.leader_loop_recovery = leader_loop_recovery
        # Opt-in: disabled removes Qwen's per-turn tool-call cap for every role
        # (Single, master, workers); the other always-on guards stay.
        if tool_call_cap not in TOOL_CALL_CAP_MODES:
            raise ValueError("tool_call_cap must be native_default or disabled")
        self.tool_call_cap = tool_call_cap
        # Opt-in role-level communication policies (Team only); off keeps earlier runs' behavior.
        if role_policy_mode not in ROLE_POLICY_MODES:
            raise ValueError("role_policy_mode must be off, global, master or propose")
        self.role_policy_mode = role_policy_mode
        self.stop_hook = validate_stop_hook(stop_hook)
        self.teammate_limit = validate_teammate_limit(teammate_limit)
        self.policy_package = validate_policy_package(policy_package) if policy_package is not None else None
        self.policy_protocol = policy_protocol(self.policy_package) if self.policy_package is not None else None
        if self.policy_package is not None and role_policy_mode not in PROTOCOL_MODES[self.policy_protocol]:
            raise ValueError(f"{self.policy_protocol} implements mode(s) {', '.join(PROTOCOL_MODES[self.policy_protocol])}")

    def system_texts(self, assignment_policy: str, communication_policy: str, *, is_team: bool,
                     status_file: str | None, resources: dict | None = None) -> tuple[str, str]:
        """The master's (or the Single agent's) system text and the workers' system note, as run() sends them."""
        checkin = is_team and self.checkin_interval_seconds is not None
        agent_rules = is_team and self.policy_protocol == AGENT_RULES_PROTOCOL
        delivery = self.team_message_delivery if is_team else None
        disclosure = self.runtime_disclosure
        recovery = self.leader_loop_recovery and is_team
        policy_mode = self.role_policy_mode if is_team else "off"
        worker_note = "\n\n".join(note for note in (
            WORKER_DELIVERY_NOTE if delivery == "between_tool_calls" else "",
            runtime_rules("worker", tool_call_cap=self.tool_call_cap) if is_team and disclosure else "") if note)
        system = (master_system(assignment_policy, communication_policy,
                                (agent_rules_text.CHECKIN_SYSTEM.format(interval_minutes=f"{self.checkin_interval_seconds / 60:g}",
                                                                        status_file=status_file) if agent_rules else
                                 CHECKIN_SYSTEM.format(interval_minutes=f"{self.checkin_interval_seconds / 60:g}",
                                                       status_file=status_file) if self.policy_package is not None else
                                 checkin_system(self.checkin_interval_seconds, status_file)) if checkin else None,
                                delivery=delivery,
                                rules=runtime_rules("master", leader_loop_recovery=recovery,
                                                    checkin_interval_seconds=self.checkin_interval_seconds
                                                    if checkin else None,
                                                    tool_call_cap=self.tool_call_cap) if disclosure else None,
                                communication_in_runtime=policy_mode != "off" or self.policy_package is not None,
                                communication_section=(agent_rules_text.MASTER_COMMUNICATION_SECTION.format(
                                    communication=communication_policy) if agent_rules else None),
                                team_size=team_size_note(self.teammate_limit))
                  if is_team else single_system(runtime_rules("single", tool_call_cap=self.tool_call_cap)
                                                if disclosure else None))
        if self.policy_package is not None and is_team and not agent_rules:
            system = system.replace(RUNTIME_COMMUNICATION_POLICY, role_package_runtime.RP_RUNTIME_POINTER)
        if is_team and self.teammate_limit != TEAMMATE_LIMIT:
            # Check-in descriptions state the slot count; the default keeps earlier texts unchanged.
            system = system.replace("At most 10 teammates can be active", f"At most {self.teammate_limit} teammates can be active")
        if self.policy_protocol == AGENT_RULES_PROTOCOL and resources and resources.get("cpus") and resources.get("memory_mb"):
            # Both arms of an agent_rules_v1 batch learn the container's shared CPU and memory (2026-10-18).
            note = agent_rules_text.resources_note(resources, team=is_team)
            system = system + "\n\n" + note
            if is_team:
                worker_note = "\n\n".join(part for part in (worker_note, note) if part)
        return system, worker_note

    def run(self, workspace: Workspace, instruction: str, assignment_policy: str,
            communication_policy: str, output_dir: Path, provider: ProviderConfig,
            timeout_seconds: float, *, execution_mode: str = "team",
            max_model_calls: int | None = None, source_boundary: str | None = None) -> dict:
        if execution_mode not in {"single", "team"}:
            raise ValueError("execution_mode must be single or team")
        validate_model_call_limit(max_model_calls)
        is_team = execution_mode == "team"
        if not math.isfinite(timeout_seconds) or timeout_seconds < 1:
            raise ValueError("Task timeout must be finite and at least one second")
        if not instruction.strip():
            raise ValueError("Task instruction is required")
        if is_team and (not assignment_policy.strip() or not communication_policy.strip()):
            raise ValueError("Both policies are required for team execution")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "launch.json").exists():
            raise ValueError("Use a fresh output directory for each task run")
        runtime = PurePosixPath(workspace.runtime_root) / ("qwen-" + uuid.uuid4().hex)
        for directory in ("home", "profile", "runtime"):
            workspace.write_file(str(runtime / directory / ".keep"), "")
        workspace.write_file(str(runtime / "empty-settings.json"), "{}\n")
        native_request_timeout = native_request_timeout_seconds(provider)
        team_stall_timeout_seconds = team_inactivity_timeout_seconds(
            self.team_stall_timeout_seconds, provider, is_team=is_team)
        settings = {"telemetry": {"enabled": False}, "experimental": {"agentTeam": is_team},
                    "model": {"maxSessionTurns": -1, "maxToolCalls": -1,
                              "generationConfig": {"maxRetries": 0}},
                    "tools": {"shell": {"enableInteractiveShell": False},
                              "toolSearch": {"enabled": False}}}
        if self.tool_call_cap == "disabled":
            settings["model"]["maxToolCallsPerTurn"] = 0  # Native: 0 means no per-turn cap.
        workspace.write_file(str(runtime / "qwen_bootstrap.cjs"),
                             Path(__file__).with_name("qwen_bootstrap.cjs").read_text())
        workspace.write_file(str(runtime / "qwen_tool_errors.mjs"),
                             Path(__file__).with_name("qwen_tool_errors.mjs").read_text())
        for name in ("qwen_native_compat.mjs", "qwen_native_patch.mjs"):
            workspace.write_file(str(runtime / name), Path(__file__).with_name(name).read_text())
        if self.stop_hook is not None:
            workspace.write_file(str(runtime / "qwen_stop_hook.mjs"),
                                 Path(__file__).with_name("qwen_stop_hook.mjs").read_text())
        if is_team and team_stall_timeout_seconds is not None:
            workspace.write_file(str(runtime / "qwen_stall_timeout.mjs"),
                                 Path(__file__).with_name("qwen_stall_timeout.mjs").read_text())
        checkin = is_team and self.checkin_interval_seconds is not None
        agent_rules = is_team and self.policy_protocol == AGENT_RULES_PROTOCOL
        if agent_rules and not checkin:
            # Rule triggers come from the check-in loop's task/Git samples (M1 review Q1).
            raise ValueError("agent_rules_v1 Team runs require runtime check-ins")
        if agent_rules and self.team_message_delivery != "between_tool_calls":
            # Message and shutdown observations exist only with this delivery (plan v4 M2).
            raise ValueError("agent_rules_v1 Team runs require team_message_delivery between_tool_calls")
        # A dedicated directory outside the delivery repository: readable by the
        # master, never part of a submission.
        status_file = str(runtime / "status" / "team_checkin.json") if checkin else None
        # agent_rules_v1: the relay rewrites these read-only files for the master.
        policy_files = ({"policies": str(runtime / "status" / agent_rules_runtime.POLICIES_FILE),
                         "lessons": str(runtime / "status" / agent_rules_runtime.LESSONS_FILE),
                         "board": str(runtime / "status" / agent_rules_runtime.BOARD_FILE)} if agent_rules else None)
        if checkin:
            workspace.write_file(str(runtime / "qwen_team_checkin.mjs"),
                                 Path(__file__).with_name("qwen_team_checkin.mjs").read_text())
        if checkin or agent_rules:
            workspace.write_file(str(runtime / "status" / ".keep"), "")
        registry = WorkerRegistry(workspace)
        delivery = self.team_message_delivery if is_team else None
        disclosure = self.runtime_disclosure
        recovery = self.leader_loop_recovery and is_team  # implies check-ins (validated in __init__)
        system, worker_note = self.system_texts(assignment_policy, communication_policy, is_team=is_team,
                                                status_file=status_file, resources=getattr(workspace, "resources", None))
        policy_mode = self.role_policy_mode if is_team else "off"
        relay = TeamRelay(provider, registry, communication_policy if is_team else "", output / "relay",
                          timeout_seconds, self.upstream, execution_mode=execution_mode,
                          require_caller_metadata=True, max_model_calls=max_model_calls,
                          worker_system_note=worker_note, role_policy_mode=policy_mode, policy_package=self.policy_package,
                          **({"policy_files": policy_files} if policy_files else {}),
                          **({"teammate_limit": self.teammate_limit} if self.teammate_limit != TEAMMATE_LIMIT else {}),
                          **({"source_boundary": source_boundary} if source_boundary is not None else {}))
        server = start_server(relay, workspace.relay_bind_host)
        try:
            relay_url = (workspace.configure_relay(server.server_port)
                         if hasattr(workspace, "configure_relay") else
                         f"http://{workspace.relay_connect_host}:{server.server_port}/v1")
            model_providers = native_model_providers(provider, relay_url)
            if model_providers:
                settings["modelProviders"] = model_providers
            workspace.write_file(str(runtime / "profile/settings.json"), json.dumps(settings))
        except BaseException:
            relay.close()
            relay.stop_policy_files()
            server.shutdown()
            server.server_close()
            raise
        if max_model_calls is None:
            budget_text = ("No fixed team-wide token, request, cost or per-worker evaluation allowance is imposed. "
                           "Finish when the task is ready for delivery; do not add unnecessary work to fill the time."
                           if is_team else
                           "No fixed token, request, cost or evaluation-count allowance is imposed. "
                           "Finish when the task is ready for delivery; do not add unnecessary work to fill the time.")
        else:
            budget_text = (f"The whole task has a shared limit of {max_model_calls} logical model calls, "
                           "including every agent's coding requests and summaries. Transport retries of one "
                           "request do not consume another logical call. This shared limit is our experiment's "
                           "mapping of a model-call budget, not an official multi-agent benchmark protocol. "
                           "Complete integration and checks within this allowance and finish when ready.")
        if self.stop_hook is not None:
            if max_model_calls is None:
                budget_text = budget_text.replace(
                    "Finish when the task is ready for delivery; do not add unnecessary work to fill the time.",
                    self.stop_hook["budget_notice"])
            else:
                budget_text = budget_text.replace(
                    "Complete integration and checks within this allowance and finish when ready.",
                    self.stop_hook["budget_notice"])
        prompt = (f"Complete this repository task using the team and the supplied policies.\n\n"
                  f"TASK:\n{instruction}\n\nDelivery repository: {workspace.repo_path}\n"
                  f"Worker worktree root: {workspace.worker_root}\n"
                  "Workers should commit changes in their own worktrees and hand off commit IDs "
                  "and relevant integration instructions. Assign final integration to a worker, "
                  "including tests in the delivery repository. Only the final delivery repository "
                  "is submitted for benchmark evaluation. Do not access private evaluation tests, "
                  "previous runs, outer experiment outputs or credentials.\n"
                  f"Task wall-time allowance: {timeout_seconds:g} seconds. {budget_text}") if is_team else (
                  f"Complete this repository task directly.\n\nTASK:\n{instruction}\n\n"
                  f"Delivery repository: {workspace.repo_path}\n"
                  "Inspect, implement and test in this repository; it is submitted for benchmark "
                  "evaluation. Do not access private evaluation tests, previous runs, outer "
                  "experiment outputs or credentials.\n"
                  f"Task wall-time allowance: {timeout_seconds:g} seconds. {budget_text}")
        default_path = (os.environ.get("PATH", os.defpath) if workspace.relay_connect_host == "127.0.0.1"
                        else "/opt/ac-node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        env = {"PATH": getattr(workspace, "path_env", default_path), "LANG": "en_US.UTF-8", "NO_COLOR": "1",
               "AGENT_COMPANY_QWEN_HOME": str(runtime / "home"), "QWEN_HOME": str(runtime / "profile"),
               "QWEN_RUNTIME_DIR": str(runtime / "runtime"), "QWEN_CODE_ENABLE_AGENT_TEAM": "1" if is_team else "0",
               "QWEN_CODE_SYSTEM_SETTINGS_PATH": str(runtime / "empty-settings.json"),
               "QWEN_CODE_SYSTEM_DEFAULTS_PATH": str(runtime / "empty-settings.json"),
               "QWEN_CODE_API_TIMEOUT_MS": str(math.ceil(native_request_timeout * 1000)),
               "QWEN_CODE_UNATTENDED_RETRY": "false",
               "AGENT_COMPANY_QWEN_CLI": self.qwen_cli,
               "AGENT_COMPANY_ROOT_OWNER": "master" if is_team else "single",
               "AGENT_COMPANY_RELAY_URL": relay_url,
               # Absent when off, so earlier launches stay byte-identical.
               **({"AGENT_COMPANY_ROLE_POLICY_MODE": policy_mode} if policy_mode != "off" else {}),
               # Protocol identity follows the package for both arms (M1 review minor 7).
               **({"AGENT_COMPANY_ROLE_PACKAGE": "1"} if self.policy_protocol == ROLE_POLICY_PROTOCOL else {}),
               **({"AGENT_COMPANY_POLICY_PROTOCOL": AGENT_RULES_PROTOCOL}
                  if self.policy_protocol == AGENT_RULES_PROTOCOL else {}),
               **({"AGENT_COMPANY_TEAMMATE_LIMIT": str(self.teammate_limit)}
                  if is_team and self.teammate_limit != TEAMMATE_LIMIT else {}),
               "NODE_COMPILE_CACHE": str(runtime / "node-cache")}
        command = [self.node, "--expose-gc", "--require", str(runtime / "qwen_bootstrap.cjs"),
                   "--import", str(runtime / "qwen_native_compat.mjs"),
                   "--import", str(runtime / "qwen_tool_errors.mjs")]
        started_ms = time.time() * 1000
        if checkin or self.stop_hook is not None or self.policy_package is not None:
            env.update({"AGENT_COMPANY_TASK_STARTED_EPOCH_MS": f"{started_ms:.0f}",
                        "AGENT_COMPANY_TASK_DEADLINE_EPOCH_MS": f"{started_ms + timeout_seconds * 1000:.0f}"})
        if self.stop_hook is not None:
            env["AGENT_COMPANY_STOP_HOOK_REASON"] = self.stop_hook["reason"]
        if checkin:
            env.update({"AGENT_COMPANY_CHECKIN_SECONDS": f"{self.checkin_interval_seconds:g}",
                        "AGENT_COMPANY_DELIVERY_REPO": str(workspace.repo_path),
                        "AGENT_COMPANY_STATUS_FILE": status_file})
        if delivery == "between_tool_calls":
            env["AGENT_COMPANY_TEAM_MESSAGE_DELIVERY"] = delivery
        if disclosure:
            env["AGENT_COMPANY_RUNTIME_DISCLOSURE"] = "1"
        if recovery:
            env["AGENT_COMPANY_LEADER_LOOP_RECOVERY"] = "1"
        if self.tool_call_cap != "native_default":
            env["AGENT_COMPANY_TOOL_CALL_CAP"] = self.tool_call_cap  # Recorded by the compat layer.
        if is_team and team_stall_timeout_seconds is not None:
            env["QWEN_TEAM_STALL_TIMEOUT_SECONDS"] = str(team_stall_timeout_seconds)
            command += ["--import", str(runtime / "qwen_stall_timeout.mjs")]
        command += [self.qwen_cli,
                   "--safe-mode", "--auth-type", "openai", "--openai-base-url",
                   relay_url,
                   "--openai-api-key", relay.token, "--model", provider.model,
                   "--output-format", "stream-json", "--approval-mode", "auto-edit",
                   "--allowed-tools", *sorted(COORD_TOOLS | WORKER_TOOLS if is_team else SINGLE_TOOLS)]
        if not is_team:
            command += ["--exclude-tools", *sorted(SINGLE_EXCLUDED_TOOLS)]
        command += ["--max-wall-time", f"{timeout_seconds:g}s", "--max-tool-calls", "-1"]
        if is_team:
            command += ["--max-subagent-depth", "1"]
        if checkin or agent_rules:
            command += ["--include-directories", str(runtime / "status")]
        command += ["--system-prompt", system, "-p", prompt]
        rates = provider.rates
        launch = {"execution_mode": execution_mode, "policy_applied": is_team,
                  "network": getattr(workspace, "network_policy_status", {"enforced": False}),
                  "command": command, "environment": env, "cwd": workspace.repo_path,
                  "model": provider.model, "provider_base_url": provider.base_url,
                  "extra_body": provider.extra_body, "max_output_tokens": provider.max_output_tokens,
                  "rates": None if rates is None else {key: str(value) for key, value in vars(rates).items()},
                  "timeout_seconds": timeout_seconds,
                  "max_model_calls": max_model_calls,
                  "model_call_limit_scope": "task-shared" if max_model_calls is not None else None,
                  "model_call_counting": "admitted relay requests including summaries; transport retries excluded",
                  "official_multi_agent_call_limit_claimed": False,
                  "upstream_request_deadline_seconds": provider.request_timeout_seconds,
                  "first_progress_timeout_seconds": provider.first_progress_timeout_seconds,
                  "progress_idle_timeout_seconds": provider.progress_idle_timeout_seconds,
                  "native_request_timeout_seconds": native_request_timeout,
                  "native_sdk_max_retries": 0,
                  "relay_max_attempts": provider.max_attempts,
                  "relay_retry_backoff_seconds": provider.retry_backoff_seconds,
                  "native_caller_metadata_required": True,
                  "native_unattended_retry": False,
                  "tool_error_feedback": "native_tool_result",
                  "native_outer_retry_note": ("Native outer transport retries have no supported disable setting; "
                                              "terminal provider failures use the relay's non-retry HTTP response."),
                  "native_team_inactivity_timeout_seconds": ((team_stall_timeout_seconds or NATIVE_TEAM_INACTIVITY_DEFAULT_SECONDS)
                                                             if is_team else None),
                  "teammate_limit": self.teammate_limit if is_team else None,
                  "team_message_delivery": delivery,
                  "runtime_disclosure": disclosure,
                  "stop_hook": self.stop_hook,
                  # None: not applicable (Single); the recovery exists only for a Team master.
                  "leader_loop_recovery": recovery if is_team else None,
                  # Applies to every role. Native setting written: absent (adaptive
                  # default) or model.maxToolCallsPerTurn=0 (no per-turn cap).
                  "tool_call_cap": self.tool_call_cap,
                  "native_max_tool_calls_per_turn": settings["model"].get("maxToolCallsPerTurn"),
                  # None: not applicable (Single). Off keeps every request as before.
                  "role_policy_mode": policy_mode if is_team else None,
                  "role_policy_protocol": self.policy_protocol,
                  "policy_package": self.policy_package,
                  # Only agent_rules_v1 adds keys, so earlier launch records stay byte-identical.
                  **({"policy_files": policy_files} if agent_rules else {}),
                  **({"model_visible_text_status": agent_rules_text.TEXT_STATUS}
                     if self.policy_protocol == AGENT_RULES_PROTOCOL else {}),
                  "role_policy_disclosure_chars": ({role: len(agent_rules_runtime.disclosure(
                                                        role, policies_file=policy_files["policies"],
                                                        lessons_file=policy_files["lessons"], board_file=policy_files["board"]))
                                                    for role in ("master", "worker")} if agent_rules else
                                                   {role: len((role_package_runtime.disclosure if self.policy_package is not None
                                                               else role_policy_disclosure)(policy_mode, role))
                                                    for role in ("master", "worker")} if is_team else None),
                  "role_policy_disclosure_version": (agent_rules_runtime.DISCLOSURE_VERSION if agent_rules else
                                                     "role_package_v1_R3" if self.policy_package is not None and is_team else
                                                     ROLE_POLICY_DISCLOSURE_VERSION if is_team and policy_mode != "off" else None),
                  "team_checkin": ({"interval_seconds": self.checkin_interval_seconds, "sender": "runtime:checkin",
                                    "milestones": list(CHECKIN_MILESTONES), "status_file": status_file,
                                    "delivery": "native leader callback and replaceable check-in queue slot; observational, no automatic scaling"}
                                   if checkin else None),
                  "runtime_dir": str(runtime), "assignment_policy": assignment_policy if is_team else None,
                  "communication_policy": communication_policy if is_team else None,
                  "master_tools": sorted(MASTER_TOOLS) if is_team else None,
                  "adaptations": ["isolated startup settings", "role-specific tools",
                                  "master read-only inspection tools (read_file, glob, grep_search)",
                                  "supported named-teammate spawn schema and native tool-error feedback",
                                  team_size_note(self.teammate_limit),
                                  "dynamic caller-owned worktrees", "worker system communication policy on every request",
                                  "native failure notice and pending automatic report delivery before final inbox drain",
                                  "native request owner/purpose and compaction prefix repair",
                                  "configurable native Team inactivity timeout",
                                  "nonstream provider replies converted to SSE",
                                  *(["runtime team check-ins to the master (observational)"] if checkin else []),
                                  *(["teammate messages delivered between tool calls"]
                                    if delivery == "between_tool_calls" else []),
                                  *(["runtime waiting and loop-guard rules disclosed to master and workers"]
                                    if disclosure else []),
                                  *(["master status-file read loop ends only the master's turn while a teammate runs"]
                                    if recovery else []),
                                  *(["no per-turn tool-call cap"] if self.tool_call_cap == "disabled" else []),
                                  *(["per-teammate communication policies (agent_rules_v1): fixed system cards, "
                                     "policy state in each coding request's tail, runtime-written policy and lesson files"]
                                    if agent_rules else
                                    [f"role-level communication policies ({policy_mode}): versioned C, control "
                                     "messages and policy state injected on every coding request"]
                                    if policy_mode != "off" else [])]
                  if is_team else [
                                  "isolated startup settings", "single coding-agent tools",
                                  "native tool-error feedback for rejected calls",
                                  "native Team disabled and subagent tools excluded",
                                  "native request owner/purpose and compaction prefix repair",
                                  "nonstream provider replies converted to SSE",
                                  *(["runtime loop-guard rules disclosed"] if disclosure else []),
                                  *(["no per-turn tool-call cap"] if self.tool_call_cap == "disabled" else [])],
                  "master_boundary": ("coordination plus read-only inspection tools; no edit/write/shell; "
                                      "enforced by tool advertisement and response validation, not an OS sandbox") if is_team else None,
                  "native_team_scheduler_and_mailbox_modified": is_team,
                  "native_compatibility_repairs": ["worker failure inbox delivery before final drain",
                                                   *(["pending native automatic reports settled before final drain"] if is_team else []),
                                                   *(["inbox delivery uses the current leader wake-up callback"] if is_team else []),
                                                   "leading analysis block only in compaction",
                                                   "explicit native request owner and purpose",
                                                   "replaceable runtime check-in slot in native leader queue",
                                                   "non-PTY shell result waits up to 200 ms after exit for output",
                                                   *(["master status-file read loop guard ends the turn into native "
                                                      "teammate waiting (opt-in)"] if recovery else [])],
                  "scripted_provider": self.upstream is not None}
        relay.save(output / "launch.json", launch)
        started = time.monotonic()
        execution, error = {}, None
        tail = (LifecycleTail(output / "qwen_stderr.txt", relay.record_worker_terminal,
                              on_spawn_result=relay.record_native_spawn_result,
                              **({"on_event": relay.record_lifecycle_event} if agent_rules else {})) if is_team else None)
        collaboration_tail = (CollaborationTail(output / "qwen_stderr.txt", relay.collaboration)
                              if is_team and relay.collaboration is not None else None)
        if collaboration_tail is not None:
            relay.poll_collaboration = collaboration_tail.poll
        if tail is not None:
            relay.poll_worker_lifecycle = tail.poll
            relay.require_native_spawn_results = True
            tail.start()
        try:
            execution = workspace.run_agent(command, env=env, timeout_seconds=timeout_seconds,
                stdout_path=output / "qwen_stdout.jsonl", stderr_path=output / "qwen_stderr.txt")
        except Exception as exc:
            error = relay.redact(json.dumps(exception_diagnostic(exc)))
            relay.event("launcher_error", error=error)
        finally:
            relay.close()
            relay.drain(provider.request_timeout_seconds + 2)
            server.shutdown()
            server.server_close()
        role_policy = None
        if tail is not None:
            # Consume the remaining lifecycle lines, then void what the run left pending.
            tail.stop()
            relay.record_worker_lifecycle_health(tail.summary())
        if collaboration_tail is not None:
            collaboration_tail.poll()
        if relay.role_policy is not None:
            role_policy = {"mode": policy_mode, "lifecycle_tail": tail.summary()}
            try:
                role_policy["voided_at_run_end"] = relay.role_policy.end_run()
                relay.save(output / "relay" / "role_policy_state.json", relay.role_policy.export())
                if relay.agent_rules and not relay.finish_policy_files():
                    role_policy["policy_files_flush"] = "timed_out"
            except Exception as exc:
                role_policy["finalization_error"] = relay.redact(f"{type(exc).__name__}: {exc}")
                # Lost policy evidence is an evidence failure, not a normal completion (review 10-18 F1).
                relay.record_evidence_failure(exc)
            if relay.rules is not None:
                # Its own step: a store finalization failure must not skip rule settlement.
                try:
                    role_policy["rule_obligations"] = relay.finish_rule_engine(output / "relay" / "rule_obligations.json")
                except Exception as exc:
                    role_policy["rule_finalization_error"] = relay.redact(f"{type(exc).__name__}: {exc}")
                    relay.record_evidence_failure(exc)
        relay.stop_policy_files()
        cli_result = native_result(output / "qwen_stdout.jsonl")
        cli_result["lifecycle"] = native_lifecycle(output / "qwen_stderr.txt")
        relay_summary = relay.summary()
        result = {"model": provider.model, "execution": execution, "launcher_error": error,
                  "native_result": cli_result,
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "provider_incidents": provider_incident_summary(relay_summary), **relay_summary,
                  "runtime_dir": str(runtime), "scripted_provider": self.upstream is not None,
                  "stop_reason": runtime_stop_reason(execution, cli_result, error, relay_summary),
                  **({"role_policy_run": role_policy} if role_policy is not None else {})}
        relay.save(output / "summary.json", result)
        return result
