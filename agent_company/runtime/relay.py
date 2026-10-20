"""Provider transport and observation for native Qwen single/team execution.

This adapts model options, tool visibility, and spawn arguments. It does not
implement a second scheduler or message bus. Prices are caller-supplied estimates.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, field
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path, PurePosixPath
import re
import selectors
import socket
import threading
import time
import uuid

from agent_company.runtime.team import WorkerRegistry, validate_worker_name
from agent_company.policy.role_policy import (MODES as ROLE_POLICY_MODES, ROLE_MODES, PolicyPersistError, PolicyRejection,
                          RolePolicyStore, SpawnCall, parse_send_message, parse_spawn)
from agent_company.policy.role_policy_runtime import render_injection
from agent_company.policy import role_package_store as package_store
from agent_company.policy import role_package_runtime as package_runtime
from agent_company.policy import agent_rules, agent_policy_store
from agent_company.policy import agent_rules_runtime
from agent_company.policy import agent_rules_text
from agent_company.policy.role_package import AGENT_RULES_PROTOCOL, PROTOCOL_MODES, policy_protocol
from agent_company.runtime.collaboration import CollaborationRuntime, Rejection as CoordRejection
from agent_company.runtime.rule_engine import RuleEngine, message_hash, party
from agent_company.runtime.reflection import KINDS as REFLECTION_KINDS, ReflectionFunnel
from agent_company.runtime.collaboration_text import DELIVERY_PROTOCOL, COLLABORATION_PROTOCOL, COORD_PROTOCOL
from agent_company.policy.source_boundary import SOURCE_BOUNDARY, append_system_text
from agent_company.runtime.transport import (InvalidProviderResponse, OpenAIChatTransport, RequestCancelled, RequestControl,
                        RequestDeadlineExceeded, RequestProgressTimeout, TransportError)

COORD_TOOLS = {"team_create", "agent", "list_agents", "send_message", "request_shutdown",
               "task_create", "task_list", "task_update"}
# Read-only inspection (2026-09-26): the master may read, list and search files before
# and during decomposition, but still cannot edit, write or run commands.
READ_ONLY_TOOLS = {"read_file", "glob", "grep_search"}
MASTER_TOOLS = COORD_TOOLS | READ_ONLY_TOOLS
SINGLE_TOOLS = READ_ONLY_TOOLS | {"edit", "write_file", "run_shell_command"}
WORKER_TOOLS = SINGLE_TOOLS | {"send_message", "task_create", "task_list", "task_update"}
# Native CLI denies, in addition to relay filtering/response validation. Depth 0
# is not a supported way to disable subagents in the released Qwen CLI.
SINGLE_EXCLUDED_TOOLS = COORD_TOOLS | {"team_delete", "team_plan_approval"}
TOOL_ERROR_FIELD = "__agent_company_tool_error"
MODEL_CALL_LIMIT_MESSAGE = "model_call_limit_exceeded: shared logical model-call budget exhausted"
TEAMMATE_LIMIT = 10


def validate_teammate_limit(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("teammate_limit must be a positive integer")
    return value


def teammate_limit_message(limit: int = TEAMMATE_LIMIT) -> str:
    return (f"Teammate limit reached: at most {limit} teammates can be active at the same time. "
            "Reuse an existing teammate with send_message, or wait for one to shut down.")


def team_size_note(limit: int = TEAMMATE_LIMIT) -> str:
    return (f"At most {limit} teammates can be active at the same time; a teammate that has shut down or failed frees its slot. "
            "Choose team size to suit the task; all teammates "
            "share the run's existing CPU, memory, wall-time and any configured model-call budget. "
            "More teammates do not add resources or extend the deadline.")


# The default texts are byte-identical to earlier runs.
TEAMMATE_LIMIT_MESSAGE = teammate_limit_message()
TEAM_SIZE_NOTE = team_size_note()


def validate_model_call_limit(value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("max_model_calls must be a positive integer or None")


class EvidenceWriteError(OSError):
    """A local audit write failed; this is never a provider retry signal."""


class DuplicateToolArguments(ValueError):
    """Tool arguments repeat a JSON key, so no single intended value exists."""

    def __init__(self, tool: str, repeated: dict[str, int]):
        self.tool, self.repeated = tool, dict(repeated)
        shown = ", ".join(f"{json.dumps(key[:64], ensure_ascii=False)} x{count}"
                          for key, count in sorted(repeated.items(), key=lambda item: -item[1])[:8])
        super().__init__(f"Arguments for {tool} repeat JSON keys ({shown}). Each key must appear once; "
                         "arguments with repeated keys are ambiguous and are never executed.")


def parse_tool_arguments(raw: str):
    """Parse a tool argument string without JSON's last-value-wins rule.

    Returns the parsed value and {key: occurrences} for every key repeated inside
    any single object at any depth. Invalid JSON still raises ValueError.
    """
    repeated: Counter = Counter()

    def pairs(items):
        counts = Counter(key for key, _ in items)
        repeated.update({key: count for key, count in counts.items() if count > 1})
        return dict(items)

    return json.loads(raw, object_pairs_hook=pairs), dict(repeated)


def strict_tool_arguments(name: str, raw: str) -> dict:
    args, repeated = parse_tool_arguments(raw)
    if repeated:
        raise DuplicateToolArguments(name, repeated)
    if not isinstance(args, dict):
        raise ValueError(f"Arguments for {name} must be a JSON object")
    return args


def model_visible_arguments(raw: str) -> str:
    """What the model's own history shows for a call the relay annotated or rewrote.

    A valid, unambiguous object is shown exactly as the model wrote it. Invalid,
    non-object or duplicate-key text is omitted as ``{}``: it was never executable,
    can be very long, and the failed tool result states why it was rejected.
    """
    try:
        args, repeated = parse_tool_arguments(raw)
    except ValueError:
        return "{}"
    return raw if isinstance(args, dict) and not repeated else "{}"


def describe_team_spawn(tool: dict) -> None:
    """Advertise only the native teammate mode this adapter can execute.

    The released CLI also advertises read-only/Explore/fork agents, but the
    experiment already rejects those modes. Keep the model-facing declaration
    consistent with that boundary without changing native execution or policies.
    This function receives a private copy, never the captured Qwen request.
    """
    function = tool["function"]
    function["description"] = (
        "Spawn a named, writable general-purpose teammate in the active native team. "
        "Call team_create first. Every teammate inherits the master's model and runs "
        "concurrently in an automatically assigned Git worktree. Use these same teammates "
        "for investigation, implementation, testing and integration; an investigation-only "
        "assignment is expressed in the prompt, not a separate agent type or permission mode. "
        "Provide a concrete task, write scope, dependencies and expected handoff. "
        "Reuse existing teammates with send_message for follow-up work. "
        "Omit working_dir, including for integration: the assigned startup worktree "
        "and the delivery repository are different. A teammate can run git -C in "
        "the delivery repository after starting in its own worktree. "
        "Only the parameters below are supported; omit all other native agent options."
    )
    function["parameters"] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,63}$",
                     "description": "Short unique teammate name; not master, leader, all or broadcast."},
            "description": {"type": "string", "description": "Short description of the assignment."},
            "prompt": {"type": "string", "minLength": 1,
                       "description": "Concrete assignment, boundaries and expected deliverable."},
            "subagent_type": {"type": "string", "enum": ["general-purpose"],
                              "description": "The only supported teammate type; may be omitted."},
        },
        "required": ["name", "description", "prompt"],
        "additionalProperties": False,
    }


def describe_team_create(tool: dict, limit: int = TEAMMATE_LIMIT) -> None:
    """Keep the native tool's count wording consistent with the Team adapter."""
    function = tool["function"]
    description = function.get("description", "")
    description = re.sub(r"\(max [^)]+ teammates per team\)",
                         f"(at most {limit} active teammates at a time)", description)
    function["description"] = description + "\n\n" + team_size_note(limit)


@dataclass(frozen=True)
class TokenRates:
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal
    currency: str = "USD"

    def __post_init__(self):
        for key in ("input_per_million", "cached_input_per_million", "output_per_million"):
            value = Decimal(str(getattr(self, key)))
            if not value.is_finite() or value < 0:
                raise ValueError("Token rates must be finite and nonnegative")
            object.__setattr__(self, key, value)


@dataclass
class ProviderConfig:
    model: str
    base_url: str
    api_key: str = field(default="", repr=False)
    max_output_tokens: int | None = None
    request_timeout_seconds: float = 120
    extra_body: dict = field(default_factory=dict)
    rates: TokenRates | None = None
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    inspection_max_output_tokens: int = 4096
    stream_response: bool = False
    # Opt-in per-attempt waits. request_timeout_seconds remains the immutable
    # logical request limit, including all retries and backoff.
    first_progress_timeout_seconds: float | None = None
    progress_idle_timeout_seconds: float | None = None

    def __post_init__(self):
        for name in ("first_progress_timeout_seconds", "progress_idle_timeout_seconds"):
            value = getattr(self, name)
            if value is not None:
                try:
                    valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                             and math.isfinite(value) and value > 0)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError(f"{name} must be null or a finite positive number")
        if not isinstance(self.stream_response, bool):
            raise ValueError("stream_response must be bool")
        if (isinstance(self.inspection_max_output_tokens, bool)
                or not isinstance(self.inspection_max_output_tokens, int)
                or self.inspection_max_output_tokens < 1):
            raise ValueError("inspection_max_output_tokens must be a positive integer")
        if (not self.model or isinstance(self.request_timeout_seconds, bool)
                or not isinstance(self.request_timeout_seconds, (int, float))
                or not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0):
            raise ValueError("A model and finite positive request timeout are required")
        if self.max_output_tokens is not None and (isinstance(self.max_output_tokens, bool) or
                not isinstance(self.max_output_tokens, int) or self.max_output_tokens < 1):
            raise ValueError("max_output_tokens must be a positive integer")
        if (isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int)
                or self.max_attempts < 1):
            raise ValueError("max_attempts must be a positive integer")
        if (isinstance(self.retry_backoff_seconds, bool)
                or not isinstance(self.retry_backoff_seconds, (int, float))
                or not math.isfinite(self.retry_backoff_seconds) or self.retry_backoff_seconds < 0):
            raise ValueError("retry_backoff_seconds must be finite and nonnegative")
        if {"model", "messages", "tools", "stream"} & self.extra_body.keys():
            raise ValueError("extra_body cannot override model, messages, tools, or stream")


def usage_and_cost(raw: dict, rates: TokenRates | None) -> tuple[dict, Decimal | None]:
    prompt, output = raw["prompt_tokens"], raw["completion_tokens"]
    detail = raw.get("prompt_tokens_details") or {}
    if not isinstance(detail, dict):
        raise ValueError("Invalid prompt_tokens_details")
    cached = raw.get("prompt_cache_hit_tokens", detail.get("cached_tokens", 0))
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (prompt, output, cached)) or cached > prompt:
        raise ValueError("Invalid token usage")
    if "prompt_cache_miss_tokens" in raw and raw["prompt_cache_miss_tokens"] != prompt - cached:
        raise ValueError("Inconsistent cache usage")
    usage = {"prompt_tokens": prompt, "completion_tokens": output,
             "cached_prompt_tokens": cached, "total_tokens": prompt + output}
    cost = None if rates is None else ((prompt - cached) * rates.input_per_million +
           cached * rates.cached_input_per_million + output * rates.output_per_million) / Decimal(1_000_000)
    return usage, cost


def role_of(body: dict) -> str:
    system = "\n".join(str(m.get("content", "")) for m in body.get("messages", [])
                       if m.get("role") in {"system", "developer"})
    matches = re.findall(r'^You are (?:read-only )?agent "([^"\n]+)" in team', system, re.MULTILINE)
    if len(set(matches)) > 1:
        raise ValueError("Ambiguous native teammate identity")
    return matches[0] if matches else "master"


def validate_completion(response):
    """Reject provider failures/empty envelopes before tool or SSE processing."""
    if not isinstance(response, dict):
        raise ValueError("Provider completion must be a JSON object")
    if response.get("error") is not None:
        raise ValueError("Provider returned an error instead of a completion")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Provider completion has no choices")
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("Provider choice has no assistant message")
        message = choice["message"]
        if message.get("role") != "assistant":
            raise ValueError("Provider completion message is not from the assistant")
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ValueError("Provider tool_calls must be a list")
        if not any(isinstance(message.get(key), str) for key in ("content", "refusal", "reasoning_content")) and not calls:
            raise ValueError("Provider assistant message contains no content or tool calls")
        for call in calls or []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("Provider tool call has no function")
            function = call["function"]
            if (not isinstance(function.get("name"), str) or not function["name"].strip()
                    or not isinstance(function.get("arguments"), str)):
                raise ValueError("Provider tool call must have a name and JSON argument string")


class TeamRelay:
    def __init__(self, provider: ProviderConfig, registry: WorkerRegistry, communication_policy: str,
                 output_dir: Path, timeout_seconds: float, upstream=None, *, execution_mode: str = "team",
                 require_caller_metadata: bool = False, max_model_calls: int | None = None,
                 source_boundary: str = SOURCE_BOUNDARY, worker_system_note: str = "",
                 role_policy_mode: str = "off", policy_package: dict | None = None,
                 policy_files: dict | None = None, teammate_limit: int = TEAMMATE_LIMIT):
        if execution_mode not in {"single", "team"}:
            raise ValueError("execution_mode must be single or team")
        if role_policy_mode != "off" and (role_policy_mode not in ROLE_POLICY_MODES or execution_mode != "team"):
            raise ValueError("role_policy_mode must be off, or global/master/propose for a Team run")
        validate_model_call_limit(max_model_calls)
        self.teammate_limit = validate_teammate_limit(teammate_limit)
        if not isinstance(source_boundary, str) or not source_boundary.strip():
            raise ValueError("A fixed benchmark source boundary is required")
        self.source_boundary = source_boundary
        # Runtime facts for teammates (e.g. message delivery); empty keeps earlier requests unchanged.
        if not isinstance(worker_system_note, str):
            raise ValueError("worker_system_note must be text")
        self.worker_system_note = worker_system_note
        self.execution_mode = execution_mode
        self.max_model_calls = max_model_calls
        self.require_caller_metadata = require_caller_metadata
        self.provider, self.registry = provider, registry
        self.communication_policy = communication_policy
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.upstream = upstream if upstream is not None else OpenAIChatTransport(provider)
        self.token = "local-team-" + uuid.uuid4().hex
        self.deadline = time.monotonic() + timeout_seconds
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._closed = False
        self._attempts = 0
        self._inflight: dict[int, str] = {}
        self._inflight_started: dict[int, float] = {}
        self._controls: dict[int, RequestControl] = {}
        # One native call owns its role until every sequential retry has ended.
        # Attempt IDs and accounting remain per actual provider call.
        self._active_requests: dict[str, RequestControl] = {}
        self._records: list[dict] = []
        self._deadline_rejections: set[str] = set()
        self._logical_model_calls = 0
        self._logical_calls_by_role: Counter = Counter()
        self._logical_calls_by_purpose: Counter = Counter()
        self._model_call_budget_rejections: Counter = Counter()
        self._budget_evidence_errors: list[dict] = []
        # (role, call id) -> [{"name", "delivered", "visible"}] for calls whose
        # delivered arguments differ from what the model wrote. In memory only;
        # the original text is already in each attempt's provider response file.
        self._model_views: dict[tuple[str, str], list[dict]] = {}
        # Reserve before native dispatch, across whole response batches. An idle
        # teammate retains its slot; only native terminal evidence or a matched
        # failed spawn result releases it. Registry entries are historical.
        self._spawn_leases: dict[int, dict] = {}
        self._spawn_sequence = 0
        self._terminal_workers: set[str] = set()
        self.poll_worker_lifecycle = None
        self.require_native_spawn_results = False
        self._worker_lifecycle_tail = None
        self._history_totals: Counter = Counter()
        # Role-level communication policies (docs/stage1_role_policy_harness_plan_20261009.md).
        # Off keeps every earlier request byte-identical. Lock order: the store calls
        # self.event() while holding its own lock, so the relay never calls the store
        # while holding self._lock.
        self.role_policy_mode = role_policy_mode
        self.policy_package = copy.deepcopy(policy_package)
        # The package's schema selects the protocol; mode never does (plan v4 section 2).
        self.policy_protocol = policy_protocol(policy_package) if policy_package is not None else None
        self.agent_rules = self.policy_protocol == AGENT_RULES_PROTOCOL
        if policy_package is not None and execution_mode == "team" and role_policy_mode not in PROTOCOL_MODES[self.policy_protocol]:
            raise ValueError(f"{self.policy_protocol} implements mode(s) {', '.join(PROTOCOL_MODES[self.policy_protocol])}")
        # agent_rules_v1 read-only files: {"policies", "lessons", "board": path} in the task runtime.
        self.policy_files = dict(policy_files or {})
        if self.policy_files and (not self.agent_rules or not {"policies", "lessons"} <= set(self.policy_files)
                                  or not set(self.policy_files) <= {"policies", "lessons", "board"}):
            raise ValueError("policy_files are the agent_rules_v1 policies, lessons and board paths")
        # Files are written by one background thread from the latest store state,
        # never on a request's return path (M1 review minor 10).
        self._files_lock = threading.Lock()
        self._files_changed = threading.Condition(threading.Lock())
        self._files_requested = self._files_written = 0
        self._files_thread = None
        self._files_stop = False
        # (role, source_request, tool_call_id) -> text appended to that call's native result.
        self._policy_receipts: dict[tuple[str, int | None, str], dict] = {}
        self.collaboration = (CollaborationRuntime(role_policy_mode, self._write_policy_batch,
                                                   team=execution_mode == "team", protocol=self.policy_protocol)
                              if policy_package is not None else None)
        self.poll_collaboration = None
        self._policy_counts: Counter = Counter()
        # (role, source_request, tool_call_id) -> committed call. A provider may reuse call
        # ids, so a result is matched by the position of the answered assistant message in
        # the role's history and by the call's complete arguments.
        self._policy_spawns: dict[tuple[str, int | None, str], dict] = {}
        self._policy_notifications: dict[tuple[str, int | None, str], dict] = {}
        self.role_policy = (agent_policy_store.AgentPolicyStore(policy_package, self._write_policy_batch)
                            if self.agent_rules and execution_mode == "team" else
                            package_store.RolePackageStore(role_policy_mode, policy_package, self._write_policy_batch)
                            if policy_package is not None and execution_mode == "team" else
                            RolePolicyStore(role_policy_mode, communication_policy, self._write_policy_batch)
                            if role_policy_mode != "off" else None)
        # agent_rules_v1 rule obligations (plan v4 section 7) and team observations for check-ins.
        self.rules = None
        self.timeout_seconds = timeout_seconds
        self._rule_sends: dict[tuple[str, int, str], dict] = {}
        self._message_pairs: Counter = Counter()
        self._shutdown_requests: dict[str, float] = {}
        self._git_coverage: dict[str, tuple] = {}
        self._response_times: dict[int, float] = {}
        # (previous, current) native sample times in seconds: per worktree, and for the task board.
        self._git_samples: dict[str, tuple] = {}
        self._delivery_sampled_at = None
        self._task_samples: tuple = (None, None)
        self._terminal_at: dict[str, float] = {}
        self._run_end_epoch = time.time() + timeout_seconds
        # agent_rules_v1 spawn gate (plan v4 section 8): lessons the master has seen at a hold,
        # whether the next hold shows the whole library (first hold, or after compaction),
        # and whether the spawn right after a hold must go through.
        self._gate_seen: set[str] = set()
        self._gate_full_next = True
        self._gate_release = False
        self._gate_held_at: float | None = None
        # Board posts that existed at each agent's previous tail (new posts are marked).
        self._board_seen: dict[str, int] = {}
        self.reflection = None
        if self.agent_rules and execution_mode == "team":
            self.rules = RuleEngine(self.role_policy.policy_of, self._rule_event, started_at=time.time(),
                                    duration_seconds=timeout_seconds)
            self.collaboration.git_listener = self._rule_git
            self.collaboration.task_listener = self._rule_task
            self.collaboration.notice_listener = self._rule_notice
            self.collaboration.sample_listener = self._rule_sample
            self.reflection = ReflectionFunnel(self._rule_event)
        self._publish_policy_files()

    def request_control(self, owner: str | None) -> RequestControl:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            with self._lock:
                self._deadline_rejections.add(owner or "unknown")
            raise RequestDeadlineExceeded("task_deadline_exceeded: task budget exhausted")
        return RequestControl(min(self.provider.request_timeout_seconds, remaining))

    def redact(self, text: str) -> str:
        return text.replace(self.provider.api_key, "[REDACTED]") if self.provider.api_key else text

    def save(self, path: Path, data):
        path.write_text(self.redact(json.dumps(data, ensure_ascii=False, indent=2)) + "\n")

    def event(self, event: str, **fields):
        with self._lock:
            with (self.output / "events.jsonl").open("a") as stream:
                stream.write(self.redact(json.dumps({"time": time.time(), "event": event, **fields}, ensure_ascii=False)) + "\n")

    def _write_policy_batch(self, record: dict) -> None:
        """Role-policy batches go to events.jsonl as one line each (replayable)."""
        self.event(record["event"], **{key: value for key, value in record.items() if key != "event"})

    def caller(self, body, owner=None, purpose=None):
        if self.require_caller_metadata and (owner is None or purpose is None):
            raise ValueError("Native caller owner and purpose are required")
        purpose = "coding" if purpose is None else purpose
        if not isinstance(purpose, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", purpose):
            raise ValueError("Invalid native request purpose")
        if owner is None:
            role = role_of(body)  # Legacy direct API only; native runs require headers.
        else:
            if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", owner):
                raise ValueError("Invalid native caller owner")
            role = owner
            # A concrete teammate declaration must agree, but a generic summary
            # system prompt is not evidence that its owner is the master.
            declared = role_of(body)
            if declared != "master" and declared != role:
                raise ValueError("Native caller owner disagrees with teammate identity")
        if self.execution_mode == "single":
            if role not in {"master", "single"}:
                raise ValueError("Native teammates are disabled in single mode")
            role = "single"
        else:
            if role != "master" and not self.registry.contains(role):
                raise ValueError("Unregistered teammate identity")
        return role, purpose

    def payload(self, body, *, owner=None, purpose=None, history=None, policy=None):
        role, purpose = self.caller(body, owner, purpose)
        allowed = SINGLE_TOOLS if self.execution_mode == "single" else (
            MASTER_TOOLS if role == "master" else WORKER_TOOLS)
        payload = copy.deepcopy(body)
        counts = self.restore_model_history(role, payload.get("messages"))
        self._observe_spawn_results(role, payload.get("messages"))
        if history is not None:
            history.update(counts)
        snapshot = None
        if self.role_policy is not None:
            if role != "master" and self.role_policy.record_first_request(role) == "bound":
                self._publish_policy_files()
            # Observe native results first: a receipt never alters what is observed.
            self._observe_policy_tool_results(role, payload.get("messages"))
            if self.agent_rules:
                if self.rules is not None and role != "master" and self.poll_worker_lifecycle is not None:
                    # Native send records (with broadcast recipients) first (M2 review N7).
                    self.poll_worker_lifecycle()
                self._observe_rule_sends(role, payload.get("messages"))
                self._apply_policy_receipts(role, payload.get("messages"))
            if purpose == "coding":
                snapshot = self.role_policy.snapshot(role)
                if policy is not None:
                    policy["snapshot"] = snapshot
                    # The response becomes the next assistant message of this history:
                    # its ordinal locates the call's result in later requests.
                    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                    policy["assistant_index"] = sum(1 for message in messages
                                                    if isinstance(message, dict) and message.get("role") == "assistant")
        payload["model"] = self.provider.model
        payload["stream"] = self.provider.stream_response
        payload.pop("stream_options", None)
        if payload["stream"]:
            payload["stream_options"] = {"include_usage": True}
        payload["tools"] = [t for t in payload.get("tools", []) if t.get("function", {}).get("name") in allowed]
        for tool in payload["tools"]:
            if purpose == "coding" and tool["function"]["name"] == "agent":
                describe_team_spawn(tool)
                if self.agent_rules:
                    tool["function"]["description"] += " " + agent_rules_text.SPAWN_TOOL_NOTE
            if purpose == "coding" and tool["function"]["name"] == "team_create":
                describe_team_create(tool, self.teammate_limit)
        if snapshot is not None:
            # Versioned C, fixed disclosure and role-policy state, rendered from the
            # snapshot this request carries (also for the master, whose static
            # system prompt omits C in these modes).
            injection = (agent_rules_runtime.disclosure(role, **self._file_names())
                         if self.agent_rules else
                         package_runtime.render_injection(snapshot) if self.policy_package is not None
                         else render_injection(snapshot))
            if self.collaboration is not None and not self.agent_rules:
                self.collaboration.bind(snapshot)
            append_system_text(payload["messages"], injection)
            if policy is not None:
                policy["injected_chars"] = len(injection)
            if role != "master" and self.worker_system_note:
                append_system_text(payload["messages"], self.worker_system_note)
        elif purpose == "coding" and self.execution_mode == "team" and role != "master":
            # Native compaction may summarize the initial assignment. Keep the
            # run's policy in the model-facing system prompt on every request,
            # without modifying native history or the message delivery loop.
            suffix = "Communication policy for this task:\n" + self.communication_policy
            append_system_text(payload["messages"], suffix)
            if self.worker_system_note:
                append_system_text(payload["messages"], self.worker_system_note)
        native_max_tokens = payload.get("max_tokens")
        if purpose == "coding":
            payload.update(copy.deepcopy(self.provider.extra_body))
        else:
            # Native auxiliary generation options (e.g. summary output/thinking
            # limits) override provider defaults, not the other way around.
            defaults = copy.deepcopy(self.provider.extra_body)
            if isinstance(payload.get("thinking"), dict) and payload["thinking"].get("type") == "disabled":
                defaults.pop("thinking", None)
                defaults.pop("reasoning_effort", None)
            payload = {**defaults, **payload}
        if self.provider.max_output_tokens is not None:
            if purpose == "coding":
                payload["max_tokens"] = (
                    min(self.provider.max_output_tokens, native_max_tokens)
                    if type(native_max_tokens) is int and native_max_tokens > 0
                    else self.provider.max_output_tokens)
            elif "max_tokens" not in payload:
                payload["max_tokens"] = self.provider.max_output_tokens
        # This fixed boundary is independent of A/C and survives fresh sessions
        # and native compaction. Docker enforces the network restriction; the
        # instruction explains it and is not used as the enforcement mechanism.
        append_system_text(payload["messages"], self.source_boundary)
        if self.policy_package is not None and purpose == "coding":
            append_system_text(payload["messages"], DELIVERY_PROTOCOL)
        if (self.collaboration is not None and purpose == "coding" and self.execution_mode == "team"
                and not self.agent_rules):
            append_system_text(payload["messages"], COLLABORATION_PROTOCOL + "\n\n" + COORD_PROTOCOL)
        return role, payload

    def _file_names(self):
        return {"policies_file": self.policy_files.get("policies", "<policies file>"),
                "lessons_file": self.policy_files.get("lessons", "<lessons file>"),
                "board_file": self.policy_files.get("board", "<board file>")}

    def restore_model_history(self, role, messages) -> dict:
        """Show each call the relay rewrote or annotated as the model wrote it.

        Native Qwen keeps the delivered arguments in history: spawns with the
        assigned working_dir and appended note, rejected calls with the adapter
        error field. A history call is replaced only when this relay recorded it
        for the same role, call ID, tool name and delivered arguments; anything
        else is left unchanged and counted. Native execution used the delivered
        version and is not affected.
        """
        counts: Counter = Counter()
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
                    continue
                try:
                    current = json.loads(function["arguments"])
                except ValueError:
                    continue
                identifier = call.get("id")
                with self._lock:
                    views = list(self._model_views.get((role, identifier), ())
                                 if isinstance(identifier, str) else ())
                visible = {view["visible"] for view in views
                           if view["name"] == function.get("name") and view["delivered"] == current}
                if len(visible) == 1:
                    function["arguments"] = visible.pop()
                    counts["restored"] += 1
                elif views:
                    counts["mismatched"] += 1
                elif isinstance(current, dict) and (TOOL_ERROR_FIELD in current or (
                        function.get("name") == "agent" and "working_dir" in current)):
                    counts["unrecorded"] += 1
        return dict(counts)

    def _record_model_views(self, role, response, delivered):
        """Remember what the model wrote for every call whose delivery differs."""
        for raw_choice, choice in zip(response.get("choices") or [], delivered.get("choices") or []):
            raw_calls = raw_choice.get("message", {}).get("tool_calls") or []
            calls = choice.get("message", {}).get("tool_calls") or []
            for raw_call, call in zip(raw_calls, calls):
                original, sent = raw_call["function"]["arguments"], call["function"]["arguments"]
                if sent == original:
                    continue
                view = {"name": call["function"]["name"], "delivered": json.loads(sent),
                        "visible": self.redact(model_visible_arguments(original))}
                with self._lock:
                    views = self._model_views.setdefault((role, call.get("id")), [])
                    if view not in views:
                        views.append(view)

    def record_worker_terminal(self, worker, source):
        """Only the native lifecycle tail may declare an instance ended."""
        with self._lock:
            self._terminal_workers.add(worker)
            released = [key for key, entry in self._spawn_leases.items()
                        if entry.get("native_name") == worker]
            for key in released:
                self._spawn_leases.pop(key)
            self.event("teammate_slots_released", worker=worker, source=source,
                       released=len(released), active_teammates=len(self._spawn_leases))
        if self.role_policy is not None:
            self.role_policy.record_terminal(worker, source=source)
            self._publish_policy_files()
        if self.rules is not None:
            self.rules.worker_ended(worker, at=self._terminal_at.pop(worker, None))
            self.reflection.worker_ended(worker)
            with self._lock:
                self._shutdown_requests.pop(worker, None)
        if self.collaboration is not None:
            self.collaboration.terminal(worker, source)

    def record_native_spawn_result(self, record, source):
        """Bind actual native names even if compaction removes tool history."""
        with self._lock:
            entry = next(((key, value) for key, value in self._spawn_leases.items()
                          if value["worker"] == record["requested_name"]
                          and value["tool_call_id"] == record["tool_call_id"]
                          and value.get("native_name") is None), None)
            if entry is None:
                return  # already resolved by the exact native tool result
            key, value = entry
            value["native_name"] = record["worker"]
            # agent_rules_v1 binds policy to the requested name; a different native
            # identity would leave that teammate without its policy (M1 review Q2).
            mismatch = self.agent_rules and record["worker"] != record["requested_name"]
            if not record["ok"] or record["worker"] in self._terminal_workers:
                self._spawn_leases.pop(key)
            self.event("teammate_native_spawn_result", worker=record["worker"], source=source,
                       requested_name=record["requested_name"], ok=record["ok"],
                       active_teammates=len(self._spawn_leases))
            if mismatch:
                self.event("agent_policy_name_mismatch", requested_name=record["requested_name"], worker=record["worker"])
                self._budget_evidence_failure(
                    f"Native teammate name {record['worker']!r} differs from requested {record['requested_name']!r}; "
                    "its agent_rules_v1 policy is bound to the requested name", "master", "native-lifecycle")

    def record_worker_lifecycle_health(self, tail):
        """Keep the common Team lifecycle diagnostics in final evidence, including off."""
        with self._lock:
            self._worker_lifecycle_tail = copy.deepcopy(tail)
            if tail.get("callback_errors") or tail.get("invalid_lines") or tail.get("partial_line_pending"):
                self._budget_evidence_failure(
                    "Native teammate lifecycle evidence is incomplete: " + json.dumps(tail),
                    "master", "native-lifecycle")

    def _observe_spawn_results(self, role, messages):
        # Match native tool results to the exact dispatched assistant call, not
        # text quoted by a worker. Native lifecycle records are authoritative;
        # this fallback also covers a native validation failure before invocation.
        if not isinstance(messages, list):
            return
        with self._lock:
            tracked = {(entry["assistant_index"], entry["tool_call_id"]): (key, entry)
                       for key, entry in self._spawn_leases.items()
                       if entry["role"] == role and entry.get("native_name") is None}
            ordinal, calls = -1, {}
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "assistant":
                    ordinal += 1
                    calls = {call.get("id"): call_fingerprint(call)
                             for call in message.get("tool_calls") or [] if isinstance(call, dict)}
                    continue
                identifier = message.get("tool_call_id")
                if message.get("role") != "tool" or (ordinal, identifier) not in tracked:
                    continue
                key, entry = tracked.pop((ordinal, identifier))
                if entry["fingerprint"] is None or calls.get(identifier) != entry["fingerprint"]:
                    continue
                text = tool_result_text(message.get("content"))
                match = re.match(r'^Teammate "([^"\n]+)" is now running concurrently\.', text)
                if match and self.require_native_spawn_results:
                    # The native success text prints the requested name, while
                    # the lifecycle record has its sanitized actual identity.
                    continue
                if match:
                    entry["native_name"] = match[1]
                if not match or match[1] in self._terminal_workers:
                    self._spawn_leases.pop(key)
                self.event("teammate_spawn_result", worker=entry["worker"],
                           native_name=match[1] if match else None,
                           outcome="running" if match else "failed",
                           active_teammates=len(self._spawn_leases))

    def _discard_spawn_leases(self, source_request, tool_call_ids=None):
        with self._lock:
            for key in [key for key, entry in self._spawn_leases.items()
                        if entry["source_request"] == source_request and (tool_call_ids is None or entry["tool_call_id"] in tool_call_ids)]:
                self._spawn_leases.pop(key)

    def prepare_response(self, role, payload, response, *, purpose="coding", policy=None):
        delivered = copy.deepcopy(response)
        if policy is not None:
            policy.pop("lesson_gate", None)  # decided again for each provider response
        if self.policy_package is not None and self.execution_mode == "team":
            identifiers = [call.get("id") for choice in delivered.get("choices", [])
                           for call in choice.get("message", {}).get("tool_calls") or []]
            if len(set(identifiers)) != len(identifiers):
                # Ambiguous native tool sequence cannot use per-call rejection.
                # Reuse the store's existing format diagnostic, before any effect.
                raise ValueError("Duplicate tool call id in one response.")
        allowed = {t["function"]["name"] for t in payload.get("tools", [])}
        pending = []
        policy_calls = []
        local_rejections, coord_calls = [], []
        spawn_prompts = {}
        staging = self.role_policy is not None and policy is not None and policy.get("snapshot") is not None
        for choice in delivered.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls") or []:
                if purpose != "coding":
                    raise ValueError("Auxiliary model requests cannot execute coding tools")
                function = call["function"]
                name = function["name"]
                if name not in allowed:
                    raise ValueError(f"Disallowed {role} tool: {name}")
                # Checked before any spawn or execution; never last-value-wins.
                args = strict_tool_arguments(name, function["arguments"])
                if TOOL_ERROR_FIELD in args:
                    raise ValueError(f"Omit reserved adapter field {TOOL_ERROR_FIELD} in corrected tool calls")
                if name == "agent":
                    worker = validate_worker_name(args.get("name"))
                    if args.get("subagent_type", "general-purpose") != "general-purpose":
                        raise ValueError("Only native general-purpose teammates are supported")
                    if args.get("model") is not None or args.get("read_only") or args.get("isolation") is not None:
                        raise ValueError("Teammates must inherit the same model and use assigned writable worktrees")
                    if args.get("run_in_background") is False:
                        raise ValueError("Native named teammates must run concurrently")
                    expected = str(PurePosixPath(self.registry.workspace.worker_root) / worker)
                    if args.get("working_dir") not in (None, expected):
                        raise ValueError(
                            f"Unexpected teammate working directory. Omit working_dir when spawning {worker}; "
                            f"its startup directory is assigned automatically as {expected}. "
                            f"The delivery repository {self.registry.workspace.repo_path} is a different location. "
                            f"After starting, the worker can use git -C {json.dumps(self.registry.workspace.repo_path)} "
                            "to integrate there. Do not set the delivery repository as the spawn working_dir.")
                    if not isinstance(args.get("prompt"), str) or not args["prompt"].strip():
                        raise ValueError("A teammate assignment prompt is required")
                    pending.append((call, function, args, worker))
                    if staging:
                        try:
                            spawn_call = self._policy_spawn_call(call, worker, args["prompt"], snapshot=policy["snapshot"])
                            policy_calls.append(spawn_call)
                            if self.agent_rules:
                                spawn_prompts[call.get("id")] = spawn_call.policy.prompt
                        except package_store.PackageRejection as exc:
                            local_rejections.append(exc)
                elif staging and name == "send_message":
                    if self.collaboration is not None:
                        coord = self.collaboration.parse_coord(role, call.get("id"), args)
                        if isinstance(coord, CoordRejection):
                            local_rejections.append(coord)
                            continue
                        if coord is not None:
                            coord_calls.append(coord)
                            continue
                    try:
                        control_call = (agent_rules.parse_control(call.get("id"), args.get("to"), args.get("message"), actor=role)
                                        if self.agent_rules else
                                        package_store.parse_send_message(
                            call.get("id"), args.get("to"), args.get("message"),
                            mode=self.role_policy_mode, actor=role, role_id=policy["snapshot"].role_id)
                            if self.policy_package is not None else
                            parse_send_message(call.get("id"), args.get("to"), args.get("message")))
                    except (package_store.PackageRejection, agent_rules.ControlRejection) as exc:
                        local_rejections.append(exc)
                        continue
                    if control_call is not None:
                        policy_calls.append(control_call)
        if staging and self.agent_rules and role == "master" and pending:
            gate = self._lesson_gate()
            if gate is not None and gate["hold"]:
                order = list(dict.fromkeys(call.get("id") for call, _, _, _ in pending))
                # The lessons go into the first held result only (M3 review 3).
                local_rejections.extend(agent_rules.ControlRejection(
                    "lesson_gate", gate["text"] if index == 0 else agent_rules_runtime.lesson_gate_same_text(),
                    tool_call_id=identifier) for index, identifier in enumerate(order))
                policy_calls = [c for c in policy_calls
                                if not (isinstance(c, agent_rules.SpawnCall) and c.tool_call_id in set(order))]
                gate["calls"] = order
            if gate is not None:
                policy["lesson_gate"] = gate
        # Whole-response policy validation before any spawn side effect; the commit
        # happens at the end of handle() (plan section 3.2).
        # Only agent_rules_v1 stores take the reflection prompt a lesson answers.
        extra = ({"lesson_trigger": self.reflection.lesson_trigger(role) if role != "master" else "voluntary"}
                 if staging and self.reflection is not None else {})
        staged = (self.role_policy.stage(role, policy["snapshot"], policy_calls,
                                         source_request=policy.get("source_request"), **extra)
                  if staging else None)
        if self.policy_package is not None:
            local_rejections.extend(staged.rejections if staged is not None else ())
            rejected_ids = {r.tool_call_id for r in local_rejections}
            pending = [item for item in pending if item[0].get("id") not in rejected_ids]
        if pending:
            # The file is already written by native before it asks for another
            # response. Poll now instead of waiting for the background interval.
            if self.poll_worker_lifecycle is not None:
                self.poll_worker_lifecycle()
            messages = payload.get("messages") or []
            assistant_index = sum(1 for message in messages
                                  if isinstance(message, dict) and message.get("role") == "assistant")
            source_request = policy.get("source_request") if policy is not None else None
            reserved = []
            with self._lock:
                if len(self._spawn_leases) + len(pending) > self.teammate_limit:
                    raise ValueError(teammate_limit_message(self.teammate_limit))
                try:
                    for call, function, args, worker in pending:
                        self._spawn_sequence += 1
                        reserved.append(self._spawn_sequence)
                        self._spawn_leases[self._spawn_sequence] = {
                            "role": role, "worker": worker, "native_name": None,
                            "assistant_index": assistant_index, "tool_call_id": call.get("id"),
                            "fingerprint": call_fingerprint(call), "source_request": source_request}
                        args["working_dir"] = self.registry.register(worker)
                        args["subagent_type"] = "general-purpose"
                        if self.agent_rules:
                            args["prompt"] = spawn_prompts.get(call.get("id"), args["prompt"])
                        args["prompt"] += ("\n\n" + agent_rules_text.SPAWN_SUFFIX if self.agent_rules else
                                           "\n\n" + package_runtime.RP_SPAWN_SUFFIX if self.policy_package is not None else
                                           "\n\nThe run's communication policy is supplied in your system instructions "
                                           "on every model request, including after context compaction."
                                           + (" Your role policy, if one is set, is supplied there as well."
                                              if self.role_policy_mode in ROLE_MODES else ""))
                        function["arguments"] = json.dumps(args, ensure_ascii=False)
                        try:
                            self.event("worker_registered", name=worker, **self.registry.metadata()[worker],
                                       active_teammates=len(self._spawn_leases), teammate_limit=self.teammate_limit)
                        except Exception as exc:
                            raise EvidenceWriteError(str(exc)) from exc
                except BaseException:
                    for key in reserved:
                        self._spawn_leases.pop(key, None)
                    raise
        if self.policy_package is not None and policy is not None:
            policy["local_rejections"] = local_rejections
            policy["coord_calls"] = coord_calls
            policy["response"] = response
        if staged is not None and staged.calls:
            policy["staged"], policy["response"] = staged, response
        return self._reject_individual_calls(delivered, local_rejections)

    # -- agent_rules_v1 spawn gate (plan v4 section 8) -------------------------------------
    def _lesson_gate(self):
        """Hold this response's spawns while the master has unseen lessons.

        Decided before staging; recorded only when the delivered response carries the
        hold (``_settle_lesson_gate``). The spawn right after a hold always goes through.
        """
        lessons = self.role_policy.lessons()
        with self._lock:
            if self._gate_release:
                return {"hold": False}
            unseen = [lesson for lesson in lessons if lesson["id"] not in self._gate_seen]
            if not unseen:
                return None
            full = self._gate_full_next
            shown = lessons if full else unseen
        return {"hold": True, "full": full, "given": [lesson["id"] for lesson in shown],
                "text": agent_rules_runtime.lesson_gate_text(shown, full=full)}

    def _settle_lesson_gate(self, policy, delivered, control):
        gate = policy.get("lesson_gate")
        if gate is None:
            return
        calls = [(call.get("id"), call["function"]["name"],
                  strict_tool_arguments(call["function"]["name"], call["function"]["arguments"]))
                 for choice in delivered.get("choices") or []
                 for call in (choice.get("message") or {}).get("tool_calls") or []]
        errors = {identifier: args for identifier, _, args in calls}
        now = time.time()
        with self._lock:
            previous = (set(self._gate_seen), self._gate_full_next, self._gate_release, self._gate_held_at)
            if gate["hold"]:
                if not all("(lesson_gate)" in str(errors.get(i, {}).get(TOOL_ERROR_FIELD, "")) for i in gate["calls"]):
                    return  # the hold never reached the master (e.g. the whole response was rejected)
                self._gate_seen.update(gate["given"])
                self._gate_full_next, self._gate_release, self._gate_held_at = False, True, now
                counter = "lesson_gate_holds"
                event = {"event": "lesson_gate_held", "calls": gate["calls"], "full": gate["full"],
                         "lessons_given": gate["given"], "count": len(gate["given"])}
            else:
                spawned = [i for i, name, args in calls if name == "agent" and TOOL_ERROR_FIELD not in args]
                if not spawned:
                    return  # nothing was spawned: the next spawn still goes through
                self._gate_release = False
                counter = "lesson_gate_releases"
                event = {"event": "lesson_gate_released", "calls": spawned,
                         "delay_s": round(now - self._gate_held_at, 3) if self._gate_held_at else None}
            self._policy_counts[counter] += 1
        control.lesson_gate_undo = (previous, counter)
        name = event.pop("event")
        self.event(name, source_request=policy.get("source_request"), **event)

    def _undo_lesson_gate(self, control):
        """The response carrying a hold or a release never reached the native CLI."""
        undo = getattr(control, "lesson_gate_undo", None)
        if undo is None:
            return
        control.lesson_gate_undo = None
        previous, counter = undo
        with self._lock:
            self._gate_seen, self._gate_full_next, self._gate_release, self._gate_held_at = previous
            self._policy_counts[counter] -= 1
        self.event("lesson_gate_undone", counter=counter)

    def _policy_spawn_call(self, call, worker, prompt, *, snapshot=None):
        if self.agent_rules:
            if not isinstance(call.get("id"), str) or not call["id"]:
                raise ValueError("A tool call id is required")
            return agent_rules.SpawnCall(call["id"], worker, agent_rules.parse_spawn_policy(prompt))
        if self.policy_package is not None:
            return package_store.parse_spawn(call.get("id"), worker, prompt, role_ids=[r.role_id for r in snapshot.roles])
        if self.role_policy_mode in ROLE_MODES:
            return parse_spawn(call.get("id"), worker, prompt)
        if not isinstance(call.get("id"), str) or not call["id"]:
            raise ValueError("A tool call id is required")
        return SpawnCall(call["id"], worker, None, None)  # global mode: binding only, markers are text

    def _apply_policy_receipts(self, role, messages):
        """Show each committed control's result beside its native tool result.

        Provider copy only, and identical on every later request, so the cached
        prefix is unchanged. Matched like spawn results: the answered assistant
        message's position in this role's history plus the call's arguments.
        """
        with self._lock:
            tracked = {(entry["assistant_index"], key[2]): entry
                       for key, entry in self._policy_receipts.items() if key[0] == role}
        if not tracked or not isinstance(messages, list):
            return
        ordinal, calls = -1, {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                ordinal += 1
                calls = {call.get("id"): call_fingerprint(call) for call in message.get("tool_calls") or []
                         if isinstance(call, dict)}
                continue
            entry = tracked.get((ordinal, message.get("tool_call_id"))) if message.get("role") == "tool" else None
            if entry is None or entry["fingerprint"] is None or calls.get(message["tool_call_id"]) != entry["fingerprint"]:
                continue
            content, addition = message.get("content"), "\n\n" + entry["text"]
            if isinstance(content, str):
                message["content"] = content + addition
            elif isinstance(content, list):
                message["content"] = content + [{"type": "text", "text": addition}]

    # -- agent_rules_v1 rule engine ---------------------------------------------------
    def _rule_event(self, event, **fields):
        self.event(event, **fields)

    def _queue_reminders(self, reminders):
        for worker, obligation_id, content in reminders:
            self.collaboration.queue_notice(worker, "rule_reminder", content, key=obligation_id)

    def _rule_notice(self, actor, kind, key):
        """Collaboration lock held: a reminder whose obligation is already settled is not shown."""
        if kind == "rule_reminder" and self.rules is not None:
            return self.rules.notice_injected(actor, key)
        if kind in REFLECTION_KINDS.values() and self.reflection is not None:
            return self.reflection.injected(actor, key)
        return True

    def _rule_git(self, actor, value, source):
        """Git sample rows (collaboration lock held): new commits trigger commit rules."""
        if self.rules is None:
            return
        incomplete = tuple(value.get("sampling_incomplete") or ())
        sample = source.get("native_sample") if isinstance(source, dict) else None
        # git_emit_failed is carried over from an earlier emission; this row itself was read (R4-2).
        unreadable = set(incomplete) & {"git_read_failed", "unreadable_commit_range"}
        readable = isinstance(sample, (int, float)) and not isinstance(sample, bool) and not unreadable
        if actor == "master":
            # The delivery repository has no rules, but the evidence record needs its last
            # readable sample to tell "not sampled" from "never appeared" (M5 review R2-4).
            if readable:
                self._delivery_sampled_at = sample / 1000
            return
        if readable:
            # Only a readable sample bounds when later commits can have appeared (M2 review N2).
            previous, current = self._git_samples.get(actor, (None, None))
            if current != sample / 1000:
                self._git_samples[actor] = (current, sample / 1000)
            self.rules.note_sample(actor, at=sample / 1000)
        if incomplete and self._git_coverage.get(actor) != incomplete:
            self._git_coverage[actor] = incomplete
            self.rules.coverage(actor, "git_sampling_incomplete", reasons=list(incomplete), source=source)
        if value.get("new_commit") and isinstance(value.get("commit"), str):
            previous = self._git_samples.get(actor, (None, None))[0]
            self._queue_reminders(self.rules.observe_commit(
                actor, value["commit"], value.get("changed_paths") or [], committed_at=value.get("committed_at"),
                not_before=previous, source=source))

    def _rule_sample(self, kind, actor, source):
        """Native sample times (collaboration lock held): when each source was last read."""
        sample = source.get("native_sample") if isinstance(source, dict) else None
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            return
        at = sample / 1000
        if kind == "task" and self._task_samples[1] != at:
            self._task_samples = (self._task_samples[1], at)

    def _rule_task(self, task_id, previous, canonical, source):
        """Task-state changes: a new owner claims it, completion by its owner is task_done."""
        owner = canonical.get("owner")
        now = time.time()
        # Only task_create/task_update results are transitions; a list is a snapshot.
        from_tool = isinstance(source, dict) and source.get("tool") in {"task_create", "task_update"}
        if self.rules is not None and from_tool and previous is not None and isinstance(previous.get("owner"), str):
            # A reassignment away, or a reopened completion, lets the same transition fire again (review 10-18 F4).
            if party(previous["owner"]) != party(owner):
                self.rules.release_task(previous["owner"], task_id, "task_claimed")
            if previous.get("status") == "completed" and canonical.get("status") != "completed":
                self.rules.release_task(previous["owner"], task_id, "task_done")
        if self.rules is None or not isinstance(owner, str) or owner.lower() in {"leader", "master"}:
            return
        if from_tool:
            # The task call ran when its response was delivered, before or with any report in it.
            occurred = self._response_times.get(source.get("source_request"), now)
        else:
            # Sometime after the previous task-board snapshot, which still showed the old state.
            occurred = self._task_samples[0] if self._task_samples[0] is not None else now
        reminders = []
        if previous is None:
            if not from_tool:
                # First seen in a periodic sample: the transition time is unknown.
                self.rules.coverage(owner, "task_first_seen_in_sample", task_id=task_id,
                                    status=canonical.get("status"))
                return
            reminders += self.rules.observe_task(owner, task_id, "task_claimed", occurred_at=occurred, source=source, from_tool=from_tool)
            if canonical.get("status") == "completed":
                reminders += self.rules.observe_task(owner, task_id, "task_done", occurred_at=occurred, source=source, from_tool=from_tool)
        else:
            if owner != previous.get("owner"):
                reminders += self.rules.observe_task(owner, task_id, "task_claimed", occurred_at=occurred, source=source, from_tool=from_tool)
            if canonical.get("status") == "completed" and previous.get("status") != "completed":
                reminders += self.rules.observe_task(owner, task_id, "task_done", occurred_at=occurred, source=source, from_tool=from_tool)
        self._queue_reminders(reminders)
        if canonical.get("status") == "completed" and (previous is None and from_tool or
                                                       previous is not None and previous.get("status") != "completed"):
            self._queue_reflection(owner, "task_done", {"task_id": str(task_id)})

    def record_lifecycle_event(self, record, source):
        """Native observations from compat lifecycle lines (stderr tail thread)."""
        event = record.get("event")
        if event == "team_message_queued" and self.agent_rules:
            sender, recipient = record.get("from"), record.get("to")
            if (isinstance(sender, str) and isinstance(recipient, str)
                    and self.registry.contains(sender) and self.registry.contains(recipient)):
                with self._lock:
                    self._message_pairs[(sender, recipient)] += 1
            return
        if self.rules is None:
            return
        at = record.get("at") / 1000 if isinstance(record.get("at"), (int, float)) else None
        if event in {"team_message_sent", "team_broadcast"} and isinstance(record.get("hash"), str):
            if event == "team_message_sent" and isinstance(record.get("from"), str):
                self.rules.resolve_native(record["from"], record.get("to"), record["hash"],
                                          "sent" if record.get("ok") else "send_failed")
            elif event == "team_broadcast" and isinstance(record.get("from"), str):
                outcomes = [r for r in record.get("recipients") or [] if isinstance(r, dict) and isinstance(r.get("name"), str)]
                if outcomes and all(r.get("outcome") != "unknown" for r in outcomes):
                    accepted = [r["name"] for r in outcomes if r.get("outcome") == "accepted"]
                    outcome = "sent" if len(accepted) == len(outcomes) else "send_partial" if accepted else "send_failed"
                    self.rules.resolve_native(record["from"], "*", record["hash"], outcome, recipients=accepted)
            # Native outcomes for the offline join with message_facts; no message text (M3 review Q2).
            # Written after the rule engine has the result, so a failed write never loses it (round 2 N2).
            sender = record.get("from")
            # compat's broadcast record has no "to"; "*" pairs it with its message_facts (M5 review R2-1).
            self.event("message_outcome", native_event=event, hash=record["hash"],
                       to="*" if event == "team_broadcast" else record.get("to"),
                       sender="master" if isinstance(sender, str) and sender.lower() in {"leader", "master"} else sender,
                       ok=record.get("ok"), recipients=[{"name": r.get("name"), "outcome": r.get("outcome")}
                                                        for r in record.get("recipients") or [] if isinstance(r, dict)]
                       if event == "team_broadcast" else None, source=source)
            return
        worker = record.get("worker")
        if not isinstance(worker, str):
            return
        if event == "worker_terminal":
            self._terminal_at.setdefault(worker, at if at is not None else time.time())
            return
        reminders = []
        if event == "team_message_observed":
            for item in record.get("items") or []:
                sender = item.get("from") if isinstance(item, dict) else None
                if not isinstance(sender, str) or item.get("control") or ":" in sender:
                    continue  # controls, runtime notices and unknown senders never trigger rules
                sender = agent_rules.LEADER if sender.lower() in {"leader", "master"} else sender
                reminders += self.rules.observe_message(worker, sender, occurred_at=at, source=source)
        elif event == "shutdown_notice_observed":
            with self._lock:
                self._shutdown_requests.setdefault(worker, at if at is not None else time.time())
            reminders += self.rules.observe_shutdown(worker, occurred_at=at, source=source)
        self._queue_reminders(reminders)
        if event == "shutdown_notice_observed":
            # One prompt per observed request; a repeat within ten minutes is cooled down (M3 review 6).
            self._queue_reflection(worker, "shutdown", {"observed_at": at if at is not None else time.time()})

    def _record_rule_activity(self, role, policy, delivered):
        """The response's delivery time, and a teammate's send candidates (one shared timestamp)."""
        position, source_request = policy.get("assistant_index"), policy.get("source_request")
        at = time.time()
        with self._lock:
            self._response_times[source_request] = at
        if role == "master":
            return
        sent = policy_call_arguments(policy.get("response") or delivered)
        for choice in delivered.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls") or []:
                name = call["function"]["name"]
                args = strict_tool_arguments(name, call["function"]["arguments"])
                if TOOL_ERROR_FIELD in args:
                    continue
                if name != "send_message" or position is None or not isinstance(args.get("message"), str):
                    continue
                try:
                    control = agent_rules.parse_control(call.get("id"), args.get("to"), args["message"], actor=role)
                except agent_rules.ControlRejection:
                    control = True
                if isinstance(control, agent_rules.BoardCall) and call.get("id") in policy.get("accepted_controls", ()):
                    # On the board at once; the message itself still reaches its recipient.
                    self.rules.record_board(role, args["message"], at=at)
                elif control is not None:
                    continue  # policy controls and lessons are not reports
                key = (position, call.get("id"))
                with self._lock:
                    self._rule_sends[(role, *key)] = {"fingerprint": sent.get(call.get("id")),
                                                      "source_request": source_request}
                self.rules.record_send(role, key, to=args.get("to"), message=args["message"], at=at)

    def _record_message_facts(self, role, policy, delivered):
        """Who sent which commits and paths to whom (plan v4 section 11), for every agent.

        Delivered calls only; the native outcome is joined offline by sender and content
        hash. No message text is kept: hash, recipient and extracted tokens only.
        """
        known = frozenset(self.registry.metadata())
        accepted = policy.get("accepted_controls", ())
        for choice in delivered.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls") or []:
                if call["function"]["name"] != "send_message":
                    continue
                args = strict_tool_arguments("send_message", call["function"]["arguments"])
                message = args.get("message")
                if TOOL_ERROR_FIELD in args or not isinstance(message, str):
                    continue
                try:
                    control = agent_rules.parse_control(call.get("id"), args.get("to"), message, actor=role)
                except agent_rules.ControlRejection:
                    continue
                board = isinstance(control, agent_rules.BoardCall)
                if control is not None and not (board and call.get("id") in accepted):
                    continue  # policy controls and lessons carry no task information
                facts = agent_rules.related_facts(message, known)
                self.event("message_facts", sender=role, to=args.get("to"), board=board,
                           source_request=policy.get("source_request"), tool_call_id=call.get("id"),
                           hash=message_hash(message), commits=facts["commits"], paths=facts["paths"])

    def _record_board_reads(self, role, policy, delivered):
        """Approximate: a delivered call whose arguments name the board file (plan v4 8.1).

        A shell command or read_file that mentions it is counted; whether the agent
        actually read the whole file, or read it some other way, is not known.
        """
        name = PurePosixPath(self.policy_files.get("board", agent_rules_runtime.BOARD_FILE)).name
        for choice in delivered.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                if call["function"]["name"] not in {"read_file", "read_many_files", "run_shell_command", "grep_search"}:
                    continue  # naming or editing the file is not reading it (M3 review 12)
                if name in arguments and TOOL_ERROR_FIELD not in arguments:
                    self.event("board_read_attempt", role=role, tool=call["function"]["name"],
                               tool_call_id=call.get("id"), source_request=policy.get("source_request"),
                               approximate=True)

    @staticmethod
    def _reflection_outputs(role, policy):
        """What this delivered response committed, for the reflection funnel."""
        outputs = {"own_policy_changes": [], "proposals_for_others": [], "lessons": [], "board_posts": []}
        for change in policy.get("committed_changes") or ():
            if change["event"] == "agent_policy_proposed" and change["proposal"]["proposer"] == role:
                proposal = change["proposal"]
                outputs["own_policy_changes" if proposal["target"] == "self" else "proposals_for_others"].append(
                    proposal["proposal_id"])
            elif change["event"] == "agent_lesson_recorded":
                outputs["lessons"].append({"id": change["lesson"]["id"], "text": change["lesson"]["text"]})
            elif change["event"] == "agent_board_posted":
                outputs["board_posts"].append(change["post"]["id"])
        return outputs

    def _queue_reflection(self, worker, kind, detail=None):
        """A worker's reflection prompt (plan v4 section 9), anchored at its next request."""
        if self.reflection is None or not self.registry.contains(worker):
            return
        queued = self.reflection.queue(worker, kind, detail)
        if queued is not None:
            prompt_id, content = queued
            self.collaboration.queue_notice(worker, REFLECTION_KINDS[kind], content, key=prompt_id)

    def _observe_rule_sends(self, role, messages):
        """Resolve recorded sends from their native results in this teammate's history."""
        if self.rules is None or role == "master" or not isinstance(messages, list):
            return
        with self._lock:
            tracked = {(key[1], key[2]): entry["fingerprint"] for key, entry in self._rule_sends.items() if key[0] == role}
        if not tracked:
            return
        ordinal, calls = -1, {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                ordinal += 1
                calls = {call.get("id"): call_fingerprint(call) for call in message.get("tool_calls") or []
                         if isinstance(call, dict)}
                continue
            key = (ordinal, message.get("tool_call_id"))
            if message.get("role") != "tool" or key not in tracked:
                continue
            fingerprint = tracked.pop(key)
            if fingerprint is None or calls.get(key[1]) != fingerprint:
                continue
            with self._lock:
                self._rule_sends.pop((role, *key), None)
            self.rules.resolve_send(role, key, notification_result(tool_result_text(message.get("content"))))

    def _drop_rule_sends(self, role, source_request):
        """Sends of a response that never reached the native CLI were never executed."""
        with self._lock:
            dropped = [key for key, entry in self._rule_sends.items()
                       if key[0] == role and entry["source_request"] == source_request]
            for key in dropped:
                self._rule_sends.pop(key, None)
        for key in dropped:
            self.rules.drop_send(role, key[1:])

    def finish_rule_engine(self, path):
        """Run-end settlement: every obligation is classified and the record saved."""
        if self.rules is None:
            return None
        self.rules.finish(at=min(time.time(), self._run_end_epoch))
        stats = self.rules.stats()
        reflection = self.reflection.finish()
        with self.rules._lock:
            obligations = copy.deepcopy(list(self.rules.obligations.values()))
        with self._lock:
            gate = {key: self._policy_counts.get(key, 0) for key in ("lesson_gate_holds", "lesson_gate_releases")}
        self.save(path, {"stats": stats, "obligations": obligations, "reflection": reflection, "lesson_gate": gate,
                         "delivery_last_sample_at": self._delivery_sampled_at,
                         "windows": {"coalesce_seconds": 600, "window_seconds": 900,
                                     "window_responses": 4, "window_responses_meaning":
                                     "the reminder request's own response plus three later delivered responses"}})
        return stats["totals"]

    def _spawn_voided(self, role, entry):
        """A spawn that never started keeps no receipt and no pending reminder."""
        if not self.agent_rules:
            return
        with self._lock:
            self._policy_receipts.pop((role, entry["source_request"], entry["tool_call_id"]), None)
        self.collaboration.cancel_notices("master", "policy_missing", entry["instance"])
        if self.rules is not None:
            self.rules.worker_ended(entry["instance"])
            self.reflection.worker_ended(entry["instance"])
        self._publish_policy_files()

    def _publish_policy_files(self):
        """Ask the writer thread to rewrite the master's read-only files.

        Never called while holding the relay or store lock. The store, not the
        files, is authoritative: every request rewrites both files from the
        latest state, so a teammate's local edit does not persist either.
        """
        if not self.agent_rules or self.role_policy is None or not self.policy_files:
            return
        with self._files_changed:
            if self._files_stop:
                return
            self._files_requested += 1
            if self._files_thread is None:
                self._files_thread = threading.Thread(target=self._files_loop, name="agent-policy-files", daemon=True)
                self._files_thread.start()
            self._files_changed.notify_all()

    def _files_loop(self):
        while True:
            with self._files_changed:
                while self._files_written >= self._files_requested and not self._files_stop:
                    self._files_changed.wait()
                if self._files_written >= self._files_requested:
                    return  # stopped and nothing left to write
                target = self._files_requested
            try:
                self._write_policy_files()
            except Exception as exc:
                # Reading or rendering failed: record it and keep serving later requests.
                try:
                    self.event("agent_policy_file_write_failed", file="all", error=self.redact(str(exc))[:200])
                except Exception:
                    pass
            finally:
                with self._files_changed:
                    self._files_written = max(self._files_written, target)
                    self._files_changed.notify_all()

    def _write_policy_files(self):
        with self._files_lock:
            state = self.role_policy.export()
            contents = {"policies": agent_rules_runtime.render_policies_file(state),
                        "lessons": agent_rules_runtime.render_lessons_file(state),
                        "board": agent_rules_runtime.render_board_file(state)}
            contents = {key: value for key, value in contents.items() if key in self.policy_files}
            workspace = self.registry.workspace
            write = getattr(workspace, "write_file_atomic", None) or workspace.write_file
            for key, content in contents.items():
                try:
                    write(self.policy_files[key], content)
                except Exception as exc:
                    try:
                        self.event("agent_policy_file_write_failed", file=key, error=self.redact(str(exc))[:200])
                    except Exception:
                        pass

    def finish_policy_files(self, timeout_seconds: float = 30) -> bool:
        """Final rewrite after the run's policy state has closed, then stop the writer."""
        self._publish_policy_files()
        written = self.flush_policy_files(timeout_seconds)
        self.stop_policy_files()
        return written

    def stop_policy_files(self):
        """Let the writer thread exit once pending writes are done; idempotent."""
        with self._files_changed:
            self._files_stop = True
            self._files_changed.notify_all()

    def flush_policy_files(self, timeout_seconds: float = 30) -> bool:
        """Wait until every requested rewrite so far has been attempted."""
        deadline = time.monotonic() + timeout_seconds
        with self._files_changed:
            target = self._files_requested
            while self._files_written < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._files_changed.wait(remaining)
        return True

    def _reject_individual_calls(self, response, rejections):
        if not rejections:
            return response
        delivered = copy.deepcopy(response)
        names = {call.get("id"): call["function"]["name"] for choice in response.get("choices", [])
                 for call in choice.get("message", {}).get("tool_calls") or []}
        errors = {r.tool_call_id: str(r) if isinstance(r, CoordRejection) else
                  agent_rules_runtime.rejection_text(r, binding=names.get(r.tool_call_id) == "agent")
                  if self.agent_rules else
                  package_runtime.rejection_text(r, binding=names.get(r.tool_call_id) == "agent")
                  for r in rejections}
        for choice in delivered.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls") or []:
                if call.get("id") in errors:
                    args = strict_tool_arguments(call["function"]["name"], call["function"]["arguments"])
                    args[TOOL_ERROR_FIELD] = errors[call["id"]]
                    call["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
        return delivered

    def rejected_tool_response(self, response, error):
        """Deliver rejected calls to native tool validation, never to execution.

        Qwen's HTTP error path does not retain the call or error in model history.
        The startup adapter turns this marker into an ordinary native validation
        failure before constructing any invocation. Keep call IDs/names; the
        model's next request shows its own arguments (restore_model_history) and
        the failed tool result carries this feedback.
        Preserve the existing all-or-none boundary for each provider response.
        """
        delivered = copy.deepcopy(response)
        feedback = (f"Tool response rejected before execution: {error}\n"
                    "No tools from this response were executed. Correct the rejected call and "
                    "resubmit any still-needed calls.")
        for choice in delivered["choices"]:
            for call in choice["message"].get("tool_calls") or []:
                function = call["function"]
                try:
                    args, repeated = parse_tool_arguments(function["arguments"])
                except ValueError:
                    args, repeated = None, {}
                if repeated:
                    # Never collapse repeated keys into one value, even for a call
                    # that will not run; the provider response file keeps the text.
                    args = {}
                elif not isinstance(args, dict):
                    args = {"__agent_company_original_arguments": function["arguments"]}
                args[TOOL_ERROR_FIELD] = feedback
                function["arguments"] = json.dumps(args, ensure_ascii=False)
        return delivered

    def handle(self, body, *, control=None, owner=None, purpose=None):
        role, purpose = self.caller(body, owner, purpose)
        history: dict = {}
        policy: dict = {}
        role, payload = self.payload(body, owner=role, purpose=purpose, history=history, policy=policy)
        with self._lock:
            self._history_totals.update(history)
            if self._closed or time.monotonic() >= self.deadline:
                raise RuntimeError("Task runtime has ended")
            # A native session has one outstanding turn. A timed-out connection
            # must not launch another paid request while its predecessor runs.
            # Other workers remain independent; this is not a team-wide barrier.
            if role in self._active_requests:
                raise ValueError(f"An upstream request for {role} is still in flight")
            if control is None:
                control = RequestControl(min(self.provider.request_timeout_seconds,
                                             self.deadline - time.monotonic()))
            control.check()
            # Reserve one logical request atomically across all roles/purposes.
            # Sequential HTTP retries below reuse this reservation and payload.
            if self.max_model_calls is not None and self._logical_model_calls >= self.max_model_calls:
                self._model_call_budget_rejections[role] += 1
                try:
                    if "snapshot" in policy:
                        self._policy_injected(role, purpose, policy, admitted=False)
                    self.event("model_call_limit_rejected", role=role, purpose=purpose,
                               logical_model_calls=self._logical_model_calls,
                               max_model_calls=self.max_model_calls, upstream_started=False)
                    self.save(self.output / "usage.json", self.summary())
                except Exception as exc:
                    return self._budget_evidence_failure(exc, role, purpose)
                return 424, {"error": {"message": MODEL_CALL_LIMIT_MESSAGE,
                                       "type": "model_call_limit_exceeded",
                                       "code": "model_call_limit_exceeded"}}
            self._logical_model_calls += 1
            self._logical_calls_by_role[role] += 1
            self._logical_calls_by_purpose[purpose] += 1
            # Run-wide logical request number shared by retries, snapshot and policy batch.
            policy["source_request"] = self._logical_model_calls
            if self.max_model_calls is not None:
                remaining = self.max_model_calls - self._logical_model_calls
                if remaining <= 20:
                    append_system_text(payload["messages"],
                        f"Shared logical model-call budget: this is admitted call "
                        f"{self._logical_model_calls} of {self.max_model_calls}; "
                        f"{remaining} further calls remain across all agents and summaries. "
                        "Calls already admitted to other agents are included in the count. "
                        "Transport retries do not consume additional logical calls. "
                        "Finish and integrate the best available implementation promptly.")
                try:
                    self.event("logical_model_call_admitted", role=role, purpose=purpose,
                               logical_model_call=self._logical_model_calls,
                               max_model_calls=self.max_model_calls, remaining_model_calls=remaining)
                except Exception as exc:
                    return self._budget_evidence_failure(exc, role, purpose)
            self._active_requests[role] = control
        status, delivered, record = 424, {}, None
        try:
            if self.collaboration is not None and purpose == "coding":
                if self.poll_collaboration is not None:
                    self.poll_collaboration()
                if self.rules is not None and role != "master":
                    # start / wall-time triggers and window checks; reminders join this request.
                    self._queue_reminders(self.rules.coding_request(role))
                tail_text = ((self._agent_rules_tail(role, policy["snapshot"], policy)
                             if self.agent_rules else package_runtime.render_tail(policy["snapshot"]))
                             if policy.get("snapshot") is not None else None)
                self.collaboration.decorate(role, payload["messages"], self.deadline - time.monotonic(),
                                            policy.get("source_request"), state_tail=tail_text)
            if "snapshot" in policy:
                # Inside the cleanup scope: a failed evidence write must not leave the role in flight.
                try:
                    self._policy_injected(role, purpose, policy, admitted=True)
                except Exception as exc:
                    status, delivered = self._policy_evidence_failure(exc, upstream_started=False)
                    return status, delivered
            for retry_index in range(self.provider.max_attempts):
                control.check()
                status, delivered, record = self._attempt(
                    body, payload, role, purpose, control, retry_index, history, policy,
                    source_request=policy.get("source_request"))
                if not self._retryable(record) or retry_index + 1 >= self.provider.max_attempts:
                    break
                delay = self.provider.retry_backoff_seconds * (2 ** (retry_index + 1) - 1)
                retry_after = record.get("provider_retry_after_seconds")
                if retry_after is not None:
                    delay = max(delay, retry_after)
                # Never renew the original call deadline or detach a previous
                # transport before issuing another sequential request.
                if time.monotonic() + delay >= min(control.deadline, self.deadline):
                    break
                try:
                    self.event("provider_retry_scheduled", role=role, purpose=purpose,
                               after_attempt=record["attempt"], retry_index=retry_index + 1,
                               delay_seconds=delay, source_request=policy.get("source_request"))
                except Exception as exc:
                    # A retry cannot proceed when its evidence cannot be kept.
                    record.update(error=self.redact(str(exc)), error_kind="evidence_write_failed",
                                  relay_http_status=424)
                    status, delivered = 424, {"error": {"message": record["error"],
                                                       "code": "evidence_write_failed"}}
                    break
                until = time.monotonic() + delay
                while time.monotonic() < until:
                    control.check()
                    time.sleep(min(0.05, max(0, until - time.monotonic())))
        except PolicyPersistError as exc:
            status, delivered = self._policy_evidence_failure(exc, upstream_started=False)
        except (RequestCancelled, RequestDeadlineExceeded) as exc:
            # Cancellation between attempts is not an additional paid attempt.
            kind = "request_cancelled" if isinstance(exc, RequestCancelled) else "provider_timeout"
            status, delivered = 424, {"error": {"message": self.redact(str(exc)), "type": kind, "code": kind}}
            try:
                self.event("native_request_ended", role=role, purpose=purpose, error_kind=kind,
                           error=self.redact(str(exc)), source_request=policy.get("source_request"))
            except Exception as write_exc:
                status, delivered = self._policy_evidence_failure(write_exc, upstream_started=record is not None)
        finally:
            with self._changed:
                self._active_requests.pop(role, None)
                self._changed.notify_all()
                try:
                    self.save(self.output / "usage.json", self.summary())
                except Exception as exc:
                    if record is not None:
                        record.update(error=self.redact(str(exc)), error_kind="evidence_write_failed",
                                      relay_http_status=424)
                    status, delivered = 424, {"error": {"message": self.redact(str(exc)),
                                                       "code": "evidence_write_failed"}}
        if status == 200 and self.collaboration is not None and purpose == "coding":
            try:
                control.check()
                valid = {call.get("id") for choice in delivered.get("choices", [])
                         for call in choice.get("message", {}).get("tool_calls") or []
                         if TOOL_ERROR_FIELD not in strict_tool_arguments(call["function"]["name"], call["function"]["arguments"])}
                self.collaboration.commit_calls(role, [c for c in policy.get("coord_calls", [])
                                                       if c.tool_call_id in valid],
                                                policy.get("response", delivered), policy.get("source_request"),
                                                policy.get("assistant_index", 0))
                self.collaboration.track_response(role, policy.get("response", delivered), policy.get("source_request"),
                                                  policy.get("assistant_index", 0))
                for rejection in policy.get("local_rejections", []):
                    # agent_rules_v1 keeps the reason (it names a refused rule line) for the evidence
                    # record; the lesson gate's hold text is recorded by the gate itself (M5).
                    detail = (getattr(rejection, "detail", None) if self.agent_rules and rejection.code != "lesson_gate"
                              else None)
                    self.event("role_package_call_rejected", role=role, source_request=policy.get("source_request"),
                               tool_call_id=rejection.tool_call_id, code=rejection.code,
                               **({"detail": detail[:600]} if isinstance(detail, str) else {}))
            except (RequestCancelled, RequestDeadlineExceeded) as exc:
                kind = "request_cancelled" if isinstance(exc, RequestCancelled) else "task_deadline_exceeded"
                status, delivered = 424, {"error": {"message": self.redact(str(exc)), "code": kind}}
                try:
                    self.event("native_request_ended", role=role, purpose=purpose, error_kind=kind,
                               source_request=policy.get("source_request"), collaboration_committed=False)
                except Exception as write_exc:
                    status, delivered = self._policy_evidence_failure(write_exc, upstream_started=True)
            except (PolicyPersistError, OSError) as exc:
                status, delivered = self._policy_evidence_failure(exc, upstream_started=True)
        if status == 200 and policy.get("staged") is not None:
            # Last veto point: collaboration/attempt evidence is already durable.
            # After this returns a committed batch, neither cancellation nor later
            # evidence failure may turn the delivered response into a 424.
            status, delivered = self._commit_role_policy(role, purpose, control, policy, delivered)
        if status == 200 and self.rules is not None and purpose == "coding":
            # Registered only after the last veto point (M2 review m3). Nothing here may turn an
            # accepted, committed response into a failure: each recorder fails on its own (M3 review M2).
            control.rule_delivery = (role, policy.get("source_request"))
            recorders = [(self._record_rule_activity, role, policy, delivered),
                         (self._record_message_facts, role, policy, delivered),
                         (self._record_board_reads, role, policy, delivered),
                         (self._commit_board_seen, role, policy, control)]
            recorders += ([(self._settle_lesson_gate, policy, delivered, control)] if role == "master" else
                          [(self.rules.coding_response, role),
                           (lambda: self.reflection.response(role, self._reflection_outputs(role, policy)),)])
            for recorder, *values in recorders:
                try:
                    recorder(*values)
                except Exception as exc:
                    self._policy_evidence_failure(exc, upstream_started=True)
                    self._save_postcommit_failure()
        if status == 200 and self.rules is not None and role == "master" and purpose == "chat-compression":
            with self._lock:
                self._gate_full_next = True  # the next hold shows every lesson again
        rejected = any(isinstance(args, dict) and TOOL_ERROR_FIELD in args
                       for choice in delivered.get("choices") or []
                       for call in (choice.get("message") or {}).get("tool_calls") or []
                       for args, _ in [parse_tool_arguments(call["function"]["arguments"])])
        if status != 200 or rejected:
            rejected_ids = None
            if status == 200 and self.policy_package is not None:
                rejected_ids = {call.get("id") for choice in delivered.get("choices", [])
                                for call in choice.get("message", {}).get("tool_calls") or []
                                if TOOL_ERROR_FIELD in strict_tool_arguments(call["function"]["name"], call["function"]["arguments"])}
            self._discard_spawn_leases(policy.get("source_request"), rejected_ids)
        return status, delivered

    def _agent_rules_tail(self, role, snapshot, policy):
        """This request's tail. Its board posts stop being new only once a response to it is delivered."""
        with self.collaboration._lock:
            roster = copy.deepcopy(self.collaboration.roster)
        unseen = 0
        if role == "master":
            lessons = self.role_policy.lessons()
        with self._lock:
            ended = frozenset(self._terminal_workers)
            seen = self._board_seen.get(role, 0)
            if role == "master":
                unseen = sum(lesson["id"] not in self._gate_seen for lesson in lessons)
        # Started teammates that the latest team sample may not show yet (M3 review 8b).
        known = tuple(w.name for w in snapshot.workers if not w.ended and (w.confirmed or w.lifecycle == "active"))
        names = self._file_names()
        text = agent_rules_runtime.render_tail(
            snapshot, remaining_seconds=self.deadline - time.monotonic(), lessons_file=names["lessons_file"],
            board_file=names["board_file"], roster=roster, ended=ended, known=known, board_seen=seen,
            lessons_unseen=unseen)
        policy["board_seen_next"] = snapshot.board_count
        policy["tail"] = {"tail_chars": len(text), "roster_at": (roster or {}).get("at"),
                          "board_shown": [p.post_id for p in snapshot.board_recent],
                          "board_new": [p.post_id for p in snapshot.board_recent if p.number > seen],
                          **({"lessons_unseen": unseen} if role == "master" else
                             {"roster_rows": len(agent_rules_runtime.roster_rows(role, roster, ended, known))})}
        return text

    def _commit_board_seen(self, role, policy, control):
        if "board_seen_next" not in policy:
            return
        with self._lock:
            control.board_seen_undo = (role, self._board_seen.get(role))
            self._board_seen[role] = policy["board_seen_next"]

    def _undo_board_seen(self, control):
        undo = getattr(control, "board_seen_undo", None)
        if undo is None:
            return
        control.board_seen_undo = None
        role, previous = undo
        with self._lock:
            if previous is None:
                self._board_seen.pop(role, None)
            else:
                self._board_seen[role] = previous

    # -- role policies ---------------------------------------------------------------

    def _policy_injected(self, role, purpose, policy, *, admitted):
        snapshot = policy["snapshot"]
        with self._lock:
            self._policy_counts["injected_requests" if admitted else "unadmitted_injections"] += 1
        if self.agent_rules:
            self.event("agent_policy_injected", role=role, purpose=purpose, admitted=admitted,
                       source_request=policy.get("source_request") if admitted else None,
                       versions={w.name: w.policy.version for w in snapshot.workers
                                 if role == "master" and not w.ended or w.name == role},
                       proposal_ids=sorted(p.proposal_id for p in snapshot.proposals if p.status == "pending"),
                       injected_chars=policy.get("injected_chars"),
                       # What the current-request tail showed (plan v4 sections 8.1, 8.2; M3 review 11).
                       **({"tail": policy["tail"]} if admitted and "tail" in policy else {}))
            return
        self.event("role_policy_injected", role=role, purpose=purpose, admitted=admitted,
                   source_request=policy.get("source_request") if admitted else None,
                   mode=snapshot.mode, global_version=snapshot.global_policy.version,
                   slot_versions=({slot.slot_id: slot.policy.version for slot in snapshot.slots}
                                  if self.policy_package is None else None),
                   role_versions=({view.role_id: view.policy.version for view in snapshot.roles}
                                  if self.policy_package is not None else None),
                   proposal_ids=sorted(p.proposal_id for p in snapshot.proposals if p.status == "pending"),
                   injected_chars=policy.get("injected_chars"))

    def _policy_rejected(self, stage, role, exc, response, source_request):
        calls = sum(len((choice.get("message") or {}).get("tool_calls") or [])
                    for choice in response.get("choices") or [] if isinstance(choice, dict))
        with self._lock:
            self._policy_counts[f"rejected_{stage}"] += 1
            self._policy_counts[f"rejected_{exc.code}"] += 1
        # Every call of the response is returned unexecuted; discarded_calls counts them all.
        self.event("role_policy_message_rejected", stage=stage, role=role, code=exc.code,
                   tool_call_id=exc.tool_call_id, source_request=source_request, discarded_calls=calls)

    def _commit_role_policy(self, role, purpose, control, policy, delivered):
        staged = policy["staged"]

        def check():
            control.check()
            if self._closed or time.monotonic() >= self.deadline:
                raise RequestDeadlineExceeded("task_deadline_exceeded: task budget exhausted")
        try:
            result = self.role_policy.commit(staged, check=check)
        except PolicyRejection as exc:
            response = policy["response"]
            error = self.redact(type(exc).__name__ + ": " + str(exc))
            rejected = json.loads(self.redact(json.dumps(self.rejected_tool_response(response, error),
                                                         ensure_ascii=False)))
            self._record_model_views(role, response, rejected)
            try:
                self._policy_rejected("late", role, exc, response, staged.source_request)
                # _qwen_response.json is what the CLI received: replace it with the
                # rejection and keep the staged (never returned) version beside it.
                delivered_path = self.output / f"attempt_{policy['attempt']:05d}_{role}_qwen_response.json"
                staged_path = delivered_path.with_name(delivered_path.stem + "_policy_staged.json")
                if delivered_path.exists():
                    staged_path.write_text(delivered_path.read_text())
                self.save(delivered_path, rejected)
            except Exception as write_exc:
                return self._policy_evidence_failure(write_exc, upstream_started=True)
            return 200, rejected
        except PolicyPersistError as exc:
            return self._policy_evidence_failure(exc, upstream_started=True)
        except (RequestCancelled, RequestDeadlineExceeded) as exc:
            kind = "request_cancelled" if isinstance(exc, RequestCancelled) else "task_deadline_exceeded"
            try:
                self.event("native_request_ended", role=role, purpose=purpose, error_kind=kind,
                           error=self.redact(str(exc)), role_policy_batch_committed=False,
                           source_request=staged.source_request)
            except Exception as write_exc:
                return self._policy_evidence_failure(write_exc, upstream_started=True)
            return 424, {"error": {"message": self.redact(str(exc)), "type": kind, "code": kind}}
        # Publishing the store state is irreversible. Later audit-file failures
        # are run-level evidence failures; accepted tool calls still reach native.
        policy["committed_batch"] = result.batch
        if self.agent_rules:
            policy["committed_changes"], policy["accepted_controls"] = result.changes, set(result.accepted_call_ids)
        if result.batch is not None:
            control.role_policy_delivery = (role, staged.source_request)
        if self.policy_package is not None and result.rejections:
            delivered = self._reject_individual_calls(delivered, result.rejections)
            known = {(r.tool_call_id, r.code, r.detail) for r in policy.get("local_rejections", [])}
            late = [r for r in result.rejections if (r.tool_call_id, r.code, r.detail) not in known]
            policy.setdefault("local_rejections", []).extend(late)
            self._record_model_views(role, policy["response"], delivered)
            try:
                self.save(self.output / f"attempt_{policy['attempt']:05d}_{role}_qwen_response.json", delivered)
                for rejection in late:
                    self.event("role_package_call_rejected", role=role, source_request=staged.source_request,
                               tool_call_id=rejection.tool_call_id, code=rejection.code)
            except Exception as exc:
                failed = self._policy_evidence_failure(exc, upstream_started=True)
                if result.batch is None:
                    return failed
                self._save_postcommit_failure()
        sent = policy_call_arguments(policy["response"])
        position = policy.get("assistant_index")
        with self._lock:
            self._policy_counts["committed_batches"] += int(result.batch is not None)
            for change in result.changes:
                self._policy_counts[change["event"]] += 1
                if change["event"] in {"role_policy_planned", "agent_policy_planned"}:
                    instance = (change["worker"]["name"] if change["event"] == "agent_policy_planned" else
                                change.get("instance", change.get("binding", {}).get("instance")))
                    self._policy_spawns[(role, staged.source_request, change["tool_call_id"])] = {
                        "instance": instance, "source_request": staged.source_request,
                        "tool_call_id": change["tool_call_id"], "assistant_index": position,
                        "fingerprint": sent.get(change["tool_call_id"])}
            for call in staged.calls:
                if (not isinstance(call, (SpawnCall, package_store.SpawnCall, agent_rules.SpawnCall)) and
                        (self.policy_package is None or call.tool_call_id in result.accepted_call_ids)):
                    self._policy_notifications[(role, staged.source_request, call.tool_call_id)] = {
                        "kind": type(call).__name__, "source_request": staged.source_request,
                        "tool_call_id": call.tool_call_id, "assistant_index": position,
                        "fingerprint": sent.get(call.tool_call_id)}
            for call_id, receipt in (getattr(result, "receipts", None) or {}).items():
                if call_id in result.accepted_call_ids:
                    self._policy_receipts[(role, staged.source_request, call_id)] = {
                        "assistant_index": position, "fingerprint": sent.get(call_id), "text": receipt}
        if self.agent_rules:
            for change in result.changes:
                if change["event"] == "agent_policy_planned" and self.rules is not None:
                    self.rules.worker_spawned(change["worker"]["name"])
                if change["event"] == "agent_policy_planned" and change["policy_missing"]:
                    self.collaboration.queue_notice("master", "policy_missing", agent_rules_text.MISSING_POLICY_REMINDER.format(
                        name=change["worker"]["name"]), key=change["worker"]["name"])
            if result.batch is not None:
                self._publish_policy_files()
        return 200, delivered

    def _save_postcommit_failure(self):
        """Best effort durable run failure; never replace an accepted response."""
        try:
            self.save(self.output / "usage.json", self.summary())
        except Exception:
            # summary() still exposes the failure if storage remains unavailable.
            pass

    def response_not_delivered(self, control):
        """HTTP delivery failed: retire unstarted bindings without undoing policies.

        Native evidence can already have confirmed a worker in a partial response;
        such workers retain their binding and slot until a real lifecycle event.
        """
        rule_delivery = getattr(control, "rule_delivery", None)
        if rule_delivery is not None and self.rules is not None:
            control.rule_delivery = None
            self._drop_rule_sends(*rule_delivery)
        if self.rules is not None:
            self._undo_lesson_gate(control)
            self._undo_board_seen(control)
        with self._lock:
            delivery = getattr(control, "role_policy_delivery", None)
            if delivery is None or self.policy_package is None:
                return
            control.role_policy_delivery = None
            role, source_request = delivery
            pending = [(key, dict(value)) for key, value in self._policy_spawns.items()
                       if key[:2] == (role, source_request)]
        voided = []
        for key, entry in pending:
            try:
                # Same order as store persistence: store -> relay. Reuse these
                # locks so native confirmation cannot land between the check,
                # persisted void and lease release. Both are reentrant here.
                with self.role_policy._lock, self._lock:
                    native_confirmed = any(
                        lease["source_request"] == source_request
                        and lease["tool_call_id"] == entry["tool_call_id"]
                        and lease.get("native_name") is not None
                        for lease in self._spawn_leases.values())
                    if native_confirmed:
                        # Success can precede tool-history/first worker request.
                        continue
                    outcome = self.role_policy.record_spawn_result(
                        entry["instance"], ok=False, tool_call_id="response_not_delivered")
                    if outcome == "voided":
                        self._policy_spawns.pop(key, None)
                        self._discard_spawn_leases(source_request, {entry["tool_call_id"]})
                        voided.append(entry["instance"])
            except Exception as exc:
                self._policy_evidence_failure(exc, upstream_started=True)
                self._save_postcommit_failure()
        with self._lock:
            for key in [key for key in self._policy_notifications if key[:2] == (role, source_request)]:
                self._policy_notifications.pop(key, None)
            for key in [key for key in self._policy_receipts if key[:2] == (role, source_request)]:
                self._policy_receipts.pop(key, None)
        if self.agent_rules:
            for instance in voided:
                self.collaboration.cancel_notices("master", "policy_missing", instance)
                if self.rules is not None:
                    self.rules.worker_ended(instance)
        self._publish_policy_files()
        try:
            self.event("role_policy_response_not_delivered", role=role, source_request=source_request)
        except Exception as exc:
            self._policy_evidence_failure(exc, upstream_started=True)
            self._save_postcommit_failure()

    def record_evidence_failure(self, exc):
        """Run-end policy or rule settlement failed: counted like any other evidence write failure."""
        # Never an empty message (e.g. OSError()): stop reasons count only non-empty errors.
        self._policy_evidence_failure(RuntimeError(f"{type(exc).__name__}: {exc}"), upstream_started=False)

    def _policy_evidence_failure(self, exc, *, upstream_started):
        message = self.redact(str(exc))
        with self._lock:
            self._policy_counts["evidence_failures"] += 1
            self._budget_evidence_errors.append({"error_kind": "evidence_write_failed", "error": message,
                                                 "role_policy": True, "upstream_started": upstream_started})
        return 424, {"error": {"message": message, "code": "evidence_write_failed",
                               "type": "evidence_write_failed"}}

    def _observe_policy_tool_results(self, role, messages):
        """Native results of committed policy messages and spawns: evidence, plus spawn failure.

        Native passes a failed tool's bare error message to the model, so success is
        recognized by native's fixed success text and anything else is a failure.
        A result counts only when it answers the assistant message at the recorded
        position of this role's history and that call has the committed arguments;
        an earlier call that reused the id is never taken for it. After a compaction
        the position no longer matches and the result stays unknown.
        """
        with self._lock:
            tracked = {(value["assistant_index"], value["tool_call_id"]): (key, value, "spawn")
                       for key, value in self._policy_spawns.items() if key[0] == role}
            tracked.update({(value["assistant_index"], value["tool_call_id"]): (key, value, "notice")
                            for key, value in self._policy_notifications.items() if key[0] == role})
        if not tracked or not isinstance(messages, list):
            return
        ordinal, calls = -1, {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                ordinal += 1
                calls = {call.get("id"): call_fingerprint(call) for call in message.get("tool_calls") or []
                         if isinstance(call, dict)}
                continue
            if message.get("role") != "tool" or (ordinal, message.get("tool_call_id")) not in tracked:
                continue
            identifier = message["tool_call_id"]
            key, entry, kind = tracked.pop((ordinal, identifier))
            if entry["fingerprint"] is None or calls.get(identifier) != entry["fingerprint"]:
                continue
            text = tool_result_text(message.get("content"))
            if kind == "spawn":
                running = text.startswith(f'Teammate "{entry["instance"]}" is now running')
                outcome = self.role_policy.record_spawn_result(entry["instance"], ok=running, tool_call_id=identifier)
                with self._lock:
                    self._policy_spawns.pop(key, None)
                    self._policy_counts[f"spawn_result_{outcome}"] += 1
                if outcome == "voided":
                    self._spawn_voided(role, entry)
                self.event("role_policy_spawn_result", instance=entry["instance"], tool_call_id=identifier,
                           source_request=entry["source_request"], outcome=outcome,
                           detail=None if running else text[:200])
            else:
                result = notification_result(text)
                with self._lock:
                    self._policy_notifications.pop(key, None)
                    self._policy_counts[f"notification_{result}"] += 1
                self.event(f"role_policy_message_{result}", role=role, tool_call_id=identifier,
                           kind=entry["kind"], source_request=entry["source_request"],
                           detail=None if result == "sent" else text[:200])

    def _budget_evidence_failure(self, error, role, purpose):
        """A local evidence failure is fatal, not a successful budget stop."""
        message = self.redact(str(error))
        self._budget_evidence_errors.append({"role": role, "purpose": purpose,
            "error_kind": "evidence_write_failed", "error": message,
            "upstream_started": False})
        return 424, {"error": {"message": message, "code": "evidence_write_failed",
                               "type": "evidence_write_failed"}}

    @staticmethod
    def _retryable(record):
        if not record.get("upstream_started"):
            return False
        if record.get("error_kind") in {"provider_transport_error", "provider_first_progress_timeout",
                                        "provider_progress_idle_timeout"}:
            # A slow/disconnected error body must not turn a known authentication/configuration
            # rejection into an automatic retry. Received HTTP status is evidence
            # even when no complete response body was returned.
            return record.get("provider_http_status") in {
                None, 200, 408, 429, 500, 502, 503, 504, 520, 522, 524, 525}
        # Deliberately not driven by words such as '503' in an error message.
        return (record.get("error_kind") == "provider_error"
                and record.get("provider_http_status") in {408, 429, 500, 502, 503, 504, 520, 522, 524, 525})

    @staticmethod
    def _provider_retry_after(response):
        """Read only the provider's structured numeric retry_after hint."""
        if not isinstance(response, dict):
            return None
        hint = response.get("retry_after")
        if hint is None and isinstance(response.get("error"), dict):
            hint = response["error"].get("retry_after")
        if isinstance(hint, bool) or not isinstance(hint, (int, float)):
            return None
        try:
            seconds = float(hint)
        except (OverflowError, ValueError):
            return None
        return seconds if math.isfinite(seconds) and seconds >= 0 else None

    def _attempt(self, body, payload, role, purpose, control, retry_index, history=None, policy=None, *,
                 source_request=None):
        with self._lock:
            control.check()
            self._attempts += 1
            identifier = self._attempts
            self._inflight[identifier] = role
            self._inflight_started[identifier] = time.time()
            self._controls[identifier] = control
        started = time.time()
        prefix = self.output / f"attempt_{identifier:05d}_{role}"
        status, response, delivered, error = None, {}, {}, None
        failure_kind = None
        usage, cost, usage_error = None, None, None
        relay_status, upstream_started, record = 200, False, {}
        awaiting_provider = False
        rejection = None
        observation: dict = {}

        def fail(exc, kind):
            nonlocal error, failure_kind, delivered, relay_status
            failure_kind = kind
            error = self.redact(type(exc).__name__ + ": " + str(exc))
            delivered = {"error": {"message": error, "type": failure_kind,
                                   "code": failure_kind, "provider_http_status": status}}
            relay_status = 424
            # The same record may already be in _records when a final log fails.
            record.update(error=error, error_kind=kind, relay_http_status=relay_status)

        def write_evidence(writer, *args, **kwargs):
            try:
                writer(*args, **kwargs)
                return True
            except Exception as exc:
                fail(exc, "evidence_write_failed")
                return False

        try:
            try:
                initial_saved = (
                    write_evidence(self.save, Path(str(prefix) + "_qwen_request.json"), body)
                    and write_evidence(self.save, Path(str(prefix) + "_provider_request.json"), payload)
                    and write_evidence(self.event, "provider_attempt_started", attempt=identifier, role=role,
                                       purpose=purpose, retry_index=retry_index, source_request=source_request,
                                       worker_system_policy_injected=purpose == "coding" and self.execution_mode == "team" and role != "master",
                                       model_history=history or {})
                )
                if initial_saved:
                    with self._lock:
                        initial_saved = write_evidence(self.save, self.output / "usage.json", self.summary())
                if initial_saved:
                    control.check()
                    upstream_started = True
                    awaiting_provider = True
                    if isinstance(self.upstream, OpenAIChatTransport):
                        status, response = self.upstream(payload, control=control, observation=observation)
                    else:
                        # Scripted test providers retain their original one-argument API.
                        status, response = self.upstream(payload)
                    awaiting_provider = False
                    if write_evidence(self.save, Path(str(prefix) + "_provider_response.json"), response):
                        control.check()  # Retain returned usage, but never spawn after cancellation.
                        if status != 200 or isinstance(response, dict) and response.get("error") is not None:
                            failure_kind = "provider_error"
                            detail = response.get("error") if isinstance(response, dict) else None
                            message = detail.get("message") if isinstance(detail, dict) else detail
                            raise RuntimeError(str(message or f"Provider HTTP failure ({status})"))
                        try:
                            validate_completion(response)
                        except ValueError:
                            failure_kind = "invalid_provider_response"
                            raise
                        try:
                            delivered = self.prepare_response(role, payload, response, purpose=purpose,
                                                              policy=policy)
                            if policy is not None and "staged" in policy:
                                policy["attempt"] = identifier
                        except (ValueError, TypeError, KeyError) as exc:
                            # Preserve the failure as native tool history. This is
                            # not a transport retry or an extra relay model call.
                            failure_kind = "tool_response_rejected"
                            error = self.redact(type(exc).__name__ + ": " + str(exc))
                            if isinstance(exc, DuplicateToolArguments):
                                rejection = {"reason": "duplicate_keys", "tool": exc.tool,
                                             "repeated_keys": {key[:64]: count for key, count in
                                                               sorted(exc.repeated.items())[:8]}}
                            elif isinstance(exc, PolicyRejection):
                                rejection = {"reason": "role_policy", "code": exc.code,
                                             "tool_call_id": exc.tool_call_id}
                                self._policy_rejected("initial", role, exc, response,
                                                      policy.get("source_request") if policy else None)
                            delivered = self.rejected_tool_response(response, error)
            except Exception as exc:
                if status is None:
                    status = observation.get("provider_http_status")
                if isinstance(exc, RequestDeadlineExceeded):
                    failure_kind = ("task_deadline_exceeded" if time.monotonic() >= self.deadline
                                    else "provider_timeout")
                elif isinstance(exc, RequestCancelled):
                    failure_kind = "request_cancelled"
                elif isinstance(exc, RequestProgressTimeout):
                    failure_kind = ("provider_first_progress_timeout" if exc.phase == "first_progress"
                                    else "provider_progress_idle_timeout")
                elif isinstance(exc, InvalidProviderResponse):
                    failure_kind = "invalid_provider_response"
                elif isinstance(exc, EvidenceWriteError):
                    failure_kind = "evidence_write_failed"
                if failure_kind is None:
                    failure_kind = ("provider_transport_error"
                                    if awaiting_provider and isinstance(exc, (TransportError, ConnectionError, OSError))
                                    else "provider_client_error" if awaiting_provider
                                    else "relay_execution_error")
                fail(exc, failure_kind)
            raw_usage = response.get("usage") if isinstance(response, dict) else None
            if raw_usage is not None:
                try:
                    usage, cost = usage_and_cost(raw_usage, self.provider.rates)
                except (KeyError, ValueError, TypeError) as exc:
                    usage_error = str(exc)
            delivered = json.loads(self.redact(json.dumps(delivered, ensure_ascii=False)))
            if relay_status == 200 and isinstance(response, dict):
                self._record_model_views(role, response, delivered)
            if rejection is not None:
                record.update(tool_rejection=rejection)
            if observation:
                # Transport produces only times/counts/enums, never partial text.
                # An incomplete attempt's usage remains unknown even if a later
                # attempt recovers; these counters are not token estimates.
                record["transport"] = observation
            # Which model snapshot answered: provider metadata only, never message content.
            # The `deepseek-flash` alias moved to a new snapshot on 2026-10-17.
            identity = {key: response.get(key) for key in ("model", "system_fingerprint")
                        if isinstance(response, dict) and isinstance(response.get(key), str)}
            if identity:
                record["provider_identity"] = {key: self.redact(value)[:200] for key, value in identity.items()}
            record.update(attempt=identifier, role=role, purpose=purpose, retry_index=retry_index,
                          source_request=source_request, started_at=started,
                          elapsed_seconds=round(time.time() - started, 3), provider_http_status=status,
                          relay_http_status=relay_status, upstream_started=upstream_started,
                          usage=usage, raw_usage=raw_usage,
                          provider_retry_after_seconds=(self._provider_retry_after(response)
                                                        if failure_kind == "provider_error" else None),
                          estimated_cost=str(cost) if cost is not None else None,
                          error=error, error_kind=failure_kind, usage_error=usage_error)
            write_evidence(self.save, Path(str(prefix) + "_qwen_response.json"), delivered)
            with self._lock:
                self._inflight.pop(identifier, None)
                self._inflight_started.pop(identifier, None)
                self._controls.pop(identifier, None)
                self._records.append(record)
                write_evidence(self.event, "provider_attempt_finished", **record)
                write_evidence(self.save, self.output / "usage.json", self.summary())
            # Late evidence failures override success, retaining status and known usage.
            return relay_status, delivered, record
        finally:
            with self._changed:
                self._inflight.pop(identifier, None)
                self._inflight_started.pop(identifier, None)
                self._controls.pop(identifier, None)

    def close(self):
        with self._lock:
            self._closed = True
            controls = list(self._active_requests.values())
        for control in controls:
            control.cancel("Relay closed")

    def drain(self, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        with self._changed:
            while self._active_requests and time.monotonic() < deadline:
                self._changed.wait(timeout=min(1, max(0, deadline - time.monotonic())))

    def team_status(self) -> dict:
        """Per-role request accounting for runtime check-ins: observations, not progress."""
        now = time.time()
        # Rule and task state first: their locks precede the relay lock (no reverse order).
        rule_rates = self.rules.rates_by_worker() if self.rules is not None else None
        overview = self.role_policy.snapshot("master") if self.agent_rules and self.rules is not None else None
        busy_owners = set()
        if self.agent_rules and self.collaboration is not None:
            with self.collaboration._lock:
                busy_owners = {t.get("owner") for t in self.collaboration.tasks.values() if t.get("status") == "in_progress"}
        with self._lock:
            roles: dict[str, dict] = {}

            def row(role):
                return roles.setdefault(role, {
                    "logical_calls": self._logical_calls_by_role.get(role, 0), "attempts": 0, "retries": 0,
                    "failed_attempts": 0, "known_tokens": 0, "unknown_usage_attempts": 0,
                    "last_finished_ago_s": None, "in_flight_s": None})
            for record in self._records:
                entry = row(record["role"])
                entry["attempts"] += 1
                entry["retries"] += int(record.get("retry_index", 0) > 0)
                entry["failed_attempts"] += int(bool(record["error"]))
                if record["usage"] is None:
                    entry["unknown_usage_attempts"] += 1
                else:
                    entry["known_tokens"] += record["usage"].get("total_tokens", 0)
                finished = record["started_at"] + record["elapsed_seconds"]
                entry["last_finished_ago_s"] = round(now - finished)
            for identifier, role in self._inflight.items():
                started = self._inflight_started.get(identifier)
                if started is not None:
                    entry = row(role)
                    entry["in_flight_s"] = max(entry["in_flight_s"] or 0, round(now - started))
            status = {"at": now, "roles": roles, "workers": self.registry.metadata(),
                      "active_teammates": len(self._spawn_leases), "teammate_limit": self.teammate_limit,
                      "note": "Known tokens are lower bounds when usage is missing; attempts are provider calls."}
            if self.agent_rules:
                live = [w for w in self.registry.metadata() if w not in self._terminal_workers]
                idle = []
                for worker in live:
                    entry = roles.get(worker)
                    quiet = (entry is not None and entry["in_flight_s"] is None
                             and entry["last_finished_ago_s"] is not None and entry["last_finished_ago_s"] >= 900)
                    if quiet and worker not in busy_owners:
                        idle.append({"worker": worker, "idle_s": entry["last_finished_ago_s"]})
                status.update(
                    remaining_s=max(0, round(self.deadline - time.monotonic())),
                    pending_shutdowns=[{"worker": w, "requested_ago_s": round(now - at)}
                                       for w, at in sorted(self._shutdown_requests.items())
                                       if w not in self._terminal_workers and now - at >= 300],
                    idle_workers=idle,
                    message_pairs=[{"from": a, "to": b, "count": n}
                                   for (a, b), n in self._message_pairs.most_common(10)],
                    rule_rates=rule_rates)
                if overview is not None:
                    status.update(
                        pending_proposals=sum(p.status == "pending" for p in overview.proposals),
                        recent_self_changes=sum(p.status == "applied" for p in overview.proposals),
                        lessons_total=overview.lesson_count, board_total=overview.board_count)
            return status

    def summary(self):
        with self._lock:
            usage = Counter()
            for record in self._records:
                usage.update(record["usage"] or {})
            known = [Decimal(r["estimated_cost"]) for r in self._records if r["estimated_cost"] is not None]
            return {"execution_mode": self.execution_mode, "policy_applied": self.execution_mode == "team",
                    "max_model_calls": self.max_model_calls,
                    "logical_model_calls": self._logical_model_calls,
                    "logical_calls_by_role": dict(self._logical_calls_by_role),
                    "logical_calls_by_purpose": dict(self._logical_calls_by_purpose),
                    "model_call_budget_rejections": sum(self._model_call_budget_rejections.values()),
                    "model_call_budget_rejections_by_role": dict(self._model_call_budget_rejections),
                    "provider_attempts": self._attempts, "completed_attempts": len(self._records),
                    "inflight_attempts": len(self._inflight), "active_requests": len(self._active_requests),
                    "usage": dict(usage),
                    "deadline_rejections": sorted(self._deadline_rejections),
                    "usage_complete": not self._active_requests and all(r["usage"] is not None for r in self._records),
                    "known_estimated_cost": str(sum(known, Decimal(0))) if self.provider.rates else None,
                    "cost_complete": self.provider.rates is not None and not self._active_requests and len(known) == len(self._records),
                    "currency": self.provider.rates.currency if self.provider.rates else None,
                    "provider_billed_cost": None, "calls_by_role": dict(Counter(r["role"] for r in self._records)),
                    # Snapshot changes within a run show up as more than one key here.
                    **({"provider_identities": dict(Counter(
                        f'{r["provider_identity"].get("model")}|{r["provider_identity"].get("system_fingerprint")}'
                        for r in self._records if r.get("provider_identity")))}
                       if any(r.get("provider_identity") for r in self._records) else {}),
                    # Occurrences across requests: a call restored in 5 later requests counts 5.
                    "model_history": {"rewritten_calls": len(self._model_views), **self._history_totals},
                    "errors": ([r for r in self._records if r["error"] or r["usage_error"]]
                               + list(self._budget_evidence_errors)),
                    "workers": self.registry.snapshot(), "worker_metadata": self.registry.metadata(),
                    **({"native_lifecycle_tail": self._worker_lifecycle_tail}
                       if self._worker_lifecycle_tail is not None else {}),
                    **({"role_policy": {"mode": self.role_policy_mode,
                                        **({"protocol": self.policy_protocol} if self.agent_rules else {}),
                                        "counts": dict(sorted(self._policy_counts.items())),
                                        "unresolved_notifications": len(self._policy_notifications),
                                        "unresolved_spawn_results": len(self._policy_spawns)}}
                       if self.role_policy is not None else {})}


def tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def notification_result(text: str) -> str:
    """Classify a native send_message result as sent, send_partial or send_failed.

    Native success texts are fixed; a failed send reaches the model as the bare
    error message (no "Failed to send message" prefix), so any other text is a failure.
    """
    if text.startswith("Message sent to ") or text.startswith("Message broadcast to all teammates"):
        return "sent"
    if text.startswith("Message broadcast delivered to "):
        return "send_partial"
    return "send_failed"


def call_fingerprint(call: dict) -> tuple | None:
    """A call's name and complete arguments, independent of JSON formatting."""
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return None
    try:
        args = json.loads(function.get("arguments") or "{}")
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    return function["name"], json.dumps(args, sort_keys=True, ensure_ascii=False)


def policy_call_arguments(response: dict) -> dict:
    """tool_call_id -> fingerprint for the provider response a policy batch came from."""
    return {call.get("id"): call_fingerprint(call)
            for choice in response.get("choices") or [] if isinstance(choice, dict)
            for call in (choice.get("message") or {}).get("tool_calls") or [] if isinstance(call, dict)}


def stream_response(response):
    validate_completion(response)
    common = {key: response[key] for key in ("id", "created", "model") if key in response}
    common["object"] = "chat.completion.chunk"
    for choice in response.get("choices", []):
        delta = dict(choice["message"])
        if delta.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
        index = choice.get("index", 0)
        yield {**common, "choices": [{"index": index, "delta": delta, "finish_reason": None}]}
        yield {**common, "choices": [{"index": index, "delta": {}, "finish_reason": choice.get("finish_reason")}]}
    if response.get("usage") is not None:
        yield {**common, "choices": [], "usage": response["usage"]}


def start_server(relay: TeamRelay, host="127.0.0.1"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            # Read-only request accounting for team check-ins; same bearer token.
            if (self.path not in {"/v1/agent-company/team-status", "/agent-company/team-status"}
                    or relay.execution_mode != "team"
                    or self.headers.get("Authorization") != "Bearer " + relay.token):
                self.send_error(403)
                return
            encoded = json.dumps(relay.team_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            body = {}
            control = None
            owner = self.headers.get("X-Agent-Company-Owner")
            purpose = self.headers.get("X-Agent-Company-Purpose")
            watch_done = threading.Event()
            client_gone = threading.Event()
            watcher = None
            if self.path not in {"/v1/chat/completions", "/chat/completions"} or self.headers.get("Authorization") != "Bearer " + relay.token:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("Invalid request body length")
                body = json.loads(self.rfile.read(length))
                control = relay.request_control(owner)

                def watch_client():
                    try:
                        with selectors.DefaultSelector() as selector:
                            selector.register(self.connection, selectors.EVENT_READ)
                            while not watch_done.wait(0.05):
                                if selector.select(0) and not self.connection.recv(1, socket.MSG_PEEK):
                                    client_gone.set()
                                    control.cancel("Client disconnected")
                                    relay.event("client_disconnected", upstream_cancel_requested=True)
                                    return
                    except (OSError, ValueError):
                        client_gone.set()
                        control.cancel("Client connection closed")

                watcher = threading.Thread(target=watch_client, daemon=True)
                watcher.start()
                status, response = relay.handle(body, control=control, owner=owner, purpose=purpose)
            except Exception as exc:
                error = relay.redact(type(exc).__name__ + ": " + str(exc))
                with relay._lock:
                    conflicting = [i for i, role in relay._inflight.items()
                                   if role == owner][:8]
                relay.event("request_rejected", error=error,
                            owner=relay.redact(str(owner)[:80]) if owner is not None else None,
                            purpose=relay.redact(str(purpose)[:80]) if purpose is not None else None,
                            identity_source="native_headers" if owner is not None else "legacy_prompt",
                            conflicting_attempts=conflicting)
                expired = isinstance(exc, RequestDeadlineExceeded)
                status, response = (408 if expired else 400), {"error": {
                    "message": error,
                    "code": "task_deadline_exceeded" if expired else "request_rejected"}}
            finally:
                watch_done.set()
                if watcher is not None:
                    watcher.join(timeout=1)
            if client_gone.is_set():
                relay.response_not_delivered(control)
                return
            try:
                if status == 200 and body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in stream_response(response):
                        self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    encoded = json.dumps(response).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("x-should-retry", "false")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                self.wfile.flush()
            except OSError:
                relay.response_not_delivered(control)
                try:
                    relay.event("client_disconnected", provider_response_preserved=True)
                except Exception as write_exc:
                    relay._policy_evidence_failure(write_exc, upstream_started=True)
                    relay._save_postcommit_failure()
    server = ThreadingHTTPServer((host, 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
