"""Relay-owned role-level communication policy state (Stage-1, S1).

Specification: docs/stage1_role_policy_harness_plan_20261009.md (v3.3), sections
3.1 (state objects), 3.2 (message conventions) and 3.6 (events). Decisions in force
(2026-10-10): D1 approved role policies override conflicting communication defaults
of C but no fixed runtime boundary; D4 succession only from a relay-confirmed
terminal instance, keeping slot and version and voiding its pending proposal; D9
H-adaptive baseline (mode ``global``); D10 option C (compat change, S2).

This module has no relay, native-CLI or model dependency. S2 wires parsing into
``prepare_response()``, the late commit into ``handle()``, lifecycle inputs into the
first-request path and the stderr reader, and renders ``PolicySnapshot`` objects
into requests. Mode ``off`` never constructs a store.

Invariants implemented here:
- The relay is the only authority. Native notification outcomes never roll back
  committed state; there is no accepted/sent state machine.
- ``stage()`` validates a whole response on a copy. ``commit()`` re-simulates the same
  calls on the latest state, persists one batch record, and only then publishes it.
  A failed simulation, cancellation check or write leaves state unchanged.
- Base versions come from the request snapshot; an explicit ``base=v<k>`` must equal
  it, and the snapshot version must still be current at commit.
- One pending proposal per slot. Identical CHANGE/REASON/EVIDENCE from the same
  instance on the same base renotifies; any difference replaces.
- A decision binds to the exact proposal id carried in the master's snapshot and is
  never redirected to a replacement.
- Slots and instances are separate. Succession keeps the slot and its current
  version; the predecessor's pending proposal is voided when it terminates.
- Replay applies the persisted change records with the same function the live path
  uses, so live and replayed state cannot diverge silently.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
from typing import Callable, Iterable

from agent_company.runtime.team import validate_worker_name

MODES = ("global", "master", "propose")
ROLE_MODES = ("master", "propose")
MASTER = "master"
LEADER = "leader"
BROADCAST = "*"

ROLE_POLICY_MAX_CHARS = 2000
PROPOSAL_MAX_CHARS = 2000      # CHANGE/REASON/EVIDENCE including their labels
NOTE_MAX_CHARS = 2000
# Not fixed by the plan. The configured starting C (P1 v001 is about 3.3k chars)
# is exempt; this bounds in-run replacements only. Confirm before S2 disclosure.
GLOBAL_POLICY_MAX_CHARS = 6000

BATCH_EVENT = "role_policy_batch"
BATCH_SCHEMA = 1

ERROR_CODES = frozenset({
    "stale_version", "stale_proposal", "unknown_slot", "unknown_proposal", "not_authorized",
    "malformed_header", "malformed_body", "too_long", "duplicate_name", "duplicate_tool_call",
    "predecessor_not_terminal", "stale_succession", "conflicting_markers", "invalid_name",
    "run_ended",
})


class PolicyRejection(ValueError):
    """A model-visible rejection of one call; the relay rejects the whole response."""

    def __init__(self, code: str, message: str, *, tool_call_id: str | None = None, **details):
        if code not in ERROR_CODES:
            raise ValueError(f"Unknown role-policy error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code, self.tool_call_id, self.details = code, tool_call_id, details


class PolicyPersistError(OSError):
    """The batch record could not be written; nothing was published."""


class ReplayError(ValueError):
    """The event log cannot reliably rebuild role-policy state."""


# ---------------------------------------------------------------------------
# Parsing (pure). Header recognition is exact; near misses are rejected so a
# mistyped control message is never silently delivered as ordinary text.

_KEYWORDS = {"role-policy": "ROLE-POLICY", "policy-proposal": "POLICY-PROPOSAL",
             "policy-decision": "POLICY-DECISION", "global-policy": "GLOBAL-POLICY"}
# Hyphen/underscore-joined tokens only: "Role policy question" stays ordinary text,
# "role_policy" or "**POLICY-PROPOSAL**" is a malformed control header.
_CANDIDATE = re.compile(r"^[\s*#>`_~]*(role|policy|global)[-_](policy|proposal|decision)\b", re.I)
_BASE = r"(?:[ \t]+base=v(?P<base>0|[1-9][0-9]{0,8}))?"
_SLOT = r"[A-Za-z][A-Za-z0-9_-]{0,63}"
_ROLE_HEADER = re.compile(rf"ROLE-POLICY[ \t]+(?P<slot>{_SLOT}){_BASE}[ \t]*")
_PROPOSAL_HEADER = re.compile(rf"POLICY-PROPOSAL{_BASE}[ \t]*")
_GLOBAL_HEADER = re.compile(rf"GLOBAL-POLICY{_BASE}[ \t]*")
_DECISION_HEADER = re.compile(
    r"POLICY-DECISION[ \t]+#(?P<id>[1-9][0-9]{0,8})[ \t]+(?P<decision>(?i:approve|reject|amend))[ \t]*")
_LABEL = re.compile(r"^(CHANGE|REASON|EVIDENCE):", re.M)
_SUCCEEDS = re.compile(rf"SUCCEEDS:[ \t]*(?P<name>{_SLOT})")
_SUCCEEDS_NEAR = re.compile(r"succeeds[ \t]*:.*", re.I)


@dataclass(frozen=True)
class RolePolicyCall:
    tool_call_id: str
    to: str
    slot: str
    base: int | None
    text: str


@dataclass(frozen=True)
class ProposalCall:
    tool_call_id: str
    to: str
    base: int | None
    change: str
    reason: str
    evidence: str


@dataclass(frozen=True)
class DecisionCall:
    tool_call_id: str
    to: str
    proposal_id: int
    decision: str               # approve | reject | amend
    note: str | None
    amended_text: str | None


@dataclass(frozen=True)
class GlobalPolicyCall:
    tool_call_id: str
    to: str
    base: int | None
    text: str


@dataclass(frozen=True)
class SpawnCall:
    tool_call_id: str
    name: str
    initial_policy: str | None
    succeeds: str | None


ControlCall = RolePolicyCall | ProposalCall | DecisionCall | GlobalPolicyCall
PolicyCall = ControlCall | SpawnCall


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _tool_call_id(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("A tool call id is required")
    return value


def _limited(text: str, limit: int, what: str, tool_call_id: str) -> str:
    if len(text) > limit:
        raise PolicyRejection("too_long", f"{what} has {len(text)} characters; the limit is {limit}.",
                              tool_call_id=tool_call_id, limit=limit, length=len(text))
    return text


def _is_control_header(line: str) -> bool:
    match = _CANDIDATE.match(line)
    return bool(match) and f"{match.group(1)}-{match.group(2)}".lower() in _KEYWORDS


def parse_send_message(tool_call_id: str, to, message) -> ControlCall | None:
    """Return the control call in one send_message, or None for an ordinary message."""
    tool_call_id = _tool_call_id(tool_call_id)
    if not isinstance(message, str):
        return None
    text = _normalize(message)
    first, _, body = text.partition("\n")
    if not _is_control_header(first):
        return None
    header = first.strip()
    to = to if isinstance(to, str) else ""
    if match := _ROLE_HEADER.fullmatch(header):
        policy = body.strip()
        if not policy:
            raise PolicyRejection("malformed_body", "ROLE-POLICY needs the complete new policy from the second line.",
                                  tool_call_id=tool_call_id)
        return RolePolicyCall(tool_call_id, to, match["slot"], _base(match),
                              _limited(policy, ROLE_POLICY_MAX_CHARS, "The role policy", tool_call_id))
    if match := _GLOBAL_HEADER.fullmatch(header):
        policy = body.strip()
        if not policy:
            raise PolicyRejection("malformed_body", "GLOBAL-POLICY needs the complete new C from the second line.",
                                  tool_call_id=tool_call_id)
        return GlobalPolicyCall(tool_call_id, to, _base(match),
                                _limited(policy, GLOBAL_POLICY_MAX_CHARS, "The global policy", tool_call_id))
    if match := _PROPOSAL_HEADER.fullmatch(header):
        change, reason, evidence = _proposal_fields(body, tool_call_id)
        return ProposalCall(tool_call_id, to, _base(match), change, reason, evidence)
    if match := _DECISION_HEADER.fullmatch(header):
        decision = match["decision"].lower()
        note, amended = _decision_body(decision, body, tool_call_id)
        return DecisionCall(tool_call_id, to, int(match["id"]), decision, note, amended)
    raise PolicyRejection(
        "malformed_header",
        "Unrecognized control header. Use exactly one of: 'ROLE-POLICY <slot>', 'POLICY-PROPOSAL', "
        "'POLICY-DECISION #<id> approve|reject|amend', 'GLOBAL-POLICY', each optionally followed by "
        "' base=v<k>' where allowed, alone on the first line without formatting.",
        tool_call_id=tool_call_id)


def _base(match) -> int | None:
    return int(match["base"]) if match["base"] is not None else None


def _proposal_fields(body: str, tool_call_id: str) -> tuple[str, str, str]:
    content = body.strip()
    usage = "POLICY-PROPOSAL needs CHANGE:, REASON: and EVIDENCE:, in this order, each starting a line."
    labels = list(_LABEL.finditer(content))
    if not labels or labels[0].start() != 0 or labels[0].group(1) != "CHANGE":
        raise PolicyRejection("malformed_body", usage, tool_call_id=tool_call_id)
    _limited(content, PROPOSAL_MAX_CHARS, "The proposal (CHANGE/REASON/EVIDENCE with labels)", tool_call_id)
    # The first REASON after CHANGE and the first EVIDENCE after that; later
    # label-like lines are content.
    reason = next((m for m in labels if m.group(1) == "REASON"), None)
    evidence = next((m for m in labels if m.group(1) == "EVIDENCE" and reason and m.start() > reason.start()), None)
    if reason is None or evidence is None:
        raise PolicyRejection("malformed_body", usage, tool_call_id=tool_call_id)
    fields = (content[labels[0].end():reason.start()].strip(),
              content[reason.end():evidence.start()].strip(),
              content[evidence.end():].strip())
    if not all(fields):
        raise PolicyRejection("malformed_body", "CHANGE, REASON and EVIDENCE must each be nonempty.",
                              tool_call_id=tool_call_id)
    return fields


def _decision_body(decision: str, body: str, tool_call_id: str) -> tuple[str | None, str | None]:
    content = body.strip()
    if decision == "amend":
        if not content:
            raise PolicyRejection("malformed_body", "amend needs the complete replacement policy from the second line.",
                                  tool_call_id=tool_call_id)
        if _is_control_header(content.partition("\n")[0]):
            raise PolicyRejection("malformed_body", "Write the amended policy directly; do not nest another control header.",
                                  tool_call_id=tool_call_id)
        return None, _limited(content, ROLE_POLICY_MAX_CHARS, "The amended policy", tool_call_id)
    if not content:
        return None, None
    if not content.startswith("NOTE:"):
        raise PolicyRejection("malformed_body", "approve/reject may only be followed by an optional 'NOTE:' line.",
                              tool_call_id=tool_call_id)
    note = content[len("NOTE:"):].strip()
    return (_limited(note, NOTE_MAX_CHARS, "The note", tool_call_id) or None), None


def parse_spawn(tool_call_id: str, name, prompt) -> SpawnCall:
    """Read the optional ROLE-POLICY: block and SUCCEEDS: line of a spawn prompt.

    Call only in modes master/propose; in global mode the relay passes no markers.
    The prompt itself is not modified.
    """
    tool_call_id = _tool_call_id(tool_call_id)
    lines = _normalize(prompt).split("\n") if isinstance(prompt, str) else []
    block: list[str] | None = None
    blocks, succeeds = [], []
    for line in lines:
        stripped = line.strip()
        if match := _SUCCEEDS.fullmatch(stripped):
            succeeds.append(match["name"])
            continue
        if stripped == "ROLE-POLICY:":
            if block is not None or blocks:
                raise PolicyRejection("malformed_body", "A spawn prompt may contain only one ROLE-POLICY: block.",
                                      tool_call_id=tool_call_id)
            block = []
            continue
        if stripped == "END-ROLE-POLICY":
            if block is None:
                raise PolicyRejection("malformed_body", "END-ROLE-POLICY without a preceding ROLE-POLICY: line.",
                                      tool_call_id=tool_call_id)
            blocks.append(block)
            block = None
            continue
        if (stripped.lower() in {"role-policy:", "end-role-policy", "role_policy:", "end_role_policy"}
                or _SUCCEEDS_NEAR.fullmatch(stripped) and len(stripped) <= 80):
            raise PolicyRejection("malformed_header",
                                  "Spawn markers are exact lines: 'ROLE-POLICY:', 'END-ROLE-POLICY', 'SUCCEEDS: <name>'.",
                                  tool_call_id=tool_call_id)
        if block is not None:
            block.append(line)
    if block is not None:
        blocks.append(block)
    if len(succeeds) > 1:
        raise PolicyRejection("malformed_body", "A spawn prompt may contain only one SUCCEEDS: line.",
                              tool_call_id=tool_call_id)
    if blocks and succeeds:
        raise PolicyRejection("conflicting_markers",
                              "Use either a ROLE-POLICY: block for a new slot or SUCCEEDS: for succession, not both.",
                              tool_call_id=tool_call_id)
    initial = None
    if blocks:
        initial = "\n".join(blocks[0]).strip()
        if not initial:
            raise PolicyRejection("malformed_body", "The ROLE-POLICY: block is empty.", tool_call_id=tool_call_id)
        _limited(initial, ROLE_POLICY_MAX_CHARS, "The initial role policy", tool_call_id)
    if not isinstance(name, str):
        raise PolicyRejection("invalid_name", "A teammate name is required.", tool_call_id=tool_call_id)
    return SpawnCall(tool_call_id, name, initial, succeeds[0] if succeeds else None)


# ---------------------------------------------------------------------------
# Immutable views. A snapshot is everything one request carries; S2 renders it
# with a pure function, so equal state gives byte-identical injection text.

@dataclass(frozen=True)
class VersionView:
    version: int                # 0 = no role policy, C applies
    text: str | None
    provenance: str | None
    actor: str | None
    committed_at: str | None
    proposal_id: int | None


@dataclass(frozen=True)
class SlotView:
    slot_id: str
    instance: str | None
    lifecycle: str | None
    policy: VersionView


@dataclass(frozen=True)
class ProposalView:
    proposal_id: int
    slot_id: str
    proposer: str
    base_version: int
    change: str
    reason: str
    evidence: str
    created_at: str
    status: str
    note: str | None
    decided_version: int | None
    replaced_by: int | None
    void_reason: str | None


@dataclass(frozen=True)
class PolicySnapshot:
    mode: str
    role: str                   # "master" or the worker instance name
    global_policy: VersionView
    slots: tuple[SlotView, ...]         # master: every slot; worker: its own slot (0 or 1)
    proposals: tuple[ProposalView, ...]  # master: pending; worker: own pending and latest settled

    def slot(self, slot_id: str) -> SlotView | None:
        return next((view for view in self.slots if view.slot_id == slot_id), None)

    @property
    def pending_ids(self) -> frozenset[int]:
        return frozenset(view.proposal_id for view in self.proposals if view.status == "pending")


# ---------------------------------------------------------------------------
# Mutable state. Every mutation goes through _apply_change, live and replay alike.

@dataclass
class _Version:
    version: int
    text: str
    provenance: str
    proposal_id: int | None
    actor: str
    committed_at: str
    source_request: int | None
    tool_call_id: str | None


@dataclass
class _Slot:
    slot_id: str
    current_instance: str | None
    versions: list[_Version] = field(default_factory=list)

    @property
    def version(self) -> int:
        return self.versions[-1].version if self.versions else 0


@dataclass
class _Binding:
    instance: str
    slot_id: str
    lifecycle: str              # planned | active | terminal | voided
    predecessor: str | None
    end_reason: str | None = None


@dataclass
class _Proposal:
    proposal_id: int
    slot_id: str
    proposer: str
    base_version: int
    change: str
    reason: str
    evidence: str
    created_at: str
    source_request: int | None
    tool_call_id: str
    status: str = "pending"     # pending | approved | rejected | amended | voided
    decided_at: str | None = None
    note: str | None = None
    decided_version: int | None = None
    replaced_by: int | None = None
    void_reason: str | None = None


@dataclass
class _State:
    mode: str
    global_versions: list[_Version] = field(default_factory=list)
    slots: dict[str, _Slot] = field(default_factory=dict)
    bindings: dict[str, _Binding] = field(default_factory=dict)
    proposals: dict[int, _Proposal] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)
    next_proposal: int = 1
    closed: bool = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _apply_change(state: _State, change: dict) -> None:
    """Apply one persisted change. Raises ReplayError on any inconsistency."""
    kind = change.get("event")
    if kind == "role_policy_initialized":
        _require(not state.global_versions and change["mode"] == state.mode, "duplicate or mismatched initialization")
        state.global_versions.append(_Version(1, change["global_policy"], "initial", None, "runtime",
                                              change["committed_at"], None, None))
    elif kind == "global_policy_committed":
        _require(change["version"] == state.global_versions[-1].version + 1, "global version gap")
        state.global_versions.append(_Version(change["version"], change["text"], "master_set", None,
                                              change["actor"], change["committed_at"],
                                              change["source_request"], change["tool_call_id"]))
    elif kind == "role_policy_planned":
        instance, slot_id = change["instance"], change["slot"]
        existing = state.bindings.get(instance)
        _require(existing is None or existing.lifecycle == "voided", f"instance {instance} already bound")
        slot = state.slots.get(slot_id)
        if slot is None:
            _require(change["predecessor"] is None and slot_id == instance, f"unknown slot {slot_id}")
            slot = state.slots[slot_id] = _Slot(slot_id, None)
        _require(slot.current_instance == change["predecessor"], f"slot {slot_id} succession mismatch")
        slot.current_instance = instance
        state.bindings[instance] = _Binding(instance, slot_id, "planned", change["predecessor"])
    elif kind == "role_policy_bound":
        binding = state.bindings.get(change["instance"])
        _require(binding is not None and binding.lifecycle == "planned", "bind without a planned instance")
        binding.lifecycle = "active"
    elif kind == "role_policy_binding_ended":
        binding = state.bindings.get(change["instance"])
        reason = change["reason"]
        _require(binding is not None and binding.lifecycle in {"planned", "active"}, "end of an unbound instance")
        _require(not any(state.proposals[pid].proposer == binding.instance for pid in state.pending.values()),
                 "instance ended with a pending proposal")
        binding.end_reason = reason
        if reason == "terminal":
            binding.lifecycle = "terminal"
        elif reason in {"spawn_failed", "never_started"}:
            _require(binding.lifecycle == "planned", "a started instance cannot fail to spawn")
            binding.lifecycle = "voided"
            slot = state.slots[binding.slot_id]
            _require(slot.current_instance == binding.instance, "voided instance is not current")
            slot.current_instance = binding.predecessor
        else:
            raise ReplayError(f"unknown binding end reason {reason!r}")
    elif kind == "role_policy_committed":
        slot = state.slots.get(change["slot"])
        _require(slot is not None and change["version"] == slot.version + 1, "role policy version gap")
        slot.versions.append(_Version(change["version"], change["text"], change["provenance"], change["proposal_id"],
                                      change["actor"], change["committed_at"], change["source_request"],
                                      change["tool_call_id"]))
    elif kind == "role_policy_proposed":
        slot_id = change["slot"]
        _require(slot_id in state.slots and slot_id not in state.pending, "second pending proposal")
        _require(change["proposal_id"] == state.next_proposal, "proposal id gap")
        proposal = _Proposal(change["proposal_id"], slot_id, change["proposer"], change["base_version"],
                             change["change"], change["reason"], change["evidence"], change["created_at"],
                             change["source_request"], change["tool_call_id"])
        state.proposals[proposal.proposal_id] = proposal
        state.pending[slot_id] = proposal.proposal_id
        state.next_proposal += 1
    elif kind == "role_policy_proposal_renotified":
        proposal = state.proposals.get(change["proposal_id"])
        _require(proposal is not None and proposal.status == "pending", "renotify of a settled proposal")
    elif kind == "role_policy_proposal_voided":
        proposal = state.proposals.get(change["proposal_id"])
        _require(proposal is not None and proposal.status == "pending", "void of a settled proposal")
        proposal.status, proposal.void_reason = "voided", change["reason"]
        proposal.replaced_by, proposal.decided_at = change.get("replaced_by"), change["at"]
        del state.pending[proposal.slot_id]
    elif kind == "role_policy_decided":
        proposal = state.proposals.get(change["proposal_id"])
        _require(proposal is not None and proposal.status == "pending", "decision on a settled proposal")
        status = {"approve": "approved", "reject": "rejected", "amend": "amended"}[change["decision"]]
        if status != "rejected":
            _require(state.slots[proposal.slot_id].version == change["decided_version"],
                     "decision does not match the committed version")
        proposal.status, proposal.note = status, change["note"]
        proposal.decided_at, proposal.decided_version = change["decided_at"], change["decided_version"]
        del state.pending[proposal.slot_id]
    elif kind == "role_policy_run_ended":
        _require(not state.pending, "run ended with pending proposals")
        state.closed = True
    else:
        raise ReplayError(f"unknown role-policy change {kind!r}")


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version_view(version: _Version | None) -> VersionView:
    if version is None:
        return VersionView(0, None, None, None, None, None)
    return VersionView(version.version, version.text, version.provenance, version.actor,
                       version.committed_at, version.proposal_id)


def _proposal_view(proposal: _Proposal) -> ProposalView:
    return ProposalView(proposal.proposal_id, proposal.slot_id, proposal.proposer, proposal.base_version,
                        proposal.change, proposal.reason, proposal.evidence, proposal.created_at, proposal.status,
                        proposal.note, proposal.decided_version, proposal.replaced_by, proposal.void_reason)


def _slot_view(state: _State, slot: _Slot) -> SlotView:
    binding = state.bindings.get(slot.current_instance) if slot.current_instance else None
    return SlotView(slot.slot_id, slot.current_instance, binding.lifecycle if binding else None,
                    _version_view(slot.versions[-1] if slot.versions else None))


@dataclass(frozen=True)
class StagedBatch:
    actor: str
    snapshot: PolicySnapshot
    calls: tuple[PolicyCall, ...]
    source_request: int | None
    preview: tuple[dict, ...]   # changes as simulated at stage time; commit recomputes


@dataclass(frozen=True)
class CommitResult:
    batch: int | None           # None when the response had no policy calls
    changes: tuple[dict, ...]


class RolePolicyStore:
    """Authoritative role-policy state for one run (modes global/master/propose)."""

    def __init__(self, mode: str, initial_global_policy: str, writer: Callable[[dict], None], *,
                 clock: Callable[[], float] = time.time):
        if mode not in MODES:
            raise ValueError("Role-policy mode must be global, master or propose; off has no store")
        if not isinstance(initial_global_policy, str) or not initial_global_policy.strip():
            raise ValueError("The starting communication policy is required")
        self._lock = threading.Lock()
        self._writer, self._clock = writer, clock
        self._next_batch = 1
        state = _State(mode)
        changes: list[dict] = []
        self._emit(state, changes, {"event": "role_policy_initialized", "mode": mode,
                                    "global_policy": initial_global_policy.strip(),
                                    "committed_at": _iso(clock())})
        self._persist("init", "runtime", None, changes)
        self._state = state

    # -- construction from an event log -------------------------------------------------

    @classmethod
    def replay(cls, records: Iterable[dict], writer: Callable[[dict], None], *,
               clock: Callable[[], float] = time.time) -> "RolePolicyStore":
        """Rebuild state from persisted batch records; other events are ignored.

        Raises ReplayError for a missing, duplicated or inconsistent batch rather than
        continuing with partial state.
        """
        batches = [record for record in records if isinstance(record, dict) and record.get("event") == BATCH_EVENT]
        if not batches:
            raise ReplayError("No role-policy batches in the log")
        state: _State | None = None
        for expected, record in enumerate(batches, start=1):
            if record.get("schema") != BATCH_SCHEMA or record.get("batch") != expected:
                raise ReplayError(f"Missing or out-of-order role-policy batch before {record.get('batch')!r}")
            changes = record.get("changes")
            if not isinstance(changes, list) or not changes:
                raise ReplayError(f"Batch {expected} has no changes")
            if state is None:
                if (record.get("kind") != "init" or changes[0].get("event") != "role_policy_initialized"
                        or changes[0].get("mode") not in MODES):
                    raise ReplayError("The first batch must initialize the store")
                state = _State(changes[0]["mode"])
            try:
                for change in changes:
                    _apply_change(state, change)
            except (KeyError, TypeError, IndexError) as exc:
                raise ReplayError(f"Batch {expected} is incomplete: {exc!r}") from exc
        store = cls.__new__(cls)
        store._lock, store._writer, store._clock = threading.Lock(), writer, clock
        store._next_batch, store._state = len(batches) + 1, state
        return store

    @staticmethod
    def read_log(path: Path) -> list[dict]:
        """Parse an events.jsonl; any unreadable line makes the log unreliable."""
        records = []
        for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ReplayError(f"Unreadable event log line {number}") from exc
        return records

    # -- reads ---------------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._state.mode

    def snapshot(self, role: str) -> PolicySnapshot:
        """The view one request carries. Pure data; equal state gives equal snapshots."""
        with self._lock:
            state = self._state
            global_view = _version_view(state.global_versions[-1])
            if role == MASTER:
                slots = tuple(_slot_view(state, state.slots[key]) for key in sorted(state.slots))
                proposals = tuple(_proposal_view(state.proposals[state.pending[key]]) for key in sorted(state.pending))
                return PolicySnapshot(state.mode, role, global_view, slots, proposals)
            binding = state.bindings.get(role)
            if binding is None or binding.lifecycle == "voided":
                return PolicySnapshot(state.mode, role, global_view, (), ())
            own = [p for p in state.proposals.values() if p.proposer == role]
            pending = [p for p in own if p.status == "pending"]
            settled = [p for p in own if p.status != "pending"]
            views = pending + ([max(settled, key=lambda p: p.proposal_id)] if settled else [])
            return PolicySnapshot(state.mode, role, global_view,
                                  (_slot_view(state, state.slots[binding.slot_id]),),
                                  tuple(_proposal_view(p) for p in sorted(views, key=lambda p: p.proposal_id)))

    def export(self) -> dict:
        """Canonical dump of the full state, for tests and offline projection."""
        with self._lock:
            state = copy.deepcopy(self._state)
        return {"mode": state.mode, "closed": state.closed, "next_proposal": state.next_proposal,
                "next_batch": self._next_batch,
                "global_versions": [vars(v) for v in state.global_versions],
                "slots": {k: {"current_instance": s.current_instance, "versions": [vars(v) for v in s.versions]}
                          for k, s in sorted(state.slots.items())},
                "bindings": {k: vars(b) for k, b in sorted(state.bindings.items())},
                # String keys so the dump is identical after a JSON round trip.
                "proposals": {str(k): vars(p) for k, p in sorted(state.proposals.items())},
                "pending": dict(sorted(state.pending.items()))}

    # -- response path: stage in prepare_response(), commit at the end of handle() -------

    def stage(self, actor: str, snapshot: PolicySnapshot, calls: Iterable[PolicyCall], *,
              source_request: int | None = None) -> StagedBatch:
        """Validate all policy calls of one response on a copy of the current state.

        Raises PolicyRejection; nothing is written or published either way.
        """
        calls = tuple(calls)
        self._check_snapshot(actor, snapshot)
        with self._lock:
            state = copy.deepcopy(self._state)
        changes = self._simulate(state, actor, snapshot, calls, source_request) if calls else []
        return StagedBatch(actor, snapshot, calls, source_request, tuple(changes))

    def commit(self, staged: StagedBatch, *, check: Callable[[], None] | None = None) -> CommitResult:
        """Re-simulate on the latest state, persist one batch, then publish.

        ``check`` runs under the lock after simulation and before the write (the relay's
        cancellation/deadline check); any exception it raises aborts without state.
        The writer and ``check`` run while the store lock is held. S2 must therefore
        never call into the store while holding a lock that the writer or check takes
        (the relay's ``_lock`` behind ``event()``), or the two orders can deadlock.
        """
        if not staged.calls:
            return CommitResult(None, ())
        with self._lock:
            state = copy.deepcopy(self._state)
            changes = self._simulate(state, staged.actor, staged.snapshot, staged.calls, staged.source_request)
            if check is not None:
                check()
            batch = self._persist("response", staged.actor, staged.source_request, changes)
            self._state = state
            return CommitResult(batch, tuple(copy.deepcopy(changes)))

    # -- lifecycle inputs (first request, stderr terminal, spawn tool result, run end) ----

    def record_first_request(self, instance: str) -> str:
        """planned -> active at the instance's first coding request."""
        with self._lock:
            binding = self._state.bindings.get(instance)
            if binding is None:
                return "unknown"
            if binding.lifecycle != "planned":
                return binding.lifecycle
            self._lifecycle(instance, [{"event": "role_policy_bound", "instance": instance,
                                        "at": _iso(self._clock())}])
            return "bound"

    def record_terminal(self, instance: str, *, source: str | None = None) -> str:
        """A relay-consumed worker_terminal; voids the instance's pending proposal. Idempotent."""
        with self._lock:
            state = self._state
            binding = state.bindings.get(instance)
            if binding is None:
                return "unknown"
            if binding.lifecycle not in {"planned", "active"}:
                return binding.lifecycle
            at = _iso(self._clock())
            changes = [{"event": "role_policy_proposal_voided", "proposal_id": pid, "reason": "proposer_terminal",
                        "replaced_by": None, "at": at}
                       for pid in sorted(state.pending.values()) if state.proposals[pid].proposer == instance]
            changes.append({"event": "role_policy_binding_ended", "instance": instance, "reason": "terminal",
                            "source": source, "at": at})
            self._lifecycle(instance, changes)
            return "terminal"

    def record_spawn_result(self, instance: str, *, ok: bool, tool_call_id: str | None = None) -> str:
        """The master's agent tool result. A failed spawn frees the name and restores the slot."""
        with self._lock:
            binding = self._state.bindings.get(instance)
            if binding is None:
                return "unknown"
            if ok:
                return "ok"
            if binding.lifecycle != "planned":
                return "conflict"  # the instance already made a request; the failure report is not adopted
            self._lifecycle(instance, [{"event": "role_policy_binding_ended", "instance": instance,
                                        "reason": "spawn_failed", "source": tool_call_id,
                                        "at": _iso(self._clock())}])
            return "voided"

    def end_run(self) -> int:
        """Void pending proposals (run_ended) and never-started planned instances; close writes.

        Returns the number of voided proposals. A planned instance that never made a
        coding request is ended as never_started, so the ledger and its replay do not
        keep a binding the native team never ran.
        """
        with self._lock:
            state = self._state
            if state.closed:
                return 0
            at = _iso(self._clock())
            changes = [{"event": "role_policy_proposal_voided", "proposal_id": state.pending[key],
                        "reason": "run_ended", "replaced_by": None, "at": at} for key in sorted(state.pending)]
            voided = len(changes)
            changes += [{"event": "role_policy_binding_ended", "instance": name, "reason": "never_started",
                         "source": "run_end", "at": at}
                        for name, binding in sorted(state.bindings.items()) if binding.lifecycle == "planned"]
            self._lifecycle(None, changes + [{"event": "role_policy_run_ended", "at": at}])
            return voided

    # -- internals -----------------------------------------------------------------------

    def _lifecycle(self, actor: str | None, changes: list[dict]) -> None:
        """Caller holds the lock."""
        state = copy.deepcopy(self._state)
        for change in changes:
            _apply_change(state, change)
        self._persist("lifecycle", actor or "runtime", None, changes)
        self._state = state

    def _persist(self, kind: str, actor: str, source_request: int | None, changes: list[dict]) -> int:
        record = {"event": BATCH_EVENT, "schema": BATCH_SCHEMA, "batch": self._next_batch, "kind": kind,
                  "actor": actor, "source_request": source_request, "changes": copy.deepcopy(changes)}
        try:
            self._writer(record)
        except Exception as exc:
            raise PolicyPersistError(f"Role-policy batch {self._next_batch} was not persisted: {exc}") from exc
        self._next_batch += 1
        return record["batch"]

    @staticmethod
    def _emit(state: _State, changes: list[dict], change: dict) -> None:
        _apply_change(state, change)
        changes.append(change)

    def _check_snapshot(self, actor: str, snapshot: PolicySnapshot) -> None:
        if not isinstance(snapshot, PolicySnapshot) or snapshot.role != actor or snapshot.mode != self._state.mode:
            raise ValueError("The snapshot must be the one injected into this actor's request")

    def _simulate(self, state: _State, actor: str, snapshot: PolicySnapshot, calls: tuple,
                  source_request: int | None) -> list[dict]:
        if state.closed:
            raise PolicyRejection("run_ended", "The run has ended; policy state is closed.")
        changes: list[dict] = []
        seen: set[str] = set()
        now = _iso(self._clock())
        for call in calls:
            if call.tool_call_id in seen:
                raise PolicyRejection("duplicate_tool_call", "Duplicate tool call id in one response.",
                                      tool_call_id=call.tool_call_id)
            seen.add(call.tool_call_id)
            context = (state, changes, actor, snapshot, call, source_request, now)
            if isinstance(call, SpawnCall):
                self._spawn(*context)
            elif isinstance(call, RolePolicyCall):
                self._set_role(*context)
            elif isinstance(call, GlobalPolicyCall):
                self._set_global(*context)
            elif isinstance(call, ProposalCall):
                self._propose(*context)
            elif isinstance(call, DecisionCall):
                self._decide(*context)
            else:
                raise TypeError(f"Unsupported policy call {type(call).__name__}")
        return changes

    def _stale(self, call, what: str, snapshot_version: int, current: int, given: int | None):
        raise PolicyRejection(
            "stale_version",
            f"{what}: your request snapshot shows version {snapshot_version}"
            + (f" but the header says base=v{given}" if given is not None and given != snapshot_version else "")
            + (f"; the current version is {current}" if snapshot_version != current else "")
            + ". Re-read the next supplied snapshot before retrying.",
            tool_call_id=call.tool_call_id, snapshot_version=snapshot_version, current_version=current, base=given)

    def _check_base(self, call, what: str, snapshot_version: int, current: int) -> None:
        if (call.base is not None and call.base != snapshot_version) or snapshot_version != current:
            self._stale(call, what, snapshot_version, current, call.base)

    def _spawn(self, state, changes, actor, snapshot, call: SpawnCall, source_request, now):
        if actor != MASTER:
            raise PolicyRejection("not_authorized", "Only the master spawns teammates.", tool_call_id=call.tool_call_id)
        try:
            validate_worker_name(call.name)
        except ValueError as exc:
            raise PolicyRejection("invalid_name", str(exc), tool_call_id=call.tool_call_id) from exc
        if state.mode not in ROLE_MODES and (call.initial_policy is not None or call.succeeds is not None):
            raise PolicyRejection("not_authorized", "Spawn policy markers are not available in this mode.",
                                  tool_call_id=call.tool_call_id)
        existing = state.bindings.get(call.name)
        if existing is not None and existing.lifecycle != "voided":
            raise PolicyRejection("duplicate_name",
                                  f"The name {call.name} was already used in this run; choose a new name.",
                                  tool_call_id=call.tool_call_id)
        if call.succeeds is not None:
            predecessor = state.bindings.get(call.succeeds)
            if predecessor is None or predecessor.lifecycle == "voided":
                raise PolicyRejection("unknown_slot", f"No teammate {call.succeeds} to succeed.",
                                      tool_call_id=call.tool_call_id)
            if predecessor.lifecycle != "terminal":
                raise PolicyRejection("predecessor_not_terminal",
                                      f"{call.succeeds} has not been confirmed terminal by the runtime yet; retry later.",
                                      tool_call_id=call.tool_call_id)
            slot = state.slots[predecessor.slot_id]
            if slot.current_instance != predecessor.instance:
                raise PolicyRejection("stale_succession",
                                      f"Slot {slot.slot_id} is already held by {slot.current_instance}.",
                                      tool_call_id=call.tool_call_id, current_instance=slot.current_instance)
            self._emit(state, changes, {"event": "role_policy_planned", "instance": call.name,
                                        "slot": slot.slot_id, "predecessor": predecessor.instance,
                                        "source_request": source_request, "tool_call_id": call.tool_call_id})
            return
        slot = state.slots.get(call.name)
        if slot is not None and slot.current_instance is not None:
            raise PolicyRejection("duplicate_name", f"{call.name} names an existing slot held by {slot.current_instance}.",
                                  tool_call_id=call.tool_call_id)
        self._emit(state, changes, {"event": "role_policy_planned", "instance": call.name, "slot": call.name,
                                    "predecessor": None, "source_request": source_request,
                                    "tool_call_id": call.tool_call_id})
        if call.initial_policy is not None:
            self._emit(state, changes, {
                "event": "role_policy_committed", "slot": call.name, "version": state.slots[call.name].version + 1,
                "text": call.initial_policy, "provenance": "spawn_initial", "proposal_id": None, "actor": actor,
                "committed_at": now, "source_request": source_request, "tool_call_id": call.tool_call_id})

    def _set_role(self, state, changes, actor, snapshot, call: RolePolicyCall, source_request, now):
        if actor != MASTER or state.mode not in ROLE_MODES:
            raise PolicyRejection("not_authorized", "ROLE-POLICY is not available to you in this mode.",
                                  tool_call_id=call.tool_call_id)
        if call.to == BROADCAST:
            raise PolicyRejection("not_authorized", "ROLE-POLICY is single-recipient; send it to the slot's instance.",
                                  tool_call_id=call.tool_call_id)
        view, slot = snapshot.slot(call.slot), state.slots.get(call.slot)
        if view is None or slot is None:
            raise PolicyRejection("unknown_slot", f"Slot {call.slot} is not in your supplied snapshot.",
                                  tool_call_id=call.tool_call_id)
        binding = state.bindings.get(slot.current_instance) if slot.current_instance else None
        running = binding is not None and binding.lifecycle in {"planned", "active"}
        if not running or call.to != slot.current_instance:
            holder = slot.current_instance if running else "none"
            raise PolicyRejection("not_authorized",
                                  f"Send ROLE-POLICY {call.slot} to the slot's current running instance ({holder}).",
                                  tool_call_id=call.tool_call_id)
        self._check_base(call, f"ROLE-POLICY {call.slot}", view.policy.version, slot.version)
        version = slot.version + 1
        self._emit(state, changes, {
            "event": "role_policy_committed", "slot": call.slot, "version": version, "text": call.text,
            "provenance": "master_set", "proposal_id": None, "actor": actor, "committed_at": now,
            "source_request": source_request, "tool_call_id": call.tool_call_id})
        if call.slot in state.pending:
            self._emit(state, changes, {"event": "role_policy_proposal_voided", "proposal_id": state.pending[call.slot],
                                        "reason": "policy_superseded", "replaced_by": None, "at": now})

    def _set_global(self, state, changes, actor, snapshot, call: GlobalPolicyCall, source_request, now):
        if actor != MASTER or state.mode != "global":
            raise PolicyRejection("not_authorized", "GLOBAL-POLICY is not available to you in this mode.",
                                  tool_call_id=call.tool_call_id)
        if call.to != BROADCAST:
            raise PolicyRejection("not_authorized", 'GLOBAL-POLICY must be broadcast with to="*".',
                                  tool_call_id=call.tool_call_id)
        current = state.global_versions[-1].version
        self._check_base(call, "GLOBAL-POLICY", snapshot.global_policy.version, current)
        self._emit(state, changes, {
            "event": "global_policy_committed", "version": current + 1, "text": call.text, "actor": actor,
            "committed_at": now, "source_request": source_request, "tool_call_id": call.tool_call_id})

    def _propose(self, state, changes, actor, snapshot, call: ProposalCall, source_request, now):
        if actor == MASTER or state.mode != "propose":
            raise PolicyRejection("not_authorized", "POLICY-PROPOSAL is not available to you in this mode.",
                                  tool_call_id=call.tool_call_id)
        if call.to != LEADER:
            raise PolicyRejection("not_authorized", 'Send POLICY-PROPOSAL to "leader".', tool_call_id=call.tool_call_id)
        binding = state.bindings.get(actor)
        view = snapshot.slots[0] if snapshot.slots else None
        if binding is None or view is None or binding.slot_id != view.slot_id:
            raise PolicyRejection("unknown_slot", "No role slot is bound to you in your supplied snapshot.",
                                  tool_call_id=call.tool_call_id)
        slot = state.slots[binding.slot_id]
        if binding.lifecycle != "active" or slot.current_instance != actor:
            raise PolicyRejection("not_authorized", "Only the slot's current running instance can propose.",
                                  tool_call_id=call.tool_call_id)
        self._check_base(call, "POLICY-PROPOSAL", view.policy.version, slot.version)
        base = slot.version
        pending_id = state.pending.get(slot.slot_id)
        if pending_id is not None:
            pending = state.proposals[pending_id]
            if (pending.proposer == actor and pending.base_version == base
                    and (pending.change, pending.reason, pending.evidence) == (call.change, call.reason, call.evidence)):
                self._emit(state, changes, {"event": "role_policy_proposal_renotified", "proposal_id": pending_id,
                                            "source_request": source_request, "tool_call_id": call.tool_call_id})
                return
            self._emit(state, changes, {"event": "role_policy_proposal_voided", "proposal_id": pending_id,
                                        "reason": "replaced", "replaced_by": state.next_proposal, "at": now})
        self._emit(state, changes, {
            "event": "role_policy_proposed", "proposal_id": state.next_proposal, "slot": slot.slot_id,
            "proposer": actor, "base_version": base, "change": call.change, "reason": call.reason,
            "evidence": call.evidence, "created_at": now, "source_request": source_request,
            "tool_call_id": call.tool_call_id})

    def _decide(self, state, changes, actor, snapshot, call: DecisionCall, source_request, now):
        if actor != MASTER or state.mode != "propose":
            raise PolicyRejection("not_authorized", "POLICY-DECISION is not available to you in this mode.",
                                  tool_call_id=call.tool_call_id)
        proposal = state.proposals.get(call.proposal_id)
        if proposal is None:
            raise PolicyRejection("unknown_proposal", f"There is no proposal #{call.proposal_id}.",
                                  tool_call_id=call.tool_call_id)
        if call.to != proposal.proposer:
            raise PolicyRejection("not_authorized", f"Send the decision on #{call.proposal_id} to {proposal.proposer}.",
                                  tool_call_id=call.tool_call_id)
        current_pending = state.pending.get(proposal.slot_id)
        binding = state.bindings.get(proposal.proposer)
        slot = state.slots[proposal.slot_id]
        if (proposal.status != "pending" or call.proposal_id not in snapshot.pending_ids
                or binding is None or binding.lifecycle != "active" or proposal.base_version != slot.version):
            reason = ("it is not in your supplied snapshot" if proposal.status == "pending"
                      and call.proposal_id not in snapshot.pending_ids else f"it is {proposal.status}")
            raise PolicyRejection(
                "stale_proposal",
                f"Proposal #{call.proposal_id} cannot be decided: {reason}. "
                + (f"The current pending proposal for slot {proposal.slot_id} is #{current_pending}; "
                   "read its full content in your next snapshot and decide again." if current_pending
                   else f"Slot {proposal.slot_id} has no pending proposal."),
                tool_call_id=call.tool_call_id, current_pending=current_pending)
        decided_version = None
        if call.decision in {"approve", "amend"}:
            decided_version = slot.version + 1
            approved = call.decision == "approve"
            self._emit(state, changes, {
                "event": "role_policy_committed", "slot": slot.slot_id, "version": decided_version,
                "text": proposal.change if approved else call.amended_text,
                "provenance": "proposal_approved" if approved else "proposal_amended",
                "proposal_id": proposal.proposal_id, "actor": actor, "committed_at": now,
                "source_request": source_request, "tool_call_id": call.tool_call_id})
        self._emit(state, changes, {
            "event": "role_policy_decided", "proposal_id": proposal.proposal_id, "decision": call.decision,
            "note": call.note, "decided_version": decided_version, "decided_at": now, "actor": actor,
            "source_request": source_request, "tool_call_id": call.tool_call_id})
