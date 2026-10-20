"""Run-local shared-role policy state with per-call rejection and late CAS.

The relay supplies its existing event writer and cancellation check. Changes are
simulated on private copies, journaled as one batch, then published under a lock.
Native delivery never undoes a committed policy. Legacy slot stores are untouched.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import re
import threading
import time
from typing import Callable, Iterable

from agent_company.policy.role_package import (
    MAX_COMMUNICATION_CHARS, MAX_PENDING_CHARS, MAX_ROLE_CHARS, MAX_TOTAL_ROLE_CHARS,
    INITIAL_ROLES, ROLE_ID_PATTERN, validate_role_package,
)
from agent_company.policy.role_policy import PolicyPersistError, ReplayError, VersionView
from agent_company.runtime.team import validate_worker_name

BATCH_EVENT = "role_package_batch"
MASTER = "master"
MODES = ("off", "propose")
_BASE = r"(?:[ \t]+base=v(?P<base>0|[1-9][0-9]{0,8}))?"
_ID = r"[A-Za-z][A-Za-z0-9_-]{0,63}"
_CANDIDATE = re.compile(r"^[\s*#>`_~]*(role|policy|global)[-_](policy|proposal|decision)\b", re.I)
_UPDATE = re.compile(rf"(?P<kind>GLOBAL-POLICY|ROLE-POLICY(?:[ \t]+(?P<role>{ROLE_ID_PATTERN})))[ \t]+(?P<op>append|replace){_BASE}[ \t]*")
_PROPOSE = re.compile(rf"POLICY-PROPOSAL[ \t]+(?P<target>C|role:{ROLE_ID_PATTERN})[ \t]+(?P<op>append|replace){_BASE}[ \t]*")
_DECIDE = re.compile(r"POLICY-DECISION[ \t]+#(?P<id>[1-9][0-9]{0,8})[ \t]+(?P<decision>approve|reject|amend)[ \t]*")


class PackageRejection(ValueError):
    """One tool call was rejected; independent valid calls can still commit."""

    def __init__(self, code: str, detail: str, *, tool_call_id: str, **details):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.tool_call_id, self.details = code, detail, tool_call_id, details


@dataclass(frozen=True)
class UpdateCall:
    tool_call_id: str
    to: str
    target: str
    operation: str
    text: str
    base: int | None = None


@dataclass(frozen=True)
class ProposalCall:
    tool_call_id: str
    to: str
    target: str
    operation: str
    change: str
    reason: str
    evidence: str
    base: int | None = None
    raw_body: str | None = None


@dataclass(frozen=True)
class DecisionCall:
    tool_call_id: str
    to: str
    proposal_id: int
    decision: str
    note: str | None = None
    amended_text: str | None = None
    raw_body: str | None = None


@dataclass(frozen=True)
class SpawnCall:
    tool_call_id: str
    name: str
    role_id: str
    binding_source: str
    succeeds: str | None = None


PackageCall = UpdateCall | ProposalCall | DecisionCall | SpawnCall


def _fail(call, code, detail, **details):
    raise PackageRejection(code, detail, tool_call_id=call if isinstance(call, str) else call.tool_call_id,
                           **details)


def _id(tool_call_id):
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("A tool call id is required")
    return tool_call_id


def malformed_header_detail(mode: str, actor: str, role_id: str | None) -> str:
    if mode == "off":
        return "Policy controls are not available in mode off; send an ordinary message instead."
    if actor == MASTER:
        return ("Use GLOBAL-POLICY append|replace, ROLE-POLICY <role_id> append|replace, or "
                "POLICY-DECISION #<id> approve|reject|amend, alone on the first line; "
                "base=v<k> is optional except on POLICY-DECISION.")
    if role_id is None:
        raise RuntimeError("A worker must have an authenticated role binding before parsing controls")
    return (f"Use POLICY-PROPOSAL C append|replace or POLICY-PROPOSAL role:{role_id} append|replace, "
            "optionally followed by base=v<k>, alone on the first line.")


def parse_send_message(tool_call_id: str, to, message, *, mode="propose", actor=MASTER,
                       role_id=None) -> UpdateCall | ProposalCall | DecisionCall | None:
    _id(tool_call_id)
    if not isinstance(message, str):
        return None
    first, _, body = message.partition("\n")
    if not _CANDIDATE.match(first):
        return None
    header = first.strip()
    recipient = to if isinstance(to, str) else ""
    if match := _UPDATE.fullmatch(header):
        target = f"role:{match['role']}" if match["role"] else "C"
        return UpdateCall(tool_call_id, recipient, target, match["op"], body,
                          int(match["base"]) if match["base"] is not None else None)
    if match := _PROPOSE.fullmatch(header):
        return ProposalCall(tool_call_id, recipient, match["target"], match["op"], "", "", "",
                            int(match["base"]) if match["base"] is not None else None, raw_body=body)
    if match := _DECIDE.fullmatch(header):
        return DecisionCall(tool_call_id, recipient, int(match["id"]), match["decision"], raw_body=body)
    _fail(tool_call_id, "malformed_header", malformed_header_detail(mode, actor, role_id))


def _validated_body(call, *, target=None):
    """Body diagnostics follow mode/actor/recipient checks, never teach denied controls."""
    body = call.raw_body
    if body is None:
        return call
    if isinstance(call, ProposalCall):
        usage = ("POLICY-PROPOSAL needs CHANGE:, REASON: and EVIDENCE:, in that order, each starting a line. "
                 "REASON and EVIDENCE must be nonempty. CHANGE may be empty only for a role replacement "
                 "that clears the role policy.")
        fields = re.fullmatch(r"CHANGE:[ \t]?(.*?)\r?\nREASON:[ \t]?(.*?)\r?\nEVIDENCE:[ \t]?(.*)", body, re.S)
        if not fields or not fields[2].strip() or not fields[3].strip():
            _fail(call, "malformed_body", usage)
        change, reason, evidence = fields.groups()
        if not change.strip() and not (call.target != "C" and call.operation == "replace"):
            _fail(call, "malformed_body", usage)
        if len(body) > 2000:
            _fail(call, "too_long", f"The proposal (CHANGE/REASON/EVIDENCE with labels) has {len(body)} characters; the limit is 2000.")
        return replace(call, change=change, reason=reason, evidence=evidence, raw_body=None)
    note, amended = None, None
    if call.decision == "amend":
        # NOTE is explanation, never role/C policy text carried into later runs.
        marker = re.search(r"(?m)^NOTE:[ \t]?", body)
        amended = body[:marker.start()].removesuffix("\n").removesuffix("\r") if marker else body
        note = body[marker.end():] if marker else None
        if marker and (not amended.strip() or not re.fullmatch(r"[^\r\n]*(?:\r?\n)?", note)):
            _fail(call, "malformed_body", f"Amend supplies the complete replacement policy for {target} below the header, without another control header. A role amendment may be empty to clear its policy; C must remain nonempty.")
    elif body:
        if not body.startswith("NOTE:"):
            _fail(call, "malformed_body", "approve/reject may only be followed by an optional 'NOTE:' line.")
        note = body[5:].lstrip(" \t")
    if note is not None and len(note) > 2000:
        _fail(call, "too_long", f"The note has {len(note)} characters; the limit is 2000.")
    return replace(call, note=note, amended_text=amended, raw_body=None)


def parse_spawn(tool_call_id: str, name, prompt, *, role_ids=INITIAL_ROLES) -> SpawnCall:
    _id(tool_call_id)
    if not isinstance(name, str):
        _fail(tool_call_id, "invalid_name", "A teammate name is required.")
    try:
        validate_worker_name(name)
    except ValueError as exc:
        _fail(tool_call_id, "invalid_name", str(exc))
    if not isinstance(prompt, str):
        raise ValueError("A spawn prompt is required")
    roles, predecessors = [], []
    syntax = "Spawn markers are exact complete lines: ROLE-TYPE: <role_id> and, optionally, SUCCEEDS: <previous-name>."
    for line in prompt.splitlines():
        line = line.strip()
        if re.match(r"(?i)ROLE[-_]POLICY:", line):
            _fail(tool_call_id, "inline_policy_not_supported", "Spawn prompts cannot carry policy text in this run. Remove the ROLE-POLICY: block; the teammate receives its role type's current policy automatically.")
        if re.match(r"(?i)[*#>`_~]*(?:ROLE[-_ ]TYPE):", line):
            match = re.fullmatch(rf"ROLE-TYPE:[ \t]*({ROLE_ID_PATTERN})", line)
            if not match:
                if line.startswith("ROLE-TYPE:"):
                    _fail(tool_call_id, "invalid_role_id", f"That role type is not in the role directory. Use one of: {', '.join(sorted(role_ids))}.")
                _fail(tool_call_id, "malformed_header", syntax)
            roles.append(match[1])
        if re.match(r"(?i)[*#>`_~]*SUCCEEDS:", line):
            match = re.fullmatch(rf"SUCCEEDS:[ \t]*({_ID})", line)
            if not match:
                _fail(tool_call_id, "malformed_header", syntax)
            predecessors.append(match[1])
    if len(roles) > 1:
        _fail(tool_call_id, "malformed_body", "A spawn prompt may contain only one ROLE-TYPE: line.")
    if len(predecessors) > 1:
        _fail(tool_call_id, "malformed_body", "A spawn prompt may contain only one SUCCEEDS: line.")
    if roles == [MASTER]:
        _fail(tool_call_id, "reserved_role", "The master role type belongs to the coordination-only root and cannot be assigned to a worker.")
    return SpawnCall(tool_call_id, name, roles[0] if roles else "implementer",
                     "explicit" if roles else "default", predecessors[0] if predecessors else None)


@dataclass(frozen=True)
class RoleView:
    role_id: str
    policy: VersionView


@dataclass(frozen=True)
class BindingView:
    instance: str
    role_id: str
    slot_id: str
    lifecycle: str
    binding_source: str
    predecessor: str | None
    spawn_order: int
    confirmed: bool


@dataclass(frozen=True)
class ProposalView:
    proposal_id: int
    target: str
    operation: str
    proposer: str
    proposer_role_id: str
    base_version: int
    change: str
    reason: str
    evidence: str
    created_at: str
    source_request: int | None
    tool_call_id: str
    status: str = "pending"
    note: str | None = None
    decided_version: int | None = None
    replaced_by: int | None = None
    void_reason: str | None = None


@dataclass(frozen=True)
class PackageSnapshot:
    mode: str
    actor: str
    role_id: str
    global_policy: VersionView
    roles: tuple[RoleView, ...]
    bindings: tuple[BindingView, ...]
    proposals: tuple[ProposalView, ...]

    @property
    def role(self):
        return self.actor

    @property
    def pending_ids(self):
        return frozenset(p.proposal_id for p in self.proposals if p.status == "pending")

    def target(self, target):
        if target == "C":
            return self.global_policy
        return next((r.policy for r in self.roles if f"role:{r.role_id}" == target), None)


def render_pending(proposals: Iterable[ProposalView]) -> str:
    """The exact approved master tail; also the admission-budget measurement."""
    rows = [f"Proposal #{p.proposal_id} for {p.target}, operation {p.operation}, from {p.proposer}, "
            f"base version {p.base_version}, submitted at {p.created_at}:\n"
            f"CHANGE: {p.change}\nREASON: {p.reason}\nEVIDENCE: {p.evidence}"
            for p in proposals if p.status == "pending"]
    return "Pending proposals:\n" + "\n\n".join(rows) if rows else "Pending proposals: none."


@dataclass(frozen=True)
class StagedBatch:
    actor: str
    snapshot: PackageSnapshot
    calls: tuple[PackageCall, ...]
    source_request: int | None
    preview: tuple[dict, ...]
    rejections: tuple[PackageRejection, ...]

    @property
    def accepted_call_ids(self):
        return tuple(c.tool_call_id for c in self.calls)


@dataclass(frozen=True)
class CommitResult:
    batch: int | None
    changes: tuple[dict, ...]
    rejections: tuple[PackageRejection, ...]
    accepted_call_ids: tuple[str, ...]


def _now(clock):
    return datetime.fromtimestamp(clock(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _view(value):
    return VersionView(value["version"], value["text"], value["provenance"], value["actor"],
                       value["committed_at"], value.get("proposal_id"))


class RolePackageStore:
    def __init__(self, mode: str, policy_package: dict, writer: Callable[[dict], None], *, clock=time.time):
        if mode not in MODES:
            raise ValueError("New policy packages support off and propose only")
        package = validate_role_package(policy_package)
        self._writer, self._clock, self._lock = writer, clock, threading.RLock()
        self._next_batch = 1
        self._state = {}
        change = {"event": "role_package_initialized", "mode": mode, "package": package,
                  "at": _now(clock)}
        state = {}
        self._apply(state, change)
        self._persist("init", "runtime", None, [change])
        self._state = state

    @property
    def mode(self):
        return self._state["mode"]

    def snapshot(self, actor: str) -> PackageSnapshot:
        with self._lock:
            s = self._state
            if actor != MASTER and actor not in s["bindings"]:
                raise RuntimeError("Worker requested a package snapshot without a committed binding")
            binding = s["bindings"].get(actor)
            if binding and binding["lifecycle"] == "voided":
                raise RuntimeError("Voided worker requested a policy snapshot")
            role_id = MASTER if actor == MASTER else binding["role_id"]
            roles = tuple(RoleView(r, _view(s["targets"][f"role:{r}"][-1]))
                          for r in sorted(s["package"]["roles"])
                          if not s["package"]["roles"][r]["retired"])
            proposals = list(s["proposals"].values())
            if actor == MASTER:
                proposals = [p for p in proposals if p["status"] == "pending"]
            else:
                own = [p for p in proposals if p["proposer"] == actor]
                latest = {}
                for p in own:
                    if p["status"] != "pending":
                        latest[p["target"]] = p
                proposals = [p for p in own if p["status"] == "pending"] + list(latest.values())
            return PackageSnapshot(self.mode, actor, role_id, _view(s["targets"]["C"][-1]), roles,
                                   tuple(BindingView(**copy.deepcopy(b)) for b in s["bindings"].values()),
                                   tuple(ProposalView(**copy.deepcopy(p)) for p in sorted(proposals, key=lambda p: p["proposal_id"])))

    request_snapshot = snapshot

    def export(self):
        with self._lock:
            return copy.deepcopy(self._state)

    export_state = export

    def stage(self, actor: str, snapshot: PackageSnapshot, calls: Iterable[PackageCall], *, source_request=None):
        if snapshot.actor != actor or snapshot.mode != self.mode:
            raise ValueError("Policy snapshot does not belong to this actor/mode")
        with self._lock:
            state = copy.deepcopy(self._state)
        _, changes, accepted, errors = self._simulate(state, actor, snapshot, tuple(calls), source_request)
        return StagedBatch(actor, snapshot, tuple(accepted), source_request, tuple(changes), tuple(errors))

    def commit(self, staged: StagedBatch, *, check=None):
        with self._lock:
            state, changes, accepted, errors = self._simulate(copy.deepcopy(self._state), staged.actor,
                                                              staged.snapshot, staged.calls, staged.source_request)
            if check is not None:
                check()
            batch = self._persist("response", staged.actor, staged.source_request, changes) if changes else None
            if changes:
                self._state = state
            return CommitResult(batch, tuple(copy.deepcopy(changes)), staged.rejections + tuple(errors),
                                tuple(c.tool_call_id for c in accepted))

    def _simulate(self, state, actor, snapshot, calls, source_request):
        changes, accepted, errors, seen = [], [], [], set()
        for call in calls:
            candidate, own_changes = copy.deepcopy(state), []
            try:
                if call.tool_call_id in seen:
                    _fail(call, "duplicate_tool_call", "Duplicate tool call id in one response.")
                seen.add(call.tool_call_id)
                if state["closed"]:
                    _fail(call, "run_ended", "The run has ended; policy state is closed.")
                self._execute(candidate, own_changes, actor, snapshot, call, source_request)
            except PackageRejection as exc:
                errors.append(exc)
                continue
            state = candidate
            changes.extend(own_changes)
            accepted.append(call)
        return state, changes, accepted, errors

    def _emit(self, state, changes, change):
        self._apply(state, change)
        changes.append(change)

    def _base(self, state, snapshot, call, target, given=None):
        view = snapshot.target(target)
        current = state["targets"][target][-1]["version"]
        if view is None or view.version != current or given is not None and given != view.version:
            _fail(call, "stale_version", f"{target} is now version {current}; your request showed version {view.version if view else 0} (base: {'v'+str(given) if given is not None else 'omitted'}). Nothing changed. Check the current text in your instructions and resubmit if still needed.")
        return current

    def _target(self, state, call, target):
        active = sorted(r for r, value in state["package"]["roles"].items() if not value["retired"])
        if target != "C" and target.removeprefix("role:") not in active:
            _fail(call, "unknown_role", f"That role type is not in the role directory. Use one of: {', '.join(active)}.")

    def _resulting_text(self, state, call, target, operation, text, *, amended=False):
        if operation == "append" and not text.strip():
            _fail(call, "malformed_body", "Append needs nonempty policy text from the second line.")
        if target == "C" and not text.strip():
            _fail(call, "malformed_body", "C must remain nonempty. Supply the complete replacement C below the header.")
        current = state["targets"][target][-1]["text"]
        result = (current + ("\n" if current else "") + text) if operation == "append" else text
        limit = MAX_COMMUNICATION_CHARS if target == "C" else MAX_ROLE_CHARS
        if len(result) > limit:
            detail = (f"The amended {target} policy has {len(result)} characters; the limit is {limit}." if amended else
                      f"The resulting {target} policy has {len(result)} characters; the limit is {limit}. Append is checked after combining with the current text.")
            _fail(call, "too_long", detail)
        if target != "C":
            total = sum(len(v[-1]["text"]) for t, v in state["targets"].items() if t != "C") - len(current) + len(result)
            if total > MAX_TOTAL_ROLE_CHARS:
                _fail(call, "too_long", f"The resulting role-policy texts total {total} characters; the limit is 16000. This change was not applied.")
        return result

    def _acl(self, actor, call, kind, allowed):
        if not allowed:
            _fail(call, "not_authorized", f"{kind} is not available to {'master' if actor == MASTER else 'worker'} in mode {self.mode}.")

    def _execute(self, state, changes, actor, snapshot, call, source_request):
        at = _now(self._clock)
        common = {"actor": actor, "source_request": source_request, "tool_call_id": call.tool_call_id, "at": at}
        if isinstance(call, SpawnCall):
            self._spawn(state, changes, actor, call, common)
            return
        if isinstance(call, UpdateCall):
            kind = "GLOBAL-POLICY" if call.target == "C" else "ROLE-POLICY"
            self._acl(actor, call, kind, self.mode == "propose" and actor == MASTER)
            self._target(state, call, call.target)
            if call.to != "*":
                _fail(call, "not_authorized", f'Send this shared policy update with to="*". Its target is {call.target}, independent of notification recipients.')
            self._base(state, snapshot, call, call.target, call.base)
            text = self._resulting_text(state, call, call.target, call.operation, call.text)
            self._update(state, changes, call.target, text, common, "master_direct", operation=call.operation)
            return
        if isinstance(call, ProposalCall):
            self._acl(actor, call, "POLICY-PROPOSAL", self.mode == "propose" and actor != MASTER)
            binding = state["bindings"].get(actor)
            if not binding:
                raise RuntimeError("Proposal actor has no role binding")
            if (binding["lifecycle"] != "active" or call.to != "leader"
                    or call.target not in {"C", f"role:{binding['role_id']}"}):
                _fail(call, "not_authorized", f"An active worker may propose C or its own role type {binding['role_id']}, addressed to leader.")
            call = _validated_body(call)
            base = self._base(state, snapshot, call, call.target, call.base)
            self._resulting_text(state, call, call.target, call.operation, call.change)
            old = next((p for p in state["proposals"].values() if p["status"] == "pending"
                        and (p["proposer"], p["target"]) == (actor, call.target)), None)
            if old and all(old[k] == getattr(call, k) for k in ("operation", "change", "reason", "evidence")) and old["base_version"] == base:
                self._emit(state, changes, {"event": "role_policy_proposal_renotified", "proposal_id": old["proposal_id"], **common})
                return
            pid = state["next_proposal"]
            if old:
                self._void(state, changes, old["proposal_id"], "replaced", at, replaced_by=pid)
            value = asdict(ProposalView(pid, call.target, call.operation, actor, binding["role_id"], base,
                                       call.change, call.reason, call.evidence, at, source_request, call.tool_call_id))
            self._emit(state, changes, {"event": "role_policy_proposed", "proposal": value, **common})
            pending = render_pending(ProposalView(**p) for p in state["proposals"].values() if p["status"] == "pending")
            if len(pending) > MAX_PENDING_CHARS:
                _fail(call, "pending_display_full", f"Too much proposal text is already waiting for the master's decision ({len(pending)} of 8000 characters). Your proposal was not submitted and any earlier one stays pending; shorten it or resubmit after the master decides.")
            return
        if isinstance(call, DecisionCall):
            self._acl(actor, call, "POLICY-DECISION", self.mode == "propose" and actor == MASTER)
            proposal = state["proposals"].get(call.proposal_id)
            if proposal is None:
                _fail(call, "unknown_proposal", f"There is no proposal #{call.proposal_id}.")
            if call.proposal_id not in snapshot.pending_ids or proposal["status"] != "pending":
                _fail(call, "stale_proposal", f"Proposal #{call.proposal_id} is no longer pending (replaced, superseded by a policy update, or its proposer ended). Decide only proposals in your current pending list; a decision never transfers to another ID.")
            if call.to != proposal["proposer"]:
                _fail(call, "not_authorized", f"Send the decision on #{call.proposal_id} to its proposer {proposal['proposer']}.")
            target = proposal["target"]
            call = _validated_body(call, target=target)
            self._base(state, snapshot, call, target, proposal["base_version"])
            version = None
            if call.decision != "reject":
                amended = call.decision == "amend"
                body = call.amended_text if amended else proposal["change"]
                if body is None or amended and _CANDIDATE.match(body.partition("\n")[0]):
                    _fail(call, "malformed_body", f"Amend supplies the complete replacement policy for {target} below the header, without another control header. A role amendment may be empty to clear its policy; C must remain nonempty.")
                text = self._resulting_text(state, call, target, "replace" if amended else proposal["operation"], body, amended=amended)
                self._update(state, changes, target, text, common, "master_amend" if amended else "subagent_proposal",
                             proposal_id=call.proposal_id, operation="replace" if amended else proposal["operation"],
                             proposer_role_id=proposal["proposer_role_id"])
                version = state["targets"][target][-1]["version"]
            self._emit(state, changes, {"event": "role_policy_decided", "proposal_id": call.proposal_id,
                                       "decision": call.decision, "note": call.note, "decided_version": version, **common})
            return
        raise TypeError("Unknown package control call")

    def _spawn(self, state, changes, actor, call, common):
        if actor != MASTER:
            raise RuntimeError("Only the master can dispatch a teammate")
        active = sorted(r for r, value in state["package"]["roles"].items() if not value["retired"])
        if call.role_id not in active:
            _fail(call, "invalid_role_id", f"That role type is not in the role directory. Use one of: {', '.join(active)}.")
        previous = state["bindings"].get(call.name)
        if previous and previous["lifecycle"] != "voided":
            holder = state["slots"].get(previous["slot_id"]) or previous["instance"]
            _fail(call, "duplicate_name", f"{call.name} is already in use by {holder}; choose a new name.")
        predecessor = None
        if call.succeeds:
            predecessor = state["bindings"].get(call.succeeds)
            if predecessor is None:
                _fail(call, "unknown_slot", f"No teammate {call.succeeds} to succeed.")
            if predecessor["lifecycle"] != "terminal":
                _fail(call, "predecessor_not_terminal", f"{call.succeeds} has not ended yet. Omit SUCCEEDS: a new teammate with the same ROLE-TYPE receives the current policy without it.")
            if predecessor["role_id"] != call.role_id:
                _fail(call, "role_type_mismatch", f"The resolved role type {call.role_id} does not match predecessor {call.succeeds}'s type {predecessor['role_id']}. Specify the matching ROLE-TYPE or omit SUCCEEDS to create a new slot.")
            holder = state["slots"][predecessor["slot_id"]]
            if holder != call.succeeds:
                _fail(call, "stale_succession", f"That position is already held by {holder}. Omit SUCCEEDS and use a new name.")
        binding = asdict(BindingView(call.name, call.role_id, predecessor["slot_id"] if predecessor else call.name,
                                    "planned", call.binding_source, call.succeeds, state["next_spawn"], False))
        self._emit(state, changes, {"event": "role_policy_planned", "binding": binding, **common})

    def _update(self, state, changes, target, text, common, provenance, *, proposal_id=None,
                operation="replace", proposer_role_id=None):
        previous = state["targets"][target][-1]
        value = {"version": previous["version"] + 1, "text": text, "provenance": provenance,
                 "actor": common["actor"], "committed_at": common["at"], "proposal_id": proposal_id,
                 "source_request": common["source_request"], "tool_call_id": common["tool_call_id"]}
        self._emit(state, changes, {"event": "role_policy_committed", "target": target,
                                   "before_text": previous["text"], "before_version": previous["version"],
                                   "value": value, "operation": operation, "proposer_role_id": proposer_role_id,
                                   **common})
        for p in list(state["proposals"].values()):
            if p["status"] == "pending" and p["target"] == target and p["proposal_id"] != proposal_id:
                self._void(state, changes, p["proposal_id"], "policy_superseded", common["at"])

    def _void(self, state, changes, pid, reason, at, replaced_by=None):
        self._emit(state, changes, {"event": "role_policy_proposal_voided", "proposal_id": pid,
                                   "reason": reason, "replaced_by": replaced_by, "at": at})

    def record_first_request(self, instance):
        with self._lock:
            binding = self._state["bindings"].get(instance)
            if binding is None or binding["lifecycle"] == "voided":
                raise RuntimeError("Worker request has no valid package binding")
            if binding["lifecycle"] != "planned":
                return binding["lifecycle"]
            self._lifecycle(instance, [{"event": "role_policy_bound", "instance": instance, "at": _now(self._clock)}])
            return "bound"

    def record_spawn_result(self, instance, *, ok, tool_call_id=None):
        with self._lock:
            binding = self._state["bindings"].get(instance)
            if binding is None:
                return "unknown"
            if binding["lifecycle"] != "planned":
                return "ok" if ok and binding["confirmed"] else "conflict"
            if not ok and binding["confirmed"]:
                return "conflict"
            if ok:
                self._lifecycle(instance, [{"event": "role_policy_spawn_confirmed", "instance": instance,
                                           "source": tool_call_id, "at": _now(self._clock)}])
                return "ok"
            self._lifecycle(instance, [{"event": "role_policy_binding_ended", "instance": instance,
                                       "reason": "spawn_failed", "source": tool_call_id, "at": _now(self._clock)}])
            return "voided"

    def record_terminal(self, instance, *, source=None):
        with self._lock:
            binding = self._state["bindings"].get(instance)
            if binding is None:
                return "unknown"
            if binding["lifecycle"] not in {"planned", "active"}:
                return binding["lifecycle"]
            at, changes, state = _now(self._clock), [], copy.deepcopy(self._state)
            for p in list(state["proposals"].values()):
                if p["status"] == "pending" and p["proposer"] == instance:
                    self._void(state, changes, p["proposal_id"], "proposer_terminal", at)
            changes.append({"event": "role_policy_binding_ended", "instance": instance, "reason": "terminal", "source": source, "at": at})
            self._lifecycle(instance, changes)
            return "terminal"

    mark_terminal = record_terminal

    def end_run(self):
        with self._lock:
            if self._state["closed"]:
                return 0
            at, changes, state = _now(self._clock), [], copy.deepcopy(self._state)
            pending = [p for p in state["proposals"].values() if p["status"] == "pending"]
            for p in pending:
                self._void(state, changes, p["proposal_id"], "run_ended", at)
            changes.extend({"event": "role_policy_binding_ended", "instance": name, "reason": "never_started", "source": "run_end", "at": at}
                           for name, binding in state["bindings"].items() if binding["lifecycle"] == "planned" and not binding["confirmed"])
            changes.append({"event": "role_policy_run_ended", "at": at})
            self._lifecycle("runtime", changes)
            return len(pending)

    def _lifecycle(self, actor, changes):
        state = copy.deepcopy(self._state)
        for change in changes:
            self._apply(state, change)
        self._persist("lifecycle", actor, None, changes)
        self._state = state

    def _persist(self, kind, actor, source_request, changes):
        record = {"event": BATCH_EVENT, "schema": 1, "batch": self._next_batch, "kind": kind,
                  "actor": actor, "source_request": source_request, "changes": copy.deepcopy(changes)}
        try:
            self._writer(record)
        except Exception as exc:
            raise PolicyPersistError(f"Role-package batch {self._next_batch} was not persisted") from exc
        self._next_batch += 1
        return record["batch"]

    @staticmethod
    def _apply(state, change):
        kind = change["event"]
        if state.get("closed"):
            raise ReplayError("Event after package run end")
        if kind == "role_package_initialized":
            if state:
                raise ReplayError("Duplicate package initialization")
            package = validate_role_package(change["package"])
            if change["mode"] not in MODES:
                raise ReplayError("Unsupported package mode")

            def initial(text):
                return [{"version": 0, "text": text, "provenance": "initial", "actor": "runtime",
                         "committed_at": change["at"], "proposal_id": None, "source_request": None,
                         "tool_call_id": None}]
            state.update(mode=change["mode"], package=package, targets={"C": initial(package["team"]["communication"])},
                         bindings={}, slots={}, proposals={}, next_proposal=1, next_spawn=1, closed=False)
            state["targets"].update({f"role:{r}": initial(v["policy"]) for r, v in package["roles"].items()})
        elif kind == "role_policy_planned":
            b = copy.deepcopy(change["binding"])
            old = state["bindings"].get(b["instance"])
            if old and old["lifecycle"] != "voided" or b["spawn_order"] != state["next_spawn"]:
                raise ReplayError("Invalid package binding order")
            if b["predecessor"] and state["slots"].get(b["slot_id"]) != b["predecessor"]:
                raise ReplayError("Invalid package succession")
            state["bindings"][b["instance"]] = b
            state["slots"][b["slot_id"]] = b["instance"]
            state["next_spawn"] += 1
        elif kind in {"role_policy_bound", "role_policy_spawn_confirmed"}:
            b = state["bindings"][change["instance"]]
            if b["lifecycle"] not in {"planned", "active"}:
                raise ReplayError("Confirming a finished binding")
            b["confirmed"] = True
            if kind == "role_policy_bound":
                b["lifecycle"] = "active"
        elif kind == "role_policy_binding_ended":
            b = state["bindings"][change["instance"]]
            if b["lifecycle"] not in {"planned", "active"}:
                raise ReplayError("Ending a finished binding")
            b["lifecycle"] = "terminal" if change["reason"] == "terminal" else "voided"
            if b["lifecycle"] == "terminal":
                b["confirmed"] = True
            if b["lifecycle"] == "voided":
                state["slots"][b["slot_id"]] = b["predecessor"]
        elif kind == "role_policy_committed":
            versions = state["targets"][change["target"]]
            if versions[-1]["version"] != change["before_version"] or versions[-1]["text"] != change["before_text"] or change["value"]["version"] != change["before_version"] + 1:
                raise ReplayError("Package policy version/text mismatch")
            versions.append(copy.deepcopy(change["value"]))
        elif kind == "role_policy_proposed":
            p = copy.deepcopy(change["proposal"])
            if p["proposal_id"] != state["next_proposal"]:
                raise ReplayError("Package proposal sequence mismatch")
            state["proposals"][p["proposal_id"]] = p
            state["next_proposal"] += 1
        elif kind == "role_policy_proposal_renotified":
            if state["proposals"][change["proposal_id"]]["status"] != "pending":
                raise ReplayError("Renotifying a finished proposal")
        elif kind in {"role_policy_proposal_voided", "role_policy_decided"}:
            p = state["proposals"][change["proposal_id"]]
            if p["status"] != "pending":
                raise ReplayError("Finishing a nonpending proposal")
            if kind == "role_policy_proposal_voided":
                p.update(status="voided", void_reason=change["reason"], replaced_by=change["replaced_by"])
            else:
                p.update(status={"approve": "approved", "reject": "rejected", "amend": "amended"}[change["decision"]],
                         note=change["note"], decided_version=change["decided_version"])
        elif kind == "role_policy_run_ended":
            if any(p["status"] == "pending" for p in state["proposals"].values()):
                raise ReplayError("Run ended with pending proposals")
            state["closed"] = True
        else:
            raise ReplayError("Unknown package event")

    @classmethod
    def replay(cls, records, writer, *, clock=time.time):
        batches = [r for r in records if r.get("event") == BATCH_EVENT]
        if not batches:
            raise ReplayError("No role-package batches")
        state = {}
        for expected, batch in enumerate(batches, 1):
            if batch.get("schema") != 1 or batch.get("batch") != expected or not batch.get("changes"):
                raise ReplayError("Missing or invalid role-package batch")
            try:
                for change in batch["changes"]:
                    cls._apply(state, change)
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayError("Invalid role-package event") from exc
        store = cls.__new__(cls)
        store._state, store._next_batch = state, len(batches) + 1
        store._writer, store._clock, store._lock = writer, clock, threading.RLock()
        return store
