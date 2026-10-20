"""Run-local per-teammate policy state for agent_rules_v1 (plan v4 sections 4-5).

Same transaction discipline as RolePackageStore: every response's controls are
simulated on private copies (per-call rejection), journaled as one batch through
the relay's event writer, then published under a lock; replay rebuilds the state.
Only teammates have policies; the master follows M. The store is authoritative:
native delivery of a control message never decides whether a change happened.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time
from typing import Callable, Iterable

from agent_company.policy import agent_rules as rules
from agent_company.policy import agent_rules_text as text
from agent_company.policy.agent_rules import (BoardCall, ControlRejection, DecisionCall, EditCall, LessonCall,
                                              ProposalCall, RuleError, SpawnCall)
from agent_company.policy.role_package import AGENT_RULES_PROTOCOL, validate_agent_rules_package
from agent_company.policy.role_policy import PolicyPersistError, ReplayError

BATCH_EVENT = "agent_policy_batch"
MASTER = "master"
ENDED = frozenset({"terminal", "voided"})
RECENT_SELF_CHANGES = 10
RECENT_OWN_PROPOSALS = 5
# Latest board posts by others shown at the end of a teammate's requests (plan v4 8.1).
BOARD_TAIL_POSTS = 5
LESSON_TRIGGERS = frozenset({"shutdown", "task_done", "voluntary"})


def _now(clock):
    return datetime.fromtimestamp(clock(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rule_key(rule_id: str) -> int:
    return int(rule_id[1:])


def _fail(call, code, detail, **details):
    raise ControlRejection(code, detail, tool_call_id=call if isinstance(call, str) else call.tool_call_id, **details)


def _error(code, **values):
    return text.ERRORS[code].format(**values)


@dataclass(frozen=True)
class PolicyVersion:
    version: int
    rules: tuple[tuple[str, str], ...]
    notes: str
    source: str
    actor: str
    at: str
    proposal_id: int | None = None
    lineage: dict | None = None

    @classmethod
    def of(cls, value: dict) -> "PolicyVersion":
        ordered = tuple(sorted(value["rules"].items(), key=lambda item: _rule_key(item[0])))
        return cls(value["version"], ordered, value["notes"], value["source"], value["actor"], value["at"],
                   value.get("proposal_id"), copy.deepcopy(value.get("lineage")))


@dataclass(frozen=True)
class WorkerView:
    name: str
    lifecycle: str
    confirmed: bool
    spawn_order: int
    predecessor: str | None
    policy: PolicyVersion

    @property
    def ended(self):
        return self.lifecycle in ENDED


@dataclass(frozen=True)
class ProposalView:
    proposal_id: int
    proposer: str
    target: str
    operation: str
    rule_id: str | None
    body: str
    reason: str
    base: dict
    created_at: str
    status: str
    note: str | None = None
    applied: dict | None = None
    void_reason: str | None = None

    @property
    def label(self):
        return self.operation + (f" {self.rule_id}" if self.rule_id else "")


@dataclass(frozen=True)
class BoardPost:
    post_id: str
    author: str
    to: str
    at: str
    text: str

    @property
    def number(self):
        return int(self.post_id[1:])


@dataclass(frozen=True)
class AgentSnapshot:
    actor: str
    workers: tuple[WorkerView, ...]
    proposals: tuple[ProposalView, ...]
    lesson_count: int
    board_count: int = 0
    # Teammates: the latest posts by others, oldest first. The master: none.
    board_recent: tuple[BoardPost, ...] = ()

    mode = "propose"

    def worker(self, name):
        return next((w for w in self.workers if w.name == name), None)

    def version_of(self, name):
        view = self.worker(name)
        return view.policy.version if view else None

    @property
    def own(self):
        view = self.worker(self.actor)
        return view.policy if view else None

    @property
    def pending_ids(self):
        return frozenset(p.proposal_id for p in self.proposals if p.status == "pending")


@dataclass(frozen=True)
class StagedBatch:
    actor: str
    snapshot: AgentSnapshot
    calls: tuple
    source_request: int | None
    preview: tuple[dict, ...]
    rejections: tuple[ControlRejection, ...]
    receipts: dict
    lesson_trigger: str = "voluntary"

    @property
    def accepted_call_ids(self):
        return tuple(c.tool_call_id for c in self.calls)


@dataclass(frozen=True)
class CommitResult:
    batch: int | None
    changes: tuple[dict, ...]
    rejections: tuple[ControlRejection, ...]
    accepted_call_ids: tuple[str, ...]
    receipts: dict


def _not_ended(state):
    return [name for name, w in state["workers"].items() if w["lifecycle"] not in ENDED]


def apply_operation(current: dict, operation: str, rule_id: str | None, lines: list[str], notes: str | None,
                    *, call, target: str) -> tuple[dict, int, str, dict]:
    """New (rules, next_rule, notes) plus the diff a later revoke reverses."""
    current_rules, next_rule, current_notes = dict(current["rules"]), current["next_rule"], current["notes"]
    if operation == "add":
        added = {}
        for line in lines:
            added[f"R{next_rule}"] = line
            next_rule += 1
        current_rules.update(added)
        return current_rules, next_rule, current_notes, {"added": added}
    if operation in ("replace", "remove"):
        if rule_id not in current_rules:
            _fail(call, "unknown_rule", _error("unknown_rule", target=target, value=rule_id))
        old = current_rules[rule_id]
        if operation == "remove":
            del current_rules[rule_id]
            return current_rules, next_rule, current_notes, {"removed": {rule_id: old}}
        current_rules[rule_id] = lines[0]
        return current_rules, next_rule, current_notes, {"replaced": {rule_id: [old, lines[0]]}}
    if operation == "notes":
        return current_rules, next_rule, notes or "", {"notes": [current_notes, notes or ""]}
    raise ValueError("Unknown policy operation")


def diff_slots(diff) -> set:
    """The rule ids a version's diff changed, plus "notes" when it changed the notes."""
    diff = diff or {}
    slots = {rule_id for key in ("added", "replaced", "removed") for rule_id in diff.get(key) or {}}
    return slots | ({"notes"} if "notes" in diff else set())


def touched_after(worker: dict, version: int, slots: set) -> list[dict]:
    """Versions after `version` that changed any of these slots, oldest first."""
    return [v for v in worker["versions"] if v["version"] > version and diff_slots(v.get("diff")) & slots]


def revert_diff(current: dict, diff: dict, *, call, target, proposal_id) -> tuple[dict, str, dict]:
    """Compensating change for an applied proposal, only if nothing touched it since."""
    current_rules, current_notes = dict(current["rules"]), current["notes"]

    def conflict():
        _fail(call, "revoke_conflict", _error("revoke_conflict", target=target, value=proposal_id))
    if "added" in diff:
        if any(current_rules.get(k) != v for k, v in diff["added"].items()):
            conflict()
        for k in diff["added"]:
            del current_rules[k]
        return current_rules, current_notes, {"removed": dict(diff["added"])}
    if "replaced" in diff:
        (rule_id, (old, new)), = diff["replaced"].items()
        if current_rules.get(rule_id) != new:
            conflict()
        current_rules[rule_id] = old
        return current_rules, current_notes, {"replaced": {rule_id: [new, old]}}
    if "removed" in diff:
        if any(k in current_rules for k in diff["removed"]):
            conflict()
        current_rules.update(diff["removed"])
        return current_rules, current_notes, {"added": dict(diff["removed"])}
    if "notes" in diff:
        old, new = diff["notes"]
        if current_notes != new:
            conflict()
        return current_rules, old, {"notes": [new, old]}
    raise ValueError("Unknown policy diff")


class AgentPolicyStore:
    protocol = AGENT_RULES_PROTOCOL
    mode = "propose"

    def __init__(self, policy_package: dict, writer: Callable[[dict], None], *, clock=time.time):
        package = validate_agent_rules_package(policy_package)
        self._writer, self._clock, self._lock = writer, clock, threading.RLock()
        self._next_batch = 1
        change = {"event": "agent_policy_initialized", "package": package, "at": _now(clock)}
        state = {}
        self._apply(state, change)
        self._persist("init", "runtime", None, [change])
        self._state = state

    # -- reading ----------------------------------------------------------------
    def snapshot(self, actor: str) -> AgentSnapshot:
        with self._lock:
            s = self._state
            if actor != MASTER:
                worker = s["workers"].get(actor)
                if worker is None:
                    raise RuntimeError("Worker requested a policy snapshot without a committed binding")
                if worker["lifecycle"] == "voided":
                    raise RuntimeError("Voided worker requested a policy snapshot")
            workers = tuple(WorkerView(w["name"], w["lifecycle"], w["confirmed"], w["spawn_order"],
                                       w["predecessor"], PolicyVersion.of(w["versions"][-1]))
                            for w in sorted(s["workers"].values(), key=lambda w: w["spawn_order"]))
            values = sorted(s["proposals"].values(), key=lambda p: p["proposal_id"])
            if actor == MASTER:
                pending = [p for p in values if p["status"] == "pending"]
                recent = [p for p in values if p["status"] == "applied"][-RECENT_SELF_CHANGES:]
                chosen = pending + recent
            else:
                own = [p for p in values if p["proposer"] == actor]
                chosen = ([p for p in own if p["status"] == "pending"]
                          + [p for p in own if p["status"] != "pending"][-RECENT_OWN_PROPOSALS:])
            proposals = tuple(self._proposal_view(p) for p in sorted(chosen, key=lambda p: p["proposal_id"]))
            others = [] if actor == MASTER else [p for p in s["board"] if p["author"] != actor][-BOARD_TAIL_POSTS:]
            recent = tuple(BoardPost(p["id"], p["author"], p["to"], p["at"], p["text"]) for p in others)
            return AgentSnapshot(actor, workers, proposals, len(s["lessons"]), len(s["board"]), recent)

    request_snapshot = snapshot

    def lessons(self):
        """Every recorded lesson, oldest first (for the spawn gate)."""
        with self._lock:
            return copy.deepcopy(self._state["lessons"])

    def policy_of(self, name):
        """(version, ((rule_id, text), ...)) for a teammate that has not ended, else None."""
        with self._lock:
            worker = self._state["workers"].get(name)
            if worker is None or worker["lifecycle"] in ENDED:
                return None
            current = worker["versions"][-1]
            return current["version"], tuple(sorted(current["rules"].items(), key=lambda item: _rule_key(item[0])))

    @staticmethod
    def _proposal_view(p):
        return ProposalView(p["proposal_id"], p["proposer"], p["target"], p["operation"], p["rule_id"],
                            p["body"], p["reason"], copy.deepcopy(p["base"]), p["created_at"], p["status"],
                            p.get("note"), copy.deepcopy(p.get("applied")), p.get("void_reason"))

    def export(self):
        with self._lock:
            return copy.deepcopy(self._state)

    export_state = export

    # -- staging and commit -------------------------------------------------------
    def stage(self, actor, snapshot, calls: Iterable, *, source_request=None, lesson_trigger="voluntary"):
        """``lesson_trigger``: the reflection prompt this response may answer (plan v4 section 8)."""
        if snapshot.actor != actor:
            raise ValueError("Policy snapshot does not belong to this actor")
        if lesson_trigger not in LESSON_TRIGGERS:
            raise ValueError("Unknown lesson trigger")
        calls = tuple(calls)
        with self._lock:
            state = copy.deepcopy(self._state)
        _, changes, accepted, errors, receipts = self._simulate(state, actor, snapshot, calls, source_request,
                                                                lesson_trigger)
        return StagedBatch(actor, snapshot, tuple(accepted), source_request, tuple(changes), tuple(errors), receipts,
                           lesson_trigger)

    def commit(self, staged: StagedBatch, *, check=None):
        with self._lock:
            state, changes, accepted, errors, receipts = self._simulate(
                copy.deepcopy(self._state), staged.actor, staged.snapshot, staged.calls, staged.source_request,
                staged.lesson_trigger)
            if check is not None:
                check()
            batch = self._persist("response", staged.actor, staged.source_request, changes) if changes else None
            if changes:
                self._state = state
            return CommitResult(batch, tuple(copy.deepcopy(changes)), staged.rejections + tuple(errors),
                                tuple(c.tool_call_id for c in accepted), receipts)

    def _simulate(self, state, actor, snapshot, calls, source_request, lesson_trigger="voluntary"):
        changes, accepted, errors, receipts, seen = [], [], [], {}, set()
        # A rule may name a teammate spawned anywhere in the same response.
        batch_names = frozenset(c.name for c in calls if isinstance(c, SpawnCall))
        # Versions this actor's earlier calls in the same response produced count as
        # already seen (M1 review MJ1); concurrent changes by others still conflict.
        produced: dict[str, int] = {}
        for call in calls:
            candidate, own = copy.deepcopy(state), []
            try:
                if call.tool_call_id in seen:
                    _fail(call, "duplicate_tool_call", _error("duplicate_tool_call"))
                seen.add(call.tool_call_id)
                if state["closed"]:
                    _fail(call, "run_ended", _error("run_ended"))
                receipt = self._execute(candidate, own, actor, snapshot, call, source_request, batch_names, produced,
                                        lesson_trigger)
            except ControlRejection as exc:
                errors.append(exc)
                continue
            state = candidate
            changes.extend(own)
            accepted.append(call)
            for change in own:
                if change["event"] == "agent_policy_committed":
                    produced[change["worker"]] = change["value"]["version"]
                elif change["event"] == "agent_policy_planned":
                    produced[change["worker"]["name"]] = change["initial"]["version"]
            if receipt:
                receipts[call.tool_call_id] = receipt
        return state, changes, accepted, errors, receipts

    def _emit(self, state, changes, change):
        self._apply(state, change)
        changes.append(change)

    def _known(self, state, batch_names):
        return frozenset(_not_ended(state)) | batch_names

    def _validated_lines(self, call, lines, *, owner, known):
        out = []
        for line in lines:
            try:
                rule = rules.parse_rule(line)
                rules.check_roster(rule, owner=owner, known=known)
            except RuleError as exc:
                _fail(call, exc.code, f"{exc.detail} (rule: {line})")
            out.append(rule.text)
        return out

    def _commit_version(self, state, changes, name, *, rules_map, next_rule, notes, source, common,
                        operation, diff, proposal_id=None, revokes=None):
        worker = state["workers"][name]
        previous = worker["versions"][-1]
        value = {"version": previous["version"] + 1, "rules": rules_map, "next_rule": next_rule, "notes": notes,
                 "source": source, "actor": common["actor"], "at": common["at"], "proposal_id": proposal_id,
                 "lineage": None, "operation": operation, "revokes": revokes, "diff": diff,
                 "source_request": common["source_request"], "tool_call_id": common["tool_call_id"]}
        self._emit(state, changes, {"event": "agent_policy_committed", "worker": name,
                                    "before_version": previous["version"], "diff": diff, "value": value, **common})
        return value["version"]

    def _execute(self, state, changes, actor, snapshot, call, source_request, batch_names, produced,
                 lesson_trigger="voluntary"):
        at = _now(self._clock)
        common = {"actor": actor, "source_request": source_request, "tool_call_id": call.tool_call_id, "at": at}
        if isinstance(call, SpawnCall):
            return self._spawn(state, changes, actor, call, common, batch_names)
        if isinstance(call, EditCall):
            return self._edit(state, changes, actor, snapshot, call, common, batch_names, produced)
        if isinstance(call, ProposalCall):
            return self._propose(state, changes, actor, snapshot, call, common, batch_names, produced)
        if isinstance(call, DecisionCall):
            return self._decide(state, changes, actor, call, common, batch_names)
        if isinstance(call, LessonCall):
            return self._lesson(state, changes, actor, call, common, lesson_trigger)
        if isinstance(call, BoardCall):
            return self._board(state, changes, actor, call, common)
        raise TypeError("Unknown policy control call")

    # -- spawn ----------------------------------------------------------------------
    def _spawn(self, state, changes, actor, call, common, batch_names):
        if actor != MASTER:
            raise RuntimeError("Only the master can dispatch a teammate")
        if call.name.lower() in rules.RESERVED_NAMES:
            _fail(call, "invalid_name", text.ACL["reserved_name"].format(name=call.name))
        previous = state["workers"].get(call.name)
        if previous and previous["lifecycle"] != "voided":
            _fail(call, "duplicate_name", _error("duplicate_name", value=call.name))
        parsed, warnings = call.policy, []
        for code, values in parsed.warnings:
            warnings.append({"code": code, "detail": text.SPAWN_PROBLEMS[code].format(**values)})
        base = {"version": 0, "rules": {}, "next_rule": 1, "notes": "", "source": "initial", "lineage": None}
        predecessor = None
        if parsed.succeeds:
            pred = state["workers"].get(parsed.succeeds)
            if pred is None or pred["lifecycle"] == "voided":
                warnings.append({"code": "unknown_predecessor",
                                 "detail": text.SPAWN_PROBLEMS["unknown_predecessor"].format(name=parsed.succeeds)})
            elif pred["lifecycle"] != "terminal":
                warnings.append({"code": "predecessor_active",
                                 "detail": text.SPAWN_PROBLEMS["predecessor_active"].format(name=parsed.succeeds)})
            else:
                last = pred["versions"][-1]
                predecessor = parsed.succeeds
                base = {"version": last["version"] + 1, "rules": dict(last["rules"]), "next_rule": last["next_rule"],
                        "notes": last["notes"], "source": "inherit",
                        "lineage": {"from": predecessor, "version": last["version"]}}
        known = self._known(state, batch_names)
        skipped = []
        for rule_id, line in list(base["rules"].items()):
            try:
                rules.check_roster(rules.parse_rule(line), owner=call.name, known=known)
            except RuleError as exc:
                del base["rules"][rule_id]
                skipped.append({"line": f"{rule_id}: {line}", "code": exc.code, "detail": exc.detail, "inherited": True})
        for line in parsed.rules:
            try:
                rule = rules.parse_rule(line)
                rules.check_roster(rule, owner=call.name, known=known)
            except RuleError as exc:
                skipped.append({"line": line, "code": exc.code, "detail": exc.detail})
                continue
            base["rules"][f"R{base['next_rule']}"] = rule.text
            base["next_rule"] += 1
        if parsed.notes is not None:
            base["notes"] = parsed.notes
        initial = {**base, "actor": actor, "at": common["at"], "proposal_id": None,
                   "operation": "spawn", "revokes": None,
                   "source_request": common["source_request"], "tool_call_id": common["tool_call_id"]}
        # No usable block and no inheritance: the master is reminded to write one.
        missing = predecessor is None and parsed.block != "ok"
        worker = {"name": call.name, "spawn_order": state["next_spawn"], "predecessor": predecessor}
        self._emit(state, changes, {"event": "agent_policy_planned", "worker": worker, "initial": initial,
                                    "block": parsed.block, "skipped": skipped, "warnings": warnings,
                                    "policy_missing": missing, **common})
        lines = [text.SPAWN_RECEIPT.format(
            name=call.name, version=initial["version"], rules=len(initial["rules"]),
            inherited=text.SPAWN_RECEIPT_INHERITED.format(predecessor=predecessor, from_version=initial["version"] - 1)
            if predecessor else "")]
        lines += [text.SPAWN_SKIPPED.format(line=s["line"], detail=s["detail"]) for s in skipped]
        lines += [text.SPAWN_WARNING.format(detail=w["detail"]) for w in warnings]
        return "\n".join(lines)

    # -- master edits ---------------------------------------------------------------
    def _targets(self, state, call, names, *, per_target=False):
        if names == ("*",):
            return sorted(_not_ended(state), key=lambda n: state["workers"][n]["spawn_order"])
        if per_target and len(names) > 1:
            return list(names)  # each name is checked in _edit_one (plan v4 5.2: per-target results)
        for name in names:
            worker = state["workers"].get(name)
            if worker is None or worker["lifecycle"] in ENDED:
                _fail(call, "unknown_target", _error("unknown_target", value=name))
        return list(names)

    def _edit(self, state, changes, actor, snapshot, call, common, batch_names, produced):
        if actor != MASTER:
            _fail(call, "not_authorized", text.ACL["worker_agent_policy"])
        if call.base is not None and call.targets == ("*",) or call.base is not None and len(call.targets) != 1:
            _fail(call, "base_requires_one_target", _error("base_requires_one_target"))
        targets = self._targets(state, call, call.targets, per_target=True)
        if not targets:
            _fail(call, "unknown_target", _error("unknown_target", value="*"))
        lines, notes = rules.edit_body(call, call.operation, call.body)
        results, applied, failures = [], 0, []
        for name in targets:
            candidate, own = copy.deepcopy(state), []
            try:
                version = self._edit_one(candidate, own, snapshot, call, name, lines, notes, common, batch_names, produced)
            except ControlRejection as exc:
                failures.append(exc)
                results.append(text.TARGET_REJECTED.format(name=name, code=exc.code, detail=exc.detail))
                continue
            state.update(candidate)
            changes.extend(own)
            applied += 1
            results.append(text.TARGET_APPLIED.format(name=name, version=version))
        if not applied:
            # Nothing changed: reject the call so native never delivers a no-op notice.
            if len(failures) == 1:
                raise failures[0]
            _fail(call, "not_applied", "; ".join(results))
        return text.AGENT_POLICY_RECEIPT.format(results="; ".join(results))

    def _edit_one(self, state, changes, snapshot, call, name, lines, notes, common, batch_names, produced):
        worker = state["workers"].get(name)
        if worker is None or worker["lifecycle"] in ENDED:
            _fail(call, "unknown_target", _error("unknown_target", value=name))
        current = worker["versions"][-1]
        # An explicit base is always checked, also for a teammate spawned in this response.
        expected = call.base if call.base is not None else produced.get(name, snapshot.version_of(name))
        if expected is not None and expected != current["version"]:
            _fail(call, "stale_version", _error("stale_version", target=name, current=current["version"],
                                                  shown=f"{expected}" + (" (base)" if call.base is not None else "")))
        if call.operation == "revoke":
            return self._revoke(state, changes, call, name, common)
        valid = self._validated_lines(call, lines, owner=name, known=self._known(state, batch_names))
        new_rules, next_rule, new_notes, diff = apply_operation(current, call.operation, call.rule_id, valid, notes,
                                                                call=call, target=name)
        return self._commit_version(state, changes, name, rules_map=new_rules, next_rule=next_rule, notes=new_notes,
                                    source="master", common=common,
                                    operation=call.operation + (f" {call.rule_id}" if call.rule_id else ""), diff=diff)

    def _revoke(self, state, changes, call, name, common):
        proposal = state["proposals"].get(call.proposal_id)
        if proposal is None:
            _fail(call, "unknown_proposal", _error("unknown_proposal", value=call.proposal_id))
        applied = (proposal.get("applied") or {}).get(name)
        if applied is None or name in proposal.get("revoked", []):
            _fail(call, "revoke_not_applicable", _error("revoke_not_applicable", value=call.proposal_id, target=name))
        worker = state["workers"][name]
        for later in touched_after(worker, applied["version"], diff_slots(applied["diff"])):
            undone = state["proposals"].get(later.get("revokes") or later.get("proposal_id"))
            undone_applied = ((undone or {}).get("applied") or {}).get(name) or {}
            revoked_later = (later.get("revokes") is None and undone is not None and name in undone.get("revoked", [])
                             and undone_applied.get("version") == later["version"])
            revoke_of_later = later.get("revokes") is not None and undone_applied.get("version", -1) > applied["version"]
            if not (revoked_later or revoke_of_later):
                # Changed again after the revoked change, even if back to the same text (plan v4 5.2).
                _fail(call, "revoke_conflict", _error("revoke_conflict", target=name, value=call.proposal_id))
        current = worker["versions"][-1]
        new_rules, new_notes, diff = revert_diff(current, applied["diff"], call=call, target=name,
                                                 proposal_id=call.proposal_id)
        version = self._commit_version(state, changes, name, rules_map=new_rules, next_rule=current["next_rule"],
                                       notes=new_notes, source="revoke", common=common,
                                       operation=f"revoke #{call.proposal_id}", diff=diff,
                                       proposal_id=call.proposal_id, revokes=call.proposal_id)
        self._emit(state, changes, {"event": "agent_policy_revoked", "proposal_id": call.proposal_id,
                                    "worker": name, "version": version, **common})
        return version

    # -- worker proposals -------------------------------------------------------------
    def _propose(self, state, changes, actor, snapshot, call, common, batch_names, produced):
        if actor == MASTER:
            _fail(call, "not_authorized", text.ACL["master_proposal"])
        worker = state["workers"].get(actor)
        if worker is None or worker["lifecycle"] != "active":
            _fail(call, "not_authorized", text.ACL["inactive_proposer"])
        if call.to != rules.LEADER:
            _fail(call, "wrong_recipient", text.ACL["proposal_to_leader"])
        target = actor if call.target in ("self", actor) else call.target
        if target.lower() in rules.RESERVED_NAMES and target != "*":
            _fail(call, "not_authorized", text.ACL["proposal_for_master"])
        lines, notes = rules.edit_body(call, call.operation, call.body)
        pid = state["next_proposal"]
        value = {"proposal_id": pid, "proposer": actor, "target": "self" if target == actor else target,
                 "operation": call.operation, "rule_id": call.rule_id, "body": call.body, "reason": call.reason,
                 "base": {}, "base_text": {}, "created_at": common["at"], "source_request": common["source_request"],
                 "tool_call_id": call.tool_call_id, "status": "pending", "note": None, "applied": None,
                 "revoked": [], "void_reason": None, "replaced_by": None}
        if target == actor:
            current = worker["versions"][-1]
            shown = produced.get(actor, snapshot.own.version if snapshot.own else None)
            if shown != current["version"]:
                _fail(call, "stale_version", _error("stale_version", target=actor, current=current["version"], shown=shown))
            valid = self._validated_lines(call, lines, owner=actor, known=self._known(state, batch_names))
            new_rules, next_rule, new_notes, diff = apply_operation(current, call.operation, call.rule_id, valid,
                                                                    notes, call=call, target=actor)
            value.update(base={actor: current["version"]}, status="applied")
            self._emit(state, changes, {"event": "agent_policy_proposed", "proposal": value, **common})
            version = self._commit_version(state, changes, actor, rules_map=new_rules, next_rule=next_rule,
                                           notes=new_notes, source="self", common=common,
                                           operation=value["operation"] + (f" {call.rule_id}" if call.rule_id else ""),
                                           diff=diff, proposal_id=pid)
            self._emit(state, changes, {"event": "agent_policy_proposal_applied", "proposal_id": pid,
                                        "applied": {actor: {"version": version, "diff": diff}}, **common})
            return text.PROPOSAL_RECEIPT_APPLIED.format(id=pid, version=version)
        names = self._targets(state, call, ("*",) if target == "*" else (target,))
        for line in lines:
            try:
                rules.parse_rule(line)
            except RuleError as exc:
                _fail(call, exc.code, f"{exc.detail} (rule: {line})")
        for name in names:
            current = state["workers"][name]["versions"][-1]
            value["base"][name] = current["version"]
            if call.operation in ("replace", "remove"):
                if call.rule_id in current["rules"]:
                    value["base_text"][name] = current["rules"][call.rule_id]
                elif target != "*":
                    _fail(call, "unknown_rule", _error("unknown_rule", target=name, value=call.rule_id))
            elif call.operation == "notes":
                value["base_text"][name] = current["notes"]
        # A newer proposal replaces a pending one only when both change the same rule or
        # the notes; separate add proposals are independent (M1 review minor 2).
        key = (actor, value["target"], value["operation"], value["rule_id"])
        old = None if call.operation == "add" else next(
            (p for p in state["proposals"].values() if p["status"] == "pending"
             and (p["proposer"], p["target"], p["operation"], p["rule_id"]) == key), None)
        if old is not None:
            self._void(state, changes, old["proposal_id"], "replaced", common["at"], replaced_by=pid)
        self._emit(state, changes, {"event": "agent_policy_proposed", "proposal": value, **common})
        return text.PROPOSAL_RECEIPT_PENDING.format(id=pid)

    def _decide(self, state, changes, actor, call, common, batch_names):
        if actor != MASTER:
            _fail(call, "not_authorized", text.ACL["worker_decision"])
        proposal = state["proposals"].get(call.proposal_id)
        if proposal is None:
            _fail(call, "unknown_proposal", _error("unknown_proposal", value=call.proposal_id))
        if proposal["status"] != "pending":
            _fail(call, "stale_proposal", _error("stale_proposal", value=call.proposal_id))
        if call.to != proposal["proposer"]:
            _fail(call, "wrong_recipient", text.ACL["decision_to_proposer"].format(id=call.proposal_id,
                                                                                   proposer=proposal["proposer"]))
        operation, note, lines, notes = proposal["operation"], None, [], None
        body = call.body
        if call.decision in ("approve", "reject"):
            stripped = body.strip()
            if stripped:
                if "\n" in stripped or not stripped.startswith("NOTE:"):
                    _fail(call, "malformed_body", text.BODY_ERRORS["decision_body"])
                note = stripped[5:].strip()
        if call.decision == "reject":
            self._emit(state, changes, {"event": "agent_policy_decided", "proposal_id": call.proposal_id,
                                        "decision": "reject", "note": note, "applied": {}, **common})
            return text.DECISION_RECEIPT.format(id=call.proposal_id, results="rejected")
        if call.decision == "amend":
            if operation == "remove":
                _fail(call, "malformed_body", text.BODY_ERRORS["amend_remove"])
            content = body.splitlines()
            if operation != "notes" and content and content[-1].startswith("NOTE:"):
                note = content.pop()[5:].strip()
            source_body = "\n".join(content)
            if not source_body.strip() and operation != "notes":
                _fail(call, "malformed_body", text.BODY_ERRORS["amend_body"])
        else:
            source_body = proposal["body"]
        lines, notes = rules.edit_body(call, operation, source_body)
        target = proposal["target"]
        if target == "*":
            # Only teammates active when the proposal was submitted, as the master saw it.
            names = [n for n in proposal["base"] if state["workers"][n]["lifecycle"] not in ENDED]
            if not names:
                _fail(call, "unknown_target", _error("unknown_target", value="*"))
        else:
            names = self._targets(state, call, (proposal["proposer"] if target == "self" else target,))
        results, applied, failures = [], {}, []
        for name in names:
            candidate, own = copy.deepcopy(state), []
            try:
                current = candidate["workers"][name]["versions"][-1]
                if (call.decision == "amend" and target == "*" and operation == "replace"
                        and name not in proposal["base_text"]):
                    _fail(call, "proposal_target_changed", _error("proposal_target_changed"))
                if call.decision == "approve" and operation in ("replace", "remove", "notes"):
                    slot = "notes" if operation == "notes" else proposal["rule_id"]
                    base = proposal["base"].get(name)
                    if base is None or touched_after(candidate["workers"][name], base, {slot}):
                        _fail(call, "proposal_target_changed", _error("proposal_target_changed"))
                valid = self._validated_lines(call, lines, owner=name, known=self._known(candidate, batch_names))
                new_rules, next_rule, new_notes, diff = apply_operation(current, operation, proposal["rule_id"], valid,
                                                                        notes, call=call, target=name)
                version = self._commit_version(candidate, own, name, rules_map=new_rules, next_rule=next_rule,
                                               notes=new_notes, source="amend" if call.decision == "amend" else "approve",
                                               common=common, operation=operation + (f" {proposal['rule_id']}" if proposal["rule_id"] else ""),
                                               diff=diff, proposal_id=call.proposal_id)
            except ControlRejection as exc:
                failures.append(exc)
                results.append(text.TARGET_REJECTED.format(name=name, code=exc.code, detail=exc.detail))
                continue
            state.update(candidate)
            changes.extend(own)
            applied[name] = {"version": version, "diff": diff}
            results.append(text.TARGET_APPLIED.format(name=name, version=version))
        if not applied:
            if len(failures) == 1:
                raise failures[0]
            _fail(call, "not_applied", "; ".join(results))
        self._emit(state, changes, {"event": "agent_policy_decided", "proposal_id": call.proposal_id,
                                    "decision": call.decision, "note": note, "applied": applied,
                                    "amended_body": source_body if call.decision == "amend" else None, **common})
        return text.DECISION_RECEIPT.format(id=call.proposal_id, results="; ".join(results))

    # -- lessons ----------------------------------------------------------------------
    def _lesson(self, state, changes, actor, call, common, trigger="voluntary"):
        if actor == MASTER:
            raise RuntimeError("The relay treats a master LESSON: as an ordinary message")
        worker = state["workers"].get(actor)
        if worker is None or worker["lifecycle"] != "active":
            _fail(call, "not_authorized", text.ACL["inactive_proposer"])
        if call.to != rules.LEADER:
            _fail(call, "wrong_recipient", text.ACL["lesson_to_leader"])
        lesson = {"id": f"L{state['next_lesson']}", "author": actor, "at": common["at"], "text": call.text,
                  "trigger": trigger, "related": rules.related_facts(call.text, frozenset(state["workers"])),
                  "source_request": common["source_request"], "tool_call_id": call.tool_call_id}
        self._emit(state, changes, {"event": "agent_lesson_recorded", "lesson": lesson, **common})
        return text.LESSON_RECEIPT.format(id=lesson["id"])

    # -- board ------------------------------------------------------------------------
    def _board(self, state, changes, actor, call, common):
        if actor != MASTER:
            worker = state["workers"].get(actor)
            if worker is None or worker["lifecycle"] != "active":
                _fail(call, "not_authorized", text.ACL["inactive_poster"])
        # The post rides on an ordinary message: native must be able to deliver it (M3 review M1).
        to = call.to.strip() if isinstance(call.to, str) else ""
        live = {name.lower() for name in _not_ended(state)}
        deliverable = (to == rules.EVERYONE or to.lower() == rules.LEADER and actor != MASTER
                       or to.lower() in live and to.lower() != actor.lower())
        if not deliverable:
            _fail(call, "wrong_recipient", text.ACL["board_recipient"])
        post = {"id": f"B{state['next_board']}", "author": actor, "to": call.to, "at": common["at"],
                "text": call.text, "related": rules.related_facts(call.text, frozenset(state["workers"])),
                "source_request": common["source_request"], "tool_call_id": call.tool_call_id}
        self._emit(state, changes, {"event": "agent_board_posted", "post": post, **common})
        return text.BOARD_RECEIPT.format(id=post["id"])

    def _void(self, state, changes, pid, reason, at, replaced_by=None):
        self._emit(state, changes, {"event": "agent_policy_proposal_voided", "proposal_id": pid,
                                    "reason": reason, "replaced_by": replaced_by, "at": at})

    # -- lifecycle --------------------------------------------------------------------
    def record_first_request(self, instance):
        with self._lock:
            worker = self._state["workers"].get(instance)
            if worker is None or worker["lifecycle"] == "voided":
                raise RuntimeError("Worker request has no valid policy binding")
            if worker["lifecycle"] != "planned":
                return worker["lifecycle"]
            self._lifecycle(instance, [{"event": "agent_policy_bound", "instance": instance, "at": _now(self._clock)}])
            return "bound"

    def record_spawn_result(self, instance, *, ok, tool_call_id=None):
        with self._lock:
            worker = self._state["workers"].get(instance)
            if worker is None:
                return "unknown"
            if worker["lifecycle"] != "planned":
                return "ok" if ok and worker["confirmed"] else "conflict"
            if not ok and worker["confirmed"]:
                return "conflict"
            if ok:
                self._lifecycle(instance, [{"event": "agent_policy_spawn_confirmed", "instance": instance,
                                            "source": tool_call_id, "at": _now(self._clock)}])
                return "ok"
            self._lifecycle(instance, [{"event": "agent_policy_binding_ended", "instance": instance,
                                        "reason": "spawn_failed", "source": tool_call_id, "at": _now(self._clock)}])
            return "voided"

    def record_terminal(self, instance, *, source=None):
        with self._lock:
            worker = self._state["workers"].get(instance)
            if worker is None:
                return "unknown"
            if worker["lifecycle"] not in {"planned", "active"}:
                return worker["lifecycle"]
            at, changes, state = _now(self._clock), [], copy.deepcopy(self._state)
            for p in list(state["proposals"].values()):
                if p["status"] == "pending" and p["proposer"] == instance:
                    self._void(state, changes, p["proposal_id"], "proposer_terminal", at)
                elif p["status"] == "pending" and p["target"] == instance:
                    self._void(state, changes, p["proposal_id"], "target_terminal", at)
            changes.append({"event": "agent_policy_binding_ended", "instance": instance, "reason": "terminal",
                            "source": source, "at": at})
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
            changes.extend({"event": "agent_policy_binding_ended", "instance": name, "reason": "never_started",
                            "source": "run_end", "at": at}
                           for name, w in state["workers"].items() if w["lifecycle"] == "planned" and not w["confirmed"])
            changes.append({"event": "agent_policy_run_ended", "at": at})
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
            raise PolicyPersistError(f"Agent-policy batch {self._next_batch} was not persisted") from exc
        self._next_batch += 1
        return record["batch"]

    # -- replay -----------------------------------------------------------------------
    @staticmethod
    def _apply(state, change):
        kind = change["event"]
        if state.get("closed"):
            raise ReplayError("Event after policy run end")
        if kind == "agent_policy_initialized":
            if state:
                raise ReplayError("Duplicate policy initialization")
            state.update(package=validate_agent_rules_package(change["package"]), workers={}, proposals={},
                         lessons=[], next_proposal=1, next_spawn=1, next_lesson=1, closed=False,
                         board=[], next_board=1)
        elif kind == "agent_policy_planned":
            w = change["worker"]
            old = state["workers"].get(w["name"])
            if old and old["lifecycle"] != "voided" or w["spawn_order"] != state["next_spawn"]:
                raise ReplayError("Invalid policy binding order")
            if w["predecessor"] and state["workers"].get(w["predecessor"], {}).get("lifecycle") != "terminal":
                raise ReplayError("Invalid policy succession")
            state["workers"][w["name"]] = {"name": w["name"], "lifecycle": "planned", "confirmed": False,
                                           "spawn_order": w["spawn_order"], "predecessor": w["predecessor"],
                                           "versions": [copy.deepcopy(change["initial"])]}
            state["next_spawn"] += 1
        elif kind in {"agent_policy_bound", "agent_policy_spawn_confirmed"}:
            w = state["workers"][change["instance"]]
            if w["lifecycle"] not in {"planned", "active"}:
                raise ReplayError("Confirming a finished binding")
            w["confirmed"] = True
            if kind == "agent_policy_bound":
                w["lifecycle"] = "active"
        elif kind == "agent_policy_binding_ended":
            w = state["workers"][change["instance"]]
            if w["lifecycle"] not in {"planned", "active"}:
                raise ReplayError("Ending a finished binding")
            w["lifecycle"] = "terminal" if change["reason"] == "terminal" else "voided"
            if w["lifecycle"] == "terminal":
                w["confirmed"] = True
        elif kind == "agent_policy_committed":
            w = state["workers"][change["worker"]]
            if w["lifecycle"] in ENDED:
                raise ReplayError("Policy change for an ended teammate")
            versions = w["versions"]
            if versions[-1]["version"] != change["before_version"] or change["value"]["version"] != change["before_version"] + 1:
                raise ReplayError("Policy version mismatch")
            versions.append(copy.deepcopy(change["value"]))
        elif kind == "agent_policy_proposed":
            p = copy.deepcopy(change["proposal"])
            if p["proposal_id"] != state["next_proposal"]:
                raise ReplayError("Proposal sequence mismatch")
            state["proposals"][p["proposal_id"]] = p
            state["next_proposal"] += 1
        elif kind == "agent_policy_proposal_applied":
            p = state["proposals"][change["proposal_id"]]
            if p["status"] != "applied" or p["applied"] is not None:
                raise ReplayError("Applying a proposal twice")
            p["applied"] = copy.deepcopy(change["applied"])
        elif kind in {"agent_policy_proposal_voided", "agent_policy_decided"}:
            p = state["proposals"][change["proposal_id"]]
            if p["status"] != "pending":
                raise ReplayError("Finishing a nonpending proposal")
            if kind == "agent_policy_proposal_voided":
                p.update(status="voided", void_reason=change["reason"], replaced_by=change["replaced_by"])
            else:
                p.update(status={"approve": "approved", "reject": "rejected", "amend": "amended"}[change["decision"]],
                         note=change["note"], applied=copy.deepcopy(change["applied"]) or None)
        elif kind == "agent_policy_revoked":
            p = state["proposals"][change["proposal_id"]]
            if change["worker"] not in (p.get("applied") or {}) or change["worker"] in p["revoked"]:
                raise ReplayError("Invalid revocation")
            p["revoked"].append(change["worker"])
            if set(p["revoked"]) >= set(p["applied"]):
                p["status"] = "revoked"
        elif kind == "agent_lesson_recorded":
            lesson = copy.deepcopy(change["lesson"])
            if lesson["id"] != f"L{state['next_lesson']}":
                raise ReplayError("Lesson sequence mismatch")
            state["lessons"].append(lesson)
            state["next_lesson"] += 1
        elif kind == "agent_board_posted":
            post = copy.deepcopy(change["post"])
            if post["id"] != f"B{state['next_board']}":
                raise ReplayError("Board sequence mismatch")
            state["board"].append(post)
            state["next_board"] += 1
        elif kind == "agent_policy_run_ended":
            if any(p["status"] == "pending" for p in state["proposals"].values()):
                raise ReplayError("Run ended with pending proposals")
            state["closed"] = True
        else:
            raise ReplayError("Unknown agent-policy event")

    @classmethod
    def replay(cls, records, writer, *, clock=time.time):
        batches = [r for r in records if r.get("event") == BATCH_EVENT]
        if not batches:
            raise ReplayError("No agent-policy batches")
        state = {}
        for expected, batch in enumerate(batches, 1):
            if batch.get("schema") != 1 or batch.get("batch") != expected or not batch.get("changes"):
                raise ReplayError("Missing or invalid agent-policy batch")
            try:
                for change in batch["changes"]:
                    cls._apply(state, change)
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayError("Invalid agent-policy event") from exc
        store = cls.__new__(cls)
        store._state, store._next_batch = state, len(batches) + 1
        store._writer, store._clock, store._lock = writer, clock, threading.RLock()
        return store
