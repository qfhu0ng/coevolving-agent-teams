"""Deterministic rule obligations for agent_rules_v1 (plan v4 section 7).

The relay reports observable events (commits, task transitions, messages that
reached a teammate's request, shutdown requests, wall-time crossings, first
requests), each with the time it happened, and the teammates' sends with their
native outcome. This engine decides which rule an event triggers, what to
remind, and whether a matching message was sent. It never interprets meaning:
it checks recipients, labeled lines and formats only.

Executed sends are kept for each teammate's lifetime, so a report that happened
after the event but before the event was observed still counts (M2 review B1).
Every obligation ends in one of four classes: satisfied, missed, censored (its
window could not finish) or unobservable (the evidence cannot decide).

Locking: tail -> collaboration -> rule engine -> store -> relay. The engine
never calls the collaboration runtime; the relay queues reminders after this
engine's lock is released, and the collaboration runtime asks
``notice_injected`` (with its own lock held) whether a reminder is still due.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import re
import threading
import time
from typing import Callable

from agent_company.policy import agent_rules
from agent_company.policy import agent_rules_text as text

COALESCE_SECONDS = 600
WINDOW_SECONDS = 900
# The response to the request that first showed the reminder, plus three later ones.
WINDOW_RESPONSES = 4
HEX = re.compile(r"[0-9a-fA-F]{7,40}\Z")
OUTCOMES = ("satisfied", "missed", "censored", "unobservable")


def message_hash(message: str) -> str:
    """Same digest as compat's messageHash (UTF-8 sha256, first 16 hex digits)."""
    return hashlib.sha256(str(message).encode("utf-8")).hexdigest()[:16]


def party(name) -> str | None:
    """Recipient names as native resolves them: the master is `leader`, names ignore case."""
    if not isinstance(name, str) or not name:
        return None
    lowered = name.lower()
    return agent_rules.LEADER if lowered in {"leader", "master"} else lowered


def labeled_lines(message: str) -> dict[str, list[str]]:
    """Field -> values of `LABEL: value` lines (label case-insensitive)."""
    found: dict[str, list[str]] = defaultdict(list)
    for line in message.splitlines():
        match = re.match(r"^[\s*>`_-]*([A-Za-z]+)[\s*`_]*:[\s*`_]*(.*?)[\s*`_]*$", line)
        if match and match[1].lower() in agent_rules.FIELDS and match[2].strip():
            found[match[1].lower()].append(match[2].strip())
    return found


def field_problems(obligation: dict, message: str) -> list[str]:
    """Missing or malformed fields for one obligation; empty means complete."""
    values, problems = labeled_lines(message), []
    for field in obligation["fields"]:
        if not values.get(field):
            problems.append(f"{field}_missing")
            continue
        if field == "sha":
            tokens = [t for v in values["sha"] for t in re.split(r"[\s,;]+", v) if t]
            shas = [t.lower() for t in tokens if HEX.fullmatch(t)]
            if not shas:
                problems.append("sha_format")
            elif obligation["trigger"] == "commit":
                commits = [e["detail"]["sha"].lower() for e in obligation["events"]]
                if not any(sum(c.startswith(token) for c in commits) == 1 for token in shas):
                    problems.append("sha_not_triggering_commit")
        elif field == "paths":
            tokens = [t.strip("`'\"") for v in values["paths"] for t in re.split(r"[\s,;]+", v) if t]
            if not tokens or any(not agent_rules.valid_glob(t) for t in tokens):
                problems.append("paths_format")
    return problems


class RuleEngine:
    def __init__(self, policies: Callable[[str], tuple | None], writer: Callable[..., None], *,
                 started_at: float, duration_seconds: float, clock: Callable[[], float] = time.time):
        """``policies(worker)`` returns (version, ((rule_id, text), ...)) or None."""
        self.policies, self.writer, self.clock = policies, writer, clock
        self.started_at, self.duration = started_at, duration_seconds
        self._lock = threading.RLock()
        self.obligations: dict[str, dict] = {}
        self.sequence = 0
        self.send_sequence = 0
        self.responses: Counter = Counter()
        self.lifetimes: dict[str, dict] = {}
        self.started: set[str] = set()
        self.seen_commits: dict[str, set] = defaultdict(set)
        self.fired_tasks: set[tuple] = set()
        self.released_tasks: set[tuple] = set()
        self.wall_fired: dict[str, set] = defaultdict(set)
        self.pending_sends: dict[tuple, dict] = {}
        self.sends: dict[str, list] = defaultdict(list)
        self.last_sample: dict[str, float] = {}
        # Sends already settled from history, by (worker, recipient, hash): a later native
        # record for the same send is consumed here, never matched to a newer identical send.
        self.history_settled: Counter = Counter()
        # Partial broadcasts settled from history whose recipients may still arrive natively.
        self.awaiting_recipients: dict[tuple, list] = defaultdict(list)
        self.coverage_counts: Counter = Counter()
        self.finished = False
        self._rules_cache: dict[str, agent_rules.Rule] = {}

    # -- helpers --------------------------------------------------------------------
    def _write(self, event, **fields):
        self.writer(event, **fields)

    def _rule(self, rule_text):
        if rule_text not in self._rules_cache:
            self._rules_cache[rule_text] = agent_rules.parse_rule(rule_text)
        return self._rules_cache[rule_text]

    def _alive(self, worker, at=None):
        life = self.lifetimes.get(worker)
        if life is None:
            return False
        return life["ended_at"] is None if at is None else (life["started_at"] <= at and
                                                             (life["ended_at"] is None or at <= life["ended_at"]))

    def _open(self, worker):
        return [o for o in self.obligations.values() if o["worker"] == worker and o["status"] == "open"]

    @staticmethod
    def _matches(rule, event):
        if rule.trigger != event["kind"]:
            return False
        if event["kind"] == "commit" and rule.paths:
            return any(agent_rules.glob_match(p, path) for p in rule.paths for path in event["detail"]["paths"])
        if event["kind"] == "message":
            return rule.sender in (None, agent_rules.EVERYONE) or party(rule.sender) == party(event["detail"]["sender"])
        if event["kind"] == "wall_time":
            return rule.fraction == event["detail"]["fraction"]
        return True

    def _reminder(self, obligation):
        event = obligation["events"][0]
        detail = dict(event["detail"])
        if "sha" in detail:
            detail["sha"] = detail["sha"][:12]
        recipient = text.RULE_RECIPIENTS.get(obligation["recipient"], obligation["recipient"])
        fields = (text.RULE_ACTION_FIELDS.format(labels=", ".join(agent_rules.FIELD_LABELS[f] for f in obligation["fields"]))
                  if obligation["fields"] else "")
        if obligation["recipient"] == agent_rules.BOARD:
            labels = ", ".join(agent_rules.FIELD_LABELS[f] for f in obligation["fields"])
            action = text.RULE_ACTION_BOARD.format(
                fields=text.RULE_ACTION_BOARD_FIELDS.format(labels=labels) if obligation["fields"] else "")
        else:
            action = text.RULE_ACTION.format(recipient=recipient, fields=fields)
        return text.RULE_REMINDER.format(rule_id=obligation["rule_id"],
                                         event=text.RULE_EVENTS[event["kind"]].format(**detail), action=action)

    def _coverage(self, worker, reason, **detail):
        self.coverage_counts[(worker, reason)] += 1
        self._write("rule_coverage", worker=worker, reason=reason, **detail)

    # -- triggers ---------------------------------------------------------------------
    def _trigger(self, worker, kind, detail, *, occurred_at, observed_at, source=None):
        """Create or extend obligations; return reminders as (worker, obligation_id, text)."""
        if self.finished:
            return []
        if not self._alive(worker):
            if worker in self.lifetimes:
                self._coverage(worker, "event_after_end", trigger=kind, detail=detail,
                               occurred_in_lifetime=self._alive(worker, occurred_at))
            return []
        policy = self.policies(worker)
        if policy is None:
            return []
        version, rules = policy
        event = {"kind": kind, "detail": detail, "occurred_at": occurred_at, "observed_at": observed_at,
                 "source": source}
        reminders = []
        for rule_id, rule_text in rules:
            rule = self._rule(rule_text)
            if not self._matches(rule, event):
                continue
            # A fixed window from the first trigger, whatever has happened to it since.
            latest = max((o for o in self.obligations.values()
                          if (o["worker"], o["rule_id"], o["version"]) == (worker, rule_id, version)),
                         key=lambda o: o["created_at"], default=None)
            if latest is not None and observed_at - latest["created_at"] < COALESCE_SECONDS:
                latest["events"].append(copy.deepcopy(event))
                self._write("rule_coalesced", obligation_id=latest["id"], worker=worker, rule_id=rule_id,
                            version=version, trigger=kind, detail=detail, status=latest["status"])
                if latest["status"] == "open":
                    self._recheck(latest, coalesced=True)
                continue
            self.sequence += 1
            obligation = {"id": f"O{self.sequence}", "worker": worker, "rule_id": rule_id, "version": version,
                          "rule": rule_text, "trigger": kind, "recipient": party(rule.recipient),
                          "fields": list(rule.fields), "events": [copy.deepcopy(event)], "created_at": observed_at,
                          "first_occurred_at": occurred_at, "reminded_at": None, "reminder_base": None,
                          "status": "open", "incomplete_attempts": 0, "checked_sends": [], "incomplete_sends": [],
                          "unknown_sends": [],
                          "resolution": None}
            self.obligations[obligation["id"]] = obligation
            self._write("rule_triggered", obligation_id=obligation["id"], worker=worker, rule_id=rule_id,
                        version=version, rule=rule_text, trigger=kind, detail=detail,
                        occurred_at=occurred_at, observed_at=observed_at, source=source)
            recipient = obligation["recipient"]
            if (recipient not in (agent_rules.LEADER, agent_rules.EVERYONE, agent_rules.BOARD)
                    and not self._alive_name(recipient)):
                self._resolve(obligation, "censored", reason="recipient_not_active")
                continue
            self._recheck(obligation)
            if obligation["status"] == "open":
                reminders.append((worker, obligation["id"], self._reminder(obligation)))
        return reminders

    def _alive_name(self, name, at=None):
        return any(self._alive(worker, at) for worker in self.lifetimes if party(worker) == name)

    def worker_spawned(self, worker, *, at=None):
        """Lifetime starts at the committed spawn: events before its first request still count."""
        with self._lock:
            now = self.clock() if at is None else at
            if worker not in self.lifetimes or self.lifetimes[worker]["ended_at"] is not None:
                self.lifetimes[worker] = {"started_at": now, "ended_at": None}
                self.started.discard(worker)

    def coding_request(self, worker, *, at=None):
        """An admitted normal coding request of a teammate: start, wall-time and window checks."""
        with self._lock:
            now = self.clock() if at is None else at
            reminders = []
            if worker not in self.lifetimes:
                self.lifetimes[worker] = {"started_at": now, "ended_at": None}
            if worker not in self.started:
                self.started.add(worker)
                reminders += self._trigger(worker, "start", {}, occurred_at=now, observed_at=now)
            for fraction in agent_rules.WALL_TIME_FRACTIONS:
                crossing = self.started_at + self.duration * fraction / 100
                if (crossing <= now and fraction not in self.wall_fired[worker]
                        and crossing >= self.lifetimes[worker]["started_at"]):
                    self.wall_fired[worker].add(fraction)
                    reminders += self._trigger(worker, "wall_time", {"fraction": fraction},
                                               occurred_at=crossing, observed_at=now)
            self._expire(worker, now)
            return reminders

    def coding_response(self, worker, *, at=None):
        """A delivered response: one opportunity to act. Failed requests are not counted."""
        with self._lock:
            now = self.clock() if at is None else at
            for obligation in self._open(worker):
                if obligation["reminded_at"] is not None and obligation["reminder_base"] is None:
                    obligation["reminder_base"] = self.responses[worker]
            self.responses[worker] += 1
            # No expiry here: this response's own sends are still pending (M2 review N3).

    def observe_commit(self, worker, sha, paths, *, committed_at=None, not_before=None, observed_at=None,
                       source=None):
        """``not_before``: the previous sample of this worktree, which did not show the commit yet."""
        with self._lock:
            now = self.clock() if observed_at is None else observed_at
            if not isinstance(sha, str) or sha in self.seen_commits[worker]:
                return []
            self.seen_commits[worker].add(sha)
            occurred = min(committed_at, now) if isinstance(committed_at, (int, float)) else now
            # Merged-in commits keep old committer times; they appeared here only after the
            # previous sample (or the spawn), never earlier (M2 review N2).
            floors = [t for t in (not_before, self.lifetimes.get(worker, {}).get("started_at"))
                      if isinstance(t, (int, float))]
            if floors:
                occurred = min(max([occurred, *floors]), now)
            return self._trigger(worker, "commit", {"sha": sha, "paths": list(paths or [])},
                                 occurred_at=occurred, observed_at=now, source=source)

    def note_sample(self, worker, *, at=None):
        with self._lock:
            self.last_sample[worker] = self.clock() if at is None else at

    def observe_task(self, owner, task_id, kind, *, occurred_at=None, at=None, source=None, from_tool=False):
        with self._lock:
            now = self.clock() if at is None else at
            key = (party(owner), str(task_id), kind)
            if key in self.fired_tasks or key in self.released_tasks and not from_tool:
                return []  # a released key fires again only from a task tool, never from a sample
            if owner not in self.lifetimes:
                # Assigned before its owner's spawn committed: not a trigger, and not lost silently (M3 review 10).
                self._coverage(owner, "event_before_spawn", trigger=kind, task_id=str(task_id))
                return []
            self.released_tasks.discard(key)
            self.fired_tasks.add(key)
            when = min(occurred_at, now) if isinstance(occurred_at, (int, float)) else now
            return self._trigger(owner, kind, {"task_id": str(task_id)}, occurred_at=when, observed_at=now,
                                 source=source)

    def release_task(self, owner, task_id, kind):
        """A task tool reversed this transition (owner moved away, or completion reopened): it may fire again.

        Reversals seen only in periodic samples keep the key, so stale-sample flaps never re-trigger.
        """
        with self._lock:
            key = (party(owner), str(task_id), kind)
            if key in self.fired_tasks:
                self.fired_tasks.discard(key)
                self.released_tasks.add(key)

    def observe_message(self, worker, sender, *, occurred_at=None, source=None):
        with self._lock:
            now = self.clock()
            when = min(occurred_at, now) if isinstance(occurred_at, (int, float)) else now
            return self._trigger(worker, "message", {"sender": sender}, occurred_at=when, observed_at=now, source=source)

    def observe_shutdown(self, worker, *, occurred_at=None, source=None):
        with self._lock:
            now = self.clock()
            when = min(occurred_at, now) if isinstance(occurred_at, (int, float)) else now
            return self._trigger(worker, "shutdown_request", {}, occurred_at=when, observed_at=now, source=source)

    def notice_injected(self, worker, obligation_id) -> bool:
        """Whether a queued reminder is still due; its window starts at first injection."""
        with self._lock:
            obligation = self.obligations.get(obligation_id)
            if obligation is None or obligation["status"] != "open":
                if obligation is not None:
                    self._write("rule_reminder_withdrawn", obligation_id=obligation_id, worker=worker,
                                status=obligation["status"])
                return False
            if obligation["reminded_at"] is None:
                obligation["reminded_at"] = self.clock()
                self._write("rule_reminded", obligation_id=obligation_id, worker=worker)
            return True

    # -- sends ----------------------------------------------------------------------------
    def record_send(self, worker, key, *, to, message, at=None):
        """A delivered send_message call; it counts once its native outcome is known."""
        with self._lock:
            self.pending_sends[(worker, *key)] = {"worker": worker, "to": party(to), "message": message,
                                                  "hash": message_hash(message),
                                                  "sent_at": self.clock() if at is None else at}

    def record_board(self, worker, message, *, at=None):
        """A committed BOARD: post: it is on the board at once, so it needs no native outcome."""
        with self._lock:
            self._executed({"worker": worker, "to": agent_rules.BOARD, "message": message,
                            "hash": message_hash(message), "sent_at": self.clock() if at is None else at},
                           "sent", None)

    def drop_send(self, worker, key):
        """The response carrying this send never reached the native CLI."""
        with self._lock:
            self.pending_sends.pop((worker, *key), None)

    def pending_keys(self, worker):
        with self._lock:
            return [key[1:] for key in self.pending_sends if key[0] == worker]

    def resolve_send(self, worker, key, outcome):
        """From the send's native tool result in the sender's history: sent | send_partial | send_failed."""
        with self._lock:
            send = self.pending_sends.pop((worker, *key), None)
            if send is None:
                return
            self.history_settled[(worker, send["to"], send["hash"])] += 1
            executed = self._executed(send, outcome, None)
            if send["to"] == agent_rules.EVERYONE:
                # Every broadcast settled from history queues, so its own late native record
                # finds it (oldest first); only partial ones still need recipients (R4-1).
                self.awaiting_recipients[(worker, agent_rules.EVERYONE, send["hash"])].append(executed)

    def resolve_native(self, worker, to, digest, outcome, *, recipients=None):
        """From compat's native execution record (also for a sender aborted right after it)."""
        with self._lock:
            target = party(to)
            accepted = {party(r) for r in recipients} if recipients is not None else None
            settled = (worker, target, digest)
            if self.history_settled[settled] > 0:
                # This record belongs to a send already settled from history (oldest first).
                self.history_settled[settled] -= 1
                queue = self.awaiting_recipients.get((worker, target, digest)) if target == agent_rules.EVERYONE else None
                waiting = queue.pop(0) if queue else None
                if (waiting is not None and accepted is not None and waiting["outcome"] == "send_partial"
                        and waiting["recipients"] is None):
                    self._attach_recipients(waiting, accepted)
                return
            key = next((k for k, s in sorted(self.pending_sends.items(), key=lambda item: item[1]["sent_at"])
                        if s["worker"] == worker and s["hash"] == digest and s["to"] == target), None)
            if key is not None:
                self._executed(self.pending_sends.pop(key), outcome, accepted)

    def _attach_recipients(self, executed, accepted):
        """Native per-recipient results for a partial broadcast first settled from history."""
        executed["recipients"] = accepted
        for obligation in self._open(executed["worker"]):
            if executed["id"] in obligation["unknown_sends"]:
                obligation["unknown_sends"].remove(executed["id"])
                obligation["checked_sends"].remove(executed["id"])
                self._check(obligation, executed)

    def _executed(self, send, outcome, recipients):
        if outcome == "send_failed":
            return None
        self.send_sequence += 1
        executed = {**send, "id": self.send_sequence, "outcome": outcome, "recipients": recipients}
        if send["to"] == agent_rules.EVERYONE and outcome == "sent" and recipients is None:
            executed["recipients"] = self._recipients_at(send["worker"], send["sent_at"])
        self.sends[send["worker"]].append(executed)
        for obligation in self._open(send["worker"]):
            self._check(obligation, executed)
        return executed

    def _recipients_at(self, sender, at):
        """Who a full broadcast reached: teammates alive at that moment, plus the master."""
        alive = {party(name) for name in self.lifetimes if self._alive(name, at)}
        return (alive - {party(sender)}) | {agent_rules.LEADER}

    def _recheck(self, obligation, *, coalesced=False):
        """A new or extended obligation looks at reports already sent (a new commit may match)."""
        for send in self.sends[obligation["worker"]]:
            if obligation["status"] != "open":
                return
            self._check(obligation, send, again=coalesced)

    def _reached(self, obligation, send):
        """True, False, or None when a partial broadcast's recipients are unknown."""
        wanted, to = obligation["recipient"], send["to"]
        if wanted == agent_rules.BOARD:
            return to == agent_rules.BOARD
        if wanted == agent_rules.EVERYONE:
            if to != agent_rules.EVERYONE:
                return False
            if send["outcome"] == "sent":
                return True
            if send["recipients"] is None:
                return None
            # Ended teammates stay in the native roster and always fail (M2 review N1).
            return self._recipients_at(send["worker"], send["sent_at"]) <= send["recipients"]
        if to == wanted:
            return True
        if to == agent_rules.EVERYONE:
            if send["recipients"] is not None:
                return wanted in send["recipients"]
            return None
        return False

    def _check(self, obligation, send, *, again=False):
        sid = send["id"]
        if send["sent_at"] < obligation["first_occurred_at"]:
            return
        first = sid not in obligation["checked_sends"]
        if not first and not (again and sid in obligation["incomplete_sends"]):
            return
        if first:
            obligation["checked_sends"].append(sid)
        reached = self._reached(obligation, send)
        if reached is None:
            if sid not in obligation["unknown_sends"]:
                obligation["unknown_sends"].append(sid)
            return
        if not reached:
            return
        problems = field_problems(obligation, send["message"])
        if problems:
            if first:  # one attempt per send, however often it is rechecked
                obligation["incomplete_attempts"] += 1
                obligation["incomplete_sends"].append(sid)
                self._write("rule_send_incomplete", obligation_id=obligation["id"], worker=obligation["worker"],
                            problems=problems)
            return
        spontaneous = obligation["reminded_at"] is None or send["sent_at"] < obligation["reminded_at"]
        self._resolve(obligation, "satisfied", spontaneous=spontaneous,
                      latency_s=round(send["sent_at"] - obligation["first_occurred_at"], 3))

    # -- endings ------------------------------------------------------------------------------
    def _window_done(self, obligation, now):
        base = obligation["reminder_base"]
        return (obligation["reminded_at"] is not None and base is not None
                and now - obligation["reminded_at"] >= WINDOW_SECONDS
                and self.responses[obligation["worker"]] - base >= WINDOW_RESPONSES)

    def _undecided(self, obligation):
        """A send that could satisfy it still has no known outcome, or reached unknown recipients."""
        # Judged now, against the obligation's current (possibly coalesced) commit set.
        unknown = set(obligation["unknown_sends"])
        if any(not field_problems(obligation, s["message"]) for s in self.sends[obligation["worker"]] if s["id"] in unknown):
            return True
        if obligation["recipient"] == agent_rules.BOARD:
            return False  # board posts are settled at commit, never pending
        # A send missing required fields can never satisfy it, whatever its outcome (plan v4 7.5).
        return any(send["worker"] == obligation["worker"] and send["sent_at"] >= obligation["first_occurred_at"]
                   and (send["to"] == obligation["recipient"] or send["to"] == agent_rules.EVERYONE)
                   and not field_problems(obligation, send["message"])
                   for send in self.pending_sends.values())

    def _expire(self, worker, now):
        for obligation in self._open(worker):
            if self._window_done(obligation, now):
                undecided = self._undecided(obligation)
                self._resolve(obligation, "unobservable" if undecided else "missed",
                              reason="undecided_send" if undecided else "window_ended")

    def _end_open(self, obligation, now, reason):
        undecided = self._undecided(obligation)
        if self._window_done(obligation, now):
            self._resolve(obligation, "unobservable" if undecided else "missed",
                          reason="undecided_send" if undecided else "window_ended")
        elif undecided:
            self._resolve(obligation, "unobservable", reason="undecided_send")
        else:
            self._resolve(obligation, "censored", reason=reason)

    def worker_ended(self, worker, *, at=None):
        with self._lock:
            now = self.clock() if at is None else at
            life = self.lifetimes.get(worker)
            if life is not None and life["ended_at"] is None:
                life["ended_at"] = now
                for fraction in agent_rules.WALL_TIME_FRACTIONS:
                    crossing = self.started_at + self.duration * fraction / 100
                    if life["started_at"] <= crossing <= now and fraction not in self.wall_fired[worker]:
                        self._coverage(worker, "wall_time_unobserved", fraction=fraction)
                last = self.last_sample.get(worker)
                self._coverage(worker, "final_git_interval", last_sample_at=last, ended_at=now,
                               unsampled_seconds=round(now - last, 3) if last is not None else None)
            for obligation in self._open(worker):
                self._end_open(obligation, now, "worker_ended" if obligation["reminded_at"] else "never_reminded")
            name = party(worker)
            for obligation in [o for o in self.obligations.values() if o["status"] == "open" and o["recipient"] == name]:
                if not self._alive_name(name):
                    self._end_open(obligation, now, "recipient_ended")

    def finish(self, *, at=None):
        """Deterministic run-end settlement; no obligation stays open."""
        with self._lock:
            if self.finished:
                return
            now = self.clock() if at is None else at
            for worker, life in self.lifetimes.items():
                if life["ended_at"] is None:
                    last = self.last_sample.get(worker)
                    self._coverage(worker, "final_git_interval", last_sample_at=last, ended_at=now, alive_at_end=True,
                                   unsampled_seconds=round(now - last, 3) if last is not None else None)
            for obligation in [o for o in self.obligations.values() if o["status"] == "open"]:
                self._end_open(obligation, now, "run_ended" if obligation["reminded_at"] else "never_reminded")
            self.finished = True
            self._write("rule_engine_finished", stats=self._stats())

    def coverage(self, worker, reason, **detail):
        with self._lock:
            self._coverage(worker, reason, **detail)

    def _resolve(self, obligation, status, **resolution):
        obligation["status"] = status
        obligation["resolution"] = {"at": self.clock(), **resolution,
                                    "incomplete_attempts": obligation["incomplete_attempts"]}
        self._write(f"rule_{status}", obligation_id=obligation["id"], worker=obligation["worker"],
                    rule_id=obligation["rule_id"], version=obligation["version"],
                    commits=[e["detail"]["sha"] for e in obligation["events"] if e["kind"] == "commit"],
                    **obligation["resolution"])

    # -- statistics ---------------------------------------------------------------------------
    def _stats(self):
        rows: dict[tuple, dict] = {}
        for o in self.obligations.values():
            key = (o["worker"], o["rule_id"], o["version"])
            row = rows.setdefault(key, {"worker": o["worker"], "rule_id": o["rule_id"], "version": o["version"],
                                        "rule": o["rule"], "kind": "fields" if o["fields"] else "send_only",
                                        "open": 0, "satisfied_spontaneous": 0, "incomplete_attempts": 0,
                                        "coalesced_events": 0, **dict.fromkeys(OUTCOMES, 0)})
            row["open" if o["status"] == "open" else o["status"]] += 1
            row["incomplete_attempts"] += o["incomplete_attempts"]
            row["coalesced_events"] += len(o["events"]) - 1
            if o["status"] == "satisfied" and o["resolution"].get("spontaneous"):
                row["satisfied_spontaneous"] += 1
        for row in rows.values():
            decided = row["satisfied"] + row["missed"]
            row["rate"] = round(row["satisfied"] / decided, 4) if decided else None
            row["denominator"] = decided
        totals = {}
        for kind in ("send_only", "fields"):
            chosen = [r for r in rows.values() if r["kind"] == kind]
            counts = {name: sum(r[name] for r in chosen) for name in (*OUTCOMES, "open")}
            decided = counts["satisfied"] + counts["missed"]
            totals[kind] = {**counts, "denominator": decided,
                            "rate": round(counts["satisfied"] / decided, 4) if decided else None}
        coverage = [{"worker": worker, "reason": reason, "count": count}
                    for (worker, reason), count in sorted(self.coverage_counts.items())]
        return {"rows": sorted(rows.values(), key=lambda r: (r["worker"], r["rule_id"], r["version"])),
                "totals": totals, "coverage": coverage, "metric": "observable_obligation_fulfillment_rate"}

    def stats(self):
        with self._lock:
            return self._stats()

    def rates_by_worker(self):
        """Compact per-teammate view for check-ins: satisfied/decided over all its rules."""
        with self._lock:
            out = {}
            for o in self.obligations.values():
                row = out.setdefault(o["worker"], {"satisfied": 0, "missed": 0, "censored": 0, "unobservable": 0, "open": 0})
                row["open" if o["status"] == "open" else o["status"]] += 1
            for row in out.values():
                decided = row["satisfied"] + row["missed"]
                row["rate"] = round(row["satisfied"] / decided, 4) if decided else None
            return out
