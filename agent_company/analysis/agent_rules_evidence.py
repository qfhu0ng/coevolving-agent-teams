"""Evidence record of one agent_rules_v1 Team run for the outer optimizer and reviewer.

Plan v4 sections 11 and 12 (item 4). Built after the run from the relay's own
records only: the replayable policy batches, policy injections, rule obligations,
reflection funnel, lesson gate, board, collaboration Git observations, message facts
and native send outcomes. No request, response, reasoning, stdout or hidden test is
read. Message text is never read here; information use joins hashes, extracted
commits and paths.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from agent_company.analysis.evidence_projection import number, redact, safe_file, text_leaf
from agent_company.evolution import agent_rules_outer_text as outer_text
from agent_company.policy.agent_policy_store import AgentPolicyStore
from agent_company.policy.role_policy import ReplayError

SCHEMA = 1
ROW_LIMIT = 400
PATH_TAIL = ".,;:)('\"`"
PATH_HEAD = "('\"`"


def _events(path: Path, root: Path) -> tuple[list[dict], int]:
    if not safe_file(path, root):
        return [], 0
    rows, bad = [], 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                bad += 1
    return rows, bad


def _party(name):
    if not isinstance(name, str):
        return None
    return "master" if name.lower() in {"leader", "master"} else name


def _clean_path(token: str) -> str:
    """Sentence punctuation around a path token; a leading dot belongs to the path (M5 review R2-2)."""
    token = token.rstrip(PATH_TAIL).lstrip(PATH_HEAD)
    return token[2:] if token.startswith("./") else token


def _redacted(value):
    """Every string inside, redacted like the text leaves: rule text and its PATHS are agent-written."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    if isinstance(value, dict):
        return {key: _redacted(item) for key, item in value.items()}
    return value


def _diff(diff):
    """A version's diff with the same treatment as the version: notes as text leaves (external review #6)."""
    if not isinstance(diff, dict):
        return _redacted(diff)
    return {key: [text_leaf(item) for item in value] if key == "notes" and isinstance(value, list)
            else _redacted(value) for key, value in diff.items()}


def _epoch(iso):
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _limited(rows: list, coverage: dict, key: str) -> list:
    if len(rows) > ROW_LIMIT:
        coverage[f"{key}_rows_omitted"] = len(rows) - ROW_LIMIT
    return rows[:ROW_LIMIT]


class Run:
    """Times and lifetimes shared by every part of the record (relay seconds)."""

    def __init__(self, events: list[dict]):
        times = [number(e.get("time")) for e in events if number(e.get("time")) is not None]
        self.start = min(times) if times else None
        self.lifetimes: dict[str, dict] = {}
        for event in events:
            if event.get("event") != "agent_policy_batch":
                continue
            for change in event.get("changes") or []:
                kind, at = change.get("event"), number(event.get("time"))
                if kind == "agent_policy_planned" and isinstance(change.get("worker"), dict):
                    name = change["worker"].get("name")
                    self.lifetimes[name] = {"spawned": at, "first_request": None, "ended": None}
                elif kind == "agent_policy_bound" and change.get("instance") in self.lifetimes:
                    self.lifetimes[change["instance"]]["first_request"] = at
                elif kind == "agent_policy_binding_ended" and change.get("instance") in self.lifetimes:
                    self.lifetimes[change["instance"]]["ended"] = at
        self._names = {name.lower(): name for name in self.lifetimes if isinstance(name, str)}
        # Each worktree's commits as sampled: native sample time when known, else the relay's write time.
        self.sightings: dict[str, list] = defaultdict(list)
        self.commit_paths: dict[str, set] = {}
        for e in events:
            if e.get("event") != "collaboration_git_observed" or not isinstance(e.get("commit"), str):
                continue
            source = e.get("source") if isinstance(e.get("source"), dict) else {}
            sample, written = number(source.get("native_sample")), number(e.get("time"))
            at = sample / 1000 if sample is not None else written
            # A native sample precedes its relay record by at most a refresh interval or so; anything
            # else is not a usable clock, so the relay's write time stands (M5 review minor 6).
            if written is not None and (at is None or not written - 600 <= at <= written + 60):
                at = written
            sha, paths = e["commit"].lower(), {p for p in e.get("paths") or [] if isinstance(p, str)}
            self.sightings[_party(e.get("actor"))].append((at or 0, sha, paths))
            self.commit_paths.setdefault(sha, set()).update(paths)
        self.known_paths = set().union(*self.commit_paths.values()) if self.commit_paths else set()
        self.last_sample = {owner: max(t for t, _, _ in rows) for owner, rows in self.sightings.items()}
        for e in events:  # the rule engine's final sample time per worker
            if e.get("event") == "rule_coverage" and e.get("reason") == "final_git_interval":
                at = number(e.get("last_sample_at"))
                if at is not None:
                    owner = _party(e.get("worker"))
                    self.last_sample[owner] = max(at, self.last_sample.get(owner, at))

    def party(self, name):
        """A teammate's recorded name whatever case the message used; leader is master."""
        name = _party(name)
        return self._names.get(name.lower(), name) if name is not None else None

    def note_delivery_sample(self, at):
        """The relay's last readable sample of the delivery repository, which writes rows only on change."""
        if number(at) is not None:
            self.last_sample["master"] = max(at, self.last_sample.get("master", at))

    def commit_status(self, owner, commit, at) -> tuple[str, str | None]:
        """Where a mentioned commit stands in one worktree relative to the message time."""
        seen = self.sightings.get(owner, [])
        firsts = [t for t, sha, _ in seen if sha.startswith(commit)]
        if firsts:
            return ("present_before" if min(firsts) < at else "appeared_after"), None
        if not self.observable_after(owner, at):
            return "undetermined", "not_sampled"
        mentioned = set().union(*(p for sha, p in self.commit_paths.items() if sha.startswith(commit)))
        if any(t >= at and touched & mentioned for t, _, touched in seen):
            return "undetermined", "possible_rewrite"
        return "not_seen", None

    def rel(self, t):
        return round(t - self.start, 3) if self.start is not None and number(t) is not None else None

    def rel_iso(self, iso):
        return self.rel(_epoch(iso))

    def active_at(self, at) -> set[str]:
        return {name for name, life in self.lifetimes.items()
                if life["spawned"] is not None and life["spawned"] <= at and (life["ended"] is None or life["ended"] > at)}

    def observable_after(self, owner, at) -> bool:
        life = self.lifetimes.get(owner)
        if life is not None and life["ended"] is not None and life["ended"] < at:
            return False
        return self.last_sample.get(owner) is not None and self.last_sample[owner] >= at


def _policies(state: dict, events: list[dict], run: Run) -> list[dict]:
    shown = {}
    for e in events:
        if e.get("event") == "agent_policy_injected" and e.get("admitted") and e.get("role") != "master":
            for worker, version in (e.get("versions") or {}).items():
                shown.setdefault((worker, version), number(e.get("time")))
    out = []
    for worker in sorted(state["workers"].values(), key=lambda w: w["spawn_order"]):
        versions = [{"version": v["version"], "at": v["at"], "t_s": run.rel_iso(v["at"]),
                     "first_shown_t_s": run.rel(shown.get((worker["name"], v["version"]))),
                     "source": v["source"], "actor": v["actor"], "operation": v.get("operation"),
                     "proposal_id": v.get("proposal_id"), "lineage": v.get("lineage"),
                     "rules": _redacted(dict(sorted(v["rules"].items(), key=lambda r: int(r[0][1:])))),
                     "notes": text_leaf(v["notes"]), "diff": _diff(v.get("diff"))}
                    for v in worker["versions"]]
        out.append({"worker": worker["name"], "lifecycle": worker["lifecycle"], "spawn_order": worker["spawn_order"],
                    "predecessor": worker["predecessor"], "versions": versions})
    return out


def _proposals(state: dict, run: Run) -> list[dict]:
    return [{"id": p["proposal_id"], "proposer": p["proposer"], "target": p["target"], "operation": p["operation"],
             "rule_id": p["rule_id"], "body": text_leaf(p["body"]), "reason": text_leaf(p["reason"]),
             "status": p["status"], "note": text_leaf(p.get("note")), "created_at": p["created_at"],
             "t_s": run.rel_iso(p["created_at"]), "base_versions": p["base"], "void_reason": p.get("void_reason"),
             "applied_versions": {name: entry.get("version") for name, entry in (p.get("applied") or {}).items()},
             "revoked_for": p.get("revoked", [])}
            for p in sorted(state["proposals"].values(), key=lambda p: p["proposal_id"])]


def _refused(events: list[dict], run: Run, coverage: dict) -> dict:
    """What the agents wrote that the runtime did not accept: skipped spawn rules and refused calls."""
    skipped = []
    for batch in (e for e in events if e.get("event") == "agent_policy_batch"):
        for change in batch.get("changes") or []:
            if change.get("event") != "agent_policy_planned" or not isinstance(change.get("worker"), dict):
                continue
            skipped += [{"t_s": run.rel(batch.get("time")), "worker": change["worker"].get("name"),
                         "line": text_leaf(item.get("line")), "code": item.get("code"), "detail": text_leaf(item.get("detail")),
                         "inherited": bool(item.get("inherited"))}
                        for item in change.get("skipped") or [] if isinstance(item, dict)]
    refused = [{"t_s": run.rel(e.get("time")), "role": run.party(e.get("role")), "code": e.get("code"),
                "detail": text_leaf(e.get("detail"))}
               for e in events if e.get("event") == "role_package_call_rejected" and e.get("code") != "lesson_gate"]
    by_code = Counter(row["code"] for row in skipped + refused)
    return {"spawn_rules_skipped": _limited(skipped, coverage, "spawn_rules_skipped"),
            "calls_refused": _limited(refused, coverage, "calls_refused"),
            "by_code": dict(sorted(by_code.items(), key=lambda item: str(item[0])))}


def _timeline(run: Run, coverage: dict) -> dict:
    """Active teammates over time and the commits each worktree showed (first sighting each)."""
    changes = []
    for name, life in run.lifetimes.items():
        changes += [(life[key], name, key) for key in ("spawned", "first_request", "ended") if life[key] is not None]
    active, rows = set(), []
    for time, worker, change in sorted(changes, key=lambda row: row[0]):
        if change == "spawned":
            active.add(worker)
        elif change == "ended":
            active.discard(worker)
        rows.append({"t_s": run.rel(time), "worker": worker, "change": change, "active_teammates": len(active)})
    commits, seen = [], set()
    for owner, sightings in run.sightings.items():
        for at, sha, paths in sightings:
            if (owner, sha) not in seen:
                seen.add((owner, sha))
                commits.append({"t_s": run.rel(at), "worktree_of": owner, "commit": sha[:12], "changed_paths": len(paths)})
    commits.sort(key=lambda row: (row["t_s"] is None, row["t_s"] or 0))
    return {"teammates": _limited(rows, coverage, "teammates"),
            "worktree_commits": _limited(commits, coverage, "worktree_commits"),
            "note": outer_text.TIMELINE_NOTE}


def information_use(events: list[dict], run: Run) -> dict:
    """Whether commits and paths named in a message later appeared in the recipient's work (plan v4 section 11)."""
    outcomes = defaultdict(list)
    for e in events:
        if e.get("event") == "message_outcome" and isinstance(e.get("hash"), str):
            to = "*" if e.get("to") == "*" or e.get("native_event") == "team_broadcast" else run.party(e.get("to"))
            outcomes[(run.party(e.get("sender")), e["hash"], to)].append(e)
    rows, totals, coverage = [], Counter(), Counter()
    by_pair = defaultdict(Counter)
    for fact in (e for e in events if e.get("event") == "message_facts"):
        sender, at, to = run.party(fact.get("sender")), number(fact.get("time")) or 0, fact.get("to")
        key = (sender, fact.get("hash"), "*" if to == "*" else run.party(to))
        queue = outcomes.get(key) or []
        outcome = queue.pop(0) if queue else None
        if outcome is None:
            recipients, delivery = None, "unknown"
        elif to == "*":
            recipients = [run.party(r["name"]) for r in outcome.get("recipients") or []
                          if isinstance(r, dict) and isinstance(r.get("name"), str) and r.get("outcome") == "accepted"]
            delivery = "broadcast"
        else:
            recipients, delivery = ([run.party(to)] if outcome.get("ok") else []), ("sent" if outcome.get("ok") else "failed")
        kind = "board" if fact.get("board") else "broadcast" if to == "*" else "direct"
        if kind == "board":
            # A post reaches every other active worker through the board, besides its native recipient.
            readers = sorted(run.active_at(at) - {sender})
            recipients = list(dict.fromkeys((recipients or []) + readers))
        raw_commits = [c.lower() for c in fact.get("commits") or [] if isinstance(c, str)]
        commits = [c for c in raw_commits if any(k.startswith(c) for k in run.commit_paths)]
        raw_paths = {_clean_path(p) for p in fact.get("paths") or [] if isinstance(p, str)}
        paths = {p for p in raw_paths if p in run.known_paths}
        coverage["tokens_not_matching_a_commit"] += len(raw_commits) - len(commits)
        coverage["tokens_not_matching_a_changed_path"] += len(raw_paths) - len(paths)
        if not commits and not paths:
            continue
        results = []
        for recipient in recipients or []:
            seen = run.sightings.get(recipient, [])
            observable = run.observable_after(recipient, at)
            for commit in commits:
                status, reason = run.commit_status(recipient, commit, at)
                entry = {"recipient": recipient, "commit": commit[:12], "status": status}
                if reason:
                    entry["undetermined_reason"] = reason
                    totals[f"commit_undetermined_{reason}"] += 1
                if recipient != "master":
                    entry["in_delivery"] = run.commit_status("master", commit, at)[0]
                results.append(entry)
                totals[f"commit_{status}"] += 1
                totals[f"{kind}_commit_{status}"] += 1
                by_pair[(sender, recipient)][f"commit_{status}"] += 1
            if paths:
                if not observable:
                    results.append({"recipient": recipient, "paths": _redacted(sorted(paths)[:20]), "status": "undetermined",
                                    "undetermined_reason": "not_sampled"})
                    totals["paths_undetermined"] += 1
                    continue
                touched_after = any(t >= at and touched & paths for t, _, touched in seen)
                touched_ever = any(touched & paths for _, _, touched in seen)
                results.append({"recipient": recipient, "paths": _redacted(sorted(paths)[:20]),
                                "touched_after": touched_after, "touched_at_any_time": touched_ever})
                for prefix in ("", f"{kind}_"):
                    totals[f"{prefix}path_mentions"] += 1
                    totals[f"{prefix}paths_touched_after"] += int(touched_after)
                totals["paths_touched_at_any_time"] += int(touched_ever)
                by_pair[(sender, recipient)]["path_mentions"] += 1
                by_pair[(sender, recipient)]["paths_touched_after"] += int(touched_after)
        totals[f"messages_{kind}"] += 1
        if outcome is None:
            totals["messages_without_native_outcome"] += 1
        # The recipient comes from the agent's own `to` argument: redacted like it (review of #6).
        rows.append({"t_s": run.rel(fact.get("time")), "sender": sender, "to": _redacted(to), "kind": kind, "delivery": delivery,
                     "results": [{**r, "recipient": _redacted(r["recipient"])} for r in results]})
    limits: dict = {}
    return {"totals": dict(totals),
            "by_sender_recipient": [{"sender": s, "recipient": _redacted(r), **dict(c)}
                                    for (s, r), c in sorted(by_pair.items())],
            "messages": _limited(rows, limits, "messages"), "coverage": {**dict(coverage), **limits},
            "definition": outer_text.INFORMATION_USE_DEFINITION}


def _resources(root: Path) -> dict:
    """Container memory during inference from the resource observer's samples (2026-10-18)."""
    rows, bad = _events(root / "observations.jsonl", root)
    inference = [r for r in rows if r.get("memory_scope") == "inference_container"]
    peaks = [number(r.get("memory_peak_bytes"), integer=True) for r in inference]
    kills = [number((r.get("memory_events") or {}).get("oom_kill"), integer=True)
             for r in inference if isinstance(r.get("memory_events"), dict)]
    peaks, kills = [v for v in peaks if v is not None], [v for v in kills if v is not None]
    return {"inference_samples": len(inference), "memory_peak_bytes": max(peaks) if peaks else None,
            "processes_killed_for_memory": max(kills) if kills else None,
            "samples_without_memory": len(inference) - len(peaks), "invalid_lines": bad}


def _funnel(events: list[dict], state: dict, run: Run) -> list[dict]:
    proposals = state["proposals"]

    def later(ids):
        # What later happened to a proposal the prompt led to (plan v4 section 9: revoked or modified).
        return {str(pid): {"status": proposals[pid]["status"], "revoked_for": proposals[pid].get("revoked", [])}
                for pid in ids or [] if pid in proposals}
    return [{"t_s": run.rel(e.get("time")), "worker": e.get("worker"), "kind": e.get("kind"), "detail": e.get("detail"),
             "queued_t_s": run.rel(e.get("queued_at")), "shown_t_s": run.rel(e.get("shown_at")),
             "close_reason": e.get("close_reason"), "outcome": e.get("outcome"), "responses": e.get("responses"),
             "produced": e.get("produced"), "own_policy_changes": e.get("own_policy_changes"),
             "proposals_for_others": e.get("proposals_for_others"), "lessons": e.get("lessons"),
             "board_posts": e.get("board_posts"),
             "later": later((e.get("own_policy_changes") or []) + (e.get("proposals_for_others") or []))}
            for e in events if e.get("event") == "reflection_funnel"]


def agent_rules_record(run_dir: str | Path) -> dict | None:
    """The agent_rules_v1 record of one Team trial, or None when the trial used another protocol."""
    root = Path(run_dir).resolve(strict=True)
    relay = root / "team" / "relay"
    events, bad_lines = _events(relay / "events.jsonl", root)
    batches = [e for e in events if e.get("event") == "agent_policy_batch"]
    if not batches:
        return None
    coverage: dict = {"invalid_event_lines": bad_lines}
    try:
        state = AgentPolicyStore.replay(batches, lambda record: None).export()
    except ReplayError as exc:
        return {"schema": SCHEMA, "protocol": "agent_rules_v1", "replay": "failed",
                "replay_error": type(exc).__name__, "notes": list(outer_text.EVIDENCE_NOTES)}
    run = Run(events)
    settled: dict = {}
    obligations_path = relay / "rule_obligations.json"
    if safe_file(obligations_path, root):
        try:
            value = json.loads(obligations_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                settled = value
            else:
                coverage["rule_obligations_not_an_object"] = True
        except ValueError:
            coverage["rule_obligations_unreadable"] = True
    else:
        coverage["rule_obligations_missing"] = True
    run.note_delivery_sample(settled.get("delivery_last_sample_at"))
    obligations = [{"id": o.get("id"), "worker": o.get("worker"), "rule_id": o.get("rule_id"),
                    "version": o.get("version"), "rule": _redacted(o.get("rule")), "trigger": o.get("trigger"),
                    "status": o.get("status"), "events": len(o.get("events") or []),
                    "first_occurred_t_s": run.rel(o.get("first_occurred_at")),
                    "reminded": o.get("reminded_at") is not None, "incomplete_attempts": o.get("incomplete_attempts"),
                    "resolution": {k: v for k, v in (o.get("resolution") or {}).items()
                                   if k in {"reason", "spontaneous", "latency_s"}}}
                   for o in settled.get("obligations") or [] if isinstance(o, dict)]
    gate = [{"t_s": run.rel(e.get("time")), "event": e["event"], "calls": len(e.get("calls") or []),
             "lessons_given": e.get("lessons_given"), "full": e.get("full"), "delay_s": e.get("delay_s")}
            for e in events if e.get("event") in {"lesson_gate_held", "lesson_gate_released", "lesson_gate_undone"}]
    reads = Counter(e.get("role") for e in events if e.get("event") == "board_read_attempt")
    record = {
        "schema": SCHEMA, "protocol": "agent_rules_v1", "replay": "ok", "notes": list(outer_text.EVIDENCE_NOTES),
        "policies": _policies(state, events, run),
        "proposals": _proposals(state, run),
        "runtime_refused": _refused(events, run, coverage),
        "lessons": [{"id": lesson["id"], "author": lesson["author"], "at": lesson["at"],
                     "t_s": run.rel_iso(lesson["at"]), "trigger": lesson["trigger"], "text": text_leaf(lesson["text"])}
                    for lesson in state["lessons"]],
        "lesson_gate": gate,
        "board": [{"id": p["id"], "author": p["author"], "to": p["to"], "at": p["at"], "t_s": run.rel_iso(p["at"]),
                   "text": text_leaf(p["text"])} for p in state.get("board") or []],
        "board_read_attempts": dict(reads),
        "rule_obligations": {"stats": _redacted(settled.get("stats")), "windows": settled.get("windows"),
                             "obligations": _limited(obligations, coverage, "obligations")},
        "reflection": {"stats": settled.get("reflection"),
                       "prompts": _limited(_funnel(events, state, run), coverage, "prompts")},
        "organization_timeline": _timeline(run, coverage),
        "resources": _resources(root),
        "information_use": information_use(events, run),
        "coverage": coverage,
    }
    json.dumps(record, allow_nan=False)
    return record
