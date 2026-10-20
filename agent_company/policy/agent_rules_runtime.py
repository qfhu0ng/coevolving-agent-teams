"""agent_rules_v1 rendering: fixed system cards, request tails and read-only files.

System cards never change during a run (plan v4 section 6), so neither spawning
nor a policy change rewrites anyone's system prompt. Policies and status live in
the current-request tail; the full record lives in three runtime-written files.
"""
from __future__ import annotations

from agent_company.policy import agent_rules_text as text
from agent_company.policy.agent_policy_store import AgentSnapshot, PolicyVersion

# Recorded run identity only; model_visible_text_status carries the approval (M4, 2026-10-17).
DISCLOSURE_VERSION = "agent_rules_v1"
POLICIES_FILE = "agent_policies.md"
LESSONS_FILE = "team_lessons.md"
BOARD_FILE = "team_board.md"
BOARD_LINE_CHARS = 100
ROSTER_SUBJECT_CHARS = 60


def worker_card(board_file: str = "<board file>") -> str:
    return text.WORKER_CARD.format(board_file=board_file)


def master_card(policies_file: str, *, lessons_file: str = "<lessons file>", board_file: str = "<board file>") -> str:
    return text.MASTER_CARD.format(policies_file=policies_file, lessons_file=lessons_file, board_file=board_file)


def disclosure(role: str, *, policies_file: str = "<policies file>", lessons_file: str = "<lessons file>",
               board_file: str = "<board file>") -> str:
    if role == "master":
        return master_card(policies_file, lessons_file=lessons_file, board_file=board_file)
    return worker_card(board_file)


def _clock(at: str) -> str:
    """HH:MM:SS of a store timestamp (UTC), or the value itself."""
    return at[11:19] if isinstance(at, str) and len(at) >= 19 and at[10] == "T" else str(at)


def _first_line(value: str, limit: int) -> str:
    line = next((part.strip() for part in str(value).splitlines() if part.strip()), "")
    return line if len(line) <= limit else line[:limit - 3].rstrip() + "..."


def render_policy(policy: PolicyVersion, *, own: bool = True) -> str:
    header, empty = ((text.WORKER_TAIL_HEADER, text.WORKER_TAIL_EMPTY) if own else
                     (text.FILE_POLICY_HEADER, text.FILE_POLICY_EMPTY))
    if policy.rules:
        lines = [header.format(version=policy.version)]
        lines += [f"{rule_id}: {rule}" for rule_id, rule in policy.rules]
    else:
        lines = [empty.format(version=policy.version)]
    lines += [text.NOTES_HEADER, policy.notes] if policy.notes else [text.NO_NOTES]
    return "\n".join(lines)


def describe_diff(diff: dict | None, *, full_notes: bool) -> list[str]:
    """What one change did, from the diff the store recorded with that version."""
    if not diff:
        return []
    lines = [text.DIFF_ADDED.format(rule_id=k, text=v) for k, v in diff.get("added", {}).items()]
    lines += [text.DIFF_REMOVED.format(rule_id=k, text=v) for k, v in diff.get("removed", {}).items()]
    lines += [text.DIFF_REPLACED.format(rule_id=k, old=old, new=new) for k, (old, new) in diff.get("replaced", {}).items()]
    if "notes" in diff:
        old, new = diff["notes"]
        lines += ([text.DIFF_NOTES_OLD, old or text.EMPTY_NOTES, text.DIFF_NOTES_NEW, new or text.EMPTY_NOTES]
                  if full_notes else [text.DIFF_NOTES_SHORT])
    return lines


def _proposal_status(p) -> str:
    target = "self" if p.target == "self" else p.target
    version = next(iter((p.applied or {}).values()), {}).get("version")
    line = text.PROPOSAL_STATUS[p.status].format(id=p.proposal_id, target=target, operation=p.label,
                                                 version=version, reason=text.VOID_REASONS.get(p.void_reason, ""))
    return line + (text.PROPOSAL_NOTE.format(note=p.note) if p.note else "")


def roster_rows(actor: str, roster: dict | None, ended=frozenset(), known=()) -> list[dict]:
    """Active teammates other than ``actor`` (plan v4 section 8.2), as rendered.

    ``roster``: {"at": ISO time, "members": [{"name", "status", "shutdown_pending",
    "tasks": [{"id", "subject"}] or None}]} from the latest team sample, or None.
    ``known``: teammates the runtime has started but the sample may not show yet.
    """
    rows, seen = [], {actor.lower()}
    for member in (roster or {}).get("members") or []:
        name, status = member.get("name"), member.get("status")
        if not isinstance(name, str) or name.lower() in seen:
            continue
        seen.add(name.lower())
        if name in ended or status in {"completed", "failed", "cancelled"}:
            continue
        rows.append({"name": name, "status": status, "shutdown_pending": bool(member.get("shutdown_pending")),
                     "tasks": member.get("tasks")})
    rows += [{"name": name, "starting": True} for name in known
             if name.lower() not in seen and name not in ended]
    return rows


def render_roster(actor: str, roster: dict | None, ended=frozenset(), known=()) -> str:
    rows = roster_rows(actor, roster, ended, known)
    if roster is None and not rows:
        return text.WORKER_TAIL_ROSTER_UNKNOWN
    at = _clock(roster.get("at") or "") if roster else None
    if not rows:
        return text.WORKER_TAIL_ROSTER_NONE.format(at=at)
    lines = [text.WORKER_TAIL_ROSTER.format(at=at) if roster else text.WORKER_TAIL_ROSTER_PENDING]
    for row in rows:
        if row.get("starting"):
            lines.append(text.WORKER_TAIL_ROSTER_STARTING.format(name=row["name"]))
            continue
        if row["tasks"] is None:
            tasks = text.WORKER_TAIL_ROSTER_TASKS_UNKNOWN
        else:
            tasks = ", ".join(text.WORKER_TAIL_ROSTER_TASK.format(
                id=t.get("id"), subject=_first_line(t.get("subject") or "", ROSTER_SUBJECT_CHARS))
                for t in row["tasks"]) or text.WORKER_TAIL_ROSTER_NO_TASKS
        lines.append(text.WORKER_TAIL_ROSTER_ROW.format(
            name=row["name"], status=text.WORKER_TAIL_ROSTER_STATUS.get(row["status"], row["status"] or "unknown"),
            shutdown=text.WORKER_TAIL_ROSTER_SHUTDOWN if row["shutdown_pending"] else "", tasks=tasks))
    return "\n".join(lines)


def render_board_tail(snapshot: AgentSnapshot, *, seen: int, board_file: str) -> str:
    """Latest posts by others, always shown (the tail is not kept in history); new ones marked."""
    if not snapshot.board_recent:
        return text.WORKER_TAIL_BOARD_EMPTY.format(count=snapshot.board_count, board_file=board_file)
    rows = [text.WORKER_TAIL_BOARD_ROW.format(id=post.post_id, new=text.WORKER_TAIL_BOARD_NEW if post.number > seen else "",
                                              author=post.author, at=_clock(post.at),
                                              line=_first_line(post.text, BOARD_LINE_CHARS))
            for post in snapshot.board_recent]
    return "\n".join([text.WORKER_TAIL_BOARD.format(count=snapshot.board_count, board_file=board_file), *rows])


def render_worker_tail(snapshot: AgentSnapshot, *, roster: dict | None = None, ended=frozenset(), known=(),
                       board_seen: int = 0, board_file: str = "<board file>") -> str:
    policy = snapshot.own
    if policy is None:
        raise RuntimeError("A teammate tail needs that teammate's policy")
    statuses = [_proposal_status(p) for p in snapshot.proposals]
    return "\n\n".join([render_policy(policy), "\n".join(statuses) if statuses else text.NO_PROPOSALS,
                         render_roster(snapshot.actor, roster, ended, known),
                         render_board_tail(snapshot, seen=board_seen, board_file=board_file)])


def render_master_tail(snapshot: AgentSnapshot, *, remaining_seconds: float | None, lessons_file: str,
                       board_seen: int = 0, board_file: str = "<board file>", lessons_unseen: int = 0) -> str:
    active = [w for w in snapshot.workers if not w.ended]
    if active:
        rows = [text.MASTER_TAIL_POLICIES] + [
            text.MASTER_TAIL_ROW.format(name=w.name, version=w.policy.version, rules=len(w.policy.rules),
                                        notes=len(w.policy.notes),
                                        source=text.SOURCE_LABELS.get(w.policy.source, w.policy.source))
            for w in active]
    else:
        rows = [text.MASTER_TAIL_NO_TEAMMATES]
    pending = [p for p in snapshot.proposals if p.status == "pending"]
    blocks = ["\n".join(rows)]
    if pending:
        blocks.append(text.MASTER_TAIL_PENDING + "\n" + "\n\n".join(
            text.MASTER_TAIL_PROPOSAL.format(id=p.proposal_id, proposer=p.proposer, target=p.target,
                                             operation=p.label,
                                             base=", ".join(f"{k} v{v}" for k, v in sorted(p.base.items())) or "none",
                                             body=p.body or "(empty)", reason=p.reason)
            for p in pending))
    else:
        blocks.append(text.MASTER_TAIL_NO_PENDING)
    applied = [p for p in snapshot.proposals if p.status == "applied"]
    if applied:
        blocks.append(text.MASTER_TAIL_SELF + "\n" + "\n".join(
            text.MASTER_TAIL_SELF_ROW.format(
                id=p.proposal_id, name=p.proposer, operation=p.label,
                version=(p.applied or {}).get(p.proposer, {}).get("version"),
                change="; ".join(describe_diff((p.applied or {}).get(p.proposer, {}).get("diff"), full_notes=False)))
            for p in applied))
    blocks.append(text.MASTER_TAIL_LESSONS.format(count=snapshot.lesson_count, unseen=lessons_unseen,
                                                  lessons_file=lessons_file))
    blocks.append(text.MASTER_TAIL_BOARD.format(count=snapshot.board_count, new=max(0, snapshot.board_count - board_seen),
                                                board_file=board_file))
    if remaining_seconds is not None:
        blocks.append(text.MASTER_TAIL_REMAINING.format(seconds=max(0, int(remaining_seconds))))
    return "\n\n".join(blocks)


def render_tail(snapshot: AgentSnapshot, *, remaining_seconds=None, lessons_file="<lessons file>",
                board_file="<board file>", roster=None, ended=frozenset(), known=(), board_seen=0,
                lessons_unseen=0) -> str:
    """``board_seen``: board posts that existed at this agent's previous delivered tail."""
    if snapshot.actor == "master":
        return render_master_tail(snapshot, remaining_seconds=remaining_seconds, lessons_file=lessons_file,
                                  board_seen=board_seen, board_file=board_file, lessons_unseen=lessons_unseen)
    return render_worker_tail(snapshot, roster=roster, ended=ended, known=known, board_seen=board_seen,
                              board_file=board_file)


def render_policies_file(state: dict) -> str:
    """Every teammate's current policy and its full change log, for read_file."""
    out = [f"# {text.POLICIES_FILE_TITLE}", ""]
    workers = sorted(state["workers"].values(), key=lambda w: w["spawn_order"])
    if not workers:
        out.append(text.POLICIES_FILE_EMPTY)
    for worker in workers:
        current = PolicyVersion.of(worker["versions"][-1])
        status = text.POLICIES_FILE_STATUS["ended" if worker["lifecycle"] in {"terminal", "voided"} else "active"]
        out += [text.POLICIES_FILE_WORKER.format(name=worker["name"], status=status), render_policy(current, own=False),
                "", text.POLICIES_FILE_CHANGES]
        for value in worker["versions"]:
            origin = text.SOURCE_LABELS.get(value["source"], value["source"])
            extra = []
            if value.get("lineage"):
                extra.append(text.POLICIES_FILE_FROM.format(predecessor=value["lineage"]["from"],
                                                            version=value["lineage"]["version"]))
            if value.get("proposal_id"):
                extra.append(text.POLICIES_FILE_PROPOSAL.format(proposal_id=value["proposal_id"]))
            out.append(text.POLICIES_FILE_CHANGE.format(version=value["version"], at=value["at"],
                                                        operation=value["operation"], origin=origin)
                       + (f" ({', '.join(extra)})" if extra else ""))
            out += ["  " + line for detail in describe_diff(value.get("diff"), full_notes=True)
                    for line in detail.splitlines() or [""]]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_lessons_file(state: dict) -> str:
    out = [f"# {text.LESSONS_FILE_TITLE}", ""]
    if not state["lessons"]:
        out.append(text.LESSONS_FILE_EMPTY)
    for lesson in state["lessons"]:
        out += [text.LESSON_ENTRY.format(id=lesson["id"], author=lesson["author"], trigger=lesson["trigger"],
                                         at=lesson["at"], text=lesson["text"]), ""]
    return "\n".join(out).rstrip() + "\n"


def render_board_file(state: dict) -> str:
    out = [f"# {text.BOARD_FILE_TITLE}", ""]
    if not state.get("board"):
        out.append(text.BOARD_FILE_EMPTY)
    for post in state.get("board") or []:
        out += [text.BOARD_ENTRY.format(id=post["id"], author=post["author"], to=post["to"] or "(none)",
                                        at=post["at"], text=post["text"]), ""]
    return "\n".join(out).rstrip() + "\n"


def lesson_gate_same_text() -> str:
    return text.LESSON_GATE_SAME


def lesson_gate_text(lessons: list[dict], *, full: bool) -> str:
    """The spawn gate's result: lessons as each teammate's account, not instructions (plan v4 8)."""
    rows = [text.LESSON_GATE_ENTRY.format(id=lesson["id"], author=lesson["author"],
                                          trigger=text.LESSON_TRIGGERS.get(lesson.get("trigger"), lesson.get("trigger")),
                                          at=_clock(lesson["at"]), text=lesson["text"])
            for lesson in lessons]
    return text.LESSON_GATE.format(lessons="\n".join([text.LESSON_GATE_ALL if full else text.LESSON_GATE_NEW, *rows]))


def rejection_text(rejection, *, binding=False) -> str:
    return (text.BINDING_FAILURE if binding else text.CONTROL_FAILURE).format(code=rejection.code,
                                                                              detail=rejection.detail)
