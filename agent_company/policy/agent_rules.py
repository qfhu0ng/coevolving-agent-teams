"""agent_rules_v1 grammar: rules, spawn policy blocks and control headers.

Parsing, validation and rendering only; no I/O and no runtime state. Rules name
only events the runtime can observe exactly (plan v4 section 4.2); the only
action is SEND. Roster checks (does a recipient exist?) belong to the store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
import re

from agent_company.policy import agent_rules_text as text

TRIGGERS = ("commit", "task_done", "task_claimed", "message", "shutdown_request", "wall_time", "start")
WALL_TIME_FRACTIONS = (50, 75, 90)
FIELDS = ("sha", "paths", "checks", "remaining", "blocker", "eta")
FIELD_LABELS = {name: name.upper() + ":" for name in FIELDS}
NAME = r"[A-Za-z][A-Za-z0-9_-]{0,63}"
NAME_RE = re.compile(NAME + r"\Z")
LEADER = "leader"
EVERYONE = "*"
# The team board (plan v4 section 8.1): a recipient only, satisfied by the owner's own BOARD: post.
BOARD = "board"
RESERVED_NAMES = frozenset({"master", "leader", "all", "broadcast", BOARD})


class RuleError(ValueError):
    """One rule line is invalid; carries a stable code for evidence and feedback."""

    def __init__(self, code: str, **values):
        self.code, self.values = code, values
        super().__init__(f"{code}: {self.detail}")

    @property
    def detail(self) -> str:
        return text.ERRORS[self.code].format(**{k: v for k, v in self.values.items()})


class ControlRejection(ValueError):
    """One tool call is rejected before execution; other calls may still run."""

    def __init__(self, code: str, detail: str, *, tool_call_id: str, **details):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.tool_call_id, self.details = code, detail, tool_call_id, details


@dataclass(frozen=True)
class Rule:
    trigger: str
    recipient: str
    fraction: int | None = None
    sender: str | None = None
    paths: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        head = "WHEN " + self.trigger
        if self.fraction is not None:
            head += f" {self.fraction}"
        if self.trigger == "message":
            head += " FROM " + (self.sender or EVERYONE)
        if self.paths:
            head += " PATHS " + ", ".join(self.paths)
        tail = " THEN SEND TO " + self.recipient
        if self.fields:
            tail += " WITH " + ", ".join(self.fields)
        return head + tail


def _party(value: str) -> str:
    if value in (EVERYONE, LEADER, BOARD):
        return value
    if not NAME_RE.fullmatch(value) or value.lower() in RESERVED_NAMES:
        raise RuleError("invalid_party", value=value)
    return value


def valid_glob(pattern: str) -> bool:
    if not pattern or "\\" in pattern or "\x00" in pattern or any(c.isspace() for c in pattern):
        return False
    path = PurePosixPath(pattern)
    return not path.is_absolute() and ".." not in path.parts


def _normal_glob(pattern: str) -> str:
    return pattern + "**" if pattern.endswith("/") else pattern


def glob_match(pattern: str, path: str) -> bool:
    """`**` matches any number of path segments, `*` and `?` stay within one."""
    parts = [p for p in _normal_glob(pattern).split("/") if p != ""]
    names = [p for p in path.split("/") if p != ""]

    def match(i: int, j: int) -> bool:
        if i == len(parts):
            return j == len(names)
        if parts[i] == "**":
            return any(match(i + 1, k) for k in range(j, len(names) + 1))
        return j < len(names) and fnmatchcase(names[j], parts[i]) and match(i + 1, j + 1)
    return match(0, 0)


_THEN = re.compile(r"[ \t]+THEN[ \t]+", re.I)
_SEND = re.compile(r"SEND[ \t]+TO[ \t]+(?P<to>\S+)(?:[ \t]+WITH[ \t]+(?P<fields>.+?))?[ \t]*\Z", re.I)


def parse_rule(line: str) -> Rule:
    """Parse one rule line; raise RuleError with the first problem found."""
    source = line.strip()
    head = re.match(r"WHEN[ \t]+", source, re.I)
    parts = _THEN.split(source[head.end():], maxsplit=1) if head else []
    if len(parts) != 2:
        raise RuleError("rule_syntax")
    left, right = parts
    tokens = left.split()
    trigger = tokens.pop(0).lower() if tokens else ""
    if trigger not in TRIGGERS:
        raise RuleError("unknown_trigger", value=trigger or "(missing)")
    fraction = None
    if tokens and tokens[0].isdigit():
        if trigger != "wall_time":
            raise RuleError("fraction_not_allowed")
        fraction = int(tokens.pop(0))
    if trigger == "wall_time" and fraction not in WALL_TIME_FRACTIONS:
        raise RuleError("wall_time_fraction")
    sender, paths = None, ()
    if tokens and tokens[0].upper() == "FROM":
        if trigger != "message":
            raise RuleError("from_not_allowed")
        if len(tokens) < 2:
            raise RuleError("rule_syntax")
        sender = _party(tokens[1])
        if sender == BOARD:
            raise RuleError("board_sender")
        tokens = tokens[2:]
    if tokens and tokens[0].upper() == "PATHS":
        if trigger != "commit":
            raise RuleError("paths_not_allowed")
        globs = [g.strip() for g in " ".join(tokens[1:]).split(",")]
        if not globs or any(not valid_glob(g) for g in globs):
            raise RuleError("invalid_glob")
        paths, tokens = tuple(dict.fromkeys(globs)), []
    if tokens:
        if tokens[0].upper() == "FROM":
            raise RuleError("from_not_allowed")
        raise RuleError("rule_syntax")
    send = _SEND.fullmatch(right.strip())
    if not send:
        raise RuleError("rule_syntax")
    recipient = _party(send["to"])
    fields = ()
    if send["fields"] is not None:
        names = [f.strip().lower() for f in send["fields"].split(",")]
        for name in names:
            if name not in FIELDS:
                raise RuleError("unknown_field", value=name or "(empty)")
        if len(set(names)) != len(names):
            raise RuleError("duplicate_field", value=next(n for n in names if names.count(n) > 1))
        fields = tuple(names)
    return Rule(trigger, recipient, fraction, (sender or EVERYONE) if trigger == "message" else None, paths, fields)


def check_roster(rule: Rule, *, owner: str, known: frozenset[str]) -> None:
    """Parties must be the master, everyone, or a teammate that has not ended."""
    if rule.recipient == owner:
        raise RuleError("self_recipient")
    for party in (rule.recipient, rule.sender):
        if party not in (None, EVERYONE, LEADER, BOARD) and party not in known:
            raise RuleError("unknown_party", value=party)


def rule_lines(body: str) -> list[str]:
    return [line.strip() for line in body.splitlines() if line.strip()]


def notes_text(body: str) -> str:
    """Free text kept as written, without surrounding blank lines."""
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# -- spawn prompt blocks ----------------------------------------------------------

_SUCCEEDS = re.compile(rf"SUCCEEDS:[ \t]*(?P<name>{NAME})[ \t]*\Z")
_NEAR_SUCCEEDS = re.compile(r"(?i)^[^A-Za-z]*succeeds\b")
_NEAR_MARKER = re.compile(r"(?i)^[^A-Za-z]*(?:end[-_ ]?policy|policy)[^A-Za-z]*\Z")


@dataclass(frozen=True)
class SpawnPolicy:
    """What a spawn prompt says about the new teammate's policy."""
    block: str  # ok | missing | unclosed | duplicate
    rules: tuple[str, ...] = ()
    notes: str | None = None
    succeeds: str | None = None
    warnings: tuple[tuple[str, dict], ...] = ()
    prompt: str = ""


def parse_spawn_policy(prompt: str) -> SpawnPolicy:
    """Never raises for policy problems: the teammate still starts (plan v4 5.1)."""
    lines = prompt.splitlines(keepends=True)
    bare = [line.rstrip("\r\n").strip() for line in lines]
    starts = [i for i, line in enumerate(bare) if line == "POLICY:"]
    ends = [i for i, line in enumerate(bare) if line == "END-POLICY"]
    warnings: list[tuple[str, dict]] = []
    block, rules, notes, removed = "missing", (), None, set()
    if len(starts) > 1:
        block = "duplicate"
        warnings.append(("duplicate_block", {}))
    elif starts:
        after = [i for i in ends if i > starts[0]]
        if not after:
            block = "unclosed"
            warnings.append(("unclosed_block", {}))
        else:
            block, start, end = "ok", starts[0], after[0]
            inner = list(range(start + 1, end))
            split = next((i for i in inner if bare[i] == "NOTES:"), None)
            rules = tuple(bare[i] for i in inner if (split is None or i < split) and bare[i])
            if split is not None:
                notes = notes_text("".join(lines[i] for i in inner if i > split))
            removed.update(range(start, end + 1))
    if any(i not in removed for i in ends) and block != "unclosed":
        if block != "duplicate":
            warnings.append(("stray_end", {}))
    inside = removed
    if any(_NEAR_MARKER.match(line) and line not in ("POLICY:", "END-POLICY")
           for i, line in enumerate(bare) if i not in inside):
        warnings.append(("near_miss_block", {}))
    exact = [(i, _SUCCEEDS.fullmatch(line)) for i, line in enumerate(bare) if i not in inside]
    exact = [(i, m) for i, m in exact if m]
    near = [i for i, line in enumerate(bare) if i not in inside and _NEAR_SUCCEEDS.match(line)
            and not _SUCCEEDS.fullmatch(line)]
    succeeds = None
    if len(exact) > 1:
        warnings.append(("duplicate_succeeds", {}))
    elif exact:
        succeeds = exact[0][1]["name"]
        if block not in ("unclosed", "duplicate"):
            # A malformed block leaves the prompt exactly as written (plan v4 5.1).
            removed.add(exact[0][0])
    if near:
        warnings.append(("malformed_succeeds", {}))
    kept = "".join(line for i, line in enumerate(lines) if i not in removed)
    return SpawnPolicy(block, rules, notes, succeeds, tuple(warnings), kept if removed else prompt)


# -- send_message controls --------------------------------------------------------

_CANDIDATE = re.compile(r"^[\s*#>`_~]*(?:agent[-_ ]?policy|policy[-_ ]?proposal|policy[-_ ]?decision)\b", re.I)
_LESSON_CANDIDATE = re.compile(r"^[\s*#>`_~]*lesson[ \t]*:", re.I)
_BOARD_CANDIDATE = re.compile(r"^[\s*#>`_~]*board[ \t]*:", re.I)
_TARGETS = rf"(?P<targets>\*|{NAME}(?:[ \t]*,[ \t]*{NAME})*)"
_OPERATION = (r"(?P<op>add|notes|(?P<rid_op>replace|remove)[ \t]+R(?P<rid>[1-9][0-9]{0,5})"
              r"|revoke[ \t]+#(?P<pid>[1-9][0-9]{0,8}))")
_EDIT = re.compile(rf"AGENT-POLICY[ \t]+{_TARGETS}[ \t]+{_OPERATION}(?:[ \t]+base=v(?P<base>0|[1-9][0-9]{{0,8}}))?[ \t]*\Z")
_PROPOSE = re.compile(rf"POLICY-PROPOSAL[ \t]+(?P<target>self|\*|{NAME})[ \t]+"
                      r"(?P<op>add|notes|(?P<rid_op>replace|remove)[ \t]+R(?P<rid>[1-9][0-9]{0,5}))[ \t]*\Z")
_DECIDE = re.compile(r"POLICY-DECISION[ \t]+#(?P<id>[1-9][0-9]{0,8})[ \t]+(?P<decision>approve|reject|amend)[ \t]*\Z")


@dataclass(frozen=True)
class EditCall:
    tool_call_id: str
    to: str
    targets: tuple[str, ...]  # ("*",) for everyone not ended at commit time
    operation: str            # add | replace | remove | notes | revoke
    rule_id: str | None = None
    proposal_id: int | None = None
    base: int | None = None
    body: str = ""


@dataclass(frozen=True)
class ProposalCall:
    tool_call_id: str
    to: str
    target: str               # self | * | teammate name
    operation: str            # add | replace | remove | notes
    rule_id: str | None
    body: str
    reason: str


@dataclass(frozen=True)
class DecisionCall:
    tool_call_id: str
    to: str
    proposal_id: int
    decision: str
    body: str = ""


@dataclass(frozen=True)
class LessonCall:
    tool_call_id: str
    to: str
    text: str


@dataclass(frozen=True)
class BoardCall:
    tool_call_id: str
    to: str
    text: str


@dataclass(frozen=True)
class SpawnCall:
    tool_call_id: str
    name: str
    policy: SpawnPolicy = field(default_factory=lambda: SpawnPolicy("missing"))


ControlCall = EditCall | ProposalCall | DecisionCall | LessonCall | BoardCall | SpawnCall


def _reject(tool_call_id, code, detail):
    raise ControlRejection(code, detail, tool_call_id=tool_call_id)


def _operation(match):
    if match["op"] in ("add", "notes"):
        return match["op"], None
    if match.groupdict().get("pid"):
        return "revoke", None
    return match["rid_op"], "R" + match["rid"]


def parse_control(tool_call_id: str, to, message, *, actor: str | None = None) -> ControlCall | None:
    """Recognize a control by its exact first line; near misses are rejected, not sent.

    The master has no lessons, so for it any LESSON: line is an ordinary message.
    Anyone may post to the board; the post is also delivered to its recipient.
    """
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("A tool call id is required")
    if not isinstance(message, str):
        return None
    first, _, body = message.partition("\n")
    first = first.rstrip("\r")
    recipient = to if isinstance(to, str) else ""
    if actor == "master" and _LESSON_CANDIDATE.match(first):
        return None
    if first.startswith("LESSON:"):
        lesson = (first[len("LESSON:"):].strip() + ("\n" + body if body.strip() else "")).strip()
        if not lesson:
            _reject(tool_call_id, "malformed_body", text.BODY_ERRORS["lesson_empty"])
        return LessonCall(tool_call_id, recipient, lesson)
    if _LESSON_CANDIDATE.match(first):
        _reject(tool_call_id, "malformed_header", text.HEADER_SYNTAX["lesson"])
    if first.startswith("BOARD:"):
        post = (first[len("BOARD:"):].strip() + ("\n" + body if body.strip() else "")).strip()
        if not post:
            _reject(tool_call_id, "malformed_body", text.BODY_ERRORS["board_empty"])
        return BoardCall(tool_call_id, recipient, post)
    if _BOARD_CANDIDATE.match(first):
        _reject(tool_call_id, "malformed_header", text.HEADER_SYNTAX["board"])
    if not _CANDIDATE.match(first):
        return None
    header = first.strip()
    if match := _EDIT.fullmatch(header):
        operation, rule_id = _operation(match)
        targets = tuple(dict.fromkeys(t.strip() for t in match["targets"].split(",")))
        return EditCall(tool_call_id, recipient, targets, operation, rule_id,
                        int(match["pid"]) if match["pid"] else None,
                        int(match["base"]) if match["base"] is not None else None, body)
    if match := _PROPOSE.fullmatch(header):
        operation, rule_id = _operation(match)
        lines = body.rstrip().splitlines()
        if not lines or not lines[-1].startswith("REASON:") or not lines[-1][7:].strip():
            _reject(tool_call_id, "malformed_body", text.BODY_ERRORS["proposal_reason"])
        return ProposalCall(tool_call_id, recipient, match["target"], operation, rule_id,
                            "\n".join(lines[:-1]), lines[-1][7:].strip())
    if match := _DECIDE.fullmatch(header):
        return DecisionCall(tool_call_id, recipient, int(match["id"]), match["decision"], body)
    kind = ("agent_policy" if re.match(r"^[\s*#>`_~]*agent", first, re.I) else
            "proposal" if re.search(r"proposal", first, re.I) else "decision")
    _reject(tool_call_id, "malformed_header", text.HEADER_SYNTAX[kind])


def edit_body(call, operation: str, body: str) -> tuple[list[str], str | None]:
    """(rule lines, notes) for one operation's body, or a malformed_body rejection."""
    lines = rule_lines(body)
    if operation == "add":
        if not lines:
            _reject(call.tool_call_id, "malformed_body", text.BODY_ERRORS["add_needs_rules"])
        return lines, None
    if operation == "replace":
        if len(lines) != 1:
            _reject(call.tool_call_id, "malformed_body", text.BODY_ERRORS["replace_needs_one_rule"])
        return lines, None
    if operation in ("remove", "revoke"):
        if lines:
            _reject(call.tool_call_id, "malformed_body", text.BODY_ERRORS[f"{operation}_takes_no_body"])
        return [], None
    if operation == "notes":
        return [], notes_text(body)
    raise ValueError("Unknown policy operation")


def related_facts(lesson: str, known: frozenset[str]) -> dict:
    """Deterministic extraction only; no model call and no interpretation."""
    words = set(re.findall(NAME, lesson))
    commits = sorted(set(re.findall(r"\b[0-9a-f]{7,40}\b", lesson)))
    paths = sorted({p for p in re.findall(r"(?<![\w/.-])([\w.-]+(?:/[\w.-]+)+)", lesson) if valid_glob(p)})
    return {"agents": sorted(words & known), "paths": paths[:20], "commits": commits[:20]}
