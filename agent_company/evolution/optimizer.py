"""A joint strategy optimizer with selective training-trace access."""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Callable

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.programbench.total_selection import MODE as TOTAL_MODE, EVIDENCE_INSTRUCTION, OBJECTIVE_INSTRUCTION, NOTE_INSTRUCTION
from agent_company.evolution.loop import PolicyProposal, ProposalRequest
from agent_company.runtime.relay import OpenAIChatTransport, ProviderConfig, TeamRelay, usage_and_cost
from agent_company.evolution.outer_budget import OuterBudget, WorkingMemory, require_complete_output
from agent_company.evolution.outer_submission import submission_tool_choice
from agent_company.evolution.trace import without_reasoning
from agent_company.evolution import agent_rules_outer_text as rules_text
from agent_company.runtime.transport import InvalidProviderResponse, RequestCancelled, RequestControl, TransportError
from agent_company.runtime_io import write_json_atomic


class PolicyOptimizationError(RuntimeError):
    def __init__(self, message: str, *, usage: dict, cost_usd: float | None,
                 elapsed_seconds: float, output_dir: Path, known_cost_usd: float | None = None):
        super().__init__(message)
        self.usage = usage
        self.cost_usd = cost_usd
        self.elapsed_seconds = elapsed_seconds
        self.output_dir = str(output_dir)
        self.known_cost_usd = known_cost_usd


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}


PAGE = {"offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1}}
LOCATION = {"path": {"type": "string", "description": "Training-relative file or task/subdirectory path."},
            "name": {"type": "string", "description": "Literal filename substring, e.g. 00720 or execution.json."},
            "source_kind": {"type": "string", "enum": ["all", "artifact", "prompt_copy"]}}
TOOLS = [
    _tool("list_training_files", "Locate files by task/subdirectory and filename substring; paginate within that selection.",
          {**LOCATION, **PAGE}, []),
    _tool("read_training_file", "Read a bounded character page from a training file.",
          {"path": {"type": "string"}, **PAGE}, ["path"]),
    _tool("search_training_files", "Find literal text, defaulting to non-request artifacts. Opt into prompt_copy to inspect model exposure; copies are not independent incidents.",
          {"query": {"type": "string"}, **LOCATION, **PAGE}, ["query"]),
    _tool("submit_policy", "Finish with both full replacement strategies and their joint rationale.",
          {"assignment_policy": {"type": "string"}, "communication_policy": {"type": "string"},
           "rationale": {"type": "string"}},
          ["assignment_policy", "communication_policy", "rationale"]),
]

NOTE_TOOL = _tool("record_research_note", "Save a concise observation or hypothesis with current-batch evidence paths; survives context rotation.",
    {"note": {"type": "string"}, "evidence_refs": {"type": "array", "items": {"type": "string"}}},
    ["note", "evidence_refs"])
TOOLS.insert(-1, NOTE_TOOL)

READ_JSON_TOOL = _tool(
    "read_training_json",
    "Open one part of a training JSON file by RFC 6901 pointer, e.g. /by_role or /entries. "
    "Array items are returned whole; page with start/count. Oversized objects list their keys.",
    {"path": {"type": "string"}, "pointer": {"type": "string"},
     "start": {"type": "integer", "minimum": 0}, "count": {"type": "integer", "minimum": 1}},
    ["path"])
CITATIONS = {"evidence_citations": {
    "type": "array",
    "description": "Verbatim support for the rationale's evidence claims. Each quote (20–600 characters) "
                   "is checked against the cited file; any unmatched quote rejects the submission.",
    "items": {"type": "object", "additionalProperties": False, "required": ["path", "quote", "supports"],
              "properties": {"path": {"type": "string"}, "quote": {"type": "string"},
                             "supports": {"type": "string"}}}}}

EXECUTION_NOTES = {"execution_notes": {
    "type": "array",
    "description": "Executability protocol: how each new or changed rule (and any known remaining conflict) "
                   "executes on target_harness. Empty only when nothing changed and nothing conflicts.",
    "items": {"type": "object", "additionalProperties": False,
              "required": ["policy_part", "rule_quote", "actor", "observable_trigger", "existing_action",
                           "capability_refs", "timing_and_fallback", "cost_and_limitations"],
              "properties": {
                  "policy_part": {"type": "string", "enum": ["assignment", "communication"]},
                  "rule_quote": {"type": "string", "description": "Short verbatim quote of the rule from the "
                                                                  "submitted policy text."},
                  "actor": {"type": "string"}, "observable_trigger": {"type": "string"},
                  "existing_action": {"type": "string"},
                  "capability_refs": {"type": "array", "items": {"type": "string"},
                                      "description": "JSON pointers into target_harness, e.g. /wakeups."},
                  "timing_and_fallback": {"type": "string"}, "cost_and_limitations": {"type": "string"}}}}}
EXECUTION_NOTE_FIELDS = tuple(EXECUTION_NOTES["execution_notes"]["items"]["required"])
MAX_EXECUTION_NOTES = 40
EXECUTABILITY_GUIDE = """
Executability protocol. The request carries target_harness: the configuration and
implementation the candidate will run under. Design A/C within it. submit_policy takes
execution_notes: for each new or changed mechanism, the rule quoted verbatim from your
policy, its actor, how the actor observes the trigger, the existing action used, the
target_harness fields that support it (JSON pointers), the timing tolerance and fallback,
and the cost. Notes are checked for structure, quotes and pointers only; they are claims
for the reviewer, not proof. Put tolerances and fallbacks in the policy text itself.
"""

WORKING_MEMORY_GUIDE = """
Working-memory research mode. evidence_index maps every available training file: sizes,
JSON key sizes and descriptive role/tool counts (shape, not findings). Plan coverage across
all training tasks before deep reading; do not spend the session on the first task. Model
calls, not pages, are the scarcer budget: request several independent evidence reads in
one turn. Open specific parts with read_training_json pointers or read_training_file pages;
follow next_offset/next_start when a page was shortened. The budget notice reports
reading_coverage per task. Evidence results leave the working context after a short
window and the context may rotate: save findings promptly with record_research_note.
Notes are the durable memory.
"""

CITATION_GUIDE = """
Evidence citations. submit_policy takes evidence_citations: cited training path, a quote
copied verbatim from that file (20–600 characters) and the claim it supports. Quotes are
checked against the files; a submission with any unmatched quote is rejected and must be
corrected. Cite only text you actually read. Rationale identifiers that never appeared in
evidence shown to you are reported to reviewers as unsupported.
"""

# Reference-like tokens (containing a digit or underscore), e.g. attempt ids.
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,}")
# The task relay's retryable provider statuses (TeamRelay._retryable).
RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504, 520, 522, 524, 525})


def _tools(working_memory: bool, citations: bool) -> list[dict]:
    if not (working_memory or citations):
        return TOOLS  # Default protocol: unchanged tool surface.
    tools = copy.deepcopy(TOOLS)
    if working_memory:
        tools[0]["function"]["parameters"]["properties"]["digest"] = {
            "type": "boolean", "description": "Add each file's descriptive shape."}
        tools.insert(2, READ_JSON_TOOL)
    submit = tools[-1]["function"]["parameters"]
    submit["properties"].update(CITATIONS)
    if citations:
        submit["required"] = submit["required"] + ["evidence_citations"]
    return tools


def unmatched_identifiers(rationale: str, shown: str) -> list[str]:
    """Reference-like rationale tokens absent from everything the model was shown.

    A disclosure for reviewers, not a gate: coined names can be legitimate, but
    PB1's rationale cited attempt ids and worker names found in no evidence.
    """
    tokens = {token for token in IDENTIFIER.findall(rationale)
              if re.search(r"[A-Za-z]", token) and re.search(r"[\d_]", token)}
    return sorted(token for token in tokens if token not in shown)[:50]

SYSTEM = """Jointly improve the assignment and communication strategies of coding agents.
The model, tool implementation, role permissions, resource limits and benchmark are fixed.
Assignment controls decomposition, worker count, scope, dependencies, reassignment
and final integration. Communication controls triggers, recipients, information
content, requests and handoffs. Keep the two strategies consistent. The master
remains coordination-only and delegates implementation and testing to workers.
Submit both full policy texts, even if one is unchanged in a particular update.
Learn reusable behavior from
training task outcomes and execution evidence. Do not include task solutions,
repository-specific patches, hidden test contents or benchmark answers in policy.
Never claim an observed message caused a score change solely from time ordering.
Consider useful exchanges, failed/missing exchanges, late messages, duplicated
work, unbalanced assignments, idle workers, integration failures and successes.

Follow selection_objective: same wall-clock cap, quality first, no score loss for
savings; then the supplied efficiency metric (normally complete total tokens),
then agent time at equal efficiency. Tokens are not dollars or GPU compute.
Use efficiency summaries to locate excess rounds, repeated work and handoffs.
Repeated commands are not proven waste: inspect revisions and outcomes. Provider
times overlap; they are not task latency or idle time. Request copies are not actions.
Simplify or DELETE redundant roles/messages/rules; more process is not inherently
better. Preserve necessary tests, error feedback, source restrictions and shutdown.
Policy cannot turn the coordination-only master into a coding single agent.
Propose a coherent testable change. In rationale give evidence references, observed
facts versus causal hypotheses, additions/deletions, expected score/calls/cost/time
effects, risks and replay falsification criteria. Do not invent savings. Unchanged
full policies are allowed if evidence is insufficient. Keep task details out of policy.

Investigate failed collaboration, not message counts alone: duplicated investigation,
stale interfaces after a message, blocked workers without targeted requests, handoffs
that never enter delivery, or a master acting as an unnecessary message relay.
Let A and C jointly change worker count, reuse, dependencies, direct peer requests,
escalation, integration and necessary independent checks. Do not mandate extra roles,
more messages or fewer workers. Broad coherent changes are allowed when grounded.
Save observations, hypotheses and unresolved questions using record_research_note.
Notes survive a bounded fresh submission session without adding model-call rounds.

Start from the compact batch results and current-attempt evidence entrypoints. Inspect
only relevant training evidence with the supplied list/read/search tools. These
tools are the entire evidence interface; no test-set artifacts are available.
Look for receipt and subsequent action when making a claim about a message.
Keep both policies applicable to unseen tasks, and distinguish transport limits
from behavior the agents can change. Follow the selection_mode described in the
request: review_replay tests a candidate on the same training batch before
adoption, whereas legacy_no_replay adopts valid submissions without that check.
Same-batch replay is training selection, not held-out generalization evidence.
A proposed change is a hypothesis, not a verified gain. Submit both complete policies and a
concise evidence-based rationale covering their interaction with submit_policy.
"""
SYSTEM += "\n" + EVIDENCE_INSTRUCTION + "\n"


AGENT_RULES_REPLACEMENTS = (
    ("Jointly improve the assignment and communication strategies of coding agents.", rules_text.OPTIMIZER_OPENING),
    ("Assignment controls decomposition, worker count, scope, dependencies, reassignment\n"
     "and final integration. Communication controls triggers, recipients, information\n"
     "content, requests and handoffs.", rules_text.OPTIMIZER_PARTS),
    ("Consider useful exchanges, failed/missing exchanges, late messages, duplicated\n"
     "work, unbalanced assignments, idle workers, integration failures and successes.\n", ""),
    ("Simplify or DELETE redundant roles/messages/rules; more process is not inherently\nbetter.",
     rules_text.OPTIMIZER_SIMPLIFY),
    ("Investigate failed collaboration, not message counts alone: duplicated investigation,\n"
     "stale interfaces after a message, blocked workers without targeted requests, handoffs\n"
     "that never enter delivery, or a master acting as an unnecessary message relay.\n"
     "Let A and C jointly change worker count, reuse, dependencies, direct peer requests,\n"
     "escalation, integration and necessary independent checks. Do not mandate extra roles,\n"
     "more messages or fewer workers. Broad coherent changes are allowed when grounded.", rules_text.OPTIMIZER_NO_MANDATE),
    ("Start from the compact batch results and current-attempt evidence entrypoints. Inspect", rules_text.OPTIMIZER_START),
)


def agent_rules_system(system: str) -> str:
    """The optimizer system for agent_rules_v1 M generation (plan v4 section 12, items 2 and 6).

    The earlier objective and selection-mode paragraphs describe replay and token-efficiency
    selection, which this entry does not use; they are replaced, and the object paragraph is added.
    """
    start = system.index("Follow selection_objective:")
    end = system.index("Repeated commands are not proven waste", start)
    system = system[:start] + rules_text.OPTIMIZER_OBJECTIVE + "\n" + system[end:]
    system = system.replace("expected score/calls/cost/time\neffects", "expected score effects")
    system = system.replace("replay falsification criteria", "falsification criteria")
    start = system.index("Follow the selection_mode described in the")
    end = system.index("A proposed change", start)
    system = system[:start] + rules_text.OPTIMIZER_PROTOCOL + "\n" + system[end:]
    system = system.replace("all training tasks before deep reading", "all available evidence directories before deep reading")
    system = system.replace("the first task", "the first directory")
    system = system.replace("reading_coverage per task", "reading_coverage per evidence directory")
    # The object is M, not rules all agents read; roles no longer exist; and the earlier lists of problems to
    # look for and of levers to change are removed for this protocol (user, 2026-10-18). Method sentences stay.
    for old, new in AGENT_RULES_REPLACEMENTS:
        if old not in system:
            raise ValueError("Optimizer system changed; agent_rules_v1 replacement text not found")
        system = system.replace(old, new)
    return system + "\n" + rules_text.OPTIMIZER_OBJECT + "\n"


def retry_delay(provider: ProviderConfig, transport_retries: int, retries: int,
                control: RequestControl, response=None) -> float | None:
    """Delay before resending, or None when retries or the shared deadline are spent."""
    if retries >= transport_retries:
        return None
    delay = provider.retry_backoff_seconds * (2 ** (retries + 1) - 1)
    hint = TeamRelay._provider_retry_after(response)
    if hint is not None:
        delay = max(delay, hint)
    if time.monotonic() + delay >= control.deadline:
        return None
    return delay


def send_with_retries(completion: Callable, payload: dict, provider: ProviderConfig, *,
                      transport_retries: int, sleep: Callable[[float], None], usage: dict,
                      account: Callable[[object], None],
                      record_retry: Callable[[int, dict], None]) -> tuple[int, object, RequestControl]:
    """Send one logical outer-loop request, resending it after a retryable failure.

    Shared by the optimizer and the reviewer. All attempts share one
    request_timeout deadline (backoff included); every attempt counts in
    usage["provider_attempts"]; an attempt whose usage the provider did not
    report counts in usage["unknown_usage_attempts"]; account(raw_usage) records
    a reply's usage, including a retryable error reply's. Cancellation,
    deadlines, invalid responses and programming errors are never resent.
    The caller persists the returned response, then calls control.check() so a
    late reply is rejected after its usage is kept. record_retry(n, record) is
    called once when retry n is scheduled ("resent": false) and again when it is
    actually sent ("resent": true).
    """
    control = RequestControl(provider.request_timeout_seconds)
    retries, scheduled = 0, None
    while True:
        # Check again after backoff, before counting or sending a retry.
        control.check()
        if scheduled is not None:
            # Only now is the resend sent: a deadline or cancellation during
            # backoff leaves the record "resent": false for recovered_usage.
            record_retry(retries, {**scheduled, "resent": True})
        usage["provider_attempts"] += 1
        response = None
        try:
            if isinstance(completion, OpenAIChatTransport):
                status, response = completion(payload, control=control)
            else:
                # Preserve the one-argument API of injected completions.
                status, response = completion(payload)
        except Exception as exc:
            usage["unknown_usage_attempts"] += 1
            # TimeoutError (including RequestDeadlineExceeded) is an OSError;
            # InvalidProviderResponse is a TransportError. Neither is a
            # retryable connection failure.
            if (isinstance(exc, (RequestCancelled, TimeoutError, InvalidProviderResponse))
                    or not isinstance(exc, (TransportError, ConnectionError, OSError))):
                raise
            delay = retry_delay(provider, transport_retries, retries, control)
            if delay is None:
                raise
            failure = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            account(response.get("usage") if isinstance(response, dict) else None)
            delay = (retry_delay(provider, transport_retries, retries, control, response)
                     if status in RETRYABLE_HTTP else None)
            if delay is None:
                return status, response, control
            failure = {"http_status": status, "response": response}
        # The same request is resent; a failed attempt's usage stays unknown.
        retries += 1
        scheduled = {**failure, "delay_seconds": delay}
        record_retry(retries, {**scheduled, "resent": False})
        control.check()
        sleep(min(delay, max(0, control.deadline - time.monotonic())))


def recovered_usage(directory: Path, rates) -> tuple[dict, float]:
    """Usage and known cost of an interrupted session, from its saved records.

    request_NNN.json marks a step's first send. Each retry_NNN_MM.json is one
    failed attempt (a retryable HTTP reply may carry its usage) and says whether
    the resend it scheduled was sent. The step's final attempt, whose usage is in
    response_NNN.json, exists only if nothing was retried or the last resend was
    sent; records from before "resent" existed count as sent. Attempts without
    readable provider usage are unknown, not zero.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0,
             "total_tokens": 0, "provider_attempts": 0, "unknown_usage_attempts": 0}
    cost = 0.0

    def add(read):
        nonlocal cost
        usage["provider_attempts"] += 1
        try:
            normalized, amount = usage_and_cost(read(), rates)
        except (OSError, KeyError, TypeError, ValueError, AttributeError):
            usage["unknown_usage_attempts"] += 1
            return
        for name, value in normalized.items():
            usage[name] += value
        cost += float(amount) if amount is not None else 0.0

    def load(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # Unreadable: its usage is unknown and its resend assumed sent.

    for request in sorted(directory.glob("request_*.json")):
        step = request.name[len("request_"):-len(".json")]
        records = [load(path) for path in sorted(directory.glob(f"retry_{step}_*.json"))]
        for record in records:
            add(lambda record=record: record["response"]["usage"])
        last = records[-1] if records else None
        if not records or not isinstance(last, dict) or last.get("resent", True):
            response = request.with_name(f"response_{step}.json")
            add(lambda response=response: json.loads(response.read_text(encoding="utf-8"))["usage"])
    return usage, cost


def redact_secret(value, secret: str | None):
    """Remove a credential from nested data, raw and as JSON-escaped text."""
    if isinstance(value, str):
        if secret:
            for encoded in (secret, json.dumps(secret, ensure_ascii=False)[1:-1],
                            json.dumps(secret, ensure_ascii=True)[1:-1]):
                value = value.replace(encoded, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact_secret(key, secret): redact_secret(item, secret) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secret(item, secret) for item in value]
    return value


class StrategyOptimizer:
    """Each batch starts a fresh context; interactions remain on disk without reasoning (B1)."""

    def __init__(self, provider: ProviderConfig, output_dir: Path, *,
                 completion: Callable | None = None,
                 max_tool_rounds: int = 32, max_context_chars: int = 180_000,
                 max_policy_chars: int = 24_000,
                 working_memory: WorkingMemory | None = None,
                 require_citations: bool = False,
                 transport_retries: int = 0, sleep: Callable[[float], None] = time.sleep):
        """working_memory, require_citations and transport_retries are opt-in.

        Leaving them unset keeps the tools, paging and single-attempt requests
        of earlier Stage1 runs. Every entry point now appends the evidence-not-
        instructions clause to the earlier system prompt; this is a prompt change.
        The shared record_research_note invalid-reference error also changed for
        legacy entries to refer to the supplied evidence interface.
        Non-TOTAL followup change_history uses the selected-baseline source wording.
        Every study entry point uses the model-minimal layer projection, with only
        reasons retained under coverage.score_layers.
        The TOTAL protocol also replaces the objective and protocol wording.
        transport_retries resends the same request after a transport error or a
        retryable HTTP status, like the task relay: exponential backoff (or the
        provider's structured retry_after), all within one request_timeout.
        """
        if min(max_tool_rounds, max_context_chars, max_policy_chars) < 1:
            raise ValueError("Optimizer context and interaction limits must be positive")
        if not isinstance(require_citations, bool):
            raise ValueError("require_citations must be true or false")
        if isinstance(transport_retries, bool) or not isinstance(transport_retries, int) or transport_retries < 0:
            raise ValueError("transport_retries must be a nonnegative integer")
        self.transport_retries, self.sleep = transport_retries, sleep
        self.provider, self.output_dir = provider, Path(output_dir)
        self.completion = completion or OpenAIChatTransport(provider)
        self.max_tool_rounds, self.max_context_chars = max_tool_rounds, max_context_chars
        self.max_policy_chars = max_policy_chars
        self.working_memory, self.require_citations = working_memory, require_citations
        self.audited = working_memory is not None or require_citations
        self.tools = _tools(working_memory is not None, require_citations)
        self.system = (SYSTEM + (WORKING_MEMORY_GUIDE if working_memory else "")
                       + (CITATION_GUIDE if require_citations else ""))

    def _citations(self, value, evidence: TrainingEvidence) -> list[dict]:
        """Verbatim-quote check for claimed evidence; wording, not truth, is verified."""
        if value in (None, []) and not self.require_citations:
            return []
        if not isinstance(value, list) or not 1 <= len(value) <= 40:
            raise ValueError("evidence_citations must list 1–40 verbatim quotes from files you read")
        verified, failures = [], []
        for index, item in enumerate(value):
            try:
                if (not isinstance(item, dict) or set(item) - {"path", "quote", "supports"}
                        or any(not isinstance(item.get(key), str) or not item[key].strip()
                               for key in ("path", "quote", "supports"))
                        or len(item["supports"]) > 500):
                    raise ValueError("needs nonempty path, quote and supports (at most 500 characters)")
                check = evidence.verify_quote(item["path"], item["quote"])
                if not check["verified"]:
                    raise ValueError("quote not found verbatim in the cited file")
                verified.append({"path": check["path"], "quote": item["quote"], "supports": item["supports"]})
            except (ValueError, TypeError, OSError) as exc:
                failures.append({"index": index, "error": str(exc),
                                 "path": item.get("path") if isinstance(item, dict) else None})
        if failures:
            raise ValueError("Citation check failed; correct or remove these citations and resubmit: "
                             + json.dumps(failures, ensure_ascii=False)[:4000])
        return verified

    @staticmethod
    def _execution_notes(raw, assignment: str, communication: str, harness: dict) -> list[dict]:
        """Structure-only checks: fields, verbatim rule quotes, resolvable target_harness pointers."""
        from agent_company.evolution.harness_capabilities import resolve_pointer
        if not isinstance(raw, list) or len(raw) > MAX_EXECUTION_NOTES:
            raise ValueError(f"execution_notes must be a list of at most {MAX_EXECUTION_NOTES} notes")
        texts = {"assignment": " ".join(assignment.split()), "communication": " ".join(communication.split())}
        notes = []
        for index, note in enumerate(raw):
            if not isinstance(note, dict) or set(note) != set(EXECUTION_NOTE_FIELDS):
                raise ValueError(f"execution_notes[{index}] must have exactly: " + ", ".join(EXECUTION_NOTE_FIELDS))
            if not isinstance(note["policy_part"], str) or note["policy_part"] not in texts:
                raise ValueError(f"execution_notes[{index}].policy_part must be assignment or communication")
            for name in EXECUTION_NOTE_FIELDS:
                if name not in {"policy_part", "capability_refs"} and (
                        not isinstance(note[name], str) or not note[name].strip()):
                    raise ValueError(f"execution_notes[{index}].{name} must be nonempty text")
            quote = " ".join(note["rule_quote"].split())
            if not 8 <= len(quote) <= 600 or quote not in texts[note["policy_part"]]:
                raise ValueError(f"execution_notes[{index}].rule_quote must be 8-600 characters copied "
                                 f"verbatim from the submitted {note['policy_part']} policy")
            refs = note["capability_refs"]
            if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
                raise ValueError(f"execution_notes[{index}].capability_refs must list target_harness pointers")
            for ref in refs:
                try:
                    resolve_pointer(harness, ref)
                except KeyError:
                    raise ValueError(f"execution_notes[{index}] cites {ref!r}, which is not a target_harness field") from None
            notes.append(dict(note))
        return notes

    def recover(self, request: ProposalRequest) -> PolicyProposal:
        """Recover a durable submission; otherwise retain partial usage without a new call."""
        directory = self.output_dir / request.storage_key
        committed = directory / "proposal.json"
        if committed.exists():
            return PolicyProposal(**json.loads(committed.read_text()))
        usage, cost = recovered_usage(directory, self.provider.rates)
        known = cost if self.provider.rates and self.provider.rates.currency == "USD" else None
        raise PolicyOptimizationError(
            "Optimizer interrupted before durable submission; previous policy pair retained without replay",
            usage=usage, cost_usd=None, known_cost_usd=known,
            elapsed_seconds=0, output_dir=directory)

    def __call__(self, request: ProposalRequest) -> PolicyProposal:
        started = time.monotonic()
        directory = self.output_dir / request.storage_key
        directory.mkdir(parents=True, exist_ok=False)
        prompt = request.as_prompt_data()
        if self.working_memory is not None:
            prompt["evidence_index"] = request.evidence.index()
        protocol = request.executability_protocol
        system, tools_for_request = self.system, self.tools
        if protocol:
            system = self.system + EXECUTABILITY_GUIDE
            tools_for_request = copy.deepcopy(self.tools)
            submit = tools_for_request[-1]["function"]["parameters"]
            submit["properties"].update(EXECUTION_NOTES)
            submit["required"] = submit["required"] + ["execution_notes"]
        if prompt.get("selection_objective", {}).get("rule_version") == TOTAL_MODE:
            start = system.index("Follow selection_objective:")
            end = system.index("Repeated commands are not proven waste", start)
            system = system[:start] + OBJECTIVE_INSTRUCTION + "\n" + system[end:]
            system = system.replace("expected score/calls/cost/time\neffects", "expected score effects")
            system = system.replace("replay falsification criteria", "falsification criteria")
            start = system.index("Follow the selection_mode described in the")
            end = system.index("A proposed change", start)
            system = system[:start] + (
                "This request uses the official total-score confirmation protocol in selection_objective. "
                "Submission proposes a candidate; it does not adopt it. Same-task evidence does not "
                "establish held-out generalization.\n") + system[end:]
            system = system.replace("all training tasks before deep reading", "all available evidence directories before deep reading")
            system = system.replace("the first task", "the first directory")
            system = system.replace("reading_coverage per task", "reading_coverage per evidence directory")
            tools_for_request = copy.deepcopy(tools_for_request)
            for tool in tools_for_request:
                if tool["function"]["name"] == "record_research_note":
                    tool["function"]["description"] = NOTE_INSTRUCTION
        if prompt.get("selection_objective", {}).get("rule_version") == rules_text.MODE:
            system, tools_for_request = agent_rules_system(system), copy.deepcopy(tools_for_request)
            for tool in tools_for_request:
                if tool["function"]["name"] == "record_research_note":
                    tool["function"]["description"] = NOTE_INSTRUCTION
                elif tool["function"]["name"] == "submit_policy":
                    tool["function"]["description"] = rules_text.OPTIMIZER_SUBMIT_NOTE
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0,
                 "total_tokens": 0, "provider_attempts": 0, "unknown_usage_attempts": 0}
        cost = 0.0
        rates = self.provider.rates
        budget = OuterBudget(system, self.max_context_chars, self.max_tool_rounds,
                             "submit_policy", self.provider.max_output_tokens,
                             self.provider.inspection_max_output_tokens, memory=self.working_memory)
        if self.audited:
            budget.totals, budget.measure = request.evidence.totals(), request.evidence.measure

        def write(name, value, *, conversation=False):
            # Reasoning stays in memory for the provider and never reaches disk (plan v4 12, B1).
            text = json.dumps(without_reasoning(value, message=conversation), ensure_ascii=False, indent=2)
            if self.provider.api_key:
                text = text.replace(self.provider.api_key, "[REDACTED]")
            (directory / name).write_text(text + "\n", encoding="utf-8")

        def costs():
            if rates is None or rates.currency != "USD" or usage["unknown_usage_attempts"]:
                return None
            return cost

        def known_cost():
            return cost if rates is not None and rates.currency == "USD" else None

        def account(raw):
            nonlocal cost
            if raw is None:
                usage["unknown_usage_attempts"] += 1
                return
            try:
                normalized, amount = usage_and_cost(raw, rates)
                for name, tokens in normalized.items():
                    usage[name] += tokens
                if amount is not None:
                    cost += float(amount)
            except (KeyError, TypeError, ValueError):
                usage["unknown_usage_attempts"] += 1

        try:
            for step in range(self.max_tool_rounds):
                submission_only = budget.prepare(messages, step)
                write("research_notes.json", budget.state())
                if self.audited:
                    write("reading_ledger.json", budget.reading_audit())
                consolidating = budget.phase == "consolidate"
                tools = (tools_for_request[-1:] if submission_only else
                         tools_for_request[-2:] if consolidating else tools_for_request)  # [..., note, submit]
                payload = {**self.provider.extra_body, "model": self.provider.model,
                           "messages": messages, "tools": tools,
                           "stream": self.provider.stream_response}
                if payload["stream"]:
                    payload["stream_options"] = {"include_usage": True}
                if submission_only:
                    submission_tool_choice(payload, self.provider, "submit_policy")
                if budget.output_limit() is not None:
                    payload["max_tokens"] = budget.output_limit()
                write(f"request_{step:03d}.json", payload)
                status, response, control = send_with_retries(
                    self.completion, payload, self.provider, transport_retries=self.transport_retries,
                    sleep=self.sleep, usage=usage, account=account,
                    record_retry=lambda retries, record: write(f"retry_{step:03d}_{retries:02d}.json", record))
                # A failing trace write must not erase usage already observed
                # in the provider's reply and reported to the controller.
                write(f"response_{step:03d}.json", response)
                # Injected completions cannot be preempted, but a late success
                # must not be accepted. Keep its response and usage for accounting.
                control.check()
                if status != 200:
                    raise RuntimeError(f"Optimizer provider returned HTTP {status}")
                message = response["choices"][0]["message"]
                # DeepSeek thinking requires the actual reasoning_content on
                # later tool requests, even after a plain-text assistant turn.
                # It stays in this in-memory list only; write() drops it.
                messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    require_complete_output(response, "Optimizer")
                    messages.append({"role": "user", "content":
                                     "Use submit_policy to return both complete replacement policies."})
                    write(f"conversation_{step:03d}.json", messages, conversation=True)
                    continue
                submitted = None
                results = []
                for index, call in enumerate(calls):
                    try:
                        name = call["function"]["name"]
                        args = json.loads(call["function"]["arguments"])
                        if name == "submit_policy":
                            assignment, policy = args["assignment_policy"], args["communication_policy"]
                            rationale = args["rationale"]
                            if (any(not isinstance(text, str) or not text.strip()
                                    or len(text) > self.max_policy_chars for text in (assignment, policy))
                                    or not isinstance(rationale, str)):
                                raise ValueError("Both policies must be nonempty text within the size limit")
                            citations = (self._citations(args.get("evidence_citations"), request.evidence)
                                         if self.audited else [])
                            notes = (self._execution_notes(args.get("execution_notes"), assignment, policy,
                                                           request.target_harness) if protocol else None)
                            submitted = (assignment, policy, rationale, citations, notes)
                            result = {"submitted": True}
                        elif submission_only:
                            raise ValueError("Evidence reading is closed to reserve the final submission; use submit_policy")
                        elif consolidating and name != "record_research_note":
                            raise ValueError("Evidence reading is paused for this consolidation call; "
                                             "record notes now")
                        elif name == "list_training_files":
                            result = request.evidence.list_files(**args)
                        elif name == "record_research_note":
                            result = budget.record_note(evidence=request.evidence, **args)
                        elif name == "read_training_file":
                            result = request.evidence.read_file(**args)
                        elif name == "read_training_json" and self.working_memory is not None:
                            result = request.evidence.read_json(**args)
                        elif name == "search_training_files":
                            result = request.evidence.search(**args)
                        else:
                            raise ValueError("Unknown evidence tool")
                    except (KeyError, ValueError, TypeError, OSError) as exc:
                        result = {"error": str(exc)}
                    results.append({"tool_call_id": call["id"], "result": result})
                    budget.append_result(messages, call["id"], result, len(calls) - index - 1)
                write(f"tool_results_{step:03d}.json", results)
                write("research_notes.json", budget.state())
                write(f"conversation_{step:03d}.json", messages, conversation=True)
                require_complete_output(response, "Optimizer")
                if submitted:
                    assignment, policy, rationale, citations, notes = submitted
                    if self.provider.api_key:
                        assignment = assignment.replace(self.provider.api_key, "[REDACTED]")
                        policy = policy.replace(self.provider.api_key, "[REDACTED]")
                        rationale = rationale.replace(self.provider.api_key, "[REDACTED]")
                    elapsed = time.monotonic() - started
                    audit = {}
                    if self.audited:
                        shown = "\n".join([json.dumps(budget.initial, ensure_ascii=False), *budget.seen,
                                           json.dumps(budget.notes, ensure_ascii=False)])
                        audit = {**budget.reading_audit(), "citations": citations,
                                 "unmatched_rationale_identifiers": unmatched_identifiers(rationale, shown)}
                        # Verified quotes come from evidence, which may contain this key. Only the
                        # optimizer knows it; the audit goes to proposal records and the reviewer.
                        audit = redact_secret(audit, self.provider.api_key)
                        write("reading_ledger.json", audit)
                    write("summary.json", {"status": "completed", "usage": usage,
                                          "cost_usd": costs(), "elapsed_seconds": elapsed,
                                          "known_cost_usd": known_cost(),
                                          "rationale": rationale,
                                          **({"reading_audit": audit} if self.audited else {})})
                    if notes is not None:
                        notes = redact_secret(notes, self.provider.api_key)
                    proposal = PolicyProposal(communication_policy=policy, assignment_policy=assignment,
                                              rationale=rationale, usage=usage,
                                              cost_usd=costs(), elapsed_seconds=elapsed,
                                              known_cost_usd=known_cost(), reading_audit=audit,
                                              execution_notes=notes)
                    # One atomic record commits both strategies with their accounting.
                    write_json_atomic(directory / "proposal.json", without_reasoning(vars(proposal)))
                    return proposal
            raise ValueError("Optimizer interaction limit reached without a submitted policy")
        except Exception as exc:
            elapsed = time.monotonic() - started
            message = str(exc)
            if self.provider.api_key:
                message = message.replace(self.provider.api_key, "[REDACTED]")
            try:
                write("summary.json", {"status": "failed", "error": message, "usage": usage,
                                      "cost_usd": costs(), "known_cost_usd": known_cost(),
                                      "elapsed_seconds": elapsed})
            except Exception as log_error:
                detail = str(log_error)
                if self.provider.api_key:
                    detail = detail.replace(self.provider.api_key, "[REDACTED]")
                message += "; failed to persist optimizer summary: " + detail
            raise PolicyOptimizationError(message, usage=usage, cost_usd=costs(),
                                          elapsed_seconds=elapsed, output_dir=directory,
                                          known_cost_usd=known_cost()) from exc
