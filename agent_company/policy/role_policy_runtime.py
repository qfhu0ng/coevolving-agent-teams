"""Model-facing rendering and runtime inputs for role policies (Stage-1, S2).

Specification: docs/stage1_role_policy_harness_plan_20261009.md v3.3, sections 3.3
(injection), 3.3.1 (prefix cache) and 9 (disclosure drafts). State lives in
role_policy.RolePolicyStore; this module only turns snapshots into text and feeds
lifecycle facts from the CLI's stderr back to the store.

Rendering is a pure function of (mode, role, snapshot): no current time, age,
request number or counter is rendered, slots and proposals are in a stable order, and
equal snapshots give byte-identical text (section 3.3, "deterministic representation").
The disclosure blocks are the section 9 drafts verbatim; a test compares them with
the document so the two cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Callable

from agent_company.policy.role_policy import MASTER, PolicySnapshot, ProposalView, SlotView, VersionView

# Section 9 drafts, verbatim (tests/test_role_policy_runtime.py checks the document).
DISCLOSURE_B = 'Policy state is maintained by the relay. The policy/proposal snapshot supplied in this request is authoritative; a native policy notification may be late or missing. Committed state is not rolled back if native message sending fails. Check the next supplied snapshot before retrying an update; do not treat a send error as proof that nothing changed.\nUpdates affect later new requests, not a request already in progress or its transport retries. Displayed timestamps are fixed submission/commit times, not the current time or an age. When idle, end your turn and wait for normal messages or check-ins; do not repeatedly poll for policy or proposal changes. Injection itself does not wake an agent or extend a deadline.\nUse only the control-message formats allowed below for your role and mode. Put the exact header on the first line of send_message.message. Optional base=v<k> must match the supplied snapshot; the relay also checks that this base is still current. Missing state is not permission to guess a version. A rejected control call can reject every call in the same response, including unrelated coordination. Prefer a separate response for policy controls; after a stale rejection, use the next supplied snapshot and reconsider the operation.\nThese are communication instructions, not new tool permissions. Existing safety rules, source boundaries, cancellation, and the hard task deadline always apply. Ordinary task messages remain available.'
DISCLOSURE_G_M = 'Mode: global. The versioned communication policy C in the supplied snapshot applies to you and all workers. To replace it, call send_message with to="*" and message starting with GLOBAL-POLICY, optionally followed by base=v<k>. From the second line, provide the complete replacement C, at most 6000 characters. This control must be broadcast, not sent to just one worker.\nBroadcasting attempts native notification to the current member list. It can include terminal members, have no recipients, or fail for some recipients; none of these outcomes undoes a committed update. New workers receive current C in their first request. Notification success is not proof that a worker has received a model request or followed C. Role-policy controls and formal proposal controls are not available in this mode.'
DISCLOSURE_G_W = 'Mode: global. Follow the versioned communication policy C in the supplied snapshot. The master can update C for the whole team. You cannot write formal policy state in this mode. Send ordinary suggestions to the leader when useful and continue your task under current C; a suggestion or update notification is not itself an approved policy change.'
DISCLOSURE_R_M = "Mode: role policies. The supplied snapshot lists slots, current instances, versions, and full role policies. A role's approved communication policy overrides conflicting communication defaults in C or incidental assignment instructions, but not fixed runtime boundaries. Without a role policy, C applies.\nTo replace a slot's policy, send_message to that slot's current instance. Start the message with ROLE-POLICY <slot>, optionally followed by base=v<k>; from the second line provide the complete new policy, at most 2000 characters. Use the target slot's version in your supplied snapshot, not another slot's version. This control is single-recipient, never broadcast.\nFor a new slot, an agent spawn prompt may contain a line ROLE-POLICY: followed by its complete initial policy, ending at END-ROLE-POLICY or the prompt end. The text limit is also 2000 characters. For succession, instead include a line SUCCEEDS: <previous-name>. Use a new worker name, and only a predecessor the relay confirms as terminal. Do not combine the two markers. Succession keeps the slot and current policy version, not the previous conversation, workspace, or pending proposal.\nUse ordinary messages for task coordination. A worker's ordinary suggestion does not change formal policy state; set the policy explicitly if you choose to adopt it. Global-policy updates are not available in this mode."
DISCLOSURE_R_W = 'Mode: role policies. For your communication choices, follow your approved role policy in the supplied snapshot. It overrides conflicting communication defaults in C or incidental assignment instructions, not fixed runtime boundaries. If no role policy exists, follow C. You cannot write formal policy state in this mode. Send ordinary suggestions to the leader when useful, and continue working under your current policy.'
DISCLOSURE_P_M = "Formal worker proposals are enabled. Your supplied snapshot includes each pending proposal's ID, slot, proposer, base version, and full CHANGE/REASON/EVIDENCE. Judge that content, not a possibly stale notification. To decide, send_message to the proposer with first line POLICY-DECISION #<id> approve, POLICY-DECISION #<id> reject, or POLICY-DECISION #<id> amend. Approve/reject may have a NOTE: below the header, at most 2000 characters; amend must put the complete replacement policy below it, also at most 2000 characters, without another control header. Decisions are single-recipient, never broadcast. Approve when the four review questions hold; amend when the direction is right but the scope is too broad or the requirement is impractical; reject when evidence is missing or the proposal repeats an existing rule without a substantive change. Give the reason for an approval or rejection in NOTE, and for a rejection say what information is missing. The worker keeps working under its current policy while you decide.\nA decision applies only to that pending ID. If it has been replaced or its base is stale, the whole response may be rejected. A reported replacement ID is diagnostic, not permission to transfer approval; read its full content in the next snapshot and decide again. Approval is a formal instruction, not a new messaging capability or proof of compliance."
DISCLOSURE_P_W = 'Mode: role policies with proposals. Follow your approved role policy from the supplied snapshot; it overrides conflicting communication defaults in C or incidental assignment instructions, not fixed runtime boundaries. If no role policy exists, follow C.\nWhen concrete task evidence suggests a better rule for your own outgoing communication, you may send_message to="leader" with first line POLICY-PROPOSAL, optionally followed by base=v<k>. Below it provide CHANGE: with the complete replacement policy; REASON: the problem the current rule causes, why this change should fix it, and any added cost or risk; and EVIDENCE: the specific messages, tasks, or files, what you expect to observe afterwards, and at which related event it can be checked. These fields including their labels must total at most 2000 characters. The relay derives your slot and base from your request snapshot. Do not propose rules for other roles; request those through ordinary messages to the master instead. If your change depends on cooperation from another role, say so; proposing that another role confirm something is not a commitment from that role. Only the master can approve a proposal.\nContinue working and communicating under the current approved policy while a proposal is pending; you do not need to wait for a decision. At most one proposal is pending for your slot. Resending identical CHANGE/REASON/EVIDENCE under the same base only retries notification and keeps its ID and submission time. Changing any of these fields replaces the old pending proposal with a new ID, even if the master previously received it. Repeated replacement can delay a decision. Read the proposal status in your next supplied snapshot; do not infer approval, rejection, or failure from native send results. The proposal itself does not change your active policy before approval.'
DISCLOSURE_Q = 'Changing communication policy: do it only for a concrete coordination problem you can point to (such as duplicate notices, an incomplete handoff, a message to the wrong owner, or an interface change that reached its users late) or because the task changed so the current rule no longer fits (a new dependency, the start of integration). Consider whether the rule is inadequate, was not followed, or a message has simply not arrived yet. If a rule was clearly not followed, remind first. A rule that already covers the case may still be simplified, clarified, or given a different trigger when a concrete problem shows it does not work in practice. If the cause is not established, state your hypothesis; you do not need to prove the root cause first. You usually cannot see whether a message was delivered or read; say so rather than assume. Fix one problem per change; "communicate better" is not a change. Cite evidence a teammate can locate (a message\'s sender and time, a task, a commit, a file, a check-in), and say what you expect to observe and at which related event it can be checked, such as the next handoff or check-in; do not poll for it. The same standard applies to ordinary policy suggestions.'
DISCLOSURE_V = 'Before you change a policy or decide a proposal, check four questions. Grounded: is the problem observed, or a guess, and if its cause is not established, is the hypothesis stated? On target: does the change address that problem? Remind first when an existing rule was clearly not followed, and do not approve a repeat of an existing rule without a substantive change; simplifying, clarifying, or retriggering a rule is a substantive change. Executable: do permissions, message delivery and wake-up conditions allow it? Worth it: what improvement could be observed, and could it add useless reports or cause important messages to be missed? Change only when all four hold, and explain a direct change or an amendment in a short ordinary message, not in the policy text.\nDo not assume a committed change affected workers immediately. Review it against reports and task events that arrive naturally: when a related event occurs, record one conclusion in an ordinary message starting with POLICY-REVIEW <slot or C> v<k>: improved, no change, side effects, or cannot judge yet, plus the facts. If you cannot tell whether behavior followed the new version, write cannot judge yet; do not poll for it. If a change clearly made things worse, set the earlier rule again as a new version; nothing rolls back automatically, and one before/after comparison does not prove a policy works.'
DISCLOSURE_D10_C = 'Your end-of-turn text is forwarded to the leader automatically. Sending a formal proposal during the turn does not change that. A proposal is not a final task report; keep your task results clear and separate.'

# v3.4 (2026-10-13): Q (when and how to change policy) for every role, V (review and follow-up) for the master,
# and fuller P-M / P-W field guidance. Recorded in launch.json.
DISCLOSURE_VERSION = "v3.4"
BLOCK_SEPARATOR = "\n\n"
LIFECYCLE_PREFIX = "Agent Company lifecycle: "


def disclosure(mode: str, role: str) -> str:
    """The fixed disclosure for one role in one mode (section 9.4 assembly); '' for off."""
    if mode == "off":
        return ""
    master = role == MASTER
    blocks = {"global": ([DISCLOSURE_G_M, DISCLOSURE_V] if master else [DISCLOSURE_G_W]),
              "master": ([DISCLOSURE_R_M, DISCLOSURE_V] if master else [DISCLOSURE_R_W]),
              # D10 option C (2026-10-10): the compat change keeps native final-report forwarding.
              "propose": ([DISCLOSURE_R_M, DISCLOSURE_P_M, DISCLOSURE_V] if master else [DISCLOSURE_P_W, DISCLOSURE_D10_C])}
    if mode not in blocks:
        raise ValueError(f"Unknown role-policy mode {mode!r}")
    return BLOCK_SEPARATOR.join([DISCLOSURE_B, DISCLOSURE_Q, *blocks[mode]])


def _communication_policy(view: VersionView) -> str:
    return f"Communication policy C (version {view.version}):\n{view.text}"


def _holder(slot: SlotView) -> str:
    if slot.instance is None:
        return "no current instance"
    state = "terminal" if slot.lifecycle == "terminal" else "running"
    return f"instance {slot.instance} ({state})"


def _policy_line(slot: SlotView) -> str:
    policy = slot.policy
    if policy.version == 0:
        return "no role policy; C applies."
    return (f"role policy version {policy.version}, committed by {policy.actor} at {policy.committed_at} "
            f"({policy.provenance}{f', proposal #{policy.proposal_id}' if policy.proposal_id else ''}):\n{policy.text}")


def _proposal_status(proposal: ProposalView) -> str:
    if proposal.status == "pending":
        return (f"Your proposal #{proposal.proposal_id} (base version {proposal.base_version}) is pending; "
                f"submitted at {proposal.created_at}, the fixed submission time.")
    if proposal.status == "voided":
        detail = {"replaced": f"replaced by #{proposal.replaced_by}",
                  "policy_superseded": "superseded by a master update to your role policy",
                  "proposer_terminal": "voided when its proposer ended",
                  "run_ended": "voided at run end"}.get(proposal.void_reason, proposal.void_reason)
        return f"Your proposal #{proposal.proposal_id} was {detail}."
    outcome = {"approved": f"approved as version {proposal.decided_version}",
               "amended": f"amended by the master into version {proposal.decided_version}",
               "rejected": "rejected"}[proposal.status]
    note = f" Note: {proposal.note}" if proposal.note else ""
    return f"Your proposal #{proposal.proposal_id} was {outcome}.{note}"


def render_role_state(snapshot: PolicySnapshot) -> str:
    """The dynamic role-policy block; '' in global mode, where C itself is the state."""
    if snapshot.mode == "global":
        return ""
    if snapshot.role == MASTER:
        lines = ["Role-policy snapshot (slots in name order):"]
        if not snapshot.slots:
            lines.append("No slots yet.")
        for slot in snapshot.slots:
            lines.append(f"Slot {slot.slot_id}: {_holder(slot)}; {_policy_line(slot)}")
        if snapshot.mode == "propose":
            lines.append("Pending proposals:" if snapshot.proposals else "Pending proposals: none.")
            for proposal in snapshot.proposals:
                lines.append(f"Proposal #{proposal.proposal_id} for slot {proposal.slot_id} from {proposal.proposer}, "
                             f"base version {proposal.base_version}, submitted at {proposal.created_at}:\n"
                             f"CHANGE: {proposal.change}\nREASON: {proposal.reason}\nEVIDENCE: {proposal.evidence}")
        return "\n".join(lines)
    if not snapshot.slots:
        return f"No role slot is bound to {snapshot.role}; C applies."
    (slot,) = snapshot.slots
    policy = (f"Your {_policy_line(slot)}" if slot.policy.version else "No role policy is set for your slot; C applies.")
    lines = [f"Your role slot: {slot.slot_id}; you are {snapshot.role}. {policy}"]
    if snapshot.mode == "propose":
        lines.extend([_proposal_status(p) for p in snapshot.proposals] or ["You have no pending proposal."])
    return "\n".join(lines)


def render_injection(snapshot: PolicySnapshot) -> str:
    """Versioned C, the fixed disclosure, then the role-policy state; a pure function of the snapshot."""
    parts = (_communication_policy(snapshot.global_policy), disclosure(snapshot.mode, snapshot.role),
             render_role_state(snapshot))
    return BLOCK_SEPARATOR.join(part for part in parts if part)


class LifecycleTail:
    """Incrementally read compat lifecycle lines from the CLI stderr file.

    Only complete lines are consumed; a partial last line waits for its newline. Each
    line is handled once (by line number). worker_terminal events go to ``on_terminal``
    with the source ``stderr:<line>``. Optional ``on_spawn_result`` receives native
    dispatch outcomes, independently of compactable model history. The file is
    written by the CLI process (directly or through docker exec) on the host.
    """

    def __init__(self, path: Path, on_terminal: Callable[[str, str], object], *,
                 on_spawn_result: Callable[[dict, str], object] | None = None,
                 on_event: Callable[[dict, str], object] | None = None,
                 clock: Callable[[], float] = time.time):
        self.path, self.on_terminal, self.clock = Path(path), on_terminal, clock
        self.on_spawn_result = on_spawn_result
        # Optional: every other well-formed lifecycle record (agent_rules_v1 observations).
        self.on_event = on_event
        self._offset, self._line, self._buffer = 0, 0, b""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.terminal_events = 0
        self.spawn_events = 0
        self.invalid_lines = 0
        self.callback_errors: list[str] = []
        self.max_consume_delay_seconds: float | None = None

    def poll(self) -> int:
        """Consume complete lines; return the number of lifecycle events handled."""
        with self._lock:
            try:
                with self.path.open("rb") as stream:
                    stream.seek(self._offset)
                    data = stream.read()
            except FileNotFoundError:
                return 0
            self._offset += len(data)
            chunks = (self._buffer + data).split(b"\n")
            self._buffer = chunks.pop()
            handled = 0
            for raw in chunks:
                self._line += 1
                text = raw.decode("utf-8", errors="replace")
                if not text.startswith(LIFECYCLE_PREFIX):
                    continue
                try:
                    record = json.loads(text[len(LIFECYCLE_PREFIX):])
                except ValueError:
                    self.invalid_lines += 1
                    continue
                if (isinstance(record, dict) and record.get("event") == "worker_spawn_result"
                        and self.on_spawn_result is not None):
                    if (not isinstance(record.get("requested_name"), str)
                            or not isinstance(record.get("tool_call_id"), str)
                            or type(record.get("ok")) is not bool
                            or (record["ok"] and not isinstance(record.get("worker"), str))):
                        self.invalid_lines += 1
                        continue
                    try:
                        self.on_spawn_result(record, f"stderr:{self._line}")
                    except Exception as exc:
                        self.callback_errors.append(f"{type(exc).__name__}: {exc}"[:300])
                    self.spawn_events += 1
                    handled += 1
                    continue
                if isinstance(record, dict) and self.on_event is not None:
                    # Terminal records too: the relay keeps their time before on_terminal runs.
                    try:
                        self.on_event(record, f"stderr:{self._line}")
                    except Exception as exc:
                        self.callback_errors.append(f"{type(exc).__name__}: {exc}"[:300])
                    if record.get("event") != "worker_terminal":
                        continue
                if not isinstance(record, dict) or record.get("event") != "worker_terminal":
                    continue
                worker = record.get("worker")
                if (not isinstance(worker, str)
                        or record.get("status") not in {"completed", "failed", "cancelled"}):
                    self.invalid_lines += 1
                    continue
                at = record.get("at")
                if isinstance(at, (int, float)) and not isinstance(at, bool):
                    delay = max(0.0, self.clock() - at / 1000)
                    self.max_consume_delay_seconds = max(self.max_consume_delay_seconds or 0.0, delay)
                try:
                    self.on_terminal(worker, f"stderr:{self._line}")
                except Exception as exc:  # recorded; a lifecycle callback must not stop the run
                    self.callback_errors.append(f"{type(exc).__name__}: {exc}"[:300])
                self.terminal_events += 1
                handled += 1
            return handled

    def start(self, interval_seconds: float = 0.5) -> None:
        def loop():
            while not self._stop.wait(interval_seconds):
                self.poll()
        self._thread = threading.Thread(target=loop, name="role-policy-lifecycle-tail", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and consume every complete line written so far."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.poll()

    def summary(self) -> dict:
        return {"terminal_events": self.terminal_events, "spawn_events": self.spawn_events,
                "invalid_lines": self.invalid_lines,
                "lines_read": self._line, "partial_line_pending": bool(self._buffer),
                "callback_errors": self.callback_errors[:20],
                "max_consume_delay_seconds": (round(self.max_consume_delay_seconds, 3)
                                              if self.max_consume_delay_seconds is not None else None)}
