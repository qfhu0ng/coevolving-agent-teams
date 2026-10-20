"""Approved role_package_v1 disclosures and pure request-snapshot rendering.

Literal text is copied from approved R3 runtime blocks. Runtime execution never
reads the design directory. System and transient request-tail are separate.
"""
from __future__ import annotations

from agent_company.policy.role_package_store import PackageSnapshot, ProposalView, render_pending

BLOCK_SEPARATOR = "\n\n"
DISCLOSURE_VERSION = "role_package_v1"
RP_CORE = 'Policy state: the policy text and versions shown in these system instructions are authoritative and are refreshed on every request. Notifications about policy changes can be late, missing or failed; that never undoes a committed change. A change applies to requests that start after it is committed. C is the team communication policy and applies to everyone. Your role policy is shared by every teammate with your role type, including later ones; it adds to C and takes precedence where they conflict, as it does over communication instructions in a task assignment. Neither overrides the fixed protocols, tool permissions, source boundaries or deadlines. The assignment policy, protocols and role directory are fixed for this run. Ordinary task messages remain available. When idle, end your turn; do not poll for policy changes.'
RP_CONTROL_SYNTAX = 'Policy controls: put the exact header alone on the first line of send_message.message and the policy text from the second line. append adds a nonempty rule after the current text on a new line; replace supplies the complete new text. A role policy may be replaced with empty text, leaving only C; C cannot be empty. base=v<k> is optional; without it, the version shown in this request is used. Either way, the change is rejected if that version is no longer current. A rejected control fails only that call; other valid calls in the same response may still run, so do not assume related calls succeeded together.'
RP_Q = 'Policy changes and suggestions should address a specific coordination problem you can point to (for example duplicate notices, an incomplete handoff, a message to the wrong owner, or an interface change that reached its users late) or a task change that makes a rule no longer fit. If an existing rule was simply not followed, a reminder may be enough. A stated hypothesis is acceptable; you do not need to prove the root cause, and you usually cannot see whether a message was read. Fix one problem per change; "communicate better" is not a rule.'
RP_LIMITS = 'Resulting texts must stay within 6000 characters for C, 2000 for each role policy and 16000 for all role policies together; append is checked after combining. Pending proposals are shown in full up to 8000 characters in total; a proposal that would exceed this is rejected and any earlier one stays pending.'
RP_BINDING_MASTER = "When spawning a teammate, put one line ROLE-TYPE: <role_id> in the agent prompt, choosing the type that matches the teammate's main function: implementer builds assigned features; integrator combines work in the delivery repository; verifier builds and checks combined results; prober explores the reference behavior and reports differences. If you omit the line, the teammate becomes an implementer. The master type is yours and cannot be given to a teammate. A role type selects which shared communication policy the teammate follows; it does not change tool permissions, and several teammates may share a type.\nA new teammate automatically receives its type's current policy, without another teammate's conversation, worktree, uncommitted work or pending proposals, so a replacement needs only the same ROLE-TYPE and a new name; SUCCEEDS is not needed. Do not put policy text in a spawn prompt; a teammate's policy comes only from its role type."
RP_OFF = 'Mode: off. C and all role policies are fixed for this run. No online policy controls or formal proposals are allowed. Use ordinary messages for coordination and suggestions, and continue under the supplied policies and protocol.'
RP_OFF_REVIEW_GRAMMAR = 'Policy review (mode off): write it as one line in your reply text, not as a message to teammates. Start with POLICY-REVIEW C v<k>: or POLICY-REVIEW role:<role_id> v<k>:, using the version shown above, then one label (no issue, issue observed or cannot judge yet) and the supporting facts. Review C unless a specific role is involved. Use cannot judge yet when evidence is missing. This records an observation only; policies cannot be changed in this mode.'
RP_GLOBAL_MASTER = 'Mode: global. You may update C for the whole team. Send to="*" with first line GLOBAL-POLICY append or GLOBAL-POLICY replace, optionally followed by base=v<k>, and put the policy text below it. Role policies and the directory are fixed. Worker suggestions use ordinary messages; formal proposals are not available.'
RP_GLOBAL_WORKER = 'Mode: global. Follow C and your supplied role policy. The master may update C; role policies are fixed. You cannot write formal policy state or submit formal proposals. Send useful suggestions to the leader as ordinary messages and continue under the current snapshot.'
RP_ROLE_CONTROLS = 'To update C, send to="*" with first line GLOBAL-POLICY append or GLOBAL-POLICY replace. To update a role, send to="*" with first line ROLE-POLICY <role_id> append or ROLE-POLICY <role_id> replace. Optionally append base=v<k> for that target, then put the policy text below the header. A role update changes the shared policy for every current and future instance of that type, including when none is active. You may update the master role\'s policy as well. You cannot change A, the protocol, resources, tool permissions or the role directory.'
RP_MASTER_MASTER = 'Mode: master. You may update C and existing role policies. Worker suggestions use ordinary messages; formal proposals are not available.'
RP_MASTER_WORKER = 'Mode: master. Follow C and your supplied role policy. The master may update C and shared role policies. You cannot write formal policy state or submit formal proposals. Send useful suggestions to the leader as ordinary messages and continue under the current snapshot.'
RP_BROADCAST = 'Broadcast notifications can be late, reach no one or fail for some recipients; this never undoes a committed update, and a delivered notification does not show that a teammate read or followed it.'
RP_PROPOSE_MASTER = "Mode: propose. You may update C and existing role policies. Workers may formally propose a change to C or their own role type. A role proposal affects the shared policy for all instances of that type, not just its proposer.\nYour snapshot displays each pending proposal's exact ID, target, operation, proposer, base version and complete CHANGE/REASON/EVIDENCE. Decide that proposal by sending to its proposer with first line POLICY-DECISION #<id> approve, POLICY-DECISION #<id> reject, or POLICY-DECISION #<id> amend. Approve/reject may have an optional NOTE: below the header, at most 2000 characters. Approve applies the proposal's stated append or replace operation. Amend replaces the whole target text, not just the proposed part: below the header write the complete new policy, including any existing text you keep, without another control header. An empty role amendment clears that role policy; an amendment to C must be nonempty.\nDecide each pending proposal at your next opportunity, using the four review questions below. Keep approved or amended policy text reusable; retain task-specific evidence in NOTE or the cited proposal rather than copying it into the policy. Give the approval or rejection reason in NOTE, and identify missing information when rejecting. A decision binds to the exact ID in your snapshot and never transfers to a replacement ID or new base. Updating a target removes other pending proposals on its old version from your list, and shutting down a worker discards its pending proposals, so decide a worker's proposals before asking it to shut down. A removed proposal can be resubmitted by its worker; if that worker has ended, you may apply the idea directly and cite the proposal ID. Workers continue under their current policies while you decide. Approval is a formal instruction, not proof of compliance."
RP_PROPOSE_WORKER = 'Mode: propose. You can propose changes to the team communication policy C or to your own role type\'s policy; the leader (master) decides. When you run into a coordination problem, such as a late or missing handoff, duplicated work, a report that reached the wrong teammate or an unclear owner, propose a short rule that would have prevented it. Small, specific proposals are welcome; a rejection has no penalty.\nSend to="leader" with the first line POLICY-PROPOSAL role:<your_role_id> append or POLICY-PROPOSAL C append (use replace instead of append only to rewrite or remove existing text), optionally followed by base=v<k>. Below it write three fields, each starting a line, in this order:\nCHANGE: the rule to add, or the complete new text for replace. Write a reusable rule without this task\'s file names, format names, outputs or solution details. Only a role replace may leave CHANGE empty, which clears your role policy.\nREASON: the problem and why the rule should help.\nEVIDENCE: where a teammate can check it (a message, commit, file, task or collaboration-event ID), whether you observed it or were told, and if you can, the next event where the effect could be seen.\nAll three fields must fit in 2000 characters. A role change applies to every teammate with your role type, now and later. Ask the master in an ordinary message for changes to other roles. Keep working under the current policies without waiting for a decision. You can have one pending proposal per target; a different one for the same target replaces it. Check its status in your next request, not in send results. If a policy update superseded it, resubmit against the new version if it is still needed.'
RP_REVIEW = 'Before changing a policy or deciding a proposal, check four questions. Grounded: is the problem observed, or a guess, and if its cause is not established, is the hypothesis stated? On target: does the change address that problem? Remind first when an existing rule was clearly not followed, and do not approve a repeat of an existing rule without a substantive change; simplifying, clarifying, or retriggering a rule is a substantive change. Executable: do permissions, message delivery and wake-up conditions allow it? Worth it: what improvement could be observed, and could it add useless reports or cause important messages to be missed? Change only when all four hold, and explain a direct change or an amendment in your reply text or in NOTE, not in the policy text.\nDo not assume a committed change affected agents immediately. Record each review as one line in your reply text, not as a message to teammates, starting with POLICY-REVIEW C v<k>: or POLICY-REVIEW role:<role_id> v<k>:. To judge an earlier change when a related event occurs, use improved, no effect, side effects or cannot judge yet. To record a check-in without a change, use no issue, issue observed (say why you kept the rule) or cannot judge yet. Add the facts, and do not poll for them. If a change clearly made things worse, replace the target with the earlier text as a new version (an empty earlier role policy clears it). Nothing rolls back automatically, and one before/after comparison does not prove a policy works.'
RP_CARRY_FORWARD = 'Approved rules stay in effect for the rest of this run and may be kept for future teams after an independent review, so write them to be useful beyond this task. Unapproved proposals are kept only as records.'
RP_RUNTIME_POINTER = 'Communication policy: the current communication policy C, the role directory, role policies and, where allowed, policy-control formats are supplied in your system instructions on every request. You have the master role type; each teammate has the role type set when it was spawned. Shared collaboration records are appended as data at the end of each request. The assignment policy and protocols stay fixed for this run.'
RP_SPAWN_SUFFIX = 'The team communication policy C, your role type and its current shared policy are supplied in your system instructions on every request, including after context compaction. Follow those; a message about a policy change does not by itself change it.'
RP_CONTROL_FAILURE = 'Policy control rejected before execution ({code}): {detail}\nThis call did not run and changed no policy state; other valid calls in the same response may still have run. Fix it using the policy state shown in your current instructions. If the detail says the control is not available to you, do not retry it; use an ordinary message instead.'
RP_BINDING_FAILURE = 'Teammate assignment rejected before native dispatch ({code}): {detail}\nThis teammate was not started by this call. Independent valid calls in the same response may still execute. Correct this assignment without resubmitting unrelated successful calls.'


def disclosure(mode: str, role: str) -> str:
    """Only the two implemented new-package modes; no legacy fallback."""
    master = role == "master"
    if mode == "off":
        blocks = ([RP_CORE, RP_Q, RP_BINDING_MASTER, RP_OFF, RP_OFF_REVIEW_GRAMMAR]
                  if master else [RP_CORE, RP_Q, RP_OFF])
    elif mode == "propose":
        blocks = ([RP_CORE, RP_CONTROL_SYNTAX, RP_Q, RP_LIMITS, RP_BINDING_MASTER, RP_PROPOSE_MASTER,
                   RP_ROLE_CONTROLS, RP_BROADCAST, RP_REVIEW, RP_CARRY_FORWARD] if master else
                  [RP_CORE, RP_CONTROL_SYNTAX, RP_Q, RP_PROPOSE_WORKER, RP_CARRY_FORWARD])
    else:
        raise ValueError("New policy-package disclosure supports off and propose only")
    return BLOCK_SEPARATOR.join(blocks)


def render_snapshot(snapshot: PackageSnapshot) -> str:
    """Stable system state: C, active directory and role text, not proposals or lifecycle."""
    head = (f"Role directory: {', '.join(r.role_id for r in snapshot.roles)}.\n"
            f"Communication policy C (version {snapshot.global_policy.version}):\n{snapshot.global_policy.text}")
    if snapshot.actor == "master":
        rows = [f"Role {r.role_id} (version {r.policy.version}):\n{r.policy.text}" if r.policy.text else
                f"Role {r.role_id} (version {r.policy.version}): empty; only C applies." for r in snapshot.roles]
        confirmed = sorted((b for b in snapshot.bindings if b.confirmed), key=lambda b: b.spawn_order)
        bindings = ("Teammate role types (in spawn order): " + ", ".join(f"{b.instance}={b.role_id}" for b in confirmed)
                    if confirmed else "No teammates yet.")
        state = "Your role type: master.\nShared role policies (role IDs in order):\n" + "\n".join(rows) + "\n" + bindings
    else:
        policy = next((r.policy for r in snapshot.roles if r.role_id == snapshot.role_id), None)
        if policy is None:
            raise RuntimeError("Bound worker role is not in the active role directory")
        lines = [f"You are {snapshot.actor}; your role type is {snapshot.role_id}."]
        if snapshot.mode == "propose":
            lines.append(f"To propose a rule for your role, start with: POLICY-PROPOSAL role:{snapshot.role_id} append")
        lines.append(f"Your shared role policy (version {policy.version}):\n{policy.text}" if policy.text else
                     f"Your shared role policy (version {policy.version}) is empty; only C applies.")
        state = "\n".join(lines)
    return BLOCK_SEPARATOR.join((head, state))


def render_injection(snapshot: PackageSnapshot) -> str:
    """Fixed instructions plus dynamic system state; no request-tail data."""
    return BLOCK_SEPARATOR.join((disclosure(snapshot.mode, snapshot.actor), render_snapshot(snapshot)))


def _proposal_status(p: ProposalView) -> str:
    stem = f"Your proposal #{p.proposal_id} for {p.target}"
    if p.status == "pending":
        return (f"{stem} (base version {p.base_version}) is pending; submitted at {p.created_at}, "
                "the fixed submission time.")
    if p.status == "voided":
        detail = {"replaced": f"replaced by #{p.replaced_by}",
                  "policy_superseded": "superseded by an update to the target policy; resubmit against the current version if it is still needed",
                  "proposer_terminal": "voided when its proposer ended", "run_ended": "voided at run end"}[p.void_reason]
        return f"{stem} was {detail}."
    outcome = {"approved": f"approved as version {p.decided_version}",
               "amended": f"amended by the master into version {p.decided_version}",
               "rejected": "rejected"}[p.status]
    return f"{stem} was {outcome}." + (f" Note: {p.note}" if p.note else "")


def render_tail(snapshot: PackageSnapshot) -> str:
    """One latest pending/status block, attached to a request copy by the relay."""
    if snapshot.mode == "off":
        return ""
    if snapshot.actor == "master":
        return render_pending(snapshot.proposals)
    return "\n".join(_proposal_status(p) for p in snapshot.proposals) or "You have no proposal status to report."


def rejection_text(rejection, *, binding=False):
    return (RP_BINDING_FAILURE if binding else RP_CONTROL_FAILURE).format(code=rejection.code, detail=rejection.detail)
