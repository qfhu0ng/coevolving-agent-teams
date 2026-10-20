"""Scripted model responses for role-policy runs of the installed Qwen CLI (S2c).

No pytest imports. Only model responses are scripted; the CLI runs the team, the
native message delivery, final-report forwarding, broadcast and shutdown.

Variants
- proposal: alpha sends a formal POLICY-PROPOSAL, then ends its turn with a text
  report, which native forwards (any nonempty round text clears native's
  explicit-report mark). The master approves the proposal from its injected
  snapshot; alpha then sees v2.
- proposal_silent / ordinary_silent: alpha writes its report and sends a leader
  message in the same response, then gives a reasoning-only response. Native
  re-requests that text-free response; the next nonempty text ("Idle.") is
  forwarded whichever kind of leader message came first (plan section 10.1).
- global: the master broadcasts GLOBAL-POLICY; alpha's next request carries C v2.
  alpha messages the leader once under C v1 and once (its acknowledgement) under v2.
- master_set: master changes alpha from v1 to v2; alpha explicitly messages its ACK.
- master_set_text_ack: same update, but alpha ends with a text ACK and relies on
  native automatic leader forwarding (the final-report race regression).
- spawn_active_limit: ten teammates occupy all slots; the next spawn is rejected.
  After two shut down, two new teammates can start even though the old roster remains.
"""
from collections import defaultdict
import json
import threading
import time
import uuid

from agent_company.runtime.relay import role_of

VARIANTS = ("proposal", "proposal_silent", "ordinary_silent", "global", "master_set", "master_set_text_ack", "spawn_active_limit")
LIMIT_WORKERS = [f"w{i}" for i in range(12)]
TURN_NOTICE = "alpha completed a turn without a model-visible final answer"
PROPOSAL = ("POLICY-PROPOSAL\nCHANGE: Message the leader only for blockers and the final report.\n"
            "REASON: The scripted build needs no coordination.\nEVIDENCE: Task assignment from the leader.")


def reply(payload, tools=(), content="", reasoning=None):
    message = {"role": "assistant", "content": content, "tool_calls": list(tools)}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()),
                 "model": payload["model"], "choices": [{"index": 0, "message": message,
                 "finish_reason": "tool_calls" if tools else "stop"}],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def call(name, args):
    return {"id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def system(payload):
    return "\n".join(text(message) for message in payload["messages"] if message.get("role") == "system")


def user_texts(payload):
    return [text(message) for message in payload["messages"] if message.get("role") == "user"]


class RolePolicyScenario:
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.lock = threading.Lock()
        self.steps = defaultdict(int)
        self.master_payloads, self.worker_payloads = [], []
        self.decision_sent = self.broadcast_sent = self.master_set_sent = self.shutdown_sent = False
        self.worker_after_update = None   # alpha's first request after the decision/broadcast message
        self.proposal_call_id = None
        self.spawn_batches = []
        self.shutdown_seen = set()
        self.native_stderr = None

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            if sum(self.steps.values()) > 60:
                raise RuntimeError("Scripted role-policy fixture did not converge")
        return self.master(payload, step) if role == "master" else self.worker(payload, step)

    def master_saw(self, marker):
        return any(marker in body for payload in self.master_payloads for body in user_texts(payload))

    def master(self, payload, step):
        self.master_payloads.append(payload)
        if step == 0:
            return reply(payload, [call("team_create", {"team_name": "policy-" + uuid.uuid4().hex[:6]})])
        if self.variant == "spawn_active_limit":
            return self.limited_master(payload, step)
        if step == 1:
            prompt = "Run the scripted build and report."
            if self.variant != "global":
                prompt += "\nROLE-POLICY:\nReport progress to the leader after each step.\nEND-ROLE-POLICY"
            return reply(payload, [call("agent", {"name": "alpha", "subagent_type": "general-purpose",
                                                  "description": "Scripted build", "prompt": prompt})])
        if self.shutdown_sent:
            return reply(payload, content="Team shut down.")
        if self.variant == "proposal" and not self.decision_sent and "Proposal #1 for slot alpha" in system(payload):
            self.decision_sent = True
            return reply(payload, [call("send_message", {"to": "alpha",
                                                         "message": "POLICY-DECISION #1 approve\nNOTE: Agreed."})])
        if self.variant in {"master_set", "master_set_text_ack"} and not self.master_set_sent and self.master_saw("ALPHA_REPORT"):
            self.master_set_sent = True
            return reply(payload, [call("send_message", {"to": "alpha", "message":
                "ROLE-POLICY alpha base=v1\nMessage the leader only for blockers and the final report."})])
        if self.variant == "global" and not self.broadcast_sent:
            self.broadcast_sent = True
            return reply(payload, [call("send_message", {"to": "*", "message":
                                                         "GLOBAL-POLICY base=v1\nReport only blockers and results."})])
        if self.variant.endswith("_silent"):
            # The forwarded text is what the test expects; shut down as soon as either arrives.
            if self.master_saw("Idle.") or self.master_saw(TURN_NOTICE):
                self.shutdown_sent = True
                return reply(payload, [call("request_shutdown", {"to": "alpha"})])
            return reply(payload, content="Waiting.")
        updated = self.decision_sent or self.broadcast_sent or self.master_set_sent
        if updated and self.master_saw("ALPHA_ACK"):
            self.shutdown_sent = True
            return reply(payload, [call("request_shutdown", {"to": "alpha"})])
        return reply(payload, content="Waiting.")

    def limited_master(self, payload, step):
        if step == 1:
            names = LIMIT_WORKERS[:10]
        elif step == 2:
            names = LIMIT_WORKERS[10:11]  # must be rejected while ten are alive/idle
        elif step == 3:
            return reply(payload, [call("request_shutdown", {"to": name}) for name in LIMIT_WORKERS[:2]])
        elif len(self.spawn_batches) == 2:
            # Do not end the master's turn while the lifecycle tail catches up:
            # idle teammates allow native to exit after a pure-text response.
            # Only the scripted provider waits; production has no added delay.
            self.wait_for_native_terminals(LIMIT_WORKERS[:2])
            names = LIMIT_WORKERS[10:]
        else:
            if self.shutdown_sent:
                return reply(payload, content="Team shut down.")
            if all(self.master_saw(f"WORKER_CWD_READY:{name}:") for name in LIMIT_WORKERS):
                self.shutdown_sent = True
                return reply(payload, [call("request_shutdown", {"to": name}) for name in LIMIT_WORKERS[2:]])
            return reply(payload, content="Waiting for the workers' reports.")
        self.spawn_batches.append(names)
        return reply(payload, [call("agent", {"name": name, "subagent_type": "general-purpose",
                                              "description": name, "prompt": "Run pwd and report your working directory."})
                               for name in names])

    def wait_for_native_terminals(self, names, timeout=5):
        if self.native_stderr is None:
            raise RuntimeError("Active-limit fixture needs the saved native lifecycle path")
        wanted = set(names)
        deadline = time.monotonic() + timeout
        while True:
            observed = set()
            try:
                saved = self.native_stderr.read_text(encoding="utf-8")
            except FileNotFoundError:
                saved = ""
            for line in saved.splitlines():
                if not line.startswith("Agent Company lifecycle: "):
                    continue
                try:
                    event = json.loads(line.split("Agent Company lifecycle: ", 1)[1])
                except json.JSONDecodeError:
                    continue  # The native writer may not have finished this line.
                if event.get("event") == "worker_terminal" and event.get("status") in {
                    "completed", "failed", "cancelled"
                }:
                    observed.add(event.get("worker"))
            if wanted <= observed:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Scripted fixture did not observe native terminal records: {sorted(wanted - observed)}")
            time.sleep(0.01)

    def worker(self, payload, step):
        self.worker_payloads.append(payload)
        latest = (user_texts(payload) or [""])[-1]
        if "shutdown" in latest.lower():
            self.shutdown_seen.add(role_of(payload))
            return reply(payload, [call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        if self.variant == "spawn_active_limit":
            if step == 0:
                return reply(payload, [call("run_shell_command", {"command": "pwd -P",
                                                                   "description": "Verify this worker's assigned directory"})])
            return reply(payload, content=f"WORKER_CWD_READY:{role_of(payload)}: working directory checked.")
        if self.variant == "global" and self.worker_after_update is not None:
            return reply(payload, content="Acknowledged.")  # the acknowledgement was already sent
        if self.variant in {"master_set", "master_set_text_ack"} and "Your role policy version 2" in system(payload):
            if self.worker_after_update is not None:
                return reply(payload, content="Acknowledged.")
            self.worker_after_update = payload
            if self.variant == "master_set_text_ack":
                return reply(payload, content="ALPHA_ACK: master-set role policy received.")
            # Exercise the policy update, without relying on automatic final-text
            # forwarding racing the master's next idle/completion check.
            return reply(payload, [call("send_message", {"to": "leader",
                "message": "ALPHA_ACK: master-set role policy received."})])
        if "POLICY-DECISION" in latest or "GLOBAL-POLICY" in latest:
            self.worker_after_update = self.worker_after_update or payload
            if self.variant == "global":
                return reply(payload, [call("send_message", {"to": "leader",
                                                             "message": "ALPHA_ACK: policy update received."})])
            return reply(payload, content="ALPHA_ACK: policy update received.")
        if step == 0:
            if self.variant == "global":
                return reply(payload, [call("send_message", {"to": "leader", "message": "Starting the build."}),
                                       call("run_shell_command", {"description": "Scripted build",
                                                                  "command": "python3 -c 'import time; time.sleep(3)'"})])
            message = PROPOSAL if self.variant.startswith("proposal") else "Progress note: halfway through the build."
            sent = call("send_message", {"to": "leader", "message": message})
            self.proposal_call_id = sent["id"]
            report = "ALPHA_REPORT: build finished." if self.variant.endswith("_silent") else ""
            return reply(payload, [sent], content=report)
        if step == 1:
            if self.variant.endswith("_silent"):
                return reply(payload, content="", reasoning="The build is complete.")
            return reply(payload, content="ALPHA_REPORT: build finished.")
        return reply(payload, content="Idle.")
