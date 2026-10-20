"""Per-turn tool-call volume scenario shared by the local test and container checks.

No pytest imports. Only model responses are scripted; the installed Qwen CLI
runs the tools and its always-on per-turn cap (2026-10-02).
"""
from collections import defaultdict
import threading
import uuid

from agent_company.runtime.relay import role_of
from leader_loop_scenario import call, reply, text


class CallVolumeScenario:
    """One role makes about 120 tool calls in a single turn without a consecutive repeat.

    Each response carries the same shell command plus three distinct globs, so
    the identical call recurs (25 times by call 100) but never twice in a row:
    Qwen's adaptive default stops the turn at call 101; with no cap it finishes.
    mode: "single" (the Single agent itself) or "worker" (one Team teammate).
    """

    RESPONSES = 30

    def __init__(self, mode):
        if mode not in ("single", "worker"):
            raise ValueError(mode)
        self.mode = mode
        self.lock = threading.Lock()
        self.steps = defaultdict(int)
        self.batches = 0            # tool-call responses sent by the busy role
        self.finished = threading.Event()
        self.shutdown_sent = False

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            if sum(self.steps.values()) > 120:
                raise RuntimeError("Scripted call-volume fixture did not converge")
        if self.mode == "single" or role != "master":
            return self.busy(payload)
        return self.master(payload, step)

    def busy(self, payload):
        users = [text(message) for message in payload["messages"] if message.get("role") == "user"]
        if self.mode == "worker" and users and "shutdown" in users[-1].lower():
            return reply(payload, [call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        if self.batches < self.RESPONSES:
            self.batches += 1
            index = self.batches
            return reply(payload, [call("run_shell_command", {"description": "Repeated check", "command": "true"}),
                                   *(call("glob", {"pattern": f"volume-{index}-{part}-*.none"}) for part in "abc")])
        self.finished.set()
        return reply(payload, content="VOLUME_DONE: all calls finished")

    def master(self, payload, step):
        if step == 0:
            return reply(payload, [call("team_create", {"team_name": "volume-" + uuid.uuid4().hex[:6]})])
        if step == 1:
            return reply(payload, [call("agent", {"name": "alpha", "subagent_type": "general-purpose",
                                                  "description": "Many calls", "prompt": "Run the checks."})])
        if self.finished.is_set() and not self.shutdown_sent:
            self.shutdown_sent = True
            return reply(payload, [call("request_shutdown", {"to": "alpha"})])
        return reply(payload, content="Noted.")
