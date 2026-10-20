"""Busy-teammate delivery scenario shared by the local test and the container check.

No pytest imports: the Linux container validation runs it with the project
environment. Only model responses are scripted; the real CLI does the rest.
"""
from collections import defaultdict
import json
import re
import threading
import time
import uuid

from agent_company.runtime.relay import role_of


def reply(payload, tools=(), content=""):
    return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()),
                 "model": payload["model"], "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": content, "tool_calls": list(tools)},
                 "finish_reason": "tool_calls" if tools else "stop"}],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def raw_call(name, args):
    return {"id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def after_last_tool_call(payload):
    """Everything the model sees after its most recent tool-call message."""
    messages = payload["messages"]
    last = max(index for index, message in enumerate(messages)
               if message.get("role") == "assistant" and message.get("tool_calls"))
    return "\n".join(text(message) for message in messages[last + 1:])


def last_user(payload):
    users = [text(message) for message in payload["messages"] if message.get("role") == "user"]
    return users[-1] if users else ""


class BusyWorkerScenario:
    def __init__(self, between_tool_calls):
        self.between = between_tool_calls
        self.steps = defaultdict(int)
        self.lock = threading.Lock()
        self.seen = threading.Event()
        self.approved = threading.Event()
        self.nonce = "MIDROUND_" + uuid.uuid4().hex
        self.team_name = "delivery-" + uuid.uuid4().hex[:10]
        self.checks = []
        self.worker_payloads = []

    def master(self, payload, step):
        if step == 0:
            return reply(payload, [raw_call("team_create", {"team_name": self.team_name})])
        if step == 1:
            return reply(payload, [raw_call("agent", {"name": "busy", "description": "Runs commands",
                                                      "prompt": "Run the commands you are given, one at a time."})])
        if step == 2:
            return reply(payload, [raw_call("send_message", {"to": "busy", "message": self.nonce}),
                                   raw_call("task_create", {"subject": "Delivery check",
                                                            "description": "Confirm mid-round task notices."})])
        if step == 3:
            found = re.findall(r"#(\d+)", after_last_tool_call(payload))
            return reply(payload, [raw_call("task_update", {"taskId": found[0] if found else "1",
                                                            "owner": "busy", "status": "in_progress"})])
        if step == 4:
            assert self.seen.wait(40), "Worker never reached its second step"
            return reply(payload, [raw_call("request_shutdown", {"to": "busy"})])
        assert self.approved.wait(40), "Worker did not approve shutdown"
        return reply(payload, content="Delivery scenario complete.")

    def worker(self, payload, step):
        self.worker_payloads.append(payload)
        tail = after_last_tool_call(payload) if step > 0 else ""
        prompt = last_user(payload)
        if prompt.startswith("The team leader has requested that you shut down"):
            self.checks.append("shutdown_prompt")
            self.approved.set()
            return reply(payload, [raw_call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        assert not prompt.startswith("You have been assigned task") or not self.between, \
            "A task assigned while busy must not start a new round"
        if step == 0:
            return reply(payload, [raw_call("run_shell_command", {"command": "sleep 8 # intentional-sleep: keep the teammate busy while messages arrive",
                                                                  "description": "First long step"})])
        if step == 1:
            if self.between:
                assert "<team_message_" in tail and self.nonce in tail, "Message not delivered after the tool call"
                assert "arrived while you were working" in tail, "Task notice not delivered mid-round"
                self.checks.append("mid_round_delivery")
            else:
                assert self.nonce not in json.dumps(payload["messages"]), "Native mode delivered mid-round"
                self.checks.append("native_waited")
            self.seen.set()
            return reply(payload, [raw_call("run_shell_command", {"command": "sleep 6 # intentional-sleep: keep the teammate busy while shutdown is requested",
                                                                  "description": "Second long step"})])
        if step == 2 and self.between:
            assert re.search(r"Runtime notice \[shutdown [0-9a-f]{8}\]: the team leader has requested that you shut down", tail)
            self.checks.append("shutdown_notice")
            return reply(payload, content="ENDING_TURN")
        if not self.between and prompt.startswith("You have been assigned task"):
            self.checks.append("native_task_round")
        return reply(payload, content="WORKER_STEP_DONE")

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            assert sum(self.steps.values()) <= 30, "Scenario did not converge"
        return self.master(payload, step) if role == "master" else self.worker(payload, step)
