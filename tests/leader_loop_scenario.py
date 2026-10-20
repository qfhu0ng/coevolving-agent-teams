"""Master status-read loop scenario shared by the local test and container checks.

No pytest imports. Only model responses are scripted; the installed Qwen CLI
runs the team, tools, loop guard, waiting and wake-ups (2026-10-02).
"""
from collections import defaultdict
import json
import re
import threading
import time
import uuid

from agent_company.runtime.relay import role_of

VARIANTS = ("status", "other", "mixed", "idle", "again")


def reply(payload, tools=(), content=""):
    return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()),
                 "model": payload["model"], "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": content, "tool_calls": list(tools)},
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


def unpaired_tool_calls(payload):
    """Call IDs without exactly one result, and results without a call."""
    calls = [item["id"] for message in payload["messages"] if message.get("role") == "assistant"
             for item in message.get("tool_calls") or ()]
    results = [message.get("tool_call_id") for message in payload["messages"] if message.get("role") == "tool"]
    return sorted({identifier for identifier in calls if results.count(identifier) != 1}
                  | (set(results) - set(calls)))


class StatusLoopScenario:
    """The master repeats one read_file in consecutive responses while a worker runs.

    status: the runtime status file (scope A). other: a repository file.
    mixed: the stopping read shares its response with another call.
    idle: the reads start only after the worker has finished its turn.
    again: after the first wake-up, while the worker still runs, the master
    polls the status file again until the guard stops it a second time.
    """

    def __init__(self, variant="status", worker_seconds=8.0):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.worker_seconds = variant, worker_seconds
        self.lock = threading.Lock()
        self.steps = defaultdict(int)
        self.master_requests = []     # (monotonic time, payload) for every master coding request
        self.worker_payloads = []
        self.read_ids = []            # IDs of the repeated read calls, in order
        self.read_times = []
        self.resumed = None           # first master payload carrying the recovery notice
        self.notices = []             # recovery notice markers seen by the master, in order
        self.worker_done = threading.Event()
        self.shutdown_sent = False

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            if sum(self.steps.values()) > 60:
                raise RuntimeError("Scripted leader-loop fixture did not converge")
        return self.master(payload, step) if role == "master" else self.worker(payload, step)

    def worker(self, payload, step):
        self.worker_payloads.append(payload)
        users = [text(message) for message in payload["messages"] if message.get("role") == "user"]
        if users and "shutdown" in users[-1].lower():
            return reply(payload, [call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        if step == 0:
            return reply(payload, [call("run_shell_command", {"description": "Slow scripted build",
                "command": f"python3 -c 'import time; time.sleep({self.worker_seconds})' && echo BUILT"})])
        self.worker_done.set()
        return reply(payload, content="ALPHA_DONE: build finished")

    def target(self, payload):
        system = "\n".join(text(message) for message in payload["messages"] if message.get("role") == "system")
        if self.variant == "other":
            prompt = "\n".join(text(message) for message in payload["messages"] if message.get("role") == "user")
            return re.search(r"Delivery repository: (\S+)", prompt).group(1) + "/original.txt"
        return re.search(r"read_file at (.+?\.json);", system).group(1)

    def master(self, payload, step):
        self.master_requests.append((time.monotonic(), payload))
        latest = text(payload["messages"][-1])
        notice = re.search(r"Runtime notice \[(leader-loop-\d+)\]:", latest)
        if notice and notice.group(1) not in self.notices:
            self.notices.append(notice.group(1))
            self.resumed = self.resumed or payload
        if step == 0:
            return reply(payload, [call("team_create", {"team_name": "loop-" + uuid.uuid4().hex[:6]})])
        if step == 1:
            return reply(payload, [call("agent", {"name": "alpha", "subagent_type": "general-purpose",
                                                  "description": "Slow build", "prompt": "Run the build."})])
        polling_again = self.variant == "again" and len(self.notices) == 1 and not self.worker_done.is_set()
        if (self.resumed is not None and not polling_again) or self.shutdown_sent:
            if not self.shutdown_sent and self.worker_done.is_set():
                self.shutdown_sent = True
                return reply(payload, [call("request_shutdown", {"to": "alpha"})])
            return reply(payload, content="Alpha delivered; team shut down." if self.shutdown_sent else "Waiting.")
        if self.variant == "idle" and not self.read_ids:
            # Start polling only once the worker is idle and its report is queued.
            if not self.worker_done.wait(60):
                raise RuntimeError("Worker did not finish")
            time.sleep(1.5)
        read = call("read_file", {"file_path": self.target(payload), "limit": 18})
        self.read_ids.append(read["id"])
        self.read_times.append(time.monotonic())
        extra = [call("glob", {"pattern": "*.txt"})] if self.variant == "mixed" and len(self.read_ids) >= 5 else []
        return reply(payload, [read, *extra])
