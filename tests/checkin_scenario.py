"""Scripted check-in provider shared by local tests and container checks.

No pytest dependency and no external model calls.
"""
from collections import defaultdict
import json
import threading
import time
import uuid

from agent_company.runtime.relay import role_of


def call(name, args):
    return {"id": "test-" + uuid.uuid4().hex, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class CheckinScenario:
    """A slow worker keeps the team busy; the master is woken by check-ins."""

    def __init__(self, ws, status_file_holder):
        self.ws = ws
        self.steps = defaultdict(int)
        self.lock = threading.Lock()
        self.checkin_texts = []
        self.read_status = None
        self.done = threading.Event()
        self.shutdown_sent = None
        self.status_file_holder = status_file_holder

    def __call__(self, payload):
        role = role_of(payload)
        text = json.dumps(payload["messages"])
        users = [str(m.get("content", "")) for m in payload["messages"] if m.get("role") == "user"]
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
        if sum(self.steps.values()) > 80:
            raise RuntimeError("Scripted check-in fixture did not converge")
        calls, content = [], ""
        if role != "master":
            if "shutdown" in (users[-1] if users else "").lower():
                calls = [call("send_message", {"to": "leader", "message": "shutdown_approved"})]
            elif step == 0:
                calls = [call("run_shell_command", {"description": "Slow scripted build", "command":
                    "python3 -c 'import time; time.sleep(5)' && printf alpha > alpha.txt && git add alpha.txt && "
                    "git -c user.name=T -c user.email=t@localhost -c commit.gpgsign=false commit -qm alpha && echo BUILT"})]
            else:
                if "BUILT" not in text:
                    raise RuntimeError("The slow scripted build did not run: " + text[-400:])
                content = "ALPHA_DONE: committed alpha.txt"
                self.done.set()
        else:
            latest = payload["messages"][-1]
            latest_text = str(latest.get("content", ""))
            is_checkin = latest.get("role") == "user" and '<teammate_message from="runtime:checkin">' in latest_text
            if step == 0:
                calls = [call("team_create", {"team_name": "checkin-" + uuid.uuid4().hex[:6]})]
            elif step == 1:
                calls = [call("agent", {"name": "alpha", "subagent_type": "general-purpose",
                                        "description": "Slow build", "prompt": "Build alpha and commit."})]
            elif self.read_status == "requested" and latest.get("role") == "tool":
                self.read_status = "read" if '"workers"' in latest_text else "unreadable"
                content = "Check-in noted; nothing changes."
            elif self.done.is_set() and not self.shutdown_sent:
                self.shutdown_sent = time.time() * 1000
                calls = [call("request_shutdown", {"to": "alpha"})]
            elif is_checkin:
                self.checkin_texts.append(latest_text)
                if self.read_status is None:
                    self.read_status = "requested"
                    calls = [call("read_file", {"file_path": self.status_file_holder["path"]})]
                else:
                    content = "Check-in noted; nothing changes."
            elif self.shutdown_sent:
                content = "Alpha delivered; team shut down."
            else:
                content = "Waiting for alpha."
        return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": payload["model"],
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "tool_calls": calls},
                                  "finish_reason": "tool_calls" if calls else "stop"}],
                     "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
