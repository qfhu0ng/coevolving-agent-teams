"""Scripted Single/off-Team file roundtrip for the real native CLI, without pytest."""
from collections import defaultdict
import threading

from agent_company.runtime.relay import COORD_TOOLS, role_of
from role_policy_scenario import call, reply, user_texts


class BasicSmokeScenario:
    def __init__(self, execution_mode, marker_path="/workspace/solution/native-smoke.txt"):
        if execution_mode not in ("single", "team"):
            raise ValueError(execution_mode)
        self.execution_mode = execution_mode
        self.marker_path = marker_path
        self.steps = defaultdict(int)
        self.lock = threading.Lock()
        self.file_verified = False
        self.shutdown_sent = False

    def __call__(self, payload):
        role = "single" if self.execution_mode == "single" else role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            if sum(self.steps.values()) > 40:
                raise RuntimeError("Scripted file smoke did not converge")
        if role == "master":
            return self.master(payload, step)
        if role == "single":
            tools = {tool["function"]["name"] for tool in payload.get("tools", [])}
            assert not tools & COORD_TOOLS, "Single received coordination tools"
        elif "shutdown" in (user_texts(payload) or [""])[-1].lower():
            return reply(payload, [call("send_message", {"to": "leader", "message": "shutdown_approved"})])
        if step == 0:
            return reply(payload, [call("write_file", {"file_path": self.marker_path, "content": "NATIVE_OK\n"})])
        if step == 1:
            return reply(payload, [call("read_file", {"file_path": self.marker_path})])
        if step == 2:
            results = [message for message in payload["messages"] if message.get("role") == "tool"]
            assert results and "NATIVE_OK" in str(results[-1].get("content", "")), "Native file readback failed"
            self.file_verified = True
        return reply(payload, content="BASELINE_SMOKE_FILE_VERIFIED" if self.file_verified else "Waiting.")

    def master(self, payload, step):
        if step == 0:
            return reply(payload, [call("team_create", {"team_name": "baseline-smoke"})])
        if step == 1:
            return reply(payload, [call("agent", {"name": "alpha", "subagent_type": "general-purpose",
                                                   "description": "Synthetic file roundtrip",
                                                   "prompt": "Write and read back the synthetic smoke marker, then report."})])
        if self.shutdown_sent:
            return reply(payload, content="Team shut down.")
        if any("BASELINE_SMOKE_FILE_VERIFIED" in body for body in user_texts(payload)):
            self.shutdown_sent = True
            return reply(payload, [call("request_shutdown", {"to": "alpha"})])
        return reply(payload, content="Waiting for the file roundtrip report.")
