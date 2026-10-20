"""agent_rules_v1 through the real native Qwen CLI with scripted model replies (plan v4 M4, zero API).

Only model replies are scripted. The CLI runs the team, native tools, git, message
delivery, check-ins and shutdown; the relay applies the approved texts. These runs
prove delivery and state transitions, not that a real model would follow a policy.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path
import re
import threading


from agent_company.policy import agent_rules_text as T
from agent_company.policy.role_package import create_m0_package
from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig, role_of
from role_policy_scenario import call, reply, system, text
from test_rejection_integration import native_qwen
from test_team import workspace

ROOT = Path(__file__).resolve().parents[1]
M0 = ROOT / "configs/stage1/policies/agent_rules_m0_20261017"
SHUTDOWN = "The team leader has requested that you shut down."
COMMIT = ("printf '{name}\\n' > {name}.txt && git add {name}.txt && git -c user.name=Fixture "
          "-c user.email=fixture@localhost -c commit.gpgsign=false commit -qm '{name} work' && git rev-parse HEAD")
ALPHA_BLOCK = ("POLICY:\nWHEN commit THEN SEND TO leader WITH sha\nWHEN task_done THEN SEND TO board WITH sha\n"
               "NOTES:\nOwn alpha.txt.\nEND-POLICY\n")
BETA_BLOCK = "POLICY:\nWHEN message FROM alpha THEN SEND TO alpha\nEND-POLICY\n"


def wait(seconds, label):
    # Native blocks foreground commands that start with sleep.
    return call("run_shell_command", {"command": f"python3 -c 'import time; time.sleep({seconds}); print(\"{label}\")'",
                                      "description": "Wait for the runtime"})


def texts(payload, roles=("user", "tool")):
    return "\n".join(text(m) for m in payload["messages"] if m.get("role") in roles)


def tail(payload):
    return text(payload["messages"][-1])


class AgentRulesScenario:
    def __init__(self):
        self.lock = threading.Lock()
        self.steps = defaultdict(int)
        self.requests = defaultdict(list)
        self.flags = set()
        self.ids = {}
        self.outputs = {}
        self.spawns = {}  # call id -> spawn call, to resend one the lesson gate held

    def once(self, flag):
        if flag in self.flags:
            return False
        self.flags.add(flag)
        return True

    def __call__(self, payload):
        role = role_of(payload)
        with self.lock:
            step = self.steps[role]
            self.steps[role] += 1
            self.requests[role].append(copy.deepcopy(payload))
            assert sum(self.steps.values()) < 400, "Bounded native fixture failed to converge"
            handler = getattr(self, role, None) or self.worker_default
            return handler(payload, step)

    # -- master ------------------------------------------------------------------------
    def master(self, payload, step):
        seen, now = texts(payload), tail(payload)
        if step == 0:
            return reply(payload, [call("team_create", {"team_name": "rules-fixture"})])
        if step == 1:
            return reply(payload, [call("task_create", {"subject": "Docx tables fixture",
                                                        "description": "Writing scope: alpha.txt",
                                                        "activeForm": "Working on the fixture"})])
        if step == 2:
            return reply(payload, [self.spawn("alpha", ALPHA_BLOCK)])
        if step == 3:
            return reply(payload, [self.spawn("beta", BETA_BLOCK)])
        if step == 4:
            return reply(payload, [call("task_update", {"taskId": "1", "owner": "alpha", "status": "in_progress"})])
        for message in payload["messages"]:
            held = message.get("tool_call_id") in self.spawns and "(lesson_gate)" in text(message)
            if message.get("role") == "tool" and held and self.once("resent " + message["tool_call_id"]):
                again = call("agent", json.loads(self.spawns[message["tool_call_id"]]["function"]["arguments"]))
                self.spawns[again["id"]] = again
                return reply(payload, [again])
        pending = re.search(r"#(\d+) from beta for alpha, operation add", now)
        if pending and self.once("decided"):
            return reply(payload, [call("send_message", {"to": "beta", "message": f"POLICY-DECISION #{pending[1]} approve"})])
        lessons = re.search(r"Lessons recorded: (\d+)", now)
        if lessons and int(lessons[1]) >= 1 and self.once("gamma_tried"):
            return reply(payload, [self.spawn("gamma", "POLICY:\nWHEN start THEN SEND TO leader\nEND-POLICY\n")])
        if 'Teammate "gamma" is now running' in seen and self.once("files_read"):
            card = system(payload)
            paths = [re.search(r"change log is in (\S+) \(read_file\)", card)[1],
                     re.search(r"Lessons from teammates are in (\S+?)\.(?:\s|$)", card)[1],
                     re.search(r"the whole board is in (\S+?)\.(?:\s|$)", card)[1]]
            self.outputs["master_files"] = paths
            return reply(payload, [call("read_file", {"file_path": p}) for p in paths])
        if "files_read" in self.flags and self.once("beta_updated"):
            return reply(payload, [call("send_message", {"to": "beta", "message":
                                                         "AGENT-POLICY beta notes\nCheck the board before starting a task."})])
        if "beta_updated" in self.flags and self.once("unowned_task"):
            return reply(payload, [call("task_create", {"subject": "Unowned follow-up", "description": "Not assigned.",
                                                        "activeForm": "Waiting"})])
        if "unowned_task" in self.flags and self.once("broadcast"):
            return reply(payload, [call("send_message", {"to": "*", "message": "BOARD: Freeze new features; integrate and verify."})])
        done = all(f"DONE_{name}" in seen for name in ("alpha", "beta", "gamma"))
        if "Collaboration observations:" in seen:
            self.flags.add("checkin_seen")
        if done and "checkin_seen" in self.flags and self.once("alpha_shutdown"):
            return reply(payload, [call("request_shutdown", {"to": "alpha"})])
        alpha_listed = re.search(r"^- alpha: version", now, re.M)
        if "alpha_shutdown" in self.flags and "shutdown_approved" in seen and not alpha_listed and self.once("delta"):
            delta = call("agent", {"name": "delta", "subagent_type": "general-purpose",
                                   "description": "Continue alpha", "prompt": "Continue the fixture.\nSUCCEEDS: alpha\n"})
            self.spawns[delta["id"]] = delta
            return reply(payload, [delta])
        if "DONE_delta" in seen and self.once("final_shutdown"):
            return reply(payload, [call("request_shutdown", {"to": name}) for name in ("beta", "gamma", "delta")])
        if "final_shutdown" in self.flags and len(re.findall(r'from="(beta|gamma|delta)">\s*shutdown_approved', seen)) >= 3:
            return reply(payload, content="Team complete.")
        if "alpha_shutdown" in self.flags and not alpha_listed and "delta" not in self.flags:
            return reply(payload, [call("glob", {"pattern": f"*.{step}"})])  # keep turning until alpha has ended
        return reply(payload, content="Waiting for teammates.")

    def spawn(self, name, block):
        spawn = call("agent", {"name": name, "subagent_type": "general-purpose", "description": f"{name} fixture",
                               "prompt": f"Complete the {name} fixture.\n{block}"})
        self.spawns[spawn["id"]] = spawn
        return spawn

    # -- workers -------------------------------------------------------------------------
    def shutdown(self, payload, name, lesson=None):
        if SHUTDOWN not in texts(payload):
            return None
        calls = []
        if lesson and T.REFLECT_SHUTDOWN in texts(payload) and self.once(f"{name}_shutdown_lesson"):
            calls.append(call("send_message", {"to": "leader", "message": "LESSON: " + lesson}))
        calls.append(call("send_message", {"to": "leader", "message": "shutdown_approved"}))
        return reply(payload, calls)

    def alpha(self, payload, step):
        seen, now = texts(payload), tail(payload)
        stop = self.shutdown(payload, "alpha", "Hand over alpha.txt with its last commit.")
        if stop:
            return stop
        if step == 0:
            assert "Your communication policy (version 0):" in now and "R1: WHEN commit THEN SEND TO leader WITH sha" in now
            return reply(payload, [call("send_message", {"to": "leader", "message":
                                                         "POLICY-PROPOSAL self notes\nOwn alpha.txt and its tests.\nREASON: scope"})])
        if "alpha_commit" not in self.flags and "WHEN commit THEN SEND TO beta" not in now:
            # Commit only once beta's approved rule is in force, so the one commit triggers R1 and R3 alike;
            # otherwise the commit sample can precede the approval (seen once in Docker, 2026-10-18).
            return reply(payload, [wait(1, f"approval {step}")])
        if self.once("alpha_commit"):
            return reply(payload, [call("run_shell_command", {"command": COMMIT.format(name="alpha"),
                                                              "description": "Commit the fixture change"})])
        reminder = re.search(r"Rule R1 applies now: a new commit ([0-9a-f]{12})", seen)
        if not reminder:
            return reply(payload, [wait(2, f"wait {step}")])
        if self.once("alpha_reported"):
            self.outputs["alpha_sha"] = reminder[1]
            return reply(payload, [call("send_message", {"to": "leader", "message": f"Committed.\nSHA: {reminder[1]}"}),
                                   call("send_message", {"to": "beta", "message": f"Reader interface ready.\nSHA: {reminder[1]}"})])
        if self.once("alpha_task_done"):
            return reply(payload, [call("task_update", {"taskId": "1", "status": "completed"})])
        if T.REFLECT_TASK_DONE.format(task_id="1") in seen and self.once("alpha_lesson"):
            sha = self.outputs["alpha_sha"]
            return reply(payload, [call("send_message", {"to": "leader", "message": "LESSON: Run the fixture check before reporting."}),
                                   call("send_message", {"to": "leader", "message": f"BOARD: alpha fixture done\nSHA: {sha}"})])
        if "alpha_lesson" not in self.flags:
            return reply(payload, [wait(1, f"prompt {step}")])
        if "WHEN commit THEN SEND TO beta" not in now:
            return reply(payload, [wait(1, f"policy {step}")])
        if self.once("alpha_done"):
            self.outputs["alpha_final_tail"] = now
            return reply(payload, [call("send_message", {"to": "leader", "message": "DONE_alpha"})])
        return reply(payload, content="alpha idle")

    def beta(self, payload, step):
        seen, now = texts(payload), tail(payload)
        stop = self.shutdown(payload, "beta")
        if stop:
            return stop
        if step == 0:
            return reply(payload, [call("send_message", {"to": "leader", "message":
                                                         "POLICY-PROPOSAL alpha add\nWHEN commit THEN SEND TO beta WITH sha\n"
                                                         "REASON: I consume its interface"})])
        if "Rule R1 applies now: a message from alpha reached you" in seen and self.once("beta_ack"):
            return reply(payload, [call("send_message", {"to": "alpha", "message": "Ack."})])
        if "[B1]" in now and self.once("beta_board_seen"):
            self.outputs["board_path"] = re.search(r"all in (\S+)\):", now)[1]
            self.outputs["beta_board_tail"] = now
            return self.read_board(payload)
        if "beta_board_seen" in self.flags and "beta_board" not in self.flags:
            reads = [text(m) for m in payload["messages"] if m.get("role") == "tool"
                     and m.get("tool_call_id") in (self.ids["beta_read"], self.ids["beta_shell"])]
            if len(reads) == 2 and all("alpha fixture done" in r for r in reads):
                self.flags.add("beta_board")
            elif self.outputs["board_reads"] < 10:
                # The relay rewrites the read-only files in the background (one docker exec per file
                # in Docker), so a read right after the tail shows a post can still find the old file;
                # a real model reads it a turn later. Wait, then read again.
                if self.once(f"board_wait_{self.outputs['board_reads']}"):
                    return reply(payload, [wait(1, f"board {step}")])
                return self.read_board(payload)
        if "beta_board" in self.flags and self.once("beta_done"):
            self.outputs["beta_roster"] = [p for p in self.requests["beta"] if "Active teammates (as of" in tail(p)]
            return reply(payload, [call("send_message", {"to": "leader", "message": "DONE_beta"})])
        if "beta_done" not in self.flags:
            return reply(payload, [wait(2, f"beta {step}")])
        return reply(payload, content="beta idle")

    def read_board(self, payload):
        board = self.outputs["board_path"]
        read, shell = (call("read_file", {"file_path": board}),
                       call("run_shell_command", {"command": f"cat {board}", "description": "Read the board"}))
        self.ids["beta_read"], self.ids["beta_shell"] = read["id"], shell["id"]
        self.outputs["board_reads"] = self.outputs.get("board_reads", 0) + 1
        return reply(payload, [read, shell])

    def gamma(self, payload, step):
        stop = self.shutdown(payload, "gamma")
        if stop:
            return stop
        if self.once("gamma_done"):
            return reply(payload, [call("send_message", {"to": "leader", "message": "DONE_gamma"})])
        if "final_shutdown" not in self.flags:
            # Keeps one teammate working, so the master waits for check-ins and messages.
            return reply(payload, [wait(3, f"gamma {step}")])
        return reply(payload, content="gamma idle")

    def delta(self, payload, step):
        stop = self.shutdown(payload, "delta")
        if stop:
            return stop
        if self.once("delta_done"):
            self.outputs["delta_tail"] = tail(payload)
            return reply(payload, [call("send_message", {"to": "leader", "message": "DONE_delta"})])
        return reply(payload, content="delta idle")

    def worker_default(self, payload, step):
        return reply(payload, content="idle")


def test_agent_rules_full_cycle_through_the_native_cli(tmp_path):
    cli, node = native_qwen()
    scenario = AgentRulesScenario()
    package = create_m0_package((M0 / "assignment.md").read_text(), (M0 / "communication.md").read_text())
    result = None
    try:
        result = QwenTeamRunner(str(cli), node=node, upstream=scenario, role_policy_mode="propose", policy_package=package,
                                checkin_interval_seconds=30, team_message_delivery="between_tool_calls",
                                runtime_disclosure=True, leader_loop_recovery=True, tool_call_cap="disabled").run(
            workspace(tmp_path), "Complete the synthetic fixture.", package["team"]["assignment"],
            package["team"]["communication"], tmp_path / "run",
            ProviderConfig("native-scripted-model", "http://unused.invalid", request_timeout_seconds=10),
            timeout_seconds=600)
    finally:
        (tmp_path / "requests.json").write_text(json.dumps(dict(scenario.requests), ensure_ascii=False, indent=1))
        (tmp_path / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    assert_full_cycle(tmp_path / "run", scenario, result, tmp_path)


def assert_full_cycle(run, scenario, result, evidence):
    """Every check of the full cycle; also run against the real ProgramBench Docker workspace before launch."""
    assert result["stop_reason"] == "native_cli_completed", result.get("stop_reason")
    assert result["execution"]["returncode"] == 0 and result["usage_complete"] and not result["errors"]
    events = [json.loads(line) for line in (run / "relay/events.jsonl").read_text().splitlines()]
    # Tool listeners attach before a teammate's first turn (native patch, 2026-10-18).
    assert '"event":"agent_start_patch_applied"' in (run / "qwen_stderr.txt").read_text()
    named = defaultdict(list)
    for event in events:
        named[event["event"]].append(event)

    # Approved texts reach the agents; old fixed protocols are not assembled.
    master_system, alpha_system = system(scenario.requests["master"][0]), system(scenario.requests["alpha"][0])
    assert T.MASTER_CARD.split("{")[0] in master_system and (M0 / "communication.md").read_text().strip() in master_system
    assert T.WORKER_CARD.split("{")[0] in alpha_system and "Fixed collaboration protocol;" not in alpha_system
    assert "COORD" not in master_system and "Assignment principles" not in alpha_system

    # Self change applies at once; beta's proposal for alpha is approved and reaches alpha.
    assert "Own alpha.txt and its tests." in scenario.outputs["alpha_final_tail"]
    assert "WHEN commit THEN SEND TO beta WITH sha" in scenario.outputs["alpha_final_tail"]

    # Rule obligations: commit reported with its SHA, board post after task_done, message FROM.
    rules = json.loads((run / "relay/rule_obligations.json").read_text())
    status = {(o["worker"], o["rule_id"]): o["status"] for o in rules["obligations"]}
    assert status[("alpha", "R1")] == "satisfied" and status[("alpha", "R2")] == "satisfied"  # commit, task_done -> board
    assert status[("alpha", "R3")] == "satisfied"  # beta's approved rule: the commit, with its SHA, to beta
    assert status[("beta", "R1")] == "satisfied" and status[("gamma", "R1")] == "satisfied"

    # Lessons: the gate held gamma once with the lesson, then let the re-spawn through.
    # gamma's spawn is held with the whole library; later delta's with only the lesson written since.
    held = named["lesson_gate_held"]
    assert [(h["lessons_given"], h["full"]) for h in held] == [(["L1"], True), (["L2"], False)]
    assert len(named["lesson_gate_released"]) == 2
    lessons = {change["lesson"]["id"]: change["lesson"]["trigger"] for batch in named["agent_policy_batch"]
               for change in batch["changes"] if change["event"] == "agent_lesson_recorded"}
    assert lessons["L1"] == "task_done" and "shutdown" in lessons.values()
    funnel = {(e["worker"], e["kind"]): e["outcome"] for e in named["reflection_funnel"]}
    assert funnel[("alpha", "task_done")] == "output"

    # The board: the post reached beta's tail as new, and beta could read the file (plan Q1).
    beta_tails = [tail(p) for p in scenario.requests["beta"] if "[B1]" in tail(p)]
    assert "[B1] new alpha" in beta_tails[0] and all("[B1] new" not in t for t in beta_tails[1:])
    results = {m["tool_call_id"]: text(m) for p in scenario.requests["beta"] for m in p["messages"]
               if m.get("role") == "tool" and m.get("tool_call_id") in (scenario.ids["beta_read"], scenario.ids["beta_shell"])}
    (evidence / "q1_worker_reads.json").write_text(json.dumps(
        {"board_reads": scenario.outputs["board_reads"], "latest": results}, ensure_ascii=False, indent=1))
    assert "alpha fixture done" in results[scenario.ids["beta_shell"]]
    assert "alpha fixture done" in results[scenario.ids["beta_read"]], results[scenario.ids["beta_read"]][:300]

    # The master read its three files; the roster showed real teammates; check-in paragraph appeared.
    master_seen = "\n".join(texts(p) for p in scenario.requests["master"])
    assert "# Teammate policies (written by the runtime" in master_seen and "# Lessons from teammates" in master_seen
    assert "# Team board" in master_seen and "Collaboration observations:" in master_seen
    assert "Container memory: " in master_seen  # numbers in Docker; unknown where cgroup v2 is unreadable
    assert scenario.outputs["beta_roster"] and "- alpha:" in tail(scenario.outputs["beta_roster"][-1])

    # Succession: delta started from alpha's latest policy.
    assert "R1: WHEN commit THEN SEND TO leader WITH sha" in scenario.outputs["delta_tail"]
    assert "WHEN commit THEN SEND TO beta WITH sha" in scenario.outputs["delta_tail"]

    # Native send records reach the relay; the unowned task was never auto-claimed.
    outcomes = named["message_outcome"]
    assert any(o["sender"] == "alpha" and o["ok"] for o in outcomes) and any(o["sender"] == "master" for o in outcomes)
    broadcasts = [o for o in outcomes if o["native_event"] == "team_broadcast" and o["sender"] == "master"]
    assert broadcasts and {r["name"] for r in broadcasts[0]["recipients"]} >= {"alpha", "beta", "gamma"}
    board = [change["post"] for batch in named["agent_policy_batch"] for change in batch["changes"]
             if change["event"] == "agent_board_posted"]
    assert [(post["id"], post["author"], post["to"]) for post in board] == [("B1", "alpha", "leader"), ("B2", "master", "*")]
    unowned = [e for e in named["collaboration_task_observed"] if e.get("task_id") == "2"]
    assert unowned and all(not e.get("owner") for e in unowned)
