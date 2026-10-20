"""Role-policy rendering, disclosure drift guard and stderr lifecycle tail (S2a).

Specification: docs/stage1_role_policy_harness_plan_20261009.md v3.3 sections 3.3, 9.
"""
import json
from pathlib import Path
import re

import pytest

from agent_company.policy import role_policy

from agent_company.policy import role_policy_runtime as runtime
from agent_company.policy.role_policy import RolePolicyStore, parse_send_message, parse_spawn
from agent_company.policy.role_policy_runtime import LifecycleTail, disclosure, render_injection, render_role_state

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "docs/stage1_role_policy_harness_plan_20261009.md"
BLOCKS = {"B": "### 9.1 固定公共块 B", "Q": "修改依据块 Q", "V": "审批与复查块 V", "G_M": "master 块 G-M", "G_W": "worker 块 G-W", "R_M": "master 块 R-M",
          "R_W": "worker 块 R-W", "P_M": "master 附加块 P-M", "P_W": "worker 块 P-W", "D10_C": "#### D10-C"}


def plan_block(fragment):
    text = PLAN.read_text()
    start = text.index(fragment)
    return re.compile(r"```text\n(.*?)\n```", re.S).search(text, start).group(1).strip("\n")


@pytest.mark.parametrize("name,fragment", BLOCKS.items())
def test_disclosure_blocks_match_the_plan_verbatim(name, fragment):
    assert getattr(runtime, f"DISCLOSURE_{name}") == plan_block(fragment)


def test_disclosure_assembly_and_lengths_match_section_9_4():
    table = PLAN.read_text().split("### 9.4", 1)[1]
    expected = {mode: tuple(int(n) for n in re.search(rf"\| {mode} \|[^|]*\| (\d+) \|[^|]*\| (\d+)", table).groups())
                for mode in ("global", "master")}
    for mode, (master, worker) in expected.items():
        assert (len(disclosure(mode, "master")), len(disclosure(mode, "w1"))) == (master, worker)
    propose = re.search(r"\| propose \|[^|]*\| (\d+) \|[^|]*\| (\d+) / (\d+)", table).groups()
    assert (len(disclosure("propose", "master")), len(disclosure("propose", "w1"))) == (int(propose[0]), int(propose[1]))
    assert disclosure("off", "master") == disclosure("off", "w1") == ""
    assert disclosure("propose", "w1").endswith(runtime.DISCLOSURE_D10_C)
    assert "POLICY-DECISION" not in disclosure("master", "master") and "GLOBAL-POLICY" not in disclosure("propose", "w1")
    with pytest.raises(ValueError):
        disclosure("mystery", "master")


@pytest.mark.parametrize("mode", ["global", "master", "propose"])
def test_change_standard_reaches_every_role_and_review_only_the_master(mode):
    """v3.4: Q follows B for every role; V is the master's last block and never a worker's."""
    for role in ("master", "w1"):
        text = disclosure(mode, role)
        assert text.startswith(runtime.DISCLOSURE_B + runtime.BLOCK_SEPARATOR + runtime.DISCLOSURE_Q)
    assert disclosure(mode, "master").endswith(runtime.BLOCK_SEPARATOR + runtime.DISCLOSURE_V)
    assert runtime.DISCLOSURE_V not in disclosure(mode, "w1")


def test_the_review_prefix_named_in_v_is_ordinary_text_not_a_control_message():
    """The POLICY-REVIEW convention must never be parsed (or rejected) as a control header."""
    prefix = re.search(r"starting with (POLICY-REVIEW) <slot or C> v<k>:", runtime.DISCLOSURE_V).group(1)
    for first_line in (f"{prefix} C v2: improved", f"{prefix} tester v3: cannot judge yet", f"**{prefix}** C v1: no change"):
        assert parse_send_message("t1", "leader", first_line + "\nFacts: the next handoff named the consumer.") is None


# Every length limit the parser enforces, the disclosure that states it to the role that
# can hit it, and the sentence that carries the number. Values come from the parser.
DISCLOSED_LIMITS = [
    ("ROLE_POLICY_MAX_CHARS", "DISCLOSURE_R_M", r"complete new policy, at most (\d+) characters"),
    ("ROLE_POLICY_MAX_CHARS", "DISCLOSURE_R_M", r"The text limit is also (\d+) characters"),
    ("NOTE_MAX_CHARS", "DISCLOSURE_P_M", r"NOTE: below the header, at most (\d+) characters"),
    ("ROLE_POLICY_MAX_CHARS", "DISCLOSURE_P_M", r"replacement policy below it, also at most (\d+) characters"),
    ("PROPOSAL_MAX_CHARS", "DISCLOSURE_P_W", r"must total at most (\d+) characters"),
    ("GLOBAL_POLICY_MAX_CHARS", "DISCLOSURE_G_M", r"complete replacement C, at most (\d+) characters"),
]


@pytest.mark.parametrize("constant,block,pattern", DISCLOSED_LIMITS)
def test_every_enforced_limit_is_disclosed_with_the_enforced_value(constant, block, pattern):
    """Enforced limits must be stated to the role that can hit them, with the value in force."""
    match = re.search(pattern, getattr(runtime, block))
    assert match, (block, pattern)
    assert int(match.group(1)) == getattr(role_policy, constant)


def test_every_parser_limit_constant_is_disclosed():
    enforced = {name for name in dir(role_policy) if name.endswith("_MAX_CHARS")}
    assert enforced == {constant for constant, _, _ in DISCLOSED_LIMITS}


@pytest.mark.parametrize("message,constant", [
    ("POLICY-DECISION #1 approve\nNOTE: {}", "NOTE_MAX_CHARS"),
    ("POLICY-DECISION #1 amend\n{}", "ROLE_POLICY_MAX_CHARS"),
    ("ROLE-POLICY decoder\n{}", "ROLE_POLICY_MAX_CHARS"),
    ("GLOBAL-POLICY\n{}", "GLOBAL_POLICY_MAX_CHARS"),
])
def test_limits_accept_the_boundary_and_reject_beyond_it_with_the_limit_in_the_feedback(message, constant):
    limit = getattr(role_policy, constant)
    assert parse_send_message("t", "decoder", message.format("x" * limit)) is not None
    with pytest.raises(role_policy.PolicyRejection) as info:
        parse_send_message("t", "decoder", message.format("x" * (limit + 1)))
    assert info.value.code == "too_long" and info.value.details["limit"] == limit
    assert f"the limit is {limit}" in str(info.value)


def test_proposal_limit_counts_the_labels():
    limit = role_policy.PROPOSAL_MAX_CHARS
    frame = "CHANGE: \nREASON: r\nEVIDENCE: e"
    body = "CHANGE: " + "x" * (limit - len(frame)) + "\nREASON: r\nEVIDENCE: e"
    assert len(body) == limit and parse_send_message("t", "leader", "POLICY-PROPOSAL\n" + body) is not None
    with pytest.raises(role_policy.PolicyRejection) as info:
        parse_send_message("t", "leader", "POLICY-PROPOSAL\n" + body.replace("x", "xx", 1))
    assert info.value.details["limit"] == limit


def tick():
    now = [1_760_000_000.0]

    def clock():
        now[0] += 30
        return now[0]
    return clock


def propose_store():
    store = RolePolicyStore("propose", "Start C.", [].append, clock=tick())

    def respond(actor, calls):
        return store.commit(store.stage(actor, store.snapshot(actor), calls))
    respond("master", [parse_spawn("s1", "frontend", "Task"),
                       parse_spawn("s2", "decoder", "Task\nROLE-POLICY:\nTell frontend about flag changes.")])
    store.record_first_request("decoder")
    return store, respond


def test_rendering_is_a_pure_function_of_state():
    store, respond = propose_store()
    proposal = "POLICY-PROPOSAL\nCHANGE: New rule\nREASON: Rework\nEVIDENCE: Task 4"
    respond("decoder", [parse_send_message("p1", "leader", proposal)])
    master, worker = render_injection(store.snapshot("master")), render_injection(store.snapshot("decoder"))
    assert render_injection(store.snapshot("master")) == master  # clock advanced between calls
    respond("decoder", [parse_send_message("p2", "leader", proposal)])  # renotify: no state change
    assert render_injection(store.snapshot("master")) == master
    assert render_injection(store.snapshot("decoder")) == worker
    respond("decoder", [parse_send_message("p3", "leader", proposal.replace("Task 4", "Task 5"))])
    assert render_injection(store.snapshot("master")) != master


def test_master_and_worker_views():
    store, respond = propose_store()
    respond("decoder", [parse_send_message("p1", "leader", "POLICY-PROPOSAL\nCHANGE: New rule\nREASON: R\nEVIDENCE: E")])
    master = render_role_state(store.snapshot("master"))
    assert master.index("Slot decoder:") < master.index("Slot frontend:")
    assert "Slot frontend: instance frontend (running); no role policy; C applies." in master
    assert "Proposal #1 for slot decoder from decoder, base version 1" in master
    assert "CHANGE: New rule\nREASON: R\nEVIDENCE: E" in master
    injection = render_injection(store.snapshot("master"))
    assert injection.startswith("Communication policy C (version 1):\nStart C.\n\n" + runtime.DISCLOSURE_B)
    assert injection.index(runtime.DISCLOSURE_P_M) < injection.index("Role-policy snapshot")
    worker = render_role_state(store.snapshot("decoder"))
    assert worker.startswith("Your role slot: decoder; you are decoder. Your role policy version 1")
    assert "Your proposal #1 (base version 1) is pending" in worker and "Slot frontend" not in worker
    assert render_role_state(store.snapshot("frontend")) == (
        "Your role slot: frontend; you are frontend. No role policy is set for your slot; C applies.\n"
        "You have no pending proposal.")
    respond("master", [parse_send_message("d1", "decoder", "POLICY-DECISION #1 reject\nNOTE: not yet")])
    assert "Your proposal #1 was rejected. Note: not yet" in render_role_state(store.snapshot("decoder"))
    store.record_terminal("decoder")
    assert "instance decoder (terminal)" in render_role_state(store.snapshot("master"))


def test_global_mode_renders_only_versioned_c():
    store = RolePolicyStore("global", "Start C.", [].append, clock=tick())
    store.commit(store.stage("master", store.snapshot("master"),
                             [parse_send_message("g1", "*", "GLOBAL-POLICY\nSecond C.")]))
    assert render_role_state(store.snapshot("master")) == ""
    text = render_injection(store.snapshot("w1"))
    assert text == ("Communication policy C (version 2):\nSecond C.\n\n" + disclosure("global", "w1"))


def lifecycle(event, **fields):
    if event == "worker_terminal":
        fields.setdefault("status", "completed")
    return "Agent Company lifecycle: " + json.dumps({"event": event, "at": 1_760_000_000_000, **fields}) + "\n"


def test_lifecycle_tail_reads_complete_lines_once(tmp_path):
    path = tmp_path / "qwen_stderr.txt"
    seen = []
    tail = LifecycleTail(path, lambda worker, source: seen.append((worker, source)), clock=lambda: 1_760_000_002.5)
    assert tail.poll() == 0  # file not created yet
    with path.open("w") as stream:
        stream.write("native noise\n" + lifecycle("team_message_queued", to="decoder"))
        stream.write(lifecycle("worker_terminal", worker="decoder", status="failed"))
        stream.write(lifecycle("worker_terminal", worker="frontend")[:30])  # partial
    assert tail.poll() == 1 and seen == [("decoder", "stderr:3")]
    assert tail.summary()["partial_line_pending"]
    with path.open("a") as stream:
        stream.write(lifecycle("worker_terminal", worker="frontend")[30:])
        stream.write("Agent Company lifecycle: {not json\n")
        stream.write(lifecycle("worker_terminal", worker=7))
    assert tail.poll() == 1 and seen[-1] == ("frontend", "stderr:4")
    assert tail.poll() == 0
    summary = tail.summary()
    assert (summary["terminal_events"], summary["invalid_lines"], summary["lines_read"]) == (2, 2, 6)
    assert summary["max_consume_delay_seconds"] == 2.5 and not summary["partial_line_pending"]


def test_lifecycle_tail_feeds_the_store_and_survives_callback_errors(tmp_path):
    store, respond = propose_store()
    respond("decoder", [parse_send_message("p1", "leader", "POLICY-PROPOSAL\nCHANGE: C\nREASON: R\nEVIDENCE: E")])
    path = tmp_path / "qwen_stderr.txt"
    path.write_text(lifecycle("worker_terminal", worker="decoder") + lifecycle("worker_terminal", worker="ghost"))

    def on_terminal(worker, source):
        if store.record_terminal(worker, source=source) == "unknown":
            raise KeyError(worker)
    tail = LifecycleTail(path, on_terminal)
    tail.start(interval_seconds=0.01)
    tail.stop()
    assert store.snapshot("master").proposals == ()
    assert store.snapshot("decoder").proposals[0].void_reason == "proposer_terminal"
    assert tail.summary()["callback_errors"] == ["KeyError: 'ghost'"]
