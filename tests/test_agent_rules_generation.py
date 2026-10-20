"""Plan v4 section 12: the generate-only entry, end to end with scripted provider replies (zero API).

Covers the B1 acceptance (reasoning sent back verbatim over several tool rounds, never on disk,
through the production assembly path) and the item 6 acceptance (new projections readable,
forbidden material not, capabilities match the runtime, unchecked parts not shown as passed,
no runner or deployment call).
"""
import json
from pathlib import Path
import time

import pytest

from agent_company.analysis.agent_rules_evidence import agent_rules_record
from agent_company.evolution import agent_rules_generation, agent_rules_outer_text as outer_text
from agent_company.evolution.harness_capabilities import target_harness
from agent_company.policy import agent_rules_runtime
from agent_company.policy.role_package import create_m0_package, validate_agent_rules_package
from agent_company.runtime.relay import ProviderConfig, TeamRelay
from agent_company.runtime.team import WorkerRegistry
from test_team import call as relay_call, request as relay_request, response as relay_response, workspace

SENTINEL = "SENTINEL-OUTER-REASONING-91c4"
FORBIDDEN = "FORBIDDEN-MARKER-5d2e"
INFERENCE = {"agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 61440, "memory_swap_mb": 61440,
             "max_model_calls": None, "team_checkin_seconds": 1800, "team_message_delivery": "between_tool_calls",
             "runtime_disclosure": True, "leader_loop_recovery": True, "tool_call_cap": "disabled",
             "role_policy_mode": "propose", "role_policy_protocol": "agent_rules_v1"}
LESSON = "Run the fixture check before reporting a fix."
NEW_RULE = "When a worker finishes a task, it posts the commit and the checks to the board."


def provider():
    return ProviderConfig(model="fake-only", base_url="https://provider.invalid/v1",
                          api_key="fake-test-key-never-used-on-network", max_output_tokens=4096, max_attempts=1)


class Upstream:
    def __init__(self):
        self.queue = []

    def __call__(self, payload):
        return 200, self.queue.pop(0)


def m0_package():
    return create_m0_package("# Assignment principles\n\nA0", "# Communication principles\n\nC0")


def recorded_trial(tmp_path, package=None) -> Path:
    """A trial directory whose relay records come from the real relay, plus forbidden material."""
    package = package or m0_package()
    trial = tmp_path / "trial"
    relay_dir = trial / "team" / "relay"
    up = Upstream()
    files = {name: str(tmp_path / f"runtime/status/{name}.md") for name in ("policies", "lessons", "board")}
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      "C", relay_dir, 5000, up, role_policy_mode="propose",
                      policy_package=package, policy_files=files)

    def turn(actor, calls=()):
        tools = (("agent", "send_message", "task_create", "task_update") if actor == "master"
                 else ("send_message", "task_update"))
        up.queue.append(relay_response(list(calls)))
        return relay.handle(relay_request(actor, tools))

    prompt = "Implement the reader.\nPOLICY:\nWHEN commit THEN SEND TO leader WITH sha\nEND-POLICY\n"
    turn("master", [relay_call("agent", {"name": "alpha", "description": "work", "prompt": prompt})])
    turn("alpha")
    sha = "ab12cd34" * 5
    relay.collaboration.observe_git("alpha", {"commit": sha, "new_commit": True, "changed_paths": ["src/A.hs"],
                                              "committed_at": None, "paths": ["src/A.hs"]},
                                    {"native_sample": int(time.time() * 1000)})
    turn("alpha", [relay_call("send_message", {"to": "leader", "message": f"Committed src/A.hs.\nSHA: {sha[:9]}"}),
                   relay_call("send_message", {"to": "leader", "message": "LESSON: " + LESSON}),
                   relay_call("send_message", {"to": "leader", "message": "BOARD: reader done"})])
    relay.record_lifecycle_event({"event": "team_message_sent", "from": "alpha", "to": "leader",
                                  "hash": __import__("agent_company.runtime.rule_engine", fromlist=["x"]).message_hash(
                                      f"Committed src/A.hs.\nSHA: {sha[:9]}"), "ok": True}, "stderr:1")
    relay.collaboration.observe_git("master", {"commit": sha, "new_commit": True, "changed_paths": ["src/A.hs"],
                                               "committed_at": None, "paths": ["src/A.hs"]},
                                     {"native_sample": int(time.time() * 1000) + 1})
    relay.finish_rule_engine(relay_dir / "rule_obligations.json")
    (trial / "run_request.json").write_text(json.dumps({"assignment_policy": package["team"]["assignment"],
                                                       "communication_policy": package["team"]["communication"],
                                                       "policy_package": package}))
    (trial / "task_result.json").write_text(json.dumps({
        "status": "completed", "score": 0.5, "task_id": "fixture", "repo_id": "fixture/repo",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cached_prompt_tokens": 0, "total_tokens": 12},
        "metrics": {"verifier": {"status": "completed", "passed_tests": 1, "valid_tests": 2, "reward": 0.5}}}))
    # Material the outer loop must never see.
    (relay_dir / "attempt_00001_master_provider_request.json").write_text(json.dumps({"messages": FORBIDDEN}))
    (trial / "team" / "qwen_stdout.txt").write_text(FORBIDDEN)
    (trial / "hidden_tests").mkdir()
    (trial / "hidden_tests" / "test.txt").write_text(FORBIDDEN)
    return trial


def tool(name, args, identifier):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Script:
    """Scripted assistant turns, each with its own reasoning sentinel; records every payload."""

    def __init__(self, label, turns):
        self.label, self.turns, self.payloads = label, list(turns), []

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        step = len(self.payloads)
        turn = self.turns.pop(0)
        calls = turn(self.payloads[-1]) if callable(turn) else turn
        message = {"role": "assistant", "content": None, "reasoning_content": f"{SENTINEL}-{self.label}-{step}",
                   "tool_calls": calls}
        return 200, {"choices": [{"message": message, "finish_reason": "tool_calls"}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    def reasoning_seen(self, index):
        return [m.get("reasoning_content") for m in self.payloads[index]["messages"] if m["role"] == "assistant"]


def optimizer_script():
    candidate_c = "# Communication principles\n\n" + NEW_RULE
    submit = {"assignment_policy": "# Assignment principles\n\nKeep one owner per module.",
              "communication_policy": candidate_c, "rationale": "The lesson shows unchecked fix reports.",
              "evidence_citations": [{"path": "run_1/agent_rules.json", "quote": LESSON,
                                      "supports": "A worker recorded unchecked fix reports."}],
              "execution_notes": [{"policy_part": "communication", "rule_quote": NEW_RULE, "actor": "master",
                                   "observable_trigger": "task_done", "existing_action": "AGENT-POLICY add",
                                   "capability_refs": ["/online_policy/rule_triggers", "/online_policy/board"],
                                   "timing_and_fallback": "At spawn.", "cost_and_limitations": "One post per task."}]}
    return Script("optimizer", [
        [tool("list_training_files", {}, "o1")],
        [tool("read_training_json", {"path": "run_1/agent_rules.json", "pointer": "/organization_timeline"}, "o2"),
         tool("read_training_json", {"path": "run_1/agent_rules.json", "pointer": "/information_use/totals"}, "o3")],
        [tool("search_training_files", {"query": FORBIDDEN}, "o4"),
         tool("read_training_file", {"path": "run_1/agent_rules.json"}, "o5")],
        [tool("record_research_note", {"note": "Lessons show unchecked fixes.", "evidence_refs": ["run_1/agent_rules.json"]}, "o6")],
        [tool("submit_policy", submit, "o7")],
    ])


def reviewer_script():
    submit = {"recommendation": "uncertain", "rationale": "One run only.",
              "executability": [{"policy_part": "communication", "rule_quote": NEW_RULE,
                                 "capability_refs": ["/online_policy/board"], "verdict": "supported",
                                 "reason": "Board posts exist."}],
              "task_specificity": {"checked": ["assignment"], "findings": []}}
    return Script("reviewer", [
        [tool("list_training_files", {}, "r1")],
        [tool("read_training_json", {"path": "run_1/agent_rules.json", "pointer": "/lessons"}, "r2")],
        [tool("record_research_note", {"note": "Checked the lesson.", "evidence_refs": ["run_1/agent_rules.json"]}, "r3")],
        [tool("submit_review", submit, "r4")],
    ])


def tool_results(script):
    """Each tool result as the scripted model first saw it (later requests may show it elided), by call id."""
    seen = {}
    for payload in script.payloads:
        for m in payload["messages"]:
            if m.get("role") == "tool":
                seen.setdefault(m["tool_call_id"], m["content"])
    return seen


@pytest.fixture
def no_runs(monkeypatch):
    from agent_company.programbench.runner import ProgramBenchRunner
    from agent_company.runtime.qwen import QwenTeamRunner

    def forbidden(*args, **kwargs):
        raise AssertionError("The generate-only entry must not start a run")
    monkeypatch.setattr(ProgramBenchRunner, "__call__", forbidden)
    monkeypatch.setattr(QwenTeamRunner, "run", forbidden)


def test_generation_entry_end_to_end(tmp_path, no_runs):
    trial = recorded_trial(tmp_path)
    m0 = m0_package()
    package_path = tmp_path / "m0.json"
    package_path.write_text(json.dumps(m0))
    optimizer, reviewer = optimizer_script(), reviewer_script()
    report = agent_rules_generation.generate(
        package=m0, trials=[trial], inference=INFERENCE, provider=provider(), output=tmp_path / "gen",
        candidate_id="M1", package_path=package_path, code_version="fixture", native_version="0.23.1",
        optimizer_completion=optimizer, reviewer_completion=reviewer)
    out = tmp_path / "gen"
    assert report["status"] == "reviewed" and report["run_dispatched"] is False and report["adopted"] is False
    assert report["requires_user_decision"] is True

    # B1: every later request carries all earlier reasoning verbatim; nothing on disk has it.
    for script, label in ((optimizer, "optimizer"), (reviewer, "reviewer")):
        assert len(script.payloads) >= 4
        for index in range(1, len(script.payloads)):
            assert script.reasoning_seen(index) == [f"{SENTINEL}-{label}-{n}" for n in range(1, index + 1)]
    leaked = [str(p) for p in out.rglob("*") if p.is_file() and SENTINEL in p.read_text(encoding="utf-8")]
    assert leaked == []

    # The candidate is a valid schema 2 package one revision after M0, pointing at its review.
    package = validate_agent_rules_package(json.loads((out / "package.json").read_text()))
    assert package["package_id"] == "M1" and package["revision"] == 1 and package["review_ref"] == "review.json"
    assert package["parent"] == {"package_id": m0["package_id"], "revision": 0}
    assert NEW_RULE in package["team"]["communication"]
    generation = json.loads((out / "generation.json").read_text())
    assert generation["run_dispatched"] is False and len(generation["incumbent"]["sha256"]) == 64

    # Unchecked parts are not passed; the communication part was not checked for task specificity.
    assessment = report["review"]["assessment"]
    assert assessment["task_specificity"]["unchecked"] == ["communication"]
    assert assessment["counts"]["supported"] == 1 and assessment["unassessed_notes"] == []

    # The new projections are readable through the evidence tools; forbidden material is not.
    results = tool_results(optimizer)
    assert '"teammates"' in results["o2"] and '"commit_appeared_after"' in results["o3"]
    assert json.loads(results["o4"]).get("matches") in ([], None) and FORBIDDEN not in results["o4"]
    evidence_files = sorted(str(p.relative_to(out / "training_evidence")) for p in (out / "training_evidence").rglob("*")
                            if p.is_file())
    assert evidence_files == ["run_1/agent_rules.json", "run_1/master_strategy.json", "run_1/projection.json",
                              "run_1/task_summary.json"]
    assert all(FORBIDDEN not in (out / "training_evidence" / f).read_text() for f in evidence_files)

    # The capability description the optimizer received is the agent_rules_v1 runtime, with the real cards.
    prompt = json.loads(optimizer.payloads[0]["messages"][1]["content"])
    harness = prompt["target_harness"]
    assert harness == target_harness(INFERENCE, execution_mode="team", provider=provider(), code_version="fixture",
                                     native_version="0.23.1", policy_protocol="agent_rules_v1")
    assert harness["online_policy"]["master_disclosure"]["text"] == agent_rules_runtime.master_card(
        "<policies file>", lessons_file="<lessons file>", board_file="<board file>")
    assert prompt["selection_mode"] == outer_text.MODE and prompt["instruction"] == outer_text.GENERATION_INSTRUCTION
    # The systems actually sent are the approved ones (tests/test_agent_rules_outer_text_approval.py).
    approved = json.loads((Path(__file__).parent / "fixtures/agent_rules_outer_texts_approved_20261018.json").read_text())
    for sent, key in ((optimizer.payloads[0]["messages"][0]["content"], "optimizer_system"),
                      (reviewer.payloads[0]["messages"][0]["content"], "reviewer_system")):
        # Followed only by the per-call budget notice, which is runtime data.
        assert sent == approved[key] or sent.startswith(approved[key] + "\n\nResearch-loop budget: {"), key


def test_agent_rules_record_contents(tmp_path):
    record = agent_rules_record(recorded_trial(tmp_path))
    assert record["replay"] == "ok" and record["protocol"] == "agent_rules_v1"
    assert [w["worker"] for w in record["policies"]] == ["alpha"]
    assert record["lessons"][0]["text"]["text"] == LESSON and record["board"][0]["id"] == "B1"
    assert record["rule_obligations"]["obligations"][0]["status"] == "satisfied"
    use = record["information_use"]
    # alpha told the master about its commit, which later appeared in the delivery repository.
    assert use["totals"]["commit_appeared_after"] == 1 and use["totals"]["messages_direct"] == 1
    assert [row["change"] for row in record["organization_timeline"]["teammates"]] == ["spawned", "first_request"]
    assert record["policies"][0]["versions"][0]["first_shown_t_s"] is not None  # the version reached alpha
    assert record["lessons"][0]["t_s"] is not None and record["board"][0]["t_s"] is not None
    # The sentence-final path "src/A.hs." still matches the changed path once its period is stripped.
    paths = [r for m in use["messages"] for r in m["results"] if "paths" in r]
    assert paths and paths[0]["paths"] == ["src/A.hs"] and paths[0]["touched_after"] is True
    assert FORBIDDEN not in json.dumps(record)


def test_other_protocols_have_no_agent_rules_record(tmp_path):
    trial = tmp_path / "trial"
    (trial / "team" / "relay").mkdir(parents=True)
    (trial / "team" / "relay" / "events.jsonl").write_text(json.dumps({"event": "provider_attempt_started"}) + "\n")
    assert agent_rules_record(trial) is None
    with pytest.raises(ValueError, match="not an agent_rules_v1"):
        agent_rules_generation.build_evidence([trial], tmp_path / "evidence", m0_package())


def load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_agent_rules_m",
                                                  Path(__file__).resolve().parents[1] / "scripts/generate_agent_rules_m.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def script_inputs(tmp_path):
    trial = recorded_trial(tmp_path)
    package = tmp_path / "m0.json"
    package.write_text(json.dumps(m0_package()))
    provider_file = tmp_path / "provider.json"
    provider_file.write_text("{}")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"inference": INFERENCE, "provider_config": str(provider_file)}))
    return ["--package", str(package), "--trial", str(trial), "--profile", str(profile)]


def test_script_without_authorization_only_checks_inputs(tmp_path, capsys, no_runs):
    code = load_script().main(script_inputs(tmp_path) + ["--candidate-id", "M1", "--output", str(tmp_path / "gen")])
    shown = json.loads(capsys.readouterr().out)
    assert code == 0 and shown["mode"] == "check_only" and shown["model_calls"] == 0 and shown["problems"] == []
    assert not (tmp_path / "gen").exists()


def test_script_reports_problems_and_refuses_draft_texts(tmp_path, capsys, no_runs, monkeypatch):
    script = load_script()
    (tmp_path / "gen").mkdir()
    # The guard still holds if the texts ever return to draft.
    monkeypatch.setattr(outer_text, "TEXT_STATUS", "draft_pending_user_approval")
    code = script.main(script_inputs(tmp_path) + ["--candidate-id", "M0", "--output", str(tmp_path / "gen"),
                                                  "--authorize-model-calls"])
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert code == 1 and len(problems) == 3  # same id, existing output, draft texts
    with pytest.raises(ValueError, match="not approved"):
        agent_rules_generation.generate(package=create_m0_package("A", "C"), trials=[], inference=INFERENCE,
                                        provider=provider(), output=tmp_path / "x", candidate_id="M1")


def test_failures_keep_usage_and_mark_the_unreviewed_package(tmp_path, no_runs):
    trial = recorded_trial(tmp_path)
    m0 = m0_package()

    def broken_reviewer(payload):
        return 500, {"error": {"message": "boom"}}
    report = agent_rules_generation.generate(
        package=m0, trials=[trial], inference=INFERENCE, provider=provider(), output=tmp_path / "gen",
        candidate_id="M1", optimizer_completion=optimizer_script(), reviewer_completion=broken_reviewer)
    out = tmp_path / "gen"
    assert report["status"] == "review_failed" and report["usage"]["optimizer"]["provider_attempts"] == 5
    assert "reviewer" in report["usage"] and (out / "package.unreviewed.json").exists()
    assert not (out / "package.json").exists()
    summary = json.loads((out / "training_evidence/run_1/task_summary.json").read_text())
    assert summary["task_id"] == "fixture" and (summary["official_passed_tests"], summary["official_valid_tests"]) == (1, 2)


def test_a_rejected_proposal_keeps_its_usage_and_a_leftover_build_is_named(tmp_path, no_runs, monkeypatch):
    trial = recorded_trial(tmp_path)
    m0 = m0_package()

    def reject(result):
        raise ValueError("rejected")
    monkeypatch.setattr(agent_rules_generation, "_validate_outer_result", reject)
    report = agent_rules_generation.generate(
        package=m0, trials=[trial], inference=INFERENCE, provider=provider(), output=tmp_path / "gen",
        candidate_id="M1", optimizer_completion=optimizer_script(), reviewer_completion=reviewer_script())
    assert report["status"] == "proposal_failed" and report["usage"]["optimizer"]["provider_attempts"] == 5
    (tmp_path / "next.building").mkdir()
    with pytest.raises(ValueError, match="earlier failed build"):
        agent_rules_generation.generate(
            package=m0, trials=[trial], inference=INFERENCE, provider=provider(), output=tmp_path / "next",
            candidate_id="M1", optimizer_completion=optimizer_script(), reviewer_completion=reviewer_script())


def shown_texts(value, path=""):
    """Every harness text marked as shown to the agents, with its JSON path."""
    if isinstance(value, dict):
        if value.get("shown_to_agents") is True and isinstance(value.get("text"), str):
            yield path, value["text"]
        for key, item in value.items():
            yield from shown_texts(item, f"{path}/{key}")


@pytest.mark.parametrize("limit", [10, 16])
def test_every_shown_harness_text_is_what_the_agents_receive(tmp_path, limit):
    from agent_company.programbench.comparison import runner_inference_options
    from agent_company.runtime.qwen import QwenTeamRunner
    inference = {**INFERENCE, **({"teammate_limit": limit} if limit != 10 else {})}
    harness = target_harness(inference, execution_mode="team", policy_protocol="agent_rules_v1")
    files = {"policies": str(tmp_path / "s/agent_policies.md"), "lessons": str(tmp_path / "s/team_lessons.md"),
             "board": str(tmp_path / "s/team_board.md")}
    status = str(tmp_path / "s/team_checkin.json")
    assignment, communication = "ASSIGNMENT-TEXT", "COMMUNICATION-TEXT"
    package = create_m0_package(assignment, communication)
    # The runner the ProgramBench runner builds from this inference (programbench/runner.py),
    # and its own assembly of the master system and the workers' note (M5 review R2-6).
    options = runner_inference_options(inference)
    runner = QwenTeamRunner("qwen", checkin_interval_seconds=options["team_checkin_seconds"],
                            team_message_delivery=options["team_message_delivery"],
                            runtime_disclosure=options["runtime_disclosure"],
                            leader_loop_recovery=options["leader_loop_recovery"],
                            tool_call_cap=options["tool_call_cap"], role_policy_mode=options["role_policy_mode"],
                            policy_package=package, teammate_limit=options.get("teammate_limit", 10))
    master_system, worker_note = runner.system_texts(assignment, communication, is_team=True, status_file=status,
                                                     resources={"cpus": float(inference["cpus"]), "memory_mb": inference["memory_mb"]})
    payloads = []

    def upstream(payload):
        payloads.append(payload)
        return 200, relay_response([])
    relay = TeamRelay(ProviderConfig("fake", "http://localhost", max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      communication, tmp_path / "relay", 5000, upstream, role_policy_mode="propose",
                      policy_package=package, policy_files=files, worker_system_note=worker_note,
                      **({"teammate_limit": limit} if limit != 10 else {}))
    up = [relay_response([relay_call("agent", {"name": "alpha", "description": "w", "prompt": "Work.\n"})])]
    relay.upstream = lambda payload: (payloads.append(payload), (200, up.pop(0) if up else relay_response([])))[1]
    relay.handle(relay_request("master", ("agent", "send_message")))
    relay.handle(relay_request("alpha", ("send_message",)))
    master_seen = master_system + "\n" + json.dumps(payloads[0], ensure_ascii=False)
    worker_seen = json.dumps(payloads[1], ensure_ascii=False)
    replacements = {"<policies file>": files["policies"], "<lessons file>": files["lessons"],
                    "<board file>": files["board"], "<the run's runtime status file>": status,
                    "<assignment_policy>": assignment, "<communication_policy>": communication}
    for path, text in shown_texts(harness):
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        where = worker_seen if "worker" in path else master_seen
        assert json.dumps(text, ensure_ascii=False)[1:-1] in where or text in where, path
    # The resource sentence reaches every worker too, not only the master.
    assert harness["online_policy"]["resources_disclosure"]["text"] in worker_seen
    spawn = next(t for t in payloads[0]["tools"] if t["function"]["name"] == "agent")["function"]["description"]
    assert spawn == harness["roles_and_tools"]["spawn_tool_description"]


def test_an_inference_the_runtime_refuses_has_no_agent_rules_harness():
    with pytest.raises(ValueError, match="role_policy_protocol"):
        target_harness({**INFERENCE, "role_policy_protocol": None}, execution_mode="team", policy_protocol="agent_rules_v1")
    bad = {key: value for key, value in INFERENCE.items() if key != "team_message_delivery"}
    with pytest.raises(ValueError, match="between_tool_calls"):
        target_harness(bad, execution_mode="team", policy_protocol="agent_rules_v1")


def test_rules_mode_systems_drop_roles_and_the_removed_hint_lists():
    from agent_company.evolution import optimizer as O, reviewer as R
    legacy = O.SYSTEM + O.WORKING_MEMORY_GUIDE + O.CITATION_GUIDE + O.EXECUTABILITY_GUIDE
    system = O.agent_rules_system(legacy)
    for gone in ("roles", "Consider useful exchanges", "Investigate failed collaboration", "Let A and C jointly change",
                 "current-attempt evidence entrypoints", "coding agents.", "replay", "legacy_no_replay"):
        assert gone not in system, gone
    for kept in (outer_text.OPTIMIZER_OPENING, outer_text.OPTIMIZER_PARTS, outer_text.OPTIMIZER_SIMPLIFY,
                 outer_text.OPTIMIZER_NO_MANDATE, "Look for receipt and subsequent action",
                 "Repeated commands are not proven waste", outer_text.OPTIMIZER_OBJECT):
        assert kept in system, kept
    assert "Consider useful exchanges" in legacy and "roles" in legacy  # the earlier protocol keeps its text
    review = R.agent_rules_review_system(R.SYSTEM + R.EXECUTABILITY_GUIDE, True)
    assert "roles" not in review and outer_text.REVIEWER_OPENING in review and outer_text.REVIEWER_DETASK in review


def test_each_run_carries_the_master_strategy_it_used_and_must_match_the_incumbent(tmp_path, no_runs):
    # External review #7: runs under different master strategies were indistinguishable in the evidence.
    m0 = m0_package()
    other = {**create_m0_package("# Assignment principles\n\nA9", "# Communication principles\n\nC9"),
             "package_id": "M1", "revision": 1, "parent": {"package_id": m0["package_id"], "revision": 0}}
    for name in "abc":
        (tmp_path / name).mkdir()
    first, second = recorded_trial(tmp_path / "a"), recorded_trial(tmp_path / "b", other)
    report = agent_rules_generation.generate(
        package=m0, trials=[first, second], inference=INFERENCE, provider=provider(), output=tmp_path / "gen",
        candidate_id="M2", optimizer_completion=optimizer_script(), reviewer_completion=reviewer_script())
    assert report["status"] == "reviewed"
    root = tmp_path / "gen/training_evidence"
    shown = [json.loads((root / f"run_{n}/task_summary.json").read_text())["master_strategy"] for n in (1, 2)]
    assert [s["same_as_current"] for s in shown] == [True, False]
    strategy = json.loads((root / "run_2/master_strategy.json").read_text())
    assert (strategy["assignment_policy"], strategy["communication_policy"]) == (other["team"]["assignment"],
                                                                                 other["team"]["communication"])
    assert "run_2/master_strategy.json" in json.loads((root / "run_2/task_summary.json").read_text())["evidence_paths"]
    generation = json.loads((tmp_path / "gen/generation.json").read_text())
    # One hash for the incumbent and for what each run used, whatever file formatting they came from.
    assert generation["trial_strategies"][0]["canonical_sha256"] == generation["incumbent"]["canonical_sha256"]
    assert generation["trial_strategies"][1]["canonical_sha256"] != generation["incumbent"]["canonical_sha256"]
    # A Single trial's request carries the package but no texts: it is not a Team trial.
    single = tmp_path / "single"
    single.mkdir()
    (single / "run_request.json").write_text(json.dumps({"assignment_policy": "", "communication_policy": "",
                                                         "policy_package": m0}))
    with pytest.raises(ValueError, match="not an agent_rules_v1 Team trial"):
        agent_rules_generation.source_strategy(single)
    (single / "run_request.json").write_text('{"policy_package": {}, "policy_package": {}}')
    with pytest.raises(ValueError, match="Duplicate"):
        agent_rules_generation.source_strategy(single)
    # No run used the incumbent: refused before anything is written.
    with pytest.raises(ValueError, match="no recorded run used the incumbent"):
        agent_rules_generation.generate(
            package=m0, trials=[second], inference=INFERENCE, provider=provider(), output=tmp_path / "x",
            candidate_id="M1", optimizer_completion=optimizer_script(), reviewer_completion=reviewer_script())
    # The same id and revision with other texts is an inconsistent package.
    with pytest.raises(ValueError, match="differ from the incumbent of that id"):
        agent_rules_generation.generate(
            package=m0, trials=[first, recorded_trial(tmp_path / "c", {**other, "package_id": m0["package_id"], "revision": 0, "parent": None})],
            inference=INFERENCE, provider=provider(), output=tmp_path / "y", candidate_id="M1",
            optimizer_completion=optimizer_script(), reviewer_completion=reviewer_script())
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_the_free_check_reports_a_trial_without_the_incumbent_strategy(tmp_path, capsys, no_runs):
    args = script_inputs(tmp_path)
    other = create_m0_package("A9", "C9")
    (tmp_path / "m0.json").write_text(json.dumps(other))
    code = load_script().main(args + ["--candidate-id", "M1", "--output", str(tmp_path / "gen")])
    shown = json.loads(capsys.readouterr().out)
    # A package file whose M0 texts differ from the ones the run used.
    assert code == 1 and shown["problems"] == [
        "run 1 used M0 revision 0 with texts that differ from the incumbent of that id and revision",
        "no recorded run used the incumbent master strategy"]
    run_request = Path(args[args.index("--trial") + 1]) / "run_request.json"
    run_request.unlink()
    load_script().main(args + ["--candidate-id", "M1", "--output", str(tmp_path / "gen")])
    assert "no readable master strategy" in json.loads(capsys.readouterr().out)["problems"][0]


def slow_reviewer_script(reading_calls):
    """Reads while reading is allowed, spends the first submit-only call on a read, then submits (10-19 failure)."""
    submit = reviewer_script().turns[-1][0]["function"]["arguments"]
    reads = [[tool("list_training_files", {}, f"s{n}")] for n in range(reading_calls)]
    return Script("reviewer", reads + [[tool("search_training_files", {"query": "late read"}, "late")],
                                       [tool("submit_review", json.loads(submit), "final")]])


def test_the_reviewer_keeps_two_submit_only_calls(tmp_path, no_runs):
    from agent_company.evolution.reviewer import PolicyReviewError, StrategyReviewer
    assert (agent_rules_generation.REVIEWER_ROUNDS, agent_rules_generation.REVIEWER_FINAL_CALLS) == (16, 2)
    reviewer = slow_reviewer_script(14)
    report = agent_rules_generation.generate(
        package=m0_package(), trials=[recorded_trial(tmp_path)], inference=INFERENCE, provider=provider(),
        output=tmp_path / "gen", candidate_id="M1", optimizer_completion=optimizer_script(),
        reviewer_completion=reviewer)
    assert report["status"] == "reviewed"
    # The last two requests offer only submit_review; the ignored late read is refused, not executed.
    offered = [[t["function"]["name"] for t in p["tools"]] for p in reviewer.payloads]
    assert offered[-1] == offered[-2] == ["submit_review"] and len(offered[-3]) > 1
    results = json.loads((tmp_path / "gen/reviewer/batch_0001/tool_results_014.json").read_text())
    assert "use submit_review" in results[0]["result"]["error"]
    # The earlier setting, one submit-only call, loses the review on the same behaviour.
    with pytest.raises(ValueError):
        StrategyReviewer(provider(), tmp_path / "bad", max_tool_rounds=8, final_calls=9)
    old = slow_reviewer_script(7)
    trial_root = tmp_path / "gen/training_evidence"
    request = agent_rules_generation._proposal_request(
        m0_package(), trial_root, [trial_root / "run_1"],
        (json.loads((trial_root / "run_1/task_summary.json").read_text()),),
        target_harness(INFERENCE, execution_mode="team", provider=provider(), policy_protocol="agent_rules_v1"))
    candidate = json.loads((tmp_path / "gen/candidate.json").read_text())
    proposal = agent_rules_generation.PolicyProposal(**{k: candidate[k] for k in (
        "assignment_policy", "communication_policy", "rationale", "execution_notes")})
    with pytest.raises(PolicyReviewError, match="limit reached"):
        StrategyReviewer(provider(), tmp_path / "old", completion=old, max_tool_rounds=8)(
            agent_rules_generation.AgentRulesReviewRequest(request, proposal))


def test_a_failed_review_is_rerun_on_the_same_candidate_without_the_optimizer(tmp_path, no_runs):
    trial = recorded_trial(tmp_path)
    m0 = m0_package()
    first_requests = []

    def broken_reviewer(payload):
        first_requests.append(json.loads(json.dumps(payload)))
        return 500, {"error": {"message": "boom"}}
    report = agent_rules_generation.generate(
        package=m0, trials=[trial], inference=INFERENCE, provider=provider(), output=tmp_path / "gen",
        candidate_id="M1", code_version="fixture", optimizer_completion=optimizer_script(),
        reviewer_completion=broken_reviewer)
    out = tmp_path / "gen"
    assert report["status"] == "review_failed"
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    reviewer = reviewer_script()
    result = agent_rules_generation.review_candidate(package=m0, generation=out, inference=INFERENCE,
                                                     provider=provider(), reviewer_completion=reviewer)
    assert result["status"] == "reviewed" and result["entry"] == "agent_rules_review"
    assert result["run_dispatched"] is False and result["adopted"] is False
    # The reviewer sees the request the failed review saw: same candidate, evidence and harness. Only key order
    # inside the candidate's execution notes may differ, because candidate.json is stored with sorted keys.
    assert (json.loads(reviewer.payloads[0]["messages"][1]["content"])
            == json.loads(first_requests[0]["messages"][1]["content"]))
    # Nothing written by the generation changed; the new review only added files.
    assert all(path.read_bytes() == data for path, data in before.items())
    package = validate_agent_rules_package(json.loads((out / "package.json").read_text()))
    unreviewed = json.loads((out / "package.unreviewed.json").read_text())
    assert package["review_ref"] == "review_2.json" and package["team"] == unreviewed["team"]
    assert json.loads((out / "review_2.json").read_text())["status"] == "completed"
    assert json.loads((out / "report_review_2.json").read_text())["review"]["recommendation"] == "uncertain"
    # A reviewed candidate is not reviewed again, and nothing is overwritten.
    with pytest.raises(ValueError, match="already has a completed review"):
        agent_rules_generation.review_candidate(package=m0, generation=out, inference=INFERENCE,
                                                provider=provider(), reviewer_completion=reviewer_script())
    other = create_m0_package("A9", "C9")
    problems = agent_rules_generation.review_problems(out, other, provider(), label="3")
    assert "--package is not the incumbent this candidate was generated from" in problems
    # A profile whose inference differs from the generation's gives another harness description: refused.
    changed = {**INFERENCE, "agent_timeout_sec": 3600, "team_checkin_seconds": 600}
    assert agent_rules_generation.review_problems(out, m0, provider(), label="3", inference=INFERENCE) == [
        "package.json exists: this candidate already has a completed review"]
    assert ("--profile does not give the harness description the candidate was generated against"
            in agent_rules_generation.review_problems(out, m0, provider(), label="3", inference=changed))
    assert agent_rules_generation.review_problems(out, m0, provider(), label="../x") == [
        "--label must be 1-20 letters, digits, '_' or '-'"]


def test_the_review_script_without_authorization_only_checks(tmp_path, capsys, no_runs):
    from agent_company.runtime.provider import public_provider
    trial = recorded_trial(tmp_path)

    def broken_reviewer(payload):
        return 500, {"error": {"message": "boom"}}
    agent_rules_generation.generate(
        package=m0_package(), trials=[trial], inference=INFERENCE, provider=provider(), output=tmp_path / "gen",
        candidate_id="M1", optimizer_completion=optimizer_script(), reviewer_completion=broken_reviewer)
    package, provider_file, profile = tmp_path / "m0.json", tmp_path / "provider.json", tmp_path / "profile.json"
    package.write_text(json.dumps(m0_package()))
    provider_file.write_text(json.dumps(public_provider(provider())))
    profile.write_text(json.dumps({"inference": INFERENCE, "provider_config": str(provider_file)}))
    import importlib.util
    spec = importlib.util.spec_from_file_location("review_agent_rules_m",
                                                  Path(__file__).resolve().parents[1] / "scripts/review_agent_rules_m.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = ["--generation", str(tmp_path / "gen"), "--package", str(package), "--profile", str(profile)]
    assert script.main(args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["problems"] == [] and shown["model_calls"] == 0 and shown["reviewer_rounds"] == 16
    assert not (tmp_path / "gen/reviewer_2").exists() and not (tmp_path / "gen/review_2.json").exists()
    package.write_text(json.dumps(create_m0_package("A9", "C9")))
    assert script.main(args) == 1
    assert "--package is not the incumbent" in json.loads(capsys.readouterr().out)["problems"][0]


def scripted_review(kinds):
    """'read' reads, 'bad' submits with a harness reference that does not exist (as on 10-19), 'good' submits."""
    good = json.loads(reviewer_script().turns[-1][0]["function"]["arguments"])
    bad = json.loads(json.dumps(good))
    bad["executability"][0]["capability_refs"] = ["/online_policy/task_done"]
    turns = []
    for n, kind in enumerate(kinds):
        if kind == "read":
            turns.append([tool("list_training_files", {}, f"k{n}")])
        else:
            turns.append([tool("submit_review", good if kind == "good" else bad, f"k{n}")])
    return Script("reviewer", turns)


def review_once(tmp_path, reviewer, **options):
    from agent_company.evolution.reviewer import StrategyReviewer
    tmp_path.mkdir(parents=True)
    trial = recorded_trial(tmp_path)
    agent_rules_generation.generate(
        package=m0_package(), trials=[trial], inference=INFERENCE, provider=provider(), output=tmp_path / "gen",
        candidate_id="M1", optimizer_completion=optimizer_script(), reviewer_completion=scripted_review(["good"]))
    root = tmp_path / "gen/training_evidence"
    request = agent_rules_generation._proposal_request(
        m0_package(), root, [root / "run_1"], (json.loads((root / "run_1/task_summary.json").read_text()),),
        target_harness(INFERENCE, execution_mode="team", provider=provider(), policy_protocol="agent_rules_v1"))
    candidate = json.loads((tmp_path / "gen/candidate.json").read_text())
    proposal = agent_rules_generation.PolicyProposal(**{k: candidate[k] for k in (
        "assignment_policy", "communication_policy", "rationale", "execution_notes")})
    return StrategyReviewer(provider(), tmp_path / "review", completion=reviewer, **options)(
        agent_rules_generation.AgentRulesReviewRequest(request, proposal))


def test_a_refused_final_submission_gets_correction_calls(tmp_path, no_runs):
    from agent_company.evolution.reviewer import PolicyReviewError
    assert agent_rules_generation.REVIEWER_REPAIR_CALLS == 2
    # The 10-19 second review: reads, wastes the first submit-only call, submits with a bad reference.
    pattern = ["read"] * 14 + ["read", "bad", "good"]
    reviewer = scripted_review(pattern)
    review = review_once(tmp_path / "a", reviewer, max_tool_rounds=16, final_calls=2, repair_calls=2)
    assert review.recommendation == "uncertain" and len(reviewer.payloads) == 17
    offered = [[t["function"]["name"] for t in p["tools"]] for p in reviewer.payloads]
    assert offered[-1] == offered[-2] == offered[-3] == ["submit_review"]
    batch = tmp_path / "a/review/batch_0001"
    assert "not a target_harness field" in json.loads((batch / "tool_results_015.json").read_text())[0]["result"]["error"]
    assert json.loads((batch / "submission_repairs.json").read_text())["granted"] == 1
    # The same behaviour without correction calls fails, as it did on 10-19.
    with pytest.raises(PolicyReviewError, match="limit reached"):
        review_once(tmp_path / "b", scripted_review(pattern), max_tool_rounds=16, final_calls=2)
    # Corrections are capped: two submit-only calls plus two corrections, so four refusals end the review.
    with pytest.raises(PolicyReviewError, match="limit reached"):
        review_once(tmp_path / "c", scripted_review(["read"] * 14 + ["bad"] * 4 + ["good"]),
                    max_tool_rounds=16, final_calls=2, repair_calls=2)
    assert json.loads((tmp_path / "c/review/batch_0001/submission_repairs.json").read_text())["granted"] == 2
    # A refused submission while reading is still open earns nothing extra: the review still has its calls.
    early = scripted_review(["read", "bad"] + ["read"] * 12 + ["good"])
    review_once(tmp_path / "d", early, max_tool_rounds=16, final_calls=2, repair_calls=2)
    assert len(early.payloads) == 15 and not (tmp_path / "d/review/batch_0001/submission_repairs.json").exists()



def test_correction_calls_count_unparsable_submissions_and_not_a_corrected_response(tmp_path, no_runs):
    good = json.loads(reviewer_script().turns[-1][0]["function"]["arguments"])
    bad = json.loads(json.dumps(good))
    bad["executability"][0]["capability_refs"] = ["/online_policy/task_done"]
    broken = {"id": "x", "type": "function", "function": {"name": "submit_review", "arguments": "{not json"}}
    reads = [[tool("list_training_files", {}, f"m{n}")] for n in range(14)]
    # Unparsable arguments in a submit-only call are a refused submission: each earns a correction call.
    garbled = Script("reviewer", reads + [[broken], [broken], [tool("submit_review", good, "ok")]])
    review_once(tmp_path / "a", garbled, max_tool_rounds=16, final_calls=2, repair_calls=2)
    assert json.loads((tmp_path / "a/review/batch_0001/submission_repairs.json").read_text())["granted"] == 2
    assert len(garbled.payloads) == 17
    # A refused and a valid submission in one response: the valid one is the review, nothing is granted.
    both = Script("reviewer", reads + [[tool("submit_review", bad, "b1"), tool("submit_review", good, "b2")]])
    review_once(tmp_path / "b", both, max_tool_rounds=16, final_calls=2, repair_calls=2)
    assert len(both.payloads) == 15 and not (tmp_path / "b/review/batch_0001/submission_repairs.json").exists()
