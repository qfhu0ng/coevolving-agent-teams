"""Runtime disclosure and master status-read loop recovery without a model (2026-10-02).

The real-CLI scenarios are in test_leader_loop_recovery_integration.py.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from agent_company.analysis import evidence_projection as projection
from agent_company.programbench.runner import ProgramBenchRunner
from agent_company.runtime.qwen import (LOOP_GUARD_RULES, QwenTeamRunner, checkin_refresh_seconds, master_system,
                                       runtime_rules, single_system)
from agent_company.runtime.relay import ProviderConfig

ROOT = Path(__file__).parents[1]
COMPAT = ROOT / "agent_company/runtime/qwen_native_compat.mjs"
PATCH = ROOT / "agent_company/runtime/qwen_native_patch.mjs"
CHUNKS = ROOT / "external/qwen-study/package/chunks"


def node():
    found = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not found:
        pytest.skip("Node with module.registerHooks is required")
    return found


def lifecycle(stderr):
    prefix = "Agent Company lifecycle: "
    return [json.loads(line[len(prefix):]) for line in stderr.splitlines() if line.startswith(prefix)]


def anchors():
    script = f"const m = await import({json.dumps(PATCH.as_uri())}); console.log(JSON.stringify(m.LEADER_LOOP_ANCHORS));"
    return json.loads(subprocess.run([node(), "--input-type=module", "-e", script], check=True,
                                     capture_output=True, text=True).stdout)


# ----- native source facts -----------------------------------------------------

def native_source():
    if not CHUNKS.is_dir():
        pytest.skip("Installed Qwen package not present")
    # Decode like the compat layer's readFileSync(..., 'utf8'): a copied package can
    # carry non-UTF-8 files such as macOS "._*.js" metadata, which Node never loads.
    return {path.name: path.read_text(encoding="utf-8", errors="replace") for path in CHUNKS.glob("*.js")}


def test_disclosed_numbers_match_the_installed_guard():
    source = native_source()
    core = next(text for text in source.values() if "var TOOL_CALL_LOOP_THRESHOLD" in text)
    for constant in ("TOOL_CALL_LOOP_THRESHOLD = 5;", "GLOBAL_DUPLICATE_THRESHOLD = 6;",
                     "DEFAULT_MAX_TOOL_CALLS_PER_TURN = 100;", "ADAPTIVE_CAP_HARD_MULTIPLIER = 10;",
                     "var STAGNATION_THRESHOLD = 8;", "SHELL_COMMAND_STAGNATION_THRESHOLD = STAGNATION_THRESHOLD;"):
        assert constant in core, constant
    assert re.search(r"STATEFUL_READ_TOOLS = /\* @__PURE__ \*/ new Set\(\[\s*\"task_list\"\s*\]\)", core) \
        or 'STATEFUL_READ_TOOLS = new Set(["task_list"])' in core
    assert re.search(r"\(status\|diff\|ls-files\)", core)
    assert any("skipLoopDetection: settings.model?.skipLoopDetection ?? true" in text for text in source.values())
    rules = " ".join(projection.RUNTIME_RULES["loop_guards"])
    for text in (LOOP_GUARD_RULES, rules):
        assert "fifth consecutive call" in text and "more than 100 tool calls" in text
        assert "at least 6 times" in text and "more than 1000 tool calls" in text and "task_list" in text
    assert "Eight consecutive" in runtime_rules("worker") and "Eight consecutive" in rules


def test_every_leader_loop_anchor_is_unique_in_the_installed_cli():
    source = native_source()
    found = {part: [name for name, text in source.items() if text.count(anchor) == 1]
             for part, anchor in anchors().items()}
    assert all(len(names) == 1 for names in found.values()), found
    assert found["response"] == found["detect"]
    assert found["exit"] == found["resume"] == found["settle"]


def patch(source, **options):
    script = (f"const m = await import({json.dumps(PATCH.as_uri())});"
              "let s = ''; for await (const c of process.stdin) s += c;"
              f"process.stdout.write(m.patchNativeSource(s, {json.dumps(options)}));")
    return subprocess.run([node(), "--input-type=module", "-e", script], input=source,
                          capture_output=True, text=True)


def test_patch_is_opt_in_exact_and_fails_closed_on_drift(tmp_path):
    source = native_source()
    by_part = {part: next(name for name, text in source.items() if anchor in text) for part, anchor in anchors().items()}
    core, runner = source[by_part["detect"]], source[by_part["exit"]]
    for text in (core, runner):
        off = patch(text)
        assert off.returncode == 0 and "leaderLoop" not in off.stdout and "LeaderLoop" not in off.stdout
    patched_core = patch(core, leaderLoopRecovery=True).stdout
    patched_runner = patch(runner, leaderLoopRecovery=True).stdout
    assert patched_core.count(".leaderLoopRecoverable(") == 1 and patched_core.count("agentCompanyResponseCalls") == 3
    assert patched_runner.count(".recoverLeaderLoop(") == 1 and patched_runner.count(".leaderResumeBatch(") == 1
    assert patched_runner.count(".leaderRecoveryUnresolved()") == 1
    for name, text in (("core.mjs", patched_core), ("runner.mjs", patched_runner)):
        (tmp_path / name).write_text(text)
        subprocess.run([node(), "--check", str(tmp_path / name)], check=True)
    drifted = patch(runner.replace(anchors()["resume"], anchors()["resume"].replace("splice(0)", "splice(0, 1)")),
                    leaderLoopRecovery=True)
    assert drifted.returncode != 0 and "anchor changed" in drifted.stderr


# ----- compat switches and bridge (fixture bundle; no CLI) ---------------------

@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    chunks = root / "chunks"
    chunks.mkdir(parents=True)
    (root / "package.json").write_text('{"type":"module"}')
    (root / "cli.js").write_text("")
    (chunks / "identity.js").write_text("""
import { AsyncLocalStorage } from 'node:async_hooks';
var teammateIdentityStore = new AsyncLocalStorage();
export function getTeammateContext() { return teammateIdentityStore.getStore(); }
export function runAs(context, fn) { return teammateIdentityStore.run(context, fn); }
""")
    (chunks / "manager.js").write_text("""
export var TeamManager = class {
  async drainLeaderInbox() {}
  async cleanup() {}
  async spawnTeammate(config3) {
    const name = config3.name;
    this.teamEventEmitter.emit("teammate_joined" /* TEAMMATE_JOINED */, {});
  }
  async executeTeammate(name, signal, updateOutput) {
    const teamManager = this;
      await teamManager.spawnTeammate({
        name,
      });
    return {};
  }
  async pollLeaderInbox() {
    const callback = this.leaderMessageCallback, newMessages = [];
    callback(
      this.formatLeaderEnvelope(newMessages).join("\\n\\n"),
    );
  }
};
export class Config { getAgentsSettings() { return {team: {maxTeammates: 10}}; } }
export async function listTasks(teamName, filters) { return []; }
// Native in-process backend anchor (teammate start hook, 2026-10-18).
export var InProcessBackend = class {
  async spawnAgent(config3) {
    const interactive = {};
    this.agents.set(config3.agentId, interactive);
    this.agentOrder.push(config3.agentId);
  }
};
""")
    (chunks / "queue.js").write_text("export function q(pendingTeammateMessages, formatted) { "
                                     "pendingTeammateMessages.push(formatted); }\n")
    # Counted as text only; never imported, so these anchors are not executed.
    (chunks / "leader.js").write_text("/*\n" + "\n".join(anchors().values()) + "\n*/\nexport {};\n")
    return root


def run_compat(bundle, script, **env):
    base = {"AGENT_COMPANY_QWEN_CLI": str(bundle / "cli.js"), "AGENT_COMPANY_ROOT_OWNER": "master",
            "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:12345/v1", "AGENT_COMPANY_CHECKIN_SECONDS": "1800",
            "AGENT_COMPANY_TASK_STARTED_EPOCH_MS": "1", "AGENT_COMPANY_TASK_DEADLINE_EPOCH_MS": "9999999999999",
            "AGENT_COMPANY_DELIVERY_REPO": str(bundle), "AGENT_COMPANY_STATUS_FILE": str(bundle / "status.json")}
    base.update(env)
    environment = {key: value for key, value in {**os.environ, **base}.items() if value is not None}
    return subprocess.run([node(), "--import", str(COMPAT), "--input-type=module", "-e", script,
                           (bundle / "chunks/identity.js").as_uri()],
                          env=environment, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize("env, message", [
    ({"AGENT_COMPANY_LEADER_LOOP_RECOVERY": "yes"}, "must be 0 or 1"),
    ({"AGENT_COMPANY_RUNTIME_DISCLOSURE": "true"}, "must be 0 or 1"),
    ({"AGENT_COMPANY_LEADER_LOOP_RECOVERY": "1", "AGENT_COMPANY_ROOT_OWNER": "single"}, "Team runs with runtime check-ins"),
    ({"AGENT_COMPANY_LEADER_LOOP_RECOVERY": "1", "AGENT_COMPANY_CHECKIN_SECONDS": None}, "Team runs with runtime check-ins"),
])
def test_invalid_switches_fail_before_the_cli(bundle, env, message):
    result = run_compat(bundle, "console.log('CLI_EXECUTED');", **env)
    assert result.returncode != 0 and message in result.stderr and "CLI_EXECUTED" not in result.stdout


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_changed_native_anchor_fails_before_the_cli(bundle, change):
    leader = bundle / "chunks/leader.js"
    if change == "missing":
        leader.write_text(leader.read_text().replace(anchors()["settle"], ""))
    else:
        (bundle / "chunks/copy.js").write_text("/*\n" + anchors()["detect"] + "\n*/\n")
    result = run_compat(bundle, "console.log('CLI_EXECUTED');", AGENT_COMPANY_LEADER_LOOP_RECOVERY="1")
    assert result.returncode != 0 and "Leader loop recovery requires" in result.stderr
    assert "CLI_EXECUTED" not in result.stdout


BRIDGE_SCRIPT = r'''
import assert from 'node:assert/strict';
import path from 'node:path';
const {runAs} = await import(process.argv[1]);
await import(new URL('./queue.js', process.argv[1]));  // The check-in queue patch is checked first.
const bridge = globalThis[Symbol.for('agent-company.native')];
const status = process.env.AGENT_COMPANY_STATUS_FILE;
const loop = 'consecutive_identical_tool_calls';
const read = (file, name = 'read_file') => ({type: 'tool_call_request', value: {name, args: {file_path: file, limit: 18}}});
assert.equal(bridge.leaderLoopRecoverable(read(status), loop, 1), true);
const dotted = path.join(path.dirname(status), '.') + '/./' + path.basename(status);
assert.equal(bridge.leaderLoopRecoverable(read(dotted), loop, 1), true);
assert.equal(bridge.leaderLoopRecoverable(read(path.basename(status)), loop, 1), false);   // relative
assert.equal(bridge.leaderLoopRecoverable(read(status + '.copy'), loop, 1), false);        // another JSON file
assert.equal(bridge.leaderLoopRecoverable(read(status, 'grep_search'), loop, 1), false);
assert.equal(bridge.leaderLoopRecoverable(read(status), 'turn_tool_call_cap', 1), false);
assert.equal(bridge.leaderLoopRecoverable(read(status), loop, 2), false);
assert.equal(runAs({agentName: 'alpha'}, () => bridge.leaderLoopRecoverable(read(status), loop, 1)), false);

const team = statuses => ({teamFile: {members: statuses.map((_, i) => ({agentId: 'a' + i}))},
  getAgentFromBackend: id => ({getStatus: () => statuses[Number(id.slice(1))]})});
const client = {};
assert.equal(bridge.recoverLeaderLoop(client, 0, team(['running'])), false, 'only a flagged halt');
for (const [queued, members] of [[1, team(['running'])], [0, team(['idle', 'completed'])], [0, undefined]]) {
  client.agentCompanyRecoverableLoop = true;
  assert.equal(bridge.recoverLeaderLoop(client, queued, members), false);
  assert.equal(client.agentCompanyRecoverableLoop, false, 'the flag is consumed');
}
assert.equal(bridge.leaderRecoveryUnresolved(), false);
assert.deepEqual(bridge.leaderResumeBatch(['plain']), ['plain'], 'no notice without a recovery');
client.agentCompanyRecoverableLoop = true;
assert.equal(bridge.recoverLeaderLoop(client, 0, team(['running', 'idle'])), true);
const checkin = '<teammate_message from="runtime:checkin">\ncheckin-1: observations\n</teammate_message>';
const batch = bridge.leaderResumeBatch([checkin]);
assert.equal(batch.length, 2);
assert.match(batch[0], /^Runtime notice \[leader-loop-4\]: .*not executed/);
assert.equal(batch[1], checkin);
assert.deepEqual(bridge.leaderResumeBatch(['next']), ['next'], 'one notice per recovery');
assert.equal(bridge.leaderRecoveryUnresolved(), false);
client.agentCompanyRecoverableLoop = true;
assert.equal(bridge.recoverLeaderLoop(client, 0, team(['running'])), true);
assert.equal(bridge.leaderRecoveryUnresolved(), true);
assert.equal(bridge.leaderRecoveryUnresolved(), false);
assert.throws(() => bridge.headers(process.env.AGENT_COMPANY_RELAY_URL, {messages: []}),
              /Leader loop recovery patches were not applied/);
console.log('BRIDGE_OK');
'''


def test_bridge_accepts_only_the_scope_a_case_and_resumes_once(bundle):
    result = run_compat(bundle, BRIDGE_SCRIPT, AGENT_COMPANY_LEADER_LOOP_RECOVERY="1")
    assert result.returncode == 0 and "BRIDGE_OK" in result.stdout, result.stderr[-3000:]
    events = lifecycle(result.stderr)
    guards = [event for event in events if event["event"] == "leader_loop_guard"]
    assert [event.get("reason") for event in guards] == [
        None, None, "not_status_file_read", "not_status_file_read", "not_status_file_read", "other_loop_type",
        "other_calls_in_response", "not_master"]
    assert guards[-1]["owner"] == "alpha" and all("file_path" not in json.dumps(event) for event in guards)
    refused = [event["reason"] for event in events if event["event"] == "leader_loop_not_recovered"]
    assert refused == ["calls_already_queued", "no_running_teammate", "no_team"]
    assert [event["recovery_id"] for event in events if event["event"] == "leader_loop_recovered"] == [
        "leader-loop-4", "leader-loop-5"]
    resumed = [event for event in events if event["event"] == "leader_loop_resumed"]
    assert len(resumed) == 1 and resumed[0]["checkin"] is True and resumed[0]["messages"] == 1
    assert [event["recovery_id"] for event in events if event["event"] == "leader_loop_unresolved"] == ["leader-loop-5"]
    loaded = next(event for event in events if event["event"] == "compatibility_loaded")
    assert loaded["leader_loop_recovery"] is True and loaded["runtime_disclosure"] is False


@pytest.mark.parametrize("disclosure, recovery", [(False, False), (True, False), (True, True)])
def test_status_file_note_only_with_disclosure(bundle, disclosure, recovery):
    result = run_compat(bundle, "console.log('LOADED');",
                        AGENT_COMPANY_RUNTIME_DISCLOSURE="1" if disclosure else None,
                        AGENT_COMPANY_LEADER_LOOP_RECOVERY="1" if recovery else None)
    assert result.returncode == 0, result.stderr[-2000:]
    view = json.loads((bundle / "status.json").read_text())
    if not disclosure:
        assert "runtime_note" not in view
        return
    assert next(iter(view)) == "runtime_note"
    note = view["runtime_note"]
    assert "refreshed about every 60 s" in note and "about every 30 minutes" in note
    assert "fifth consecutive identical tool call" in note and "ending your turn ends the run" in note
    # Review 2026-10-02: the exception must state every condition the gate checks.
    assert ("is the only tool call in its response and a teammate is running, only your turn ends" in note) is recovery


# ----- prompts, runner settings and evidence -----------------------------------

def test_prompts_are_unchanged_unless_disclosure_is_enabled():
    checkin = "Runtime team check-ins: fixture."
    default = master_system("A", "C", checkin, "native_idle")
    assert default == master_system("A", "C", checkin, "native_idle", rules=None)
    assert "Runtime rules" not in default and "Runtime rules" not in single_system()
    assert single_system() == single_system(None)
    rules = runtime_rules("master", checkin_interval_seconds=1800)
    disclosed = master_system("A", "C", checkin, "native_idle", rules=rules)
    sections = disclosed.split("\n\n")
    assert sections.count(rules) == 1
    assert [section for section in sections if section != rules] == default.split("\n\n")
    assert single_system(runtime_rules("single")).startswith(single_system())


def test_role_rules_state_each_role_s_real_consequence():
    master = runtime_rules("master", checkin_interval_seconds=1800)
    assert "stops every teammate" in master and "Exception in this run" not in master
    assert "end your turn without calling tools" in master and "about every 30 minutes" in master
    assert "delivered with your next tool results" in master and "ending your turn ends the run" in master
    assert "refreshed about every 60 seconds" in master and checkin_refresh_seconds(1800) == 60
    exception = runtime_rules("master", checkin_interval_seconds=1800, leader_loop_recovery=True)
    assert "Exception in this run" in exception and "it is the only tool call in that response" in exception
    assert "including a response with any other tool call, still ends the run" in exception
    assert "check-in" not in runtime_rules("master").split("\n")[1]  # No check-in schedule without check-ins.
    worker, single = runtime_rules("worker"), runtime_rules("single")
    # Review 2026-10-02: a queued message can start a turn that ends normally, which clears the error.
    assert "a turn that then ends normally clears the error; otherwise you are marked failed" in worker
    assert "however long that takes" in worker and "do not restart the counts" in worker
    assert "stops every teammate" not in worker
    assert "The whole task normally runs as one turn" in single and "teammate" not in single
    assert "git status" not in master  # The master has no shell.
    with pytest.raises(ValueError):
        runtime_rules("reviewer")


def test_runner_switches_are_validated():
    with pytest.raises(ValueError, match="requires runtime check-ins"):
        QwenTeamRunner("cli.js", leader_loop_recovery=True)
    with pytest.raises(ValueError, match="boolean"):
        QwenTeamRunner("cli.js", runtime_disclosure="yes")
    QwenTeamRunner("cli.js", checkin_interval_seconds=1800, runtime_disclosure=True, leader_loop_recovery=True)
    provider = ProviderConfig("model", "http://localhost")
    with pytest.raises(ValueError, match="requires team_checkin_seconds"):
        ProgramBenchRunner(provider, Path("q"), Path("n"), leader_loop_recovery=True)
    with pytest.raises(ValueError, match="boolean"):
        ProgramBenchRunner(provider, Path("q"), Path("n"), runtime_disclosure=1)


def test_projection_reports_settings_as_recorded_never_assumed():
    rules = projection.runtime_rules_projection
    assert rules(None)["effective"] == {"runtime_disclosure": "unknown", "leader_loop_recovery": "unknown",
                                        "tool_call_cap": "unknown", "teammate_limit": "unknown"}
    old = [{"event": "compatibility_loaded", "root_owner": "master"}]
    assert rules(old)["effective"]["leader_loop_recovery"] == "unknown" and rules(old)["leader_loop"] is None
    off = [{"event": "compatibility_loaded", "runtime_disclosure": True, "leader_loop_recovery": False}] * 2
    assert rules(off)["effective"] == {"runtime_disclosure": True, "leader_loop_recovery": False,
                                       "tool_call_cap": "unknown", "teammate_limit": "unknown"}  # Unrecorded settings stay unknown.
    capped = [{"event": "compatibility_loaded", "tool_call_cap": "disabled"}] * 2
    assert rules(capped)["effective"]["tool_call_cap"] == "disabled"
    assert rules(capped[:1] + [{"event": "compatibility_loaded", "tool_call_cap": "native_default"}])[
        "effective"]["tool_call_cap"] == "conflicting"
    assert rules(off)["leader_loop"] is None
    mixed = off[:1] + [{"event": "compatibility_loaded", "runtime_disclosure": False, "leader_loop_recovery": False}]
    assert rules(mixed)["effective"]["runtime_disclosure"] == "conflicting"
    on = [{"event": "compatibility_loaded", "runtime_disclosure": False, "leader_loop_recovery": True},
          {"event": "leader_loop_guard", "loop_type": "consecutive_identical_tool_calls", "recoverable": True},
          {"event": "leader_loop_guard", "loop_type": "turn_tool_call_cap", "recoverable": False},
          {"event": "leader_loop_recovered", "recovery_id": "leader-loop-1"},
          {"event": "leader_loop_resumed", "recovery_id": "leader-loop-1", "waited_s": 40, "checkin": True},
          {"event": "leader_loop_not_recovered", "reason": "no_running_teammate"},
          {"event": "leader_loop_unresolved", "recovery_id": "leader-loop-2"}]
    leader = rules(on)["leader_loop"]
    assert leader["guard_stops"] == {"consecutive_identical_tool_calls": 1, "turn_tool_call_cap": 1}
    assert (leader["recovered"], leader["resumed"], leader["resumed_by_checkin"], leader["unresolved"]) == (1, 1, 1, 1)
    assert leader["not_recovered"] == {"no_running_teammate": 1} and leader["resume_wait_s"]["max"] == 40
    assert rules(on)["scope"].startswith("Runtime mechanics only")


def test_disclosure_follows_the_tool_call_cap_mode():
    # 2026-10-02: no per-turn cap for long-horizon runs; the consecutive guard stays.
    for role in ("master", "worker", "single"):
        default, disabled = runtime_rules(role), runtime_rules(role, tool_call_cap="disabled")
        assert "more than 100 tool calls" in default and "more than 1000 tool calls" in default
        assert "100" not in disabled and "1000" not in disabled
        assert "There is no limit on the number of tool calls in a turn." in disabled
        assert "the fifth consecutive call of one tool with identical arguments" in disabled
    assert "accumulate across the task" not in runtime_rules("single", tool_call_cap="disabled")
    assert "Eight consecutive" in runtime_rules("worker", tool_call_cap="disabled")
    with pytest.raises(ValueError):
        runtime_rules("single", tool_call_cap="1000")
    with pytest.raises(ValueError, match="tool_call_cap"):
        QwenTeamRunner("cli.js", tool_call_cap=0)
    with pytest.raises(ValueError, match="tool_call_cap"):
        ProgramBenchRunner(ProviderConfig("model", "http://localhost"), Path("q"), Path("n"), tool_call_cap=None)
