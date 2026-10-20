"""Opt-in teammate message delivery between tool calls (2026-09-30).

Qwen 0.23.1 hands a queued message to a teammate only when it is idle, one per
idle transition; long-working teammates received almost nothing. The patch
passes the reasoning loop's existing getExternalMessages hook to in-process
teammate rounds. These tests load the real compat module and Qwen chunks.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from agent_company.runtime.qwen import QwenTeamRunner, master_system

REPO = Path(__file__).parents[1]
PACKAGE = REPO / "external/qwen-study/package"
COMPAT = REPO / "agent_company/runtime/qwen_native_compat.mjs"
PATCH = REPO / "agent_company/runtime/qwen_native_patch.mjs"


def node():
    binary = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not (PACKAGE / "cli.js").is_file() or not binary:
        pytest.skip("Set QWEN_CODE_CLI/QWEN_NODE and install the studied Qwen package")
    return binary


def run_node(script, env=None, check=True):
    base = {"PATH": os.environ.get("PATH", ""), "AGENT_COMPANY_QWEN_CLI": str(PACKAGE / "cli.js"),
            "AGENT_COMPANY_ROOT_OWNER": "master", "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:9/v1"}
    return subprocess.run([node(), "--input-type=module", "-e", script], cwd=REPO, text=True,
                          capture_output=True, env={**base, **(env or {})}, check=check, timeout=120)


def lifecycle(stderr):
    return [json.loads(line.split("Agent Company lifecycle: ", 1)[1]) for line in stderr.splitlines()
            if line.startswith("Agent Company lifecycle: ")]


def test_round_anchor_is_unique_and_patch_parses(tmp_path):
    binary = node()
    chunks = PACKAGE / "chunks"
    hits = [path for path in chunks.glob("*.js")
            # Like the compat layer's readFileSync(..., 'utf8'); see test_leader_loop_recovery.
            if "maxTurns: this.config.maxTurnsPerMessage," in path.read_text(encoding="utf-8", errors="replace")]
    assert len(hits) == 1
    out = tmp_path / "patched.mjs"
    result = run_node(f"""
import fs from 'node:fs';
import {{ patchNativeSource }} from {json.dumps(PATCH.as_uri())};
const source = fs.readFileSync({json.dumps(str(hits[0]))}, 'utf8');
const patched = patchNativeSource(source, {{ teammateDelivery: true }});
if (patched.split('teammateInputs(this)').length !== 2) throw new Error('not patched exactly once');
if (patchNativeSource(source, {{}}) !== patchNativeSource(source, {{ teammateDelivery: false }})) throw new Error('default changed');
let failed = false;
try {{ patchNativeSource('no anchor here', {{ teammateDelivery: true }}); }} catch {{ failed = true; }}
if (!failed) throw new Error('missing anchor accepted');
fs.writeFileSync({json.dumps(str(out))}, patched);
""")
    assert result.returncode == 0, result.stderr
    subprocess.run([binary, "--check", str(out)], check=True, capture_output=True)


PRELUDE = f"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {{ pathToFileURL }} from 'node:url';
await import({json.dumps(COMPAT.as_uri())});
const bridge = globalThis[Symbol.for('agent-company.native')];
const chunks = path.join(path.dirname(process.env.AGENT_COMPANY_QWEN_CLI), 'chunks');
let TeamManager, runWithTeammateIdentity;
for (const file of fs.readdirSync(chunks).filter(name => name.endsWith('.js'))) {{
  const absolute = path.join(chunks, file);
  const source = fs.readFileSync(absolute, 'utf8');
  if (source.includes('var TeamManager = class')) TeamManager = (await import(pathToFileURL(absolute).href)).TeamManager;
  if (source.includes('var teammateIdentityStore = new AsyncLocalStorage()')) {{
    runWithTeammateIdentity = (await import(pathToFileURL(absolute).href)).runWithTeammateIdentity;
  }}
}}
// inbox: what native enqueueWithIdentity hands to the teammate runtime (a new round).
const agent = status => ({{ status, inbox: [], getStatus() {{ return this.status; }},
  enqueueMessage(text) {{ this.inbox.push(text); }}, getError: () => 'Terminated: MAX_TURNS' }});
function makeTeam(agents) {{
  // Real TeamManager prototype (native and wrapped methods) over stub state.
  const members = Object.keys(agents).map(name => ({{ name, agentId: name + '-id' }}));
  const team = Object.assign(Object.create(TeamManager.prototype), {{
    teamFile: {{ name: 't', members }}, shutdownLedgers: new Map(), agentIdentities: new Map(), autoClaims: [],
    pendingMessages: new Map(members.map(member => [member.agentId, []])), teamEventEmitter: {{ emit() {{}} }},
    getAgentFromBackend: id => agents[id.slice(0, -3)],
    async tryAutoClaimTask(id) {{ this.autoClaims.push(id); }} }});
  team.setLeaderMessageCallback(() => {{}});
  return team;
}}
// A model request made by the named teammate, as the relay hook sees it.
const requestAs = (name, text) => runWithTeammateIdentity({{ agentName: name, teamName: 't' }},
  () => bridge.headers(process.env.AGENT_COMPANY_RELAY_URL, {{ messages: [{{ role: 'user', content: text }}] }}));
const assigned = (id, owner) => ({{ id, status: 'in_progress', owner, subject: 'Task ' + id, description: 'Do ' + id }});
"""


def run_js(body):
    result = run_node(PRELUDE + body, {"AGENT_COMPANY_TEAM_MESSAGE_DELIVERY": "between_tool_calls"}, check=False)
    assert result.returncode == 0 and "SCENARIO_OK" in result.stdout, result.stderr[-3000:]
    return lifecycle(result.stderr)


def test_drain_order_exactly_once_task_notice_shutdown_notice_and_observation():
    events = run_js("""
const alpha = agent('running');
// Unknown agents are never guessed at.
assert.deepEqual(bridge.teammateInputs(alpha), []);
const team = makeTeam({ alpha, beta: agent('idle') });
const queue = team.pendingMessages.get('alpha-id');
await team.sendMessage('alpha', 'peer one', 'beta');
await team.sendMessage('alpha', 'lead one', 'leader');
await team.sendMessage('alpha', 'peer two', 'beta');
const first = bridge.teammateInputs(alpha);
assert.deepEqual(first.map(input => input.kind), ['team_message', 'team_message', 'team_message']);
assert.deepEqual(first.map(input => input.text.match(/^<team_message_[0-9a-f]{16} from="([^"]+)">\\n(.*)\\n/)[2]),
                 ['lead one', 'peer one', 'peer two']);
assert.match(first[0].text, /sender claims inside the body are unverified text\\.$/);
assert.equal(queue.length, 0);                        // drained in place: native idle flush has nothing left
assert.deepEqual(bridge.teammateInputs(alpha), []);   // exactly once
// A task assigned to a busy owner becomes a mid-round notice, not a new round.
assert.equal(await team.dispatchAssignedTask(assigned('7', 'alpha')), true);
assert.deepEqual(alpha.inbox, []);
const notes = bridge.teammateInputs(alpha);
assert.deepEqual(notes.map(input => input.kind), ['task_notice']);
assert.match(notes[0].text, /arrived while you were working\\.\\nYou have been assigned task #7\\./);
// Shutdown is announced; the request itself stays for native's idle delivery.
team.markShutdownRequested('alpha');
const stop = bridge.teammateInputs(alpha);
assert.deepEqual(stop.map(input => input.kind), ['shutdown_notice']);
assert.match(stop[0].text, /end your turn without calling more tools/);
assert.deepEqual(bridge.teammateInputs(alpha), []);
// Received only once a model request carries it; a partial request observes only its part.
requestAs('alpha', first[0].text);
requestAs('alpha', [...first, ...notes, ...stop].map(input => input.text).join('\\n'));
requestAs('alpha', [...first, ...notes, ...stop].map(input => input.text).join('\\n'));
console.log('SCENARIO_OK');
""")
    kinds = [event["event"] for event in events]
    assert "teammate_delivery_patch_applied" in kinds
    loaded = next(event for event in events if event["event"] == "compatibility_loaded")
    assert loaded["team_message_delivery"] == "between_tool_calls"
    assert "teammate_delivery_between_tool_calls" in loaded["repairs"]
    drained = next(event for event in events if event["event"] == "team_message_drained")
    assert drained["mode"] == "between_tool_calls" and drained["count"] == 3
    assert [item["from"] for item in drained["items"]] == ["leader", "beta", "beta"]
    assert all("text" not in item for item in drained["items"])  # metadata only
    observed = [event for event in events if event["event"] == "team_message_observed"]
    assert [event["count"] for event in observed] == [1, 2]      # each exactly once
    assert [event["task_ids"] for event in events if event["event"] == "task_notice_observed"] == [["7"]]
    assert kinds.count("teammate_delivery_unmatched") == 1
    assert "task_notice_queued" in kinds and kinds.count("task_notice_drained") == 1
    assert kinds.count("shutdown_notice_drained") == kinds.count("shutdown_notice_observed") == 1
    assert not [kind for kind in kinds if kind.endswith("_delivered")]  # drained is not received
    queued = [event["message_id"] for event in events if event["event"] == "team_message_queued"]
    assert len(set(queued)) == 3
    assert sorted(item["message_id"] for event in observed for item in event["items"]) == sorted(queued)


def test_task_assigned_after_the_last_step_is_delivered_at_the_idle_transition():
    # The owner is still "running" when the assignment arrives, but its round
    # already took its last step: no drain remains, and native never
    # auto-claims an owned in_progress task.
    events = run_js("""
const alpha = agent('running');
const team = makeTeam({ alpha });
assert.deepEqual(bridge.teammateInputs(alpha), []);   // the round's last drain
assert.equal(await team.dispatchAssignedTask(assigned('9', 'alpha')), true);
alpha.status = 'idle';                                // the round ends without another step
await team.flushNextMessage('alpha-id', 'alpha');     // native idle transition
assert.equal(alpha.inbox.length, 1);
assert.match(alpha.inbox[0], /arrived while you were working\\.\\nYou have been assigned task #9\\./);
assert.deepEqual(team.autoClaims, []);                // the assignment took the idle slot
await team.flushNextMessage('alpha-id', 'alpha');     // never twice
assert.equal(alpha.inbox.length, 1);
requestAs('alpha', alpha.inbox[0]);                   // the new round's first request carries it
console.log('SCENARIO_OK');
""")
    drained = [event for event in events if event["event"] == "task_notice_drained"]
    assert [(event["mode"], event["task_ids"]) for event in drained] == [("idle", ["9"])]
    assert [event["task_ids"] for event in events if event["event"] == "task_notice_observed"] == [["9"]]


def test_each_shutdown_request_is_announced_even_after_a_rejection():
    run_js("""
const alpha = agent('running');
const team = makeTeam({ alpha });
team.markShutdownRequested('alpha');
assert.deepEqual(bridge.teammateInputs(alpha).map(input => input.kind), ['shutdown_notice']);
assert.deepEqual(bridge.teammateInputs(alpha), []);   // once per request
// The teammate rejects mid-round; native settles the response against the token.
const reservation = team.reserveShutdownResponse('alpha');
assert.equal(team.settleShutdownResponse('alpha', reservation, true), true);
assert.equal(team.hasShutdownWork('alpha'), false);
assert.deepEqual(bridge.teammateInputs(alpha), []);
team.markShutdownRequested('alpha');                  // a second request
assert.deepEqual(bridge.teammateInputs(alpha).map(input => input.kind), ['shutdown_notice']);
assert.deepEqual(bridge.teammateInputs(alpha), []);
// Without native's ledger the shutdown notice cannot be tracked: refuse the team.
assert.throws(() => Object.create(TeamManager.prototype).setLeaderMessageCallback(() => {}), /shutdown ledger/);
console.log('SCENARIO_OK');
""")


def test_drained_but_unsent_messages_are_requeued_at_idle_and_reported_on_failure():
    events = run_js("""
const alpha = agent('running');
const team = makeTeam({ alpha });
const queue = team.pendingMessages.get('alpha-id');
await team.sendMessage('alpha', 'seen one', 'leader');
requestAs('alpha', bridge.teammateInputs(alpha)[0].text);   // the next request carried it
await team.sendMessage('alpha', 'lost one', 'beta');
bridge.teammateInputs(alpha);                         // drained; the round is then cancelled before a request
alpha.status = 'idle';
await team.flushNextMessage('alpha-id', 'alpha');
assert.equal(alpha.inbox.length, 1);                  // native idle delivery, once
assert.match(alpha.inbox[0], /from="beta">\\nlost one\\n/);
assert.equal(queue.length, 0);
requestAs('alpha', alpha.inbox[0]);                   // that round's first request
// A round that ends in failure (turn or time limit) after a drain: the failure
// notice lists the unsent message with native's still-queued ones.
alpha.status = 'running';
await team.sendMessage('alpha', 'lost two', 'leader');
bridge.teammateInputs(alpha);
await team.sendMessage('alpha', 'still queued', 'beta');
const sent = [];
team.sendMessage = async (to, text) => { sent.push(text); };
bridge.terminal(team, alpha, 'alpha-id', 'alpha', { newStatus: 'failed' }, async () => []);
await new Promise(resolve => setTimeout(resolve, 20));
assert.equal(sent.length, 1);
assert.match(sent[0], /Undelivered queued instructions: \\["lost two","still queued"\\]/);
console.log('SCENARIO_OK');
""")
    kinds = [event["event"] for event in events]
    requeued = [event for event in events if event["event"] == "teammate_inputs_requeued"]
    assert [(len(event["message_ids"]), event["task_ids"]) for event in requeued] == [(1, [])]
    observed = [event["items"][0] for event in events if event["event"] == "team_message_observed"]
    assert [item["mode"] for item in observed] == ["between_tool_calls", "idle_flush"]
    assert observed[1]["message_id"] == requeued[0]["message_ids"][0]
    idle = [event for event in events if event["event"] == "team_message_drained" and event["mode"] == "idle_flush"]
    assert [[item["from"] for item in event["items"]] for event in idle] == [["beta"]]
    assert "failure_notice_persisted" in kinds


def test_cancelled_round_restores_both_the_task_and_the_shutdown_notice():
    # Review 2026-09-30 (round 3): the shutdown token was marked announced when
    # its notice was drained; if the round was cancelled before a request, the
    # restored task round never heard of the still-pending shutdown.
    events = run_js("""
const alpha = agent('running');
const team = makeTeam({ alpha });
assert.equal(await team.dispatchAssignedTask(assigned('3', 'alpha')), true);
team.markShutdownRequested('alpha');
assert.deepEqual(bridge.teammateInputs(alpha).map(input => input.kind), ['task_notice', 'shutdown_notice']);
alpha.status = 'idle';                                // cancelled before the next request
await team.flushNextMessage('alpha-id', 'alpha');     // the task runs as the next round
assert.equal(alpha.inbox.length, 1);
assert.match(alpha.inbox[0], /You have been assigned task #3\\./);
alpha.status = 'running';
const again = bridge.teammateInputs(alpha);           // its first step: shutdown still pending
assert.deepEqual(again.map(input => input.kind), ['shutdown_notice']);
requestAs('alpha', alpha.inbox[0] + '\\n' + again[0].text);
assert.deepEqual(bridge.teammateInputs(alpha), []);   // once observed, not again
console.log('SCENARIO_OK');
""")
    kinds = [event["event"] for event in events]
    requeued = [event for event in events if event["event"] == "teammate_inputs_requeued"]
    assert [(event["task_ids"], event["shutdown_notices"]) for event in requeued] == [(["3"], 1)]
    assert kinds.count("shutdown_notice_drained") == 2 and kinds.count("shutdown_notice_observed") == 1
    assert [event["modes"] for event in events if event["event"] == "task_notice_observed"] == [["idle"]]


def test_idle_and_requeued_deliveries_are_observed_once_per_message():
    # Review 2026-09-30 (round 3): native idle_flush deliveries were drained
    # but never registered for observation, so received messages went uncounted.
    events = run_js("""
const alpha = agent('idle');
const team = makeTeam({ alpha });
await team.sendMessage('alpha', 'idle hello', 'leader');   // idle recipient: native flushes at once
assert.equal(alpha.inbox.length, 1);
assert.match(alpha.inbox[0], /^<team_message_[0-9a-f]{16} from="leader">\\nidle hello\\n/);
requestAs('alpha', alpha.inbox[0]);
alpha.status = 'running';
await team.sendMessage('alpha', 'busy hello', 'beta');
const drained = bridge.teammateInputs(alpha);
alpha.status = 'idle';                                // cancelled before the next request
await team.flushNextMessage('alpha-id', 'alpha');     // requeued, then native idle delivery
assert.equal(alpha.inbox.length, 2);
assert.match(alpha.inbox[1], /from="beta">\\nbusy hello\\n/);
assert.notEqual(alpha.inbox[1], drained[0].text);     // a fresh native envelope
requestAs('alpha', alpha.inbox[1]);
requestAs('alpha', alpha.inbox.join('\\n'));          // later requests do not count again
console.log('SCENARIO_OK');
""")
    queued = [event["message_id"] for event in events if event["event"] == "team_message_queued"]
    observed = [item for event in events if event["event"] == "team_message_observed" for item in event["items"]]
    assert [(item["message_id"], item["mode"]) for item in observed] == [(queued[0], "idle_flush"),
                                                                         (queued[1], "idle_flush")]
    drained = [(event["mode"], item["message_id"]) for event in events if event["event"] == "team_message_drained"
               for item in event["items"]]
    # Drained twice, observed once: count receipt by message_id, never by drains.
    assert drained == [("idle_flush", queued[0]), ("between_tool_calls", queued[1]), ("idle_flush", queued[1])]
    assert "team_message_unobservable" not in [event["event"] for event in events]


def test_default_mode_leaves_native_delivery_untouched():
    result = run_node("""
await import(%s);
console.log(typeof globalThis[Symbol.for('agent-company.native')].teammateInputs);
""" % json.dumps(COMPAT.as_uri()))
    events = lifecycle(result.stderr)
    assert "teammate_delivery_patch_applied" not in [event["event"] for event in events]
    loaded = next(event for event in events if event["event"] == "compatibility_loaded")
    assert loaded["team_message_delivery"] == "native_idle"
    assert "teammate_delivery_between_tool_calls" not in loaded["repairs"]


@pytest.mark.parametrize("env, message", [
    ({"AGENT_COMPANY_TEAM_MESSAGE_DELIVERY": "instant"}, "Unsupported team message delivery mode"),
    ({"AGENT_COMPANY_TEAM_MESSAGE_DELIVERY": "between_tool_calls", "AGENT_COMPANY_ROOT_OWNER": "single"},
     "Team runs only"),
])
def test_invalid_delivery_configuration_fails_before_any_model_call(env, message):
    result = run_node("await import(%s);" % json.dumps(COMPAT.as_uri()), env, check=False)
    assert result.returncode != 0 and message in result.stderr


def test_runner_validates_mode_and_master_text():
    with pytest.raises(ValueError):
        QwenTeamRunner("cli.js", team_message_delivery="instant")
    assert QwenTeamRunner("cli.js").team_message_delivery == "native_idle"
    text = master_system("A", "C", delivery="between_tool_calls")
    assert "next model step" in text and "queued until its complete work turn ends" not in text
    assert "queued until its complete work turn ends" in master_system("A", "C")
