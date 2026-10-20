"""No-provider native private-function repairs and lifecycle evidence."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from agent_company.runtime.qwen import native_lifecycle, runtime_stop_reason
from test_rejection_integration import native_qwen


@pytest.mark.parametrize("owner", ["master", "single"])
def test_teammate_count_override_is_team_only_and_preserves_other_native_settings(owner):
    cli, node = native_qwen()
    compat = Path(__file__).parents[1] / "agent_company/runtime/qwen_native_compat.mjs"
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const chunks = process.argv[1];
const file = fs.readdirSync(chunks).find(name => name.endsWith('.js')
  && fs.readFileSync(path.join(chunks, name), 'utf8').includes('var TeamManager = class'));
const {Config, TeamManager} = await import(pathToFileURL(path.join(chunks, file)).href);
const original = {maxParallelAgents: 3, team: {maxTeammates: 2, unrelatedSetting: 'kept'}};
const config = Object.create(Config.prototype);
config.agentsSettings = original;
const settings = config.getAgentsSettings();
const master = process.env.AGENT_COMPANY_ROOT_OWNER === 'master';
assert.equal(settings.team.maxTeammates, master ? Infinity : 2);
assert.equal(settings.maxParallelAgents, 3);
assert.equal(settings.team.unrelatedSetting, 'kept');
assert.deepEqual(original, {maxParallelAgents: 3, team: {maxTeammates: 2, unrelatedSetting: 'kept'}});
if (master) {
  assert.notEqual(settings, original);
  assert.notEqual(settings.team, original.team);
  const team = new TeamManager({}, {name: 'unit-limit', members: []}, null,
    {maxTeammates: config.getAgentsSettings().team.maxTeammates});
  assert.equal(team.maxTeammates, Infinity);
  team.taskUpdateUnsubscribe();
} else {
  assert.equal(settings, original);
}
config.agentsSettings = undefined;
assert.equal(config.getAgentsSettings()?.team?.maxTeammates, master ? Infinity : undefined);
'''
    result = subprocess.run([node, "--import", str(compat), "--input-type=module", "-e", script,
                             str(cli.parent / "chunks")], check=True, capture_output=True, text=True,
                            env={**os.environ, "AGENT_COMPANY_QWEN_CLI": str(cli),
                                 "AGENT_COMPANY_ROOT_OWNER": owner,
                                 "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:12345/v1"}, timeout=30)
    prefix = "Agent Company lifecycle: "
    loaded = next(json.loads(line[len(prefix):]) for line in result.stderr.splitlines()
                  if line.startswith(prefix) and json.loads(line[len(prefix):])["event"] == "compatibility_loaded")
    assert loaded["teammate_limit"] == (10 if owner == "master" else None)
    # The real package's shell service is loaded (and patched) before this record.
    assert "shell_output_drain" in loaded["repairs"]


def test_compaction_patch_retains_literal_analysis_tags_and_rejects_truncation():
    cli, node = native_qwen()
    module = Path(__file__).parents[1] / "agent_company/runtime/qwen_native_patch.mjs"
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
const {stripLeadingAnalysis, patchNativeSource} = await import(process.argv[1]);
const snapshot = '<state_snapshot>quote `<analysis>` and </analysis> intact</state_snapshot>';
assert.equal(stripLeadingAnalysis('<analysis>private</analysis>\n' + snapshot), snapshot);
assert.equal(stripLeadingAnalysis(snapshot), snapshot);
assert.equal(stripLeadingAnalysis('<analysis>truncated'), '');
assert.equal(stripLeadingAnalysis('<analysis>x</analysis><state_snapshot>truncated'), '<state_snapshot>truncated');
let changed = 0;
for (const filename of fs.readdirSync(process.argv[2])) {
  if (!filename.endsWith('.js')) continue;
  const original = fs.readFileSync(path.join(process.argv[2], filename), 'utf8');
  if (patchNativeSource(original) !== original) changed++;
}
assert.equal(changed, 4);
assert.throws(() => patchNativeSource('function stripAnalysisBlock(rawSummary) { changed }'), /anchor changed/);
'''
    subprocess.run([node, "--input-type=module", "-e", script, module.as_uri(), str(cli.parent / "chunks")],
                   check=True, capture_output=True, text=True)


def test_unobserved_or_failed_notification_is_not_cli_completion(tmp_path):
    stderr = tmp_path / "stderr"
    events = [{"event": "worker_terminal", "status": "failed", "worker": "scout"},
              {"event": "failure_notice_queued", "notice_id": "n1", "worker": "scout"}]
    prefix = "Agent Company lifecycle: "
    stderr.write_text("".join(prefix + json.dumps(event) + "\n" for event in events))
    lifecycle = native_lifecycle(stderr)
    cli = {"explicit_error": False, "missing_result": False, "lifecycle": lifecycle}
    execution = {"returncode": 0, "timed_out": False}
    assert runtime_stop_reason(execution, cli, None, {"errors": []}) == "unhandled_worker_failure"
    stderr.write_text(stderr.read_text() + prefix + json.dumps({
        "event": "failure_notice_observed", "notice_id": "n1"}) + "\n")
    cli["lifecycle"] = native_lifecycle(stderr)
    assert runtime_stop_reason(execution, cli, None, {"errors": []}) == "native_cli_completed"
    # A rejected provider response is not evidence corruption. The bad attempt
    # stays rejected, but a later valid completion is not permanently vetoed.
    assert runtime_stop_reason(execution, cli, None, {"errors": [
        {"error": "Invalid JSON", "error_kind": "invalid_provider_response"}]}) == "native_cli_completed"


def test_native_context_side_query_and_terminal_drain(tmp_path):
    cli, node = native_qwen()
    compat = Path(__file__).parents[1] / "agent_company/runtime/qwen_native_compat.mjs"
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const bridge = globalThis[Symbol.for('agent-company.native')];
const envelope = (from, text) => `<teammate_message from="${from}">\n${text}\n</teammate_message>`;
const leaderQueue = [envelope('runtime:system', 'failure') + '\n\n' + envelope('runtime:checkin', 'old: stale')];
bridge.enqueueLeaderMessage(leaderQueue, envelope('alpha', 'worker report'));
bridge.enqueueLeaderMessage(leaderQueue, envelope('runtime:checkin', 'new: fresh'));
assert.equal(leaderQueue.length, 3);
assert(leaderQueue[0].includes('failure') && !leaderQueue.join('\n').includes('old: stale'));
assert(leaderQueue[1].includes('worker report'));
for (let i = 0; i < 10; i++) bridge.enqueueLeaderMessage(leaderQueue, envelope('runtime:checkin', `next-${i}: fresh`));
assert.equal(leaderQueue.length, 3, 'Busy leader check-ins must not accumulate');
assert(leaderQueue[2].includes('next-9: fresh'));
const inFlight = leaderQueue.splice(0);
bridge.enqueueLeaderMessage(leaderQueue, envelope('runtime:checkin', 'later: newest'));
assert(inFlight[2].includes('next-9: fresh') && leaderQueue.length === 1, 'Never mutate an already-drained request');
const recorded = [];
const originalWrite = process.stderr.write.bind(process.stderr);
process.stderr.write = function (text, ...args) { recorded.push(String(text)); return originalWrite(text, ...args); };
const chunks = process.argv[1];
async function find(needle) {
  for (const name of fs.readdirSync(chunks)) {
    const file = path.join(chunks, name);
    if (name.endsWith('.js') && fs.readFileSync(file, 'utf8').includes(needle)) return import(pathToFileURL(file).href);
  }
  throw new Error('Native source missing');
}
const {runWithTeammateIdentity} = await find('var teammateIdentityStore = new AsyncLocalStorage()');
const {runSideQuery} = await find('async function runSideQuery(config, options) {');
const {TeamManager} = await find('var TeamManager = class');
const base = 'http://127.0.0.1:12345/v1';
const seen = [];
const config = {getBaseLlmClient() { return {async generateText(opts) {
  await Promise.resolve();
  seen.push(bridge.headers(base, {messages: []}));
  assert.equal(opts.systemInstruction, 'Summarize only');
  return {text: '<state_snapshot>ok</state_snapshot>'};
}}}};
await Promise.all(['alpha','beta'].map(name => runWithTeammateIdentity({agentName:name}, () =>
  runSideQuery(config, {model: 'test', purpose: 'chat-compression', skipOutputLanguagePreference: true,
    contents: [], systemInstruction: 'Summarize only', abortSignal: new AbortController().signal}))));
assert.deepEqual(seen.map(h => h['X-Agent-Company-Owner']).sort(), ['alpha','beta']);
assert(seen.every(h => h['X-Agent-Company-Purpose'] === 'chat-compression'));
assert.equal(bridge.headers(base, {messages: []})['X-Agent-Company-Owner'], 'master');
assert.equal(bridge.headers(base, {messages: []})['X-Agent-Company-Purpose'], 'coding');
assert.throws(() => bridge.headers('https://external.invalid/v1', {}), /bypasses/);

const sent = [];
const manager = {teamFile:{name:'unit'}, pendingMessages:new Map(),
 async sendMessage(to,text,from) { await new Promise(resolve => setTimeout(resolve, 15)); sent.push({to,text,from}); },
 async pollLeaderInbox() { assert.equal(sent.length, 2, 'Final drain raced async terminal writes'); }};
const agent = {getError: () => 'Connection failed'};
const tasks = async () => [{id:'1',subject:'implement',status:'in_progress',owner:'alpha'}];
bridge.terminal(manager,agent,'alpha@unit','alpha',{newStatus:'failed'},tasks);
bridge.terminal(manager,agent,'beta@unit','beta',{newStatus:'failed'},tasks);
bridge.terminal(manager,agent,'normal@unit','normal',{newStatus:'cancelled'},tasks);
bridge.terminal(manager,agent,'user@unit','user',{newStatus:'failed',roundCancelledByUser:true},tasks);
await TeamManager.prototype.drainLeaderInbox.call(manager);
assert(sent.every(message => message.from === 'runtime:system' && message.to === 'leader'));
assert(sent[0].text.includes('implement'));
const noticeMessages = sent.map(message => ({role:'user',content:`<teammate_message from="runtime:system">\n${message.text}\n</teammate_message>`}));
bridge.withPurpose('chat-compression', () => bridge.headers(base, {messages:noticeMessages}));
assert(!recorded.some(line => line.includes('failure_notice_observed')));
bridge.headers(base, {messages: sent.map(message => ({role:'user',content:message.text}))});
assert(!recorded.some(line => line.includes('failure_notice_observed')), 'Unframed IDs are not runtime envelopes');
bridge.headers(base, {messages: noticeMessages});

const broken = {teamFile:{name:'broken'},pendingMessages:new Map(),
 async sendMessage() {throw new Error('Disk write failed');}, async pollLeaderInbox() {throw new Error('Must not finalize');}};
bridge.terminal(broken,agent,'failed@unit','failed',{newStatus:'failed'},async () => []);
await assert.rejects(TeamManager.prototype.drainLeaderInbox.call(broken), /Disk write failed/);
'''
    result = subprocess.run([node, "--import", str(compat), "--input-type=module", "-e", script,
                             str(cli.parent / "chunks")], check=True, capture_output=True, text=True,
                            env={**os.environ, "AGENT_COMPANY_QWEN_CLI": str(cli),
                                 "AGENT_COMPANY_ROOT_OWNER": "master",
                                 "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:12345/v1"})
    stderr = tmp_path / "stderr"
    stderr.write_text(result.stderr)
    lifecycle = native_lifecycle(stderr)
    events = lifecycle["events"]
    assert sum(event["event"] == "failure_notice_observed" for event in events) == 2
    assert len(lifecycle["notification_errors"]) == 1
    assert len(lifecycle["unobserved_failure_notices"]) == 1


@pytest.mark.parametrize("fixed", [False, True])
def test_automatic_leader_write_is_awaited_before_final_drain(fixed):
    """Same barrier-controlled native race: old drain loses text, compat waits."""
    cli, node = native_qwen()
    root = Path(__file__).parents[1]
    imports = ["--import", str(root / "agent_company/runtime/qwen_native_compat.mjs")] if fixed else []
    result = subprocess.run([node, *imports, str(root / "tests/native_final_report_repro.mjs"),
                             str(cli.parent / "chunks"), "after" if fixed else "before"],
                            check=True, capture_output=True, text=True, timeout=30,
                            env={**os.environ, "AGENT_COMPANY_QWEN_CLI": str(cli),
                                 "AGENT_COMPANY_ROOT_OWNER": "master",
                                 "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:12345/v1"})
    evidence = json.loads(result.stdout)
    assert evidence["lost"] is not fixed


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("detach", [False, True])
def test_inbox_poll_uses_current_callback_after_io_and_respects_detach(fixed, detach):
    cli, node = native_qwen()
    root = Path(__file__).parents[1]
    imports = ["--import", str(root / "agent_company/runtime/qwen_native_compat.mjs")] if fixed else []
    result = subprocess.run([node, *imports, str(root / "tests/native_leader_wakeup_repro.mjs"),
                             str(cli.parent / "chunks"), "after" if fixed else "before", "detach" if detach else "wait"],
                            check=True, capture_output=True, text=True, timeout=30,
                            env={**os.environ, "AGENT_COMPANY_QWEN_CLI": str(cli),
                                 "AGENT_COMPANY_ROOT_OWNER": "master",
                                 "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:12345/v1"})
    assert json.loads(result.stdout)["awakened"] is (fixed and not detach)
