"""Opt-in runtime team check-ins: scheduling, relay accounting, native delivery.

Check-ins give the coordination-only master periodic decision opportunities with
an observational snapshot. They never spawn, stop or reassign workers.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

from agent_company.analysis.evidence_projection import project_evidence
from agent_company.runtime.qwen import QwenTeamRunner, master_system
from agent_company.runtime.relay import ProviderConfig, TeamRelay, role_of, start_server
from agent_company.runtime.team import WorkerRegistry
from checkin_scenario import CheckinScenario
from test_team import call, request, response, workspace

MODULE = Path(__file__).parents[1] / "agent_company/runtime/qwen_team_checkin.mjs"


def node_or_skip():
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not node:
        pytest.skip("node is required for the check-in module tests")
    return node


def run_node(script):
    subprocess.run([node_or_skip(), "--input-type=module", "-e", script, MODULE.as_uri()],
                   check=True, capture_output=True, text=True)


def test_compact_snapshot_keeps_the_time_base_of_its_ages():
    # Review 2026-10-01: *_ago_s are measured from the sample's start, but the
    # compact record kept only its finish time. A 90-second sample made a
    # 5-second-old activity look 5 seconds before the finish instead of 95.
    run_node(r'''
import assert from 'node:assert/strict';
const {compactCheckin} = await import(process.argv[1]);
const view = {at: '2026-09-30T06:01:30.000Z', sample_started_at: '2026-09-30T06:00:00.000Z',
  elapsed_s: 1890, remaining_s: 19710, unassigned_open_tasks: 0,
  delivery: {commits_since_launch: 2, last_commit_ago_s: 30, last_commit_subject: 'private subject', head: 'abc'},
  workers: [{name: 'alpha', status: 'running', last_activity_ago_s: 5, shutdown_pending: false,
             tasks: [{id: '1', status: 'in_progress', subject: 'private task'}],
             pending_tools: [{name: 'run_shell_command', announced_ago_s: 3}],
             git: {commits_since_worker_creation: 1, last_commit_subject: 'private', head: 'abc'}}],
  unknown: ['relay']};
const compact = compactCheckin(view);
assert.equal(compact.sample_started_at, view.sample_started_at);
assert.equal(compact.at, view.at);
assert.equal(compact.workers[0].last_activity_ago_s, 5);
assert.deepEqual(compact.workers[0].pending_tools, ['run_shell_command']);
assert.equal(compact.workers[0].open_tasks, 1);
assert(!JSON.stringify(compact).includes('private') && !JSON.stringify(compact).includes('abc'));
''')


def test_a_tool_call_announced_before_spawn_returns_is_observed():
    # 2026-10-18 Linux: a scripted first reply announced the teammate's only tool call before
    # spawnTeammate returned, so the post-spawn listener missed it. The listener now attaches
    # when the native in-process backend registers the agent, before its first turn.
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {EventEmitter} from 'node:events';
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const tools = new Map();
const listeners = new Set();
const attachSource = source.slice(source.indexOf('  function attach(team, member) {'), source.indexOf('\n  const spawn = TeamManager.prototype.spawnTeammate;'));
const {attach} = new Function('tools', 'editTools', 'Date', 'agentStartListeners', `${attachSource}; return {attach};`)(
  tools, new Set(['edit','write_file']), Date, listeners);
const emitter = new EventEmitter();
const agent = {getEventEmitter: () => emitter};
// The native backend registers the agent, then starts it; the first call comes before spawn returns.
for (const listener of listeners) listener('alpha', agent);
emitter.emit('tool_call', {callId: 'shell', name: 'run_shell_command', timestamp: 1});
attach({getAgentFromBackend: () => agent}, {name: 'alpha', agentId: 'alpha'});  // the post-spawn fallback
const state = tools.get('alpha');
assert.equal(state.commands, 1);
assert.deepEqual([...state.pending.keys()], ['shell']);
assert.equal(emitter.listenerCount('tool_call'), 1, 'attached once');
// A failed spawn retried under the same id starts a new agent with a new emitter.
const retry = new EventEmitter();
for (const listener of listeners) listener('alpha', {getEventEmitter: () => retry});
retry.emit('tool_call', {callId: 'edit', name: 'edit', timestamp: 2});
assert.equal(tools.get('alpha').edits, 1);
assert.deepEqual([...tools.get('alpha').pending.keys()], ['edit']);
''')


def test_the_start_hook_precedes_the_teammates_first_turn_in_the_real_native_source():
    chunks = MODULE.parents[2] / "external/qwen-study/package/chunks"
    if not chunks.is_dir() or not shutil.which("node"):
        pytest.skip("Needs the installed native package and Node")
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
const {patchNativeSource} = await import(new URL('./qwen_native_patch.mjs', process.argv[1]));
const chunks = ''' + json.dumps(str(chunks)) + r''';
const name = fs.readdirSync(chunks).find(file => fs.readFileSync(path.join(chunks, file), 'utf8').includes('var InProcessBackend = class'));
const patched = patchNativeSource(fs.readFileSync(path.join(chunks, name), 'utf8'));
const spawn = patched.indexOf('async spawnAgent(config3) {', patched.indexOf('var InProcessBackend = class'));
const hook = patched.indexOf('.agentStarting(config3.agentId, interactive);', spawn);
const start = patched.indexOf('interactive.start(context2)', spawn);
assert(spawn > 0 && hook > spawn && start > hook, 'hook must run after registration and before start()');
''')


def test_native_tool_observations_report_pending_results_not_execution_time():
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {EventEmitter} from 'node:events';
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const tools = new Map();
let now = 1_000_000;
class Clock extends Date { static now() { return now; } }
const attachSource = source.slice(source.indexOf('  function attach(team, member) {'), source.indexOf('\n  const spawn = TeamManager.prototype.spawnTeammate;'));
const {attach} = new Function('tools', 'editTools', 'Date', 'agentStartListeners', `${attachSource}; return {attach, observeTools};`)(
  tools, new Set(['edit','write_file']), Clock, new Set());
const emitter = new EventEmitter();
const member = {name: 'alpha', agentId: 'alpha', joinedAt: now, cwd: '/worker'};
const team = {teamFile: {name: 'team', members: [member]}, lastActivityAt: new Map(), pendingMessages: new Map(),
  getAgentFromBackend: () => ({getEventEmitter: () => emitter, getStatus: () => 'running'})};
attach(team, member);
// Native emits the whole batch before scheduling and results only after the
// batch completes. The first read may already be done while the shell runs.
emitter.emit('tool_call', {callId: 'read', name: 'read_file', timestamp: now});
emitter.emit('tool_call', {callId: 'shell', name: 'run_shell_command', timestamp: now});
now += 600_000;
const snapshotSource = source.slice(source.indexOf('  async function snapshot('), source.indexOf('\n  function publish('));
const snapshot = new Function('env', `const {tools, listTasks, relayToken, relayStatus, commits, repo, worktree, startMs, deadlineMs, refreshMs, statusFile, Date, runtimeNote, rolePackage, agentRules, checkinTotals} = env; return (${snapshotSource});`)({
  tools, listTasks: async () => [], relayToken: 'fixture', relayStatus: async () => ({workers:{alpha:{creation_head:'abc'}}}),
  commits: async () => ({}), repo: '/delivery', worktree: async () => ({}), startMs: 1_000_000, deadlineMs: 9_000_000,
  refreshMs: 60_000, statusFile: '/status', Date: Clock, runtimeNote: null, rolePackage: false,
  agentRules: false, checkinTotals: null});
const view = await snapshot(team, {}, now);
const worker = view.workers[0];
assert.deepEqual(worker.pending_tools, [{name:'read_file',announced_ago_s:600},{name:'run_shell_command',announced_ago_s:600}]);
assert(!('current_tool' in worker) && !('last_tool' in worker));
assert.equal(worker.tool_counts.commands, 1);
now += 600_000;
emitter.emit('tool_result', {callId: 'read', name: 'read_file', durationMs: 10});
emitter.emit('tool_result', {callId: 'shell', name: 'run_shell_command', durationMs: 1_200_000});
now += 1000;
const completed = (await snapshot(team, {}, now)).workers[0];
assert.deepEqual(completed.pending_tools, []);
assert.deepEqual(completed.last_tool_result, {name:'run_shell_command',received_ago_s:1});
''')


def test_native_spawn_returns_while_observation_collection_is_still_pending():
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const wrapper = source.slice(source.indexOf('  const spawn = TeamManager.prototype.spawnTeammate;'), source.indexOf('\n  const cleanup = TeamManager.prototype.cleanup;'));
let release;
const collection = new Promise(resolve => { release = resolve; });
let polls = 0;
class TeamManager { constructor() { this.teamFile = {members: []}; } async spawnTeammate() { return 'spawned'; } }
new Function('TeamManager', 'poll', 'attach', 'setInterval', `const managers = new Set(); const scheduler = {stopped: false}; let timer = null; const intervalMs = 1000; ${wrapper}`)(
  TeamManager, async () => { polls++; await collection; }, () => {}, () => ({unref() {}}));
let returned = false;
const spawning = new TeamManager().spawnTeammate().then(value => { returned = true; return value; });
await new Promise(resolve => setImmediate(resolve));
assert.equal(polls, 1);
assert(returned, 'spawn was blocked by Git/relay observation sampling');
release();
assert.equal(await spawning, 'spawned');
''')


def test_native_checkin_callback_failure_retracts_without_sent_or_observed_event():
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
const {CheckinScheduler, compactCheckin, runCheckin, SENDER} = await import(process.argv[1]);
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const pollSource = source.slice(source.indexOf('  async function poll('), source.indexOf('\n  // Exists even before team_create'));
for (const failure of ['unavailable', 'throws', 'rejects']) {
  const scheduler = new CheckinScheduler({intervalMs:1000,startMs:0,deadlineMs:100_000,milestones:[],finalGuardMs:1});
  const events = [];
  let clock = 1000, wakes = 0;
  class Clock extends Date { static now() { return clock; } }
  const team = {formatLeaderEnvelope: messages => messages.map(m => `<teammate_message from="${m.from}">\n${m.text}\n</teammate_message>`)};
  if (failure === 'throws') team.leaderMessageCallback = () => {throw new Error('callback rejected');};
  if (failure === 'rejects') team.leaderMessageCallback = async () => {throw new Error('callback rejected');};
  const poll = new Function('env', `const {scheduler, runCheckin, compactCheckin, SENDER, team, Date, record} = env;
    const managers = new Set([team]); let polling = false, lastSkipReason = null;
    const lastSnapshotMs = 0, refreshMs = 250, timer = null, checkinSeconds = 1;
    const teamState = () => ({active:true,windingDown:false});
    const snapshot = async (team, decision) => ({id:decision.id,workers:[],unknown:[]});
    const publish = () => {}; const packageCheckin = view => view.id + ': fixture'; const checkinText = packageCheckin;
    return (${pollSource});`)({scheduler,runCheckin,compactCheckin,SENDER,team,Date:Clock,record:(event,data) => events.push({event,...data})});
  await poll();
  assert.equal(scheduler.pendingId, null, failure);
  assert.deepEqual(events.map(e => e.event), ['checkin_error']);
  assert.match(events[0].error, /unavailable|callback rejected/);
  // Failure releases the existing polling lock. A later native callback can
  // queue/wake normally; retries are not silently suppressed for an interval.
  const queued = [];
  team.leaderMessageCallback = (formatted, text) => {queued.push(formatted); wakes++; assert.equal(text, 'checkin-2: fixture');};
  clock = 1100;
  await poll();
  assert.equal(wakes, 1);
  assert.equal(scheduler.pendingId, 'checkin-2');
  assert.equal(events.at(-1).event, 'checkin_sent');
  // The sent snapshot is kept as lifecycle evidence: counts and states only.
  assert.deepEqual(events.at(-1).snapshot.workers, []);
  assert.equal(events.at(-1).snapshot.delivery, null);
  assert(!events.some(e => e.event === 'checkin_observed'));
  assert(queued[0].includes('<teammate_message from="runtime:checkin">\ncheckin-2:'));
}
''')


def test_worker_git_observation_excludes_inherited_commits_and_keeps_unknown(tmp_path):
    ws = workspace(tmp_path)
    initial = ws.read_head(ws.repo_path)
    commit = ["git", "-c", "user.name=T", "-c", "user.email=t@localhost", "-c", "commit.gpgsign=false",
              "commit", "--allow-empty", "-qm"]
    for subject in ("earlier worker one", "earlier worker two"):
        subprocess.run([*commit, subject], cwd=ws.repo_path, check=True)
    registry = WorkerRegistry(ws)
    worker = registry.register("late")
    creation = registry.metadata()["late"]["creation_head"]
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const {repo, worker, initial, creation} = FIXTURE;
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const functions = source.slice(source.indexOf('  async function commits('), source.indexOf('\n  async function relayStatus('));
const git = async (args, cwd) => execFileSync('git', ['-C', cwd, ...args], {encoding:'utf8'}).trim();
const {commits, worktree} = new Function('git', 'baseline', 'path', 'repo', `const rolePackage = false; const sampledHeads = new Map(); ${functions}; return {commits, worktree};`)(git, initial, path, repo);
assert.equal((await commits(repo, Date.now())).commits_since_launch, 2);
let observed = await worktree(worker, Date.now(), creation);
assert.equal(observed.commits_since_worker_creation, 0);
assert.equal(observed.last_commit_ago_s, null);
assert(!('commits_since_launch' in observed));
execFileSync('git', ['-C', worker, '-c','user.name=T','-c','user.email=t@localhost','-c','commit.gpgsign=false','commit','--allow-empty','-qm','own work']);
observed = await worktree(worker, Date.now(), creation);
assert.equal(observed.commits_since_worker_creation, 1);
assert.equal(observed.creation_head, creation);
const unknown = await worktree(worker, Date.now(), null);
assert.equal(unknown.commits_since_worker_creation, null);
assert.equal(unknown.creation_head, null);
assert.equal(unknown.last_commit_ago_s, null);
// A missing Git object must fail this observation, never fall back to the
// task-start baseline or a timestamp scan. snapshot's guard labels it unknown.
await assert.rejects(worktree(worker, Date.now(), 'missing-creation-head'));
'''.replace("FIXTURE", json.dumps({"repo": ws.repo_path, "worker": worker, "initial": initial, "creation": creation})))


def test_package_git_sampling_covers_intermediate_and_merge_commits(tmp_path):
    ws = workspace(tmp_path)
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const repo = FIXTURE;
const command = args => execFileSync('git', ['-C', repo, '-c', 'user.name=T', '-c', 'user.email=t@localhost',
  '-c', 'commit.gpgsign=false', ...args], {encoding:'utf8'}).trim();
const initial = command(['rev-parse', 'HEAD']);
const branch = command(['branch', '--show-current']);
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const functions = source.slice(source.indexOf('  async function commits('), source.indexOf('\n  async function relayStatus('));
const git = async (args, cwd) => {
  try { return execFileSync('git', ['-C', cwd, ...args], {encoding:'utf8', stdio:['ignore','pipe','pipe']}).trim(); }
  catch (error) { error.code = error.status; throw error; }
};
const {commits, emitGitMetadata} = new Function('git', 'baseline', 'path', 'repo', `const rolePackage = true; const sampledHeads = new Map(); const sampledGitFailures = new Map(); ${functions}; return {commits, emitGitMetadata};`)(git, initial, path, repo);
const collect = async () => {
  const result = await commits(repo, Date.now());
  await emitGitMetadata('master', repo, result, Date.now(), async () => {});
  return result;
};
assert.deepEqual((await collect()).commit_samples, []);
fs.writeFileSync(path.join(repo, 'intermediate.pyc'), 'synthetic cache');
command(['add', 'intermediate.pyc']); command(['commit', '-qm', 'intermediate cache']);
const cache = command(['rev-parse', 'HEAD']);
fs.unlinkSync(path.join(repo, 'intermediate.pyc'));
command(['add', '-u']); command(['commit', '-qm', 'remove intermediate cache']);
const removed = command(['rev-parse', 'HEAD']);
let result = await collect();
assert.deepEqual(result.commit_samples.map(row => row.commit), [cache, removed]);
assert.deepEqual(result.commit_samples[0].added_paths, ['intermediate.pyc']);
assert.deepEqual(result.commit_samples[1].deleted_paths, ['intermediate.pyc']);
assert.deepEqual(result.sampling_incomplete, []);
assert.deepEqual((await collect()).commit_samples, []);
command(['checkout', '-qb', 'fixture-side']);
fs.writeFileSync(path.join(repo, 'side.txt'), 'side'); command(['add', 'side.txt']); command(['commit', '-qm', 'side']);
const side = command(['rev-parse', 'HEAD']); command(['checkout', '-q', branch]);
fs.writeFileSync(path.join(repo, 'main.txt'), 'main'); command(['add', 'main.txt']); command(['commit', '-qm', 'main']);
const main = command(['rev-parse', 'HEAD']); command(['merge', '--no-ff', '-qm', 'merge fixture', 'fixture-side']);
const merged = command(['rev-parse', 'HEAD']);
result = await collect();
assert.deepEqual(new Set(result.commit_samples.map(row => row.commit)), new Set([side, main, merged]));
assert.deepEqual(new Set(result.commit_samples.find(row => row.commit === merged).changed_paths), new Set(['side.txt','main.txt']));
command(['reset', '--hard', initial]);
result = await collect();
assert.deepEqual(result.sampling_incomplete, ['nonancestor_previous_head']);
assert.deepEqual((await collect()).sampling_incomplete, []);
const fresh = new Function('git', 'baseline', 'path', 'repo', `const rolePackage = true; const sampledHeads = new Map(); ${functions}; return commits;`)(git, null, path, repo);
assert.deepEqual((await fresh(repo, Date.now())).sampling_incomplete, ['missing_previous_head']);
'''.replace('FIXTURE', json.dumps(ws.repo_path)))


@pytest.mark.parametrize('rewrite', ['reset', 'rebase', 'missing', 'unrelated'])
def test_package_git_history_rewrite_resumes_after_one_gap_report(tmp_path, rewrite):
    ws = workspace(tmp_path)
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFile, execFileSync} from 'node:child_process';
const {repo, rewrite} = FIXTURE;
const command = args => execFileSync('git', ['-C', repo, '-c', 'user.name=T', '-c', 'user.email=t@localhost',
  '-c', 'commit.gpgsign=false', ...args], {encoding:'utf8', stdio:['ignore','pipe','pipe']}).trim();
const initial = command(['rev-parse', 'HEAD']), branch = command(['branch', '--show-current']);
const add = (file, subject) => {
  fs.writeFileSync(path.join(repo, file), subject);
  command(['add', file]); command(['commit', '-qm', subject]);
  return command(['rev-parse', 'HEAD']);
};
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const functions = source.slice(source.indexOf('  async function commits('), source.indexOf('\n  async function relayStatus('));
const exits = [];
// Use the production execFile error shape: Git's real exit status is error.code.
const git = (args, cwd) => new Promise((resolve, reject) => execFile('git', ['-C', cwd, ...args],
  {encoding:'utf8'}, (error, stdout) => {
    if (error && args[0] === 'merge-base') exits.push({args, code:error.code});
    error ? reject(error) : resolve(stdout.trim());
  }));
const {commits, emitGitMetadata, sampledHeads} = new Function('git', 'baseline', 'path', 'repo',
  `const rolePackage = true; const sampledHeads = new Map(); const sampledGitFailures = new Map();
   ${functions}; return {commits, emitGitMetadata, sampledHeads};`)(git, rewrite === 'missing' ? null : initial, path, repo);
const collect = async () => {
  const result = await commits(repo, Date.now()), rows = [];
  await emitGitMetadata('master', repo, result, Date.now(), async row => rows.push(row));
  return {result, rows};
};
let expected, warnings, oldHead = null;
if (rewrite === 'missing') {
  expected = []; warnings = ['missing_previous_head'];
} else {
  await collect();
  const oldOne = add('old-one.txt', 'old one');
  oldHead = add('old-two.txt', 'old two');
  await collect();
  assert.equal(sampledHeads.get(repo), oldHead);
  if (rewrite === 'reset') {
    command(['reset', '--hard', initial]);
    expected = [add('after-reset.pyc', 'cache after reset')];
    warnings = ['nonancestor_previous_head'];
  } else if (rewrite === 'rebase') {
    command(['checkout', '-qb', 'fixture-upstream', initial]);
    const upstream = add('upstream.txt', 'upstream');
    command(['checkout', '-q', branch]); command(['rebase', 'fixture-upstream']);
    expected = command(['rev-list', '--reverse', '--topo-order', `${initial}..HEAD`]).split('\n');
    assert.equal(expected.length, 3);
    assert(expected.includes(upstream) && !expected.includes(oldOne) && !expected.includes(oldHead));
    warnings = ['nonancestor_previous_head'];
  } else {
    command(['checkout', '--orphan', 'fixture-unrelated']); command(['rm', '-rf', '.']);
    add('unrelated.pyc', 'new root');
    expected = []; warnings = ['nonancestor_previous_head', 'no_common_ancestor'];
  }
}
const current = command(['rev-parse', 'HEAD']);
let result = await commits(repo, Date.now());
assert.deepEqual(result.commit_samples.map(row => row.commit), expected);
assert.deepEqual(result.sampling_incomplete, warnings);
if (rewrite === 'reset') assert.deepEqual(result.commit_samples[0].added_paths, ['after-reset.pyc']);
if (rewrite !== 'missing') assert(exits.some(row => row.args[1] === '--is-ancestor' && row.code === 1));
if (rewrite === 'unrelated') assert(exits.some(row => row.args.length === 3 && row.code === 1));
// Even a recoverable history gap must not consume its baseline before output succeeds.
await assert.rejects(emitGitMetadata('master', repo, result, Date.now(), async () => {
  throw new Error('injected gap emission failure');
}), /injected gap emission failure/);
assert.equal(sampledHeads.get(repo) ?? null, oldHead);
({result} = await collect());
assert.deepEqual(result.sampling_incomplete, warnings);
assert.equal(sampledHeads.get(repo), current);
let next = await collect();
assert.deepEqual(next.result.commit_samples, []);
assert.deepEqual(next.result.sampling_incomplete, []);
assert(next.rows.every(row => row.git.sampling_incomplete.length === 0));
// A new cache and its deletion remain separate, visible commits after recovery.
const added = add('after-recovery.pyc', 'visible after history recovery');
fs.unlinkSync(path.join(repo, 'after-recovery.pyc'));
command(['add', '-u']); command(['commit', '-qm', 'delete recovered cache']);
const removed = command(['rev-parse', 'HEAD']);
next = await collect();
assert.deepEqual(next.result.commit_samples.map(row => row.commit), [added, removed]);
assert.deepEqual(next.result.commit_samples[0].added_paths, ['after-recovery.pyc']);
assert.deepEqual(next.result.commit_samples[1].deleted_paths, ['after-recovery.pyc']);
assert(next.rows.every(row => row.git.sampling_incomplete.length === 0));
assert.equal(sampledHeads.get(repo), removed);
'''.replace('FIXTURE', json.dumps({'repo': ws.repo_path, 'rewrite': rewrite})))


@pytest.mark.parametrize('failure', ['status', 'ancestry', 'is_ancestor', 'merge_base', 'range', 'emit'])
def test_package_git_failed_sample_keeps_commit_interval_for_next_emission(tmp_path, failure):
    ws = workspace(tmp_path)
    worker = WorkerRegistry(ws).register('alpha')
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const {repo, worker, failure} = FIXTURE;
const command = args => execFileSync('git', ['-C', worker, '-c', 'user.name=T', '-c', 'user.email=t@localhost',
  '-c', 'commit.gpgsign=false', ...args], {encoding:'utf8'}).trim();
const initial = command(['rev-parse', 'HEAD']);
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const functions = source.slice(source.indexOf('  async function commits('), source.indexOf('\n  async function relayStatus('));
let fail = false, fallbackReads = 0;
const git = async (args, cwd) => {
  if (fail && (failure === 'status' && args[0] === 'status'
      || failure === 'ancestry' && args.includes('--not')
      || failure === 'range' && args[0] === 'diff-tree')) throw new Error('injected read failure');
  if (args[0] === 'merge-base' && args.length === 3) fallbackReads++;
  if (fail && (failure === 'is_ancestor' && args[1] === '--is-ancestor'
      || failure === 'merge_base' && args[0] === 'merge-base' && args.length === 3)) {
    // Real Git read failure (128), distinct from --is-ancestor's exit 1.
    try { execFileSync('git', ['-C', cwd, 'rev-parse', '--verify', 'missing-git-object'], {stdio:'pipe'}); }
    catch (error) { error.code = error.status; assert.equal(error.code, 128); throw error; }
  }
  try { return execFileSync('git', ['-C', cwd, ...args], {encoding:'utf8', stdio:['ignore','pipe','pipe']}).trim(); }
  catch (error) { error.code = error.status; throw error; }
};
const {worktree, emitGitMetadata, sampledHeads} = new Function('git', 'baseline', 'path', 'repo',
  `const rolePackage = true; const sampledHeads = new Map(); const sampledGitFailures = new Map();
   ${functions}; return {worktree, emitGitMetadata, sampledHeads};`)(git, initial, path, repo);
await emitGitMetadata('alpha', worker, await worktree(worker, Date.now(), initial), Date.now(), async () => {});
assert.equal(sampledHeads.get(worker), initial);
if (failure === 'merge_base') {
  fs.writeFileSync(path.join(worker, 'abandoned.txt'), 'abandoned');
  command(['add', 'abandoned.txt']); command(['commit', '-qm', 'abandoned branch']);
  await emitGitMetadata('alpha', worker, await worktree(worker, Date.now(), initial), Date.now(), async () => {});
  command(['reset', '--hard', initial]);
}
const previous = sampledHeads.get(worker);
fs.writeFileSync(path.join(worker, 'between.pyc'), 'synthetic cache');
command(['add', 'between.pyc']); command(['commit', '-qm', 'add between samples']);
const first = command(['rev-parse', 'HEAD']);
fs.unlinkSync(path.join(worker, 'between.pyc'));
command(['add', '-u']); command(['commit', '-qm', 'delete between samples']);
const second = command(['rev-parse', 'HEAD']);
fail = true;
let failed = null;
try { failed = await worktree(worker, Date.now(), initial); }
catch (error) { assert.match(error.message, /injected read failure/); }
const firstEmission = [];
if (failure === 'emit') {
  let writes = 0;
  await assert.rejects(emitGitMetadata('alpha', worker, failed, Date.now(), async row => {
    firstEmission.push(row);
    if (++writes === 2) throw new Error('injected emit failure');
  }), /injected emit failure/);
  assert(failed.sampling_incomplete.includes('git_emit_failed'));
} else {
  await emitGitMetadata('alpha', worker, failed, Date.now(), async row => firstEmission.push(row));
  assert(firstEmission.some(row => row.git.sampling_incomplete.length > 0));
}
assert.equal(sampledHeads.get(worker), previous, 'failed sample consumed unread/unemitted commits');
if (failure === 'is_ancestor') assert.equal(fallbackReads, 0, 'read failure was mistaken for rewritten history');
if (failure === 'is_ancestor' || failure === 'merge_base') assert(failed.sampling_incomplete.includes('unreadable_commit_range'));
fail = false;
const next = await worktree(worker, Date.now(), initial);
assert.deepEqual(next.commit_samples.map(row => row.commit), [first, second]);
const rows = [];
await emitGitMetadata('alpha', worker, next, Date.now(), async row => rows.push(row));
assert.deepEqual(rows.map(row => row.git.commit), [first, second]);
if (failure === 'emit') assert(rows.every(row => row.git.sampling_incomplete.includes('git_emit_failed')));
assert.equal(sampledHeads.get(worker), second);
assert.deepEqual((await worktree(worker, Date.now(), initial)).commit_samples, []);
'''.replace('FIXTURE', json.dumps({'repo': ws.repo_path, 'worker': worker, 'failure': failure})))


def test_package_git_emission_waits_for_write_callback_not_backpressure():
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const functions = source.slice(source.indexOf('  function emitCollaboration('), source.indexOf('\n  async function relayStatus('));
const pending = [];
const fakeProcess = {stderr: {write: (line, callback) => { pending.push({line, callback}); return false; }}};
const {emitGitMetadata, sampledHeads} = new Function('process', `const sampledHeads = new Map(); const sampledGitFailures = new Map();
  ${functions}; return {emitGitMetadata, sampledHeads};`)(fakeProcess);
const sha = 'a'.repeat(40), row = {head:sha, sampling_incomplete:[], commit_samples:[]};
let settled = false;
const emitted = emitGitMetadata('alpha', '/work', row, 1).then(() => settled = true);
await new Promise(resolve => setImmediate(resolve));
assert.equal(settled, false);
assert.equal(sampledHeads.has('/work'), false);
pending.shift().callback();
await emitted;
assert.equal(sampledHeads.get('/work'), sha);
const next = {head:'b'.repeat(40), sampling_incomplete:[], commit_samples:[]};
const failed = emitGitMetadata('alpha', '/work', next, 2);
pending.shift().callback(new Error('injected callback failure'));
await assert.rejects(failed, /injected callback failure/);
assert.equal(sampledHeads.get('/work'), sha);
assert(next.sampling_incomplete.includes('git_emit_failed'));
''')


def test_package_git_snapshot_keeps_actor_worktree_binding_across_membership_changes():
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const snapshotSource = source.slice(source.indexOf('  async function snapshot('), source.indexOf('\n  function publish('));
const member = {name:'alpha', agentId:'alpha', cwd:'/alpha-worktree', joinedAt:1};
const team = {teamFile:{name:'fixture', members:[member]}, getAgentFromBackend: () => ({getStatus: () => 'running'})};
const emitted = [];
const snapshot = new Function('env', `const {tools, listTasks, relayToken, commits, repo, worktree,
  startMs, deadlineMs, refreshMs, statusFile, runtimeNote, rolePackage, emitCollaboration, emitGitMetadata,
  agentRules, checkinTotals} = env;
  return (${snapshotSource});`)({tools:new Map(), relayToken:null, repo:'/delivery',
  listTasks: async () => { member.cwd = '/wrong-worktree'; team.teamFile.members = []; return []; },
  commits: async () => ({head:'a'.repeat(40)}),
  worktree: async cwd => {assert.equal(cwd, '/alpha-worktree'); return {head:'b'.repeat(40)};},
  startMs:0, deadlineMs:Date.now() + 60_000, refreshMs:60_000, statusFile:'/status', runtimeNote:null,
  rolePackage:true, agentRules:false, checkinTotals:null, emitCollaboration:async () => {},
  emitGitMetadata:async (actor, cwd, metadata) => emitted.push({actor,cwd,head:metadata.head})});
const view = await snapshot(team, {}, Date.now());
assert.deepEqual(emitted, [
  {actor:'master', cwd:'/delivery', head:'a'.repeat(40)},
  {actor:'alpha', cwd:'/alpha-worktree', head:'b'.repeat(40)},
]);
assert.equal(view.workers[0].name, 'alpha');
assert(!JSON.stringify(view).includes('/alpha-worktree'));
assert(!JSON.stringify(view).includes('/wrong-worktree'));
''')


def test_package_git_untracked_products_do_not_count_as_unfinished_tracked_work(tmp_path):
    ws = workspace(tmp_path)
    run_node(r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const repo = FIXTURE;
const git = async (args, cwd) => execFileSync('git', ['-C', cwd, ...args], {encoding:'utf8'}).trim();
const command = args => execFileSync('git', ['-C', repo, '-c', 'user.name=T', '-c', 'user.email=t@localhost',
  '-c', 'commit.gpgsign=false', ...args], {encoding:'utf8'}).trim();
fs.writeFileSync(path.join(repo, 'tracked.txt'), 'tracked');
command(['add', 'tracked.txt']); command(['commit', '-qm', 'tracked fixture']);
const initial = command(['rev-parse', 'HEAD']);
const source = fs.readFileSync(new URL('./qwen_native_compat.mjs', process.argv[1]), 'utf8');
const functions = source.slice(source.indexOf('  async function commits('), source.indexOf('\n  async function relayStatus('));
const {worktree} = new Function('git', 'baseline', 'path', 'repo', `const rolePackage = true;
  const sampledHeads = new Map(); const sampledGitFailures = new Map(); ${functions}; return {worktree};`)(git, initial, path, repo);
fs.writeFileSync(path.join(repo, 'untracked-build.o'), 'build');
let row = await worktree(repo, Date.now(), initial);
assert.equal(row.uncommitted_paths, 1);
assert.equal(row.tracked_uncommitted_paths, 0);
fs.writeFileSync(path.join(repo, 'tracked.txt'), 'modified');
row = await worktree(repo, Date.now(), initial);
assert.equal(row.uncommitted_paths, 2);
assert.equal(row.tracked_uncommitted_paths, 1);
command(['checkout', '--', 'tracked.txt']);
command(['mv', 'tracked.txt', 'renamed.txt']);
row = await worktree(repo, Date.now(), initial);
assert.equal(row.tracked_uncommitted_paths, 1, 'rename source and destination count as one changed entry');
'''.replace('FIXTURE', json.dumps(ws.repo_path)))


def test_scheduler_sends_periodically_skips_idle_teams_and_replaces_expired_checkins():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler} = await import(process.argv[1]);
const s = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 100_000, milestones: [], minSpacingMs: 100, finalGuardMs: 10});
const active = {active: true, windingDown: false};
assert.equal(s.poll(500, active).action, 'none');
const first = s.poll(1000, active);
assert.deepEqual([first.action, first.reason, first.id], ['send', 'periodic', 'checkin-1']);
// Unread by the master: expiry replaces the queued notification with a fresh
// ID, without claiming that the old notification was observed.
assert.deepEqual(s.poll(2000, active), {action: 'send', reason: 'periodic', milestones: [], id: 'checkin-2', replacesId: 'checkin-1'});
s.observed('checkin-1');
assert.equal(s.pendingId, 'checkin-2');
s.observed('checkin-2');
assert.equal(s.poll(3000, active).action, 'send');
s.observed('checkin-3');
// An idle team has returned control to the master; the tick is consumed.
assert.deepEqual(s.poll(4000, {active: false, windingDown: false}), {action: 'skip', reason: 'no_active_teammates'});
assert.equal(s.poll(4500, active).action, 'none');
''')


def test_milestones_merge_into_a_nearby_checkin_and_wait_for_an_unread_one():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler} = await import(process.argv[1]);
const active = {active: true, windingDown: false};
// Halfway (50 s) lands 5 s before the periodic tick at 55 s: merged.
let s = new CheckinScheduler({intervalMs: 55_000, startMs: 0, deadlineMs: 100_000, milestones: [0.5], minSpacingMs: 10_000, finalGuardMs: 1});
assert.equal(s.poll(50_000, active).action, 'none');
let sent = s.poll(55_000, active);
assert.equal(sent.reason, 'periodic');
assert.deepEqual(sent.milestones, ['half of the task wall time has elapsed']);
// Far from any periodic tick: sent on its own, exactly once.
s = new CheckinScheduler({intervalMs: 90_000, startMs: 0, deadlineMs: 100_000, milestones: [0.5, 0.75], minSpacingMs: 10_000, finalGuardMs: 1});
sent = s.poll(50_000, active);
assert.deepEqual([sent.action, sent.reason, sent.milestones.length], ['send', 'milestone', 1]);
// The last-quarter milestone waits while the halfway check-in is unread.
assert.equal(s.poll(75_000, active).action, 'none');
s.observed(sent.id);
assert.deepEqual(s.poll(76_000, active).milestones, ['the last quarter of the task wall time has started']);
''')


def test_scheduler_stops_for_deadline_but_only_pauses_for_cooperative_shutdown():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler} = await import(process.argv[1]);
let s = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 10_000, milestones: [], finalGuardMs: 2000});
assert.deepEqual(s.poll(8000, {active: true, windingDown: false}), {action: 'stop', reason: 'deadline'});
assert.equal(s.poll(8500, {active: true, windingDown: false}).action, 'none');
s = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 10_000, milestones: [], finalGuardMs: 1});
assert.deepEqual(s.poll(1000, {active: true, windingDown: true}), {action: 'skip', reason: 'shutdown_in_progress'});
assert.equal(s.stopped, null, 'A cooperative shutdown request may be rejected');
// A rejected shutdown, or a newly created replacement, resumes check-ins.
assert.equal(s.poll(2000, {active: true, windingDown: false}).action, 'send');
''')


def test_delayed_snapshot_cannot_send_after_shutdown_deadline_or_cleanup():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler, runCheckin} = await import(process.argv[1]);
for (const ending of ['shutdown', 'deadline', 'cleanup', 'inactive']) {
  const scheduler = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 10_000,
    milestones: [], minSpacingMs: 10, finalGuardMs: 1});
  let clock = 1000;
  let team = {active: true, windingDown: false};
  let release, samplingStarted;
  const sampling = new Promise(resolve => samplingStarted = resolve);
  const sent = [];
  const work = runCheckin({scheduler, now: () => clock, teamState: () => team,
    snapshot: decision => {samplingStarted(); return new Promise(resolve => release = () => resolve({id: decision.id}));},
    publish: () => {}, send: async view => sent.push(view)});
  await sampling;
  if (ending === 'shutdown') team = {active: true, windingDown: true};
  if (ending === 'deadline') clock = 10_000;
  if (ending === 'cleanup') scheduler.stop('cleanup');
  if (ending === 'inactive') team = {active: false, windingDown: false};
  release();
  await work;
  assert.deepEqual(sent, [], ending + ': sampling must not bypass a changed lifecycle');
  assert.equal(scheduler.pendingId, null, ending + ': an unsent message cannot stay pending');
  if (ending === 'shutdown' || ending === 'inactive') {
    // A refused shutdown or a replacement worker does not permanently disable it.
    assert.equal(scheduler.poll(2000, {active: true, windingDown: false}).action, 'send');
  }
}
''')


def test_cancelled_milestone_remains_available_after_shutdown_is_rejected():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler, runCheckin} = await import(process.argv[1]);
const scheduler = new CheckinScheduler({intervalMs: 90_000, startMs: 0, deadlineMs: 100_000,
  milestones: [0.5], minSpacingMs: 10_000, finalGuardMs: 1});
let team = {active: true, windingDown: false};
await runCheckin({scheduler, now: () => 50_000, teamState: () => team,
  snapshot: async decision => {team = {...team, windingDown: true}; return {id: decision.id};},
  publish: () => {}, send: async () => assert.fail('Shutdown became pending during sampling')});
assert.equal(scheduler.pendingId, null);
const resumed = scheduler.poll(51_000, {active: true, windingDown: false});
assert.equal(resumed.action, 'send');
assert.deepEqual(resumed.milestones, ['half of the task wall time has elapsed']);
''')


def test_refresh_publishes_before_first_notification_without_waking_master():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler, runCheckin} = await import(process.argv[1]);
const scheduler = new CheckinScheduler({intervalMs: 1800_000, startMs: 0, deadlineMs: 21600_000,
  milestones: []});
const snapshots = [];
let clock = 1000;
const options = {scheduler, now: () => clock, teamState: () => ({active: true, windingDown: false}),
  refreshDue: true, snapshot: async () => ({at: clock}), publish: view => snapshots.push(view),
  send: async () => assert.fail('Refreshing a status file must not wake the master')};
await runCheckin(options);
clock = 61_000;
await runCheckin(options);
assert.deepEqual(snapshots.map(view => view.at), [1000, 61_000]);
assert.equal(scheduler.pendingId, null);
assert.equal(scheduler.sequence, 0);
''')


def test_failed_snapshot_does_not_leave_an_unread_notification_that_was_never_sent():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler, runCheckin} = await import(process.argv[1]);
const scheduler = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 10_000,
  milestones: [], finalGuardMs: 1});
await assert.rejects(runCheckin({scheduler, now: () => 1000,
  teamState: () => ({active: true, windingDown: false}),
  snapshot: async () => {throw new Error('temporary collector failure');},
  publish: () => {}, send: async () => assert.fail('A failed snapshot cannot be sent')}),
  /temporary collector failure/);
assert.equal(scheduler.pendingId, null);
assert.equal(scheduler.poll(1100, {active: true, windingDown: false}).action, 'send');
''')


def test_checkin_text_reports_facts_and_unknowns_without_verdicts():
    run_node(r'''
import assert from 'node:assert/strict';
const {formatCheckin} = await import(process.argv[1]);
const text = formatCheckin({id: 'checkin-3', reason: 'periodic', milestones: [], elapsed_s: 3600, remaining_s: 18000,
  slots: {used: 3, max: 10}, delivery: {commits_since_launch: 2, last_commit_ago_s: 900, last_commit_subject: 'merge core'},
  unknown: ['relay'], status_file: '/rt/status/team_checkin.json',
  workers: [{name: 'core', status: 'running', last_activity_ago_s: 30, queued_messages: 1, shutdown_pending: false,
             pending_tools: [{name: 'run_shell_command', announced_ago_s: 1200}], tool_counts: {edits: 4, commands: 9},
             joined_ago_s: 7200, git: {commits_since_worker_creation: 2, last_commit_ago_s: 2400, uncommitted_paths: 3, commits_not_in_delivery_history: 1},
             relay: {failed_attempts: 2}, tasks: [{id: '2', status: 'in_progress'}]},
            {name: 'docs', status: 'running', joined_ago_s: 2000, git: {commits_since_worker_creation: 0, uncommitted_paths: 0}}]}, 1800);
assert(text.startsWith('checkin-3: Runtime team check-in (periodic).'));
assert(text.includes('Teammate slots used 3/10'));
assert(text.includes('no new worktree commit for 40m'));
assert(text.includes('docs: running') && text.includes("no commit since this worker's worktree was created; worker joined 33m ago"));
assert(text.includes('Delivery repository: 2 new commit(s) since launch; last 15m ago ("merge core")'));
assert(text.includes('run_shell_command awaiting result (20m since announcement)'));
assert(text.includes('run_shell_command announced 20m ago; no result observed yet (execution may not have started)'));
assert(!text.includes('running for'));
assert(text.includes('1') && text.includes('cherry-pick'));
assert(!text.includes('not yet in the delivery repository'));
assert(text.includes('2 failed model request attempt(s)'));
assert(text.includes('Unavailable sources: relay'));
assert(!/\b(stalled|stuck|idle worker|should be replaced)\b/i.test(text.replace('not a verdict that anyone is stuck', '')));
''')


def test_checkin_text_distinguishes_active_count_from_unknown_count():
    run_node(r'''
import assert from 'node:assert/strict';
const {formatCheckin} = await import(process.argv[1]);
const snapshot = {id: 'checkin-1', reason: 'periodic', elapsed_s: 60, remaining_s: 600,
  slots: {used: 10, max: 10}, workers: [], unknown: [], delivery: {}};
const text = formatCheckin(snapshot, 60);
assert(text.includes('Teammate slots used 10/10.'));
assert(text.includes('shut-down teammates remain in the roster but do not count toward this limit.'));
assert(!text.includes('12/null') && !text.includes('Infinity'));
const unknown = formatCheckin({...snapshot, slots: {used: 12, max: null}}, 60);
assert(!unknown.includes('no fixed teammate-count limit'));
''')


def test_checkin_text_distinguishes_unavailable_sources_from_observed_empty_sources():
    run_node(r'''
import assert from 'node:assert/strict';
const {formatCheckin} = await import(process.argv[1]);
const text = formatCheckin({id: 'checkin-1', reason: 'periodic', elapsed_s: 60, remaining_s: 3600,
  slots: {used: 2, max: 10}, unknown: ['task_board'], workers: [
    {name: 'missing', status: 'running', tool_counts: null, tasks: null, git: null, pending_tools: null},
    {name: 'observed', status: 'running', tool_counts: {calls: 0, edits: 0, commands: 0}, tasks: [], git: {}, pending_tools: []},
  ]}, 1800);
const missing = text.split('\n').find(line => line.startsWith('- missing:'));
const observed = text.split('\n').find(line => line.startsWith('- observed:'));
assert(/tool: unknown/.test(missing), missing);
assert(/tasks: unknown/.test(missing), missing);
assert(!/no observed tool call yet|tasks: none/.test(missing), missing);
assert(/no observed tool call yet/.test(observed), observed);
assert(/tasks: none/.test(observed), observed);
''')


def test_relay_team_status_is_authenticated_team_only_accounting(tmp_path):
    ws = workspace(tmp_path)
    registry = WorkerRegistry(ws)
    spawn = response([call("agent", {"name": "alpha", "prompt": "Implement alpha"})],
                     {"prompt_tokens": 10, "completion_tokens": 5})
    replies = iter([(200, spawn), (200, response([], {"prompt_tokens": 7, "completion_tokens": 3})),
                    (200, response([])), (525, {"error": {"message": "scripted provider error"}})])
    relay = TeamRelay(ProviderConfig("model", "http://localhost", max_attempts=1), registry, "policy", tmp_path / "logs", 60,
                      lambda _: next(replies))
    assert relay.handle(request())[0] == 200
    assert relay.handle(request("alpha"))[0] == 200
    assert relay.handle(request("alpha"))[0] == 200
    assert relay.handle(request("alpha"))[0] >= 400
    status_view = relay.team_status()
    assert status_view["workers"] == registry.metadata()
    assert status_view["workers"]["alpha"]["creation_head"] == ws.read_head(ws.repo_path)
    status = status_view["roles"]
    assert status["master"]["attempts"] == 1 and status["master"]["known_tokens"] == 15
    assert status["alpha"]["attempts"] == 3 and status["alpha"]["known_tokens"] == 10
    assert status["alpha"]["failed_attempts"] == 1 and "errors" not in status["alpha"]
    assert status["alpha"]["unknown_usage_attempts"] == 2 and status["alpha"]["in_flight_s"] is None
    server = start_server(relay)
    url = f"http://127.0.0.1:{server.server_port}/v1/agent-company/team-status"
    try:
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(url, timeout=5)
        assert denied.value.code == 403
        reply = urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + relay.token}),
                                       timeout=5)
        assert json.loads(reply.read())["roles"]["alpha"]["attempts"] == 3
    finally:
        server.shutdown()
        server.server_close()


def test_master_prompt_is_unchanged_unless_checkins_are_enabled():
    assert "runtime:checkin" not in master_system("A", "C")
    with pytest.raises(ValueError):
        QwenTeamRunner("cli.js", checkin_interval_seconds=0)


def test_shared_checkin_setting_does_not_enable_notifications_or_status_on_single(tmp_path):
    from test_qwen import SingleAgentScenario

    default = Path(__file__).parents[1] / "external/qwen-study/package/cli.js"
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(default)))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Set QWEN_CODE_CLI and QWEN_NODE to execute the native check-in integration test")
    ws = workspace(tmp_path)
    scenario = SingleAgentScenario(ws)
    output = tmp_path / "single"
    result = QwenTeamRunner(str(cli.resolve()), node=node, upstream=scenario, checkin_interval_seconds=1).run(
        ws, "Create single.txt and verify it.", "", "", output,
        ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=40),
        timeout_seconds=90, execution_mode="single", max_model_calls=None)
    stderr = (output / "qwen_stderr.txt").read_text()
    assert result["execution"]["returncode"] == 0, stderr
    assert scenario.steps == 5 and scenario.observations == ["read_original", "write_readback", "shell_verified"]
    launch = json.loads((output / "launch.json").read_text())
    assert launch["team_checkin"] is None and launch["max_model_calls"] is None
    assert "AGENT_COMPANY_CHECKIN_SECONDS" not in launch["environment"]
    assert "--include-directories" not in launch["command"]
    assert "checkin_enabled" not in stderr and "checkin_sent" not in stderr
    assert "runtime:checkin" not in json.dumps(launch)
    assert not list(Path(result["runtime_dir"]).rglob("team_checkin.json"))
    assert result["workers"] == {}


def test_native_checkins_reach_the_master_and_stop_at_shutdown(tmp_path):
    default = Path(__file__).parents[1] / "external/qwen-study/package/cli.js"
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(default)))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Set QWEN_CODE_CLI and QWEN_NODE to execute the native check-in integration test")
    ws = workspace(tmp_path)
    holder = {}
    scenario = CheckinScenario(ws, holder)
    runner = QwenTeamRunner(str(cli.resolve()), node=node, upstream=scenario, checkin_interval_seconds=1)
    original = runner.run

    # The status file path is chosen at launch; expose it to the scripted master.
    def run(*args, **kwargs):
        import agent_company.runtime.qwen as module
        real = module.checkin_system

        def capture(interval, status_file):
            holder["path"] = status_file
            return real(interval, status_file)
        module.checkin_system = capture
        try:
            return original(*args, **kwargs)
        finally:
            module.checkin_system = real
    result = run(ws, "Build alpha.txt with one slow worker.", "Assign one worker.", "Message only on decisions.",
                 tmp_path / "run", ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=40),
                 timeout_seconds=90)
    stderr = (tmp_path / "run/qwen_stderr.txt").read_text()
    assert result["execution"]["returncode"] == 0, stderr
    events = [json.loads(line.split("Agent Company lifecycle: ", 1)[1]) for line in stderr.splitlines()
              if line.startswith("Agent Company lifecycle: ")]
    names = [event["event"] for event in events]
    assert "checkin_enabled" in names and "checkin_error" not in names, stderr
    assert "agent_start_patch_applied" in names, stderr
    sent = [event for event in events if event["event"] == "checkin_sent"]
    # Each sent snapshot survives in the lifecycle log (the status file is overwritten).
    snapshots = [event["snapshot"] for event in sent]
    assert all({"sample_started_at", "at", "elapsed_s", "remaining_s", "delivery", "workers", "unknown"} <= set(item)
               for item in snapshots)
    assert all(item["sample_started_at"] <= item["at"] for item in snapshots)
    assert any(worker["name"] == "alpha" for item in snapshots for worker in item["workers"])
    assert "subject" not in json.dumps(snapshots)
    trial = tmp_path / "trial"
    (trial / "team/relay").mkdir(parents=True)
    (trial / "team/qwen_stderr.txt").write_text(stderr)
    projected = project_evidence(trial)["checkins"]
    assert projected["status"] == "recorded" and projected["snapshots_status"] == "recorded_for_sent_checkins"
    assert projected["sent"] == len(sent) and len(projected["snapshots"]) == len(sent)
    assert any(worker["name"] == "alpha" for item in projected["snapshots"] for worker in item["workers"])
    observed = {event["checkin_id"] for event in events if event["event"] == "checkin_observed"}
    # A busy master may never see an older queued snapshot that was replaced;
    # supersession accounts for it but must not be reported as observation.
    superseded = {event["supersedes_id"] for event in sent if "supersedes_id" in event}
    assert sent and {event["checkin_id"] for event in sent} <= observed | superseded | {sent[-1]["checkin_id"]}
    # The master actually saw the observational snapshot, including the slow tool.
    seen = "\n".join(scenario.checkin_texts)
    assert "Runtime team check-in" in seen and "Teammate slots used 1/10." in seen
    assert "- alpha: running" in seen and "run_shell_command awaiting result" in seen
    assert scenario.read_status == "read", "The master could not read the status file on demand"
    status = json.loads(Path(holder["path"]).read_text())
    assert status["slots"]["max"] == 10
    assert status["workers"][0]["name"] == "alpha" and "relay" not in status["unknown"]
    assert status["workers"][0]["relay"]["attempts"] >= 1
    assert not str(holder["path"]).startswith(str(ws.repo_path))
    assert not list(Path(ws.repo_path).rglob("team_checkin.json"))
    # No check-in after the shutdown decision, and the team still finished.
    assert scenario.shutdown_sent, "The master never reached its shutdown decision"
    assert not any(event["at"] > scenario.shutdown_sent for event in sent)
    launch = json.loads((tmp_path / "run/launch.json").read_text())
    assert launch["teammate_limit"] == 10
    assert launch["team_checkin"]["interval_seconds"] == 1
    assert "--include-directories" in launch["command"]
    assert (Path(ws.repo_path) / "alpha.txt").exists() is False  # Worker committed in its own worktree only.


class EarlyStatusScenario:
    """Read the file before any tick and observe a real cherry-pick without a wake-up."""

    def __init__(self, ws, status_file_holder):
        self.ws = ws
        self.holder = status_file_holder
        self.stage = "initial"
        self.worker_started = False
        self.worker_done = threading.Event()
        self.initial_status = None
        self.refreshed_status = None
        self.responses = 0

    def __call__(self, payload):
        self.responses += 1
        if self.responses > 50:
            raise RuntimeError("Early-status native fixture did not converge")
        role = role_of(payload)
        latest = payload["messages"][-1]
        latest_text = str(latest.get("content", ""))
        calls, content = [], ""
        if role != "master":
            users = [str(m.get("content", "")) for m in payload["messages"] if m.get("role") == "user"]
            if "shutdown" in (users[-1] if users else "").lower():
                calls = [call("send_message", {"to": "leader", "message": "shutdown_approved"})]
            elif not self.worker_started:
                self.worker_started = True
                delivery = shlex.quote(str(self.ws.repo_path))
                git_config = "-c user.name=T -c user.email=t@localhost -c commit.gpgsign=false"
                command = (
                    "python3 -c 'import time; time.sleep(4)' && printf alpha > alpha.txt && git add alpha.txt && "
                    f"git {git_config} commit -qm alpha && "
                    f"git -C {delivery} {git_config} commit --allow-empty -qm integration && "
                    f"git -C {delivery} {git_config} cherry-pick "
                    '"$(git rev-parse HEAD)" && echo BUILT'
                )
                calls = [call("run_shell_command", {"description": "Create and integrate a scripted commit", "command": command})]
            else:
                assert "BUILT" in json.dumps(payload["messages"]), latest_text
                self.worker_done.set()
                content = "ALPHA_DONE: alpha.txt was committed and cherry-picked into delivery."
        elif self.stage == "initial":
            # This is the master's first model request, before team_create.
            self.initial_status = json.loads(Path(self.holder["path"]).read_text())
            calls = [call("read_file", {"file_path": self.holder["path"]})]
            self.stage = "initial_read"
        elif self.stage == "initial_read":
            assert latest.get("role") == "tool" and '"workers"' in latest_text, latest_text
            calls = [call("team_create", {"team_name": "early-status-" + uuid.uuid4().hex[:6]})]
            self.stage = "created"
        elif self.stage == "created":
            calls = [call("agent", {"name": "alpha", "description": "Implement and integrate alpha",
                                    "prompt": "Create alpha.txt in your worktree, commit and integrate it into delivery."})]
            self.stage = "working"
        elif self.stage == "working" and self.worker_done.is_set():
            # A file refresh must occur independently of the 20-second wake-up.
            until = time.monotonic() + 7
            while time.monotonic() < until:
                view = json.loads(Path(self.holder["path"]).read_text())
                if view.get("delivery", {}).get("commits_since_launch") == 2:
                    self.refreshed_status = view
                    break
                time.sleep(0.05)
            assert self.refreshed_status, "The independent sampler did not refresh the completed integration"
            calls = [call("read_file", {"file_path": self.holder["path"]})]
            self.stage = "refreshed_read"
        elif self.stage == "refreshed_read":
            assert latest.get("role") == "tool" and '"commits_not_in_delivery_history"' in latest_text, latest_text
            calls = [call("request_shutdown", {"to": "alpha"})]
            self.stage = "shutdown"
        elif self.stage == "shutdown":
            content = "Alpha was integrated and the team is shut down."
        else:
            content = "Waiting for alpha."
        return 200, {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": payload["model"],
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "tool_calls": calls},
                                  "finish_reason": "tool_calls" if calls else "stop"}],
                     "usage": {"prompt_tokens": 50, "completion_tokens": 5}}


def test_native_status_is_readable_before_first_checkin_and_cherrypick_is_not_reported_unlanded(tmp_path, monkeypatch):
    default = Path(__file__).parents[1] / "external/qwen-study/package/cli.js"
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(default)))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Set QWEN_CODE_CLI and QWEN_NODE to execute the native check-in integration test")
    import agent_company.runtime.qwen as module
    ws = workspace(tmp_path)
    holder = {}
    real = module.checkin_system

    def capture(interval, status_file):
        holder["path"] = status_file
        return real(interval, status_file)

    monkeypatch.setattr(module, "checkin_system", capture)
    scenario = EarlyStatusScenario(ws, holder)
    runner = QwenTeamRunner(str(cli.resolve()), node=node, upstream=scenario, checkin_interval_seconds=20)
    result = runner.run(ws, "Create and integrate alpha.txt.", "Assign one worker.", "Message only on decisions.",
                        tmp_path / "run", ProviderConfig("native-scripted-model", "http://localhost", request_timeout_seconds=40),
                        timeout_seconds=90)
    stderr = (tmp_path / "run/qwen_stderr.txt").read_text()
    assert result["execution"]["returncode"] == 0, stderr
    assert scenario.stage == "shutdown", stderr
    assert scenario.initial_status["workers"] == []
    assert scenario.initial_status["at"] < scenario.refreshed_status["at"]
    worker = scenario.refreshed_status["workers"][0]
    assert worker["git"]["commits_not_in_delivery_history"] == 1
    assert "unlanded_commits" not in worker["git"]
    # An empty integration commit makes the cherry-pick's SHA different, while
    # its resulting tree is exactly the worker tree. History inequality is not
    # missing code; the user-visible formatter must retain that distinction.
    delivery_tree = subprocess.check_output(["git", "-C", str(ws.repo_path), "rev-parse", "HEAD^{tree}"], text=True).strip()
    worker_tree = subprocess.check_output(["git", "-C", str(ws.repo_path), "rev-parse", worker["git"]["head"] + "^{tree}"], text=True).strip()
    assert delivery_tree == worker_tree
    assert (Path(ws.repo_path) / "alpha.txt").read_text() == "alpha"
    assert '"event":"checkin_sent"' not in stderr, "File refresh must not require a model wake-up"
    assert '"event":"checkin_error"' not in stderr, stderr


def test_container_memory_is_read_from_the_own_cgroup_and_shown_only_in_agent_rules_checkins(tmp_path):
    proc, cgroup = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/task\n")
    (cgroup / "task").mkdir(parents=True)
    for name, value in (("memory.current", str(31 * 2**30)), ("memory.peak", str(44 * 2**30)),
                        ("memory.max", str(80 * 2**30)), ("memory.events", "low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n")):
        (cgroup / "task" / name).write_text(value)
    run_node(r'''
import assert from 'node:assert/strict';
const {containerMemory, memoryLine, formatCheckin, compactCheckin} = await import(process.argv[1]);
const memory = containerMemory(''' + json.dumps(str(proc)) + ", " + json.dumps(str(cgroup)) + r''');
assert.deepEqual(memory, {current_bytes: 31 * 2 ** 30, peak_bytes: 44 * 2 ** 30, limit_bytes: 80 * 2 ** 30, oom_kills: 1});
assert.equal(memoryLine(memory),
  'Container memory: 31.0 GiB in use of 80.0 GiB, peak 44.0 GiB; processes killed for lack of memory so far: 1.');
assert.deepEqual(containerMemory('/nonexistent', '/nonexistent'),
  {current_bytes: null, peak_bytes: null, limit_bytes: null, oom_kills: null});
assert.equal(memoryLine(containerMemory('/nonexistent')),
  'Container memory: unknown in use of unknown, peak unknown; processes killed for lack of memory so far: unknown.');
const view = {id: 'checkin-1', reason: 'periodic', elapsed_s: 60, remaining_s: 600, slots: {used: 1, max: 16},
  delivery: {commits_since_launch: 0}, workers: [], unknown: []};
assert(!formatCheckin(view, 1800).includes('Container memory'));  // earlier protocols unchanged
const withMemory = formatCheckin({...view, container_memory: memory, team_observations: null}, 1800);
assert(withMemory.split('\n').includes(memoryLine(memory)));
assert.deepEqual(compactCheckin({...view, container_memory: memory}).container_memory, memory);
assert(!('container_memory' in compactCheckin(view)));
''')
