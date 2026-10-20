"""agent_rules_v1 behavior inside the native compat layer, using the actual source (zero API)."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'agent_company/runtime'
SOURCE = (RUNTIME / 'qwen_native_compat.mjs').read_text()


def node():
    path = shutil.which('node')
    if not path:
        pytest.skip('Node is needed for the native compat source')
    return path


def run(script):
    # On stdin: a script that inlines the compat source exceeds Linux's 128 KiB per-argument limit.
    result = subprocess.run([node(), '--input-type=module'], input=script, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_any_package_protocol_turns_on_fine_grained_samples():
    assert 'const rolePackage = rolePackageV1 || agentRules;' in SOURCE
    sampling = SOURCE[SOURCE.index('  async function commits('):SOURCE.index('\n  async function relayStatus(')]
    assert sampling.count('if (rolePackage) {') == 2
    snapshot = SOURCE[SOURCE.index('const completedMs = Date.now();'):]
    assert snapshot.split('\n', 2)[1].strip() == 'if (rolePackage) {'
    # Behavior that belongs to role_package_v1 alone never follows the new protocol.
    assert 'rolePackageV1 ? PACKAGE_PROPOSAL_HEADER' in SOURCE
    if 'const roleTarget' in SOURCE:  # role-target broadcast metadata, when present
        assert 'const roleTarget = rolePackageV1 ?' in SOURCE


@pytest.mark.parametrize('agent_rules', [True, False])
def test_agent_rules_checkins_have_no_policy_review(agent_rules):
    package = re.search(r'  function packageCheckin\(view\) \{.*?\n  \}', SOURCE, re.S).group(0)
    dispatch = re.search(r'  const checkinText = .*?;\n', SOURCE).group(0)
    view = {'id': 'checkin-1', 'reason': 'periodic', 'elapsed_s': 100, 'remaining_s': 200,
            'slots': {'used': 1, 'max': 10}, 'workers': [], 'delivery': {}, 'unknown': []}
    script = (f'import {{formatCheckin}} from {json.dumps((RUNTIME / "qwen_team_checkin.mjs").as_uri())};\n'
              f'const rolePackage=true,agentRules={json.dumps(agent_rules)},rolePolicyMode="propose",checkinSeconds=1800;\n'
              + dispatch + package + f'\nconst view={json.dumps(view)};\n'
              'console.log(JSON.stringify({text:checkinText(view),plain:formatCheckin(view,1800)}));')
    value = run(script)
    if agent_rules:
        assert value['text'] == value['plain'] and 'POLICY-REVIEW' not in value['text']
    else:
        assert 'POLICY-REVIEW' in value['text']


def test_worker_controls_to_leader_are_not_task_reports():
    line = re.search(r'^const AGENT_RULES_CONTROL = .*;$', SOURCE, re.M).group(0)
    samples = {'POLICY-PROPOSAL self add': True, 'POLICY-PROPOSAL integ replace R3': True,
               'POLICY-PROPOSAL * remove R12': True, 'POLICY-PROPOSAL self notes': True, 'LESSON: commit first': True,
               'POLICY-PROPOSAL role:implementer append': False, 'Lesson: x': False, 'Done with the reader.': False,
               'POLICY-PROPOSAL self add extra': False, 'BOARD: interface fixed': False, 'Board: x': False}
    value = run(line + f'\nconsole.log(JSON.stringify(Object.fromEntries({json.dumps(list(samples))}'
                '.map(text => [text, AGENT_RULES_CONTROL.test(text)]))));')
    assert value == samples


def test_new_commit_rows_carry_their_own_paths_and_time(tmp_path):
    from test_team import workspace
    repo = workspace(tmp_path).repo_path
    script = '''
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const repo = REPO;
const command = args => execFileSync('git', ['-C', repo, '-c', 'user.name=T', '-c', 'user.email=t@localhost',
  '-c', 'commit.gpgsign=false', ...args], {encoding:'utf8'}).trim();
const initial = command(['rev-parse', 'HEAD']);
const functions = SOURCE.slice(SOURCE.indexOf('  async function commits('), SOURCE.indexOf('\\n  async function relayStatus('));
const git = async (args, cwd) => execFileSync('git', ['-C', cwd, ...args], {encoding:'utf8'}).trim();
const {commits, emitGitMetadata} = new Function('git', 'baseline', 'path', 'repo',
  `const rolePackage = true; const sampledHeads = new Map(); const sampledGitFailures = new Map(); ${functions}; return {commits, emitGitMetadata};`)(git, initial, path, repo);
const rows = [];
const collect = async () => { const result = await commits(repo, Date.now()); await emitGitMetadata('alpha', repo, result, 1, async row => rows.push(row.git)); };
await collect();
fs.writeFileSync(path.join(repo, 'a.txt'), 'a'); command(['add', 'a.txt']); command(['commit', '-qm', 'a']);
fs.mkdirSync(path.join(repo, 'src')); fs.writeFileSync(path.join(repo, 'src/b.txt'), 'b');
command(['add', 'src/b.txt']); command(['commit', '-qm', 'b']);
await collect();
console.log(JSON.stringify(rows.map(r => ({commit: r.commit, new_commit: r.new_commit, changed_paths: r.changed_paths,
  timed: typeof r.committed_at === 'number' && r.committed_at > 0}))));
'''.replace('REPO', json.dumps(repo)).replace('SOURCE', json.dumps(SOURCE))
    rows = run(script)
    assert [r['new_commit'] for r in rows] == [False, True, True]
    assert [r['changed_paths'] for r in rows] == [[], ['a.txt'], ['src/b.txt']]
    assert [r['timed'] for r in rows] == [False, True, True]


def test_control_messages_are_marked_and_auto_claim_is_off():
    line = re.search(r'^const AGENT_RULES_ANY_CONTROL = .*;$', SOURCE, re.M).group(0)
    samples = {'AGENT-POLICY alpha add': True, 'POLICY-DECISION #3 approve': True, 'POLICY-PROPOSAL self add': True,
               'LESSON: x': True, 'Please review the reader.': False, 'AGENT-POLICYX': False,
               'BOARD: x': True, 'Board: x': False, 'See the BOARD: x': False}
    value = run(line + f'\nconsole.log(JSON.stringify(Object.fromEntries({json.dumps(list(samples))}'
                '.map(text => [text, AGENT_RULES_ANY_CONTROL.test(text)]))));')
    assert value == samples
    assert '...(agentRules ? { control: AGENT_RULES_ANY_CONTROL.test(firstLine(item.text)) } : {})' in SOURCE
    block = SOURCE[SOURCE.index('if (agentRules) {\n  // agent_rules_v1 (plan v4 section 10)'):]
    assert 'manager.prototype.tryAutoClaimTask = async function () {};' in block.split('\n}\n', 1)[0]


def test_message_hash_matches_the_relay_and_shutdown_text_is_recognized():
    from agent_company.runtime.rule_engine import message_hash
    function = re.search(r'^function messageHash\(message\) \{.*?\n\}', SOURCE, re.S | re.M).group(0)
    pattern = re.search(r'^const NATIVE_SHUTDOWN_TEXT = .*;$', SOURCE, re.M).group(0)
    samples = ['SHA: ab12cd34', '中文 汇报\nPATHS: src/a.hs', 'emoji 🙂 and tabs\t', '']
    native = ('The team leader has requested that you shut down. Please finish your current work and use '
              'send_message to reply to "leader" with either "shutdown_approved" or "shutdown_rejected: <reason>".')
    value = run("import {createHash} from 'node:crypto';\n" + function + '\n' + pattern +
                f'\nconsole.log(JSON.stringify({{hashes: {json.dumps(samples)}.map(messageHash), '
                f'shutdown: NATIVE_SHUTDOWN_TEXT.test({json.dumps(native)}), other: NATIVE_SHUTDOWN_TEXT.test("Please shut down")}}));')
    assert value['hashes'] == [message_hash(sample) for sample in samples]
    assert value['shutdown'] is True and value['other'] is False


def test_new_native_records_exist_only_for_agent_rules():
    send = SOURCE[SOURCE.index('  TeamManager.prototype.sendMessage = async function'):]
    send = send[:send.index('\n  };\n')]
    assert "if (!agentRules || automatic || broadcastInner.consume(from, message)) {" in send
    assert "const inner = agentRules ? broadcastInner.begin(from, message, recipients.length) : null;" in SOURCE
    assert 'AsyncLocalStorage()' not in SOURCE.split('const broadcastInner')[0].split('NATIVE_SHUTDOWN_TEXT')[1]
    assert "record('team_message_sent', { ...base, ok: true });" in send
    enqueue = SOURCE[SOURCE.index('  TeamManager.prototype.enqueueWithIdentity = function'):]
    assert enqueue.split('\n')[1].strip().startswith('if (agentRules && typeof message === ')
    assert "...(agentRules ? { hash: messageHash(message) } : {})" in SOURCE


def test_broadcast_inner_sends_are_counted_per_sender_and_content():
    """M2 review N8: a count, not an async context, so woken rounds never inherit it."""
    function = re.search(r'^function messageHash\(message\) \{.*?\n\}', SOURCE, re.S | re.M).group(0)
    counter = re.search(r'^const broadcastInner = \{.*?\n\};', SOURCE, re.S | re.M).group(0)
    value = run("import {createHash} from 'node:crypto';\nconst records = []; const record = (event, data) => records.push([event, data]);\n"
                + function + '\n' + counter + '''
const leftover = broadcastInner.begin('x', 'y', 2); broadcastInner.consume('x', 'y'); broadcastInner.end(leftover);
const key = broadcastInner.begin('Alpha', 'hi', 2);
const seen = [broadcastInner.consume('alpha', 'hi'), broadcastInner.consume('beta', 'hi'),
              broadcastInner.consume('alpha', 'other'), broadcastInner.consume('alpha', 'hi'), broadcastInner.consume('alpha', 'hi')];
broadcastInner.begin('alpha', 'again', 3); broadcastInner.end(key);
console.log(JSON.stringify({seen, left: [...broadcastInner.pending.values()], records}));''')
    assert value == {'seen': [True, False, False, True, False], 'left': [3],
                     'records': [['broadcast_inner_count_mismatch', {'left': 1}]]}


def test_merged_commits_are_new_rows_with_their_own_paths(tmp_path):
    from test_team import workspace
    repo = workspace(tmp_path).repo_path
    script = '''
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const repo = REPO;
const command = args => execFileSync('git', ['-C', repo, '-c', 'user.name=T', '-c', 'user.email=t@localhost',
  '-c', 'commit.gpgsign=false', ...args], {encoding:'utf8'}).trim();
const initial = command(['rev-parse', 'HEAD']);
const main = command(['branch', '--show-current']);
const functions = SOURCE.slice(SOURCE.indexOf('  async function commits('), SOURCE.indexOf('\\n  async function relayStatus('));
const git = async (args, cwd) => execFileSync('git', ['-C', cwd, ...args], {encoding:'utf8'}).trim();
const {commits, emitGitMetadata} = new Function('git', 'baseline', 'path', 'repo',
  `const rolePackage = true; const sampledHeads = new Map(); const sampledGitFailures = new Map(); ${functions}; return {commits, emitGitMetadata};`)(git, initial, path, repo);
const rows = [];
const collect = async () => { const result = await commits(repo, Date.now()); await emitGitMetadata('alpha', repo, result, 1, async row => rows.push(row.git)); };
await collect();
command(['checkout', '-qb', 'side']); fs.writeFileSync(path.join(repo, 'side.txt'), 's'); command(['add', 'side.txt']); command(['commit', '-qm', 'side']);
command(['checkout', '-q', main]); fs.writeFileSync(path.join(repo, 'main.txt'), 'm'); command(['add', 'main.txt']); command(['commit', '-qm', 'main']);
command(['merge', '-q', '--no-edit', 'side']);
const merged = command(['rev-parse', 'HEAD']);
await collect();
console.log(JSON.stringify({merged, rows: rows.slice(1).map(r => ({commit: r.commit, new_commit: r.new_commit, paths: r.changed_paths}))}));
'''.replace('REPO', json.dumps(repo)).replace('SOURCE', json.dumps(SOURCE))
    value = run(script)
    rows = value['rows']
    assert len(rows) == 3 and all(r['new_commit'] for r in rows)
    merge = next(r for r in rows if r['commit'] == value['merged'])
    assert set(merge['paths']) == {'side.txt', 'main.txt'}


def test_agent_rules_checkins_add_collaboration_observations():
    base = {'id': 'checkin-2', 'reason': 'periodic', 'elapsed_s': 100, 'remaining_s': 200,
            'slots': {'used': 2, 'max': 10}, 'workers': [], 'delivery': {}, 'unknown': []}
    observed = {'pending_shutdowns': [{'worker': 'alpha', 'requested_ago_s': 420}],
                'idle_workers': [], 'message_pairs': [{'from': 'alpha', 'to': 'beta', 'count': 5}],
                'rule_rates': {'alpha': {'satisfied': 3, 'missed': 1, 'open': 2}, 'beta': {'satisfied': 0, 'missed': 0}},
                'pending_proposals': 1, 'recent_self_changes': 2, 'lessons_total': 4, 'lessons_new': 1,
                'board_total': 6, 'board_new': 3}
    script = (f'import {{formatCheckin}} from {json.dumps((RUNTIME / "qwen_team_checkin.mjs").as_uri())};\n'
              f'const base={json.dumps(base)};\n'
              f'console.log(JSON.stringify({{plain:formatCheckin(base,1800),'
              f'rules:formatCheckin({{...base, team_observations:{json.dumps(observed)}}},1800),'
              f'missing:formatCheckin({{...base, team_observations:null}},1800)}}));')
    value = run(script)
    assert 'Collaboration observations' not in value['plain'] and 'AGENT-POLICY' not in value['plain']
    rules = value['rules']
    for line in ('- Shutdown requested over 5 minutes ago and not finished: alpha (7m).',
                 '- Idle over 15 minutes with no in-progress task: none.',
                 '- Messages between teammates, most frequent pairs (counts only): alpha->beta 5.',
                 '- Rule obligations met so far (met/decided; not-yet-decidable ones excluded): alpha 3/4 (2 open), beta none decided.',
                 '- Policy proposals waiting for you: 1; recent changes teammates applied to their own policies: 2',
                 '- Lessons: 4 in total, 1 new since the previous check-in. Board posts: 6 in total, 3 new since the previous check-in.'):
        assert line in rules
    assert rules.endswith("also decide whether to change any teammate's policy with AGENT-POLICY. No reply is needed if nothing changes.")
    assert 'Collaboration observations: unavailable' in value['missing']


def test_team_samples_and_checkin_baselines_are_agent_rules_only():
    snapshot = SOURCE[SOURCE.index('  async function snapshot('):SOURCE.index('\n  function publish(')]
    block = snapshot[snapshot.index('if (agentRules && team) {'):]
    assert "kind: 'team'" in block and "task.status === 'in_progress'" in block and '.slice(0, 60)' in block
    assert '...(agentRules ? { team_observations: relay ? {' in snapshot
    poll = SOURCE[SOURCE.index('  async function poll('):]
    assert poll.index('if (view.team_observations) checkinTotals.sent(view.id, view.team_observations);') < poll.index('return callback(')
    observe = SOURCE[SOURCE.index('    observe(text) {'):]
    assert observe.index('scheduler.observed(id);') < observe.index('if (agentRules) checkinTotals.observed(id);')
    assert "tasks: worker.tasks == null ? null :" in block


def test_checkin_totals_move_only_when_the_master_sees_a_checkin():
    script = (f'import {{CheckinTotals}} from {json.dumps((RUNTIME / "qwen_team_checkin.mjs").as_uri())};\n'
              'const totals = new CheckinTotals(); const out = [];\n'
              'out.push(totals.newSince({lessons_total: 2, board_total: 5}));\n'
              'totals.sent("checkin-1", {lessons_total: 2, board_total: 5});\n'
              '// checkin-1 is replaced by checkin-2 before the master reads it\n'
              'totals.sent("checkin-2", {lessons_total: 3, board_total: 7});\n'
              'out.push(totals.newSince({lessons_total: 3, board_total: 7}));\n'
              'totals.observed("checkin-2");\n'
              'out.push(totals.newSince({lessons_total: 4, board_total: 7}));\n'
              'totals.observed("checkin-9");\n'
              'out.push(totals.newSince({lessons_total: null}));\n'
              'console.log(JSON.stringify(out));')
    assert run(script) == [{'lessons_new': 2, 'board_new': 5}, {'lessons_new': 3, 'board_new': 7},
                           {'lessons_new': 1, 'board_new': 0}, {'lessons_new': None, 'board_new': None}]
