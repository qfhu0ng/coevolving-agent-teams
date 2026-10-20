"""Native Qwen non-PTY shell output must drain before its result is settled."""

import os
from pathlib import Path
import subprocess

import pytest

from test_rejection_integration import native_qwen


ROOT = Path(__file__).parents[1]
COMPAT = ROOT / "agent_company/runtime/qwen_native_compat.mjs"
PATCH = ROOT / "agent_company/runtime/qwen_native_patch.mjs"


def native_script(script, *arguments):
    cli, node = native_qwen()
    return subprocess.run(
        [node, "--import", str(COMPAT), "--input-type=module", "-e", script,
         str(cli.parent / "chunks"), *arguments],
        check=True, capture_output=True, text=True, timeout=30,
        env={"PATH": os.environ.get("PATH", os.defpath), "LANG": "en_US.UTF-8",
             "AGENT_COMPANY_QWEN_CLI": str(cli), "AGENT_COMPANY_ROOT_OWNER": "single",
             "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:9/v1"},
    )


NATIVE_SERVICE = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const chunks = process.argv[1];
const file = fs.readdirSync(chunks).find(name => name.endsWith('.js') &&
  fs.readFileSync(path.join(chunks, name), 'utf8').includes('var ShellExecutionService = class'));
const {ShellExecutionService: service} = await import(pathToFileURL(path.join(chunks, file)).href);
'''


@pytest.mark.parametrize("scenario", [
    "exit_data_close", "cancel", "deadline", "spawn_error", "background", "background_after_exit",
    "cancel_after_exit", "exit_no_close", "pre_aborted",
])
def test_native_fallback_drains_late_streams_and_preserves_other_endings(scenario):
    # Only cp.spawn and process.kill are faked. The actual installed service is
    # loaded through our normal compat hook, including its decoder and cleanup.
    native_script(NATIVE_SERVICE + r'''
import cp from 'node:child_process';
import {EventEmitter, getEventListeners} from 'node:events';
import {PassThrough} from 'node:stream';
import {syncBuiltinESMExports} from 'node:module';
const scenario = process.argv[2];
const children = [];
const kills = [];
cp.spawn = () => {
  const child = new EventEmitter();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.pid = scenario === 'spawn_error' ? undefined : 424242;
  child.exitCode = null;
  child.signalCode = null;
  child.kill = signal => { kills.push(['child', signal]); return true; };
  children.push(child);
  return child;
};
syncBuiltinESMExports();
process.kill = (pid, signal) => { kills.push([pid, signal]); return true; };
const controller = new AbortController();
const events = [];
const postEvents = [];
const settlements = [];
if (scenario === 'pre_aborted') {
  controller.abort();
  const h = await service.execute('pwd -P', '/tmp', e => events.push(e), controller.signal, false, {});
  assert.equal((await h.result).aborted, true);
  assert.equal(children.length, 0);
} else {
  const signal = controller.signal;
  const h = service.childProcessFallback('pwd -P', '/tmp', e => events.push(e), signal,
    false, 1024 * 1024, 'cat', {
      onData: e => postEvents.push(e), onSettle: e => settlements.push(e),
    });
  const child = children[0];
  let settled = false;
  h.result.then(() => { settled = true; });
  const lateOutput = () => {
    child.stdout.emit('data', Buffer.from('late-out\n'));
    child.stderr.emit('data', Buffer.from('late-error\n'));
  };
  const close = (code = 0, signal = null) => {
    child.exitCode = code;
    child.signalCode = signal;
    child.emit('close', code, signal);
  };
  if (scenario === 'spawn_error') {
    child.emit('error', new Error('spawn ENOENT'));
    const result = await h.result;
    close(-2);
    assert.equal(result.exitCode, 1);
    assert.equal(result.error.message, 'spawn ENOENT');
    assert.equal(result.aborted, false);
    assert.equal(getEventListeners(signal, 'abort').length, 0);
  } else if (scenario === 'background') {
    child.stdout.emit('data', Buffer.from('before\n'));
    controller.abort({kind: 'background'});
    const result = await h.result;
    assert.equal(result.promoted, true);
    assert.equal(result.aborted, false);
    assert.equal(result.output, 'before');
    assert.equal(child.listenerCount('exit'), 0);
    assert.equal(child.listenerCount('close'), 1); // Only post-promotion close remains.
    assert.equal(getEventListeners(signal, 'abort').length, 0);
    lateOutput();
    close();
    assert.deepEqual(postEvents.map(e => e.chunk), ['late-out\n', 'late-error\n']);
    assert.equal(settlements.length, 1);
    assert.deepEqual(events, []);
  } else if (scenario === 'exit_no_close') {
    // A descendant keeps the pipes open: exit settles after the bounded drain
    // with output delivered inside it; later output is not part of the result.
    child.exitCode = 0;
    // The drain timer starts at exit, so measure from there.
    const started = performance.now();
    child.emit('exit', 0, null);
    lateOutput();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(settled, false, 'exit waits for close up to the drain bound');
    const result = await h.result;
    const waited = performance.now() - started;
    assert.ok(waited >= 150 && waited < 2000, String(waited));
    child.stdout.emit('data', Buffer.from('after-settle\n'));
    assert.equal(result.output, 'late-out\nlate-error');
    assert.equal(result.exitCode, 0);
    assert.equal(result.aborted, false);
    assert.equal(child.listenerCount('close'), 0);
    assert.equal(getEventListeners(signal, 'abort').length, 0);
    assert.deepEqual(kills, []);
  } else {
    if (scenario === 'cancel') controller.abort({kind: 'cancel'});
    if (scenario === 'deadline') controller.abort(new DOMException('fixture deadline', 'TimeoutError'));
    const cancelled = scenario === 'cancel' || scenario === 'deadline';
    if (cancelled) assert.deepEqual(kills, [[-child.pid, 'SIGTERM']]);
    child.exitCode = cancelled ? null : 0;
    child.signalCode = cancelled ? 'SIGTERM' : null;
    child.emit('exit', child.exitCode, child.signalCode);
    // As native: exit ends the command, so a later abort neither kills nor promotes.
    if (scenario === 'background_after_exit') controller.abort({kind: 'background'});
    if (scenario === 'cancel_after_exit') controller.abort({kind: 'cancel'});
    assert.equal(getEventListeners(signal, 'abort').length, 0);
    await Promise.resolve();
    assert.equal(settled, false, 'exit must not settle before the pipes drain');
    lateOutput();
    close(child.exitCode, child.signalCode);
    const result = await h.result;
    assert.equal(result.output, 'late-out\nlate-error');
    assert.equal(result.rawOutput.toString(), 'late-out\nlate-error\n');
    assert.equal(result.exitCode, cancelled ? null : 0);
    assert.equal(result.aborted, cancelled);
    assert.equal(result.promoted, undefined);
    assert.equal(child.listenerCount('close'), 0);
    assert.equal(getEventListeners(signal, 'abort').length, 0);
    assert.equal(events.length, 1);
    assert.deepEqual(postEvents, []);
    assert.deepEqual(settlements, []);
    if (!cancelled) assert.deepEqual(kills, []);
  }
  assert.equal(service.activeChildProcesses.size, 0);
}
// Native cancellation schedules a later SIGKILL check. Observe the real end of
// its timer rather than adding a sleep to the fixture or masking an extra kill.
process.once('beforeExit', () => {
  assert.deepEqual(kills, ['cancel', 'deadline'].includes(scenario) ? [[-424242, 'SIGTERM']] : []);
  assert.equal(service.activeChildProcesses.size, 0);
});
''', scenario)


def test_real_native_fallback_collects_concurrent_fast_command_output(tmp_path):
    native_script(NATIVE_SERVICE + r'''
const directory = fs.realpathSync(process.argv[2]);
const results = await Promise.all(Array.from({length: 12}, async () => {
  const h = service.childProcessFallback('pwd -P; printf "stderr-marker\\n" >&2', directory,
    () => {}, new AbortController().signal, false, 1024 * 1024, 'cat');
  return h.result;
}));
for (const result of results) {
  assert.equal(result.exitCode, 0);
  assert.equal(result.output, directory + '\nstderr-marker');
  assert.equal(result.executionMethod, 'child_process');
}
assert.equal(service.activeChildProcesses.size, 0);
''', str(tmp_path))


DESCENDANT = r"""
import cp from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
let exited;
let ready;
const parentExited = new Promise(resolve => { exited = resolve; });
const descendantReady = new Promise(resolve => { ready = resolve; });
const spawn = cp.spawn;
cp.spawn = (...args) => {
  const child = spawn(...args);
  let seen = '';
  child.stdout.on('data', data => { seen += data; if (seen.includes('descendant-ready')) ready(); });
  child.once('exit', () => exited(performance.now()));
  return child;
};
syncBuiltinESMExports();
const kills = [];
const kill = process.kill.bind(process);
process.kill = (pid, signal) => { kills.push([pid, signal]); return kill(pid, signal); };
const quote = value => "'" + value.replaceAll("'", "'\\''") + "'";
// The descendant ignores SIGTERM, holds stdout/stderr and has its own bounded
// lifetime (10 s), far beyond the drain bound the assertions use.
const descendant = "process.on('SIGTERM',()=>{}); process.stdout.write('descendant-ready\\n'); " +
  "setTimeout(() => process.exit(0), 10000);";
"""


@pytest.mark.parametrize("group", ["same_group", "new_session"])
def test_real_native_exit_does_not_wait_for_pipe_holding_descendants(tmp_path, group):
    # new_session: the descendant leaves the shell's process group (setsid), so
    # a timeout's group kill cannot reach it; waiting for 'close' would hang.
    native_script(NATIVE_SERVICE + DESCENDANT + r"""
const detached = process.argv[3] === 'new_session';
const parent = "const {spawn}=require('node:child_process'); const child=spawn(process.execPath,['-e'," +
  JSON.stringify(descendant) + "],{stdio:['ignore',1,2],detached:" + detached + "}); child.unref(); " +
  "console.log('descendant-pid', child.pid); process.stderr.write('parent-done\\n'); process.exit(0);";
const controller = new AbortController();
const deadline = AbortSignal.timeout(3000);  // like the tool's foreground timeout
deadline.addEventListener('abort', () => controller.abort(deadline.reason), {once: true});
const h = service.childProcessFallback(quote(process.execPath) + ' -e ' + quote(parent), process.argv[2],
  () => {}, controller.signal, false, 1024 * 1024, 'cat');
let descendantPid;
try {
  const result = await h.result;
  const settledAt = performance.now();
  const exitAt = await parentExited;
  descendantPid = Number(/descendant-pid (\d+)/.exec(result.output)?.[1]);
  assert.ok(settledAt - exitAt < 1500, 'settled ' + (settledAt - exitAt) + ' ms after the shell exit');
  assert.equal(result.exitCode, 0);
  assert.equal(result.aborted, false);
  assert.ok(result.output.includes('parent-done'), 'output written before exit is kept');
  assert.deepEqual(kills, []);
  assert.equal(service.activeChildProcesses.size, 0);
  kill(descendantPid, 0);  // Still alive: nothing waited for it.
} finally {
  if (descendantPid) try { kill(descendantPid, 'SIGKILL'); } catch {}
}
""", str(tmp_path), group)


@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_real_native_cancel_with_pipe_holding_descendant_settles_after_shell_exit(tmp_path, reason):
    native_script(NATIVE_SERVICE + DESCENDANT + r"""
const controller = new AbortController();
const h = service.childProcessFallback(quote(process.execPath) + ' -e ' + quote(descendant) + ' & wait',
  process.argv[2], () => {}, controller.signal, false, 1024 * 1024, 'cat');
let settled = false;
h.result.then(() => { settled = true; });
try {
  // The shell is alive (it waits), so the call must not settle on its own.
  await descendantReady;
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(settled, false);
  if (process.argv[3] === 'cancel') controller.abort({kind: 'cancel'});
  else controller.abort(new DOMException('fixture deadline', 'TimeoutError'));
  const result = await h.result;
  const exitAt = await parentExited;
  assert.ok(performance.now() - exitAt < 1500);
  assert.equal(result.aborted, true);
  assert.equal(result.exitCode, null);
  assert.equal(result.output, 'descendant-ready');
  assert.deepEqual(kills[0], [-h.pid, 'SIGTERM']);
  assert.ok(kills.every(([pid]) => pid === -h.pid));
  assert.equal(service.activeChildProcesses.size, 0);
} finally {
  try { kill(-h.pid, 'SIGKILL'); } catch {}
}
""", str(tmp_path), reason)


def test_real_native_backgrounded_job_does_not_hold_the_foreground_result(tmp_path):
    # 'cmd &' in a foreground call: the job inherits the pipes. Native settles
    # at shell exit; the repair waits only for its bounded drain, not the job.
    native_script(NATIVE_SERVICE + r'''
const started = performance.now();
const h = service.childProcessFallback('sleep 3 & echo started', process.argv[2],
  () => {}, new AbortController().signal, false, 1024 * 1024, 'cat');
try {
  const result = await h.result;
  const waited = performance.now() - started;
  assert.ok(waited < 1000, 'settled after ' + waited + ' ms');
  assert.equal(result.output, 'started');
  assert.equal(result.exitCode, 0);
  assert.equal(result.aborted, false);
  assert.equal(service.activeChildProcesses.size, 0);
} finally {
  try { process.kill(-h.pid, 'SIGKILL'); } catch {}
}
''', str(tmp_path))


def test_shell_drain_patch_stays_in_fallback_and_rejects_anchor_drift():
    cli, node = native_qwen()
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
const {patchNativeSource} = await import(process.argv[1]);
const chunks = process.argv[2];
const owners = fs.readdirSync(chunks).filter(name => name.endsWith('.js')).map(name =>
  fs.readFileSync(path.join(chunks, name), 'utf8')).filter(s => s.includes('var ShellExecutionService = class'));
assert.equal(owners.length, 1);
const source = owners[0];
const start = source.indexOf('  static childProcessFallback(');
const end = source.indexOf('  static executeWithPty(', start);
const patched = patchNativeSource(source);
// The PTY implementation itself remains byte-identical.
const ptyEnd = source.indexOf('\n};', end);
assert.ok(patched.includes(source.slice(end, ptyEnd)));
const fallback = source.slice(start, end);
for (const [anchor, drifted] of [
  ['const handleExit = /* @__PURE__ */ __name((code, signal) => {', 'const handleExit = /* @__PURE__ */ __name((code, signal2) => {'],
  ['            aborted: abortSignal.aborted,\n', '            aborted: abortSignal?.aborted,\n'],
  ['          handleExit(code, signal);\n        }, "exitHandler");', '          handleExit(code, signal, 0);\n        }, "exitHandler");'],
]) {
  assert.equal(fallback.split(anchor).length, 2);
  const changed = source.slice(0, start) + fallback.replace(anchor, drifted) + source.slice(end);
  assert.throws(() => patchNativeSource(changed), /anchor changed/);
}
// Only the fallback's abort snapshot is replaced; the PTY path's copy stays.
const snapshot = 'aborted: abortSignal.aborted,';
assert.equal(source.split(snapshot).length, 3);
assert.equal(patched.split(snapshot).length, 2);
assert.ok(patched.indexOf(snapshot) > patched.indexOf('  static executeWithPty('));
'''
    subprocess.run([node, "--input-type=module", "-e", script, PATCH.as_uri(), str(cli.parent / "chunks")],
                   check=True, capture_output=True, text=True, timeout=30)
