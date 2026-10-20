"""No-model checks for replacing an unobserved runtime check-in safely."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


MODULE = Path(__file__).parents[1] / "agent_company/runtime/qwen_team_checkin.mjs"


def run_node(script):
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not node:
        pytest.skip("node is required for check-in scheduler tests")
    result = subprocess.run(
        [node, "--input-type=module", "-e", script, MODULE.as_uri()],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_missing_observation_is_replaced_each_period_with_a_fresh_id():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler, runCheckin} = await import(process.argv[1]);
const interval = 1800_000;
const scheduler = new CheckinScheduler({intervalMs: interval, startMs: 0, deadlineMs: 21600_000,
  milestones: [], finalGuardMs: 60_000});
const decisions = [];
let clock = interval, slot = null;
for (; clock <= 5 * interval; clock += interval) {
  await runCheckin({scheduler, now: () => clock,
    teamState: () => ({active: true, windingDown: false}),
    snapshot: async decision => ({id: decision.id, at: clock}), publish: () => {},
    // The binding owns one replaceable notification slot, including recovery
    // when the old notification disappeared before it reached a model request.
    send: async (view, decision) => {
      if (slot) assert.equal(decision.replacesId, slot.id);
      slot = view;
      decisions.push(decision);
    }});
  assert.equal(slot.at, clock, 'The pending notification must contain the latest snapshot');
  if (clock === 2 * interval) slot = null; // Simulated lost notification, not an observation.
}
assert.deepEqual(decisions.map(d => d.id), ['checkin-1', 'checkin-2', 'checkin-3', 'checkin-4', 'checkin-5']);
assert.deepEqual(decisions.map(d => d.replacesId ?? null), [null, 'checkin-1', 'checkin-2', 'checkin-3', 'checkin-4']);
assert.equal(scheduler.pendingId, 'checkin-5', 'Expiry must not count as observation');
for (const old of decisions.slice(0, -1)) scheduler.observed(old.id);
assert.equal(scheduler.pendingId, 'checkin-5', 'A late old confirmation cannot clear a replacement');
scheduler.observed('checkin-5');
assert.equal(scheduler.pendingId, null);
''')


def test_young_pending_milestone_coalesces_then_is_preserved_in_replacement():
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler} = await import(process.argv[1]);
const scheduler = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 10_000,
  milestones: [0.09, 0.15], minSpacingMs: 50, finalGuardMs: 1});
const team = {active: true, windingDown: false};
const first = scheduler.poll(900, team);
assert.equal(first.reason, 'milestone');
assert.deepEqual(scheduler.poll(1000, team), {action: 'coalesce', reason: 'periodic', pendingId: first.id});
assert.equal(scheduler.poll(1500, team).action, 'none');
const replacement = scheduler.poll(2000, team);
assert.equal(replacement.replacesId, first.id);
assert.deepEqual(replacement.milestones, ['9% of the task wall time has elapsed', '15% of the task wall time has elapsed']);
assert.equal(scheduler.poll(2100, team).action, 'none');
''')


@pytest.mark.parametrize("failure_phase", ["snapshot", "publish", "send"])
def test_failed_replacement_restores_old_pending_and_unsent_milestones(failure_phase):
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler, runCheckin} = await import(process.argv[1]);
const phase = PHASE;
const scheduler = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 10_000,
  milestones: [0.15], minSpacingMs: 50, finalGuardMs: 1});
const team = {active: true, windingDown: false};
const first = scheduler.poll(1000, team);
const fail = () => {throw new Error('synthetic ' + phase + ' failure');};
await assert.rejects(runCheckin({scheduler, now: () => 2000, teamState: () => team,
  snapshot: async decision => phase === 'snapshot' ? fail() : {id: decision.id},
  publish: () => {if (phase === 'publish') fail();},
  send: async () => {if (phase === 'send') fail();}}), /synthetic/);
assert.equal(scheduler.pendingId, first.id);
assert.equal(scheduler.lastSentMs, 1000);
const retry = scheduler.poll(2001, team);
assert.equal(retry.replacesId, first.id);
assert.notEqual(retry.id, first.id);
assert.deepEqual(retry.milestones, ['15% of the task wall time has elapsed']);
'''.replace("PHASE", repr(failure_phase)))


@pytest.mark.parametrize("ending", ["shutdown", "inactive", "deadline", "cleanup"])
def test_delayed_replacement_cannot_bypass_lifecycle_and_preserves_prior_pending(ending):
    run_node(r'''
import assert from 'node:assert/strict';
const {CheckinScheduler, runCheckin} = await import(process.argv[1]);
const ending = ENDING;
const scheduler = new CheckinScheduler({intervalMs: 1000, startMs: 0, deadlineMs: 10_000,
  milestones: [], finalGuardMs: 1});
let clock = 1000, team = {active: true, windingDown: false};
const first = scheduler.poll(clock, team);
clock = 2000;
let release, started;
const sampling = new Promise(resolve => started = resolve);
const pending = runCheckin({scheduler, now: () => clock, teamState: () => team,
  snapshot: decision => {started(); return new Promise(resolve => release = () => resolve({id: decision.id}));},
  publish: () => {}, send: async () => assert.fail('Lifecycle changed while collecting replacement')});
await sampling;
if (ending === 'shutdown') team = {active: true, windingDown: true};
if (ending === 'inactive') team = {active: false, windingDown: false};
if (ending === 'deadline') clock = 10_000;
if (ending === 'cleanup') scheduler.stop('cleanup');
release();
await pending;
assert.equal(scheduler.pendingId, first.id, 'The old notification still awaits observation');
assert.equal(scheduler.lastSentMs, 1000);
if (ending === 'shutdown' || ending === 'inactive') {
  const resumed = scheduler.poll(2001, {active: true, windingDown: false});
  assert.equal(resumed.replacesId, first.id);
} else {
  assert.equal(scheduler.poll(10_001, {active: true, windingDown: false}).action, 'none');
}
'''.replace("ENDING", repr(ending)))
