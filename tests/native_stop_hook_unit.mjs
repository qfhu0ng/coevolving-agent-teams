import assert from 'node:assert/strict';
import { installNativeStopHook } from '../agent_company/runtime/qwen_stop_hook.mjs';

function fixture({ safe = true } = {}) {
  return class NativeConfig {
    callbacks = [];
    loaded = [];
    isSafeMode() { return safe; }
    getDisableAllHooks() { return this.isSafeMode(); }
    getUserHooks() { return { Stop: ['untrusted-user-hook'] }; }
    getProjectHooks() { return { Stop: ['untrusted-project-hook'] }; }
    getExtensions() { return [{ hooks: { Stop: ['untrusted-extension-hook'] } }]; }
    getSessionId() { return 'root-session'; }
    getTeamManager() { return { hasActiveTeammates: () => this.activeTeammates === true,
      pendingAsyncWork: this.pendingAsyncWork }; }
    getStopHookBlockingCap() { return 8; }
    getHookSystem() { return this.hookSystem; }
    async initialize() {
      if (!this.getDisableAllHooks()) {
        this.loaded = [this.getUserHooks(), this.getProjectHooks(), ...this.getExtensions()].filter(Boolean);
        this.hookSystem = { getAllHooks: () => this.loaded,
          addFunctionHook: (...args) => this.callbacks.push(args) };
      }
      return 'native-initialization-result';
    }
  };
}

{
  const Config = fixture();
  const initialize = Config.prototype.initialize;
  assert.equal(installNativeStopHook(Config, {}), false);
  assert.equal(Config.prototype.initialize, initialize);
}

for (const owner of ['single', 'master']) {
  let time = 1000;
  let pendingLeaderMessages = false;
  const records = [];
  const Config = fixture();
  installNativeStopHook(Config, { owner, reason: 'TEST_ONLY_SENTINEL {remaining_seconds}',
    deadlineEpochMs: 2001000, now: () => time, record: (...args) => records.push(args),
    hasPendingLeaderMessages: () => pendingLeaderMessages });
  const config = new Config();
  assert.equal(await config.initialize(), 'native-initialization-result');
  assert.equal(config.isSafeMode(), true);
  assert.equal(config.getStopHookBlockingCap(), 8);
  assert.deepEqual(config.loaded, []);
  assert.equal(config.getUserHooks(), undefined);
  assert.equal(config.getProjectHooks(), undefined);
  assert.deepEqual(config.getExtensions(), []);
  assert.equal(config.callbacks.length, 1);
  const [session, event, matcher, callback, errorText, options] = config.callbacks[0];
  assert.equal(session, 'root-session');
  assert.equal(event, 'Stop');
  assert.equal(matcher, '*');
  assert.equal(errorText, undefined);
  assert.equal(options.id, 'agent-company-root-stop');
  const input = { session_id: session, stop_hook_active: true };
  assert.deepEqual(callback(input, {}), { decision: 'block', reason: 'TEST_ONLY_SENTINEL 2000' });
  time = 1100000;
  assert.deepEqual(callback(input, {}), { decision: 'block', reason: 'TEST_ONLY_SENTINEL 901' });
  time = 1101000;
  assert.deepEqual(callback(input, {}), { continue: true });
  assert.equal(records.at(-1)[1].allow_reason, 'final_fifteen_minutes');
  time = 1102000;
  assert.deepEqual(callback(input, {}), { continue: true });
  time = 2001000;
  assert.deepEqual(callback(input, {}), { continue: true });
  assert.equal(records.at(-1)[1].allow_reason, 'deadline');
  time = 1000;
  assert.deepEqual(callback(input, { signal: { aborted: true } }), { continue: true });
  assert.equal(records.at(-1)[1].allow_reason, 'aborted');
  assert.deepEqual(callback({ session_id: 'worker-session' }, {}), { continue: true });
  assert.equal(records.at(-1)[1].allow_reason, 'other_session');
  config.activeTeammates = true;
  assert.deepEqual(callback(input, {}), owner === 'master' ? { continue: true }
    : { decision: 'block', reason: 'TEST_ONLY_SENTINEL 2000' });
  assert.equal(records.at(-1)[1].allow_reason, owner === 'master' ? 'active_teammates' : null);
  config.activeTeammates = false;
  config.pendingAsyncWork = new Set(['automatic report write']);
  assert.deepEqual(callback(input, {}), owner === 'master' ? { continue: true }
    : { decision: 'block', reason: 'TEST_ONLY_SENTINEL 2000' });
  assert.equal(records.at(-1)[1].allow_reason, owner === 'master' ? 'pending_team_work' : null);
  assert.equal(config.pendingAsyncWork.size, 1);
  config.pendingAsyncWork.clear();
  pendingLeaderMessages = true;
  assert.deepEqual(callback(input, {}), owner === 'master' ? { continue: true }
    : { decision: 'block', reason: 'TEST_ONLY_SENTINEL 2000' });
  assert.equal(records.at(-1)[1].allow_reason, owner === 'master' ? 'pending_leader_messages' : null);
  assert.equal(pendingLeaderMessages, true);
  assert.equal(records[0][0], 'stop_hook_installed');
  assert.equal(records.filter(([name]) => name === 'stop_hook_decision').length, 10);
}

{
  const Config = fixture({ safe: false });
  installNativeStopHook(Config, { owner: 'single', reason: 'TEST_ONLY_SENTINEL', deadlineEpochMs: 1000 });
  await assert.rejects(new Config().initialize(), /requires native safe mode/);
}
assert.throws(() => installNativeStopHook(fixture(), {
  owner: 'worker', reason: 'TEST_ONLY_SENTINEL', deadlineEpochMs: 1000 }), /root session/);
assert.throws(() => installNativeStopHook(fixture(), {
  owner: 'single', reason: 'TEST_ONLY_SENTINEL', deadlineEpochMs: NaN }), /task deadline/);
console.log('Native Stop hook unit checks passed (no model calls).');
