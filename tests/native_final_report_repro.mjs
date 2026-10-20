// Deterministic reproduction against the installed TeamManager, no provider.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [chunks, expected] = process.argv.slice(2);
const file = fs.readdirSync(chunks).find(name => name.endsWith('.js') &&
  fs.readFileSync(path.join(chunks, name), 'utf8').includes('var TeamManager = class'));
const {TeamManager} = await import(pathToFileURL(path.join(chunks, file)).href);
let release;
const blockedWrite = new Promise(resolve => { release = resolve; });
const inbox = [], received = [];
let drained = false;
const fake = {pendingAsyncWork: new Set(), pendingMessages: new Map(),
  async sendMessage(to, message, from, summary, automatic) {
    assert.equal(to, 'leader'); assert.equal(automatic, true);
    await blockedWrite; inbox.push(message);
  },
  async pollLeaderInbox() { received.push(...inbox.splice(0)); },
};
// Native reports a turn after the worker becomes idle, without awaiting I/O.
TeamManager.prototype.fireAndForget.call(fake, 'reportFinalAnswer(alpha)',
  fake.sendMessage('leader', 'ALPHA_ACK: updated policy', 'alpha', undefined, true));
const drain = TeamManager.prototype.drainLeaderInbox.call(fake).then(() => { drained = true; });
await new Promise(resolve => setImmediate(resolve));
assert.equal(drained, expected === 'before', 'Finalization must wait for automatic leader persistence');
release();
await drain;
await Promise.allSettled([...fake.pendingAsyncWork]);
assert.equal(received.includes('ALPHA_ACK: updated policy'), expected === 'after');
console.log(JSON.stringify({phase: expected, automatic_write_delayed: true,
  ack_seen_before_finalization: received.length === 1, lost: received.length === 0}));
