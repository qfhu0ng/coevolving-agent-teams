// Deterministic native poll/wait interleaving; no timers used to order I/O.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {EventEmitter} from 'node:events';
import {pathToFileURL} from 'node:url';
const [chunks, expected, detach] = process.argv.slice(2);
const file = fs.readdirSync(chunks).find(name => name.endsWith('.js') &&
  fs.readFileSync(path.join(chunks, name), 'utf8').includes('var TeamManager = class'));
const {TeamManager} = await import(pathToFileURL(path.join(chunks, file)).href);
let release;
const ready = new Promise(resolve => { release = resolve; });
const received = [];
const fake = {leaderMessageCallback: message => received.push(message),
  teamEventEmitter: new EventEmitter(), allTeammatesTerminated: () => false,
  async consumeLeaderInbox() { await ready; return [{from: 'w11', text: 'ready'}]; },
  formatLeaderEnvelope: () => ['w11 ready'], formatLeaderDisplay: () => 'w11 ready'};
const controller = new AbortController();
const poll = TeamManager.prototype.pollLeaderInbox.call(fake);
let reason;
const waiting = TeamManager.prototype.waitForTeammateActivity.call(fake, 10000, controller.signal)
  .then(value => { reason = value; });
if (detach === 'detach') fake.leaderMessageCallback = null;
release();
await poll;
await new Promise(resolve => setImmediate(resolve));
assert.equal(received.length, detach === 'detach' && expected === 'after' ? 0 : 1);
assert.equal(reason, expected === 'after' && detach !== 'detach' ? 'message' : undefined);
console.log(JSON.stringify({phase: expected, detached: detach === 'detach',
  queued_messages: received.length, awakened: reason === 'message'}));
controller.abort();
await waiting;
