// Native failure delivery and explicit request ownership. This changes the
// terminal-message bridge, not assignment/recovery policy or worker tools.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash, randomBytes } from 'node:crypto';
import { registerHooks } from 'node:module';
import { CHECKIN_QUEUE_ANCHOR, LEADER_LOOP_ANCHORS, POLICY_REPORT_ANCHOR, TEAMMATE_ROUND_ANCHOR, patchNativeSource,
         stripLeadingAnalysis } from './qwen_native_patch.mjs';

// Node reports loaded modules by real path; compare against the real directory
// so a symlinked Qwen install still receives its source patches.
const chunks = fs.realpathSync(path.join(path.dirname(path.resolve(process.env.AGENT_COMPANY_QWEN_CLI)), 'chunks'));
const purpose = new AsyncLocalStorage();
const rootOwner = process.env.AGENT_COMPANY_ROOT_OWNER;
const relayURL = process.env.AGENT_COMPANY_RELAY_URL;
if (!['master', 'single'].includes(rootOwner) || !relayURL) throw new Error('Explicit native root owner and relay URL required');
const checkinSeconds = Number(process.env.AGENT_COMPANY_CHECKIN_SECONDS || 0);
const requireCheckinQueue = checkinSeconds > 0 && rootOwner === 'master';
let checkinQueuePath = null;
let checkinQueueApplied = false;
let agentStartApplied = false;  // in-process teammates announce themselves before their first turn
if (requireCheckinQueue) {
  // --import runs before the CLI entry point loads its leader queue. Validate
  // that one source can be adapted now; the load hook below verifies the actual
  // loaded source too. A missing/changed queue must fail before any model call.
  let count = 0;
  for (const file of fs.readdirSync(chunks).filter(name => name.endsWith('.js'))) {
    const absolute = path.join(chunks, file);
    const matches = fs.readFileSync(absolute, 'utf8').split(CHECKIN_QUEUE_ANCHOR).length - 1;
    count += matches;
    if (matches) checkinQueuePath = absolute;
  }
  if (count !== 1) throw new Error(`Team check-ins require exactly one native leader queue patch anchor; found ${count}`);
}
// Opt-in teammate message delivery. native_idle keeps Qwen's behavior: a
// message waits until the recipient is idle, one per idle transition.
// between_tool_calls also delivers queued messages after each tool batch.
const deliveryMode = process.env.AGENT_COMPANY_TEAM_MESSAGE_DELIVERY || 'native_idle';
if (!['native_idle', 'between_tool_calls'].includes(deliveryMode)) {
  throw new Error(`Unsupported team message delivery mode: ${deliveryMode}`);
}
if (deliveryMode !== 'native_idle' && rootOwner !== 'master') {
  throw new Error('Teammate message delivery applies to Team runs only');
}
const teammateDelivery = deliveryMode === 'between_tool_calls';
let teammateRoundPath = null;
let teammateRoundApplied = false;
if (teammateDelivery) {
  // Fail before any model call when the installed CLI's round loop changed.
  let count = 0;
  for (const file of fs.readdirSync(chunks).filter(name => name.endsWith('.js'))) {
    const absolute = path.join(chunks, file);
    const matches = fs.readFileSync(absolute, 'utf8').split(TEAMMATE_ROUND_ANCHOR).length - 1;
    count += matches;
    if (matches) teammateRoundPath = absolute;
  }
  if (count !== 1) throw new Error(`Teammate message delivery requires exactly one native round anchor; found ${count}`);
}
// Opt-in runtime disclosure (a note in the status file) and leader loop
// recovery (2026-10-02); independent switches, both off by default.
function flag(name) {
  const value = process.env[name];
  if (value !== undefined && !['0', '1'].includes(value)) throw new Error(`${name} must be 0 or 1`);
  return value === '1';
}
const runtimeDisclosure = flag('AGENT_COMPANY_RUNTIME_DISCLOSURE');
const rolePackageV1 = flag('AGENT_COMPANY_ROLE_PACKAGE');
// agent_rules_v1 (plan v4): per-teammate policies; the relay owns all policy state.
const policyProtocol = process.env.AGENT_COMPANY_POLICY_PROTOCOL;
if (policyProtocol !== undefined && policyProtocol !== 'agent_rules_v1') {
  throw new Error(`Unsupported policy protocol: ${policyProtocol}`);
}
const agentRules = policyProtocol === 'agent_rules_v1';
if (agentRules && rolePackageV1) throw new Error('agent_rules_v1 and role_package_v1 are exclusive');
// Any package protocol: fine-grained task/Git collaboration samples. Behavior that
// belongs to role_package_v1 alone (role headers, POLICY-REVIEW) checks rolePackageV1.
const rolePackage = rolePackageV1 || agentRules;
// Configurable worker slots (plan v4 section 10); absent keeps the earlier 10.
const teammateLimitSetting = process.env.AGENT_COMPANY_TEAMMATE_LIMIT;
if (teammateLimitSetting !== undefined && !/^[1-9][0-9]{0,3}$/.test(teammateLimitSetting)) {
  throw new Error('AGENT_COMPANY_TEAMMATE_LIMIT must be a positive integer');
}
const teammateLimit = teammateLimitSetting === undefined ? 10 : Number(teammateLimitSetting);
// The native setting itself is written by qwen.py; this records the requested mode.
const toolCallCap = process.env.AGENT_COMPANY_TOOL_CALL_CAP || 'native_default';
if (!['native_default', 'disabled'].includes(toolCallCap)) throw new Error(`Unsupported tool call cap mode: ${toolCallCap}`);
const leaderLoopRecovery = flag('AGENT_COMPANY_LEADER_LOOP_RECOVERY');
// Scope A: only the master's repeated reads of this run's status file.
const leaderStatusFile = process.env.AGENT_COMPANY_STATUS_FILE;
if (leaderLoopRecovery && (!requireCheckinQueue || !leaderStatusFile || !path.isAbsolute(leaderStatusFile))) {
  throw new Error('Leader loop recovery applies to Team runs with runtime check-ins and an absolute status file');
}
const leaderLoopPaths = {};
const leaderLoopApplied = { detect: false, runner: false };
if (leaderLoopRecovery) {
  // Fail before any model call when the installed CLI's guard or runner changed.
  for (const [part, anchor] of Object.entries(LEADER_LOOP_ANCHORS)) {
    let count = 0;
    for (const file of fs.readdirSync(chunks).filter(name => name.endsWith('.js'))) {
      const absolute = path.join(chunks, file);
      const matches = fs.readFileSync(absolute, 'utf8').split(anchor).length - 1;
      count += matches;
      if (matches) leaderLoopPaths[part] = absolute;
    }
    if (count !== 1) throw new Error(`Leader loop recovery requires exactly one native ${part} anchor; found ${count}`);
  }
  if (leaderLoopPaths.response !== leaderLoopPaths.detect) {
    throw new Error('Leader loop recovery requires the response and detect anchors in one module');
  }
  if (leaderLoopPaths.resume !== leaderLoopPaths.exit || leaderLoopPaths.settle !== leaderLoopPaths.exit) {
    throw new Error('Leader loop recovery requires the runner exit, resume and settle anchors in one module');
  }
}
// Role-level communication policies (2026-10-10); the relay owns all policy state.
// Here: D10 option C in propose mode, and per-recipient broadcast metadata.
const rolePolicyMode = process.env.AGENT_COMPANY_ROLE_POLICY_MODE || 'off';
if (!['off', 'global', 'master', 'propose'].includes(rolePolicyMode)) {
  throw new Error(`Unsupported role policy mode: ${rolePolicyMode}`);
}
const policyProposalReports = rolePolicyMode === 'propose';
let policyReportPath;
let policyReportApplied = false;
if (policyProposalReports) {
  let count = 0;
  for (const file of fs.readdirSync(chunks).filter(name => name.endsWith('.js'))) {
    const absolute = path.join(chunks, file);
    const matches = fs.readFileSync(absolute, 'utf8').split(POLICY_REPORT_ANCHOR).length - 1;
    count += matches;
    if (matches) policyReportPath = absolute;
  }
  if (count !== 1) throw new Error(`Policy proposal reporting requires exactly one native anchor; found ${count}`);
}
const POLICY_PROPOSAL_HEADER = /^POLICY-PROPOSAL(?:[ \t]+base=v(?:0|[1-9][0-9]{0,8}))?$/;
const PACKAGE_PROPOSAL_HEADER = /^POLICY-PROPOSAL[ \t]+(?:C|role:[a-z][a-z0-9_-]{0,31})[ \t]+(?:append|replace)(?:[ \t]+base=v(?:0|[1-9][0-9]{0,8}))?$/;
// agent_rules_v1 controls and board posts never trigger "message" rules (plan v4 sections 4.2, 9).
const AGENT_RULES_ANY_CONTROL = /^(?:AGENT-POLICY|POLICY-PROPOSAL|POLICY-DECISION)[ \t]|^(?:LESSON|BOARD):/;
// agent_rules_v1 worker -> leader controls are not task reports: a proposal or a lesson. A board
// post is an ordinary message to its recipient, so it counts as a report (user decision 10-17).
const AGENT_RULES_CONTROL = /^(?:POLICY-PROPOSAL[ \t]+(?:self|\*|[A-Za-z][A-Za-z0-9_-]{0,63})[ \t]+(?:add|notes|(?:replace|remove)[ \t]+R[1-9][0-9]{0,5})[ \t]*$|LESSON:)/;
// agent_rules_v1: the relay matches a native send to the model's call by sender,
// recipient and this content hash (no message text leaves compat in records).
function messageHash(message) {
  return createHash('sha256').update(String(message)).digest('hex').slice(0, 16);
}
const NATIVE_SHUTDOWN_TEXT = /^The team leader has requested that you shut down\./;
// Native broadcast calls sendMessage once per recipient; those inner sends are part of the
// broadcast's own record, never separate direct sends (M2 review N4). A count per sender
// and content, not an async context: a round that a broadcast wakes must not inherit it (N8).
const broadcastInner = {
  pending: new Map(),
  begin(from, message, count) {
    const key = `${String(from).toLowerCase()}\0${messageHash(message)}`;
    this.pending.set(key, (this.pending.get(key) || 0) + count);
    return key;
  },
  consume(from, message) {
    const key = `${String(from).toLowerCase()}\0${messageHash(message)}`;
    const left = this.pending.get(key) || 0;
    if (left <= 0) return false;
    if (left === 1) this.pending.delete(key); else this.pending.set(key, left - 1);
    return true;
  },
  end(key) {
    // Native calls sendMessage for every recipient before its first await; a leftover
    // count means native changed, so keep evidence of it (M2 review R4-3).
    const left = this.pending.get(key) || 0;
    if (left > 0) record('broadcast_inner_count_mismatch', { left });
    this.pending.delete(key);
  },
};
function firstLine(message) {
  return typeof message === 'string' ? message.split(/\r\n|\r|\n/, 1)[0].trim() : '';
}
const notices = new Map();
const writes = new WeakMap();
let checkins = null;  // Opt-in runtime team check-ins; see the end of this module.
let leaderMessageQueue = null;  // Read-only view of the existing native queue for Stop decisions.
let teammateContext;
let sequence = 0;
let leaderRecovery = null;               // the recovery awaiting a teammate message or check-in
let leaderRecoverySequence = 0;
const leaderNotices = new Map();         // notice marker -> recovery id, until seen in a master request
function record(event, data = {}) {
  // stderr is retained even when the runtime directory cannot be recovered.
  process.stderr.write('Agent Company lifecycle: ' + JSON.stringify({ event, at: Date.now(), ...data }) + '\n');
}

// ----- Teammate delivery state (between_tool_calls only) ----------------------
// "Drained" means removed from a queue and handed to the teammate's round (or,
// at idle, to its next round); "observed" means its unique marker appeared in
// one of that teammate's coding requests. A round can end after a drain without
// another request (cancel, turn or time limit), so only observed inputs count
// as received, and a message drained twice (requeued) is counted once by its
// message_id. Observed is not provider acceptance or model compliance.
const teamManagers = new Set();          // TeamManager instances owning in-process teammates
const agentStartListeners = new Set();   // Called with each in-process teammate before its first turn
const taskNotices = new Map();           // agentId -> [{ taskId, text, marker, queuedAt }] assigned while busy
const announcedShutdowns = new WeakSet(); // native shutdown ledger tokens with a notice in flight or observed
const unobserved = new Map();            // teammate name -> drained inputs not yet seen in a model request
const idleLabels = new Map();            // agentId -> [{ from, text, marker }] native idle envelopes not yet matched
const trackedQueues = new WeakSet();
const unmatchedAgents = new WeakSet();
let messageSequence = 0;

function requireShutdownLedgers(team) {
  // Shutdown notices are keyed by native ledger tokens (one per request).
  if (!(team.shutdownLedgers instanceof Map)) {
    throw new Error('Native shutdown ledger unavailable; teammate delivery cannot track shutdown requests');
  }
}

function unannouncedShutdown(team, name) {
  // One notice per shutdown request, so a teammate that rejected one request
  // is told again about the next instead of waiting for its turn to end. A
  // notice that never reached a request releases its tokens (requeueUnobserved).
  const tokens = team.shutdownLedgers.get(name)?.outstandingTokens || [];
  const fresh = tokens.filter(token => !token.consumed && !announcedShutdowns.has(token));
  for (const token of fresh) announcedShutdowns.add(token);
  return fresh;
}

function tagMessage(item) {
  if (!item.agentCompanyMessageId) {
    item.agentCompanyMessageId = `msg-${++messageSequence}`;
    item.agentCompanyQueuedAt = Date.now();
  }
  return item;
}

function trackQueue(team, member, queue) {
  // Native creates each recipient queue once and only push()es to it from
  // sendMessage, before any idle flush: tag every message there.
  if (trackedQueues.has(queue)) return;
  trackedQueues.add(queue);
  const push = queue.push;
  queue.push = function (...items) {
    const status = team.getAgentFromBackend?.(member.agentId)?.getStatus?.() ?? null;
    for (const item of items) {
      tagMessage(item);
      record('team_message_queued', { to: member.name, message_id: item.agentCompanyMessageId, from: item.from || null,
                                      priority: item.priority, recipient_status: status,
                                      chars: String(item.text).length, queue_length: this.length + 1 });
    }
    return push.apply(this, items);
  };
}

function requestText(request) {
  return (request.messages || []).map(message => typeof message.content === 'string'
    ? message.content : (message.content || []).map(part => part.text || '').join('\n')).join('\n');
}

function awaitObservation(name, entries) {
  if (entries.length === 0) return;
  let pending = unobserved.get(name);
  if (!pending) unobserved.set(name, pending = []);
  pending.push(...entries);
}

function observeTeammateRequest(name, request) {
  const pending = unobserved.get(name);
  const text = requestText(request);
  const seen = pending.filter(entry => text.includes(entry.marker));
  if (seen.length === 0) return;
  const rest = pending.filter(entry => !seen.includes(entry));
  if (rest.length) unobserved.set(name, rest); else unobserved.delete(name);
  const now = Date.now();
  const messages = seen.filter(entry => entry.kind === 'team_message');
  const notes = seen.filter(entry => entry.kind === 'task_notice');
  if (messages.length) {
    record('team_message_observed', { worker: name, count: messages.length, items: messages.map(entry => ({
      ...itemMeta(entry.item, now), mode: entry.mode, since_drained_ms: now - entry.drainedAt })) });
  }
  if (notes.length) {
    record('task_notice_observed', { worker: name, task_ids: notes.map(entry => entry.note.taskId),
                                     modes: notes.map(entry => entry.mode),
                                     waited_ms: notes.map(entry => now - entry.note.queuedAt) });
  }
  if (seen.some(entry => entry.kind === 'shutdown_notice')) record('shutdown_notice_observed', { worker: name });
}

function requeueUnobserved(team, agentId, name) {
  // The round ended after a drain without sending it; give it back so native
  // idle delivery (messages), the idle task path (notices) or the next drain
  // (shutdown notices, while the request is still pending) delivers it. Idle
  // deliveries are not included: they were handed to the next round, which
  // may not have started yet.
  const pending = unobserved.get(name);
  const ended = (pending || []).filter(entry => entry.mode === 'between_tool_calls');
  if (ended.length === 0) return;
  const rest = pending.filter(entry => !ended.includes(entry));
  if (rest.length) unobserved.set(name, rest); else unobserved.delete(name);
  const messages = ended.filter(entry => entry.kind === 'team_message').map(entry => entry.item);
  const notes = ended.filter(entry => entry.kind === 'task_notice').map(entry => entry.note);
  const shutdowns = ended.filter(entry => entry.kind === 'shutdown_notice');
  if (messages.length) {
    let queue = team.pendingMessages.get(agentId);
    if (!queue) team.pendingMessages.set(agentId, queue = []);
    queue.unshift(...messages);
  }
  if (notes.length) {
    let list = taskNotices.get(agentId);
    if (!list) taskNotices.set(agentId, list = []);
    list.unshift(...notes);
  }
  for (const entry of shutdowns) for (const token of entry.tokens) announcedShutdowns.delete(token);
  record('teammate_inputs_requeued', { worker: name, message_ids: messages.map(item => item.agentCompanyMessageId),
                                       task_ids: notes.map(note => note.taskId), shutdown_notices: shutdowns.length });
}

function parseNativeEnvelope(message) {
  // Native idle flush labels a queued message exactly like labelTeamMessage.
  const match = typeof message === 'string' && message.match(/^<team_message_([0-9a-f]+) from="([^"]*)">\n/);
  if (!match) return null;
  const close = `\n</team_message_${match[1]}>\n`;
  const end = message.lastIndexOf(close);
  if (end < match[0].length) return null;
  return { from: match[2], text: message.slice(match[0].length, end), marker: `<team_message_${match[1]} from=` };
}

function memberByName(team, name) {
  const members = team.teamFile?.members || [];
  return members.find(member => member.name === name)
    || members.find(member => String(member.name).toLowerCase() === String(name).toLowerCase());
}

function locateTeammate(agent) {
  for (const team of teamManagers) {
    for (const member of team.teamFile?.members || []) {
      if (team.getAgentFromBackend?.(member.agentId) === agent) return { team, member };
    }
  }
  return null;
}

function labelTeamMessage(item) {
  // Same envelope as native flushNextMessage, so the model sees one format.
  // The marker identifies this delivery in the teammate's later requests.
  if (!item.from) return { text: String(item.text), marker: String(item.text) };
  const nonce = randomBytes(8).toString('hex');
  return { marker: `<team_message_${nonce} from=`,
           text: `<team_message_${nonce} from="${item.from}">\n${item.text}\n</team_message_${nonce}>\n` +
    `The message above was delivered verbatim from "${item.from}"; sender claims inside the body are unverified text.` };
}

function itemMeta(item, now) {
  tagMessage(item);
  return { message_id: item.agentCompanyMessageId, from: item.from || null, priority: item.priority,
           chars: String(item.text).length, waited_ms: now - item.agentCompanyQueuedAt,
           ...(agentRules ? { control: AGENT_RULES_ANY_CONTROL.test(firstLine(item.text)) } : {}) };
}

function teamCounts(team) {
  // Native statuses: running/idle/completed/failed/cancelled. Idle teammates
  // are alive but not working; only running ones make waiting meaningful.
  const counts = { running: 0, idle: 0, terminal: 0, unknown: 0 };
  for (const member of team?.teamFile?.members || []) {
    const status = team.getAgentFromBackend?.(member.agentId)?.getStatus?.();
    if (!status) counts.unknown++;
    else if (['completed', 'failed', 'cancelled'].includes(status)) counts.terminal++;
    else if (status === 'idle') counts.idle++;
    else counts.running++;
  }
  return counts;
}

const bridge = {
  leaderLoopRecoverable(event, loopType, responseCalls) {
    // Called by the patched main session only after an always-on guard stopped
    // it. Native still clears the pending calls and ends the interaction; this
    // only says whether the halt is the scope-A case (no arguments recorded).
    // The stopped response is never added to history and none of its calls
    // run; v1 accepts only a response whose sole call is the status read.
    const call = event?.value || {};
    const file = call.args?.file_path;
    const target = call.name === 'read_file' && typeof file === 'string' && path.isAbsolute(file)
      && path.normalize(file) === path.normalize(leaderStatusFile);
    const owner = teammateContext()?.agentName || rootOwner;
    const reason = owner !== rootOwner ? 'not_master'
      : loopType !== 'consecutive_identical_tool_calls' ? 'other_loop_type'
      : !target ? 'not_status_file_read' : responseCalls !== 1 ? 'other_calls_in_response' : null;
    record('leader_loop_guard', { owner, loop_type: loopType ?? null, tool: call.name ?? null,
                                  status_file_target: target, response_calls: responseCalls ?? null,
                                  recoverable: reason === null, ...(reason ? { reason } : {}) });
    return reason === null;
  },
  recoverLeaderLoop(client, queuedCalls, team) {
    // Called by the patched headless runner when a turn ended in a loop event.
    // true: end the turn without tool calls, i.e. native teammate waiting.
    const flagged = client?.agentCompanyRecoverableLoop === true;
    if (client) client.agentCompanyRecoverableLoop = false;
    if (!flagged) return false;
    const id = `leader-loop-${++leaderRecoverySequence}`;
    const counts = teamCounts(team);
    // Calls already handed to the runner would run once the turn continued;
    // implied by a one-call response, checked again here.
    const reason = queuedCalls > 0 ? 'calls_already_queued'
      : !team ? 'no_team' : counts.running === 0 ? 'no_running_teammate' : null;
    if (reason) {
      record('leader_loop_not_recovered', { recovery_id: id, reason, queued_calls: queuedCalls, ...counts });
      return false;
    }
    leaderRecovery = { id, team, at: Date.now() };
    record('leader_loop_recovered', { recovery_id: id, ...counts });
    return true;
  },
  leaderResumeBatch(batch) {
    // The next master prompt after a recovery: real teammate messages or a check-in.
    if (!leaderRecovery || batch.length === 0) return batch;
    const recovery = leaderRecovery;
    leaderRecovery = null;
    const waited = Math.round((Date.now() - recovery.at) / 1000);
    const marker = `Runtime notice [${recovery.id}]:`;
    leaderNotices.set(marker, recovery.id);
    record('leader_loop_resumed', { recovery_id: recovery.id, waited_s: waited, messages: batch.length,
                                    checkin: batch.some(text => String(text).includes('<teammate_message from="runtime:checkin">')) });
    return [`${marker} your previous turn was stopped by Qwen's always-on loop guard at the fifth consecutive ` +
      'identical read_file call of the runtime status file; that response was discarded and the call was not ' +
      'executed. Instead of exiting, ' +
      `the runtime ended that turn and waited without model calls for ${waited} s until the message(s) below ` +
      'arrived. Ending a turn without tool calls waits the same way.', ...batch];
  },
  leaderRecoveryUnresolved() {
    // No message and no active teammate after the wait: report the original
    // loop failure rather than a successful completion.
    if (!leaderRecovery) return false;
    const recovery = leaderRecovery;
    leaderRecovery = null;
    record('leader_loop_unresolved', { recovery_id: recovery.id, waited_s: Math.round((Date.now() - recovery.at) / 1000),
                                       ...teamCounts(recovery.team) });
    return true;
  },
  teammateInputs(agent) {
    // Called by the patched teammate round after each tool batch (and when the
    // model answers without tools). Returning [] leaves native behavior intact.
    const found = locateTeammate(agent);
    if (!found) {
      if (!unmatchedAgents.has(agent)) {
        unmatchedAgents.add(agent);
        record('teammate_delivery_unmatched', { managers: teamManagers.size });
      }
      return [];
    }
    const { team, member } = found;
    const inputs = [];
    const now = Date.now();
    const queue = team.pendingMessages?.get(member.agentId);
    if (queue && queue.length > 0) {
      // Native priority (shutdown/leader before peers), send order within one.
      const batch = queue.splice(0, queue.length).map((item, index) => ({ item, index }))
        .sort((a, b) => (a.item.priority - b.item.priority) || (a.index - b.index)).map(entry => entry.item);
      const handed = [];
      for (const item of batch) {
        const { text, marker } = labelTeamMessage(item);
        inputs.push({ kind: 'team_message', text });
        handed.push({ kind: 'team_message', mode: 'between_tool_calls', item, marker, drainedAt: now });
      }
      awaitObservation(member.name, handed);
      record('team_message_drained', { worker: member.name, mode: 'between_tool_calls', count: batch.length,
                                       items: batch.map(item => itemMeta(item, now)) });
    }
    const notes = taskNotices.get(member.agentId);
    if (notes && notes.length > 0) {
      const batch = notes.splice(0, notes.length);
      for (const note of batch) inputs.push({ kind: 'task_notice', text: note.text });
      awaitObservation(member.name, batch.map(note => ({ kind: 'task_notice', mode: 'between_tool_calls', note,
                                                         marker: note.marker, drainedAt: now })));
      record('task_notice_drained', { worker: member.name, mode: 'between_tool_calls',
                                      task_ids: batch.map(note => note.taskId),
                                      waited_ms: batch.map(note => now - note.queuedAt) });
    }
    const tokens = team.hasShutdownWork?.(member.name) ? unannouncedShutdown(team, member.name) : [];
    if (tokens.length > 0) {
      // Do not consume the shutdown request: native hands the request prompt to
      // a teammate only when it is idle. Ask the teammate to end its turn.
      const marker = `Runtime notice [shutdown ${randomBytes(4).toString('hex')}]:`;
      inputs.push({ kind: 'shutdown_notice', text: `${marker} the team leader has requested that you shut down. ` +
        'Finish the step you are on, then end your turn without calling more tools; the shutdown request will then be ' +
        'delivered for you to confirm.' });
      awaitObservation(member.name, [{ kind: 'shutdown_notice', mode: 'between_tool_calls', tokens, marker, drainedAt: now }]);
      record('shutdown_notice_drained', { worker: member.name });
    }
    return inputs;
  },
  stripLeadingAnalysis,
  stripLeadingAnalysis,
  enqueueLeaderMessage(queue, formatted) {
    leaderMessageQueue = queue;
    // Native escapes envelope delimiters inside teammate text. Only a real
    // runtime:checkin envelope can occupy this replaceable queue slot; other
    // messages (including failure notices) retain their contents and order.
    const checkinEnvelope = /<teammate_message from="runtime:checkin">\n[\s\S]*?<\/teammate_message>/g;
    if (formatted.includes('<teammate_message from="runtime:checkin">\n')) {
      for (let index = queue.length - 1; index >= 0; index--) {
        const remaining = queue[index].replace(checkinEnvelope, '').trim();
        if (!remaining) queue.splice(index, 1);
        else if (remaining !== queue[index].trim()) queue[index] = remaining;
      }
    }
    queue.push(formatted);
  },
  withPurpose(value, fn) { return purpose.run(value || 'auxiliary', fn); },
  // D10 option C: true exempts a valid proposal from native's explicit-report mark.
  policyProposalMessage(message, sender) {
    const proposal = policyProposalReports && (agentRules ? AGENT_RULES_CONTROL : rolePackageV1 ? PACKAGE_PROPOSAL_HEADER : POLICY_PROPOSAL_HEADER).test(firstLine(message));
    if (proposal) record('policy_proposal_not_report', { worker: sender });
    return proposal;
  },
  headers(baseURL, request) {
    if (requireCheckinQueue && !checkinQueueApplied) {
      throw new Error('Team check-in leader queue patch was not applied before a model request');
    }
    if (teammateDelivery && !teammateRoundApplied) {
      throw new Error('Teammate message delivery patch was not applied before a model request');
    }
    if (leaderLoopRecovery && !(leaderLoopApplied.detect && leaderLoopApplied.runner)) {
      throw new Error('Leader loop recovery patches were not applied before a model request');
    }
    if (policyProposalReports && !policyReportApplied) {
      throw new Error('Policy proposal reporting patch was not applied before a model request');
    }
    // Never forward these local routing fields to a non-relay endpoint.
    if (String(baseURL).replace(/\/$/, '') !== relayURL.replace(/\/$/, '')) throw new Error('Native model endpoint bypasses the task relay');
    const owner = teammateContext()?.agentName || rootOwner;
    const requestPurpose = purpose.getStore() || 'coding';
    if (owner !== rootOwner && requestPurpose === 'coding' && unobserved.has(owner)) {
      observeTeammateRequest(owner, request);
    }
    if (owner === rootOwner && requestPurpose === 'coding' && (notices.size || checkins?.pending() || leaderNotices.size)) {
      const text = requestText(request);
      checkins?.observe(text);
      for (const [marker, id] of leaderNotices) {
        if (text.includes(marker)) {
          leaderNotices.delete(marker);
          record('leader_loop_notice_observed', { recovery_id: id });
        }
      }
      for (const [id, notice] of notices) {
        // Match the real native envelope, not an ID quoted by a teammate or a
        // copy in a side-query summary that the coordinator has not yet read.
        if (!notice.observed && text.includes(`<teammate_message from="runtime:system">\n${id}:`)) {
          notice.observed = true;
          record('failure_notice_observed', { notice_id: id, worker: notice.worker });
        }
      }
    }
    return { 'X-Agent-Company-Owner': owner, 'X-Agent-Company-Purpose': requestPurpose };
  },
  agentStarting(agentId, agent) {
    // Runs inside native spawnAgent, outside its rollback: a listener must never throw there.
    for (const listener of agentStartListeners) {
      try {
        listener(agentId, agent);
      } catch (error) {
        record('agent_start_listener_error', { agent_id: agentId, error: String(error).slice(0, 500) });
      }
    }
  },
  spawnResult(callId, requestedName, name, ok) {
    record('worker_spawn_result', { tool_call_id: callId, requested_name: requestedName, worker: name, ok });
  },
  terminal(manager, agent, agentId, name, event, listTasks) {
    record('worker_terminal', { worker: name, agent_id: agentId, status: event.newStatus,
      user_cancelled: Boolean(event.roundCancelledByUser) });
    // Drained between tool calls but never sent (e.g. the round hit its turn or
    // time limit): as undelivered as native's queue, so report them with it.
    const unsent = (unobserved.get(name) || []).filter(entry => entry.kind === 'team_message').map(entry => entry.item);
    unobserved.delete(name);
    taskNotices.delete(agentId);
    idleLabels.delete(agentId);
    if (event.newStatus !== 'failed' || event.roundCancelledByUser) return;
    const id = `native-worker-failure-${++sequence}`;
    const error = String(agent.getError?.() || agent.getLastRoundError?.() || 'Unknown terminal failure').slice(0, 2000);
    const queued = [...unsent, ...(manager.pendingMessages.get(agentId) || [])].map(item => String(item.text).slice(0, 200));
    notices.set(id, { worker: name, observed: false });
    record('failure_notice_queued', { notice_id: id, worker: name, error });
    // Start the task snapshot before native unassignment. It is best effort:
    // native task ownership may concurrently change, so report that explicitly.
    const work = (async () => {
      let tasks;
      try {
        tasks = (await listTasks(manager.teamFile.name)).filter(task =>
          (task.owner === agentId || task.owner === name) && task.status !== 'completed')
          .map(task => ({ id: task.id, subject: task.subject, status: task.status }));
      } catch { tasks = null; }
      const text = `${id}: Native runtime reports worker ${name} FAILED. Error: ${error}\n` +
        `Unfinished owned tasks at terminal snapshot (ownership may already be released): ${JSON.stringify(tasks)}.\n` +
        `Undelivered queued instructions: ${JSON.stringify(queued)}.\n` +
        'This is a runtime status notice, not a worker completion report. Inspect task_list and delivery evidence; choose replacement, reassignment, cancellation, or explicitly accept partial delivery within the remaining task budget.';
      // ':' is disallowed in worker names, so no real teammate can own this
      // sender identity; native envelope escaping prevents embedded forgery.
      await manager.sendMessage('leader', text, 'runtime:system', `Worker ${name} failed`, true);
      record('failure_notice_persisted', { notice_id: id, worker: name });
    })().catch(error => {
      record('failure_notice_delivery_error', { notice_id: id, worker: name, error: String(error).slice(0, 2000) });
      // Native finalization must throw, not silently return a success record.
      throw error;
    });
    let pending = writes.get(manager);
    if (!pending) writes.set(manager, pending = []);
    pending.push(work);
    // The drain owns propagation; prevent an early unhandled rejection.
    work.catch(() => {});
  },
};
globalThis[Symbol.for('agent-company.native')] = bridge;

// Provenance only: patchNativeSource throws on shell anchor drift, but sources
// without a shell service (e.g. synthetic chunks) are not an error here.
let shellDrainApplied = false;
registerHooks({ load(url, context, nextLoad) {
  const loaded = nextLoad(url, context);
  if (url.startsWith('file:') && path.dirname(fileURLToPath(url)) === chunks && loaded.format === 'module') {
    const source = typeof loaded.source === 'string' ? loaded.source : Buffer.from(loaded.source).toString('utf8');
    const requiredQueue = requireCheckinQueue && fileURLToPath(url) === checkinQueuePath;
    const requiredRound = teammateDelivery && fileURLToPath(url) === teammateRoundPath;
    const leaderDetect = leaderLoopRecovery && fileURLToPath(url) === leaderLoopPaths.detect;
    const leaderRunner = leaderLoopRecovery && fileURLToPath(url) === leaderLoopPaths.exit;
    const policyReport = policyProposalReports && fileURLToPath(url) === policyReportPath;
    const patched = patchNativeSource(source, { requireCheckinQueue: requiredQueue, teammateDelivery: requiredRound,
                                                leaderLoopRecovery: leaderDetect || leaderRunner,
                                                policyProposalReports: policyReport });
    if (source.includes('var ShellExecutionService = class')) shellDrainApplied = true;
    if (source.includes('var InProcessBackend = class')) {
      agentStartApplied = true;
      record('agent_start_patch_applied', { module: path.basename(fileURLToPath(url)) });
    }
    if (policyReport) {
      policyReportApplied = true;
      record('policy_report_patch_applied', { module: path.basename(policyReportPath) });
    }
    for (const [part, applies] of [['detect', leaderDetect], ['runner', leaderRunner]]) {
      if (!applies) continue;
      leaderLoopApplied[part] = true;
      record('leader_loop_patch_applied', { part, module: path.basename(fileURLToPath(url)) });
    }
    if (requiredQueue) {
      checkinQueueApplied = true;
      record('checkin_queue_patch_applied', { module: path.basename(checkinQueuePath) });
    }
    if (requiredRound) {
      teammateRoundApplied = true;
      record('teammate_delivery_patch_applied', { module: path.basename(teammateRoundPath), mode: deliveryMode });
    }
    return { ...loaded, source: patched };
  }
  return loaded;
} });
let manager;
let nativeConfig;
for (const file of fs.readdirSync(chunks).filter(name => name.endsWith('.js'))) {
  const absolute = path.join(chunks, file);
  const source = fs.readFileSync(absolute, 'utf8');
  if (source.includes('var teammateIdentityStore = new AsyncLocalStorage()')) {
    teammateContext = (await import(pathToFileURL(absolute).href)).getTeammateContext;
  }
  if (source.includes('var TeamManager = class')) {
    const native = await import(pathToFileURL(absolute).href);
    manager = native.TeamManager;
    nativeConfig = native.Config;
  }
}
if (!teammateContext || !manager) throw new Error('Native teammate identity or manager unavailable');
if (process.env.AGENT_COMPANY_STOP_HOOK_REASON !== undefined) {
  const { installNativeStopHook } = await import('./qwen_stop_hook.mjs');
  installNativeStopHook(nativeConfig, {
    owner: rootOwner,
    reason: process.env.AGENT_COMPANY_STOP_HOOK_REASON,
    deadlineEpochMs: Number(process.env.AGENT_COMPANY_TASK_DEADLINE_EPOCH_MS),
    hasPendingLeaderMessages: () => Boolean(leaderMessageQueue?.length),
    record,
  });
}
if (rootOwner === 'master') {
  // Team creation reads this getter, not settings.json's agents.team. Use an
  // unbounded roster: completed members stay there. The relay enforces ten
  // active dispatches using native terminal events, not historical roster size.
  // Retain other settings and do not mutate the underlying config object.
  const getAgentsSettings = nativeConfig?.prototype.getAgentsSettings;
  if (typeof getAgentsSettings !== 'function') throw new Error('Native team configuration unavailable');
  nativeConfig.prototype.getAgentsSettings = function () {
    const settings = getAgentsSettings.call(this);
    return { ...settings, team: { ...settings?.team, maxTeammates: Infinity } };
  };
}
const drain = manager.prototype.drainLeaderInbox;
manager.prototype.drainLeaderInbox = async function () {
  // A worker becomes idle before fireAndForget(reportFinalAnswer) persists its
  // automatic text. Wait BEFORE the final inbox snapshot, not only in cleanup
  // after leader exit. Native reports async failures to the leader callback.
  // As in native cleanup, include work scheduled by earlier pending work.
  await Promise.all(writes.get(this) || []);
  while (this.pendingAsyncWork?.size > 0) {
    await Promise.allSettled([...this.pendingAsyncWork]);
  }
  return drain.call(this);
};
if (teammateDelivery) {
  if (!teammateRoundApplied) throw new Error('Teammate message delivery patch was not applied to the loaded round loop');
  installTeammateDelivery(manager);
}
if (policyProposalReports && !policyReportApplied) {
  throw new Error('Policy proposal reporting patch was not applied to the loaded team manager');
}
if (rolePolicyMode !== 'off') installBroadcastRecord(manager);
if (agentRules) {
  // agent_rules_v1 (plan v4 section 10): an unowned task is never claimed by whichever
  // teammate happens to be idle. Owned tasks keep their native dispatch path.
  manager.prototype.tryAutoClaimTask = async function () {};
  record('auto_claim_disabled', { protocol: policyProtocol });
}
record('compatibility_loaded', { root_owner: rootOwner, team_message_delivery: deliveryMode,
  runtime_disclosure: runtimeDisclosure, leader_loop_recovery: leaderLoopRecovery, tool_call_cap: toolCallCap,
  teammate_limit: rootOwner === 'master' ? teammateLimit : null,
  role_policy_mode: rolePolicyMode,
  repairs: [...(rootOwner === 'master' ? ['automatic_report_final_drain', 'leader_inbox_current_callback'] : []), 'terminal_notice', 'summary_prefix', 'request_identity', ...(checkinQueueApplied ? ['checkin_queue_slot'] : []),
            ...(teammateRoundApplied ? ['teammate_delivery_between_tool_calls'] : []),
            ...(policyReportApplied ? ['policy_proposal_not_report'] : []),
            ...(shellDrainApplied ? ['shell_output_drain'] : []),
            ...(agentStartApplied ? ['teammate_start_tool_listener'] : []),
            ...(rolePolicyMode !== 'off' ? ['broadcast_recipient_record'] : [])] });

// Per-recipient broadcast outcomes as metadata (no message text). The recipient
// list mirrors native broadcast(); "accepted" means native sendMessage resolved
// (queued or written), not that the recipient made a model request.
function installBroadcastRecord(TeamManager) {
  const nativeBroadcast = TeamManager.prototype.broadcast;
  if (typeof nativeBroadcast !== 'function') throw new Error('Native team broadcast unavailable');
  TeamManager.prototype.broadcast = async function (message, fromName) {
    const from = String(fromName);
    const recipients = this.teamFile.members.filter(m => m.name.toLowerCase() !== from.toLowerCase()).map(m => m.name);
    if (from.toLowerCase() !== 'leader') recipients.push('leader');
    const base = { from, global_policy: firstLine(message).startsWith('GLOBAL-POLICY'),
      ...(agentRules ? { hash: messageHash(message) } : {}) };
    let result;
    try {
      const inner = agentRules ? broadcastInner.begin(from, message, recipients.length) : null;
      try {
        result = await nativeBroadcast.call(this, message, fromName);
      } finally {
        if (inner) broadcastInner.end(inner);
      }
    } catch (error) {
      record('team_broadcast', { ...base, total: null, error: String(error?.message || error).slice(0, 200),
                                 recipients: recipients.map(name => ({ name, outcome: 'unknown' })) });
      throw error;
    }
    const failed = new Set(result?.failedRecipients || []);
    const consistent = result?.total === recipients.length;
    record('team_broadcast', { ...base, total: result?.total ?? null,
      recipients: recipients.map(name => ({ name, outcome: !consistent ? 'unknown' : failed.has(name) ? 'failed' : 'accepted' })) });
    return result;
  };
}

function installTeammateDelivery(TeamManager) {
  const nativeSetCallback = TeamManager.prototype.setLeaderMessageCallback;
  TeamManager.prototype.setLeaderMessageCallback = function (callback) {
    // Bound once the leader creates its team, before any teammate is spawned.
    if (callback) {
      requireShutdownLedgers(this);
      teamManagers.add(this);
    }
    return nativeSetCallback.call(this, callback);
  };
  const nativeSpawn = TeamManager.prototype.spawnTeammate;
  TeamManager.prototype.spawnTeammate = async function (...args) {
    requireShutdownLedgers(this);
    teamManagers.add(this);
    return nativeSpawn.apply(this, args);
  };

  const nativeSend = TeamManager.prototype.sendMessage;
  TeamManager.prototype.sendMessage = async function (toName, message, from, summary, automatic = false) {
    const member = typeof toName === 'string' ? memberByName(this, toName) : null;
    const queue = member ? this.pendingMessages?.get(member.agentId) : null;
    if (member && queue) trackQueue(this, member, queue);
    if (!agentRules || automatic || broadcastInner.consume(from, message)) {
      return nativeSend.call(this, toName, message, from, summary, automatic);
    }
    // Native execution of a model send, recorded even when native then aborts the
    // sender (shutdown approval), so its final reply can still be counted.
    const base = { from: from ?? null, to: String(toName), hash: messageHash(message) };
    let result;
    try {
      result = await nativeSend.call(this, toName, message, from, summary, automatic);
    } catch (error) {
      record('team_message_sent', { ...base, ok: false });
      throw error;
    }
    record('team_message_sent', { ...base, ok: true });
    return result;
  };

  const nativeEnqueue = TeamManager.prototype.enqueueWithIdentity;
  TeamManager.prototype.enqueueWithIdentity = function (agentId, agent, message) {
    if (agentRules && typeof message === 'string' && NATIVE_SHUTDOWN_TEXT.test(message)) {
      // Native hands an idle teammate the plain shutdown text with no envelope.
      const name = this.teamFile?.members?.find(member => member.agentId === agentId)?.name;
      if (name) awaitObservation(name, [{ kind: 'shutdown_notice', mode: 'idle', marker: message, tokens: [] }]);
    }
    // Native idle flush generates its own envelope nonce; keep it so the
    // delivery can be observed like a between-tool-calls one.
    const label = parseNativeEnvelope(message);
    if (label) {
      let labels = idleLabels.get(agentId);
      if (!labels) idleLabels.set(agentId, labels = []);
      labels.push(label);
    }
    return nativeEnqueue.call(this, agentId, agent, message);
  };

  const nativeFlush = TeamManager.prototype.flushNextMessage;
  TeamManager.prototype.flushNextMessage = async function (agentId, agentName) {
    const agent = this.getAgentFromBackend?.(agentId);
    if (agent && agent.getStatus?.() === 'idle') {
      requeueUnobserved(this, agentId, agentName);
      const notes = taskNotices.get(agentId);
      if (notes && notes.length > 0) {
        // Assigned after the owner's last step but before it went idle, so no
        // step remained to carry it, and native never auto-claims an owned
        // task. Native would have run the assignment before idling: run it now.
        const batch = notes.splice(0, notes.length);
        const now = Date.now();
        this.enqueueWithIdentity(agentId, agent, batch.map(note => note.text).join('\n\n'));
        awaitObservation(agentName, batch.map(note => ({ kind: 'task_notice', mode: 'idle', note, marker: note.marker,
                                                         drainedAt: now })));
        record('task_notice_drained', { worker: agentName, mode: 'idle', task_ids: batch.map(note => note.taskId),
                                        waited_ms: batch.map(note => now - note.queuedAt) });
        return;
      }
    }
    const queue = this.pendingMessages?.get(agentId);
    const before = queue ? [...queue] : [];
    const result = await nativeFlush.call(this, agentId, agentName);
    const after = new Set(queue || []);
    const now = Date.now();
    const drained = before.filter(item => !after.has(item));
    if (drained.length > 0) {
      // Native starts a new round with it, so that round's first request carries it.
      const labels = idleLabels.get(agentId) || [];
      const entries = [];
      for (const item of drained) {
        const index = labels.findIndex(label => label.from === item.from && label.text === item.text);
        const marker = index >= 0 ? labels.splice(index, 1)[0].marker : (item.from ? null : String(item.text));
        if (marker) entries.push({ kind: 'team_message', mode: 'idle_flush', item, marker, drainedAt: now });
        else record('team_message_unobservable', { worker: agentName, message_id: itemMeta(item, now).message_id });
      }
      awaitObservation(agentName, entries);
      record('team_message_drained', { worker: agentName, mode: 'idle_flush', count: drained.length,
                                       items: drained.map(item => itemMeta(item, now)) });
    }
    return result;
  };

  const nativeDispatch = TeamManager.prototype.dispatchAssignedTask;
  TeamManager.prototype.dispatchAssignedTask = async function (task) {
    // Native queues a whole new round behind a busy owner's current one, which
    // then consumes the owner's next idle transition. Deliver it mid-round, or
    // at the idle transition if the owner has already taken its last step.
    if (task?.status === 'in_progress' && task.owner) {
      const member = memberByName(this, task.owner);
      const agent = member ? this.getAgentFromBackend?.(member.agentId) : null;
      if (agent && agent.getStatus?.() === 'running' && !this.hasShutdownWork?.(member.name)) {
        let notes = taskNotices.get(member.agentId);
        if (!notes) taskNotices.set(member.agentId, notes = []);
        const prompt = this.buildTaskPrompt(task);
        notes.push({ taskId: task.id, queuedAt: Date.now(), marker: prompt.match(/<task_content_[0-9a-f]+>/)?.[0] || prompt,
                     text: 'Runtime notice: this task-board assignment arrived while you were working.\n' + prompt });
        record('task_notice_queued', { worker: member.name, task_id: task.id });
        return true;
      }
    }
    return nativeDispatch.call(this, task);
  };
}

// ----- Opt-in runtime team check-ins -----------------------------------------
// Gives the coordination-only master periodic decision opportunities with an
// observational snapshot. Native waits only re-check every 120 s and never call
// the leader's model unless a message arrives; this uses its native callback
// and wake-up path, with one replaceable pending check-in. It never spawns,
// stops or reassigns, and leaves worker/failure inbox messages unchanged.
if (requireCheckinQueue) {
  // Check-ins observe teammates' tool calls from their first turn (same chunk as TeamManager).
  if (!agentStartApplied) throw new Error('Teammate start patch was not applied to the loaded in-process backend');
  checkins = await setupCheckins(manager);
}

async function setupCheckins(TeamManager) {
  const { CheckinScheduler, CheckinTotals, compactCheckin, containerMemory, formatCheckin, runCheckin, SENDER } = await import(new URL('./qwen_team_checkin.mjs', import.meta.url).href);
  const { execFile } = await import('node:child_process');
  // agent_rules_v1 check-ins carry no POLICY-REVIEW reminder (plan v4 section 9).
  const checkinText = view => agentRules ? formatCheckin(view, checkinSeconds) : packageCheckin(view);
  function packageCheckin(view) {
    let value = formatCheckin(view, checkinSeconds);
    if (!rolePackage) return value;
    const first = value.indexOf('\n');
    const header = "{checkin_id}: Runtime team check-in ({reason}). The observations below are not completion reports or a verdict that anyone is stuck. Unknown means the source was unavailable. A mode-specific policy-review reminder follows the observations.".replace('{checkin_id}', view.id).replace('{reason}', view.reason);
    value = header + (first >= 0 ? value.slice(first) : '');
    // The legacy renderer owns this final template suffix. Never replace text
    // inside an observation (for example a commit subject quoting the suffix).
    const suffix = ' No reply is needed if nothing changes.';
    if (value.endsWith(suffix)) value = value.slice(0, -suffix.length);
    return value + '\n\n' + (rolePolicyMode === 'propose' ? "Review recent collaboration events and decide whether to change C or a role policy. Use an allowed control for any change; if you make no change, write one line in your reply text, not a message to teammates: POLICY-REVIEW C v<k>: or POLICY-REVIEW role:<role_id> v<k>:, then no issue, issue observed or cannot judge yet, and the reason, including unavailable evidence when relevant." : "Review recent collaboration events and write one line in your reply text, not a message to teammates: POLICY-REVIEW C v<k>: or POLICY-REVIEW role:<role_id> v<k>:, then no issue, issue observed or cannot judge yet, and your observations or unavailable evidence. This mode keeps C and role policies fixed.");
  }

  let listTasks;
  for (const file of fs.readdirSync(chunks).filter(name => name.endsWith('.js'))) {
    const absolute = path.join(chunks, file);
    if (fs.readFileSync(absolute, 'utf8').includes('async function listTasks(teamName, filters)')) {
      listTasks = (await import(pathToFileURL(absolute).href)).listTasks;
    }
  }
  const startMs = Number(process.env.AGENT_COMPANY_TASK_STARTED_EPOCH_MS);
  const deadlineMs = Number(process.env.AGENT_COMPANY_TASK_DEADLINE_EPOCH_MS);
  const repo = process.env.AGENT_COMPANY_DELIVERY_REPO;
  const statusFile = process.env.AGENT_COMPANY_STATUS_FILE;
  if (!(deadlineMs > startMs) || !repo || !statusFile || typeof listTasks !== 'function') {
    throw new Error('Team check-ins require task times, delivery repository, status file and native task board');
  }
  const intervalMs = checkinSeconds * 1000;
  // File observations are refreshed independently of model wake-ups. With the
  // normal 30-minute check-in, the file is refreshed once per minute.
  const refreshMs = Math.max(250, Math.min(60_000, intervalMs / 4));
  // Opt-in disclosure: data in the file, but still model-visible input when read.
  const runtimeNote = runtimeDisclosure ? (
    `This file is refreshed about every ${refreshMs / 1000} s; reading it again within that interval returns the same snapshot. ` +
    'To wait, end your turn without tool calls: while a teammate is working, the runtime waits without model calls ' +
    'and wakes you for a teammate message, an automatic report when a teammate ends its work turn, or a check-in ' +
    `(about every ${checkinSeconds % 60 ? `${checkinSeconds} s` : `${checkinSeconds / 60} minutes`} and once each at half ` +
    'and three quarters of the wall time). If no teammate is working and no message is queued, ending your turn ' +
    'ends the run. ' +
    'Qwen stops a turn at the fifth consecutive identical tool call, including read_file of this file; for the ' +
    'master that ends the whole run' + (leaderLoopRecovery
      ? ', except that in this run, when the fifth identical read of this file is the only tool call in its ' +
        'response and a teammate is running, only your turn ends and you wait.'
      : '.')) : null;
  const scheduler = new CheckinScheduler({ intervalMs, startMs, deadlineMs,
    minSpacingMs: Math.min(5 * 60_000, intervalMs / 2), finalGuardMs: Math.min(60_000, 0.02 * (deadlineMs - startMs)) });
  const tokenIndex = process.argv.indexOf('--openai-api-key');
  const relayToken = tokenIndex >= 0 ? process.argv[tokenIndex + 1] : null;
  const managers = new Set();
  const tools = new Map();  // Call announcements and result delivery, not execution timing.
  const terminal = new Set(['completed', 'failed', 'cancelled']);
  const editTools = new Set(['edit', 'write_file']);
  let timer = null;
  let polling = false;
  let lastSnapshotMs = 0;
  let lastSkipReason = null;
  // agent_rules_v1: lesson and board totals at the previous check-in the master saw (plan v4 section 9).
  const checkinTotals = new CheckinTotals();

  function attach(team, member) {
    observeTools(member.agentId, team.getAgentFromBackend?.(member.agentId)?.getEventEmitter?.());
  }
  agentStartListeners.add((agentId, agent) => {
    // Native registers an id only when no live agent holds it, so every start is a new
    // instance (a retry after a failed spawn reuses the id with a new emitter).
    tools.delete(agentId);
    observeTools(agentId, agent?.getEventEmitter?.());
  });

  function observeTools(agentId, emitter) {
    if (tools.has(agentId)) return;
    if (!emitter) return;  // Reported as unknown tool activity.
    const state = { pending: new Map(), lastResult: null, edits: 0, commands: 0, calls: 0 };
    tools.set(agentId, state);
    emitter.on('tool_call', event => {
      state.calls++;
      if (editTools.has(event.name)) state.edits++;
      if (event.name === 'run_shell_command') state.commands++;
      state.pending.set(event.callId, { name: event.name, announcedMs: event.timestamp || Date.now() });
    });
    emitter.on('tool_result', event => {
      const call = state.pending.get(event.callId);
      if (!call) return;
      state.pending.delete(event.callId);
      state.lastResult = { name: call.name, receivedMs: Date.now() };
    });
  }

  const spawn = TeamManager.prototype.spawnTeammate;
  TeamManager.prototype.spawnTeammate = async function (...args) {
    const result = await spawn.apply(this, args);
    managers.add(this);
    for (const member of this.teamFile.members) attach(this, member);
    if (!timer && !scheduler.stopped) {
      // Qwen may bootstrap a child CLI with the same imports. Only the process
      // owning an actual TeamManager refreshes the file; a parent bootstrap
      // must not overwrite the child's roster with an empty snapshot.
      timer = setInterval(() => { void poll(); }, Math.max(250, Math.min(15_000, intervalMs / 4)));
      timer.unref();  // Observations must never keep a finished CLI alive.
    }
    // Refresh without holding up the native spawn response. poll owns its
    // concurrency guard, lifecycle checks and error reporting.
    void poll(true);
    return result;
  };

  const cleanup = TeamManager.prototype.cleanup;
  TeamManager.prototype.cleanup = async function (...args) {
    // Mark shutdown before native cleanup awaits: an in-flight Git/relay sample
    // must not deliver a new leader message once cleanup has started.
    if (managers.has(this)) {
      scheduler.stop('cleanup');
      if (timer) clearInterval(timer);
    }
    return cleanup.apply(this, args);
  };

  function git(args, cwd) {
    return new Promise((resolve, reject) => execFile('git', ['--no-optional-locks', '-C', cwd, ...args],
      { timeout: 5000, env: { ...process.env, GIT_OPTIONAL_LOCKS: '0' } },
      (error, stdout) => (error ? reject(error) : resolve(String(stdout).trim()))));
  }

  // Delivery HEAD at launch: "new" commits are those after it, whatever their timestamps.
  const baseline = await git(['rev-parse', 'HEAD'], repo).catch(() => null);
  const sampledHeads = new Map();
  const sampledGitFailures = new Map();

  async function commits(cwd, nowMs, startHead = baseline, countKey = 'commits_since_launch') {
    const count = startHead ? Number(await git(['rev-list', '--count', `${startHead}..HEAD`], cwd)) : null;
    const out = { [countKey]: count, last_commit_ago_s: null, last_commit_subject: null, head: null };
    if (count > 0) {
      const [sha, seconds, ...subject] = (await git(['log', '-1', '--format=%H%x09%ct%x09%s'], cwd)).split('\t');
      Object.assign(out, { head: sha, last_commit_ago_s: Math.max(0, Math.round(nowMs / 1000 - Number(seconds))),
        last_commit_subject: subject.join('\t').slice(0, 100) });
    }
    if (rolePackage) {
      out.head = await git(['rev-parse', 'HEAD'], cwd);
      const previous = sampledHeads.get(cwd) ?? startHead;
      out.commit_samples = [];
      out.sampling_incomplete = [];
      if (!previous) out.sampling_incomplete.push('missing_previous_head');
      else {
        try {
          let rangeStart = previous;
          try {
            await git(['merge-base', '--is-ancestor', previous, out.head], cwd);
          } catch (error) {
            // Exit 1 means readable histories diverged; other failures must
            // retain the old sample head so the missing interval is retried.
            if (error.code !== 1 || error.killed || error.signal) throw error;
            out.sampling_incomplete.push('nonancestor_previous_head');
            try {
              rangeStart = await git(['merge-base', previous, out.head], cwd);
            } catch (error) {
              if (error.code !== 1 || error.killed || error.signal) throw error;
              // Unrelated histories have no recoverable interval. Emit the
              // current head with an explicit gap before re-anchoring.
              out.sampling_incomplete.push('no_common_ancestor');
              rangeStart = null;
            }
          }
          const range = rangeStart
            ? await git(['rev-list', '--reverse', '--topo-order', `${rangeStart}..${out.head}`], cwd) : '';
          for (const sha of range.split('\n').filter(Boolean)) {
            // Metadata only. -m includes merge-parent differences; explicit
            // no-renames keeps the status/path pairs unambiguous. Deduplicate
            // paths from multiple parents without inferring patch authorship.
            const changes = (await git(['diff-tree', '--no-ext-diff', '--no-textconv', '--no-renames',
              '--no-commit-id', '--name-status', '-r', '-m', '--root', '-z', sha], cwd)).split('\0');
            const row = {commit: sha, changed_paths: [], added_paths: [], deleted_paths: [], committed_at: null};
            try {
              const seconds = Number(await git(['show', '-s', '--format=%ct', sha], cwd));
              if (Number.isFinite(seconds) && seconds > 0) row.committed_at = seconds;
            } catch {}
            for (let i = 0; i + 1 < changes.length; i += 2) {
              if (!changes[i + 1]) continue;
              row.changed_paths.push(changes[i + 1]);
              if (changes[i] === 'D') row.deleted_paths.push(changes[i + 1]);
              if (changes[i] === 'A') row.added_paths.push(changes[i + 1]);
            }
            for (const key of ['changed_paths', 'added_paths', 'deleted_paths']) row[key] = [...new Set(row[key])];
            out.commit_samples.push(row);
          }
        } catch {
          out.sampling_incomplete.push('unreadable_commit_range');
        }
      }
      for (const key of ['changed_paths', 'added_paths', 'deleted_paths']) {
        out[key] = [...new Set(out.commit_samples.flatMap(row => row[key]))];
      }
      out.tracked_caches = (await git(['ls-files', '-z', '--', '*.pyc', '*.pyo', '**/__pycache__/**'], cwd))
        .split('\0').filter(Boolean);
    }
    return out;
  }

  async function worktree(cwd, nowMs, creationHead) {
    // Creation-relative ancestry is not authorship: later merges/cherry-picks
    // can include changes produced by another teammate.
    const out = await commits(cwd, nowMs, creationHead, 'commits_since_worker_creation');
    out.creation_head = creationHead;
    const status = await git(['status', '--porcelain', '--untracked-files=normal'], cwd);
    out.uncommitted_paths = status ? status.split('\n').length : 0;
    if (rolePackage) {
      // Untracked build products are not unfinished tracked work. NUL output
      // keeps filenames intact; a rename/copy has a second pathname, not a
      // second changed entry.
      const tracked = (await git(['status', '--porcelain', '-z', '--untracked-files=no'], cwd))
        .split('\0').filter(Boolean);
      out.tracked_uncommitted_paths = 0;
      for (let index = 0; index < tracked.length; index++) {
        out.tracked_uncommitted_paths++;
        if (/[RC]/.test(tracked[index].slice(0, 2))) index++;
      }
    }
    if (path.resolve(cwd) !== path.resolve(repo)) {
      // Object ancestry only: cherry-pick changes SHA while delivering the same
      // patch. Do not call this "unlanded work" or infer integration failure.
      const deliveryHead = await git(['rev-parse', 'HEAD'], repo);
      out.commits_not_in_delivery_history = Number(await git(['rev-list', '--count', 'HEAD', '--not', deliveryHead], cwd));
    }
    return out;
  }

  function emitCollaboration(record) {
    return new Promise((resolve, reject) => {
      // false means backpressure, not failure. Only the write callback confirms
      // this metadata was emitted; a failed sample must not consume its range.
      process.stderr.write('Agent Company collaboration: ' + JSON.stringify(record) + '\n',
        error => error ? reject(error) : resolve());
    });
  }

  async function emitGitMetadata(actor, cwd, metadata, sampleAt, emit = emitCollaboration) {
    const incomplete = [...new Set([...(metadata?.sampling_incomplete || []),
      ...(sampledGitFailures.get(cwd) || []), ...(metadata ? [] : ['git_read_failed'])])];
    const fresh = Boolean(metadata?.commit_samples?.length);
    const samples = fresh ? metadata.commit_samples : [{commit: metadata?.head ?? null}];
    try {
      // new_commit/changed_paths/committed_at: one row per commit first read in this
      // interval, with that commit's own paths only (agent_rules_v1 commit triggers).
      for (const sample of samples) await emit({ kind: 'git', sample: sampleAt, actor, git: { commit: sample.commit,
        new_commit: fresh, changed_paths: sample.changed_paths || [], committed_at: sample.committed_at ?? null,
        paths: [...new Set([...(sample.changed_paths || []), ...(sample.commit === metadata?.head ? metadata?.tracked_caches || [] : [])])],
        deleted_paths: sample.deleted_paths || [], added_paths: sample.added_paths || [],
        sampling_incomplete: incomplete,
        new_commit_count: metadata?.commits_since_worker_creation ?? metadata?.commits_since_launch ?? null,
        uncommitted_paths: metadata?.uncommitted_paths ?? null,
        tracked_uncommitted_paths: metadata?.tracked_uncommitted_paths ?? null,
        commits_not_in_delivery_history: metadata?.commits_not_in_delivery_history ?? null } });
      // A successfully emitted history gap establishes a new baseline. Read
      // or emission failures never consume the interval, including mixed cases.
      if (metadata && (metadata.sampling_incomplete || []).every(reason =>
        ['missing_previous_head', 'nonancestor_previous_head', 'no_common_ancestor'].includes(reason))) {
        sampledHeads.set(cwd, metadata.head);
        sampledGitFailures.delete(cwd);
      }
    } catch (error) {
      sampledGitFailures.set(cwd, ['git_emit_failed']);
      if (metadata) metadata.sampling_incomplete = [...new Set([...incomplete, 'git_emit_failed'])];
      throw error;
    }
  }

  async function relayStatus() {
    const url = relayURL.replace(/\/$/, '') + '/agent-company/team-status';
    const response = await fetch(url, { headers: { Authorization: 'Bearer ' + relayToken },
                                        signal: AbortSignal.timeout(3000) });
    if (!response.ok) throw new Error('relay status HTTP ' + response.status);
    return response.json();
  }

  async function snapshot(team, decision, nowMs) {
    const unknown = [];
    // Native membership can change while Git/relay reads await. Keep each
    // sampled actor tied to the worktree read for this snapshot, off-model.
    const members = (team?.teamFile.members || []).map(({name, agentId, cwd, joinedAt}) =>
      ({name, agentId, cwd, joinedAt}));
    const guard = async (label, work) => { try { return await work(); } catch { unknown.push(label); return null; } };
    const tasks = team ? await guard('task_board', () => listTasks(team.teamFile.name)) : null;
    const relay = relayToken ? await guard('relay', relayStatus) : (unknown.push('relay'), null);
    const delivery = await guard('delivery_git', () => commits(repo, nowMs));
    const workers = [];
    for (const member of members) {
      const agent = team.getAgentFromBackend?.(member.agentId);
      const state = tools.get(member.agentId);
      if (!state) unknown.push(`tools:${member.name}`);
      const pending = state ? [...state.pending.values()].sort((a, b) => a.announcedMs - b.announcedMs) : null;
      const creationHead = relay?.workers?.[member.name]?.creation_head ?? null;
      if (!creationHead) unknown.push(`creation_head:${member.name}`);
      const last = team.lastActivityAt?.get(member.agentId);
      workers.push({
        name: member.name, status: agent?.getStatus?.() ?? 'unknown',
        joined_ago_s: Number.isFinite(member.joinedAt) ? Math.max(0, Math.round((nowMs - member.joinedAt) / 1000)) : null,
        last_activity_ago_s: last ? Math.max(0, Math.round((nowMs - last) / 1000)) : null,
        queued_messages: team.pendingMessages ? (team.pendingMessages.get(member.agentId) || []).length : null,
        shutdown_pending: Boolean(team.hasShutdownWork?.(member.name)),
        pending_tools: pending?.map(call => ({ name: call.name, announced_ago_s: Math.max(0, Math.round((nowMs - call.announcedMs) / 1000)) })) ?? null,
        last_tool_result: state?.lastResult ? { name: state.lastResult.name, received_ago_s: Math.max(0, Math.round((nowMs - state.lastResult.receivedMs) / 1000)) } : null,
        tool_counts: state ? { edits: state.edits, commands: state.commands, calls: state.calls } : null,
        tasks: tasks ? tasks.filter(task => task.owner === member.agentId || task.owner === member.name)
          .map(task => ({ id: task.id, subject: String(task.subject || '').slice(0, 80), status: task.status })) : null,
        git: member.cwd ? await guard(`git:${member.name}`, () => worktree(member.cwd, nowMs, creationHead))
          : (unknown.push(`git:${member.name}`), null),
        relay: relay?.roles?.[member.name] ?? null,
      });
    }
    const completedMs = Date.now();
    if (rolePackage) {
      const names = new Map(members.map(member => [member.agentId, member.name]));
      for (const task of tasks || []) {
        const match = String(task.description || '').match(/^(?:writing scope|write scope|scope|paths):\s*([^\n]+)/im);
        const paths = match ? match[1].split(',').map(value => value.trim().replace(/^[`"]|[`"]$/g, '')) : null;
        await emitCollaboration({ kind: 'task', sample: completedMs, task: { task_id: String(task.id), status: task.status,
          owner: names.get(task.owner) || task.owner || null,
          paths: paths?.every(value => value && !value.startsWith('/') && !value.split('/').includes('..')) ? paths : null } });
      }
      for (const [actor, cwd, metadata] of [['master', repo, delivery],
        ...workers.map((worker, index) => [worker.name, members[index].cwd, worker.git])]) {
        try { await emitGitMetadata(actor, cwd, metadata, completedMs); }
        catch { unknown.push(`git_emit:${actor}`); }
      }
    }
    if (agentRules && team) {
      // Teammates' request tails list who is active and on what (plan v4 section 8.2):
      // names, native status and in-progress task subjects only.
      try {
        await emitCollaboration({ kind: 'team', sample: completedMs, team: { members: workers.map(worker => ({
          name: worker.name, status: worker.status, shutdown_pending: worker.shutdown_pending,
          // null when the task board could not be read: unknown, not "no tasks".
          tasks: worker.tasks == null ? null : worker.tasks.filter(task => task.status === 'in_progress')
            .map(task => ({ id: String(task.id), subject: String(task.subject || '').slice(0, 60) })) })) } });
      } catch { unknown.push('team_emit'); }
    }
    return {
      ...(runtimeNote ? { runtime_note: runtimeNote } : {}),
      id: decision.id || decision.pendingId || null,
      at: new Date(completedMs).toISOString(), sample_started_at: new Date(nowMs).toISOString(),
      refresh_interval_s: refreshMs / 1000, team_created: Boolean(team),
      reason: decision.reason || 'status_refresh',
      milestones: decision.milestones || [],
      elapsed_s: Math.round((completedMs - startMs) / 1000), remaining_s: Math.max(0, Math.round((deadlineMs - completedMs) / 1000)),
      slots: { used: relay?.active_teammates ?? null, max: relay?.teammate_limit ?? null },
      ...(agentRules ? { container_memory: containerMemory() } : {}),
      unassigned_open_tasks: tasks ? tasks.filter(task => !task.owner && task.status !== 'completed').length : null,
      delivery, workers, master_relay: relay?.roles?.master ?? null, unknown, status_file: statusFile,
      ...(agentRules ? { team_observations: relay ? {
        pending_shutdowns: relay.pending_shutdowns ?? null, idle_workers: relay.idle_workers ?? null,
        message_pairs: relay.message_pairs ?? null, rule_rates: relay.rule_rates ?? null,
        pending_proposals: relay.pending_proposals ?? null, recent_self_changes: relay.recent_self_changes ?? null,
        lessons_total: relay.lessons_total ?? null, board_total: relay.board_total ?? null,
        ...checkinTotals.newSince(relay),
      } : null } : {}),
    };
  }

  function publish(view) {
    try {
      // Outside the delivery repository, so it can never enter a submission.
      fs.mkdirSync(path.dirname(statusFile), { recursive: true });
      fs.writeFileSync(statusFile + '.tmp', JSON.stringify(view, null, 2));
      fs.renameSync(statusFile + '.tmp', statusFile);
      lastSnapshotMs = Date.now();
    } catch (error) {
      record('checkin_status_write_error', { error: String(error).slice(0, 500) });
    }
  }

  function teamState(team) {
    if (!team) return { active: false, windingDown: false };
    const alive = team.teamFile.members.filter(member => {
      const status = team.getAgentFromBackend?.(member.agentId)?.getStatus?.();
      return status && !terminal.has(status);
    });
    return { active: team.hasActiveTeammates(),
             windingDown: alive.length > 0 && alive.every(member => team.hasShutdownWork?.(member.name)) };
  }

  async function poll(forceRefresh = false) {
    if (polling) return;
    polling = true;
    try {
      const team = [...managers].pop();
      const { decision, view } = await runCheckin({
        scheduler, now: Date.now, teamState: () => teamState(team),
        snapshot: (decision, nowMs) => snapshot(team, decision, nowMs), publish,
        refreshDue: forceRefresh || Date.now() - lastSnapshotMs >= refreshMs,
        send: (view) => {
          const callback = team.leaderMessageCallback;
          if (typeof callback !== 'function') throw new Error('Native leader callback is unavailable');
          const message = { from: SENDER, text: checkinText(view) };
          // Before the native send, so a check-in read at once still moves the baseline (round 2 N3).
          if (view.team_observations) checkinTotals.sent(view.id, view.team_observations);
          // Do not append to the disk inbox: it can be drained into a private
          // CLI queue while the master is busy. The patched native callback
          // replaces that queue's check-in slot and wakes native waiters.
          return callback(team.formatLeaderEnvelope([message]).join('\n\n'), message.text);
        },
      });
      if (decision.action === 'none') return;
      if (decision.action === 'stop' || decision.action === 'skip') {
        if (decision.action === 'stop' || decision.reason !== lastSkipReason) {
          record(decision.action === 'stop' ? 'checkin_stopped' : 'checkin_skipped', { reason: decision.reason });
        }
        lastSkipReason = decision.reason;
        if (decision.action === 'stop') clearInterval(timer);
        return;
      }
      lastSkipReason = null;
      if (decision.action === 'coalesce') {
        record('checkin_coalesced', { pending_id: decision.pendingId, reason: decision.reason, at: view.at });
        return;
      }
      record('checkin_sent', { checkin_id: decision.id, reason: decision.reason, milestones: decision.milestones,
                               ...(decision.replacesId ? { supersedes_id: decision.replacesId } : {}),
                               workers: view.workers.length, unknown: view.unknown, chars: checkinText(view).length,
                               snapshot: compactCheckin(view) });
    } catch (error) {
      record('checkin_error', { error: String(error).slice(0, 500) });
    } finally {
      polling = false;
    }
  }

  // Exists even before team_create / the first periodic notification. No team
  // has been created yet, so there are zero registered workers, not stale ones.
  publish(await snapshot(null, { reason: 'initial' }, Date.now()));
  process.once('exit', () => { if (timer) clearInterval(timer); });
  record('checkin_enabled', { interval_s: checkinSeconds, refresh_interval_s: refreshMs / 1000, status_file: statusFile });
  return {
    pending: () => scheduler.pendingId,
    observe(text) {
      const id = scheduler.pendingId;
      if (id && text.includes(`<teammate_message from="${SENDER}">\n${id}:`)) {
        scheduler.observed(id);
        if (agentRules) checkinTotals.observed(id);
        record('checkin_observed', { checkin_id: id });
      }
    },
  };
}
