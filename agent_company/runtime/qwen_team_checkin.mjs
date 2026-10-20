import fs from 'node:fs';
import path from 'node:path';
// Runtime team check-ins for the coordination-only master: when to deliver an
// observational team snapshot, and how to word it. Logic only, except containerMemory(),
// which reads the container's own cgroup files; the native binding
// (qwen_native_compat.mjs) supplies clocks, snapshots and delivery.
//
// A check-in reports facts gathered from several sources (native lifecycle and
// tool events, the task board, read-only git, relay request accounting). It
// never judges a worker as stalled or failed, and unknown sources stay unknown.

export const SENDER = 'runtime:checkin';

/** cgroup v2 memory of this process's own cgroup, the task container (agent_rules_v1).
 * The same counters as the resource observer (programbench/observation.py); a field that
 * cannot be read is null and shown as unknown. memory.max "max" (no limit) is null too. */
export function containerMemory(procRoot = '/proc', cgroupRoot = '/sys/fs/cgroup') {
  const out = { current_bytes: null, peak_bytes: null, limit_bytes: null, oom_kills: null };
  let base = null;
  try {
    const line = fs.readFileSync(path.join(procRoot, 'self/cgroup'), 'utf8').split('\n').find(row => row.startsWith('0::'));
    const member = line?.slice(3);
    if (member?.startsWith('/') && !member.split('/').includes('..')) base = path.join(cgroupRoot, member);
  } catch { return out; }
  if (!base) return out;
  const read = name => { try { return fs.readFileSync(path.join(base, name), 'utf8').trim(); } catch { return null; } };
  const count = value => (/^\d+$/.test(value ?? '') ? Number(value) : null);
  out.current_bytes = count(read('memory.current'));
  out.peak_bytes = count(read('memory.peak'));
  out.limit_bytes = count(read('memory.max'));
  const kill = read('memory.events')?.split('\n').map(row => row.split(' ')).find(pair => pair[0] === 'oom_kill');
  out.oom_kills = kill ? count(kill[1]) : null;
  return out;
}

export function memoryLine(memory) {
  const gib = bytes => (Number.isFinite(bytes) ? `${(bytes / 2 ** 30).toFixed(1)} GiB` : 'unknown');
  return `Container memory: ${gib(memory?.current_bytes)} in use of ${gib(memory?.limit_bytes)}, `
    + `peak ${gib(memory?.peak_bytes)}; processes killed for lack of memory so far: ${memory?.oom_kills ?? 'unknown'}.`;
}

export class CheckinScheduler {
  constructor({ intervalMs, startMs, deadlineMs, milestones = [0.5, 0.75], minSpacingMs = 5 * 60_000,
                finalGuardMs = 60_000 }) {
    if (!(intervalMs > 0) || !(deadlineMs > startMs)) throw new Error('Invalid check-in schedule');
    Object.assign(this, { intervalMs, startMs, deadlineMs, minSpacingMs, finalGuardMs });
    this.milestones = milestones.map(fraction => ({ fraction, label: milestoneLabel(fraction),
      atMs: startMs + fraction * (deadlineMs - startMs), announced: false }));
    this.nextPeriodicMs = startMs + intervalMs;
    this.lastSentMs = null;
    this.pendingMilestones = [];
    this.pendingId = null;   // Sent, not yet seen in a master request.
    this.pendingMessageMilestones = [];
    this.reservation = null; // Allows a pre-send lifecycle change to undo this poll.
    this.stopped = null;     // Reason once permanently stopped.
    this.sequence = 0;
  }

  /** Decide one poll. `team` = {active, windingDown}. Returns an action. */
  poll(nowMs, team) {
    if (this.stopped) return { action: 'none' };
    if (nowMs >= this.deadlineMs - this.finalGuardMs) return this.stop('deadline');
    // Cooperative shutdown can be rejected or followed by a replacement. Only
    // native cleanup / the deadline permanently ends runtime supervision.
    if (team.windingDown) return { action: 'skip', reason: 'shutdown_in_progress' };
    for (const milestone of this.milestones) {
      if (!milestone.announced && nowMs >= milestone.atMs) {
        milestone.announced = true;
        this.pendingMilestones.push(milestone.label);
      }
    }
    const periodicDue = nowMs >= this.nextPeriodicMs;
    // A milestone that would land within minSpacing of the next periodic
    // check-in, or while an earlier one is still unread, waits and is merged.
    const milestoneDue = this.pendingMilestones.length > 0 && this.pendingId === null
      && this.nextPeriodicMs - nowMs > this.minSpacingMs
      && (this.lastSentMs === null || nowMs - this.lastSentMs >= this.minSpacingMs);
    if (!periodicDue && !milestoneDue) return { action: 'none' };
    const previousPeriodicMs = this.nextPeriodicMs;
    if (periodicDue) this.nextPeriodicMs = nowMs + this.intervalMs;
    // An idle team has already returned control to the master.
    if (!team.active) return { action: 'skip', reason: 'no_active_teammates' };
    const reason = periodicDue ? 'periodic' : 'milestone';
    if (this.pendingId && nowMs - this.lastSentMs < this.intervalMs) {
      return { action: 'coalesce', reason, pendingId: this.pendingId };
    }
    // An unobserved notification cannot block later decision opportunities for
    // the rest of the task. The native binding replaces its queued notification;
    // it must not append another unread message or count expiry as observation.
    const replacesId = this.pendingId;
    const newlyDueMilestones = this.pendingMilestones.splice(0);
    const milestones = [...this.pendingMessageMilestones, ...newlyDueMilestones];
    this.reservation = { previousPeriodicMs, previousSentMs: this.lastSentMs,
      previousPendingId: this.pendingId, previousMessageMilestones: this.pendingMessageMilestones,
      newlyDueMilestones };
    this.lastSentMs = nowMs;
    this.pendingId = `checkin-${++this.sequence}`;
    this.pendingMessageMilestones = milestones;
    return { action: 'send', reason, milestones, id: this.pendingId, ...(replacesId ? { replacesId } : {}) };
  }

  observed(id) {
    if (id === this.pendingId) {
      this.pendingId = null;
      this.pendingMessageMilestones = [];
      this.reservation = null;
    }
  }

  retract(decision) {
    if (decision.id !== this.pendingId || !this.reservation) return;
    this.nextPeriodicMs = this.reservation.previousPeriodicMs;
    this.lastSentMs = this.reservation.previousSentMs;
    this.pendingMilestones.unshift(...this.reservation.newlyDueMilestones);
    this.pendingId = this.reservation.previousPendingId;
    this.pendingMessageMilestones = this.reservation.previousMessageMilestones;
    this.reservation = null;
  }

  /** Re-check immediately before sending: async sampling can outlive a task. */
  beforeSend(nowMs, team, decision) {
    let blocked = null;
    if (this.stopped) blocked = { action: 'stop', reason: this.stopped };
    else if (nowMs >= this.deadlineMs - this.finalGuardMs) blocked = this.stop('deadline');
    else if (team.windingDown) blocked = { action: 'skip', reason: 'shutdown_in_progress' };
    else if (!team.active) blocked = { action: 'skip', reason: 'no_active_teammates' };
    if (blocked) {
      this.retract(decision);
      return blocked;
    }
    return decision;
  }

  stop(reason) {
    this.stopped = reason;
    return { action: 'stop', reason };
  }
}

/** Production polling path, with injectable observation functions for tests.
 * Refreshing the status file never calls a model. The send guard deliberately
 * runs after both async collection and publication, without an intervening
 * await before handing off to the native leader callback.
 */
export async function runCheckin({ scheduler, now, teamState, snapshot, publish, send, refreshDue = false }) {
  const sampledAtMs = now();
  const decision = scheduler.poll(sampledAtMs, teamState());
  if (decision.action === 'stop' || scheduler.stopped) return { decision };
  if (!refreshDue && ['none', 'skip'].includes(decision.action)) return { decision };
  try {
    const view = await snapshot(decision, sampledAtMs);
    publish(view);
    if (decision.action !== 'send') return { decision, view };
    const finalDecision = scheduler.beforeSend(now(), teamState(), decision);
    if (finalDecision.action !== 'send') return { decision: finalDecision, view };
    await send(view, decision);
    return { decision, view };
  } catch (error) {
    if (decision.action === 'send') scheduler.retract(decision);
    throw error;
  }
}

function milestoneLabel(fraction) {
  if (fraction === 0.5) return 'half of the task wall time has elapsed';
  if (fraction === 0.75) return 'the last quarter of the task wall time has started';
  return `${Math.round(fraction * 100)}% of the task wall time has elapsed`;
}

const minutes = seconds => (seconds === null || seconds === undefined ? 'unknown'
  : `${Math.round(seconds / 60)}m`);

/** Observed facts worth the master's attention; never a verdict. */
export function signals(worker, intervalSeconds) {
  const out = [];
  const git = worker.git || {};
  if (worker.status === 'failed') out.push('native status failed (see the runtime:system failure notice)');
  if (git.commits_since_worker_creation === 0 && worker.joined_ago_s >= intervalSeconds) {
    out.push(`no commit since this worker's worktree was created; worker joined ${minutes(worker.joined_ago_s)} ago`);
  } else if (git.commits_since_worker_creation > 0 && typeof git.last_commit_ago_s === 'number'
             && git.last_commit_ago_s >= intervalSeconds) {
    out.push(`no new worktree commit for ${minutes(git.last_commit_ago_s)}`);
  }
  if (git.commits_not_in_delivery_history > 0) {
    out.push(`${git.commits_not_in_delivery_history} original commit SHA(s) absent from delivery history `
      + '(ancestry only; cherry-picked or otherwise integrated code may already be delivered)');
  }
  for (const tool of worker.pending_tools || []) {
    if (tool.announced_ago_s >= intervalSeconds / 2) {
      out.push(`${tool.name} announced ${minutes(tool.announced_ago_s)} ago; no result observed yet (execution may not have started)`);
    }
  }
  if (worker.queued_messages > 0) out.push(`${worker.queued_messages} queued message(s) not yet delivered`);
  if (worker.relay && worker.relay.failed_attempts > 0) out.push(`${worker.relay.failed_attempts} failed model request attempt(s)`);
  return out;
}

export function compactCheckin(snapshot) {
  // The sent snapshot as lifecycle evidence (stderr), since the status file is
  // overwritten on every refresh. Counts, states and tool names only: no commit
  // subjects, task subjects or relay text. The master's check-in text is unchanged.
  const count = value => (typeof value === 'number' && Number.isFinite(value) ? value : null);
  const delivery = snapshot.delivery;
  return {
    // Every *_ago_s is measured from sample_started_at (sampling start), not from at (finish).
    sample_started_at: snapshot.sample_started_at ?? null, at: snapshot.at ?? null,
    elapsed_s: count(snapshot.elapsed_s), remaining_s: count(snapshot.remaining_s),
    delivery: delivery ? { commits_since_launch: count(delivery.commits_since_launch),
                           last_commit_ago_s: count(delivery.last_commit_ago_s) } : null,
    unassigned_open_tasks: count(snapshot.unassigned_open_tasks),
    workers: (snapshot.workers || []).map(worker => {
      const git = worker.git;
      return {
        name: worker.name, status: worker.status, shutdown_pending: worker.shutdown_pending === true,
        last_activity_ago_s: count(worker.last_activity_ago_s), queued_messages: count(worker.queued_messages),
        pending_tools: Array.isArray(worker.pending_tools) ? worker.pending_tools.map(call => call.name) : null,
        open_tasks: Array.isArray(worker.tasks) ? worker.tasks.filter(task => task.status !== 'completed').length : null,
        git: git ? { commits_since_worker_creation: count(git.commits_since_worker_creation),
                     commits_not_in_delivery_history: count(git.commits_not_in_delivery_history),
                     uncommitted_paths: count(git.uncommitted_paths), last_commit_ago_s: count(git.last_commit_ago_s) } : null,
      };
    }),
    unknown: Array.isArray(snapshot.unknown) ? snapshot.unknown : null,
    ...('container_memory' in snapshot ? { container_memory: Object.fromEntries(
      ['current_bytes', 'peak_bytes', 'limit_bytes', 'oom_kills'].map(key => [key, count(snapshot.container_memory?.[key])])) } : {}),
  };
}

export function formatCheckin(snapshot, intervalSeconds) {
  // The configured limit, as the relay reports it; 10 when unknown (earlier runs' text).
  const limit = Number.isInteger(snapshot.slots?.max) ? snapshot.slots.max : 10;
  const teamSize = `Teammate slots used ${snapshot.slots.used}/${snapshot.slots.max}. `
    + `At most ${limit} teammates can be active at the same time; shut-down teammates remain in the roster but do not count toward this limit.`;
  const lines = [
    `${snapshot.id}: Runtime team check-in (${snapshot.reason}). Observations only — not instructions, `
      + 'not completion reports, and not a verdict that anyone is stuck. Unknown means the source was unavailable.',
    `Elapsed ${minutes(snapshot.elapsed_s)}, remaining ${minutes(snapshot.remaining_s)}. `
      + teamSize,
  ];
  if (snapshot.at) lines.push(`Snapshot collected at ${snapshot.at}`
    + (snapshot.sample_started_at ? ` (collection started ${snapshot.sample_started_at}; sources are not atomic)` : '') + '.');
  for (const label of snapshot.milestones || []) lines.push(`Milestone: ${label}.`);
  // agent_rules_v1 only: earlier protocols' check-ins stay byte-identical.
  if ('container_memory' in snapshot) lines.push(memoryLine(snapshot.container_memory));
  const delivery = snapshot.delivery || {};
  lines.push(`Delivery repository: ${delivery.commits_since_launch ?? 'unknown'} new commit(s) since launch`
    + (delivery.commits_since_launch > 0 ? `; last ${minutes(delivery.last_commit_ago_s)} ago ("${delivery.last_commit_subject}")` : '')
    + '.');
  for (const worker of snapshot.workers) {
    const git = worker.git || {};
    const tool = worker.pending_tools?.length ? worker.pending_tools.map(pending =>
      `${pending.name} awaiting result (${minutes(pending.announced_ago_s)} since announcement)`).join(', ')
      : (worker.last_tool_result ? `last ${worker.last_tool_result.name} result ${minutes(worker.last_tool_result.received_ago_s)} ago`
        : (worker.tool_counts == null ? 'unknown' : 'no observed tool call yet'));
    const tasks = worker.tasks == null ? 'unknown'
      : (worker.tasks.map(task => `#${task.id} ${task.status}`).join(', ') || 'none');
    lines.push(`- ${worker.name}: ${worker.status}${worker.shutdown_pending ? ' (shutdown requested)' : ''}; `
      + `activity ${minutes(worker.last_activity_ago_s)} ago; tool: ${tool}; `
      + `edit attempts ${worker.tool_counts?.edits ?? 'unknown'}, command attempts ${worker.tool_counts?.commands ?? 'unknown'}; `
      + `worktree: ${git.commits_since_worker_creation ?? 'unknown'} new commit(s) since worker creation`
      + (git.commits_since_worker_creation > 0 ? `, last ${minutes(git.last_commit_ago_s)} ago` : '')
      + `, ${git.uncommitted_paths ?? 'unknown'} uncommitted path(s); `
      + `tasks: ${tasks}.`);
    const facts = signals(worker, intervalSeconds);
    if (facts.length) lines.push(`  observed: ${facts.join('; ')}.`);
  }
  if ('team_observations' in snapshot) lines.push(...collaborationLines(snapshot.team_observations));
  if (snapshot.unknown?.length) lines.push(`Unavailable sources: ${snapshot.unknown.join(', ')}.`);
  if (snapshot.status_file) lines.push(`Latest full snapshot (may be newer than this message): ${snapshot.status_file}`);
  if ('team_observations' in snapshot) {
    // agent_rules_v1 (plan v4 section 9): the same decision, plus whether to change a teammate's policy.
    lines.push('Decide from these observations together with the task and the code whether to continue, message a '
      + 'worker, reassign work, add a worker, or ask a worker to hand off and shut down. Based on the collaboration '
      + 'problems you have seen and handled since the previous check-in, also decide whether to change any '
      + "teammate's policy with AGENT-POLICY. No reply is needed if nothing changes.");
  } else {
    lines.push('Decide from these observations together with the task and the code whether to continue, message a '
      + 'worker, reassign work, add a worker, or ask a worker to hand off and shut down. No reply is needed if nothing changes.');
  }
  return lines.join('\n');
}

/** agent_rules_v1: lesson and board totals as of the last check-in the master actually saw.
 * A sent check-in that a later one replaced before the master read it never moves the
 * baseline, so "new since the previous check-in" keeps what arrived before it (M3 review 7). */
export class CheckinTotals {
  constructor() {
    this.baseline = { lessons: 0, board: 0 };
    this.sentTotals = new Map();
  }

  newSince(observed) {
    const delta = (key, total) => (typeof total === 'number' ? total - this.baseline[key] : null);
    return { lessons_new: delta('lessons', observed?.lessons_total), board_new: delta('board', observed?.board_total) };
  }

  sent(id, observed) {
    if (!id || !observed) return;
    this.sentTotals.set(id, { lessons: observed.lessons_total, board: observed.board_total });
  }

  observed(id) {
    const totals = this.sentTotals.get(id);
    this.sentTotals.clear();
    if (!totals) return;
    if (typeof totals.lessons === 'number') this.baseline.lessons = totals.lessons;
    if (typeof totals.board === 'number') this.baseline.board = totals.board;
  }
}

/** agent_rules_v1 collaboration observations: counts and times only, never message text. */
export function collaborationLines(observed) {
  if (!observed) return ['Collaboration observations: unavailable (relay status could not be read).'];
  const list = (items, render) => (items == null ? 'unknown' : items.length ? items.map(render).join(', ') : 'none');
  const rates = observed.rule_rates == null ? 'unknown'
    : Object.keys(observed.rule_rates).length === 0 ? 'no rule has triggered yet'
      : Object.entries(observed.rule_rates).map(([name, row]) => {
        const decided = (row.satisfied || 0) + (row.missed || 0);
        return `${name} ${decided ? `${row.satisfied}/${decided}` : 'none decided'}${row.open ? ` (${row.open} open)` : ''}`;
      }).join(', ');
  const count = value => (typeof value === 'number' ? value : 'unknown');
  return [
    'Collaboration observations:',
    `- Shutdown requested over 5 minutes ago and not finished: ${list(observed.pending_shutdowns, row => `${row.worker} (${minutes(row.requested_ago_s)})`)}.`,
    `- Idle over 15 minutes with no in-progress task: ${list(observed.idle_workers, row => `${row.worker} (${minutes(row.idle_s)})`)}.`,
    `- Messages between teammates, most frequent pairs (counts only): ${list(observed.message_pairs, row => `${row.from}->${row.to} ${row.count}`)}.`,
    `- Rule obligations met so far (met/decided; not-yet-decidable ones excluded): ${rates}.`,
    `- Policy proposals waiting for you: ${count(observed.pending_proposals)}; recent changes teammates applied to their own policies: ${count(observed.recent_self_changes)} (details at the end of your requests).`,
    `- Lessons: ${count(observed.lessons_total)} in total, ${count(observed.lessons_new)} new since the previous check-in. Board posts: ${count(observed.board_total)} in total, ${count(observed.board_new)} new since the previous check-in.`,
  ];
}
