// Opt-in, controller-owned Stop callback for the native root session.
// Safe mode stays enabled. No settings, project, extension, HTTP or command
// hooks are loaded, and no SubagentStop callback is installed.
export function installNativeStopHook(Config, { owner, reason, deadlineEpochMs,
  now = Date.now, record = () => {}, hasPendingLeaderMessages = () => false } = {}) {
  if (reason === undefined || reason === null || reason === '') return false;
  if (typeof reason !== 'string' || !reason.trim()) throw new Error('Stop hook reason must be nonempty text');
  if (!['single', 'master'].includes(owner)) throw new Error('Stop hook applies only to a native root session');
  if (!Number.isFinite(deadlineEpochMs) || deadlineEpochMs <= 0) throw new Error('Stop hook requires the existing task deadline');
  if (typeof Config?.prototype?.initialize !== 'function') throw new Error('Native Config.initialize unavailable');
  const initialize = Config.prototype.initialize;
  const installed = new WeakSet();

  Config.prototype.initialize = async function (...args) {
    if (this.isSafeMode?.() !== true) throw new Error('Controller Stop hook requires native safe mode');
    // Let native create its HookSystem and MessageBus, but present no external
    // hook sources. These own-instance getters also keep later native reloads
    // empty. All remaining safe-mode gates (MCP, context files, skills, etc.)
    // still see isSafeMode() === true.
    this.getDisableAllHooks = () => false;
    this.getUserHooks = () => undefined;
    this.getProjectHooks = () => undefined;
    this.getExtensions = () => [];
    const result = await initialize.apply(this, args);
    if (installed.has(this)) return result;
    const system = this.getHookSystem?.();
    if (typeof system?.addFunctionHook !== 'function' || typeof system?.getAllHooks !== 'function') {
      throw new Error('Native session hook API unavailable');
    }
    if (system.getAllHooks().length !== 0) throw new Error('Unexpected configured hooks in isolated native session');
    const sessionId = this.getSessionId();
    system.addFunctionHook(sessionId, 'Stop', '*', (input, context) => {
      // Qwen 0.23.1 sends stop_hook_active:true even on the first root Stop.
      // Native owns its unchanged consecutive-block cap; do not use that flag
      // as an initial/recursive-call discriminator.
      const remainingMs = deadlineEpochMs - now();
      let allowReason = context?.signal?.aborted ? 'aborted'
        : remainingMs <= 0 ? 'deadline'
        : remainingMs <= 900000 ? 'final_fifteen_minutes'
        : input?.session_id !== sessionId ? 'other_session' : null;
      if (!allowReason && owner === 'master') {
        const manager = this.getTeamManager?.();
        // Stop runs before headless's normal teammate wait and final inbox
        // drain. Hand those cases back without consuming or changing queues.
        if (manager?.hasActiveTeammates?.() === true) allowReason = 'active_teammates';
        else if (manager?.pendingAsyncWork?.size > 0) allowReason = 'pending_team_work';
        else if (hasPendingLeaderMessages()) allowReason = 'pending_leader_messages';
      }
      const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
      record('stop_hook_decision', { owner, session_id: sessionId,
        decision: allowReason ? 'allow' : 'block', allow_reason: allowReason,
        remaining_seconds: remainingSeconds,
        aborted: Boolean(context?.signal?.aborted), stop_hook_active: input?.stop_hook_active ?? null });
      return allowReason ? { continue: true } : { decision: 'block',
        reason: reason.replaceAll('{remaining_seconds}', String(remainingSeconds)) };
    }, undefined, { id: 'agent-company-root-stop', timeout: 1000 });
    installed.add(this);
    record('stop_hook_installed', { owner, session_id: sessionId, deadline_epoch_ms: deadlineEpochMs });
    return result;
  };
  return true;
}
