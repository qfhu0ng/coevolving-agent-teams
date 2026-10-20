// Narrow source repairs for private functions in Qwen 0.23.1. No installed
// package is edited. Exact anchors deliberately fail on an incompatible CLI.
export const CHECKIN_QUEUE_ANCHOR = 'pendingTeammateMessages.push(formatted);';
// AgentInteractive.runOneRound (in-process teammates) omits the reasoning
// loop's getExternalMessages option that background subagents pass, so queued
// team messages can only arrive when the teammate is idle.
export const TEAMMATE_ROUND_ANCHOR = 'maxTurns: this.config.maxTurnsPerMessage,';
// Opt-in leader loop recovery (2026-10-02). An always-on loop guard on the
// leader exits the whole CLI, and exit cleanup stops every teammate. Only the
// narrow case the compat hook accepts (repeated reads of the runtime status
// file while a teammate is running) ends the leader's turn and uses the
// native teammate wait instead; every other halt keeps native behavior.
export const LEADER_LOOP_ANCHORS = {
  // Turn.run: a streamed response's tool calls all arrive in its finish chunk.
  response: '        const functionCalls2 = resp.functionCalls ?? [];\n' +
            '        for (const fnCall of functionCalls2) {\n' +
            '          const event = this.handlePendingFunctionCall(fnCall);',
  // Main-session detection point: the only place that still sees the call.
  detect: 'const alwaysOnLoop = !duplicateLoopGuardRequest && this.loopDetector.checkAlwaysOnSafeties(event);',
  // Headless runner: exit on loop, else finalize a tool-free turn and wait.
  exit: '        if (loopDetected) {\n          return emitLoopDetectedResult();\n        }\n' +
        '        let shouldFinalizeTurn = toolCallRequests.length === 0;',
  // Next prompt built from queued teammate messages.
  resume: '          const batch = pendingTeammateMessages.splice(0);',
  // After the wait: no message and no active teammate would otherwise end as success.
  settle: '          if (teamManager) {\n            await teamManager.drainLeaderInbox();\n          }\n' +
          '          if (pendingTeammateMessages.length > 0) {\n            continue;\n          }',
};

// Role policies, D10 option C (2026-10-10). A formal POLICY-PROPOSAL to the leader
// is a control message, not a task report: native otherwise marks any explicit
// worker -> leader message as "reported this turn" and then skips forwarding the
// worker's end-of-turn text. Only a valid proposal header is exempted.
export const POLICY_REPORT_ANCHOR = '      if (sender && !automatic) {\n' +
                                    '        this.explicitLeaderReports.add(sender.agentId);\n      }';

// Bounded wait for 'close' after a non-PTY shell exits. Same 200 ms the native
// PTY path allows its post-exit drain (SIGKILL_TIMEOUT_MS) and promotion drain
// (PROMOTE_DRAIN_TIMEOUT_MS); bytes written before exit are already in the pipe,
// and the setImmediate after the timer guarantees one more poll phase.
export const SHELL_EXIT_DRAIN_MS = 200;

export function stripLeadingAnalysis(raw) {
  const text = raw.trim();
  if (!text.startsWith('<analysis>')) return text;
  const end = text.indexOf('</analysis>');
  return end < 0 ? '' : text.slice(end + '</analysis>'.length).trim();
}

function replaceOnce(source, before, after) {
  if (source.split(before).length !== 2) throw new Error('Qwen compatibility patch anchor changed: ' + before.slice(0, 100));
  return source.replace(before, after);
}

export function patchNativeSource(source, { requireCheckinQueue = false, teammateDelivery = false,
                                           leaderLoopRecovery = false, policyProposalReports = false } = {}) {
  const hook = 'globalThis[Symbol.for("agent-company.native")]';
  if (source.includes('var TeamManager = class')) {
    // Bind relay dispatch reservations to native instance names independently
    // of model history (which can be compacted before the next master request).
    source = replaceOnce(source,
      '    this.teamEventEmitter.emit("teammate_joined" /* TEAMMATE_JOINED */, {',
      `    ${hook}.spawnResult(config3.agentCompanyCallId, config3.name, name, true);\n` +
      '    this.teamEventEmitter.emit("teammate_joined" /* TEAMMATE_JOINED */, {');
    source = replaceOnce(source,
      '      await teamManager.spawnTeammate({\n        name,',
      '      await teamManager.spawnTeammate({\n        agentCompanyCallId: this.callId,\n        name,');
    source = replaceOnce(source,
      '  async executeTeammate(name, signal, updateOutput) {',
      '  async executeTeammate(name, signal, updateOutput) {\n' +
      '    const result = await this.agentCompanyExecuteTeammate(name, signal, updateOutput);\n' +
      `    if (result.error) ${hook}.spawnResult(this.callId, name, null, false);\n` +
      '    return result;\n  }\n  async agentCompanyExecuteTeammate(name, signal, updateOutput) {');
    // A wait can install its wake-up wrapper while inbox I/O is pending.
    // Calling a callback captured before that await queues the report but leaves
    // the leader asleep. Respect callback detachment as well as replacement.
    source = replaceOnce(source,
      '    callback(\n      this.formatLeaderEnvelope(newMessages).join("\\n\\n"),',
      '    this.leaderMessageCallback?.(\n      this.formatLeaderEnvelope(newMessages).join("\\n\\n"),');
  }
  if (source.includes('var InProcessBackend = class')) {
    // Before start(): a scripted or very fast first reply can otherwise announce the
    // teammate's first tool call before spawnTeammate returns and check-ins attach
    // their listener, so that call is never observed (2026-10-18, Linux). The compat
    // layer records agent_start_patch_applied; real CLI tests require it.
    source = replaceOnce(source,
      '    this.agents.set(config3.agentId, interactive);\n    this.agentOrder.push(config3.agentId);\n',
      '    this.agents.set(config3.agentId, interactive);\n    this.agentOrder.push(config3.agentId);\n' +
      `    ${hook}.agentStarting(config3.agentId, interactive);\n`);
  }
  if (source.includes('var ShellExecutionService = class')) {
    // A child can exit before its stdout/stderr pipes have drained; native
    // settles at exit and loses that output (observed with concurrent Linux
    // pwd). Exit still ends the command exactly as native does (abort snapshot,
    // abort listener removed, no later cancel kill or promotion); only the
    // result waits for 'close', bounded because a backgrounded descendant may
    // hold the pipes. Scope: non-PTY fallback only.
    const start = source.indexOf('  static childProcessFallback(');
    const end = source.indexOf('  static executeWithPty(', start);
    if (start < 0 || end < start) throw new Error('Qwen compatibility patch anchor changed: shell fallback');
    const original = source.slice(start, end);
    let repaired = replaceOnce(original,
      '        const handleExit = /* @__PURE__ */ __name((code, signal) => {\n          const { finalBuffer } = cleanup();',
      '        let agentCompanySettled = false;\n        let agentCompanyExitAborted;\n' +
      '        const handleExit = /* @__PURE__ */ __name((code, signal) => {\n' +
      '          if (agentCompanySettled) return;\n          agentCompanySettled = true;\n' +
      '          const { finalBuffer } = cleanup();');
    repaired = replaceOnce(repaired, '            aborted: abortSignal.aborted,\n',
      '            aborted: agentCompanyExitAborted ?? abortSignal.aborted,\n');
    repaired = replaceOnce(repaired,
      '            this.activeChildProcesses.delete(child.pid);\n          }\n          handleExit(code, signal);\n        }, "exitHandler");',
      '            this.activeChildProcesses.delete(child.pid);\n          }\n' +
      '          if (exited) return;\n' +
      '          exited = true;\n' +
      '          agentCompanyExitAborted = abortSignal.aborted;\n' +
      '          abortSignal.removeEventListener("abort", abortHandler);\n' +
      '          let drainTimer = null;\n' +
      '          const settle = () => {\n' +
      '            clearTimeout(drainTimer);\n' +
      '            child.off("close", settle);\n' +
      '            handleExit(code, signal);\n' +
      '          };\n' +
      '          child.once("close", settle);\n' +
      `          drainTimer = setTimeout(() => setImmediate(settle), ${SHELL_EXIT_DRAIN_MS});\n` +
      '        }, "exitHandler");');
    source = replaceOnce(source, original, repaired);
  }
  if (policyProposalReports) {
    source = replaceOnce(source, POLICY_REPORT_ANCHOR,
      `      if (sender && !automatic && !${hook}.policyProposalMessage(message, sender.name)) {\n` +
      '        this.explicitLeaderReports.add(sender.agentId);\n      }');
  }
  if (leaderLoopRecovery && source.includes(LEADER_LOOP_ANCHORS.detect)) {
    // Native still clears pending calls, ends the interaction and yields the
    // loop event; this only records whether the halt is the recoverable case,
    // including how many tool calls the stopped response contained.
    source = replaceOnce(source, LEADER_LOOP_ANCHORS.response,
      `        this.agentCompanyResponseCalls = (this.agentCompanyResponseCalls ?? 0) + (resp.functionCalls?.length ?? 0);\n` +
      LEADER_LOOP_ANCHORS.response);
    source = replaceOnce(source, LEADER_LOOP_ANCHORS.detect,
      `${LEADER_LOOP_ANCHORS.detect}\n          this.agentCompanyRecoverableLoop = alwaysOnLoop && ` +
      `${hook}.leaderLoopRecoverable(event, this.loopDetector.getLastLoopType(), turn.agentCompanyResponseCalls);`);
  }
  if (leaderLoopRecovery && source.includes(LEADER_LOOP_ANCHORS.exit)) {
    source = replaceOnce(source, LEADER_LOOP_ANCHORS.exit,
      `        if (loopDetected && ${hook}.recoverLeaderLoop(llmClient, toolCallRequests.length, config.getTeamManager())) {\n` +
      `          loopDetected = false;\n        }\n${LEADER_LOOP_ANCHORS.exit}`);
    source = replaceOnce(source, LEADER_LOOP_ANCHORS.resume,
      `          const batch = ${hook}.leaderResumeBatch(pendingTeammateMessages.splice(0));`);
    source = replaceOnce(source, LEADER_LOOP_ANCHORS.settle,
      `${LEADER_LOOP_ANCHORS.settle}\n          if (${hook}.leaderRecoveryUnresolved()) {\n` +
      `            return emitLoopDetectedResult();\n          }`);
  }
  if (requireCheckinQueue || source.includes(CHECKIN_QUEUE_ANCHOR)) {
    source = replaceOnce(source, CHECKIN_QUEUE_ANCHOR,
      `${hook}.enqueueLeaderMessage(pendingTeammateMessages, formatted);`);
  }
  if (teammateDelivery) {
    // Opt-in: drain the teammate's queued messages after each tool batch
    // through the loop's existing external-input path (never mid-tool).
    source = replaceOnce(source, TEAMMATE_ROUND_ANCHOR,
      `getExternalMessages: () => ${hook}.teammateInputs(this),\n          ${TEAMMATE_ROUND_ANCHOR}`);
  }
  if (source.includes('function stripAnalysisBlock(rawSummary) {')) {
    source = replaceOnce(source,
      'function stripAnalysisBlock(rawSummary) {\n  let result = rawSummary.replace(/<analysis>[\\s\\S]*?<\\/analysis>\\s*/g, "");\n  result = result.replace(/<analysis>[\\s\\S]*$/g, "");\n  return result.trim();\n}',
      `function stripAnalysisBlock(rawSummary) { return ${hook}.stripLeadingAnalysis(rawSummary); }`);
    source = replaceOnce(source, '  async compress(chat, opts) {',
      `  async compress(chat, opts) { return ${hook}.withPurpose("chat-compression", () => this.agentCompanyCompress(chat, opts)); }\n  async agentCompanyCompress(chat, opts) {`);
    source = replaceOnce(source, '      if (isTerminalStatus(event.newStatus)) {\n        this.fireAndForget(',
      `      if (isTerminalStatus(event.newStatus)) {\n        ${hook}.terminal(this, agent, agentId, agentName, event, listTasks);\n        this.fireAndForget(`);
  }
  if (source.includes('async function runSideQuery(config, options) {')) {
    source = replaceOnce(source, 'async function runSideQuery(config, options) {',
      `async function runSideQuery(config, options) { return ${hook}.withPurpose(options.purpose, () => agentCompanySideQuery(config, options)); }\nasync function agentCompanySideQuery(config, options) {`);
  }
  if (source.includes('var ContentGenerationPipeline = class')) {
    source = replaceOnce(source, '              signal: perRequestAc?.signal\n',
      `              signal: perRequestAc?.signal,\n              headers: ${hook}.headers(this.client.baseURL, openaiRequest)\n`);
    source = replaceOnce(source, '            { signal: perRequestAc.signal }',
      `            { signal: perRequestAc.signal, headers: ${hook}.headers(this.client.baseURL, openaiRequest) }`);
  }
  return source;
}
