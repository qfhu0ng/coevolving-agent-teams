"""Check-in queue compatibility fails before model dispatch; no provider used."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).parents[1]
COMPAT = ROOT / "agent_company/runtime/qwen_native_compat.mjs"
PATCH = ROOT / "agent_company/runtime/qwen_native_patch.mjs"
ANCHOR = "pendingTeammateMessages.push(formatted);"
QUEUE = "export function enqueue(pendingTeammateMessages, formatted) { " + ANCHOR + " }\n"


@pytest.fixture
def native_fixture(tmp_path):
    """Small native module surface, with the CLI queue loaded after --import."""
    bundle = tmp_path / "bundle"
    chunks = bundle / "chunks"
    chunks.mkdir(parents=True)
    (bundle / "package.json").write_text('{"type":"module"}')
    (bundle / "cli.js").write_text("")
    (chunks / "identity.js").write_text("""
import { AsyncLocalStorage } from 'node:async_hooks';
var teammateIdentityStore = new AsyncLocalStorage();
export function getTeammateContext() { return teammateIdentityStore.getStore(); }
""")
    (chunks / "manager.js").write_text("""
export var Config = class { getAgentsSettings() { return {}; } };
export var TeamManager = class {
  async drainLeaderInbox() {}
  async cleanup() {}
  async spawnTeammate(config3) {
    const name = config3.name;
    this.teamEventEmitter.emit("teammate_joined" /* TEAMMATE_JOINED */, {});
  }
  async executeTeammate(name, signal, updateOutput) {
    const teamManager = this;
      await teamManager.spawnTeammate({
        name,
      });
    return {};
  }
  async pollLeaderInbox() {
    const callback = this.leaderMessageCallback, newMessages = [];
    callback(
      this.formatLeaderEnvelope(newMessages).join("\\n\\n"),
    );
  }
};
export async function listTasks(teamName, filters) { return []; }
// Native in-process backend anchor (teammate start hook, 2026-10-18).
export var InProcessBackend = class {
  async spawnAgent(config3) {
    const interactive = {};
    this.agents.set(config3.agentId, interactive);
    this.agentOrder.push(config3.agentId);
  }
};
""")
    (chunks / "queue.js").write_text(QUEUE)
    return bundle


def run_fixture(bundle, script, *, owner="master", seconds="1800"):
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node with module.registerHooks is required")
    return subprocess.run(
        [node, "--import", str(COMPAT), "--input-type=module", "-e", script,
         (bundle / "chunks/queue.js").as_uri()],
        env={**os.environ, "AGENT_COMPANY_QWEN_CLI": str(bundle / "cli.js"),
             "AGENT_COMPANY_ROOT_OWNER": owner,
             "AGENT_COMPANY_RELAY_URL": "http://127.0.0.1:12345/v1",
             "AGENT_COMPANY_CHECKIN_SECONDS": seconds,
             "AGENT_COMPANY_TASK_STARTED_EPOCH_MS": "1",
             "AGENT_COMPANY_TASK_DEADLINE_EPOCH_MS": "9999999999999",
             "AGENT_COMPANY_DELIVERY_REPO": str(bundle),
             "AGENT_COMPANY_STATUS_FILE": str(bundle / "status.json")},
        capture_output=True, text=True, timeout=20,
    )


def events(result):
    prefix = "Agent Company lifecycle: "
    return [json.loads(line[len(prefix):]) for line in result.stderr.splitlines()
            if line.startswith(prefix)]


@pytest.mark.parametrize("variant,count", [("missing", 0), ("drifted", 0),
                                           ("duplicate_same", 2), ("duplicate_chunk", 2)])
def test_checkin_guard_rejects_missing_drifted_or_duplicate_anchor_before_cli(native_fixture, variant, count):
    queue = native_fixture / "chunks/queue.js"
    if variant == "missing":
        queue.write_text("export const noQueue = true;\n")
    elif variant == "drifted":
        queue.write_text(QUEUE.replace(ANCHOR, "pendingTeammateMessages.push(String(formatted));"))
    elif variant == "duplicate_same":
        queue.write_text(QUEUE.replace(ANCHOR, ANCHOR + ANCHOR))
    else:
        (native_fixture / "chunks/second_queue.js").write_text(QUEUE)
    result = run_fixture(native_fixture, "console.log('CLI_EXECUTED');")
    assert result.returncode != 0
    assert f"exactly one native leader queue patch anchor; found {count}" in result.stderr
    assert "CLI_EXECUTED" not in result.stdout
    assert not events(result), "Structural mismatch must stop before runtime startup, not retry"


def test_checkin_guard_requires_actual_load_and_logs_only_successful_application(native_fixture):
    result = run_fixture(native_fixture, r'''
import assert from 'node:assert/strict';
const bridge = globalThis[Symbol.for('agent-company.native')];
const base = process.env.AGENT_COMPANY_RELAY_URL;
assert.throws(() => bridge.headers(base, {messages: []}), /not applied before a model request/);
const {enqueue} = await import(process.argv[1]);
const queue = ['worker report'];
const envelope = id => `<teammate_message from="runtime:checkin">\n${id}: snapshot\n</teammate_message>`;
enqueue(queue, envelope('old'));
enqueue(queue, envelope('new'));
assert.deepEqual(queue, ['worker report', envelope('new')], 'Actual loaded code must use the single-slot callback');
assert.equal(bridge.headers(base, {messages: []})['X-Agent-Company-Owner'], 'master');
console.log('REQUEST_ALLOWED');
''')
    assert result.returncode == 0, result.stderr
    assert "REQUEST_ALLOWED" in result.stdout
    lifecycle = events(result)
    loaded = next(event for event in lifecycle if event["event"] == "compatibility_loaded")
    assert "checkin_queue_slot" not in loaded["repairs"], "Preflight is not proof of actual load"
    applied = [event for event in lifecycle if event["event"] == "checkin_queue_patch_applied"]
    assert len(applied) == 1 and applied[0]["module"] == "queue.js"


def test_checkin_guard_blocks_model_boundary_if_queue_was_never_loaded(native_fixture):
    result = run_fixture(native_fixture, """
const bridge = globalThis[Symbol.for('agent-company.native')];
bridge.headers(process.env.AGENT_COMPANY_RELAY_URL, {messages: []});
console.log('MODEL_DISPATCH');
""")
    assert result.returncode != 0
    assert "not applied before a model request" in result.stderr
    assert "MODEL_DISPATCH" not in result.stdout
    assert not any(event["event"] == "checkin_queue_patch_applied" for event in events(result))


def test_checkin_guard_rechecks_source_when_loaded_after_preflight(native_fixture):
    result = run_fixture(native_fixture, r'''
import fs from 'node:fs';
// The --import preflight saw the correct fixture; actual loading sees drift.
fs.writeFileSync(new URL(process.argv[1]), 'export function enqueue(queue, value) { queue.push(value); }');
await import(process.argv[1]);
console.log('MODEL_DISPATCH');
''')
    assert result.returncode != 0
    assert "Qwen compatibility patch anchor changed" in result.stderr
    assert "MODEL_DISPATCH" not in result.stdout
    assert not any(event["event"] == "checkin_queue_patch_applied" for event in events(result))


@pytest.mark.parametrize("owner,seconds", [("master", "0"), ("single", "1800")])
@pytest.mark.parametrize("anchor_count", [0, 2])
def test_checkin_guard_does_not_block_single_or_disabled_team(native_fixture, owner, seconds, anchor_count):
    if anchor_count == 0:
        (native_fixture / "chunks/queue.js").write_text("export const noQueue = true;\n")
    else:
        (native_fixture / "chunks/second_queue.js").write_text(QUEUE)
    result = run_fixture(native_fixture, """
const bridge = globalThis[Symbol.for('agent-company.native')];
bridge.headers(process.env.AGENT_COMPANY_RELAY_URL, {messages: []});
console.log('REQUEST_ALLOWED');
""", owner=owner, seconds=seconds)
    assert result.returncode == 0, result.stderr
    assert "REQUEST_ALLOWED" in result.stdout
    assert not (native_fixture / "status.json").exists()
    assert not any(event["event"] in {"checkin_enabled", "checkin_queue_patch_applied"}
                   for event in events(result))


def test_current_native_bundle_has_one_strictly_patchable_checkin_queue():
    from test_rejection_integration import native_qwen

    cli, node = native_qwen()
    result = subprocess.run([node, "--input-type=module", "-e", r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
const {CHECKIN_QUEUE_ANCHOR, patchNativeSource} = await import(process.argv[1]);
let count = 0;
for (const name of fs.readdirSync(process.argv[2]).filter(name => name.endsWith('.js'))) {
  const source = fs.readFileSync(path.join(process.argv[2], name), 'utf8');
  const matches = source.split(CHECKIN_QUEUE_ANCHOR).length - 1;
  count += matches;
  if (matches) {
    const patched = patchNativeSource(source, {requireCheckinQueue: true});
    assert(!patched.includes(CHECKIN_QUEUE_ANCHOR));
    assert(patched.includes('.enqueueLeaderMessage(pendingTeammateMessages, formatted);'));
  }
}
assert.equal(count, 1);
assert.throws(() => patchNativeSource('', {requireCheckinQueue: true}), /anchor changed/);
assert.throws(() => patchNativeSource(CHECKIN_QUEUE_ANCHOR.repeat(2), {requireCheckinQueue: true}), /anchor changed/);
''', PATCH.as_uri(), str(cli.parent / "chunks")], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
