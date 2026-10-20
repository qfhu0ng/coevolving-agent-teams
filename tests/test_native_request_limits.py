"""Released Qwen conversion and a live local relay, with no external requests."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from agent_company.runtime.qwen import QwenTeamRunner, native_model_providers
from agent_company.runtime.relay import ProviderConfig, TeamRelay, role_of, start_server
from agent_company.runtime.team import WorkerRegistry
from basic_smoke_scenario import BasicSmokeScenario
from role_policy_scenario import reply as scenario_reply
from test_rejection_integration import native_qwen, reply
from test_team import request, workspace


def deepseek_provider():
    return ProviderConfig(
        "deepseek-flash", "https://api.deepseek.com", max_output_tokens=65536,
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        request_timeout_seconds=5, stream_response=True)


def test_real_native_converter_to_relay_preserves_request_intent(tmp_path):
    """Use native CLI settings resolution, registry, converter and SDK transport.

    The cold/shared requests represent the native compactor's two call shapes;
    this checks parameter conversion, not summary quality or branch selection.
    """
    cli, node = native_qwen()
    observed = []

    def upstream(payload):
        observed.append(copy.deepcopy(payload))
        return reply(payload, content="<state_snapshot>offline summary</state_snapshot>")

    provider = deepseek_provider()
    relay = TeamRelay(provider, WorkerRegistry(workspace(tmp_path)), "", tmp_path / "relay", 60,
                      upstream, execution_mode="single", require_caller_metadata=True)
    server = start_server(relay, "127.0.0.1")
    base = f"http://127.0.0.1:{server.server_port}/v1"
    settings = {"model": {"generationConfig": {"maxRetries": 0}},
                "modelProviders": native_model_providers(provider, base)}
    setup = tmp_path / "settings.json"
    setup.write_text(json.dumps(settings))
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [chunks, settingsFile, base, token] = process.argv.slice(1);
async function find(needle) {
  const name = fs.readdirSync(chunks).find(name => name.endsWith('.js') &&
    fs.readFileSync(path.join(chunks, name), 'utf8').includes(needle));
  assert(name, needle);
  return import(pathToFileURL(path.join(chunks, name)).href);
}
const {resolveCliGenerationConfig} = await find('function resolveCliGenerationConfig(inputs) {');
const {ModelsConfig, Config, computeThresholds} = await find('var ModelsConfig = class');
const {parseModelReasoningCapabilities} = await find('function parseModelReasoningCapabilities(value) {');
const {ContentGenerationPipeline, determineProvider} = await find('var ContentGenerationPipeline = class');
const {clampOutputTokensToWindow} = await find('function clampOutputTokensToWindow(');
const {preloadRuntimeFetchModule} = await find('async function preloadRuntimeFetchModule() {');
// Match the native CLI's provider initialization before constructing its SDK.
await preloadRuntimeFetchModule();
const settings = JSON.parse(fs.readFileSync(settingsFile, 'utf8'));
const resolved = resolveCliGenerationConfig({settings, selectedAuthType: 'openai', env: process.env,
  argv: {model:'deepseek-flash', openaiBaseUrl:base, openaiApiKey:token}});
assert.equal(resolved.baseUrl, base);
assert.equal(resolved.generationConfig.maxRetries, 0);
assert.equal(resolved.generationConfig.timeout, settings.modelProviders.openai[0].generationConfig.timeout);
assert.equal(resolved.generationConfig.contextWindowSize, 1000000);
assert.equal(resolved.generationConfig.maxOutputTokens, undefined);
// Exercise the released accessor and threshold calculation without overriding
// autoCompactThreshold: the series uses Qwen's native default of 0.85.
assert.equal(settings.model.autoCompactThreshold, undefined);
const defaultCompactThreshold = Config.prototype.getAutoCompactThreshold.call({});
assert.equal(defaultCompactThreshold, undefined);
const thresholds = computeThresholds(resolved.generationConfig.contextWindowSize, defaultCompactThreshold);
assert.deepEqual(thresholds, {warn:830000, auto:850000, hard:977000, effectiveWindow:980000});
const models = new ModelsConfig({initialAuthType:'openai', modelProvidersConfig:settings.modelProviders,
  generationConfig:resolved.generationConfig, initialRegistryBaseUrl:base});
const capability = parseModelReasoningCapabilities(models.getResolvedModel('openai','deepseek-flash',base).capabilities.reasoning);
assert.deepEqual(capability, {thinking:true, disableField:'thinking', efforts:['low','high','max']});
const cliConfig = {
  getResolvedModelConfig: (...args) => models.getResolvedModel(...args),
  getCliVersion: () => 'offline-test', getProxy: () => undefined, getSessionId: () => 'offline-test'
};
const contentGeneratorConfig = {...resolved.generationConfig, authType:'openai',
  model:'deepseek-flash', baseUrl:base, apiKey:token};
const pipeline = new ContentGenerationPipeline({contentGeneratorConfig, cliConfig,
  provider:determineProvider(contentGeneratorConfig, cliConfig)});
const emitted = [];
const context = () => ({model:'deepseek-flash'});
const messages = [{role:'user', parts:[{text:'Synthetic repository history.'}]}];
const cases = [
  ['cold', 'chat-compression', {maxOutputTokens:20000, thinkingConfig:{includeThoughts:false}}],
  ['shared', 'chat-compression', {maxOutputTokens:20000, tools:[{functionDeclarations:[
    {name:'read_file', description:'Read a file', parameters:{type:'OBJECT', properties:{file_path:{type:'STRING'}}}}
  ]}], toolConfig:{functionCallingConfig:{mode:'NONE'}}}],
  ['coding', 'coding', {}],
  ['pretty_1m', 'coding', {maxOutputTokens:clampOutputTokensToWindow(32000,
    resolved.generationConfig.contextWindowSize,110478)}],
  ['escalated_1m', 'coding', {maxOutputTokens:clampOutputTokensToWindow(64000,
    resolved.generationConfig.contextWindowSize,110478)}],
  ['near_edge_1m', 'coding', {maxOutputTokens:clampOutputTokensToWindow(32000,
    resolved.generationConfig.contextWindowSize,944000)}],
  // Keep regression coverage for smaller legitimate native window budgets.
  ['latex2', 'coding', {maxOutputTokens:clampOutputTokensToWindow(32000,131072,90870)}],
  ['single', 'coding', {maxOutputTokens:clampOutputTokensToWindow(32000,131072,97076)}],
  ['escalated', 'coding', {maxOutputTokens:clampOutputTokensToWindow(64000,131072,1000)}]
];
for (const [name,purpose,config] of cases) {
  const payload = await pipeline.buildRequest({model:'deepseek-flash', contents:messages, config,
    ...(name === 'shared' ? {promptCacheSharing:true} : {})}, 'offline-'+name, context(), true);
  assert.equal(payload.max_tokens, ['coding','pretty_1m'].includes(name) ? 32000 :
    name === 'escalated_1m' ? 64000 : name === 'near_edge_1m' ? 6000 :
    name === 'latex2' ? 30202 : name === 'single' ? 23996 :
    name === 'escalated' ? 64000 : 20000);
  assert.equal(payload.thinking?.type, name === 'cold' ? 'disabled' : undefined);
  const stream = await pipeline.client.chat.completions.create(payload, {headers:{
    'X-Agent-Company-Owner':'single', 'X-Agent-Company-Purpose':purpose}});
  let content = '';
  for await (const chunk of stream) content += chunk.choices[0]?.delta?.content ?? '';
  assert.equal(content, '<state_snapshot>offline summary</state_snapshot>');
  emitted.push({name,purpose,payload});
}
process.stdout.write(JSON.stringify({capability, contextWindowSize:resolved.generationConfig.contextWindowSize,
  thresholds, emitted}));
'''
    try:
        result = subprocess.run(
            [node, "--input-type=module", "-e", script, str(cli.parent / "chunks"), str(setup), base, relay.token],
            check=False, capture_output=True, text=True, timeout=60,
            env={key: value for key, value in os.environ.items()
                 if key not in {"DEEPSEEK_API_KEY", "STAGE1_GLM_API_KEY", "QWEN_CODE_MAX_OUTPUT_TOKENS"}})
    finally:
        relay.close()
        server.shutdown()
        server.server_close()
    (tmp_path / "node_stdout.txt").write_text(result.stdout)
    (tmp_path / "node_stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    native = json.loads(result.stdout)
    (tmp_path / "native_emitted.json").write_text(json.dumps(native, indent=2) + "\n")
    (tmp_path / "upstream_emitted.json").write_text(json.dumps(observed, indent=2) + "\n")
    assert native["contextWindowSize"] == 1_000_000
    assert native["thresholds"] == {
        "warn": 830_000, "auto": 850_000, "hard": 977_000, "effectiveWindow": 980_000}
    assert len(observed) == len(native["emitted"]) == 9
    for before, after in zip(native["emitted"], observed, strict=True):
        assert after["max_tokens"] == before["payload"]["max_tokens"]
        assert before["payload"]["stream"] is after["stream"] is True
        if before["name"] == "cold":
            assert after["thinking"] == {"type": "disabled"}
            assert "reasoning_effort" not in after
        else:
            assert after["thinking"] == {"type": "enabled"}
            assert after["reasoning_effort"] == "high"
    assert observed[1]["tool_choice"] == "none"
    assert len(observed[1]["tools"]) == 1
    assert relay.summary()["provider_attempts"] == 9


def test_native_cli_uses_registered_capability_and_native_output_default(tmp_path):
    """Also traverse QwenTeamRunner's written settings and actual CLI startup."""
    cli, node = native_qwen()
    observed = []

    def upstream(payload):
        observed.append(copy.deepcopy(payload))
        return reply(payload, content="Offline delivery complete.")

    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=upstream).run(
        workspace(tmp_path), "Report the synthetic task complete.", "", "", output,
        deepseek_provider(), 40, execution_mode="single")
    assert result["stop_reason"] == "native_cli_completed", result
    launch = json.loads((output / "launch.json").read_text())
    settings = json.loads((Path(result["runtime_dir"]) / "profile/settings.json").read_text())
    assert "QWEN_CODE_MAX_OUTPUT_TOKENS" not in launch["environment"]
    registered = settings["modelProviders"]["openai"][0]
    assert registered["baseUrl"] == launch["environment"]["AGENT_COMPANY_RELAY_URL"]
    assert registered["generationConfig"] == {
        "maxRetries": 0, "timeout": int(launch["environment"]["QWEN_CODE_API_TIMEOUT_MS"]),
        "contextWindowSize": 1_000_000}
    assert "autoCompactThreshold" not in settings["model"]
    assert len(observed) == 1
    native = json.loads(next((output / "relay").glob("*_qwen_request.json")).read_text())
    assert native["max_tokens"] == observed[0]["max_tokens"] == 32000
    assert observed[0]["thinking"] == {"type": "enabled"}
    assert observed[0]["reasoning_effort"] == "high"


def truncated_reply(payload):
    status, body = scenario_reply(payload, reasoning="Synthetic reasoning stopped at its output limit.")
    body["choices"][0]["finish_reason"] = "length"
    return status, body


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_native_length_recovery_continues_file_work_without_budget_override(tmp_path, execution_mode):
    """The released CLI must recover a reasoning-only truncation before doing real tools."""
    cli, node = native_qwen()
    ws = workspace(tmp_path)
    marker = str(Path(ws.repo_path) / "native-length-marker.txt")
    scenario = BasicSmokeScenario(execution_mode, marker)
    target = "single" if execution_mode == "single" else "alpha"
    target_requests = []

    def upstream(payload):
        role = "single" if execution_mode == "single" else role_of(payload)
        if role == target:
            target_requests.append(copy.deepcopy(payload))
            if len(target_requests) == 1:
                return truncated_reply(payload)
        return scenario(payload)

    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=upstream).run(
        ws, "Complete the synthetic file roundtrip.", "Assign one worker.",
        "Report the completed file roundtrip to the leader.", output,
        deepseek_provider(), 60, execution_mode=execution_mode, max_model_calls=40)
    assert result["stop_reason"] == "native_cli_completed", result
    assert not result["errors"] and result["usage_complete"]
    assert scenario.file_verified and Path(marker).read_text() == "NATIVE_OK\n"
    assert len(target_requests) >= 4  # truncated, write, read, readback proof
    assert target_requests[0]["max_tokens"] == 32000
    assert target_requests[1]["max_tokens"] == 64000
    assert all(0 < payload["max_tokens"] <= 64000 for payload in target_requests)
    native_requests = [json.loads(path.read_text()) for path in
                       sorted((output / "relay").glob(f"*_{target}_qwen_request.json"))]
    assert [item["max_tokens"] for item in native_requests] == [
        item["max_tokens"] for item in target_requests]
    if execution_mode == "team":
        assert scenario.shutdown_sent
        responses = [json.loads(path.read_text()) for path in
                     (output / "relay").glob("*_alpha_provider_response.json")]
        messages = [json.loads(tool["function"]["arguments"]).get("message")
                    for response in responses for choice in response["choices"]
                    for tool in choice["message"].get("tool_calls", [])
                    if tool["function"]["name"] == "send_message"]
        assert "shutdown_approved" in messages
        lifecycle = result["native_result"]["lifecycle"]["events"]
        terminals = [event for event in lifecycle if event.get("event") == "worker_terminal"]
        # Native shutdown may cancel an idle worker after its approval.
        assert any(event.get("worker") == "alpha" and event.get("status") in {"completed", "cancelled"}
                   for event in terminals), lifecycle
        assert not any(event.get("status") == "failed" for event in terminals)


def test_native_persistent_length_recovery_is_bounded(tmp_path):
    """Native owns the finite retry/continuation budget; relay never raises its requests."""
    cli, node = native_qwen()
    observed = []

    def upstream(payload):
        observed.append(copy.deepcopy(payload))
        return truncated_reply(payload)

    output = tmp_path / "run"
    result = QwenTeamRunner(str(cli), node=node, upstream=upstream).run(
        workspace(tmp_path), "Complete the synthetic task.", "", "", output,
        deepseek_provider(), 40, execution_mode="single", max_model_calls=20)
    assert not result["execution"]["timed_out"]
    assert len(observed) == 5  # initial, escalated replay, three native continuations
    assert observed[0]["max_tokens"] == 32000
    assert all(0 < payload["max_tokens"] <= 64000 for payload in observed[1:])
    native_requests = [json.loads(path.read_text()) for path in
                       sorted((output / "relay").glob("*_single_qwen_request.json"))]
    assert [item["max_tokens"] for item in native_requests] == [item["max_tokens"] for item in observed]


@pytest.mark.parametrize("native_limit", [None, True, False, 0, -1, 1.5, "30202", 1, 23996, 30202, 65536, 99999])
def test_coding_budget_uses_only_positive_native_integer(tmp_path, native_limit):
    relay = TeamRelay(deepseek_provider(), WorkerRegistry(workspace(tmp_path)), "policy", tmp_path / "relay", 30)
    body = request()
    if native_limit is not None:
        body["max_tokens"] = native_limit
    original = copy.deepcopy(body)
    try:
        _, payload = relay.payload(body)
    finally:
        relay.close()
    valid = type(native_limit) is int and native_limit > 0
    assert payload["max_tokens"] == (min(65536, native_limit) if valid else 65536)
    assert body == original
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"


def test_reasoning_capability_does_not_relabel_other_providers():
    assert native_model_providers(ProviderConfig("other", "https://example.invalid"), "http://localhost/v1") == {}
