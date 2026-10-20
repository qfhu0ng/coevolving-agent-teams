"""Selected DS settings through local SSE and the real Qwen CLI; no provider API."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from agent_company.runtime.provider import load_provider
from agent_company.runtime.qwen import QwenTeamRunner
from test_progress_recovery import emit
from test_rejection_integration import native_qwen, reply, tool_results
from test_stream_transport import chunk, headers
from test_team import call, workspace
from test_transport import peer


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["single", "team"])
def test_ds_streaming_round_trips_reasoning_tools_and_final_cache_usage(tmp_path, monkeypatch, mode):
    cli, node = native_qwen()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "SYNTHETIC_DS_KEY")
    provider = load_provider(ROOT / "configs/stage1/deepseek_flash.json")
    ws = workspace(tmp_path)
    planned = [call("read_file", {"file_path": str(Path(ws.repo_path) / "original.txt")}),
               call("glob", {"pattern": "*.txt", "path": ws.repo_path})]
    reasons = ["Synthetic reasoning for first read.", "Synthetic reasoning for file listing.",
               "Synthetic final reasoning."]
    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
             "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 4,
             "completion_tokens_details": {"reasoning_tokens": 2}}

    def serve(handler, state):
        payload = state.requests[-1]["body"]
        step = len(state.requests) - 1
        assert step < 3, "Native thinking/tool history did not finish in three requests"
        assert payload["model"] == "deepseek-flash"
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high" and payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["tools"] and payload["max_tokens"] == 32000
        history = [message for message in payload["messages"] if message["role"] == "assistant"]
        assert [message.get("reasoning_content") for message in history] == reasons[:step]
        if step:
            tool_results(payload, planned[:step])
        _, result = reply(payload, [planned[step]] if step < 2 else [],
                          content="Inspection complete." if step == 2 else "", usage=usage)
        # Exercise reasoning deltas separately from content/tool arguments and usage.
        headers(handler)
        emit(handler, chunk({"reasoning_content": reasons[step][:10]}))
        result["choices"][0]["message"]["reasoning_content"] = reasons[step][10:]
        delta = result["choices"][0]["message"]
        if delta.get("tool_calls"):
            delta["tool_calls"] = [{"index": index, **tool}
                                   for index, tool in enumerate(delta["tool_calls"])]
        emit(handler, chunk(delta))
        # DeepSeek puts usage on the final finish chunk, not a usage-only chunk.
        final = chunk({}, result["choices"][0]["finish_reason"])
        final["usage"] = usage
        emit(handler, final)
        emit(handler, "[DONE]")

    output = tmp_path / "run"
    with peer(serve) as (url, state):
        result = QwenTeamRunner(str(cli), node=node).run(
            ws, "Read original.txt and list the text files, then finish.",
            "Inspect the repository.", "Return concrete findings.", output,
            replace(provider, base_url=url), timeout_seconds=30, execution_mode=mode,
            max_model_calls=4)
        assert len(state.requests) == 3
    assert result["stop_reason"] == "native_cli_completed", result
    assert result["provider_attempts"] == result["completed_attempts"] == 3
    assert result["usage_complete"] and result["usage"]["cached_prompt_tokens"] == 24
    responses = sorted((output / "relay").glob("*_provider_response.json"))
    assert len(responses) == 3
    assert all(json.loads(path.read_text())["usage"] == usage for path in responses)


def test_native_converter_retains_reasoning_from_a_previous_non_tool_turn():
    """DeepSeek requires old reasoning even when that assistant turn used no tool."""
    cli, node = native_qwen()
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const chunks = process.argv[1];
const file = fs.readdirSync(chunks).find(name => name.endsWith('.js')
  && fs.readFileSync(path.join(chunks, name), 'utf8').includes('var OpenAIContentConverter ='));
const {OpenAIContentConverter: converter} = await import(pathToFileURL(path.join(chunks, file)).href);
const context = {model: 'deepseek-flash'};
const response = converter.convertOpenAIResponseToLlm({choices: [{index: 0,
  message: {role: 'assistant', content: 'First turn complete.', reasoning_content: 'Prior no-tool reasoning.'},
  finish_reason: 'stop'}]}, context);
const messages = converter.convertLlmRequestToOpenAI({contents: [
  {role: 'user', parts: [{text: 'First request.'}]},
  response.candidates[0].content,
  {role: 'user', parts: [{text: 'Now use a tool.'}]}
]}, context);
const assistant = messages.find(message => message.role === 'assistant');
assert.equal(assistant.reasoning_content, 'Prior no-tool reasoning.');
assert.equal(assistant.content, 'First turn complete.');
assert.deepEqual(messages.at(-1).content, [{type: 'text', text: 'Now use a tool.'}]);
'''
    result = subprocess.run([node, "--input-type=module", "-e", script, str(cli.parent / "chunks")],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
