"""Real Qwen CLI checks for the 1M window; every provider reply is scripted.

The pretty regression feeds the native token ledger the recorded 91,408 input
and 18,248 output counters. It exercises the real next-turn decision and tool
execution, not a fabricated 110K-character prompt or an external model call.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path

import pytest

from agent_company.runtime import qwen
from agent_company.runtime.relay import role_of
from basic_smoke_scenario import BasicSmokeScenario
from test_native_request_limits import deepseek_provider
from test_rejection_integration import native_qwen
from test_team import workspace


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_scripted_case(base, execution_mode, *, pretty_usage=False):
    """Keep the formal runtime switches; script only the synthetic file task."""
    base.mkdir(parents=True)
    cli, node = native_qwen()
    ws = workspace(base)
    marker = str(Path(ws.repo_path) / "native-window-marker.txt")
    scenario = BasicSmokeScenario(execution_mode, marker)
    observed = defaultdict(list)
    target = "single" if execution_mode == "single" else "alpha"
    injection = None

    def upstream(payload):
        nonlocal injection
        role = "single" if execution_mode == "single" else role_of(payload)
        observed[role].append(copy.deepcopy(payload))
        status, body = scenario(payload)
        if pretty_usage and role == target and len(observed[role]) == 1:
            # The same usage anchor which preceded pretty's failed compaction.
            body["usage"] = {"prompt_tokens": 91408, "completion_tokens": 18248,
                             "total_tokens": 109656,
                             "completion_tokens_details": {"reasoning_tokens": 17676}}
            body["choices"][0]["message"]["reasoning_content"] = "Synthetic usage-ledger fixture."
            injection = copy.deepcopy(body)
        return status, body

    output = base / "run"
    result = qwen.QwenTeamRunner(
        str(cli), node=node, upstream=upstream, checkin_interval_seconds=1800,
        team_message_delivery="between_tool_calls", runtime_disclosure=True,
        leader_loop_recovery=True, tool_call_cap="disabled", role_policy_mode="propose",
    ).run(ws, "Complete the synthetic file roundtrip.", "Assign one worker.",
          "Report the completed file roundtrip to the leader.", output,
          deepseek_provider(), 60, execution_mode=execution_mode, max_model_calls=40)
    save(base / "provider_requests.json", observed)
    save(base / "result.json", result)
    save(base / "injected_usage_response.json", injection)
    assert result["stop_reason"] == "native_cli_completed", result
    assert not result["execution"]["timed_out"] and not result["errors"], result
    assert scenario.file_verified and Path(marker).read_text() == "NATIVE_OK\n"
    if execution_mode == "team":
        assert scenario.shutdown_sent
        assert not any(event.get("status") == "failed" for event in
                       result["native_result"]["lifecycle"]["events"])
    return ws, result, dict(observed), injection


def model_visible(payload):
    return {"system": [message for message in payload["messages"] if message.get("role") == "system"],
            "tools": payload.get("tools", [])}


def normalized_system(system, workspace_root, runtime_dir):
    """Only replace the per-run local workspace and native UUID directory.

    Do not normalize model instructions, tool schemas, role names or budgets.
    The original provider payloads remain beside the normalized comparison.
    """
    return json.loads(json.dumps(system).replace(str(runtime_dir), "<NATIVE_RUNTIME>")
                      .replace(str(workspace_root), "<WORKSPACE>"))


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_window_change_preserves_actual_cli_system_and_tools(tmp_path, monkeypatch, execution_mode):
    # Both synthetic repositories must have the same initial commit. Keep the
    # native Git disclosure intact rather than normalizing away its contents.
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-10-15T00:00:00+00:00")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-10-15T00:00:00+00:00")
    registered = qwen.native_model_providers

    def retry8_registration(provider, relay_url):
        value = copy.deepcopy(registered(provider, relay_url))
        for models in value.values():
            for model in models:
                model["generationConfig"].pop("contextWindowSize", None)
        return value

    with monkeypatch.context() as old:
        old.setattr(qwen, "native_model_providers", retry8_registration)
        before = run_scripted_case(tmp_path / "retry8", execution_mode)
    after = run_scripted_case(tmp_path / "retry9", execution_mode)
    comparisons = {}
    expected = {"single"} if execution_mode == "single" else {"master", "alpha"}
    assert before[2].keys() == after[2].keys() == expected
    for role in sorted(expected):
        old_view, new_view = (model_visible(case[2][role][0]) for case in (before, after))
        old_system = normalized_system(old_view["system"], tmp_path / "retry8", before[1]["runtime_dir"])
        new_system = normalized_system(new_view["system"], tmp_path / "retry9", after[1]["runtime_dir"])
        comparisons[role] = {
            "tools_identical": old_view["tools"] == new_view["tools"],
            "system_identical_after_path_normalization": old_system == new_system,
            "tool_names": [tool["function"]["name"] for tool in new_view["tools"]],
            "retry8_system": old_system, "retry9_system": new_system,
            "normalization": ["per-run workspace path", "native runtime UUID directory"],
        }
        save(tmp_path / f"{role}_model_visible.json", {"retry8": old_view, "retry9": new_view})
    save(tmp_path / "comparison.json", comparisons)
    assert all(row["tools_identical"] for row in comparisons.values()), comparisons
    assert all(row["system_identical_after_path_normalization"] for row in comparisons.values()), comparisons
    for _, result, _, _ in (before, after):
        settings = json.loads((Path(result["runtime_dir"]) / "profile/settings.json").read_text())
        assert settings["tools"]["toolSearch"] == {"enabled": False}


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_pretty_usage_jump_continues_in_real_cli_without_compression(tmp_path, execution_mode):
    _, result, observed, injection = run_scripted_case(tmp_path / execution_mode, execution_mode,
                                                     pretty_usage=True)
    target = "single" if execution_mode == "single" else "alpha"
    requests = observed[target]
    assert injection is not None and len(requests) >= 3
    # This anchor exceeds the old 108,072 hard edge even before pending tools,
    # but remains far below the approved 850,000 automatic compression line.
    anchor = injection["usage"]["prompt_tokens"] + injection["usage"]["completion_tokens"]
    assert 108072 < anchor == 109656 < 850000
    issued = injection["choices"][0]["message"]["tool_calls"][0]
    next_messages = requests[1]["messages"]
    assert any(message.get("tool_call_id") == issued["id"] for message in next_messages)
    assert any(tool.get("id") == issued["id"] for message in next_messages
               for tool in message.get("tool_calls", []))
    assert all(0 < payload["max_tokens"] <= 32000 for payload in requests)
    events = [json.loads(line) for line in (tmp_path / execution_mode / "run/relay/events.jsonl")
              .read_text().splitlines() if line.strip()]
    purposes = [row.get("purpose") for row in events if row.get("event") == "provider_attempt_started"]
    assert purposes and set(purposes) == {"coding"}, purposes
    errors = (tmp_path / execution_mode / "run/qwen_stderr.txt").read_text()
    assert "COMPRESSION_FAILED" not in errors and "Context is too large" not in errors
    save(tmp_path / "pretty_regression.json", {
        "external_model_calls": 0, "synthetic_usage_not_real_token_generation": True,
        "prompt_tokens": 91408, "completion_tokens": 18248, "anchor_after_response": anchor,
        "old_hard_limit": 108072, "new_auto_limit": 850000,
        "next_coding_request_retained_tool_call_and_result": True,
        "provider_purposes": purposes, "stop_reason": result["stop_reason"],
    })
