"""Opt-in progress clocks and their native wait envelope, without model calls."""
import json
from types import SimpleNamespace

import pytest

import agent_company.runtime.qwen as qwen_module
from agent_company.runtime.provider import public_provider, load_provider
from agent_company.runtime.qwen import QwenTeamRunner, provider_incident_summary
from agent_company.runtime.relay import ProviderConfig
from test_team import workspace


FIELDS = ("first_progress_timeout_seconds", "progress_idle_timeout_seconds")


def test_legacy_provider_has_no_progress_timeout():
    provider = ProviderConfig("test-model", "http://localhost")
    assert all(getattr(provider, name) is None for name in FIELDS)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [None, 0.125, 600])
def test_progress_limits_are_independently_optional(field, value):
    provider = ProviderConfig("test-model", "http://localhost", **{field: value})
    assert getattr(provider, field) == value
    assert getattr(provider, next(name for name in FIELDS if name != field)) is None


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [0, -1, True, False, float("nan"), float("inf"),
                                    -float("inf"), "600", 10 ** 1000])
def test_invalid_progress_limits_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        ProviderConfig("test-model", "http://localhost", **{field: value})


def test_provider_file_and_public_settings_keep_progress_limits(tmp_path, monkeypatch):
    path = tmp_path / "provider.json"
    path.write_text(json.dumps({"model": "test-model", "base_url": "http://localhost",
        "api_key_env": "TEST_PROGRESS_KEY", "request_timeout_seconds": 1800,
        "first_progress_timeout_seconds": 600, "progress_idle_timeout_seconds": 600}))
    monkeypatch.setenv("TEST_PROGRESS_KEY", "synthetic-secret")
    provider = load_provider(path)
    public = public_provider(provider)
    assert public["request_timeout_seconds"] == 1800
    assert all(public[name] == 600 for name in FIELDS)
    assert "api_key" not in public and "synthetic-secret" not in json.dumps(public)


@pytest.mark.parametrize("mode,limits,stall,expected", [
    ("team", {}, 1800, 1800),
    ("team", {}, None, 600),
    ("team", {"first_progress_timeout_seconds": 600}, 1800, 2040),
    ("team", {"progress_idle_timeout_seconds": 600}, None, 2040),
    ("team", dict.fromkeys(FIELDS, 600), 3000, 3000),
    ("single", dict.fromkeys(FIELDS, 600), 1800, None),
])
def test_native_wait_envelope_only_changes_when_opted_in(tmp_path, monkeypatch,
                                                       mode, limits, stall, expected):
    ws = workspace(tmp_path)
    captured = {}
    monkeypatch.setattr(qwen_module, "start_server", lambda *args: SimpleNamespace(
        server_port=12345, shutdown=lambda: None, server_close=lambda: None))

    def run_agent(command, *, env, timeout_seconds, stdout_path, stderr_path):
        captured.update(command=command, env=env, timeout_seconds=timeout_seconds)
        stdout_path.write_text(json.dumps({"type": "result", "result": "ok", "is_error": False}) + "\n")
        stderr_path.write_text("")
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(ws, "run_agent", run_agent)
    result = QwenTeamRunner("unused-cli", upstream=lambda _: pytest.fail("No model call expected"),
        team_stall_timeout_seconds=stall).run(ws, "A local test", "Assignment", "Communication",
        tmp_path / "run", ProviderConfig("test-model", "http://localhost",
            request_timeout_seconds=1800, **limits), timeout_seconds=21600, execution_mode=mode)
    launch = json.loads((tmp_path / "run/launch.json").read_text())
    assert result["stop_reason"] == "native_cli_completed"
    assert launch["upstream_request_deadline_seconds"] == 1800
    assert launch["native_request_timeout_seconds"] == 1980
    assert launch["native_team_inactivity_timeout_seconds"] == expected
    assert captured["env"]["QWEN_CODE_API_TIMEOUT_MS"] == "1980000"
    assert captured["timeout_seconds"] == 21600
    if mode == "team" and (limits or stall is not None):
        assert float(captured["env"]["QWEN_TEAM_STALL_TIMEOUT_SECONDS"]) == expected
        assert any(arg.endswith("/qwen_stall_timeout.mjs") for arg in captured["command"])
    else:
        assert "QWEN_TEAM_STALL_TIMEOUT_SECONDS" not in captured["env"]
    for field in FIELDS:
        assert launch[field] == limits.get(field)


def test_progress_failures_are_separate_provider_incidents():
    kinds = ["provider_first_progress_timeout", "provider_progress_idle_timeout", "provider_timeout"]
    summary = provider_incident_summary({"errors": [
        {"error": "synthetic", "error_kind": kind, "role": "builder"} for kind in kinds]})
    assert summary == {"count": 3, "by_kind": dict.fromkeys(kinds, 1), "by_role": {"builder": 3}}
