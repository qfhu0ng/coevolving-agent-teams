"""Shared deployment profile: all roles, unknown prices, no inherited DS options."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_company.runtime.provider import public_provider, load_provider
from agent_company.evolution.optimizer import StrategyOptimizer
from agent_company.runtime.relay import TeamRelay, usage_and_cost
from agent_company.runtime.relay import ProviderConfig
from agent_company.evolution.reviewer import StrategyReviewer

PROFILE = Path(__file__).resolve().parents[1] / "configs/stage1/self_hosted_glm.json"


def test_profile_requires_env_and_does_not_embed_key(monkeypatch):
    monkeypatch.delenv("STAGE1_GLM_API_KEY", raising=False)
    assert "api_key" not in json.loads(PROFILE.read_text())
    with pytest.raises(ValueError, match="STAGE1_GLM_API_KEY"):
        load_provider(PROFILE)


def test_all_roles_use_same_profile_without_changing_permissions(monkeypatch, tmp_path):
    monkeypatch.setenv("STAGE1_GLM_API_KEY", "synthetic-secret")
    provider = load_provider(PROFILE)
    assert provider.request_timeout_seconds == 600
    assert provider.max_attempts == 11  # Initial attempt plus at most ten retries.
    assert provider.retry_backoff_seconds == 1
    assert provider.inspection_max_output_tokens == 32768
    assert provider.extra_body == {} and provider.rates is None
    assert "synthetic-secret" not in json.dumps(public_provider(provider))
    registry = SimpleNamespace(contains=lambda name: name in {"worker-a", "worker-b"})
    relay = TeamRelay(provider, registry, "C", tmp_path / "relay", 30,
                      require_caller_metadata=True)
    body = {"model": "old-model", "messages": [{"role": "system", "content": "Synthetic test"}],
            "tools": [{"type": "function", "function": {"name": n}} for n in
                      ("send_message", "run_shell_command", "agent")]}
    for role in ("master", "worker-a", "worker-b"):
        for purpose in ("coding", "summary"):
            actual_role, payload = relay.payload(body, owner=role, purpose=purpose)
            assert actual_role == role
            assert payload["model"] == "RadixArk/GLM-5.3-NVFP4"
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
            assert "thinking" not in payload and "reasoning_effort" not in payload
            names = [t["function"]["name"] for t in payload["tools"]]
            assert ("run_shell_command" in names) == (role != "master")
    for cls in (StrategyOptimizer, StrategyReviewer):
        outer = cls(provider, tmp_path / cls.__name__)
        assert outer.provider is provider
        assert outer.completion.config.base_url == provider.base_url
    _, cost = usage_and_cost({"prompt_tokens": 10, "completion_tokens": 2}, provider.rates)
    assert cost is None


@pytest.mark.parametrize("limit", [0, -1, True, None, 1.5])
def test_invalid_inspection_allowance_is_rejected(limit):
    with pytest.raises(ValueError, match="inspection_max_output_tokens"):
        ProviderConfig("model", "https://provider.invalid", inspection_max_output_tokens=limit)
