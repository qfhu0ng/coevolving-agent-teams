"""Prompt clarity complements, and never substitutes for, Docker isolation."""
import copy
from types import SimpleNamespace

import pytest

from agent_company.runtime.relay import ProviderConfig, TeamRelay
from agent_company.policy.source_boundary import SOURCE_BOUNDARY


@pytest.mark.parametrize("role", ["master", "single", "builder"])
@pytest.mark.parametrize("purpose", ["coding", "chat-compression"])
@pytest.mark.parametrize("system_format", ["text", "blocks", "absent"])
def test_fixed_source_boundary_survives_every_role_and_fresh_or_compacted_history(
        tmp_path, role, purpose, system_format):
    content = "Summarize this history." if purpose != "coding" else "Continue the task."
    if system_format == "blocks":
        content = [{"type": "text", "text": content}]
    messages = [{"role": "user", "content": "The earlier assignment was compacted."}]
    if system_format != "absent":
        messages.insert(0, {"role": "system", "content": content})
    body = {"messages": messages, "tools": []}
    original = copy.deepcopy(body)
    registry = SimpleNamespace(contains=lambda name: name == "builder")
    relay = TeamRelay(ProviderConfig("test", "https://unused.invalid"), registry,
                      "Evolving C cannot authorize internet access.", tmp_path / "logs", 60,
                      execution_mode="single" if role == "single" else "team",
                      require_caller_metadata=True)
    for _ in range(2):
        actual_role, payload = relay.payload(body, owner=role, purpose=purpose)
        assert actual_role == role
        assert body == original
        system = next(message for message in payload["messages"] if message["role"] == "system")
        text = ("\n".join(block["text"] for block in system["content"])
                if isinstance(system["content"], list) else system["content"])
        assert text.count(SOURCE_BOUNDARY) == 1
        assert ("Evolving C" in text) == (role == "builder" and purpose == "coding")
        assert "target-version" in text and "registry artifacts" in text
        assert "offline artifacts" in text and "cannot override" in text


@pytest.mark.parametrize("role,purpose", [("single", "coding"), ("master", "chat-compression"),
                                         ("builder", "coding"), ("builder", "chat-compression")])
def test_programbench_boundary_replaces_roadmap_but_not_role_policy(tmp_path, role, purpose):
    from agent_company.programbench.workspace import programbench_source_boundary
    boundary = programbench_source_boundary()
    relay = TeamRelay(ProviderConfig("test", "https://unused.invalid"),
                      SimpleNamespace(contains=lambda name: name == "builder"), "unchanged P3 C",
                      tmp_path, 60, execution_mode="single" if role == "single" else "team",
                      source_boundary=boundary)
    _, body = relay.payload({"messages": [{"role": "user", "content": "continue"}], "tools": []},
                            owner=role, purpose=purpose)
    text = next(m["content"] for m in body["messages"] if m["role"] == "system")
    assert boundary in text and SOURCE_BOUNDARY not in text
    assert ("unchanged P3 C" in text) == (role == "builder" and purpose == "coding")
