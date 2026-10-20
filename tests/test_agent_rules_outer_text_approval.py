"""Every agent_rules_v1 outer-loop text equals the user's 2026-10-18 approval (plan v4 M5)."""
import json
from pathlib import Path

from agent_company.evolution import agent_rules_outer_text as text, optimizer, reviewer
from agent_company.evolution.harness_capabilities import target_harness
from agent_company.evolution.outer_budget import REASONING_STUB
from test_agent_rules_generation import INFERENCE

ROOT = Path(__file__).resolve().parents[1]
APPROVED = json.loads((ROOT / "tests/fixtures/agent_rules_outer_texts_approved_20261018.json").read_text())


def test_text_module_matches_the_approval_exactly():
    assert text.TEXT_STATUS == "approved_20261018"
    current = json.loads(json.dumps({name: getattr(text, name) for name in dir(text)
                                     if name.isupper() and name != "TEXT_STATUS"}))
    # A new or changed model-visible string needs a new approval and a new fixture.
    assert sorted(current) == sorted(APPROVED["constants"])
    assert {name: value for name, value in current.items() if value != APPROVED["constants"][name]} == {}


def test_assembled_systems_and_harness_are_the_approved_texts():
    # The inline replacements in agent_rules_system / agent_rules_review_system are pinned here too.
    assert optimizer.agent_rules_system(optimizer.SYSTEM + optimizer.WORKING_MEMORY_GUIDE + optimizer.CITATION_GUIDE
                                        + optimizer.EXECUTABILITY_GUIDE) == APPROVED["optimizer_system"]
    assert reviewer.agent_rules_review_system(reviewer.SYSTEM + reviewer.EXECUTABILITY_GUIDE, True) == APPROVED["reviewer_system"]
    harness = target_harness(INFERENCE, execution_mode="team", policy_protocol="agent_rules_v1")
    assert json.loads(json.dumps(harness["online_policy"])) == APPROVED["online_policy"]
    assert REASONING_STUB == APPROVED["reasoning_stub"]
