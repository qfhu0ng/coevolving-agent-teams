"""Every agent_rules_v1 model-visible text equals the user's approvals (2026-10-17 M4; resources 2026-10-18)."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from agent_company.policy import agent_rules_text as text
from agent_company.policy.role_package import create_m0_package

ROOT = Path(__file__).resolve().parents[1]
APPROVED = json.loads((ROOT / "tests/fixtures/agent_rules_texts_approved_20261018.json").read_text())
M0 = ROOT / "configs/stage1/policies/agent_rules_m0_20261017"


def test_text_module_matches_the_approval_exactly():
    assert text.TEXT_STATUS == "approved_20261018"
    current = {name: getattr(text, name) for name in dir(text) if name.isupper() and name != "TEXT_STATUS"}
    # A new or changed model-visible string needs a new approval and a new fixture.
    assert sorted(current) == sorted(APPROVED["constants"])
    assert {name: value for name, value in current.items() if value != APPROVED["constants"][name]} == {}


def test_m0_files_are_the_approved_texts_and_form_a_valid_package():
    assignment, communication = (M0 / "assignment.md").read_text(), (M0 / "communication.md").read_text()
    assert assignment == APPROVED["m0"]["assignment"] and communication == APPROVED["m0"]["communication"]
    assert "roles" not in assignment
    package = create_m0_package(assignment, communication)
    assert package["team"] == {"assignment": assignment, "communication": communication}


def test_agent_rules_checkin_text_is_the_approved_rendering():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the check-in renderer")
    module = (ROOT / "agent_company/runtime/qwen_team_checkin.mjs").as_uri()
    checkin = APPROVED["checkin"]
    script = (f"import {{formatCheckin, collaborationLines}} from {json.dumps(module)};"
              f"console.log(JSON.stringify([formatCheckin({json.dumps(checkin['view'])}, {checkin['interval_seconds']}),"
              f" collaborationLines(null)]));")
    rendered, unavailable = json.loads(subprocess.run([node, "--input-type=module", "-e", script], check=True,
                                                      capture_output=True, text=True).stdout)
    assert rendered == checkin["text"] and unavailable == checkin["unavailable"]


def test_the_checkin_memory_line_is_the_approved_rendering():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the check-in renderer")
    module = (ROOT / "agent_company/runtime/qwen_team_checkin.mjs").as_uri()
    cases = APPROVED["checkin_memory_lines"]
    script = (f"import {{memoryLine}} from {json.dumps(module)};"
              f"console.log(JSON.stringify({json.dumps([case['memory'] for case in cases])}.map(memoryLine)));")
    rendered = json.loads(subprocess.run([node, "--input-type=module", "-e", script], check=True,
                                         capture_output=True, text=True).stdout)
    assert rendered == [case["text"] for case in cases]
