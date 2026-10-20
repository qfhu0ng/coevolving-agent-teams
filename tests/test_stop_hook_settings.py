"""Stop-hook settings traverse planning without starting a paid runtime."""
import json
import subprocess
from pathlib import Path

import pytest

from agent_company.programbench.comparison import prepare_comparison, runner_inference_options
from agent_company.programbench.runner import ProgramBenchRunner
from agent_company.runtime.qwen import QwenTeamRunner, validate_stop_hook
from agent_company.runtime.relay import ProviderConfig
from test_rejection_integration import native_qwen
from test_programbench_total_evidence import trial, write
from agent_company.programbench.total_evidence import compare_trials


HOOK = {"reason": "TEST_ONLY_CONTINUE {remaining_seconds}", "budget_notice": "TEST_ONLY_BUDGET"}


def test_new_profiles_preserve_user_approved_text():
    expected = {
        "reason": "Time remaining: {remaining_seconds} seconds. Continue useful work on the task within your existing role permissions. Do not end the session solely because the current implementation appears complete: compare the delivered program with the reference executable on inputs and options you have not tried yet, and address the differences and remaining issues you find. When less than 900 seconds remain, stop starting new work; make sure the delivery builds with compile.sh and all completed work is in the delivery repository, then finish.",
        "budget_notice": "The runtime will remind you to continue if you try to finish before the wall-time allowance expires. Use the remaining time for useful work within your existing role permissions.",
    }
    configs = Path(__file__).resolve().parents[1] / "configs/stage1"
    names = [f"first_round_deepseek_pandoc_{mode}.json" for mode in ("off", "global", "master", "propose")]
    names += ["baseline_deepseek_pandoc.template.json", "programbench_pandoc_evolved_followup.template.json"]
    for name in names:
        assert json.loads((configs / name).read_text())["inference"]["stop_hook"] == expected


@pytest.mark.parametrize("location", ["inference", "native"])
def test_stop_hook_changes_cannot_be_compared_as_same_conditions(tmp_path, location):
    # Full native settings compare same-mode runs; Single/Team share inference
    # but legitimately differ in several native Team-only switches.
    a = trial(tmp_path, "first", "incumbent", 10, single=location == "inference")
    b = trial(tmp_path, "second", "team", 20)
    assert compare_trials(a, b)["decision"] == "needs_confirmation"
    path = b.parent.parent / "run_settings.json" if location == "inference" else b / "team/launch.json"
    data = json.loads(path.read_text())
    (data["inference"] if location == "inference" else data)["stop_hook"] = HOOK
    write(path, data)
    assert compare_trials(a, b)["decision"] == "inconclusive"


def test_native_stop_callback_deadline_abort_and_isolation():
    _, node = native_qwen()
    result = subprocess.run([node, str(Path(__file__).with_name("native_stop_hook_unit.mjs"))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", [True, {}, "yes", {"reason": "missing notice"},
    {"reason": "missing placeholder", "budget_notice": "notice"},
    {"reason": "{remaining_seconds}", "budget_notice": " "},
    {"reason": "{remaining_seconds}{remaining_seconds}", "budget_notice": "notice"},
    {**HOOK, "extra": True}])
def test_invalid_stop_settings_rejected_before_execution(value):
    with pytest.raises(ValueError, match="stop_hook"):
        QwenTeamRunner("unused-cli", stop_hook=value)
    with pytest.raises(ValueError, match="stop_hook"):
        ProgramBenchRunner(ProviderConfig("fixture", "http://unused"), Path("qwen"),
                           Path("node"), stop_hook=value)


def test_stop_settings_copy_preserves_exact_text_and_default():
    source = dict(HOOK)
    result = validate_stop_hook(source)
    assert result == source and result is not source
    assert validate_stop_hook(None) is None
    source["reason"] = "changed"
    assert result == HOOK


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_profile_plan_and_runner_keep_same_stop_text(tmp_path, execution_mode):
    task = tmp_path / "synthetic__task.1111111"
    task.mkdir()
    (task / "task.yaml").write_text("repository: synthetic/task\ncommit: '" + "1" * 40 + "'\nlanguage: c\n")
    (task / "tests.json").write_text("Do not read official test results while planning.")
    (tmp_path / "instruction.md").write_text("Synthetic task")
    (tmp_path / "a.md").write_text("Synthetic assignment")
    (tmp_path / "c.md").write_text("Synthetic communication")
    profile = {"tasks": [{"task_id": task.name, "path": task.name, "instructions": "instruction.md"}],
        "arms": [{"name": execution_mode, "execution_mode": execution_mode, "policy_version": 0,
                  "assignment_policy_path": "a.md" if execution_mode == "team" else None,
                  "communication_policy_path": "c.md" if execution_mode == "team" else None}],
        "provider_config": "provider-not-read.json", "parallelism": 1,
        "inference": {"agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 61440,
                      "memory_swap_mb": 61440, "max_model_calls": None, "stop_hook": HOOK},
        "evaluation": {"docker_cpus": 10}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    plan = prepare_comparison(path, tmp_path, repeat_index=1).plan
    options = runner_inference_options(plan["inference"])
    runner = ProgramBenchRunner(ProviderConfig("fixture", "http://unused"),
                               Path("qwen"), Path("node"), **options)
    assert plan["inference"]["stop_hook"] == runner.stop_hook == HOOK
    assert not (tmp_path / "provider-not-read.json").exists()
