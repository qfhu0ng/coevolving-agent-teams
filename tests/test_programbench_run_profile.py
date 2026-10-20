"""The selected pair uses baseline resources, not the earlier Roadmap defaults."""
import json
from pathlib import Path

from agent_company.programbench.task import (
    OFFICIAL_EVALUATOR_CPUS,
    OFFICIAL_INFERENCE_CPUS,
    OFFICIAL_INFERENCE_MEMORY_MB,
    OFFICIAL_INFERENCE_TIMEOUT_SEC,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_profiles_use_committable_policy_and_instruction_inputs():
    profiles = sorted((ROOT / "configs/stage1").glob("programbench_*.json"))
    assert profiles
    for path in profiles:
        profile = json.loads(path.read_text())
        policy_rows = profile.get("arms", [profile])
        for row in policy_rows:
            if row.get("execution_mode") == "single":
                assert row.get("assignment_policy_path") in (None, "")
                assert row.get("communication_policy_path") in (None, "")
                continue
            if path.name.endswith(".template.json") and row.get("policy_version", 0) > 0 \
                    and "assignment_policy_path" not in row and "communication_policy_path" not in row:
                # A template's candidate arm gets its generated A/C only when the followup
                # runner instantiates it; every other arm of the template is checked below.
                assert profile.get("evolution_enabled") is True and row["execution_mode"] == "team"
                continue
            if row.get("study_role") == "evolved":
                # This arm has no policy input: the one-task runner generates
                # its A/C after the fresh baseline Team run (P0 or P3), before
                # dispatching the arm.
                baseline = [arm for arm in policy_rows if arm.get("study_role") == "baseline"]
                assert profile.get("evolution_enabled") is True
                assert row["execution_mode"] == "team"
                assert len(baseline) == 1
                assert row["policy_version"] > baseline[0]["policy_version"]
                assert row.get("assignment_policy_path") is None
                assert row.get("communication_policy_path") is None
                assert len(profile["tasks"]) == 1 and len(policy_rows) == 3
                assert {arm["study_role"] for arm in policy_rows} == {"single", "baseline", "evolved"}
                continue
            for key in ("assignment_policy_path", "communication_policy_path"):
                relative = Path(row[key])
                assert relative.parts[:3] == ("configs", "stage1", "policies")
                assert not relative.is_absolute() and ".." not in relative.parts
                assert (ROOT / relative).is_file()
                assert (ROOT / relative).read_text().strip()
        for task in profile.get("tasks", []):
            assert task["instructions"] == "configs/stage1/programbench_instruction.md"
            assert (ROOT / task["instructions"]).is_file()


def test_five_task_profile_enables_checkins_only_for_the_p3_team_runtime():
    profile = json.loads((ROOT / "configs/stage1/programbench_agensh5_single_vs_p3_readonly_master.json").read_text())
    assert profile["experiment_status"] == "stopped_by_user_20260928"
    assert profile["launch_requires_new_user_authorization"] is True
    assert profile["experiment_record"] == "runs/stage1_glm_programbench_agensh5_checkin_20260928/START.md"
    assert len(profile["tasks"]) == 5
    assert [(arm["name"], arm["execution_mode"], arm["policy_version"]) for arm in profile["arms"]] == [
        ("single", "single", "none"), ("p3_master_readonly_checkin", "team", "P3")]
    assert profile["inference"]["team_checkin_seconds"] == 1800
    assert profile["inference"]["max_model_calls"] is None
    assert profile["inference"]["agent_timeout_sec"] == 21600
    assert profile["parallelism"] == 2 and profile["evolution_enabled"] is False


def test_no_call_cap_profile_only_changes_budget_scope_and_purpose():
    capped = json.loads((ROOT / "configs/stage1/programbench_fd_glm_p3.json").read_text())
    uncapped = json.loads((ROOT / "configs/stage1/programbench_fd_no_call_cap_glm_p3.json").read_text())
    assert capped["inference"]["max_model_calls"] == 1000
    assert uncapped["inference"]["max_model_calls"] is None
    assert uncapped["inference"]["agent_timeout_sec"] == 21600
    assert uncapped["inference"]["step_scope"] == "all_roles_and_purposes_metered_without_logical_call_cap"
    for data in (capped, uncapped):
        data.pop("purpose")
        data["inference"].pop("max_model_calls")
        data["inference"].pop("step_scope")
    assert capped == uncapped


def test_pb1_profile_records_stopped_unadopted_four_team_comparison():
    profile = json.loads((ROOT / "configs/stage1/programbench_ac_pb1_heldout_20260924.json").read_text())
    assert profile["experiment_status"] == "user_stopped_for_api_maintenance_20260924"
    assert profile["candidate_status"] == "experimental_not_adopted"
    assert profile["restart_requires_new_user_authorization"] is True
    assert [task["task_id"] for task in profile["tasks"]] == ["chmln__sd.87d1ba5", "lz4__lz4.1519f46"]
    assert [(arm["execution_mode"], arm["policy_version"]) for arm in profile["arms"]] == [
        ("team", "P3"), ("team", "PB1")]
    assert len(profile["tasks"]) * len(profile["arms"]) == 4
    assert profile["parallelism"] == 2 and profile["evolution_enabled"] is False
    assert profile["inference"]["max_model_calls"] is None
    assert profile["inference"]["agent_timeout_sec"] == 21600
    assert profile["selection"]["candidate_auto_adopted"] is False
    assert profile["selection"]["policy_updated_from_heldout"] is False


def test_public_provider_profile_has_only_environment_credential_reference():
    provider = json.loads((ROOT / "configs/stage1/self_hosted_glm.json").read_text())
    assert "api_key" not in provider
    assert provider["api_key_env"] == "STAGE1_GLM_API_KEY"
    assert provider["rates"] is None


def test_pair_profile_matches_adapter_resource_defaults():
    profile = json.loads((ROOT / "configs/stage1/programbench_glm_p3.json").read_text())
    assert profile["execution_modes"] == ["single", "team"]
    assert profile["runs_per_mode"] == 1 and profile["evolution_enabled"] is False
    assert profile["policy_version"] == "P3"
    assert (ROOT / profile["assignment_policy_path"]).is_file()
    assert (ROOT / profile["communication_policy_path"]).is_file()
    limits = profile["inference"]
    assert limits["agent_timeout_sec"] == OFFICIAL_INFERENCE_TIMEOUT_SEC == 21600
    assert limits["cpus"] == OFFICIAL_INFERENCE_CPUS == 20
    assert limits["memory_mb"] == OFFICIAL_INFERENCE_MEMORY_MB == 61440
    assert limits["memory_swap_mb"] == limits["memory_mb"]
    assert limits["max_model_calls"] == 1000
    assert limits["resource_scope"] == "whole_task_shared_by_all_agents"
    assert profile["evaluation"]["docker_cpus"] == OFFICIAL_EVALUATOR_CPUS == 10
    provider = json.loads((ROOT / profile["provider_config"]).read_text())
    assert provider["model"] == "RadixArk/GLM-5.3-NVFP4"
    assert provider["max_attempts"] == 11
    assert provider["request_timeout_seconds"] == 600
