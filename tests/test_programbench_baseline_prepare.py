"""Offline baseline profile preparation; no assets/provider/process required."""
import json
from pathlib import Path
import socket
import subprocess

import pytest

from agent_company.programbench.baseline import baseline_profiles, first_round_profiles, write_baseline_profiles
from agent_company.programbench.comparison import prepare_comparison
from agent_company.programbench.total_selection import MODE as TOTAL_MODE
from scripts import prepare_programbench_baseline as cli
from scripts import prepare_programbench_first_round as first_cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("process attempted"))
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: pytest.fail("network attempted"))


def test_defaults_are_three_independent_pandoc_p0_global_six_hour_pairs():
    plan = baseline_profiles(ROOT)
    assert plan["status"] == "profiles_rendered_not_launched" and plan["round_count"] == 3
    assert not plan["automatic_execution"] and plan["external_model_calls"] == 0
    for index, item in enumerate(plan["profiles"], 1):
        profile = item["profile"]
        assert profile["repeat_index"] == index
        # The planner derives the pair from the effective round, never a copy.
        assert all("pair_id" not in task for task in profile["tasks"]) and "pair_id" not in profile
        assert profile["inference"]["agent_timeout_sec"] == 21600
        assert profile["inference"]["role_policy_mode"] == "global"
        assert profile["inference"]["observation_interval_seconds"] == 60
        assert profile["parallelism"] == 2 and profile["evolution_enabled"] is False
        assert "Each rendered profile requests one paired round only" in profile["limitations"]
        assert "Single runs once initially" not in profile["limitations"]
        single, team = profile["arms"]
        assert single["execution_mode"] == "single" and single["assignment_policy_path"] is None
        assert team["name"] == "team_p0" and "fresh_p0_20261001" in team["assignment_policy_path"]
        assert (ROOT / team["assignment_policy_path"]).is_file()
    plan["profiles"][0]["profile"]["tasks"][0]["instructions"] = "changed"
    assert plan["profiles"][1]["profile"]["tasks"][0]["instructions"] != "changed"


def test_default_cli_reads_only_template_and_does_not_write(monkeypatch, capsys):
    original = Path.read_text
    reads = []

    def template_only(path, *args, **kwargs):
        reads.append(path.name)
        assert path.name == "baseline_deepseek_pandoc.template.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", template_only)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("default plan attempted a write"))
    assert cli.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["round_count"] == 3 and plan["task_assets_validated"] is False
    assert reads == ["baseline_deepseek_pandoc.template.json"]


@pytest.mark.parametrize("first_round", [False, True])
def test_written_round_profiles_are_inputs_to_the_existing_comparison_planner(tmp_path, first_round):
    plan = first_round_profiles(ROOT) if first_round else baseline_profiles(ROOT)
    for item in plan["profiles"]:
        profile = item["profile"]
        task = profile["tasks"][0]
        directory = tmp_path / task["path"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "task.yaml").write_text("repository: jgm/pandoc\ncommit: '5caad90" + "0" * 33 + "'\nlanguage: haskell\n")
        (directory / "tests.json").write_text("MUST NOT READ")
        instruction = tmp_path / task["instructions"]
        instruction.parent.mkdir(parents=True, exist_ok=True)
        instruction.write_text("Synthetic public task description.")
        for key in ("assignment_policy_path", "communication_policy_path"):
            policy = tmp_path / profile["arms"][-1][key]
            policy.parent.mkdir(parents=True, exist_ok=True)
            policy.write_text((ROOT / profile["arms"][-1][key]).read_text())
    paths = write_baseline_profiles(plan, tmp_path / "profiles")
    for index, path in enumerate(paths, 1):
        prepared = prepare_comparison(path, tmp_path, **({"repeat_index": index} if first_round else {})).plan
        assert prepared["repeat_index"] == index
        expected = 1 if first_round and index < 4 else 2
        assert prepared["new_rollouts"] == expected and len(prepared["arms"]) == expected
        assert prepared["tasks"][0]["pair_id"] == f"jgm__pandoc.5caad90#r{index}"
        assert prepared["runs_per_task_arm"] == 1 and len(prepared["waves"]) == 1
        assert prepared["inference"]["observation_interval_seconds"] == 60
    # A CLI --repeat-index override moves the pair with the round; no stale r1.
    if first_round:
        for index, path in enumerate(paths, 1):
            with pytest.raises(ValueError, match="positive repeat_index"):
                prepare_comparison(path, tmp_path, repeat_index=0)
            with pytest.raises(ValueError, match="must equal first_round_wave"):
                prepare_comparison(path, tmp_path, repeat_index=index + 1)
    else:
        overridden = prepare_comparison(paths[0], tmp_path, repeat_index=7).plan
        assert overridden["repeat_index"] == overridden["tasks"][0]["repeat_index"] == 7
        assert overridden["tasks"][0]["pair_id"] == "jgm__pandoc.5caad90#r7"
    with pytest.raises(FileExistsError):
        write_baseline_profiles(plan, tmp_path / "profiles")


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [True], [1.5]])
def test_invalid_rounds_are_rejected(indices):
    with pytest.raises(ValueError, match="unique nonnegative integers"):
        baseline_profiles(ROOT, indices)


def test_first_round_profiles_preserve_shared_conditions_and_do_not_pool_modes():
    template = json.loads((ROOT / "configs/stage1/baseline_deepseek_pandoc.template.json").read_text())
    plan = first_round_profiles(ROOT)
    assert plan["mode_order"] == ["off", "global", "master", "propose"]
    assert plan["pair_count"] == 1 and plan["run_count"] == 5 and plan["wave_count"] == 4
    assert not plan["automatic_execution"]
    for index, item in enumerate(plan["profiles"], 1):
        profile = item["profile"]
        assert item["recommended_repeat_index"] == index
        assert profile["selection_protocol"] == TOTAL_MODE
        assert profile["first_round_wave"] == index
        assert "repeat_index" not in profile and "pair_id" not in profile
        assert profile["inference"]["role_policy_mode"] == item["mode"]
        paired = item["mode"] == "propose"
        assert [arm["execution_mode"] for arm in profile["arms"]] == (["single", "team"] if paired else ["team"])
        assert profile["parallelism"] == (2 if paired else 1)
        assert profile["arms"][-1]["name"] == f"team_p0_{item['mode']}"
        # Saved profiles match the renderer; normalize the authorized wave
        # shape and Team mode while preserving all inference/task conditions.
        assert json.loads((ROOT / "configs/stage1" / item["filename"]).read_text()) == profile
        profile["inference"]["role_policy_mode"] = template["inference"]["role_policy_mode"]
        profile["arms"][-1]["name"] = template["arms"][1]["name"]
        if not paired:
            profile["arms"].insert(0, template["arms"][0])
        profile["parallelism"] = template["parallelism"]
        assert "five runs" in profile["purpose"]
        assert "Single runs once initially" in profile["limitations"]
        profile["purpose"] = template["purpose"]
        profile["limitations"] = template["limitations"]
        profile.pop("selection_protocol")
        profile.pop("first_round_wave")
        assert profile == template


def test_first_round_default_cli_is_read_only(monkeypatch, capsys):
    original = Path.read_text

    def template_only(path, *args, **kwargs):
        assert path.name == "baseline_deepseek_pandoc.template.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", template_only)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("default plan attempted a write"))
    assert first_cli.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode_order"] == ["off", "global", "master", "propose"]
    assert plan["external_model_calls"] == 0 and not plan["task_assets_validated"]
