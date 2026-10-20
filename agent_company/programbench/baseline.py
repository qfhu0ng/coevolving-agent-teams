"""Render independent baseline profiles; no execution or provider configuration I/O."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable

from agent_company.programbench.total_selection import MODE as TOTAL_MODE

TEMPLATE = Path("configs/stage1/baseline_deepseek_pandoc.template.json")


def baseline_profiles(project_root: Path, repeat_indices: Iterable[int] = (1, 2, 3)) -> dict:
    indices = list(repeat_indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)):
        raise ValueError("repeat_indices must contain unique nonnegative integers")
    template = json.loads((Path(project_root) / TEMPLATE).read_text(encoding="utf-8"))
    profiles = []
    for index in indices:
        profile = deepcopy(template)
        # No per-task pair_id: the planner derives <task>#r<round> from the
        # effective round, so a CLI --repeat-index override cannot go stale.
        profile["repeat_index"] = index
        profiles.append({"filename": f"baseline_deepseek_pandoc_r{index}.json", "profile": profile})
    return {"status": "profiles_rendered_not_launched", "profiles": profiles,
            "round_count": len(indices), "runs_per_task_arm_per_profile": 1,
            "external_model_calls": 0, "automatic_execution": False,
            "task_assets_validated": False,
            "note": "Rendering does not inspect task assets or provider settings. Validate each saved profile with the existing comparison CLI; execution requires separate authorization."}


FIRST_ROUND_MODES = ("off", "global", "master", "propose")


def first_round_profiles(project_root: Path) -> dict:
    """Three solo Teams followed by one concurrent Single/propose pair: five runs."""
    template = json.loads((Path(project_root) / TEMPLATE).read_text(encoding="utf-8"))
    profiles = []
    for index, mode in enumerate(FIRST_ROUND_MODES, start=1):
        profile = deepcopy(template)
        profile["selection_protocol"] = TOTAL_MODE
        profile["first_round_wave"] = index
        profile["inference"]["role_policy_mode"] = mode
        shape = "the only Single concurrently with propose Team" if mode == "propose" else f"one standalone {mode} Team"
        profile["purpose"] = (
            f"First round wave {index}/4: {shape}, six hours per arm, fresh P0. "
            "The complete first round is five runs: one Single plus off/global/master/propose Teams, "
            "in that wave order. Explicit repeat_index and separate authorization are required. "
            "No optimizer, reviewer, evolution, automatic repetition, resume or adoption.")
        profile["limitations"] = (
            "Single runs once initially, paired only with propose; off/global/master comparisons with that Single cross time windows. "
            "Mode ordering is descriptive. Different Team modes are separate conditions; no Single range update. "
            "Valid official zero scores remain zero and missing evaluations remain unknown. Shared resources do not imply equal tokens. "
            "A triggered confirmation uses a new concurrent Single/selected-Team pair with a new explicit repeat_index. "
            "Thresholds follow docs/stage1_single_task_plan_20261014.md. S0, each wave, confirmation and evolution require separate authorization.")
        if mode != "propose":
            profile["arms"] = [arm for arm in profile["arms"] if arm["execution_mode"] == "team"]
            profile["parallelism"] = 1
        for arm in profile["arms"]:
            if arm["execution_mode"] == "team":
                arm["name"] = f"team_p0_{mode}"
        profiles.append({"filename": f"first_round_deepseek_pandoc_{mode}.json", "profile": profile,
                         "mode": mode, "recommended_repeat_index": index})
    return {"status": "profiles_rendered_not_launched", "profiles": profiles,
            "mode_order": list(FIRST_ROUND_MODES), "pair_count": 1, "wave_count": 4, "run_count": 5,
            "runs_per_task_arm_per_profile": 1, "external_model_calls": 0,
            "automatic_execution": False, "task_assets_validated": False,
            "note": "Authorize each wave separately in the listed order: off, global and master run as solo Teams; propose and the only Single run concurrently. Explicit --repeat-index is required for execution. Comparisons with the first three Teams cross API time windows; mode ranking is descriptive only. Single has no range update. Confirmation is a new simultaneous Single/selected-Team pair with a new index. Do not overlap solo Team waves without checking server capacity and launcher concurrency."}


def write_baseline_profiles(plan: dict, directory: Path) -> list[Path]:
    """Explicit local artifact write, never a launch; refuse existing output."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    for item in plan["profiles"]:
        path = directory / item["filename"]
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(item["profile"], ensure_ascii=False, indent=2) + "\n")
        paths.append(path)
    return paths
