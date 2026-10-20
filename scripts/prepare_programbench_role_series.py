#!/usr/bin/env python3
"""Create local role-series inputs from existing P0; never generate v1 or launch.

Run with the execution checkout as project-root and a new directory outside it
(normally the deployment's profiles/). v1 is supplied later after review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_company.policy.role_package import create_p0_package


def prepare_inputs(project_root: Path, output: Path) -> dict:
    root, output = Path(project_root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Role-series inputs must be outside the execution checkout")
    source = root / "configs/stage1/policies/fresh_p0_20261001"
    package = create_p0_package(
        (source / "assignment_v000.md").read_bytes().decode("utf-8"),
        (source / "communication_v000.md").read_bytes().decode("utf-8"))
    profiles = []
    for batch in (1, 2):
        profile = json.loads((root / f"configs/stage1/role_series_pandoc_batch{batch}.template.json").read_text())
        for arm in profile["arms"]:
            name = "p0" if arm["policy_version"] == 0 else "v1"
            arm["policy_package_path"] = str(output / f"role_package_{name}.json")
        profiles.append(profile)
    output.mkdir(parents=True, exist_ok=False)
    (output / "role_package_p0.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    for batch, profile in enumerate(profiles, 1):
        (output / f"role_series_pandoc_batch{batch}.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    return {"status": "inputs_prepared_not_launched", "directory": str(output),
            "profiles": [str(output / f"role_series_pandoc_batch{batch}.json") for batch in (1, 2)],
            "p0_package": str(output / "role_package_p0.json"),
            "v1_package": str(output / "role_package_v1.json"), "v1_created": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--write-profiles", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_inputs(args.project_root, args.write_profiles), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
