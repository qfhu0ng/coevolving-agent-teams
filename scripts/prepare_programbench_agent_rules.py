#!/usr/bin/env python3
"""Create agent_rules_v1 batch inputs; never generate a candidate or launch.

Batch 1 (default): the approved M0 package and the Single + M0 Team profile.
--task-id selects the ProgramBench task (default Pandoc; user, 2026-10-19: also ctags). Only the
template's task entry changes; the task must exist under the execution checkout.
Batch 2: the reviewed candidate M1 package and its review, copied byte for byte, and the
M1 Team profile (user, 2026-10-19: M0 is not rerun). The candidate must be revision 1 of
the approved M0, with its review file beside it.

Run with the execution checkout as project-root and a new directory outside it
(normally the deployment's profiles/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_company.policy.role_package import (create_m0_package, load_policy_package,  # noqa: E402
                                               validate_agent_rules_package)

M0 = "configs/stage1/policies/agent_rules_m0_20261017"
PANDOC = "jgm__pandoc.5caad90"
TASKS = "external/programbench-study/src/programbench/data/tasks"


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")


def prepare_inputs(project_root: Path, output: Path, *, batch: int = 1, candidate: Path | None = None,
                   task_id: str = PANDOC) -> dict:
    root, output = Path(project_root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("agent_rules_v1 inputs must be outside the execution checkout")
    if batch not in (1, 2) or (batch == 2) != (candidate is not None):
        raise ValueError("Batch 2, and only batch 2, takes the reviewed candidate package")
    source = root / M0
    package = validate_agent_rules_package(create_m0_package(
        (source / "assignment.md").read_bytes().decode("utf-8"),
        (source / "communication.md").read_bytes().decode("utf-8")))
    profile = json.loads((root / f"configs/stage1/agent_rules_pandoc_batch{batch}.template.json")
                         .read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", task_id) or not (root / TASKS / task_id / "task.yaml").is_file():
        raise ValueError("Unknown ProgramBench task")
    if len(profile["tasks"]) != 1:
        raise ValueError("The agent_rules_v1 template has exactly one task")
    profile["tasks"][0].update(task_id=task_id, path=f"{TASKS}/{task_id}")
    stem = "agent_rules_pandoc" if task_id == PANDOC else f"agent_rules_{task_id}"
    if batch == 1:
        for arm in profile["arms"]:
            arm["policy_package_path"] = str(output / "agent_rules_package_m0.json")
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "agent_rules_package_m0.json", package)
        _write(output / f"{stem}_batch1.json", profile)
        return {"status": "inputs_prepared_not_launched", "batch": 1, "task_id": task_id, "directory": str(output),
                "profile": str(output / f"{stem}_batch1.json"),
                "m0_package": str(output / "agent_rules_package_m0.json"), "candidate_created": False}
    candidate = Path(candidate).resolve()
    m1 = validate_agent_rules_package(load_policy_package(candidate))
    review_ref = m1.get("review_ref")
    if (m1["revision"] != 1 or m1["parent"] != {"package_id": package["package_id"], "revision": 0}
            or not isinstance(review_ref, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", review_ref)
            or not (candidate.parent / review_ref).is_file() or m1["team"] == package["team"]
            or json.loads((candidate.parent / review_ref).read_text(encoding="utf-8")).get("status") != "completed"):
        raise ValueError("The candidate must be a reviewed revision 1 of M0 with new texts and its completed review "
                         "beside it")
    for arm in profile["arms"]:
        arm["policy_package_path"] = str(output / "agent_rules_package_m1.json")
    output.mkdir(parents=True, exist_ok=False)
    # Byte-for-byte copies, so the launch record's hash is the generation's own package.
    (output / "agent_rules_package_m1.json").write_bytes(candidate.read_bytes())
    (output / review_ref).write_bytes((candidate.parent / review_ref).read_bytes())
    _write(output / f"{stem}_batch2.json", profile)
    digest = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ("agent_rules_package_m1.json", review_ref)}
    return {"status": "inputs_prepared_not_launched", "batch": 2, "task_id": task_id, "directory": str(output),
            "profile": str(output / f"{stem}_batch2.json"),
            "m1_package": str(output / "agent_rules_package_m1.json"), "m1_review": str(output / review_ref),
            "candidate_source": str(candidate), "sha256": digest, "candidate_created": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--write-profiles", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--task-id", default=PANDOC, help="ProgramBench task (default: %(default)s)")
    parser.add_argument("--candidate-package", type=Path,
                        help="Batch 2 only: the reviewed candidate package (a generation's package.json)")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_inputs(args.project_root, args.write_profiles, batch=args.batch,
                                    candidate=args.candidate_package, task_id=args.task_id),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
