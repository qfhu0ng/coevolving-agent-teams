#!/usr/bin/env python3
"""Generate and review one candidate master strategy from recorded agent_rules_v1 Team trials (plan v4 section 12, item 6).

It never runs, adopts or schedules the candidate. Without --authorize-model-calls it runs
every check that costs nothing and prints what it would read (exit 0 when they pass): the
optimizer and reviewer make real, billed provider calls, which need the user's explicit
authorization, and they are refused while the outer-loop texts are still drafts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_company.analysis.agent_rules_evidence import agent_rules_record  # noqa: E402
from agent_company.deploy.code_version import code_version, code_version_label  # noqa: E402
from agent_company.evolution import agent_rules_outer_text as outer_text  # noqa: E402
from agent_company.evolution.agent_rules_generation import generate, source_strategy, strategy_problems  # noqa: E402
from agent_company.evolution.harness_capabilities import native_version, target_harness  # noqa: E402
from agent_company.policy.role_package import load_policy_package, validate_agent_rules_package  # noqa: E402
from agent_company.runtime.provider import load_provider, public_provider  # noqa: E402


def generator_checkout() -> dict:
    """Which code generated the candidate: HEAD and whether the tree had uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {"head": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Incumbent schema 2 package")
    parser.add_argument("--trial", type=Path, action="append", required=True,
                        help="Recorded agent_rules_v1 Team trial directory; repeat for cumulative evidence")
    parser.add_argument("--profile", type=Path, required=True,
                        help="Profile whose inference and provider_config the candidate will use")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-package", type=Path, help="Installed Qwen package, for the native version")
    parser.add_argument("--authorize-model-calls", action="store_true",
                        help="Make the real optimizer and reviewer provider calls (billed)")
    args = parser.parse_args(argv)
    problems = []
    try:
        package = validate_agent_rules_package(load_policy_package(args.package))
    except (OSError, ValueError) as exc:
        parser.error(f"--package is not a schema 2 agent_rules_v1 package: {exc}")
    if args.candidate_id == package["package_id"] or not args.candidate_id.strip():
        problems.append("--candidate-id must differ from the incumbent package_id")
    if args.output.exists():
        problems.append("--output already exists")
    if args.output.with_name(args.output.name + ".building").exists():
        problems.append("--output.building is left from an earlier failed build; check and remove it first")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    inference = profile.get("inference")
    provider_path = (ROOT / profile["provider_config"]).resolve() if isinstance(profile.get("provider_config"), str) else None
    try:
        target_harness(inference, execution_mode="team", policy_protocol="agent_rules_v1")
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        problems.append(f"profile inference does not describe an agent_rules_v1 Team: {exc}")
    if provider_path is None or not provider_path.is_file():
        problems.append("profile provider_config is not a readable file")
    records = {str(trial): trial.is_dir() and agent_rules_record(trial) is not None for trial in args.trial}
    if not all(records.values()):
        problems.append("every --trial must be a recorded agent_rules_v1 Team trial")
    sources = []
    for trial in args.trial:
        try:
            sources.append(source_strategy(trial))
        except (OSError, ValueError) as exc:
            problems.append(f"no readable master strategy in a --trial run_request.json: {exc}")
    if len(sources) == len(args.trial):
        problems.extend(strategy_problems(package, sources))
    if args.authorize_model_calls and not outer_text.TEXT_STATUS.startswith("approved_"):
        problems.append("the outer-loop texts are not approved yet; real calls are refused")
    summary = {"mode": "check_only" if not args.authorize_model_calls else "authorized",
               "incumbent": package["package_id"], "trials": records, "would_write": str(args.output),
               "trial_strategies": [{"package_id": s["package_id"], "revision": s["revision"]} for s in sources],
               "text_status": outer_text.TEXT_STATUS, "problems": problems, "model_calls": 0}
    if problems or not args.authorize_model_calls:
        print(json.dumps(summary, indent=2))
        return 1 if problems else 0
    provider = load_provider(provider_path)
    report = generate(package=package, trials=args.trial, inference=inference, provider=provider, output=args.output,
                      candidate_id=args.candidate_id, package_path=args.package,
                      code_version=code_version_label(code_version(ROOT)), generator=generator_checkout(),
                      native_version=native_version(args.qwen_package) if args.qwen_package else None)
    print(json.dumps({**report, "provider": public_provider(provider)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "reviewed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
