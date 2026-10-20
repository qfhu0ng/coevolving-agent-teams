#!/usr/bin/env python3
"""Review an existing agent_rules_v1 candidate master strategy again after its review failed.

The optimizer is not called and the candidate is not changed, run, adopted or scheduled. Without
--authorize-model-calls it runs every check that costs nothing and prints what it would do (exit 0
when they pass); the reviewer makes real, billed provider calls, which need the user's explicit
authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_company.deploy.code_version import code_version, code_version_label  # noqa: E402
from agent_company.evolution import agent_rules_outer_text as outer_text  # noqa: E402
from agent_company.evolution.agent_rules_generation import (REVIEWER_FINAL_CALLS, REVIEWER_REPAIR_CALLS,  # noqa: E402
                                                            REVIEWER_ROUNDS, review_candidate, review_problems)
from agent_company.evolution.harness_capabilities import target_harness  # noqa: E402
from agent_company.policy.role_package import load_policy_package, validate_agent_rules_package  # noqa: E402
from agent_company.runtime.provider import load_provider, public_provider  # noqa: E402
from agent_company.runtime.relay import ProviderConfig, TokenRates  # noqa: E402


def provider_without_key(path: Path) -> ProviderConfig:
    """The provider's public fields for the free checks; the credential is read only for real calls."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "api_key" in data:
        raise ValueError("Use api_key_env, not a credential value, in provider configuration")
    data.pop("api_key_env", None)
    if data.get("rates") is not None:
        data["rates"] = TokenRates(**data["rates"])
    return ProviderConfig(api_key="", **data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True, help="Generation output whose review failed")
    parser.add_argument("--package", type=Path, required=True, help="The incumbent package it was generated from")
    parser.add_argument("--profile", type=Path, required=True,
                        help="The profile it was generated with (inference and provider_config)")
    parser.add_argument("--label", default="2",
                        help="Suffix of this review's files (reviewer_<label>/, review_<label>.json, ...); default 2")
    parser.add_argument("--authorize-model-calls", action="store_true",
                        help="Make the real reviewer provider calls (billed)")
    args = parser.parse_args(argv)
    try:
        package = validate_agent_rules_package(load_policy_package(args.package))
    except (OSError, ValueError) as exc:
        parser.error(f"--package is not a schema 2 agent_rules_v1 package: {exc}")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    inference = profile.get("inference")
    provider_path = (ROOT / profile["provider_config"]).resolve() if isinstance(profile.get("provider_config"), str) else None
    problems = []
    try:
        target_harness(inference, execution_mode="team", policy_protocol="agent_rules_v1")
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        problems.append(f"profile inference does not describe an agent_rules_v1 Team: {exc}")
    provider = None
    if provider_path is None or not provider_path.is_file():
        problems.append("profile provider_config is not a readable file")
    else:
        provider = provider_without_key(provider_path)
        problems.extend(review_problems(args.generation, package, provider, args.label, inference))
    if args.authorize_model_calls and not outer_text.TEXT_STATUS.startswith("approved_"):
        problems.append("the outer-loop texts are not approved yet; real calls are refused")
    summary = {"mode": "check_only" if not args.authorize_model_calls else "authorized",
               "generation": str(args.generation), "incumbent": package["package_id"],
               "reviewer_rounds": REVIEWER_ROUNDS, "reviewer_final_calls": REVIEWER_FINAL_CALLS,
               "reviewer_repair_calls": REVIEWER_REPAIR_CALLS,
               "would_write": [f"reviewer_{args.label}/", f"review_{args.label}.json",
                               f"report_review_{args.label}.json", "package.json (if reviewed)"],
               "text_status": outer_text.TEXT_STATUS, "problems": problems, "model_calls": 0}
    if problems or not args.authorize_model_calls:
        print(json.dumps(summary, indent=2))
        return 1 if problems else 0
    provider = load_provider(provider_path)
    report = review_candidate(package=package, generation=args.generation, inference=inference, provider=provider,
                              code_version=code_version_label(code_version(ROOT)), label=args.label)
    print(json.dumps({**report, "provider": public_provider(provider)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "reviewed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
