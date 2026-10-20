#!/usr/bin/env python3
"""Perform one explicit operation on one final ProgramBench deployment.

deploy, verify, prepare, status and explicitly scripted smoke use no model API.
launch, probe and real smoke are paid operations requiring separate user approval.
No operation chains another one or retries an unclear response.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.deploy.deployment import Deployment, deploy, remote_verify
from agent_company.deploy.launch import (
    ARMS, RUNNER_FLAGS, STEPS, launch, preparation, prepare, status,
)
from agent_company.deploy.ssh import ssh
from agent_company.runtime_io import write_json_atomic

DEFAULT_ASSETS = Path(os.environ.get("AGENT_COMPANY_REMOTE_ROOT", "/srv/agent-company"))


def _explicit_runner_arguments(argv, runner):
    flags = RUNNER_FLAGS.get(runner, set())
    result = []
    for index, token in enumerate(argv):
        flag = token.split("=", 1)[0]
        if flag not in flags:
            continue
        result.append(token)
        if "=" not in token:
            result.append(argv[index + 1])
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--task-id", default="jgm__pandoc.5caad90")
    actions = parser.add_mutually_exclusive_group()
    for name in ("deploy", "verify", "launch", "smoke", "probe", "status"):
        actions.add_argument("--" + name, action="store_true")
    parser.add_argument("--prepare", action="store_true",
                        help="Read-only runner plan, or combine with --smoke/--probe for preparation preflight")
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--commit")
    parser.add_argument("--local-record-dir", type=Path)
    parser.add_argument("--runner", choices=("comparison", "followup"))
    parser.add_argument("--step", choices=STEPS)
    parser.add_argument("--profile")
    parser.add_argument("--trigger-report", help="Launcher-only saved confirmation trigger; never passed to runner")
    parser.add_argument("--paused-round", help="Launcher-only paused round for approved dispatch; never passed to runner")
    for flag in ("repeat-index", "parallelism", "pair-id", "source-run", "evolution-round", "batch-index"):
        parser.add_argument("--" + flag)
    for flag in ("previous-round", "approved-trial", "history-trial"):
        parser.add_argument("--" + flag, action="append", default=[])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--sequence", type=int)
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument("--scripted", action="store_true")
    kind.add_argument("--real", action="store_true")
    parser.add_argument("--max-calls", type=int, choices=(40,), default=40)
    parser.add_argument("--timeout", type=int, choices=(420,), default=420)
    return parser


def _remote_call(deployment, argv, *, paid=False, transport=None):
    transport = transport or ssh
    remote_argv = []
    arguments = iter(argv)
    for token in arguments:
        if token == "--local-record-dir":
            next(arguments)
        elif not token.startswith("--local-record-dir="):
            remote_argv.append(token)
    command = [str(deployment.python), "-u", str(deployment.code / "scripts/launch_programbench_formal.py"),
               *remote_argv, "--remote"]
    payload = None
    key = None
    if paid:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key.strip():
            raise ValueError("DEEPSEEK_API_KEY is not set; credential files are never read")
        payload = json.dumps({"api_key": key})
    shell = (f"cd {shlex.quote(str(deployment.code))} && "
             "env -u DEEPSEEK_API_KEY -u STAGE1_GLM_API_KEY PYTHONDONTWRITEBYTECODE=1 "
             f"PYTHONPATH={shlex.quote(f'{deployment.evaluator}:{deployment.code}')} " + shlex.join(command))
    try:
        response = transport(shell, stdin=payload, timeout=2000)
        text = response.stdout
        if key:
            text = text.replace(key, "[REDACTED]")
        report = json.loads(text)
        if not isinstance(report, dict):
            raise ValueError("Expected JSON response")
        if response.returncode:
            return {"status": "failed", "remote_returncode": response.returncode,
                    "result": report, "next_action": "stop_and_inspect_no_retry"}
        return report
    except (OSError, subprocess.SubprocessError, ValueError):
        raise RuntimeError("Remote result unclear; inspect this deployment, never resend automatically") from None


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(raw)
    if not any((args.prepare, args.deploy, args.verify, args.launch, args.smoke, args.probe, args.status)):
        parser.error("Select an operation")
    if args.prepare and any((args.deploy, args.verify, args.launch, args.status)):
        parser.error("--prepare can only be combined with --smoke or --probe")
    preparation_plan = args.prepare and (args.smoke or args.probe)
    paid = not args.prepare and (args.launch or args.probe or args.real)
    if args.deploy and (args.remote or not args.commit):
        parser.error("--deploy requires --commit and is a workstation operation")
    if args.smoke and (not args.arm or args.sequence is None or not (args.scripted or args.real)):
        parser.error("--smoke requires --arm, --sequence and explicit --scripted or --real")
    if args.probe and args.sequence is None:
        parser.error("--probe requires --sequence")
    if not args.smoke and (args.scripted or args.real or args.arm):
        parser.error("--arm, --scripted and --real are smoke-only options")
    if ((args.prepare and not preparation_plan) or args.launch) and not all((args.runner, args.step, args.profile, args.repeat_index is not None)):
        parser.error("--prepare/--launch requires runner, step, profile and explicit repeat-index")
    if args.runner == "followup" and not args.source_run:
        parser.error("followup requires --source-run")
    supplied = {token.split("=", 1)[0] for token in raw}
    forbidden = set.union(*RUNNER_FLAGS.values()) - RUNNER_FLAGS.get(args.runner, set())
    if supplied & forbidden:
        parser.error("Argument is not supported by the selected runner")
    try:
        deployment = Deployment(args.final, args.assets_root, args.task_id)
        if args.deploy:
            folder = args.local_record_dir or Path("runs") / args.final.name
            result = deploy(Path(__file__).resolve().parents[1], deployment, args.commit,
                            local_record_dir=folder)
        elif not args.remote:
            result = _remote_call(deployment, raw, paid=paid)
            if args.verify and result.get("status") == "verified":
                folder = args.local_record_dir or Path("runs") / args.final.name
                folder.mkdir(parents=True, exist_ok=True)
                write_json_atomic(folder / "VERIFY.json", result)
        elif args.verify:
            result = remote_verify(deployment)
        elif args.status:
            result = status(deployment)
        elif args.smoke or args.probe:
            kind = "cache_probe" if args.probe else "smoke_scripted" if args.scripted else "smoke_real"
            result = preparation(deployment, kind=kind, arm=args.arm, sequence=args.sequence,
                                 payload=lambda: json.load(sys.stdin), prepare_only=args.prepare)
        else:
            spec = {"runner": args.runner, "step": args.step, "profile": args.profile,
                    "trigger_report": args.trigger_report,
                    "paused_round": args.paused_round,
                    "arguments": _explicit_runner_arguments(raw, args.runner)}
            result = (launch(deployment, spec, lambda: json.load(sys.stdin)) if args.launch
                      else prepare(deployment, spec))
    except Exception as exc:
        # Paid operations may hold the stdin key. Zero-API failures can retain
        # their bounded diagnostic so an operator can identify the stop reason.
        result = {"status": "failed", "error_type": type(exc).__name__,
                  "next_action": "stop_and_inspect_no_retry"}
        if not paid:
            result["error"] = str(exc)[:1000]
        print(json.dumps(result))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "failed" or result.get("returncode", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
