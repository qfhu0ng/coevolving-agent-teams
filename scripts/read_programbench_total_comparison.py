#!/usr/bin/env python3
"""Read saved trials and print total-score trigger/confirmation evidence; never run agents."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_company.programbench.total_evidence import compare_trials, confirm_trials


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--history-trial", type=Path, action="append", default=[])
    parser.add_argument("--trigger", type=Path, help="Saved first-comparison report; its threshold is reused")
    parser.add_argument("--series-root", type=Path, help="Explicit offline series root; paid_runs is inferred automatically")
    args = parser.parse_args(argv)
    if args.series_root is not None and args.trigger is None:
        parser.error("--series-root is only used with --trigger")
    if args.trigger:
        if args.history_trial:
            parser.error("Confirmation uses the saved trigger, not an updated history threshold")
        result = confirm_trials(json.loads(args.trigger.read_text()), args.incumbent, args.candidate,
                                series_root=args.series_root)
    else:
        result = compare_trials(args.incumbent, args.candidate, history_paths=args.history_trial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
