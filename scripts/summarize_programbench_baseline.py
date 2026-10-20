#!/usr/bin/env python3
"""Read local comparison result directories and print a descriptive JSON report.

Does not call providers, run evaluators, read trajectories or modify run files.
Input directories may come from independently launched round profiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.analysis.baseline_summary import summarize_comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--left-arm", required=True)
    parser.add_argument("--right-arm", required=True)
    parser.add_argument("--expected-repeat-indices", nargs="+", type=int,
                        help="Include wholly missing rounds as unknown; otherwise summarize only supplied plans")
    parser.add_argument("--layer-rules", type=Path, action="append",
                        help="Optional explicit rules, once per task; other tasks' layers remain unknown")
    args = parser.parse_args(argv)
    report = summarize_comparisons(args.run_directories, left_arm=args.left_arm, right_arm=args.right_arm,
                                   expected_repeat_indices=args.expected_repeat_indices,
                                   layer_rule_paths=args.layer_rules)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
