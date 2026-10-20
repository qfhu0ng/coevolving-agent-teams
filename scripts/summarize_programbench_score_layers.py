#!/usr/bin/env python3
"""Read filtered evaluator metadata; print names only or explicit score layers.

No evaluator invocation, hidden tests, raw eval JSON, provider or file writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.analysis.score_layers import load_rules, load_score_layers, names_only


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--names-only", action="store_true",
                       help="Task ID, official status, record counts and sorted test IDs; no per-test outcomes")
    modes.add_argument("--layer-rules", type=Path, help="Exact test assignments; omitted rules stay unknown")
    args = parser.parse_args(argv)
    try:
        report = (names_only(args.run_dir) if args.names_only else load_score_layers(
            args.run_dir, load_rules(args.layer_rules) if args.layer_rules else None))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
