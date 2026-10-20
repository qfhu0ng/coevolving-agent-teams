#!/usr/bin/env python3
"""Print independent Pandoc baseline profiles; optionally save them locally.

The default is read-only. This command has no launch, deployment or provider
operation. Each saved profile is a separate input to the existing comparison CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_company.programbench.baseline import baseline_profiles, write_baseline_profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--repeat-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--write-profiles", type=Path, help="Explicitly write profiles into a new local directory")
    args = parser.parse_args(argv)
    result = baseline_profiles(args.project_root, args.repeat_indices)
    if args.write_profiles is not None:
        result["written_profiles"] = [str(path.resolve()) for path in write_baseline_profiles(result, args.write_profiles)]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
