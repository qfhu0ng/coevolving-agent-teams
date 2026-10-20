#!/usr/bin/env python3
"""Print three solo Team profiles and one Single/propose pair (five runs); never deploy or launch."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_company.programbench.baseline import first_round_profiles, write_baseline_profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--write-profiles", type=Path, help="Write into a new local directory only")
    args = parser.parse_args(argv)
    result = first_round_profiles(args.project_root)
    if args.write_profiles is not None:
        result["written_profiles"] = [str(path.resolve()) for path in write_baseline_profiles(result, args.write_profiles)]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
