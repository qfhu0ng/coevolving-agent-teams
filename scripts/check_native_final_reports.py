#!/usr/bin/env python3
"""Read explicitly named native run directories; print diagnostics without modifying runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_company.analysis.native_final_report import inspect_final_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Directories containing launch.json, summary.json and relay/")
    args = parser.parse_args(argv)
    print(json.dumps({"runs": [inspect_final_report(path) for path in args.runs]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
