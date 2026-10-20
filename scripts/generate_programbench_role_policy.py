"""Replay role-policy generation artifacts without model calls or run dispatch.

The phase 2/3 live diagnostic/review/optimizer backend is deliberately not yet
connected. This explicit replay entry cannot launch an experiment.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.policy.role_package import load_role_package
from agent_company.programbench.role_generation import replay_generation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--replay-result", required=True, type=Path,
                        help="Saved structured candidate/review/report; no new model review")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = replay_generation(load_role_package(args.parent),
                               json.loads(args.replay_result.read_bytes().decode("utf-8")), args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
