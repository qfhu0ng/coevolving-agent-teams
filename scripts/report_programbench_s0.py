"""Print offline S0 readings; never execute a run or regenerate model projections."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_company.analysis.s0_report import summarize_s0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="Saved team/relay/events.jsonl")
    parser.add_argument("--projection", type=Path, help="Optional existing allowlist evidence projection JSON")
    parser.add_argument("--launch", type=Path, help="Saved team/launch.json; only recorded schedule epochs and timeout are used")
    args = parser.parse_args()
    print(json.dumps(summarize_s0(args.events, projection_path=args.projection, launch_path=args.launch), ensure_ascii=False,
                     allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
