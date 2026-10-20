"""Project existing ProgramBench Team evidence without exposing model reasoning.

The projection (agent_company.analysis.evidence_projection) uses only the
Python standard library. It never imports a runner,
launches a model, writes source data, reads request history, or reads test bodies.
--run-dir is one trial directory containing task_result.json and team/relay/.
Collaboration text is an outward tool argument, not an assistant reasoning block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.analysis.evidence_projection import project_evidence
from agent_company.analysis.score_layers import load_rules


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layer-rules", type=Path, help="Explicit rules; absent layers remain unknown")
    args = parser.parse_args(argv)
    rules = load_rules(args.layer_rules) if args.layer_rules else None
    print(json.dumps(project_evidence(args.run_dir, layer_rules=rules), ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
