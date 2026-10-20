#!/usr/bin/env python3
"""Native Qwen communication evolution: download, inspect, run-task, or run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_company.roadmap.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
