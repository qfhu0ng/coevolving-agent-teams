"""Run the official verifier for a ProgramBench trial that skipped it because inference cleanup failed.

Usage: complete_programbench_verification.py <trial directory>  (run/<task id>/<arm>)

The task and evaluator settings come from the run's run_settings.json. No model calls. The
trial's containers must already be gone; see runner.complete_verification.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_company.programbench.runner import complete_verification  # noqa: E402
from agent_company.programbench.task import ProgramBenchTask  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trial", type=Path)
    trial = parser.parse_args(argv).trial.resolve(strict=True)
    settings = json.loads((trial.parent.parent / "run_settings.json").read_text(encoding="utf-8"))
    rows = [row for row in settings["tasks"] if Path(row["path"]).name == trial.parent.name]
    if len(rows) != 1:
        raise SystemExit(f"run_settings.json has no single task for {trial.parent.name}")
    inference = settings["inference"]
    task = ProgramBenchTask.load(rows[0]["path"], instruction_path=rows[0]["instructions"],
                                 agent_timeout_sec=inference["agent_timeout_sec"], cpus=inference["cpus"],
                                 memory_mb=inference["memory_mb"])
    result = complete_verification(trial, task, evaluator_cpus=settings["evaluation"]["docker_cpus"],
                                   docker=settings.get("docker") or "docker")
    verifier = result["metrics"]["verifier"]
    print(json.dumps({"status": result["status"], "score": result["score"],
                      "verifier_status": verifier.get("status"), "passed_tests": verifier.get("passed_tests"),
                      "valid_tests": verifier.get("valid_tests")}))


if __name__ == "__main__":
    main()
