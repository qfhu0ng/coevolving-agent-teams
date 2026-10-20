"""The script reads the trial's task and evaluator settings from its run's run_settings.json."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/complete_programbench_verification.py"


def load():
    spec = importlib.util.spec_from_file_location("complete_programbench_verification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_and_evaluator_settings_come_from_the_run(tmp_path, monkeypatch, capsys):
    module = load()
    trial = tmp_path / "run/jgm__pandoc.5caad90/team_m1"
    trial.mkdir(parents=True)
    settings = {"tasks": [{"path": "/tasks/other", "instructions": "/x"},
                          {"path": "/tasks/jgm__pandoc.5caad90", "instructions": "/tasks/i.md"}],
                "inference": {"agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 81920},
                "evaluation": {"docker_cpus": 10}, "docker": "/deploy/tools/docker"}
    (tmp_path / "run/run_settings.json").write_text(json.dumps(settings))
    seen = {}

    def task_load(path, **kwargs):
        seen["task"] = (path, kwargs)
        return "TASK"

    def complete(directory, task, **kwargs):
        seen["complete"] = (directory, task, kwargs)
        return {"status": "completed", "score": 0.5, "metrics": {"verifier": {
            "status": "completed", "passed_tests": 1, "valid_tests": 2}}}

    monkeypatch.setattr(module.ProgramBenchTask, "load", task_load)
    monkeypatch.setattr(module, "complete_verification", complete)
    module.main([str(trial)])
    assert seen["task"] == ("/tasks/jgm__pandoc.5caad90", {"instruction_path": "/tasks/i.md",
                                                            "agent_timeout_sec": 21600, "cpus": 20,
                                                            "memory_mb": 81920})
    assert seen["complete"] == (trial.resolve(), "TASK", {"evaluator_cpus": 10, "docker": "/deploy/tools/docker"})
    assert json.loads(capsys.readouterr().out)["passed_tests"] == 1
    settings["tasks"] = settings["tasks"][:1]
    (tmp_path / "run/run_settings.json").write_text(json.dumps(settings))
    with pytest.raises(SystemExit):
        module.main([str(trial)])
