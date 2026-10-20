"""Apply committed evaluator adaptations to our copied ProgramBench evaluator.

The task image and task test archives are untouched. The installed evaluator and
its plugin are measured as part of the formal deployment identity.
"""
from pathlib import Path
import shutil


def install(evaluator_root: Path) -> None:
    path = Path(evaluator_root) / "programbench/eval/eval.py"
    source = path.read_text(encoding="utf-8")
    replacements = {
        'run_args=[*DOCKER_RUN_ARGS, "--init"],':
            'run_args=[*DOCKER_RUN_ARGS, "--init", "--memory", "61440m", "--memory-swap", "61440m"],',
        '            run_cmd = "chmod +x ./eval/run.sh && ./eval/run.sh"':
            '            env.copy_in(Path(__file__).with_name("ac_pytest_attempt_timeout.py"),\n'
            '                        f"{WORKSPACE_DIR}/ac_pytest_attempt_timeout.py")\n'
            '            run_cmd = ("export PYTEST_PLUGINS=ac_pytest_attempt_timeout "\n'
            '                       "PYTHONPATH=/workspace:${PYTHONPATH:-} && "\n'
            '                       "chmod +x ./eval/run.sh && ./eval/run.sh")',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError(f"Unsupported evaluator adaptation point: {before.strip()}")
        source = source.replace(before, after)
    path.write_text(source, encoding="utf-8", newline="")
    shutil.copyfile(Path(__file__).with_name("evaluator_pytest_timeout.py"),
                    path.with_name("ac_pytest_attempt_timeout.py"))
