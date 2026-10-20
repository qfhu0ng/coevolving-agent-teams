"""Zero-API synthetic pytest attempts; never executes task candidate code."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest

from agent_company.programbench import evaluator_compat
from agent_company.programbench import evaluator_pytest_timeout as plugin


ROOT = Path(__file__).resolve().parents[1]


def plugin_environment(tmp_path):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"DEEPSEEK_API_KEY", "STAGE1_GLM_API_KEY", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}}
    sources = os.environ.get("AC_EVALUATOR_PLUGIN_DIR")
    if sources:
        for module in ("pytest_timeout", "pytest_rerunfailures"):
            shutil.copyfile(Path(sources) / f"{module}.py", tmp_path / f"{module}.py")
    elif any(importlib.util.find_spec(name) is None for name in ("pytest_timeout", "pytest_rerunfailures")):
        pytest.skip("Requires real pytest-timeout/rerunfailures (or captured plugin sources)")
    paths = [str(Path(value).resolve()) for value in environment.get("PYTHONPATH", "").split(os.pathsep) if value]
    environment.update(PYTHONPATH=os.pathsep.join((str(tmp_path), str(ROOT), *paths)),
                       PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return environment


@pytest.mark.parametrize("workers", [0, 10])
@pytest.mark.parametrize("scenario", ["ordinary_then_hang", "all_hang", "setup", "teardown", "child"])
def test_actual_plugins_bound_each_attempt_and_preserve_later_test(tmp_path, scenario, workers):
    if workers and importlib.util.find_spec("xdist") is None:
        pytest.skip("Requires real pytest-xdist")
    environment = plugin_environment(tmp_path)
    source = '''import os, sys, time, subprocess
from pathlib import Path
import pytest
COUNT = Path(__file__).with_name("attempts.txt")
@pytest.fixture(autouse=True)
def attempt(request):
    if request.node.name != "test_bad":
        yield
        return
    count = int(COUNT.read_text()) + 1 if COUNT.exists() else 1
    COUNT.write_text(str(count))
    if SCENARIO == "setup":
        time.sleep(5)
    yield count
    if SCENARIO == "teardown":
        time.sleep(5)
def test_bad(attempt):
    if SCENARIO == "ordinary_then_hang" and attempt == 1:
        assert False, "first ordinary failure"
    if SCENARIO == "child":
        subprocess.run([sys.executable, "-c", "import os,time; from pathlib import Path; Path('child_' + str(os.getpid())).write_text(str(os.getpid())); time.sleep(5)"], check=True)
    elif SCENARIO != "setup":
        if SCENARIO == "teardown":
            assert False, "ordinary failure before hanging teardown"
        time.sleep(5)
def test_after():
    assert True
'''.replace("SCENARIO", repr(scenario))
    (tmp_path / "test_synthetic.py").write_text(source)
    command = [sys.executable, "-m", "pytest", "-p", "pytest_timeout", "-p", "pytest_rerunfailures",
               "-p", "agent_company.programbench.evaluator_pytest_timeout", "test_synthetic.py", "-q",
               "--timeout=0.03", "--timeout-method=thread", "--ac-attempt-timeout=0.15",
               "--reruns=2", "--reruns-delay=0", "--junitxml=result.xml"]
    if workers:
        command += ["-p", "xdist.plugin", "-n", str(workers)]
    started = time.monotonic()
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    (tmp_path / "command.json").write_text(json.dumps(command))
    (tmp_path / "pytest.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 1, result.stdout + result.stderr
    assert time.monotonic() - started < 12
    assert (tmp_path / "attempts.txt").read_text() == "3"
    cases = list(ET.parse(tmp_path / "result.xml").getroot().iter("testcase"))
    assert len(cases) == 2
    assert len({(case.get("classname"), case.get("name")) for case in cases}) == 2
    assert any(case.get("name") == "test_after" and not list(case) for case in cases)
    bad = [case for case in cases if case.get("name") == "test_bad"]
    assert bad and all(case.find("failure") is not None or case.find("error") is not None for case in bad)
    assert "INTERNALERROR" not in result.stdout and "worker crashed" not in result.stdout
    assert "ProgramBench attempt exceeded its timeout" in result.stdout
    assert "PermissionError" not in result.stdout
    for child in tmp_path.glob("child_*"):
        with pytest.raises(ProcessLookupError):
            os.kill(int(child.read_text()), 0)


def test_fixture_cleanup_after_timeout_does_not_get_another_test_budget(tmp_path):
    environment = plugin_environment(tmp_path)
    (tmp_path / "test_cleanup.py").write_text('''import time, pytest
@pytest.fixture
def hung(request):
    request.addfinalizer(lambda: time.sleep(10))
    request.addfinalizer(lambda: time.sleep(10))
    yield
    time.sleep(10)
def test_hung(hung):
    time.sleep(10)
def test_after():
    assert True
''')
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "pytest_timeout", "-p", "pytest_rerunfailures",
        "-p", "agent_company.programbench.evaluator_pytest_timeout", "test_cleanup.py", "-q", "--timeout=.03",
        "--ac-attempt-timeout=.15", "--reruns=0", "--junitxml=result.xml"], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stdout + result.stderr
    assert time.monotonic() - started < 5
    assert "ProgramBench attempt exceeded" in result.stdout
    cases = list(ET.parse(tmp_path / "result.xml").getroot().iter("testcase"))
    assert len(cases) == 2 and len({case.get("name") for case in cases}) == 2
    assert any(case.get("name") == "test_after" and not list(case) for case in cases)
    assert "ProgramBench teardown diagnostics" in (tmp_path / "result.xml").read_text()


def test_setup_and_call_share_one_attempt_budget(tmp_path):
    environment = plugin_environment(tmp_path)
    (tmp_path / "test_shared.py").write_text('''import time, pytest
@pytest.fixture
def slow_setup():
    time.sleep(.35)
def test_shared(slow_setup):
    time.sleep(.35)
''')
    command = [sys.executable, "-m", "pytest", "-p", "pytest_timeout", "-p", "pytest_rerunfailures",
               "-p", "agent_company.programbench.evaluator_pytest_timeout", "test_shared.py", "-q",
               "--timeout=30", "--ac-attempt-timeout=.5", "--reruns=0", "--junitxml=result.xml"]
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stdout + result.stderr
    cases = list(ET.parse(tmp_path / "result.xml").getroot().iter("testcase"))
    assert len(cases) == 1 and cases[0].find("failure") is not None
    assert "ProgramBench attempt exceeded its timeout" in result.stdout


def test_descendant_selection_excludes_sibling_workers(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    parent = os.getpid()
    mapping = {parent: 2, 10001: parent, 10002: 10001, 10003: 2, 10004: 10003}
    for pid, ppid in mapping.items():
        folder = proc / str(pid)
        folder.mkdir(parents=True)
        (folder / "stat").write_text(f"{pid} (synthetic child) S {ppid} 0 0")
    original = plugin.Path
    monkeypatch.setattr(plugin, "Path", lambda value: proc if value == "/proc" else original(value))
    assert plugin._descendants() == [10002, 10001]
    calls = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))
    plugin._kill_children()
    assert calls == [(10002, signal.SIGKILL), (10001, signal.SIGKILL)]


def test_installer_changes_only_our_evaluator_copy_and_sets_resources(tmp_path):
    root = tmp_path / "evaluator_fix"
    path = root / "programbench/eval/eval.py"
    path.parent.mkdir(parents=True)
    source = '''class Evaluator:
    def _new_env(self):
        return ContainerEnvironment(
            run_args=[*DOCKER_RUN_ARGS, "--init"],
        )
    def _run_test_branch(self, env):
        if True:
            run_cmd = "chmod +x ./eval/run.sh && ./eval/run.sh"
            return run_cmd
'''
    path.write_text(source)
    evaluator_compat.install(root)
    result = path.read_text()
    assert '"--memory", "61440m", "--memory-swap", "61440m"' in result
    assert 'PYTEST_PLUGINS=ac_pytest_attempt_timeout' in result
    assert 'env.copy_in(Path(__file__)' in result
    assert path.with_name("ac_pytest_attempt_timeout.py").read_bytes() == Path(plugin.__file__).read_bytes()
    captures = []
    namespace = {"ContainerEnvironment": lambda **kwargs: kwargs, "DOCKER_RUN_ARGS": [],
                 "__file__": str(path), "Path": Path, "WORKSPACE_DIR": "/workspace"}
    exec(compile(result, str(path), "exec"), namespace)
    evaluator = namespace["Evaluator"]()
    assert evaluator._new_env()["run_args"] == ["--init", "--memory", "61440m", "--memory-swap", "61440m"]
    class Environment:
        def copy_in(self, local, remote):
            captures.append((local, remote))
    command = evaluator._run_test_branch(Environment())
    assert "PYTHONPATH=/workspace:${PYTHONPATH:-}" in command
    assert captures == [(path.with_name("ac_pytest_attempt_timeout.py"), "/workspace/ac_pytest_attempt_timeout.py")]
    with pytest.raises(ValueError, match="Unsupported evaluator"):
        evaluator_compat.install(root)


@pytest.mark.parametrize("timeout", [0, -1, 61, float("inf"), float("nan")])
def test_invalid_attempt_timeout_is_rejected(timeout):
    class Config:
        def getoption(self, name):
            return timeout
    with pytest.raises(pytest.UsageError):
        plugin.pytest_configure(Config())
