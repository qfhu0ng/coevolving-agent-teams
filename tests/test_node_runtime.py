"""Runtime mounts retain the task image and only add explicit compatibility files."""

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from agent_company.runtime.node_runtime import node_runtime_mounts
from agent_company.evolution.loop import RunRequest, TaskSpec
from agent_company.runtime.relay import ProviderConfig
from agent_company.roadmap.runner import RoadmapRunner
from test_roadmap import make_task


@pytest.fixture
def node_directory(tmp_path):
    directory = tmp_path / "node-v22.23.2-linux-arm64"
    (directory / "bin").mkdir(parents=True)
    (directory / "bin/node").write_text("runtime fixture; not executed\n")
    return directory


@pytest.mark.parametrize("platform", ["linux/amd64", "linux/arm64"])
def test_standard_runtime_directory_is_mounted_readonly(node_directory, platform):
    assert node_runtime_mounts(node_directory, platform) == [
        (node_directory.resolve(), "/opt/ac-node", True),
    ]


def test_relative_runtime_directory_is_resolved(node_directory, monkeypatch):
    monkeypatch.chdir(node_directory.parent)
    assert node_runtime_mounts(Path(node_directory.name), "linux/amd64") == [
        (node_directory.resolve(), "/opt/ac-node", True),
    ]


def test_amd64_task_mounts_explicit_arm64_compatibility_readonly(node_directory):
    compatibility = node_directory / "compat/aarch64-linux-gnu"
    compatibility.mkdir(parents=True)
    loader = compatibility / "ld-linux-aarch64.so.1"
    loader.write_text("loader fixture; not executed\n")
    (compatibility / "libc.so.6").write_text("library fixture; not executed\n")

    assert node_runtime_mounts(node_directory, "linux/amd64") == [
        (node_directory.resolve(), "/opt/ac-node", True),
        (compatibility.resolve(), "/lib/aarch64-linux-gnu", True),
        (loader.resolve(), "/lib/ld-linux-aarch64.so.1", True),
    ]


def test_incomplete_compatibility_directory_does_not_silently_run(node_directory):
    (node_directory / "compat/aarch64-linux-gnu").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        node_runtime_mounts(node_directory, "linux/amd64")


@pytest.mark.parametrize("platform", ["linux/arm64", "linux/arm/v7"])
@pytest.mark.parametrize("loader_present", [False, True])
def test_other_task_platforms_do_not_mount_amd64_compatibility(
    node_directory, platform, loader_present,
):
    compatibility = node_directory / "compat/aarch64-linux-gnu"
    compatibility.mkdir(parents=True)
    if loader_present:
        (compatibility / "ld-linux-aarch64.so.1").write_text("not executed\n")

    assert node_runtime_mounts(node_directory, platform) == [
        (node_directory.resolve(), "/opt/ac-node", True),
    ]


@pytest.mark.parametrize("execution_mode", ["single", "team"])
def test_runner_passes_compatibility_mounts_to_both_execution_modes(
    tmp_path, monkeypatch, node_directory, execution_mode,
):
    import agent_company.roadmap.runner as module

    task = make_task(tmp_path)
    package = tmp_path / "qwen"
    package.mkdir()
    (package / "cli.js").write_text("// not executed\n")
    compatibility = node_directory / "compat/aarch64-linux-gnu"
    compatibility.mkdir(parents=True)
    (compatibility / "ld-linux-aarch64.so.1").write_text("not executed\n")
    observed = {}

    def mount_spy(directory, platform):
        observed["mount_helper"] = (directory, platform)
        return node_runtime_mounts(directory, platform)

    class Workspace:
        repo_path = "/app"

        def __init__(self, supplied_task, *, mounts, platform, container_name):
            observed.update(task=supplied_task, mounts=mounts, platform=platform,
                            container_name=container_name)
            self.resources = {"cpus": supplied_task.cpus, "memory_mb": supplied_task.memory_mb,
                              "shared_by_all_agents": True}

        def start(self):
            observed["started"] = True

        def run(self, command):
            return SimpleNamespace(stdout="initial-revision\n" if command == [
                "git", "rev-parse", "HEAD",
            ] else "")

        def stop(self):
            observed["stopped"] = True

    class NativeRunner:
        def __init__(self, qwen_cli, *, node, **kwargs):
            observed["native_paths"] = (qwen_cli, node)

        def run(self, workspace, instruction, assignment, communication, output, provider,
                timeout, *, execution_mode):
            observed["execution"] = (instruction, assignment, communication, timeout, execution_mode)
            return {"stop_reason": "native_cli_completed", "usage": {}}

    monkeypatch.setattr(module, "node_runtime_mounts", mount_spy)
    monkeypatch.setattr(module, "DockerWorkspace", Workspace)
    monkeypatch.setattr(module, "QwenTeamRunner", NativeRunner)
    monkeypatch.setattr(module, "score", lambda *args: {"status": "completed", "reward": 0})
    runner = RoadmapRunner(ProviderConfig("scripted-only", "http://localhost"), package,
                           node_directory, platform="linux/amd64")
    request = RunRequest(TaskSpec(task.task_id, task.repository_id, {"path": str(task.path)}),
                         "development", 0, "communication", "assignment", tmp_path / "result",
                         execution_mode=execution_mode)

    result = runner(request)

    assert result.status == "completed", result.error
    assert observed["mount_helper"] == (node_directory.resolve(), "linux/amd64")
    assert observed["mounts"] == [
        (package.resolve(), "/opt/ac-qwen", True),
        *node_runtime_mounts(node_directory, "linux/amd64"),
    ]
    assert observed["task"].describe() == task.describe()
    assert observed["platform"] == "linux/amd64"
    assert observed["native_paths"] == ("/opt/ac-qwen/cli.js", "/opt/ac-node/bin/node")
    assert observed["execution"] == (
        task.instruction, "assignment" if execution_mode == "team" else "",
        "communication" if execution_mode == "team" else "", task.agent_timeout_sec,
        execution_mode,
    )
    assert result.metrics["resources"] == {
        "cpus": task.cpus, "memory_mb": task.memory_mb, "shared_by_all_agents": True,
    }
    assert observed["started"] and observed["stopped"]


def test_bootstrap_reports_the_actual_node_process(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the startup adapter")
    bootstrap = Path(__file__).parents[1] / "agent_company/runtime/qwen_bootstrap.cjs"
    fields = ("JSON.stringify({version:process.version,arch:process.arch,platform:process.platform,"
              "libuv:process.versions.uv,execPath:process.execPath})")
    completed = subprocess.run(
        [node, "--require", str(bootstrap), "-p", fields],
        env={"AGENT_COMPANY_QWEN_HOME": str(tmp_path.resolve())},
        text=True, capture_output=True, timeout=20, check=True,
    )
    prefix = "Agent Company: Node runtime "
    records = [line.removeprefix(prefix) for line in completed.stderr.splitlines()
               if line.startswith(prefix)]
    assert len(records) == 1
    assert json.loads(records[0]) == json.loads(completed.stdout)
