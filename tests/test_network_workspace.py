"""Network installation is fail-closed and outlives task process restarts."""
import subprocess

import pytest

from agent_company.runtime.workspace import DockerWorkspace
from test_roadmap import make_task


@pytest.fixture
def rig(tmp_path, monkeypatch):
    events = []

    class Boundary:
        image = "trusted-test-helper"

        def __init__(self, docker):
            assert docker == "docker"

        def prepare(self):
            events.append(("prepare",))

        def configure(self, container_id, relay_ip=None, relay_port=None):
            events.append(("filter", container_id, relay_ip, relay_port))

    workspace = DockerWorkspace(make_task(tmp_path), container_name="ac-stage1-owned")

    def docker(args, **kwargs):
        events.append(tuple(args))
        output = ""
        if args[0] == "create":
            output = "namespace-id\n"
        elif args[0] == "run":
            output = "task-id\n"
        elif args[:2] == ["exec", "namespace-id"]:
            output = "192.0.2.1 STREAM host.docker.internal\n"
        elif args[0] == "inspect":
            output = "{}"
        return subprocess.CompletedProcess(args, 0, output, "")

    def command(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "/app\n" if args == ["pwd"] else "", "")

    monkeypatch.setattr("agent_company.runtime.workspace.NetworkBoundary", Boundary)
    monkeypatch.setattr(workspace, "_docker", docker)
    monkeypatch.setattr(workspace, "run", command)
    return workspace, events


def test_restriction_precedes_task_start_and_survives_quiescence(rig):
    workspace, events = rig
    workspace.start()
    create = next(event for event in events if event[0] == "create")
    run = next(event for event in events if event[0] == "run")
    assert events.index(("filter", "namespace-id", None, None)) < events.index(run)
    assert create[create.index("--name") + 1] == "ac-stage1-owned-network"
    assert "ALL" in create and "--read-only" in create
    assert "NET_ADMIN" not in create  # Holder itself never administers its network.
    assert run[run.index("--network") + 1] == "container:namespace-id"
    assert "NET_ADMIN" in run and "NET_RAW" in run and "no-new-privileges" in run
    assert "--add-host" not in run
    assert workspace.network_policy_status["enforced"]
    assert workspace.configure_relay(34567) == "http://192.0.2.1:34567/v1"
    assert events[-1] == ("filter", "namespace-id", "192.0.2.1", 34567)
    workspace.finish_agent()
    assert events[-2:] == [("stop", "-t", "0", "task-id"), ("start", "task-id")]
    assert workspace._network_container_id == "namespace-id"
    with pytest.raises(RuntimeError, match="active agent"):
        workspace.configure_relay(34568)
    workspace.stop()
    assert events[-2:] == [("rm", "-f", "task-id"), ("rm", "-f", "namespace-id")]
    assert not workspace.network_policy_status["enforced"]


def test_failed_filter_never_starts_task_and_cleans_holder(rig, monkeypatch):
    workspace, events = rig

    def fail(*args, **kwargs):
        raise RuntimeError("filter unavailable")

    monkeypatch.setattr("agent_company.runtime.workspace.NetworkBoundary.configure", fail)
    with pytest.raises(RuntimeError, match="filter unavailable"):
        workspace.start()
    assert not any(event[0] == "run" for event in events)
    assert events[-1] == ("rm", "-f", "namespace-id")
    assert workspace.container_id is None and workspace._network_container_id is None


def test_failed_reinstall_never_launches_agent(rig, tmp_path, monkeypatch):
    workspace, _ = rig
    workspace.start()

    def fail(*args, **kwargs):
        raise RuntimeError("filter unavailable")

    def never(*args, **kwargs):
        pytest.fail("Agent launched without network restriction")

    monkeypatch.setattr(workspace, "_apply_network_boundary", fail)
    monkeypatch.setattr(workspace, "_long_command", never)
    with pytest.raises(RuntimeError, match="filter unavailable"):
        workspace.run_agent(["qwen"], env={}, timeout_seconds=10,
                            stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")
    assert workspace.agent_finished
    workspace.stop()


def test_a_removal_that_times_out_is_retried_and_a_vanished_container_counts_as_removed(rig, monkeypatch):
    workspace, events = rig
    workspace.start()
    old = workspace._docker
    calls = []

    def busy(args, **kwargs):
        calls.append((tuple(args), kwargs.get("timeout")))
        if args == ["rm", "-f", "namespace-id"] and sum(c[0] == tuple(args) for c in calls) == 1:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])  # the daemon was still busy
        if args == ["rm", "-f", "namespace-id"]:
            raise subprocess.CalledProcessError(1, args, "", "Error: No such container: namespace-id")
        if args[:2] == ["container", "inspect"]:
            raise subprocess.CalledProcessError(1, args, "", "Error: No such container: namespace-id")
        return old(args, **kwargs)

    monkeypatch.setattr(workspace, "_docker", busy)
    workspace.stop()
    assert [c for c in calls if c[0][0] == "rm"] == [
        (("rm", "-f", "task-id"), 120), (("rm", "-f", "namespace-id"), 120), (("rm", "-f", "namespace-id"), 300)]
    assert workspace.container_id is None and workspace._network_container_id is None


@pytest.mark.parametrize("case", ["two_timeouts", "still_exists", "first_attempt_error"])
def test_a_removal_that_keeps_failing_still_fails_and_keeps_ownership(rig, monkeypatch, case):
    workspace, events = rig
    workspace.start()
    old = workspace._docker
    attempts = []

    def failing(args, **kwargs):
        if args == ["rm", "-f", "namespace-id"]:
            attempts.append(kwargs["timeout"])
            if case == "two_timeouts" or len(attempts) == 1 and case == "still_exists":
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            raise subprocess.CalledProcessError(1, args, "", "Error response from daemon: busy")
        if args[:2] == ["container", "inspect"]:
            return subprocess.CompletedProcess(args, 0, "namespace-id\n", "")
        return old(args, **kwargs)

    monkeypatch.setattr(workspace, "_docker", failing)
    with pytest.raises(subprocess.TimeoutExpired if case == "two_timeouts" else subprocess.CalledProcessError):
        workspace.stop()
    assert workspace._network_container_id == "namespace-id"
    assert attempts == ([120] if case == "first_attempt_error" else [120, 300])


def test_failed_task_cleanup_preserves_namespace_ownership(rig, monkeypatch):
    workspace, events = rig
    workspace.start()
    old = workspace._docker

    def fail(args, **kwargs):
        if args == ["rm", "-f", "task-id"]:
            raise RuntimeError("daemon unavailable")
        return old(args, **kwargs)

    monkeypatch.setattr(workspace, "_docker", fail)
    with pytest.raises(RuntimeError, match="daemon unavailable"):
        workspace.stop()
    assert workspace._network_container_id == "namespace-id"
    assert ("rm", "-f", "namespace-id") not in events
