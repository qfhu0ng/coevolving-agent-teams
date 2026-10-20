"""Zero-API allocation tests with real planner, records and OS file locks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace

import pytest

from agent_company.deploy import launch
from agent_company.programbench.comparison import prepare_comparison
from test_programbench_formal_launch import (
    forbid_mutation_and_credentials, rig as _formal_rig, spec, write,
)


rig = _formal_rig


@pytest.fixture
def live_controllers(rig, monkeypatch):
    live = set()
    observed = []

    def start(command, **options):
        output = Path(command[command.index("--output") + 1])
        assert not output.exists(), "Launcher must leave run/ creation to the runner"
        profile = Path(command[command.index("--profile") + 1])
        index = int(command[command.index("--repeat-index") + 1])
        plan = prepare_comparison(profile, rig.deployment.code, repeat_index=index).plan
        pid = 78000 + len(observed)
        observed.append((command, options, pid))
        live.add(pid)
        write(output / "run_settings.json", plan)
        write(output / "progress.json", {
            "state": "running", "parallelism": plan["parallelism"],
            "running": [{"task_id": plan["tasks"][0]["task_id"], "arm_id": arm["arm_id"]}
                        for arm in plan["arms"]], "pending": [], "rows": [],
            "cleanup_attention": [], "requires_user_action": False,
        })
        return SimpleNamespace(pid=pid)

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(launch, "pid_alive", lambda pid: pid in live)
    monkeypatch.setattr(launch, "active_runner_processes", lambda _: sorted(live))
    return SimpleNamespace(live=live, observed=observed, start=start)


def test_two_single_arm_operations_launch_and_third_refuses_before_side_effects(rig, live_controllers, monkeypatch):
    first = launch.launch(rig.deployment, spec(rig, "off"), rig.payload)
    second = launch.launch(rig.deployment, spec(rig, "global"), rig.payload)
    assert first["reserved_arm_count"] == second["reserved_arm_count"] == 1
    assert len(live_controllers.live) == 2
    prepared = launch.prepare(rig.deployment, spec(rig, "master"))
    assert prepared["runtime"]["active_arm_count"] == 2
    assert prepared["runtime"]["new_arm_count"] == 1
    assert prepared["runtime"]["max_parallel_arms"] == 2
    assert not prepared["runtime"]["conditions"]["arm_slots_available"]
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Timing or capacity"):
        launch.launch(rig.deployment, spec(rig, "master"), payload)
    assert len(live_controllers.observed) == 2


@pytest.mark.parametrize("incoming", ["off", "global", "master"])
def test_two_arm_operation_blocks_every_new_operation(rig, live_controllers, monkeypatch, incoming):
    record = launch.launch(rig.deployment, spec(rig, "propose"), rig.payload)
    assert record["reserved_arm_count"] == 2
    prepared = launch.prepare(rig.deployment, spec(rig, incoming))
    assert prepared["runtime"]["active_arm_count"] == 2
    assert not prepared["runtime"]["conditions"]["arm_slots_available"]
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Timing or capacity"):
        launch.launch(rig.deployment, spec(rig, incoming), payload)


def test_one_active_arm_cannot_admit_a_new_two_arm_operation(rig, live_controllers, monkeypatch):
    launch.launch(rig.deployment, spec(rig, "off"), rig.payload)
    prepared = launch.prepare(rig.deployment, spec(rig, "propose"))
    assert prepared["runtime"]["active_arm_count"] == 1
    assert prepared["runtime"]["new_arm_count"] == 2
    assert not prepared["runtime"]["satisfied"]
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Timing or capacity"):
        launch.launch(rig.deployment, spec(rig, "propose"), payload)


@pytest.mark.parametrize("available,allowed", [(None, False), (61440 * 1024 - 1, False),
                                                (61440 * 1024, True)])
def test_memory_reserves_only_the_new_arm_and_also_checks_single_arm_steps(
        rig, live_controllers, monkeypatch, available, allowed):
    launch.launch(rig.deployment, spec(rig, "off"), rig.payload)
    host = launch.capacity(rig.deployment)
    monkeypatch.setattr(launch, "capacity", lambda _: {**host, "mem_available_kb": available})
    prepared = launch.prepare(rig.deployment, spec(rig, "global"))
    assert prepared["runtime"]["active_arm_count"] == 1
    assert prepared["runtime"]["conditions"]["concurrent_memory_available"] is allowed
    assert prepared["runtime"]["satisfied"] is allowed
    if allowed:
        # Existing arm's allocation is already reflected in MemAvailable.
        launch.launch(rig.deployment, spec(rig, "global"), rig.payload)
        assert len(live_controllers.live) == 2
    else:
        payload = forbid_mutation_and_credentials(rig, monkeypatch)
        with pytest.raises(ValueError, match="Timing or capacity"):
            launch.launch(rig.deployment, spec(rig, "global"), payload)


def test_existing_s0_runs_still_occupy_slots_when_first_round_skips_s0(rig, live_controllers, monkeypatch):
    for mode in ("propose", "global"):
        prepared = launch.prepare(rig.deployment, spec(rig, mode, step="s0_" + mode))
        assert prepared["runtime"]["satisfied"]
        assert "s0_completed_without_hard_failure" not in prepared["runtime"]["conditions"]
        launch.launch(rig.deployment, spec(rig, mode, step="s0_" + mode), rig.payload)
    assert len(live_controllers.live) == 2
    prepared = launch.prepare(rig.deployment, spec(rig, "off"))
    assert "s0_completed_without_hard_failure" not in prepared["runtime"]["conditions"]
    assert not prepared["runtime"]["conditions"]["arm_slots_available"]
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Timing or capacity"):
        launch.launch(rig.deployment, spec(rig, "off"), payload)


@pytest.mark.parametrize("state", ["running", "completed"])
def test_live_controller_keeps_full_reservation_across_progress_gaps(rig, live_controllers, monkeypatch, state):
    launch.launch(rig.deployment, spec(rig, "propose"), rig.payload)
    output = rig.deployment.final / "paid_runs/wave4_propose_r4/run"
    progress = launch.read_json(output / "progress.json")
    progress.update(state=state, running=[], pending=[])
    write(output / "progress.json", progress)
    prepared = launch.prepare(rig.deployment, spec(rig, "off"))
    assert prepared["runtime"]["active_arm_count"] == 2
    assert not prepared["runtime"]["satisfied"]
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError, match="Timing or capacity"):
        launch.launch(rig.deployment, spec(rig, "off"), payload)


def test_live_start_before_run_directory_is_visible_keeps_reservation(rig, live_controllers, monkeypatch):
    def no_output(command, **options):
        live_controllers.live.add(78999)
        return SimpleNamespace(pid=78999)
    monkeypatch.setattr(subprocess, "Popen", no_output)
    record = launch.launch(rig.deployment, spec(rig, "propose"), rig.payload)
    assert record["reserved_arm_count"] == 2
    assert not (rig.deployment.final / "paid_runs/wave4_propose_r4/run").exists()
    prepared = launch.prepare(rig.deployment, spec(rig, "off"))
    assert prepared["runtime"]["active_arm_count"] == 2
    assert not prepared["runtime"]["satisfied"]


@pytest.mark.parametrize("mutation", ["missing_reservation", "invalid_reservation", "unknown_state",
                                      "unknown_blocked_live", "unknown_blocked_dead", "untracked_runner", "dead_controller"])
def test_ambiguous_live_execution_refuses_before_side_effects(rig, live_controllers, monkeypatch, mutation):
    record = launch.launch(rig.deployment, spec(rig, "off"), rig.payload)
    root = rig.deployment.final / "paid_runs/wave1_off_r1"
    if mutation in {"missing_reservation", "invalid_reservation"}:
        saved = launch.read_json(root / "launch.json")
        if mutation == "missing_reservation":
            saved.pop("reserved_arm_count")
        else:
            saved["reserved_arm_count"] = 0
        write(root / "launch.json", saved)
    elif mutation in {"unknown_state", "unknown_blocked_live", "unknown_blocked_dead"}:
        progress = launch.read_json(root / "run/progress.json")
        state = "not_a_known_runner_state" if mutation == "unknown_state" else "blocked_unknown_state"
        write(root / "run/progress.json", {**progress, "state": state, "running": []})
        if mutation == "unknown_blocked_dead":
            live_controllers.live.remove(record["pid"])
    elif mutation == "untracked_runner":
        live_controllers.live.add(79999)
    else:
        live_controllers.live.remove(record["pid"])
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    with pytest.raises(ValueError):
        launch.launch(rig.deployment, spec(rig, "global"), payload)


@pytest.mark.parametrize("kind", ["smoke_scripted", "smoke_real", "cache_probe"])
def test_preparations_remain_exclusive_even_with_only_one_running_arm(rig, live_controllers, monkeypatch, kind):
    launch.launch(rig.deployment, spec(rig, "off"), rig.payload)
    payload = forbid_mutation_and_credentials(rig, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Preparation entered occupied deployment"))
    with pytest.raises(ValueError, match="still running|unconfirmed"):
        launch.preparation(rig.deployment, kind=kind, arm="single", sequence=1, payload=payload)


def test_real_file_lock_serializes_last_slot_check_and_process_allocation(rig, live_controllers, monkeypatch):
    launch.launch(rig.deployment, spec(rig, "off"), rig.payload)
    entered, release = Event(), Event()

    def held_start(command, **options):
        entered.set()
        assert release.wait(5), "Test did not release the synthetic process-start boundary"
        return live_controllers.start(command, **options)

    monkeypatch.setattr(subprocess, "Popen", held_start)
    with ThreadPoolExecutor(max_workers=1) as executor:
        winner = executor.submit(launch.launch, rig.deployment, spec(rig, "global"), rig.payload)
        try:
            assert entered.wait(5), "First contender did not reach process start"
            plan_count = len(rig.observed["planning"])
            credential_count = len(rig.observed["credentials"])
            # Distinct names and indexes still compete for the same final slot.
            with pytest.raises(BlockingIOError):
                launch.launch(rig.deployment, spec(rig, "master"), rig.payload)
            assert len(rig.observed["planning"]) == plan_count
            assert len(rig.observed["credentials"]) == credential_count
            assert not (rig.deployment.final / "paid_runs/wave3_master_r3").exists()
        finally:
            release.set()
        result = winner.result(timeout=5)
    assert result["name"] == "wave2_global_r2"
    assert len(live_controllers.live) == 2
    prepared = launch.prepare(rig.deployment, spec(rig, "master"))
    assert prepared["runtime"]["active_arm_count"] == 2
    assert not prepared["runtime"]["satisfied"]
    assert all(json.loads(path.read_text())["reserved_arm_count"] == 1
               for path in (rig.deployment.final / "paid_runs").glob("*/launch.json"))


def test_confirmed_terminal_controller_releases_its_reserved_slot(rig, live_controllers):
    old = launch.launch(rig.deployment, spec(rig, "off"), rig.payload)
    path = rig.deployment.final / "paid_runs/wave1_off_r1/run/progress.json"
    progress = launch.read_json(path)
    write(path, {**progress, "state": "completed", "running": []})
    live_controllers.live.remove(old["pid"])
    prepared = launch.prepare(rig.deployment, spec(rig, "propose"))
    assert prepared["runtime"]["active_arm_count"] == 0
    assert prepared["runtime"]["satisfied"]
    record = launch.launch(rig.deployment, spec(rig, "propose"), rig.payload)
    assert record["reserved_arm_count"] == 2


@pytest.mark.parametrize("kind", ["smoke_real", "cache_probe"])
def test_preparation_holds_real_lock_until_exit_and_excludes_every_other_start(rig, monkeypatch, kind):
    entered, release = Event(), Event()

    def foreground(command, **options):
        entered.set()
        assert release.wait(5), "Test did not release synthetic preparation"
        return SimpleNamespace(returncode=0, stdout="synthetic result", stderr="")

    monkeypatch.setattr(subprocess, "run", foreground)
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(launch.preparation, rig.deployment, kind=kind,
                                  arm="single", sequence=1, payload=rig.payload)
        try:
            assert entered.wait(5), "Preparation did not reach foreground execution"
            credentials = len(rig.observed["credentials"])
            with pytest.raises(BlockingIOError):
                launch.launch(rig.deployment, spec(rig), rig.payload)
            with pytest.raises(BlockingIOError):
                launch.preparation(rig.deployment, kind="smoke_real", arm="team_p0_propose",
                                   sequence=2, payload=rig.payload)
            assert len(rig.observed["credentials"]) == credentials
            assert not (rig.deployment.final / "paid_runs").exists()
        finally:
            release.set()
        assert running.result(timeout=5)["returncode"] == 0
