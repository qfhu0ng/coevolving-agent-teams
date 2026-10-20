"""Generic DS S0 setup: synthetic credentials and local fixtures, no remote execution."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from agent_company.programbench import comparison

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_programbench_s0_deepseek.py"
TASK_ID = "example__synthetic.abcdef0"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load():
    return load_file("ds_s0_launcher_test", LAUNCHER)


def test_templates_do_not_choose_a_task_and_only_differ_in_mode():
    module = load()
    templates = {mode: json.loads((ROOT / module.TEMPLATES[mode]).read_text()) for mode in module.MODES}
    for mode, data in templates.items():
        assert data["tasks"] == []
        assert data["provider_config"] == "configs/stage1/deepseek_flash.json"
        assert data["inference"] == module.inference(mode)
        assert data["evolution_enabled"] is False and data["parallelism"] == 1
        bound = module.build_profile(ROOT, mode, TASK_ID)
        assert bound["tasks"] == [{"task_id": TASK_ID,
                                   "path": f"external/programbench-study/src/programbench/data/tasks/{TASK_ID}",
                                   "instructions": "configs/stage1/programbench_instruction.md"}]
        assert f"Selected public task: {TASK_ID}" in bound["purpose"]
        assert "No default task is selected" not in bound["purpose"]
        assert "No task has been selected" not in bound["limitations"]
        assert "does not authorize deployment, real model calls or execution" in bound["purpose"]
        assert data["tasks"] == []  # Binding never rewrites the template on disk.
        with pytest.raises(ValueError, match="tasks and arms must be nonempty"):
            comparison.prepare_comparison(ROOT / module.TEMPLATES[mode], ROOT)
    first, second = (deepcopy(templates[mode]) for mode in module.MODES)
    for data in (first, second):
        data.pop("purpose")
        data["inference"].pop("role_policy_mode")
        data["arms"][0].pop("name")
    assert first == second


@pytest.mark.skipif(not (ROOT / "external/programbench-study/src/programbench/data/tasks/jgm__pandoc.5caad90").is_dir(),
                    reason="needs the local ProgramBench task data under external/")
@pytest.mark.parametrize("mode", ["propose", "global"])
def test_pandoc_review_profile_matches_renderer_and_public_plan(mode):
    module = load()
    task_id = "jgm__pandoc.5caad90"
    path = ROOT / module.PROFILES[mode]
    profile = json.loads(path.read_text())
    assert profile == module.build_profile(ROOT, mode, task_id)
    plan = comparison.prepare_comparison(path, ROOT).plan
    assert [task["task_id"] for task in plan["tasks"]] == [task_id]
    assert [(arm["arm_id"], arm["execution_mode"], arm["policy_version"]) for arm in plan["arms"]] == [
        (f"team_p0_{mode}", "team", 0)]
    assert plan["inference"] == module.inference(mode)
    assert "fresh P0 A/C" in profile["limitations"]
    assert "Saved P1" not in profile["limitations"]
    assert all("fresh_p0_20261001" in profile["arms"][0][key] for key in
               ("assignment_policy_path", "communication_policy_path"))
    assert plan["new_rollouts"] == 1 and plan["reused_rollouts"] == 0
    assert plan["fresh_start"] is True and plan["runs_per_task_arm"] == 1
    assert plan["evolution_enabled"] is False and plan["automatic_restarts"] is False


def test_pandoc_snapshot_profiles_match_reviewed_files():
    module = load()
    data, _ = module.snapshot(ROOT, module.Selection("jgm__pandoc.5caad90", "review-only"))
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for path in module.PROFILES.values():
            assert archive.extractfile(str(path)).read() == (ROOT / path).read_bytes()


def test_ds_provider_selects_streaming_thinking_and_shared_deadline(monkeypatch):
    from agent_company.runtime.provider import load_provider
    module = load()
    data = json.loads((ROOT / module.PROVIDER_CONFIG).read_text())
    assert "api_key" not in data and data["api_key_env"] == "DEEPSEEK_API_KEY"
    assert data == module.PROVIDER
    assert data["model"] == "deepseek-flash" and data["stream_response"] is True
    assert [data[key] for key in ("request_timeout_seconds", "first_progress_timeout_seconds",
                                 "progress_idle_timeout_seconds", "max_attempts")] == [1800, 300, 300, 3]
    assert data["extra_body"] == {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    assert data["retry_backoff_seconds"] == 1 and data["max_output_tokens"] == 65536
    monkeypatch.setenv("DEEPSEEK_API_KEY", "SYNTHETIC_DS_KEY")
    monkeypatch.setenv("STAGE1_GLM_API_KEY", "DO_NOT_USE")
    provider = load_provider(ROOT / module.PROVIDER_CONFIG)
    assert provider.api_key == "SYNTHETIC_DS_KEY" and provider.model == "deepseek-flash"


@pytest.mark.parametrize("argv", [["--deploy"], ["--prepare", "global"],
                                  ["--smoke", "propose", "--scripted"], ["--launch", "global"]])
def test_cli_has_no_implicit_task_or_deployment(argv, monkeypatch):
    module = load()
    monkeypatch.setattr(module, "ssh", lambda *a, **k: pytest.fail("must not connect"))
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("task,deployment", [("../escape", "test"), (TASK_ID, "../escape"),
                                             ("", "test"), (TASK_ID, "")])
def test_selection_rejects_unsafe_or_missing_path_components(task, deployment):
    module = load()
    with pytest.raises(ValueError):
        module.Selection(task, deployment)


@pytest.fixture(params=["propose", "global"])
def deployed(request, tmp_path, monkeypatch):
    module = load()
    selection = module.Selection(TASK_ID, "unit-test-only")
    monkeypatch.setattr(module, "PROJECT", tmp_path / "server")
    root = module.branch(selection, request.param)
    code = root / "code"
    for path in [module.PROVIDER_CONFIG, *module.TEMPLATES.values(),
                 "configs/stage1/programbench_instruction.md",
                 *(str(Path(module.POLICY) / name) for name in module.POLICY_SHA256)]:
        target = code / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    for mode in module.MODES:
        profile = code / module.PROFILES[mode]
        profile.write_text(json.dumps(module.build_profile(code, mode, TASK_ID)))
    task = code / "external/programbench-study/src/programbench/data/tasks" / TASK_ID
    task.mkdir(parents=True)
    (task / "task.yaml").write_text("repository: example/synthetic\ncommit: " + "abcdef0" + "1" * 33 + "\n")
    evaluator = root / "evaluator_fix/programbench/eval/eval.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# synthetic evaluator\n")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unmocked launch"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unmocked process"))
    return module, selection, request.param, root, code


def test_prepare_reads_public_inputs_without_keys_or_writes(deployed, monkeypatch):
    module, selection, mode, root, code = deployed
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("STAGE1_GLM_API_KEY", raising=False)
    before = set(root.rglob("*"))
    prepared = module.prepare(selection, mode)
    assert prepared["status"] == "prepared_not_launched"
    assert set(root.rglob("*")) == before
    assert prepared["plan"]["tasks"][0]["task_id"] == TASK_ID
    assert prepared["plan"]["provider_config"] == str(code / module.PROVIDER_CONFIG)
    assert prepared["plan"]["inference"] == module.inference(mode)


def test_launch_is_once_only_and_uses_ds_stdin_key(deployed, monkeypatch):
    module, selection, mode, root, code = deployed
    seen = []

    def popen(command, **kwargs):
        seen.append(command)
        assert kwargs["env"]["DEEPSEEK_API_KEY"] == "SYNTHETIC_DS_KEY"
        assert "STAGE1_GLM_API_KEY" not in kwargs["env"]
        assert kwargs["cwd"] == str(code) and kwargs["start_new_session"] is True
        return SimpleNamespace(pid=1234)

    monkeypatch.setenv("STAGE1_GLM_API_KEY", "DO_NOT_FORWARD")
    monkeypatch.setattr(subprocess, "Popen", popen)
    record = module.remote_launch(selection, mode, {"api_key": "SYNTHETIC_DS_KEY"})
    assert json.loads((root / "launch.json").read_text())["arms"] == [f"team_p0_{mode}"]
    assert record["pid"] == 1234
    for path in root.rglob("*"):
        if path.is_file():
            assert "SYNTHETIC_DS_KEY" not in path.read_text()
    with pytest.raises(FileExistsError):
        module.remote_launch(selection, mode, {"api_key": "SYNTHETIC_DS_KEY"})
    assert len(seen) == 1


@pytest.mark.parametrize("artifact", ["run", "launch.json", "controller.log"])
def test_existing_run_is_never_overwritten(deployed, artifact):
    module, selection, mode, root, _ = deployed
    target = root / artifact
    target.mkdir() if artifact == "run" else target.write_text("keep")
    with pytest.raises(FileExistsError):
        module.prepare(selection, mode)
    assert target.is_dir() if artifact == "run" else target.read_text() == "keep"


def test_legacy_launch_checks_before_reading_stdin(deployed):
    module, selection, mode, root, _ = deployed
    (root / "run").mkdir()
    with pytest.raises(FileExistsError):
        module.remote_launch(selection, mode, lambda: pytest.fail("stdin read before public checks"))


@pytest.mark.parametrize("operation", ["--remote-launch", "--remote-smoke", "--remote-probe"])
def test_legacy_cli_defers_stdin_to_remote_checks(monkeypatch, operation):
    module = load()
    monkeypatch.setattr(module.json, "load", lambda _: pytest.fail("eager stdin read"))
    def receive(*args):
        assert callable(args[-1])
        return {"returncode": 0}
    monkeypatch.setattr(module, "remote_launch", receive)
    monkeypatch.setattr(module, "remote_preparation", receive)
    args = ["--task-id", TASK_ID, "--deployment-id", "unit-test", operation]
    if operation != "--remote-probe":
        args.append("propose")
    assert module.main(args) == 0


def test_zero_api_preparation_scrubs_keys_and_real_step_uses_ds_only(deployed, monkeypatch):
    module, selection, mode, root, code = deployed
    monkeypatch.setenv("DEEPSEEK_API_KEY", "DO_NOT_INHERIT")
    monkeypatch.setenv("STAGE1_GLM_API_KEY", "DO_NOT_INHERIT_EITHER")
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=0, stdout='{"passed": true}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    module.remote_preparation(selection, "smoke_scripted", mode, None)
    module.remote_preparation(selection, "smoke_real", mode, {"api_key": "SYNTHETIC_DS_KEY"})
    assert "DEEPSEEK_API_KEY" not in seen[0][1] and "STAGE1_GLM_API_KEY" not in seen[0][1]
    assert seen[1][1]["DEEPSEEK_API_KEY"] == "SYNTHETIC_DS_KEY" and "STAGE1_GLM_API_KEY" not in seen[1][1]
    assert "--scripted" in seen[0][0] and "--scripted" not in seen[1][0]
    assert all(command[command.index("--profile") + 1] == str(code / module.PROFILES[mode])
               for command, _ in seen)


def test_snapshot_contains_explicit_overlays_and_bound_profiles():
    module = load()
    selection = module.Selection(TASK_ID, "unit-test-only")
    data, manifest = module.snapshot(ROOT, selection)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert archive.getnames().count("scripts/launch_programbench_s0_deepseek.py") == 1
        files = {member.name: archive.extractfile(member).read() for member in archive.getmembers()
                 if member.isfile()}
    assert set(files) == set(manifest["files"])
    assert all(hashlib.sha256(data).hexdigest() == manifest["files"][name] for name, data in files.items())
    assert str(module.PROVIDER_CONFIG) in files
    for mode in module.MODES:
        assert json.loads(files[str(module.PROFILES[mode])])["tasks"][0]["task_id"] == TASK_ID
    assert "scripts/s0_native_smoke.py" in files and "scripts/s0_cache_probe.py" in files
    assert files["scripts/launch_programbench_s0_deepseek.py"] == LAUNCHER.read_bytes()
    assert not any(name.startswith(("local_docs/", "runs/")) or "/.env" in name for name in files)


def test_remote_call_only_reads_key_for_paid_steps(monkeypatch):
    module = load()
    selection = module.Selection(TASK_ID, "unit-test-only")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    seen = []

    def ssh(remote, *, stdin=None, timeout=120):
        seen.append((remote, stdin))
        return SimpleNamespace(returncode=0, stdout='{"status": "ok"}', stderr="")

    monkeypatch.setattr(module, "ssh", ssh)
    reply, ok = module.remote_call(selection, "--remote-prepare", "global")
    assert ok and json.loads(seen[-1][1]) == {}
    assert "--task-id" in seen[-1][0] and "--deployment-id" in seen[-1][0]
    assert str(module.branch(selection, "global") / "code/scripts/launch_programbench_s0_deepseek.py") in seen[-1][0]
    assert "local_docs/" not in seen[-1][0]
    with pytest.raises(ValueError):
        module.remote_call(selection, "--remote-smoke", "global", with_credential=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "SYNTHETIC_DS_KEY")
    module.remote_call(selection, "--remote-smoke", "global", with_credential=True)
    assert json.loads(seen[-1][1]) == {"api_key": "SYNTHETIC_DS_KEY"}
    assert "SYNTHETIC_DS_KEY" not in seen[-1][0]


def test_smoke_requires_explicit_scripted_or_real_choice(monkeypatch):
    module = load()
    monkeypatch.setattr(module, "remote_call", lambda *a, **k: pytest.fail("must not connect"))
    with pytest.raises(SystemExit) as error:
        module.main(["--task-id", TASK_ID, "--deployment-id", "unit-test-only", "--smoke", "global"])
    assert error.value.code == 2


@pytest.mark.parametrize("operation,flags,paid", [("--smoke", ["global", "--scripted"], False),
                                                   ("--smoke", ["global", "--real"], True),
                                                   ("--prepare", ["global"], False)])
def test_cli_preserves_step_isolation_and_failure_code(monkeypatch, capsys, operation, flags, paid):
    module = load()
    calls = []

    def remote_call(selection, op, mode=None, **kwargs):
        calls.append((op, mode, kwargs.get("with_credential", False)))
        return {"returncode": 1}, False

    monkeypatch.setattr(module, "remote_call", remote_call)
    assert module.main(["--task-id", TASK_ID, "--deployment-id", "unit-test-only", operation, *flags]) == 1
    assert len(calls) == 1 and calls[0][2] is paid
    assert json.loads(capsys.readouterr().out) == {"returncode": 1}


@pytest.mark.parametrize("bad_global", [False, True])
def test_deploy_checks_both_branch_manifests_without_remote_execution(tmp_path, monkeypatch, bad_global):
    module = load()
    selection = module.Selection(TASK_ID, "unit-test-only")
    manifest = {"head": "test-head", "files": {"agent_company/x.py": "f" * 64},
                "overlays": {}, "generated_profiles": []}
    monkeypatch.setattr(module, "snapshot", lambda *a: (b"synthetic-archive", manifest))
    seen = []

    def ssh(remote, **kwargs):
        seen.append(remote)
        hashes = {"propose": f"{'f' * 64}  ./agent_company/x.py\n",
                  "global": f"{('e' if bad_global else 'f') * 64}  ./agent_company/x.py\n"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(hashes), stderr="")

    monkeypatch.setattr(module, "ssh", ssh)
    if bad_global:
        with pytest.raises(RuntimeError, match="global"):
            module.deploy(tmp_path, selection)
        assert not (tmp_path / module.local_run(selection) / "DEPLOYMENT.json").exists()
    else:
        result = module.deploy(tmp_path, selection)
        assert result["verified_files_by_mode"] == {"propose": 1, "global": 1}
        assert result["formal_run_started"] is False and result["external_model_calls"] == 0
    assert len(seen) == 1
    assert 'test ! -e "$S"' in seen[0] and 'test ! -e "$B"' in seen[0]
    assert f"ln -s {module.EXTERNAL}" in seen[0]


def test_remote_tests_keep_pipefail_remove_credentials_and_prepare_both_modes(monkeypatch):
    module = load()
    selection = module.Selection(TASK_ID, "unit-test-only")
    seen = []
    prepared = []

    def ssh(remote, **kwargs):
        seen.append(remote)
        return SimpleNamespace(returncode=1, stdout="synthetic failure", stderr="")

    def remote_call(selection, operation, mode, **kwargs):
        prepared.append(mode)
        return {"status": "prepared_not_launched"}, True

    monkeypatch.setattr(module, "ssh", ssh)
    monkeypatch.setattr(module, "remote_call", remote_call)
    result = module.remote_tests(selection)
    assert result["ok"] is False and prepared == list(module.MODES)
    assert "pipefail" in seen[0] and "-u DEEPSEEK_API_KEY" in seen[0] and "-u STAGE1_GLM_API_KEY" in seen[0]
    assert "tests/test_role_policy_integration.py" in seen[0]
    assert "tests/test_programbench_s0_deepseek_launch.py" not in seen[0]


@pytest.mark.parametrize("mode,variant", [("propose", "proposal"), ("global", "global")])
def test_generic_smoke_reuses_safe_projection_and_update_checks(tmp_path, mode, variant):
    from test_role_policy_integration import run
    from agent_company.analysis.evidence_projection import project_evidence
    smoke = load_file("generic_s0_smoke_test", ROOT / "scripts/s0_native_smoke.py")
    scenario, result, lifecycle, events, output = run(tmp_path, variant, mode)
    trial = smoke.copy_trial(output, tmp_path / "trial")
    assert not any(path.is_symlink() for path in trial.rglob("*"))
    policies = project_evidence(trial)["role_policies"]
    assert (policies["status"], policies["ledger"]["replay"]) == ("on", "ok")
    assert smoke.update_reached_worker(mode, scenario) is True
    scenario.worker_after_update = scenario.worker_payloads[0]
    assert smoke.update_reached_worker(mode, scenario) is False


def test_generic_smoke_filters_out_compaction(tmp_path):
    smoke = load_file("generic_s0_filter_test", ROOT / "scripts/s0_native_smoke.py")
    relay = tmp_path / "relay"
    relay.mkdir()
    events = [{"event": "provider_attempt_started", "attempt": n, "purpose": purpose}
              for n, purpose in [(1, "coding"), (2, "chat-compression"), (3, "coding")]]
    (relay / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    for n, role, content in [(1, "master", "master-injection"), (2, "master", "no-injection"),
                             (3, "alpha", "worker-injection")]:
        (relay / f"attempt_{n:05d}_{role}_provider_request.json").write_text(json.dumps(
            {"messages": [{"role": "system", "content": content}]}))
    assert smoke.system_texts(relay, True) == ["master-injection"]
    assert smoke.system_texts(relay, False) == ["worker-injection"]


def test_scripted_smoke_main_never_loads_credentials_and_labels_protocol_only(deployed, monkeypatch, capsys):
    module, selection, mode, root, code = deployed
    smoke = load_file("generic_s0_main_test", ROOT / "scripts/s0_native_smoke.py")
    monkeypatch.setattr(smoke, "CODE", code)
    monkeypatch.setattr(smoke, "load_provider", lambda _: pytest.fail("scripted must not load credentials"))
    calls = []

    class FakeWorkspace:
        container_id = None

        def __init__(self, task, **kwargs):
            assert task.task_id == TASK_ID

        def start(self):
            calls.append("start")
            # Like the real workspace when setup fails after docker run: identity observed, then raise.
            self.image_identity = {"reference": "programbench/x:tag", "image_id": "sha256:" + "d" * 64,
                                   "error": None, "source": "synthetic"}
            raise OSError("simulated container start failure; no Docker call")

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(smoke, "ProgramBenchWorkspace", FakeWorkspace)
    output = root / "fake-smoke"
    assert smoke.main(["--profile", str(code / module.PROFILES[mode]), "--assets-root", str(root),
                       "--output", str(output), "--scripted"]) == 1
    report = json.loads((output / "summary.json").read_text())
    assert calls == ["start", "stop"]
    assert report["external_model_calls"] == 0 and report["passed"] is False
    assert report["cleanup_confirmed"] and report["exception_type"] == "OSError"
    assert report["image_identity"]["image_id"] == "sha256:" + "d" * 64
    assert report["diagnostic_overrides"]["task"] == "scripted role-policy protocol"
    assert report["diagnostic_overrides"]["provider_request_timeout_seconds"] == 40
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("outcome", ["success", "partial", "all_failed", "http_500", "error_object", "missing_cache"])
def test_generic_cache_probe_persists_each_call_and_reports_failures(deployed, monkeypatch, capsys, outcome):
    from agent_company.runtime.relay import ProviderConfig
    module, selection, mode, root, code = deployed
    probe = load_file("generic_s0_cache_test", ROOT / "scripts/s0_cache_probe.py")
    monkeypatch.setattr(probe, "CODE", code)
    monkeypatch.setattr(probe, "load_provider", lambda _: ProviderConfig(
        "deepseek-flash", "https://unused.invalid", api_key="SYNTHETIC_DS_KEY"))
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    output = root / "cache-test"
    payloads = []

    def transport(payload, *, observation):
        # The in-flight call is persisted before transport; completed rows remain.
        persisted = json.loads((output / "cache_probe.json").read_text())
        assert persisted["calls"] == len(payloads) and persisted["complete"] is False
        assert persisted["started_calls"] == len(payloads) + 1 and persisted["status"] == "running"
        assert persisted["current_call"]["phase"] == ("A_unchanged" if len(payloads) < 3 else "B_after_policy_update")
        assert persisted["current_call"]["index"] == len(payloads) % 3
        assert isinstance(observation, dict)
        payloads.append(deepcopy(payload))
        index = len(payloads)
        if outcome == "all_failed" or (outcome == "partial" and index == 3):
            raise TimeoutError("SYNTHETIC_DS_KEY must not be stored")
        if outcome in {"http_500", "error_object"}:
            return (500 if outcome == "http_500" else 200), {"error": {"message": "SYNTHETIC_DS_KEY"}}
        usage = {"prompt_tokens": 100, "completion_tokens": 1}
        if outcome != "missing_cache":
            usage["prompt_cache_hit_tokens"] = 0
        return 200, {"model": "synthetic-response", "usage": usage}

    def transport_factory(provider):
        assert provider.max_attempts == 1 and provider.request_timeout_seconds == 180
        return transport

    monkeypatch.setattr(probe, "OpenAIChatTransport", transport_factory)
    result = probe.main(["--profile", str(code / module.PROFILES[mode]), "--output", str(output)])
    report = json.loads((output / "cache_probe.json").read_text())
    expected = ((0, "ok", 6, 0) if outcome == "success" else
                (1, "partial", 2, 1) if outcome == "partial" else (1, "failed", 0, 1))
    assert (result, report["status"], report["succeeded_calls"], report["failed_calls"]) == expected
    assert report["complete"] is (outcome == "success")
    assert report["planned_calls"] == 6 and report["current_call"] is None
    assert report["calls"] == report["started_calls"] == len(payloads) == expected[2] + expected[3]
    assert report["cache_reported"] is (expected[2] > 0)
    assert all(row["cached_tokens"] == 0 for row in report["rows"] if row["ok"])
    assert all(payload == payloads[0] for payload in payloads[:3])
    if outcome == "success":
        assert all(payload == payloads[3] for payload in payloads[3:])
        assert payloads[0]["messages"][0] != payloads[3]["messages"][0]
    else:
        failed = report["rows"][-1]
        assert failed["ok"] is False and failed["cached_tokens"] is None
        if outcome == "missing_cache":
            assert failed["http_status"] == 200 and failed["cache_field"] is None
        else:
            assert failed["error_type"] == ("TimeoutError" if outcome in {"partial", "all_failed"} else outcome)
    assert "SYNTHETIC_DS_KEY" not in (output / "cache_probe.json").read_text()
    printed = capsys.readouterr().out
    assert "SYNTHETIC_DS_KEY" not in printed
    assert json.loads(printed) == report
