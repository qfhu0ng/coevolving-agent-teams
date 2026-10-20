"""Official identity, condition, history and confirmation binding without new hashes."""
import hashlib
import json

import pytest

from agent_company.programbench.total_evidence import compare_trials, confirm_trials, series_repeat_indices
from agent_company.deploy.code_version import code_version


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def trial(root, run, arm, passed, *, mode="propose", version=0, single=False, ids=None, repeat_index=None):
    path = root / run / "task" / arm
    settings_path = path.parent.parent / "run_settings.json"
    if repeat_index is None:
        repeat_index = (json.loads(settings_path.read_text())["repeat_index"] if settings_path.is_file()
                        else len(list(root.rglob("run_settings.json"))) + 1)
    round_metadata = {"repeat_index": repeat_index, "pair_id": f"task#r{repeat_index}"}
    write(path / "run_request.json", {"task": {"task_id": "task", "repo_id": "repo"},
          "execution_mode": "single" if single else "team", "policy_version": version,
          "assignment_policy": "" if single else f"A{version}", "communication_policy": "" if single else f"C{version}",
          **round_metadata})
    write(path / "task_result.json", {"status": "failed", "score": passed / 100,
          "metrics": {"verifier": {"status": "completed", "reward": passed / 100,
                                   "passed_tests": passed, "valid_tests": 100}}})
    write(path / "verifier/filtered_test_results.json", {"schema_version": 1, "task_id": "task",
          "official_status": "completed", "reward": passed / 100, "valid_tests": 100,
          "tests": [{"test_id": str(i)} for i in (ids or range(100))]})
    write(path.parent.parent / "run_settings.json", {"provider": {"model": "test"},
          "code_version": {"kind": "git", "commit": "a" * 40},
          "qwen_version": "0.23.1", **round_metadata,
          "inference": {"role_policy_mode": mode, "agent_timeout_sec": 21600}, "evaluation": {},
          "parallelism": 1, "waves": [[{"task_id": "task", "arm_id": arm}]]})
    write(path / ("single" if single else "team") / "launch.json", {"native_compatibility_repairs": ["synthetic"]})
    return path


def concurrent(a, b):
    root = a.parent.parent
    settings = json.loads((root / "run_settings.json").read_text())
    settings.update(parallelism=2, waves=[[{"task_id": "task", "arm_id": p.name} for p in (a,b)]])
    write(root / "run_settings.json", settings)
    write(root / "progress.json", {"rows": [{"task_id": "task", "arm_id": p.name,
          "dispatched_at": "2026-10-14T00:00:00+00:00", "finished_at": "2026-10-14T06:00:00+00:00"} for p in (a,b)]})


def test_rejects_same_total_different_universe_or_task(tmp_path):
    a=trial(tmp_path,"a","single",10,single=True)
    b=trial(tmp_path,"b","team",20,ids=range(100,200))
    assert compare_trials(a,b)["decision"] == "inconclusive"
    b=trial(tmp_path,"b","team",20)
    assert compare_trials(a,b)["decision"] == "needs_confirmation"
    request=json.loads((b/"run_request.json").read_text())
    request["task"]["repo_id"]="other"
    write(b/"run_request.json", request)
    assert compare_trials(a,b)["decision"] == "inconclusive"


def test_confirmation_must_be_new_matching_concurrent_runs(tmp_path):
    a=trial(tmp_path,"first","single",10,single=True)
    b=trial(tmp_path,"first","team",20)
    concurrent(a,b)
    trigger=compare_trials(a,b)
    assert confirm_trials(trigger,a,b)["decision"] == "inconclusive"
    c=trial(tmp_path,"confirm","single",10,single=True)
    d=trial(tmp_path,"confirm","team",13)
    concurrent(c,d)
    assert confirm_trials(trigger,c,d)["decision"] == "confirmed"
    settings=json.loads((d.parent.parent/"run_settings.json").read_text())
    settings["inference"]["role_policy_mode"]="global"
    write(d.parent.parent/"run_settings.json",settings)
    assert confirm_trials(trigger,c,d)["decision"] == "inconclusive"


def test_history_deduplicates_and_excludes_single_other_modes_and_policies(tmp_path):
    a=trial(tmp_path,"a","single",10,single=True)
    b=trial(tmp_path,"b","team",20)
    global_run=trial(tmp_path,"g","team",90,mode="global")
    candidate=trial(tmp_path,"p1","team",99,version=1)
    single2=trial(tmp_path,"s2","single",99,single=True)
    report=compare_trials(a,b,history_paths=[b,b,global_run,candidate,single2])
    assert report["threshold_passed_tests"] == 5 and report["team_history_counts"] == [0,1]
    r2=trial(tmp_path,"r2","team",10)
    r3=trial(tmp_path,"r3","team",30)
    report=compare_trials(a,b,history_paths=[r2,r3])
    assert report["threshold_passed_tests"] == 40 and report["decision"] == "not_distinguished"


def test_evolution_cannot_combine_incumbent_and_candidate_range(tmp_path):
    a=trial(tmp_path,"a","team0",10)
    b=trial(tmp_path,"b","team1",20,version=1)
    r2=trial(tmp_path,"r2","team1",40,version=1)
    report=compare_trials(a,b,history_paths=[r2])
    assert report["team_history_counts"] == [1,2] and report["threshold_passed_tests"] == 5


def test_duplicate_progress_rows_do_not_prove_concurrency(tmp_path):
    a=trial(tmp_path,"first","single",10,single=True)
    b=trial(tmp_path,"first","team",20)
    concurrent(a,b)
    trigger=compare_trials(a,b)
    c=trial(tmp_path,"confirm","single",10,single=True)
    d=trial(tmp_path,"confirm","team",13)
    concurrent(c,d)
    path=c.parent.parent/"progress.json"
    rows=json.loads(path.read_text())["rows"]
    write(path,{"rows":[rows[0],rows[0]]})
    assert confirm_trials(trigger,c,d)["decision"] == "inconclusive"


def test_changed_runtime_repairs_cannot_trigger_an_evolution_comparison(tmp_path):
    a=trial(tmp_path,"a","team0",10)
    b=trial(tmp_path,"b","team1",20,version=1)
    write(b/"team/launch.json",{"native_compatibility_repairs":["changed"]})
    assert compare_trials(a,b)["decision"] == "inconclusive"


def test_recorded_result_identity_must_match_request(tmp_path):
    a=trial(tmp_path,"a","single",10,single=True)
    b=trial(tmp_path,"b","team",20)
    path=b/"task_result.json"
    data=json.loads(path.read_text())
    data["policy_version"]=99
    write(path,data)
    assert compare_trials(a,b)["decision"] == "inconclusive"


@pytest.mark.parametrize("key,original,conflict", [("repeat_index", 1, 2), ("pair_id", "task#r1", "task#r2")])
@pytest.mark.parametrize("location", ["result", "settings"])
def test_round_identity_conflicts_are_unknown(tmp_path, key, original, conflict, location):
    a = trial(tmp_path, "a", "single", 10, single=True)
    b = trial(tmp_path, "b", "team", 20)
    request_path = b / "run_request.json"
    request = json.loads(request_path.read_text())
    request[key] = original
    write(request_path, request)
    path = b / "task_result.json" if location == "result" else b.parent.parent / "run_settings.json"
    record = json.loads(path.read_text())
    record[key] = conflict
    write(path, record)
    assert compare_trials(a, b)["decision"] == "inconclusive"


def test_confirmation_cannot_reuse_trigger_round_number(tmp_path):
    a = trial(tmp_path, "first", "single", 10, single=True, repeat_index=1)
    b = trial(tmp_path, "first", "team", 20, repeat_index=1)
    c = trial(tmp_path, "confirm", "single", 10, single=True, repeat_index=1)
    d = trial(tmp_path, "confirm", "team", 13, repeat_index=1)
    concurrent(a, b)
    concurrent(c, d)
    trigger = compare_trials(a, b)
    assert confirm_trials(trigger, c, d)["decision"] == "inconclusive"
    c = trial(tmp_path, "confirm", "single", 10, single=True, repeat_index=2)
    d = trial(tmp_path, "confirm", "team", 13, repeat_index=2)
    concurrent(c, d)
    assert confirm_trials(trigger, c, d)["decision"] == "confirmed"


def relocate_settings(path, deployment, *, commit="a" * 40):
    file = path.parent.parent / "run_settings.json"
    data = json.loads(file.read_text())
    data.update(project_root=f"/deploy/{deployment}/code", qwen_package=f"/deploy/{deployment}/qwen",
                node_directory=f"/deploy/{deployment}/node", code_version={"kind": "git", "commit": commit})
    write(file, data)


def test_cross_deployment_initial_confirmation_and_followup_use_code_not_paths(tmp_path):
    a = trial(tmp_path, "a", "single", 10, single=True)
    b = trial(tmp_path, "b", "team", 20)
    c = trial(tmp_path, "confirmation", "single", 10, single=True)
    d = trial(tmp_path, "confirmation", "team", 13)
    for index, p in enumerate((a, b, c, d)):
        relocate_settings(p, str(index))
    concurrent(c, d)
    trigger = compare_trials(a, b)
    assert trigger["decision"] == "needs_confirmation"
    assert trigger["incumbent_execution_mode"] == "single" and trigger["candidate_execution_mode"] == "team"
    assert confirm_trials(trigger, c, d)["decision"] == "confirmed"
    candidate = trial(tmp_path, "followup", "team1", 40, version=1)
    relocate_settings(candidate, "followup")
    assert compare_trials(b, candidate)["decision"] == "needs_confirmation"
    relocate_settings(candidate, "same-path-different-code", commit="b" * 40)
    assert compare_trials(b, candidate)["decision"] == "inconclusive"


def test_unknown_code_versions_do_not_compare_equal(tmp_path):
    a = trial(tmp_path, "a", "single", 10, single=True)
    b = trial(tmp_path, "b", "team", 20)
    for p in (a, b):
        file = p.parent.parent / "run_settings.json"
        data = json.loads(file.read_text())
        data.pop("code_version")
        write(file, data)
    report = compare_trials(a, b)
    assert report["decision"] == "inconclusive" and "code version" in report["reason"]


@pytest.mark.parametrize("value", [None, "", " ", "unknown", 231])
@pytest.mark.parametrize("both", [False, True])
def test_unknown_qwen_versions_do_not_compare_equal(tmp_path, value, both):
    a = trial(tmp_path, "a", "single", 10, single=True)
    b = trial(tmp_path, "b", "team", 20)
    for p in (a, b) if both else (b,):
        file = p.parent.parent / "run_settings.json"
        data = json.loads(file.read_text())
        data["qwen_version"] = value
        write(file, data)
    report = compare_trials(a, b)
    assert report["decision"] == "inconclusive" and "Qwen version" in report["reason"]


def confirmation_series(root, *, confirm_index=5):
    a = trial(root, "wave4_propose_r4/run", "single", 10, single=True, repeat_index=4)
    b = trial(root, "wave1_off_r1/run", "team", 20, mode="off", repeat_index=1)
    trial(root, "wave2_global_r2/run", "team", 12, mode="global", repeat_index=2)
    trial(root, "wave3_master_r3/run", "team", 13, mode="master", repeat_index=3)
    c = trial(root, f"confirm_off_r{confirm_index}/run", "single", 10, single=True,
              mode="off", repeat_index=confirm_index)
    d = trial(root, f"confirm_off_r{confirm_index}/run", "team", 13, mode="off", repeat_index=confirm_index)
    concurrent(c, d)
    return compare_trials(a, b), c, d


@pytest.mark.parametrize("index", [2, 3])
def test_confirmation_cannot_reuse_any_other_wave_number_in_paid_series(tmp_path, index):
    trigger, c, d = confirmation_series(tmp_path / "paid_runs", confirm_index=index)
    assert trigger["decision"] == "needs_confirmation"
    result = confirm_trials(trigger, c, d)
    assert result["decision"] == "inconclusive"
    assert "already allocated in this series" in result["reason"]


def test_confirmation_scans_paid_series_and_excludes_its_own_shared_index(tmp_path):
    root = tmp_path / "paid_runs"
    trigger, c, d = confirmation_series(root)
    result = confirm_trials(trigger, c, d)
    assert result["decision"] == "confirmed"
    assert result["repeat_index_scope"] == "all_recorded_runs_in_series"
    assert result["series_root"] == str(root)
    assert set(series_repeat_indices(root)) == {1, 2, 3, 4, 5}


def test_confirmation_rejects_index_reserved_by_an_unscored_launch(tmp_path):
    root = tmp_path / "paid_runs"
    trigger, c, d = confirmation_series(root)
    write(root / "stopped_followup_r5/launch.json", {"repeat_index": 5, "status": "started"})
    result = confirm_trials(trigger, c, d)
    assert result["decision"] == "inconclusive" and "already allocated" in result["reason"]


@pytest.mark.parametrize("corruption", ["bad_json", "no_index", "conflicting_index", "empty_step"])
def test_unknown_or_inconsistent_paid_allocation_refuses_confirmation(tmp_path, corruption):
    root = tmp_path / "paid_runs"
    trigger, c, d = confirmation_series(root)
    if corruption == "empty_step":
        (root / "unknown_step").mkdir()
    else:
        record = root / "wave2_global_r2/launch.json"
        write(record, {"repeat_index": 3} if corruption == "conflicting_index" else {})
        if corruption == "bad_json":
            record.write_text("not json")
    assert confirm_trials(trigger, c, d)["decision"] == "inconclusive"


def test_explicit_offline_series_and_scope_without_series(tmp_path, capsys):
    from scripts.read_programbench_total_comparison import main

    root = tmp_path / "offline_series"
    trigger, c, d = confirmation_series(root)
    result = confirm_trials(trigger, c, d)
    assert result["decision"] == "confirmed"
    assert result["repeat_index_scope"] == "trigger_trials_only_no_series_context"
    saved = tmp_path / "trigger.json"
    write(saved, trigger)
    assert main(["--incumbent", str(c), "--candidate", str(d), "--trigger", str(saved),
                 "--series-root", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "confirmed" and result["repeat_index_scope"] == "all_recorded_runs_in_series"


def deployment(root, *, changed_file=None):
    """The real full-manifest identity path, including non-runtime files."""
    files = {"agent_company/module.py": b"source", "docs/plan.md": b"plan",
             "tests/test_module.py": b"test", "configs/stage1/profile.json": b"{}"}
    lines = []
    for relative, original in files.items():
        body = original + b" changed" if relative == changed_file else original
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        lines.append(f"{hashlib.sha256(body).hexdigest()}  ./{relative}")
    (root.parent / "code_sha256.txt").write_text("\n".join(lines) + "\n")
    value = code_version(root)
    assert value is not None and value["kind"] == "deployment_manifest"
    return value


def set_version(path, version):
    file = path.parent.parent / "run_settings.json"
    data = json.loads(file.read_text())
    data["code_version"] = version
    write(file, data)


@pytest.mark.parametrize("changed_file", ["agent_company/module.py", "docs/plan.md",
                                         "tests/test_module.py", "configs/stage1/profile.json"])
def test_full_deployment_manifest_compare_and_confirm(tmp_path, changed_file):
    first_version = deployment(tmp_path / "first_deploy/code")
    same_version = deployment(tmp_path / "copied_deploy/code")
    changed_version = deployment(tmp_path / "changed_deploy/code", changed_file=changed_file)
    assert first_version == same_version != changed_version
    a = trial(tmp_path, "first", "single", 10, single=True)
    b = trial(tmp_path, "first", "team", 20)
    c = trial(tmp_path, "confirmation", "single", 10, single=True)
    d = trial(tmp_path, "confirmation", "team", 13)
    set_version(a, first_version)
    set_version(c, same_version)
    concurrent(a, b)
    concurrent(c, d)
    trigger = compare_trials(a, b)
    assert trigger["decision"] == "needs_confirmation"
    assert confirm_trials(trigger, c, d)["decision"] == "confirmed"
    candidate = trial(tmp_path, "followup", "team1", 40, version=1)
    set_version(candidate, same_version)
    assert compare_trials(b, candidate)["decision"] == "needs_confirmation"
    set_version(candidate, changed_version)
    assert compare_trials(b, candidate)["decision"] == "inconclusive"
    set_version(c, changed_version)
    changed = confirm_trials(trigger, c, d)
    assert changed["decision"] == "inconclusive"
    assert "triggering arm conditions" in changed["reason"]


def test_read_comparison_cli_reports_initial_and_saved_threshold_confirmation(tmp_path, capsys):
    from scripts.read_programbench_total_comparison import main

    a = trial(tmp_path, "first", "single", 10, single=True)
    b = trial(tmp_path, "first", "team", 20)
    concurrent(a, b)
    assert main(["--incumbent", str(a), "--candidate", str(b)]) == 0
    trigger = json.loads(capsys.readouterr().out)
    assert trigger["decision"] == "needs_confirmation" and trigger["threshold_passed_tests"] == 5
    saved = tmp_path / "trigger.json"
    write(saved, trigger)
    c = trial(tmp_path, "confirmation", "single", 10, single=True)
    d = trial(tmp_path, "confirmation", "team", 13)
    concurrent(c, d)
    assert main(["--incumbent", str(c), "--candidate", str(d), "--trigger", str(saved)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "confirmed" and result["saved_trigger_threshold"] == 5


def test_read_comparison_cli_history_changes_initial_threshold(tmp_path, capsys):
    from scripts.read_programbench_total_comparison import main

    a = trial(tmp_path, "first", "single", 10, single=True)
    b = trial(tmp_path, "first", "team", 20)
    h1 = trial(tmp_path, "history1", "team", 5)
    h2 = trial(tmp_path, "history2", "team", 35)
    args = ["--incumbent", str(a), "--candidate", str(b), "--history-trial", str(h1),
            "--history-trial", str(h2)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "not_distinguished" and result["threshold_passed_tests"] == 60
    assert result["team_history_counts"] == [0, 3]


def test_read_comparison_cli_trigger_and_history_are_mutually_exclusive_before_read(tmp_path, capsys):
    from scripts.read_programbench_total_comparison import main

    with pytest.raises(SystemExit) as exc:
        main(["--incumbent", "missing-a", "--candidate", "missing-b", "--trigger", "missing-trigger",
              "--history-trial", "missing-history"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "saved trigger, not an updated history threshold" in captured.err


def test_read_comparison_cli_missing_history_is_unknown_not_ignored(tmp_path, capsys):
    from scripts.read_programbench_total_comparison import main

    a = trial(tmp_path, "first", "single", 10, single=True)
    b = trial(tmp_path, "first", "team", 20)
    assert main(["--incumbent", str(a), "--candidate", str(b),
                 "--history-trial", str(tmp_path / "missing")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "inconclusive" and result["adopted"] is False
