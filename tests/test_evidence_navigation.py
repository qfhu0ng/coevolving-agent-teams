import json

import pytest

from agent_company.evolution.evidence import TrainingEvidence
from agent_company.evolution.loop import _task_summary, TaskSpec, TaskResult, run_stage1


def test_twenty_thousand_files_have_direct_navigation_and_current_homepage(tmp_path):
    root = tmp_path / "train"
    task = root / "batch_001/task_001"
    relay = task / "team/relay"
    relay.mkdir(parents=True)
    for index in range(20_000):
        (relay / f"attempt_{index:05d}_worker_provider_request.json").touch()
    (relay / "attempt_00720_verifier_qwen_response.json").write_text("response")
    (relay / "events.jsonl").write_text("event")
    (task / "task_result.json").write_text("{}")
    old = root / "batch_001/task_001.interrupted/attempt_0001"
    old.mkdir(parents=True)
    (old / "task_result.json").write_text("old")
    evidence = TrainingEvidence(root, [old, task])
    home = evidence.overview()
    assert home["total_files"] == 20004
    assert home["tasks"][0]["directory"] == "batch_001/task_001"
    assert home["interrupted_attempts"][0]["directory"].endswith("attempt_0001")
    assert all(".interrupted" not in p for p in home["tasks"][0]["current_attempt_entrypoints"])
    located = evidence.list_files(path="batch_001/task_001/team/relay", name="00720",
                                 source_kind="artifact")
    assert located["total"] == 1
    assert located["files"][0]["path"].endswith("attempt_00720_verifier_qwen_response.json")


def test_prompt_copies_opt_in_and_identical_independent_events_are_not_merged(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    for index in range(100):
        (task / f"attempt_{index:05d}_master_provider_request.json").write_text("Disallowed tool")
    (task / "events.jsonl").write_text("Disallowed tool\nDisallowed tool\n")
    evidence = TrainingEvidence(tmp_path, [task])
    hits = evidence.search("Disallowed")
    assert [(hit["path"], hit["line"]) for hit in hits["matches"]] == [
        ("task/events.jsonl", 1), ("task/events.jsonl", 2)]
    copies = evidence.search("Disallowed", source_kind="prompt_copy", limit=50)
    assert len(copies["matches"]) == 50 and copies["next_offset"] == 50
    assert all(hit["source_kind"] == "prompt_copy" for hit in copies["matches"])
    assert len(evidence.search("Disallowed", source_kind="prompt_copy", offset=50, limit=50)["matches"]) == 50
    assert evidence.read_file(copies["matches"][0]["path"])["text"] == "Disallowed tool"


@pytest.mark.parametrize("operation", ["list", "search"])
def test_new_filters_cannot_escape_current_batch(tmp_path, operation):
    task, hidden = tmp_path / "train/current", tmp_path / "heldout"
    task.mkdir(parents=True)
    hidden.mkdir()
    (hidden / "answer").write_text("SECRET")
    (task / "escape").symlink_to(hidden / "answer")
    old = tmp_path / "train/previous"
    old.mkdir()
    evidence = TrainingEvidence(tmp_path / "train", [task])
    for path in ("../heldout", str(hidden), "previous", "current/escape"):
        with pytest.raises(PermissionError):
            (evidence.list_files(path=path) if operation == "list"
             else evidence.search("SECRET", path=path))
    assert evidence.list_files()["total"] == 0


def test_summary_keeps_terminal_and_score_but_separates_old_costs(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "execution.json").write_text("{}")
    (task / "team").mkdir()
    (task / "team/qwen_stdout.jsonl").write_text("{}")
    record = {"task_id": "example", "repo_id": "repo", "output_dir": str(task),
              "execution_mode": "team", "status": "failed", "score": 0.44,
              "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "cached_prompt_tokens": 0},
              "cost_usd": None, "known_cost_usd": 13, "current_attempt_cost_usd": 3,
              "interrupted_attempts": [{"cost_usd": None, "known_cost_usd": 10, "usage": {"total_tokens": 90}}],
              "metrics": {"team": {"known_estimated_cost": "3", "currency": "USD", "execution": {"returncode": 0, "elapsed_seconds": 12},
                                   "native_result": {"result": {"result": "Waiting on a worker", "subtype": "success"}}},
                          "verifier": {"status": "completed", "reward": 0.44, "reward_details": {"passed": 4}},
                          "resources": {"padding": "x" * 10000}}}
    summary = _task_summary(record, tmp_path)
    assert summary["runtime_outcome"]["last_text"] == "Waiting on a worker"
    assert summary["status"] == "failed"
    assert summary["score"] == summary["official_result"]["raw_score"] == 0.44
    assert summary["agent_elapsed_seconds"] == 12
    assert summary["current_attempt"]["known_cost_usd"] == 3
    assert summary["current_attempt"]["cost_usd"] == 3
    assert summary["interrupted_attempts"][0]["known_cost_usd"] == 10
    assert summary["usage"]["total_tokens"] == 10
    assert summary["trace_files"] == ["task/execution.json", "task/team/qwen_stdout.jsonl"]
    assert len(json.dumps(summary)) < 6000


def test_failed_proposals_do_not_claim_replay_validation(tmp_path):
    def fail(_):
        raise RuntimeError("No proposal")
    result = run_stage1(train_tasks=[TaskSpec("train", "repo")], test_tasks=[TaskSpec("test", "heldout")],
                       initial_communication="C0", assignment_policy="A0", output_dir=tmp_path,
                       run_task=lambda _: TaskResult("completed", score=1), propose_policy=fail,
                       review_policy=lambda _: pytest.fail("No candidate to review"))
    assert result["successful_proposals"] == result["executed_replay_tasks"] == result["accepted_policy_updates"] == 0
    assert "No paired replay was executed" in result["final_policy_note"]
    assert "fallback, not replay validation" in result["final_policy_note"]


@pytest.mark.parametrize("value_size", [10, 2000])
def test_inference_image_never_changes_the_optimizer_summary(tmp_path, value_size):
    task = tmp_path / "task"
    task.mkdir()
    names = [f"metric_{index:02d}" for index in range(20)]
    # Runner insertion order: inference_image arrives as the 12th metric key.
    metrics = {name: "x" * value_size for name in names[:11]}
    metrics["inference_image"] = {"reference": "programbench/x:tag", "image_id": "sha256:" + "a" * 64,
                                  "error": None, "source": "docker inspect"}
    metrics.update({name: "x" * value_size for name in names[11:]})
    record = {"task_id": "example", "repo_id": "repo", "output_dir": str(task), "execution_mode": "team",
              "status": "completed", "score": 0.5, "metrics": metrics}
    without = {**record, "metrics": {k: v for k, v in metrics.items() if k != "inference_image"}}
    summary = _task_summary(record, tmp_path)
    assert json.dumps(summary, sort_keys=True) == json.dumps(_task_summary(without, tmp_path), sort_keys=True)
    assert "inference_image" not in json.dumps(summary) and "sha256:" not in json.dumps(summary)
    oversize = summary["metrics"].get("detail") == "Read result_file for full data"
    assert oversize is (value_size > 1000)
    if oversize:
        assert summary["metrics"]["keys"] == names[:16]
