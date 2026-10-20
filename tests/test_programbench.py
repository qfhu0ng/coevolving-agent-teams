"""Synthetic ProgramBench adapter tests; no benchmark tests, Docker or model calls."""
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from agent_company.programbench import task as pb


@pytest.fixture
def task(tmp_path):
    directory = tmp_path / pb.SMOKE_TASK_ID
    directory.mkdir()
    (directory / "task.yaml").write_text(
        "repository: tomnomnom/gron\ncommit: 88a6234ea2d0c487090988182ad9a7cdf6def924\nlanguage: go\n")
    # An unreadable-for-the-parser decoy proves public loading never parses this.
    (directory / "tests.json").write_text("not json; this file must not be read")
    instructions = tmp_path / "official_agent_instructions.md"
    instructions.write_text("Implement the documented behavior using the reference black box.")
    return pb.ProgramBenchTask.load(directory, instruction_path=instructions)


def test_public_loader_preserves_explicit_instructions_and_resource_provenance(task):
    assert task.task_id == pb.SMOKE_TASK_ID
    assert task.image == "programbench/tomnomnom_1776_gron.88a6234:task_cleanroom_v6"
    assert (task.agent_timeout_sec, task.cpus, task.memory_mb) == (21600, 20, 61440)
    assert task.memory_mb * 1024 ** 2 == 60 * 1024 ** 3  # Docker 60g.
    assert task.metadata["resource_limits_source"] == "official_baseline"
    assert task.metadata["inference_resource_overrides"] == {}
    assert task.metadata["resource_limits_scope"] == "task-shared"
    assert task.metadata["verifier_aggregate_timeout_enforced"] is False
    assert task.instruction == task.instruction_path.read_text()
    assert "instruction" not in task.describe()
    json.dumps(task.describe())


@pytest.mark.parametrize("override", [{"agent_timeout_sec": 7200}, {"cpus": 2}, {"memory_mb": 8192},
                                      {"agent_timeout_sec": 7200, "cpus": 2, "memory_mb": 8192}])
def test_inference_overrides_are_retained_and_labeled_custom(task, override):
    custom = pb.ProgramBenchTask.load(task.path, instruction_path=task.instruction_path, **override)
    assert all(getattr(custom, name) == value for name, value in override.items())
    assert custom.metadata["resource_limits_source"] == "custom"
    assert custom.metadata["inference_resource_overrides"] == override
    assert custom.metadata["official_inference_resource_limits"] == {
        "agent_timeout_sec": 21600, "cpus": 20, "memory_mb": 61440}


def test_loader_lowercases_task_id_for_mixed_case_repository(tmp_path):
    """Official directories/images are lowercase (ffmpeg__ffmpeg.360a402) while task.yaml keeps FFmpeg/FFmpeg."""
    directory = tmp_path / "ffmpeg__ffmpeg.360a402"
    directory.mkdir()
    (directory / "task.yaml").write_text(
        "repository: FFmpeg/FFmpeg\ncommit: 360a4025fb2582d52d871ea2129d6b659598bb49\nlanguage: c\n")
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Reimplement the documented behavior.")
    loaded = pb.ProgramBenchTask.load(directory, instruction_path=instructions)
    assert loaded.task_id == "ffmpeg__ffmpeg.360a402"
    assert loaded.image == "programbench/ffmpeg_1776_ffmpeg.360a402:task_cleanroom_v6"
    assert loaded.repository_id == "FFmpeg/FFmpeg"
    # A separate parent avoids colliding with the lowercase directory on case-insensitive filesystems.
    wrong = tmp_path / "wrong_case" / "FFmpeg__FFmpeg.360a402"
    wrong.mkdir(parents=True)
    (wrong / "task.yaml").write_text((directory / "task.yaml").read_text())
    with pytest.raises(ValueError, match="disagrees"):
        pb.ProgramBenchTask.load(wrong, instruction_path=instructions)


def test_loader_rejects_hidden_instruction_source_and_invalid_resource(task):
    with pytest.raises(ValueError, match="public agent"):
        pb.ProgramBenchTask.load(task.path, instruction_path=task.path / "tests.json")
    with pytest.raises(ValueError, match="memory_mb"):
        pb.ProgramBenchTask.load(task.path, instruction_path=task.instruction_path, memory_mb=True)


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "solution"
    directory.mkdir()
    (directory / "compile.sh").write_text("#!/bin/sh\ncp implementation.sh executable\n")
    (directory / "compile.sh").chmod(0o755)
    (directory / "implementation.sh").write_text("#!/bin/sh\nprintf '%s\\n' synthetic\n")
    (directory / "implementation.sh").chmod(0o755)
    (directory / "new-file.txt").write_text("untracked source is included")
    return directory


def test_archive_full_source_preserves_scripts_and_excludes_reference_and_prebuilt(source, tmp_path):
    for name in (".git", "reference"):
        (source / name).mkdir()
        (source / name / "private").write_text("excluded")
    (source / "executable").write_bytes(b"prebuilt")
    (source / "reference-copy").write_bytes(b"protected exact path")
    target = tmp_path / "submission.tar.gz"
    report = pb.create_submission(source, target, excluded_paths=("reference-copy",))
    with tarfile.open(target) as archive:
        assert set(archive.getnames()) == {"compile.sh", "implementation.sh", "new-file.txt"}
        assert archive.getmember("compile.sh").mode & 0o111
        assert archive.getmember("implementation.sh").mode & 0o111
    assert report["files"] == 3
    assert set(report["excluded_paths"]) == {".git", "reference", "executable", "reference-copy"}
    with pytest.raises(FileExistsError):
        pb.create_submission(source, target)


def test_archive_rejects_symlink_before_writing(source, tmp_path):
    (source / "escape").symlink_to(tmp_path)
    target = tmp_path / "submission.tar.gz"
    with pytest.raises(ValueError, match="links"):
        pb.create_submission(source, target)
    assert not target.exists()


@pytest.mark.parametrize("name", ["../escape", "/outside", "nested/../../escape", "evil\\path"])
def test_archive_validation_rejects_traversal(tmp_path, name):
    archive_path = tmp_path / "hostile.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="inside"):
        pb.validate_submission(archive_path)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_validation_rejects_links_and_special_files(tmp_path, kind):
    archive_path = tmp_path / "hostile.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("compile.sh")
        member.type = kind
        member.linkname = "/workspace/reference/executable"
        archive.addfile(member)
    with pytest.raises(ValueError, match="links"):
        pb.validate_submission(archive_path)


def test_submission_requires_compile_and_outside_destination(source, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        pb.create_submission(source, source / "submission.tar.gz")
    (source / "compile.sh").unlink()
    with pytest.raises(ValueError, match="compile.sh"):
        pb.create_submission(source, tmp_path / "submission.tar.gz")


class FakeEvaluationResult:
    """Tiny official-API double; fixtures are invented and contain no benchmark data."""
    def __init__(self, data):
        self.test_results = [SimpleNamespace(**row) for row in data.get("test_results", [])]
        self.test_branches = data.get("test_branches", ["active"])
        self.test_branch_errors = data.get("test_branch_errors", {})
        self.error_code = data.get("error_code")
        self.warnings = data.get("warnings", [])

    @classmethod
    def model_validate_json(cls, text):
        return cls(json.loads(text))

    def for_branches(self, active):
        result = deepcopy(self)
        result.test_results = [row for row in result.test_results if row.branch in active]
        result.test_branch_errors = {k: v for k, v in result.test_branch_errors.items() if k in active}
        result.test_branches = active
        return result

    def without_ignored(self, ignored):
        result = deepcopy(self)
        result.test_results = [row for row in result.test_results if f"{row.branch}/{row.name}" not in ignored]
        return result


class FakeSummary:
    @classmethod
    def from_eval_result(cls, task_id, result):
        passed = sum(row.status == "passed" for row in result.test_results)
        count = len(result.test_results)
        return SimpleNamespace(score=passed / count if count else 0, n_resolved=passed, n_tests=count,
                               error_code=result.error_code, test_branch_errors=result.test_branch_errors,
                               n_system_errors=sum(row.status == "system_error" for row in result.test_results),
                               n_warnings=len(result.warnings))


@pytest.fixture
def official(monkeypatch):
    calls = []

    def batch(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["output"]) / kwargs["sources"][0].name / pb.SMOKE_TASK_ID
        path.mkdir(parents=True)
        (path / f"{pb.SMOKE_TASK_ID}.eval.json").write_text(json.dumps({
            "test_results": [{"name": "invented", "branch": "active", "status": "passed"}]}))

    api = (FakeEvaluationResult, FakeSummary, batch,
           lambda instance: ["active"], lambda instance: {"active/ignored"},
           lambda **kwargs: [{"instance_id": pb.SMOKE_TASK_ID}], 1.0, 0.95)
    monkeypatch.setattr(pb, "_official_api", lambda: api)
    return calls


@pytest.fixture(autouse=True)
def evaluator_images(monkeypatch):
    """Evaluator tests never inspect a real Docker daemon."""
    calls = []

    def image_id(docker, reference):
        calls.append((docker, reference))
        return {"image_id": EVAL_IMAGE_ID, "error": None}

    monkeypatch.setattr(pb, "_docker_image_id", image_id)
    return calls


EVAL_IMAGE_ID = "sha256:" + "b" * 64
REAL_DOCKER_IMAGE_ID = pb._docker_image_id


def evaluated(tmp_path, **fields):
    path = tmp_path / "invented.eval.json"
    data = {"test_results": [{"name": "one", "branch": "active", "status": "passed"}]}
    data.update(fields)
    path.write_text(json.dumps(data))
    return path


def test_official_info_filtering_removes_ignored_tests_branches_and_stale_errors(official, tmp_path):
    path = evaluated(tmp_path, test_branches=["active", "ignored_branch"], test_results=[
        {"name": "one", "branch": "active", "status": "passed"},
        {"name": "ignored", "branch": "active", "status": "failure"},
        {"name": "two", "branch": "ignored_branch", "status": "system_error"}],
        test_branch_errors={"ignored_branch": ["stale"]}, warnings=["branch ignored_branch stale"])
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, path)
    assert (result["status"], result["reward"], result["valid_tests"]) == ("completed", 1.0, 1)
    assert result["resolved"] is True and result["almost"] is True


def test_filtered_metadata_uses_official_filtering_and_never_exports_bodies(official, tmp_path):
    path = evaluated(tmp_path, test_branches=["active", "ignored_branch"], test_results=[
        {"name": "one", "branch": "active", "status": "passed", "extra": {"text": "HIDDEN_BODY"}},
        {"name": "ignored", "branch": "active", "status": "failure"},
        {"name": "two", "branch": "ignored_branch", "status": "failure"}])
    sidecar = tmp_path / "filtered.json"
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, path, filtered_results_path=sidecar)
    saved = json.loads(sidecar.read_text())
    assert saved["tests"] == [{"test_id": "active/one", "status": "passed"}]
    assert saved["reward"] == result["reward"] == 1
    assert saved["valid_tests"] == result["valid_tests"] == 1
    assert "HIDDEN_BODY" not in sidecar.read_text()


@pytest.mark.parametrize("code,expected_status,expected_reward", [
    ("compile_failed", "completed", 0), (None, "incomplete_evaluation", None)])
def test_filtered_metadata_preserves_official_not_run_distinction(
        official, tmp_path, code, expected_status, expected_reward):
    path = evaluated(tmp_path, error_code=code,
                     test_results=[{"name": "one", "branch": "active", "status": "not_run"}])
    sidecar = tmp_path / "filtered.json"
    pb.summarize_official_result(pb.SMOKE_TASK_ID, path, filtered_results_path=sidecar)
    saved = json.loads(sidecar.read_text())
    assert saved["official_status"] == expected_status and saved["reward"] == expected_reward
    assert saved["submission_error"] is (code == "compile_failed")


def test_filtered_metadata_write_failure_does_not_change_official_reward(official, tmp_path, monkeypatch):
    def denied(*args):
        raise PermissionError("PRIVATE_PATH")
    monkeypatch.setattr(pb, "write_json_atomic", denied)
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, evaluated(tmp_path),
                                          filtered_results_path=tmp_path / "filtered.json")
    assert result["reward"] == 1 and result["status"] == "completed"
    assert result["filtered_results_error"] == "PermissionError"
    assert "PRIVATE_PATH" not in json.dumps(result)


@pytest.mark.parametrize("status,expected", [("failure", "completed"),
                                            ("system_error", "verifier_error"),
                                            ("not_run", "incomplete_evaluation")])
def test_low_score_is_distinct_from_infrastructure_or_missing_results(official, tmp_path, status, expected):
    path = evaluated(tmp_path, test_results=[{"name": "one", "branch": "active", "status": status}])
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, path)
    assert result["status"] == expected
    assert result["official_test_fraction"] == 0.0
    assert result["reward"] == (0.0 if status == "failure" else None)


@pytest.mark.parametrize("code,expected_reward", [("compile_failed", 0.0),
                                                  ("copy_executable_failed", 0.0),
                                                  ("wipe_workspace_failed", None)])
def test_submission_build_failure_is_distinct_from_verifier_setup_failure(official, tmp_path, code, expected_reward):
    path = evaluated(tmp_path, error_code=code,
                     test_results=[{"name": "one", "branch": "active", "status": "not_run"}])
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, path)
    assert result["reward"] == expected_reward
    assert result["error_code"] == code


def test_no_tests_and_no_artifact_never_manufacture_zero(official, tmp_path):
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, evaluated(tmp_path, test_results=[]))
    assert result["status"] == "empty_evaluation" and result["reward"] is None
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, tmp_path / "missing.json")
    assert result["status"] == "missing_evaluation" and result["reward"] is None


def test_almost_uses_official_threshold_and_keeps_exact_fraction(official, tmp_path):
    rows = [{"name": str(i), "branch": "active", "status": "passed" if i < 19 else "failure"}
            for i in range(20)]
    result = pb.summarize_official_result(pb.SMOKE_TASK_ID, evaluated(tmp_path, test_results=rows))
    assert result["reward"] == 0.95 and result["almost"] is True and result["resolved"] is False


def test_evaluator_uses_fresh_official_batch_and_separate_run_and_result_dirs(
        task, source, official, evaluator_images, tmp_path, monkeypatch):
    monkeypatch.delenv("PROGRAMBENCH_DOCKER_EXECUTABLE", raising=False)
    monkeypatch.delenv("PROGRAMBENCH_DOCKER_ORG", raising=False)
    archive = tmp_path / "submission.tar.gz"
    pb.create_submission(source, archive)
    output = tmp_path / "verification"
    result = pb.evaluate_submission(task, archive, output)
    assert result["status"] == "completed" and result["reward"] == 1.0
    assert len(official) == 1
    call = official[0]
    assert call["workers"] == call["branch_workers"] == 1
    assert call["docker_cpus"] == 10 and call["force"] is False
    assert task.cpus == 20  # Evaluation never inherits the inference CPU limit.
    assert result["evaluator_resource_limits_source"] == "official_default"
    assert result["official_compile_timeout_seconds"] == 900
    assert result["official_test_branch_timeout_seconds"] == 3600
    assert call["image_tag"] == pb.CLEANROOM_TAG
    reference = f"programbench/{task.task_id.replace('__', '_1776_')}:{pb.CLEANROOM_TAG}"
    assert reference == task.image
    assert evaluator_images == [("docker", reference)] * 2
    assert result["evaluator_image"] == {
        "reference": reference, "before": {"image_id": EVAL_IMAGE_ID, "error": None},
        "after": {"image_id": EVAL_IMAGE_ID, "error": None},
        "source": "docker inspect --type image .Id before and after the official evaluator",
        "scope": "tagged compile image; per-branch committed images are not observed"}
    assert call["output"] != call["sources"][0]
    assert "branch_retries" not in call  # Preserve official default, not our own retry harness.
    assert (output / "result.json").is_file()
    assert json.loads((output / "filtered_test_results.json").read_text())["tests"] == [
        {"test_id": "active/invented", "status": "passed"}]
    with pytest.raises(FileExistsError):
        pb.evaluate_submission(task, archive, output)
    assert len(official) == 1


def test_evaluator_cpu_override_is_independent_and_labeled_custom(task, source, official, tmp_path):
    archive = tmp_path / "submission.tar.gz"
    pb.create_submission(source, archive)
    result = pb.evaluate_submission(task, archive, tmp_path / "verification", docker_cpus=2)
    assert official[0]["docker_cpus"] == 2
    assert result["docker_cpus"] == 2
    assert result["evaluator_resource_limits_source"] == "custom"
    assert result["official_evaluator_docker_cpus"] == 10
    assert task.cpus == 20


def test_evaluator_failure_hides_exception_bodies_and_keeps_unknown_score(task, source, official, tmp_path, monkeypatch):
    archive = tmp_path / "submission.tar.gz"
    pb.create_submission(source, archive)
    api = list(pb._official_api())

    def broken(**kwargs):
        raise RuntimeError("PRIVATE TEST BODY OR SECRET")

    api[2] = broken
    monkeypatch.setattr(pb, "_official_api", lambda: api)
    result = pb.evaluate_submission(replace(task, verifier_timeout_sec=1), archive, tmp_path / "verification")
    assert result["status"] == "verifier_error" and result["reward"] is None
    assert result["error_code"] == "RuntimeError"
    assert result["evaluator_image"]["after"] == {"image_id": EVAL_IMAGE_ID, "error": None}
    assert "PRIVATE" not in json.dumps(result)


def test_explicit_docker_uses_official_subprocess_argv_and_private_environment(
        task, source, official, evaluator_images, tmp_path, monkeypatch, capsys):
    archive = tmp_path / "submission.tar.gz"
    pb.create_submission(source, archive)
    output = tmp_path / "verification with spaces"
    wrapper = "/synthetic path/docker-wrapper"
    monkeypatch.setenv("PROGRAMBENCH_DOCKER_EXECUTABLE", "parent-default")
    monkeypatch.setenv("PRIVATE_TEST_ENV", "PRIVATE_VALUE")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv[:3] == [sys.executable, "-c", pb._OFFICIAL_EVAL_SUBPROCESS]
        assert argv[3:] == [str(output / "submission_run"), str(output / "official_eval"),
                           "3", pb.CLEANROOM_TAG, "^" + re.escape(task.task_id) + "$"]
        assert kwargs["env"]["PROGRAMBENCH_DOCKER_EXECUTABLE"] == wrapper
        assert kwargs["env"]["PRIVATE_TEST_ENV"] == "PRIVATE_VALUE"
        assert kwargs["check"] is True and kwargs["stderr"] == subprocess.STDOUT
        assert "timeout" not in kwargs and "shell" not in kwargs
        assert "PRIVATE_VALUE" not in repr(argv)
        kwargs["stdout"].write("synthetic official evaluator output\n")
        # Stand in for the official batch run in the child, without Docker.
        pb._official_api()[2](sources=[Path(argv[3])], output=Path(argv[4]),
                              workers=1, branch_workers=1, docker_cpus=int(argv[5]),
                              image_tag=argv[6], filter_spec=argv[7], force=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pb.subprocess, "run", run)
    result = pb.evaluate_submission(task, archive, output, docker_cpus=3, docker_executable=wrapper)
    assert len(calls) == len(official) == 1
    assert os.environ["PROGRAMBENCH_DOCKER_EXECUTABLE"] == "parent-default"
    assert result["status"] == "completed" and result["reward"] == 1.0
    assert result["docker_executable"] == wrapper
    assert result["evaluator_invocation"] == "official_api_subprocess"
    assert [docker for docker, _ in evaluator_images] == [wrapper, wrapper]
    assert (output / "official_eval.log").read_text() == "synthetic official evaluator output\n"
    assert "PRIVATE_" not in json.dumps(result)
    assert "PRIVATE_" not in (output / "result.json").read_text()
    assert capsys.readouterr() == ("", "")


def test_official_subprocess_failure_is_unknown_and_redacts_command_output(
        task, source, official, tmp_path, monkeypatch):
    archive = tmp_path / "submission.tar.gz"
    pb.create_submission(source, archive)

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, output="PRIVATE_HIDDEN_TEST")

    monkeypatch.setattr(pb.subprocess, "run", fail)
    result = pb.evaluate_submission(task, archive, tmp_path / "verification", docker_executable="wrapper")
    assert result["status"] == "verifier_error" and result["reward"] is None
    assert result["error_code"] == "CalledProcessError"
    assert "PRIVATE_" not in json.dumps(result)
    assert not official


def test_evaluator_image_is_inspected_before_and_after_the_official_batch(
        task, source, official, tmp_path, monkeypatch):
    archive = tmp_path / "submission.tar.gz"
    pb.create_submission(source, archive)
    monkeypatch.setenv("PROGRAMBENCH_DOCKER_EXECUTABLE", "env-docker")
    monkeypatch.setenv("PROGRAMBENCH_DOCKER_ORG", "mirror")
    seen = []
    answers = iter([{"image_id": None, "error": "CalledProcessError"},
                    {"image_id": EVAL_IMAGE_ID, "error": None}])

    def image_id(docker, reference):
        seen.append((docker, reference, len(official)))
        return next(answers)

    monkeypatch.setattr(pb, "_docker_image_id", image_id)
    result = pb.evaluate_submission(task, archive, tmp_path / "verification")
    reference = f"mirror/{task.task_id.replace('__', '_1776_')}:{pb.CLEANROOM_TAG}"
    # Batch count 0 then 1: one inspect before, one after the official evaluator.
    assert seen == [("env-docker", reference, 0), ("env-docker", reference, 1)]
    assert result["status"] == "completed" and result["reward"] == 1.0
    assert result["evaluator_image"]["reference"] == reference
    assert result["evaluator_image"]["before"] == {"image_id": None, "error": "CalledProcessError"}
    assert result["evaluator_image"]["after"] == {"image_id": EVAL_IMAGE_ID, "error": None}
    assert json.loads((tmp_path / "verification" / "result.json").read_text())["evaluator_image"] == \
        result["evaluator_image"]


def test_verifier_elapsed_excludes_image_provenance_inspects(task, source, official, tmp_path, monkeypatch):
    archive = tmp_path / "submission.tar.gz"
    pb.create_submission(source, archive)
    clock = [100.0]

    def image_id(docker, reference):
        clock[0] += 1000  # A slow inspect must not count as evaluator time.
        return {"image_id": EVAL_IMAGE_ID, "error": None}

    monkeypatch.setattr(pb, "_docker_image_id", image_id)
    monkeypatch.setattr(pb, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    result = pb.evaluate_submission(task, archive, tmp_path / "verification")
    assert clock[0] == 2100.0
    assert result["verifier_elapsed_seconds"] == 0


def test_docker_image_id_helper_argv_validation_and_redaction(monkeypatch):
    calls = []
    outputs = iter([subprocess.CompletedProcess([], 0, EVAL_IMAGE_ID + "\n", ""),
                    subprocess.CompletedProcess([], 0, "[]\n", ""),
                    subprocess.CalledProcessError(1, ["docker"], output="PRIVATE_OUT", stderr="PRIVATE_ERR"),
                    subprocess.TimeoutExpired(["docker"], 30, output="PRIVATE_OUT")])

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        value = next(outputs)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(pb.subprocess, "run", run)
    helper = REAL_DOCKER_IMAGE_ID  # The autouse fixture replaced the module attribute.
    assert helper("wrapper", "programbench/x:tag") == {"image_id": EVAL_IMAGE_ID, "error": None}
    assert calls[0][0] == ["wrapper", "inspect", "--type", "image", "--format", "{{.Id}}", "programbench/x:tag"]
    assert calls[0][1] == {"text": True, "capture_output": True, "check": True, "timeout": 30}
    assert helper("wrapper", "r") == {"image_id": None, "error": "unexpected_format"}
    failed = helper("wrapper", "r")
    assert failed == {"image_id": None, "error": "CalledProcessError"}
    assert helper("wrapper", "r") == {"image_id": None, "error": "TimeoutExpired"}
    assert "PRIVATE" not in json.dumps(failed)
    assert pb.is_image_id(EVAL_IMAGE_ID) and not pb.is_image_id(EVAL_IMAGE_ID + "0")
    assert not pb.is_image_id(None) and not pb.is_image_id("sha256:" + "g" * 64)


@pytest.mark.parametrize("executable", ["", "  ", "nul\0byte", True, Path("wrapper")])
def test_invalid_docker_executable_fails_before_evaluator_output(task, source, tmp_path, executable):
    output = tmp_path / "verification"
    with pytest.raises(ValueError, match="docker_executable"):
        pb.evaluate_submission(task, tmp_path / "absent.tar.gz", output, docker_executable=executable)
    assert not output.exists()
