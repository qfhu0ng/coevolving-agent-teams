import copy
import importlib.util
import json
from pathlib import Path

import pytest

from agent_company.policy.role_package import create_p0_package, load_role_package
from agent_company.programbench.role_generation import GenerationResult, generate_only, replay_generation


def packages():
    parent = create_p0_package("Delegate work.", "Report checked revisions.\r\n")
    candidate = copy.deepcopy(parent)
    candidate.update(package_id="v1", revision=1, parent={"package_id": "P0", "revision": 0})
    candidate["roles"]["implementer"]["policy"] = "Use an explicit handoff.\r\n"
    return parent, candidate


@pytest.fixture(autouse=True)
def forbid_dispatch(monkeypatch):
    from agent_company.programbench.runner import ProgramBenchRunner
    from agent_company.runtime import provider

    def forbidden(*args, **kwargs):
        raise AssertionError("Generation boundary attempted execution or credential loading")

    monkeypatch.setattr(ProgramBenchRunner, "__init__", forbidden)
    monkeypatch.setattr(ProgramBenchRunner, "__call__", forbidden)
    monkeypatch.setattr(provider, "load_provider", forbidden)


@pytest.mark.parametrize("status", ["proposed", "not_needed", "no_events", "blocked"])
def test_scripted_generation_keeps_evidence_and_never_dispatches(tmp_path, status):
    parent, candidate = packages()
    before = copy.deepcopy(parent)
    review = {"fixture": True, "status": "rejected" if status == "blocked" else "unreviewed"}

    def scripted(received):
        received["team"]["assignment"] = "Mutation must not affect parent."
        return GenerationResult(status, candidate if status in {"proposed", "blocked"} else None,
                                review, {"sources": ["scripted fixture"]})

    record = generate_only(parent, tmp_path / "output", scripted)
    assert parent == before
    assert record["dispatch"] == "not_performed" and record["official_score"] is None
    assert record["fresh_model_review"] is False and record["capture_kind"] == "scripted"
    assert json.loads((tmp_path / "output/review.json").read_text()) == review
    if status in {"proposed", "blocked"}:
        assert load_role_package(tmp_path / "output/candidate.json") == candidate
        assert load_role_package(tmp_path / "output/candidate.json")["roles"]["implementer"]["policy"].endswith("\r\n")
    else:
        assert not (tmp_path / "output/candidate.json").exists()


def test_v2_remains_a_draft_and_replay_does_not_claim_model_review(tmp_path):
    _, parent = packages()
    candidate = copy.deepcopy(parent)
    candidate.update(package_id="v2", revision=2, parent={"package_id": "v1", "revision": 1})
    record = replay_generation(parent, {"status": "proposed", "candidate": candidate,
        "review": {"original_session": "fixture"}, "report": {}}, tmp_path / "v2")
    assert record["capture_kind"] == "artifact_replay" and not record["fresh_model_review"]
    assert "ready" not in record and "approved" not in record
    assert record["dispatch"] == "not_performed"


def test_invalid_parent_and_existing_directory_refuse_without_overwrite(tmp_path):
    parent, candidate = packages()
    candidate["parent"]["revision"] = 9
    with pytest.raises(ValueError, match="Candidate parent"):
        generate_only(parent, tmp_path / "out", lambda _: GenerationResult("proposed", candidate, {}, {}))
    assert not (tmp_path / "out").exists()
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        generate_only(parent, tmp_path / "out", lambda _: pytest.fail("Must not invoke generator"))


def test_cli_requires_explicit_replay_and_preserves_result(tmp_path, capsys):
    path = Path(__file__).parents[1] / "scripts/generate_programbench_role_policy.py"
    spec = importlib.util.spec_from_file_location("generation_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, candidate = packages()
    (tmp_path / "parent.json").write_text(json.dumps(parent))
    saved = {"status": "blocked", "candidate": candidate, "review": {"status": "unknown"}, "report": {}}
    (tmp_path / "result.json").write_text(json.dumps(saved))
    module.main(["--parent", str(tmp_path / "parent.json"), "--replay-result", str(tmp_path / "result.json"),
                 "--output", str(tmp_path / "out")])
    assert json.loads(capsys.readouterr().out)["state"] == "blocked"
    with pytest.raises(SystemExit) as exc:
        module.main(["--run"])
    assert exc.value.code == 2
