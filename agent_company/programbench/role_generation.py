"""Generation-only artifact boundary for the role-policy series.

Phase 1/4 installs the boundary and exercises it with scripted components or
saved structured results. Diagnosis, carry review and the supplementary model
optimizer are phase 2/3 components, not implemented by copying these artifacts.
No runner, comparison launcher, provider or credential loader is used here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from agent_company.policy.role_package import validate_role_package


@dataclass(frozen=True)
class GenerationResult:
    status: str
    candidate: dict | None
    review: dict
    report: dict


def generate_only(parent: dict, output: str | Path, generate: Callable[[dict], GenerationResult],
                  *, capture_kind: str = "scripted") -> dict:
    """Capture generation/review output without any route to dispatch a run.

    This does not approve the candidate, infer missing per-rule reviews or mark
    it ready to run. The later evolution controller owns those semantics; both
    raw candidate and review remain inspectable even when blocked.
    """
    parent = validate_role_package(parent)
    if capture_kind not in {"scripted", "artifact_replay"}:
        raise ValueError("The phase 1/4 entry supports scripted or artifact replay only")
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(folder)
    result = generate(copy.deepcopy(parent))
    if not isinstance(result, GenerationResult) or result.status not in {
        "proposed", "not_needed", "no_events", "blocked"
    }:
        raise ValueError("Invalid structured generation result")
    if not isinstance(result.review, dict) or not isinstance(result.report, dict):
        raise ValueError("Review and report must be structured records")
    candidate = validate_role_package(result.candidate) if result.candidate is not None else None
    if candidate is not None and candidate["parent"] != {
        "package_id": parent["package_id"], "revision": parent["revision"]
    }:
        raise ValueError("Candidate parent does not match the selected lineage parent")
    if result.status == "proposed" and candidate is None:
        raise ValueError("A proposed result needs its complete candidate package")
    record = {"capture_kind": capture_kind, "fresh_model_review": False,
              "state": result.status, "dispatch": "not_performed",
              "candidate_artifact": "candidate.json" if candidate is not None else None,
              "review_artifact": "review.json", "report_artifact": "report.json",
              "official_score": None}
    artifacts = {"parent.json": parent, "review.json": result.review,
                 "report.json": result.report, "generation.json": record}
    if candidate is not None:
        artifacts["candidate.json"] = candidate
    # Serialize before writing, without newline translation of embedded policy
    # strings. No source artifact or earlier output directory is overwritten.
    serialized = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                  for name, value in artifacts.items()}
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in serialized.items():
        with (folder / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    return record


def replay_generation(parent: dict, saved_result: dict, output: str | Path) -> dict:
    """Replay supplied structured output; never claim a new reviewer session."""
    if not isinstance(saved_result, dict) or set(saved_result) != {"status", "candidate", "review", "report"}:
        raise ValueError("Replay input needs status, candidate, review and report")
    result = GenerationResult(**copy.deepcopy(saved_result))
    return generate_only(parent, output, lambda _: result, capture_kind="artifact_replay")
