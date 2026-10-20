"""Download the public RoadmapBench task interface, without oracle solutions."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable


DATASET = "UnipatAI/RoadmapBench"
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download_task(
    task_id: str,
    destination: Path,
    *,
    revision: str = "main",
    fetch: Callable[[str], bytes] = _fetch,
) -> Path:
    """Fetch instructions/config/tests only; the official image provides code.

    Tests stay in this host-side directory. The runtime never mounts the task
    directory into an agent's container. A caller may choose a published dataset
    revision; downloading does not make these tasks an approved research split.
    """
    if not SLUG.fullmatch(task_id):
        raise ValueError("Expected one RoadmapBench task slug")
    quoted_revision = urllib.parse.quote(revision, safe="")
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{quoted_revision}"
    listing = (
        f"https://huggingface.co/api/datasets/{DATASET}/tree/"
        f"{quoted_revision}/{task_id}/tests?recursive=true&limit=1000"
    )
    entries = json.loads(fetch(listing))
    if not isinstance(entries, list):
        raise ValueError("Invalid dataset tree response")
    # A task's verifier directory is small. Refuse a potentially truncated tree
    # rather than silently shipping only the first part of a verifier.
    if len(entries) >= 1000:
        raise ValueError("Verifier tree needs pagination; use an official dataset download")
    files = [f"{task_id}/instruction.md", f"{task_id}/task.toml"]
    for entry in entries:
        if entry.get("type") != "file":
            continue
        path = PurePosixPath(entry["path"])
        if ".." in path.parts or path.parts[:2] != (task_id, "tests"):
            raise ValueError("Dataset returned a path outside the task verifier")
        files.append(str(path))
    if f"{task_id}/tests/test.sh" not in files:
        raise ValueError("Official task has no tests/test.sh")
    target = Path(destination).resolve() / task_id
    for name in files:
        relative = PurePosixPath(name).relative_to(task_id)
        local = target.joinpath(*relative.parts)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(fetch(f"{base}/{urllib.parse.quote(name, safe='/')}"))
    (target / "data-source.json").write_text(
        json.dumps(
            {"dataset": DATASET, "revision": revision, "task_id": task_id,
             "downloaded_at": datetime.now(timezone.utc).isoformat(),
             "files": files, "oracle_downloaded": False},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return target
