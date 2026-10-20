"""Record source identity at execution time using Git or the existing deployment manifest.

Deployment directories move and archives have no .git. Never infer an archived
checkout's version from a parent repository; unknown provenance is not equality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


def code_version(project_root: Path) -> dict | None:
    root = Path(project_root).resolve()
    manifest = root.parent / "code_sha256.txt"
    if root.name == "code" and manifest.is_file():
        try:
            files = {}
            for line in manifest.read_text().splitlines():
                digest, relative = line.split(None, 1)
                relative = relative.removeprefix("./")
                path = root / relative
                if (not re.fullmatch(r"[0-9a-f]{64}", digest) or relative in files
                        or not path.resolve().is_relative_to(root)
                        or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                    return None
                files[relative] = digest
            if not files or not any(p.startswith("agent_company/") for p in files):
                return None
            return {"kind": "deployment_manifest", "files": dict(sorted(files.items()))}
        except (OSError, ValueError):
            return None
    if not (root / ".git").exists():
        return None
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return None
        if git("status", "--porcelain", "--untracked-files=normal"):
            return None
        head = git("rev-parse", "HEAD")
        return {"kind": "git", "commit": head} if re.fullmatch(r"[0-9a-f]{40,64}", head) else None
    except (OSError, subprocess.SubprocessError):
        return None


def known_code_version(value) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("kind") == "git":
        return isinstance(value.get("commit"), str) and bool(re.fullmatch(r"[0-9a-f]{40,64}", value["commit"]))
    files = value.get("files")
    return (value.get("kind") == "deployment_manifest" and isinstance(files, dict) and bool(files)
            and all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                    for k, v in files.items()))


def code_version_label(value) -> str | None:
    """Short display label only; comparisons still use the complete identity.

    Deployed archives omit Git metadata, and file counts cannot distinguish two
    different manifests of the same size. Canonicalize the existing manifest for
    this display-only label; comparisons still use every recorded path/digest.
    """
    if not known_code_version(value):
        return None
    if value["kind"] == "git":
        return f"git:{value['commit'][:12]}"
    canonical = json.dumps(value["files"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"deployment_manifest:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]}"
