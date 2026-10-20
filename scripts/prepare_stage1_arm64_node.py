"""Prepare a native ARM64 agent runtime without changing an amd64 task image.

Inputs must already be local. The source container is created only for ``docker cp``;
it is never started, and is removed before this command returns.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import uuid


def _docker(*arguments: str) -> str:
    return subprocess.run(
        ["docker", *arguments], check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip()


def _extract_node(archive_path: Path, output_dir: Path) -> tuple[str, str]:
    """Strip the official top-level directory, retaining safe npm symlinks."""
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Safe extraction requires Python 3.11.8+ or Python 3.12+")
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        paths = [PurePosixPath(member.name) for member in members]
        if any(not path.parts or path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("Unsafe path in Node archive")
        roots = {path.parts[0] for path in paths}
        if len(roots) != 1:
            raise ValueError("Node archive must contain one official top-level directory")
        root = roots.pop()
        match = re.fullmatch(r"node-v(\d+\.\d+\.\d+)-linux-arm64", root)
        if match is None:
            raise ValueError("Expected an official node-vVERSION-linux-arm64 archive")

        def strip_root(name: str) -> str:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root:
                raise ValueError(f"Unsafe archive path: {name!r}")
            return str(PurePosixPath(*path.parts[1:]))

        # Check all pathnames before extracting any member. The standard data filter
        # additionally resolves symlinks against the actual extraction directory.
        for member in members:
            strip_root(member.name)
            if member.islnk():
                strip_root(member.linkname)

        def safe_member(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo | None:
            relative = strip_root(member.name)
            if relative == ".":
                if not member.isdir():
                    raise ValueError("The archive root must be a directory")
                return None
            selected = copy.copy(member)
            selected.name = relative
            if selected.islnk():
                selected.linkname = strip_root(selected.linkname)
            return tarfile.data_filter(selected, destination)

        archive.extractall(output_dir, members=members, filter=safe_member)
    if not (output_dir / "bin/node").is_file():
        raise ValueError("Node archive does not contain bin/node")
    return root, match.group(1)


def prepare_runtime(*, node_archive: Path, output_dir: Path,
                    source_image: str = "node:20-bookworm-slim") -> dict:
    """Use a fresh output directory; failed attempts leave their partial files intact."""
    node_archive = node_archive.resolve(strict=True)
    # lexists semantics: even a dangling output symlink must not be reused.
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing runtime: {output_dir}")
    if not source_image or source_image.startswith("-"):
        raise ValueError("source_image must be a local Docker image reference")
    inspected = json.loads(_docker("image", "inspect", source_image))[0]
    if inspected.get("Os") != "linux" or inspected.get("Architecture") != "arm64":
        raise ValueError("The local source image must be linux/arm64")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = output_dir.resolve()
    archive_root, node_version = _extract_node(node_archive, output_dir)
    compat_dir = output_dir / "compat"
    compat_dir.mkdir()
    source_name = "ac-stage1-node-source-" + uuid.uuid4().hex
    source_id = _docker(
        "create", "--pull=never", "--platform", "linux/arm64", "--network", "none",
        "--name", source_name, "--entrypoint", "/bin/true", inspected["Id"],
    )
    try:
        _docker("cp", f"{source_id}:/lib/aarch64-linux-gnu", str(compat_dir))
        _docker("cp", "-L", f"{source_id}:/etc/os-release", str(compat_dir / "os-release"))
    finally:
        # Never remove a user container: this ID was returned by our own create,
        # and this program has no run/start/exec path. No force-removal is needed.
        _docker("rm", source_id)

    library_dir = compat_dir / "aarch64-linux-gnu"
    if not (library_dir / "ld-linux-aarch64.so.1").is_file():
        raise ValueError("Source image did not provide the ARM64 dynamic loader")
    provenance = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "node_archive": str(node_archive),
        "archive_root": archive_root,
        "node_version": node_version,
        "node_architecture": "arm64",
        "source_image": {
            "reference": source_image,
            "id": inspected["Id"],
            "os": inspected["Os"],
            "architecture": inspected["Architecture"],
        },
        "source_distribution": (compat_dir / "os-release").read_text(),
        "source_container_started": False,
        "source_container_removed": True,
        "runtime_directory": str(output_dir),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-archive", required=True, type=Path,
                        help="Already-downloaded official node-vVERSION-linux-arm64 tar archive")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New runtime directory; existing paths are never overwritten")
    parser.add_argument("--source-image", default="node:20-bookworm-slim",
                        help="Already-local linux/arm64 image providing glibc and C++ libraries")
    args = parser.parse_args()
    try:
        result = prepare_runtime(node_archive=args.node_archive, output_dir=args.output_dir,
                                 source_image=args.source_image)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, f"Unable to prepare ARM64 Node runtime: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
