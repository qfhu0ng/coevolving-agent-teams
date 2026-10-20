import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import prepare_stage1_arm64_node as prepare


ROOT = "node-v22.23.2-linux-arm64"


def make_archive(tmp_path, extra=()):
    path = tmp_path / "node.tar.xz"
    with tarfile.open(path, "w:xz") as archive:
        for name, content in (("bin/node", b"node fixture"),
                              ("lib/node_modules/npm/bin/npm-cli.js", b"npm fixture")):
            member = tarfile.TarInfo(f"{ROOT}/{name}")
            member.mode = 0o755
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        npm = tarfile.TarInfo(f"{ROOT}/bin/npm")
        npm.type = tarfile.SYMTYPE
        npm.linkname = "../lib/node_modules/npm/bin/npm-cli.js"
        archive.addfile(npm)
        for member in extra:
            archive.addfile(member)
    return path


class FakeDocker:
    def __init__(self, *, architecture="arm64", fail_copy=False):
        self.architecture = architecture
        self.fail_copy = fail_copy
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("image", "inspect"):
            return json.dumps([{"Id": "sha256:local-source", "Os": "linux",
                                "Architecture": self.architecture}])
        if args[0] == "create":
            return "temporary-source-id"
        if args[0] == "cp":
            if self.fail_copy:
                raise subprocess.CalledProcessError(1, ["docker", *args], stderr="copy failed")
            if args[-2].endswith(":/lib/aarch64-linux-gnu"):
                destination = Path(args[-1]) / "aarch64-linux-gnu"
                destination.mkdir()
                (destination / "ld-linux-aarch64.so.1").write_bytes(b"loader fixture")
                (destination / "libc.so.6").write_bytes(b"libc fixture")
            else:
                Path(args[-1]).write_text('ID=debian\nVERSION_ID="12"\n')
            return ""
        assert args == ("rm", "temporary-source-id")
        return "temporary-source-id"


def test_prepare_local_only_preserves_npm_symlink_and_provenance(tmp_path, monkeypatch):
    docker = FakeDocker()
    monkeypatch.setattr(prepare, "_docker", docker)
    archive = make_archive(tmp_path)
    target = tmp_path / "runtime"
    result = prepare.prepare_runtime(node_archive=archive, output_dir=target)
    assert (target / "bin/node").stat().st_mode & 0o111
    assert (target / "bin/npm").is_symlink()
    assert (target / "bin/npm").read_bytes() == b"npm fixture"
    assert (target / "compat/aarch64-linux-gnu/ld-linux-aarch64.so.1").is_file()
    assert result == json.loads((target / "provenance.json").read_text())
    assert result["node_version"] == "22.23.2"
    assert result["source_image"] == {"reference": "node:20-bookworm-slim",
        "id": "sha256:local-source", "os": "linux", "architecture": "arm64"}
    assert 'ID=debian' in result["source_distribution"]
    assert result["source_container_started"] is False
    assert result["source_container_removed"] is True
    create = next(call for call in docker.calls if call[0] == "create")
    assert "--pull=never" in create
    assert create[-1] == "sha256:local-source"
    assert create[create.index("--network") + 1] == "none"
    assert {call[0] for call in docker.calls} == {"image", "create", "cp", "rm"}
    assert docker.calls[-1] == ("rm", "temporary-source-id")


@pytest.mark.parametrize("dangling_symlink", [False, True])
def test_existing_output_never_overwritten(tmp_path, monkeypatch, dangling_symlink):
    target = tmp_path / "runtime"
    if dangling_symlink:
        target.symlink_to(tmp_path / "missing")
    else:
        target.mkdir()
        (target / "keep").write_text("user data")
    docker = FakeDocker()
    monkeypatch.setattr(prepare, "_docker", docker)
    with pytest.raises(FileExistsError):
        prepare.prepare_runtime(node_archive=make_archive(tmp_path), output_dir=target)
    assert not docker.calls
    if not dangling_symlink:
        assert (target / "keep").read_text() == "user data"


def test_rejects_x64_source_before_creating_output_or_container(tmp_path, monkeypatch):
    docker = FakeDocker(architecture="amd64")
    monkeypatch.setattr(prepare, "_docker", docker)
    with pytest.raises(ValueError, match="linux/arm64"):
        prepare.prepare_runtime(node_archive=make_archive(tmp_path), output_dir=tmp_path / "runtime")
    assert [call[0] for call in docker.calls] == ["image"]
    assert not (tmp_path / "runtime").exists()


def test_failed_copy_removes_only_created_source_and_keeps_partial_output(tmp_path, monkeypatch):
    docker = FakeDocker(fail_copy=True)
    monkeypatch.setattr(prepare, "_docker", docker)
    target = tmp_path / "runtime"
    with pytest.raises(subprocess.CalledProcessError):
        prepare.prepare_runtime(node_archive=make_archive(tmp_path), output_dir=target)
    assert docker.calls[-1] == ("rm", "temporary-source-id")
    assert (target / "bin/node").is_file()
    assert not (target / "provenance.json").exists()


@pytest.mark.parametrize("attack", ["parent", "absolute", "symlink", "symlink_chain", "hardlink", "fifo"])
def test_archive_cannot_escape_fresh_runtime(tmp_path, monkeypatch, attack):
    target = tmp_path / "runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("user data")
    member = tarfile.TarInfo(f"{ROOT}/bad")
    extras = []
    if attack == "parent":
        member.name = f"{ROOT}/../outside/sentinel"
    elif attack == "absolute":
        member.name = str(sentinel)
    elif attack == "symlink":
        member.type = tarfile.SYMTYPE
        member.linkname = "../outside/sentinel"
    elif attack == "symlink_chain":
        member.type = tarfile.SYMTYPE
        member.linkname = "."
        child = tarfile.TarInfo(f"{ROOT}/bad/escape")
        child.type = tarfile.SYMTYPE
        child.linkname = "../outside/sentinel"
        extras.append(member)
        member = child
    elif attack == "hardlink":
        member.type = tarfile.LNKTYPE
        member.linkname = f"{ROOT}/../outside/sentinel"
    else:
        member.type = tarfile.FIFOTYPE
    extras.append(member)
    docker = FakeDocker()
    monkeypatch.setattr(prepare, "_docker", docker)
    with pytest.raises((ValueError, tarfile.TarError)):
        prepare.prepare_runtime(node_archive=make_archive(tmp_path, extras), output_dir=target)
    assert sentinel.read_text() == "user data"
    assert [call[0] for call in docker.calls] == ["image"]


def test_missing_local_image_does_not_pull_or_create_output(tmp_path, monkeypatch):
    calls = []

    def missing(*args):
        calls.append(args)
        raise subprocess.CalledProcessError(1, ["docker", *args])

    monkeypatch.setattr(prepare, "_docker", missing)
    target = tmp_path / "runtime"
    with pytest.raises(subprocess.CalledProcessError):
        prepare.prepare_runtime(node_archive=make_archive(tmp_path), output_dir=target)
    assert calls == [("image", "inspect", "node:20-bookworm-slim")]
    assert not target.exists()
