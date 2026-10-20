"""Source provenance uses existing Git/manifest identities, never a deployment path."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from agent_company.deploy.code_version import code_version, code_version_label, known_code_version


def snapshot(root, body=b"source", *, spaces=" "):
    path = root / "agent_company/module.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(body)
    (root.parent / "code_sha256.txt").write_text(hashlib.sha256(body).hexdigest() + spaces + "./agent_company/module.py\n")


def test_relocated_snapshots_compare_existing_file_digests(tmp_path):
    a, b = tmp_path / "deploy_a/code", tmp_path / "deploy_b/code"
    snapshot(a)
    snapshot(b, spaces="  ")
    assert known_code_version(code_version(a))
    assert code_version(a) == code_version(b)
    (b / "agent_company/module.py").write_bytes(b"changed overlay")
    assert code_version(b) is None


@pytest.mark.parametrize("entry", ["", "broken", "0" * 64 + " ./missing.py\n", "0" * 64 + " ../outside.py\n"])
def test_unusable_manifest_is_unknown(tmp_path, entry):
    root = tmp_path / "code"
    snapshot(root)
    (root.parent / "code_sha256.txt").write_text(entry)
    assert code_version(root) is None


def test_duplicate_manifest_path_is_unknown(tmp_path):
    root = tmp_path / "code"
    snapshot(root)
    manifest = root.parent / "code_sha256.txt"
    manifest.write_text(manifest.read_text() * 2)
    assert code_version(root) is None


def test_archive_without_manifest_never_discovers_parent_git(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    archive = tmp_path / "runs/example/code"
    archive.mkdir(parents=True)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("parent Git discovery"))
    assert code_version(archive) is None


@pytest.mark.parametrize("dirty,wrong_root,known", [(False, False, True), (True, False, False), (False, True, False)])
def test_only_exact_clean_git_root_can_record_head(tmp_path, monkeypatch, dirty, wrong_root, known):
    (tmp_path / ".git").write_text("gitdir: /synthetic/worktree")
    calls = []
    def git(command, **kwargs):
        calls.append(command)
        assert command[:3] == ["git", "-C", str(tmp_path)]
        args = command[3:]
        if args == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=str(tmp_path.parent if wrong_root else tmp_path) + "\n")
        if args[0] == "status":
            return SimpleNamespace(stdout=" M source.py\n" if dirty else "")
        assert args == ["rev-parse", "HEAD"]
        return SimpleNamespace(stdout="a" * 40 + "\n")
    monkeypatch.setattr("subprocess.run", git)
    value = code_version(tmp_path)
    assert known_code_version(value) is known
    if not known:
        assert value is None and not any(c[-1] == "HEAD" for c in calls)


def test_version_label_is_display_only_and_does_not_replace_complete_manifest(tmp_path):
    root = tmp_path / "code"
    snapshot(root)
    version = code_version(root)
    canonical = json.dumps(version["files"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert code_version_label(version) == "deployment_manifest:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    assert "run_settings" not in code_version_label(version)
    assert version["files"] == {"agent_company/module.py": hashlib.sha256(b"source").hexdigest()}
    assert code_version_label({"kind": "git", "commit": "a" * 40}) == "git:aaaaaaaaaaaa"
    assert code_version_label(None) is None


def test_manifest_display_label_is_order_independent_but_distinguishes_equal_counts():
    files = {"agent_company/z.py": "a" * 64, "agent_company/规则.py": "b" * 64}
    version = {"kind": "deployment_manifest", "files": files}
    reversed_version = {"kind": "deployment_manifest", "files": dict(reversed(files.items()))}
    changed_version = {"kind": "deployment_manifest", "files": {**files, "agent_company/z.py": "c" * 64}}
    assert code_version_label(version) == code_version_label(reversed_version)
    assert code_version_label(version) != code_version_label(changed_version)
    assert version["files"] == files
