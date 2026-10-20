from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "agent_company"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

FORBIDDEN_PACKAGE_IMPORT_ROOTS = {"docs", "runs", "scripts", "tests"}

# Frozen at the start of P1. P2 removes entries as shared CLI logic moves into
# agent_company; no new edge may be added.
SCRIPT_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def test_package_does_not_import_repository_interfaces_or_outputs() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for imported in sorted(_absolute_imports(path)):
            root = imported.split(".", 1)[0]
            if root in FORBIDDEN_PACKAGE_IMPORT_ROOTS:
                violations.append(f"{path.relative_to(REPO_ROOT)} -> {imported}")
    assert violations == []


def test_script_import_edges_match_frozen_migration_allowlist() -> None:
    observed: set[tuple[str, str]] = set()
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        for imported in _absolute_imports(path):
            if imported == "scripts" or imported.startswith("scripts."):
                observed.add((path.name, imported))

    unexpected = sorted(observed - SCRIPT_IMPORT_ALLOWLIST)
    stale_allowlist = sorted(SCRIPT_IMPORT_ALLOWLIST - observed)
    assert unexpected == [], f"new script-to-script imports are forbidden: {unexpected}"
    assert stale_allowlist == [], (
        "remove migrated edges from SCRIPT_IMPORT_ALLOWLIST: "
        f"{stale_allowlist}"
    )


def test_canonical_runtime_entry_points_are_importable() -> None:
    from agent_company.deploy.launch import launch, prepare, status
    from agent_company.runtime.qwen import QwenTeamRunner
    from agent_company.runtime.relay import ProviderConfig

    assert all(callable(item) for item in (launch, prepare, status))
    assert callable(QwenTeamRunner)
    assert callable(ProviderConfig)
