"""Shared helpers for the explicit ProgramBench paid-launch scripts.

Standard library only: no network, processes, writes or credential persistence.
Launch scripts import these from the package rather than from one another.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex

CREDENTIAL_ENV = "STAGE1_GLM_API_KEY"


def branch_path(project_root: Path, run_name: str) -> Path:
    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_name):
        raise ValueError("run-name must be one explicit safe directory name")
    project_root = Path(project_root)
    if not project_root.is_absolute():
        raise ValueError("project-root must be an absolute server project path")
    return project_root / "runs" / run_name


def read_named_credential(credential_file: Path | None = None) -> str:
    if credential_file is None:
        value = os.environ.get(CREDENTIAL_ENV)
    else:
        values = []
        for line in Path(credential_file).read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith(CREDENTIAL_ENV + "="):
                tokens = shlex.split(line.partition("=")[2], comments=True)
                if len(tokens) != 1:
                    raise ValueError("Configured provider credential is malformed")
                values.append(tokens[0])
        value = values[0] if len(values) == 1 else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Exactly one nonempty configured provider credential is required")
    return value
