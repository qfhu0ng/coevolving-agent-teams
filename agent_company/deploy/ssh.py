"""Credential-free SSH setup shared by the explicit ProgramBench launchers."""
from __future__ import annotations

import os
import subprocess

# The experiment server: set these for your own machine. Only the key path is read, never the key.
HOST = os.environ.get("AGENT_COMPANY_SSH_HOST", "user@experiment-server")
KEY = os.path.expanduser(os.environ.get("AGENT_COMPANY_SSH_KEY", "~/.ssh/agent_company_server"))
KNOWN = os.path.expanduser(os.environ.get("AGENT_COMPANY_SSH_KNOWN_HOSTS", "~/.ssh/agent_company_known_hosts"))
CREDENTIAL_ENV_NAMES = ("DEEPSEEK_API_KEY", "STAGE1_GLM_API_KEY")


def clean_environment() -> dict:
    return {key: value for key, value in os.environ.items() if key not in CREDENTIAL_ENV_NAMES}


def credential(payload):
    """Read a supplied payload only when called, after the launcher's checks."""
    try:
        value = payload() if callable(payload) else payload
    except Exception:
        raise ValueError("Cannot read credential payload from stdin") from None
    key = value.get("api_key") if isinstance(value, dict) else None
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Missing credential on stdin")
    return key


def ssh(remote, *, stdin=None, timeout=120, host=HOST, key=KEY, known=KNOWN):
    command = ["ssh", "-T", "-F", "/dev/null", "-i", str(key), "-o", "IdentitiesOnly=yes",
               "-o", "BatchMode=yes", "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
               "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
               "-o", f"UserKnownHostsFile={known}", host, remote]
    return subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=timeout,
                          env=clean_environment())
