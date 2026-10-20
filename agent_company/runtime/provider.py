"""Load a provider configuration and describe it without credential values."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from agent_company.runtime.relay import ProviderConfig, TokenRates


def load_provider(path: Path) -> ProviderConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "api_key" in data:
        raise ValueError("Use api_key_env, not a credential value, in provider configuration")
    key_env = data.pop("api_key_env", None)
    api_key = ""
    if key_env:
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise ValueError(f"Provider credential environment variable {key_env} is not set")
    if data.get("rates") is not None:
        data["rates"] = TokenRates(**data["rates"])
    return ProviderConfig(api_key=api_key, **data)


def public_provider(provider):
    result = asdict(provider)
    result.pop("api_key")
    if result["rates"]:
        result["rates"] = {key: str(value) for key, value in result["rates"].items()}
    return result
