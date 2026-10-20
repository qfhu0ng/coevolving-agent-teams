"""Validated, byte-preserving policy packages; legacy A/C inputs never opt in here."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

ROLE_POLICY_PROTOCOL = "role_package_v1"
# Schema 2: per-teammate policies written by the master; the package holds only M.
AGENT_RULES_PROTOCOL = "agent_rules_v1"
PACKAGE_PROTOCOLS = (ROLE_POLICY_PROTOCOL, AGENT_RULES_PROTOCOL)
# Team modes each package protocol implements; agent_rules_v1 has one mixed mode.
PROTOCOL_MODES = {ROLE_POLICY_PROTOCOL: ("off", "propose"), AGENT_RULES_PROTOCOL: ("propose",)}
FIXED_PROTOCOL = "collaboration_protocol_v1"
ROLE_ID_PATTERN = r"[a-z][a-z0-9_-]{0,31}"
ROLE_ID_RE = re.compile(ROLE_ID_PATTERN + r"\Z", re.ASCII)
INITIAL_ROLES = ("master", "integrator", "verifier", "prober", "implementer")
RESERVED_ROLES = frozenset({"master", "implementer"})
MAX_ROLES = 16
MAX_ROLE_CHARS = 2000
MAX_TOTAL_ROLE_CHARS = 16000
MAX_COMMUNICATION_CHARS = 6000
MAX_PENDING_CHARS = 8000


def _object(value, required, optional=(), *, where):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{where} must be an object")
    if not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f"{where} has missing or unsupported fields")


def _text(value, *, where, nonempty=False, limit=None):
    if not isinstance(value, str) or nonempty and not value.strip():
        raise ValueError(f"{where} must be {'nonempty ' if nonempty else ''}text")
    # JSON permits lone surrogates; the saved package and the request must be UTF-8.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where} must be valid UTF-8 text") from exc
    if limit is not None and len(value) > limit:
        raise ValueError(f"{where} exceeds {limit} characters")


def _reference(value, where):
    _object(value, ("package_id", "revision"), where=where)
    _text(value["package_id"], where=f"{where}.package_id", nonempty=True)
    if type(value["revision"]) is not int or value["revision"] < 0:
        raise ValueError(f"{where}.revision must be a nonnegative integer")


def validate_role_package(value: object) -> dict:
    """Validate the explicit v1 shape; return a detached copy without normalizing text.

    Change records are retained JSON evidence, not runtime instructions. Their full
    per-rule review schema belongs to the later evolution pipeline.
    """
    _object(value, ("schema_version", "package_id", "revision", "parent", "protocol",
                    "team", "roles", "changes", "review_ref"),
            ("attempted_parent", "fallback_reason"), where="policy package")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported policy package schema_version")
    _reference({"package_id": value["package_id"], "revision": value["revision"]}, "package")
    if value["parent"] is not None:
        _reference(value["parent"], "parent")
    if value.get("attempted_parent") is not None:
        _reference(value["attempted_parent"], "attempted_parent")
    if value.get("fallback_reason") is not None:
        _text(value["fallback_reason"], where="fallback_reason", nonempty=True)
    _object(value["protocol"], ("id",), where="protocol")
    if value["protocol"]["id"] != FIXED_PROTOCOL:
        raise ValueError("Unsupported fixed collaboration protocol")
    _object(value["team"], ("assignment", "communication"), where="team")
    _text(value["team"]["assignment"], where="team.assignment", nonempty=True)
    _text(value["team"]["communication"], where="team.communication", nonempty=True,
          limit=MAX_COMMUNICATION_CHARS)
    roles = value["roles"]
    if not isinstance(roles, dict) or not RESERVED_ROLES <= roles.keys() or len(roles) > MAX_ROLES:
        raise ValueError("Role directory must contain master and implementer and at most 16 entries")
    for role_id, role in roles.items():
        if not isinstance(role_id, str) or not ROLE_ID_RE.fullmatch(role_id):
            raise ValueError("Invalid role ID")
        _object(role, ("policy", "retired"), where=f"roles.{role_id}")
        _text(role["policy"], where=f"roles.{role_id}.policy", limit=MAX_ROLE_CHARS)
        if type(role["retired"]) is not bool or role_id in RESERVED_ROLES and role["retired"]:
            raise ValueError("Role retired must be boolean; reserved roles cannot retire")
    if sum(len(r["policy"]) for r in roles.values()) > MAX_TOTAL_ROLE_CHARS:
        raise ValueError("Role texts, including retired roles, exceed 16000 characters")
    if not isinstance(value["changes"], list) or any(not isinstance(c, dict) for c in value["changes"]):
        raise ValueError("changes must be a list of change records")
    if value["review_ref"] is not None:
        _text(value["review_ref"], where="review_ref", nonempty=True)
    # Reject non-JSON values/NaN in opaque provenance too, without rewriting it.
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Policy package must contain only valid UTF-8 JSON data") from exc
    return copy.deepcopy(value)


def validate_agent_rules_package(value: object) -> dict:
    """Schema 2 (agent_rules_v1): M only, no roles; texts kept byte-for-byte."""
    _object(value, ("schema_version", "package_id", "revision", "parent", "protocol",
                    "team", "changes", "review_ref"),
            ("attempted_parent", "fallback_reason"), where="policy package")
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise ValueError("Unsupported policy package schema_version")
    _reference({"package_id": value["package_id"], "revision": value["revision"]}, "package")
    for key in ("parent", "attempted_parent"):
        if value.get(key) is not None:
            _reference(value[key], key)
    if value.get("fallback_reason") is not None:
        _text(value["fallback_reason"], where="fallback_reason", nonempty=True)
    _object(value["protocol"], ("id",), where="protocol")
    if value["protocol"]["id"] != AGENT_RULES_PROTOCOL:
        raise ValueError("A schema 2 package must declare protocol agent_rules_v1")
    _object(value["team"], ("assignment", "communication"), where="team")
    for key in ("assignment", "communication"):
        _text(value["team"][key], where=f"team.{key}", nonempty=True)
    if not isinstance(value["changes"], list) or any(not isinstance(c, dict) for c in value["changes"]):
        raise ValueError("changes must be a list of change records")
    if value["review_ref"] is not None:
        _text(value["review_ref"], where="review_ref", nonempty=True)
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Policy package must contain only valid UTF-8 JSON data") from exc
    return copy.deepcopy(value)


def validate_policy_package(value: object) -> dict:
    """Dispatch on schema_version; each schema keeps its own strict validation."""
    version = value.get("schema_version") if isinstance(value, dict) else None
    if type(version) is int and version == 2:
        return validate_agent_rules_package(value)
    return validate_role_package(value)


def policy_protocol(package: dict) -> str:
    """The runtime protocol a validated package selects; never inferred from mode."""
    return AGENT_RULES_PROTOCOL if validate_policy_package(package)["schema_version"] == 2 else ROLE_POLICY_PROTOCOL


def load_policy_package(path: str | Path) -> dict:
    raw = Path(path).read_bytes().decode("utf-8")
    return validate_policy_package(json.loads(raw, object_pairs_hook=_unique_pairs))


def create_m0_package(assignment: str, communication: str, *, package_id: str = "M0") -> dict:
    """The M0 texts themselves need the user's word-by-word approval (plan v4 section 14)."""
    return validate_agent_rules_package({
        "schema_version": 2, "package_id": package_id, "revision": 0, "parent": None,
        "protocol": {"id": AGENT_RULES_PROTOCOL},
        "team": {"assignment": assignment, "communication": communication},
        "changes": [], "review_ref": None,
    })


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate policy-package key: {key}")
        value[key] = item
    return value


def load_role_package(path: str | Path) -> dict:
    """Read exact UTF-8 text, rejecting duplicate JSON keys and non-finite numbers."""
    raw = Path(path).read_bytes().decode("utf-8")
    return validate_role_package(json.loads(raw, object_pairs_hook=_unique_pairs))


def create_p0_package(assignment: str, communication: str, *, package_id: str = "P0") -> dict:
    return validate_role_package({
        "schema_version": 1, "package_id": package_id, "revision": 0, "parent": None,
        "protocol": {"id": FIXED_PROTOCOL},
        "team": {"assignment": assignment, "communication": communication},
        "roles": {role_id: {"policy": "", "retired": False} for role_id in INITIAL_ROLES},
        "changes": [], "review_ref": None,
    })


def package_policy_identity(package: dict) -> dict:
    """Only input policy/runtime text determines policy identity, not review metadata."""
    value = validate_policy_package(package)
    return {key: value[key] for key in ("protocol", "team", "roles") if key in value}
