"""New package validation leaves legacy A/C untouched and preserves policy bytes."""
import json

import pytest

from agent_company.policy.role_package import (
    create_p0_package, load_role_package, package_policy_identity, validate_role_package,
)


def test_p0_preserves_crlf_unicode_and_empty_roles(tmp_path):
    package = create_p0_package("A\r\n 末尾\n", "C\r\n ", package_id="series/P0")
    path = tmp_path / "package.json"
    path.write_bytes(json.dumps(package, ensure_ascii=False).encode("utf-8"))
    assert load_role_package(path) == package
    assert package["team"] == {"assignment": "A\r\n 末尾\n", "communication": "C\r\n "}
    assert len(package["roles"]) == 5
    assert all(r == {"policy": "", "retired": False} for r in package["roles"].values())
    result = validate_role_package(package)
    result["roles"]["master"]["policy"] = "Detached"
    assert package["roles"]["master"]["policy"] == ""


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(unknown=True),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(revision=-1),
    lambda p: p["protocol"].update(id="other"),
    lambda p: p["team"].update(communication=" "),
    lambda p: p["team"].update(communication="界" * 6001),
    lambda p: p["roles"]["master"].update(retired=True),
    lambda p: p["roles"]["verifier"].update(retired=0),
    lambda p: p["roles"].update(UPPER={"policy": "", "retired": False}),
    lambda p: p["roles"]["verifier"].update(policy="x" * 2001),
    lambda p: p.update(changes=[{"value": float("nan")}]),
    lambda p: p["team"].update(assignment="\ud800"),
])
def test_package_rejects_invalid_shape_or_text(mutate):
    package = create_p0_package("A", "C")
    mutate(package)
    with pytest.raises(ValueError):
        validate_role_package(package)


def test_retired_entries_still_consume_catalog_and_total_budget():
    p = create_p0_package("A", "C")
    for n in range(9):
        p["roles"][f"old{n}"] = {"policy": "x" * 2000, "retired": True}
    with pytest.raises(ValueError, match="16000"):
        validate_role_package(p)
    for role in p["roles"].values():
        role["policy"] = ""
    for n in range(3):
        p["roles"][f"new{n}"] = {"policy": "", "retired": False}
    with pytest.raises(ValueError, match="16"):
        validate_role_package(p)


def test_duplicate_json_keys_are_not_silently_replaced(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_role_package(path)


def test_policy_identity_excludes_only_lineage_review_metadata():
    p = create_p0_package("A", "C")
    other = dict(p, package_id="v1", revision=1, parent={"package_id": "P0", "revision": 0},
                 changes=[{"origin": "event1"}], review_ref="review.json")
    assert package_policy_identity(p) == package_policy_identity(other)
    other = validate_role_package(other)
    other["roles"]["verifier"]["policy"] = "Report check results."
    assert package_policy_identity(p) != package_policy_identity(other)
