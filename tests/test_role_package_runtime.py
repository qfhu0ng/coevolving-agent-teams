"""Model-visible package snapshots: latest tail, stable system and exact scope."""
import pytest

from agent_company.policy import role_package_runtime as runtime
from agent_company.policy.role_package import create_p0_package
from agent_company.policy.role_package_store import RolePackageStore, parse_send_message, parse_spawn


def store_with_workers(mode="propose"):
    package = create_p0_package("A", "C\r\nexact trailing \n")
    package["roles"]["prober"]["retired"] = True
    package["roles"]["prober"]["policy"] = "OLD_DO_NOT_INJECT"
    store = RolePackageStore(mode, package, lambda r: None, clock=lambda: 1760000000)
    for index, name in enumerate(("zeta", "alpha")):
        store.commit(store.stage("master", store.snapshot("master"),
                                [parse_spawn(f"s{index}", name, "ROLE-TYPE: verifier")]))
        store.record_first_request(name)
    return store


def test_disclosure_uses_only_approved_mode_matrix():
    assert runtime.disclosure("propose", "worker").split("\n\n")[:1] == [runtime.RP_CORE]
    assert runtime.disclosure("propose", "worker") == "\n\n".join([
        runtime.RP_CORE, runtime.RP_CONTROL_SYNTAX, runtime.RP_Q,
        runtime.RP_PROPOSE_WORKER, runtime.RP_CARRY_FORWARD])
    assert runtime.disclosure("off", "master") == "\n\n".join([
        runtime.RP_CORE, runtime.RP_Q, runtime.RP_BINDING_MASTER,
        runtime.RP_OFF, runtime.RP_OFF_REVIEW_GRAMMAR])
    for mode in ("global", "master", "unknown"):
        with pytest.raises(ValueError):
            runtime.disclosure(mode, "master")
    for mode in ("off", "propose"):
        worker = runtime.disclosure(mode, "worker")
        assert runtime.RP_BINDING_MASTER not in worker
        assert "implementer builds assigned features" not in worker
        assert "A rejected control call can reject every call" not in worker
    assert "automatically" not in runtime.RP_SPAWN_SUFFIX


def test_system_does_not_change_when_pending_status_changes():
    store = store_with_workers()
    snapshot = store.snapshot("master")
    system = runtime.render_injection(snapshot)
    assert runtime.render_tail(snapshot) == "Pending proposals: none."
    proposal = parse_send_message("p", "leader", "POLICY-PROPOSAL C append\nCHANGE: New reusable rule.\nREASON: stale handoff\nEVIDENCE: task 4")
    store.commit(store.stage("zeta", store.snapshot("zeta"), [proposal]))
    after = store.snapshot("master")
    assert runtime.render_injection(after) == system
    assert "Proposal #1 for C, operation append, from zeta" in runtime.render_tail(after)
    assert "stale handoff" not in system
    assert runtime.render_tail(snapshot) == "Pending proposals: none."
    decision = parse_send_message("d", "zeta", "POLICY-DECISION #1 reject\nNOTE: Rule already covers it.")
    store.commit(store.stage("master", after, [decision]))
    assert runtime.render_injection(store.snapshot("master")) == system
    assert runtime.render_tail(store.snapshot("master")) == "Pending proposals: none."
    assert runtime.render_tail(store.snapshot("zeta")) == "Your proposal #1 for C was rejected. Note: Rule already covers it."


def test_worker_gets_current_own_role_only_and_retired_never_active():
    store = store_with_workers()
    system = runtime.render_snapshot(store.snapshot("zeta"))
    assert "Role directory: implementer, integrator, master, verifier." in system
    assert "C\r\nexact trailing \n" in system
    assert "You are zeta; your role type is verifier." in system
    assert "POLICY-PROPOSAL role:verifier append" in system
    assert "Your shared role policy (version 0) is empty; only C applies." in system
    assert "Shared role policies" not in system
    assert "OLD_DO_NOT_INJECT" not in system
    assert "slot" not in system and "lifecycle" not in system
    master = runtime.render_snapshot(store.snapshot("master"))
    assert "Teammate role types (in spawn order): zeta=verifier, alpha=verifier" in master
    assert master.count("Role master (version 0)") == 1
    assert "OLD_DO_NOT_INJECT" not in master


def test_worker_status_keeps_only_latest_settled_per_target():
    store = store_with_workers()
    for index in range(1, 4):
        proposal = parse_send_message(f"p{index}", "leader", f"POLICY-PROPOSAL C append\nCHANGE: Rule {index}\nREASON: why\nEVIDENCE: task {index}")
        store.commit(store.stage("zeta", store.snapshot("zeta"), [proposal]))
        decision = parse_send_message(f"d{index}", "zeta", f"POLICY-DECISION #{index} reject")
        store.commit(store.stage("master", store.snapshot("master"), [decision]))
    assert runtime.render_tail(store.snapshot("zeta")) == "Your proposal #3 for C was rejected."
    assert len(store.export()["proposals"]) == 3


def test_off_has_no_pending_tail_or_worker_proposal_header():
    store = store_with_workers("off")
    for actor in ("master", "zeta"):
        assert runtime.render_tail(store.snapshot(actor)) == ""
    worker = runtime.render_injection(store.snapshot("zeta"))
    assert "To propose a rule for your role" not in worker
    assert runtime.RP_CONTROL_SYNTAX not in worker
    assert runtime.RP_OFF in worker
