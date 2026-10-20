"""Per-teammate policy store for agent_rules_v1: transactions, controls, lifecycle, replay (zero API)."""
import pytest

from agent_company.policy import agent_rules as r
from agent_company.policy.agent_policy_store import AgentPolicyStore
from agent_company.policy.role_package import (create_m0_package, load_policy_package, policy_protocol,
                                               validate_policy_package)
from agent_company.policy.role_policy import PolicyPersistError, ReplayError


def make():
    events = []
    return AgentPolicyStore(create_m0_package('Assign by module.', 'Report handoffs.'), events.append), events


def commit(store, actor, *calls, snapshot=None):
    return store.commit(store.stage(actor, snapshot or store.snapshot(actor), calls))


def spawn(store, name, prompt='Task.', start=True):
    result = commit(store, 'master', r.SpawnCall('s-' + name, name, r.parse_spawn_policy(prompt)))
    assert not result.rejections
    if start:
        assert store.record_spawn_result(name, ok=True) == 'ok'
        assert store.record_first_request(name) == 'bound'
    return result


def control(actor_to, message, call_id='c1'):
    return r.parse_control(call_id, actor_to, message)


def policy(store, name):
    return store.snapshot('master').worker(name).policy


def test_schema_v2_package_dispatch_and_protocol(tmp_path):
    package = create_m0_package('A', 'C')
    assert policy_protocol(package) == 'agent_rules_v1' and 'roles' not in package
    path = tmp_path / 'm0.json'
    path.write_text('{"schema_version": 2, "schema_version": 2}')
    with pytest.raises(ValueError, match='Duplicate'):
        load_policy_package(path)
    with pytest.raises(ValueError):
        validate_policy_package({**package, 'roles': {}})
    with pytest.raises(ValueError):
        validate_policy_package({**package, 'protocol': {'id': 'role_package_v1'}})


def test_spawn_mixed_rules_warn_and_still_start():
    store, _ = make()
    spawn(store, 'integ')
    prompt = ('Build the writer.\nPOLICY:\nWHEN commit THEN SEND TO integ WITH sha\nWHEN commit THEN WAIT\n'
              'WHEN commit THEN SEND TO ghost\nNOTES:\nScope: src/Writers/.\nEND-POLICY')
    result = spawn(store, 'impl', prompt)
    receipt = result.receipts['s-impl']
    assert 'version 0, 1 rule(s)' in receipt and receipt.count('Skipped rule') == 2
    current = policy(store, 'impl')
    assert current.rules == (('R1', 'WHEN commit THEN SEND TO integ WITH sha'),) and current.notes == 'Scope: src/Writers/.'
    planned = next(c for c in result.changes if c['event'] == 'agent_policy_planned')
    assert planned['policy_missing'] is False and len(planned['skipped']) == 2


def test_rules_may_name_teammates_spawned_later_in_the_same_response():
    store, _ = make()
    first = r.SpawnCall('s1', 'impl', r.parse_spawn_policy('POLICY:\nWHEN commit THEN SEND TO integ\nEND-POLICY'))
    second = r.SpawnCall('s2', 'integ', r.parse_spawn_policy('Integrate.'))
    result = commit(store, 'master', first, second)
    assert not result.rejections and len(store.snapshot('master').worker('impl').policy.rules) == 1
    missing = [c for c in result.changes if c['event'] == 'agent_policy_planned' and c['policy_missing']]
    assert [c['worker']['name'] for c in missing] == ['integ']


def test_duplicate_live_name_is_a_real_spawn_rejection():
    store, _ = make()
    spawn(store, 'impl')
    result = commit(store, 'master', r.SpawnCall('s2', 'impl', r.parse_spawn_policy('Again.')))
    assert [x.code for x in result.rejections] == ['duplicate_name']


def test_succession_inherits_latest_policy_and_continues_versions():
    store, _ = make()
    spawn(store, 'integ')
    spawn(store, 'old', 'POLICY:\nWHEN commit THEN SEND TO integ WITH sha\nNOTES:\nold notes\nEND-POLICY')
    assert not commit(store, 'old', control('leader', 'POLICY-PROPOSAL self add\nWHEN task_done THEN SEND TO leader\nREASON: r')).rejections
    early = spawn(store, 'early', 'SUCCEEDS: old\nTask.', start=False)
    assert 'has not ended' in early.receipts['s-early'] and policy(store, 'early').rules == ()
    store.record_terminal('old')
    result = spawn(store, 'new', 'SUCCEEDS: old\nPOLICY:\nWHEN start THEN SEND TO leader\nNOTES:\nnew notes\nEND-POLICY\nGo.')
    inherited = policy(store, 'new')
    assert inherited.version == 2 and inherited.lineage == {'from': 'old', 'version': 1}
    assert [rule_id for rule_id, _ in inherited.rules] == ['R1', 'R2', 'R3'] and inherited.notes == 'new notes'
    assert 'starting from old version 1' in result.receipts['s-new']
    unknown = spawn(store, 'other', 'SUCCEEDS: nobody\nTask.')
    assert 'no teammate nobody' in unknown.receipts['s-other']


def test_master_edit_multi_target_partial_results_and_star_scope():
    store, _ = make()
    spawn(store, 'integ')
    spawn(store, 'impl')
    edit = control('*', 'AGENT-POLICY * add\nWHEN commit THEN SEND TO integ WITH sha')
    result = commit(store, 'master', edit)
    assert not result.rejections
    assert 'integ: not applied (self_recipient)' in result.receipts['c1'] and 'impl: applied, version 1' in result.receipts['c1']
    spawn(store, 'late')
    assert policy(store, 'late').rules == ()  # * expanded at commit time only


def test_all_targets_failing_rejects_the_call():
    store, _ = make()
    spawn(store, 'integ')
    result = commit(store, 'master', control('integ', 'AGENT-POLICY integ add\nWHEN commit THEN SEND TO integ'))
    assert [x.code for x in result.rejections] == ['self_recipient'] and policy(store, 'integ').version == 0
    result = commit(store, 'master', control('integ', 'AGENT-POLICY ghost add\nWHEN commit THEN SEND TO leader'))
    assert [x.code for x in result.rejections] == ['unknown_target']


def test_version_checks_with_and_without_base_and_concurrent_self_change():
    store, _ = make()
    spawn(store, 'impl')
    old_master = store.snapshot('master')
    worker_view = store.snapshot('impl')
    assert not commit(store, 'impl', control('leader', 'POLICY-PROPOSAL self add\nWHEN start THEN SEND TO leader\nREASON: r'),
                      snapshot=worker_view).rejections
    stale = commit(store, 'master', control('impl', 'AGENT-POLICY impl add\nWHEN task_done THEN SEND TO leader'),
                   snapshot=old_master)
    assert [x.code for x in stale.rejections] == ['stale_version']
    wrong_base = commit(store, 'master', control('impl', 'AGENT-POLICY impl add base=v0\nWHEN task_done THEN SEND TO leader'))
    assert [x.code for x in wrong_base.rejections] == ['stale_version']
    ok = commit(store, 'master', control('impl', 'AGENT-POLICY impl add base=v1\nWHEN task_done THEN SEND TO leader'))
    assert not ok.rejections and policy(store, 'impl').version == 2
    again = commit(store, 'impl', control('leader', 'POLICY-PROPOSAL self remove R1\nREASON: r'), snapshot=worker_view)
    assert [x.code for x in again.rejections] == ['stale_version']
    multi_base = commit(store, 'master', control('*', 'AGENT-POLICY impl,impl2 add base=v2\nWHEN start THEN SEND TO leader'))
    assert [x.code for x in multi_base.rejections] == ['base_requires_one_target']


def test_rule_ids_are_stable_and_never_reused():
    store, _ = make()
    spawn(store, 'impl', 'POLICY:\nWHEN start THEN SEND TO leader\nWHEN task_done THEN SEND TO leader\nEND-POLICY')
    assert not commit(store, 'master', control('impl', 'AGENT-POLICY impl remove R1')).rejections
    missing = commit(store, 'master', control('impl', 'AGENT-POLICY impl replace R1\nWHEN start THEN SEND TO leader'))
    assert [x.code for x in missing.rejections] == ['unknown_rule']
    assert not commit(store, 'master', control('impl', 'AGENT-POLICY impl add\nWHEN commit THEN SEND TO leader')).rejections
    assert [rule_id for rule_id, _ in policy(store, 'impl').rules] == ['R2', 'R3']


def test_revoke_reverses_self_change_and_refuses_after_later_edit():
    store, _ = make()
    spawn(store, 'impl', 'POLICY:\nWHEN start THEN SEND TO leader\nEND-POLICY')
    assert not commit(store, 'impl', control('leader', 'POLICY-PROPOSAL self remove R1\nREASON: noisy')).rejections
    assert 'Recent changes' in __import__('agent_company.policy.agent_rules_runtime', fromlist=['x']).render_tail(
        store.snapshot('master'), remaining_seconds=10)
    assert not commit(store, 'master', control('impl', 'AGENT-POLICY impl revoke #1')).rejections
    assert policy(store, 'impl').rules == (('R1', 'WHEN start THEN SEND TO leader'),)
    twice = commit(store, 'master', control('impl', 'AGENT-POLICY impl revoke #1'))
    assert [x.code for x in twice.rejections] == ['revoke_not_applicable']
    assert not commit(store, 'impl', control('leader', 'POLICY-PROPOSAL self notes\nshort notes\nREASON: r')).rejections
    assert not commit(store, 'master', control('impl', 'AGENT-POLICY impl notes\nmaster notes')).rejections
    conflict = commit(store, 'master', control('impl', 'AGENT-POLICY impl revoke #2'))
    assert [x.code for x in conflict.rejections] == ['revoke_conflict'] and policy(store, 'impl').notes == 'master notes'


def test_cross_proposals_wait_and_decisions_apply_or_refuse():
    store, _ = make()
    spawn(store, 'integ', 'POLICY:\nWHEN task_done THEN SEND TO leader\nEND-POLICY')
    spawn(store, 'impl')
    pending = commit(store, 'impl', control('leader', 'POLICY-PROPOSAL integ add\nWHEN commit THEN SEND TO impl WITH sha\nREASON: I need SHAs'))
    assert 'waiting' in pending.receipts['c1'] and policy(store, 'integ').version == 0
    wrong_to = commit(store, 'master', control('integ', 'POLICY-DECISION #1 approve'))
    assert [x.code for x in wrong_to.rejections] == ['wrong_recipient']
    approved = commit(store, 'master', control('impl', 'POLICY-DECISION #1 approve\nNOTE: good idea'))
    assert not approved.rejections and policy(store, 'integ').rules[-1][1] == 'WHEN commit THEN SEND TO impl WITH sha'
    assert policy(store, 'integ').source == 'approve'
    commit(store, 'impl', control('leader', 'POLICY-PROPOSAL integ replace R1\nWHEN task_done THEN SEND TO *\nREASON: r'))
    commit(store, 'master', control('integ', 'AGENT-POLICY integ replace R1\nWHEN task_done THEN SEND TO impl'))
    refused = commit(store, 'master', control('impl', 'POLICY-DECISION #2 approve'))
    assert [x.code for x in refused.rejections] == ['proposal_target_changed']
    amended = commit(store, 'master', control('impl', 'POLICY-DECISION #2 amend\nWHEN task_done THEN SEND TO leader\nNOTE: keep leader'))
    assert not amended.rejections and dict(policy(store, 'integ').rules)['R1'] == 'WHEN task_done THEN SEND TO leader'
    state = store.export()['proposals'][2]
    assert state['status'] == 'amended' and state['body'] == 'WHEN task_done THEN SEND TO *'


def test_acl_and_recipients():
    store, _ = make()
    spawn(store, 'impl')
    worker_edit = commit(store, 'impl', control('impl', 'AGENT-POLICY impl add\nWHEN start THEN SEND TO leader'))
    master_prop = commit(store, 'master', control('leader', 'POLICY-PROPOSAL self add\nWHEN start THEN SEND TO leader\nREASON: r'))
    for_master = commit(store, 'impl', control('leader', 'POLICY-PROPOSAL leader add\nWHEN start THEN SEND TO leader\nREASON: r'))
    not_leader = commit(store, 'impl', control('impl', 'POLICY-PROPOSAL self add\nWHEN start THEN SEND TO leader\nREASON: r'))
    decision = commit(store, 'impl', control('impl', 'POLICY-DECISION #1 approve'))
    assert [x.rejections[0].code for x in (worker_edit, master_prop, for_master, not_leader, decision)] == [
        'not_authorized', 'not_authorized', 'not_authorized', 'wrong_recipient', 'not_authorized']


def test_lessons_and_lifecycle_voiding():
    store, events = make()
    spawn(store, 'integ')
    spawn(store, 'impl')
    lesson = commit(store, 'impl', control('leader', 'LESSON: Commit before reporting a SHA to integ.'))
    assert lesson.receipts['c1'] == 'Lesson L1 recorded.'
    assert store.export()['lessons'][0]['related']['agents'] == ['integ']
    commit(store, 'impl', control('leader', 'POLICY-PROPOSAL integ add\nWHEN start THEN SEND TO leader\nREASON: r'))
    commit(store, 'integ', control('leader', 'POLICY-PROPOSAL impl add\nWHEN start THEN SEND TO leader\nREASON: r'))
    store.record_terminal('impl')
    statuses = {p['proposal_id']: (p['status'], p['void_reason']) for p in store.export()['proposals'].values()}
    assert statuses == {1: ('voided', 'proposer_terminal'), 2: ('voided', 'target_terminal')}
    ended = commit(store, 'master', control('impl', 'AGENT-POLICY impl add\nWHEN start THEN SEND TO leader'))
    assert [x.code for x in ended.rejections] == ['unknown_target']
    assert store.end_run() == 0
    closed = commit(store, 'master', control('integ', 'AGENT-POLICY integ add\nWHEN start THEN SEND TO leader'))
    assert [x.code for x in closed.rejections] == ['run_ended']
    replayed = AgentPolicyStore.replay(events, lambda record: None)
    assert replayed.export() == store.export()


def test_spawn_failure_voids_initial_policy_and_name_is_reusable():
    store, _ = make()
    spawn(store, 'impl', 'POLICY:\nWHEN start THEN SEND TO leader\nEND-POLICY', start=False)
    assert store.record_spawn_result('impl', ok=False) == 'voided'
    assert store.snapshot('master').worker('impl').lifecycle == 'voided'
    spawn(store, 'impl')


def test_persist_failure_publishes_nothing_and_replay_rejects_gaps():
    store, events = make()
    spawn(store, 'impl')
    before = store.export()
    store._writer = lambda record: (_ for _ in ()).throw(OSError('disk'))
    with pytest.raises(PolicyPersistError):
        commit(store, 'master', control('impl', 'AGENT-POLICY impl add\nWHEN start THEN SEND TO leader'))
    assert store.export() == before
    with pytest.raises(ReplayError):
        AgentPolicyStore.replay(events[:1] + events[2:], lambda record: None)


def test_same_response_changes_count_as_seen_but_explicit_base_is_checked():
    store, _ = make()
    spawn(store, 'impl')
    two = commit(store, 'master', control('impl', 'AGENT-POLICY impl add\nWHEN start THEN SEND TO leader', 'c1'),
                 control('impl', 'AGENT-POLICY impl notes\nOwn src/ only.', 'c2'))
    assert not two.rejections and policy(store, 'impl').version == 2
    mine = commit(store, 'impl', control('leader', 'POLICY-PROPOSAL self add\nWHEN task_done THEN SEND TO leader\nREASON: r', 'p1'),
                  control('leader', 'POLICY-PROPOSAL self remove R1\nREASON: r', 'p2'))
    assert not mine.rejections and policy(store, 'impl').version == 4
    spawned = commit(store, 'master', r.SpawnCall('s-e', 'e', r.parse_spawn_policy('Task.')),
                     control('e', 'AGENT-POLICY e add base=v7\nWHEN start THEN SEND TO leader', 'c3'))
    assert [x.code for x in spawned.rejections] == ['stale_version']


def test_star_proposal_applies_only_to_its_submission_set():
    store, _ = make()
    spawn(store, 'a')
    spawn(store, 'b')
    commit(store, 'a', control('leader', 'POLICY-PROPOSAL * add\nWHEN task_done THEN SEND TO leader\nREASON: r'))
    spawn(store, 'c')
    decided = commit(store, 'master', control('a', 'POLICY-DECISION #1 approve'))
    assert 'a: applied' in decided.receipts['c1'] and 'b: applied' in decided.receipts['c1']
    assert 'c:' not in decided.receipts['c1'] and policy(store, 'c').rules == ()


def test_pending_proposals_replace_only_the_same_kind():
    store, _ = make()
    spawn(store, 'a')
    spawn(store, 'b')
    commit(store, 'a', control('leader', 'POLICY-PROPOSAL b add\nWHEN start THEN SEND TO leader\nREASON: r'))
    commit(store, 'a', control('leader', 'POLICY-PROPOSAL b notes\nScope.\nREASON: r'))
    commit(store, 'a', control('leader', 'POLICY-PROPOSAL b add\nWHEN task_done THEN SEND TO leader\nREASON: r'))
    commit(store, 'a', control('leader', 'POLICY-PROPOSAL b notes\nNarrower scope.\nREASON: r'))
    statuses = {p['proposal_id']: p['status'] for p in store.export()['proposals'].values()}
    assert statuses == {1: 'pending', 2: 'voided', 3: 'pending', 4: 'pending'}


def test_inherited_rules_get_the_roster_check():
    store, _ = make()
    spawn(store, 'b')
    spawn(store, 'new')
    spawn(store, 'old', 'POLICY:\nWHEN commit THEN SEND TO b\nWHEN task_done THEN SEND TO new\nWHEN start THEN SEND TO leader\nEND-POLICY')
    store.record_terminal('old')
    store.record_terminal('b')
    store.record_terminal('new')
    result = spawn(store, 'new2', 'SUCCEEDS: old\nTask.')
    kept = dict(policy(store, 'new2').rules)
    assert kept == {'R3': 'WHEN start THEN SEND TO leader'}
    assert result.receipts['s-new2'].count('Skipped rule: R') == 2


def test_inherited_rule_naming_the_successor_is_skipped():
    store, _ = make()
    spawn(store, 'integ')
    spawn(store, 'old', 'POLICY:\nWHEN commit THEN SEND TO integ\nEND-POLICY')
    store.record_terminal('old')
    store.record_terminal('integ')
    commit(store, 'master', r.SpawnCall('s-x', 'integ', r.parse_spawn_policy('SUCCEEDS: old\nTask.')))
    assert policy(store, 'integ').rules == ()


def test_reject_all_invalid_rules_and_multiple_successors():
    store, _ = make()
    spawn(store, 'impl')
    commit(store, 'impl', control('leader', 'POLICY-PROPOSAL * add\nWHEN start THEN SEND TO leader\nREASON: r'))
    rejected = commit(store, 'master', control('impl', 'POLICY-DECISION #1 reject\nNOTE: not needed'))
    assert rejected.receipts['c1'] == 'Decision on #1: rejected'
    assert store.export()['proposals'][1]['status'] == 'rejected' and store.export()['proposals'][1]['note'] == 'not needed'
    bad = spawn(store, 'x', 'POLICY:\nWHEN nothing THEN SEND TO y\nWHEN commit THEN WAIT\nEND-POLICY')
    planned = next(c for c in bad.changes if c['event'] == 'agent_policy_planned')
    assert policy(store, 'x').rules == () and planned['policy_missing'] is False and len(planned['skipped']) == 2
    store.record_terminal('impl')
    spawn(store, 's1', 'SUCCEEDS: impl\nPart one.')
    spawn(store, 's2', 'SUCCEEDS: impl\nPart two.')
    assert policy(store, 's1').lineage == policy(store, 's2').lineage == {'from': 'impl', 'version': 0}


def test_every_version_records_its_diff_for_the_master():
    from agent_company.policy import agent_rules_runtime as runtime
    store, _ = make()
    spawn(store, 'impl', 'POLICY:\nWHEN start THEN SEND TO leader\nWHEN task_done THEN SEND TO leader\nEND-POLICY')
    commit(store, 'impl', control('leader', 'POLICY-PROPOSAL self remove R2\nREASON: noisy'))
    commit(store, 'master', control('impl', 'AGENT-POLICY impl replace R1\nWHEN start THEN SEND TO *'))
    tail = runtime.render_tail(store.snapshot('master'), remaining_seconds=5)
    assert '#1 impl: remove R2 -> version 1; removed R2: WHEN task_done THEN SEND TO leader' in tail
    document = runtime.render_policies_file(store.export())
    assert 'removed R2: WHEN task_done THEN SEND TO leader' in document
    assert 'replaced R1: WHEN start THEN SEND TO leader -> WHEN start THEN SEND TO *' in document


def test_an_old_star_proposal_does_not_touch_a_rule_created_after_it():
    # Review 10-18 F2: a teammate without R1 at proposal time had no base, so approval removed its later R1.
    store, _ = make()
    spawn(store, 'a', 'POLICY:\nWHEN start THEN SEND TO leader\nEND-POLICY')
    spawn(store, 'alpha')
    assert not commit(store, 'a', control('leader', 'POLICY-PROPOSAL * remove R1\nREASON: noisy')).rejections
    assert not commit(store, 'alpha', control('leader', 'POLICY-PROPOSAL self add\nWHEN commit THEN SEND TO a WITH sha\nREASON: r')).rejections
    decided = commit(store, 'master', control('a', 'POLICY-DECISION #1 approve'))
    assert 'a: applied' in decided.receipts['c1'] and 'alpha: not applied (proposal_target_changed' in decided.receipts['c1']
    assert policy(store, 'alpha').rules == (('R1', 'WHEN commit THEN SEND TO a WITH sha'),) and policy(store, 'a').rules == ()


def test_approval_refuses_a_rule_changed_and_changed_back():
    store, _ = make()
    spawn(store, 'integ', 'POLICY:\nWHEN task_done THEN SEND TO leader\nEND-POLICY')
    spawn(store, 'impl')
    commit(store, 'impl', control('leader', 'POLICY-PROPOSAL integ remove R1\nREASON: r'))
    commit(store, 'master', control('integ', 'AGENT-POLICY integ replace R1\nWHEN task_done THEN SEND TO impl'))
    commit(store, 'master', control('integ', 'AGENT-POLICY integ replace R1\nWHEN task_done THEN SEND TO leader'))
    refused = commit(store, 'master', control('impl', 'POLICY-DECISION #1 approve'))
    assert [x.code for x in refused.rejections] == ['proposal_target_changed'] and len(policy(store, 'integ').rules) == 1


def test_one_ended_name_in_a_list_rejects_only_that_target():
    # Review 10-18 F9: per-target results (plan v4 5.2) also when a listed teammate has ended.
    store, _ = make()
    spawn(store, 'alpha')
    spawn(store, 'beta')
    store.record_terminal('beta')
    result = commit(store, 'master', control('alpha', 'AGENT-POLICY alpha,beta add\nWHEN start THEN SEND TO leader'))
    assert not result.rejections
    assert 'alpha: applied' in result.receipts['c1'] and 'beta: not applied (unknown_target' in result.receipts['c1']
    alone = commit(store, 'master', control('alpha', 'AGENT-POLICY beta add\nWHEN start THEN SEND TO leader'))
    assert [x.code for x in alone.rejections] == ['unknown_target']


def test_revoke_refuses_a_change_made_again_and_allows_undo_in_reverse_order():
    # Review 10-18 F10: "changed again" counts even when the text came back to the same value.
    store, _ = make()
    spawn(store, 'impl')
    commit(store, 'impl', control('leader', 'POLICY-PROPOSAL self notes\nA\nREASON: r'))
    commit(store, 'master', control('impl', 'AGENT-POLICY impl notes\nB'))
    commit(store, 'master', control('impl', 'AGENT-POLICY impl notes\nA'))
    conflict = commit(store, 'master', control('impl', 'AGENT-POLICY impl revoke #1'))
    assert [x.code for x in conflict.rejections] == ['revoke_conflict'] and policy(store, 'impl').notes == 'A'
    other, _ = make()
    spawn(other, 'impl')
    commit(other, 'impl', control('leader', 'POLICY-PROPOSAL self notes\nA\nREASON: r'))
    commit(other, 'impl', control('leader', 'POLICY-PROPOSAL self notes\nB\nREASON: r'))
    assert not commit(other, 'master', control('impl', 'AGENT-POLICY impl revoke #2')).rejections
    assert not commit(other, 'master', control('impl', 'AGENT-POLICY impl revoke #1')).rejections
    assert policy(other, 'impl').notes == ''


def test_amending_a_star_replace_skips_a_teammate_that_lacked_the_rule():
    store, _ = make()
    spawn(store, 'a', 'POLICY:\nWHEN start THEN SEND TO leader\nEND-POLICY')
    spawn(store, 'alpha')
    commit(store, 'a', control('leader', 'POLICY-PROPOSAL * replace R1\nWHEN start THEN SEND TO *\nREASON: r'))
    commit(store, 'alpha', control('leader', 'POLICY-PROPOSAL self add\nWHEN commit THEN SEND TO a WITH sha\nREASON: r'))
    amended = commit(store, 'master', control('a', 'POLICY-DECISION #1 amend\nWHEN task_done THEN SEND TO leader'))
    assert 'a: applied' in amended.receipts['c1'] and 'alpha: not applied (proposal_target_changed' in amended.receipts['c1']
    assert policy(store, 'alpha').rules == (('R1', 'WHEN commit THEN SEND TO a WITH sha'),)
