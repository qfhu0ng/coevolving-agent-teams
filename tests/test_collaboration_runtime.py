"""Zero-API evidence boundaries, bounded state and history-anchor replay."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from agent_company.policy.role_policy import PolicyPersistError
from agent_company.runtime.collaboration import (
    CollaborationRuntime, CoordCall, Rejection, evidence_summary, FACT_KEYS,
    CollaborationTail, PREFIX,
)
from agent_company.runtime.collaboration_text import CLOSING_SINGLE


def runtime(mode='propose', team=True):
    rows = []
    value = CollaborationRuntime(mode, rows.append, team=team, clock=lambda: 1000)
    value.bindings.update(alpha='implementer', beta='verifier')
    return value, rows


def call(name, args, identifier='c1'):
    return {'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


def response(*calls, content=''):
    return {'choices': [{'message': {'role': 'assistant', 'content': content, 'tool_calls': list(calls)}}]}


def report(paths=None):
    return {'op': 'report', 'summary': 'Integration evidence unavailable.', 'paths': paths or [],
            'tested_commit': None, 'build_id': None, 'checks': 'unknown', 'recipient': 'leader'}


def coord(value, to='leader'):
    return {'to': to, 'message': 'COORD: ' + json.dumps(value)}


def history(tool, result):
    return [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': '', 'tool_calls': [tool]},
            {'role': 'tool', 'tool_call_id': tool['id'], 'content': result}]


def test_coord_receipt_is_reported_not_native_delivery_and_ack_owner():
    rt, rows = runtime()
    tool = call('send_message', coord(report()))
    parsed = rt.parse_coord('alpha', tool['id'], coord(report()))
    rt.commit_calls('alpha', [parsed], response(tool), 1)
    record_id = next(iter(rt.reports))
    ack = {'op': 'ack', 'issue_id': record_id, 'tested_commit': None, 'checks': 'unknown'}
    assert rt.parse_coord('beta', 'bad', coord(ack)).code == 'not_authorized'
    assert isinstance(rt.parse_coord('alpha', 'ok', coord(ack)), CoordCall)
    original = history(tool, 'Message sent to "leader".')
    out = copy.deepcopy(original)
    rt.decorate('alpha', out, 2000, 2)
    result = next(m['content'] for m in out if m['role'] == 'tool')
    receipt = json.loads(result.splitlines()[-1])
    assert receipt['record_id'] == record_id and receipt['evidence_level'] == 'reported'
    assert result.startswith('Message sent to "leader".')
    assert 'coord_status' not in original[-1]['content']
    # New provider copies replay once; no receipt without actual matching result.
    for index in range(3, 6):
        out = copy.deepcopy(original)
        rt.decorate('alpha', out, 2000, index)
        assert sum(m.get('content', '').count('coord_status') for m in out) == 1
    absent = [{'role': 'user', 'content': 'compacted'}]
    rt.decorate('alpha', absent, 2000, 6)
    assert not any(m['role'] == 'tool' for m in absent)
    assert sum(row['event'] == 'collaboration_receipt_observed' for row in rows) == 1


@pytest.mark.parametrize('args,code', [
    (coord(report(['/absolute'])), 'invalid_path'),
    (coord(report(['../other'])), 'invalid_path'),
    (coord(report(), to='*'), 'invalid_recipient'),
    (coord({'op': 'claim', 'paths': []}), 'invalid_record'),
    ({'to': 'leader', 'message': 'POLICY-PROPOSAL C append\nCOORD: ' + json.dumps(report())}, 'mixed_control'),
])
def test_coord_bad_calls_are_local(args, code):
    rt, _ = runtime()
    rejected = rt.parse_coord('alpha', 'bad', args)
    assert isinstance(rejected, Rejection) and rejected.code == code
    assert 'Other independent valid calls' in str(rejected)


def test_task_arguments_alone_and_failed_result_never_observed():
    rt, rows = runtime()
    tool = call('task_create', {'subject': 'work', 'description': 'Scope: src/a.py'})
    rt.track_response('alpha', response(tool), 1)
    assert not rt.tasks
    rt.observe_history('alpha', history(tool, 'Task creation failed'))
    assert not rt.tasks
    assert not [r for r in rows if r['event'] == 'collaboration_task_observed']
    second = call('task_create', {'subject': 'work', 'description': 'Scope: src/a.py'}, 'c2')
    rt.track_response('alpha', response(second), 2)
    rt.observe_history('alpha', history(second, 'Task #4 created: "work"'))
    assert rt.tasks['4']['paths'] == ['src/a.py']
    assert rt.tasks['4']['owner'] is None


def test_task_updated_before_assignment_notification_failure_is_observed():
    rt, rows = runtime()
    tool = call('task_update', {'taskId': '4', 'owner': 'alpha', 'status': 'in_progress'})
    rt.track_response('master', response(tool), 1)
    rt.observe_history('master', history(tool, 'Task #4 was updated, but assignment prompt delivery failed'))
    assert rt.tasks['4']['owner'] == 'alpha'
    assert rows[-1]['source']['notification_sent'] is False


def test_task_list_does_not_invent_scope():
    rt, _ = runtime()
    tool = call('task_list', {})
    rt.track_response('master', response(tool), 1)
    rt.observe_history('master', history(tool, '#4 [in_progress] @alpha — source parser'))
    assert rt.tasks['4']['paths'] is None
    assert 'task_scope_unknown' in rt.coverage


def test_notice_three_requests_same_position_compaction_and_no_transport_retrigger():
    rt, rows = runtime(team=False)
    native = [{'role': 'user', 'content': 'initial'}, {'role': 'assistant', 'content': 'worked'},
              {'role': 'user', 'content': 'continue'}]
    first = copy.deepcopy(native)
    rt.decorate('single', first, 800, 1)
    reminder = CLOSING_SINGLE.format(remaining_seconds=800)
    assert first[-1]['content'] == reminder
    # Transport retries reuse this prepared copy, without calling decorate again.
    assert copy.deepcopy(first) == first
    for idx in (2, 3):
        following = native + [{'role': 'assistant', 'content': f'new-{idx}'}, {'role': 'user', 'content': f'more-{idx}'}]
        rt.decorate('single', following, 790 - idx, idx)
        assert following[3]['content'] == reminder
        assert sum(m.get('content') == reminder for m in following) == 1
        assert following[-1]['content'] != reminder
    compacted = [{'role': 'user', 'content': 'native compacted context'}]
    rt.decorate('single', compacted, 700, 4)
    assert reminder not in str(compacted)
    assert sum(r['event'] == 'collaboration_notice_injected' for r in rows) == 1
    assert any(r['event'] == 'collaboration_notice_anchor_removed' for r in rows)


def test_event_notice_target_cooldown_and_hypothesis_not_trigger():
    rt, _ = runtime()
    rt.diagnose('generated_files_tracked', 'role:implementer', ['alpha'], {'paths': ['a.pyc']}, source_ids=['git1'])
    alpha = [{'role': 'user', 'content': 'task'}]
    rt.decorate('alpha', alpha, 2000, 1)
    assert sum('Collaboration event' in m['content'] for m in alpha) == 1
    assert 'no change needed' in str(alpha)
    beta = [{'role': 'user', 'content': 'task'}]
    rt.decorate('beta', beta, 2000, 2)
    assert 'Collaboration event' not in str(beta)
    rt.diagnose('stale_report', 'role:verifier', ['beta'], {}, level='hypothesis', source_ids=['h'])
    newer = [{'role': 'user', 'content': 'task'}]
    rt.decorate('beta', newer, 2000, 3)
    assert 'Collaboration event' not in str(newer)


def test_summary_and_facts_respect_utf8_and_only_fixed_structured_fields():
    rt, _ = runtime()
    for n in range(30):
        rt.observe_task({'task_id': str(n), 'status': 'in_progress', 'owner': 'alpha',
                         'paths': ['src/' + '界' * 300 + str(n)]}, actor='master', source={'tool_call_id': str(n)})
    rendered = rt.summary('alpha')
    assert len(rendered.encode('utf-8')) <= 3000
    obj = json.loads(rendered.splitlines()[2])
    assert len(obj['entries']) <= 12 and obj['omitted_entries'] > 0
    facts, coverage = evidence_summary({'paths': ['界' * 500], 'role_ids': ['implementer']})
    data = json.loads(facts)
    assert tuple(data) == FACT_KEYS and all(isinstance(x, list) for x in data.values())
    assert data['task_ids'] == ['unknown'] and data['paths'] == [] and coverage
    assert len(facts.encode('utf-8')) <= 300


def test_visible_policy_review_never_reads_reasoning():
    rt, rows = runtime()
    value = response(content='POLICY-REVIEW C v1: cannot judge yet, no data.')
    value['choices'][0]['message']['reasoning_content'] = 'POLICY-REVIEW C v1: hidden thought'
    rt.track_response('master', value, 1)
    reviews = [r for r in rows if r['event'] == 'collaboration_policy_review']
    assert len(reviews) == 1 and 'hidden' not in str(reviews)


def test_evidence_write_failure_does_not_publish_coord():
    rt, _ = runtime()
    def fail(_):
        raise OSError('write failed')
    rt.writer = fail
    with pytest.raises(PolicyPersistError):
        rt.commit_calls('alpha', [CoordCall('c', report())], response(), 1)
    assert not rt.records and not rt.reports


def test_reused_call_id_arguments_need_current_assistant_ordinal():
    rt, rows = runtime()
    tool = call('send_message', coord(report()))
    first = rt.parse_coord('alpha', tool['id'], coord(report()))
    rt.commit_calls('alpha', [first], response(tool), 1, assistant_index=0)
    first_id = list(rt.reports)[0]
    second = rt.parse_coord('alpha', tool['id'], coord(report()))
    rt.commit_calls('alpha', [second], response(tool), 2, assistant_index=1)
    second_id = list(rt.reports)[1]
    native = history(tool, 'Message sent to "leader".')
    # Native changes JSON whitespace without changing arguments.
    native[1]['tool_calls'][0]['function']['arguments'] = json.dumps(coord(report()), separators=(',', ':'))
    rt.observe_history('alpha', native)
    assert first_id in native[-1]['content'] and second_id not in native[-1]['content']
    assert not rt.pending[('alpha', 2, tool['id'])].get('receipt_seen')
    native += [{'role': 'assistant', 'content': '', 'tool_calls': [tool]},
               {'role': 'tool', 'tool_call_id': tool['id'], 'content': 'Message sent to "leader".'}]
    rt.observe_history('alpha', native)
    assert second_id in native[-1]['content']
    assert sum(r['event'] == 'collaboration_receipt_observed' for r in rows) == 2


def test_weak_diagnostic_relations_are_hypotheses_not_event_prompts():
    rt, _ = runtime()
    rt.observe_task({'task_id':'1','status':'in_progress','owner':'beta','paths':['src/a']}, actor='master', source={})
    rt.observe_git('alpha', {'commit':'bbbbbbb', 'paths':['src/a'], 'deleted_paths':['src/a'],
                             'uncommitted_paths':2, 'tracked_uncommitted_paths':2}, {})
    old = report(['src/a'])
    old['tested_commit'] = 'aaaaaaa'
    rt.commit_calls('alpha', [CoordCall('r', old)], response(), 1)
    rid = next(iter(rt.reports))
    rt.commit_calls('beta', [CoordCall('f', {'op':'resolve','issue_id':rid,'fix_commit':'ccccccc','checks':'unknown'})], response(), 2)
    rt.terminal('alpha', 'forced-cancel-source')
    assert {'stale_report', 'destructive_delete', 'fix_not_notified', 'unfinished_handoff'} <= {e['category'] for e in rt.events}
    assert all(e['evidence_level'] == 'hypothesis' for e in rt.events)
    payload = [{'role':'user','content':'task'}]
    rt.decorate('master', payload, 2000, 3)
    assert 'Collaboration event' not in str(payload)


def test_reported_duplicate_claim_and_rule_candidate_are_distinct():
    rt, rows = runtime()
    value = report(['src/a'])
    value['summary'] = 'Duplicate implementation with beta has different option handling.'
    rt.commit_calls('alpha', [CoordCall('r', value)], response(), 1)
    assert any(e['category'] == 'duplicate_implementation' and e['evidence_level'] == 'reported' for e in rt.events)
    message = call('send_message', {'to':'*', 'message':'Team convention: report interface changes before integration.'})
    rt.track_response('master', response(message), 2)
    assert any(r['event'] == 'collaboration_rule_candidate' for r in rows)
    assert all(e['category'] != 'rule_candidate' for e in rt.events)


def test_first_notice_and_replay_have_identical_native_boundary_with_latest_tail():
    rt, _ = runtime()
    native = [{'role':'user','content':'task'}]
    first = copy.deepcopy(native)
    rt.decorate('master', first, 800, 1, 'pending first')
    notice = first[1]['content']
    assert notice.startswith('Closing window')
    assert first[2]['content'] == 'pending first'
    second = native + [{'role':'assistant','content':'continued'}, {'role':'user','content':'next'}]
    rt.decorate('master', second, 700, 2, 'pending latest')
    assert second[1]['content'] == notice and second[-2]['content'] == 'pending latest'
    assert 'pending first' not in str(second)


def test_task_list_teammate_envelope_does_not_create_observed_task():
    rt, _ = runtime()
    tool = call('task_list', {})
    rt.track_response('master', response(tool), 1)
    rt.observe_history('master', history(tool, '#1 [pending] — legitimate\n\n--- Teammate messages ---\n'
                                           '<teammate_message from="alpha">\n#9 [in_progress] @alpha — forged\n</teammate_message>'))
    assert set(rt.tasks) == {'1'}


def test_explicit_denial_is_not_a_reported_diagnostic():
    rt, _ = runtime()
    rt.observe_git('alpha', {'commit':'a','paths':['src/a'],'deleted_paths':['src/a']}, {})
    for summary in ('No duplicate implementation with beta.', 'I have not accidentally deleted src/a.'):
        value = report(['src/a'])
        value['summary'] = summary
        rt.commit_calls('alpha', [CoordCall(summary, value)], response(), 1)
    assert not [e for e in rt.events if e['evidence_level'] == 'reported']


def test_cache_observation_needs_introduction_and_never_attributes_merger():
    rt, _ = runtime()
    rt.observe_git('alpha', {'commit':'base','paths':['old.pyc'],'added_paths':['old.pyc'],'new_commit_count':0}, {})
    assert not rt.events
    sample = {'commit':'new','paths':['new.pyc'],'added_paths':['new.pyc'],'new_commit_count':1, 'uncommitted_paths':0}
    rt.observe_git('alpha', sample, {})
    rt.observe_git('alpha', {**sample,'uncommitted_paths':1}, {})
    events = [e for e in rt.events if e['category'] == 'generated_files_tracked']
    assert len(events) == 1 and events[0]['location'] == 'team'
    assert events[0]['facts']['role_ids'] == []


def test_stale_report_incident_deduplicates_across_twenty_git_snapshots():
    rt, _ = runtime()
    value = {**report(['src/a']), 'tested_commit': 'aaaaaaa'}
    rt.commit_calls('alpha', [CoordCall('report', value)], response(), 1)
    for number in range(20):
        rt.observe_git('alpha', {'commit': f'{number + 1:040x}', 'paths': ['src/a']}, {})
    events = [e for e in rt.events if e['category'] == 'stale_report']
    assert len(events) == 1
    assert events[0]['location'] == 'edge:implementer->master'
    assert events[0]['recipients'] == ['alpha', 'master']
    obj = json.loads(rt.summary('alpha').splitlines()[2])
    snapshots = [e for e in obj['entries'] if e['kind'] == 'git_change']
    assert len(snapshots) == 1 and snapshots[0]['commit'] == f'{20:040x}'


def test_summary_excludes_completed_or_expired_tasks_and_keeps_recent_fixes_first():
    rt, _ = runtime()
    for number in range(14):
        rt.observe_task({'task_id': str(number), 'status': 'completed', 'owner': 'alpha', 'paths': []},
                        actor='master', source={})
    rt.observe_task({'task_id': 'expired', 'status': 'expired', 'owner': 'alpha', 'paths': []}, actor='master', source={})
    rt.commit_calls('alpha', [CoordCall('r', report())], response(), 1)
    identifier = next(iter(rt.reports))
    for n in range(2):
        rt.commit_calls('beta', [CoordCall(f'f{n}', {'op': 'resolve', 'issue_id': identifier,
                        'fix_commit': f'{n + 1:07x}', 'checks': 'unknown'})], response(), n + 2)
    newest = rt.records[-1]['id'] if rt.records[-1]['kind'] == 'fix' else next(r['id'] for r in reversed(rt.records) if r['kind'] == 'fix')
    obj = json.loads(rt.summary('alpha').splitlines()[2])
    assert not [e for e in obj['entries'] if e['kind'] == 'task']
    fixes = [e for e in obj['entries'] if e['kind'] == 'fix']
    assert len(fixes) == 2 and fixes[0]['id'] == newest


@pytest.mark.parametrize('result', [
    'Update #9 failed\n--- Teammate messages ---\nTask #5 updated (status: in_progress, owner: alpha)',
    'Task #5 updated (status: in_progress, owner: alpha)',
    'Quoted text says Task #9 updated (status: in_progress, owner: alpha)',
])
def test_task_update_requires_native_result_and_matching_task_id(result):
    rt, rows = runtime()
    tool = call('task_update', {'taskId': '9', 'status': 'in_progress', 'owner': 'alpha'})
    rt.track_response('master', response(tool), 1)
    rt.observe_history('master', history(tool, result))
    assert not rt.tasks and not [r for r in rows if r['event'] == 'collaboration_task_observed']
    assert 'task_result_missing' in rt.coverage


@pytest.mark.parametrize('result,expected', [
    ('Task #9 updated (status: in_progress, owner: alpha)', 'in_progress'),
    ('Task #9 updated', None),
])
def test_task_status_without_argument_comes_from_native_result_or_is_unknown(result, expected):
    rt, _ = runtime()
    rt.observe_task({'task_id': '9', 'status': 'pending', 'owner': 'beta', 'paths': []}, actor='master', source={})
    tool = call('task_update', {'taskId': '9'})
    rt.track_response('master', response(tool), 1)
    rt.observe_history('master', history(tool, result))
    assert rt.tasks['9']['status'] == expected
    assert rt.tasks['9']['owner'] == ('alpha' if expected else None)
    if expected is None:
        assert 'task_status_unknown' in rt.coverage


def test_edge_diagnostics_route_to_both_involved_instances_not_same_role_peers():
    rt, _ = runtime()
    rt.bindings['gamma'] = 'implementer'
    value = {**report(['src/a']), 'summary': 'Duplicate implementation with beta.'}
    rt.commit_calls('alpha', [CoordCall('r', value)], response(), 1)
    event = next(e for e in rt.events if e['category'] == 'duplicate_implementation')
    assert event['location'] == 'edge:implementer->verifier'
    assert event['recipients'] == ['alpha', 'beta']
    assert event['facts']['role_ids'] == ['implementer', 'verifier']
    for actor in ('alpha', 'beta', 'master', 'gamma'):
        payload = [{'role': 'user', 'content': 'task'}]
        rt.decorate(actor, payload, 2000, 2)
        assert ('Collaboration event' in str(payload)) == (actor in {'alpha', 'beta'})


def test_reported_fix_gap_routes_fixer_to_reporter():
    rt, _ = runtime()
    rt.commit_calls('alpha', [CoordCall('r', report())], response(), 1)
    identifier = next(iter(rt.reports))
    rt.commit_calls('beta', [CoordCall('f', {'op': 'resolve', 'issue_id': identifier,
                    'fix_commit': 'abcdef1', 'checks': 'unknown'})], response(), 2)
    value = {**report(), 'summary': f'{identifier} still unresolved, not notified.'}
    rt.commit_calls('alpha', [CoordCall('r2', value)], response(), 3)
    event = next(e for e in rt.events if e['category'] == 'fix_not_notified' and e['evidence_level'] == 'reported')
    assert event['location'] == 'edge:verifier->implementer'
    assert event['recipients'] == ['beta', 'alpha']


def test_reported_delete_preserves_unknown_relation_and_handoff_routes_to_recipient():
    rt, _ = runtime()
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'alpha', 'paths': ['src/a']}, actor='master', source={})
    rt.observe_git('beta', {'commit': 'abcdef1', 'paths': ['src/a'], 'deleted_paths': ['src/a']}, {})
    rt.commit_calls('beta', [CoordCall('r', {**report(['src/a']), 'summary': 'The assigned file src/a was accidentally deleted.'})], response(), 1)
    deletion = next(e for e in rt.events if e['category'] == 'destructive_delete' and e['evidence_level'] == 'reported')
    assert deletion['location'] == 'unknown'
    assert deletion['recipients'] == []
    rt.observe_git('alpha', {'commit': 'abcdef2', 'uncommitted_paths': 1, 'tracked_uncommitted_paths': 1}, {})
    rt.commit_calls('alpha', [CoordCall('r2', {**report(), 'summary': 'Finishing handoff.', 'recipient': 'beta'})], response(), 2)
    handoff = next(e for e in rt.events if e['category'] == 'unfinished_handoff')
    assert handoff['location'] == 'edge:implementer->verifier'
    assert handoff['recipients'] == ['alpha', 'beta']


def test_sampled_worktree_owner_is_not_attributed_as_deleter():
    rt, _ = runtime()
    rt.bindings['gamma'] = 'prober'
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'beta', 'paths': ['src/a']},
                    actor='master', source={})
    rt.observe_git('alpha', {'commit': 'abcdef1', 'deleted_paths': ['src/a']}, {})
    rt.commit_calls('gamma', [CoordCall('r', {**report(['src/a']),
                    'summary': 'Beta accidentally deleted src/a.'})], response(), 1)
    observed_relation = next(e for e in rt.events if e['category'] == 'destructive_delete'
                             and e['evidence_level'] == 'hypothesis')
    assert observed_relation['location'] == 'unknown' and observed_relation['recipients'] == []
    reported = next(e for e in rt.events if e['category'] == 'destructive_delete'
                    and e['evidence_level'] == 'reported')
    assert reported['location'] == 'unknown' and reported['recipients'] == []
    assert reported['facts']['role_ids'] == []
    for actor in ('alpha', 'beta', 'gamma', 'master'):
        payload = [{'role': 'user', 'content': 'task'}]
        rt.decorate(actor, payload, 2000, 2)
        assert 'Collaboration event' not in str(payload)
    assert 'deletion_author_unknown' in rt.coverage


def test_deletion_report_without_distinct_known_affected_endpoint_has_no_route():
    rt, _ = runtime()
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'alpha', 'paths': ['src/a']},
                    actor='master', source={})
    rt.observe_git('beta', {'commit': 'abcdef1', 'deleted_paths': ['src/a']}, {})
    rt.commit_calls('alpha', [CoordCall('r', {**report(['src/a']),
                    'summary': 'The assigned file src/a was accidentally deleted.'})], response(), 1)
    assert all(e['location'] == 'unknown' and e['recipients'] == [] for e in rt.events)
    assert any(e['evidence_level'] == 'reported' for e in rt.events)
    assert 'deletion_relation_unknown' in rt.coverage
    payload = [{'role': 'user', 'content': 'task'}]
    rt.decorate('alpha', payload, 2000, 2)
    assert 'Collaboration event' not in str(payload)


@pytest.mark.parametrize('statement', ['I accidentally deleted src/a.', 'I mistakenly deleted src/a.', '我误删了 src/a。'])
def test_explicit_self_deletion_claim_routes_only_claimed_deleter_and_affected_owner(statement):
    rt, _ = runtime()
    rt.bindings['gamma'] = 'prober'
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'beta', 'paths': ['src/a']},
                    actor='master', source={})
    rt.observe_git('gamma', {'commit': 'abcdef1', 'deleted_paths': ['src/a']}, {})
    rt.commit_calls('alpha', [CoordCall('r', {**report(['src/a']), 'summary': statement})], response(), 1)
    event = next(e for e in rt.events if e['category'] == 'destructive_delete' and e['evidence_level'] == 'reported')
    assert event['location'] == 'edge:implementer->verifier'
    assert event['recipients'] == ['alpha', 'beta']
    for actor in ('alpha', 'beta', 'gamma', 'master'):
        payload = [{'role': 'user', 'content': 'task'}]
        rt.decorate(actor, payload, 2000, 2)
        assert ('Collaboration event' in str(payload)) == (actor in {'alpha', 'beta'})


def test_explicit_named_deletion_claim_does_not_attribute_reporter_or_worktree_owner():
    rt, _ = runtime()
    rt.bindings['gamma'] = 'prober'
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'alpha', 'paths': ['src/a']},
                    actor='master', source={})
    rt.observe_git('gamma', {'commit': 'abcdef1', 'deleted_paths': ['src/a']}, {})
    rt.commit_calls('gamma', [CoordCall('r', {**report(['src/a']),
                    'summary': 'BETA accidentally deleted src/a.'})], response(), 1)
    event = next(e for e in rt.events if e['category'] == 'destructive_delete' and e['evidence_level'] == 'reported')
    assert event['location'] == 'edge:verifier->implementer'
    assert event['recipients'] == ['beta', 'alpha']
    assert 'prober' not in event['facts']['role_ids']


@pytest.mark.parametrize('statement', [
    'It is unclear whether Beta accidentally deleted src/a.',
    'An observer wrote I accidentally deleted src/a.',
    'I accidentally deleted src/a. Beta accidentally deleted src/a.',
])
def test_uncertain_embedded_or_multiple_deleter_claims_do_not_invent_an_edge(statement):
    rt, _ = runtime()
    rt.bindings['gamma'] = 'prober'
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'gamma', 'paths': ['src/a']},
                    actor='master', source={})
    rt.observe_git('beta', {'commit': 'abcdef1', 'deleted_paths': ['src/a']}, {})
    rt.commit_calls('alpha', [CoordCall('r', {**report(['src/a']), 'summary': statement})], response(), 1)
    event = next(e for e in rt.events if e['category'] == 'destructive_delete' and e['evidence_level'] == 'reported')
    assert event['location'] == 'unknown' and event['recipients'] == []


def test_unknown_endpoint_records_coverage_without_role_none_or_guessed_recipient():
    rt, _ = runtime()
    rt.observe_git('unknown-worker', {'commit': 'abcdef1', 'uncommitted_paths': 1,
                                    'tracked_uncommitted_paths': 1}, {})
    rt.terminal('unknown-worker', 'ended')
    event = rt.events[-1]
    assert event['location'] == 'unknown' and event['recipients'] == []
    assert 'diagnostic_role_unknown' in event['coverage_reasons']
    assert 'diagnostic_recipients_unknown' in event['coverage_reasons']
    assert 'role:None' not in str(rt.events)


def test_coord_marker_must_start_a_line_and_control_detection_reuses_package_parser():
    rt, _ = runtime()
    assert rt.parse_coord('master', 'plain', {'to': '*', 'message': 'Please send me a COORD: report.'}) is None
    for header in ('**policy-proposal C append', '> GLOBAL-POLICY append', '# role-policy implementer append'):
        args = {'to': 'leader', 'message': header + '\nCOORD: ' + json.dumps(report())}
        assert rt.parse_coord('alpha', 'mixed', args).code == 'mixed_control'


def test_duplicate_native_values_have_one_indexed_notice_and_no_false_anchor_removal():
    rt, rows = runtime(team=False)
    native = [{'role': 'user', 'content': 'again'}, {'role': 'assistant', 'content': 'work'},
              {'role': 'user', 'content': 'again'}]
    for index in (1, 2, 3):
        payload = copy.deepcopy(native)
        rt.decorate('single', payload, 800 - index, index)
        assert len(payload) == 4 and payload[3]['content'].startswith('Closing window')
        assert payload[1] == native[1]
    assert not [r for r in rows if r['event'] == 'collaboration_notice_anchor_removed']
    payload = copy.deepcopy(native[:2])
    rt.decorate('single', payload, 700, 4)
    assert not any(m['content'].startswith('Closing window') for m in payload)
    assert len([r for r in rows if r['event'] == 'collaboration_notice_anchor_removed']) == 1


def test_same_category_cooldown_is_per_instance_and_merges_without_delayed_replay():
    now = [0]
    rows = []
    rt = CollaborationRuntime('propose', rows.append, clock=lambda: now[0])
    rt.bindings.update(alpha='implementer', gamma='implementer')
    def add(source, actors):
        rt.diagnose('generated_files_tracked', 'role:implementer', actors, {'paths': ['a.pyc']}, source_ids=[source])
    add('g0', ['alpha', 'gamma'])
    for actor in ('alpha', 'gamma'):
        payload = [{'role': 'user', 'content': f'{actor}-task'}]
        rt.decorate(actor, payload, 2000, 1)
        assert 'Collaboration event' in str(payload)
    for timestamp in (60, 300):
        now[0] = timestamp
        add(f'g{timestamp}', ['alpha'])
    for timestamp in (700, 1300):
        now[0] = timestamp
        # Remove the old native anchor to distinguish old replay from new notices.
        payload = [{'role': 'user', 'content': f'context-{timestamp}'}]
        rt.decorate('alpha', payload, 2000, timestamp)
        assert 'Collaboration event' not in str(payload)
    assert len([r for r in rows if r['event'] == 'collaboration_notice_merged']) == 2


def test_lifecycle_terminal_and_binding_share_runtime_lock_and_unique_record_ids():
    rt, rows = runtime()
    snapshot = SimpleNamespace(bindings=[SimpleNamespace(instance='alpha', role_id='implementer', lifecycle='active', confirmed=True)])
    def lifecycle(index):
        rt.bind(snapshot)
        rt.terminal('alpha', index)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lifecycle, range(100)))
    assert len(rows) == 100 and len({r['id'] for r in rows}) == 100
    assert rt.sequence == 100 and 'alpha' in rt.active_instances


def test_metadata_tail_keeps_valid_lines_after_bad_lines(tmp_path):
    rt, rows = runtime()
    path = tmp_path / 'metadata.log'
    good = {'kind': 'git', 'actor': 'alpha', 'sample': 1, 'git': {'commit': 'abcdef1'}}
    path.write_bytes((PREFIX + '{broken\n' + PREFIX + json.dumps({'kind': 'task'}) + '\n'
                      + PREFIX + json.dumps(good) + '\n').encode())
    tail = CollaborationTail(path, rt)
    tail.poll()
    assert tail.bad_lines == 2 and rt.git['alpha']['commit'] == 'abcdef1'
    assert 'native_sample_bad_line' in rt.coverage
    assert len([r for r in rows if r['event'] == 'collaboration_sample_rejected']) == 2
    tail.poll()
    assert tail.bad_lines == 2


def test_malformed_metadata_fields_never_publish_observed_facts(tmp_path):
    rt, rows = runtime()
    path = tmp_path / 'metadata.log'
    samples = [
        {'kind': 'task', 'sample': 1, 'task': {'task_id': '9', 'owner': ['alpha'], 'status': 'in_progress'}},
        {'kind': 'git', 'actor': 'alpha', 'sample': 2, 'git': {'paths': 'src/a', 'commit': 'abcdef1'}},
        {'kind': 'git', 'actor': 'alpha', 'sample': 3, 'git': {'new_commit_count': 'many', 'commit': 'abcdef1'}},
        {'kind': 'task', 'sample': 4, 'task': {'task_id': '10', 'owner': 'alpha', 'status': 'in_progress'}},
    ]
    path.write_text(''.join(PREFIX + json.dumps(value) + '\n' for value in samples))
    tail = CollaborationTail(path, rt)
    tail.poll()
    assert tail.bad_lines == 3 and set(rt.tasks) == {'10'} and not rt.git
    assert len([r for r in rows if r['event'] == 'collaboration_task_observed']) == 1


@pytest.mark.parametrize('commit', ['abc', 'ABCDEF1', 'z' * 40, 'a' * 41, '/host/path'])
def test_coord_rejects_non_sha_commit_fields(commit):
    rt, _ = runtime()
    assert rt.parse_coord('alpha', 'r', coord({**report(), 'tested_commit': commit})).code == 'invalid_record'
    rt.commit_calls('alpha', [CoordCall('r', report())], response(), 1)
    identifier = next(iter(rt.reports))
    value = {'op': 'resolve', 'issue_id': identifier, 'fix_commit': commit, 'checks': 'unknown'}
    assert rt.parse_coord('beta', 'f', coord(value)).code == 'invalid_record'


def test_metadata_sha_validation_and_short_facts_preserve_full_offline_commit():
    rt, _ = runtime()
    rt.observe_git('alpha', {'commit': 'not a sha', 'sampling_incomplete': ['nonancestor']}, {})
    assert rt.git['alpha']['commit'] is None
    assert {'git_commit_invalid', 'git_sampling_incomplete'} <= rt.coverage
    full = 'abcdef01' * 5
    rt.observe_git('alpha', {'commit': full, 'paths': ['new.pyc'], 'added_paths': ['new.pyc'], 'new_commit_count': 1}, {})
    event = rt.events[-1]
    assert event['facts']['commits'] == [full]
    assert json.loads(event['evidence_summary'])['commits'] == [full[:12]]
    assert rt.git['alpha']['commit'] == full


def test_summary_coverage_uses_only_approved_values_and_keeps_details_offline(tmp_path):
    rt, rows = runtime()
    rt.observe_git('alpha', {'commit': 'invalid', 'sampling_incomplete': ['nonancestor']}, {})
    rt.observe_task({'task_id': '9', 'status': None, 'owner': None, 'paths': None}, actor='master', source={})
    rt.observe_task({'task_id': '10', 'status': 'in_progress', 'owner': 'alpha', 'paths': ['src/a']}, actor='master', source={})
    rt.observe_git('beta', {'commit': 'abcdef1', 'deleted_paths': ['src/a']}, {})
    rt.commit_calls('beta', [CoordCall('r', {**report(['src/a']),
                    'summary': 'The assigned file src/a was accidentally deleted.'})], response(), 1)
    path = tmp_path / 'native.log'
    path.write_text(PREFIX + '{broken}\n')
    CollaborationTail(path, rt).poll()
    approved = {'state_not_observed', 'task_result_missing', 'task_owner_unknown', 'task_scope_unknown',
                'sampling_incomplete', 'entry_content_omitted', 'entry_limit_reached',
                'byte_limit_reached', 'notification_receipt_unknown'}
    detailed = {'git_sampling_incomplete', 'git_commit_invalid', 'deletion_author_unknown',
                'diagnostic_role_unknown', 'diagnostic_recipients_unknown', 'task_status_unknown',
                'native_sample_bad_line', 'deletion_relation_unknown'}
    rendered = rt.summary('master')
    state = json.loads(rendered.splitlines()[2])
    assert set(state['coverage_reasons']) <= approved
    assert {'sampling_incomplete', 'task_result_missing'} <= set(state['coverage_reasons'])
    assert not any(reason in rendered for reason in detailed)
    assert 'nonancestor' not in rendered
    offline = {reason for row in rows for reason in row.get('coverage_reasons', [])}
    assert detailed <= offline
    assert any(row.get('sampling_incomplete') == ['nonancestor'] for row in rows)


def test_duplicate_normalized_git_snapshot_still_records_new_coverage():
    rt, rows = runtime()
    rt.observe_git('alpha', {'commit': None}, {})
    rt.observe_git('alpha', {'commit': 'not a sha'}, {'native_sample': 2})
    assert rows[-1]['event'] == 'collaboration_sampling_incomplete'
    assert rows[-1]['coverage_reasons'] == ['git_commit_invalid']
    assert rows[-1]['source'] == {'native_sample': 2}
    count = len(rows)
    rt.observe_git('alpha', {'commit': 'still invalid'}, {'native_sample': 3})
    assert len(rows) == count


def test_reported_deletion_survives_head_only_samples_and_deduplicates_source_rows():
    rt, rows = runtime()
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'alpha',
                    'paths': ['src/a', 'src/b']}, actor='master', source={})
    rt.observe_git('beta', {'commit': 'abcdef1', 'deleted_paths': ['src/a', 'src/b']}, {})
    deletion_id = rt.git['beta']['id']
    rt.observe_git('beta', {'commit': 'abcdef1', 'deleted_paths': [], 'new_commit_count': 0}, {})
    assert rt.git['beta']['deleted_paths'] == []
    rt.commit_calls('beta', [CoordCall('r', {**report(['src/a', 'src/b']),
                    'summary': 'I accidentally deleted src/a and src/b.'})], response(), 1)
    events = [e for e in rt.events if e['category'] == 'destructive_delete' and e['evidence_level'] == 'reported']
    assert len(events) == 1
    assert events[0]['recipients'] == ['beta', 'alpha']
    assert deletion_id in events[0]['source_event_ids']
    assert any(row['id'] == deletion_id and row.get('deleted_paths') == ['src/a', 'src/b'] for row in rows)


def test_deletion_index_is_bounded_refreshes_paths_and_uses_latest_source(monkeypatch):
    import agent_company.runtime.collaboration as collaboration
    monkeypatch.setattr(collaboration, 'DELETION_INDEX_LIMIT', 2)
    rt, rows = runtime()
    for index, path in enumerate(('src/a', 'src/b', 'src/a', 'src/c'), 1):
        rt.observe_git('beta', {'commit': f'{index:07x}', 'deleted_paths': [path]}, {})
    assert list(rt.deletions) == ['src/a', 'src/c']
    assert rt.deletions['src/a'][0] == '0000003'
    assert rt.deletions['src/c'][0] == '0000004'
    assert rows[-1]['event'] == 'collaboration_sampling_incomplete'
    assert rows[-1]['path'] == 'src/b'
    assert rows[-1]['reason'] == 'deletion_index_limit_reached'
    rt.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'alpha',
                    'paths': ['src/a']}, actor='master', source={})
    latest = rt.deletions['src/a'][1]['id']
    rt.observe_git('beta', {'commit': '0000004'}, {})
    rt.commit_calls('beta', [CoordCall('r', {**report(['src/a']),
                    'summary': 'I accidentally deleted src/a.'})], response(), 1)
    event = next(e for e in rt.events if e['evidence_level'] == 'reported')
    assert latest in event['source_event_ids']
    assert json.loads(rt.summary('master').splitlines()[2])['coverage_reasons'] == ['sampling_incomplete']


def test_cache_merge_repeats_do_not_create_new_events_or_notices():
    rt, _ = runtime()
    now = [1000]
    rt.clock = lambda: now[0]
    for index in range(3):
        rt.observe_git('alpha', {'commit': f'{index + 1:07x}', 'added_paths': ['cache/a.pyc'],
                                'paths': ['cache/a.pyc'], 'new_commit_count': 1}, {})
        request = [{'role': 'user', 'content': f'task {index}'}]
        rt.decorate('master', request, 2000, index + 1)
        assert sum('Collaboration event' in message['content'] for message in request) == (1 if index == 0 else 0)
        now[0] += 700  # Exceed the cooldown: deduplication must not rely on it.
    assert len([e for e in rt.events if e['category'] == 'generated_files_tracked']) == 1
    rt.observe_git('beta', {'commit': '0000004', 'added_paths': ['cache/b.pyc'], 'new_commit_count': 1}, {})
    assert len([e for e in rt.events if e['category'] == 'generated_files_tracked']) == 2


def test_hypotheses_do_not_displace_recent_fix_or_active_task_from_summary():
    rt, rows = runtime()
    for number in range(6):
        rt.commit_calls('alpha', [CoordCall(f'r{number}', report())], response(), number + 1)
        identifier = list(rt.reports)[-1]
        rt.commit_calls('beta', [CoordCall(f'f{number}', {'op': 'resolve', 'issue_id': identifier,
                        'fix_commit': f'{number + 1:07x}', 'checks': 'unknown'})], response(), number + 7)
    rt.observe_task({'task_id': 'current', 'status': 'in_progress', 'owner': 'alpha', 'paths': []},
                    actor='master', source={})
    state = json.loads(rt.summary('master').splitlines()[2])
    assert not any(e['evidence_level'] == 'hypothesis' for e in state['entries'])
    assert any(e['kind'] == 'fix' for e in state['entries'])
    assert any(e['kind'] == 'task' and e.get('task_id') == 'current' for e in state['entries'])
    assert len([r for r in rows if r['event'] == 'collaboration_diagnostic' and r['evidence_level'] == 'hypothesis']) == 6


@pytest.mark.parametrize('status', ['pending', 'in_progress', 'completed'])
def test_task_update_native_status_does_not_require_owner(status):
    rt, _ = runtime()
    tool = call('task_update', {'taskId': '9'})
    rt.track_response('master', response(tool), 1)
    rt.observe_history('master', history(tool, f'Task #9 updated (status: {status}).'))
    assert rt.tasks['9']['status'] == status
    assert rt.tasks['9']['owner'] is None
    state = json.loads(rt.summary('master').splitlines()[2])
    assert any(e.get('task_id') == '9' for e in state['entries']) == (status != 'completed')
    assert 'task_status_unknown' not in rt.coverage


@pytest.mark.parametrize('tracked', [None, 0, 1])
def test_reported_handoff_requires_known_tracked_uncommitted_changes(tracked):
    rt, _ = runtime()
    sample = {'commit': 'abcdef1', 'uncommitted_paths': 20}
    if tracked is not None:
        sample['tracked_uncommitted_paths'] = tracked
    rt.observe_git('alpha', sample, {})
    rt.commit_calls('alpha', [CoordCall('r', {**report(), 'summary': 'Finishing handoff.',
                    'recipient': 'beta'})], response(), 1)
    reported = [e for e in rt.events if e['category'] == 'unfinished_handoff' and e['evidence_level'] == 'reported']
    assert bool(reported) == (tracked == 1)
    assert rt.git['alpha']['tracked_uncommitted_paths'] == tracked
    rt.terminal('alpha', 'end')
    assert bool([e for e in rt.events if e['category'] == 'unfinished_handoff']) == (tracked == 1)


@pytest.mark.parametrize('tracked', ['1', -1, True])
def test_tail_rejects_invalid_tracked_uncommitted_count(tmp_path, tracked):
    rt, _ = runtime()
    path = tmp_path / 'native.log'
    sample = {'kind': 'git', 'actor': 'alpha', 'sample': 1,
              'git': {'commit': 'abcdef1', 'tracked_uncommitted_paths': tracked}}
    path.write_text(PREFIX + json.dumps(sample) + '\n')
    tail = CollaborationTail(path, rt)
    tail.poll()
    assert tail.bad_lines == 1 and not rt.git
