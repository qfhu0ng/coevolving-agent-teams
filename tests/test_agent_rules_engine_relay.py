"""Rule obligations through the production relay composition (agent_rules_v1, zero API)."""
import json

from agent_company.runtime.relay import TOOL_ERROR_FIELD
from test_agent_rules_relay import args, history, setup, spawn, turn
from test_team import call, request, response

SHA = 'ab12cd34' * 5
RULE = 'POLICY:\nWHEN commit THEN SEND TO leader WITH sha\nWHEN message FROM leader THEN SEND TO leader\nEND-POLICY\n'


def events(relay):
    return [json.loads(line) for line in relay.output.joinpath('events.jsonl').read_text().splitlines()]


def names(relay):
    return [e['event'] for e in events(relay)]


def obligations(relay):
    return list(relay.rules.obligations.values())


def started(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up, block=RULE)
    turn(relay, up, 'alpha')
    return relay, up


def test_commit_reminder_send_and_native_result_make_a_satisfied_obligation(tmp_path):
    relay, up = started(tmp_path)
    relay.collaboration.observe_git('alpha', {'commit': SHA, 'new_commit': True, 'changed_paths': ['src/A.hs'],
                                              'committed_at': None, 'paths': ['src/A.hs']}, {'native_sample': 1})
    reminder = 'Rule R1 applies now: a new commit ab12cd34ab12 appeared in your worktree.'
    _, _, payload = turn(relay, up, 'alpha')
    assert sum(reminder in str(m.get('content')) for m in payload['messages']) == 1
    assert 'rule_reminded' in names(relay)
    report = call('send_message', {'to': 'leader', 'message': 'Committed.\nSHA: ab12cd34'})
    proposal = call('send_message', {'to': 'leader', 'message': 'POLICY-PROPOSAL self notes\nScope.\nREASON: r'})
    status, delivered, _ = turn(relay, up, 'alpha', [report, proposal])
    assert status == 200 and TOOL_ERROR_FIELD not in args(delivered)
    hist = [{'role': 'assistant', 'content': '', 'tool_calls': [report, proposal]},
            {'role': 'tool', 'tool_call_id': report['id'], 'content': 'Message sent to "leader".'},
            {'role': 'tool', 'tool_call_id': proposal['id'], 'content': 'Message sent to "leader".'}]
    turn(relay, up, 'alpha', history=hist)
    commit = next(o for o in obligations(relay) if o['trigger'] == 'commit')
    assert commit['status'] == 'satisfied' and commit['resolution']['spontaneous'] is False
    assert len(relay.rules.pending_sends) == 0


def test_message_observations_trigger_except_controls(tmp_path):
    relay, up = started(tmp_path)
    relay.record_lifecycle_event({'event': 'team_message_observed', 'worker': 'alpha',
                                  'items': [{'from': 'leader', 'control': True}]}, 'stderr:1')
    assert not [o for o in obligations(relay) if o['trigger'] == 'message']
    relay.record_lifecycle_event({'event': 'team_message_observed', 'worker': 'alpha',
                                  'items': [{'from': 'leader', 'control': False}, {'from': 'runtime:checkin'}]}, 'stderr:2')
    message = [o for o in obligations(relay) if o['trigger'] == 'message']
    assert len(message) == 1 and message[0]['events'][0]['detail'] == {'sender': 'leader'}


def test_task_transitions_from_tool_results_and_samples(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up, block='POLICY:\nWHEN task_claimed THEN SEND TO leader\nWHEN task_done THEN SEND TO leader\nEND-POLICY\n')
    collab = relay.collaboration
    collab.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'alpha', 'paths': None},
                        actor='master', source={'tool_call_id': 't1', 'source_request': 1, 'tool': 'task_create'})
    collab.observe_task({'task_id': '2', 'status': 'completed', 'owner': 'alpha', 'paths': None},
                        actor='master', source={'native_sample': 5})
    collab.observe_task({'task_id': '1', 'status': 'completed', 'owner': 'alpha', 'paths': None},
                        actor='master', source={'native_sample': 6})
    triggers = [(o['trigger'], o['events'][0]['detail']['task_id']) for o in obligations(relay)]
    assert triggers == [('task_claimed', '1'), ('task_done', '1')]
    assert any(e['event'] == 'rule_coverage' and e['reason'] == 'task_first_seen_in_sample' for e in events(relay))


def test_head_rows_and_master_rows_never_trigger_and_gaps_are_coverage(tmp_path):
    relay, up = started(tmp_path)
    collab = relay.collaboration
    collab.observe_git('alpha', {'commit': SHA, 'new_commit': False, 'changed_paths': []}, {})
    collab.observe_git('master', {'commit': SHA, 'new_commit': True, 'changed_paths': ['x']}, {})
    collab.observe_git('alpha', {'commit': None, 'new_commit': False, 'sampling_incomplete': ['nonancestor_previous_head']}, {})
    assert not obligations(relay)[1:] and [o['trigger'] for o in obligations(relay)] == []
    assert any(e['event'] == 'rule_coverage' and e['reason'] == 'git_sampling_incomplete' for e in events(relay))


def test_team_status_reports_pairs_pending_shutdowns_and_rates(tmp_path):
    relay, up = started(tmp_path)
    spawn(relay, up, 'beta', block='')
    relay.record_lifecycle_event({'event': 'team_message_queued', 'from': 'alpha', 'to': 'beta'}, 's:1')
    relay.record_lifecycle_event({'event': 'team_message_queued', 'from': 'alpha', 'to': 'beta'}, 's:2')
    relay.record_lifecycle_event({'event': 'team_message_queued', 'from': 'leader', 'to': 'beta'}, 's:3')
    import time
    relay.record_lifecycle_event({'event': 'shutdown_notice_observed', 'worker': 'beta',
                                  'at': (time.time() - 400) * 1000}, 's:4')
    relay.collaboration.observe_git('alpha', {'commit': SHA, 'new_commit': True, 'changed_paths': []}, {})
    status = relay.team_status()
    assert status['message_pairs'] == [{'from': 'alpha', 'to': 'beta', 'count': 2}]
    assert [row['worker'] for row in status['pending_shutdowns']] == ['beta']
    assert status['remaining_s'] > 0 and status['rule_rates']['alpha']['open'] >= 1
    relay.record_worker_terminal('beta', 'test')
    assert relay.team_status()['pending_shutdowns'] == []


def test_run_end_settles_and_saves_obligations(tmp_path):
    relay, up = started(tmp_path)
    relay.collaboration.observe_git('alpha', {'commit': SHA, 'new_commit': True, 'changed_paths': []}, {})
    totals = relay.finish_rule_engine(tmp_path / 'rules.json')
    saved = json.loads((tmp_path / 'rules.json').read_text())
    assert all(o['status'] != 'open' for o in saved['obligations'])
    assert totals['fields']['censored'] == 1 and saved['stats']['metric'] == 'observable_obligation_fulfillment_rate'
    assert names(relay)[-1] == 'rule_engine_finished'


def test_failed_spawn_ends_the_rule_lifetime(tmp_path):
    relay, up = setup(tmp_path)
    tool, _ = spawn(relay, up, 'beta', block=RULE)
    turn(relay, up, history=history(tool, 'Error: spawn failed for beta.'))
    assert relay.rules.lifetimes['beta']['ended_at'] is not None
    assert relay.rules.observe_commit('beta', SHA, []) == []


def worker_turn(relay, up, calls=(), hist=()):
    return turn(relay, up, 'alpha', list(calls), list(hist))


def sent(tool, text='Message sent to "leader".'):
    return {'role': 'tool', 'tool_call_id': tool['id'], 'content': text}


def test_report_before_the_sample_is_counted_and_no_reminder_is_shown(tmp_path):
    relay, up = started(tmp_path)
    report = call('send_message', {'to': 'leader', 'message': 'SHA: ab12cd34ab12'})
    worker_turn(relay, up, [report])
    worker_turn(relay, up, hist=[{'role': 'assistant', 'content': '', 'tool_calls': [report]}, sent(report)])
    import time
    relay.collaboration.observe_git('alpha', {'commit': SHA, 'new_commit': True, 'changed_paths': [],
                                              'committed_at': int(time.time()) - 30}, {})
    commit = next(o for o in obligations(relay) if o['trigger'] == 'commit')
    assert commit['status'] == 'satisfied' and commit['resolution']['spontaneous'] is True
    _, _, payload = worker_turn(relay, up)
    assert 'Rule R1 applies now' not in json.dumps(payload['messages'])


def test_task_completion_and_report_in_one_response(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up, block='POLICY:\nWHEN task_done THEN SEND TO leader WITH remaining\nEND-POLICY\n')
    worker_turn(relay, up)
    done = call('task_update', {'taskId': '1', 'status': 'completed'})
    report = call('send_message', {'to': 'leader', 'message': 'Finished task 1.\nREMAINING: nothing'})
    worker_turn(relay, up, [done, report])
    hist = [{'role': 'assistant', 'content': '', 'tool_calls': [done, report]},
            {'role': 'tool', 'tool_call_id': done['id'], 'content': 'Task #1 updated (status: completed, owner: alpha)'},
            sent(report)]
    _, _, payload = worker_turn(relay, up, hist=hist)
    task = next(o for o in obligations(relay) if o['trigger'] == 'task_done')
    assert task['status'] == 'satisfied'
    assert 'Rule R1 applies now' not in json.dumps(payload['messages'])


def test_auxiliary_requests_are_not_opportunities(tmp_path):
    relay, up = started(tmp_path)
    before = relay.rules.responses['alpha']
    body = {'messages': [{'role': 'system', 'content': 'Summarize.'}], 'tools': []}
    up.queue.append({'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'summary'},
                                  'finish_reason': 'stop'}], 'usage': None})
    assert relay.handle(body, owner='alpha', purpose='compression')[0] == 200
    assert relay.rules.responses['alpha'] == before
    worker_turn(relay, up)
    assert relay.rules.responses['alpha'] == before + 1


def test_native_execution_records_settle_final_and_broadcast_sends(tmp_path):
    from agent_company.runtime.rule_engine import message_hash
    relay, up = setup(tmp_path)
    spawn(relay, up, block='POLICY:\nWHEN shutdown_request THEN SEND TO leader WITH remaining\nEND-POLICY\n')
    worker_turn(relay, up)
    relay.record_lifecycle_event({'event': 'shutdown_notice_observed', 'worker': 'alpha', 'at': 1_000}, 's:1')
    assert relay._shutdown_requests['alpha'] == 1.0
    approval = 'shutdown_approved\nREMAINING: none'
    worker_turn(relay, up, [call('send_message', {'to': 'leader', 'message': approval})])
    relay.record_lifecycle_event({'event': 'team_message_sent', 'from': 'alpha', 'to': 'leader',
                                  'hash': message_hash(approval), 'ok': True}, 's:2')
    shutdown = next(o for o in obligations(relay) if o['trigger'] == 'shutdown_request')
    assert shutdown['status'] == 'satisfied' and shutdown['first_occurred_at'] == 1.0
    relay2, up2 = setup(tmp_path / 'b') if (tmp_path / 'b').mkdir() is None else None
    spawn(relay2, up2, block='POLICY:\nWHEN message FROM leader THEN SEND TO leader\nEND-POLICY\n')
    turn(relay2, up2, 'alpha')
    relay2.record_lifecycle_event({'event': 'team_message_observed', 'worker': 'alpha', 'at': 2_000,
                                   'items': [{'from': 'leader', 'control': False}]}, 's:3')
    turn(relay2, up2, 'alpha', [call('send_message', {'to': '*', 'message': 'ack'})])
    relay2.record_lifecycle_event({'event': 'team_broadcast', 'from': 'alpha', 'hash': message_hash('ack'),
                                   'recipients': [{'name': 'leader', 'outcome': 'accepted'},
                                                  {'name': 'beta', 'outcome': 'failed'}]}, 's:4')
    message = next(o for o in relay2.rules.obligations.values() if o['trigger'] == 'message')
    assert message['status'] == 'satisfied'


def test_sends_of_an_undelivered_response_are_dropped(tmp_path):
    from agent_company.runtime.transport import RequestControl
    from test_team import request, response
    relay, up = started(tmp_path)
    control = RequestControl(60)
    body = request('alpha', ('send_message', 'task_update'))
    up.queue.append(response([call('send_message', {'to': 'leader', 'message': 'SHA: ab12cd34'})]))
    assert relay.handle(body, control=control)[0] == 200
    assert relay.rules.pending_keys('alpha')
    relay.response_not_delivered(control)
    assert not relay.rules.pending_keys('alpha') and not relay._rule_sends


def test_sample_times_bound_merged_commits_and_sampled_task_transitions(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up, block='POLICY:\nWHEN commit THEN SEND TO leader\nWHEN task_done THEN SEND TO leader\nEND-POLICY\n')
    worker_turn(relay, up)
    collab = relay.collaboration
    import time
    t1 = time.time() - 120  # samples are always read before they are processed
    collab.observe_git('alpha', {'commit': SHA, 'new_commit': False, 'changed_paths': []}, {'native_sample': t1 * 1000})
    collab.observe_git('alpha', {'commit': 'cd' * 20, 'new_commit': True, 'changed_paths': [],
                                 'committed_at': int(t1) - 3600}, {'native_sample': (t1 + 60) * 1000})
    commit = next(o for o in obligations(relay) if o['trigger'] == 'commit')
    assert commit['first_occurred_at'] >= t1 - 1e-6  # not the merged commit's hour-old time
    collab.observe_task({'task_id': '9', 'status': 'in_progress', 'owner': 'alpha', 'paths': None},
                        actor='master', source={'native_sample': t1 * 1000})
    collab.observe_task({'task_id': '9', 'status': 'in_progress', 'owner': 'alpha', 'paths': None},
                        actor='master', source={'native_sample': (t1 + 30) * 1000})
    collab.observe_task({'task_id': '9', 'status': 'completed', 'owner': 'alpha', 'paths': None},
                        actor='master', source={'native_sample': (t1 + 60) * 1000})
    done = next(o for o in obligations(relay) if o['trigger'] == 'task_done')
    assert abs(done['first_occurred_at'] - (t1 + 30)) < 1e-6  # the previous snapshot, not the claim


def test_relay_rejected_and_control_sends_are_not_reports(tmp_path):
    relay, up = started(tmp_path)
    report = call('send_message', {'to': 'leader', 'message': 'SHA: ab12cd34'})
    disallowed = call('agent', {'name': 'x', 'description': 'd', 'prompt': 'p'})
    worker_turn(relay, up, [report, disallowed])  # whole response rejected before execution
    assert not relay.rules.pending_keys('alpha')


def test_owner_changes_and_missing_owners(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up, block='POLICY:\nWHEN task_claimed THEN SEND TO leader\nEND-POLICY\n')
    spawn(relay, up, 'beta', block='POLICY:\nWHEN task_claimed THEN SEND TO leader\nEND-POLICY\n')
    collab = relay.collaboration
    tool = {'tool_call_id': 't', 'source_request': 1, 'tool': 'task_update'}
    collab.observe_task({'task_id': '4', 'status': 'pending', 'owner': None, 'paths': None}, actor='master', source=tool)
    assert not obligations(relay)
    collab.observe_task({'task_id': '4', 'status': 'in_progress', 'owner': 'alpha', 'paths': None}, actor='master', source=tool)
    collab.observe_task({'task_id': '4', 'status': 'in_progress', 'owner': 'beta', 'paths': None}, actor='master', source=tool)
    assert [(o['worker'], o['trigger']) for o in obligations(relay)] == [('alpha', 'task_claimed'), ('beta', 'task_claimed')]


def test_unreadable_git_samples_do_not_move_the_commit_bound(tmp_path):
    import time
    relay, up = started(tmp_path)
    collab, t0 = relay.collaboration, time.time() - 300
    relay.rules.lifetimes['alpha']['started_at'] = t0 - 600  # spawned before these samples
    collab.observe_git('alpha', {'commit': SHA, 'new_commit': False, 'changed_paths': []}, {'native_sample': t0 * 1000})
    collab.observe_git('alpha', {'commit': None, 'new_commit': False, 'sampling_incomplete': ['git_read_failed']},
                       {'native_sample': (t0 + 60) * 1000})
    collab.observe_git('alpha', {'commit': 'ef' * 20, 'new_commit': True, 'changed_paths': [], 'committed_at': int(t0) - 900},
                       {'native_sample': (t0 + 120) * 1000})
    commit = next(o for o in obligations(relay) if o['trigger'] == 'commit')
    assert abs(commit['first_occurred_at'] - t0) < 1e-6  # the last readable sample, not the failed one


def test_a_task_moved_back_or_reopened_by_a_task_tool_triggers_again_but_a_sample_flap_does_not(tmp_path):
    # Review 10-18 F4: the (worker, task, kind) key is released when a task tool reverses the transition.
    import time
    block = 'POLICY:\nWHEN task_claimed THEN SEND TO leader\nWHEN task_done THEN SEND TO leader\nEND-POLICY\n'

    class Clock:
        now = time.time()

        def __call__(self):
            return self.now
    relay, up = setup(tmp_path)
    clock = Clock()
    relay.rules.clock = relay.reflection.clock = clock
    spawn(relay, up, 'alpha', block=block)
    spawn(relay, up, 'beta', block=block)
    turn(relay, up, 'alpha')
    turn(relay, up, 'beta')
    start = clock.now
    steps = [(0, 'master', 'alpha', 'in_progress', 'task_create'), (60, 'master', 'beta', 'in_progress', 'task_update'),
             (20 * 60, 'master', 'alpha', 'in_progress', 'task_update'), (21 * 60, 'alpha', 'alpha', 'completed', 'task_update'),
             (40 * 60, 'master', 'alpha', 'in_progress', 'task_update'), (60 * 60, 'alpha', 'alpha', 'completed', 'task_update')]
    for n, (offset, actor, owner, status, tool) in enumerate(steps):
        clock.now = start + offset
        relay.collaboration.observe_task({'task_id': '1', 'owner': owner, 'status': status, 'paths': None}, actor=actor,
                                         source={'tool_call_id': f't{n}', 'source_request': 1000 + n, 'tool': tool})
    triggers = [(o['worker'], o['trigger']) for o in relay.rules.obligations.values()]
    assert triggers == [('alpha', 'task_claimed'), ('beta', 'task_claimed'), ('alpha', 'task_claimed'),
                        ('alpha', 'task_done'), ('alpha', 'task_done')]
    # A stale periodic sample that flips a completed task back does not release the key.
    clock.now += 11 * 60
    relay.collaboration.observe_task({'task_id': '1', 'owner': 'alpha', 'status': 'in_progress', 'paths': None},
                                     actor='master', source={'native_sample': 5})
    clock.now += 30
    relay.collaboration.observe_task({'task_id': '1', 'owner': 'alpha', 'status': 'completed', 'paths': None},
                                     actor='master', source={'tool_call_id': 't9', 'source_request': 2000, 'tool': 'task_list'})
    assert len(relay.rules.obligations) == 5


def _task_rig(tmp_path):
    import time
    block = 'POLICY:\nWHEN task_claimed THEN SEND TO leader\nWHEN task_done THEN SEND TO leader\nEND-POLICY\n'

    class Clock:
        now = time.time()

        def __call__(self):
            return self.now
    relay, up = setup(tmp_path)
    clock = Clock()
    relay.rules.clock = relay.reflection.clock = clock
    spawn(relay, up, 'alpha', block=block)
    spawn(relay, up, 'beta', block=block)
    turn(relay, up, 'alpha')
    turn(relay, up, 'beta')

    def tool(n, owner, status, name='task_update', actor='master'):
        relay.collaboration.observe_task({'task_id': '1', 'owner': owner, 'status': status, 'paths': None}, actor=actor,
                                         source={'tool_call_id': f't{n}', 'source_request': 1000 + n, 'tool': name})

    def sample(owner, status):
        relay.collaboration.observe_task({'task_id': '1', 'owner': owner, 'status': status, 'paths': None},
                                         actor='master', source={'native_sample': clock.now * 1000})
    return relay, clock, tool, sample


def test_a_stale_sample_never_refires_a_released_task_transition(tmp_path):
    # Fix review 10-18: a board snapshot read before a reassign or reopen but tailed after it.
    relay, clock, tool, sample = _task_rig(tmp_path)
    tool(0, 'alpha', 'in_progress', 'task_create')
    clock.now += 20 * 60
    tool(1, 'beta', 'in_progress')
    clock.now += 2
    sample('alpha', 'in_progress')
    clock.now += 60
    sample('beta', 'in_progress')
    assert [(o['worker'], o['trigger']) for o in relay.rules.obligations.values()] == [
        ('alpha', 'task_claimed'), ('beta', 'task_claimed')]
    (tmp_path / 'reopen').mkdir()
    other, clock, tool, sample = _task_rig(tmp_path / 'reopen')
    tool(0, 'alpha', 'in_progress', 'task_create')
    clock.now += 60
    tool(1, 'alpha', 'completed', actor='alpha')
    clock.now += 20 * 60
    tool(2, 'alpha', 'in_progress')
    clock.now += 2
    sample('alpha', 'completed')
    assert [(o['worker'], o['trigger']) for o in other.rules.obligations.values()] == [
        ('alpha', 'task_claimed'), ('alpha', 'task_done')]


def test_owner_name_case_does_not_release_a_task_transition(tmp_path):
    relay, clock, tool, sample = _task_rig(tmp_path)
    tool(0, 'alpha', 'in_progress', 'task_create')
    for n in range(1, 4):
        clock.now += 11 * 60
        tool(n, 'Alpha', 'in_progress')  # the master typed a capital; native stores the sanitized name
        clock.now += 30
        sample('alpha', 'in_progress')
    assert [(o['worker'], o['trigger']) for o in relay.rules.obligations.values()] == [('alpha', 'task_claimed')]
