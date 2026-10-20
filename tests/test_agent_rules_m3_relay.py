"""M3 through the production relay (agent_rules_v1, zero API): spawn gate, board, roster, reflection."""
import json

from agent_company.policy import agent_rules_text as text
from agent_company.runtime.relay import TOOL_ERROR_FIELD
from test_agent_rules_relay import args, history, setup, spawn, turn
from test_team import call, request, response

SHA = 'ab12cd34' * 5


def events(relay):
    return [json.loads(line) for line in relay.output.joinpath('events.jsonl').read_text().splitlines()]


def named(relay, name):
    return [e for e in events(relay) if e['event'] == name]


def bound(relay, up, *names):
    for name in names:
        spawn(relay, up, name)
        turn(relay, up, name)


def lesson(relay, up, worker, body, hist=()):
    tool = call('send_message', {'to': 'leader', 'message': 'LESSON: ' + body})
    status, delivered, _ = turn(relay, up, worker, [tool], hist)
    assert status == 200 and TOOL_ERROR_FIELD not in args(delivered)
    return tool


def spawn_call(name):
    return call('agent', {'name': name, 'description': 'work', 'prompt': 'Do it.\nPOLICY:\nWHEN start THEN SEND TO leader\nEND-POLICY\n'})


def error(delivered, index=0):
    return args(delivered, index).get(TOOL_ERROR_FIELD, '')


def tail(payload):
    return payload['messages'][-1]['content']


# -- spawn gate --------------------------------------------------------------------------
def test_unseen_lessons_hold_every_spawn_once_then_the_next_spawn_goes_through(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    lesson(relay, up, 'alpha', 'Run the build before reporting.')
    note = call('send_message', {'to': 'alpha', 'message': 'Keep going.'})
    status, delivered, _ = turn(relay, up, calls=[spawn_call('beta'), note, spawn_call('gamma')])
    assert status == 200
    held = error(delivered, 0)
    assert '(lesson_gate)' in held and 'All lessons so far:' in held and 'Run the build before reporting.' in held
    assert text.LESSON_GATE_SAME in error(delivered, 2) and '(lesson_gate)' in error(delivered, 2)
    assert 'Run the build' not in error(delivered, 2) and TOOL_ERROR_FIELD not in args(delivered, 1)
    assert not relay._spawn_leases or all(e['worker'] == 'alpha' for e in relay._spawn_leases.values())
    assert 'beta' not in relay.role_policy.export()['workers']
    gate = named(relay, 'lesson_gate_held')
    assert len(gate) == 1 and gate[0]['lessons_given'] == ['L1'] and gate[0]['full'] is True
    # A lesson that arrives now waits for the next hold; this spawn is not held.
    lesson(relay, up, 'alpha', 'Merge from the delivery repository first.')
    status, delivered, _ = turn(relay, up, calls=[spawn_call('beta')])
    assert TOOL_ERROR_FIELD not in args(delivered)
    released = named(relay, 'lesson_gate_released')
    assert len(released) == 1 and released[0]['delay_s'] is not None
    status, delivered, _ = turn(relay, up, calls=[spawn_call('gamma')])
    held = error(delivered)
    assert 'Lessons since the previous hold:' in held and '[L2]' in held and '[L1]' not in held


def test_no_unseen_lessons_means_no_hold_and_compaction_shows_all_again(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    lesson(relay, up, 'alpha', 'First.')
    _, delivered, _ = turn(relay, up, calls=[spawn_call('beta')])
    assert '(lesson_gate)' in error(delivered)
    _, delivered, _ = turn(relay, up, calls=[spawn_call('beta')])
    _, delivered, _ = turn(relay, up, calls=[spawn_call('gamma')])
    assert TOOL_ERROR_FIELD not in args(delivered)  # nothing new since the hold
    up.queue.append(response([]))
    status, _ = relay.handle(request('master', ()), owner='master', purpose='chat-compression')
    assert status == 200
    lesson(relay, up, 'alpha', 'Second.')
    _, delivered, _ = turn(relay, up, calls=[spawn_call('delta')])
    held = error(delivered)
    assert 'All lessons so far:' in held and '[L1]' in held and '[L2]' in held


def test_a_hold_that_never_reached_native_is_undone(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    lesson(relay, up, 'alpha', 'First.')
    body = request('master', ('agent', 'send_message'))
    up.queue.append(response([spawn_call('beta')]))
    from agent_company.runtime.relay import RequestControl
    control = RequestControl(60)
    status, delivered = relay.handle(body, control=control)
    assert '(lesson_gate)' in error(delivered)
    relay.response_not_delivered(control)
    assert relay._gate_seen == set() and relay._gate_release is False
    _, delivered, _ = turn(relay, up, calls=[spawn_call('beta')])
    assert 'All lessons so far:' in error(delivered)


# -- reflection ---------------------------------------------------------------------------
def complete(relay, task_id, owner='alpha'):
    relay.collaboration.observe_task({'task_id': task_id, 'status': 'in_progress', 'owner': owner, 'paths': None},
                                     actor='master', source={'tool': 'task_create', 'tool_call_id': 'c' + task_id,
                                                             'source_request': 1})
    relay.collaboration.observe_task({'task_id': task_id, 'status': 'completed', 'owner': owner, 'paths': None},
                                     actor='master', source={'tool': 'task_update', 'tool_call_id': 'u' + task_id,
                                                             'source_request': 2})


def test_task_completion_prompts_reflection_and_the_lesson_records_it(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    complete(relay, '3')
    _, _, payload = turn(relay, up, 'alpha')
    prompt = text.REFLECT_TASK_DONE.format(task_id='3')
    assert sum(prompt == m.get('content') for m in payload['messages']) == 1
    lesson(relay, up, 'alpha', 'Claim one module at a time.')
    assert relay.role_policy.lessons()[0]['trigger'] == 'task_done'
    complete(relay, '4')  # within ten minutes: cooled down
    assert named(relay, 'reflection_prompt_cooled_down')
    record = relay.finish_rule_engine(tmp_path / 'rules.json')
    saved = json.loads((tmp_path / 'rules.json').read_text())
    assert record is not None and saved['reflection']['task_done']['with_output'] == 1
    funnel = named(relay, 'reflection_funnel')
    assert funnel[0]['lessons'][0]['id'] == 'L1' and funnel[0]['outcome'] == 'output'


def test_shutdown_request_prompts_lessons_once(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    relay.record_lifecycle_event({'event': 'shutdown_notice_observed', 'worker': 'alpha', 'at': 1}, 'stderr:1')
    _, _, payload = turn(relay, up, 'alpha')
    assert sum(text.REFLECT_SHUTDOWN == m.get('content') for m in payload['messages']) == 1
    _, _, payload = turn(relay, up, 'alpha')
    assert sum(text.REFLECT_SHUTDOWN == m.get('content') for m in payload['messages']) == 1  # anchored, not repeated
    assert len(named(relay, 'reflection_prompt_shown')) == 1


# -- board -------------------------------------------------------------------------------
def test_board_posts_reach_their_recipient_the_file_and_other_tails(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha', 'beta')
    _, _, before = turn(relay, up, 'beta')
    assert 'Board: no posts by others yet (0 post(s)' in tail(before)
    tool = call('send_message', {'to': 'leader', 'message': 'BOARD: reader interface fixed\nPATHS: src/Reader.hs'})
    status, delivered, _ = turn(relay, up, 'alpha', [tool])
    assert status == 200 and TOOL_ERROR_FIELD not in args(delivered)
    _, _, mine = turn(relay, up, 'alpha', history=history(tool, 'Message sent to "leader".'))
    assert text.BOARD_RECEIPT.format(id='B1') in json.dumps(mine['messages'])
    assert '[B1]' not in tail(mine)  # one's own posts are not listed
    _, _, first = turn(relay, up, 'beta')
    assert '[B1] new alpha' in tail(first) and 'reader interface fixed' in tail(first)
    _, _, second = turn(relay, up, 'beta')
    assert '[B1] alpha' in tail(second) and '[B1] new' not in tail(second)
    _, _, master = turn(relay, up)
    assert 'Board posts: 1, 1 new since your previous request.' in tail(master)
    _, _, master = turn(relay, up)
    assert 'Board posts: 1, 0 new since your previous request.' in tail(master)
    assert relay.flush_policy_files()
    assert 'reader interface fixed' in (tmp_path / 'runtime/status/team_board.md').read_text()


def test_board_rules_are_met_by_posts_and_reminders_name_the_board(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up, 'alpha', block='POLICY:\nWHEN commit THEN SEND TO board WITH sha\nEND-POLICY\n')
    turn(relay, up, 'alpha')
    relay.collaboration.observe_git('alpha', {'commit': SHA, 'new_commit': True, 'changed_paths': ['src/A.hs'],
                                              'committed_at': None, 'paths': ['src/A.hs']}, {'native_sample': 1})
    _, _, payload = turn(relay, up, 'alpha')
    assert any('team board' in str(m.get('content')) and 'Rule R1' in str(m.get('content')) for m in payload['messages'])
    tool = call('send_message', {'to': 'leader', 'message': 'BOARD: parser\nSHA: ab12cd34'})
    turn(relay, up, 'alpha', [tool])
    obligation = next(iter(relay.rules.obligations.values()))
    assert obligation['status'] == 'satisfied'


def test_message_facts_and_board_reads_are_recorded_without_text(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    report = call('send_message', {'to': 'leader', 'message': 'Fixed in ' + SHA[:9] + ', see src/Reader/Docx.hs'})
    control = call('send_message', {'to': 'leader', 'message': 'POLICY-PROPOSAL self notes\nScope.\nREASON: r'})
    turn(relay, up, 'alpha', [report, control])
    reader = call('read_file', {'absolute_path': str(tmp_path / 'runtime/status/team_board.md')})
    body = request('alpha', ('send_message', 'read_file'))
    up.queue.append(response([reader]))
    assert relay.handle(body)[0] == 200
    facts = named(relay, 'message_facts')
    assert len(facts) == 1 and facts[0]['commits'] == [SHA[:9]] and facts[0]['paths'] == ['src/Reader/Docx.hs']
    assert 'message' not in facts[0] and facts[0]['sender'] == 'alpha' and facts[0]['board'] is False
    reads = named(relay, 'board_read_attempt')
    assert len(reads) == 1 and reads[0]['tool'] == 'read_file' and reads[0]['approximate'] is True


# -- roster -------------------------------------------------------------------------------
def test_worker_tail_lists_active_teammates_from_the_latest_team_sample(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha', 'beta')
    _, _, payload = turn(relay, up, 'alpha')
    assert 'Active teammates (status not observed yet):\n- beta: just started, not observed yet' in tail(payload)
    relay.collaboration.observe_team({'members': [
        {'name': 'alpha', 'status': 'running', 'shutdown_pending': False, 'tasks': []},
        {'name': 'beta', 'status': 'idle', 'shutdown_pending': False, 'tasks': [{'id': '7', 'subject': 'Docx tables'}]}]},
        {'native_sample': 1_700_000_000_000})
    _, _, payload = turn(relay, up, 'alpha')
    assert 'Active teammates (as of 22:13:20 UTC):\n- beta: idle; in progress: #7 Docx tables' in tail(payload)
    relay.record_worker_terminal('beta', 'stderr:2')
    _, _, payload = turn(relay, up, 'alpha')
    assert 'none besides you' in tail(payload)


def test_team_status_carries_policy_lesson_and_board_counts(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    lesson(relay, up, 'alpha', 'First.')
    turn(relay, up, 'alpha', [call('send_message', {'to': 'leader', 'message': 'BOARD: x'})])
    status = relay.team_status()
    assert status['lessons_total'] == 1 and status['board_total'] == 1
    assert status['pending_proposals'] == 0 and status['recent_self_changes'] == 0


# -- review fixes (M3_REVIEW_SUBAGENT.md) ---------------------------------------------------
def test_a_failing_recorder_never_fails_a_committed_response(tmp_path, monkeypatch):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    original = relay.event

    def flaky(event, **fields):
        if event == 'message_facts':
            raise OSError('disk full')
        return original(event, **fields)
    monkeypatch.setattr(relay, 'event', flaky)
    tool = call('send_message', {'to': 'leader', 'message': 'BOARD: parser done'})
    status, delivered, _ = turn(relay, up, 'alpha', [tool])
    assert status == 200 and TOOL_ERROR_FIELD not in args(delivered)
    assert relay.role_policy.export()['board'][0]['id'] == 'B1'
    assert relay.summary()['role_policy']['counts']['evidence_failures'] == 1


def test_board_marks_survive_a_failed_request(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha', 'beta')
    turn(relay, up, 'beta', [call('send_message', {'to': 'leader', 'message': 'BOARD: first'})])
    up.queue.append({'error': {'message': 'overloaded'}})
    original = relay.upstream
    relay.upstream = lambda payload: (up.payloads.append(payload), (500, up.queue.pop(0)))[1]
    status, _ = relay.handle(request('alpha', ('send_message', 'task_update')))
    relay.upstream = original
    assert status == 424 and '[B1] new beta' in tail(up.payloads[-1])
    _, _, payload = turn(relay, up, 'alpha')
    assert '[B1] new beta' in tail(payload)
    _, _, payload = turn(relay, up, 'alpha')
    assert '[B1] beta' in tail(payload) and '[B1] new' not in tail(payload)


def test_master_tail_counts_unseen_lessons_and_holds_are_undone_with_their_counts(tmp_path):
    from agent_company.runtime.relay import RequestControl
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    lesson(relay, up, 'alpha', 'First.')
    _, _, payload = turn(relay, up)
    assert 'Lessons recorded: 1, 1 not yet shown to you when a spawn was held.' in tail(payload)
    up.queue.append(response([spawn_call('beta')]))
    control = RequestControl(60)
    relay.handle(request('master', ('agent', 'send_message')), control=control)
    assert relay.summary()['role_policy']['counts']['lesson_gate_holds'] == 1
    relay.response_not_delivered(control)
    assert relay.summary()['role_policy']['counts']['lesson_gate_holds'] == 0
    _, delivered, _ = turn(relay, up, calls=[spawn_call('beta')])
    assert '(lesson_gate)' in error(delivered)
    _, _, payload = turn(relay, up)
    assert 'Lessons recorded: 1, 0 not yet shown to you when a spawn was held.' in tail(payload)
    injected = [e for e in named(relay, 'agent_policy_injected') if e['role'] == 'master']
    assert injected[-1]['tail']['lessons_unseen'] == 0 and injected[-1]['tail']['tail_chars'] > 0


def test_a_release_spawn_rejected_per_call_keeps_the_release(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    lesson(relay, up, 'alpha', 'First.')
    _, delivered, _ = turn(relay, up, calls=[spawn_call('beta')])
    assert '(lesson_gate)' in error(delivered)
    lesson(relay, up, 'alpha', 'Second.')
    _, delivered, _ = turn(relay, up, calls=[spawn_call('board')])  # reserved name: rejected per call
    assert '(invalid_name)' in error(delivered)
    _, delivered, _ = turn(relay, up, calls=[spawn_call('beta')])
    assert TOOL_ERROR_FIELD not in args(delivered)


def test_board_posts_to_board_are_refused_before_native(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    _, delivered, _ = turn(relay, up, 'alpha', [call('send_message', {'to': 'board', 'message': 'BOARD: x'})])
    assert '(wrong_recipient)' in error(delivered) and relay.role_policy.export()['board'] == []
    assert not named(relay, 'message_facts')


def test_native_send_outcomes_and_tail_contents_are_recorded(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha', 'beta')
    relay.record_lifecycle_event({'event': 'team_message_sent', 'from': 'leader', 'to': 'alpha', 'hash': 'h1', 'ok': True},
                                 'stderr:1')
    relay.record_lifecycle_event({'event': 'team_broadcast', 'from': 'alpha', 'to': '*', 'hash': 'h2',
                                  'recipients': [{'name': 'beta', 'outcome': 'accepted'}]}, 'stderr:2')
    outcomes = named(relay, 'message_outcome')
    assert [(o['sender'], o['hash'], o['ok']) for o in outcomes] == [('master', 'h1', True), ('alpha', 'h2', None)]
    assert outcomes[1]['recipients'] == [{'name': 'beta', 'outcome': 'accepted'}]
    turn(relay, up, 'beta', [call('send_message', {'to': 'leader', 'message': 'BOARD: first'})])
    turn(relay, up, 'alpha')
    shown = [e for e in named(relay, 'agent_policy_injected') if e['role'] == 'alpha'][-1]['tail']
    assert shown['board_shown'] == ['B1'] and shown['board_new'] == ['B1'] and shown['roster_rows'] == 1


def test_only_reading_tools_count_as_board_reads(tmp_path):
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    path = str(tmp_path / 'runtime/status/team_board.md')
    body = request('alpha', ('send_message', 'read_file', 'write_file', 'run_shell_command'))
    up.queue.append(response([call('write_file', {'file_path': path, 'content': 'x'}),
                              call('run_shell_command', {'command': 'cat ' + path})]))
    assert relay.handle(body)[0] == 200
    assert [e['tool'] for e in named(relay, 'board_read_attempt')] == ['run_shell_command']


def test_a_hold_inside_a_wholly_rejected_response_is_not_counted_as_seen(tmp_path):
    from agent_company.runtime.relay import RequestControl
    relay, up = setup(tmp_path)
    bound(relay, up, 'alpha')
    lesson(relay, up, 'alpha', 'First.')
    raw = response([spawn_call('beta')])
    gate = relay._lesson_gate()
    policy = {'lesson_gate': {**gate, 'calls': [raw['choices'][0]['message']['tool_calls'][0]['id']]}, 'source_request': 9}
    late = relay.rejected_tool_response(raw, 'PolicyRejection: stale')  # what a late whole-response rejection delivers
    relay._settle_lesson_gate(policy, late, RequestControl(60))
    assert relay._gate_seen == set() and not named(relay, 'lesson_gate_held')


def test_a_failed_outcome_write_still_settles_the_send(tmp_path, monkeypatch):
    relay, up = setup(tmp_path)
    spawn(relay, up, 'alpha', block='POLICY:\nWHEN start THEN SEND TO leader\nEND-POLICY\n')
    turn(relay, up, 'alpha')
    tool = call('send_message', {'to': 'leader', 'message': 'started'})
    turn(relay, up, 'alpha', [tool])
    original = relay.event

    def flaky(event, **fields):
        if event == 'message_outcome':
            raise OSError('disk full')
        return original(event, **fields)
    monkeypatch.setattr(relay, 'event', flaky)
    from agent_company.runtime.rule_engine import message_hash
    try:
        relay.record_lifecycle_event({'event': 'team_message_sent', 'from': 'alpha', 'to': 'leader',
                                      'hash': message_hash('started'), 'ok': True}, 'stderr:1')
    except OSError:
        pass
    assert next(iter(relay.rules.obligations.values()))['status'] == 'satisfied'
