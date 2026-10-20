"""Production relay composition for agent_rules_v1, zero API."""
import json

import pytest

from agent_company.policy import agent_rules_text as text
from agent_company.policy import agent_rules_runtime
from agent_company.policy.role_package import create_m0_package
from agent_company.runtime.collaboration_text import COLLABORATION_PROTOCOL, COORD_PROTOCOL, DELIVERY_PROTOCOL
from agent_company.runtime.relay import TOOL_ERROR_FIELD, ProviderConfig, TeamRelay
from agent_company.runtime.team import WorkerRegistry
from test_team import call, request, response, workspace


class Upstream:
    def __init__(self):
        self.queue, self.payloads = [], []

    def __call__(self, payload):
        self.payloads.append(payload)
        return 200, self.queue.pop(0)


def setup(tmp_path):
    upstream = Upstream()
    files = {'policies': str(tmp_path / 'runtime/status/agent_policies.md'),
             'lessons': str(tmp_path / 'runtime/status/team_lessons.md'),
             'board': str(tmp_path / 'runtime/status/team_board.md')}
    relay = TeamRelay(ProviderConfig('fake', 'http://localhost', max_attempts=1),
                      WorkerRegistry(workspace(tmp_path)), 'COMMUNICATION-PRINCIPLES', tmp_path / 'relay', 5000,
                      upstream, role_policy_mode='propose', policy_package=create_m0_package('A', 'COMMUNICATION-PRINCIPLES'),
                      policy_files=files)
    return relay, upstream


def turn(relay, upstream, actor='master', calls=(), history=()):
    tools = ('agent', 'send_message', 'task_create', 'task_update', 'task_list') if actor == 'master' else ('send_message', 'task_update')
    body = request(actor, tools)
    body['messages'].extend(history)
    upstream.queue.append(response(list(calls)))
    status, result = relay.handle(body)
    return status, result, upstream.payloads[-1]


def args(result, index=0):
    return json.loads(result['choices'][0]['message']['tool_calls'][index]['function']['arguments'])


def spawn(relay, upstream, name='alpha', block='POLICY:\nWHEN commit THEN SEND TO leader WITH sha\nEND-POLICY\n'):
    tool = call('agent', {'name': name, 'description': 'work', 'prompt': 'Implement the reader.\n' + block + 'Report back.'})
    status, delivered, _ = turn(relay, upstream, calls=[tool])
    assert status == 200 and TOOL_ERROR_FIELD not in args(delivered)
    return tool, delivered


def history(tool, result_text):
    return [{'role': 'assistant', 'content': '', 'tool_calls': [tool]},
            {'role': 'tool', 'tool_call_id': tool['id'], 'content': result_text}]


def test_agent_rules_requires_its_only_mode(tmp_path):
    with pytest.raises(ValueError, match='propose'):
        TeamRelay(ProviderConfig('fake', 'http://localhost'), WorkerRegistry(workspace(tmp_path)), 'C', tmp_path / 'r',
                  50, Upstream(), role_policy_mode='off', policy_package=create_m0_package('A', 'C'))


def test_master_system_is_fixed_across_spawns_and_policy_changes(tmp_path):
    relay, up = setup(tmp_path)
    _, _, before = turn(relay, up)
    spawn(relay, up)
    turn(relay, up, calls=[call('send_message', {'to': 'alpha', 'message': 'AGENT-POLICY alpha add\nWHEN task_done THEN SEND TO leader'})])
    _, _, after = turn(relay, up)
    assert before['messages'][0] == after['messages'][0]
    system = after['messages'][0]['content']
    assert text.MASTER_CARD.split(':')[0] in system and 'agent_policies.md' in system
    assert DELIVERY_PROTOCOL in system and COLLABORATION_PROTOCOL not in system and COORD_PROTOCOL not in system
    spawn_tool = next(t for t in after['tools'] if t['function']['name'] == 'agent')
    assert text.SPAWN_TOOL_NOTE in spawn_tool['function']['description']
    assert 'runtime_collaboration_state' not in json.dumps(after['messages'])


def test_spawn_prompt_loses_block_and_worker_sees_policy_only_in_tail(tmp_path):
    relay, up = setup(tmp_path)
    _, delivered = spawn(relay, up)
    prompt = args(delivered)['prompt']
    assert 'POLICY:' not in prompt and 'END-POLICY' not in prompt
    assert prompt.startswith('Implement the reader.\nReport back.') and prompt.endswith(text.SPAWN_SUFFIX)
    _, _, first = turn(relay, up, 'alpha')
    system = first['messages'][0]['content']
    assert (agent_rules_runtime.worker_card(str(tmp_path / 'runtime/status/team_board.md')) in system
            and 'COMMUNICATION-PRINCIPLES' not in system)
    assert 'R1: WHEN commit THEN SEND TO leader WITH sha' in first['messages'][-1]['content']
    turn(relay, up, calls=[call('send_message', {'to': 'alpha', 'message': 'AGENT-POLICY alpha notes\nOwn src/Readers only.'})])
    _, _, second = turn(relay, up, 'alpha')
    assert second['messages'][0] == first['messages'][0]
    assert 'version 1' in second['messages'][-1]['content'] and 'Own src/Readers only.' in second['messages'][-1]['content']


def test_spawn_receipt_is_appended_to_the_native_result(tmp_path):
    relay, up = setup(tmp_path)
    tool, _ = spawn(relay, up, block='POLICY:\nWHEN commit THEN SEND TO ghost\nEND-POLICY\n')
    _, _, later = turn(relay, up, history=history(tool, 'Teammate "alpha" is now running concurrently.'))
    result = next(m['content'] for m in later['messages'] if m.get('role') == 'tool')
    assert result.startswith('Teammate "alpha" is now running') and 'Skipped rule: WHEN commit THEN SEND TO ghost' in result
    _, _, again = turn(relay, up, history=history(tool, 'Teammate "alpha" is now running concurrently.'))
    assert next(m['content'] for m in again['messages'] if m.get('role') == 'tool') == result


def test_missing_block_reminder_is_anchored_once(tmp_path):
    relay, up = setup(tmp_path)
    tool, _ = spawn(relay, up, block='')
    hist = history(tool, 'Teammate "alpha" is now running concurrently.')
    _, _, first = turn(relay, up, history=hist)
    reminder = text.MISSING_POLICY_REMINDER.format(name='alpha')
    assert [m['content'] for m in first['messages']].count(reminder) == 1
    _, _, second = turn(relay, up, history=hist)
    assert [m['content'] for m in second['messages']].count(reminder) == 1
    index = lambda payload: [m['content'] for m in payload['messages']].index(reminder)
    assert index(first) == index(second)


def test_controls_are_rejected_per_call_and_partial_edits_report_per_target(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up, 'alpha')
    spawn(relay, up, 'beta', block='')
    bad = call('send_message', {'to': 'alpha', 'message': 'AGENT-POLICY alpha append\nx'})
    edit = call('send_message', {'to': '*', 'message': 'AGENT-POLICY alpha,beta add\nWHEN task_done THEN SEND TO beta'})
    task = call('task_list', {})
    status, delivered, _ = turn(relay, up, calls=[bad, edit, task])
    assert status == 200
    assert 'malformed_header' in args(delivered, 0)[TOOL_ERROR_FIELD]
    assert TOOL_ERROR_FIELD not in args(delivered, 1) and TOOL_ERROR_FIELD not in args(delivered, 2)
    _, _, later = turn(relay, up, history=[{'role': 'assistant', 'content': '', 'tool_calls': [bad, edit, task]},
                                           {'role': 'tool', 'tool_call_id': edit['id'], 'content': 'Message broadcast to all teammates.'}])
    result = next(m['content'] for m in later['messages'] if m.get('role') == 'tool')
    assert 'alpha: applied, version 1' in result and 'beta: not applied (self_recipient)' in result


def test_worker_self_change_lesson_and_master_lesson_is_ordinary(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up)
    turn(relay, up, 'alpha')
    proposal = call('send_message', {'to': 'leader', 'message': 'POLICY-PROPOSAL self add\nWHEN task_done THEN SEND TO leader\nREASON: report done'})
    lesson = call('send_message', {'to': 'leader', 'message': 'LESSON: Commit before reporting a SHA.'})
    status, delivered, _ = turn(relay, up, 'alpha', [proposal, lesson])
    assert status == 200 and all(TOOL_ERROR_FIELD not in args(delivered, i) for i in range(2))
    _, _, worker = turn(relay, up, 'alpha')
    assert 'version 1' in worker['messages'][-1]['content'] and 'was applied as version 1' in worker['messages'][-1]['content']
    master_lesson = call('send_message', {'to': 'alpha', 'message': 'LESSON: master note'})
    status, delivered, master = turn(relay, up, calls=[master_lesson])
    assert TOOL_ERROR_FIELD not in args(delivered)
    tail = master['messages'][-1]['content']
    assert 'Lessons recorded: 1' in tail and '#1 alpha: add -> version 1' in tail and 'Remaining wall time' in tail
    assert len(relay.role_policy.export()['lessons']) == 1


def test_policy_and_lesson_files_are_published(tmp_path):
    relay, up = setup(tmp_path)
    policies, lessons = (tmp_path / 'runtime/status/agent_policies.md'), (tmp_path / 'runtime/status/team_lessons.md')
    assert relay.flush_policy_files()
    assert 'No teammates yet.' in policies.read_text() and 'No lessons yet.' in lessons.read_text()
    spawn(relay, up)
    assert relay.flush_policy_files() and 'R1: WHEN commit THEN SEND TO leader WITH sha' in policies.read_text()
    turn(relay, up, 'alpha')
    turn(relay, up, 'alpha', [call('send_message', {'to': 'leader', 'message': 'LESSON: Pin the commit you tested.'})])
    assert relay.flush_policy_files()
    assert 'L1 from alpha' in lessons.read_text() and 'Pin the commit you tested.' in lessons.read_text()
    turn(relay, up, 'alpha', [call('send_message', {'to': 'leader', 'message': 'POLICY-PROPOSAL self remove R1\nREASON: noisy'})])
    assert relay.flush_policy_files() and 'removed R1: WHEN commit THEN SEND TO leader WITH sha' in policies.read_text()
    policies.write_text('tampered')
    relay.record_worker_terminal('alpha', 'test')
    assert relay.flush_policy_files() and '## alpha (ended)' in policies.read_text()
    assert not list(policies.parent.glob('*.tmp'))


def test_launch_state_summary_records_protocol(tmp_path):
    relay, up = setup(tmp_path)
    turn(relay, up)
    assert relay.summary()['role_policy']['protocol'] == 'agent_rules_v1'


def test_failed_spawn_drops_receipt_and_pending_reminder(tmp_path):
    relay, up = setup(tmp_path)
    tool, _ = spawn(relay, up, 'beta', block='')
    _, _, later = turn(relay, up, history=history(tool, 'Error: spawn failed for beta.'))
    result = next(m['content'] for m in later['messages'] if m.get('role') == 'tool')
    assert result == 'Error: spawn failed for beta.'
    assert text.MISSING_POLICY_REMINDER.format(name='beta') not in json.dumps(later['messages'])
    assert relay.role_policy.snapshot('master').worker('beta').lifecycle == 'voided'


def test_malformed_block_reaches_the_worker_unchanged(tmp_path):
    relay, up = setup(tmp_path)
    prompt = 'Implement.\nPOLICY:\nWHEN commit THEN SEND TO leader\nSUCCEEDS: nobody\nKeep going.'
    tool = call('agent', {'name': 'alpha', 'description': 'work', 'prompt': prompt})
    _, delivered, _ = turn(relay, up, calls=[tool])
    assert args(delivered)['prompt'] == prompt + '\n\n' + text.SPAWN_SUFFIX
    assert relay.role_policy.snapshot('master').worker('alpha').policy.rules == ()


def test_new_protocol_runs_no_diagnostics_or_summary():
    from agent_company.runtime.collaboration import CollaborationRuntime
    rows = []
    runtime = CollaborationRuntime('propose', rows.append, protocol='agent_rules_v1')
    runtime.observe_task({'task_id': '1', 'status': 'in_progress', 'owner': 'a', 'paths': ['src/x.hs']}, actor='master', source={})
    runtime.observe_task({'task_id': '2', 'status': 'in_progress', 'owner': 'b', 'paths': ['src/x.hs']}, actor='master', source={})
    runtime.observe_git('a', {'commit': 'a' * 40, 'paths': ['__pycache__/x.pyc'], 'added_paths': ['__pycache__/x.pyc'],
                              'deleted_paths': ['src/x.hs'], 'new_commit_count': 1}, {})
    runtime.terminal('a', 'test')
    messages = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'U'}]
    runtime.decorate('master', messages, 10_000, 1, state_tail='TAIL')
    assert not [row for row in rows if row['event'] == 'collaboration_diagnostic']
    assert messages[-1] == {'role': 'user', 'content': 'TAIL'} and len(messages) == 3
    assert runtime.parse_coord('a', 'c1', {'to': 'leader', 'message': 'COORD: {}'}) is None


def test_native_name_mismatch_is_reported_as_a_run_error(tmp_path):
    relay, up = setup(tmp_path)
    tool, _ = spawn(relay, up)
    relay.record_native_spawn_result({'requested_name': 'alpha', 'tool_call_id': tool['id'], 'worker': 'alpha_1',
                                      'ok': True}, 'test')
    errors = relay.summary()['errors']
    assert any("differs from requested 'alpha'" in error.get('error', '') for error in errors)


def test_file_writer_survives_render_errors_and_stops_after_finish(tmp_path, monkeypatch):
    from agent_company.policy import agent_rules_runtime
    relay, up = setup(tmp_path)
    assert relay.flush_policy_files()
    original, calls = agent_rules_runtime.render_policies_file, []

    def flaky(state):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('render failed once')
        return original(state)
    monkeypatch.setattr(agent_rules_runtime, 'render_policies_file', flaky)
    spawn(relay, up)
    assert relay.flush_policy_files()
    spawn(relay, up, 'beta')
    assert relay.flush_policy_files() and '## beta' in (tmp_path / 'runtime/status/agent_policies.md').read_text()
    assert any(json.loads(line).get('event') == 'agent_policy_file_write_failed'
               for line in (tmp_path / 'relay/events.jsonl').read_text().splitlines())
    thread = relay._files_thread
    assert relay.finish_policy_files()
    thread.join(timeout=5)
    assert not thread.is_alive()
    relay._publish_policy_files()  # after stop: ignored, no new thread
    assert relay._files_thread is thread and not thread.is_alive()
