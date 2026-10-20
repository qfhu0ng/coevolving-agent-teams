"""Production relay composition for package state and collaboration, zero API."""
import json

import pytest

from agent_company.policy.role_package import create_p0_package
from agent_company.runtime.relay import TOOL_ERROR_FIELD, ProviderConfig, TeamRelay
from agent_company.runtime.team import WorkerRegistry
from test_team import call, request, response, workspace


class Upstream:
    def __init__(self):
        self.queue, self.payloads = [], []

    def __call__(self, payload):
        self.payloads.append(payload)
        return 200, self.queue.pop(0)


def setup(tmp_path, mode='propose'):
    upstream = Upstream()
    relay = TeamRelay(ProviderConfig('fake', 'http://localhost', max_attempts=1),
                      WorkerRegistry(workspace(tmp_path)), 'C', tmp_path / 'relay', 5000,
                      upstream, role_policy_mode=mode, policy_package=create_p0_package('A', 'C'))
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


def spawn(relay, upstream, name='alpha', role='implementer'):
    tool = call('agent', {'name': name, 'description': 'work', 'prompt': 'Implement assigned task.\nROLE-TYPE: ' + role})
    assert turn(relay, upstream, calls=[tool])[0] == 200
    return tool


@pytest.mark.parametrize('mode', ['off', 'propose'])
def test_off_and_propose_binding_and_role_shared_system(tmp_path, mode):
    relay, up = setup(tmp_path, mode)
    tool = spawn(relay, up)
    status, _, payload = turn(relay, up, 'alpha')
    assert status == 200
    system = payload['messages'][0]['content']
    assert 'Your role type is' not in system  # exact lower-case approved binding line below
    assert 'your role type is implementer' in system
    assert 'Fixed collaboration protocol' in system
    assert 'D10' not in system
    assert payload['messages'][-1]['role'] == 'user'
    assert 'runtime_collaboration_state' in str(payload['messages'])
    assert 'ROLE-TYPE: implementer' in args(turn(relay, up, calls=[call('agent', {'name':'beta','prompt':'Work.\nROLE-TYPE: implementer','description':'work'})])[1])['prompt']
    assert tool['function']['name'] == 'agent'


def test_invalid_binding_does_not_drop_independent_spawn_or_reserve_slot(tmp_path):
    relay, up = setup(tmp_path)
    bad = call('agent', {'name': 'bad', 'description':'bad', 'prompt': 'ROLE-TYPE: not_registered'})
    good = call('agent', {'name':'alpha', 'description':'ok', 'prompt':'ROLE-TYPE: implementer'})
    status, delivered, _ = turn(relay, up, calls=[bad, good])
    assert status == 200
    assert TOOL_ERROR_FIELD in args(delivered, 0)
    assert TOOL_ERROR_FIELD not in args(delivered, 1)
    assert 'working_dir' in args(delivered, 1)
    assert relay.registry.contains('alpha') and not relay.registry.contains('bad')
    assert len(relay._spawn_leases) == 1


def test_local_coord_error_keeps_task_call_and_policy_valid(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up)
    bad = call('send_message', {'to': '*', 'message': 'COORD: ' + json.dumps({'op':'bad'})})
    task = call('task_update', {'taskId':'1', 'status':'in_progress'})
    status, delivered, _ = turn(relay, up, 'alpha', [bad, task])
    assert status == 200
    assert TOOL_ERROR_FIELD in args(delivered, 0) and TOOL_ERROR_FIELD not in args(delivered, 1)


def test_policy_commit_shared_role_and_pending_not_in_system(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up)
    proposal = call('send_message', {'to':'leader', 'message': 'POLICY-PROPOSAL role:implementer append\nCHANGE: Report changed interfaces.\nREASON: Coordination gap.\nEVIDENCE: collab-2.'})
    assert turn(relay, up, 'alpha', [proposal])[0] == 200
    _, _, payload = turn(relay, up)
    assert 'Report changed interfaces.' not in payload['messages'][0]['content']
    assert 'Report changed interfaces.' in str(payload['messages'][1:])
    approved = call('send_message', {'to':'alpha', 'message':'POLICY-DECISION #1 approve'})
    assert turn(relay, up, calls=[approved])[0] == 200
    spawn(relay, up, 'beta')
    _, _, beta = turn(relay, up, 'beta')
    assert 'Report changed interfaces.' in beta['messages'][0]['content']


def test_coord_real_response_receipt_and_correction_does_not_lose_nonpolicy(tmp_path):
    relay, up = setup(tmp_path)
    spawn(relay, up)
    record = {'op':'report','summary':'Need recheck.','paths':[], 'tested_commit':None,'build_id':None,'checks':'unknown','recipient':'leader'}
    tool = call('send_message', {'to':'leader', 'message':'COORD: ' + json.dumps(record)})
    status, _, _ = turn(relay, up, 'alpha', [tool])
    assert status == 200
    hist = [{'role':'assistant','content':'','tool_calls':[tool]}, {'role':'tool','tool_call_id':tool['id'],'content':'Message sent to "leader".'}]
    _, _, later = turn(relay, up, 'alpha', history=hist)
    result = next(m['content'] for m in later['messages'] if m['role'] == 'tool')
    assert 'coord_status' in result and 'reported' in result
    assert len(relay.collaboration.reports) == 1


def test_single_package_has_no_team_store_or_coord_protocol(tmp_path):
    up = Upstream()
    relay = TeamRelay(ProviderConfig('fake', 'http://localhost', max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      '', tmp_path/'relay', 5000, up, execution_mode='single', policy_package=create_p0_package('A','C'))
    up.queue.append(response([]))
    assert relay.handle(request('master', ('read_file',)))[0] == 200
    assert relay.role_policy is None
    assert 'COORD:' not in str(up.payloads[-1])


@pytest.mark.parametrize('other_kind', ['control', 'ordinary'])
def test_duplicate_call_id_is_whole_response_rejected_before_commit(tmp_path, other_kind):
    relay, up = setup(tmp_path)
    first = call('send_message', {'to':'*','message':'GLOBAL-POLICY append\nReport integration changes.'})
    other = call('send_message', {'to':'*','message':'GLOBAL-POLICY append\nReport test changes.'}) if other_kind == 'control' else call('task_list', {})
    other['id'] = first['id']
    status, delivered, _ = turn(relay, up, calls=[first, other])
    assert status == 200 and TOOL_ERROR_FIELD in args(delivered, 0) and TOOL_ERROR_FIELD in args(delivered, 1)
    assert relay.role_policy.snapshot('master').global_policy.version == 0
    assert not relay._spawn_leases


def test_partial_rejection_is_recorded_once_and_binding_feedback_is_exact(tmp_path):
    relay, up = setup(tmp_path)
    bad = call('agent', {'name':'bad','description':'bad','prompt':'ROLE-TYPE: unknown'})
    good = call('agent', {'name':'alpha','description':'work','prompt':'ROLE-TYPE: implementer'})
    _, delivered, _ = turn(relay, up, calls=[bad,good])
    assert 'Teammate assignment rejected before native dispatch' in args(delivered, 0)[TOOL_ERROR_FIELD]
    events = [json.loads(line) for line in (tmp_path/'relay/events.jsonl').read_text().splitlines()]
    errors = [e for e in events if e['event'] == 'role_package_call_rejected' and e['tool_call_id'] == bad['id']]
    assert len(errors) == 1


def test_transport_retry_reuses_frozen_notice_and_current_state(tmp_path):
    import copy
    payloads = []
    def upstream(payload):
        payloads.append(copy.deepcopy(payload))
        if len(payloads) == 1:
            relay.collaboration.observe_task({'task_id':'new','status':'pending','owner':None,'paths':None},
                                             actor='master', source={'scripted':'between provider attempts'})
            return 503, {'error': {'message':'scripted transient'}}
        return 200, response([])
    relay = TeamRelay(ProviderConfig('fake','http://localhost',max_attempts=2,retry_backoff_seconds=0),
                      WorkerRegistry(workspace(tmp_path)), 'C', tmp_path/'relay', 800, upstream,
                      role_policy_mode='off', policy_package=create_p0_package('A','C'))
    assert relay.handle(request('master', ('task_list',)))[0] == 200
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    events = [json.loads(line) for line in (tmp_path/'relay/events.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event'] == 'collaboration_notice_injected']) == 1
    assert relay.summary()['logical_model_calls'] == 1


@pytest.mark.parametrize("mode", ["off", "propose"])
def test_delivery_protocol_is_stable_system_for_every_team_role(tmp_path, mode):
    from agent_company.runtime.collaboration_text import DELIVERY_PROTOCOL, COLLABORATION_PROTOCOL
    relay, up = setup(tmp_path, mode)
    spawn(relay, up)
    for actor in ("master", "alpha"):
        for _ in range(2):  # fresh/native compacted request with no original task prompt
            _, _, payload = turn(relay, up, actor)
            system = payload["messages"][0]["content"]
            assert system.count(DELIVERY_PROTOCOL) == 1
            assert system.index(relay.source_boundary) < system.index(DELIVERY_PROTOCOL) < system.index(COLLABORATION_PROTOCOL)
            assert all(DELIVERY_PROTOCOL not in str(m) for m in payload["messages"][1:])


def test_single_delivery_system_does_not_reject_duplicate_tool_ids(tmp_path):
    from agent_company.runtime.collaboration_text import DELIVERY_PROTOCOL
    up = Upstream()
    relay = TeamRelay(ProviderConfig('fake', 'http://localhost', max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      '', tmp_path/'relay', 5000, up, execution_mode='single', policy_package=create_p0_package('A','C'))
    first, second = call('read_file', {'file_path':'one'}), call('read_file', {'file_path':'two'})
    second['id'] = first['id']
    up.queue.append(response([first, second]))
    status, result = relay.handle(request('master', ('read_file',)))
    assert status == 200
    assert TOOL_ERROR_FIELD not in args(result, 0) and TOOL_ERROR_FIELD not in args(result, 1)
    system = up.payloads[-1]['messages'][0]['content']
    assert system.count(DELIVERY_PROTOCOL) == 1
    assert system.index(relay.source_boundary) < system.index(DELIVERY_PROTOCOL)


def policy_and_spawn():
    return [call('send_message', {'to':'*','message':'GLOBAL-POLICY append\nReport integration changes.'}),
            call('agent', {'name':'alice','description':'work','prompt':'ROLE-TYPE: implementer'})]


def read_events(relay):
    return [json.loads(line) for line in (relay.output/'events.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('failure', ['cancel', 'deadline', 'coord_write', 'track_write'])
def test_precommit_abort_never_publishes_policy_or_binding_and_releases_lease(tmp_path, monkeypatch, failure):
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    original = relay.collaboration.commit_calls
    def before_policy(*a, **k):
        original(*a, **k)
        if failure == 'cancel':
            control.cancel('scripted disconnect')
        elif failure == 'deadline':
            control._deadline = 0
        elif failure == 'coord_write':
            raise OSError('scripted collaboration disk failure')
    monkeypatch.setattr(relay.collaboration, 'commit_calls', before_policy)
    if failure == 'track_write':
        monkeypatch.setattr(relay.collaboration, 'track_response', lambda *a, **k: (_ for _ in ()).throw(OSError('track failure')))
    up.queue.append(response(policy_and_spawn()))
    status, _ = relay.handle(request('master', ('agent','send_message')), control=control)
    assert status == 424
    assert relay.role_policy.snapshot('master').global_policy.version == 0
    assert not relay.role_policy.export()['bindings']
    assert not relay._spawn_leases and not relay._policy_spawns


@pytest.mark.parametrize('failure', ['cancel', 'deadline'])
def test_after_commit_cancel_or_deadline_cannot_replace_accepted_response(tmp_path, monkeypatch, failure):
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    original = relay.role_policy.commit
    def commit(*a, **k):
        result = original(*a, **k)
        if failure == 'cancel':
            control.cancel('scripted disconnect after durable commit')
        else:
            control._deadline = 0
        return result
    monkeypatch.setattr(relay.role_policy, 'commit', commit)
    up.queue.append(response(policy_and_spawn()))
    status, delivered = relay.handle(request('master', ('agent','send_message')), control=control)
    assert status == 200 and TOOL_ERROR_FIELD not in args(delivered, 1)
    assert relay.role_policy.snapshot('master').global_policy.version == 1
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'planned'
    # A definite HTTP delivery failure retires the unstarted binding, not C.
    relay.response_not_delivered(control)
    assert relay.role_policy.snapshot('master').global_policy.version == 1
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'voided'
    assert not relay._spawn_leases and not relay._policy_spawns
    assert any(c['event'] == 'role_policy_binding_ended' for e in read_events(relay) for c in e.get('changes', []))
    assert spawn(relay, up, 'alice')


def test_partial_delivery_does_not_void_already_confirmed_worker(tmp_path):
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    up.queue.append(response(policy_and_spawn()))
    assert relay.handle(request('master', ('agent','send_message')), control=control)[0] == 200
    relay.role_policy.record_first_request('alice')
    relay.response_not_delivered(control)
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'active'
    assert len(relay._spawn_leases) == 1


def test_not_delivered_is_idempotent_for_one_request(tmp_path, monkeypatch):
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    up.queue.append(response(policy_and_spawn()))
    assert relay.handle(request('master', ('agent', 'send_message')), control=control)[0] == 200
    cleaned = []
    original = relay.role_policy.record_spawn_result
    def record(instance, **fields):
        cleaned.append(instance)
        return original(instance, **fields)
    monkeypatch.setattr(relay.role_policy, 'record_spawn_result', record)
    relay.response_not_delivered(control)
    relay.response_not_delivered(control)
    assert control.role_policy_delivery is None
    assert cleaned == ['alice']
    assert len([e for e in read_events(relay) if e['event'] == 'role_policy_response_not_delivered']) == 1
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'voided'
    assert not relay._spawn_leases


def test_http_half_close_after_commit_never_sends_voided_spawn(tmp_path, monkeypatch):
    import socket
    import threading
    from agent_company.runtime.relay import start_server
    relay, up = setup(tmp_path)
    committed, disconnected = threading.Event(), threading.Event()
    original_commit, original_event = relay.role_policy.commit, relay.event
    def commit(*a, **k):
        result = original_commit(*a, **k)
        committed.set()
        assert disconnected.wait(3), 'scripted client half-close was not observed'
        return result
    def event(event_name, **fields):
        result = original_event(event_name, **fields)
        if event_name == 'client_disconnected':
            disconnected.set()
        return result
    monkeypatch.setattr(relay.role_policy, 'commit', commit)
    monkeypatch.setattr(relay, 'event', event)
    up.queue.append(response(policy_and_spawn()))
    server = start_server(relay)
    try:
        with socket.create_connection(server.server_address, timeout=5) as client:
            body = json.dumps(request('master', ('agent', 'send_message'))).encode()
            headers = ('POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\n'
                       f'Authorization: Bearer {relay.token}\r\nContent-Length: {len(body)}\r\n'
                       'X-Agent-Company-Owner: master\r\nX-Agent-Company-Purpose: coding\r\n\r\n')
            client.sendall(headers.encode() + body)
            assert committed.wait(3)
            client.shutdown(socket.SHUT_WR)
            assert client.recv(65536) == b''  # no 200 or spawn after its binding was voided
        assert relay.role_policy.snapshot('master').global_policy.version == 1
        assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'voided'
        assert not relay._spawn_leases
        assert len([e for e in read_events(relay) if e['event'] == 'role_policy_response_not_delivered']) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_http_high_descriptor_is_not_mistaken_for_disconnected(tmp_path, monkeypatch):
    import fcntl
    import io
    import socket
    import time
    from types import SimpleNamespace
    import agent_company.runtime.relay as module
    relay, up = setup(tmp_path)
    captured = {}
    def server_factory(address, handler):
        captured['handler'] = handler
        return SimpleNamespace(serve_forever=lambda: None)
    monkeypatch.setattr(module, 'ThreadingHTTPServer', server_factory)
    module.start_server(relay)
    source, peer = socket.socketpair()
    connection = socket.socket(fileno=fcntl.fcntl(source, fcntl.F_DUPFD, 1024))
    try:
        assert connection.fileno() >= 1024
        handler = captured['handler'].__new__(captured['handler'])
        body = json.dumps(request('master', ('agent', 'send_message'))).encode()
        handler.path = '/v1/chat/completions'
        handler.headers = {'Authorization': 'Bearer ' + relay.token, 'Content-Length': str(len(body)),
                           'X-Agent-Company-Owner': 'master', 'X-Agent-Company-Purpose': 'coding'}
        handler.rfile, handler.wfile, handler.connection = io.BytesIO(body), io.BytesIO(), connection
        statuses = []
        handler.send_response = statuses.append
        handler.send_header = lambda *a: None
        handler.end_headers = lambda: None
        original = relay.handle
        def delayed_handle(*a, **k):
            time.sleep(0.15)  # watcher must examine the real high fd while the request is active
            return original(*a, **k)
        monkeypatch.setattr(relay, 'handle', delayed_handle)
        up.queue.append(response(policy_and_spawn()))
        handler.do_POST()
        assert statuses == [200]
        assert 'alice' in handler.wfile.getvalue().decode()
        assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'planned'
        assert not any(e['event'] == 'client_disconnected' for e in read_events(relay))
    finally:
        connection.close()
        source.close()
        peer.close()


@pytest.mark.parametrize('with_spawn', [False, True])
@pytest.mark.parametrize('failed_writer', ['response', 'event'])
def test_late_rejection_evidence_failure_has_commit_sensitive_status(tmp_path, monkeypatch, with_spawn, failed_writer):
    from agent_company.policy.role_package_store import parse_send_message
    relay, up = setup(tmp_path)
    original_commit, original_save, original_event = relay.role_policy.commit, relay.save, relay.event
    def competing_commit(staged, **kwargs):
        snapshot = relay.role_policy.snapshot('master')
        other = parse_send_message('competing', '*', 'GLOBAL-POLICY append\nConcurrent update')
        original_commit(relay.role_policy.stage('master', snapshot, [other]))
        result = original_commit(staged, **kwargs)
        def fail_rewrite(path, value):
            if failed_writer == 'response' and path.name.endswith('_qwen_response.json'):
                raise OSError('scripted late rejection save failure')
            return original_save(path, value)
        def fail_event(name, **fields):
            if failed_writer == 'event' and name == 'role_package_call_rejected':
                raise OSError('scripted late rejection event failure')
            return original_event(name, **fields)
        monkeypatch.setattr(relay, 'save', fail_rewrite)
        monkeypatch.setattr(relay, 'event', fail_event)
        return result
    monkeypatch.setattr(relay.role_policy, 'commit', competing_commit)
    calls = policy_and_spawn() if with_spawn else policy_and_spawn()[:1]
    up.queue.append(response(calls))
    status, delivered = relay.handle(request('master', ('agent','send_message')))
    assert status == (200 if with_spawn else 424)
    assert relay.summary()['role_policy']['counts']['evidence_failures'] == 1
    assert relay.role_policy.snapshot('master').global_policy.text == 'C\nConcurrent update'
    if with_spawn:
        assert TOOL_ERROR_FIELD in args(delivered, 0) and TOOL_ERROR_FIELD not in args(delivered, 1)
        assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'planned'
        assert len(relay._spawn_leases) == 1
    else:
        assert not relay._spawn_leases


def test_whole_response_capacity_rejection_has_no_local_rejection_event(tmp_path):
    from agent_company.runtime.relay import TEAMMATE_LIMIT
    relay, up = setup(tmp_path)
    for n in range(TEAMMATE_LIMIT):
        spawn(relay, up, f'worker{n}')
    bad = call('agent', {'name':'bad','description':'work','prompt':'ROLE-TYPE: missing'})
    good = call('agent', {'name':'alice','description':'work','prompt':'ROLE-TYPE: implementer'})
    status, delivered, _ = turn(relay, up, calls=[bad,good])
    assert status == 200 and TOOL_ERROR_FIELD in args(delivered, 1)
    assert not any(e['event'] == 'role_package_call_rejected' and e['tool_call_id'] == bad['id'] for e in read_events(relay))
    assert 'alice' not in relay.role_policy.export()['bindings']
    assert len(relay._spawn_leases) == TEAMMATE_LIMIT


@pytest.mark.parametrize('error_type', [BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError])
@pytest.mark.parametrize('audit_fails', [False, True])
def test_http_broken_pipe_voids_only_undelivered_planned_bindings(tmp_path, monkeypatch, error_type, audit_fails):
    import io
    import socket
    from types import SimpleNamespace
    import agent_company.runtime.relay as module
    relay, up = setup(tmp_path)
    captured = {}
    def server_factory(address, handler):
        captured['handler'] = handler
        return SimpleNamespace(serve_forever=lambda: None)
    monkeypatch.setattr(module, 'ThreadingHTTPServer', server_factory)
    module.start_server(relay)
    handler = captured['handler'].__new__(captured['handler'])
    body = json.dumps(request('master', ('agent','send_message'))).encode()
    handler.path = '/v1/chat/completions'
    handler.headers = {'Authorization':'Bearer ' + relay.token, 'Content-Length':str(len(body)),
                       'X-Agent-Company-Owner':'master', 'X-Agent-Company-Purpose':'coding'}
    handler.rfile = io.BytesIO(body)
    handler.connection, peer = socket.socketpair()
    handler.send_response = lambda *a: None
    handler.send_header = lambda *a: None
    handler.end_headers = lambda: None
    class BrokenWriter:
        def write(self, chunk):
            raise error_type('scripted HTTP write failure')
    handler.wfile = BrokenWriter()
    original_event = relay.event
    if audit_fails:
        def event(event_name, **fields):
            if event_name == 'client_disconnected':
                raise OSError('scripted disconnect audit failure')
            return original_event(event_name, **fields)
        monkeypatch.setattr(relay, 'event', event)
    up.queue.append(response(policy_and_spawn()))
    try:
        handler.do_POST()
    finally:
        handler.connection.close()
        peer.close()
    assert relay.role_policy.snapshot('master').global_policy.version == 1
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'voided'
    assert not relay._spawn_leases and not relay._policy_spawns
    assert any(e['event'] == 'role_policy_response_not_delivered' for e in read_events(relay))


@pytest.mark.parametrize('phase', ['commit', 'collaboration', 'outer_attempt'])
def test_cancel_audit_failure_cannot_escape_lease_cleanup(tmp_path, monkeypatch, phase):
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    original_event = relay.event
    def event(event_name, **fields):
        if event_name == 'native_request_ended':
            raise OSError('scripted cancellation audit failure')
        return original_event(event_name, **fields)
    monkeypatch.setattr(relay, 'event', event)
    if phase == 'commit':
        original_commit = relay.role_policy.commit
        def commit(*a, **k):
            control.cancel('scripted precommit cancellation')
            return original_commit(*a, **k)
        monkeypatch.setattr(relay.role_policy, 'commit', commit)
    elif phase == 'collaboration':
        original_save = relay.save
        def save(path, value):
            result = original_save(path, value)
            if path.name == 'usage.json' and not relay._active_requests:
                control.cancel('scripted precollaboration cancellation')
            return result
        monkeypatch.setattr(relay, 'save', save)
    else:
        original_attempt = relay._attempt
        def attempt(*a, **k):
            original_attempt(*a, **k)
            control.cancel('scripted outer cancellation after staging')
            control.check()
        monkeypatch.setattr(relay, '_attempt', attempt)
    up.queue.append(response(policy_and_spawn()))
    status, delivered = relay.handle(request('master', ('agent','send_message')), control=control)
    assert status == 424 and delivered['error']['code'] == 'evidence_write_failed'
    assert relay.role_policy.snapshot('master').global_policy.version == 0
    assert not relay.role_policy.export()['bindings']
    assert not relay._spawn_leases and not relay._policy_spawns
    assert not relay._active_requests


def test_partial_delivery_preserves_native_spawn_success_before_first_request(tmp_path):
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    calls = policy_and_spawn()
    up.queue.append(response(calls))
    assert relay.handle(request('master', ('agent','send_message')), control=control)[0] == 200
    relay.record_native_spawn_result({'requested_name':'alice', 'worker':'alice',
                                     'tool_call_id':calls[1]['id'], 'ok':True}, 'native lifecycle')
    relay.response_not_delivered(control)
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'planned'
    assert len(relay._spawn_leases) == 1 and len(relay._policy_spawns) == 1
    relay.role_policy.record_first_request('alice')
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'active'


def test_precollaboration_deadline_is_timeout_not_evidence_failure(tmp_path, monkeypatch):
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    original_save = relay.save
    def save(path, value):
        result = original_save(path, value)
        if path.name == 'usage.json' and not relay._active_requests:
            control._deadline = 0
        return result
    monkeypatch.setattr(relay, 'save', save)
    up.queue.append(response(policy_and_spawn()))
    status, delivered = relay.handle(request('master', ('agent','send_message')), control=control)
    assert status == 424 and delivered['error']['code'] == 'task_deadline_exceeded'
    assert relay.summary()['role_policy']['counts'].get('evidence_failures', 0) == 0
    assert not relay._spawn_leases and not relay.role_policy.export()['bindings']


def test_delivery_cleanup_is_atomic_against_native_confirmation(tmp_path, monkeypatch):
    import threading
    from agent_company.runtime.transport import RequestControl
    relay, up = setup(tmp_path)
    control = RequestControl(5000)
    calls = policy_and_spawn()
    up.queue.append(response(calls))
    assert relay.handle(request('master', ('agent','send_message')), control=control)[0] == 200
    attempted, finished = threading.Event(), threading.Event()
    threads = []
    original_writer = relay.role_policy._writer
    def native_callback():
        attempted.set()
        relay.record_native_spawn_result({'requested_name':'alice', 'worker':'alice',
                                         'tool_call_id':calls[1]['id'], 'ok':True}, 'native lifecycle')
        finished.set()
    def writer(record):
        if any(c['event'] == 'role_policy_binding_ended' for c in record.get('changes', [])):
            thread = threading.Thread(target=native_callback)
            threads.append(thread)
            thread.start()
            assert attempted.wait(1)
            assert not finished.wait(0.05)  # confirmation cannot mutate lease mid-void
        return original_writer(record)
    monkeypatch.setattr(relay.role_policy, '_writer', writer)
    relay.response_not_delivered(control)
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert finished.is_set()
    assert relay.role_policy.export()['bindings']['alice']['lifecycle'] == 'voided'
    assert not relay._spawn_leases and not relay._policy_spawns
