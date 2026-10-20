"""Exercise the bounded diagnostic with a fake transport, never a provider."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import pytest

from agent_company.runtime.relay import ProviderConfig
from agent_company.runtime.transport import RequestProgressTimeout
from scripts import s0_cache_probe as probe


def harness(tmp_path, monkeypatch, responses, observations=None):
    output = tmp_path / 'probe'
    provider = ProviderConfig(
        'requested-alias', 'http://unused', api_key='synthetic-key', stream_response=True,
        extra_body={'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'})
    prepared = SimpleNamespace(
        tasks=[SimpleNamespace(task_id='synthetic__task.1234567', instruction='Public instruction')],
        plan={'provider_config': 'unused.json',
              'inference': {'role_policy_mode': 'propose', 'memory_mb': 61440,
                            'cpus': 20, 'agent_timeout_sec': 3600},
              'arms': [{'arm_id': 'team', 'execution_mode': 'team'}]},
        policies={'team': ('Assign independent work.', 'Share blockers.')})
    monkeypatch.setattr(probe, 'prepare_comparison', lambda *_: prepared)
    monkeypatch.setattr(probe, 'load_provider', lambda _: provider)
    calls, snapshots, sleeps = [], [], []
    monkeypatch.setattr(probe.time, 'sleep', sleeps.append)

    class Transport:
        def __init__(self, settings):
            assert settings.max_attempts == 1
            assert settings.request_timeout_seconds == 180
            assert settings.first_progress_timeout_seconds == 120
            assert settings.progress_idle_timeout_seconds == 120

        def __call__(self, payload, *, observation):
            # The pending attempt must already exist even for the first call.
            snapshots.append(json.loads((output / 'cache_probe.json').read_text()))
            index = len(calls)
            calls.append(deepcopy(payload))
            if index >= len(responses):
                pytest.fail('The probe made an unexpected transport call after its terminal outcome')
            if observations is not None:
                observation.update(deepcopy(observations[index]))
            response = responses[index]
            if isinstance(response, BaseException):
                raise response
            return deepcopy(response)

    monkeypatch.setattr(probe, 'OpenAIChatTransport', Transport)
    return SimpleNamespace(
        run=lambda: probe.main(['--profile', str(tmp_path / 'profile.json'), '--output', str(output)]),
        read=lambda: json.loads((output / 'cache_probe.json').read_text()),
        output=output, calls=calls, snapshots=snapshots, sleeps=sleeps)


def success(model='reported-version', cached=0):
    return 200, {'model': model, 'usage': {'prompt_tokens': 12, 'completion_tokens': 2,
                                         'prompt_cache_hit_tokens': cached},
                 'choices': [{'message': {'content': 'PRIVATE_OUTPUT synthetic-key',
                                           'reasoning_content': 'PRIVATE_THOUGHT synthetic-key'}}]}


def assert_utc(value):
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() == timedelta(0)
    return parsed


def assert_pending_snapshots(run):
    for number, snapshot in enumerate(run.snapshots):
        assert snapshot['calls'] == number
        assert snapshot['started_calls'] == number + 1
        assert len(snapshot['rows']) == number
        assert snapshot['status'] == 'running'
        assert snapshot['complete'] is False
        assert snapshot['request_settings']['max_tokens'] == 16
        pending = snapshot['current_call']
        assert set(pending) == {'phase', 'index', 'started_at_utc'}
        assert pending['phase'] == ('A_unchanged' if number < 3 else 'B_after_policy_update')
        assert pending['index'] == number % 3
        assert_utc(pending['started_at_utc'])


def test_six_valid_measurements_preserve_payloads_and_record_each_attempt(tmp_path, monkeypatch, capsys):
    run = harness(tmp_path, monkeypatch, [success(), success(cached=8), *[success() for _ in range(4)]])
    assert run.run() == 0
    report = run.read()
    assert (report['calls'], report['started_calls'], report['planned_calls']) == (6, 6, 6)
    assert report['status'] == 'ok' and report['complete'] is True
    assert report['current_call'] is None
    assert report['succeeded_calls'] == 6 and report['failed_calls'] == 0
    assert report['provider_model'] == 'requested-alias'
    assert report['response_models'] == ['reported-version']
    assert report['cache_reported'] is True
    assert report['rows'][0]['cached_tokens'] == 0  # A zero cache hit is a valid measurement.
    assert report['request_settings'] == {
        'max_attempts': 1, 'max_tokens': 16, 'request_timeout_seconds': 180,
        'first_progress_timeout_seconds': 120, 'progress_idle_timeout_seconds': 120}
    assert_pending_snapshots(run)
    for row, snapshot in zip(report['rows'], run.snapshots, strict=True):
        assert row['started_at_utc'] == snapshot['current_call']['started_at_utc']
        assert assert_utc(row['finished_at_utc']) >= assert_utc(row['started_at_utc'])
        assert row['max_tokens'] == 16 and row['timeout_phase'] is None
        assert row['error_type'] is None and row['issues'] == []
    assert run.calls[0] == run.calls[1] == run.calls[2]
    assert run.calls[3] == run.calls[4] == run.calls[5] and run.calls[0] != run.calls[3]
    assert all(payload['stream_options'] == {'include_usage': True} for payload in run.calls)
    assert all(payload['thinking'] == {'type': 'enabled'} for payload in run.calls)
    assert all(payload['reasoning_effort'] == 'high' for payload in run.calls)
    assert all(payload['max_tokens'] == 16 and payload['stream'] is True for payload in run.calls)
    assert run.sleeps == [2] * 5  # Neither failures nor the final successful call sleep.
    printed = capsys.readouterr().out
    assert json.loads(printed) == report
    assert 'PRIVATE_' not in printed and 'synthetic-key' not in printed


@pytest.mark.parametrize('position', [1, 2, 4, 6])
@pytest.mark.parametrize(('failure', 'error_type', 'http_status'), [
    (RuntimeError('PRIVATE_ERROR synthetic-key'), 'RuntimeError', None),
    ((401, {'error': {'message': 'PRIVATE_ERROR synthetic-key'}}), 'http_401', 401),
    ((200, {'error': {'message': 'PRIVATE_ERROR synthetic-key'}}), 'error_object', 200),
    ((200, ['PRIVATE_ERROR synthetic-key']), 'error_object', 200),
])
def test_failure_stops_without_another_call_or_sleep(
        tmp_path, monkeypatch, capsys, position, failure, error_type, http_status):
    run = harness(tmp_path, monkeypatch, [*[success() for _ in range(position - 1)], failure])
    assert run.run() == 1
    report = run.read()
    assert report['status'] == ('failed' if position == 1 else 'partial')
    assert report['complete'] is (position == 6) and report['current_call'] is None
    assert report['calls'] == report['started_calls'] == len(run.calls) == position
    assert report['succeeded_calls'] == position - 1 and report['failed_calls'] == 1
    assert run.sleeps == [2] * (position - 1)
    assert_pending_snapshots(run)
    failed = report['rows'][-1]
    assert failed['error_type'] == error_type and failed['http_status'] == http_status
    assert failed['timeout_phase'] is None
    assert failed['response_model'] is None
    assert all(failed[field] is None for field in (
        'prompt_tokens', 'completion_tokens', 'cache_field', 'cached_tokens', 'usage_keys'))
    assert assert_utc(failed['finished_at_utc']) >= assert_utc(failed['started_at_utc'])
    printed = capsys.readouterr().out
    assert json.loads(printed) == report
    assert 'PRIVATE_' not in printed and 'synthetic-key' not in printed


@pytest.mark.parametrize('phase', ['first_progress', 'progress_idle'])
@pytest.mark.parametrize('http_status', [None, 200])
def test_progress_timeout_preserves_phase_and_known_http_status_without_inventing_usage(
        tmp_path, monkeypatch, phase, http_status):
    observation = {'stop_reason': phase, 'response_complete': False, 'elapsed_seconds': 120.013,
                   'bytes_received': 128, 'sse_events': 2, 'response_format': 'sse',
                   'progress_observable': True, 'reasoning_content_progress_events': 1}
    if http_status is not None:
        observation['provider_http_status'] = http_status
    run = harness(tmp_path, monkeypatch, [RequestProgressTimeout(phase)], [observation])
    assert run.run() == 1
    row = run.read()['rows'][0]
    assert row['error_type'] == 'RequestProgressTimeout' and row['timeout_phase'] == phase
    assert row['http_status'] == http_status
    assert row['transport_observation'] == observation
    assert all(row[field] is None for field in ('prompt_tokens', 'completion_tokens', 'cached_tokens'))
    assert run.sleeps == []


def test_other_exception_does_not_claim_a_progress_timeout_phase(tmp_path, monkeypatch):
    failure = TimeoutError('PRIVATE_ERROR synthetic-key')
    failure.phase = 'first_progress'
    run = harness(tmp_path, monkeypatch, [failure], [{'provider_http_status': 503}])
    assert run.run() == 1
    row = run.read()['rows'][0]
    assert row['http_status'] == 503
    assert row['timeout_phase'] is None and row['error_type'] == 'TimeoutError'


def test_transport_observation_keeps_diagnostics_and_drops_arbitrary_text(tmp_path, monkeypatch, capsys):
    kept = {'request_started_at': 1.0, 'headers_received_at': 1.1, 'headers_elapsed_seconds': 0.1,
            'first_byte_at': 1.2, 'first_byte_elapsed_seconds': 0.2,
            'first_progress_at': 1.3, 'first_progress_elapsed_seconds': 0.3,
            'last_progress_at': 1.4, 'last_progress_elapsed_seconds': 0.4,
            'max_progress_gap_seconds': 0.1, 'elapsed_seconds': 0.5,
            'bytes_received': 512, 'sse_events': 4, 'provider_http_status': 200,
            'response_format': 'sse', 'progress_observable': True, 'response_complete': True,
            'stop_reason': 'completed', 'content_progress_events': 1,
            'reasoning_content_progress_events': 2, 'reasoning_progress_events': 0,
            'refusal_progress_events': 0, 'tool_name_progress_events': 0,
            'tool_arguments_progress_events': 0}
    unsafe = {'headers': {'Authorization': 'Bearer synthetic-key'},
              'content': 'PRIVATE_CONTENT', 'reasoning_content': 'PRIVATE_THOUGHT',
              'tool_arguments': 'PRIVATE_ARGUMENTS', 'exception': 'PRIVATE_ERROR synthetic-key'}
    run = harness(tmp_path, monkeypatch, [success() for _ in range(6)], [{**kept, **unsafe}] * 6)
    assert run.run() == 0
    assert all(row['transport_observation'] == kept for row in run.read()['rows'])
    printed = capsys.readouterr().out
    assert 'PRIVATE_' not in printed and 'synthetic-key' not in printed
    assert 'PRIVATE_' not in (run.output / 'cache_probe.json').read_text()
    assert 'synthetic-key' not in (run.output / 'cache_probe.json').read_text()


def test_invalid_observation_values_are_not_serialized(tmp_path, monkeypatch):
    invalid = {'provider_http_status': 'synthetic-key', 'stop_reason': 'PRIVATE_ERROR',
               'response_format': 'PRIVATE_FORMAT', 'elapsed_seconds': float('nan'),
               'max_progress_gap_seconds': float('inf'), 'headers_elapsed_seconds': {'secret': 'synthetic-key'}}
    run = harness(tmp_path, monkeypatch, [RuntimeError('synthetic-key')], [invalid])
    assert run.run() == 1
    row = run.read()['rows'][0]
    assert row['transport_observation'] == {} and row['http_status'] is None


@pytest.mark.parametrize('position', [1, 2, 4])
@pytest.mark.parametrize('interruption', [KeyboardInterrupt('PRIVATE_INTERRUPT'), SystemExit(17)])
def test_base_exception_propagates_with_pending_attempt_snapshot(
        tmp_path, monkeypatch, capsys, position, interruption):
    run = harness(tmp_path, monkeypatch, [*[success() for _ in range(position - 1)], interruption])
    with pytest.raises(type(interruption)) as raised:
        run.run()
    assert raised.value is interruption
    report = run.read()
    assert report == run.snapshots[-1]
    assert report['calls'] == position - 1 and report['started_calls'] == position
    assert report['succeeded_calls'] == position - 1 and report['failed_calls'] == 0
    assert report['status'] == 'running' and report['complete'] is False
    assert len(run.calls) == position and run.sleeps == [2] * (position - 1)
    assert_pending_snapshots(run)
    assert capsys.readouterr().out == ''
    assert 'PRIVATE_INTERRUPT' not in (run.output / 'cache_probe.json').read_text()


@pytest.mark.parametrize('model', [None, '', '  ', False, 3])
def test_missing_or_invalid_model_stops_measurement(tmp_path, monkeypatch, model):
    run = harness(tmp_path, monkeypatch, [success(model=model)])
    assert run.run() == 1
    report = run.read()
    row = report['rows'][0]
    assert row['error_type'] == 'invalid_measurement'
    assert row['issues'] == ['missing_response_model'] and row['response_model'] is None
    assert report['status'] == 'failed' and report['complete'] is False
    assert len(run.calls) == 1 and run.sleeps == []


def test_absent_model_stops_measurement(tmp_path, monkeypatch):
    response = success()[1]
    del response['model']
    run = harness(tmp_path, monkeypatch, [(200, response)])
    assert run.run() == 1
    assert run.read()['rows'][0]['issues'] == ['missing_response_model']


@pytest.mark.parametrize('usage', [None, [], 'PRIVATE_USAGE synthetic-key', {}])
def test_missing_or_invalid_usage_stops_without_replacing_unknown_with_zero(tmp_path, monkeypatch, usage):
    response = success()[1]
    response['usage'] = usage
    run = harness(tmp_path, monkeypatch, [(200, response)])
    assert run.run() == 1
    report = run.read()
    row = report['rows'][0]
    assert row['error_type'] == 'invalid_measurement'
    assert row['issues'] == ['invalid_prompt_tokens', 'invalid_completion_tokens',
                             'invalid_cached_tokens', 'missing_cache_field']
    assert all(row[key] is None for key in ('prompt_tokens', 'completion_tokens', 'cached_tokens', 'cache_field'))
    assert report['cache_reported'] is False and run.sleeps == []
    assert 'PRIVATE_USAGE' not in json.dumps(report) and 'synthetic-key' not in json.dumps(report)


@pytest.mark.parametrize('field', ['prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens'])
def test_absent_usage_field_is_unknown_and_stops(tmp_path, monkeypatch, field):
    response = success()[1]
    del response['usage'][field]
    run = harness(tmp_path, monkeypatch, [(200, response)])
    assert run.run() == 1
    row = run.read()['rows'][0]
    output_field = 'cached_tokens' if field == 'prompt_cache_hit_tokens' else field
    expected = ['invalid_' + output_field]
    if output_field == 'cached_tokens':
        expected.append('missing_cache_field')
    assert row['error_type'] == 'invalid_measurement' and row['issues'] == expected
    assert row[output_field] is None and len(run.calls) == 1 and run.sleeps == []


@pytest.mark.parametrize('field', ['prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens'])
@pytest.mark.parametrize('invalid', [None, True, False, -1, 1.5, 'PRIVATE_COUNT synthetic-key',
                                   float('nan'), float('inf')])
def test_token_counts_must_be_nonnegative_integers_without_bool_coercion(
        tmp_path, monkeypatch, field, invalid):
    response = success()[1]
    response['usage'][field] = invalid
    run = harness(tmp_path, monkeypatch, [(200, response)])
    assert run.run() == 1
    report = run.read()
    row = report['rows'][0]
    output_field = 'cached_tokens' if field == 'prompt_cache_hit_tokens' else field
    assert row['error_type'] == 'invalid_measurement'
    assert row['issues'] == ['invalid_' + output_field] and row[output_field] is None
    assert report['status'] == 'failed' and report['calls'] == 1
    assert 'PRIVATE_COUNT' not in json.dumps(report) and 'synthetic-key' not in json.dumps(report)


def test_cache_hits_cannot_exceed_prompt_tokens(tmp_path, monkeypatch):
    run = harness(tmp_path, monkeypatch, [success(cached=13)])
    assert run.run() == 1
    row = run.read()['rows'][0]
    assert row['error_type'] == 'invalid_measurement'
    assert row['issues'] == ['cached_tokens_exceed_prompt_tokens']
    assert row['prompt_tokens'] == 12 and row['cached_tokens'] == 13
    assert run.sleeps == []


@pytest.mark.parametrize('position', [2, 4])
def test_response_model_change_stops_before_next_call_in_either_phase(tmp_path, monkeypatch, position):
    run = harness(tmp_path, monkeypatch, [*[success() for _ in range(position - 1)],
                                         success(model='different-version')])
    assert run.run() == 1
    report = run.read()
    assert report['status'] == 'partial' and report['complete'] is False
    assert report['rows'][-1]['error_type'] == 'invalid_measurement'
    assert report['rows'][-1]['issues'] == ['response_model_changed']
    assert report['response_models'] == ['different-version', 'reported-version']
    assert len(run.calls) == position and run.sleeps == [2] * (position - 1)


def test_cached_tokens_fallback_and_zero_counts_are_valid(tmp_path, monkeypatch):
    response = success()[1]
    response['usage'] = {'prompt_tokens': 0, 'completion_tokens': 0,
                         'prompt_tokens_details': {'cached_tokens': 0}}
    run = harness(tmp_path, monkeypatch, [(200, response)] * 6)
    assert run.run() == 0
    report = run.read()
    assert report['status'] == 'ok' and report['cache_reported'] is True
    assert all(row['cache_field'] == 'prompt_tokens_details.cached_tokens' for row in report['rows'])
    assert all(row['prompt_tokens'] == row['completion_tokens'] == row['cached_tokens'] == 0
               for row in report['rows'])


def test_provider_cache_field_takes_precedence_over_nested_fallback(tmp_path, monkeypatch):
    response = success()[1]
    response['usage']['prompt_tokens_details'] = {'cached_tokens': 9}
    run = harness(tmp_path, monkeypatch, [(200, response)] * 6)
    assert run.run() == 0
    assert all(row['cache_field'] == 'prompt_cache_hit_tokens' and row['cached_tokens'] == 0
               for row in run.read()['rows'])
