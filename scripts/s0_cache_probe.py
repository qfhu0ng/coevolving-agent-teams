"""Explicit-profile prompt-cache diagnostic; makes at most six real provider calls.

Not part of the zero-API mechanism smoke. Requires separate authorization. Phase A
sends one fixed master request three times; phase B commits one policy change and
sends the new request three times. Only response model identity, usage, cache-field
presence and content-free transport diagnostics are kept, never generated text.
Each attempt is recorded before calling the transport and is not retried. A failed
request or unknown/invalid measurement is saved and stops the batch immediately.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from agent_company.runtime.provider import load_provider
from agent_company.programbench.comparison import prepare_comparison, runner_inference_options
from agent_company.runtime.qwen import master_system
from agent_company.policy.role_policy import RolePolicyStore, parse_send_message, parse_spawn
from agent_company.policy.role_policy_runtime import render_injection
from agent_company.policy.source_boundary import append_system_text
from agent_company.runtime.transport import OpenAIChatTransport, RequestProgressTimeout


def transport_observation(observation):
    """Keep only the existing transport's content-free timing/counter fields."""
    fields = {
        'request_started_at', 'provider_http_status', 'bytes_received', 'sse_events',
        'headers_received_at', 'headers_elapsed_seconds', 'first_byte_at', 'first_byte_elapsed_seconds',
        'first_progress_at', 'first_progress_elapsed_seconds', 'last_progress_at', 'last_progress_elapsed_seconds',
        'max_progress_gap_seconds', 'elapsed_seconds', 'response_format', 'progress_observable',
        'response_complete', 'stop_reason', 'content_progress_events', 'reasoning_content_progress_events',
        'reasoning_progress_events', 'refusal_progress_events', 'tool_name_progress_events',
        'tool_arguments_progress_events',
    }
    enums = ('unknown', 'sse', 'json', 'in_flight', 'completed', 'http_error', 'cancelled',
             'total_deadline', 'first_progress', 'progress_idle', 'invalid_response', 'transport_error')
    return {key: value for key, value in observation.items() if key in fields and (
        type(value) in (int, bool) or type(value) is float and math.isfinite(value)
        or isinstance(value, str) and value in enums)}


def token_count(value):
    return value if type(value) is int and value >= 0 else None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cache_field(usage: dict):
    if 'prompt_cache_hit_tokens' in usage:
        return 'prompt_cache_hit_tokens', usage['prompt_cache_hit_tokens']
    details = usage.get('prompt_tokens_details')
    if isinstance(details, dict) and 'cached_tokens' in details:
        return 'prompt_tokens_details.cached_tokens', details['cached_tokens']
    return None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    prepared = prepare_comparison(args.profile.resolve(), CODE)
    plan = prepared.plan
    if "role_policy_protocol" in plan['inference']:
        parser.error('This legacy cache probe does not support role_policy_protocol')
    if len(prepared.tasks) != 1 or len(plan['arms']) != 1 or plan['arms'][0]['execution_mode'] != 'team':
        parser.error('--profile must select exactly one ProgramBench task and one Team arm')
    mode = runner_inference_options(plan['inference'])['role_policy_mode']
    if mode not in ('global', 'master', 'propose'):
        parser.error('--profile must enable role policies')
    (arm,) = plan['arms']
    (task,) = prepared.tasks
    assignment, communication = prepared.policies[arm['arm_id']]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provider = replace(load_provider(Path(plan['provider_config'])), max_attempts=1,
                       request_timeout_seconds=180, first_progress_timeout_seconds=120,
                       progress_idle_timeout_seconds=120)
    transport = OpenAIChatTransport(provider)
    store = RolePolicyStore(mode, communication, lambda record: None)

    def commit(actor, calls):
        store.commit(store.stage(actor, store.snapshot(actor), calls))

    initial = 'Work on the assigned component.'
    if mode != 'global':
        initial += '\nROLE-POLICY:\nReport interface changes to downstream.'
    commit('master', [parse_spawn('s1', 'component', initial), parse_spawn('s2', 'downstream', 'Use the interface.')])
    store.record_first_request('component')
    store.record_first_request('downstream')
    static = master_system(assignment, communication, communication_in_runtime=True)
    user = ('Context for this probe (do not act on it):\n' + task.instruction + '\n\n' + assignment
            + '\n\nReply with the single word OK.')

    def request():
        messages = [{'role': 'system', 'content': static}, {'role': 'user', 'content': user}]
        append_system_text(messages, render_injection(store.snapshot('master')))
        payload = {'model': provider.model, 'messages': messages, 'max_tokens': 16,
                   'stream': provider.stream_response, **provider.extra_body}
        if provider.stream_response:
            payload['stream_options'] = {'include_usage': True}
        return payload

    rows = []
    started_calls = 0
    current_call = None
    path = output / 'cache_probe.json'

    def save(terminal=False):
        succeeded = sum(row['ok'] for row in rows)
        failed = len(rows) - succeeded
        status = ('running' if not terminal else 'ok' if failed == 0
                  else 'failed' if succeeded == 0 else 'partial')
        summary = {'provider_model': provider.model, 'profile': str(args.profile.resolve()),
                   'task_id': task.task_id, 'mode': mode, 'planned_calls': 6, 'calls': len(rows),
                   'complete': len(rows) == 6, 'started_calls': started_calls, 'current_call': current_call,
                   'request_settings': {'max_attempts': provider.max_attempts,
                       'max_tokens': payload['max_tokens'],
                       'request_timeout_seconds': provider.request_timeout_seconds,
                       'first_progress_timeout_seconds': provider.first_progress_timeout_seconds,
                       'progress_idle_timeout_seconds': provider.progress_idle_timeout_seconds},
                   'succeeded_calls': succeeded, 'failed_calls': failed, 'status': status, 'rows': rows,
                   'cache_reported': any(row.get('cache_field') for row in rows if row['ok']),
                   'response_models': sorted({row['response_model'] for row in rows
                                              if row.get('response_model') is not None}),
                   'note': 'Diagnostic only; no cache-hit threshold or retries. A failure or invalid measurement stops '
                           'the batch. started_calls counts attempts recorded before transport invocation, not HTTP sends; '
                           'current_call is unfinished evidence, not proof of a live process. Output text is not stored.'}
        path.write_text(json.dumps(summary, indent=2) + '\n')
        return summary

    for phase in ('A_unchanged', 'B_after_policy_update'):
        if phase.startswith('B'):
            control = (parse_send_message('m1', '*', 'GLOBAL-POLICY\nReport only blockers and results.')
                       if mode == 'global' else
                       parse_send_message('m1', 'component', 'ROLE-POLICY component\nMessage downstream first.'))
            commit('master', [control])
        payload = request()
        system_chars = len(payload['messages'][0]['content'])
        for index in range(3):
            current_call = {'phase': phase, 'index': index, 'started_at_utc': utc_now()}
            started_calls += 1
            save()  # Persist the current attempt before any HTTP request can be made.
            started = time.monotonic()
            row = {'phase': phase, 'index': index, 'system_chars': system_chars,
                   'response_model': None, 'started_at_utc': current_call['started_at_utc'],
                   'max_tokens': payload['max_tokens'], 'timeout_phase': None, 'issues': []}
            unknown = dict(prompt_tokens=None, completion_tokens=None, cache_field=None, cached_tokens=None,
                           usage_keys=None)
            observation = {}
            try:
                status, response = transport(payload, observation=observation)
            except Exception as exc:
                row.update(ok=False, http_status=transport_observation(observation).get('provider_http_status'),
                           error_type=type(exc).__name__, **unknown,
                           latency_s=round(time.monotonic() - started, 3))
                if isinstance(exc, RequestProgressTimeout):
                    row['timeout_phase'] = exc.phase
            else:
                latency = round(time.monotonic() - started, 3)
                error_object = not isinstance(response, dict) or response.get('error') is not None
                if status != 200 or error_object:
                    row.update(ok=False, http_status=status, latency_s=latency, **unknown,
                               error_type=f'http_{status}' if status != 200 else 'error_object')
                else:
                    usage = response.get('usage')
                    usage = usage if isinstance(usage, dict) else {}
                    response_model = response.get('model')
                    field, cached = cache_field(usage)
                    row.update(http_status=status, latency_s=latency,
                               response_model=(response_model if isinstance(response_model, str)
                                               and response_model.strip() else None),
                               prompt_tokens=token_count(usage.get('prompt_tokens')),
                               completion_tokens=token_count(usage.get('completion_tokens')),
                               cache_field=field, cached_tokens=token_count(cached), usage_keys=sorted(usage))
                    if row['response_model'] is None:
                        row['issues'].append('missing_response_model')
                    elif rows and row['response_model'] != rows[0]['response_model']:
                        row['issues'].append('response_model_changed')
                    for name in ('prompt_tokens', 'completion_tokens', 'cached_tokens'):
                        if row[name] is None:
                            row['issues'].append('invalid_' + name)
                    if field is None:
                        row['issues'].append('missing_cache_field')
                    if row['cached_tokens'] is not None and row['prompt_tokens'] is not None and (
                            row['cached_tokens'] > row['prompt_tokens']):
                        row['issues'].append('cached_tokens_exceed_prompt_tokens')
                    row.update(ok=not row['issues'], error_type='invalid_measurement' if row['issues'] else None)
            row.update(finished_at_utc=utc_now(), transport_observation=transport_observation(observation))
            rows.append(row)
            current_call = None
            summary = save(terminal=not row['ok'] or len(rows) == 6)
            if not row['ok']:
                print(json.dumps(summary), flush=True)
                return 1
            if len(rows) < 6:
                time.sleep(2)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
