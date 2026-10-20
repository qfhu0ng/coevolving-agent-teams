"""Configurable active-teammate limit (plan v4 section 10); the default keeps earlier texts (zero API)."""
import json

import pytest

from agent_company.policy.role_package import create_m0_package
from agent_company.runtime.relay import (TEAM_SIZE_NOTE, TEAMMATE_LIMIT_MESSAGE, TOOL_ERROR_FIELD, ProviderConfig,
                                         TeamRelay, describe_team_create, team_size_note, teammate_limit_message)
from agent_company.runtime.team import WorkerRegistry
from test_agent_rules_launch import launch
from test_team import call, request, response, workspace


def test_default_texts_are_unchanged():
    assert team_size_note() == TEAM_SIZE_NOTE and teammate_limit_message() == TEAMMATE_LIMIT_MESSAGE
    assert 'At most 10 teammates' in TEAM_SIZE_NOTE and 'at most 10 teammates' in TEAMMATE_LIMIT_MESSAGE
    tool = {'function': {'description': 'Create a team (max 5 teammates per team).'}}
    describe_team_create(tool)
    assert '(at most 10 active teammates at a time)' in tool['function']['description']


def test_relay_enforces_and_describes_the_configured_limit(tmp_path):
    replies, payloads = [], []

    def upstream(payload):
        payloads.append(payload)
        return 200, replies.pop(0)
    relay = TeamRelay(ProviderConfig('fake', 'http://localhost', max_attempts=1), WorkerRegistry(workspace(tmp_path)),
                      'C', tmp_path / 'relay', 500, upstream, teammate_limit=2)
    spawns = [call('agent', {'name': name, 'description': 'w', 'prompt': 'Work.'}) for name in ('a', 'b', 'c')]
    replies.append(response(spawns[:2]))
    assert relay.handle(request('master', ('agent', 'team_create')))[0] == 200
    replies.append(response(spawns[2:]))
    _, delivered = relay.handle(request('master', ('agent', 'team_create')))
    error = json.loads(delivered['choices'][0]['message']['tool_calls'][0]['function']['arguments'])[TOOL_ERROR_FIELD]
    assert 'at most 2 teammates' in error
    created = next(t for t in payloads[-1]['tools'] if t['function']['name'] == 'team_create')
    assert team_size_note(2) in created['function']['description']
    assert relay.team_status()['teammate_limit'] == 2
    (tmp_path / 'x').mkdir()
    with pytest.raises(ValueError):
        TeamRelay(ProviderConfig('fake', 'http://localhost'), WorkerRegistry(workspace(tmp_path / 'x')), 'C',
                  tmp_path / 'r2', 5, upstream, teammate_limit=0)


def test_launch_states_the_configured_limit_everywhere(tmp_path):
    record = launch(tmp_path, create_m0_package('A', 'C'), checkin_interval_seconds=1800,
                    team_message_delivery='between_tool_calls', teammate_limit=32)
    system = record['command'][record['command'].index('--system-prompt') + 1]
    assert record['teammate_limit'] == 32 and record['environment']['AGENT_COMPANY_TEAMMATE_LIMIT'] == '32'
    assert team_size_note(32) in system and 'At most 10 teammates' not in system
    assert 'At most 32 teammates can be active at the same time; shut-down teammates' in system


def test_default_launch_has_no_limit_variable(tmp_path):
    record = launch(tmp_path, create_m0_package('A', 'C'), checkin_interval_seconds=1800,
                    team_message_delivery='between_tool_calls')
    assert record['teammate_limit'] == 10 and 'AGENT_COMPANY_TEAMMATE_LIMIT' not in record['environment']


def test_profile_and_runner_accept_the_setting():
    from agent_company.programbench.comparison import runner_inference_options
    base = {'memory_mb': 1024, 'cpus': 2, 'agent_timeout_sec': 60}
    assert runner_inference_options({**base, 'teammate_limit': 32})['teammate_limit'] == 32
    assert 'teammate_limit' not in runner_inference_options(base)
    with pytest.raises(ValueError):
        runner_inference_options({**base, 'teammate_limit': True})
