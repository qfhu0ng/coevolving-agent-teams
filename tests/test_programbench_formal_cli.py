"""Public launcher parsing and diagnostics without SSH or model calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import launch_programbench_formal as cli
from agent_company.deploy.deployment import Deployment


@pytest.mark.parametrize('abbreviation', ['--history', '--approved'])
def test_abbreviations_fail_before_remote_call(monkeypatch, abbreviation):
    monkeypatch.setattr(cli, '_remote_call', lambda *a, **k: pytest.fail('remote called'))
    with pytest.raises(SystemExit) as error:
        cli.main(['--final', '/server/final', '--prepare', '--runner', 'followup',
                  '--step', 'followup', '--profile', '/server/profile', '--repeat-index', '5',
                  '--source-run', '/server/source', abbreviation, '/server/trial'])
    assert error.value.code == 2


@pytest.mark.parametrize('local_option', [
    ['--local-record-dir', '/Users/workstation/private path'],
    ['--local-record-dir=/Users/workstation/private path'],
])
def test_local_record_directory_never_reaches_remote(local_option):
    observed = []
    def transport(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"verified"}')
    cli._remote_call(Deployment(Path('/server/final'), Path('/server/assets')),
                     ['--final', '/server/final', '--verify', *local_option], transport=transport)
    assert len(observed) == 1
    assert 'local-record-dir' not in observed[0][0]
    assert '/Users/' not in observed[0][0]
    assert observed[0][1]['stdin'] is None


@pytest.mark.parametrize('arguments, paid', [
    (['--verify'], False), (['--status'], False),
    (['--prepare', '--runner', 'comparison', '--step', 's0_propose',
      '--profile', '/server/profile', '--repeat-index', '0'], False),
    (['--smoke', '--arm', 'single', '--sequence', '1', '--scripted'], False),
    (['--probe', '--sequence', '1'], True),
    (['--smoke', '--arm', 'single', '--sequence', '1', '--real'], True),
    (['--launch', '--runner', 'comparison', '--step', 's0_propose',
      '--profile', '/server/profile', '--repeat-index', '0'], True),
])
def test_only_zero_api_failures_include_bounded_reason(monkeypatch, capsys, arguments, paid):
    reason = 'failure condition ' + 'x' * 1200
    def fail(*args, **kwargs):
        raise RuntimeError(reason)
    monkeypatch.setattr(cli, '_remote_call', fail)
    assert cli.main(['--final', '/server/final', *arguments]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['error_type'] == 'RuntimeError'
    if paid:
        assert 'error' not in report and reason[:20] not in json.dumps(report)
    else:
        assert report['error'] == reason[:1000]


def test_deploy_failure_includes_reason(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError('Remote deploy failed (exit 9): disk full')
    monkeypatch.setattr(cli, 'deploy', fail)
    assert cli.main(['--final', '/server/final', '--deploy', '--commit', 'a' * 40]) == 1
    assert json.loads(capsys.readouterr().out)['error'].endswith('disk full')


def test_paused_round_is_launcher_only(monkeypatch, capsys):
    observed = []
    def prepare(deployment, spec):
        observed.append(spec)
        return {'status': 'prepared_not_launched'}
    monkeypatch.setattr(cli, 'prepare', prepare)
    assert cli.main(['--final', '/server/final', '--remote', '--prepare', '--runner', 'comparison',
                     '--step', 'approved', '--profile', '/server/final/profiles/approved.json',
                     '--repeat-index', '7', '--paused-round', '/server/final/paid_runs/followup_r6/run']) == 0
    assert observed[0]['paused_round'].endswith('/followup_r6/run')
    assert observed[0]['arguments'] == ['--repeat-index', '7']
    capsys.readouterr()


@pytest.mark.parametrize('operation', [
    ['--probe', '--sequence', '2'],
    ['--smoke', '--real', '--arm', 'team_p0_propose', '--sequence', '2'],
])
def test_preparation_prepare_is_zero_api_and_never_reads_stdin(monkeypatch, capsys, operation):
    observed = []
    def preparation(deployment, **kwargs):
        assert kwargs['prepare_only'] is True
        observed.append(kwargs)
        return {'status': 'prepared_not_launched'}
    monkeypatch.setattr(cli, 'preparation', preparation)
    monkeypatch.setattr(cli.sys, 'stdin', SimpleNamespace(read=lambda *args: pytest.fail('stdin read')))
    assert cli.main(['--final', '/server/final', '--remote', '--prepare', *operation]) == 0
    assert observed and json.loads(capsys.readouterr().out)['status'] == 'prepared_not_launched'


@pytest.mark.parametrize('operation', [
    ['--probe', '--sequence', '2'],
    ['--smoke', '--real', '--arm', 'single', '--sequence', '1'],
])
def test_preparation_prepare_remote_call_is_not_paid(monkeypatch, capsys, operation):
    def remote(deployment, raw, *, paid):
        assert paid is False and '--prepare' in raw
        raise RuntimeError('preflight reason')
    monkeypatch.setattr(cli, '_remote_call', remote)
    assert cli.main(['--final', '/server/final', '--prepare', *operation]) == 1
    assert json.loads(capsys.readouterr().out)['error'] == 'preflight reason'


@pytest.mark.parametrize('operation', ['--deploy', '--launch', '--verify', '--status'])
def test_prepare_rejects_ambiguous_operation_before_remote(monkeypatch, operation):
    monkeypatch.setattr(cli, '_remote_call', lambda *a, **k: pytest.fail('remote called'))
    with pytest.raises(SystemExit) as error:
        cli.main(['--final', '/server/final', '--prepare', operation])
    assert error.value.code == 2
