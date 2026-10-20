"""Provider-reported model snapshot is kept as attempt metadata (zero API)."""
from agent_company.runtime.relay import ProviderConfig, TeamRelay
from agent_company.runtime.team import WorkerRegistry
from test_team import request, response, workspace


def test_model_and_fingerprint_are_recorded_and_summarized(tmp_path):
    replies = []

    def upstream(payload):
        return 200, replies.pop(0)
    relay = TeamRelay(ProviderConfig('deepseek-flash', 'http://localhost', max_attempts=1),
                      WorkerRegistry(workspace(tmp_path)), '', tmp_path / 'relay', 500, upstream, execution_mode='single')
    for fingerprint in ('fp_old', 'fp_new', 'fp_new'):
        replies.append({**response([]), 'model': 'deepseek-flash', 'system_fingerprint': fingerprint})
        assert relay.handle(request('master', ('read_file',)))[0] == 200
    replies.append(response([]))
    relay.handle(request('master', ('read_file',)))
    records = relay._records
    assert records[0]['provider_identity'] == {'model': 'deepseek-flash', 'system_fingerprint': 'fp_old'}
    assert 'provider_identity' not in records[-1]
    assert relay.summary()['provider_identities'] == {'deepseek-flash|fp_old': 1, 'deepseek-flash|fp_new': 2}
