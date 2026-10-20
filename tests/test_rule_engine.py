"""Rule obligations for agent_rules_v1: triggers, windows, satisfaction and settlement (zero API)."""
import pytest

from agent_company.runtime.rule_engine import RuleEngine, field_problems, labeled_lines, message_hash


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class Rig:
    def __init__(self, rules=None, duration=3600):
        self.clock, self.events, self.keys = Clock(), [], 0
        self.policies = {'impl': (1, tuple(rules or [('R1', 'WHEN commit THEN SEND TO integ WITH sha')]))}
        self.engine = RuleEngine(lambda worker: self.policies.get(worker), self.write, started_at=self.clock.now,
                                 duration_seconds=duration, clock=self.clock)
        for name in ('impl', 'integ'):
            self.engine.worker_spawned(name)
            self.engine.coding_request(name)

    def write(self, event, **fields):
        self.events.append({'event': event, **fields})

    def names(self):
        return [e['event'] for e in self.events]

    def obligation(self, index=0):
        return list(self.engine.obligations.values())[index]

    def send(self, to, message, outcome='sent', worker='impl', resolve=True):
        self.keys += 1
        key = ('r', f'c{self.keys}')
        self.engine.record_send(worker, key, to=to, message=message)
        if resolve:
            self.engine.resolve_send(worker, key, outcome)
        return key

    def remind(self, worker='impl'):
        return [self.engine.notice_injected(worker, o['id']) for o in self.engine.obligations.values()
                if o['worker'] == worker]

    def turns(self, count, step=0, worker='impl'):
        """count requests, each followed by its delivered response."""
        for _ in range(count):
            self.clock.now += step
            self.engine.coding_request(worker)
            self.engine.coding_response(worker)


def test_commit_reminder_and_satisfaction_after_reminder():
    rig = Rig()
    reminders = rig.engine.observe_commit('impl', 'a' * 40, ['src/A.hs'])
    assert reminders == [('impl', 'O1', 'Rule R1 applies now: a new commit aaaaaaaaaaaa appeared in your worktree. '
                                       'Send integ a message with the labeled lines SHA:.')]
    assert rig.remind() == [True]
    rig.clock.now += 30
    rig.send('integ', 'Done.\nSHA: aaaaaaa')
    obligation = rig.obligation()
    assert obligation['status'] == 'satisfied' and obligation['resolution']['spontaneous'] is False
    assert obligation['resolution']['latency_s'] == 30


def test_report_sent_before_the_sample_counts_and_withdraws_the_reminder():
    """M2 review B1: commit at T0, report at T0+40, sample observed later."""
    rig = Rig()
    rig.clock.now += 40
    rig.send('integ', 'SHA: ' + 'b' * 12)
    rig.clock.now += 20
    assert rig.engine.observe_commit('impl', 'b' * 40, [], committed_at=rig.clock.now - 60) == []
    assert rig.obligation()['status'] == 'satisfied' and rig.obligation()['resolution']['spontaneous'] is True
    assert rig.remind() == [False] and 'rule_reminder_withdrawn' in rig.names()


def test_a_send_before_the_event_itself_never_counts():
    rig = Rig()
    rig.send('integ', 'SHA: ' + 'c' * 12)
    rig.clock.now += 10
    rig.engine.observe_commit('impl', 'c' * 40, [], committed_at=rig.clock.now)
    assert rig.obligation()['status'] == 'open'


def test_task_done_and_report_in_one_response():
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO leader WITH remaining')])
    response_at = rig.clock.now
    rig.send('leader', 'Task done.\nREMAINING: none')
    rig.clock.now += 5
    assert rig.engine.observe_task('impl', 3, 'task_done', occurred_at=response_at) == []
    assert rig.obligation()['status'] == 'satisfied'
    assert rig.engine.observe_task('impl', 3, 'task_done', occurred_at=response_at) == []  # deduped per task/owner
    assert len(rig.engine.obligations) == 1


def test_coalescing_absorbs_triggers_into_a_settled_obligation_within_ten_minutes():
    rig = Rig([('R1', 'WHEN message FROM integ THEN SEND TO integ')])
    rig.engine.observe_message('impl', 'integ')
    rig.clock.now += 20
    rig.send('integ', 'ack')
    rig.clock.now += 40
    assert rig.engine.observe_message('impl', 'integ') == []  # no new obligation, no reminder
    assert len(rig.engine.obligations) == 1 and len(rig.obligation()['events']) == 2
    assert rig.events[-1]['event'] == 'rule_coalesced' and rig.events[-1]['status'] == 'satisfied'
    rig.clock.now += 600
    assert len(rig.engine.observe_message('impl', 'integ')) == 1  # 660 s after the first


def test_coalesced_commit_rechecks_earlier_reports():
    rig = Rig()
    rig.engine.observe_commit('impl', '1' * 40, [])
    rig.clock.now += 30
    rig.send('integ', 'SHA: 2222222')  # names the second commit, sampled later
    rig.clock.now += 30
    rig.engine.observe_commit('impl', '2' * 40, [], committed_at=rig.clock.now - 50)
    assert rig.obligation()['status'] == 'satisfied' and rig.obligation()['incomplete_attempts'] == 1


def test_window_counts_delivered_responses_after_the_reminder():
    rig = Rig()
    rig.engine.observe_commit('impl', 'd' * 40, [])
    rig.engine.coding_request('impl')
    rig.remind()
    rig.engine.coding_response('impl')  # the reminder request's own response
    rig.send('integ', 'SHA: ' + 'e' * 12)
    assert rig.obligation()['incomplete_attempts'] == 1 and 'rule_send_incomplete' in rig.names()
    rig.clock.now += 1000
    rig.turns(2)
    assert rig.obligation()['status'] == 'open'  # 3 responses so far, 4 needed
    rig.turns(1)
    assert rig.obligation()['status'] == 'open'  # judged at the next request, after this response's sends
    rig.engine.coding_request('impl')
    assert rig.obligation()['status'] == 'missed' and rig.obligation()['resolution']['incomplete_attempts'] == 1


def test_failed_requests_do_not_count_and_time_alone_is_not_enough():
    rig = Rig()
    rig.engine.observe_commit('impl', 'f' * 40, [])
    rig.engine.coding_request('impl')
    rig.remind()
    for _ in range(6):
        rig.engine.coding_request('impl')  # no delivered response
    rig.clock.now += 5000
    rig.engine.coding_request('impl')
    assert rig.obligation()['status'] == 'open' and rig.obligation()['reminder_base'] is None
    rig.engine.worker_ended('impl')
    assert rig.obligation()['status'] == 'censored' and rig.obligation()['resolution']['reason'] == 'worker_ended'


def test_expired_obligation_stays_missed_when_the_worker_ends_later():
    rig = Rig()
    rig.engine.observe_commit('impl', '0f' * 20, [])
    rig.remind()
    rig.clock.now += 901
    rig.turns(4)
    rig.engine.coding_request('impl')
    assert rig.obligation()['status'] == 'missed'
    rig.engine.worker_ended('impl')
    assert rig.obligation()['status'] == 'missed'


def test_unreminded_and_recipient_endings():
    rig = Rig()
    rig.engine.observe_commit('impl', '9' * 40, [])
    rig.engine.worker_ended('integ')
    assert rig.obligation()['status'] == 'censored' and rig.obligation()['resolution']['reason'] == 'recipient_ended'
    rig.clock.now += 601  # past the coalescing window of O1
    rig.engine.observe_commit('impl', '8' * 40, [])
    assert rig.obligation(1)['status'] == 'censored' and rig.obligation(1)['resolution']['reason'] == 'recipient_not_active'
    assert rig.remind() == [False, False]
    alone = Rig()
    alone.engine.observe_commit('impl', '7' * 40, [])
    alone.engine.worker_ended('impl')
    assert alone.obligation()['resolution']['reason'] == 'never_reminded'


def test_final_response_send_with_unknown_outcome_is_unobservable_not_censored():
    """M2 review M2: the sender is aborted right after its last reply."""
    rig = Rig([('R1', 'WHEN shutdown_request THEN SEND TO leader WITH remaining')])
    rig.engine.observe_shutdown('impl')
    rig.remind()
    rig.send('leader', 'shutdown_approved\nREMAINING: none', resolve=False)
    rig.engine.worker_ended('impl')
    assert rig.obligation()['status'] == 'unobservable' and rig.obligation()['resolution']['reason'] == 'undecided_send'


def test_native_execution_record_resolves_a_send_without_history():
    rig = Rig([('R1', 'WHEN shutdown_request THEN SEND TO leader WITH remaining')])
    rig.engine.observe_shutdown('impl')
    message = 'shutdown_approved\nREMAINING: none'
    rig.send('Leader', message, resolve=False)
    rig.engine.resolve_native('impl', 'leader', message_hash(message), 'sent')
    assert rig.obligation()['status'] == 'satisfied' and not rig.engine.pending_sends


def test_broadcasts_partial_results_and_failures():
    rig = Rig()
    rig.engine.observe_commit('impl', 'a1' * 20, [])
    rig.send('integ', 'SHA: a1a1a1a', outcome='send_failed')
    assert rig.obligation()['status'] == 'open'
    rig.send('*', 'SHA: a1a1a1a')
    assert rig.obligation()['status'] == 'satisfied'
    exact = Rig()
    exact.engine.observe_commit('impl', 'b2' * 20, [])
    key = exact.send('*', 'SHA: b2b2b2b', resolve=False)
    exact.engine.resolve_native('impl', '*', message_hash('SHA: b2b2b2b'), 'send_partial', recipients=['other'])
    assert exact.obligation()['status'] == 'open' and key
    unknown = Rig()
    unknown.engine.observe_commit('impl', 'c3' * 20, [])
    unknown.remind()
    unknown.send('*', 'SHA: c3c3c3c', outcome='send_partial')
    unknown.clock.now += 901
    unknown.turns(4)
    unknown.engine.coding_request('impl')
    assert unknown.obligation()['status'] == 'unobservable'


def test_star_rules_need_a_full_broadcast():
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO *')])
    rig.engine.observe_task('impl', 4, 'task_done')
    rig.remind()
    rig.send('*', 'Task 4 is done.', outcome='send_partial')
    rig.engine.resolve_native('impl', '*', message_hash('Task 4 is done.'), 'send_partial', recipients=['leader'])
    rig.clock.now += 901
    rig.turns(4)
    rig.engine.coding_request('impl')
    assert rig.obligation()['status'] == 'missed'


def test_send_only_rules_and_recipient_names_ignore_case():
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO leader'), ('R2', 'WHEN task_done THEN SEND TO integ')])
    reminders = rig.engine.observe_task('impl', 4, 'task_done')
    assert [r[1] for r in reminders] == ['O1', 'O2'] and 'the master (to="leader")' in reminders[0][2]
    rig.send('Leader', 'Task 4 is done.')
    rig.send('INTEG', 'Task 4 is done.')
    assert [o['status'] for o in rig.engine.obligations.values()] == ['satisfied', 'satisfied']


@pytest.mark.parametrize('rules,call,expected', [
    ([('R1', 'WHEN commit PATHS src/Readers/** THEN SEND TO integ')], ('commit', ['src/Writers/A.hs']), 0),
    ([('R1', 'WHEN commit PATHS src/Readers/** THEN SEND TO integ')], ('commit', ['docs/x', 'src/Readers/B.hs']), 1),
    ([('R1', 'WHEN message FROM integ THEN SEND TO integ')], ('message', 'leader'), 0),
    ([('R1', 'WHEN message FROM integ THEN SEND TO integ')], ('message', 'Integ'), 1),
    ([('R1', 'WHEN message FROM * THEN SEND TO integ')], ('message', 'leader'), 1),
    ([('R1', 'WHEN task_claimed THEN SEND TO leader')], ('task', 'task_done'), 0),
    ([('R1', 'WHEN shutdown_request THEN SEND TO leader WITH remaining')], ('shutdown', None), 1),
])
def test_trigger_filters(rules, call, expected):
    rig = Rig(rules)
    kind, value = call
    result = (rig.engine.observe_commit('impl', 'ab' * 20, value) if kind == 'commit' else
              rig.engine.observe_message('impl', value) if kind == 'message' else
              rig.engine.observe_task('impl', 1, value) if kind == 'task' else rig.engine.observe_shutdown('impl'))
    assert len(result) == expected


def test_start_and_wall_time_only_within_lifetime_and_once():
    rig = Rig([('R1', 'WHEN wall_time 50 THEN SEND TO leader'), ('R2', 'WHEN start THEN SEND TO leader')], duration=1000)
    assert [o['trigger'] for o in rig.engine.obligations.values()] == ['start']
    rig.clock.now += 600
    assert [r[1] for r in rig.engine.coding_request('impl')] == ['O2']
    assert rig.engine.coding_request('impl') == []
    rig.policies['late'] = rig.policies['impl']
    rig.engine.worker_spawned('late')
    rig.engine.coding_request('late')
    assert [o['trigger'] for o in rig.engine.obligations.values() if o['worker'] == 'late'] == ['start']


def test_events_before_first_request_count_and_versions_are_snapshotted():
    rig = Rig([('R1', 'WHEN task_claimed THEN SEND TO leader')])
    rig.policies['new'] = rig.policies['impl']
    rig.engine.worker_spawned('new')
    assert len(rig.engine.observe_task('new', 7, 'task_claimed')) == 1
    rig.policies['new'] = (2, (('R1', 'WHEN task_claimed THEN SEND TO integ'),))
    rig.engine.observe_task('new', 8, 'task_claimed')
    first, second = [o for o in rig.engine.obligations.values() if o['worker'] == 'new']
    assert (first['version'], first['recipient'], second['version'], second['recipient']) == (1, 'leader', 2, 'integ')
    rig.policies['new'] = (3, ())  # rule removed: no new obligations, old ones keep their rule
    assert rig.engine.observe_task('new', 9, 'task_claimed') == []
    assert first['rule'] == 'WHEN task_claimed THEN SEND TO leader'


def test_events_after_a_lifetime_and_unobserved_tails_are_coverage():
    rig = Rig([('R1', 'WHEN wall_time 50 THEN SEND TO leader')], duration=1000)
    rig.engine.note_sample('impl')
    rig.clock.now += 600
    rig.engine.worker_ended('impl')
    rig.engine.observe_commit('impl', 'de' * 20, [], committed_at=rig.clock.now - 1)
    reasons = [e['reason'] for e in rig.events if e['event'] == 'rule_coverage']
    assert reasons == ['wall_time_unobserved', 'final_git_interval', 'event_after_end']
    assert rig.events[-1]['occurred_in_lifetime'] is True
    rig.engine.finish()
    assert {(c['worker'], c['reason']) for c in rig.engine.stats()['coverage']} >= {('impl', 'event_after_end')}


def test_finish_settles_everything_and_reports_rates():
    rig = Rig([('R1', 'WHEN commit THEN SEND TO integ WITH sha'), ('R2', 'WHEN task_done THEN SEND TO leader')])
    rig.engine.observe_commit('impl', 'aa' * 20, [])
    rig.remind()
    rig.send('integ', 'SHA: aaaaaaa')
    rig.engine.observe_task('impl', 2, 'task_done')
    rig.remind()
    rig.clock.now += 901
    rig.turns(4)
    rig.engine.coding_request('impl')
    rig.clock.now += 700
    rig.engine.observe_commit('impl', 'bb' * 20, [])
    rig.engine.finish()
    stats = rig.engine.stats()
    assert all(row['open'] == 0 for row in stats['rows'])
    assert stats['totals']['fields'] == {'satisfied': 1, 'missed': 0, 'censored': 1, 'unobservable': 0, 'open': 0,
                                         'denominator': 1, 'rate': 1.0}
    assert stats['totals']['send_only']['missed'] == 1 and stats['totals']['send_only']['rate'] == 0.0
    assert rig.names()[-1] == 'rule_engine_finished'
    assert rig.engine.observe_commit('impl', 'cc' * 20, []) == []


def test_field_checks():
    obligation = {'fields': ['sha', 'paths', 'checks'], 'trigger': 'commit',
                  'events': [{'detail': {'sha': 'abcdef1234567'}}, {'detail': {'sha': 'abcdef9999999'}}]}
    assert field_problems(obligation, 'SHA: abcdef1\nPATHS: src/a.hs\nCHECKS: 12/12') == []
    assert field_problems(obligation, 'SHA: abcdef\nPATHS: src/a.hs\nCHECKS: ok') == ['sha_format']
    assert field_problems(obligation, 'SHA: abcdef12\nPATHS: ../x\nCHECKS: ok') == ['paths_format']
    assert field_problems(obligation, 'sha: abcdef12\npaths: a.hs') == ['checks_missing']
    assert field_problems(obligation, '**SHA:** abcdef1\n- PATHS: a.hs\nCHECKS: x') == []
    assert labeled_lines('- ETA: 10 min\nnote: x\nREMAINING:\n') == {'eta': ['10 min']}


def test_hash_matches_compat():
    # node -e "console.log(require('crypto').createHash('sha256').update('héllo').digest('hex').slice(0,16))"
    assert message_hash('héllo') == '3c48591d8d098a45'


def test_star_rule_after_a_teammate_ended_needs_only_the_living():
    """M2 review N1: ended members stay in the native roster and always fail."""
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO *')])
    rig.engine.worker_spawned('gone')
    rig.engine.worker_ended('gone')
    rig.clock.now += 10
    rig.engine.observe_task('impl', 5, 'task_done')
    rig.send('*', 'done', resolve=False)
    rig.engine.resolve_native('impl', '*', message_hash('done'), 'send_partial', recipients=['integ', 'leader'])
    assert rig.obligation()['status'] == 'satisfied'
    missing = Rig([('R1', 'WHEN task_done THEN SEND TO *')])
    missing.engine.observe_task('impl', 5, 'task_done')
    missing.send('*', 'done', resolve=False)
    missing.engine.resolve_native('impl', '*', message_hash('done'), 'send_partial', recipients=['leader'])
    assert missing.obligation()['status'] == 'open'  # integ was alive and did not receive it


def test_merged_commit_cannot_predate_the_previous_sample():
    """M2 review N2: an old committer time would make old messages count."""
    rig = Rig([('R1', 'WHEN commit THEN SEND TO leader')])
    rig.clock.now = 1000
    rig.send('leader', 'Question about the parser?')
    rig.clock.now = 2800
    rig.engine.observe_commit('impl', 'ee' * 20, [], committed_at=900, not_before=2740)
    obligation = rig.obligation()
    assert obligation['first_occurred_at'] == 2740 and obligation['status'] == 'open'


def test_compliance_in_the_last_counted_response_is_counted():
    """M2 review N3: expiry waits for the next request, after that response's sends resolve."""
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO leader')])
    rig.engine.observe_task('impl', 6, 'task_done')
    rig.engine.coding_request('impl')
    rig.remind()
    rig.engine.coding_response('impl')
    rig.clock.now += 1000
    rig.turns(2)
    rig.engine.coding_request('impl')
    key = rig.send('leader', 'Task 6 done.', resolve=False)
    rig.engine.coding_response('impl')  # 4th response: window conditions met, but no expiry here
    assert rig.obligation()['status'] == 'open'
    rig.engine.resolve_send('impl', key, 'sent')
    assert rig.obligation()['status'] == 'satisfied'


def test_identical_sends_do_not_steal_native_records():
    """M2 review N4: a late native record of send #1 must not settle send #2."""
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO leader')])
    first = rig.send('leader', 'ok', resolve=False)
    rig.engine.resolve_send('impl', first, 'sent')  # settled from history first
    second = rig.send('leader', 'ok', resolve=False)
    rig.engine.resolve_native('impl', 'leader', message_hash('ok'), 'sent')  # belongs to #1
    assert ('impl', *second) in rig.engine.pending_sends


def test_partial_broadcast_recipients_arriving_after_history():
    """M2 review N7: the native per-recipient record can come after the history result."""
    rig = Rig()
    rig.engine.observe_commit('impl', 'f0' * 20, [])
    key = rig.send('*', 'SHA: f0f0f0f', resolve=False)
    rig.engine.resolve_send('impl', key, 'send_partial')
    assert rig.obligation()['status'] == 'open' and rig.obligation()['unknown_sends']
    rig.engine.resolve_native('impl', '*', message_hash('SHA: f0f0f0f'), 'send_partial', recipients=['integ'])
    assert rig.obligation()['status'] == 'satisfied'


def test_finish_records_the_unsampled_tail_of_living_workers():
    rig = Rig()
    rig.engine.note_sample('impl', at=1000)
    rig.clock.now = 1300
    rig.engine.finish()
    tails = [e for e in rig.events if e['event'] == 'rule_coverage' and e['reason'] == 'final_git_interval']
    assert {e['worker'] for e in tails} == {'impl', 'integ'}
    assert next(e for e in tails if e['worker'] == 'impl')['unsampled_seconds'] == 300


def test_two_identical_partial_broadcasts_both_get_their_recipients():
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO integ')])
    rig.engine.observe_task('impl', 1, 'task_done')
    first = rig.send('*', 'done', resolve=False)
    second = rig.send('*', 'done', resolve=False)
    rig.engine.resolve_send('impl', first, 'send_partial')
    rig.engine.resolve_send('impl', second, 'send_partial')
    rig.engine.resolve_native('impl', '*', message_hash('done'), 'send_partial', recipients=['leader'])
    assert rig.obligation()['status'] == 'open'
    rig.engine.resolve_native('impl', '*', message_hash('done'), 'send_partial', recipients=['integ'])
    assert rig.obligation()['status'] == 'satisfied'


def test_broadcast_records_pair_with_broadcasts_in_order():
    """M2 review R4-1: a full broadcast settled first must not take a partial one's recipients."""
    rig = Rig([('R1', 'WHEN task_done THEN SEND TO integ')])
    rig.engine.observe_task('impl', 1, 'task_done')
    direct = rig.send('leader', 'same', resolve=False)
    full = rig.send('*', 'same', resolve=False)
    partial = rig.send('*', 'same', resolve=False)
    rig.engine.resolve_send('impl', direct, 'sent')
    rig.engine.resolve_send('impl', full, 'sent')       # full broadcast reached integ: satisfied
    assert rig.obligation()['status'] == 'satisfied'
    rig.engine.resolve_send('impl', partial, 'send_partial')
    queue = rig.engine.awaiting_recipients[('impl', '*', message_hash('same'))]
    assert [entry['outcome'] for entry in queue] == ['sent', 'send_partial']
    rig.engine.resolve_native('impl', '*', message_hash('same'), 'sent', recipients=['integ', 'leader'])
    assert [entry['outcome'] for entry in queue] == ['send_partial'] and queue[0]['recipients'] is None


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _fields_case(path, message, ending):
    clock = _Clock()
    sha = 'ab12cd34' * 5
    policies = {'impl': (1, (('R1', 'WHEN commit THEN SEND TO leader WITH sha'),))}
    engine = RuleEngine(lambda w: policies.get(w), lambda *a, **k: None, started_at=clock.now,
                        duration_seconds=6 * 3600, clock=clock)
    for worker in ('impl', 'other'):
        engine.worker_spawned(worker)
        engine.coding_request(worker)
    engine.observe_commit('impl', sha, ['src/A.hs'])
    oid = next(iter(engine.obligations))
    engine.coding_request('impl')
    engine.notice_injected('impl', oid)
    clock.now += 5
    if path == 'pending':
        engine.record_send('impl', ('r', 'c1'), to='leader', message=message)  # its outcome never arrives
    else:
        engine.record_send('impl', ('r', 'c1'), to='*', message=message)
        engine.resolve_send('impl', ('r', 'c1'), 'send_partial')  # recipients unknown
    engine.coding_response('impl')
    if ending == 'window':
        for _ in range(4):
            clock.now += 300
            engine.coding_request('impl')
            engine.coding_response('impl')
        engine.coding_request('impl')
    else:
        engine.worker_ended('impl')
    return engine.obligations[oid]['status']


@pytest.mark.parametrize('path', ['pending', 'partial_broadcast'])
def test_a_send_missing_required_fields_never_leaves_an_obligation_undecided(path):
    # Review 10-18 F5 (plan v4 7.5): only a send that could satisfy the rule makes it unobservable.
    assert _fields_case(path, 'I am working', 'window') == 'missed'
    assert _fields_case(path, 'I am working', 'ended') == 'censored'
    assert _fields_case(path, 'SHA: ab12cd34', 'window') == 'unobservable'
    assert _fields_case(path, 'SHA: ab12cd34', 'ended') == 'unobservable'


@pytest.mark.parametrize('attached, expected', [(True, 'satisfied'), (False, 'unobservable')])
def test_an_unknown_recipient_send_is_judged_against_the_coalesced_commits(attached, expected):
    # Fix review 10-18: the field check runs at decision time, after a later commit joined the obligation.
    clock = _Clock()
    first, second = 'ab12cd34' * 5, 'ff00ee11' * 5
    policies = {'impl': (1, (('R1', 'WHEN commit THEN SEND TO leader WITH sha'),))}
    engine = RuleEngine(lambda w: policies.get(w), lambda *a, **k: None, started_at=clock.now,
                        duration_seconds=6 * 3600, clock=clock)
    for worker in ('impl', 'other', 'gone'):
        engine.worker_spawned(worker)
        engine.coding_request(worker)
    engine.worker_ended('gone')
    engine.observe_commit('impl', first, ['a'])
    oid = next(iter(engine.obligations))
    engine.coding_request('impl')
    engine.notice_injected('impl', oid)
    clock.now += 5
    message = 'Done.\nSHA: ' + second[:8]
    engine.record_send('impl', ('r', 'c1'), to='*', message=message)
    engine.resolve_send('impl', ('r', 'c1'), 'send_partial')  # recipients unknown
    engine.coding_response('impl')
    clock.now += 30
    engine.observe_commit('impl', second, ['b'])  # coalesced into the same obligation
    if attached:
        engine.resolve_native('impl', '*', message_hash(message), 'send_partial', recipients=['leader', 'other'])
    for _ in range(4):
        clock.now += 300
        engine.coding_request('impl')
        engine.coding_response('impl')
    engine.coding_request('impl')
    assert engine.obligations[oid]['status'] == expected
