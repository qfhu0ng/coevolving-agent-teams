"""M3 units for agent_rules_v1: board grammar and store, tails, gate text, board rules, reflection funnel."""
import pytest

from agent_company.policy import agent_rules, agent_rules_runtime
from agent_company.policy import agent_rules_text as text
from agent_company.policy.agent_policy_store import AgentPolicyStore, AgentSnapshot, BoardPost
from agent_company.policy.agent_rules import BoardCall, ControlRejection, RuleError, SpawnCall, parse_spawn_policy
from agent_company.policy.role_package import create_m0_package
from agent_company.runtime.reflection import ReflectionFunnel
from agent_company.runtime.rule_engine import RuleEngine


# -- grammar -----------------------------------------------------------------------------
def test_board_posts_parse_for_anyone_and_near_misses_are_rejected():
    post = agent_rules.parse_control('c1', 'leader', 'BOARD: interface fixed\nSHA: abc1234', actor='alpha')
    assert post == BoardCall('c1', 'leader', 'interface fixed\nSHA: abc1234')
    assert isinstance(agent_rules.parse_control('c2', 'alpha', 'BOARD: from the master', actor='master'), BoardCall)
    for message, code in (('BOARD:   ', 'malformed_body'), ('Board: lower case', 'malformed_header'),
                          ('**BOARD:** bold', 'malformed_header')):
        with pytest.raises(ControlRejection) as error:
            agent_rules.parse_control('c3', 'leader', message, actor='alpha')
        assert error.value.code == code
    assert agent_rules.parse_control('c4', 'leader', 'The board looks fine.', actor='alpha') is None


def test_board_is_a_recipient_only_and_a_reserved_name():
    rule = agent_rules.parse_rule('WHEN commit THEN SEND TO board WITH sha')
    assert rule.recipient == 'board' and rule.text == 'WHEN commit THEN SEND TO board WITH sha'
    agent_rules.check_roster(rule, owner='alpha', known=frozenset())
    with pytest.raises(RuleError) as error:
        agent_rules.parse_rule('WHEN message FROM board THEN SEND TO leader')
    assert error.value.code == 'board_sender'
    assert 'board' in agent_rules.RESERVED_NAMES


# -- store -------------------------------------------------------------------------------
def store():
    records = []
    s = AgentPolicyStore(create_m0_package('A', 'C'), records.append, clock=lambda: 1_700_000_000)
    for name in ('alpha', 'beta'):
        snap = s.snapshot('master')
        s.commit(s.stage('master', snap, [SpawnCall('s-' + name, name, parse_spawn_policy('x'))]))
        s.record_first_request(name)
    return s, records


def post(s, actor, message, call_id, to='leader'):
    call = agent_rules.parse_control(call_id, to, message, actor=actor)
    return s.commit(s.stage(actor, s.snapshot(actor), [call]))


def test_board_posts_are_numbered_stored_and_replayed():
    s, records = store()
    result = post(s, 'alpha', 'BOARD: parser done\nSHA: abc1234\nPATHS: src/A.hs', 'b1')
    assert result.receipts == {'b1': text.BOARD_RECEIPT.format(id='B1')}
    post(s, 'master', 'BOARD: freeze at 90%', 'b2', to='alpha')
    board = s.export()['board']
    assert [(p['id'], p['author'], p['to']) for p in board] == [('B1', 'alpha', 'leader'), ('B2', 'master', 'alpha')]
    assert board[0]['related']['commits'] == ['abc1234'] and board[0]['related']['paths'] == ['src/A.hs']
    replayed = AgentPolicyStore.replay(records, lambda record: None)
    assert replayed.export()['board'] == board
    s.record_terminal('beta')
    rejected = post(s, 'beta', 'BOARD: late', 'b3')
    assert [r.code for r in rejected.rejections] == ['not_authorized']


def test_snapshot_shows_the_latest_five_posts_by_others():
    s, _ = store()
    for index in range(7):
        post(s, 'beta', f'BOARD: note {index}', f'b{index}')
    post(s, 'alpha', 'BOARD: mine', 'mine')
    snap = s.snapshot('alpha')
    assert snap.board_count == 8 and [p.post_id for p in snap.board_recent] == ['B3', 'B4', 'B5', 'B6', 'B7']
    assert s.snapshot('master').board_recent == () and s.snapshot('master').board_count == 8


def test_lessons_record_the_prompt_they_answer():
    s, _ = store()
    call = agent_rules.parse_control('l1', 'leader', 'LESSON: build first', actor='alpha')
    s.commit(s.stage('alpha', s.snapshot('alpha'), [call], lesson_trigger='task_done'))
    assert s.lessons()[0]['trigger'] == 'task_done'
    with pytest.raises(ValueError):
        s.stage('alpha', s.snapshot('alpha'), [call], lesson_trigger='incident')


# -- rendering ----------------------------------------------------------------------------
def snapshot(actor='alpha', posts=(), count=None):
    recent = tuple(BoardPost(f'B{n}', author, 'leader', f'2026-10-17T01:2{n}:00Z', body) for n, author, body in posts)
    return AgentSnapshot(actor, (), (), 0, len(posts) if count is None else count, recent)


def test_roster_lists_active_teammates_other_than_self():
    roster = {'at': '2026-10-17T03:04:05Z', 'members': [
        {'name': 'alpha', 'status': 'running', 'tasks': [{'id': '1', 'subject': 'mine'}]},
        {'name': 'beta', 'status': 'running', 'shutdown_pending': True,
         'tasks': [{'id': '2', 'subject': 'Implement the docx reader ' + 'x' * 80}]},
        {'name': 'gamma', 'status': 'idle', 'tasks': []},
        {'name': 'delta', 'status': 'completed', 'tasks': []},
        {'name': 'eps', 'status': 'running', 'tasks': []}]}
    rendered = agent_rules_runtime.render_roster('alpha', roster, ended=frozenset({'eps'}))
    lines = rendered.splitlines()
    assert lines[0] == 'Active teammates (as of 03:04:05 UTC):'
    assert lines[1].startswith('- beta: working (shutdown requested); in progress: #2 Implement the docx reader')
    assert lines[1].endswith('...') and len(lines[1].split('#2 ')[1]) == 60
    assert lines[2] == '- gamma: idle; in progress: none' and len(lines) == 3
    assert agent_rules_runtime.render_roster('alpha', None) == text.WORKER_TAIL_ROSTER_UNKNOWN
    alone = {'at': '2026-10-17T03:04:05Z', 'members': [{'name': 'alpha', 'status': 'running', 'tasks': []}]}
    assert agent_rules_runtime.render_roster('alpha', alone) == 'Active teammates (as of 03:04:05 UTC): none besides you.'


def test_board_tail_always_shows_recent_posts_and_marks_new_ones():
    snap = snapshot(posts=[(1, 'beta', 'first\nmore'), (2, 'gamma', 'y' * 150)])
    rendered = agent_rules_runtime.render_board_tail(snap, seen=1, board_file='/s/team_board.md')
    lines = rendered.splitlines()
    assert lines[0] == 'Latest board posts by others (2 post(s) on the board; all in /s/team_board.md):'
    assert lines[1] == '[B1] beta (01:21:00): first'
    assert lines[2].startswith('[B2] new gamma (01:22:00): ') and lines[2].endswith('...')
    assert len(lines[2].split('): ', 1)[1]) == 100
    empty = agent_rules_runtime.render_board_tail(snapshot(count=3), seen=3, board_file='/b')
    assert empty == 'Board: no posts by others yet (3 post(s) on the board; all in /b).'


def test_master_tail_counts_new_posts_and_gate_text_lists_lessons():
    tail = agent_rules_runtime.render_master_tail(snapshot('master', count=4), remaining_seconds=None,
                                                  lessons_file='/l', board_seen=1, board_file='/b')
    assert 'Board posts: 4, 3 new since your previous request. Full board: /b' in tail
    lessons = [{'id': 'L1', 'author': 'alpha', 'trigger': 'shutdown', 'at': '2026-10-17T01:27:00Z', 'text': 'Build first.'}]
    full = agent_rules_runtime.lesson_gate_text(lessons, full=True)
    assert full.startswith('Spawn held, not started') and 'All lessons so far:\n[L1] alpha (at shutdown, 01:27:00): Build first.' in full
    assert 'Lessons since the previous hold:' in agent_rules_runtime.lesson_gate_text(lessons, full=False)


def test_cards_name_the_board_file():
    assert '/s/team_board.md' in agent_rules_runtime.worker_card('/s/team_board.md')
    card = agent_rules_runtime.disclosure('master', policies_file='/p', lessons_file='/l', board_file='/b')
    assert '/p' in card and '/l' in card and '/b' in card and 'BOARD:' in card
    assert agent_rules_runtime.render_board_file({'board': []}).endswith('No posts yet.\n')


# -- rule engine ---------------------------------------------------------------------------
def engine(rules):
    events = []
    eng = RuleEngine(lambda worker: (1, tuple((f'R{i + 1}', r) for i, r in enumerate(rules))),
                     lambda event, **fields: events.append((event, fields)), started_at=0, duration_seconds=3600,
                     clock=lambda: 10)
    eng.worker_spawned('alpha', at=0)
    eng.worker_spawned('beta', at=0)
    return eng, events


def test_board_rules_are_met_only_by_the_owners_post():
    eng, _ = engine(['WHEN commit THEN SEND TO board WITH sha'])
    reminders = eng.observe_commit('alpha', 'ab12cd34' * 5, ['src/A.hs'], observed_at=5)
    assert len(reminders) == 1 and 'team board' in reminders[0][2]
    eng.record_board('beta', 'BOARD: x\nSHA: ab12cd34', at=6)
    eng.record_send('alpha', (0, 'c'), to='leader', message='SHA: ab12cd34', at=6)
    eng.resolve_send('alpha', (0, 'c'), 'sent')
    assert next(iter(eng.obligations.values()))['status'] == 'open'
    eng.record_board('alpha', 'BOARD: parser\nSHA: ab12cd34', at=7)
    assert next(iter(eng.obligations.values()))['status'] == 'satisfied'


def test_board_obligations_ignore_pending_and_partial_broadcasts():
    eng, _ = engine(['WHEN start THEN SEND TO board'])
    eng.coding_request('alpha', at=1)
    eng.record_send('alpha', (0, 'b'), to='*', message='hello', at=2)
    obligation = next(iter(eng.obligations.values()))
    assert obligation['status'] == 'open' and not eng._undecided(obligation)
    eng.finish(at=20)
    assert obligation['status'] == 'censored'


# -- reflection funnel -------------------------------------------------------------------
def funnel():
    now, events = [1000.0], []
    f = ReflectionFunnel(lambda event, **fields: events.append((event, fields)), clock=lambda: now[0])
    return f, events, now


def test_funnel_follows_a_prompt_through_its_window():
    f, events, now = funnel()
    prompt_id, content = f.queue('alpha', 'task_done', {'task_id': '3'})
    assert content == text.REFLECT_TASK_DONE.format(task_id='3')
    assert f.lesson_trigger('alpha') == 'voluntary'
    assert f.injected('alpha', prompt_id) and f.lesson_trigger('alpha') == 'task_done'
    f.response('alpha', {'lessons': [{'id': 'L1', 'text': 'Build first.'}], 'own_policy_changes': [4]})
    f.response('alpha', {'lessons': [{'id': 'L2', 'text': 'build  FIRST.'}]})
    f.response('alpha')
    f.response('alpha')
    closed = [fields for event, fields in events if event == 'reflection_funnel']
    assert len(closed) == 1 and closed[0]['outcome'] == 'output' and closed[0]['responses'] == 4
    assert closed[0]['own_policy_changes'] == [4] and [item['repeated_text'] for item in closed[0]['lessons']] == [False, True]
    assert f.lesson_trigger('alpha') == 'voluntary'


def test_funnel_separates_unshown_unanswered_and_silent_prompts():
    f, events, now = funnel()
    unseen, _ = f.queue('alpha', 'shutdown')
    shown, _ = f.queue('beta', 'shutdown')
    f.injected('beta', shown)
    silent, _ = f.queue('gamma', 'task_done', {'task_id': '1'})
    f.injected('gamma', silent)
    f.response('gamma')
    f.worker_ended('alpha')
    f.worker_ended('beta')
    stats = f.finish()
    outcomes = {fields['worker']: fields['outcome'] for event, fields in events if event == 'reflection_funnel'}
    assert outcomes == {'alpha': 'not_shown', 'beta': 'no_opportunity', 'gamma': 'no_output'}
    assert stats['shutdown'] == {'queued': 2, 'shown': 1, 'with_opportunity': 0, 'with_output': 0,
                                 'own_policy_changes': 0, 'proposals_for_others': 0, 'lessons': 0, 'board_posts': 0}
    assert f.queue('delta', 'shutdown') is None  # nothing after the run ends


def test_funnel_cooldown_and_repeats():
    f, events, now = funnel()
    assert f.queue('alpha', 'task_done', {'task_id': '1'}) is not None
    assert f.queue('alpha', 'task_done', {'task_id': '1'}) is None  # same task: once
    now[0] += 300
    assert f.queue('alpha', 'task_done', {'task_id': '2'}) is None
    assert any(event == 'reflection_prompt_cooled_down' for event, _ in events)
    assert f.queue('alpha', 'shutdown') is not None  # another kind has its own cooldown
    now[0] += 600
    assert f.queue('alpha', 'task_done', {'task_id': '3'}) is not None


def test_team_samples_are_read_from_compat_lines_and_malformed_ones_rejected(tmp_path):
    import json as _json
    from agent_company.runtime.collaboration import PREFIX, CollaborationRuntime, CollaborationTail
    rows = []
    runtime = CollaborationRuntime('propose', rows.append, protocol='agent_rules_v1')
    log = tmp_path / 'stderr.log'
    good = {'kind': 'team', 'sample': 1_700_000_000_000, 'team': {'members': [
        {'name': 'beta', 'status': 'running', 'shutdown_pending': False, 'tasks': [{'id': '2', 'subject': 'x'}]}]}}
    bad = {'kind': 'team', 'sample': 1, 'team': {'members': [{'name': 'beta', 'tasks': [{'id': True}]}]}}
    log.write_text(PREFIX + _json.dumps(good) + '\n' + PREFIX + _json.dumps(bad) + '\n')
    CollaborationTail(log, runtime).poll()
    assert runtime.roster == {'at': '2023-11-14T22:13:20Z', 'members': good['team']['members']}
    assert [r['event'] for r in rows] == ['collaboration_sample_rejected']


# -- review fixes (M3_REVIEW_SUBAGENT.md) ---------------------------------------------------
def test_board_posts_need_a_deliverable_recipient():
    s, _ = store()
    for to in ('board', '', 'nobody', 'alpha', 'master'):  # itself; native has no recipient "master"
        result = post(s, 'alpha', 'BOARD: x', 'b-' + (to or 'empty'), to=to)
        assert [r.code for r in result.rejections] == ['wrong_recipient'], to
    assert [r.code for r in post(s, 'master', 'BOARD: x', 'm1', to='leader').rejections] == ['wrong_recipient']
    for to in ('leader', '*', 'beta', 'BETA'):
        assert not post(s, 'alpha', 'BOARD: x', 'ok-' + to, to=to).rejections, to
    assert not post(s, 'master', 'BOARD: x', 'm2', to='*').rejections
    s.record_terminal('beta')
    assert [r.code for r in post(s, 'alpha', 'BOARD: x', 'ended', to='beta').rejections] == ['wrong_recipient']


def test_reserved_words_cannot_be_teammate_names():
    s, _ = store()
    for name in ('board', 'Board', 'leader'):
        result = s.commit(s.stage('master', s.snapshot('master'), [SpawnCall('s-' + name, name, parse_spawn_policy('x'))]))
        assert [r.code for r in result.rejections] == ['invalid_name'], name


def test_board_reminders_name_a_real_recipient_and_send_only_rules_are_met():
    eng, _ = engine(['WHEN commit THEN SEND TO board WITH sha', 'WHEN start THEN SEND TO board'])
    reminders = eng.observe_commit('alpha', 'ab12cd34' * 5, ['src/A.hs'], observed_at=5)
    assert reminders[0][2].endswith('Post on the team board: a send_message (usually to="leader") whose first line '
                                    'starts with BOARD:, including the labeled lines SHA:.')
    eng.coding_request('alpha', at=6)
    eng.record_board('alpha', 'BOARD: hello', at=7)
    start = next(o for o in eng.obligations.values() if o['trigger'] == 'start')
    assert start['status'] == 'satisfied'


def test_tasks_assigned_before_the_spawn_are_coverage_not_silence():
    eng, events = engine(['WHEN task_claimed THEN SEND TO leader'])
    assert eng.observe_task('gamma', '1', 'task_claimed') == []
    assert ('rule_coverage', {'worker': 'gamma', 'reason': 'event_before_spawn', 'trigger': 'task_claimed',
                              'task_id': '1'}) in events
    eng.worker_spawned('gamma', at=1)
    assert eng.observe_task('gamma', '1', 'task_claimed')  # no longer marked as fired


def test_roster_marks_unknown_tasks_and_started_teammates():
    roster = {'at': '2026-10-17T03:04:05Z', 'members': [{'name': 'beta', 'status': 'running', 'tasks': None}]}
    rendered = agent_rules_runtime.render_roster('alpha', roster, known=('alpha', 'beta', 'gamma'))
    assert '- beta: working; in progress: unknown (task board not read)' in rendered
    assert '- gamma: just started, not observed yet' in rendered and rendered.count('beta') == 1
    assert agent_rules_runtime.render_roster('alpha', None, known=('gamma',)).startswith(text.WORKER_TAIL_ROSTER_PENDING)


def test_shutdown_prompts_repeat_per_request_after_the_cooldown():
    f, events, now = funnel()
    assert f.queue('alpha', 'shutdown', {'observed_at': 1}) is not None
    now[0] += 60
    assert f.queue('alpha', 'shutdown', {'observed_at': 2}) is None
    now[0] += 3600
    assert f.queue('alpha', 'shutdown', {'observed_at': 3}) is not None
    assert [e for e, _ in events].count('reflection_prompt_cooled_down') == 1
