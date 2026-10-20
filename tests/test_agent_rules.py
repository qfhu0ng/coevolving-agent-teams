"""agent_rules_v1 grammar: rules, globs, spawn blocks and control headers (zero API)."""
import pytest

from agent_company.policy import agent_rules as r


@pytest.mark.parametrize('line,canonical', [
    ('WHEN commit THEN SEND TO integ WITH sha, paths', 'WHEN commit THEN SEND TO integ WITH sha, paths'),
    ('when commit paths src/**, docs/*.md then send to integ with SHA', 'WHEN commit PATHS src/**, docs/*.md THEN SEND TO integ WITH sha'),
    ('WHEN message THEN SEND TO leader', 'WHEN message FROM * THEN SEND TO leader'),
    ('WHEN message FROM integ THEN SEND TO integ WITH checks', 'WHEN message FROM integ THEN SEND TO integ WITH checks'),
    ('WHEN wall_time 75 THEN SEND TO leader WITH remaining, eta', 'WHEN wall_time 75 THEN SEND TO leader WITH remaining, eta'),
    ('WHEN task_done THEN SEND TO *', 'WHEN task_done THEN SEND TO *'),
    ('WHEN shutdown_request THEN SEND TO leader WITH remaining', 'WHEN shutdown_request THEN SEND TO leader WITH remaining'),
    ('WHEN start THEN SEND TO leader', 'WHEN start THEN SEND TO leader'),
])
def test_valid_rules_render_canonically_and_round_trip(line, canonical):
    rule = r.parse_rule(line)
    assert rule.text == canonical
    assert r.parse_rule(rule.text) == rule


@pytest.mark.parametrize('line,code', [
    ('commit -> integ', 'rule_syntax'),
    ('WHEN commit SEND TO integ', 'rule_syntax'),
    ('WHEN discovered_info THEN SEND TO integ', 'unknown_trigger'),
    ('WHEN wall_time 25 THEN SEND TO leader', 'wall_time_fraction'),
    ('WHEN wall_time THEN SEND TO leader', 'wall_time_fraction'),
    ('WHEN commit 50 THEN SEND TO leader', 'fraction_not_allowed'),
    ('WHEN commit FROM integ THEN SEND TO leader', 'from_not_allowed'),
    ('WHEN task_done PATHS src/** THEN SEND TO leader', 'paths_not_allowed'),
    ('WHEN commit PATHS ../outside THEN SEND TO leader', 'invalid_glob'),
    ('WHEN commit PATHS /abs/** THEN SEND TO leader', 'invalid_glob'),
    ('WHEN commit THEN SEND TO master', 'invalid_party'),
    ('WHEN commit THEN SEND TO 9lives', 'invalid_party'),
    ('WHEN commit THEN SEND TO integ WITH sha, diff', 'unknown_field'),
    ('WHEN commit THEN SEND TO integ WITH sha, sha', 'duplicate_field'),
    ('WHEN commit THEN WAIT FOR integ', 'rule_syntax'),
])
def test_invalid_rules_have_specific_codes(line, code):
    with pytest.raises(r.RuleError) as error:
        r.parse_rule(line)
    assert error.value.code == code
    assert error.value.detail


def test_roster_checks_self_and_unknown_parties():
    rule = r.parse_rule('WHEN commit THEN SEND TO integ')
    r.check_roster(rule, owner='impl', known=frozenset({'integ'}))
    with pytest.raises(r.RuleError, match='self_recipient'):
        r.check_roster(rule, owner='integ', known=frozenset({'integ'}))
    with pytest.raises(r.RuleError, match='unknown_party'):
        r.check_roster(rule, owner='impl', known=frozenset())
    sender = r.parse_rule('WHEN message FROM ghost THEN SEND TO leader')
    with pytest.raises(r.RuleError, match='unknown_party'):
        r.check_roster(sender, owner='impl', known=frozenset({'integ'}))
    r.check_roster(r.parse_rule('WHEN message FROM * THEN SEND TO *'), owner='impl', known=frozenset())


@pytest.mark.parametrize('pattern,path,expected', [
    ('src/**', 'src/a/b.hs', True), ('src/**', 'src', True), ('src/*', 'src/a.hs', True),
    ('src/*', 'src/a/b.hs', False), ('src/', 'src/a/b', True), ('**/*.md', 'README.md', True),
    ('**/*.md', 'docs/x/y.md', True), ('docs/*.md', 'docs/x/y.md', False), ('a?c', 'abc', True),
    ('Text/Pandoc/Readers/**', 'Text/Pandoc/Writers/HTML.hs', False),
])
def test_glob_semantics(pattern, path, expected):
    assert r.glob_match(pattern, path) is expected


def test_spawn_block_strips_markers_and_keeps_task_text():
    prompt = ('Implement the reader.\nPOLICY:\nWHEN commit THEN SEND TO integ WITH sha\n\nWHEN bogus THEN SEND TO x\n'
              'NOTES:\nScope: src/Readers/.\n\nNever touch tests.\nEND-POLICY\nSUCCEEDS: old1\nReport when done.\n')
    parsed = r.parse_spawn_policy(prompt)
    assert parsed.block == 'ok'
    assert parsed.rules == ('WHEN commit THEN SEND TO integ WITH sha', 'WHEN bogus THEN SEND TO x')
    assert parsed.notes == 'Scope: src/Readers/.\n\nNever touch tests.'
    assert parsed.succeeds == 'old1' and not parsed.warnings
    assert parsed.prompt == 'Implement the reader.\nReport when done.\n'


@pytest.mark.parametrize('prompt,block,warning', [
    ('Task only.', 'missing', None),
    ('Task\nPOLICY:\nWHEN commit THEN SEND TO integ\nMore task text', 'unclosed', 'unclosed_block'),
    ('POLICY:\nA\nEND-POLICY\nPOLICY:\nB\nEND-POLICY', 'duplicate', 'duplicate_block'),
    ('Task\nEND-POLICY', 'missing', 'stray_end'),
    ('Task\n**POLICY:**\nWHEN commit THEN SEND TO integ', 'missing', 'near_miss_block'),
])
def test_spawn_block_problems_leave_prompt_unchanged(prompt, block, warning):
    parsed = r.parse_spawn_policy(prompt)
    assert parsed.block == block
    assert parsed.prompt == prompt
    assert [code for code, _ in parsed.warnings] == ([warning] if warning else [])


def test_spawn_succeeds_variants():
    two = r.parse_spawn_policy('SUCCEEDS: a\nSUCCEEDS: b\nTask')
    assert two.succeeds is None and ('duplicate_succeeds', {}) in two.warnings and two.prompt == 'SUCCEEDS: a\nSUCCEEDS: b\nTask'
    near = r.parse_spawn_policy('Succeeds alpha\nTask')
    assert near.succeeds is None and ('malformed_succeeds', {}) in near.warnings
    only = r.parse_spawn_policy('Task\nSUCCEEDS: alpha')
    assert only.succeeds == 'alpha' and only.block == 'missing' and only.prompt == 'Task\n'


@pytest.mark.parametrize('message,expected', [
    ('AGENT-POLICY integ add\nWHEN commit THEN SEND TO leader',
     dict(targets=('integ',), operation='add', rule_id=None, base=None)),
    ('AGENT-POLICY integ, impl replace R3 base=v4\nWHEN commit THEN SEND TO leader',
     dict(targets=('integ', 'impl'), operation='replace', rule_id='R3', base=4)),
    ('AGENT-POLICY * remove R2', dict(targets=('*',), operation='remove', rule_id='R2')),
    ('AGENT-POLICY impl notes\nScope: src/', dict(targets=('impl',), operation='notes')),
    ('AGENT-POLICY impl revoke #7', dict(targets=('impl',), operation='revoke', proposal_id=7)),
])
def test_agent_policy_headers(message, expected):
    call = r.parse_control('c1', 'impl', message)
    assert isinstance(call, r.EditCall)
    for key, value in expected.items():
        assert getattr(call, key) == value


def test_proposal_decision_and_lesson_headers():
    proposal = r.parse_control('c1', 'leader', 'POLICY-PROPOSAL self replace R2\nWHEN task_done THEN SEND TO leader\nREASON: shorter')
    assert (proposal.target, proposal.operation, proposal.rule_id, proposal.body, proposal.reason) == (
        'self', 'replace', 'R2', 'WHEN task_done THEN SEND TO leader', 'shorter')
    decision = r.parse_control('c2', 'impl', 'POLICY-DECISION #3 amend\nWHEN commit THEN SEND TO integ\nNOTE: narrower')
    assert (decision.proposal_id, decision.decision) == (3, 'amend')
    lesson = r.parse_control('c3', 'leader', 'LESSON: Commit before reporting a SHA.\nOtherwise the SHA changes.')
    assert lesson.text == 'Commit before reporting a SHA.\nOtherwise the SHA changes.'
    assert r.parse_control('c4', 'leader', 'Ordinary status update.') is None
    assert r.parse_control('c5', 'leader', 'Lessons learned from the run follow.') is None


@pytest.mark.parametrize('message,code', [
    ('AGENT-POLICY integ append\nx', 'malformed_header'),
    ('**AGENT-POLICY** integ add\nx', 'malformed_header'),
    ('Policy proposal: maybe we should report SHAs', 'malformed_header'),
    ('POLICY-PROPOSAL self add\nWHEN commit THEN SEND TO leader', 'malformed_body'),
    ('POLICY-DECISION 3 approve', 'malformed_header'),
    ('Lesson: commit first', 'malformed_header'),
    ('LESSON:   ', 'malformed_body'),
])
def test_near_miss_controls_are_rejected_not_sent(message, code):
    with pytest.raises(r.ControlRejection) as error:
        r.parse_control('c1', 'leader', message)
    assert error.value.code == code and error.value.tool_call_id == 'c1'


@pytest.mark.parametrize('operation,body,ok', [
    ('add', 'WHEN commit THEN SEND TO x\nWHEN start THEN SEND TO leader', True), ('add', '\n', False),
    ('replace', 'WHEN commit THEN SEND TO x', True), ('replace', 'a\nb', False),
    ('remove', '', True), ('remove', 'leftover', False), ('revoke', 'x', False), ('notes', '', True),
])
def test_operation_bodies(operation, body, ok):
    call = r.EditCall('c1', '*', ('a',), operation)
    if ok:
        r.edit_body(call, operation, body)
    else:
        with pytest.raises(r.ControlRejection, match='malformed_body'):
            r.edit_body(call, operation, body)


def test_related_facts_are_deterministic():
    facts = r.related_facts('integ merged 0a1b2c3d into src/Text/Reader.hs; ask mdwrite', frozenset({'integ', 'mdwrite', 'x'}))
    assert facts == {'agents': ['integ', 'mdwrite'], 'paths': ['src/Text/Reader.hs'], 'commits': ['0a1b2c3d']}


def test_malformed_block_keeps_succeeds_line_but_still_names_the_predecessor():
    prompt = 'Task\nPOLICY:\nWHEN commit THEN SEND TO integ\nSUCCEEDS: old\nMore'
    parsed = r.parse_spawn_policy(prompt)
    assert parsed.block == 'unclosed' and parsed.prompt == prompt and parsed.succeeds == 'old'
    mixed = r.parse_spawn_policy('SUCCEEDS: old\nSucceeds: other\nTask')
    assert mixed.succeeds == 'old' and ('malformed_succeeds', {}) in mixed.warnings


def test_master_lesson_lines_are_ordinary_messages():
    assert r.parse_control('c1', 'alpha', 'LESSON: keep going', actor='master') is None
    assert r.parse_control('c1', 'alpha', 'Lesson: keep going', actor='master') is None
    with pytest.raises(r.ControlRejection):
        r.parse_control('c1', 'leader', 'Lesson: keep going', actor='alpha')
