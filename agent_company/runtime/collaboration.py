"""Evidence-only collaboration observations and request-local auxiliary context.

Only native tool results, supplied task-store/Git metadata and explicit COORD
claims enter this store. No file content, reasoning, hidden test or scheduler.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time

from agent_company.policy.role_policy import PolicyPersistError
from agent_company.policy.role_package_store import _CANDIDATE
from agent_company.runtime import collaboration_text as text
from agent_company.policy import agent_rules_text

CATEGORIES = frozenset({'scope_conflict', 'generated_files_tracked', 'stale_report',
                        'destructive_delete', 'fix_not_notified', 'unfinished_handoff',
                        'duplicate_implementation'})
FACT_KEYS = ('task_ids', 'coord_record_ids', 'commits', 'paths', 'role_ids')
PREFIX = 'Agent Company collaboration: '
COMMIT = re.compile(r'^[0-9a-f]{7,40}$')
DELETION_INDEX_LIMIT = 1024
SUMMARY_COVERAGE = frozenset({'state_not_observed', 'task_result_missing', 'task_owner_unknown',
                             'task_scope_unknown', 'sampling_incomplete', 'entry_content_omitted',
                             'entry_limit_reached', 'byte_limit_reached', 'notification_receipt_unknown'})


def valid_commit(value):
    return isinstance(value, str) and COMMIT.fullmatch(value) is not None


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def relative_path(value):
    return (isinstance(value, str) and bool(value) and not PurePosixPath(value).is_absolute()
            and '..' not in PurePosixPath(value).parts and '\\' not in value
            and '\x00' not in value)


def evidence_summary(facts):
    """Fixed five arrays; omit whole values without manufacturing missing facts."""
    result, reasons = {}, []
    for key in FACT_KEYS:
        values = facts.get(key)
        if key == 'commits' and values:
            values = [v[:12] for v in values if valid_commit(v)]
        result[key] = list(dict.fromkeys(v for v in values if isinstance(v, str))) if values else ['unknown']
        if not values and 'evidence_summary_fields_unknown' not in reasons:
            reasons.append('evidence_summary_fields_unknown')
    while len(compact(result).encode('utf-8')) > 300:
        key = max((k for k in FACT_KEYS if result[k] and result[k] != ['unknown']),
                  key=lambda k: sum(len(v.encode('utf-8')) for v in result[k]), default=None)
        if key is None:
            break
        result[key].pop()
        if 'evidence_summary_values_omitted' not in reasons:
            reasons.append('evidence_summary_values_omitted')
    return compact(result), reasons


def call_identity(call):
    function = call.get('function', {})
    try:
        arguments = json.loads(function.get('arguments', '{}'))
    except (TypeError, ValueError):
        return None
    return (call.get('id'), function.get('name'), arguments)


def function_name(call):
    return call.get('function', {}).get('name')


def visible(message):
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def anchor(message):
    """Native visible history identity; never inspect or retain model reasoning."""
    return {key: copy.deepcopy(message[key]) for key in ('role', 'content', 'tool_call_id', 'tool_calls')
            if key in message}


@dataclass(frozen=True)
class Rejection:
    tool_call_id: str
    code: str
    detail: str

    def __str__(self):
        return (f'COORD rejected ({self.code}): {self.detail} This record was not applied. '
                'Other independent valid calls in the response are unaffected.')


@dataclass(frozen=True)
class CoordCall:
    tool_call_id: str
    value: dict


class CollaborationRuntime:
    def __init__(self, mode, writer, *, team=True, clock=time.monotonic, protocol='role_package_v1'):
        self.mode, self.writer, self.team, self.clock = mode, writer, team, clock
        # agent_rules_v1 keeps the raw task/Git/lifecycle observations and the
        # anchored-notice mechanism, but runs no diagnostics, COORD, POLICY-REVIEW
        # or collaboration summary: agents judge problems themselves (plan v4 2).
        if protocol not in ('role_package_v1', 'agent_rules_v1'):
            raise ValueError('Unknown collaboration protocol')
        self.agent_rules = protocol == 'agent_rules_v1'
        self.queued = {}
        # agent_rules_v1 rule-engine hooks, called with this runtime's lock held
        # (lock order: collaboration -> rule engine -> relay); None means unused.
        self.git_listener = self.task_listener = self.notice_listener = self.sample_listener = None
        self._lock = threading.RLock()
        self.sequence = 0
        self.records = []
        self.reports = {}
        self.tasks = {}
        self.git = {}
        # Recent deletion evidence survives later HEAD-only snapshots. Each
        # path retains its latest source row; the A05 snapshot stays separate.
        self.deletions = {}
        self.cache_introductions = set()
        self.bindings = {}
        self.active_instances = set()
        self.pending = {}
        self.observed = set()
        self.notices = {}
        self.closed_thresholds = {}
        self.cooldowns = {}
        self.events = []
        self.delivered_events = set()
        self.coverage = {'state_not_observed'}
        # agent_rules_v1: the latest native team sample (names, statuses, in-progress
        # tasks) for teammates' request tails (plan v4 section 8.2). Memory only:
        # every provider request file already keeps the tail it was shown.
        self.roster = None

    def _write(self, event_type, **fields):
        row = {'event': event_type, 'id': f'collab-{self.sequence + 1}', **fields}
        try:
            self.writer(row)
        except Exception as exc:
            raise PolicyPersistError(str(exc)) from exc
        self.sequence += 1
        return row

    def bind(self, snapshot):
        with self._lock:
            if snapshot is not None:
                for binding in snapshot.bindings:
                    self.bindings[binding.instance] = binding.role_id
                    if binding.lifecycle == 'active' or (binding.confirmed and binding.lifecycle == 'planned'):
                        self.active_instances.add(binding.instance)
                    else:
                        self.active_instances.discard(binding.instance)

    def role_id(self, actor):
        return 'master' if actor == 'master' else ('single' if actor == 'single' else self.bindings.get(actor))

    def queue_notice(self, actor, kind, content, key=None):
        """A one-time notice anchored at the actor's next coding request."""
        with self._lock:
            self.queued.setdefault(actor, []).append((kind, content, key))

    def cancel_notices(self, actor, kind, key):
        """Withdraw queued notices that were not delivered yet."""
        with self._lock:
            self.queued[actor] = [n for n in self.queued.get(actor, []) if (n[0], n[2]) != (kind, key)]

    def parse_coord(self, actor, call_id, args):
        if self.agent_rules:
            return None
        message = args.get('message')
        if not isinstance(message, str) or not re.search(r'^COORD:', message, re.M):
            return None
        def reject(code):
            return Rejection(call_id, code, text.COORD_ERRORS[code])
        if any(_CANDIDATE.match(line) for line in message.splitlines()):
            return reject('mixed_control')
        lines = message.rstrip().splitlines()
        if not lines or not lines[-1].startswith('COORD: ') or sum(line.startswith('COORD:') for line in lines) != 1:
            return reject('invalid_record')
        try:
            value = json.loads(lines[-1][7:])
        except ValueError:
            return reject('invalid_record')
        if not isinstance(value, dict):
            return reject('invalid_record')
        fields = {'report': {'op', 'summary', 'paths', 'tested_commit', 'build_id', 'checks', 'recipient'},
                  'resolve': {'op', 'issue_id', 'fix_commit', 'checks'},
                  'ack': {'op', 'issue_id', 'tested_commit', 'checks'}}
        if value.get('op') not in fields or set(value) != fields[value['op']]:
            return reject('invalid_record')
        if args.get('to') != ('*' if actor == 'master' else 'leader'):
            return reject('invalid_recipient')
        nullable = {'tested_commit', 'build_id'}
        if any(not isinstance(v, str) and not (k in nullable and v is None)
               for k, v in value.items() if k not in {'paths'}):
            return reject('invalid_record')
        if any(value.get(k) is not None and not valid_commit(value[k])
               for k in ('tested_commit', 'fix_commit')):
            return reject('invalid_record')
        if value['op'] == 'report':
            if not isinstance(value['paths'], list) or any(not relative_path(p) for p in value['paths']):
                return reject('invalid_path')
            if value['recipient'] not in {'leader', *self.active_instances}:
                return reject('invalid_recipient')
        else:
            report = self.reports.get(value['issue_id'])
            if report is None:
                return reject('unknown_record')
            if value['op'] == 'ack' and report['actor'] != actor:
                return reject('not_authorized')
        return CoordCall(call_id, value)

    def commit_calls(self, actor, calls, response, source_request, assistant_index=0):
        """Persist accepted claims before making them visible; no native-delivery claim."""
        with self._lock:
            for call in calls:
                row = self._write('collaboration_coord', actor=actor, role_id=self.role_id(actor),
                                  source_request=source_request, tool_call_id=call.tool_call_id,
                                  evidence_level='reported', record=call.value)
                value = copy.deepcopy(call.value)
                entry = {'id': row['id'], 'kind': {'report': 'report', 'resolve': 'fix', 'ack': 'ack'}[value['op']],
                         'actor': actor, 'role_id': self.role_id(actor), 'evidence_level': 'reported',
                         'source_event_ids': [row['id']], **value}
                self.records.append(entry)
                if value['op'] == 'report':
                    self.reports[row['id']] = entry
                else:
                    self.reports[value['issue_id']][value['op']] = row['id']
                receipt = compact({'coord_status': 'accepted', 'record_id': row['id'],
                                   'state_version': self.sequence, 'evidence_level': 'reported'})
                self._track_response(actor, response, source_request, call.tool_call_id, receipt, assistant_index)
                self._diagnose_coord(entry)

    def track_response(self, actor, response, source_request, assistant_index=0):
        with self._lock:
            self._track_response(actor, response, source_request, assistant_index=assistant_index)
            if self.agent_rules:
                return
            for choice in response.get('choices', []):
                # Deliberately only visible content. Reasoning is never read.
                for line in visible(choice.get('message', {})).splitlines():
                    if re.match(r'^POLICY-REVIEW (?:C|role:[A-Za-z][A-Za-z0-9_-]*) v\d+:', line):
                        self._write('collaboration_policy_review', actor=actor, role_id=self.role_id(actor),
                                    source_request=source_request, evidence_level='reported', review=line)

    def _track_response(self, actor, response, source_request, only=None, receipt=None, assistant_index=0):
        for choice in response.get('choices', []):
            message = choice.get('message', {})
            for call in message.get('tool_calls') or []:
                identifier = call.get('id')
                if only is not None and identifier != only:
                    continue
                if call.get('function', {}).get('name') not in {'task_create', 'task_update', 'task_list', 'send_message'}:
                    continue
                key = (actor, source_request, identifier)
                old = self.pending.get(key, {})
                self.pending[key] = {**old, 'call': copy.deepcopy(call), 'source_request': source_request, 'assistant_index': assistant_index,
                                     **({'receipt': receipt} if receipt is not None else {})}
                if only is None and not self.agent_rules and function_name(call) == 'send_message' and key not in self.observed:
                    arguments = json.loads(call['function']['arguments'])
                    message = arguments.get('message', '')
                    if arguments.get('to') == '*' and isinstance(message, str) and re.search(
                            r'(?im)^(?:coordination rule|team convention|reusable rule|协作约定|团队约定):', message):
                        self._write('collaboration_rule_candidate', actor=actor, role_id=self.role_id(actor),
                                    source_request=source_request, tool_call_id=identifier,
                                    evidence_level='reported', suggested_operation='add_rule',
                                    message_reference={'source_request': source_request, 'tool_call_id': identifier})

    def observe_history(self, actor, messages):
        with self._lock:
            calls, ordinal = {}, -1
            for message in messages:
                if message.get('role') == 'assistant':
                    ordinal += 1
                    calls = {c.get('id'): call_identity(c) for c in message.get('tool_calls') or []}
                if message.get('role') != 'tool':
                    continue
                identifier = message.get('tool_call_id')
                for key, entry in list(self.pending.items()):
                    if (key[0] != actor or key[2] != identifier or ordinal != entry['assistant_index']
                            or calls.get(identifier) != call_identity(entry['call'])):
                        continue
                    if key not in self.observed:
                        self._observe_task(actor, entry, message)
                        self.observed.add(key)
                    if entry.get('receipt'):
                        result = visible(message)
                        if entry['receipt'] not in result:
                            message['content'] = result + '\n' + entry['receipt']
                        self._write_receipt_once(key, entry)

    def _write_receipt_once(self, key, entry):
        if entry.get('receipt_seen'):
            return
        self._write('collaboration_receipt_observed', actor=key[0], source_request=key[1],
                    tool_call_id=key[2], evidence_level='observed')
        entry['receipt_seen'] = True

    def _observe_task(self, actor, entry, result):
        call = entry['call']
        name = call['function']['name']
        if name not in {'task_create', 'task_update', 'task_list'}:
            return
        raw = visible(result).split('--- Teammate messages ---', 1)[0]
        args = json.loads(call['function']['arguments'])
        # Native task-update can persist before assignment notification fails.
        match = re.search(r'^Task #(\d+) (created|updated|was updated)\b', raw, re.M)
        expected = 'created' if name == 'task_create' else 'updated'
        if (name in {'task_create', 'task_update'} and match
                and match[2] in {expected, 'was updated' if expected == 'updated' else expected}
                and (name != 'task_update' or match[1] == str(args.get('taskId')))):
            task_id = match[1]
            before = self.tasks.get(task_id, {})
            fields = re.search(r'\(status: (pending|in_progress|completed)(?:, owner: ([^)]+))?\)', raw)
            native_status, native_owner = fields.groups() if fields else (None, None)
            if native_owner in {'none', 'unassigned', 'null'}:
                native_owner = None
            task = {**before, 'task_id': task_id, 'status': args.get('status', native_status),
                    'owner': args.get('owner', native_owner), 'paths': before.get('paths')}
            if 'description' in args:
                task['paths'] = self._scope(args['description'])
            self.observe_task(task, actor=actor, source={'source_request': entry['source_request'],
                              'tool_call_id': call.get('id'), 'notification_sent': 'assignment prompt' not in raw,
                              **({'tool': name} if self.agent_rules else {})})
        elif name == 'task_list':
            # List output lacks descriptions/scopes. Unknown paths stay unknown.
            for match in re.finditer(r'^#([0-9]+) \[(pending|in_progress|completed)\](?: @([^ \n]+))? — ', raw, re.M):
                self.observe_task({'task_id': match[1], 'status': match[2], 'owner': match[3], 'paths': None},
                                  actor=actor, source={'source_request': entry['source_request'],
                                                     'tool_call_id': call.get('id'),
                                                     **({'tool': name} if self.agent_rules else {})})
        else:
            self.coverage.add('task_result_missing')

    @staticmethod
    def _scope(description):
        # Require an explicit scope label; never infer from subjects or ownership.
        if not isinstance(description, str):
            return None
        match = re.search(r'(?im)^(?:writing scope|write scope|scope|paths):\s*([^\n]+)', description)
        if not match:
            return None
        values = [v.strip().strip('`\"') for v in match[1].split(',')]
        return values if values and all(relative_path(v) for v in values) else None

    def observe_task(self, task, *, actor, source):
        with self._lock:
            task_id = str(task['task_id'])
            canonical = {k: task.get(k) for k in ('status', 'owner', 'paths')}
            if self.agent_rules and self.sample_listener is not None:
                self.sample_listener('task', actor, source)  # every sighting, changed or not
            previous = self.tasks.get(task_id, {}).get('observation')
            if previous == canonical:
                return
            if self.agent_rules and self.task_listener is not None:
                self.task_listener(task_id, previous, canonical, source)
            row = self._write('collaboration_task_observed', task_id=task_id, actor=actor,
                              evidence_level='observed', **canonical, source=source,
                              coverage_reasons=['task_status_unknown'] if canonical['status'] is None else [])
            entry = {'id': row['id'], 'kind': 'task', 'task_id': task_id, **canonical,
                     'role_id': self.role_id(canonical['owner']), 'commit': None,
                     'evidence_level': 'observed', 'source_event_ids': [row['id']], 'observation': canonical}
            self.tasks[task_id] = entry
            self.records.append(entry)
            self.coverage.discard('state_not_observed')
            if not canonical['owner']:
                self.coverage.add('task_owner_unknown')
            if canonical['status'] is None:
                self.coverage.add('task_status_unknown')
            if canonical['paths'] is None:
                self.coverage.add('task_scope_unknown')
            for other in self.tasks.values():
                if (other['task_id'] != task_id and entry['status'] == other['status'] == 'in_progress'
                        and entry['owner'] and other['owner'] and entry['owner'] != other['owner']
                        and 'integrator' not in {entry['role_id'], other['role_id']}
                        and set(entry['paths'] or []) & set(other['paths'] or [])):
                    self.diagnose('scope_conflict', 'team', [entry['owner'], other['owner']],
                                  {'task_ids': [task_id, other['task_id']], 'paths': entry['paths'],
                                   'role_ids': [r for r in [entry['role_id'], other['role_id']] if r]},
                                  source_ids=[entry['id'], other['id']])

    def observe_team(self, value, source):
        """Latest team sample; never written as evidence or used for diagnostics."""
        with self._lock:
            sample = source.get('native_sample') if isinstance(source, dict) else None
            at = (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(sample / 1000))
                  if isinstance(sample, (int, float)) and not isinstance(sample, bool) else None)
            self.roster = {'at': at, 'members': copy.deepcopy(value.get('members') or [])}

    def observe_git(self, actor, value, source):
        with self._lock:
            if self.agent_rules and self.sample_listener is not None:
                self.sample_listener('git', actor, source)
            if self.agent_rules and self.git_listener is not None:
                self.git_listener(actor, {key: value.get(key) for key in
                                          ('commit', 'new_commit', 'changed_paths', 'committed_at', 'sampling_incomplete')}, source)
            allowed = {k: value.get(k) for k in ('commit', 'paths', 'added_paths', 'deleted_paths', 'new_commit_count', 'uncommitted_paths', 'tracked_uncommitted_paths', 'commits_not_in_delivery_history', 'sampling_incomplete')}
            coverage = []
            if allowed['commit'] is not None and not valid_commit(allowed['commit']):
                allowed['commit'] = None
                coverage.append('git_commit_invalid')
            if allowed['sampling_incomplete']:
                coverage.append('git_sampling_incomplete')
            new_coverage = set(coverage) - self.coverage
            self.coverage.update(coverage)
            allowed['paths'] = [p for p in allowed.get('paths') or [] if relative_path(p)]
            allowed['deleted_paths'] = [p for p in allowed.get('deleted_paths') or [] if relative_path(p)]
            allowed['added_paths'] = [p for p in allowed.get('added_paths') or [] if relative_path(p)]
            if self.git.get(actor, {}).get('observation') == allowed:
                if new_coverage:
                    self._write('collaboration_sampling_incomplete', actor=actor, source=source,
                                coverage_reasons=sorted(new_coverage))
                return
            row = self._write('collaboration_git_observed', actor=actor, role_id=self.role_id(actor),
                              evidence_level='observed', source=source, coverage_reasons=coverage, **allowed)
            entry = {'id': row['id'], 'kind': 'git_change', 'actor': actor, 'role_id': self.role_id(actor),
                     'evidence_level': 'observed', 'source_event_ids': [row['id']], **allowed, 'observation': allowed}
            self.git[actor] = entry
            self.records.append(entry)
            for path in allowed['deleted_paths']:
                self.deletions.pop(path, None)
                self.deletions[path] = (allowed['commit'], entry)
                if len(self.deletions) > DELETION_INDEX_LIMIT:
                    expired = next(iter(self.deletions))
                    commit, source_entry = self.deletions.pop(expired)
                    self.coverage.add('deletion_index_limit_reached')
                    self._write('collaboration_sampling_incomplete', reason='deletion_index_limit_reached',
                                path=expired, commit=commit, source_event_ids=[source_entry['id']])
            facts = {'paths': allowed['paths'], 'commits': [allowed['commit']] if allowed['commit'] else [],
                     'role_ids': [entry['role_id']] if entry['role_id'] else []}
            caches = [p for p in allowed['added_paths'] if ('__pycache__' in PurePosixPath(p).parts or p.endswith(('.pyc', '.pyo')))
                      and p not in self.cache_introductions]
            if caches and (allowed.get('new_commit_count') or 0) > 0:
                self.cache_introductions.update(caches)
                self.diagnose('generated_files_tracked', 'team', ['master'],
                              {**facts, 'paths': caches, 'role_ids': []}, source_ids=[row['id']])
            active = [t for t in self.tasks.values() if t['status'] == 'in_progress']
            affected = [t for t in active if set(t['paths'] or []) & set(allowed['deleted_paths'])]
            for task in affected:
                # The sampler actor owns the worktree, not necessarily this
                # commit. A sampled deletion cannot identify its author.
                self.coverage.add('deletion_author_unknown')
                self.diagnose('destructive_delete', None, [],
                              {**facts, 'paths': allowed['deleted_paths'], 'task_ids': [task['task_id']],
                               'role_ids': [task['role_id']] if task['role_id'] else []},
                              level='hypothesis', source_ids=[row['id'], task['id']],
                              coverage_reasons=['deletion_author_unknown'])
            self._diagnose_coord(None)

    def _diagnose_coord(self, entry):
        for report in self.reports.values():
            tested = report.get('tested_commit')
            tested = tested if valid_commit(tested) else None
            git = self.git.get(report['actor'], {})
            facts = {'coord_record_ids': [report['id']], 'paths': report.get('paths'),
                     'commits': [tested] if tested else [], 'role_ids': [report['role_id']] if report['role_id'] else []}
            if tested and git.get('commit') and tested != git['commit'] and not git['commit'].startswith(tested):
                # An older SHA is not proof of missing fixes (including cherry-picks).
                recipient = self._recipient(report)
                self.diagnose('stale_report', self._edge_location(report['actor'], recipient),
                              [report['actor'], recipient], facts, level='hypothesis',
                              source_ids=[report['id'], git['id']], incident_key=report['id'])
            if report.get('resolve') and not report.get('ack'):
                fixer = self._record_actor(report['resolve'])
                self.diagnose('fix_not_notified', self._edge_location(fixer, report['actor']),
                              [fixer, report['actor']], facts, level='hypothesis',
                              source_ids=[report['id'], report['resolve']])
        if not entry or entry.get('kind') != 'report':
            return
        # Explicit agent claims are reported, never an inferred semantic verdict.
        # These inputs are send_message arguments, not arbitrary assistant prose.
        statement = entry['summary'].lower()
        def claims(phrases):
            for phrase in phrases:
                for match in re.finditer(re.escape(phrase), statement):
                    before = statement[max(0, match.start() - 45):match.start()]
                    if re.search(r'\b(?:no|not|never|without)(?:\s+\w+){0,3}\s*$|(?:没有|并未|未发生|并非|不是|不存在)[^，。]{0,12}$', before):
                        continue
                    return True
            return False
        actor, paths = entry['actor'], entry.get('paths') or []
        facts = {'coord_record_ids': [entry['id']], 'paths': paths,
                 'role_ids': [entry['role_id']] if entry['role_id'] else []}
        related = [r for r in self.reports.values() if r['id'] != entry['id'] and r['id'] in statement]
        for old in related:
            if old['actor'] == actor and old.get('resolve') and any(word in statement for word in ('not notified', 'still unresolved', '未通知', '仍未解决')):
                fixer = self._record_actor(old['resolve'])
                self.diagnose('fix_not_notified', self._edge_location(fixer, actor), [fixer, actor], facts, level='reported',
                              source_ids=[old['id'], old['resolve'], entry['id']])
            if entry.get('tested_commit') and old.get('resolve') and any(word in statement for word in ('stale report', '过期汇报')):
                recipient = self._recipient(old)
                self.diagnose('stale_report', self._edge_location(old['actor'], recipient),
                              [old['actor'], recipient], facts, level='reported',
                              source_ids=[old['id'], old['resolve'], entry['id']], incident_key=old['id'])
        deletion_sources = {}
        for path in paths:
            if path in self.deletions:
                _, deletion = self.deletions[path]
                source = deletion_sources.setdefault(deletion['id'], {**deletion, 'deleted_paths': []})
                source['deleted_paths'].append(path)
        if deletion_sources and claims(('accidentally deleted', 'mistakenly deleted', '误删')):
            claimed_deleter = self._claimed_deletion_actor(entry)
            for deletion in deletion_sources.values():
                affected = [t for t in self.tasks.values() if t.get('status') == 'in_progress'
                            and set(t.get('paths') or []) & set(deletion.get('deleted_paths') or []) & set(paths)]
                # The sampled worktree and report sender never imply authorship.
                # Only an explicit literal actor claim supplies a reported edge.
                for target in affected or [None]:
                    owner = target['owner'] if target else None
                    distinct = claimed_deleter is not None and owner is not None and claimed_deleter != owner
                    location = self._edge_location(claimed_deleter, owner) if distinct else None
                    if location is None:
                        self.coverage.add('deletion_relation_unknown')
                    self.diagnose('destructive_delete', location, [claimed_deleter, owner] if distinct else [],
                                  {**facts, 'role_ids': [], 'task_ids': [target['task_id']] if target else []},
                                  level='reported',
                                  source_ids=[entry['id'], deletion['id'], *([target['id']] if target else [])],
                                  coverage_reasons=['deletion_relation_unknown'] if location is None else [])
        named = [a for a in self.bindings if a != actor and re.search(r'(?<![\w-])' + re.escape(a) + r'(?![\w-])', statement)]
        if paths and named and claims(('duplicate implementation', 'duplicated implementation', '重复实现')):
            for other in named:
                self.diagnose('duplicate_implementation', self._edge_location(actor, other), [actor, other],
                              facts, level='reported', source_ids=[entry['id']], incident_key=(entry['id'], other))
        git = self.git.get(actor)
        if (git and (git.get('tracked_uncommitted_paths') or 0) > 0
                and any(word in statement for word in ('finishing', 'handoff', '收工', '交接'))):
            recipient = self._recipient(entry)
            self.diagnose('unfinished_handoff', self._edge_location(actor, recipient), [actor, recipient], facts,
                          level='reported', source_ids=[entry['id'], git['id']])

    @staticmethod
    def _recipient(report):
        recipient = report.get('recipient')
        return 'master' if recipient == 'leader' else recipient

    def _claimed_deletion_actor(self, report):
        """Literal sentence-leading claims only; never infer a Git author."""
        statement = report.get('summary', '')
        boundary = r'(?:^|(?<=[.!?。！？\n]))[ \t]*'
        actors = set()
        if re.search(boundary + r'(?:I (?:accidentally|mistakenly) deleted\b|我误删)', statement, re.I):
            actors.add(report['actor'])
        for instance in {'master', *self.bindings}:
            if re.search(boundary + re.escape(instance) + r' accidentally deleted\b', statement, re.I):
                actors.add(instance)
        return next(iter(actors)) if len(actors) == 1 else None

    def _record_actor(self, identifier):
        return next((r.get('actor') for r in self.records if r['id'] == identifier), None)

    def _edge_location(self, sender, recipient):
        first, second = self.role_id(sender), self.role_id(recipient)
        if first is None or second is None:
            self.coverage.add('diagnostic_role_unknown')
            return None
        return f'edge:{first}->{second}'

    def terminal(self, actor, source):
        with self._lock:
            git = self.git.get(actor)
            active = [t for t in self.tasks.values() if t['owner'] == actor and t['status'] != 'completed']
            if active or (git and (git.get('tracked_uncommitted_paths') or 0) > 0):
                self.diagnose('unfinished_handoff', self._edge_location(actor, 'master'), [actor, 'master'],
                              {'task_ids': [t['task_id'] for t in active],
                               'role_ids': [self.role_id(actor)] if self.role_id(actor) else []},
                              level='hypothesis', source_ids=[t['id'] for t in active] + ([git['id']] if git else []))
            self._write('collaboration_terminal_observed', actor=actor, source=source, evidence_level='observed')

    def diagnose(self, category, location, actors, facts, *, level='observed', source_ids=(), suggested_operation='add_rule', incident_key=None, coverage_reasons=()):
        if category not in CATEGORIES or level not in {'observed', 'reported', 'hypothesis'}:
            raise ValueError('Unknown collaboration diagnostic')
        if self.agent_rules:
            return
        if suggested_operation not in {'add_rule', 'modify_rule', 'remove_rule', 'no_change_needed'}:
            raise ValueError('Unknown suggested operation')
        # Deduplicate the underlying incident, not snapshots of the same fact.
        identity = (category, incident_key if incident_key is not None else tuple(source_ids))
        previous = next((event for event in self.events if event['incident'] == identity), None)
        ranks = {'hypothesis': 0, 'reported': 1, 'observed': 2}
        if previous and ranks[previous['evidence_level']] >= ranks[level]:
            return
        known = list(dict.fromkeys(a for a in actors if a in {'master', *self.bindings}))
        if location and location.startswith('edge:'):
            facts = {**facts, 'role_ids': list(dict.fromkeys(
                [r for r in facts.get('role_ids') or [] if isinstance(r, str)]
                + [self.role_id(a) for a in known if self.role_id(a)]))}
        coverage = list(coverage_reasons)
        if location is None or location == 'role:None':
            location = 'unknown'
            coverage.append('diagnostic_role_unknown')
        recipients = ['master'] if location == 'team' else known if location != 'unknown' else []
        if not recipients:
            coverage.append('diagnostic_recipients_unknown')
        self.coverage.update(coverage)
        summary, summary_coverage = evidence_summary(facts)
        coverage.extend(summary_coverage)
        row = self._write('collaboration_diagnostic', category=category, location=location,
                          evidence_level=level, suggested_operation=suggested_operation, facts=facts,
                          evidence_summary=summary, coverage_reasons=coverage, source_event_ids=list(source_ids),
                          recipients=recipients)
        if previous:
            self.events.remove(previous)
        self.events.append({**row, 'incident': identity, 'observed_at': self.clock()})
        self.records.append({'id': row['id'], 'kind': 'diagnostic', 'evidence_level': level,
                             'source_event_ids': list(source_ids), 'category': category,
                             'location': location, 'suggested_operation': suggested_operation,
                             'facts': json.loads(summary)})

    def summary(self, actor):
        active_tasks = [t for t in self.tasks.values() if t.get('status') in {'pending', 'in_progress'}]
        active_diagnostics = {event['id'] for event in self.events}
        entries = active_tasks + list(self.git.values()) + [r for r in self.records
                   if r['kind'] not in {'task', 'git_change'}
                   and (r['kind'] != 'diagnostic' or r['id'] in active_diagnostics
                        and r['evidence_level'] != 'hypothesis')]
        def priority(e):
            recipient = e.get('recipient')
            addressed = recipient == actor or (actor == 'master' and recipient == 'leader')
            unresolved = e['kind'] == 'report' and not e.get('ack')
            return (0 if addressed and unresolved else 1 if unresolved and e.get('role_id') == self.role_id(actor)
                    else 2 if e['kind'] == 'diagnostic' else 3, -int(e['id'].split('-')[-1]))
        entries.sort(key=priority)
        result = {'state_version': self.sequence, 'role_id': self.role_id(actor), 'entries': [],
                  'omitted_entries': 0, 'coverage_reasons': sorted({
                      reason if reason in SUMMARY_COVERAGE else
                      'task_result_missing' if reason == 'task_status_unknown' else 'sampling_incomplete'
                      for reason in self.coverage})}
        def render():
            return text.STATE_TEMPLATE.replace('{json}', compact(result))
        for entry in entries:
            clean = {k: v for k, v in entry.items() if k not in {'observation', 'op', 'actor', 'summary', 'checks', 'recipient', 'resolve', 'ack', 'sampling_incomplete'}}
            if len(result['entries']) >= 12:
                result['omitted_entries'] += 1
                if 'entry_limit_reached' not in result['coverage_reasons']:
                    result['coverage_reasons'].append('entry_limit_reached')
                continue
            result['entries'].append(clean)
            if len(render().encode('utf-8')) > 2800:
                result['entries'][-1] = {k: clean[k] for k in ('id', 'kind', 'evidence_level', 'source_event_ids')}
                if 'entry_content_omitted' not in result['coverage_reasons']:
                    result['coverage_reasons'].append('entry_content_omitted')
                if len(render().encode('utf-8')) > 2800:
                    result['entries'].pop()
                    result['omitted_entries'] += 1
                    if 'byte_limit_reached' not in result['coverage_reasons']:
                        result['coverage_reasons'].append('byte_limit_reached')
        return render()

    def decorate(self, actor, messages, remaining, source_request, state_tail=None):
        """Mutate provider-copy only: anchored notices first, newest state at tail."""
        with self._lock:
            self.observe_history(actor, messages)
            native = [anchor(m) for m in messages]
            retained = []
            for notice in self.notices.get(actor, []):
                index = notice['anchor_index']
                if index < len(native) and native[index] == notice['anchor']:
                    retained.append(notice)
                else:
                    self._write('collaboration_notice_anchor_removed', actor=actor, notice_id=notice['id'],
                                reason='indexed_native_anchor_missing_or_changed')
            self.notices[actor] = retained
            new = []
            seen = self.closed_thresholds.setdefault(actor, set())
            threshold = 300 if remaining <= 300 else 900 if remaining <= 900 else None
            if threshold is not None and threshold not in seen:
                seen.update({900, 300} if threshold == 300 else {900})
                template = (text.CLOSING_SINGLE if actor == 'single' else
                            agent_rules_text.CLOSING_MASTER if actor == 'master' and self.agent_rules else
                            text.CLOSING_MASTER_PROPOSE if actor == 'master' and self.mode == 'propose' else
                            text.CLOSING_MASTER_OFF if actor == 'master' else text.CLOSING_WORKER)
                new.append(('closing', template.format(remaining_seconds=max(0, int(remaining)))))
            queued_keys = {}
            for kind, content, key in self.queued.pop(actor, []):
                queued_keys[len(new)] = key
                new.append((kind, content))
            if self.team and not self.agent_rules:
                delivered_now = False
                for event in self.events:
                    key = (actor, event['id'])
                    cooldown = (actor, event['category'])
                    if key in self.delivered_events or actor not in event['recipients'] or event['evidence_level'] == 'hypothesis':
                        continue
                    if event['observed_at'] - self.cooldowns.get(cooldown, float('-inf')) < 600:
                        self._write('collaboration_notice_merged', actor=actor, diagnostic_id=event['id'],
                                    category=event['category'], reason='recipient_category_cooldown')
                        self.delivered_events.add(key)
                        continue
                    if delivered_now:
                        continue
                    if not native:
                        self.coverage.add('notification_receipt_unknown')
                        continue
                    suffix = text.EVENT_OFF if self.mode == 'off' else text.EVENT_MASTER if actor == 'master' else text.EVENT_WORKER
                    new.insert(0, ('event', text.EVENT_PREFIX.format(event_id=event['id'], **event) + ' ' + suffix))
                    self.delivered_events.add(key)
                    self.cooldowns[cooldown] = self.clock()
                    delivered_now = True
            for position, (kind, content) in enumerate(new):
                if not native:
                    self.coverage.add('notification_receipt_unknown')
                    continue
                # Persist the locator only, not arbitrary tool-result contents.
                locator = {'role': native[-1].get('role'), 'tool_call_id': native[-1].get('tool_call_id'),
                           'assistant_index': sum(m.get('role') == 'assistant' for m in native) - 1,
                           'message_index': len(native) - 1}
                if (self.agent_rules and self.notice_listener is not None and position in queued_keys
                        and self.notice_listener(actor, kind, queued_keys[position]) is False):
                    continue  # its obligation is already settled: never shown
                row = self._write('collaboration_notice_injected', actor=actor, kind=kind,
                                  source_request=source_request, anchor=locator, content=content)
                self.notices[actor].append({'id': row['id'], 'anchor': native[-1],
                                           'anchor_index': len(native) - 1, 'content': content})
            # R3 fixes the native-history boundary on FIRST delivery and replay;
            # ephemeral pending/state are always current-request tail blocks.
            for index in reversed(range(len(native))):
                inserts = [n for n in self.notices[actor] if n['anchor_index'] == index]
                for notice in reversed(inserts):
                    messages.insert(index + 1, {'role': 'user', 'content': notice['content']})
            if state_tail:
                messages.append({'role': 'user', 'content': state_tail})
            if self.team and not self.agent_rules:
                messages.append({'role': 'user', 'content': self.summary(actor)})



class CollaborationTail:
    """Read only explicit structured metadata lines, never whole native output."""
    def __init__(self, path, runtime):
        self.path, self.runtime = Path(path), runtime
        self.offset, self.buffer = 0, b''
        self.bad_lines = 0
        self._lock = threading.RLock()

    def _bad_line(self):
        self.bad_lines += 1
        with self.runtime._lock:
            self.runtime.coverage.add('native_sample_bad_line')
            self.runtime._write('collaboration_sample_rejected', bad_line_count=self.bad_lines,
                                reason='malformed_native_sample', coverage_reasons=['native_sample_bad_line'])

    @staticmethod
    def _valid_fields(row):
        """Reject malformed metadata before the observer publishes any fact."""
        if row['kind'] == 'team':
            members = row['team'].get('members')
            return isinstance(members, list) and all(
                isinstance(m, dict) and isinstance(m.get('name'), str)
                and (m.get('status') is None or isinstance(m['status'], str))
                and type(m.get('shutdown_pending', False)) is bool
                and (m.get('tasks') is None or isinstance(m['tasks'], list))
                and all(isinstance(t, dict) and isinstance(t.get('id'), (str, int)) and not isinstance(t.get('id'), bool)
                        and isinstance(t.get('subject', ''), str) for t in m.get('tasks') or [])
                for m in members)
        if row['kind'] == 'task':
            value = row['task']
            return (isinstance(value.get('task_id'), (str, int)) and not isinstance(value.get('task_id'), bool)
                    and all(value.get(k) is None or isinstance(value[k], str) for k in ('owner', 'status'))
                    and (value.get('paths') is None or isinstance(value['paths'], list)
                         and all(isinstance(p, str) for p in value['paths'])))
        value = row['git']
        return (all(value.get(k) is None or isinstance(value[k], list)
                    and all(isinstance(p, str) for p in value[k])
                    for k in ('paths', 'added_paths', 'deleted_paths', 'sampling_incomplete', 'changed_paths'))
                and (value.get('new_commit') is None or type(value['new_commit']) is bool)
                and (value.get('committed_at') is None or type(value['committed_at']) in (int, float)
                     and value['committed_at'] >= 0)
                and all(value.get(k) is None or type(value[k]) is int and value[k] >= 0
                        for k in ('new_commit_count', 'uncommitted_paths', 'tracked_uncommitted_paths', 'commits_not_in_delivery_history')))

    def poll(self):
        with self._lock:
            try:
                with self.path.open('rb') as stream:
                    stream.seek(self.offset)
                    data = stream.read()
            except FileNotFoundError:
                return
            self.offset += len(data)
            lines = (self.buffer + data).split(b'\n')
            self.buffer = lines.pop()
            for raw in lines:
                if not raw.startswith(PREFIX.encode()):
                    continue
                try:
                    row = json.loads(raw[len(PREFIX):])
                    if (not isinstance(row, dict) or row.get('kind') not in {'task', 'git', 'team'}
                            or 'sample' not in row
                            or row['kind'] == 'team' and not isinstance(row.get('team'), dict)
                            or row['kind'] == 'task' and (not isinstance(row.get('task'), dict) or 'task_id' not in row['task'])
                            or row['kind'] == 'git' and (not isinstance(row.get('git'), dict) or not isinstance(row.get('actor'), str))):
                        raise ValueError('Invalid collaboration sample')
                    if not self._valid_fields(row):
                        raise ValueError('Invalid collaboration metadata fields')
                except (ValueError, TypeError, UnicodeError):
                    self._bad_line()
                    continue
                try:
                    if row['kind'] == 'team':
                        self.runtime.observe_team(row['team'], {'native_sample': row['sample']})
                    elif row['kind'] == 'task':
                        self.runtime.observe_task(row['task'], actor='master', source={'native_sample': row['sample']})
                    elif row['kind'] == 'git':
                        self.runtime.observe_git(row['actor'], row['git'], {'native_sample': row['sample']})
                except (ValueError, TypeError, KeyError):
                    # Persistence failures deliberately propagate as a run-level
                    # failure; only malformed per-line metadata is skipped.
                    self._bad_line()
