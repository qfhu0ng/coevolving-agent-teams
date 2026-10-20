"""Profile-selected S0 runtime smoke, separate from any benchmark rollout.

Exercises the selected arm in the profile's single ProgramBench task
container, without solving that task or running its official evaluator. Mode, A/C
and runtime switches come from the formal plan parser. --scripted makes zero
external model calls and never loads provider credentials. Its scripted responses
exercise the role-policy protocol for global/master/propose, or a file roundtrip for
Single/off Team. A bounded real-provider smoke requests
a synthetic file task but only checks mechanism wiring, not the resulting file;
it requires the explicit --real flag and separate authorization.

Only injected system text and relay/lifecycle metadata are inspected. Scripted
success establishes mechanism wiring, not real-model compliance, API health or task
quality. The active-teammate check here verifies the recorded ten-worker setting;
the separate real-CLI scenario verifies rejection at ten and reuse after shutdown.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import uuid

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / 'tests')]

from agent_company.runtime.provider import load_provider
from agent_company.programbench.comparison import prepare_comparison, runner_inference_options
from agent_company.analysis.evidence_projection import project_evidence
from agent_company.programbench.workspace import ProgramBenchWorkspace, programbench_source_boundary
from agent_company.runtime.qwen import RUNTIME_COMMUNICATION_POLICY, QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig
from agent_company.policy.role_policy import RolePolicyStore
from agent_company.policy.role_policy_runtime import DISCLOSURE_VERSION, disclosure
from role_policy_scenario import RolePolicyScenario, system

INSTRUCTION = ('Infrastructure compatibility test only; do not solve the benchmark task or inspect reference '
               'behavior. Create /workspace/solution/native-smoke.txt containing exactly NATIVE_OK followed by '
               'a newline, and read it back to verify. Use one writable teammate for this. No compile.sh or '
               'executable is needed. ')
EXERCISE = {
    'off': 'Report the file roundtrip, then complete the normal shutdown handshake and finish.',
    'master': ('After the teammate reports, use the documented ROLE-POLICY control to replace its role policy once. '
               'Have it acknowledge the update, then complete the normal shutdown handshake and finish.'),
    'propose': ('This run also checks the role-policy controls in your system instructions. When you spawn the '
                'teammate, include an initial role policy with the spawn marker. Ask the teammate to submit '
                'exactly one formal proposal for its own communication policy after it verifies the file. Decide '
                'the proposal, then complete the normal shutdown handshake and finish.'),
    'global': ('This run also checks the communication-policy control in your system instructions. After the '
               'teammate reports, replace the communication policy C once using the documented broadcast format, '
               'then complete the normal shutdown handshake and finish.'),
}


def basic_runtime_checks(launch, events, runtime, options, arm, assignment, communication):
    """Check Single/off-Team wiring without requiring an online-policy ledger."""
    single = arm['execution_mode'] == 'single'
    loaded = [event for event in events if event.get('event') == 'compatibility_loaded']
    environment = launch.get('environment', {})
    static = launch['command'][launch['command'].index('--system-prompt') + 1]
    checks = {
        'launch_records_execution_mode': launch.get('execution_mode') == arm['execution_mode'],
        'compat_loaded': bool(loaded),
        'team_enabled_matches_arm': environment.get('QWEN_CODE_ENABLE_AGENT_TEAM') == ('0' if single else '1'),
        'online_role_policies_disabled': launch.get('role_policy_mode') == (None if single else 'off')
                                         and 'AGENT_COMPANY_ROLE_POLICY_MODE' not in environment,
        'no_online_policy_state': not (runtime / 'relay/role_policy_state.json').exists(),
        'launch_records_runtime_switches': launch.get('runtime_disclosure') == options['runtime_disclosure']
                                          and launch.get('tool_call_cap') == options['tool_call_cap'],
    }
    if single:
        checks.update(single_has_no_policies=launch.get('assignment_policy') is None
                      and launch.get('communication_policy') is None and launch.get('policy_applied') is False,
                      single_has_no_team_hooks=launch.get('team_checkin') is None
                      and launch.get('team_message_delivery') is None and launch.get('leader_loop_recovery') is None)
    else:
        workers = system_texts(runtime / 'relay', False)
        checks.update(master_static_prompt_carries_a_and_c=assignment.strip() in static and communication.strip() in static,
                      worker_requests_carry_fixed_c=bool(workers) and all(communication.strip() in text for text in workers),
                      launch_records_active_teammate_limit=launch.get('teammate_limit') == 10)
    return checks


def lifecycle(path: Path) -> list[dict]:
    prefix = 'Agent Company lifecycle: '
    return [json.loads(line[len(prefix):]) for line in path.read_text(errors='replace').splitlines()
            if line.startswith(prefix)]


def coding_attempts(relay: Path) -> set[int]:
    """Attempt ids whose request purpose was coding; injection is coding-only."""
    attempts = set()
    for line in (relay / 'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('event') == 'provider_attempt_started' and event.get('purpose') == 'coding':
            attempts.add(event.get('attempt'))
    return attempts


def system_texts(relay: Path, master: bool) -> list[str]:
    """System text of coding requests only; skip compaction and auxiliary requests."""
    coding = coding_attempts(relay)
    texts = []
    for path in sorted(relay.glob('attempt_*_provider_request.json')):
        identifier, role = path.name.split('_', 2)[1], path.name.split('_', 2)[2].rsplit('_provider_request.json', 1)[0]
        if int(identifier) not in coding or (role == 'master') != master:
            continue
        body = json.loads(path.read_text())
        texts.append('\n'.join(str(m.get('content', '')) for m in body.get('messages', []) if m.get('role') == 'system'))
    return texts


def copy_trial(runtime: Path, trial: Path) -> Path:
    """Copy the evidence the projection reads; it refuses symlinks/outside files."""
    (trial / 'team').mkdir(parents=True)
    shutil.copytree(runtime / 'relay', trial / 'team/relay')
    for name in ('qwen_stderr.txt', 'summary.json'):
        if (runtime / name).is_file():
            shutil.copy2(runtime / name, trial / 'team' / name)
    return trial


def update_reached_worker(mode: str, scenario) -> bool:
    """The worker request after the decision/broadcast carries the new version."""
    payload = getattr(scenario, 'worker_after_update', None)
    if payload is None:
        return False
    text = system(payload)
    if mode == 'master':
        return ('Your role policy version 2' in text
                and 'Message the leader only for blockers and the final report.' in text)
    if mode == 'propose':
        return ('Your role policy version 2' in text and 'Your proposal #1 was approved as version 2.' in text
                and 'Message the leader only for blockers' in text)
    return 'Communication policy C (version 2):\nReport only blockers and results.' in text


def scripted_controls_committed(mode, rows):
    changes = [change for row in rows if row.get('event') == 'role_policy_batch' for change in row['changes']]
    kinds = {change['event'] for change in changes}
    if mode == 'propose':
        return {'role_policy_proposed', 'role_policy_decided'} <= kinds
    if mode == 'master':
        return any(change['event'] == 'role_policy_committed' and change.get('provenance') == 'master_set'
                   and change.get('slot') == 'alpha' and change.get('version') == 2 for change in changes)
    return mode == 'global' and 'global_policy_committed' in kinds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--arm', help='Select one named arm from a multi-arm profile; never runs both')
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, default=40)
    parser.add_argument('--timeout', type=float, default=420)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument('--scripted', action='store_true', help='zero external model calls; no credentials loaded')
    execution.add_argument('--real', action='store_true', help='bounded provider run; requires separate authorization')
    args = parser.parse_args(argv)
    if args.max_calls <= 0 or not 0 < args.timeout < float('inf'):
        parser.error('--max-calls and --timeout must be positive finite values')
    prepared = prepare_comparison(args.profile.resolve(), CODE)
    plan = prepared.plan
    if "role_policy_protocol" in plan['inference']:
        parser.error('This legacy smoke does not support role_policy_protocol; use the role-package CLI scenarios')
    if len(prepared.tasks) != 1:
        parser.error('--profile must select exactly one ProgramBench task')
    arms = [arm for arm in plan['arms'] if args.arm is None or arm['arm_id'] == args.arm]
    if len(arms) != 1:
        parser.error('Select exactly one profile arm with --arm')
    (arm,) = arms
    options = runner_inference_options(plan['inference'])
    mode = options['role_policy_mode'] if arm['execution_mode'] == 'team' else 'off'
    if mode not in EXERCISE:
        parser.error('--profile Team role_policy_mode must be off, global, master or propose')
    active_policy = arm['execution_mode'] == 'team' and mode != 'off'
    assignment, communication = prepared.policies[arm['arm_id']]
    (selected_task,) = prepared.tasks
    task = replace(selected_task, agent_timeout_sec=args.timeout)
    assets, output = args.assets_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ws = ProgramBenchWorkspace(task, docker=str(assets / 'tools/docker'), platform='linux/amd64',
        container_name='ac-s0-smoke-' + uuid.uuid4().hex[:10], mounts=[
            (assets / 'external/qwen-study/package', '/opt/ac-qwen', True),
            (assets / 'external/stage1-runtime/node-v22.23.2-linux-x64', '/opt/ac-node', True)])
    summary = {'mode': mode, 'arm': arm['arm_id'], 'execution_mode': arm['execution_mode'], 'profile': str(args.profile.resolve()),
               'task_id': task.task_id, 'image': task.image, 'passed': False, 'scripted': args.scripted,
               'official_evaluation_run': False, 'benchmark_solution_attempted': False,
               'external_model_calls': 0 if args.scripted else None,
               'formal_switches': {name: options[name] for name in (
                   'runtime_disclosure', 'leader_loop_recovery', 'tool_call_cap', 'team_checkin_seconds',
                   'team_message_delivery', 'role_policy_mode')},
               'diagnostic_overrides': {'task': (('scripted role-policy protocol' if active_policy else 'scripted file roundtrip') if args.scripted else
                                                'synthetic file instruction; file contents not checked'),
                                        'timeout_seconds': args.timeout,
                                        'max_model_calls': args.max_calls}}
    try:
        # Do not load the profile's provider or read its credentials in scripted mode.
        if not args.scripted:
            scenario = None
        elif active_policy:
            scenario = RolePolicyScenario({'propose': 'proposal', 'global': 'global', 'master': 'master_set_text_ack'}[mode])
        else:
            # Single/off Team only: global/propose smokes never import this scenario.
            from basic_smoke_scenario import BasicSmokeScenario
            scenario = BasicSmokeScenario(arm['execution_mode'])
        provider = (ProviderConfig('scripted-s0-smoke', 'http://unused.invalid', request_timeout_seconds=40)
                    if args.scripted else
                    replace(load_provider(Path(plan['provider_config'])), max_attempts=1,
                            request_timeout_seconds=120, first_progress_timeout_seconds=90,
                            progress_idle_timeout_seconds=90))
        summary['diagnostic_overrides'].update({
            'provider_max_attempts': provider.max_attempts,
            'provider_request_timeout_seconds': provider.request_timeout_seconds,
            'provider_first_progress_timeout_seconds': provider.first_progress_timeout_seconds,
            'provider_progress_idle_timeout_seconds': provider.progress_idle_timeout_seconds,
            'provider_retry_backoff_seconds': provider.retry_backoff_seconds,
        })
        ws.start()
        summary.update(container_id=ws.container_id, image_identity=ws.image_identity)
        summary['node'] = ws.run(['/opt/ac-node/bin/node', '--version']).stdout.strip()
        instruction = (INSTRUCTION + EXERCISE[mode] if arm['execution_mode'] == 'team' else
                       INSTRUCTION.replace('Use one writable teammate for this.', 'Perform the file roundtrip directly.'))
        result = QwenTeamRunner('/opt/ac-qwen/cli.js', node='/opt/ac-node/bin/node', upstream=scenario,
            checkin_interval_seconds=options['team_checkin_seconds'],
            team_message_delivery=options['team_message_delivery'],
            runtime_disclosure=options['runtime_disclosure'],
            leader_loop_recovery=options['leader_loop_recovery'],
            tool_call_cap=options['tool_call_cap'], role_policy_mode=mode,
            stop_hook=options.get('stop_hook')).run(
                ws, instruction, assignment, communication, output / 'runtime', provider,
                timeout_seconds=args.timeout, execution_mode=arm['execution_mode'], max_model_calls=args.max_calls,
                source_boundary=programbench_source_boundary())
        execution = result.get('execution', {})
        summary.update(returncode=execution.get('returncode'), timed_out=execution.get('timed_out'),
                       elapsed_seconds=execution.get('elapsed_seconds'), stop_reason=result.get('stop_reason'),
                       logical_model_calls=result.get('logical_model_calls'),
                       provider_attempts=result.get('provider_attempts'),
                       usage_complete=result.get('usage_complete'),
                       external_model_calls=0 if args.scripted else result.get('provider_attempts'))
        runtime = output / 'runtime'
        launch = json.loads((runtime / 'launch.json').read_text())
        report = project_evidence(copy_trial(runtime, output / 'trial'))
        projected = report.get('role_policies') or {}
        events = lifecycle(runtime / 'qwen_stderr.txt')
        if active_policy:
            static = launch['command'][launch['command'].index('--system-prompt') + 1]
            events = lifecycle(runtime / 'qwen_stderr.txt')
            loaded = [e for e in events if e.get('event') == 'compatibility_loaded']
            masters, workers = system_texts(runtime / 'relay', True), system_texts(runtime / 'relay', False)
            state = json.loads((runtime / 'relay/role_policy_state.json').read_text())
            replayed = RolePolicyStore.replay(RolePolicyStore.read_log(runtime / 'relay/events.jsonl'), lambda r: None)
            checks = {
                'launch_records_mode': launch.get('role_policy_mode') == mode
                                       and launch['environment'].get('AGENT_COMPANY_ROLE_POLICY_MODE') == mode,
                'compat_loaded_with_mode': bool(loaded) and all(e.get('role_policy_mode') == mode for e in loaded),
                'launch_records_active_teammate_limit': launch.get('teammate_limit') == 10,
                'compat_records_active_teammate_limit': bool(loaded) and all(
                    e.get('teammate_limit') == 10 for e in loaded),
                'launch_records_disclosure_version': launch.get('role_policy_disclosure_version') == DISCLOSURE_VERSION,
                'launch_records_disclosure_lengths': launch.get('role_policy_disclosure_chars') == {
                    role: len(disclosure(mode, role)) for role in ('master', 'worker')},
                'master_static_prompt_omits_c': RUNTIME_COMMUNICATION_POLICY in static and communication.strip() not in static,
                'master_requests_carry_versioned_c_and_disclosure': bool(masters) and all(
                    'Communication policy C (version' in text and disclosure(mode, 'master') in text for text in masters),
                'worker_requests_carry_disclosure': bool(workers) and all(
                    disclosure(mode, 'worker') in text for text in workers),
                'ledger_replays_to_final_state': replayed.export() == state,
                'projection_reports_role_policies': projected.get('status') == 'on'
                                                    and projected.get('ledger', {}).get('replay') == 'ok',
            }
            if mode == 'propose':
                checks['policy_report_patch_applied'] = any(e.get('event') == 'policy_report_patch_applied' for e in events)
            if args.scripted:
                checks['scripted_controls_committed'] = scripted_controls_committed(
                    mode, RolePolicyStore.read_log(runtime / 'relay/events.jsonl'))
                checks['update_reached_worker_request'] = update_reached_worker(mode, scenario)
                checks['updated_report_reached_master_request'] = scenario.master_saw('ALPHA_ACK')
                checks['scripted_shutdown_handshake_completed'] = (scenario.shutdown_sent
                    and scenario.shutdown_seen == {'alpha'} and scenario.master_saw('shutdown_approved'))
        else:
            checks = basic_runtime_checks(launch, events, runtime, options, arm, assignment, communication)
            if args.scripted:
                checks['scripted_file_roundtrip'] = scenario.file_verified
                if arm['execution_mode'] == 'team':
                    checks['scripted_shutdown_requested'] = scenario.shutdown_sent
        summary['checks'] = checks
        summary['observed'] = {key: projected.get(key) for key in (
            'proposals', 'notifications', 'rejections', 'spawn_results', 'broadcasts', 'injection')}
        summary['observed']['versions'] = [{k: row.get(k) for k in ('scope', 'slot', 'version', 'provenance',
                                                                 'first_carry_latency_s')}
                                         for row in projected.get('versions') or []]
        summary['observed']['later_version_carried'] = any(
            (row.get('version') or 0) >= 2 and row.get('first_carry_latency_s') is not None
            for row in projected.get('versions') or [])
        summary['prompt_cache'] = report.get('prompt_cache', {}).get('total')
        assert all(checks.values()), checks
        assert summary['returncode'] == 0 and not summary['timed_out'], 'CLI did not finish cleanly'
        assert summary['stop_reason'] == 'native_cli_completed', summary['stop_reason']
        assert ws.network_policy_status['enforced']
        summary.update(passed=True, network_enforced=True)
    except Exception as exc:
        summary['exception_type'] = type(exc).__name__
        if isinstance(exc, AssertionError):
            summary['exception'] = str(exc)[:500]
    finally:
        try:
            try:
                if ws.container_id:
                    ws.finish_agent()
            finally:
                ws.stop()
            summary['cleanup_confirmed'] = True
        except Exception as exc:
            summary['cleanup_exception_type'] = type(exc).__name__
        # stop() keeps the identity; a start that failed after docker run still records it.
        summary.setdefault('image_identity', getattr(ws, 'image_identity', None))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, default=str), flush=True)
    return 0 if summary['passed'] and summary.get('cleanup_confirmed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
