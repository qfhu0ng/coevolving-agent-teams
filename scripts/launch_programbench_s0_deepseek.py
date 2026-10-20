#!/usr/bin/env python3
"""Task-explicit DeepSeek S0 preparation and separately authorized execution steps.

Every command requires --task-id and --deployment-id; there is no task or date default.
No command automatically runs the next step. --deploy, --remote-tests, --prepare and
--smoke MODE --scripted do not load or transmit provider credentials. --smoke requires
an explicit --scripted or --real choice. Real smoke,
the six-call cache probe and formal launch are separate, paid operations.

The selected DeepSeek configuration uses streaming and explicit thinking/high:
300-second first-progress/idle timers and a shared 1800-second logical deadline
including retries/backoff; max_attempts=3 is a cap. Scripted smoke does not certify
the provider or these production timing windows.
"""
import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_company.deploy import ssh as launch_transport
from agent_company.deploy.s0_contract import MODES, POLICY, POLICY_SHA256, inference  # noqa: F401 (re-exported)


PROJECT = Path(os.environ.get('AGENT_COMPANY_REMOTE_ROOT', '/srv/agent-company'))
HOST, KEY, KNOWN = launch_transport.HOST, launch_transport.KEY, launch_transport.KNOWN
TEMPLATES = {mode: Path(f'configs/stage1/s0_deepseek_{mode}.template.json') for mode in MODES}
PROFILES = {mode: Path(f'configs/stage1/s0_deepseek_{mode}.json') for mode in MODES}
PROVIDER_CONFIG = Path('configs/stage1/deepseek_flash.json')
PROVIDER = {
    'model': 'deepseek-flash', 'base_url': 'https://api.deepseek.com',
    'api_key_env': 'DEEPSEEK_API_KEY', 'max_output_tokens': 65536,
    'request_timeout_seconds': 1800, 'first_progress_timeout_seconds': 300,
    'progress_idle_timeout_seconds': 300, 'inspection_max_output_tokens': 32768,
    'stream_response': True, 'max_attempts': 3, 'retry_backoff_seconds': 1.0,
    'extra_body': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}, 'rates': None,
}
CREDENTIAL_ENV_NAMES = ('DEEPSEEK_API_KEY', 'STAGE1_GLM_API_KEY')
TASKS = Path('external/programbench-study/src/programbench/data/tasks')
INSTRUCTIONS = 'configs/stage1/programbench_instruction.md'
# Reuse the existing isolated evaluator/dependency artifacts, not the old task or run.
EVALUATOR_SOURCE = PROJECT / 'runs/stage1_kimi_programbench_ffmpeg_p1_rerun2_20261003/evaluator_fix'
TEST_DEPENDENCIES = PROJECT / 'runs/stage1_kimi_programbench_ffmpeg_p1_rerun2_20261003/validation/offline_test_dependencies.zip'
QWEN_PACKAGE = PROJECT / 'external/qwen-study/package'
NODE_DIRECTORY = PROJECT / 'external/stage1-runtime/node-v22.23.2-linux-x64'
EXTERNAL = PROJECT / 'external'
SNAPSHOT_PATHS = ['agent_company', 'configs/stage1', 'scripts', 'tests', 'docs', 'pyproject.toml']
CLEAN_PATHS = ['agent_company', 'configs/stage1', 'scripts', 'tests', 'pyproject.toml']
LAUNCHER = 'scripts/launch_programbench_s0_deepseek.py'
OVERLAYS = {name: name for name in (
    LAUNCHER, 'tests/test_programbench_s0_deepseek_launch.py',
    'scripts/s0_native_smoke.py', 'scripts/s0_cache_probe.py',
    *(str(path) for path in TEMPLATES.values()), str(PROVIDER_CONFIG),
)}
ROLE_POLICY_TESTS = [
    'tests/test_role_policy.py', 'tests/test_role_policy_runtime.py',
    'tests/test_role_policy_relay.py', 'tests/test_role_policy_integration.py',
    'tests/test_role_policy_projection.py', 'tests/test_programbench_ac_evidence.py',
    'tests/test_harness_capabilities.py', 'tests/test_executability_protocol.py',
    'tests/test_programbench.py', 'tests/test_programbench_workspace.py',
    'tests/test_programbench_runner.py', 'tests/test_team.py',
    'tests/test_native_compat.py', 'tests/test_team_checkin.py',
    'tests/test_qwen.py', 'tests/test_runtime_outcomes.py',
    'tests/test_relay_failures.py', 'tests/test_stream_transport.py',
    'tests/test_progress_transport.py', 'tests/test_progress_recovery.py',
    'tests/test_deepseek_streaming.py', 'tests/test_s0_cache_probe.py',
    'tests/test_shell_output_drain.py',
    'tests/test_s0_report.py', 'tests/test_native_final_report.py',
    'tests/test_code_version.py', 'tests/test_programbench_total_evidence.py',
    'tests/test_programbench_total_selection.py', 'tests/test_programbench_total_prompts.py',
    'tests/test_programbench_policy_comparison.py', 'tests/test_programbench_evolved_followup.py',
    'tests/test_programbench_baseline_prepare.py', 'tests/test_programbench_baseline_smoke.py',
    'tests/test_programbench_disabled_launchers.py', 'tests/test_programbench_qualitative_triad.py',
    'tests/test_programbench_qualitative_triad_launcher.py',
]


def _identifier(value, name):
    if (not isinstance(value, str) or len(value) > 100
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is None):
        raise ValueError(f'{name} must be an explicit safe directory identifier')
    return value


@dataclass(frozen=True)
class Selection:
    task_id: str
    deployment_id: str

    def __post_init__(self):
        _identifier(self.task_id, 'task-id')
        _identifier(self.deployment_id, 'deployment-id')


def _mode(mode):
    if mode not in MODES:
        raise ValueError('Unknown S0 mode')
    return mode


def run_name(selection: Selection, mode: str) -> str:
    return f'stage1_deepseek_programbench_s0_{selection.task_id}_{_mode(mode)}_{selection.deployment_id}'


def branch(selection: Selection, mode: str) -> Path:
    return PROJECT / 'runs' / run_name(selection, mode)


def local_run(selection: Selection) -> Path:
    return Path('runs') / f'stage1_deepseek_programbench_s0_{selection.task_id}_{selection.deployment_id}'


def build_profile(root: Path, mode: str, task_id: str) -> dict:
    """Pure rendering: bind the explicit task and its description without authorizing execution."""
    _identifier(task_id, 'task-id')
    profile = json.loads((root / TEMPLATES[_mode(mode)]).read_text(encoding='utf-8'))
    if profile.get('tasks') != []:
        raise ValueError('S0 templates must not contain an implicit task')
    if (profile.get('provider_config') != str(PROVIDER_CONFIG)
            or profile.get('inference') != inference(mode)):
        raise ValueError('S0 template provider/runtime differs from the selected settings')
    profile['tasks'] = [{'task_id': task_id, 'path': str(TASKS / task_id), 'instructions': INSTRUCTIONS}]
    profile['purpose'] = profile['purpose'].replace('Task-unbound S0 template', 'Task-bound S0 profile', 1).replace(
        'Bind an explicitly selected public ProgramBench task via the generic S0 launcher. No default task is selected.',
        f'Selected public task: {task_id}. This local binding does not authorize deployment, real model calls or execution.',
        1)
    profile['limitations'] = profile['limitations'].replace(
        'No task has been selected, and this template cannot run until a public task is explicitly bound.',
        f'The public task {task_id} is bound for S0 mechanism observation only; this is not a formal task-quality experiment.',
        1)
    return profile


def _public_provider(path: Path) -> dict:
    """Read only the public configuration, never resolve its credential environment."""
    value = json.loads(path.read_text(encoding='utf-8'))
    if value != PROVIDER:
        raise ValueError('DeepSeek provider settings differ from the selected public configuration')
    return value


def clean_environment() -> dict:
    return launch_transport.clean_environment()


# ----- server side ----------------------------------------------------------------------

def prepare(selection: Selection, mode: str) -> dict:
    """Read deployed public inputs only; no credential access, writes or process launch."""
    root = branch(selection, mode)
    code = root / 'code'
    for name in ('run', 'launch.json', 'controller.log'):
        if (root / name).exists():
            raise FileExistsError('Existing launch/output; inspect, do not launch again')
    sys.path.insert(0, str(code))
    from agent_company.programbench.comparison import prepare_comparison
    plan = prepare_comparison(code / PROFILES[mode], code).plan
    arms = [(arm['arm_id'], arm['execution_mode'], arm['policy_version']) for arm in plan['arms']]
    if (plan['new_rollouts'] != 1 or plan['parallelism'] != 1 or plan['reused_rollouts'] != 0
            or plan['runs_per_task_arm'] != 1 or plan['fresh_start'] is not True
            or plan['evolution_enabled'] is not False or plan['automatic_restarts'] is not False
            or [task['task_id'] for task in plan['tasks']] != [selection.task_id]
            or arms != [(f'team_p0_{mode}', 'team', 0)] or plan['inference'] != inference(mode)):
        raise ValueError('Plan differs from the selected S0 run')
    profile = json.loads((code / PROFILES[mode]).read_text(encoding='utf-8'))
    expected_task = {'task_id': selection.task_id, 'path': str(TASKS / selection.task_id),
                     'instructions': INSTRUCTIONS}
    if profile.get('tasks') != [expected_task]:
        raise ValueError('Profile must use the selected public task path and standard instructions')
    if not (code / TASKS / selection.task_id).is_dir():
        raise FileNotFoundError('Selected task directory is unavailable')
    paths = [arm.get(field) for arm in plan['arms'] for field in ('assignment_policy_path', 'communication_policy_path')]
    if paths != [str((code / POLICY / name).resolve()) for name in POLICY_SHA256]:
        raise ValueError('S0 must use the saved fresh P0 policies')
    for name, digest in POLICY_SHA256.items():
        if hashlib.sha256((code / POLICY / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Team A/C differs from the verified fresh P0 policy')
    if Path(plan['provider_config']).resolve() != (code / PROVIDER_CONFIG).resolve():
        raise ValueError('S0 must reference the DeepSeek public provider configuration')
    provider = _public_provider(Path(plan['provider_config']))
    if plan['evaluation']['docker_cpus'] != 10:
        raise ValueError('Evaluator settings differ from the selected S0 run')
    evaluator = root / 'evaluator_fix'
    if not (evaluator / 'programbench/eval/eval.py').is_file():
        raise FileNotFoundError('Isolated evaluator missing')
    command = [str(PROJECT / '.venv/bin/python'), '-u', str(code / 'scripts/run_programbench_policy_comparison.py'),
               '--run', '--repeat-index', '0', '--profile', str(code / PROFILES[mode]), '--project-root', str(code),
               '--qwen-package', str(QWEN_PACKAGE), '--node-directory', str(NODE_DIRECTORY),
               '--docker', str(PROJECT / 'tools/docker'), '--output', str(root / 'run')]
    return {'status': 'prepared_not_launched', 'mode': mode, 'task_id': selection.task_id,
            'deployment_id': selection.deployment_id, 'plan': plan, 'command': command, 'cwd': str(code),
            'provider_recovery': provider, 'pythonpath': os.pathsep.join((str(evaluator), str(code)))}


def credential(payload):
    return launch_transport.credential(payload)


def remote_launch(selection: Selection, mode: str, payload) -> dict:
    prepared = prepare(selection, mode)
    key = credential(payload)
    from agent_company.runtime_io import write_json_atomic
    root = branch(selection, mode)
    record = {'status': 'launching', 'run_name': run_name(selection, mode), 'mode': mode,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'cwd': prepared['cwd'], 'command': prepared['command'], 'pythonpath': prepared['pythonpath'],
              'planned_rollouts': 1, 'reused_rollouts': 0, 'parallelism': 1, 'fresh_start': True,
              'task_ids': [selection.task_id], 'deployment_id': selection.deployment_id,
              'arms': [arm['arm_id'] for arm in prepared['plan']['arms']], 'inference': prepared['plan']['inference'],
              'evaluation': prepared['plan']['evaluation'], 'provider_recovery': prepared['provider_recovery'],
              'automatic_task_retries': 0, 'evolution_enabled': False, 'optimizer_sessions': 0,
              'reviewer_sessions': 0, 'single_arm': False, 'team_policy_sha256': POLICY_SHA256,
              'heldout': False, 'detached_session': True, 'credential_persisted': False}
    with (root / 'launch.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    environment = {**clean_environment(), 'DEEPSEEK_API_KEY': key, 'PYTHONPATH': prepared['pythonpath']}
    with (root / 'controller.log').open('xb') as stream:
        process = subprocess.Popen(prepared['command'], cwd=prepared['cwd'], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    record.update(status='started', pid=process.pid, started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json_atomic(root / 'launch.json', record)
    return {name: record[name] for name in ('run_name', 'mode', 'pid', 'started_at_utc', 'planned_rollouts',
                                            'parallelism', 'fresh_start', 'evolution_enabled', 'single_arm')}


def remote_preparation(selection: Selection, kind: str, mode: str, payload) -> dict:
    """Run exactly one explicitly chosen preparation step; no chained or automatic retry."""
    if kind not in ('smoke_scripted', 'smoke_real', 'cache_probe'):
        raise ValueError('Unknown preparation step')
    prepare(selection, mode)
    root = branch(selection, mode)
    code = root / 'code'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'preparation_runs' / f'{kind}_{stamp}'
    output.parent.mkdir(exist_ok=True)
    environment = clean_environment()
    environment['PYTHONPATH'] = str(code)
    key = None
    if kind == 'cache_probe':
        command = ['scripts/s0_cache_probe.py', '--profile', str(code / PROFILES[mode])]
    else:
        command = ['scripts/s0_native_smoke.py', '--profile', str(code / PROFILES[mode]),
                   '--assets-root', str(PROJECT)]
        if kind == 'smoke_scripted':
            command.append('--scripted')
        else:
            command.append('--real')
    if kind != 'smoke_scripted':
        key = credential(payload)
        environment['DEEPSEEK_API_KEY'] = key
    command = [str(PROJECT / '.venv/bin/python'), '-u', *command, '--output', str(output)]
    completed = subprocess.run(command, cwd=code, env=environment, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=1800)
    lines = [line for line in completed.stdout.splitlines() if line.startswith('{')]
    stderr = completed.stderr[-1500:] if completed.returncode else ''
    if key:
        stderr = stderr.replace(key, '[REDACTED]')
    return {'kind': kind, 'mode': mode, 'returncode': completed.returncode, 'output': str(output),
            'summary': json.loads(lines[-1]) if lines else None, 'stderr_tail': stderr}


# ----- workstation side -----------------------------------------------------------------

def ssh(remote, *, stdin=None, timeout=120):
    return launch_transport.ssh(remote, stdin=stdin, timeout=timeout, host=HOST, key=KEY, known=KNOWN)


def snapshot(root: Path, selection: Selection) -> tuple[bytes, dict]:
    """HEAD plus explicitly listed overlays/generated profiles; no credential reads."""
    status = subprocess.run(['git', '-C', str(root), 'diff', '--name-only', '-z', 'HEAD', '--', *CLEAN_PATHS],
                            capture_output=True, text=True, check=True).stdout
    dirty = [name for name in status.split('\0') if name and name not in OVERLAYS]
    if dirty:
        raise RuntimeError('Tracked deploy paths outside the explicit overlays have uncommitted changes')
    _public_provider(root / PROVIDER_CONFIG)
    generated = {str(PROFILES[mode]): (json.dumps(build_profile(root, mode, selection.task_id), indent=2) + '\n').encode()
                 for mode in MODES}
    archive = subprocess.run(['git', '-C', str(root), 'archive', '--format=tar', 'HEAD', '--', *SNAPSHOT_PATHS],
                             capture_output=True, check=True).stdout
    manifest, buffer = {}, io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source, tarfile.open(fileobj=buffer, mode='w:gz') as target:
        for member in source.getmembers():
            if member.isfile() and member.name not in OVERLAYS and member.name not in generated:
                data = source.extractfile(member).read()
                manifest[member.name] = hashlib.sha256(data).hexdigest()
                target.addfile(member, io.BytesIO(data))
        for deployed, local in OVERLAYS.items():
            path = root / local
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Overlay must be a regular local file: {local}')
            data = path.read_bytes()
            manifest[deployed] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(deployed)
            info.size, info.mode = len(data), 0o644
            target.addfile(info, io.BytesIO(data))
        for deployed, data in generated.items():
            manifest[deployed] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(deployed)
            info.size, info.mode = len(data), 0o644
            target.addfile(info, io.BytesIO(data))
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    return buffer.getvalue(), {'head': head, 'files': manifest, 'overlays': dict(OVERLAYS),
                              'generated_profiles': sorted(generated), 'task_id': selection.task_id,
                              'deployment_id': selection.deployment_id}


def deploy(root: Path, selection: Selection) -> dict:
    data, manifest = snapshot(root, selection)
    staging = PROJECT / 'runs' / f'stage1_deepseek_programbench_s0_{selection.task_id}_staging_{selection.deployment_id}'
    branches = ' '.join(shlex.quote(str(branch(selection, mode))) for mode in MODES)
    hash_files = {mode: str(branch(selection, mode) / 'code_sha256.txt') for mode in MODES}
    hash_report = ('import json; from pathlib import Path; '
                   f'print(json.dumps({{mode:Path(path).read_text() for mode,path in {hash_files!r}.items()}}))')
    task_path = TASKS / selection.task_id
    script = f'''set -eu
S={shlex.quote(str(staging))}
test ! -e "$S"
for B in {branches}; do test ! -e "$B"; done
test -f {shlex.quote(str(EVALUATOR_SOURCE / "programbench/eval/eval.py"))}
test -f {shlex.quote(str(TEST_DEPENDENCIES))}
test -d {shlex.quote(str(PROJECT / task_path))}
mkdir "$S"
base64 -d | tar -xz -C "$S"
for B in {branches}; do
  mkdir -p "$B/validation"
  cp -a "$S" "$B/code"
  ln -s {shlex.quote(str(EXTERNAL))} "$B/code/external"
  test -d "$B/code/{task_path}"
  cp -a {shlex.quote(str(EVALUATOR_SOURCE))} "$B/evaluator_fix"
  cp -a {shlex.quote(str(TEST_DEPENDENCIES))} "$B/validation/"
  (cd "$B/code" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$B/code_sha256.txt"
done
{shlex.quote(str(PROJECT / '.venv/bin/python'))} -c {shlex.quote(hash_report)}
'''
    result = ssh(f'bash -o pipefail -c {shlex.quote(script)}', stdin=base64.b64encode(data).decode(), timeout=300)
    if result.returncode:
        raise RuntimeError('Deploy failed: ' + result.stderr[-800:])
    manifests = json.loads(result.stdout)
    if set(manifests) != set(MODES):
        raise RuntimeError('Deploy did not return both branch hash manifests')
    verified = {}
    for mode in MODES:
        remote = {line.split(None, 1)[1].removeprefix('./'): line.split(None, 1)[0]
                  for line in manifests[mode].splitlines() if line.strip()}
        if remote != manifest['files']:
            raise RuntimeError(f'Deployed hashes differ for {mode}; inspect before any retry')
        verified[mode] = len(remote)
    record = {'status': 'deployed', 'base_git_commit': manifest['head'], 'files': len(manifest['files']),
              'verified_files_by_mode': verified, 'task_id': selection.task_id,
              'deployment_id': selection.deployment_id,
              'branches': [str(branch(selection, mode)) for mode in MODES],
              'overlays': sorted(OVERLAYS), 'overlay_sources': manifest['overlays'],
              'generated_profiles': manifest['generated_profiles'],
              'shared_links': {'code/external': str(EXTERNAL)}, 'evaluator_fix_source': str(EVALUATOR_SOURCE),
              'external_model_calls': 0, 'formal_run_started': False,
              'deployed_at_utc': datetime.now(timezone.utc).isoformat()}
    local = root / local_run(selection)
    (local / 'preparation').mkdir(parents=True, exist_ok=True)
    with (local / 'DEPLOYMENT.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    (local / 'preparation/code_sha256.txt').write_text(
        ''.join(f'{digest} ./{name}\n' for name, digest in sorted(manifest['files'].items())))
    return record


def remote_tests(selection: Selection) -> dict:
    code = branch(selection, 'propose') / 'code'
    zip_path = branch(selection, 'propose') / 'validation' / TEST_DEPENDENCIES.name
    no_credentials = 'env ' + ' '.join('-u ' + name for name in CREDENTIAL_ENV_NAMES)
    remote = (f'cd {shlex.quote(str(code))} && {no_credentials} PYTHONPATH={shlex.quote(f"{zip_path}:{code}")} '
              f'QWEN_CODE_CLI={shlex.quote(str(QWEN_PACKAGE / "cli.js"))} '
              f'QWEN_NODE={shlex.quote(str(NODE_DIRECTORY / "bin/node"))} '
              f'{shlex.quote(str(PROJECT / ".venv/bin/python"))} -m pytest -q -p no:cacheprovider '
              + ' '.join(ROLE_POLICY_TESTS)
              + f' 2>&1 | tee {shlex.quote(str(branch(selection, "propose") / "validation/deployed_tests.log"))}')
    result = ssh('bash -o pipefail -c ' + shlex.quote(remote), timeout=1800)
    report = {'tests_returncode': result.returncode, 'tests_tail': result.stdout[-1500:], 'prepare': {}}
    for mode in MODES:
        try:
            prepared, ok = remote_call(selection, '--remote-prepare', mode)
            report['prepare'][mode] = prepared if ok else {'failed': prepared}
        except RuntimeError as exc:
            report['prepare'][mode] = {'failed': str(exc)}
    report['ok'] = result.returncode == 0 and all('failed' not in value for value in report['prepare'].values())
    return report


def remote_call(selection: Selection, operation, mode=None, *, with_credential=False, timeout=120):
    code = branch(selection, mode or 'propose') / 'code'
    argv = [str(PROJECT / '.venv/bin/python'), str(code / LAUNCHER),
            '--task-id', selection.task_id, '--deployment-id', selection.deployment_id, operation]
    if mode:
        argv.append(mode)
    payload = '{}'
    if with_credential:
        key = os.environ.get(PROVIDER['api_key_env'], '')
        if not key.strip():
            raise ValueError('DEEPSEEK_API_KEY is not set; no credential file is read')
        payload = json.dumps({'api_key': key})
    no_credentials = 'env ' + ' '.join('-u ' + name for name in CREDENTIAL_ENV_NAMES)
    try:
        result = ssh(f'cd {shlex.quote(str(code))} && {no_credentials} {shlex.join(argv)}',
                     stdin=payload, timeout=timeout)
        reply = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        raise RuntimeError(f'{operation} unclear; inspect the branch before any retry') from None
    return reply, result.returncode == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--deployment-id', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--deploy', action='store_true')
    group.add_argument('--remote-tests', action='store_true')
    group.add_argument('--smoke', choices=MODES)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--launch', choices=MODES)
    group.add_argument('--prepare', choices=MODES)
    group.add_argument('--remote-prepare', choices=MODES)
    group.add_argument('--remote-smoke', choices=MODES)
    group.add_argument('--remote-smoke-scripted', choices=MODES)
    group.add_argument('--remote-probe', action='store_true')
    group.add_argument('--remote-launch', choices=MODES)
    smoke_kind = parser.add_mutually_exclusive_group()
    smoke_kind.add_argument('--scripted', action='store_true')
    smoke_kind.add_argument('--real', action='store_true')
    args = parser.parse_args(argv)
    if (args.scripted or args.real) and not args.smoke:
        parser.error('--scripted/--real applies only to --smoke')
    if args.smoke and not (args.scripted or args.real):
        parser.error('--smoke requires --scripted or --real; no paid default')
    try:
        selection = Selection(args.task_id, args.deployment_id)
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    ok = True
    if args.deploy:
        result = deploy(root, selection)
    elif args.remote_tests:
        result = remote_tests(selection)
        ok = result['ok']
    elif args.smoke:
        result, ok = (remote_call(selection, '--remote-smoke-scripted', args.smoke, timeout=1900) if args.scripted
                      else remote_call(selection, '--remote-smoke', args.smoke, with_credential=True, timeout=1900))
    elif args.probe:
        result, ok = remote_call(selection, '--remote-probe', with_credential=True, timeout=1900)
    elif args.launch:
        result, ok = remote_call(selection, '--remote-launch', args.launch, with_credential=True)
    elif args.prepare:
        result, ok = remote_call(selection, '--remote-prepare', args.prepare)
    elif args.remote_prepare:
        prepared = prepare(selection, args.remote_prepare)
        result = {key: prepared[key] for key in ('status', 'mode', 'task_id', 'deployment_id', 'command')}
    elif args.remote_smoke_scripted:
        result = remote_preparation(selection, 'smoke_scripted', args.remote_smoke_scripted, None)
        ok = result['returncode'] == 0
    elif args.remote_smoke:
        result = remote_preparation(selection, 'smoke_real', args.remote_smoke, lambda: json.load(sys.stdin))
        ok = result['returncode'] == 0
    elif args.remote_probe:
        result = remote_preparation(selection, 'cache_probe', 'propose', lambda: json.load(sys.stdin))
        ok = result['returncode'] == 0
    else:
        result = remote_launch(selection, args.remote_launch, lambda: json.load(sys.stdin))
    print(json.dumps(result, indent=2, default=str))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
