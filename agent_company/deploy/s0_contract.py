"""Fixed S0 settings shared by the S0 launcher script and the deployment launch checks.

The package must not import repository scripts (tests/test_architecture_boundaries.py),
so the values both sides check live here; the launcher script re-exports them.
"""
from __future__ import annotations

from pathlib import Path

MODES = ('propose', 'global')
POLICY = Path('configs/stage1/policies/fresh_p0_20261001')
POLICY_SHA256 = {
    'assignment_v000.md': 'cce38a9a63bb6b708e4fc4f33fad1e5adf97e734950a85d3a363ade03620e27d',
    'communication_v000.md': '76c8227aa1b8d0e48c62065e1a480bc8dd01bab3e42fd73d4da45a2454e86c27',
}


def mode_name(mode):
    if mode not in MODES:
        raise ValueError('Unknown S0 mode')
    return mode


def inference(mode):
    return {'agent_timeout_sec': 3600, 'cpus': 20, 'memory_mb': 61440, 'memory_swap_mb': 61440,
            'max_model_calls': None, 'team_checkin_seconds': 1800, 'team_message_delivery': 'between_tool_calls',
            'runtime_disclosure': True, 'leader_loop_recovery': True, 'tool_call_cap': 'disabled',
            'role_policy_mode': mode_name(mode)}
