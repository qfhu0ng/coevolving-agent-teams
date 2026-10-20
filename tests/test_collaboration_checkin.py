"""Execute the actual opt-in formatter over the real legacy check-in renderer."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from agent_company.runtime.collaboration_text import CHECKIN_HEADER, CHECKIN_FOOTER, CHECKIN_PROPOSE, CHECKIN_OFF


@pytest.mark.parametrize('mode,enabled', [('off',True), ('propose',True), ('off',False)])
def test_actual_package_checkin_formatter(mode, enabled):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the native formatter')
    runtime = Path(__file__).resolve().parents[1] / 'agent_company/runtime'
    source = (runtime/'qwen_native_compat.mjs').read_text()
    function = re.search(r'  function packageCheckin\(view\) \{.*?\n  \}', source, re.S).group(0)
    view = {'id':'checkin-1','reason':'periodic','elapsed_s':100,'remaining_s':200,
            'slots':{'used':1,'max':10},'workers':[],'delivery':{},'unknown':[]}
    script = (f'import {{formatCheckin}} from {json.dumps((runtime/"qwen_team_checkin.mjs").as_uri())};\n'
              f'const rolePackage={json.dumps(enabled)},rolePolicyMode={json.dumps(mode)},checkinSeconds=1800;\n'
              + function + f'\nconst view={json.dumps(view)};\nconsole.log(JSON.stringify({{current:packageCheckin(view),old:formatCheckin(view,1800)}}));')
    result = subprocess.run([node,'--input-type=module','-e',script], check=True, text=True, capture_output=True)
    value = json.loads(result.stdout)
    if not enabled:
        assert value['current'] == value['old']
    else:
        assert value['current'].startswith(CHECKIN_HEADER.format(checkin_id='checkin-1',reason='periodic'))
        assert CHECKIN_FOOTER in value['current']
        assert value['current'].endswith(CHECKIN_PROPOSE if mode == 'propose' else CHECKIN_OFF)
        assert 'No reply is needed' not in value['current']


def test_package_checkin_removes_only_the_template_footer_suffix():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the native formatter')
    runtime = Path(__file__).resolve().parents[1] / 'agent_company/runtime'
    source = (runtime / 'qwen_native_compat.mjs').read_text()
    function = re.search(r'  function packageCheckin\(view\) \{.*?\n  \}', source, re.S).group(0)
    subject = 'Keep observation. No reply is needed if nothing changes.'
    view = {'id': 'checkin-1', 'reason': 'periodic', 'elapsed_s': 1, 'remaining_s': 1000,
            'slots': {'used': 0, 'max': 10}, 'workers': [], 'unknown': [],
            'delivery': {'commits_since_launch': 1, 'last_commit_ago_s': 0, 'last_commit_subject': subject}}
    script = (f'import {{formatCheckin}} from {json.dumps((runtime / "qwen_team_checkin.mjs").as_uri())};\n'
              'const rolePackage=true,rolePolicyMode="propose",checkinSeconds=1800;\n'
              + function + f'\nconsole.log(JSON.stringify(packageCheckin({json.dumps(view)})));')
    result = subprocess.run([node, '--input-type=module', '-e', script], check=True, text=True, capture_output=True)
    value = json.loads(result.stdout)
    assert subject in value
    assert value.count('No reply is needed if nothing changes.') == 1
    assert CHECKIN_FOOTER in value and value.endswith(CHECKIN_PROPOSE)
