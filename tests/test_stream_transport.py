"""Local SSE peers only; full-call assembly, accounting and deadline boundaries."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_company.runtime.transport import (
    OpenAIChatTransport, RequestControl, RequestDeadlineExceeded,
    RequestCancelled, InvalidProviderResponse,
)
from test_transport import peer, config


def event(value):
    return ('data: ' + (value if isinstance(value, str) else json.dumps(value)) + '\r\n\r\n').encode()


def chunk(delta=None, finish=None):
    return {'id':'stream', 'model':'model', 'choices':[{'index':0,'delta':delta or {},'finish_reason':finish}]}


def headers(h):
    h.send_response(200)
    h.send_header('Content-Type', 'text/event-stream')
    h.end_headers()


def test_assemble_split_tools_usage_and_redact():
    def serve(h, state):
        headers(h)
        rows = [chunk({'content':'scripted-secret', 'tool_calls':[{'index':0,'id':'call_a',
                    'type':'function','function':{'name':'submit_', 'arguments':'{"ok":'}}]}),
                chunk({'tool_calls':[{'index':0,'function':{'name':'review','arguments':'true}'}}]}, 'tool_calls'),
                {'choices':[], 'usage':{'prompt_tokens':12,'completion_tokens':3,'total_tokens':15}}, '[DONE]']
        for row in rows:
            data = event(row)
            for start in range(0,len(data),7):
                h.wfile.write(data[start:start+7])
            h.wfile.flush()
    with peer(serve) as (url, state):
        status, result = OpenAIChatTransport(config(url))({'stream':True})
    assert status == 200 and result['usage']['total_tokens'] == 15
    message = result['choices'][0]['message']
    assert message['content'] == '[REDACTED]'
    assert message['tool_calls'][0]['function'] == {'name':'submit_review','arguments':'{"ok":true}'}


@pytest.mark.parametrize('rows', [[chunk({'content':'partial'})],
    [chunk({'content':'partial'}),'[DONE]'], ['not json'],
    [{'error':{'message':'bad'}}], [chunk({},'stop')]])
def test_incomplete_stream_is_never_a_success(rows):
    def serve(h,state):
        headers(h)
        for row in rows:
            h.wfile.write(event(row))
    with peer(serve) as (url,state):
        with pytest.raises(InvalidProviderResponse):
            OpenAIChatTransport(config(url))({'stream':True})


@pytest.mark.parametrize('cancel', [False,True])
def test_keepalives_do_not_defeat_deadline_or_cancellation(cancel):
    def serve(h,state):
        headers(h)
        while not state.stop.is_set():
            h.wfile.write(b': keepalive\n\n')
            h.wfile.flush()
            time.sleep(.01)
    with peer(serve) as (url,state):
        control = RequestControl(5 if cancel else .15)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(OpenAIChatTransport(config(url,5)), {'stream':True}, control=control)
            assert state.started.wait(1)
            if cancel:
                control.cancel('user pause')
            with pytest.raises(RequestCancelled if cancel else RequestDeadlineExceeded):
                future.result(timeout=2)
