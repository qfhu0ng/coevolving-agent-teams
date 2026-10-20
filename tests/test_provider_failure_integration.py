"""Actual Qwen -> actual relay/transport -> loopback-only provider failures."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import threading
import time

import pytest

from agent_company.runtime.qwen import QwenTeamRunner
from agent_company.runtime.relay import ProviderConfig
from test_team import workspace


@pytest.mark.parametrize("mode", ["single", "team"])
@pytest.mark.parametrize("fault", ["http200_error", "keepalive_deadline"])
def test_native_pipeline_fails_once_without_spawning_or_retrying(tmp_path, mode, fault):
    cli = Path(os.environ.get("QWEN_CODE_CLI", str(
        Path(__file__).parents[1] / "external/qwen-study/package/cli.js")))
    node = os.environ.get("QWEN_NODE") or shutil.which("node")
    if not cli.is_file() or not node:
        pytest.skip("Native Qwen CLI and Node required; no real provider is used")
    calls, peer_closed = [], threading.Event()
    cleanup = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                if fault == "http200_error":
                    raw = {"error": {"message": "Unable to start processing within the 900-second timeout"}}
                    self.wfile.write(json.dumps(raw).encode())
                    self.wfile.flush()
                else:
                    while not cleanup.wait(0.02):
                        self.wfile.write(b"\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                peer_closed.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "run"
    try:
        started = time.monotonic()
        result = QwenTeamRunner(str(cli.resolve()), node=node).run(
            workspace(tmp_path), "Inspect this repository and report what is present.",
            "Delegate inspection." if mode == "team" else "",
            "Report findings." if mode == "team" else "", output,
            ProviderConfig("loopback-only", f"http://127.0.0.1:{server.server_port}",
                           request_timeout_seconds=0.4),
            timeout_seconds=15, execution_mode=mode)
        elapsed = time.monotonic() - started
        assert len(calls) == 1, result
        assert result["stop_reason"] == "cli_error" and result["native_result"]["explicit_error"]
        assert not result["execution"]["timed_out"] and elapsed < 15
        assert result["workers"] == {} and result["inflight_attempts"] == 0
        assert result["provider_attempts"] == result["completed_attempts"] == 1
        assert not result["usage_complete"] and not result["cost_complete"]
        error = result["errors"][0]
        assert error["relay_http_status"] == 424
        if fault == "http200_error":
            assert error["provider_http_status"] == 200 and error["error_kind"] == "provider_error"
            raw = json.loads(next((output / "relay").glob("*_provider_response.json")).read_text())
            assert "900-second" in raw["error"]["message"]
        else:
            assert error["error_kind"] == "provider_timeout" and peer_closed.wait(1)
    finally:
        cleanup.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
