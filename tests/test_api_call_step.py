"""Task 3: ApiCallStep — a Process whose update is a remote model call.

Demonstrates vivarium orchestrating a *remote* simulation inside a composite.
Tested through the already-covered ``call_module`` path, against a trivial
threaded HTTP stub (no roadrunner, so no engine segfault risk).
"""
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from viva_simulation_core.core import build_core
from viva_simulation_core.api.workflow import call_module, list_modules
import viva_simulation_core.steps.api_call  # noqa: F401 — registers ApiCallStep


@pytest.fixture(scope="module")
def core():
    return build_core()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        out = {k: float(v) + 1.0 for k, v in body.get("overrides", {}).items()}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # keep test output clean
        pass


@contextmanager
def _stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/step"
    finally:
        srv.shutdown()


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_api_call_step_registered(core):
    assert "ApiCallStep" in {c.name for c in list_modules(core)}


def test_api_call_step_round_trips(core):
    with _stub_server() as url:
        out = call_module(core, "ApiCallStep",
                          config={"url": url, "observables": ["x"]},
                          inputs={"x": 5.0}, n_steps=1)
    assert out["x"] == 6.0  # the remote server returned x + 1


def test_api_call_step_dead_url_errors(core):
    with pytest.raises(Exception) as ei:
        call_module(core, "ApiCallStep",
                    config={"url": f"http://127.0.0.1:{_closed_port()}/step",
                            "observables": ["x"], "timeout": 1.0},
                    inputs={"x": 5.0}, n_steps=1)
    assert "failed" in str(ei.value).lower() or "apicallstep" in str(ei.value).lower()
