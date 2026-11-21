"""Task 6: smoke tests so the examples don't bit-rot.

Fast tests cover the engine-free tour demos + the surrogate scorecards; the
roadrunner-backed demos (serve) run under -m slow.
"""
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from examples import workflow_tour, surrogate_demo


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

    def log_message(self, *a):
        pass


@contextmanager
def _stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/step"
    finally:
        srv.shutdown()


def test_tour_list():
    assert "MathExpressionStep" in workflow_tour.demo_list()


def test_tour_call():
    assert workflow_tour.demo_call()["y"] == 13.0


def test_tour_connect_run():
    assert workflow_tour.demo_connect_run()["scale.b"] == 60.0


def test_tour_sweep():
    assert workflow_tour.demo_sweep()[3.0] == 6.0


def test_tour_api_call():
    with _stub_server() as url:
        assert workflow_tour.demo_api_call(url)["x"] == 6.0


def test_surrogate_demo_scorecards():
    sc = surrogate_demo.scorecards()
    assert set(sc) == {"persistence", "linear"}


@pytest.mark.slow
def test_tour_serve():
    r = workflow_tour.demo_serve()
    assert "S1" in r["observables"]


@pytest.mark.slow
def test_serve_demo_step_in_process():
    from examples import serve_demo
    serve_demo.step_in_process()  # runs without raising
