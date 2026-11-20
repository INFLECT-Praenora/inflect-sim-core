"""Task 5: workflow define / run / graph / delete.

Fast tests use a two-node MathExpressionStep chain (no roadrunner): inc computes
a = x + 1, scale computes b = a * 10, wired inc.a -> scale.a. With x = 5 that is
a = 6, b = 60.
"""
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from viva_simulation_core.api.app import create_app  # noqa: E402
from viva_simulation_core.api.config import ApiConfig  # noqa: E402

KEY = "key-one"
H = {"Authorization": f"Bearer {KEY}"}

SPEC = {
    "nodes": [
        {"id": "inc", "module": "MathExpressionStep",
         "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
        {"id": "scale", "module": "MathExpressionStep",
         "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}},
    ],
    "wires": [{"from": "inc.a", "to": "scale.a"}],
    "emit": ["scale.b", "inc.a"],
}


@pytest.fixture
def client():
    return TestClient(create_app(ApiConfig(api_keys=KEY, rate_capacity=1000.0,
                                           rate_refill_per_s=1000.0, request_timeout_s=20.0)))


def _define(client):
    resp = client.post("/workflows", json=SPEC, headers=H)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_define_returns_id_and_graph(client):
    data = _define(client)
    assert "workflow_id" in data
    graph = data["graph"]
    assert {n["id"] for n in graph["nodes"]} == {"inc", "scale"}
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["from"] == "inc.a"
    assert graph["edges"][0]["to"] == "scale.a"


def test_run_produces_observables(client):
    wid = _define(client)["workflow_id"]
    resp = client.post(f"/workflows/{wid}/run",
                       json={"n_steps": 1, "overrides": {"inc.x": 5.0}}, headers=H)
    assert resp.status_code == 200, resp.text
    obs = resp.json()["observables"]
    assert obs["inc.a"] == 6.0
    assert obs["scale.b"] == 60.0


def test_graph_endpoint(client):
    wid = _define(client)["workflow_id"]
    resp = client.get(f"/workflows/{wid}/graph", headers=H)
    assert resp.status_code == 200
    g = resp.json()
    assert len(g["nodes"]) == 2 and len(g["edges"]) == 1


def test_define_unknown_module_422(client):
    bad = {"nodes": [{"id": "n", "module": "NoSuchModule"}], "wires": [], "emit": []}
    assert client.post("/workflows", json=bad, headers=H).status_code == 422


def test_run_unknown_id_404(client):
    assert client.get("/workflows/does-not-exist/graph", headers=H).status_code == 404
    assert client.post("/workflows/does-not-exist/run", json={"n_steps": 1},
                       headers=H).status_code == 404


def test_delete_evicts(client):
    wid = _define(client)["workflow_id"]
    assert client.delete(f"/workflows/{wid}", headers=H).status_code == 200
    assert client.post(f"/workflows/{wid}/run", json={"n_steps": 1},
                       headers=H).status_code == 404


def test_concurrent_runs_are_correct(client):
    wid = _define(client)["workflow_id"]
    results = {}

    def run(i):
        r = client.post(f"/workflows/{wid}/run",
                        json={"n_steps": 1, "overrides": {"inc.x": float(i)}}, headers=H)
        results[i] = r.json()["observables"]["scale.b"]

    threads = [threading.Thread(target=run, args=(i,)) for i in (1, 2, 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {1: 20.0, 2: 30.0, 3: 40.0}  # b = (i + 1) * 10


def test_workflows_require_auth(client):
    assert client.post("/workflows", json=SPEC).status_code == 401
