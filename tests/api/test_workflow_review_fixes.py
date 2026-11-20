"""Final-review fixes: shared engine lock on all run paths; fan-out wiring."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import viva_simulation_core.api.workflow as wf  # noqa: E402
from viva_simulation_core.api.app import create_app  # noqa: E402
from viva_simulation_core.api.config import ApiConfig  # noqa: E402

KEY = "key-one"
H = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client():
    return TestClient(create_app(ApiConfig(api_keys=KEY, rate_capacity=1000.0,
                                           rate_refill_per_s=1000.0, request_timeout_s=20.0)))


def test_module_call_holds_the_engine_lock(client, monkeypatch):
    """Finding 1: POST /modules/{name}/call must serialize engine access under the
    shared lock (roadrunner/BioGears are not thread-safe)."""
    seen = {}
    orig = wf._run_composite

    def spy(comp, n_steps, interval):
        seen["locked"] = wf._ENGINE_LOCK.locked()
        return orig(comp, n_steps, interval)

    monkeypatch.setattr(wf, "_run_composite", spy)
    resp = client.post("/modules/MathExpressionStep/call",
                       json={"config": {"expressions": [{"out": "y", "expr": "x + 1"}]},
                             "inputs": {"x": 1.0}}, headers=H)
    assert resp.status_code == 200
    assert seen.get("locked") is True  # the lock was held while the composite ran


def test_workflow_run_holds_the_engine_lock(client, monkeypatch):
    """Finding 1: the same shared lock covers the workflow run path too."""
    seen = {}
    orig = wf._run_composite

    def spy(comp, n_steps, interval):
        seen["locked"] = wf._ENGINE_LOCK.locked()
        return orig(comp, n_steps, interval)

    monkeypatch.setattr(wf, "_run_composite", spy)
    spec = {"nodes": [{"id": "n", "module": "MathExpressionStep",
                       "config": {"expressions": [{"out": "y", "expr": "x + 1"}]}}],
            "wires": [], "emit": ["n.y"]}
    wid = client.post("/workflows", json=spec, headers=H).json()["workflow_id"]
    client.post(f"/workflows/{wid}/run", json={"n_steps": 1, "overrides": {"n.x": 1.0}}, headers=H)
    assert seen.get("locked") is True


def test_fan_out_wiring_delivers_to_all_consumers(client):
    """Finding 2: one producer output wired to two consumers must reach BOTH."""
    spec = {
        "nodes": [
            {"id": "src", "module": "MathExpressionStep",
             "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
            {"id": "c1", "module": "MathExpressionStep",
             "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}},
            {"id": "c2", "module": "MathExpressionStep",
             "config": {"expressions": [{"out": "c", "expr": "a * 100"}]}},
        ],
        "wires": [{"from": "src.a", "to": "c1.a"}, {"from": "src.a", "to": "c2.a"}],
        "emit": ["c1.b", "c2.c"],
    }
    wid = client.post("/workflows", json=spec, headers=H).json()["workflow_id"]
    resp = client.post(f"/workflows/{wid}/run",
                       json={"n_steps": 1, "overrides": {"src.x": 5.0}}, headers=H)
    assert resp.status_code == 200, resp.text
    obs = resp.json()["observables"]
    assert obs["c1.b"] == 60.0   # a = 6, both consumers see it
    assert obs["c2.c"] == 600.0
