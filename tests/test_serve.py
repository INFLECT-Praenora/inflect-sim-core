"""Task 2: the lean serve path — ServedModel + one-endpoint model_server.

Replaces the deleted twin service: an in-process stepping handle over a model
loaded from SBML/Antimony, plus a single-route ASGI app for the api-call demo.

roadrunner engines are heavy and multiple live instances in one process can
segfault, so the fast tests share ONE module-scoped engine and ``reset()`` it.
"""
from pathlib import Path

import pytest

from viva_simulation_core.serving import ServedModel, model_server

# A tiny model with an assignment rule (``total``) — exercises rule-driven
# observables without loading the 91-state sepsis model.
SMALL = "S1 -> S2; k*S1; k = 0.3; S1 = 10; S2 = 0; total := S1 + S2"

SEPSIS_SBML = (Path(__file__).resolve().parents[1] / "workspace" / "investigations"
               / "biogears-sbml-sepsis" / "sources" / "biogears_sepsis.xml")


@pytest.fixture(scope="module")
def served():
    return ServedModel(SMALL, "antimony")


def test_served_model_observables_includes_rule_target(served):
    served.reset()
    obs = served.observables()
    assert "S1" in obs and "S2" in obs
    assert "total" in obs  # assignment-rule target surfaced, not crashed on


def test_served_model_step_advances_state(served):
    served.reset()
    s0 = served.step()
    assert isinstance(s0, dict) and isinstance(s0["S1"], float)
    assert s0["S1"] < 10.0  # S1 decays
    assert abs(s0["total"] - (s0["S1"] + s0["S2"])) < 1e-6  # rule holds


def test_served_model_predict_returns_n_points(served):
    served.reset()
    traj = served.predict(3)
    assert set(("S1", "S2")).issubset(traj)
    assert len(traj["S1"]) == 3
    assert traj["S1"][0] > traj["S1"][-1]  # monotone decay over the window


def test_served_model_reset_returns_to_initial_trajectory(served):
    served.reset()
    first = served.step()["S1"]   # fresh model's first-step value
    served.predict(10)            # advance far
    served.reset()
    assert abs(served.step()["S1"] - first) < 1e-9  # reset -> same fresh trajectory


def test_served_model_overrides_seed_initial(served):
    served.reset()
    s = served.step(overrides={"S1": 20.0})
    assert s["S1"] < 20.0 and s["S1"] > 10.0  # decayed from the seeded 20, not 10


def test_model_server_step_route(served):
    served.reset()
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    client = TestClient(model_server(served))
    r = client.post("/step", json={"overrides": {}})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict) and "S1" in body


@pytest.mark.slow
def test_served_model_loads_real_sepsis_without_crashing():
    m = ServedModel(SEPSIS_SBML.read_text(encoding="utf-8"), "sbml")
    assert m.observables()          # real 91-state model, rules included
    assert isinstance(m.step(), dict)
