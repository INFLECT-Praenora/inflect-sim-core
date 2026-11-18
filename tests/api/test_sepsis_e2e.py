"""Slow end-to-end: the whole HTTP API on the real stiff sepsis SBML."""
import math
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from viva_simulation_core.api.app import create_app  # noqa: E402
from viva_simulation_core.api.config import ApiConfig  # noqa: E402

pytestmark = pytest.mark.slow

SBML = (Path(__file__).resolve().parents[2] / "workspace" / "investigations"
        / "biogears-sbml-sepsis" / "sources" / "biogears_sepsis.xml")
KEY = "sepsis-key"
H = {"Authorization": f"Bearer {KEY}"}
OBS = ["MAP", "COmeas"]


@pytest.fixture(scope="module")
def client():
    cfg = ApiConfig(api_keys=KEY, rate_capacity=1000.0, rate_refill_per_s=1000.0,
                    request_timeout_s=300.0, register_compile_timeout_s=120.0)
    with TestClient(create_app(cfg)) as c:
        yield c


@pytest.fixture(scope="module")
def twin(client):
    r = client.post("/twins", json={"model": SBML.read_text(), "model_format": "sbml"}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()


def _finite(vals):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def test_register_names(twin):
    assert {"MAP", "COmeas", "SVR"} <= set(twin["observables"])
    assert {"severity", "infectionStart"} <= set(twin["parameters"])
    assert "exact" in twin["engines_available"]


def test_step_and_predict(client, twin):
    tid = twin["twin_id"]
    t_prev = -1.0
    for dt in (1.0, 5.0, 10.0):
        r = client.post(f"/twins/{tid}/step", json={"dt_s": dt, "observables": OBS}, headers=H)
        assert r.status_code == 200, r.text
        assert r.json()["t"] > t_prev
        t_prev = r.json()["t"]
        assert _finite(r.json()["observables"].values())
    p = client.post(f"/twins/{tid}/predict", headers=H,
                    json={"observables": OBS, "horizon_s": 30.0, "interval_s": 5.0})
    assert p.status_code == 200, p.text
    body = p.json()
    assert body["engine_used"] == "exact"
    for o in OBS:
        assert len(body["observables"][o]) == 7 and _finite(body["observables"][o])


def test_predict_does_not_mutate_live_state(client, twin):
    tid = twin["twin_id"]
    for _ in range(3):
        assert client.post(f"/twins/{tid}/step", json={"dt_s": 5.0}, headers=H).status_code == 200

    def snap():
        r = client.post(f"/twins/{tid}/step", json={"dt_s": 0, "observables": OBS}, headers=H)
        assert r.status_code == 200, r.text
        return r.json()

    before = snap()
    p = client.post(f"/twins/{tid}/predict", headers=H,
                    json={"observables": OBS, "horizon_s": 30.0, "interval_s": 10.0,
                          "parameters": {"severity": 0.9}})
    assert p.status_code == 200, p.text
    after = snap()
    assert after["t"] == before["t"]
    for o in OBS:
        assert after["observables"][o] == pytest.approx(before["observables"][o], rel=1e-9, abs=1e-9)
