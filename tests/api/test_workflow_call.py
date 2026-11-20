"""Task 4: POST /modules/{name}/call — call a single module as a 1-node composite."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from viva_simulation_core.api.app import create_app  # noqa: E402
from viva_simulation_core.api.config import ApiConfig  # noqa: E402

KEY = "key-one"
H = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client():
    return TestClient(create_app(ApiConfig(api_keys=KEY, rate_capacity=1000.0,
                                            rate_refill_per_s=1000.0, request_timeout_s=20.0)))


def test_call_math_expression_module(client):
    body = {
        "config": {"expressions": [{"out": "y", "expr": "2*x + 3"}]},
        "inputs": {"x": 5.0},
        "n_steps": 1,
    }
    resp = client.post("/modules/MathExpressionStep/call", json=body, headers=H)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["outputs"]["y"] == 13.0
    assert data["provenance"]["modules"] == ["MathExpressionStep"]


def test_call_unknown_module_422(client):
    resp = client.post("/modules/NoSuchModule/call", json={"config": {}, "inputs": {}},
                       headers=H)
    assert resp.status_code == 422


def test_call_requires_auth(client):
    assert client.post("/modules/MathExpressionStep/call", json={"config": {}}).status_code == 401
