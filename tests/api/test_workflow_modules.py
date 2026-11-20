"""Task 3: /modules introspection endpoints on the unified app."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from viva_simulation_core.api.app import create_app  # noqa: E402
from viva_simulation_core.api.config import ApiConfig  # noqa: E402

KEY = "key-one"
H = {"Authorization": f"Bearer {KEY}"}


def make_cfg(**kw):
    base = dict(api_keys=KEY, max_twins=4, rate_capacity=1000.0,
                rate_refill_per_s=1000.0, request_timeout_s=20.0)
    base.update(kw)
    return ApiConfig(**base)


@pytest.fixture
def client():
    return TestClient(create_app(make_cfg()))


def test_list_modules_includes_real_viva_modules(client):
    resp = client.get("/modules", headers=H)
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()}
    for expected in ("TelluriumUTCStep", "MathExpressionStep", "BioGearsProcess", "NeuralProcess"):
        assert expected in names, f"{expected} missing from {sorted(names)[:20]}"


def test_list_modules_dedups_short_name(client):
    names = [m["name"] for m in client.get("/modules", headers=H).json()]
    # the short alias appears once, not alongside its dotted fully-qualified twin
    assert names.count("TelluriumUTCStep") == 1
    assert not any(n.endswith(".TelluriumUTCStep") for n in names)


def test_list_modules_never_500_across_all_modules(client):
    # every registered module must introspect without taking the endpoint down
    resp = client.get("/modules", headers=H)
    assert resp.status_code == 200
    assert len(resp.json()) >= 4


def test_get_one_module_card(client):
    resp = client.get("/modules/MathExpressionStep", headers=H)
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "MathExpressionStep"
    assert isinstance(card["config_schema"], dict)
    assert "expressions" in card["config_schema"]


def test_get_unknown_module_404(client):
    assert client.get("/modules/NoSuchModule", headers=H).status_code == 404


def test_modules_requires_auth(client):
    assert client.get("/modules").status_code == 401
