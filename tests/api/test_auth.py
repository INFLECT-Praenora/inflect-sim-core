import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from viva_simulation_core.api.auth import (
    TokenBucket, get_config, get_limiter, is_exempt, owner_id, rate_limit, require_key,
)
from viva_simulation_core.api.config import ApiConfig


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_app(cfg, clock):
    app = FastAPI()
    lim = TokenBucket(cfg.rate_capacity, cfg.rate_refill_per_s, clock=clock)
    app.dependency_overrides[get_config] = lambda: cfg
    app.dependency_overrides[get_limiter] = lambda: lim

    @app.get("/who")
    def who(key: str = Depends(require_key)):
        return {"key": key}

    @app.get("/step", dependencies=[Depends(rate_limit("step"))])
    def step():
        return {}

    @app.get("/healthz")
    def h():
        return {"ok": True}

    return app


CFG = ApiConfig(api_keys="k1,k2", rate_capacity=5, rate_refill_per_s=1,
                cost_step=1, cost_register=3, cost_ensemble=4)


def test_valid_key_returns_key():
    c = TestClient(make_app(CFG, Clock()))
    r = c.get("/who", headers={"Authorization": "Bearer k2"})
    assert r.status_code == 200
    oid = r.json()["key"]
    assert oid.startswith("k_") and "k2" != oid and oid == owner_id("k2")
    r2 = c.get("/who", headers={"Authorization": "Bearer k2"})
    assert r2.json()["key"] == oid  # deterministic
    r3 = c.get("/who", headers={"Authorization": "Bearer k1"})
    assert r3.json()["key"] != oid


def test_bucket_keyed_on_derived_id_isolates_keys():
    c = TestClient(make_app(CFG, Clock()))
    h1, h2 = {"Authorization": "Bearer k1"}, {"Authorization": "Bearer k2"}
    for _ in range(5):
        assert c.get("/step", headers=h1).status_code == 200
    assert c.get("/step", headers=h1).status_code == 429
    assert c.get("/step", headers=h2).status_code == 200


@pytest.mark.parametrize("hdr", [None, "Bearer nope", "Basic k1", "Bearer "])
def test_missing_or_invalid_401(hdr):
    c = TestClient(make_app(CFG, Clock()))
    r = c.get("/who", headers={"Authorization": hdr} if hdr else {})
    assert r.status_code == 401


def test_no_configured_keys_rejects_all():
    c = TestClient(make_app(ApiConfig(), Clock()))
    assert c.get("/who", headers={"Authorization": "Bearer "}).status_code == 401
    assert c.get("/who", headers={"Authorization": "Bearer x"}).status_code == 401


def test_non_ascii_key_rejected_not_error():
    c = TestClient(make_app(CFG, Clock()))
    r = c.get("/who", headers={"Authorization": "Bearer ké".encode("utf-8")})
    assert r.status_code == 401


def test_constant_time_compare_used(monkeypatch):
    import viva_simulation_core.api.auth as a
    calls = []
    real = a.secrets.compare_digest
    monkeypatch.setattr(a.secrets, "compare_digest",
                        lambda x, y: calls.append(1) or real(x, y))
    c = TestClient(make_app(CFG, Clock()))
    c.get("/who", headers={"Authorization": "Bearer k1"})
    assert len(calls) >= 2  # compares against every key


def test_bucket_trips_and_refills():
    clk = Clock()
    b = TokenBucket(5, 1, clock=clk)
    assert b.consume("a", 3) and b.consume("a", 2)
    assert not b.consume("a", 1)
    assert b.retry_after("a", 1) > 0
    assert b.consume("b", 5)  # per-key isolation
    clk.t += 2
    assert b.consume("a", 2)
    assert not b.consume("a", 1)


def test_costs_register_ensemble_exceed_step():
    c = ApiConfig()
    assert c.costs["register"] > c.costs["step"]
    assert c.costs["ensemble"] > c.costs["step"]


def test_rate_limit_429_with_retry_after():
    clk = Clock()
    c = TestClient(make_app(CFG, clk))
    h = {"Authorization": "Bearer k1"}
    for _ in range(5):
        assert c.get("/step", headers=h).status_code == 200
    r = c.get("/step", headers=h)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    clk.t += 1
    assert c.get("/step", headers=h).status_code == 200


def test_rate_limit_requires_auth():
    c = TestClient(make_app(CFG, Clock()))
    assert c.get("/step").status_code == 401


def test_exempt():
    assert is_exempt("/healthz") and is_exempt("/version")
    assert not is_exempt("/twins/abc/step")
    assert TestClient(make_app(CFG, Clock())).get("/healthz").status_code == 200
