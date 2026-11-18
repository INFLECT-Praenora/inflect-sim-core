import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from viva_simulation_core.api.app import create_app  # noqa: E402
from viva_simulation_core.api.config import ApiConfig  # noqa: E402
from viva_simulation_core.api.registry import TwinRegistry  # noqa: E402
from viva_simulation_core.api.surrogate_backend import FakeBackend, SurrogateEntry  # noqa: E402
from viva_simulation_core.api.worker import WorkerTimeout  # noqa: E402

ANT = "S1 -> S2; k*S1; k=0.1; S1=10; S2=0;"
K1, K2 = "key-one", "key-two"
H1 = {"Authorization": f"Bearer {K1}"}
H2 = {"Authorization": f"Bearer {K2}"}


def make_cfg(**kw):
    base = dict(api_keys=f"{K1},{K2}", max_ensemble_n=4, max_twins=4,
                rate_capacity=1000.0, rate_refill_per_s=1000.0,
                request_timeout_s=20.0)
    base.update(kw)
    return ApiConfig(**base)


@pytest.fixture
def make_client():
    made = []

    def _make(cfg=None, backend=None):
        app = create_app(cfg or make_cfg(), backend=backend)
        c = TestClient(app)
        c.__enter__()
        made.append(c)
        return c

    yield _make
    for c in made:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def register(c, headers=H1, **kw):
    r = c.post("/twins", json={"model": ANT, "model_format": "antimony", **kw}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


PRED = {"observables": ["S1"], "horizon_s": 10.0, "interval_s": 1.0}


def test_register_predict_step_flow(client):
    reg = register(client)
    assert {"S1", "S2"} <= set(reg["observables"]) and "k" in reg["parameters"]
    assert "exact" in reg["engines_available"]
    tid = reg["twin_id"]
    assert client.get(f"/twins/{tid}", headers=H1).status_code == 200
    p = client.post(f"/twins/{tid}/predict", json=PRED, headers=H1)
    assert p.status_code == 200, p.text
    assert p.json()["engine_used"] == "exact"
    assert len(p.json()["observables"]["S1"]) == 11
    s1 = client.post(f"/twins/{tid}/step", json={"dt_s": 2.0}, headers=H1).json()
    s2 = client.post(f"/twins/{tid}/step", json={"dt_s": 2.0}, headers=H1).json()
    assert s2["t"] > s1["t"] > 0
    r = client.post(f"/twins/{tid}/reset", headers=H1)
    assert r.status_code == 200 and r.json()["t"] == 0.0
    assert client.delete(f"/twins/{tid}", headers=H1).status_code == 200
    assert client.get(f"/twins/{tid}", headers=H1).status_code == 404


def test_targeted_observables_only(client):
    tid = register(client)["twin_id"]
    p = client.post(f"/twins/{tid}/predict", json=PRED, headers=H1).json()
    assert set(p["observables"]) == {"S1"}
    s = client.post(f"/twins/{tid}/step", json={"observables": ["S2"]}, headers=H1).json()
    assert set(s["observables"]) == {"S2"}


def test_return_alias_roundtrip(client):
    tid = register(client)["twin_id"]
    p = client.post(f"/twins/{tid}/predict", json={**PRED, "return": "endpoint"}, headers=H1)
    assert p.status_code == 200
    assert isinstance(p.json()["observables"]["S1"], float)
    assert "return_" not in p.text


def test_ensemble_stats(client):
    tid = register(client)["twin_id"]
    body = {**PRED, "return": "stats",
            "ensemble": {"n": 3, "draws": {"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}, "seed": 1}}
    r = client.post(f"/twins/{tid}/predict", json=body, headers=H1)
    assert r.status_code == 200, r.text
    st = r.json()["ensemble_stats"]
    assert {"mean", "p05", "p95"} <= set(st)
    assert st["p05"]["S1"] <= st["mean"]["S1"] <= st["p95"]["S1"]


def test_over_cap_ensemble_503(client):
    tid = register(client)["twin_id"]
    body = {**PRED, "ensemble": {"n": 5, "draws": {"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}}}
    assert client.post(f"/twins/{tid}/predict", json=body, headers=H1).status_code == 503
    assert client.post(f"/twins/{tid}/step", json={}, headers=H1).status_code == 200


def test_owner_isolation_403(client):
    tid = register(client)["twin_id"]
    assert client.get(f"/twins/{tid}", headers=H2).status_code == 403
    assert client.post(f"/twins/{tid}/step", json={}, headers=H2).status_code == 403
    assert client.post(f"/twins/{tid}/predict", json=PRED, headers=H2).status_code == 403
    assert client.post(f"/twins/{tid}/reset", headers=H2).status_code == 403
    assert client.delete(f"/twins/{tid}", headers=H2).status_code == 403
    assert client.get(f"/twins/{tid}", headers=H1).status_code == 200


def test_unknown_twin_404(client):
    assert client.get("/twins/nope", headers=H1).status_code == 404
    assert client.post("/twins/nope/step", json={}, headers=H1).status_code == 404
    assert client.delete("/twins/nope", headers=H1).status_code == 404


def test_bad_models_and_observables(make_client):
    c = make_client(make_cfg(max_model_bytes=200))
    r = c.post("/twins", json={"model": 'import "evil.xml"; S1 -> S2; k*S1; k=1; S1=1;',
                               "model_format": "antimony"}, headers=H1)
    assert r.status_code == 422
    r = c.post("/twins", json={"model": "this is not antimony {{{", "model_format": "antimony"}, headers=H1)
    assert r.status_code == 422
    r = c.post("/twins", json={"model": "x" * 500, "model_format": "antimony"}, headers=H1)
    assert r.status_code == 413
    r = c.post("/twins", json={"model": ANT, "model_format": "antimony", "observables": ["Nope"]}, headers=H1)
    assert r.status_code == 422
    tid = register(c)["twin_id"]
    r = c.post(f"/twins/{tid}/predict", json={**PRED, "observables": ["Nope"]}, headers=H1)
    assert r.status_code == 422
    r = c.post(f"/twins/{tid}/predict", json={**PRED, "horizon_s": 1e12}, headers=H1)
    assert r.status_code == 422


def test_forced_missing_surrogate_409(client):
    tid = register(client)["twin_id"]
    r = client.post(f"/twins/{tid}/predict", json={**PRED, "engine": "surrogate"}, headers=H1)
    assert r.status_code == 409


def test_surrogate_tier_used(make_client):
    entry = SurrogateEntry(checkpoint_ref="x", targets=["S1"], horizon_s=100.0,
                           param_ranges={}, state_ranges={})
    fb = FakeBackend(entry, lambda e, req: {"S1": [1.0] * 11})
    c = make_client(backend=fb)
    tid = register(c)["twin_id"]
    p = c.post(f"/twins/{tid}/predict", json=PRED, headers=H1).json()
    assert p["engine_used"] == "surrogate" and p["in_distribution"] is True
    assert p["observables"]["S1"] == [1.0] * 11
    p = c.post(f"/twins/{tid}/predict", json={**PRED, "engine": "exact"}, headers=H1).json()
    assert p["engine_used"] == "exact"


def test_timeout_504_and_server_survives(client):
    tid = register(client)["twin_id"]
    sess = client.app.state.registry.get(tid, next(iter(client.app.state.registry._sessions.values())).owner_key)
    real = sess.worker

    def boom(**kw):
        real._main.kill()
        raise WorkerTimeout("slow")

    real.predict = boom
    assert client.post(f"/twins/{tid}/predict", json=PRED, headers=H1).status_code == 504
    assert client.get("/healthz").status_code == 200
    # crashed worker is transparently rebuilt
    assert client.post(f"/twins/{tid}/step", json={}, headers=H1).status_code == 200


def test_healthz_version_exempt(client):
    assert client.get("/healthz").status_code == 200
    v = client.get("/version")
    assert v.status_code == 200 and "version" in v.json()


def test_auth_required_everywhere(client):
    assert client.post("/twins", json={"model": ANT, "model_format": "antimony"}).status_code == 401
    assert client.get("/twins/x").status_code == 401
    assert client.post("/twins/x/step", json={}).status_code == 401
    assert client.delete("/twins/x").status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert client.get("/twins/x", headers=bad).status_code == 401
    # every documented route except the exempt ones is 401 without a key
    for path, methods in client.app.openapi()["paths"].items():
        if path in ("/healthz", "/version"):
            continue
        for m in methods:
            r = client.request(m.upper(), path.replace("{twin_id}", "x"), json={})
            assert r.status_code == 401, (m, path)


def test_rate_limit_429_isolated_per_app(make_client):
    cfg = make_cfg(rate_capacity=2.0, rate_refill_per_s=0.0)
    c = make_client(cfg)
    assert c.get("/twins/x", headers=H1).status_code in (404, 200)  # GET not weighted
    c.post("/twins/x/step", json={}, headers=H1)
    c.post("/twins/x/step", json={}, headers=H1)
    assert c.post("/twins/x/step", json={}, headers=H1).status_code == 429
    c2 = make_client(cfg)  # fresh app -> fresh buckets
    assert c2.post("/twins/x/step", json={}, headers=H1).status_code == 404


def test_evicted_session_404_after_lock(client):
    tid = register(client)["twin_id"]
    reg: TwinRegistry = client.app.state.registry
    sess = next(iter(reg._sessions.values()))
    sess.evicted = True
    assert client.post(f"/twins/{tid}/step", json={}, headers=H1).status_code == 404
    sess.evicted = False


def test_single_worker_guard():
    from viva_simulation_core.api.__main__ import check_workers, main

    check_workers(1)
    with pytest.raises(ValueError):
        check_workers(2)
    with pytest.raises(SystemExit):
        main(["--workers", "2"])
