"""Narrated walkthrough of the INFLECT digital-twin API.

Runs entirely in-process via ``fastapi.testclient.TestClient`` -- no server, no
hosting.  Every request and response below is real.

    uv run python examples/digital_twin_demo.py

Full reference (routes, config, error contract): docs/api.md
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from viva_simulation_core.api.app import create_app
from viva_simulation_core.api.config import ApiConfig
from viva_simulation_core.api.surrogate_backend import FakeBackend, SurrogateEntry

# A small, fast Antimony kinetic model (S1 -> S2 -> S3).  An SBML model registers
# identically with ``"model_format": "sbml"``; see the sepsis end-to-end test for
# the stiff BioGears-derived SBML twin.
MODEL = """
S1 -> S2; k*S1
S2 -> S3; k2*S2
k = 0.1; k2 = 0.05
S1 = 10; S2 = 0; S3 = 0
"""
KEY, OTHER_KEY = "demo-key-1", "demo-key-2"
AUTH = {"Authorization": f"Bearer {KEY}"}
AUTH2 = {"Authorization": f"Bearer {OTHER_KEY}"}
PRED = {"observables": ["S1", "S2"], "horizon_s": 10.0, "interval_s": 2.0}

SECTIONS: list[str] = []


def cfg(**kw) -> ApiConfig:
    base = dict(api_keys=f"{KEY},{OTHER_KEY}", max_ensemble_n=8, max_twins=4,
                max_model_bytes=2000, request_timeout_s=20.0,
                rate_capacity=1000.0, rate_refill_per_s=1000.0)
    base.update(kw)
    return ApiConfig(**base)


def section(title: str) -> None:
    SECTIONS.append(title)
    print(f"\n{'=' * 72}\n{len(SECTIONS)}. {title}\n{'=' * 72}")


def show(resp, method: str, path: str, body=None, expect: int = 200, note: str = ""):
    print(f"\n> {method} {path}" + (f"   # {note}" if note else ""))
    if body is not None:
        print("  request :", json.dumps(body))
    text = json.dumps(resp.json())
    print(f"  response: HTTP {resp.status_code}  {text if len(text) < 600 else text[:600] + ' ...'}")
    assert resp.status_code == expect, f"expected {expect}, got {resp.status_code}: {resp.text}"
    return resp.json()


def call(c, method, path, body=None, headers=AUTH, expect=200, note=""):
    kw = {"headers": headers}
    if body is not None:
        kw["json"] = body
    return show(c.request(method, path, **kw), method, path, body, expect, note)


def register(c, headers=AUTH, **extra):
    body = {"model": MODEL, "model_format": "antimony", **extra}
    return call(c, "POST", "/twins", body, headers)


def main() -> None:
    # ------------------------------------------------------------------ 1
    section("Setup: in-process app with a known API key and small caps")
    config = cfg()
    print("ApiConfig: api_keys=<2 keys>, max_ensemble_n=8, max_twins=4, max_model_bytes=2000")
    print("(Env equivalents: INFLECT_API_API_KEYS, INFLECT_API_MAX_ENSEMBLE_N, ...)")
    with TestClient(create_app(config)) as c:
        call(c, "GET", "/healthz", headers={}, note="unauthenticated liveness")

        # -------------------------------------------------------------- 2
        section("Register a twin (Antimony)")
        reg = register(c)
        tid = reg["twin_id"]
        assert {"S1", "S2", "S3"} <= set(reg["observables"]) and "k" in reg["parameters"]
        assert reg["engines_available"] == ["exact"]
        again = register(c)
        assert again["twin_id"] == tid
        print("  (registering the same model again with the same key dedups to the same twin_id)")

        # -------------------------------------------------------------- 3
        section("GET twin metadata")
        call(c, "GET", f"/twins/{tid}")

        # -------------------------------------------------------------- 4
        section("Step the live twin: t advances; parameter and species overrides persist")
        s1 = call(c, "POST", f"/twins/{tid}/step", {"dt_s": 2.0, "observables": ["S1", "S2"]})
        s2 = call(c, "POST", f"/twins/{tid}/step", {"dt_s": 2.0, "observables": ["S1", "S2"]})
        assert s2["t"] > s1["t"] > 0
        s3 = call(c, "POST", f"/twins/{tid}/step",
                  {"dt_s": 2.0, "parameters": {"k": 0.5}, "observables": ["S1", "S2"]},
                  note="override k=0.5 (a 'treatment'); decay speeds up")
        s4 = call(c, "POST", f"/twins/{tid}/step", {"dt_s": 2.0, "observables": ["S1", "S2"]},
                  note="no override sent: k=0.5 persisted")
        s5 = call(c, "POST", f"/twins/{tid}/step",
                  {"dt_s": 0.0, "species": {"S1": 50.0}, "observables": ["S1"]},
                  note="species override: set S1=50 (a 'bolus')")
        assert abs(s5["observables"]["S1"] - 50.0) < 1e-6

        # -------------------------------------------------------------- 5
        section("Predict (non-destructive what-if) in all three return modes")
        p = call(c, "POST", f"/twins/{tid}/predict", PRED, note='"return" omitted = trajectory')
        assert p["engine_used"] == "exact" and len(p["observables"]["S1"]) == 6
        pe = call(c, "POST", f"/twins/{tid}/predict", {**PRED, "return": "endpoint"},
                  note='the JSON key is "return" (alias of return_)')
        assert isinstance(pe["observables"]["S1"], float)
        ps = call(c, "POST", f"/twins/{tid}/predict", {**PRED, "return": "stats"})
        assert ps["engine_used"] == "exact"
        pp = call(c, "POST", f"/twins/{tid}/predict",
                  {**PRED, "return": "endpoint", "parameters": {"k": 1.0}, "initial_state": {"S1": 20.0}},
                  note="per-call parameters/initial_state overrides (twin state untouched)")
        assert pp["observables"]["S1"] < 20.0

        # -------------------------------------------------------------- 6
        section("Ensembles via predict: members, draws (uniform + normal), stochastic")
        em = call(c, "POST", f"/twins/{tid}/predict",
                  {**PRED, "return": "endpoint",
                   "ensemble": {"members": [{"k": 0.05}, {"k": 0.1}, {"k": 0.3}]}},
                  note="explicit parameter sets")
        assert em["ensemble_stats"]["n"] == 3
        ed = call(c, "POST", f"/twins/{tid}/predict",
                  {**PRED, "return": "endpoint",
                   "ensemble": {"n": 6, "seed": 7,
                                "draws": {"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2},
                                          "k2": {"dist": "normal", "mean": 0.05, "sd": 0.01}}}},
                  note="seeded draws from uniform + normal priors")
        st = ed["ensemble_stats"]
        assert st["p05"]["S1"] <= st["mean"]["S1"] <= st["p95"]["S1"]
        print(f"  -> S1 endpoint  mean={st['mean']['S1']:.3f}  p05={st['p05']['S1']:.3f}  p95={st['p95']['S1']:.3f}")
        ed2 = call(c, "POST", f"/twins/{tid}/predict",
                   {**PRED, "return": "endpoint",
                    "ensemble": {"n": 6, "seed": 7,
                                 "draws": {"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2},
                                           "k2": {"dist": "normal", "mean": 0.05, "sd": 0.01}}}})
        assert ed2["ensemble_stats"]["mean"] == st["mean"], "same seed must reproduce"
        print("  -> same seed reproduces the same stats")
        es = call(c, "POST", f"/twins/{tid}/predict",
                  {**PRED, "return": "endpoint", "ensemble": {"n": 5, "stochastic": True, "seed": 3}},
                  note="stochastic (Gillespie) replicates; spread comes from intrinsic noise")
        print(f"  -> stochastic S1 endpoint mean={es['ensemble_stats']['mean']['S1']:.3f} "
              f"p05={es['ensemble_stats']['p05']['S1']:.3f} p95={es['ensemble_stats']['p95']['S1']:.3f}")

        # -------------------------------------------------------------- 9 (errors, same app)
        section("Error cases (uniform body: {\"error\": ..., \"detail\": ...} or FastAPI detail)")
        call(c, "POST", f"/twins/{tid}/predict", {**PRED, "observables": ["Nope"]}, expect=422,
             note="unknown observable")
        call(c, "POST", "/twins",
             {"model": 'import "evil.xml"; S1 -> S2; k*S1; k=1; S1=1;', "model_format": "antimony"},
             expect=422, note="unsafe model (Antimony import)")
        call(c, "POST", "/twins", {"model": "x" * 3000, "model_format": "antimony"}, expect=413,
             note="model larger than max_model_bytes")
        call(c, "GET", "/twins/does-not-exist", expect=404, note="unknown twin id")
        call(c, "POST", f"/twins/{tid}/predict",
             {**PRED, "ensemble": {"n": 9, "draws": {"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}}},
             expect=503, note="ensemble N=9 > max_ensemble_n=8")
        call(c, "POST", f"/twins/{tid}/step", {"dt_s": -1.0}, expect=422, note="negative dt_s")
        r = c.post(f"/twins/{tid}/step", content='{"dt_s": NaN}',
                   headers={**AUTH, "content-type": "application/json"})
        show(r, "POST", f"/twins/{tid}/step", {"dt_s": "NaN"}, expect=422, note="non-finite dt_s")
        call(c, "POST", f"/twins/{tid}/predict", {**PRED, "horizon_s": 1e12}, expect=422,
             note="horizon over API limit")

        # -------------------------------------------------------------- 8
        section("Auth and ownership")
        r = c.get(f"/twins/{tid}")
        show(r, "GET", f"/twins/{tid}", expect=401, note="no Authorization header")
        call(c, "GET", f"/twins/{tid}", headers={"Authorization": "Bearer wrong"}, expect=401,
             note="unknown key")
        call(c, "GET", f"/twins/{tid}", headers=AUTH2, expect=403,
             note="a different valid key cannot touch this twin")
        call(c, "POST", f"/twins/{tid}/step", {}, headers=AUTH2, expect=403)

        # -------------------------------------------------------------- 10
        section("Reset and delete")
        rs = call(c, "POST", f"/twins/{tid}/reset", note="back to t=0")
        assert rs["t"] == 0.0
        call(c, "DELETE", f"/twins/{tid}")
        call(c, "GET", f"/twins/{tid}", expect=404, note="gone")

    # ---------------------------------------------------------------- 7
    section("Tiered engine: surrogate vs exact (auto selection)")
    # FakeBackend stands in for an operator-curated checkpoint.  It matches ANY
    # model hash, so it applies to the demo twin.  The envelope: targets, horizon
    # and parameter ranges the surrogate was validated on.
    entry = SurrogateEntry(checkpoint_ref="demo-checkpoint", targets=["S1", "S2"], horizon_s=100.0,
                           param_ranges={"k": (0.05, 0.2)}, state_ranges={})
    canned = lambda e, req: {"observables": {o: [float(i) for i in range(6)] for o in req.observables},
                             "times": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]}
    with TestClient(create_app(config, backend=FakeBackend(entry, canned))) as c:
        reg = register(c)
        assert reg["engines_available"] == ["exact", "surrogate"]
        tid = reg["twin_id"]
        p = call(c, "POST", f"/twins/{tid}/predict", PRED, note="in-envelope + engine=auto")
        assert p["engine_used"] == "surrogate" and p["in_distribution"] is True
        p = call(c, "POST", f"/twins/{tid}/predict", {**PRED, "parameters": {"k": 5.0}},
                 note="k=5.0 is outside the validated range -> auto falls back")
        assert p["engine_used"] == "exact"
        p = call(c, "POST", f"/twins/{tid}/predict", {**PRED, "engine": "exact"},
                 note="caller can always force exact")
        assert p["engine_used"] == "exact"

    with TestClient(create_app(config)) as c:  # default NullBackend: no checkpoints
        tid = register(c)["twin_id"]
        call(c, "POST", f"/twins/{tid}/predict", {**PRED, "engine": "surrogate"}, expect=409,
             note="forced surrogate, but this model has no checkpoint")

    # ---------------------------------------------------------------- rate limit
    section("Rate limiting (token bucket per key)")
    with TestClient(create_app(cfg(rate_capacity=2.0, rate_refill_per_s=0.0))) as c:
        for _ in range(2):
            call(c, "POST", "/twins/x/step", {}, expect=404)
        call(c, "POST", "/twins/x/step", {}, expect=429, note="bucket empty")

    # ---------------------------------------------------------------- 11
    section("Persistence, restart, and workers (notes)")
    print("""
  * Set INFLECT_API_DB_PATH to persist twins in SQLite.  The MODEL (and owner)
    persists across a restart; live simulation state is ephemeral.  After a
    restart a twin is transparently re-warmed (recompiled) at t=0 on first use.
  * The twin registry is in-process: run exactly ONE uvicorn worker
    (`python -m viva_simulation_core.api` enforces this).
  * Not demoed here: 504 on a hung worker (see tests/api/test_app.py), and the
    real torch surrogate adapter (FakeBackend is used above).
  * Full reference and curl examples: docs/api.md
""")

    print("=" * 72)
    print(f"PASS: {len(SECTIONS)} sections ran, all self-checks held")
    for i, s in enumerate(SECTIONS, 1):
        print(f"  {i:2d}. {s}")


if __name__ == "__main__":
    main()
