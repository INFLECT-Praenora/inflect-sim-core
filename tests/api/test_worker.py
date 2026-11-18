import pytest

from viva_simulation_core.api.worker import TwinWorker, WorkerError, WorkerTimeout

ANT = "S1 -> S2; k*S1; k=0.1; S1=10; S2=0;"


@pytest.fixture
def worker():
    w = TwinWorker(ANT, "antimony", None, interval_s=1.0)
    w.start()
    try:
        yield w
    finally:
        w.close()


def test_step_advances_and_persists(worker):
    a = worker.step(dt_s=1)
    b = worker.step(dt_s=1)
    assert 0.99 < a["t"] < 1.01 and b["t"] > a["t"]
    assert b["observables"]["S1"] < a["observables"]["S1"]
    assert worker.step()["t"] == pytest.approx(3.0)  # default dt = interval


def test_step_param_persists(worker):
    worker.step(dt_s=1, parameters={"k": 0.5})
    s1 = worker.step(dt_s=1)["observables"]["S1"]  # k not re-sent
    s0 = 10 * 2.718281828 ** -0.5
    assert s1 == pytest.approx(s0 * 2.718281828 ** -0.5, rel=1e-2)


def test_species_overwrite(worker):
    r = worker.step(dt_s=1, species={"S1": 100.0})
    assert r["observables"]["S1"] > 50


def test_predict_non_mutating(worker):
    worker.step(dt_s=5)
    s = worker.snapshot_probe()
    p = worker.predict(horizon_s=100, observables=["S2"], parameters={"k": 1.0})
    assert p["observables"]["S2"][-1] > 9.9
    assert worker.snapshot_probe() == s
    p2 = worker.predict(horizon_s=10, interval_s=5, observables=["S1"], initial_state={"S1": 50.0})
    assert p2["observables"]["S1"][0] == pytest.approx(50.0)
    assert worker.snapshot_probe() == s
    # live timeline untouched and starts-from-current for predict
    assert worker.step(dt_s=0)["t"] == pytest.approx(5.0)
    p3 = worker.predict(horizon_s=1, observables=["S1"])
    assert p3["observables"]["S1"][0] == pytest.approx(10 * 2.718281828 ** -0.5, rel=1e-3)


def test_ensemble_stats_shape(worker):
    r = worker.ensemble(draws={"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}, n=4,
                        observables=["S2"], horizon_s=10, seed=1)
    assert {"mean", "p05", "p95"} <= r.keys()
    assert {"mean", "p05", "p95"} <= set(r) and "S2" in r["mean"]
    assert all(lo <= m <= hi for lo, m, hi in zip(r["p05"]["S2"], r["mean"]["S2"], r["p95"]["S2"]))
    assert r["p95"]["S2"][-1] > r["p05"]["S2"][-1]
    # live state untouched
    assert worker.step(dt_s=0)["t"] == 0.0


def test_ensemble_stochastic(worker):
    r = worker.ensemble(stochastic=True, n=3, seed=3, observables=["S2"], horizon_s=5, per_member=True)
    assert len(r["members"]) == 3
    finals = [m["S2"][-1] for m in r["members"]]
    assert len(set(finals)) > 1  # different seeds give different trajectories
    again = worker.ensemble(stochastic=True, n=3, seed=3, observables=["S2"], horizon_s=5,
                            per_member=True)
    assert again["members"] == r["members"]  # same seed reproduces
    assert worker.step(dt_s=1)["provenance"]["integrator"] == "cvode"


def test_ensemble_compiles_once():
    w = TwinWorker(ANT, "antimony", None, pool_size=2)
    w.start()
    try:
        n = 8
        draws = {"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}
        w.ensemble(draws=draws, n=n, observables=["S2"], seed=1)
        w.ensemble(draws=draws, n=n, observables=["S2"], seed=2)
        assert w.compile_count() <= w.pool_size
        assert w.compile_count() < n
    finally:
        w.close()


def test_seq_idempotent(worker):
    a = worker.step(dt_s=1, seq=7)
    b = worker.step(dt_s=1, seq=7)
    assert a == b
    assert worker.step(dt_s=1, seq=8)["t"] == pytest.approx(2.0)
    assert worker.step(dt_s=1, seq=7) == a  # older but cached retry: still a no-op
    assert worker.step(dt_s=0)["t"] == pytest.approx(2.0)  # no double-advance


def test_seq_stale_and_observable_mismatch(worker):
    worker.step(dt_s=1, seq=1, observables=["S1"])
    worker.step(dt_s=1, seq=2)
    with pytest.raises(WorkerError, match="different observables"):
        worker.step(dt_s=1, seq=1, observables=["S2"])
    for s in range(3, 25):
        worker.step(dt_s=0, seq=s)
    with pytest.raises(WorkerError, match="stale"):
        worker.step(dt_s=1, seq=1)  # evicted from the cache and older than latest
    assert worker.step(dt_s=0)["t"] == pytest.approx(2.0)


def test_ensemble_error_does_not_poison_next_call():
    w = TwinWorker(ANT, "antimony", None, pool_size=2)
    w.start()
    try:
        with pytest.raises(WorkerError):
            w.ensemble(members=[{"nope": 1}, {"nope": 2}], observables=["S2"])
        r = w.ensemble(draws={"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}, n=4,
                       observables=["S2"], seed=1)
        assert "mean" in r
        assert isinstance(w.compile_count(), int)
        # killed helper is evicted and respawned
        w._helpers[0].kill()
        with pytest.raises(WorkerError):
            w.ensemble(draws={"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}, n=4,
                       observables=["S2"])
        assert w._helpers == []
        assert "mean" in w.ensemble(draws={"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}}, n=4,
                                    observables=["S2"])
    finally:
        w.close()


EVENT_MODEL = (
    "S1 -> S2; k*S1; k=0.1; S1=10; S2=0; r := 2*time;"
    "at (time > 3): S1 = S1 + 20;"
)


def test_predict_non_mutating_with_events():
    a = TwinWorker(EVENT_MODEL, "antimony", None)
    b = TwinWorker(EVENT_MODEL, "antimony", None)
    a.start()
    b.start()
    try:
        for w in (a, b):
            w.step(dt_s=2)
        s = a.snapshot_probe()
        p = a.predict(horizon_s=6, interval_s=1, observables=["S1"])  # crosses the t=3 event
        assert max(p["observables"]["S1"]) > 20
        assert a.snapshot_probe() == s
        ta = [a.step(dt_s=1.5) for _ in range(4)]
        tb = [b.step(dt_s=1.5) for _ in range(4)]
        for x, y in zip(ta, tb):
            assert x["t"] == y["t"]
            assert x["observables"] == pytest.approx(y["observables"])
        assert ta[0]["observables"]["S1"] > 15  # event fired live (S1 would be ~7 without it)
    finally:
        a.close()
        b.close()


def test_reset(worker):
    worker.step(dt_s=3, parameters={"k": 1.0})
    worker.reset()
    r = worker.step(dt_s=1)
    assert r["t"] == pytest.approx(1.0)
    assert r["observables"]["S1"] == pytest.approx(10 * 2.718281828 ** -0.1, rel=1e-3)


def test_timeout_kills_worker():
    w = TwinWorker(ANT, "antimony", None)
    w.start()
    try:
        with pytest.raises(WorkerTimeout):
            w._call("_sleep", {"seconds": 60}, timeout=0.5)
        assert not w.is_alive()
    finally:
        w.close()
