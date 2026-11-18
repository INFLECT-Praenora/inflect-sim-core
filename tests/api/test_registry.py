import threading

import pytest

from viva_simulation_core.api.config import ApiConfig
from viva_simulation_core.api.registry import NotOwner, TwinRegistry, UnknownTwin

ANT = "S1 -> S2; k*S1; k=0.1; S1=10; S2=0;"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def reg(clock):
    r = TwinRegistry(ApiConfig(max_twins=2, twin_ttl_s=100.0), clock=clock)
    try:
        yield r
    finally:
        r.close()


def test_register_returns_warm_twin(reg):
    s = reg.register(ANT, "antimony", "k1")
    assert s.worker.is_alive()
    assert len(s.twin_id) >= 20 and s.owner_key == "k1"
    assert reg.get(s.twin_id, "k1") is s


def test_get_wrong_owner_403(reg):
    s = reg.register(ANT, "antimony", "k1")
    with pytest.raises(NotOwner):
        reg.get(s.twin_id, "k2")


def test_unknown_twin_404(reg):
    with pytest.raises(UnknownTwin):
        reg.get("nope", "k1")


def test_lru_eviction_at_capacity(reg, clock):
    a = reg.register(ANT, "antimony", "a")
    clock.t = 1
    b = reg.register(ANT, "antimony", "b")
    clock.t = 2
    reg.get(a.twin_id, "a")  # a is now more recent than b
    clock.t = 3
    c = reg.register(ANT, "antimony", "c")
    assert not b.worker.is_alive()
    with pytest.raises(UnknownTwin):
        reg.get(b.twin_id, "b")
    assert a.worker.is_alive() and c.worker.is_alive()


def test_ttl_eviction(reg, clock):
    s = reg.register(ANT, "antimony", "k1")
    clock.t = 101
    with pytest.raises(UnknownTwin):
        reg.get(s.twin_id, "k1")
    assert not s.worker.is_alive()


def test_evict_waits_on_inflight_lock(reg):
    s = reg.register(ANT, "antimony", "k1")
    done = threading.Event()
    with s.lock:
        t = threading.Thread(target=lambda: (reg.evict(s.twin_id), done.set()))
        t.start()
        assert not done.wait(0.5)  # blocked while the request holds the lock
        assert s.worker.is_alive()
    assert done.wait(10)
    t.join()
    assert not s.worker.is_alive()


def test_crashed_worker_rebuilds(reg):
    s = reg.register(ANT, "antimony", "k1")
    s.worker.step(dt_s=5)
    s.worker._main.proc.kill()
    s.worker._main.proc.join(5)
    assert not s.worker.is_alive()
    reg.rebuild_if_crashed(s)
    assert s.worker.is_alive()
    assert s.worker.step(dt_s=0)["t"] == pytest.approx(0.0)


def test_dedup_same_source_same_owner(reg):
    a = reg.register(ANT, "antimony", "k1")
    b = reg.register(ANT, "antimony", "k1")
    assert a is b and a.twin_id == b.twin_id
    c = reg.register(ANT, "antimony", "k2")
    assert c.twin_id != a.twin_id
