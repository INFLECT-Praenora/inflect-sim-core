import pytest

from viva_simulation_core.api.config import ApiConfig
from viva_simulation_core.api.persistence import TwinMeta, TwinStore
from viva_simulation_core.api.registry import NotOwner, TwinRegistry, UnknownTwin

ANT = "S1 -> S2; k*S1; k=0.1; S1=10; S2=0;"


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def _meta(tid="t1"):
    return TwinMeta(twin_id=tid, source=ANT, model_format="antimony",
                    metadata={"observables": {"S1": "S1"}}, owner_key="k1",
                    model_hash="h", created=1.0)


def test_save_load_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    a = TwinStore(db)
    a.save(_meta())
    a.save(_meta())  # upsert
    a.close()
    b = TwinStore(db)
    got = b.load_all()
    b.close()
    assert got == [_meta()]


def test_memory_store_and_delete():
    st = TwinStore(":memory:")
    st.save(_meta())
    st.delete("t1")
    assert st.load_all() == []
    st.close()


def _reg(db, clock=None, **cfg):
    return TwinRegistry(ApiConfig(**cfg), clock=clock or Clock(), store=TwinStore(db))


def test_restart_rewarms_to_t0(tmp_path):
    db = tmp_path / "t.db"
    a = _reg(db)
    tid = a.register(ANT, "antimony", "k1").twin_id
    a.close()  # shutdown must NOT delete rows
    b = _reg(db)
    try:
        with pytest.raises(NotOwner):
            b.get(tid, "k2")
        s = b.get(tid, "k1")
        assert s.worker.is_alive() and s.twin_id == tid
        assert b.get(tid, "k1") is s
        with pytest.raises(UnknownTwin):
            b.get("nope", "k1")
    finally:
        b.close()


def test_explicit_evict_deletes_row(tmp_path):
    db = tmp_path / "t.db"
    a = _reg(db)
    tid = a.register(ANT, "antimony", "k1").twin_id
    a.evict(tid)
    a.close()
    b = _reg(db)
    try:
        with pytest.raises(UnknownTwin):
            b.get(tid, "k1")
    finally:
        b.close()


def test_evict_persisted_but_not_in_memory(tmp_path):
    db = tmp_path / "t.db"
    a = _reg(db)
    tid = a.register(ANT, "antimony", "k1").twin_id
    a.close()
    b = _reg(db)
    try:
        b.evict(tid)
        with pytest.raises(UnknownTwin):
            b.get(tid, "k1")
    finally:
        b.close()


def test_ttl_eviction_keeps_row(tmp_path):
    clock = Clock()
    r = _reg(tmp_path / "t.db", clock, twin_ttl_s=100.0)
    try:
        s = r.register(ANT, "antimony", "k1")
        clock.t = 500
        r._sweep()
        assert s.evicted
        s2 = r.get(s.twin_id, "k1")
        assert s2 is not s and s2.worker.is_alive()
    finally:
        r.close()
