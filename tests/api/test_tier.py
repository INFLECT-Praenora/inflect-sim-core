import json
from types import SimpleNamespace as NS

import pytest

from viva_simulation_core.api.surrogate_backend import (
    CheckpointStore, FakeBackend, NullBackend, SurrogateEntry,
    SurrogateUnavailable, select_engine,
)


def _entry():
    return SurrogateEntry(
        checkpoint_ref="ck.pt", targets=["IL6", "TNF"], horizon_s=3600.0,
        param_ranges={"k": (0.0, 1.0)}, state_ranges={"IL6": (0.0, 10.0)},
        val_error={"rmse": 0.01},
    )


def _req(**kw):
    d = dict(observables=["IL6"], horizon_s=1800.0, parameters={"k": 0.5}, initial_state={"IL6": 1.0})
    d.update(kw)
    return NS(**d)


def test_auto_in_envelope_picks_surrogate():
    assert select_engine(_req(), _entry(), "auto") == ("surrogate", True)


def test_auto_out_of_range_falls_back_exact():
    assert select_engine(_req(parameters={"k": 5.0}), _entry(), "auto") == ("exact", True)


def test_auto_horizon_too_long_falls_back():
    assert select_engine(_req(horizon_s=9999.0), _entry(), "auto") == ("exact", True)


def test_auto_unknown_observable_falls_back():
    assert select_engine(_req(observables=["X"]), _entry(), "auto") == ("exact", True)


def test_force_surrogate_out_flags_not_in_dist():
    assert select_engine(_req(parameters={"k": 5.0}), _entry(), "surrogate") == ("surrogate", False)


def test_force_missing_raises_409():
    with pytest.raises(SurrogateUnavailable):
        select_engine(_req(), None, "surrogate")


def test_exact_always_exact():
    assert select_engine(_req(), _entry(), "exact") == ("exact", True)
    assert select_engine(_req(), None, "auto") == ("exact", True)


def test_checkpoint_store_lookup(tmp_path):
    (tmp_path / "abc.json").write_text(json.dumps({
        "checkpoint_ref": "ck.pt", "targets": ["IL6"], "horizon_s": 10.0,
        "param_ranges": {"k": [0, 1]}, "state_ranges": {}, "val_error": {"rmse": 0.1},
    }))
    store = CheckpointStore(tmp_path)
    e = store.lookup("abc")
    assert e.targets == ["IL6"] and e.param_ranges["k"] == (0.0, 1.0)
    assert store.lookup("missing") is None
    assert store.lookup("../abc") is None
    with pytest.raises(SurrogateUnavailable):
        store.predict(e, _req())


def test_null_backend_returns_none():
    assert NullBackend().lookup("abc") is None


def test_fake_backend_predict():
    e = _entry()
    fb = FakeBackend(e, lambda entry, req: {"IL6": [1.0, 2.0]})
    assert fb.lookup("any") is e
    assert fb.predict(e, _req(), seed_state={}) == {"IL6": [1.0, 2.0]}


def test_unknown_param_key_out_of_envelope():
    r = _req(parameters={"zzz": 0.5})
    assert select_engine(r, _entry(), "auto") == ("exact", True)
    assert select_engine(r, _entry(), "surrogate") == ("surrogate", False)


def test_unknown_state_key_and_state_out_of_range():
    assert select_engine(_req(initial_state={"zzz": 1.0}), _entry(), "auto") == ("exact", True)
    assert select_engine(_req(initial_state={"IL6": 11.0}), _entry(), "auto") == ("exact", True)


def test_inclusive_boundaries():
    e = _entry()
    assert select_engine(_req(horizon_s=3600.0), e, "auto") == ("surrogate", True)
    assert select_engine(_req(parameters={"k": 0.0}), e, "auto") == ("surrogate", True)
    assert select_engine(_req(parameters={"k": 1.0}), e, "auto") == ("surrogate", True)


def test_force_surrogate_in_envelope():
    assert select_engine(_req(), _entry(), "surrogate") == ("surrogate", True)


def test_non_numeric_value_out_of_envelope():
    assert select_engine(_req(parameters={"k": "0.5"}), _entry(), "auto") == ("exact", True)


def test_non_finite_sidecar_rejected(tmp_path):
    base = {"checkpoint_ref": "c", "targets": ["IL6"], "horizon_s": 10.0,
            "param_ranges": {"k": [0, 1]}, "state_ranges": {}, "val_error": {}}
    (tmp_path / "h1.json").write_text(json.dumps({**base, "horizon_s": float("inf")}))
    (tmp_path / "h2.json").write_text(json.dumps({**base, "param_ranges": {"k": [0, float("inf")]}}))
    s = CheckpointStore(tmp_path)
    assert s.lookup("h1") is None and s.lookup("h2") is None
