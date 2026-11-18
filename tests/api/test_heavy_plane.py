from types import SimpleNamespace

import pytest

from viva_simulation_core.api.config import ApiConfig
from viva_simulation_core.api.heavy_plane import HeavyPlane, LocalHeavyPlane, OverCapacity


def _req(n):
    return SimpleNamespace(ensemble=SimpleNamespace(n=n))


def _runner(twin_id, request):
    return {"twin": twin_id, "n": request.ensemble.n}


def test_local_runs_under_cap():
    plane = LocalHeavyPlane(_runner, ApiConfig(max_ensemble_n=8))
    assert plane.submit_ensemble("t1", _req(8)) == {"twin": "t1", "n": 8}


def test_over_cap_no_remote_raises():
    plane = LocalHeavyPlane(_runner, ApiConfig(max_ensemble_n=8))
    with pytest.raises(OverCapacity):
        plane.submit_ensemble("t1", _req(9))


def test_over_cap_with_remote_delegates():
    remote = LocalHeavyPlane(lambda t, r: {"remote": True}, ApiConfig(max_ensemble_n=10**6))
    calls = []
    plane = LocalHeavyPlane(lambda t, r: calls.append(t), ApiConfig(max_ensemble_n=8), remote=remote)
    assert plane.submit_ensemble("t1", _req(9)) == {"remote": True}
    assert calls == []


def test_protocol_satisfied():
    assert isinstance(LocalHeavyPlane(_runner, ApiConfig()), HeavyPlane)


def _members(k):
    return SimpleNamespace(ensemble=SimpleNamespace(n=None, members=[{}] * k))


def test_members_over_cap_not_bypassed():
    plane = LocalHeavyPlane(_runner, ApiConfig(max_ensemble_n=8))
    with pytest.raises(OverCapacity):
        plane.submit_ensemble("t1", _members(10_000))


def test_none_ensemble_and_short_members_run_locally():
    plane = LocalHeavyPlane(lambda t, r: "ok", ApiConfig(max_ensemble_n=8))
    assert plane.submit_ensemble("t1", SimpleNamespace(ensemble=None)) == "ok"
    assert plane.submit_ensemble("t1", _members(3)) == "ok"
    assert plane.submit_ensemble("t1", _members(0)) == "ok"
