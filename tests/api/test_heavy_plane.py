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
    plane = LocalHeavyPlane(_runner, ApiConfig(max_ensemble_n=8), remote=remote)
    assert plane.submit_ensemble("t1", _req(9)) == {"remote": True}


def test_protocol_satisfied():
    assert isinstance(LocalHeavyPlane(_runner, ApiConfig()), HeavyPlane)
