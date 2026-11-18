import subprocess
import sys

import numpy as np
import pytest

from viva_simulation_core.api.config import ApiConfig

MODEL = """
model toy
  S1 -> S2; k1*S1;
  S2 -> S3; k2*S2;
  k1 = 0.30; k2 = 0.12;
  S1 = 10; S2 = 0; S3 = 0;
end
"""
SPECIES = ["S1", "S2", "S3"]
H = {"Authorization": "Bearer k"}


def test_app_import_does_not_load_torch():
    code = ("import sys, viva_simulation_core.api.app; "
            "assert 'torch' not in sys.modules, 'torch leaked into app import'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_surrogate_torch_import_guard():
    from viva_simulation_core.api import surrogate_torch as st

    if not st.HAS_TORCH_BACKEND:
        with pytest.raises(Exception):
            st.TorchCheckpointBackend("/nonexistent")
        pytest.skip("viva-torch not installed")
    assert st.TorchCheckpointBackend  # importable


@pytest.mark.slow
def test_torch_surrogate_auto_and_fallback(tmp_path):
    pytest.importorskip("viva_torch")
    from fastapi.testclient import TestClient

    from viva_simulation_core.api.app import create_app
    from viva_simulation_core.api.surrogate_curate import train_and_curate
    from viva_simulation_core.api.surrogate_torch import TorchCheckpointBackend

    res = train_and_curate(source=MODEL, model_format="antimony", observables=SPECIES,
                           directory=tmp_path, timestep_s=1.0, n_trajectories=8, n_steps=20,
                           epochs=150)
    assert (tmp_path / f"{res.model_hash}.pt").is_file()
    cfg = ApiConfig(checkpoint_dir=str(tmp_path), api_keys="k", rate_capacity=1e6,
                    rate_refill_per_s=1e6)
    with TestClient(create_app(cfg, backend=TorchCheckpointBackend(tmp_path))) as c:
        r = c.post("/twins", json={"model": MODEL, "model_format": "antimony",
                                   "observables": SPECIES}, headers=H)
        assert r.status_code == 200, r.text
        tid = r.json()["twin_id"]
        assert "surrogate" in r.json()["engines_available"]
        body = {"observables": SPECIES, "horizon_s": 15.0, "interval_s": 1.0}
        ex = c.post(f"/twins/{tid}/predict", json={**body, "engine": "exact"}, headers=H).json()
        au = c.post(f"/twins/{tid}/predict", json=body, headers=H).json()
        assert au["engine_used"] == "surrogate" and au["in_distribution"] is True
        assert len(au["times"]) == len(ex["times"])
        a = np.concatenate([ex["observables"][s] for s in SPECIES])
        b = np.concatenate([au["observables"][s] for s in SPECIES])
        assert np.sqrt(np.mean((a - b) ** 2)) / (a.max() - a.min()) < 0.1
        far = c.post(f"/twins/{tid}/predict", json={**body, "horizon_s": 100.0}, headers=H).json()
        assert far["engine_used"] == "exact"


@pytest.mark.slow
def test_live_seed_outside_trained_region_gates_surrogate(tmp_path):
    pytest.importorskip("viva_torch")
    from fastapi.testclient import TestClient

    from viva_simulation_core.api.app import create_app
    from viva_simulation_core.api.surrogate_curate import train_and_curate
    from viva_simulation_core.api.surrogate_torch import TorchCheckpointBackend

    res = train_and_curate(source=MODEL, model_format="antimony", observables=SPECIES,
                           directory=tmp_path, timestep_s=1.0, n_trajectories=8, n_steps=20,
                           epochs=100)
    lo, hi = res.entry.target_bounds["S1"]
    assert lo <= 5.0 and hi >= 10.0
    cfg = ApiConfig(checkpoint_dir=str(tmp_path), api_keys="k", rate_capacity=1e6,
                    rate_refill_per_s=1e6)
    with TestClient(create_app(cfg, backend=TorchCheckpointBackend(tmp_path))) as c:
        tid = c.post("/twins", json={"model": MODEL, "model_format": "antimony",
                                     "observables": SPECIES}, headers=H).json()["twin_id"]
        body = {"observables": SPECIES, "horizon_s": 10.0, "interval_s": 1.0}
        # in-region seed: served by the surrogate
        assert c.post(f"/twins/{tid}/predict", json=body, headers=H).json()["engine_used"] == "surrogate"
        # drive S1 far outside the trained bound on the live twin
        r = c.post(f"/twins/{tid}/step", json={"dt_s": 0.0, "species": {"S1": hi * 10}}, headers=H)
        assert r.status_code == 200, r.text
        auto = c.post(f"/twins/{tid}/predict", json=body, headers=H).json()
        assert auto["engine_used"] == "exact"
        forced = c.post(f"/twins/{tid}/predict", json={**body, "engine": "surrogate"}, headers=H).json()
        assert forced["engine_used"] == "surrogate" and forced["in_distribution"] is False


@pytest.mark.slow
def test_time_grid_matches_exact_for_non_multiple_horizon(tmp_path):
    pytest.importorskip("viva_torch")
    from fastapi.testclient import TestClient

    from viva_simulation_core.api.app import create_app
    from viva_simulation_core.api.surrogate_curate import train_and_curate
    from viva_simulation_core.api.surrogate_torch import TorchCheckpointBackend

    train_and_curate(source=MODEL, model_format="antimony", observables=SPECIES,
                     directory=tmp_path, timestep_s=1.0, n_trajectories=4, n_steps=10, epochs=30)
    cfg = ApiConfig(checkpoint_dir=str(tmp_path), api_keys="k", rate_capacity=1e6,
                    rate_refill_per_s=1e6)
    with TestClient(create_app(cfg, backend=TorchCheckpointBackend(tmp_path))) as c:
        tid = c.post("/twins", json={"model": MODEL, "model_format": "antimony",
                                     "observables": SPECIES}, headers=H).json()["twin_id"]
        body = {"observables": ["S1"], "horizon_s": 2.5, "interval_s": 1.0}
        ex = c.post(f"/twins/{tid}/predict", json={**body, "engine": "exact"}, headers=H).json()
        au = c.post(f"/twins/{tid}/predict", json=body, headers=H).json()
        assert au["engine_used"] == "surrogate"
        assert au["times"] == pytest.approx(ex["times"]) and au["times"][-1] == pytest.approx(2.5)
