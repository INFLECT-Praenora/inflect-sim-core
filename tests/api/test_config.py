from viva_simulation_core.api.config import ApiConfig


def test_config_defaults():
    c = ApiConfig.from_env()
    assert c.max_model_bytes == 1024 * 1024
    assert c.max_species_reactions == 2000
    assert c.max_horizon_s == 24 * 3600
    assert c.max_steps_per_call == 10000
    assert c.max_ensemble_n == 256
    assert c.max_observables_per_call == 64
    assert c.request_timeout_s == 30
    assert c.register_compile_timeout_s == 60


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("INFLECT_API_MAX_ENSEMBLE_N", "8")
    monkeypatch.setenv("INFLECT_API_REQUEST_TIMEOUT_S", "2.5")
    c = ApiConfig.from_env()
    assert c.max_ensemble_n == 8
    assert c.request_timeout_s == 2.5


def test_negative_cost_rejected(monkeypatch):
    import pytest
    monkeypatch.setenv("INFLECT_API_COST_STEP", "-1")
    with pytest.raises(ValueError):
        ApiConfig.from_env()
