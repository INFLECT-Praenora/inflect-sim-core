"""Task 6: the tiered worked workflow (BioGears + Tellurium + NeuralProcess).

Define + graph compose all four real viva- modules (always runnable — nodes
construct lazily). The exact+surrogate+compare core runs end-to-end everywhere;
the full run including the BioGears physiology engine needs compiled pybiogears,
so it skips cleanly where that engine is not built.
"""
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("viva_torch")
from fastapi.testclient import TestClient  # noqa: E402

from viva_simulation_core.api.app import create_app  # noqa: E402
from viva_simulation_core.api.config import ApiConfig  # noqa: E402
from viva_simulation_core.api.workflow_examples import (  # noqa: E402
    train_demo_surrogate, tiered_workflow_spec)
from viva_simulation_core.api.workflow_models import WorkflowNode, WorkflowSpec  # noqa: E402

KEY = "key-one"
H = {"Authorization": f"Bearer {KEY}"}


def _has_pybiogears() -> bool:
    import importlib.util
    return importlib.util.find_spec("pybiogears") is not None


def _client():
    return TestClient(create_app(ApiConfig(api_keys=KEY, rate_capacity=1000.0,
                                           rate_refill_per_s=1000.0, request_timeout_s=120.0)))


@pytest.mark.slow
def test_tiered_defines_four_real_modules():
    """The unified API composes + graphs all four real viva- modules at once."""
    client = _client()
    with tempfile.TemporaryDirectory() as d:
        info = train_demo_surrogate(d)
        spec = tiered_workflow_spec(checkpoint=info["checkpoint"], timestep=info["timestep"],
                                    feature_keys=info["feature_keys"], include_biogears=True)
        define = client.post("/workflows", json=spec.model_dump(by_alias=True), headers=H)
        assert define.status_code == 200, define.text
        graph = define.json()["graph"]
        assert {n["id"] for n in graph["nodes"]} == {"exact", "surrogate", "compare", "physio"}
        assert {n["module"] for n in graph["nodes"]} == {
            "TelluriumUTCStep", "NeuralProcess", "MathExpressionStep", "BioGearsProcess"}


@pytest.mark.slow
def test_tiered_core_runs_exact_surrogate_compare():
    """Exact SBML + neural surrogate + derivation run end-to-end at the surrogate's
    trained timestep (no BioGears engine required)."""
    client = _client()
    with tempfile.TemporaryDirectory() as d:
        info = train_demo_surrogate(d)
        spec = tiered_workflow_spec(checkpoint=info["checkpoint"], timestep=info["timestep"],
                                    feature_keys=info["feature_keys"], include_biogears=False)
        wid = client.post("/workflows", json=spec.model_dump(by_alias=True),
                          headers=H).json()["workflow_id"]
        seed = {f"surrogate.{info['feature_keys'][0]}": 10.0}
        run = client.post(f"/workflows/{wid}/run",
                          json={"n_steps": 3, "interval": info["timestep"], "overrides": seed},
                          headers=H)
        assert run.status_code == 200, run.text
        obs = run.json()["observables"]
        total = obs["compare.surrogate_total"]
        assert total is not None and isinstance(total, (int, float))
        assert set(run.json()["provenance"]["modules"]) == {
            "TelluriumUTCStep", "NeuralProcess", "MathExpressionStep"}


@pytest.mark.slow
@pytest.mark.skipif(not _has_pybiogears(), reason="pybiogears engine not built in this environment")
def test_tiered_full_run_with_biogears():
    client = _client()
    with tempfile.TemporaryDirectory() as d:
        info = train_demo_surrogate(d)
        spec = tiered_workflow_spec(checkpoint=info["checkpoint"], timestep=info["timestep"],
                                    feature_keys=info["feature_keys"], include_biogears=True)
        wid = client.post("/workflows", json=spec.model_dump(by_alias=True),
                          headers=H).json()["workflow_id"]
        seed = {f"surrogate.{info['feature_keys'][0]}": 10.0}
        run = client.post(f"/workflows/{wid}/run",
                          json={"n_steps": 3, "interval": info["timestep"], "overrides": seed},
                          headers=H)
        assert run.status_code == 200, run.text
        assert run.json()["observables"]["physio.mean_arterial_pressure_mmHg"] is not None


@pytest.mark.slow
def test_surrogate_wrong_interval_is_422():
    """A surrogate node wired at the wrong timestep must fail cleanly (422), not
    500 — the NeuralProcess refuses any interval != its trained timestep."""
    client = _client()
    with tempfile.TemporaryDirectory() as d:
        info = train_demo_surrogate(d)
        bad_dt = info["timestep"] * 2.0
        spec = WorkflowSpec(
            nodes=[WorkflowNode(id="s", module="NeuralProcess",
                                config={"checkpoint": info["checkpoint"]}, interval=bad_dt)],
            wires=[], emit=[f"s.{info['feature_keys'][0]}"])
        wid = client.post("/workflows", json=spec.model_dump(by_alias=True),
                          headers=H).json()["workflow_id"]
        seed = {f"s.{info['feature_keys'][0]}": 10.0}
        bad = client.post(f"/workflows/{wid}/run",
                          json={"n_steps": 1, "interval": bad_dt, "overrides": seed}, headers=H)
        assert bad.status_code == 422, bad.text
