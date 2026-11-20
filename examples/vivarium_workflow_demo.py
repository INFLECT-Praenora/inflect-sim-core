"""Narrated walkthrough of the unified vivarium-workflow API — the four verbs.

Runs entirely in-process via ``fastapi.testclient.TestClient`` — no server, no
hosting. Every request and response below is real, over real registered viva-
modules (TelluriumUTCStep, viva_torch.NeuralProcess, BioGearsProcess,
MathExpressionStep).

    uv run python examples/vivarium_workflow_demo.py

Full reference (routes, config): docs/workflow-api.md
"""
from __future__ import annotations

import json
import tempfile

from fastapi.testclient import TestClient

from viva_simulation_core.api.app import create_app
from viva_simulation_core.api.config import ApiConfig
from viva_simulation_core.api.workflow_examples import train_demo_surrogate, tiered_workflow_spec

KEY = "demo-key"
H = {"Authorization": f"Bearer {KEY}"}


def show(title: str, resp) -> dict:
    print(f"\n=== {title} ===")
    print(f"HTTP {resp.status_code}")
    body = resp.json()
    text = json.dumps(body, indent=2)
    print(text if len(text) < 1400 else text[:1400] + " …")
    return body


def main() -> None:
    client = TestClient(create_app(ApiConfig(api_keys=KEY, rate_capacity=1e6,
                                             rate_refill_per_s=1e6, request_timeout_s=120.0)))

    print("The unified vivarium-workflow API — define a module, call it, connect a")
    print("workflow, call the workflow — all over process-bigraph's own primitives.")

    # 1. DEFINE / DISCOVER — the live module registry.
    mods = show("1. define/discover modules  (GET /modules)",
                client.get("/modules", headers=H))
    names = {m["name"] for m in mods}
    print(f"\n   {len(mods)} modules registered. The four this demo composes are all present: "
          + ", ".join(n for n in ("TelluriumUTCStep", "NeuralProcess", "BioGearsProcess",
                                   "MathExpressionStep") if n in names))

    # 2. CALL A MODULE — one module as a 1-node composite.
    show("2. call one module  (POST /modules/MathExpressionStep/call)  y = 2x+3, x=5",
         client.post("/modules/MathExpressionStep/call",
                     json={"config": {"expressions": [{"out": "y", "expr": "2*x + 3"}]},
                           "inputs": {"x": 5.0}, "n_steps": 1}, headers=H))

    # Train a tiny neural surrogate so the tiered workflow has a real checkpoint.
    print("\n   (training a tiny neural surrogate of the small kinetic model …)")
    with tempfile.TemporaryDirectory() as d:
        info = train_demo_surrogate(d)
        spec = tiered_workflow_spec(checkpoint=info["checkpoint"], timestep=info["timestep"],
                                    feature_keys=info["feature_keys"], include_biogears=True)

        # 3. CONNECT A WORKFLOW — exact SBML + neural surrogate + physiology + derivation.
        created = show("3. connect a workflow  (POST /workflows)  tiered engines",
                       client.post("/workflows", json=spec.model_dump(by_alias=True), headers=H))
        wid = created["workflow_id"]

        # 4. CALL THE WORKFLOW — run the exact+surrogate+compare core at the trained timestep.
        core_spec = tiered_workflow_spec(checkpoint=info["checkpoint"], timestep=info["timestep"],
                                         feature_keys=info["feature_keys"], include_biogears=False)
        core_wid = client.post("/workflows", json=core_spec.model_dump(by_alias=True),
                               headers=H).json()["workflow_id"]
        seed = {f"surrogate.{info['feature_keys'][0]}": 10.0}
        show("4. call the workflow  (POST /workflows/{id}/run)  3 steps",
             client.post(f"/workflows/{core_wid}/run",
                         json={"n_steps": 3, "interval": info["timestep"], "overrides": seed},
                         headers=H))

        # The graph — the same node/edge structure the workbench renders.
        show("the workflow graph  (GET /workflows/{id}/graph)",
             client.get(f"/workflows/{wid}/graph", headers=H))

    print("\nAll four verbs, over real viva- modules, through one FastAPI surface.")
    print("Full reference: docs/workflow-api.md")


if __name__ == "__main__":
    main()
