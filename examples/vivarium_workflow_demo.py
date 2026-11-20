"""The Workflow API in four steps — list modules, run one, connect many, run it.

Runs the whole API in-process (no server), printing every real request and the
part of each response that matters. The four modules it uses are all real:
TelluriumUTCStep (exact SBML), NeuralProcess (neural surrogate), BioGearsProcess
(physiology), MathExpressionStep (math).

    uv run python examples/vivarium_workflow_demo.py

Reference: docs/workflow-api.md
"""
from __future__ import annotations

import tempfile

from fastapi.testclient import TestClient

from viva_simulation_core.api.app import create_app
from viva_simulation_core.api.config import ApiConfig
from viva_simulation_core.api.workflow_examples import train_demo_surrogate, tiered_workflow_spec

KEY = "demo-key"
H = {"Authorization": f"Bearer {KEY}"}


def step(n: int, title: str) -> None:
    print(f"\n{'─' * 70}\n{n}. {title}\n{'─' * 70}")


def main() -> None:
    client = TestClient(create_app(ApiConfig(api_keys=KEY, rate_capacity=1e6,
                                             rate_refill_per_s=1e6, request_timeout_s=120.0)))
    print("The Workflow API: snap simulation modules together and run them over HTTP.")

    # 1. LIST — what modules can I use?
    step(1, "List the modules you can use   (GET /modules)")
    modules = client.get("/modules", headers=H).json()
    print(f"{len(modules)} modules are registered. This demo uses four of them:")
    want = {"TelluriumUTCStep": "exact SBML model runner",
            "NeuralProcess": "fast neural surrogate",
            "BioGearsProcess": "physiology engine",
            "MathExpressionStep": "evaluates a math expression"}
    have = {m["name"]: m["kind"] for m in modules}
    for name, what in want.items():
        print(f"  • {name:<20} ({have.get(name, '?'):<7}) — {what}")

    # 2. RUN ONE — call a single module.
    step(2, "Run one module   (POST /modules/MathExpressionStep/call)")
    print("Compute y = 2x + 3 at x = 5:")
    r = client.post("/modules/MathExpressionStep/call",
                    json={"config": {"expressions": [{"out": "y", "expr": "2*x + 3"}]},
                          "inputs": {"x": 5.0}}, headers=H)
    print(f"  → y = {r.json()['outputs']['y']}")

    print("\n(training a tiny neural surrogate so the workflow has a real checkpoint …)")
    with tempfile.TemporaryDirectory() as d:
        info = train_demo_surrogate(d)

        # 3. CONNECT — wire modules into a workflow.
        step(3, "Connect modules into a workflow   (POST /workflows)")
        spec = tiered_workflow_spec(checkpoint=info["checkpoint"], timestep=info["timestep"],
                                    feature_keys=info["feature_keys"], include_biogears=True)
        created = client.post("/workflows", json=spec.model_dump(by_alias=True), headers=H).json()
        graph = created["graph"]
        print("Built a workflow from four real modules:")
        for n in graph["nodes"]:
            print(f"  • {n['id']:<10} = {n['module']}")
        print("Wires (one node's output → another's input):")
        for e in graph["edges"]:
            print(f"  • {e['from']}  →  {e['to']}")

        # 4. RUN — run the workflow.
        step(4, "Run the workflow   (POST /workflows/{id}/run)")
        core = tiered_workflow_spec(checkpoint=info["checkpoint"], timestep=info["timestep"],
                                    feature_keys=info["feature_keys"], include_biogears=False)
        wid = client.post("/workflows", json=core.model_dump(by_alias=True),
                          headers=H).json()["workflow_id"]
        seed = {f"surrogate.{info['feature_keys'][0]}": 10.0}
        run = client.post(f"/workflows/{wid}/run",
                          json={"n_steps": 3, "interval": info["timestep"], "overrides": seed},
                          headers=H).json()
        print("Ran the exact-engine + surrogate + compare workflow for 3 steps:")
        for ref, val in run["observables"].items():
            print(f"  • {ref:<34} = {val:.4f}")

    print(f"\n{'─' * 70}")
    print("Four verbs — list, run one, connect, run — over real modules, one API.")
    print("Reference: docs/workflow-api.md")


if __name__ == "__main__":
    main()
