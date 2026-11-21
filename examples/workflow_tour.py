"""A tour of the simulation core: one registry, one API, many ways to run.

Every function below drives the SAME unified facade
(``from viva_simulation_core import catalog, call, run, serve``) a different way,
one per task shape. A tiny kinetic model stands in for the sepsis engines so the
tour runs anywhere without a stiff solver; swap in a real engine and the calls
are identical.

Run ``python -m examples.workflow_tour`` for the narrated version.
"""
from __future__ import annotations

from viva_simulation_core import catalog, call, run, serve

SMALL = "S1 -> S2; k*S1; k = 0.3; S1 = 10; S2 = 0; total := S1 + S2"

# a two-node workflow: inc computes a = x + 1, scale computes b = a * 10
CHAIN = {
    "nodes": [
        {"id": "inc", "module": "MathExpressionStep",
         "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
        {"id": "scale", "module": "MathExpressionStep",
         "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}},
    ],
    "wires": [{"from": "inc.a", "to": "scale.a"}],
    "emit": ["scale.b", "inc.a"],
}


def demo_list() -> list[str]:
    """Discover the registry — every module, adapter, and API-calling step."""
    return [c.name for c in catalog()]


def demo_call() -> dict:
    """Run ONE module as a 1-node composite."""
    return call("MathExpressionStep",
                config={"expressions": [{"out": "y", "expr": "2*x + 3"}]},
                inputs={"x": 5.0}, steps=1)


def demo_connect_run() -> dict:
    """COMPOSE modules into a workflow and run it — values hand off node to node."""
    return run(CHAIN, steps=1, overrides={"inc.x": 5.0})


def demo_sweep(values=(1.0, 2.0, 3.0)) -> dict:
    """ENSEMBLE: run the same spec across a parameter grid, gather the results."""
    return {v: call("MathExpressionStep",
                    config={"expressions": [{"out": "y", "expr": "2*x"}]},
                    inputs={"x": v}, steps=1)["y"] for v in values}


def demo_serve() -> dict:
    """SERVE a model as a stepping API (in-process handle)."""
    model = serve(SMALL, "antimony")
    traj = model.predict(3)
    return {"observables": list(model.observables()),
            "S1_after_3": traj["S1"][-1]}


def demo_api_call(url: str) -> dict:
    """An API-CALLING step: a composite node that calls a served model over HTTP."""
    return call("ApiCallStep", config={"url": url, "observables": ["x"]},
                inputs={"x": 5.0}, steps=1)


def main() -> None:
    print("modules:", ", ".join(demo_list()))
    print("call    :", demo_call())
    print("connect :", demo_connect_run())
    print("sweep   :", demo_sweep())
    print("serve   :", demo_serve())
    print("api-call: start a server (see examples/serve_demo.py), then demo_api_call(url)")
    print("nextflow: see examples/nextflow/ — vivarium inside a Nextflow pipeline")


if __name__ == "__main__":
    main()
