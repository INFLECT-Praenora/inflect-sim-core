"""Serve a model as a stepping API, then call it from a vivarium composite.

Two halves of the same capability:
1. ``ServedModel`` — load a model, step/predict it in-process.
2. ``model_server`` + ``ApiCallStep`` — expose it over HTTP and let a composite
   node call it (the full serve/consume loop).

Run: ``python -m examples.serve_demo``.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

from viva_simulation_core import call, serve
from viva_simulation_core.serving import ServedModel, model_server

SMALL = "S1 -> S2; k*S1; k = 0.3; S1 = 10; S2 = 0"


def step_in_process() -> None:
    model = serve(SMALL, "antimony")
    print("observables:", ", ".join(model.observables()))
    for i in range(3):
        print(f"  step {i+1}:", {k: round(v, 3) for k, v in model.step().items()})


@contextmanager
def _served_over_http(model: ServedModel, port: int = 8765):
    import uvicorn

    config = uvicorn.Config(model_server(model), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:  # wait for the socket to come up
        pass
    try:
        yield f"http://127.0.0.1:{port}/step"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def serve_and_call() -> None:
    """The full loop: serve the model over HTTP, call it from an ApiCallStep."""
    with _served_over_http(ServedModel(SMALL, "antimony")) as url:
        out = call("ApiCallStep", config={"url": url, "observables": ["S1", "S2"]},
                   inputs={"S1": 10.0, "S2": 0.0}, steps=1)
        print("api-call result:", {k: round(v, 3) for k, v in out.items()})


def main() -> None:
    print("— serve a model, step it in-process —")
    step_in_process()
    print("\n— serve it over HTTP, call it from a composite —")
    serve_and_call()


if __name__ == "__main__":
    main()
